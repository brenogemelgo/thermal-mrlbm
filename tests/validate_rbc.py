#!/usr/bin/env python3
"""Focused CUDA validation, never a production campaign. Requires nvcc and a GPU."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(ROOT))
from run_rbc import CASES, parameters, hdf5_flags
from postRayleigh import block_errors, uncertainty, load
import h5py


def command(cmd, cwd=ROOT, expect=0):
    r = subprocess.run([str(c) for c in cmd], cwd=cwd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    print(r.stdout, end='', flush=True)
    if r.returncode != expect:
        raise RuntimeError(f'exit {r.returncode}: {cmd}')
    return r.stdout


def build(work, name, source='src/main.cu', **defs):
    exe = work/name
    command(['nvcc', '-std=c++20', '-O2', '--expt-relaxed-constexpr', '--extended-lambda',
             '-arch='+os.environ.get('CUDA_ARCH', 'sm_86'), '-DCASE_RAYLEIGH_BENARD',
             *[f'-D{k}={v}' for k, v in defs.items()], ROOT/source, '-o', exe, *hdf5_flags()])
    return exe


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--work', type=Path, help='preserve test artifacts here (default: a new temporary directory)')
    a = p.parse_args()
    work = (a.work or Path(tempfile.mkdtemp(prefix='thermal-rbc-tests-'))).resolve()
    work.mkdir(parents=True, exist_ok=True)
    results = {'work': str(work), 'tests': {}}
    print('artifacts:', work, flush=True)
    for c, (_, _, grids, _, _) in CASES.items():
        for n in grids:
            q = parameters(c, n)
            assert np.isclose(q['gbeta']*q['H']**3/(q['nu']*q['kappa']), q['Ra'])
            assert np.isclose(q['nu']/q['kappa'], q['Pr'])
            assert q['tau'] > .5 and q['tau_thermal'] > .5
    results['tests']['all_parameter_sets'] = 'PASS'
    # Algebraic CPU quadrature plus GPU plate reconstruction, both with and without forcing.
    unit = build(work, 'moments-forced', 'tests/moments.cu', RBC_N=12)
    command([unit], cwd=work)
    fringe = build(work, 'moments-fringe', 'tests/moments.cu', RBC_N=7)
    command([fringe], cwd=work)
    results['tests']['fused_boundary_fringe_ownership'] = 'PASS (N=7,12; poisoned output and exact NEE projection)'
    errors = []
    flux_errors = []
    shear_errors = []
    for n in [12, 24, 48]:
        unit = build(work, f'diffusion-{n}', 'tests/moments.cu', RBC_N=n, RBC_RA=0, MRL_DOUBLE_PRECISION=1)
        log = command([unit, 'diffusion'], cwd=work)
        line = next(s for s in log.splitlines() if s.startswith('diffusion ')).split()
        errors.append(float(line[2])); flux_errors.append(float(line[3])); shear_errors.append(float(line[4]))
    rates = np.log(np.array(errors[:-1])/errors[1:])/np.log(np.array([23/11, 47/23]))
    assert min(rates) > 1.8, (errors, rates)
    shear_rates = np.log(np.array(shear_errors[:-1])/shear_errors[1:])/np.log(np.array([23/11, 47/23]))
    assert min(shear_rates) > 1.8, (shear_errors, shear_rates)
    assert flux_errors[-1] < flux_errors[0]/4
    results['tests']['nee_diffusion'] = dict(errors=errors, wall_flux_errors=flux_errors, observed_orders=rates.tolist(),
                                               shear_errors=shear_errors, shear_observed_orders=shear_rates.tolist())
    # Independent Nusselt diagnostics and RR equivalence at zero velocity.
    runs = []
    for rr in [2, 3]:
        exe = build(work, f'conduction-rr{rr}', RBC_N=12, RBC_RA=0,
                    RBC_TRANSIENT_STEPS=10, RBC_SAMPLE=5, REGULARIZATION_ORDER=rr)
        rid = f'conduction-rr{rr}'
        command([exe, '--runId', rid, '--steps', 200, '--checkpoint-every', 100], cwd=work)
        run = work/'output/rayleigh_benard'/rid
        data = load(run, 'rbc_timeseries.csv')
        for field in ['nu_bottom', 'nu_top', 'nu_volume', 'nu_kinetic', 'nu_thermal']:
            assert np.max(np.abs(data[field]-1)) < 3e-5, (field, data[field])
        assert np.max(np.abs(data['max_velocity'])) < 1e-6
        runs.append(run)
    p2 = load(runs[0], 'rbc_profiles.csv')
    p3 = load(runs[1], 'rbc_profiles.csv')
    np.testing.assert_array_equal(p2['temperature'], p3['temperature'])
    results['tests']['conduction_all_five_nu_and_rr_limits'] = 'PASS (<3e-5)'
    # A short finite-Ra run, restart inside the averaging window, and corrupted data rejection.
    exe = build(work, 'buoyant', RBC_N=16, RBC_TRANSIENT_STEPS=20, RBC_SAMPLE=10)
    command([exe, '--runId', 'full', '--steps', 300, '--checkpoint-every', 100], cwd=work)
    command([exe, '--runId', 'split', '--steps', 100, '--checkpoint-every', 100], cwd=work)
    # Simulate rows written after an older checkpoint, as in interrupted output.
    split = work/'output/rayleigh_benard/split'
    with h5py.File(split/'statistics.h5', 'r+') as out:
        table = out['statistics/timeseries']
        rows, cols = table.shape
        table.resize((rows+1, cols))
        table[rows] = np.zeros(cols)
        table[rows, 0] = 999999
    command([exe, '--runId', 'split', '--steps', 300, '--continue'], cwd=work)
    full = work/'output/rayleigh_benard/full'
    assert (full/'checkpoints/step_000000300.chk').read_bytes() == (split/'checkpoints/step_000000300.chk').read_bytes()
    for name in ['rbc_profiles.csv', 'rbc_timeseries.csv', 'rbc_convergence.csv']:
        np.testing.assert_array_equal(load(full, name), load(split, name))
    results['tests']['restart_bitwise_and_series_rollback'] = 'PASS'
    chk = split/'checkpoints/step_000000300.chk'
    original = chk.read_bytes()
    chk.write_bytes(b'OLD_FULL'+original[8:])
    log = command([exe, '--runId', 'split', '--steps', 400, '--continue'], cwd=work, expect=1)
    assert 'incompatible checkpoint' in log
    chk.write_bytes(original[:-24])
    log = command([exe, '--runId', 'split', '--steps', 400, '--continue'], cwd=work, expect=1)
    assert 'truncated checkpoint' in log
    chk.write_bytes(original)
    # Production-sized RBC-1 short integration; RBC-4 finite forcing/Pr specialization.
    for case, n, rr in [(1, 64, 2), (1, 64, 3), (2, 16, 3), (3, 16, 3), (4, 32, 3)]:
        exe = build(work, f'case{case}-{n}-rr{rr}', RBC_CASE=case, RBC_N=n,
                    RBC_TRANSIENT_STEPS=20, RBC_SAMPLE=10, REGULARIZATION_ORDER=rr)
        rid = exe.name
        command([exe, '--runId', rid, '--steps', 200, '--checkpoint-every', 200], cwd=work)
        run = work/'output/rayleigh_benard'/rid
        data = load(run, 'rbc_timeseries.csv')
        for name in data.dtype.names:
            if name != 'eta_min':
                assert np.all(np.isfinite(data[name])), (rid, name)
        assert data['mass_density'][-1] > .9
    results['tests']['short_buoyant_case_specializations'] = 'PASS (not DNS validation)'
    # Plot/report pipeline and correlated uncertainty smoke tests.
    rng = np.random.default_rng(1)
    noise = rng.normal(size=4096)
    correlated = np.empty_like(noise); correlated[0] = noise[0]
    for i in range(1, len(noise)):
        correlated[i] = .9*correlated[i-1]+noise[i]
    assert uncertainty(correlated)['effective_samples'] < len(noise)/3
    assert len(block_errors(correlated)) >= 8
    command([sys.executable, ROOT/'postRayleigh.py', 'all', runs[0], full, '--out', work/'plots'])
    assert len(list((work/'plots').rglob('*.pdf'))) >= 7
    r = json.loads((work/'plots/full/report.json').read_text())
    assert not r['dns_convergence_established']
    results['tests']['plots_and_correlation_aware_reports'] = 'PASS'
    # The preserved jet boundary code must compile against the changed moment policy.
    command(['nvcc', '-std=c++20', '-O2', '--expt-relaxed-constexpr', '--extended-lambda',
             '-arch='+os.environ.get('CUDA_ARCH', 'sm_86'), '-DCASE_JET', ROOT/'src/main.cu', '-o', work/'jet', *hdf5_flags()])
    results['tests']['jet_compile'] = 'PASS'
    (work/'validation.json').write_text(json.dumps(results, indent=2)+'\n')
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
