#!/usr/bin/env python3
"""Short two-slab CUDA checks, including one-GPU emulation and real peers when present."""
import argparse
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from build_convert_vti import hdf5_flags


def execute(command, log):
    result = subprocess.run(list(map(str, command)), cwd=ROOT, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    log.write_text(result.stdout)
    if result.returncode:
        raise RuntimeError(f'exit {result.returncode}: {command}; see {log}')
    return result.stdout


def compare(a, b, n, step, precision=4, snapshots=True):
    dtype = np.float32 if precision == 4 else np.float64
    states = [np.frombuffer((run / f'checkpoints/step_{step:09d}.chk').read_bytes(),
                           dtype=dtype, count=14 * n**3, offset=struct.calcsize('@8s6I3Q9d')) for run in (a, b)]
    result = dict(moments_bitwise_equal=np.array_equal(*states),
                  moments_max_abs=float(np.max(np.abs(states[0] - states[1]))), diagnostics={})
    def compare_moments(x, y):
        if precision == 4:
            np.testing.assert_array_equal(x, y)
        else:
            # Specializing constant slab offsets can change FMA contraction in
            # the force correction to nearly cancelled FP64 stresses. Bound
            # roundoff on the unit-density lattice scale, rather than relative
            # error in a component approaching zero. Physical fields are still
            # compared bitwise below; this is not a discretization tolerance.
            tolerance = 32 * np.finfo(np.float64).eps
            np.testing.assert_allclose(x, y, rtol=tolerance, atol=tolerance)
    compare_moments(*states)
    data_a = a / 'hdf5' if (a / 'hdf5').is_dir() else a
    data_b = b / 'hdf5' if (b / 'hdf5').is_dir() else b
    for name in ('timeseries', 'profiles', 'convergence'):
        with h5py.File(data_a / 'statistics.h5') as left, h5py.File(data_b / 'statistics.h5') as right:
            x, y = left[f'statistics/{name}'][:], right[f'statistics/{name}'][:]
        np.testing.assert_allclose(x, y, rtol=2.e-12, atol=2.e-13)
        finite = np.isfinite(x) & np.isfinite(y)
        result['diagnostics'][name] = float(np.max(np.abs(x[finite] - y[finite]), initial=0))
    if snapshots:
        for kind in ('fields', 'slices'):
            for source in (data_a / kind).glob('step_*.h5'):
                with h5py.File(source) as left, h5py.File(data_b / kind / source.name) as right:
                    for name in left['fields']:
                        np.testing.assert_array_equal(left[f'fields/{name}'][:], right[f'fields/{name}'][:])
        for source in (data_a / 'validation').glob('snapshot_*.h5'):
            with h5py.File(source) as left, h5py.File(data_b / 'validation' / source.name) as right:
                compare_moments(left['moments'][:], right['moments'][:])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work', type=Path)
    parser.add_argument('--quick', action='store_true', help='only the odd-grid production case and restarts')
    parser.add_argument('--only', nargs='+', help='select named cases from the short suite')
    parser.add_argument('--sanitizers', action='store_true', help='run four Compute Sanitizer tools on both exchange modes')
    parser.add_argument('--split-z', type=int, help='compile an uneven split (use with --quick)')
    parser.add_argument('--arch', default=os.environ.get('CUDA_ARCH', 'sm_86'))
    args = parser.parse_args()
    work = (args.work or Path(tempfile.mkdtemp(prefix='thermal-two-gpu-'))).resolve()
    if work.exists() and any(work.iterdir()):
        parser.error('--work must be new or empty')
    work.mkdir(parents=True, exist_ok=True)
    report = dict(work=str(work), cases={}, hardware_peer_test='not_run')
    configs = [('odd', 7, 3, None, False, False)] if args.quick else [
        ('minimum', 4, 3, None, False, False),
        ('odd', 7, 3, None, False, False),
        ('regular', 16, 3, None, False, False),
        ('conduction-rr2', 12, 2, None, False, True),
        ('conduction-rr3', 12, 3, None, False, True),
        ('thermal-v2', 16, 3, 2, True, True),
        ('forced-v3', 16, 3, 3, True, True),
        ('queued', 7, 3, None, False, False)]
    if args.only:
        unknown = set(args.only) - {case[0] for case in configs}
        if unknown:
            parser.error(f'unknown cases: {sorted(unknown)}')
        configs = [case for case in configs if case[0] in args.only]
    gpu_info = None
    for name, n, rr, validation, double, zero_ra in configs:
        queued = name == 'queued'
        sample = 1000 if queued else 5
        steps = 257 if queued else 60
        executable = work / name
        defines = ['-DCASE_RAYLEIGH_BENARD', '-DMRL_MULTI_GPU_TESTING', f'-DRBC_N={n}',
                   '-DRBC_TRANSIENT_STEPS=10', f'-DRBC_SAMPLE={sample}', f'-DREGULARIZATION_ORDER={rr}']
        if zero_ra:
            defines += ['-DRBC_RA=0']
        if validation:
            defines += [f'-DRBC_VALIDATION_CASE={validation}']
        if double:
            defines += ['-DMRL_DOUBLE_PRECISION']
        if args.split_z is not None:
            if not 2 <= args.split_z <= n - 2:
                parser.error('requested split leaves fewer than two local planes')
            defines += [f'-DMRL_SPLIT_Z={args.split_z}']
        execute(['nvcc', '-std=c++20', '-O3', '--restrict', '--expt-relaxed-constexpr', '--extended-lambda',
                 f'-arch={args.arch}', '-lineinfo', *defines, ROOT / 'src/main.cu', '-o', executable,
                 *hdf5_flags()], work / f'{name}-build.log')
        if gpu_info is None:
            gpu_info = json.loads(execute([executable, '--gpu-info'], work / 'gpu-info.json'))
            report['gpu_info'] = gpu_info
        modes = {'single': [], 'local': ['--emulate-two-gpus'],
                 'host': ['--emulate-two-gpus', '--halo-transport', 'host']}
        if len(gpu_info['devices']) >= 2:
            modes['two-host'] = ['--gpus', 2, '--devices', '0,1', '--halo-transport', 'host']
            if gpu_info['peer_access'][0][1] and gpu_info['peer_access'][1][0]:
                modes['two-peer'] = ['--gpus', 2, '--devices', '0,1']
                report['hardware_peer_test'] = 'executed'
            else:
                report['hardware_peer_test'] = 'SKIP: bidirectional P2P unavailable'
        else:
            report['hardware_peer_test'] = 'SKIP: only one visible GPU; emulation is not a physical peer test'

        def run(mode, end, rid=None, resume=False):
            rid = rid or f'{name}-{mode}'
            command = [executable, '--runId', rid, '--steps', end, '--checkpoint-every', 257 if queued else 30,
                       '--output-root', work / 'raw', '--hdf5-root', work / 'hdf5', '--compression', 0]
            if not queued:
                command += ['--field-every', 30, '--slice-every', 20, '--slice-velocity', '--profile-every', 17, '--progress-every', 19]
            if resume:
                command += ['--continue']
            execute(command + modes[mode], work / f'{rid}-{end}-{mode}.log')
            return work / 'raw/rayleigh_benard' / rid

        reference = run('single', steps)
        results = {}
        for mode in modes:
            if mode == 'single':
                continue
            candidate = run(mode, steps)
            results[mode] = compare(reference, candidate, n, steps, 8 if double else 4, not queued)
        if name == 'odd':
            for first, second in [('single', 'local'), ('local', 'single'), ('local', 'host')]:
                rid = f'restart-{first}-{second}'
                run(first, 30, rid)
                candidate = run(second, 60, rid, resume=True)
                results[rid] = compare(reference, candidate, n, 60)
            if args.sanitizers:
                report['sanitizers'] = {}
                for mode in ('local', 'host'):
                    for tool in ('memcheck', 'racecheck', 'synccheck', 'initcheck'):
                        rid = f'sanitizer-{mode}-{tool}'
                        execute(['compute-sanitizer', '--tool', tool, '--error-exitcode', 99,
                                 executable, '--runId', rid, '--steps', 25, '--checkpoint-every', 25,
                                 '--output-root', work / 'sanitizers', *modes[mode]], work / f'{rid}.log')
                        report['sanitizers'][f'{mode}-{tool}'] = 'PASS'
        report['cases'][name] = results
        (work / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
        print(name, 'PASS', flush=True)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
