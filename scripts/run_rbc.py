#!/usr/bin/env python3
"""Build one specialized RBC configuration; running a production case is explicit."""
import argparse
import json
from pathlib import Path
import shlex
import subprocess
try:
    from .build_convert_vti import hdf5_flags, build_converter
except ImportError:
    from build_convert_vti import hdf5_flags, build_converter

ROOT = Path(__file__).resolve().parents[1]
CASES = {1: (1e6, .7, [64, 96, 128, 512], 100, 150),
         2: (1e7, .7, [128, 192, 256, 512], 100, 200),
         3: (1e8, .7, [256, 384, 512], 200, 300),
         4: (1e8, 1., [256, 384, 512], 200, 300)}


def parameters(case, n):
    ra, pr, _, transient, average = CASES[case]
    h, uff = n - 1, .05
    nu = uff * h * (pr / ra)**.5
    kappa = nu / pr
    return dict(case=case, n=n, Ra=ra, Pr=pr, H=h, aspect_ratio=n/h,
                Uff=uff, nu=nu, kappa=kappa, gbeta=uff**2/h,
                tau=0.5+3*nu, tau_thermal=0.5+3*kappa, tff=h/uff,
                transient_steps=round(transient*h/uff),
                averaging_steps=round(average*h/uff),
                moment_buffers_GiB=2*14*n**3*4/2**30)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--case', type=int, choices=CASES, default=1)
    p.add_argument('--grid', type=int)
    p.add_argument('--rr', type=int, choices=[2, 3], default=3)
    p.add_argument('--list', action='store_true')
    p.add_argument('--test-grid', action='store_true', help='allow a non-production grid')
    p.add_argument('--conduction', action='store_true')
    p.add_argument('--validation', choices=['V1', 'V2', 'V3'],
                   help='isolated analytical validation physics (campaign sets schedules)')
    p.add_argument('--precision', choices=['single', 'double'],
                   help='default: double for analytical validation, single for production')
    p.add_argument('--output-root', type=Path, default=ROOT/'output')
    p.add_argument('--hdf5-root', type=Path, help='separate HDF5 root; default keeps the legacy layout')
    p.add_argument('--sample', type=int, default=100)
    p.add_argument('--transient-steps', type=int)
    p.add_argument('--steps', type=int, help='absolute final step, also compiled as NSTEPS')
    p.add_argument('--checkpoint-every', type=int, default=5000)
    p.add_argument('--run-id', default=None)
    p.add_argument('--continue', dest='resume', action='store_true')
    p.add_argument('--fields', action='store_true', help='legacy alias: 3D temperature at checkpoint cadence')
    for name in ('slice-every', 'field-every', 'profile-every', 'progress-every', 'output-start'):
        p.add_argument('--'+name, type=int, default=0, help='independent lattice-step schedule (0 disables interval)')
    p.add_argument('--slice-velocity', action='store_true', help='include physical uz alongside slice temperature')
    p.add_argument('--keep-checkpoints', type=int, default=0, help='0 keeps all; >=2 retains latest N atomic checkpoints')
    p.add_argument('--compression', type=int, choices=range(10), default=1, help='lossless HDF5 gzip level; 0 disables gzip')
    p.add_argument('--field-io', choices=('async', 'sync', 'serial'), default='async',
                   help='temperature snapshots: isolated background writer, synchronous encoder, or bounded plane fallback')
    p.add_argument('--io-workers', type=int, choices=range(1, 33), default=4, help='CPU compression workers (default 4)')
    p.add_argument('--arch', default='sm_86')
    p.add_argument('--gpus', type=int, choices=(1, 2), default=1)
    p.add_argument('--devices', help='CUDA ordinals: e.g. 0 for one GPU, 0,1 for two')
    p.add_argument('--split-z', type=int, help='compile-time two-GPU interface; default NZ//2; tune for unequal GPUs')
    p.add_argument('--halo-transport', choices=('peer', 'host', 'auto'), default='peer',
                   help='two-GPU transport; peer requires bidirectional cudaMemcpyPeerAsync access')
    p.add_argument('--gpu-info', action='store_true', help='build and print visible GPUs/P2P capability; do not simulate')
    p.add_argument('--build-only', action='store_true')
    a = p.parse_args()
    a.precision = a.precision or ('double' if a.validation else 'single')
    if a.list:
        print(json.dumps([parameters(c, n) for c in CASES for n in CASES[c][2]], indent=2))
        return
    n = a.grid or CASES[a.case][2][0]
    if a.split_z is not None and not 2 <= a.split_z <= n - 2:
        p.error('--split-z must leave at least two planes on each GPU')
    if n < 4 or (n not in CASES[a.case][2] and not (a.test_grid or a.validation)):
        p.error('grid must belong to the case; use --test-grid for short verification')
    if a.validation:
        if a.steps is None:
            a.steps = (n-1)**2*(1 if a.validation == 'V1' else 5)
        if a.transient_steps is None:
            a.transient_steps = 0
    if a.sample <= 0 or a.checkpoint_every <= 0 or (a.steps is not None and a.steps < 0) or (a.transient_steps is not None and a.transient_steps < 0):
        p.error('invalid interval or step count')
    if any(getattr(a, key) < 0 for key in ('slice_every', 'field_every', 'profile_every', 'progress_every', 'output_start', 'keep_checkpoints')) or a.keep_checkpoints == 1:
        p.error('output intervals must be nonnegative; retain at least two checkpoints or use 0 for all')
    build = ROOT / 'build'
    build.mkdir(exist_ok=True)
    tag = a.validation.lower() if a.validation else f'rbc{a.case}'
    exe = build / f'{tag}_n{n}_rr{a.rr}_{a.precision}'
    if a.split_z is not None:
        exe = exe.with_name(exe.name + f'_z{a.split_z}')
    cmd = ['nvcc', '-std=c++20', '-O3', '--restrict', '--expt-relaxed-constexpr',
           '--extended-lambda', f'-arch={a.arch}', '-DCASE_RAYLEIGH_BENARD',
           f'-DRBC_CASE={a.case}', f'-DRBC_N={n}', f'-DREGULARIZATION_ORDER={a.rr}',
           f'-DRBC_SAMPLE={a.sample}', f'-DRBC_STAMP={a.checkpoint_every}']
    if a.conduction or a.validation:
        cmd += ['-DRBC_RA=0']
    if a.split_z is not None:
        cmd += [f'-DMRL_SPLIT_Z={a.split_z}']
    if a.validation:
        cmd += [f'-DRBC_VALIDATION_CASE={a.validation[1:]}']
    if a.precision == 'double':
        cmd += ['-DMRL_DOUBLE_PRECISION']
    if a.transient_steps is not None:
        cmd += [f'-DRBC_TRANSIENT_STEPS={a.transient_steps}']
    if a.steps is not None:
        cmd += [f'-DRBC_STEPS={a.steps}']
    flags = hdf5_flags()
    build_converter(flags)
    cmd += ['src/main.cu', '-o', str(exe), *flags]
    print(shlex.join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)
    if a.build_only:
        return
    if a.gpu_info:
        subprocess.run([str(exe), '--gpu-info'], cwd=ROOT, check=True)
        return
    rid = a.run_id or (f'{tag}_n{n}_rr{a.rr}_{a.precision}' if a.validation else f'rbc{a.case}_n{n}_rr{a.rr}')
    run = [str(exe), '--runId', rid, '--checkpoint-every', str(a.checkpoint_every),
           '--output-root', str(a.output_root.resolve()), '--gpus', str(a.gpus),
           '--halo-transport', a.halo_transport, '--field-io', a.field_io, '--io-workers', str(a.io_workers)]
    if a.devices is not None:
        run += ['--devices', a.devices]
    if a.hdf5_root is not None:
        run += ['--hdf5-root', str(a.hdf5_root.resolve())]
    if a.resume:
        run += ['--continue']
    if a.fields:
        run += ['--fields']
    for key in ('slice_every', 'field_every', 'profile_every', 'progress_every', 'output_start', 'keep_checkpoints', 'compression'):
        run += ['--'+key.replace('_', '-'), str(getattr(a, key))]
    if a.slice_velocity:
        run += ['--slice-velocity']
    subprocess.run(run, cwd=ROOT, check=True)


if __name__ == '__main__':
    main()
