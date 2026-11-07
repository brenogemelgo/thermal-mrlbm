#!/usr/bin/env python3
"""Build one specialized RBC configuration; running a production case is explicit."""
import argparse
import json
from pathlib import Path
import shlex
import subprocess

ROOT = Path(__file__).resolve().parents[1]
CASES = {1: (1e6, .7, [64, 96, 128], 100, 150),
         2: (1e7, .7, [128, 192, 256], 100, 200),
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
    p.add_argument('--sample', type=int, default=100)
    p.add_argument('--transient-steps', type=int)
    p.add_argument('--steps', type=int, help='absolute final step, also compiled as NSTEPS')
    p.add_argument('--checkpoint-every', type=int, default=5000)
    p.add_argument('--run-id', default=None)
    p.add_argument('--continue', dest='resume', action='store_true')
    p.add_argument('--fields', action='store_true', help='also export full moment fields and VTI')
    p.add_argument('--arch', default='sm_86')
    p.add_argument('--build-only', action='store_true')
    a = p.parse_args()
    if a.list:
        print(json.dumps([parameters(c, n) for c in CASES for n in CASES[c][2]], indent=2))
        return
    n = a.grid or CASES[a.case][2][0]
    if n < 4 or (n not in CASES[a.case][2] and not a.test_grid):
        p.error('grid must belong to the case; use --test-grid for short verification')
    if a.sample <= 0 or a.checkpoint_every <= 0 or (a.steps is not None and a.steps < 0) or (a.transient_steps is not None and a.transient_steps < 0):
        p.error('invalid interval or step count')
    build = ROOT / 'build'
    build.mkdir(exist_ok=True)
    exe = build / f'rbc{a.case}_n{n}_rr{a.rr}'
    cmd = ['nvcc', '-std=c++20', '-O3', '--restrict', '--expt-relaxed-constexpr',
           '--extended-lambda', f'-arch={a.arch}', '-DCASE_RAYLEIGH_BENARD',
           f'-DRBC_CASE={a.case}', f'-DRBC_N={n}', f'-DREGULARIZATION_ORDER={a.rr}',
           f'-DRBC_SAMPLE={a.sample}', f'-DRBC_STAMP={a.checkpoint_every}']
    if a.conduction:
        cmd += ['-DRBC_RA=0']
    if a.transient_steps is not None:
        cmd += [f'-DRBC_TRANSIENT_STEPS={a.transient_steps}']
    if a.steps is not None:
        cmd += [f'-DRBC_STEPS={a.steps}']
    cmd += ['src/main.cu', '-o', str(exe)]
    print(shlex.join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)
    if a.build_only:
        return
    rid = a.run_id or f'rbc{a.case}_n{n}_rr{a.rr}'
    run = [str(exe), '--runId', rid, '--checkpoint-every', str(a.checkpoint_every)]
    if a.resume:
        run += ['--continue']
    if a.fields:
        run += ['--fields']
    subprocess.run(run, cwd=ROOT, check=True)


if __name__ == '__main__':
    main()
