#!/usr/bin/env python3
"""Short, sequential GPU benchmarks; never use production grids or schedules."""
import argparse
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import statistics
import struct
import subprocess
import tempfile

import h5py
import numpy as np

from build_convert_vti import hdf5_flags

ROOT = Path(__file__).resolve().parents[1]


def execute(command, cwd, log):
    command = list(map(str, command))
    print(shlex.join(command), flush=True)
    result = subprocess.run(command, cwd=cwd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    log.write_text(result.stdout)
    if result.returncode:
        raise RuntimeError(f'exit {result.returncode}; see {log}')
    return result.stdout


def measurements(text):
    values = {}
    for line in text.splitlines():
        pairs = re.findall(r'(\w+)=([-+\d.eE]+)(?=\s|$)', line)
        if pairs:
            prefix = line.split()[0] + '.' if line.startswith(('streamCollide ', 'reduceTiles ', 'reducePlanes ')) else ''
            values.update({prefix + key: float(value) for key, value in pairs})
    return values


def compare(before, after, n, steps):
    # The existing version-2 checkpoint header is unchanged; reject any ABI
    # mismatch before comparing the fourteen contiguous SoA moment fields.
    header = struct.Struct('@8s6I3Q9d')
    states = []
    for run in (before, after):
        raw = (run / 'checkpoints' / f'step_{steps:09d}.chk').read_bytes()
        info = header.unpack_from(raw)
        if info[:6] != (b'TMRLBM2\0', n, n, n, 14, 4):
            raise RuntimeError('unsupported benchmark checkpoint ABI')
        states.append(np.frombuffer(raw, dtype=np.float32, count=14 * n**3, offset=header.size))
    differences = dict(moments_bitwise_equal=np.array_equal(*states),
                       moments_max_abs=float(np.max(np.abs(states[0] - states[1]))), diagnostics={})
    for table in ('timeseries', 'profiles', 'convergence'):
        with h5py.File(before / 'statistics.h5') as a, h5py.File(after / 'statistics.h5') as b:
            x, y = a[f'statistics/{table}'][:], b[f'statistics/{table}'][:]
        if x.shape != y.shape or not np.array_equal(np.isfinite(x), np.isfinite(y)):
            raise RuntimeError(f'diagnostic shape/finite-value mismatch: {table}')
        finite = np.isfinite(x) & np.isfinite(y)
        differences['diagnostics'][table] = dict(
            max_abs=float(np.max(np.abs(x[finite] - y[finite]), initial=0)),
            agrees=np.allclose(x, y, rtol=1.e-11, atol=1.e-12, equal_nan=True))
    # The moment comparison is exact. Diagnostics use a roundoff allowance
    # because warp reduction reassociates FP64 sums; this is not a DNS criterion.
    if not differences['moments_bitwise_equal'] or not all(t['agrees'] for t in differences['diagnostics'].values()):
        raise RuntimeError(f'numerical regression: {differences}')
    return differences


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--before', type=Path, help='untouched source tree containing src/ and cases/; enables comparison')
    parser.add_argument('--before-fused', action='store_true', help='baseline already includes fused NEE')
    parser.add_argument('--work', type=Path, help='new or empty artifact directory')
    parser.add_argument('--grids', type=int, nargs='+', choices=(32, 64, 96, 128), default=[64, 128])
    parser.add_argument('--steps', type=int, default=200)
    parser.add_argument('--sample', type=int, default=100)
    parser.add_argument('--field-every', type=int, default=0, help='short export comparison; 0 disables snapshots')
    parser.add_argument('--field-io', choices=('async', 'sync', 'serial'), default='async', help='after tree only')
    parser.add_argument('--io-workers', type=int, choices=range(1, 33), default=4)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--arch', default=os.environ.get('CUDA_ARCH', 'sm_86'))
    args = parser.parse_args()
    if not (20 < args.steps <= 1000 and 1 <= args.sample <= args.steps - 20 and 1 <= args.repeats <= 10):
        parser.error('use 21..1000 steps, 1..steps-20 sampling, and 1..10 repeats')
    if args.field_every < 0:
        parser.error('field-every must be nonnegative')
    work = (args.work or Path(tempfile.mkdtemp(prefix='thermal-rbc-benchmark-'))).resolve()
    if work.exists() and any(work.iterdir()):
        parser.error('--work must be empty; successful results are never overwritten')
    work.mkdir(parents=True, exist_ok=True)
    checkpoint_bytes = (2 if args.before else 1) * args.repeats * sum(14 * n**3 * 4 for n in args.grids)
    required_bytes = checkpoint_bytes + 256 * 2**20
    free_bytes = shutil.disk_usage(work).free
    if free_bytes < required_bytes:
        parser.error(f'benchmark needs at least {required_bytes / 2**30:.2f} GiB free '
                     f'(checkpoints plus build margin); available {free_bytes / 2**30:.2f} GiB; '
                     'choose another --work filesystem or fewer repeats')
    report = dict(grids=args.grids, steps=args.steps, sample=args.sample, repeats=args.repeats,
                  field_every=args.field_every, field_io=args.field_io, io_workers=args.io_workers,
                  arch=args.arch, flags='-O3 --restrict -lineinfo --resource-usage', results={})
    gpu = subprocess.run(['nvidia-smi', '--query-gpu=name,driver_version,memory.total,clocks.current.sm,power.limit', '--format=csv'],
                         text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    report['gpu'] = gpu.stdout.strip()
    trees = {}
    for label, source in ([('before', args.before.resolve())] if args.before else []) + [('after', ROOT)]:
        destination = work / f'source-{label}'
        for folder in ('src', 'cases'):
            shutil.copytree(source / folder, destination / folder)
        (destination / 'tests').mkdir()
        harness = (ROOT / 'tests/benchmark_rbc.cu').read_text()
        if label == 'before' and not args.before_fused:
            harness = harness.replace('rbc::PLANE_BLOCKS', 'rbc::PROFILE_SIZE')
        (destination / 'tests/benchmark_rbc.cu').write_text(harness)
        trees[label] = destination
    for n in args.grids:
        results = report['results'][str(n)] = {}
        binaries = {}
        for label, tree in trees.items():
            results[label] = dict(solver_elapsed_s=[], solver_mlups=[], solver_compute_mlups=[], solver_output_s=[])
            common = ['nvcc', '-std=c++20', '-O3', '--restrict', '--expt-relaxed-constexpr', '--extended-lambda',
                      f'-arch={args.arch}', '-lineinfo', '--resource-usage', '-DCASE_RAYLEIGH_BENARD',
                      f'-DRBC_N={n}', '-DRBC_TRANSIENT_STEPS=20', f'-DRBC_SAMPLE={args.sample}']
            executables = {}
            for name, source in [('solver', 'src/main.cu'), ('bench', 'tests/benchmark_rbc.cu')]:
                executable = work / f'{label}-{name}-{n}.bin'
                flags = ['-DRBC_BENCH_SPLIT_NEE'] if name == 'bench' and label == 'before' and not args.before_fused else []
                execute(common + flags + [tree / source, '-o', executable] + hdf5_flags(), ROOT,
                        work / f'{label}-{name}-{n}-build.log')
                executables[name] = executable
            binaries[label] = executables
        for label, executables in binaries.items():
            folder = work / f'{label}-bench-{n}'
            folder.mkdir()
            results[label]['microbenchmark'] = measurements(execute([executables['bench'], args.steps], folder, folder / 'benchmark.log'))
        # Alternate the order of matched runs to reduce clock/thermal and
        # filesystem-cache bias; the desktop GPU still has uncontrolled users.
        for repeat in range(args.repeats):
            labels = list(binaries)
            if repeat % 2:
                labels.reverse()
            for label in labels:
                executables = binaries[label]
                folder = work / f'{label}-run-{n}-{repeat}'
                folder.mkdir()
                command = [executables['solver'], '--runId', 'deterministic', '--steps', args.steps,
                           '--checkpoint-every', args.steps]
                if args.field_every:
                    command += ['--field-every', args.field_every]
                    if label == 'after':
                        command += ['--field-io', args.field_io, '--io-workers', args.io_workers]
                text = execute(command, folder, folder / 'solver.log')
                match = re.search(r'elapsed ([\d.eE+-]+) s; ([\d.eE+-]+) MLUPS', text)
                if not match:
                    raise RuntimeError('solver timing line missing')
                results[label]['solver_elapsed_s'].append(float(match[1]))
                results[label]['solver_mlups'].append(float(match[2]))
                with h5py.File(folder/'output/rayleigh_benard/deterministic/statistics.h5') as source:
                    runtime = source['statistics/runtime'][-1]
                    results[label]['solver_compute_mlups'].append(float(runtime[10]))
                    results[label]['solver_output_s'].append(float(runtime[8]))
        for label in binaries:
            results[label]['median_solver_mlups'] = statistics.median(results[label]['solver_mlups'])
        if args.before:
            results['equivalence'] = compare(
                work / f'before-run-{n}-0/output/rayleigh_benard/deterministic',
                work / f'after-run-{n}-0/output/rayleigh_benard/deterministic', n, args.steps)
            results['end_to_end_speedup'] = results['after']['median_solver_mlups'] / results['before']['median_solver_mlups']
        (work / 'report.json').write_text(json.dumps(report, indent=2, default=lambda value: bool(value)) + '\n')
    print(json.dumps(report, indent=2, default=lambda value: bool(value)))


if __name__ == '__main__':
    main()
