#!/usr/bin/env python3
"""Short real-GPU scaling benchmark, including coupled statistics and final checkpoint IO."""
import argparse
import csv
import json
import os
from pathlib import Path
import re
import shutil
import statistics
import subprocess
import tempfile

from build_convert_vti import hdf5_flags

ROOT = Path(__file__).resolve().parents[1]


def execute(command, log):
    result = subprocess.run(list(map(str, command)), cwd=ROOT, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    log.write_text(result.stdout)
    if result.returncode:
        raise RuntimeError(f'exit {result.returncode}; see {log}')
    return result.stdout


def timing(output):
    match = re.search(r'elapsed ([\d.eE+-]+) s; ([\d.eE+-]+) MLUPS', output)
    if not match:
        raise RuntimeError('missing completed solver timing')
    values = dict(elapsed_s=float(match[1]), sustained_mlups=float(match[2]))
    progress = [line for line in output.splitlines() if 'compute_MLUPS=' in line]
    if not progress:
        raise RuntimeError('missing computation/output timing')
    pairs = dict(re.findall(r'(\w+)=([-+\d.eE]+)', progress[-1]))
    values.update(compute_mlups=float(pairs['compute_MLUPS']), output_s=float(pairs['output']))
    return values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--grid', type=int, choices=(32, 64, 96, 128, 192, 256, 384, 512), default=128)
    parser.add_argument('--steps', type=int, default=300)
    parser.add_argument('--sample', type=int, default=100)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--devices', default='0,1')
    parser.add_argument('--halo-transport', choices=('peer', 'host'), default='peer')
    parser.add_argument('--split-z', type=int)
    parser.add_argument('--only-two', action='store_true', help='omit single-GPU baseline when it cannot fit')
    parser.add_argument('--arch', default=os.environ.get('CUDA_ARCH', 'sm_86'))
    parser.add_argument('--work', type=Path)
    args = parser.parse_args()
    if not (20 < args.steps <= 1000 and 1 <= args.sample <= args.steps - 20 and 1 <= args.repeats <= 10):
        parser.error('use 21..1000 steps, 1..steps-20 sample interval, and 1..10 repeats')
    try:
        devices = [int(value) for value in args.devices.split(',')]
    except ValueError:
        parser.error('--devices requires two integer CUDA ordinals')
    if len(devices) != 2 or len(set(devices)) != 2 or min(devices) < 0:
        parser.error('--devices requires two distinct nonnegative ordinals')
    if args.split_z is not None and not 2 <= args.split_z <= args.grid - 2:
        parser.error('--split-z must leave at least two planes in each slab')
    work = (args.work or Path(tempfile.mkdtemp(prefix='thermal-two-gpu-benchmark-'))).resolve()
    if work.exists() and any(work.iterdir()):
        parser.error('--work must be new or empty')
    work.mkdir(parents=True, exist_ok=True)
    report = dict(status='not_run', grid=args.grid, steps=args.steps, sample=args.sample,
                  repeats=args.repeats, devices=devices, halo_transport=args.halo_transport,
                  split_z=args.split_z or args.grid // 2, measurements=[])

    def save():
        (work / 'report.json').write_text(json.dumps(report, indent=2) + '\n')

    binary = work / 'solver'
    defines = ['-DCASE_RAYLEIGH_BENARD', f'-DRBC_N={args.grid}', '-DREGULARIZATION_ORDER=3',
               '-DRBC_TRANSIENT_STEPS=20', f'-DRBC_SAMPLE={args.sample}']
    if args.split_z is not None:
        defines += [f'-DMRL_SPLIT_Z={args.split_z}']
    execute(['nvcc', '-std=c++20', '-O3', '--restrict', '--expt-relaxed-constexpr', '--extended-lambda',
             f'-arch={args.arch}', '-lineinfo', '--resource-usage', *defines,
             ROOT / 'src/main.cu', '-o', binary, *hdf5_flags()], work / 'build.log')
    info = json.loads(execute([binary, '--gpu-info'], work / 'gpu-info.json'))
    report['gpu_info'] = info
    if max(devices) >= len(info['devices']):
        report['status'] = 'SKIP: two requested physical GPUs are not visible; no emulated performance result'
        save()
        print(json.dumps(report, indent=2))
        return
    if args.halo_transport == 'peer' and not all(info['peer_access'][a][b] for a, b in (devices, devices[::-1])):
        report['status'] = 'FAIL: requested pair does not support bidirectional CUDA P2P'
        save()
        raise RuntimeError(report['status'])
    modes = [2] if args.only_two else [1, 2]
    required = len(modes) * args.repeats * 14 * args.grid**3 * 4 + 256 * 2**20
    if shutil.disk_usage(work).free < required:
        parser.error(f'insufficient free space; need {required / 2**30:.2f} GiB for retained checkpoints/build margin')
    # Sequential matched runs, alternating order to limit desktop clock drift.
    # No 3D fields or video; every mode includes the same GPU statistics and IO.
    try:
        for repeat in range(args.repeats):
            for gpus in modes if repeat % 2 == 0 else modes[::-1]:
                rid = f'n{args.grid}-g{gpus}-r{repeat}'
                selected = args.devices if gpus == 2 else str(devices[0])
                output = execute([binary, '--runId', rid, '--steps', args.steps,
                                  '--gpus', gpus, '--devices', selected, '--halo-transport', args.halo_transport,
                                  '--checkpoint-every', args.steps, '--compression', 0,
                                  '--output-root', work / 'output'], work / f'{rid}.log')
                row = dict(gpus=gpus, repeat=repeat, **timing(output))
                report['measurements'].append(row)
                save()
        report['medians'] = {str(g): {key: statistics.median(row[key] for row in report['measurements'] if row['gpus'] == g)
                                      for key in ('elapsed_s', 'sustained_mlups', 'compute_mlups', 'output_s')} for g in modes}
        if 1 in modes:
            report['sustained_speedup'] = report['medians']['2']['sustained_mlups'] / report['medians']['1']['sustained_mlups']
            report['compute_speedup'] = report['medians']['2']['compute_mlups'] / report['medians']['1']['compute_mlups']
        report['status'] = 'completed short benchmark; no production/DNS claim'
    except Exception as error:
        report['status'] = f'FAIL: {error}'
        raise
    finally:
        save()
        if report['measurements']:
            with (work / 'timings.csv').open('w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(report['measurements'][0]))
                writer.writeheader()
                writer.writerows(report['measurements'])
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
