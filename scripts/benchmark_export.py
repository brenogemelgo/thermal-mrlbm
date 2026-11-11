#!/usr/bin/env python3
"""Export microbenchmarks on deterministic changing fields; never a production simulation."""
import argparse
import json
import os
from pathlib import Path
import re
import statistics
import subprocess
import tempfile

import h5py
import numpy as np

from build_convert_vti import hdf5_flags

ROOT = Path(__file__).resolve().parents[1]


def execute(command, log):
    result = subprocess.run(list(map(str, command)), cwd=ROOT, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    log.write_text(result.stdout)
    if result.returncode:
        raise RuntimeError(f'exit {result.returncode}; see {log}')
    return result.stdout


def compare(reference, candidate):
    result = dict(temperature_bitwise_equal=True, raw_chunks_equal=True, metadata_equal_except_run_id=True)
    for path in reference.glob('fields/*.h5'):
        with h5py.File(path) as left, h5py.File(candidate/'fields'/path.name) as right:
            a, b = left['fields/temperature'], right['fields/temperature']
            # Compare bits plane by plane, including signed zero and NaN payloads.
            for z in range(a.shape[0]):
                if a[z].tobytes() != b[z].tobytes():
                    raise AssertionError(f'temperature changed: {path.name}, z={z}')
            assert a.dtype == b.dtype and a.shape == b.shape
            assert a.compression == b.compression and a.compression_opts == b.compression_opts
            assert a.shuffle == b.shuffle and a.fletcher32 == b.fletcher32
            assert {k: v for k, v in left.attrs.items() if k != 'run_id'} == {k: v for k, v in right.attrs.items() if k != 'run_id'}
            if a.chunks == b.chunks:
                for offset in np.ndindex(*(int(np.ceil(n/c)) for n, c in zip(a.shape, a.chunks))):
                    coordinate = tuple(i*c for i, c in zip(offset, a.chunks))
                    if a.id.read_direct_chunk(coordinate) != b.id.read_direct_chunk(coordinate):
                        result['raw_chunks_equal'] = False
            else:
                result['raw_chunks_equal'] = None
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--grids', nargs='+', type=int, choices=(64, 67, 128, 256, 512), default=[128])
    parser.add_argument('--snapshots', type=int, default=3)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--workers', nargs='+', type=int, default=[1, 4])
    parser.add_argument('--compression', nargs='+', type=int, choices=range(10), default=[1])
    parser.add_argument('--chunks', nargs='+', default=['64x128'], help='benchmark alternatives; production defaults stay 64x128')
    parser.add_argument('--arch', default=os.environ.get('CUDA_ARCH', 'sm_86'))
    parser.add_argument('--work', type=Path)
    args = parser.parse_args()
    if not 1 <= args.snapshots <= 5 or not 1 <= args.repeats <= 5 or any(not 1 <= n <= 32 for n in args.workers):
        parser.error('use 1..5 snapshots/repeats and 1..32 workers')
    try:
        chunks = [tuple(map(int, shape.lower().split('x'))) for shape in args.chunks]
        if any(len(c) != 2 or min(c) < 1 or max(c) > 512 for c in chunks):
            raise ValueError()
    except ValueError:
        parser.error('chunks must be YxX with dimensions 1..512')
    work = (args.work or Path(tempfile.mkdtemp(prefix='thermal-export-benchmark-'))).resolve()
    if work.exists() and any(work.iterdir()):
        parser.error('--work must be new or empty')
    work.mkdir(parents=True, exist_ok=True)
    report = dict(work=str(work), measurements=[], notes=[
        'Synthetic changing temperature/velocity fields, not turbulent DNS.',
        'original4 is the former four-field plane pipeline; original1 isolates the same pipeline for temperature only.',
        'sync and async use identical temperature values, filters and compression levels.',
        'HDF5 write/flush timers measure OS-buffered IO, not power-loss durability or physical drive completion.',
        'D2H GPU timings aggregate copy-engine duration; capture wall time also includes readiness/API costs.',
        'Back-to-back snapshots stress backpressure; native short-solver comparisons measure useful compute overlap.'])
    for n in args.grids:
        binary = work/f'export-n{n}'
        execute(['nvcc', '-std=c++20', '-O3', '--restrict', '--expt-relaxed-constexpr', '--extended-lambda',
                 f'-arch={args.arch}', '-DCASE_RAYLEIGH_BENARD', f'-DRBC_N={n}', ROOT/'tests/benchmark_export.cu',
                 '-o', binary, *hdf5_flags()], work/f'build-n{n}.log')
        for level in args.compression:
            for repeat in range(args.repeats):
                variants = [('original4', 1, (64, 128)), ('original1', 1, (64, 128))]
                variants += [(mode, workers, chunk) for mode in ('sync', 'async') for workers in args.workers for chunk in chunks]
                # The reference is retained for exact comparisons in every trial.
                reference = None
                for mode, workers, (cy, cx) in variants:
                    tag = f'n{n}-c{level}-r{repeat}-{mode}-w{workers}-y{cy}-x{cx}'
                    root = work/tag
                    text = execute([binary, root, mode, args.snapshots, level, workers, cy, cx], work/f'{tag}.log')
                    run = root/'rayleigh_benard'/mode
                    row = dict(grid=n, compression=level, repeat=repeat, mode=mode, workers=workers,
                               chunks=[1, min(cy, n), min(cx, n)], run=str(run))
                    row['total_s'] = float(re.search(r'completed total_s=([\d.eE+-]+)', text)[1])
                    row['submission_s'] = [float(v) for v in re.findall(r'submission step=\d+ seconds=([\d.eE+-]+)', text)]
                    if mode.startswith('original'):
                        row['stages'] = [{k: float(v) for k, v in re.findall(r'(\w+)=([\d.eE+-]+)', line)}
                                         for line in text.splitlines() if line.startswith('original ')]
                    else:
                        with h5py.File(run/'statistics.h5') as f:
                            data = f['statistics/exports'];columns = data.attrs['columns']
                            if isinstance(columns, bytes):
                                columns = columns.decode()
                            row['stages'] = [dict(zip(columns.split(','), map(float, values))) for values in data[:]]
                    payload = n**3*4*(4 if mode == 'original4' else 1)*args.snapshots
                    row['payload_MiB_s'] = payload/2**20/row['total_s']
                    row['median_snapshot_s'] = statistics.median(row['submission_s'])
                    if mode == 'original1':
                        reference = run
                    elif reference is not None:
                        row['equivalence'] = compare(reference, run)
                    report['measurements'].append(row)
                    (work/'report.json').write_text(json.dumps(report, indent=2)+'\n')
                    print(tag, f"{row['total_s']:.3f}s", flush=True)
    print(work/'report.json')


if __name__ == '__main__':
    main()
