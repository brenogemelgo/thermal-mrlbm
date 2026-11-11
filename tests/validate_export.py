#!/usr/bin/env python3
"""Focused field IO correctness, raw filter compatibility, restart and converter checks."""
import argparse
import json
import os
from pathlib import Path
import signal
import struct
import subprocess
import sys
import tempfile
import time

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
from benchmark_export import compare, execute
from build_convert_vti import build_converter, hdf5_flags


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--before', type=Path, required=True, help='saved original src/ and cases/ tree')
    parser.add_argument('--work', type=Path)
    parser.add_argument('--arch', default=os.environ.get('CUDA_ARCH', 'sm_86'))
    args = parser.parse_args()
    work = (args.work or Path(tempfile.mkdtemp(prefix='thermal-export-validation-'))).resolve()
    if work.exists() and any(work.iterdir()):
        parser.error('--work must be new or empty')
    work.mkdir(parents=True, exist_ok=True)
    report = dict(work=str(work), checks={})
    common = ['nvcc', '-std=c++20', '-O3', '--restrict', '--expt-relaxed-constexpr', '--extended-lambda',
              f'-arch={args.arch}', '-DCASE_RAYLEIGH_BENARD', '-DRBC_N=16', '-DRBC_STEPS=60',
              '-DRBC_TRANSIENT_STEPS=10', '-DRBC_SAMPLE=5']
    binaries = {}
    for label, tree in [('before', args.before.resolve()), ('after', ROOT)]:
        binary = work/label
        execute([*common, tree/'src/main.cu', '-o', binary, *hdf5_flags()], work/f'{label}-build.log')
        binaries[label] = binary

    def run(label, rid, steps=60, resume=False, mode=None):
        command = [binaries[label], '--runId', rid, '--steps', steps, '--checkpoint-every', 30,
                   '--field-every', 10, '--slice-every', 15, '--slice-velocity', '--compression', 1,
                   '--output-root', work/'raw', '--hdf5-root', work/'hdf5']
        if mode:
            command += ['--field-io', mode]
        if resume:
            command += ['--continue']
        execute(command, work/f'{rid}-{steps}.log')
        return work/'raw/rayleigh_benard'/rid

    def check_state(reference, candidate):
        a, b = [(p/'checkpoints/step_000000060.chk').read_bytes() for p in (reference, candidate)]
        offset = struct.calcsize('@8s6I3Q9d')
        np.testing.assert_array_equal(np.frombuffer(a, '<f4', 14*16**3, offset), np.frombuffer(b, '<f4', 14*16**3, offset))
        for name in ('timeseries', 'profiles', 'convergence'):
            with h5py.File(reference/'hdf5/statistics.h5') as left, h5py.File(candidate/'hdf5/statistics.h5') as right:
                np.testing.assert_array_equal(left[f'statistics/{name}'][:], right[f'statistics/{name}'][:])
        result = compare(reference/'hdf5', candidate/'hdf5')
        assert result['raw_chunks_equal']
        for path in (reference/'hdf5/slices').glob('*.h5'):
            with h5py.File(path) as left, h5py.File(candidate/'hdf5/slices'/path.name) as right:
                for key in left['fields']:
                    np.testing.assert_array_equal(left[f'fields/{key}'][:], right[f'fields/{key}'][:])
        return result

    reference = run('before', 'reference')
    for mode in ('serial', 'sync', 'async'):
        candidate = run('after', mode, mode=mode)
        report['checks'][mode] = check_state(reference, candidate)
        with h5py.File(candidate/'hdf5/fields/step_000000060.h5') as f:
            assert list(f['fields']) == ['temperature']
    for first in ('before', 'after'):
        rid = f'restart-{first}'
        candidate = run(first, rid, 30, mode='async' if first == 'after' else None)
        hashes = {p: p.read_bytes() for p in (candidate/'hdf5/fields').glob('*.h5')}
        (candidate/'hdf5/fields/step_000000040.h5.tmp').write_bytes(b'interrupted write')
        run('after', rid, 60, resume=True, mode='async')
        report['checks'][rid] = check_state(reference, candidate)
        assert all(path.read_bytes() == original for path, original in hashes.items())
        assert not list((candidate/'hdf5/fields').glob('*.tmp'))

    # FP64 odd-sized chunks exercise zero-padded edges and both lossless filter
    # pipelines, as well as changing source fields while a write is outstanding.
    binary = work/'edge-fp64'
    execute(['nvcc', '-std=c++20', '-O3', '--restrict', '--expt-relaxed-constexpr', '--extended-lambda',
             f'-arch={args.arch}', '-DCASE_RAYLEIGH_BENARD', '-DRBC_N=67', '-DMRL_DOUBLE_PRECISION',
             ROOT/'tests/benchmark_export.cu', '-o', binary, *hdf5_flags()], work/'edge-build.log')
    for level in (0, 1, 6):
        runs = {}
        for mode in ('original1', 'sync', 'async'):
            root = work/f'edge-c{level}'
            execute([binary, root, mode, 3, level, 4], work/f'edge-c{level}-{mode}.log')
            runs[mode] = root/'rayleigh_benard'/mode
        for mode in ('sync', 'async'):
            result = compare(runs['original1'], runs[mode])
            assert result['raw_chunks_equal'], result
            report['checks'][f'fp64-edges-c{level}-{mode}'] = result

    converter = build_converter()
    for rid in ('reference', 'async'):
        source = work/'raw/rayleigh_benard'/rid
        execute([converter, source, '--results-root', work/'converted', '--jobs', 2], work/f'convert-{rid}.log')
        for path in (work/'converted'/rid/'vtis').glob('*.vti'):
            raw = path.read_bytes();start = raw.index(b'\n_')+2
            length = struct.unpack_from('<Q', raw, start)[0]
            step = int(path.stem.removeprefix('step_'))
            with h5py.File(source/'hdf5/fields'/f'step_{step:09d}.h5') as f:
                assert raw[start+8:start+8+length] == f['fields/temperature'][:].tobytes()
                if rid == 'reference':
                    offset = start+8+length
                    assert struct.unpack_from('<Q', raw, offset)[0] == 3*length
                    vector = np.stack([f[f'fields/{name}'][:] for name in ('ux', 'uy', 'uz')], axis=-1)
                    assert raw[offset+8:offset+8+3*length] == vector.tobytes()
            assert (b'Name="velocity"' in raw[:start]) == (rid == 'reference')
        report['checks'][f'converter-{rid}'] = 'PASS; temperature bytes preserved'

    (work/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
