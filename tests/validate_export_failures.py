#!/usr/bin/env python3
"""Fault injection for the short export harness; does not run a solver."""
import argparse
import json
from pathlib import Path
import signal
import subprocess
import tempfile
import time

import h5py


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('binary', type=Path, help='tests/benchmark_export.cu built at N>=128')
    parser.add_argument('--work', type=Path)
    args = parser.parse_args()
    work = (args.work or Path(tempfile.mkdtemp(prefix='thermal-export-failures-'))).resolve()
    if work.exists() and any(work.iterdir()):
        parser.error('--work must be new or empty')
    work.mkdir(parents=True, exist_ok=True)
    report = {}
    for test in ('write-error', 'worker-death', 'parent-death'):
        root = work/test
        fields = root/'rayleigh_benard/async/fields'
        fields.mkdir(parents=True)
        if test == 'write-error':
            (fields/'step_000000001.h5.tmp').mkdir()
        with (work/f'{test}.log').open('w') as log:
            process = subprocess.Popen([str(args.binary.resolve()), str(root), 'async', '4', '1', '4'], stdout=log, stderr=subprocess.STDOUT)
            worker = None
            try:
                if test != 'write-error':
                    step = 0 if test == 'parent-death' else 1
                    temporary = fields/f'step_{step:09d}.h5.tmp'
                    deadline = time.monotonic()+15
                    while time.monotonic() < deadline:
                        children = Path(f'/proc/{process.pid}/task/{process.pid}/children')
                        if temporary.exists() and children.exists() and children.read_text().split():
                            worker = int(children.read_text().split()[0])
                            break
                        if process.poll() is not None:
                            raise AssertionError('export completed before fault injection')
                        time.sleep(.001)
                    assert worker is not None, 'writer did not start'
                    target = process.pid if test == 'parent-death' else worker
                    import os
                    os.kill(target, signal.SIGKILL)
                code = process.wait(timeout=15)
                assert code != 0, 'a failed writer must fail the parent'
                if worker is not None:
                    deadline = time.monotonic()+5
                    while time.monotonic() < deadline:
                        status = Path(f'/proc/{worker}/status')
                        if not status.exists() or 'State:\tZ' in status.read_text():
                            break
                        time.sleep(.01)
                    else:
                        raise AssertionError('orphan writer still running')
                if test != 'parent-death':
                    assert (fields/'step_000000000.h5').exists(), 'completed snapshot was lost'
                    assert not (fields/'step_000000001.h5').exists(), 'incomplete snapshot was published'
                for path in fields.glob('*.h5'):
                    with h5py.File(path) as source:
                        data = source['fields/temperature']
                        for z in range(data.shape[0]):
                            data[z]  # Verify HDF5 decoding and checksums.
                report[test] = 'PASS'
            finally:
                if process.poll() is None:
                    process.kill();process.wait()
    (work/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
