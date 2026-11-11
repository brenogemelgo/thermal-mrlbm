#!/usr/bin/env python3
"""Explicit, non-destructive migration of legacy reduced diagnostics for restart.

Existing binary checkpoints are unchanged. Historical field binaries/VTI remain
readable through postCommon; new snapshots use HDF5. Never run during a solver.
"""
import argparse
from pathlib import Path
import sys
import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.rbc_hdf5 import read_metadata, data_directory, TABLES


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    args = parser.parse_args()
    run = data_directory(args.run.resolve())
    for filename in ('statistics.h5', 'metadata.h5'):
        if (run/filename).exists():
            parser.error(f'{run/filename} already exists; refusing to overwrite')
    meta = read_metadata(run)
    temporary = run/'statistics.h5.tmp'
    with h5py.File(temporary, 'w') as out:
        out.attrs['schema'] = 'thermal-mrlbm-hdf5-v1'
        group = out.create_group('statistics')
        for legacy, name in TABLES.items():
            path = run/legacy
            if name == 'runtime' or not path.exists():
                continue
            data = np.atleast_1d(np.genfromtxt(path, delimiter=',', names=True))
            names = data.dtype.names
            matrix = np.empty((len(data), len(names)))
            for i, field in enumerate(names):
                if field == 'estimator':
                    # Read this string-valued legacy column independently.
                    raw = np.atleast_1d(np.genfromtxt(path, delimiter=',', names=True, dtype=None, encoding='utf-8'))
                    matrix[:, i] = [['bottom', 'top', 'volume', 'kinetic', 'thermal'].index(s) for s in raw[field]]
                else:
                    matrix[:, i] = data[field]
            target = group.create_dataset(name, data=matrix, maxshape=(None, len(names)),
                                          chunks=(128, len(names)), compression='gzip', compression_opts=1,
                                          shuffle=True, fletcher32=True)
            target.attrs['columns'] = ','.join(names)
        if 'timeseries' not in group:
            raise ValueError('legacy time series is required for restart migration')
        group.attrs['estimator_order'] = 'bottom,top,volume,kinetic,thermal'
    temporary.replace(run/'statistics.h5')
    temporary = run/'metadata.h5.tmp'
    with h5py.File(temporary, 'w') as out:
        group = out.create_group('metadata')
        for k, v in meta.items():
            group.attrs[k] = str(v).lower() if isinstance(v, bool) else str(v)
    temporary.replace(run/'metadata.h5')
    print(f'Migrated reduced diagnostics in {run}; all legacy files and checkpoints preserved.')


if __name__ == '__main__':
    main()
