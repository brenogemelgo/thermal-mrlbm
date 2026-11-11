"""HDF5 schema readers shared by the plotters; legacy output is read-only compatible."""
from pathlib import Path
import h5py
import numpy as np


TABLES = {'rbc_timeseries.csv': 'timeseries', 'rbc_profiles.csv': 'profiles',
          'rbc_convergence.csv': 'convergence', 'runtime': 'runtime'}


def data_directory(run):
    """Accept either the local run directory or its separate HDF5 directory."""
    run = Path(run)
    link = run/'hdf5'
    if link.is_symlink() and not link.exists():
        raise FileNotFoundError(f'HDF5 storage is unavailable: {link}')
    return link if link.is_dir() else run


def text(value):
    return value.decode() if isinstance(value, bytes) else str(value)


def value(raw):
    raw = text(raw)
    if raw in ('true', 'false'):
        return raw == 'true'
    try:
        return float(raw)
    except ValueError:
        return raw


def read_metadata(run):
    run = data_directory(run)
    if (run/'metadata.h5').exists():
        with h5py.File(run/'metadata.h5', 'r') as source:
            return {k: value(v) for k, v in source['metadata'].attrs.items()}
    result = {}
    for line in (run/'metadata.txt').read_text().splitlines():
        if '=' in line:
            k, v = line.split('=', 1)
            result[k.strip()] = value(v.strip())
    return result


def read_table(run, name):
    run = data_directory(run)
    if (run/'statistics.h5').exists():
        with h5py.File(run/'statistics.h5', 'r') as source:
            data = source['statistics/'+TABLES.get(name, name)]
            columns = text(data.attrs['columns']).split(',')
            result = np.empty(data.shape[0], dtype=[(c, np.float64) for c in columns])
            values = data[:]
            for i, column in enumerate(columns):
                result[column] = values[:, i]
            return result
    return np.atleast_1d(np.genfromtxt(run/name, delimiter=',', names=True))


def processed_data(run, destination, result):
    """Export useful reduced tables and uncertainty to a separate derived artifact."""
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    target = destination/'diagnostics.h5'
    temp = target.with_suffix('.h5.tmp')
    with h5py.File(temp, 'w') as output:
        output.attrs['run_id'] = Path(run).name
        output.attrs['dns_convergence_established'] = False
        output.attrs['schema'] = 'thermal-mrlbm-derived-v1'
        meta = output.create_group('metadata')
        for k, v in read_metadata(run).items():
            meta.attrs[k] = v
        for name in ('rbc_timeseries.csv', 'rbc_profiles.csv'):
            data = read_table(run, name)
            dataset = output.create_dataset(TABLES[name], data=data, compression='gzip', shuffle=True)
            dataset.attrs['origin'] = 'raw reduced diagnostics; no full fields loaded'
        if result:
            columns = ['mean', 'stderr', 'effective_samples', 'tau_samples', 'ci95_lower', 'ci95_upper']
            names = list(result['estimators'])
            rows = [[result['estimators'][n].get(k) for k in columns] for n in names]
            data = output.create_dataset('nusselt_uncertainty', data=np.array(rows, dtype=float))
            data.attrs['columns'] = ','.join(columns)
            data.attrs['estimators'] = ','.join(names)
    temp.replace(target)
