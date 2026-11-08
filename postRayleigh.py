#!/usr/bin/env python3
"""Plot Rayleigh–Bénard diagnostics and compare RBC runs.

Examples:
    python3 postRayleigh.py all output/rayleigh_benard/short
    python3 postRayleigh.py convergence output/rayleigh_benard/RUN_ID
    python3 postRayleigh.py grid output/rayleigh_benard/rbc4_n*_rr3
    python3 postRayleigh.py validation-summary output/rbc_validation --out output/rbc_validation/summary

Reads reduced CSV statistics and metadata. Writes PNG/PDF figures and JSON
reports without loading full simulation fields or asserting DNS convergence.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np

NUS = ['nu_bottom', 'nu_top', 'nu_volume', 'nu_kinetic', 'nu_thermal']
LABELS = ['Lower plate', 'Upper plate', 'Volume transport', 'Kinetic dissipation', 'Thermal dissipation']
NIEMELA = 'https://doi.org/10.1038/35009036'
KOOIJ = 'https://doi.org/10.1016/j.compfluid.2018.01.010'
plt.rcParams.update({'font.family': 'serif', 'font.size': 10, 'axes.labelsize': 11,
                     'savefig.dpi': 300, 'axes.spines.top': False, 'axes.spines.right': False})


def metadata(run):
    result = {}
    for line in (Path(run) / 'metadata.txt').read_text().splitlines():
        if '=' not in line:
            continue
        k, v = (s.strip() for s in line.split('=', 1))
        try:
            result[k] = float(v)
        except ValueError:
            result[k] = v
    if result.get('moment_format') != 'neq_post_v2':
        raise ValueError('expected version-2 non-equilibrium moment metadata')
    return result


def load(run, name):
    data = np.atleast_1d(np.genfromtxt(Path(run) / name, delimiter=',', names=True))
    if not len(data):
        raise ValueError(f'no samples in {name}')
    return data


def block_errors(x):
    """Nonoverlapping batch means at multiple sizes; plateau needs inspection."""
    x = np.asarray(x, dtype=float)
    result = []
    size = 1
    while len(x) // size >= 8:
        n = len(x) // size
        blocks = x[:n*size].reshape(n, size).mean(axis=1)
        result.append({'block_samples': size, 'blocks': n,
                       'stderr': float(blocks.std(ddof=1)/np.sqrt(n))})
        size *= 2
    return result


def uncertainty(x):
    x = np.asarray(x, dtype=float)
    if len(x) < 8:
        return {'mean': float(x.mean()), 'samples': len(x), 'stderr': None,
                'effective_samples': None, 'tau_samples': None, 'blocks': []}
    fluct = x-x.mean()
    if np.all(fluct == 0):
        tau = 1.
    else:
        nfft = 1 << (2*len(x)-1).bit_length()
        spectrum = np.fft.rfft(fluct, nfft)
        ac = np.fft.irfft(spectrum*spectrum.conj(), nfft)[:len(x)]
        ac /= np.arange(len(x), 0, -1)
        ac /= ac[0]
        stop = min(len(ac)-1, len(x)//2)
        positive = []
        for i in range(1, stop, 2):
            pair = ac[i]+ac[i+1]
            if pair <= 0:
                break
            positive.append(pair)
        tau = max(1., 1+2*sum(positive))
    neff = max(1., len(x)/tau)
    blocks = block_errors(x)
    err = float(x.std(ddof=1)/np.sqrt(neff))
    # Conservative against the batch-mean ladder; no assumed plateau.
    err = max([err]+[b['stderr'] for b in blocks])
    half = len(x)//2
    return {'mean': float(x.mean()), 'samples': len(x), 'stderr': err,
            'effective_samples': neff, 'tau_samples': tau, 'blocks': blocks,
            'half_mean_relative_difference': float((x[half:].mean()-x[:half].mean())/x.mean())}


def save(fig, out, stem):
    out.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    for ext in ['pdf', 'png']:
        fig.savefig(out / f'{stem}.{ext}', bbox_inches='tight')
    plt.close(fig)


def clean_json(obj):
    if isinstance(obj, dict):
        return {k: clean_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [clean_json(v) for v in obj]
    if isinstance(obj, (float, np.floating)) and not np.isfinite(obj):
        return None
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def report(run, out=None):
    run = Path(run)
    m = metadata(run)
    series = load(run, 'rbc_timeseries.csv')
    avg = series[series['averaging'] == 1]
    if not len(avg):
        raise ValueError(f'{run}: no post-transient samples')
    p = load(run, 'rbc_profiles.csv')
    stats = {name: uncertainty(avg[name]) for name in NUS}
    means = np.array([stats[n]['mean'] for n in NUS])
    nu = stats['nu_volume']['mean']
    h = m['H']
    nb, nt = means[:2]
    result = dict(run=str(run), Ra=m['RA'], Pr=m['PR'], grid=int(m['NZ']), rr=int(m['REGULARIZATION_ORDER']),
                  estimators=stats, estimator_relative_spread=float(np.ptp(means)/abs(means.mean())),
                  nu_z_relative_range=float(np.ptp(p['nu_z'])/abs(p['nu_z'].mean())),
                  lower_boundary_layer_cells=h/(2*nb), upper_boundary_layer_cells=h/(2*nt),
                  lower_boundary_layer_nodes=int(np.floor(h/(2*nb)))+1 if nb>0 else 0,
                  upper_boundary_layer_nodes=int(np.floor(h/(2*nt)))+1 if nt>0 else 0,
                  max_delta_over_eta=float(np.max(p['delta_over_eta_min'])),
                  averaging_duration_ff=float((avg['step'][-1]-m['timeAverageStart'])/m['T_FREEFALL']),
                  target_averaging_duration_ff=m['AVERAGING_FF'],
                  completed_step=int(series['step'][-1]),
                  dns_convergence_established=False,
                  uncertainty_note='Autocorrelation and batch estimates require stationarity and a block-error plateau; short records cannot establish convergence.')
    if m['RBC_CASE'] in [1, 2, 3] and m['RA'] > 0:
        reference = .124*m['RA']**.309
        result['reference'] = dict(Nu=reference, relative_difference=nu/reference-1, source=NIEMELA,
                                   geometry_note='Experimental helium cylinder, aspect ratio 0.5; simulation has periodic horizontal boundaries and aspect ratio NX/(NZ-1). This is a scaling comparison, not matched geometry.')
    elif m['RBC_CASE'] == 4 and m['RA'] == 1e8 and m['PR'] == 1:
        result['reference'] = dict(Nu=32.32, uncertainty=.24, relative_difference=nu/32.32-1, source=KOOIJ,
                                   geometry_note='Kooij et al. (2018), Table 1, periodic domain, Gamma=1. The on-node cubic lattice has Gamma=NX/(NZ-1), slightly above one. Published +/-0.24 is a code/estimator comparison interval, not this run\'s sampling error.')
    if out is not None:
        out = Path(out)
        out.mkdir(parents=True, exist_ok=True)
        (out/'report.json').write_text(json.dumps(clean_json(result), indent=2, allow_nan=False)+'\n')
    return result


def convergence(run, out):
    data = load(run, 'rbc_timeseries.csv')
    avg = data[data['averaging'] == 1]
    r = report(run, out)
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
    for name, label in zip(NUS, LABELS):
        ax[0].plot(data['t_ff'], data[name], alpha=.55, linewidth=.7)
        ax[1].plot(avg['t_ff'], np.cumsum(avg[name])/np.arange(1, len(avg)+1), label=label)
    ax[0].set(xlabel=r'$t/t_{ff}$', ylabel=r'Instantaneous $Nu$')
    ax[1].set(xlabel=r'$t/t_{ff}$', ylabel=r'Cumulative mean $Nu$')
    ax[1].legend(fontsize=8)
    save(fig, out, 'nusselt_convergence')
    fig, ax = plt.subplots(figsize=(5, 3.4))
    for name, label in zip(NUS, LABELS):
        blocks = r['estimators'][name]['blocks']
        if blocks:
            ax.plot([b['block_samples'] for b in blocks], [b['stderr'] for b in blocks], 'o-', label=label)
    ax.set(xscale='log', xlabel='Samples per block', ylabel=r'Standard error of mean $Nu$')
    if ax.lines:
        ax.legend(fontsize=8)
    save(fig, out, 'nusselt_block_errors')


def profiles(run, out):
    p = load(run, 'rbc_profiles.csv')
    m = metadata(run)
    z = p['z_over_h']
    fig, ax = plt.subplots(2, 3, figsize=(11, 6.6))
    ax[0, 0].plot(p['temperature'], z)
    ax[0, 0].set(xlabel=r'$\langle T\rangle$', ylabel=r'$z/H$')
    for f, label in zip(['urms', 'vrms', 'wrms'], [r'$u_{rms}$', r'$v_{rms}$', r'$w_{rms}$']):
        ax[0, 1].plot(p[f]/m['U_FREEFALL'], z, label=label)
    ax[0, 1].set(xlabel=r'$u_{rms}/U_{ff}$', ylabel=r'$z/H$'); ax[0, 1].legend()
    ax[0, 2].plot(p['trms'], z); ax[0, 2].set(xlabel=r'$T_{rms}/\Delta T$', ylabel=r'$z/H$')
    for f, label in zip(['epsilon_u', 'epsilon_t'], [r'$\epsilon_u$', r'$\epsilon_T$']):
        norm = m['H']/m['U_FREEFALL']**3 if f == 'epsilon_u' else m['H']/m['U_FREEFALL']
        ax[1, 0].plot(p[f]*norm, z, label=label)
    ax[1, 0].set_xscale('log')
    ax[1, 0].set(xlabel='Dissipation (free-fall units)', ylabel=r'$z/H$'); ax[1, 0].legend()
    for f in ['convective_flux', 'conductive_flux']:
        ax[1, 1].plot(p[f]*m['H']/m['KAPPA'], z, label=f)
    ax[1, 1].plot(p['nu_z'], z, 'k--', label=r'$Nu(z)$')
    ax[1, 1].set(xlabel='Heat flux / conduction flux', ylabel=r'$z/H$'); ax[1, 1].legend(fontsize=8)
    ax[1, 2].plot(p['delta_over_eta_min'], z)
    ax[1, 2].axvline(1, color='k', ls=':')
    ax[1, 2].set(xlabel=r'$\Delta/\eta_{min}(z)$', ylabel=r'$z/H$')
    for axis in ax.flat:
        if axis.get_xscale() == 'linear':
            axis.xaxis.set_major_locator(MaxNLocator(nbins=3))
            axis.ticklabel_format(style='sci', axis='x', scilimits=(-2, 3), useMathText=True)
    save(fig, out, 'rms_dissipation_profiles')


def scaling(runs, out):
    reports = [report(r) for r in runs]
    fig, ax = plt.subplots(figsize=(6, 4))
    ra = np.geomspace(1e6, 1e8, 100)
    ax.loglog(ra, .124*ra**.309, 'k--', label=r'Niemela: $0.124 Ra^{0.309}$ (cylinder)')
    for r in reports:
        if r['Ra'] <= 0:
            continue
        a = r['estimators']['nu_volume']
        ax.errorbar(r['Ra'], a['mean'], yerr=a['stderr'], fmt='o', label=f"Pr={r['Pr']:g}, N={r['grid']}, RR{r['rr']}")
    ax.errorbar(1e8, 32.32, yerr=.24, fmt='s', color='k', label='Kooij: periodic, Pr=1')
    ax.set(xlabel=r'$Ra$', ylabel=r'$Nu$')
    ax.legend(fontsize=7)
    save(fig, out, 'nu_ra_scaling')
    out.mkdir(parents=True, exist_ok=True)
    (out/'reference_comparisons.json').write_text(json.dumps(clean_json(reports), indent=2, allow_nan=False)+'\n')


def grid_convergence(runs, out):
    groups = {}
    for run in runs:
        r = report(run)
        groups.setdefault((r['Ra'], r['Pr'], r['rr']), []).append(r)
    fig, ax = plt.subplots(figsize=(6, 4))
    result = []
    for (ra, pr, rr), rs in groups.items():
        rs.sort(key=lambda r: r['grid'])
        values = [r['estimators']['nu_volume']['mean'] for r in rs]
        ax.errorbar([r['grid'] for r in rs], values,
                    yerr=[r['estimators']['nu_volume']['stderr'] or 0 for r in rs],
                    fmt='o-', label=f'Ra={ra:g}, Pr={pr:g}, RR{rr}')
        for coarse, fine in zip(rs[:-1], rs[1:]):
            a = coarse['estimators']['nu_volume']['mean']; b = fine['estimators']['nu_volume']['mean']
            result.append(dict(Ra=ra, Pr=pr, rr=rr, coarse=coarse['grid'], fine=fine['grid'],
                               relative_change=b/a-1, coarse_diagnostics=coarse, fine_diagnostics=fine))
    ax.set(xlabel='Grid points per direction', ylabel=r'$Nu$'); ax.legend(fontsize=8)
    save(fig, out, 'grid_convergence')
    (out/'grid_comparison.json').write_text(json.dumps(clean_json(result), indent=2, allow_nan=False)+'\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('analysis', choices=['convergence', 'profiles', 'scaling', 'grid', 'all',
                                            'validation', 'validation-summary'],
                        help='plot selection; all generates every figure and report')
    parser.add_argument('runs', nargs='+', type=Path,
                        help='simulation output directories, or a campaign root for validation-summary')
    parser.add_argument('--out', type=Path, default=Path('output/rbc_plots'), help='figure/report output directory')
    a = parser.parse_args()
    if a.analysis == 'validation-summary':
        if len(a.runs) != 1:
            parser.error('validation-summary expects one campaign root')
        from scripts.rbc_validation_analysis import summarize_campaign
        summarize_campaign(a.runs[0], a.out)
        return
    for run in a.runs:
        dest = a.out/run.name
        if a.analysis == 'validation':
            from scripts.rbc_validation_analysis import analyze_run
            analyze_run(run, dest)
            continue
        if a.analysis in ['convergence', 'all']:
            convergence(run, dest)
        if a.analysis in ['profiles', 'all']:
            profiles(run, dest)
    if a.analysis in ['scaling', 'all']:
        scaling(a.runs, a.out)
    if a.analysis in ['grid', 'all']:
        grid_convergence(a.runs, a.out)


if __name__ == '__main__':
    main()
