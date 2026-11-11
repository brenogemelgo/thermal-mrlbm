"""Post-processing only. Functions run only through explicit CLI requests."""
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import h5py

from postRayleigh import NUS, clean_json, convergence, load, metadata, profiles, report, save, uncertainty
from scripts.rbc_validation_reference import exact, force_identity, stress_tensor, NU, KAPPA
from scripts.rbc_hdf5 import data_directory


def write_json(path, value):
    path.write_text(json.dumps(clean_json(value), indent=2, allow_nan=False)+'\n')


def write_table(path, rows, columns=None):
    """Write numerical tables as columnar HDF5."""
    rows = list(rows)
    columns = columns or sorted({k for row in rows for k in row})
    target = path.with_suffix('.h5')
    temp = target.with_suffix('.h5.tmp')
    with h5py.File(temp, 'w') as out:
        out.attrs['schema'] = 'thermal-mrlbm-derived-columns-v1'
        group = out.create_group('columns')
        for column in columns:
            values = [row.get(column) for row in rows]
            if all(v is None or isinstance(v, (int, float, np.number)) for v in values):
                data = np.array([np.nan if v is None else v for v in values], dtype=float)
            else:
                data = np.array(['' if v is None else str(v) for v in values], dtype=h5py.string_dtype())
            group.create_dataset(column, data=data, compression='gzip')
    temp.replace(target)


def status(checks):
    states = [c['status'] for c in checks]
    return 'failed' if 'failed' in states else 'inconclusive' if 'inconclusive' in states else 'passed'


def check(name, valid, detail):
    return dict(name=name, status='passed' if valid else 'failed', detail=detail)


def norms(error):
    # Invariant Y cancels in the volume mean. Periodic X counts each point once;
    # the physical on-node plate volume uses trapezoidal Z weights.
    n = error.shape[-2]
    weights = np.ones(n)
    weights[[0, -1]] = .5
    weights = weights[:, None]
    size = (n-1)*error.shape[-1]
    return dict(l1=float(np.sum(np.abs(error)*weights)/size),
                l2=float(np.sqrt(np.sum(error**2*weights)/size)),
                linf=float(np.max(np.abs(error))))


def grad_t(t):
    return (np.roll(t, -1, axis=1)-np.roll(t, 1, axis=1))/2


def lap(t):
    result = np.roll(t, -1, axis=1)-2*t+np.roll(t, 1, axis=1)
    result[1:-1] += t[2:]-2*t[1:-1]+t[:-2]
    return result


def snapshots(run, target):
    entries = []
    import h5py
    for path in (data_directory(run)/'validation').glob('snapshot_*.h5'):
        with h5py.File(path, 'r') as source:
            info = {k: v.decode() if isinstance(v, bytes) else str(v) for k, v in source.attrs.items()}
        info = {k: int(v) if k != 'case' else v for k, v in info.items()}
        if info['step'] <= target:
            entries.append((info['step'], path, info))
    for path in (data_directory(run)/'validation').glob('snapshot_*.json'):
        info = json.loads(path.read_text())
        if info['step'] <= target and not path.with_suffix('.h5').exists():
            entries.append((info['step'], path, info))
    entries.sort()
    if not entries or entries[0][0] != 0 or entries[-1][0] != target:
        raise ValueError(f'{run}: initial/final committed validation snapshot missing')
    for _, path, info in entries:
        n = info['nx']
        if info['schema'] != 1 or n != info['nz'] or info['channels'] != 29:
            raise ValueError(f'incompatible slice schema: {path}')
        if path.suffix == '.h5':
            with h5py.File(path, 'r') as source:
                data = source['moments'][:]
        else:
            data = np.fromfile(path.with_suffix('.bin'), dtype=np.float64)
        if data.size != 29*n*n:
            raise ValueError(f'truncated slice: {path}')
        yield info, data.reshape(29, n, n)


def analytical(run, out, job, m):
    test, n, target = job['test'], job['n'], job['steps']
    checks = [check('validation_configuration', m.get('validation_case') == test and
                    m['NU'] == NU and m['KAPPA'] == KAPPA and m['real_bytes'] == 8 and
                    m['validation_formula_version'] == 1 and m['validation_mode_amplitude'] == .1 and
                    m['validation_peclet'] == 1 and m['validation_channel_re'] == .2,
                    'Analytical cases require fixed nu=kappa=0.02 and FP64 state.')]
    rows, initial, final = [], None, None
    identity_ok = True
    y_error = 0.
    for info, data in snapshots(run, target):
        step = info['step']
        if info['previous_step'] != (step-1 if step else 0):
            raise ValueError('expected consecutive old/new time levels in each snapshot')
        if info['case'] != test or info['nx'] != n:
            raise ValueError('snapshot case/grid differs from campaign manifest')
        now, old = data[:14], data[14:28]
        if not np.all(np.isfinite(data)) or np.any(now[0] <= 0):
            raise ValueError('nonfinite slice or nonpositive density')
        y_error = max(y_error, float(data[28].max()))
        ref = exact(test, n, step)
        physical = now[1:4]-ref['a']/2
        row = dict(step=step, diffusive_time=ref['time'])
        for key, value in norms(now[10]-ref['T']).items():
            row['temperature_'+key] = value
        row['wall_temperature_linf'] = float(np.max(np.abs((now[10]-ref['T'])[[0, -1]])))
        row['wall_velocity_linf'] = float(np.max(np.abs(physical[:, [0, -1]])))
        bottom = (3*now[10, 0]-4*now[10, 1]+now[10, 2])/2*(n-1)
        top = (-3*now[10, -1]+4*now[10, -2]-now[10, -3])/2*(n-1)
        row['wall_flux_bottom_error'] = float(np.mean(bottom)-1) if test == 'V1' else None
        row['wall_flux_top_error'] = float(np.mean(top)-1) if test == 'V1' else None
        if test == 'V1':
            row['wall_flux_bottom_abs_error'] = abs(row['wall_flux_bottom_error'])
            row['wall_flux_top_abs_error'] = abs(row['wall_flux_top_error'])
        row['velocity_l2'] = float(np.sqrt(sum(norms(physical[i]-ref['u'][i])['l2']**2 for i in range(3))))
        row['velocity_linf'] = float(np.max(np.abs(physical-ref['u'])))
        if test == 'V2' and step:
            before = old[10]
            mid = (before+now[10])/2
            midpoint = exact(test, n, step-.5)
            laplacian = lap(mid)[1:-1]
            advection = midpoint['u'][0]*grad_t(mid)
            residual = now[10]-before+advection-midpoint['source']
            denominator = float(np.sum(laplacian**2))
            row['recovered_diffusivity'] = float(np.sum(laplacian*residual[1:-1])/denominator) if denominator else None
            row['diffusivity_error'] = abs(row['recovered_diffusivity']-KAPPA) if denominator else None
            row['pde_residual_l2'] = float(np.sqrt(np.mean((residual[1:-1]-KAPPA*laplacian)**2)))*(n-1)**2/KAPPA
            omitted = (now[10]-before-midpoint['source'])[1:-1]-KAPPA*laplacian
            row['pde_residual_without_advection_l2'] = float(np.sqrt(np.mean(omitted**2)))*(n-1)**2/KAPPA
            row['reference_advection_l2'] = norms(midpoint['adv_t'])['l2']*(n-1)**2/KAPPA
        if test == 'V3':
            omega = 1/(.5+3*NU)
            c2 = stress_tensor(now)+np.einsum('izx,jzx->ijzx', ref['a'], ref['a'])/4
            stress = now[0]*(1-omega/2)/(1-omega)*c2
            amplitude = .2*NU/(n-1)
            stress_scale = NU*amplitude/(n-1)
            row['velocity_relative_l2'] = row['velocity_l2']/amplitude
            row['stress_relative_l2'] = float(np.sqrt(sum(norms(stress[i, j]-ref['stress'][i, j])['l2']**2
                                                            for i in range(3) for j in range(3))))/stress_scale
            row['stress_linf'] = float(np.max(np.abs(stress-ref['stress'])))
            if step:
                residual, bound = force_identity(old, now, step, job['rr'])
                row['force_identity_linf'], row['force_roundoff_bound'] = residual, bound
                identity_ok &= residual <= bound
                old_ref = exact(test, n, step-1)
                old_u = old[1:4]-old_ref['a']/2
                midpoint = exact(test, n, step-.5)
                momentum_residual = (physical[0]-old_u[0]-NU*lap((physical[0]+old_u[0])/2)-midpoint['a'][0])[1:-1]
                row['momentum_pde_relative_l2'] = float(np.sqrt(np.mean(momentum_residual**2)))/max(float(np.max(np.abs(midpoint['a'][0]))), np.finfo(float).tiny)
        if any(isinstance(value, (float, np.floating)) and not np.isfinite(value)
               for value in row.values() if value is not None):
            raise ValueError(f'nonfinite analytical metric at step {step}')
        rows.append(row)
        if initial is None:
            initial = (now.copy(), ref)
        final = (now.copy(), ref, physical.copy())
    checks += [check('finite_slice_data', True, 'All committed two-time-level slices are finite.'),
               check('y_invariance', y_error == 0, f'Maximum variation over all Y nodes: {y_error:g}'),
               check('exact_plate_temperature', max(r['wall_temperature_linf'] for r in rows) == 0,
                     'On-node Dirichlet temperatures must be assigned exactly.'),
               check('exact_no_slip', max(r['wall_velocity_linf'] for r in rows) == 0,
                     'The plate velocity is zero; force is also zero at both plates.')]
    if test == 'V2':
        checks += [check('nontrivial_advection', max(r.get('reference_advection_l2', 0) for r in rows) > 0,
                         'The source includes a nonzero spatial/time-dependent u*dT/dx.'),
                   check('prescribed_velocity', max(r['velocity_linf'] for r in rows) <= 32*np.finfo(float).eps,
                         'Direct prescription is checked against the independent Python reference; this is not an NS accuracy check.')]
    if test == 'V3':
        checks.append(check('local_force_identity', identity_ok,
                            'Independent D3Q27 quadrature and direct Guo collision; tolerance is a floating-point forward-error budget.'))
    checks.append(dict(name='continuum_accuracy', status='inconclusive',
                       detail='Report errors and refinement order; no universal finite-grid truncation tolerance is imposed.'))
    write_table(out/'analytical_history.h5', rows)
    now, ref, physical = final
    z = np.arange(n)/(n-1)
    omega = 1/(.5+3*NU)
    stress = now[0]*(1-omega/2)/(1-omega)*(stress_tensor(now)+np.einsum('izx,jzx->ijzx', ref['a'], ref['a'])/4)
    profile_rows = [dict(z_over_h=z[i], temperature=float(now[10, i].mean()),
                         temperature_exact=float(ref['T'][i].mean()), ux=float(physical[0, i].mean()),
                         ux_exact=float(ref['u'][0, i].mean()), stress_xz=float(stress[0, 2, i].mean()),
                         stress_xz_exact=float(ref['stress'][0, 2, i].mean())) for i in range(n)]
    write_table(out/'analytical_profiles.h5', profile_rows)
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.5))
    for key in ('temperature_l1', 'temperature_l2', 'temperature_linf'):
        ax[0].semilogy([r['diffusive_time'] for r in rows], [max(r[key], np.finfo(float).tiny) for r in rows], label=key)
    ax[0].set(xlabel='Diffusive time', ylabel='Temperature error'); ax[0].legend(fontsize=7)
    ax[1].plot(now[10].mean(axis=1), z, label='Numerical')
    ax[1].plot(ref['T'].mean(axis=1), z, 'k--', label='Exact')
    ax[1].set(xlabel='Plane mean temperature', ylabel=r'$z/H$'); ax[1].legend()
    save(fig, out, 'analytical_temperature')
    fig, ax = plt.subplots(figsize=(5, 3.5))
    image = ax.imshow(now[10]-ref['T'], origin='lower', extent=(0, n/(n-1), 0, 1), aspect='auto', cmap='RdBu_r')
    fig.colorbar(image, ax=ax, label='Temperature error')
    ax.set(xlabel=r'$x/H$', ylabel=r'$z/H$'); save(fig, out, 'temperature_error_slice')
    if test == 'V2':
        fig, ax = plt.subplots(1, 2, figsize=(9, 3.5))
        valid = [r for r in rows if r.get('recovered_diffusivity') is not None]
        ax[0].plot([r['diffusive_time'] for r in valid], [r['recovered_diffusivity'] for r in valid], 'o-')
        ax[0].axhline(KAPPA, color='k', ls='--'); ax[0].set(xlabel='Diffusive time', ylabel='Recovered diffusivity')
        for key in ('pde_residual_l2', 'pde_residual_without_advection_l2'):
            ax[1].semilogy([r['diffusive_time'] for r in valid], [r[key] for r in valid], label=key)
        ax[1].set(xlabel='Diffusive time', ylabel='PDE residual (diffusive units)'); ax[1].legend(fontsize=6)
        save(fig, out, 'diffusivity_advection_verification')
    if test == 'V3':
        fig, ax = plt.subplots(1, 3, figsize=(12, 3.5))
        ax[0].plot(physical[0].mean(axis=1), z); ax[0].plot(ref['u'][0].mean(axis=1), z, 'k--')
        ax[0].set(xlabel='Channel velocity', ylabel=r'$z/H$')
        ax[1].plot(stress[0, 2].mean(axis=1), z); ax[1].plot(ref['stress'][0, 2].mean(axis=1), z, 'k--')
        ax[1].set(xlabel='Viscous momentum-flux stress', ylabel=r'$z/H$')
        valid = [r for r in rows if 'force_identity_linf' in r]
        ax[2].semilogy([r['diffusive_time'] for r in valid], [max(r['force_identity_linf'], 1e-300) for r in valid], label='Identity residual')
        ax[2].semilogy([r['diffusive_time'] for r in valid], [r['force_roundoff_bound'] for r in valid], '--', label='Roundoff bound')
        ax[2].set(xlabel='Viscous time', ylabel='Force consistency'); ax[2].legend(fontsize=7)
        save(fig, out, 'forced_channel_velocity_stress')
    result = dict(test=test, grid=n, rr=job['rr'], h_z=1/(n-1), h_x=1/(n-1),
                  final=rows[-1], checks=checks, status=status(checks),
                  transverse_invariance_error=y_error, error_norm='volume mean with half-weight wall nodes')
    if test == 'V3':
        result['stress_convention'] = 'Non-equilibrium viscous momentum flux, sigma=-2*rho*nu*S; opposite sign to Cauchy viscous stress.'
    if test == 'V1':
        series = load(run, 'rbc_timeseries.csv')
        nu_rows = [dict(step=int(row['step']), **{key: float(row[key]) for key in NUS}) for row in series]
        write_table(out/'conduction_nusselt.h5', nu_rows)
        result['nusselt'] = {key: dict(final=float(series[key][-1]), maximum_error=float(np.max(np.abs(series[key]-1)))) for key in NUS}
        result['kinetic_nusselt_note'] = 'Ra=0 makes the kinetic-dissipation identity singular; its conduction-limit diagnostic is 1.'
        fig, ax = plt.subplots(1, 2, figsize=(9, 3.5))
        for key in NUS:
            ax[0].plot(series['step'], series[key], label=key)
        ax[0].axhline(1, color='k', ls='--'); ax[0].set(xlabel='Lattice step', ylabel=r'$Nu$'); ax[0].legend(fontsize=6)
        ax[1].plot([r['diffusive_time'] for r in rows], [r['wall_flux_bottom_error'] for r in rows], label='Lower plate')
        ax[1].plot([r['diffusive_time'] for r in rows], [r['wall_flux_top_error'] for r in rows], label='Upper plate')
        ax[1].set(xlabel='Diffusive time', ylabel='Wall flux error'); ax[1].legend()
        save(fig, out, 'conduction_nusselt_wall_flux')
    return result


def confidence(values, seed):
    values = np.asarray(values, dtype=float)
    info = uncertainty(values)
    block = max(1, int(np.ceil(2*(info['tau_samples'] or 1))))
    count = len(values)//block
    info.update(block_samples=block, independent_blocks=count, confidence_interval_95=None)
    draws = np.empty(0)
    if count >= 8 and info['tau_samples'] is not None:
        means = values[:count*block].reshape(count, block).mean(axis=1)
        rng = np.random.default_rng(seed)
        # Bound temporary memory even for long records.
        draws = np.concatenate([means[rng.integers(0, count, size=(100, count))].mean(axis=1) for _ in range(50)])
        draws += values.mean()-means.mean()
        info['confidence_interval_95'] = np.quantile(draws, [.025, .975]).tolist()
    info['confidence_note'] = 'Conditional batch-bootstrap percentile interval; stationarity and a block-size plateau still require assessment.'
    return info, draws


def turbulent(run, out, job, m):
    convergence(run, out)
    profiles(run, out)
    r = report(run)
    data = load(run, 'rbc_timeseries.csv')
    avg = data[data['averaging'] == 1]
    expected = (job['steps']-job['transient_steps'])//job['sample']
    checks = [check('averaging_schedule', len(avg) == expected and m['timeAverageCount'] == expected,
                    f'Expected {expected} post-transient samples; found {len(avg)}.'),
              check('finite_diagnostics', all(np.all(np.isfinite(data[key])) for key in NUS), 'Five Nusselt series must be finite.')]
    target_ra = 1e6 if job['test'] == 'V4' else 1e7
    eps = np.finfo(np.float32).eps
    checks.append(check('physical_parameters', m['RA'] == target_ra and
                        abs(m['PR']-.7) <= eps*.7 and abs(m['U_FREEFALL']-.05) <= eps*.05 and
                        m['H'] == job['n']-1 and m['THOT'] == 1 and m['TCOLD'] == 0,
                        'Ra, Pr, free-fall speed, plate separation and temperatures; literal rounding uses FP32 precision.'))
    scalars, bootstrap = {}, {}
    for key in NUS:
        seed = int.from_bytes(hashlib.sha256((job['run_id']+key).encode()).digest()[:8], 'little')
        scalars[key], bootstrap[key] = confidence(avg[key], seed)
    with h5py.File(out/'nusselt_bootstrap.h5', 'w') as output:
        for key, values in bootstrap.items():
            output.create_dataset(key, data=values, compression='gzip')
    rows = []
    for key, info in scalars.items():
        ci = info['confidence_interval_95']
        rows.append(dict(estimator=key, samples=len(avg), mean=info['mean'],
                         ci95_lower=ci[0] if ci else None, ci95_upper=ci[1] if ci else None,
                         effective_samples=info['effective_samples'], block_samples=info['block_samples'],
                         blocks=info['independent_blocks'], half_mean_relative_difference=info.get('half_mean_relative_difference')))
    write_table(out/'nusselt_confidence.h5', rows)
    checks.append(dict(name='statistical_stationarity', status='inconclusive', detail='Intervals and half-record drift are reported; no automatic stationarity claim.'))
    checks.append(dict(name='dns_resolution', status='inconclusive', detail='Nu(z), boundary-layer cells, local Delta/eta and dissipation are diagnostic evidence, not a DNS certificate.'))
    fig, ax = plt.subplots(figsize=(7, 3.5))
    for i, key in enumerate(NUS):
        info = scalars[key]
        ax.plot(i, info['mean'], 'o')
        if info['confidence_interval_95']:
            low, high = info['confidence_interval_95']
            ax.vlines(i, low, high)
    ax.set_xticks(range(5), [k.removeprefix('nu_') for k in NUS], rotation=20)
    ax.set(ylabel=r'$Nu$ (conditional 95% intervals)'); save(fig, out, 'nusselt_confidence')
    return dict(test=job['test'], grid=job['n'], rr=job['rr'], h_z=1/(job['n']-1), h_x=1/(job['n']-1),
                checks=checks, status=status(checks), scalars=scalars, diagnostics=r,
                dns_convergence_established=False)


def analyze_run(run, out):
    run, out = Path(run), Path(out)
    job = json.loads((run/'validation_run.json').read_text())
    if json.loads((run/'simulation_complete.json').read_text()) != job:
        raise ValueError('missing or mismatched successful simulation receipt')
    m = metadata(run)
    if (m['NX'] != job['n'] or m['NY'] != job['n'] or m['NZ'] != job['n'] or
        m['NSTEPS'] != job['steps'] or m['REGULARIZATION_ORDER'] != job['rr'] or
        m['timeAverageStart'] != job['transient_steps'] or m['timeAverageStride'] != job['sample']):
        raise ValueError('solver metadata differs from campaign configuration')
    data = load(run, 'rbc_timeseries.csv')
    final_sample = job['transient_steps']+(job['steps']-job['transient_steps'])//job['sample']*job['sample']
    if int(data['step'][-1]) != final_sample:
        raise ValueError('time series does not reach the expected final sampling step')
    out.mkdir(parents=True, exist_ok=True)
    result = analytical(run, out, job, m) if job['test'] in ('V1', 'V2', 'V3') else turbulent(run, out, job, m)
    result.update(run=str(run), run_id=job['run_id'], analysis_completed=True)
    write_json(out/'validation_result.json', result)
    write_table(out/'checks.h5', result['checks'], ['name', 'status', 'detail'])
    return result


def summarize_campaign(root, out):
    root, out = Path(root), Path(out)
    state = json.loads((root/'campaign_state.json').read_text())
    out.mkdir(parents=True, exist_ok=True)
    results, rows, all_checks = [], [], []
    for entry in state['jobs'].values():
        path = Path(entry.get('post', root/'absent'))/'validation_result.json'
        result = json.loads(path.read_text()) if path.exists() and entry['status'] == 'post_complete' else None
        if result:
            results.append((entry, result))
            all_checks.extend(dict(test=entry['test'], grid=entry['n'], rr=entry['rr'], **c) for c in result['checks'])
        else:
            all_checks.append(dict(test=entry['test'], grid=entry['n'], rr=entry['rr'], name='execution',
                                   status='failed' if entry['status'] == 'failed' else 'inconclusive',
                                   detail=entry.get('error', 'not executed')))
        outcome = result['status'] if result else 'failed' if entry['status'] == 'failed' else 'inconclusive'
        rows.append(dict(test=entry['test'], grid=entry['n'], rr=entry['rr'], execution=entry['status'],
                         status=outcome, run_id=entry['run_id'], detail=entry.get('error', 'not executed' if not result else 'see per-run checks')))
    write_table(out/'campaign_summary.h5', rows, ['test', 'grid', 'rr', 'execution', 'status', 'run_id', 'detail'])
    write_table(out/'campaign_checks.h5', all_checks, ['test', 'grid', 'rr', 'name', 'status', 'detail'])
    refinements = []
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    for axis, test in zip(axes, ['V1', 'V2', 'V3']):
        for rr in sorted({r['rr'] for _, r in results if r['test'] == test}):
            group = sorted([r for _, r in results if r['test'] == test and r['rr'] == rr], key=lambda r: r['grid'])
            metrics = ['temperature_l1', 'temperature_l2', 'temperature_linf'] if test != 'V3' else ['velocity_relative_l2', 'stress_relative_l2', 'momentum_pde_relative_l2']
            if test == 'V1':
                metrics += ['wall_flux_bottom_abs_error', 'wall_flux_top_abs_error']
            if test == 'V2':
                metrics += ['diffusivity_error', 'pde_residual_l2']
            for metric in metrics:
                pairs = [(r['h_z'], r['final'].get(metric)) for r in group]
                pairs = [(h, e) for h, e in pairs if e is not None and np.isfinite(e)]
                if pairs:
                    axis.loglog([h for h, _ in pairs], [max(e, np.finfo(float).tiny) for _, e in pairs], 'o-', label=f'{metric}, RR{rr}')
                for coarse, fine in zip(group[:-1], group[1:]):
                    ec, ef = coarse['final'].get(metric), fine['final'].get(metric)
                    order = float(np.log(ec/ef)/np.log(coarse['h_z']/fine['h_z'])) if ec and ef and ec > 0 and ef > 0 else None
                    refinements.append(dict(test=test, rr=rr, metric=metric, coarse=coarse['grid'], fine=fine['grid'],
                                            h_coarse=coarse['h_z'], h_fine=fine['h_z'], error_coarse=ec, error_fine=ef,
                                            observed_order=order, status='inconclusive',
                                            note='Linear conduction can be roundoff-limited; finite-grid order alone is not an accuracy acceptance bound.'))
        axis.set(title=test, xlabel=r'$h_z=1/(N-1)$', ylabel='Error')
        if axis.lines:
            axis.legend(fontsize=5)
    save(fig, out, 'analytical_spatial_convergence')
    write_table(out/'spatial_convergence.h5', refinements,
              ['test', 'rr', 'metric', 'coarse', 'fine', 'h_coarse', 'h_fine',
               'error_coarse', 'error_fine', 'observed_order', 'status', 'note'])
    grids = []
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    for axis, test in zip(axes, ['V4', 'V5']):
        group = sorted([(e, r) for e, r in results if r['test'] == test], key=lambda pair: pair[1]['grid'])
        for key in NUS:
            axis.plot([r['grid'] for _, r in group], [r['scalars'][key]['mean'] for _, r in group], 'o-', label=key)
            for (ce, coarse), (fe, fine) in zip(group[:-1], group[1:]):
                a, b = coarse['scalars'][key]['mean'], fine['scalars'][key]['mean']
                low = high = None
                def bootstrap_file(entry):
                    path = Path(entry['post'])/'nusselt_bootstrap.h5'
                    return h5py.File(path, 'r') if path.exists() else np.load(path.with_suffix('.npz'))
                with bootstrap_file(ce) as ca, bootstrap_file(fe) as fa:
                    if len(ca[key]) and len(fa[key]):
                        low, high = np.quantile(np.asarray(fa[key])-np.asarray(ca[key]), [.025, .975]).tolist()
                grids.append(dict(test=test, estimator=key, coarse=coarse['grid'], fine=fine['grid'],
                                  difference=b-a, relative_difference=(b-a)/a, ci95_difference_lower=low,
                                  ci95_difference_upper=high, status='inconclusive',
                                  note='Independent-run batch bootstrap; requires stationary, adequately blocked records.'))
        axis.set(title=test, xlabel='Grid points per direction', ylabel=r'$Nu$')
        if axis.lines:
            axis.legend(fontsize=6)
    save(fig, out, 'rbc_grid_differences')
    write_table(out/'rbc_grid_differences.h5', grids,
              ['test', 'estimator', 'coarse', 'fine', 'difference', 'relative_difference',
               'ci95_difference_lower', 'ci95_difference_upper', 'status', 'note'])
    comparison = []
    v3 = [r for _, r in results if r['test'] == 'V3']
    for n in sorted({r['grid'] for r in v3}):
        by_order = {r['rr']: r for r in v3 if r['grid'] == n}
        if 2 in by_order and 3 in by_order:
            for metric in ('velocity_relative_l2', 'stress_relative_l2', 'force_identity_linf'):
                comparison.append(dict(grid=n, metric=metric, rr2=by_order[2]['final'][metric], rr3=by_order[3]['final'][metric]))
    write_table(out/'rr2_rr3_comparison.h5', comparison, ['grid', 'metric', 'rr2', 'rr3'])
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    for axis, metric in zip(axes, ['velocity_relative_l2', 'stress_relative_l2']):
        for rr in (2, 3):
            group = sorted([r for r in v3 if r['rr'] == rr], key=lambda r: r['grid'])
            if group:
                axis.semilogy([r['grid'] for r in group], [r['final'][metric] for r in group], 'o-', label=f'RR{rr}')
        axis.set(xlabel='Grid points', ylabel=metric)
        if axis.lines:
            axis.legend()
    save(fig, out, 'rr2_rr3_channel_comparison')
    counts = {s: sum(row['status'] == s for row in rows) for s in ['passed', 'failed', 'inconclusive']}
    check_counts = {s: sum(c['status'] == s for c in all_checks) for s in counts}
    fig, axis = plt.subplots(figsize=(6, 3.5))
    axis.bar(list(check_counts), list(check_counts.values()), color=['#228833', '#cc3311', '#bb9911'])
    axis.set(ylabel='Number of checks'); save(fig, out, 'campaign_check_status')
    summary = dict(schema=1, counts=counts, check_counts=check_counts, checks=all_checks,
                   jobs=rows, spatial_convergence=refinements, grid_differences=grids,
                   rr_comparison=comparison, dns_convergence_established=False,
                   note='Unexecuted jobs and unsupported physical acceptance claims are inconclusive; execution failures remain failed.')
    write_json(out/'campaign_summary.json', summary)
    lines = ['# RBC pre-production campaign', '', f"Passed: {counts['passed']}; failed: {counts['failed']}; inconclusive: {counts['inconclusive']}.",
             '', f"Individual checks: {check_counts['passed']} passed, {check_counts['failed']} failed, {check_counts['inconclusive']} inconclusive.",
             '', '| Test | N | RR | Execution | Assessment |', '| --- | --- | --- | --- | --- |']
    lines += [f"| {r['test']} | {r['grid']} | {r['rr']} | {r['execution']} | {r['status']} |" for r in rows]
    lines += ['', 'See per-run checks and HDF5/JSON diagnostics. Continuum accuracy and turbulent DNS convergence require assessment; no exact RBC Nusselt acceptance value is imposed.']
    (out/'campaign_summary.md').write_text('\n'.join(lines)+'\n')
    return summary
