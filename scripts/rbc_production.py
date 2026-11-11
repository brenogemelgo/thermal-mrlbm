#!/usr/bin/env python3
"""Plan or explicitly execute the four sequential RR3 production simulations."""
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import importlib.util
import json
import math
import os
from pathlib import Path
import signal
import shlex
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
from run_rbc import parameters, hdf5_flags

GRIDS = {1: 512, 2: 512, 3: 512, 4: 512}
FIELD_INTERVALS = {1: 5000, 2: 5000, 3: 10000, 4: 10000}


def atomic_json(path, data):
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(data, indent=2, allow_nan=False)+'\n')
    temp.replace(path)


def selection(groups):
    result = set()
    for group in groups or []:
        for arg in group:
            for part in arg.split(','):
                part = part.upper().removeprefix('RBC-')
                if part not in ('1', '2', '3', '4'):
                    raise ValueError(f'unknown case: {part}')
                result.add(int(part))
    return result


def snapshot_count(end, interval, start):
    if not interval or start > end:
        return 0
    # Include the initial state only when start=0, and always the final state.
    return end//interval-max(1, math.ceil(start/interval))+1+int(start == 0)+int(end % interval != 0)


def configuration(case, args):
    n = GRIDS[case]
    p = parameters(case, n)
    steps = p['transient_steps']+p['averaging_steps']
    def interval(name, recommended):
        supplied = getattr(args, name)
        return recommended if supplied is None else supplied
    field_interval = interval('field_every', FIELD_INTERVALS[case])
    cfg = dict(schema=1, run_id=f'rbc{case}_n{n}_rr3', case=case, grid=n, rr=3, precision='single',
               physics=p, steps=steps, transient_steps=p['transient_steps'], sample=args.sample,
               slice_every=interval('slice_every', 0),
               field_every=field_interval,
               profile_every=interval('profile_every', round(p['tff'])),
               progress_every=interval('progress_every', round(p['tff']*.1)),
               checkpoint_every=interval('checkpoint_every', 100000),
               output_start=args.output_start, keep_checkpoints=args.keep_checkpoints,
               compression=args.compression, slice_velocity=args.slice_velocity,
               export_fields=['temperature'], vti_enabled=False,
               hdf5_root=str(args.hdf5_root.resolve()))
    return cfg


def storage(cfg):
    n, end = cfg['grid'], cfg['steps']
    slices = snapshot_count(end, cfg['slice_every'], cfg['output_start'])
    volumes = snapshot_count(end, cfg['field_every'], cfg['output_start'])
    checkpoints = math.ceil(end/cfg['checkpoint_every'])
    retained = min(checkpoints, cfg['keep_checkpoints'] or checkpoints)
    # Moments only; exact ABI header and reduced accumulator add a small amount.
    cp_bytes = 14*n**3*4+2*18*n*8+10*8+256
    scalar_samples = 1+cfg['transient_steps']//cfg['sample']+(end-cfg['transient_steps'])//cfg['sample']
    # Progress triggers are a union of schedules; bound its rows by the sum.
    progress_bound = scalar_samples+checkpoints+volumes+slices+math.ceil(end/(cfg['profile_every'] or end))+math.ceil(end/(cfg['progress_every'] or end))
    reduced_bound = scalar_samples*21*8+progress_bound*12*8+n*19*8+5*5*8
    fields = volumes*len(cfg['export_fields'])*n**3*4
    slice_bytes = slices*(2 if cfg['slice_velocity'] else 1)*n**2*4
    total = fields+slice_bytes+retained*cp_bytes+reduced_bound
    optional_vti = fields if cfg['case']==3 else 0
    return dict(slice_count=slices, volume_count=volumes, checkpoint_writes=checkpoints,
                retained_checkpoints=retained, scalar_samples=scalar_samples,
                fields_GiB=fields/2**30, slices_GiB=slice_bytes/2**30,
                hdf5_field_exports_GiB=(fields+slice_bytes)/2**30,
                hdf5_field_exports_TB=(fields+slice_bytes)/1.e12,
                optional_rbc3_vti_TB=optional_vti/1.e12,
                rbc3_hdf5_plus_vti_field_exports_GiB=(2*fields+slice_bytes)/2**30 if cfg['case']==3 else None,
                checkpoints_GiB=retained*cp_bytes/2**30, reduced_payload_bound_GiB=reduced_bound/2**30,
                final_payload_GiB=(total+retained*reduced_bound)/2**30,
                final_payload_with_optional_rbc3_vti_TB=(total+retained*reduced_bound+optional_vti)/1.e12,
                checkpoint_history_bound_GiB=retained*reduced_bound/2**30,
                checkpoint_transaction_peak_extra_GiB=(cp_bytes+reduced_bound)/2**30,
                snapshot_bytes_per_field=n**3*4, slice_bytes_per_field=n**2*4,
                note='Uncompressed numeric payload only; add HDF5/chunk/filesystem overhead and plots/video. Field-export figures exclude checkpoints/statistics. RBC-3 VTI comparison assumes the same fields as its source; VTI is off by default. No compression ratio assumed.')


def wants_video(cfg, args):
    return cfg['case']==3 and not args.no_video and bool(cfg['slice_every'] or cfg['field_every'])


@contextmanager
def lock(path):
    with path.open('a') as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f'another process owns {path}') from error
        yield


def execute(command, log):
    print(shlex.join(map(str, command)), flush=True)
    with log.open('a') as output:
        output.write('\n'+datetime.now(timezone.utc).isoformat()+' '+shlex.join(map(str, command))+'\n')
        output.flush()
        # Tee line-buffered solver diagnostics to the terminal as well as the log.
        with subprocess.Popen([str(c) for c in command], cwd=ROOT, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True, bufsize=1,
                              start_new_session=True) as process:
            try:
                for line in process.stdout:
                    print(line, end='', flush=True)
                    output.write(line)
                    output.flush()
                if process.wait():
                    raise subprocess.CalledProcessError(process.returncode, command)
            except BaseException:
                if process.poll() is None:
                    # Stop the launcher and its solver/compiler children together;
                    # do not leave an orphan GPU run after releasing the lock.
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait(timeout=10)
                raise


def check_dependencies(video=False, build=False):
    """Check before long work, while letting completed jobs skip without tools."""
    for package in ('numpy', 'matplotlib', 'h5py'):
        if importlib.util.find_spec(package) is None:
            raise RuntimeError(f'missing post-processing dependency: {package}')
    if video and not shutil.which('ffmpeg'):
        raise RuntimeError('ffmpeg is required for automatic movies; install it or select --no-video')
    if build:
        hdf5_flags()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--only', nargs='+', action='append', help='1,2 or RBC-1 RBC-2 (default all four)')
    parser.add_argument('--skip', nargs='+', action='append')
    parser.add_argument('--resume', action='store_true', help='restore interrupted runs; completed runs are always preserved')
    parser.add_argument('--post-only', action='store_true', help='post-process successful simulations without building/running')
    parser.add_argument('--plan', action='store_true', help='print configurations, counts and storage; never build/run')
    parser.add_argument('--output-root', type=Path, default=ROOT/'output')
    parser.add_argument('--hdf5-root', type=Path, default=ROOT/'output_hdf5',
                        help='separate HDF5 root; writes <root>/rayleigh_benard/<run_id>/')
    parser.add_argument('--results-root', type=Path, default=ROOT/'results')
    parser.add_argument('--arch', default='sm_86')
    parser.add_argument('--gpus', type=int, choices=(1, 2), default=1)
    parser.add_argument('--devices', help='visible CUDA ordinals, e.g. 0,1')
    parser.add_argument('--split-z', type=int, help='compile-time interface for unequal GPU capacities/speeds')
    parser.add_argument('--halo-transport', choices=('peer', 'host', 'auto'), default='peer')
    parser.add_argument('--field-io', choices=('async', 'sync', 'serial'), default='async')
    parser.add_argument('--io-workers', type=int, choices=range(1, 33), default=4)
    parser.add_argument('--sample', type=int, default=100)
    for option in ('slice-every', 'field-every', 'profile-every', 'progress-every', 'checkpoint-every'):
        help_text = 'override lattice-step interval for every selected case; 0 disables output except checkpoints'
        if option == 'field-every':
            help_text += '; defaults: RBC-1/2 5000, RBC-3/4 10000'
        parser.add_argument('--'+option, type=int, help=help_text)
    parser.add_argument('--output-start', type=int, default=0, help='earliest slice/volume timestep')
    parser.add_argument('--keep-checkpoints', type=int, default=2, help='0 keeps all; otherwise >=2')
    parser.add_argument('--compression', type=int, choices=range(10), default=1)
    parser.add_argument('--slice-velocity', action='store_true')
    parser.add_argument('--no-video', action='store_true', help='skip RBC-3 MP4 preview; RBC-1/2/4 never render video automatically')
    parser.add_argument('--video-fps', type=int, default=30)
    parser.add_argument('--video-stride', type=int, default=1)
    args = parser.parse_args()
    try:
        selected = (selection(args.only) if args.only else set(GRIDS))-selection(args.skip)
        if not selected:
            raise ValueError('no cases selected')
        if args.split_z is not None and any(not 2 <= args.split_z <= GRIDS[c] - 2 for c in selected):
            raise ValueError('--split-z must leave at least two planes on each GPU')
        intervals = [getattr(args, name) for name in ('slice_every', 'field_every', 'profile_every', 'progress_every', 'checkpoint_every')]
        if any(x is not None and x < 0 for x in intervals) or args.checkpoint_every == 0:
            raise ValueError('intervals must be nonnegative and checkpoints positive')
        if min(args.sample, args.video_fps, args.video_stride) <= 0 or args.output_start < 0 or args.keep_checkpoints < 0 or args.keep_checkpoints == 1:
            raise ValueError('invalid sample, video, output-start or checkpoint retention setting')
    except ValueError as error:
        parser.error(str(error))
    configs = [configuration(c, args) for c in sorted(selected)]
    if args.plan:
        execution = dict(gpus=args.gpus, devices=args.devices, halo_transport=args.halo_transport, split_z=args.split_z,
                         field_io=args.field_io, io_workers=args.io_workers)
        print(json.dumps([dict(config=c, storage=storage(c), execution=execution) for c in configs], indent=2))
        return 0
    output_root, results_root = args.output_root.resolve(), args.results_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    results_root.mkdir(parents=True, exist_ok=True)
    state_path = output_root/'rbc_production_state.json'
    with lock(output_root/'.rbc_production.lock'):
        state = json.loads(state_path.read_text()) if state_path.exists() else dict(schema=1, jobs={})
        for case, n in GRIDS.items():
            state['jobs'].setdefault(f'rbc{case}_n{n}_rr3', dict(status='not_run'))
        state['selected'] = sorted(selected)
        failure = None
        try:
            for cfg in configs:
                rid = cfg['run_id']
                run, post = output_root/'rayleigh_benard'/rid, results_root/rid
                hdf5_run = Path(cfg['hdf5_root'])/'rayleigh_benard'/rid
                entry = state['jobs'].setdefault(rid, {})
                entry.update(run=str(run), hdf5=str(hdf5_run), results=str(post), storage=storage(cfg))
                entry['execution'] = dict(gpus=args.gpus, devices=args.devices, halo_transport=args.halo_transport, split_z=args.split_z,
                                          field_io=args.field_io, io_workers=args.io_workers)
                manifest, receipt = run/'production_run.json', run/'simulation_complete.json'
                if manifest.exists():
                    previous = json.loads(manifest.read_text())
                    if previous != cfg:
                        temperature_transition = (previous.get('export_fields') == ['temperature', 'ux', 'uy', 'uz'] and
                                                  dict(previous, export_fields=['temperature']) == cfg)
                        if temperature_transition and receipt.exists():
                            # Completed historical exports keep their original configuration.
                            cfg = previous
                        elif temperature_transition and args.resume:
                            atomic_json(run/'production_export_transition.json', dict(previous=previous, current=cfg))
                            atomic_json(manifest, cfg)
                        else:
                            raise RuntimeError(f'configuration mismatch: {run}; restore original options or select another --output-root')
                else:
                    if args.post_only or (run.exists() and any(run.iterdir())):
                        raise RuntimeError(f'no owned production manifest; refusing to overwrite/adopt {run}')
                    if hdf5_run.exists() and any(hdf5_run.iterdir()):
                        raise RuntimeError(f'refusing to adopt unowned HDF5 output: {hdf5_run}')
                    run.mkdir(parents=True, exist_ok=True)
                    atomic_json(manifest, cfg)
                entry['storage'] = storage(cfg)
                post.mkdir(parents=True, exist_ok=True)
                if receipt.exists():
                    if json.loads(receipt.read_text()) != cfg:
                        raise RuntimeError(f'incompatible completion receipt: {receipt}')
                    print(f'Completed simulation skipped: {rid}', flush=True)
                elif args.post_only:
                    raise RuntimeError(f'{rid}: successful simulation receipt missing')
                else:
                    checkpoints = list((run/'checkpoints').glob('step_*.chk'))
                    existing_data = (hdf5_run/'statistics.h5').exists()
                    if (checkpoints or existing_data) and not args.resume:
                        raise RuntimeError(f'{rid}: interrupted output exists; use --resume')
                    if existing_data and not checkpoints:
                        raise RuntimeError(f'{rid}: no committed checkpoint to resume; preserve this run and choose another --output-root')
                    check_dependencies(video=wants_video(cfg, args), build=True)
                    entry['status'] = 'running'
                    atomic_json(state_path, state)
                    command = [sys.executable, ROOT/'scripts/run_rbc.py', '--case', cfg['case'], '--grid', cfg['grid'],
                               '--rr', 3, '--precision', 'single', '--steps', cfg['steps'], '--transient-steps', cfg['transient_steps'],
                               '--sample', cfg['sample'], '--run-id', rid, '--output-root', output_root,
                               '--hdf5-root', cfg['hdf5_root'], '--arch', args.arch,
                               '--gpus', args.gpus, '--halo-transport', args.halo_transport,
                               '--field-io', args.field_io, '--io-workers', args.io_workers]
                    if args.devices is not None:
                        command += ['--devices', args.devices]
                    if args.split_z is not None:
                        command += ['--split-z', args.split_z]
                    for key in ('checkpoint_every', 'slice_every', 'field_every', 'profile_every', 'progress_every',
                                'output_start', 'keep_checkpoints', 'compression'):
                        command.extend(['--'+key.replace('_', '-'), cfg[key]])
                    if cfg['slice_velocity']:
                        command.append('--slice-velocity')
                    if args.resume and checkpoints:
                        command.append('--continue')
                    execute(command, run/'execution.log')
                    if not (run/'checkpoints'/f"step_{cfg['steps']:09d}.chk").exists():
                        raise RuntimeError(f'{rid}: final checkpoint missing')
                    atomic_json(receipt, cfg)
                entry['status'] = 'postprocessing'
                atomic_json(state_path, state)
                post_options = dict(run=str(run), config=cfg,
                                    video=wants_video(cfg, args),
                                    fps=args.video_fps, stride=args.video_stride)
                post_receipt = post/'production_post_complete.json'
                if post_receipt.exists() and json.loads(post_receipt.read_text()).get('run') != str(run):
                    raise RuntimeError(f'{post}: results belong to another raw output root; choose another --results-root')
                if not args.post_only and post_receipt.exists() and json.loads(post_receipt.read_text()) == post_options:
                    print(f'Completed post-processing skipped: {rid}', flush=True)
                else:
                    check_dependencies(video=post_options['video'])
                    command = [sys.executable, ROOT/'postRayleigh.py', 'all', run, '--out', results_root]
                    if post_options['video']:
                        command += ['--video', '--video-fps', args.video_fps, '--video-stride', args.video_stride]
                    execute(command, post/'postprocess.log')
                    atomic_json(post_receipt, post_options)
                entry['status'] = 'complete'
                entry.pop('error', None)
                atomic_json(state_path, state)
        except (Exception, KeyboardInterrupt) as error:
            failure = error
            if 'entry' in locals():
                entry.update(status='failed', error=str(error) or type(error).__name__)
        finally:
            atomic_json(state_path, state)
            # Summarize execution only; the numerical report explicitly withholds DNS claims.
            atomic_json(results_root/'rbc_production_summary.json', state)
            lines = ['# RBC production campaign', '', '| Run | Execution |', '| --- | --- |']
            lines += [f"| {rid} | {entry.get('status', 'not run')} |" for rid, entry in state['jobs'].items()]
            lines += ['', 'See each run report for statistical intervals, heat-flux agreement and resolution diagnostics. Execution completion does not establish DNS convergence.']
            (results_root/'rbc_production_summary.md').write_text('\n'.join(lines)+'\n')
        if failure:
            print(f'Production campaign stopped: {failure}. Successful results retained.', file=sys.stderr)
            return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
