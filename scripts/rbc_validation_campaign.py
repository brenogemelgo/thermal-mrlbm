#!/usr/bin/env python3
"""Run the pre-production matrix sequentially, preserving checkpoints and results."""
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import json
from pathlib import Path
import shlex
import subprocess
import sys

from rbc_validation_cases import GRIDS, jobs

ROOT = Path(__file__).resolve().parents[1]


def atomic_json(path, value):
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    tmp.replace(path)


def selection(groups):
    result = set()
    for group in groups or []:
        for value in group:
            result.update(s.upper() for s in value.split(',') if s)
    bad = result-set(GRIDS)
    if bad:
        raise ValueError(f'unknown tests: {sorted(bad)}')
    return result


@contextmanager
def lock_campaign(root):
    with (root/'.campaign.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f'another campaign owns {root}') from error
        yield


def execute(cmd, log):
    print(shlex.join(map(str, cmd)), flush=True)
    with log.open('a') as out:
        out.write('\n'+datetime.now(timezone.utc).isoformat()+' '+shlex.join(map(str, cmd))+'\n')
        out.flush()
        subprocess.run([str(s) for s in cmd], cwd=ROOT, stdout=out, stderr=subprocess.STDOUT, check=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--only', nargs='+', action='append', help='V1,V2 or V1 V2; default: V1–V5')
    p.add_argument('--skip', nargs='+', action='append', help='exclude named tests')
    p.add_argument('--rr', choices=['2', '3', 'both'], default='3',
                   help='V3 order/comparison; V1,V2,V4,V5 always use RR3')
    p.add_argument('--resume', action='store_true', help='resume interrupted jobs and skip completed simulations')
    p.add_argument('--post-only', action='store_true', help='analyze completed jobs; never build or simulate')
    p.add_argument('--root', type=Path, default=ROOT/'output/rbc_validation')
    p.add_argument('--arch', default='sm_86')
    p.add_argument('--checkpoint-every', type=int, help='override checkpoint interval for every job')
    a = p.parse_args()
    if a.checkpoint_every is not None and a.checkpoint_every <= 0:
        p.error('--checkpoint-every must be positive')
    try:
        selected = (selection(a.only) if a.only else set(GRIDS))-selection(a.skip)
    except ValueError as error:
        p.error(str(error))
    if not selected:
        p.error('no tests selected')
    root = a.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    with lock_campaign(root):
        state_path = root/'campaign_state.json'
        state = json.loads(state_path.read_text()) if state_path.exists() else {'schema': 1, 'jobs': {}}
        state['last_selection'] = sorted(selected)
        state['last_options'] = {'rr': a.rr, 'resume': a.resume, 'post_only': a.post_only}
        failure = None
        try:
            for job in jobs(selected, a.rr, a.checkpoint_every):
                output_root = root/job.test
                run = output_root/'rayleigh_benard'/job.run_id
                post = output_root/'post'/job.run_id
                description = job.description()
                entry = dict(description, run=str(run), post=str(post), status='not_run')
                previous = state['jobs'].get(job.run_id)
                if previous:
                    entry.update({k: previous[k] for k in ('status', 'error') if k in previous})
                state['jobs'][job.run_id] = entry
                atomic_json(state_path, state)
                manifest = run/'validation_run.json'
                receipt = run/'simulation_complete.json'
                if manifest.exists():
                    if json.loads(manifest.read_text()) != description:
                        raise RuntimeError(f'configuration mismatch for {run}; use a separate --root')
                    if not (a.resume or a.post_only):
                        raise RuntimeError(f'{run} already exists; use --resume or --post-only')
                else:
                    if a.post_only:
                        raise RuntimeError(f'no campaign manifest for {run}')
                    if run.exists() and any(run.iterdir()):
                        raise RuntimeError(f'refusing to adopt or overwrite unowned output: {run}')
                    run.mkdir(parents=True, exist_ok=True)
                    atomic_json(manifest, description)
                if receipt.exists():
                    if json.loads(receipt.read_text()) != description:
                        raise RuntimeError(f'incompatible completion receipt: {receipt}')
                    print(f'Completed simulation preserved: {job.run_id}', flush=True)
                elif a.post_only:
                    raise RuntimeError(f'{run} has no successful simulation receipt; resume it first')
                else:
                    entry['status'] = 'running'
                    atomic_json(state_path, state)
                    cmd = [sys.executable, ROOT/'scripts/run_rbc.py', '--case', str(job.case),
                           '--grid', str(job.n), '--rr', str(job.rr), '--precision', job.precision,
                           '--steps', str(job.steps), '--transient-steps', str(job.transient_steps),
                           '--sample', str(job.sample), '--checkpoint-every', str(job.checkpoint_every),
                           '--run-id', job.run_id, '--output-root', output_root, '--arch', a.arch]
                    if job.test in ('V1', 'V2', 'V3'):
                        cmd += ['--validation', job.test, '--test-grid']
                    if a.resume and any((run/'checkpoints').glob('step_*.chk')):
                        cmd += ['--continue']
                    execute(cmd, run/'execution.log')
                    if not (run/'checkpoints'/f'step_{job.steps:09d}.chk').exists():
                        raise RuntimeError(f'final checkpoint missing for {job.run_id}')
                    atomic_json(receipt, description)
                    entry['status'] = 'simulation_complete'
                    atomic_json(state_path, state)
                # A post-processing failure never removes the successful simulation receipt.
                entry['status'] = 'postprocessing'
                atomic_json(state_path, state)
                execute([sys.executable, ROOT/'postRayleigh.py', 'validation', run,
                         '--out', output_root/'post'], run/'postprocess.log')
                numerical = json.loads((post/'validation_result.json').read_text())
                entry.update(status='post_complete', numerical_status=numerical['status'])
                entry.pop('error', None)
                atomic_json(state_path, state)
                if numerical['status'] == 'failed':
                    raise RuntimeError(f'numerical validation failed for {job.run_id}; see {post}')
        except (Exception, KeyboardInterrupt) as error:
            failure = error
            if 'entry' in locals():
                if entry['status'] != 'post_complete':
                    entry['status'] = 'failed'
                entry['error'] = str(error) or type(error).__name__
            state['last_failure'] = str(error) or type(error).__name__
        else:
            state.pop('last_failure', None)
        finally:
            # Include explicitly unexecuted cases/grids in the consolidated report.
            for job in jobs(set(GRIDS), a.rr, a.checkpoint_every):
                state['jobs'].setdefault(job.run_id, dict(job.description(), status='not_run'))
            atomic_json(state_path, state)
            try:
                execute([sys.executable, ROOT/'postRayleigh.py', 'validation-summary', root,
                         '--out', root/'summary'], root/'summary.log')
            except (Exception, KeyboardInterrupt) as error:
                atomic_json(root/'summary_failure.json', {'status': 'failed', 'error': str(error),
                            'campaign_state': str(state_path)})
                failure = failure or error
        if failure:
            print(f'Campaign stopped: {failure}. Results retained under {root}', file=sys.stderr)
            return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
