"""Read-only local queue/study summary; scheduling and monitoring stay with run-watch."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time


QUEUE = Path('/home/filip/.codex/skills/run-watch/scripts/gpu_queue.py')
DEFAULT_PREFIXES = ('cifar-beta-fresh-', 'cifar-eqprop-')
TERMINAL_JOBS = {'complete', 'cancelled', 'expired', 'failed', 'rejected'}
TERMINAL_DECISIONS = {'complete', 'search_exhausted', 'quality_stop'}


def utc(timestamp):
    return datetime.fromtimestamp(timestamp, timezone.utc).isoformat()


def short(value, length=240):
    return ' '.join(str(value or '').split())[:length]


def number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def age(timestamp, now):
    return max(0., now - timestamp) if number(timestamp) else None


def object_value(value):
    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, dict):
        raise ValueError('Expected a JSON object')
    return value


def queue_read(*arguments):
    # These are the only permitted queue commands; no transport/probe/launch calls.
    if arguments not in [('list',)] and not (len(arguments) == 2 and arguments[0] == 'status'):
        raise ValueError('Only list and per-job status are allowed')
    result = subprocess.run([sys.executable, str(QUEUE), *arguments],
                            capture_output=True, text=True, timeout=30)
    if result.returncode:
        raise RuntimeError(short(result.stderr or result.stdout))
    return object_value(json.loads(result.stdout))


def read_json(path):
    if path.stat().st_size > 4 * 1024 * 1024:
        raise ValueError('JSON summary input exceeds4 MiB')
    return object_value(json.loads(path.read_text()))


def summarize_job(row, status, now):
    job = status.get('job', row)
    attempts = status.get('attempts', [])
    if not isinstance(job, dict) or not isinstance(attempts, list):
        raise ValueError('Malformed queue job status')
    latest = max(attempts, key=lambda a: a.get('issued', 0)) if attempts else {}
    observed = object_value(latest['receipt']) if latest.get('receipt') else {}
    receipt = object_value(observed.get('receipt', {}))
    outer_progress = object_value(observed.get('progress', {}))
    progress = outer_progress.get('progress', outer_progress)
    if not isinstance(progress, dict):
        raise ValueError('Malformed attempt progress')
    keep = ('stage', 'epoch', 'epochs', 'updates', 'total_updates', 'batch', 'batches',
            'examples', 'steps', 'completed_epoch', 'next_epoch', 'next_batch', 'total_updates')
    compact = {key:progress[key] for key in keep if key in progress}
    if isinstance(progress.get('cursor'), dict):
        compact['cursor'] = {key:progress['cursor'][key] for key in
            ('completed_epoch', 'next_epoch', 'next_batch', 'examples_in_epoch', 'total_updates')
            if key in progress['cursor']}
    reason = job.get('reason') or observed.get('error') or receipt.get('error') or ''
    return dict(id=row['id'], state=job.get('state', 'unknown'), reason=short(reason),
                target=latest.get('target'), gpu_name=latest.get('gpu_name'),
                attempt_state=latest.get('state'), attempt_token=latest.get('token'),
                attempt_age_seconds=age(receipt.get('started_at', latest.get('issued')), now),
                progress_age_seconds=age(latest.get('progress_at'), now),
                progress=compact, used_seconds=job.get('used_seconds'),
                job_update_age_seconds=age(job.get('updated'), now))


def study_summary(root):
    summary = dict(root=str(root), decision=None, receipts=[], errors=[])
    try:
        decision = read_json(root / 'analysis/decision.json')
        summary['decision'] = {key:decision[key] for key in ('state', 'phase', 'selected', 'requests') if key in decision}
        if not isinstance(decision.get('state'), str):
            raise ValueError('Decision has no state')
    except (OSError, ValueError, TypeError) as exc:
        summary['errors'].append('decision.json: ' + short(exc))
    for path in sorted((root / 'analysis').glob('*-collection.json')):
        try:
            row = read_json(path)
            receipt = {key:row[key] for key in ('case', 'stage', 'validated', 'rejected', 'target') if key in row}
            receipt['file'] = path.name
            if row.get('reason'):
                receipt['reason'] = short(row['reason'])
            metrics = row.get('validation')
            if not isinstance(metrics, dict):
                metrics = row.get('final', {}).get('validation') if isinstance(row.get('final'), dict) else None
            if isinstance(metrics, dict):
                receipt['validation'] = {k:metrics[k] for k in ('accuracy', 'loss', 'examples') if number(metrics.get(k))}
            summary['receipts'].append(receipt)
        except (OSError, ValueError, TypeError) as exc:
            summary['errors'].append(path.name + ': ' + short(exc))
    if (root / 'launch/transport.py').exists() and (root / 'monitor/jobs.json').exists():
        try:
            owner = read_json(root / 'monitor/owner-status.json')
            heartbeat_age = age(owner.get('observed_at'), time.time())
            summary['slurm_owner'] = dict(state=owner.get('state'), heartbeat_age_seconds=heartbeat_age,
                                         error=owner.get('error'))
            if owner.get('state') != 'healthy' or heartbeat_age is None or heartbeat_age > 900:
                summary['errors'].append('Slurm owner failed or stale: ' + short(owner))
            status = read_json(root / 'monitor/status.json')
            summary['slurm_jobs'] = {key:{name:row.get(name) for name in
                ('job_id', 'state', 'elapsed_seconds', 'status', 'metrics_bytes', 'last_progress_at', 'reason')}
                for key,row in status.get('jobs', {}).items()}
            summary['errors'].extend('Slurm: ' + short(error) for error in status.get('errors', []))
            for key,row in summary['slurm_jobs'].items():
                reason = str(row.get('reason', '')).lower()
                if row.get('state') == 'PENDING' and ('held' in reason or 'launch failed' in reason):
                    summary['errors'].append('Slurm ' + key + ': ' + str(row.get('reason')))
        except (OSError, ValueError, TypeError) as exc:
            summary['errors'].append('Slurm owner: ' + short(exc))
    return summary


def summarize(listing, jobs, studies, now, errors=()):
    alerts = [dict(kind='read_error', message=short(error)) for error in errors]
    service = listing.get('service')
    if not isinstance(service, dict):
        service = {}
    service_age = age(service.get('observed_at'), now)
    if service_age is None:
        alerts.append(dict(kind='service_unknown', message='Queue service heartbeat is missing or malformed'))
    elif service_age > 180:
        alerts.append(dict(kind='service_stale', message=f'Queue heartbeat is {service_age:.0f}s old'))
    if service.get('alive') is False:
        alerts.append(dict(kind='service_not_alive', message='Queue service process is not alive'))
    for job in jobs:
        if job['state'] in ('attention', 'blocked', 'failed', 'expired', 'rejected'):
            alerts.append(dict(kind='job_attention', job_id=job['id'],
                               message=f"{job['state']}: {job['reason']}"))
        if job['state'] == 'running':
            progress_age = job['progress_age_seconds']
            if progress_age is None or progress_age > 1800:
                alerts.append(dict(kind='progress_stale' if progress_age is not None else 'progress_unknown',
                    job_id=job['id'], message='Running job progress age is ' +
                    (f'{progress_age:.0f}s' if progress_age is not None else 'unavailable')))
    for study in studies:
        for error in study['errors']:
            alerts.append(dict(kind='study_read_error', study=study['root'], message=error))
        if (study.get('decision') or {}).get('state') == 'blocked':
            alerts.append(dict(kind='study_blocked', study=study['root'], message='Study decision is blocked'))
    active = [job['id'] for job in jobs if job['state'] not in TERMINAL_JOBS]
    terminal = bool(studies) and all((s.get('decision') or {}).get('state') in TERMINAL_DECISIONS
                                    and not s['errors'] for s in studies)
    complete = bool(not active and terminal and not errors and service_age is not None
                    and service_age <= 180 and service.get('alive') is not False)
    return dict(timestamp_utc=utc(now), queue_service=dict(alive=service.get('alive'),
        heartbeat_age_seconds=service_age,
        observed_at=service.get('observed_at') if number(service.get('observed_at')) else None,
        admission_enabled=service.get('admission_enabled')),
        jobs=jobs, job_counts=dict(Counter(job['state'] for job in jobs)), studies=studies,
        alerts=alerts, active_job_ids=active, operational_complete=complete,
        completion_note='Operational queue/study status only; this report does not establish scientific success.',
        monitoring_owner='run-watch queue for lab jobs; dedicated Slurm timer for legacy V100', summary_cadence_seconds=1800)


def markdown(report):
    lines = [f"CIFAR EqProp queue summary — {report['timestamp_utc']}", '',
             'Operational status: ' + ('complete' if report['operational_complete'] else 'in progress or unresolved'),
             report['completion_note'], '',
             f"Queue heartbeat age: {report['queue_service']['heartbeat_age_seconds']} seconds.",
             'Monitoring owners: run-watch lab queue; dedicated legacy V100 Slurm timer.', '']
    if report['alerts']:
        lines.extend(['Attention:', ''])
        lines.extend('- ' + short(a.get('job_id') or a.get('study')) + ': ' + a['message'] for a in report['alerts'])
        lines.append('')
    lines.extend(['| Job | State | Target / GPU | Progress | Progress age | Reason |',
                  '| --- | --- | --- | --- | --- | --- |'])
    for job in report['jobs']:
        seconds = job['progress_age_seconds']
        values = [job['id'], job['state'], f"{job['target'] or '—'} / {job['gpu_name'] or '—'}",
                  json.dumps(job['progress'], sort_keys=True), '—' if seconds is None else f'{seconds:.0f}s', job['reason']]
        lines.append('| ' + ' | '.join(short(value, 700).replace('|', '\\|') for value in values) + ' |')
    for study in report['studies']:
        lines.extend(['', str(study['root']), '',
                      'Decision: ' + json.dumps(study['decision'], sort_keys=True), ''])
        if 'slurm_owner' in study:
            lines.append('Slurm owner: ' + json.dumps(study['slurm_owner'], sort_keys=True))
            for key,row in study.get('slurm_jobs', {}).items():
                lines.append('- Slurm ' + key + ': ' + json.dumps(row, sort_keys=True))
        for row in study['receipts']:
            metrics = row.get('validation', {})
            lines.append(f"- {row.get('case', row['file'])}, stage {row.get('stage', '?')}: "
                         f"validated={row.get('validated')}, rejected={row.get('rejected')}, "
                         f"accuracy={metrics.get('accuracy')}, CE={metrics.get('loss')}")
    return '\n'.join(lines) + '\n'


def atomic_text(path, text):
    temporary = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    try:
        with temporary.open('w') as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--study-root', type=Path, action='append', required=True)
    parser.add_argument('--job-prefix', action='append')
    args = parser.parse_args()
    prefixes = tuple(args.job_prefix or DEFAULT_PREFIXES)
    errors, jobs = [], []
    try:
        listing = queue_read('list')
        if not isinstance(listing.get('jobs'), list):
            raise ValueError('Queue list has no jobs array')
    except (OSError, ValueError, TypeError, RuntimeError, subprocess.SubprocessError) as exc:
        listing = {}
        errors.append('Queue list: ' + short(exc))
    for row in listing.get('jobs', []):
        if not isinstance(row, dict) or not isinstance(row.get('id'), str):
            errors.append('Malformed queue list row')
            continue
        if not row['id'].startswith(prefixes):
            continue
        try:
            jobs.append(summarize_job(row, queue_read('status', row['id']), time.time()))
        except (OSError, ValueError, TypeError, RuntimeError, subprocess.SubprocessError) as exc:
            errors.append(row['id'] + ': ' + short(exc))
            jobs.append(summarize_job(row, {}, time.time()))
    studies = [study_summary(root.expanduser().resolve()) for root in args.study_root]
    report = summarize(listing, jobs, studies, time.time(), errors)
    report['job_prefixes'] = list(prefixes)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + '\n'
    atomic_text(args.output_dir / 'latest.json', payload)
    atomic_text(args.output_dir / 'latest.md', markdown(report))
    with (args.output_dir / 'history.jsonl').open('a') as stream:
        stream.write(json.dumps(report, sort_keys=True, allow_nan=False) + '\n')
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps(dict(timestamp_utc=report['timestamp_utc'], jobs=len(jobs), alerts=len(report['alerts']),
                          operational_complete=report['operational_complete'], output_dir=str(args.output_dir))))


if __name__ == '__main__':
    main()
