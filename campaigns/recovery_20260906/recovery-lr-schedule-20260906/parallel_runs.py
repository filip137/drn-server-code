"""Bounded subprocess concurrency for independent, unchanged DRN commands."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent


def save(path, payload):
    path = Path(path)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(payload, indent=2) + '\n')
    temp.replace(path)


def run_jobs(jobs, *, out, status, cwd, env, workers):
    if not 1 <= workers <= 3:
        raise ValueError('Expected between one and three workers.')
    waiting = list(jobs)
    active = []
    failed = []
    while waiting or active:
        while waiting and len(active) < workers and not failed:
            job = waiting.pop(0)
            if job['result'].parent.exists():
                raise FileExistsError(job['result'].parent)
            log = (out / 'logs' / (job['label'] + '.log')).open('w')
            child = subprocess.Popen(job['command'], cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT)
            active.append((job, child, log))
            print('START', job['label'], child.pid, flush=True)
        for job, child, log in list(active):
            if child.poll() is None:
                continue
            log.close()
            active.remove((job, child, log))
            if child.returncode or not job['result'].exists():
                failed.append(dict(case=job['label'], exit_code=child.returncode))
                continue
            result = json.loads(job['result'].read_text())
            if result['status'] != 'complete' or result['epochs_completed'] != job['epochs'] or not result['selected_replay_passed']:
                failed.append(dict(case=job['label'], error='Terminal scientific validation failed.'))
                continue
            status['completed'].append(job.get('completed_record', job['label']))
            print('COMPLETE', job['label'], flush=True)
        status.update(status='running', heartbeat=time.time(),
                      active=[dict(case=j['label'], pid=c.pid, command=j['command'], result=str(j['result'])) for j, c, _ in active],
                      queued=[j['label'] for j in waiting], failures=failed)
        save(out / 'status.json', status)
        if failed and not active:
            status.update(status='failed', heartbeat=time.time())
            save(out / 'status.json', status)
            raise RuntimeError(f'One or more retained runs failed: {failed}')
        if waiting or active:
            time.sleep(10)


def main(args):
    plan = json.loads((HERE / 'plan.json').read_text())
    out = HERE / 'drn_screen'
    previous = json.loads((out / 'serial_supervisor_at_handoff.json').read_text())
    # An explicit operator-controlled boundary handoff must precede this entry.
    if not previous.get('handoff_verified_no_live_worker'):
        raise RuntimeError('Expected verified serial-supervisor handoff receipt.')
    status = dict(status='starting', architecture='drn', phase='screen', pid=os.getpid(),
                  expected_arms=[a['arm_id'] for a in plan['drn']['arms']], completed=[],
                  started_at=previous['started_at'], heartbeat=time.time(), workers=args.workers)
    jobs = []
    for arm in plan['drn']['arms']:
        case = arm['arm_id']
        result_path = out / 'runs' / case / 'result.json'
        record = dict(arm=case, result=str(result_path))
        if result_path.exists():
            result = json.loads(result_path.read_text())
            if result['status'] != 'complete' or result['arm'] != arm or result['epochs_completed'] != arm['epochs'] or not result['selected_replay_passed']:
                raise RuntimeError(f'Existing result is not a complete exact planned arm: {result_path}')
            status['completed'].append(record)
            continue
        jobs.append(dict(label=case, result=result_path, epochs=arm['epochs'], completed_record=record,
                         command=[sys.executable, str(HERE / 'run_drn.py'), '--plan', str(HERE / 'plan.json'),
                                  '--arm', case, '--replica', 'array-1-write-1', '--output', str(result_path.parent)]))
    env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
               PYTHONUNBUFFERED='1', EBL_MNIST_ROOT=args.mnist_root)
    run_jobs(jobs, out=out, status=status, cwd=HERE.parent / 'ibm-om-cell-aware-quantized', env=env, workers=args.workers)
    status.update(status='complete', heartbeat=time.time(), completed_at=time.time())
    save(out / 'status.json', status)
    print('SCREEN_COMPLETE', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--workers', type=int, default=3)
    p.add_argument('--mnist-root', required=True)
    main(p.parse_args())
