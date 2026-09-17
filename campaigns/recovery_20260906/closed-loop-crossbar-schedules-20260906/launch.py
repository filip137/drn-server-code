"""Bounded local CUDA queue; retain worker handles and semantic completion."""
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from run_recovery import HERE, file_hash, write_json


def worker(condition, schedule, writer, *, smoke=False):
    name = f'{condition}-{schedule}-{writer}'
    output = HERE / ('smoke' if smoke else 'runs') / name
    command = [sys.executable, str(HERE / 'run_recovery.py'),
               '--condition', condition, '--schedule', schedule,
               '--writer', writer, '--output', str(output), '--test']
    if smoke:
        command += ['--epochs', '2']
        if writer == 'closed_loop_pv':
            command += ['--maximum-batches', '128', '--evaluation-limit', '256']
    prefix = 'smoke-' if smoke else ''
    with (HERE / 'logs' / f'{prefix}{name}.log').open('x') as log:
        child = subprocess.Popen(command, cwd=HERE, stdout=log, stderr=subprocess.STDOUT)
        receipt = dict(name=name, pid=child.pid, command=command, started=time.time(),
                       log=log.name, status_path=str(output/'status.json'),
                       result_path=str(output/'result.json'), plan_sha256=file_hash(HERE/'plan.json'),
                       source_sha256=file_hash(HERE/'run_recovery.py'))
        write_json(HERE/'logs'/f'{prefix}{name}.launch.json', receipt)
        code = child.wait()
    path = output/'result.json'
    complete = code == 0 and path.exists() and json.loads(path.read_text())['status'] == 'complete'
    receipt.update(exit_code=code, semantic_complete=complete, finished=time.time())
    write_json(HERE/'logs'/f'{prefix}{name}.exit.json', receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--workers', type=int, default=2)
    args = parser.parse_args()
    schedules = ('exponential',) if args.smoke else ('constant', 'exponential')
    cases = [(condition, schedule, writer) for schedule in schedules
             for condition in ('healthy', 'faulted')
             for writer in ('open_loop', 'closed_loop_pv')]
    prefix = 'smoke_' if args.smoke else ''
    write_json(HERE/f'{prefix}launch.json', dict(pid=os.getpid(), started=time.time(),
               expected_cases=['-'.join(case) for case in cases], workers=args.workers))
    receipts = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(worker,*case,smoke=args.smoke) for case in cases]
        for future in as_completed(futures):
            receipt = future.result()
            receipts.append(receipt)
            write_json(HERE/f'{prefix}queue_status.json', dict(heartbeat=time.time(),
                       expected=len(cases), completed=receipts))
            print(receipt['name'], receipt['exit_code'], receipt['semantic_complete'], flush=True)
    complete = len(receipts)==len(cases) and all(r['semantic_complete'] for r in receipts)
    write_json(HERE/f'{prefix}terminal.json', dict(status='complete' if complete else 'failed',
               finished=time.time(), receipts=receipts))
    return 0 if complete else 1


if __name__ == '__main__':
    sys.exit(main())
