"""Bounded local job queue with exit codes and semantic artifact checks."""
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from run_recovery import HERE, file_hash, write_json


def worker(architecture, condition, writer):
    name = f'{architecture}-{condition}-{writer}'
    output = HERE / 'runs' / name
    command = [sys.executable, str(HERE / 'run_recovery.py'), '--architecture', architecture, '--condition', condition, '--writer', writer, '--output', str(output), '--test']
    with (HERE / 'logs' / f'{name}.log').open('x') as log:
        child = subprocess.Popen(command, cwd=HERE, stdout=log, stderr=subprocess.STDOUT)
        receipt = {'name': name, 'pid': child.pid, 'command': command, 'started': time.time(), 'status_path': str(output / 'status.json'), 'result_path': str(output / 'result.json'), 'log': log.name, 'source_sha256': file_hash(HERE / 'run_recovery.py'), 'plan_sha256': file_hash(HERE / 'plan.json')}
        write_json(HERE / 'logs' / f'{name}.launch.json', receipt)
        code = child.wait()
    result_path = output / 'result.json'
    complete = code == 0 and result_path.is_file() and json.loads(result_path.read_text())['status'] == 'complete'
    receipt.update(exit_code=code, semantic_complete=complete, finished=time.time())
    write_json(HERE / 'logs' / f'{name}.exit.json', receipt)
    return receipt


if __name__ == '__main__':
    (HERE / 'runs').mkdir(exist_ok=True)
    (HERE / 'logs').mkdir(exist_ok=True)
    cases = [(architecture, condition, writer) for architecture in ('crossbar', 'drn') for condition in ('healthy', 'faulted') for writer in ('open_loop', 'closed_loop_pv')]
    write_json(HERE / 'launch.json', {'pid': os.getpid(), 'started': time.time(), 'expected_cases': ['-'.join(c) for c in cases], 'max_workers': 2, 'plan_sha256': file_hash(HERE / 'plan.json')})
    receipts = []
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(worker, *case) for case in cases]
        for future in as_completed(futures):
            receipt = future.result()
            receipts.append(receipt)
            write_json(HERE / 'queue_status.json', {'heartbeat': time.time(), 'completed': receipts, 'expected': len(cases)})
            print(receipt['name'], receipt['exit_code'], receipt['semantic_complete'], flush=True)
    complete = all(r['semantic_complete'] for r in receipts) and len(receipts) == len(cases)
    if complete:
        subprocess.run([sys.executable, str(HERE / 'labs/tools/analyze_recovery.py')], cwd=HERE, check=True)
    write_json(HERE / 'terminal.json', {'status': 'complete' if complete else 'failed', 'finished': time.time(), 'receipts': receipts})
    sys.exit(0 if complete else 1)
