"""Finish the two DRN P&V arms after verified operational acceleration."""
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from run_recovery import HERE, file_hash, write_json


def worker(condition):
    name = f'drn-{condition}-closed_loop_pv'
    output = HERE / 'runs' / name
    command = [sys.executable, str(HERE/'run_fast_recovery.py'), '--condition', condition, '--output', str(output), '--test']
    with (HERE/'logs'/f'{name}-accelerated.log').open('x') as log:
        child = subprocess.Popen(command, cwd=HERE, stdout=log, stderr=subprocess.STDOUT)
        receipt = {'name': name, 'pid': child.pid, 'command': command, 'started': time.time(), 'log': log.name, 'result_path': str(output/'result.json'), 'wrapper_sha256': file_hash(HERE/'run_fast_recovery.py'), 'kernel_sha256': file_hash(HERE/'fast_drn_pulse.py'), 'parity_sha256': file_hash(HERE/'smoke/full-controller-parity.json')}
        write_json(HERE/'logs'/f'{name}-accelerated.launch.json', receipt)
        code = child.wait()
    complete = code == 0 and (output/'result.json').exists() and (output/'acceleration.json').exists()
    receipt.update(exit_code=code, semantic_complete=complete, finished=time.time())
    write_json(HERE/'logs'/f'{name}-accelerated.exit.json', receipt)
    return receipt


if __name__ == '__main__':
    preserved=[]
    for p in sorted((HERE/'runs').glob('*/result.json')):
        r=json.loads(p.read_text())
        assert r['status']=='complete'
        preserved.append({'path':str(p),'sha256':file_hash(p)})
    assert len(preserved)==6
    write_json(HERE/'accelerated_launch.json',{'pid':os.getpid(),'started':time.time(),'preserved_completed_results':preserved,'conditions':['healthy','faulted'],'plan_sha256':file_hash(HERE/'plan.json')})
    receipts=[]
    with ThreadPoolExecutor(max_workers=2) as pool:
        for future in as_completed([pool.submit(worker,c) for c in ('healthy','faulted')]):
            receipt=future.result(); receipts.append(receipt)
            write_json(HERE/'accelerated_queue_status.json',{'heartbeat':time.time(),'completed':receipts})
            print(receipt,flush=True)
    complete=all(r['semantic_complete'] for r in receipts)
    if complete:
        subprocess.run([sys.executable,str(HERE/'labs/tools/analyze_recovery.py')],check=True,cwd=HERE)
        subprocess.run([sys.executable,str(HERE/'labs/tools/analyze_writer_state.py')],check=True,cwd=HERE)
    write_json(HERE/'terminal.json',{'status':'complete' if complete else 'failed','finished':time.time(),'receipts':receipts,'preserved_completed_results':preserved})
    sys.exit(0 if complete else 1)
