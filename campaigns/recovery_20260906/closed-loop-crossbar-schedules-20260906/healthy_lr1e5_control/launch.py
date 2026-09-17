"""Launch the one declared healthy P&V control with a durable exit receipt."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent

def write(name,value):
    path=HERE/name
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2)+'\n')
    tmp.replace(path)

if __name__=='__main__':
    gate=json.loads((ROOT/'smoke/gate.json').read_text())
    for name,digest in gate['source_hashes'].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest
    plan=json.loads((HERE/'plan.json').read_text())
    for source in plan['comparison_sources'].values():
        assert hashlib.sha256(Path(source['path']).read_bytes()).hexdigest()==source['sha256']
    command=[sys.executable,str(ROOT/'run_recovery.py'),'--plan',str(HERE/'plan.json'),
             '--condition','healthy','--schedule','constant','--writer','closed_loop_pv',
             '--epochs','30','--output',str(HERE/'run'),'--test']
    with (HERE/'run.log').open('x') as log:
        child=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        receipt=dict(supervisor_pid=os.getpid(),worker_pid=child.pid,command=command,
                     started=time.time(),status_path=str(HERE/'run/status.json'),
                     result_path=str(HERE/'run/result.json'),log=str(HERE/'run.log'),
                     plan_sha256=hashlib.sha256((HERE/'plan.json').read_bytes()).hexdigest())
        write('launch.json',receipt)
        code=child.wait()
    path=HERE/'run/result.json'
    complete=code==0 and path.exists() and json.loads(path.read_text())['status']=='complete'
    receipt.update(status='complete' if complete else 'failed',exit_code=code,
                   semantic_complete=complete,finished=time.time())
    write('terminal.json',receipt)
    sys.exit(0 if complete else 1)
