"""One predeclared lower-edge DRN probe, sharing the existing three-worker limit."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from parallel_runs import save

HERE=Path(__file__).resolve().parent
out=HERE/'drn_lr_boundary'
out.mkdir(exist_ok=False)
status=dict(status='waiting_for_slot',pid=os.getpid(),arm='faulted-constant-1e-6',heartbeat=time.time())
save(out/'status.json',status)
while True:
    screen=json.loads((HERE/'drn_screen/status.json').read_text())
    if screen['status']=='failed':raise RuntimeError('Original screen failed; investigate before continuing.')
    if not screen.get('queued') and len(screen.get('active',[]))<3:break
    status['heartbeat']=time.time();save(out/'status.json',status);time.sleep(10)
target=HERE/'drn_screen/runs/faulted-constant-1e-6'
command=[sys.executable,str(HERE/'run_drn.py'),'--plan',str(HERE/'plan.with_lr_boundary.json'),
         '--arm',status['arm'],'--replica','array-1-write-1','--output',str(target)]
env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONUNBUFFERED='1',EBL_MNIST_ROOT='/home/filip/datasets/mnist')
with (out/'run.log').open('w') as log:
    child=subprocess.Popen(command,cwd=HERE.parent/'ibm-om-cell-aware-quantized',env=env,stdout=log,stderr=subprocess.STDOUT)
    status.update(status='running',child_pid=child.pid,command=command,result=str(target/'result.json'))
    while child.poll() is None:
        status['heartbeat']=time.time();save(out/'status.json',status);time.sleep(10)
if child.returncode:
    status.update(status='failed',exit_code=child.returncode,heartbeat=time.time());save(out/'status.json',status)
    raise RuntimeError('Lower-edge DRN probe failed; inspect retained log.')
result=json.loads((target/'result.json').read_text())
if result['status']!='complete' or result['epochs_completed']!=30 or not result['selected_replay_passed']:
    raise RuntimeError('Lower-edge DRN probe lacks a complete validated result.')
status.update(status='complete',heartbeat=time.time(),completed_at=time.time(),child_pid=None)
save(out/'status.json',status)
print('LOWER_RATE_PROBE_COMPLETE',flush=True)
