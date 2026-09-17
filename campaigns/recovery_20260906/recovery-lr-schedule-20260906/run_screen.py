"""Persistent sequential supervisor; numerical work remains in the native runners."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent

def save(path,payload):
    temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps(payload,indent=2)+'\n')
    temp.replace(path)

def main(args):
    plan=json.loads((HERE/'plan.json').read_text())
    section=plan[args.architecture]
    out=HERE/(args.architecture+'_screen')
    out.mkdir(exist_ok=False)
    (out/'logs').mkdir()
    env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONUNBUFFERED='1',EBL_DEFER_CURRENT_SIMULATIONS='1',EBL_MNIST_ROOT=args.mnist_root)
    status=dict(status='starting',architecture=args.architecture,pid=os.getpid(),expected_arms=[a['arm_id'] for a in section['arms']],completed=[],started_at=time.time(),heartbeat=time.time(),phase='screen')
    if args.architecture=='crossbar':
        code=HERE/'crossbar_code'
        command=[sys.executable,'-m','ebl','study','prepare','--plan',str(code/'studies'/(section['study_id']+'.json')),'--results-root',str(out/'results')]
        subprocess.run(command,cwd=code,env=env,check=True)
    else:
        code=HERE.parent/'ibm-om-cell-aware-quantized'
    save(out/'status.json',status)
    ordered=list(section['arms'])
    if args.architecture=='drn':
        ordered.sort(key=lambda a:(a['name']!='constant-1e-4',section['arms'].index(a)))
    for arm in ordered:
        arm_id=arm['arm_id']
        if args.architecture=='drn':
            command=[sys.executable,str(HERE/'run_drn.py'),'--plan',str(HERE/'plan.json'),'--arm',arm_id,'--replica','array-1-write-1','--output',str(out/'runs'/arm_id)]
            native_root=out/'runs'/arm_id
        else:
            inputs=Path(args.crossbar_inputs)
            native_root=out/'results'/section['study_id']/'runs'/arm_id
            command=[sys.executable,'-m','ebl','train','--config',str(code/'examples/mnist_analog_relu/long_kl_schedules_20260906'/(arm_id+'.json')),'--output-dir',str(native_root),'--teacher-weights',str(inputs/'teacher_weights.pt'),'--device-state',str(inputs/('hwa_healthy_p0.pt' if arm['condition']=='healthy' else 'hwa_faulted_p0.pt'))]
        status.update(status='running',current_arm=arm_id,command=command,artifact_root=str(native_root),heartbeat=time.time())
        save(out/'status.json',status)
        print('START',arm_id,flush=True)
        with (out/'logs'/(arm_id+'.log')).open('w') as log:
            child=subprocess.Popen(command,cwd=code,env=env,stdout=log,stderr=subprocess.STDOUT)
            status['child_pid']=child.pid
            while child.poll() is None:
                status['heartbeat']=time.time()
                save(out/'status.json',status)
                time.sleep(15)
        if child.returncode!=0:
            status.update(status='failed',exit_code=child.returncode,heartbeat=time.time())
            save(out/'status.json',status)
            raise RuntimeError(f'{arm_id} exited {child.returncode}; see its retained log.')
        result_files=list(native_root.glob('*/result.json')) if args.architecture=='crossbar' else [native_root/'result.json']
        if len(result_files)!=1 or not result_files[0].is_file():
            raise RuntimeError(f'Expected one terminal result for {arm_id}: {result_files}')
        result=json.loads(result_files[0].read_text())
        if result.get('status','complete')!='complete':
            raise RuntimeError(f'Non-complete result for {arm_id}')
        status['completed'].append(dict(arm=arm_id,result=str(result_files[0])))
        status.update(heartbeat=time.time(),child_pid=None)
        save(out/'status.json',status)
        print('COMPLETE',arm_id,flush=True)
    status.update(status='complete',completed_at=time.time(),heartbeat=time.time())
    save(out/'status.json',status)
    print('SCREEN_COMPLETE',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('architecture',choices=['drn','crossbar'])
    p.add_argument('--mnist-root',required=True)
    p.add_argument('--crossbar-inputs',default='')
    main(p.parse_args())
