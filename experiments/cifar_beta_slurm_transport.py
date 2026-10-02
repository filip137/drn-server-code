"""One bounded Slurm owner: inspect, collect, validate and invoke frozen stage decisions."""
from datetime import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
REMOTE_ROOT = '/lustre/fsn1/projects/rech/fmu/ucy17uy/server_code/results/cifar-eqprop-legacy-v100-20260930-v1'
REMOTE_SOURCE = '/lustre/fswork/projects/rech/fmu/ucy17uy/server_code/cifar-eqprop-legacy-v100-source-20260930-v1'
DATA = '/lustre/fsn1/projects/rech/fmu/ucy17uy/server_code/results/cifar10-l8-schedule-renewal-e50-e70-seed0-20260924-v1/inputs/cifar-10-batches-py'
STOP = datetime.fromisoformat('2026-10-06T08:00:00+02:00').timestamp()
SSH = ['ssh','-F','/home/filip/.ssh/config','-o','BatchMode=yes','-o','ConnectTimeout=15','jean-zay']
LIVE = {'RUNNING','COMPLETING','CONFIGURING'}
QUEUED = {'PENDING','REQUEUED','SUSPENDED','REQUEUE_HOLD'}
PARENTS = {'1':None,'5':'1','10':'5','20':'10','30':'20','50':'30'}
CASES = ['native_bptt','beta_0p3','beta_1','beta_0p0001','beta_0p001','beta_0p01','beta_0p03','beta_0p1','beta_3']
BUDGETS = {'1':5400,'5':14400,'10':18000,'20':28800,'30':28800,'50':54000}


def atomic(path, value):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value,indent=2,sort_keys=True)+'\n');tmp.replace(path)


def read(path, default=None):
    return json.loads(path.read_text()) if path.exists() else default


def command(argv, **kw):
    try:
        return subprocess.run(argv,check=True,text=True,capture_output=True,timeout=180,**kw).stdout.strip()
    except subprocess.CalledProcessError as exc:
        detail=(exc.stderr or exc.stdout or '').strip()[-2000:]
        raise RuntimeError(f'{shlex.join(argv)} failed ({exc.returncode}): {detail}') from exc


def remote(argv):
    return command(SSH+[shlex.join(argv)])


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def inspect(jobs):
    """Read-only Slurm adapter, adapted from the existing CIFAR probe_slurm.py."""
    result={}
    # A completed job may disappear from squeue while remaining in sacct.
    # Querying it with -j then fails; a successful current-user snapshot does not.
    live_rows=[line.split('|') for line in command(['squeue','-h','--me','-o','%i|%T|%a|%R']).splitlines()]
    for key,job in jobs.items():
        jid=str(job['job_id'])
        rows=[line.split('|') for line in command(['sacct','-nPX','-j',jid,'-o','JobID,State,ExitCode,Account,ElapsedRaw']).splitlines() if line.split('|')[0]==jid]
        queued=[row for row in live_rows if row[0]==jid]
        if not rows and not queued:raise RuntimeError('No Slurm record for '+jid)
        if (rows and rows[0][3]!='fmu@v100') or (queued and queued[0][2]!='fmu@v100'):raise RuntimeError('Slurm account mismatch')
        state=(queued[0][1] if queued else rows[0][1]).split()[0].rstrip('+')
        cell=ROOT/'cells'/job['stage']/job['case']
        status=read(cell/'status.json',{})
        result[key]=dict(job_id=jid,state=state,alive=state in LIVE,queued=state in QUEUED,
            exit_code=rows[0][2] if rows else None,elapsed_seconds=int(rows[0][4]) if rows else None,
            status=status,metrics_bytes=(cell/'metrics.jsonl').stat().st_size if (cell/'metrics.jsonl').exists() else 0,
            result_exists=(cell/'result.json').exists(),rejected=(cell/'artifacts/scientific_rejection.json').exists(),
            reason=queued[0][3] if queued else '')
    return result


def completed(row):
    return not row['alive'] and not row['queued'] and ((row['state']=='COMPLETED' and row['exit_code']=='0:0' and row['result_exists']) or row['rejected'])


def operational_error(row):
    reason = str(row.get('reason', '')).lower()
    return row.get('state') in QUEUED and ('held' in reason or 'launch failed' in reason or 'launch_failed' in reason)


def verify_runtime(actual, expected):
    if any(actual.get(k)!=expected.get(k) for k in ('torch','cuda')) or actual.get('cudnn')!=8907:
        raise RuntimeError('Runtime mismatch: '+repr(actual))


def submit(stage,case):
    if stage not in PARENTS or case not in CASES:raise ValueError('Undeclared stage/case')
    if ((case=='beta_0p3' and stage!='1') or
            (case=='beta_1' and stage not in {'1','5'})):
        raise ValueError('Matched initial extension requires review before this stage')
    path=ROOT/'monitor/jobs.json';jobs=read(path,{})
    key=case+'-'+stage
    if key in jobs:
        if not jobs[key].get('job_id'):raise RuntimeError('Uncertain prior submission; reconcile before retry: '+key)
        return jobs[key]['job_id']
    if time.time()>=STOP:raise RuntimeError('Study deadline reached')
    if stage in {'10','20','30','50'}:
        parent_stage=PARENTS[stage]
        previous=read(ROOT/'analysis'/f'{case}-{parent_stage}-collection.json',{})
        elapsed=previous.get('elapsed_seconds',0)
        if previous.get('validated') is True and not previous.get('rejected') and elapsed>0:
            previous_epochs=int(parent_stage)-int(PARENTS[parent_stage])
            predicted=1.10*elapsed*(int(stage)-int(parent_stage))/previous_epochs
            available=min(BUDGETS[stage],STOP-time.time())-60
            if predicted>available:
                raise RuntimeError(f'Measured timing requires budget review for {key}: '
                                   f'{predicted:.0f}s including10% margin versus {available:.0f}s available')
    freeze=read(ROOT/'launch/identity.json')
    for rel,sha in freeze.items():
        if digest(ROOT/rel)!=sha:raise RuntimeError('Changed frozen transport: '+rel)
    if stage!='1':
        parent=ROOT/'cells'/PARENTS[stage]/case/'checkpoints/final_model.pt'
        observed=remote(['sha256sum',REMOTE_ROOT+'/'+str(parent.relative_to(ROOT))]).split()[0]
        if observed!=digest(parent):raise RuntimeError('Remote parent differs from validated checkpoint')
    script=key+('-timed.sbatch' if stage=='10' else '.sbatch')
    if stage=='10' and not (ROOT/'launch'/script).is_file():
        raise RuntimeError('Measured stage10 allocation not prepared: '+case)
    # Persist intent before sbatch: a lost acknowledgement cannot submit a duplicate.
    jobs[key]=dict(stage=stage,case=case,budget_seconds=BUDGETS[stage],submitted_at=time.time(),job_id=None)
    atomic(path,jobs)
    jid=remote(['sbatch','--parsable',REMOTE_ROOT+'/launch/'+script]).split(';')[0]
    if not jid.isdigit():raise RuntimeError('Invalid Slurm submission acknowledgement: '+jid)
    jobs[key]['job_id']=jid;atomic(path,jobs)
    return jid


def poll():
    with (ROOT/'monitor/owner.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        jobs=read(ROOT/'monitor/jobs.json',{})
        if any(not j.get('job_id') for j in jobs.values()):raise RuntimeError('Uncertain submission: inspect jobs.json')
        observations=json.loads(command(SSH+[shlex.join(['python3',REMOTE_ROOT+'/monitor/transport.py','inspect'])],
            input=json.dumps(jobs))) if jobs else {}
        sys.path.insert(0,str(ROOT/'source'))
        from experiments import run_cifar_beta_stages as wrapper
        wrapper.ROOT=ROOT
        previous=read(ROOT/'monitor/status.json',{}).get('jobs',{})
        failures=[]
        for key,row in observations.items():
            job=jobs[key];stage,case=job['stage'],job['case']
            signature=json.dumps([row['state'],row['status'].get('progress'),row['metrics_bytes']],sort_keys=True)
            old=previous.get(key,{})
            row['signature']=signature
            row['last_progress_at']=old.get('last_progress_at',time.time()) if signature==old.get('signature') else time.time()
            receipt_path=ROOT/'analysis'/f'{case}-{stage}-collection.json'
            if receipt_path.exists():
                receipt=read(receipt_path)
                if receipt.get('job_id')!=job['job_id'] or receipt.get('validated') is not True:raise RuntimeError('Collection identity mismatch')
                wrapper.verify_environment(receipt.get('environment'))
                continue
            if operational_error(row):
                failures.append(key+': operational hold '+row.get('reason', ''))
                continue
            if completed(row):
                local=ROOT/'cells'/stage/case;local.mkdir(parents=True,exist_ok=True)
                command(['rsync','-a','-e',shlex.join(SSH[:-1]),'jean-zay:'+REMOTE_ROOT+f'/cells/{stage}/{case}/',str(local)+'/'])
                receipt=wrapper.validate(stage,case)
                atomic(receipt_path,dict(receipt,target='jean-zay',job_id=job['job_id'],slurm=row))
            elif not row['alive'] and not row['queued']:
                failures.append(key+': terminal '+row['state']+' without validated completion')
            elif row['alive'] and time.time()-row['last_progress_at']>2700:
                failures.append(key+': no semantic progress for 2700 seconds')
        atomic(ROOT/'monitor/status.json',dict(observed_at=time.time(),jobs=observations,errors=failures))
        if failures:
            atomic(ROOT/'analysis/decision.json',dict(state='blocked',phase='slurm_operational_failure',requests=[],errors=failures))
            raise RuntimeError('; '.join(failures))
        wrapper.submit=submit
        wrapper.advance()


def run(stage,case):
    if stage not in PARENTS or case not in CASES:raise ValueError('Undeclared stage/case')
    if ((case=='beta_0p3' and stage!='1') or
            (case=='beta_1' and stage not in {'1','5'})):
        raise ValueError('Matched initial extension requires review before this stage')
    for rel,sha in read(ROOT/'launch/identity.json').items():
        if digest(ROOT/rel)!=sha:raise RuntimeError('Changed frozen transport: '+rel)
    import torch
    actual=dict(torch=str(torch.__version__),cuda=torch.version.cuda,cudnn=torch.backends.cudnn.version())
    verify_runtime(actual,read(ROOT/'inputs/study.json')['expected_environment'])
    allowed=min(BUDGETS[stage]-(time.time()-float(os.environ['CIFAR_ALLOCATION_STARTED_AT'])),STOP-time.time())-30
    if allowed<600:raise RuntimeError('Insufficient time before deadline')
    atomic(ROOT/'slurm'/f'{case}-{stage}-allocation.json',dict(job_id=os.environ['SLURM_JOB_ID'],environment=actual,started_at=time.time(),budget_seconds=BUDGETS[stage],maximum_seconds=allowed))
    argv=[sys.executable,'-u','-m','experiments.train_cifar_beta_stages','--root',str(ROOT),'--case',case,'--stage',stage,
          '--output-dir',str(ROOT/'cells'/stage/case),'--dataset-root',DATA,'--target','jean-zay','--maximum-seconds',str(allowed)]
    if stage!='1':argv+=['--resume-checkpoint',str(ROOT/'cells'/PARENTS[stage]/case/'checkpoints/final_model.pt')]
    os.execv(sys.executable,argv)


if __name__=='__main__':
    mode=sys.argv[1]
    if mode=='inspect':print(json.dumps(inspect(json.load(sys.stdin))))
    elif mode=='poll':
        try:
            poll()
            atomic(ROOT/'monitor/owner-status.json',dict(observed_at=time.time(),state='healthy',error=None))
        except Exception as exc:
            atomic(ROOT/'monitor/owner-status.json',dict(observed_at=time.time(),state='blocked',error=str(exc)))
            raise
    elif mode=='run':run(*sys.argv[2:])
    else:raise ValueError(mode)
