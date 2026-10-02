"""Queue entrypoint and CPU collection for the two beta-adaptation arms."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from datetime import datetime, timezone

from experiments import reporting

SOURCE = Path(__file__).resolve().parents[1]
ROOT = SOURCE.parent


def validate(arm):
    import torch
    out=ROOT/'cells'/arm
    errors=reporting.validate_run(out)
    if errors:raise ValueError(errors)
    config=json.loads((out/'manifest.json').read_text())['resolved_config']
    rows=[json.loads(line) for line in (out/'metrics.jsonl').read_text().splitlines()]
    assert [r['epoch'] for r in rows]==list(range(1,11))
    assert all(r['steps']==r['epoch']*1407 and r['train_examples']==45000 and r['validation_examples']==5000 for r in rows)
    final=torch.load(out/'checkpoints/final_model.pt',map_location='cpu',weights_only=False)
    initial=torch.load(out/'checkpoints/initial_model.pt',map_location='cpu',weights_only=False)
    assert initial['epoch']==initial['scheduler']['last_epoch']==0 and not initial['optimizer']['state']
    assert final['epoch']==final['scheduler']['last_epoch']==10
    assert {int(v['step']) for v in final['optimizer']['state'].values()}=={14070}
    assert all(int(t)==14070 for n,t in final['model']['module'].items() if n.endswith('num_batches_tracked'))
    assert config['gradient_method']=='conv_eqprop_autograd_boundaries'
    policy=config['conv_eqprop']['policy'];assert policy==arm
    for epoch in range(11):
        observation=json.loads((out/'artifacts'/f'beta_boundary_e{epoch:03}.json').read_text())
        assert observation['epoch']==epoch and observation['examples']==256
        assert len(observation['blocks'])==3
        if epoch in [0,1,5,10]:
            assert len(observation['layers'])==8 and len(observation['batches'])==64
        else:assert observation['layers']==[]
        if arm=='fixed':assert observation['beta_state']['values']==config['conv_eqprop']['initial_betas']
    calibrations=list((out/'artifacts').glob('beta_calibration_e*.json'))
    if arm=='fixed':assert not calibrations
    else:
        assert len(calibrations)==9
        for epoch in range(2,11):
            cal=json.loads((out/'artifacts'/f'beta_calibration_e{epoch:03}.json').read_text())
            assert cal['complete'] and cal['beta_state']['calibrated_for_epoch']==epoch
            assert all(abs(b['trials'][-1]['relative_rms']/b['target']-1)<=.03 for b in cal['blocks'])
    result=json.loads((out/'result.json').read_text())
    assert result['completion']['full_epoch_budget'] and not result['completion']['official_test_read']
    assert result['terminal_metrics']['final_solver_audit_passed'] is None
    assert not (out/'artifacts/final_solver_audit.json').exists()
    return dict(arm=arm,validated=True,accuracy=rows[-1]['validation_accuracy'],loss=rows[-1]['validation_loss'])


def main(mode,arm):
    assert arm in {'fixed','epoch'}
    out=ROOT/'cells'/arm
    if mode=='validate':
        print(json.dumps(validate(arm)));return
    if mode=='collect':
        queue='/home/filip/.codex/skills/run-watch/scripts/gpu_queue.py'
        status=json.loads(subprocess.check_output([sys.executable,queue,'status',os.environ['RUN_WATCH_JOB_ID']],text=True))
        target=status['attempts'][-1]['target']
        if target!='local':
            out.mkdir(parents=True,exist_ok=True)
            remote=f'/home/filip/server_code/results/{ROOT.name}/cells/{arm}/'
            subprocess.run(['rsync','-a','-e','ssh -F /home/filip/.ssh/config -o BatchMode=yes',target+':'+remote,str(out)+'/'],check=True)
        result=validate(arm)
        reporting.atomic_write_json(ROOT/'analysis'/f'{arm}-collection.json',dict(result,job_id=os.environ['RUN_WATCH_JOB_ID'],target=target))
        subprocess.run([sys.executable,str(ROOT/'analysis/review.py')],check=True,
                       env=dict(os.environ,MPLCONFIGDIR='/tmp/cifar-beta-adaptation-mpl'))
        return
    assert mode=='run' and not out.exists()
    attempt=Path.home()/'.local/state/run-watch/queue-workers/attempts'/os.environ['RUN_WATCH_ATTEMPT']
    spec=json.loads((attempt/'spec.json').read_text());receipt=json.loads((attempt/'receipt.json').read_text())
    deadline=min(receipt['started_at']+spec['allowed_seconds'],spec.get('stop_at') or float('inf'))-120
    assert deadline-time.time()>2400
    config=SOURCE/os.environ.get('CIFAR_BETA_CONFIG_DIR','configs/cifar/beta_adaptation_20260929')/f'{arm}.json'
    command=[sys.executable,'-u','-m','experiments.train_cifar_l8_analog',str(config),
        '--mode','train','--output-dir',str(out),'--device','cuda','--require-gpu-resident',
        '--target',os.environ['CIFAR_QUEUE_TARGET'],'--assets',str(ROOT/'assets'),
        '--dataset-root',os.environ['CIFAR_DATA_ROOT'],'--source-identity',str(SOURCE/'source_identity.json'),
        '--stop-before',datetime.fromtimestamp(deadline,timezone.utc).isoformat(),
        '--initial-epoch-seconds','2400','--deadline-reserve-seconds','300']
    print('START_HYBRID_EQPROP_NO_SMOKE',arm,flush=True)
    os.execv(sys.executable,command)


if __name__=='__main__':main(*sys.argv[1:])
