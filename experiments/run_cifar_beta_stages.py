"""Queue transport and predeclared continuation decisions for fresh beta tests."""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from experiments import reporting
from experiments.cifar_beta_stage_decision import decide

SOURCE = Path(__file__).resolve().parents[1]
ROOT = SOURCE.parent
QUEUE = '/home/filip/.codex/skills/run-watch/scripts/gpu_queue.py'
PARENTS = {'1': None, '5': '1', '10': '5', '20': '10', '30': '20', '50': '30'}


def study_settings():
    path = ROOT/'inputs/study.json'
    return json.loads(path.read_text()) if path.exists() else {}


def verify_environment(actual):
    expected = study_settings().get('expected_environment')
    if expected is not None and (not isinstance(actual, dict) or
            any(key not in expected or key not in actual or actual[key] != expected[key]
                for key in ('torch', 'cuda'))):
        raise ValueError(f'Runtime differs from study environment: {actual!r} versus {expected!r}')


def validate(stage, case):
    import torch
    out = ROOT/'cells'/stage/case
    manifest = json.loads((out/'manifest.json').read_text())
    environment = manifest.get('environment')
    verify_environment(environment)
    status = json.loads((out/'status.json').read_text())
    rejection = out/'artifacts/scientific_rejection.json'
    if rejection.exists():
        rejected = json.loads(rejection.read_text())
        assert rejected['kind'] == 'nonfinite' and status['state'] == 'failed'
        assert rejected['case'] == case and str(rejected['stage']) == stage
        assert not (out/'result.json').exists()
        return dict(validated=True, rejected=True, reason=rejected['reason'], stage=stage, case=case,
                    environment=environment)
    errors = reporting.validate_run(out)
    if errors:
        raise ValueError(errors)
    result = json.loads((out/'result.json').read_text())
    final = torch.load(out/'checkpoints/final_model.pt', map_location='cpu', weights_only=False)
    initial = torch.load(out/'checkpoints/initial_model.pt', map_location='cpu', weights_only=False)
    assert stage in PARENTS
    expected_epoch, expected_steps, next_batch = int(stage), int(stage)*1407, 0
    progress = final['staged_cursor']
    assert final['epoch'] == final['scheduler']['last_epoch'] == expected_epoch
    assert progress['completed_epoch'] == expected_epoch and progress['next_batch'] == next_batch
    assert progress['total_updates'] == expected_steps
    assert {int(v['step']) for v in final['optimizer']['state'].values()} == {expected_steps}
    assert final['native_epoch_resume_safe'] == (next_batch == 0)
    assert final['case_id'] == case
    assert final['case'] == json.loads((ROOT/'inputs/cases.json').read_text())[case]
    if 'beta_search' in final['case'] or 'beta_feedback' in final['case'] or final['case'].get('measure_training_rms', False):
        from experiments.cifar_conv_eqprop import validate_beta_search_record, validate_training_beta_state
        search = final['case'].get('beta_search')
        for checkpoint in (initial, final):
            validate_training_beta_state(checkpoint.get('conv_beta_state'), final['case'],
                                         checkpoint['staged_cursor']['total_updates'])
        beta = initial['conv_beta_state']['values']
        count_key = 'search_counts' if search is not None else 'clip_counts'
        counts = {key:list(value) for key, value in initial['conv_beta_state'][count_key].items()}
        update = initial['staged_cursor']['total_updates']
        for row in reporting.iter_metrics(out/'metrics.jsonl'):
            if row.get('stage') != 'training':
                continue
            update += 1
            assert row['cursor']['total_updates'] == row['beta_successful_updates'] == update
            assert len(row['training_rms']) == 3
            assert all(rms['free_rms'] > 0 and rms['relative_rms'] >= 0 for rms in row['training_rms'])
            if search is not None:
                assert row['starting_betas'] == beta and row['betas_next'] == row['betas_used']
                assert len(row['beta_search']) == len(row['betas_used']) == 3
                for i, record in enumerate(row['beta_search']):
                    validate_beta_search_record(record, beta[i], search['targets'][i], search)
                    assert record['selected_beta'] == row['betas_used'][i]
                    selected = record['trials'][record['chosen_index']]
                    assert all(value == selected[key] for key, value in row['training_rms'][i].items())
                    counts['pairs'][i] += record['pair_count']
                    counts['tolerance_misses'][i] += int(record['tolerance_miss'])
            else:
                assert row['betas_used'] == beta
                for key in ('factor', 'beta'):
                    flags = row[key+'_clipped']
                    assert len(flags) == 3 and all(type(flag) is bool for flag in flags)
                    counts[key] = [value + flag for value, flag in zip(counts[key], flags)]
            beta = row['betas_next']
        assert update == expected_steps and beta == final['conv_beta_state']['values']
        assert counts == final['conv_beta_state'][count_key]
    initializer = torch.load(ROOT/'inputs/initializer.pt', map_location='cpu', weights_only=False)
    for name, value in final['model']['module'].items():
        if name.endswith('num_batches_tracked'):
            assert int(value) == int(initializer['model']['module'][name]) + expected_steps
    if stage == '1':
        assert initial['staged_cursor']['total_updates'] == 0 and not initial['optimizer']['state']
        for group, values in initializer['model'].items():
            assert all(torch.equal(t,initial['model'][group][name]) for name,t in values.items())
    else:
        parent_stage = PARENTS[stage]
        parent = torch.load(ROOT/'cells'/parent_stage/case/'checkpoints/final_model.pt',map_location='cpu',weights_only=False)
        assert initial['staged_cursor'] == parent['staged_cursor']
        for group,values in parent['model'].items():
            assert all(torch.equal(t,initial['model'][group][name]) for name,t in values.items())
    measurement = result['terminal_metrics']['final']
    assert measurement['validation']['examples'] == 5000
    assert not result['completion']['official_test_read']
    return dict(validated=True, rejected=False, stage=stage, case=case,
                validation=measurement['validation'], train_cohort=measurement['train_cohort'],
                cursor=progress, runtime=manifest.get('runtime'), environment=environment,
                elapsed_seconds=result['terminal_metrics']['elapsed_seconds'])


def submit(stage, case):
    path = ROOT/'launch'/f'{case}-{stage}.json'
    spec = json.loads(path.read_text())
    existing = subprocess.run([sys.executable, QUEUE, 'status', spec['id']], capture_output=True, text=True)
    if existing.returncode == 0:
        return json.loads(existing.stdout)['job']['state']
    if 'Unknown job' not in existing.stderr:
        raise RuntimeError(existing.stderr)
    if stage != '1':
        parent_stage = PARENTS[stage]
        relative = Path('cells')/parent_stage/case/'checkpoints/final_model.pt'
        parent = ROOT/relative
        digest = reporting.sha256_file(parent)
        for target, placement in spec['placements'].items():
            remote_root = Path(placement['workdir']).parent
            destination = remote_root/relative
            if target != 'local':
                # Target paths are controlled repository paths, without shell metacharacters.
                subprocess.run(['ssh','-F','/home/filip/.ssh/config','-o','BatchMode=yes',target,
                                'mkdir','-p',str(destination.parent)],check=True)
                subprocess.run(['rsync','-a','-e','ssh -F /home/filip/.ssh/config -o BatchMode=yes',
                                str(parent),target+':'+str(destination)],check=True)
            placement['prepared_files'] = [p for p in placement['prepared_files'] if p['path'] != str(destination)]
            placement['prepared_files'].append(dict(path=str(destination),sha256=digest))
        reporting.atomic_write_json(path, spec)
    subprocess.run([sys.executable, QUEUE, 'submit', str(path)], check=True)
    return 'submitted'


def advance():
    """Called after validated collection; no persistent experiment service."""
    with (ROOT/'analysis/decision.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        records = {stage: {} for stage in PARENTS}
        for stage in records:
            for path in (ROOT/'analysis').glob(f'*-{stage}-collection.json'):
                row = json.loads(path.read_text())
                verify_environment(row.get('environment'))
                records[stage][row['case']] = row
        if study_settings().get('comparison_only') is True:
            cases = json.loads((ROOT/'inputs/cases.json').read_text())
            present = [records['1'][case] for case in cases if case in records['1']]
            state = ('blocked' if not cases or any(row.get('validated') is not True for row in present)
                     else 'complete' if len(present) == len(cases) else 'pending')
            reporting.atomic_write_json(ROOT/'analysis/decision.json', dict(
                state=state, phase='comparison_1', selected=list(cases), requests=[],
                records=records, requests_processed=True))
            return
        decision = decide(records, final_stage=study_settings().get('final_stage', '10'),
                          start_branch=study_settings().get('start_branch', 'initial'))
        reporting.atomic_write_json(ROOT/'analysis/decision.json',dict(decision,records=records))
        for request in decision['requests']:
            submit(request['stage'],request['case'])
        reporting.atomic_write_json(ROOT/'analysis/decision.json',dict(decision,records=records,requests_processed=True))
        if records['1']:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            fig, axes = plt.subplots(1,2,figsize=(11,4))
            for case in sorted({c for rows in records.values() for c in rows}):
                points = [(float(stage),rows[case]['validation'])
                          for stage,rows in records.items() if case in rows and not rows[case]['rejected']]
                if points:
                    axes[0].plot([p[0] for p in points],[100*p[1]['accuracy'] for p in points],'-o',label=case)
                    axes[1].plot([p[0] for p in points],[p[1]['loss'] for p in points],'-o',label=case)
            axes[0].set_ylabel('Validation accuracy (%)'); axes[1].set_ylabel('Validation CE')
            for ax in axes: ax.set_xlabel('Epoch milestone'); ax.legend(fontsize=7)
            fig.tight_layout();fig.savefig(ROOT/'analysis/milestones.jpg',dpi=160);plt.close(fig)


def main(mode,stage=None,case=None):
    if mode == 'advance':
        advance();return
    assert stage in PARENTS
    if study_settings().get('comparison_only') is True and stage != '1':
        raise ValueError('Comparison-only studies are bounded to one epoch')
    assert case in json.loads((ROOT/'inputs/cases.json').read_text())
    out = ROOT/'cells'/stage/case
    if mode == 'validate':
        print(json.dumps(validate(stage,case)));return
    if mode in ('collect', 'collect-only'):
        status = json.loads(subprocess.check_output([sys.executable,QUEUE,'status',os.environ['RUN_WATCH_JOB_ID']],text=True))
        target = status['attempts'][-1]['target']
        if target != 'local':
            out.mkdir(parents=True,exist_ok=True)
            remote = f'/home/filip/server_code/results/{ROOT.name}/cells/{stage}/{case}/'
            subprocess.run(['rsync','-a','-e','ssh -F /home/filip/.ssh/config -o BatchMode=yes',target+':'+remote,str(out)+'/'],check=True)
        receipt = validate(stage,case)
        reporting.atomic_write_json(ROOT/'analysis'/f'{case}-{stage}-collection.json',dict(receipt,target=target,job_id=os.environ['RUN_WATCH_JOB_ID']))
        if mode == 'collect':
            advance()
        return
    assert mode == 'run' and not out.exists()
    import torch
    verify_environment(dict(torch=str(torch.__version__), cuda=torch.version.cuda))
    attempt = Path.home()/'.local/state/run-watch/queue-workers/attempts'/os.environ['RUN_WATCH_ATTEMPT']
    spec = json.loads((attempt/'spec.json').read_text());receipt=json.loads((attempt/'receipt.json').read_text())
    deadline = min(receipt['started_at']+spec['allowed_seconds'],spec.get('stop_at') or float('inf'))
    remaining = deadline-time.time()-90
    if remaining < 600:raise RuntimeError('Insufficient remaining stage allowance')
    argv=[sys.executable,'-u','-m','experiments.train_cifar_beta_stages','--root',str(ROOT),
          '--case',case,'--stage',stage,'--output-dir',str(out),'--dataset-root',os.environ['CIFAR_DATA_ROOT'],
          '--target',os.environ['CIFAR_QUEUE_TARGET'],'--maximum-seconds',str(remaining)]
    if stage!='1':
        argv += ['--resume-checkpoint',str(ROOT/'cells'/PARENTS[stage]/case/'checkpoints/final_model.pt')]
    os.execv(sys.executable,argv)


if __name__ == '__main__':main(*sys.argv[1:])
