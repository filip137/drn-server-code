"""CPU-only milestone transport checks; no queue or GPU commands are executed."""
import copy
import json
from types import SimpleNamespace

import pytest
import torch

from experiments import run_cifar_beta_stages as runner


ENVIRONMENT = {'torch': '2.11.0+cu128', 'cuda': '12.8'}


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def root(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, 'ROOT', tmp_path)
    (tmp_path/'analysis').mkdir()
    (tmp_path/'inputs').mkdir()
    return tmp_path


def checkpoint(epoch):
    return dict(epoch=epoch, scheduler={'last_epoch': epoch},
                staged_cursor=dict(completed_epoch=epoch, next_batch=0, total_updates=epoch*1407),
                optimizer={'state': {'weight': {'step': epoch*1407}} if epoch else {}},
                model={'module': {'weight': torch.tensor(float(epoch)),
                                  'bn.num_batches_tracked': torch.tensor(epoch*1407)}},
                native_epoch_resume_safe=True, case_id='beta_1', case={'beta': 1.})


def prepare_stage(root, stage):
    out = root/'cells'/stage/'beta_1'
    (out/'checkpoints').mkdir(parents=True)
    write_json(out/'manifest.json', {'environment': ENVIRONMENT, 'runtime': {'target': 'trex'}})
    write_json(out/'status.json', {'state': 'complete'})
    write_json(out/'result.json', {'completion': {'official_test_read': False},
        'terminal_metrics': {'final': {'validation': {'examples': 5000, 'accuracy': .6, 'loss': 1.},
                                      'train_cohort': {}}, 'elapsed_seconds': 1.}})
    write_json(root/'inputs/cases.json', {'beta_1': {'beta': 1.}})
    torch.save(checkpoint(0), root/'inputs/initializer.pt')
    parent_stage = runner.PARENTS[stage]
    initial = checkpoint(int(parent_stage) if parent_stage else 0)
    torch.save(initial, out/'checkpoints/initial_model.pt')
    torch.save(checkpoint(int(stage)), out/'checkpoints/final_model.pt')
    if parent_stage:
        parent = root/'cells'/parent_stage/'beta_1/checkpoints/final_model.pt'
        parent.parent.mkdir(parents=True)
        torch.save(initial, parent)
    return out


@pytest.mark.parametrize('stage', ['1', '5', '10', '20', '30', '50'])
def test_validation_all_milestones_preserves_environment_and_parent_identity(root, monkeypatch, stage):
    prepare_stage(root, stage)
    monkeypatch.setattr(runner.reporting, 'validate_run', lambda _: [])
    write_json(root/'inputs/study.json', {'final_stage': '50', 'expected_environment': ENVIRONMENT})
    receipt = runner.validate(stage, 'beta_1')
    assert receipt['environment'] == ENVIRONMENT
    assert receipt['cursor']['total_updates'] == int(stage)*1407


def test_validation_rejects_wrong_parent_model(root, monkeypatch):
    out = prepare_stage(root, '30')
    monkeypatch.setattr(runner.reporting, 'validate_run', lambda _: [])
    initial = checkpoint(20)
    initial['model']['module']['weight'] += 1
    torch.save(initial, out/'checkpoints/initial_model.pt')
    with pytest.raises(AssertionError):
        runner.validate('30', 'beta_1')


def test_instrumented_collection_requires_complete_beta_telemetry(root, monkeypatch):
    from experiments.cifar_conv_eqprop import initial_training_beta_state
    out = prepare_stage(root, '1')
    monkeypatch.setattr(runner.reporting, 'validate_run', lambda _: [])
    case = {'method': 'hybrid_eqprop', 'betas': [1., 2., 3.], 'measure_training_rms': True}
    write_json(root/'inputs/cases.json', {'beta_1': case})
    for name, updates in [('initial_model.pt', 0), ('final_model.pt', 1407)]:
        path = out/'checkpoints'/name
        state = torch.load(path, weights_only=False)
        state['case'] = case
        state['conv_beta_state'] = initial_training_beta_state(case)
        state['conv_beta_state']['successful_updates'] = updates
        torch.save(state, path)
    rows = [json.dumps(dict(stage='training', cursor={'total_updates': update},
        beta_successful_updates=update, betas_used=case['betas'], betas_next=case['betas'],
        factor_clipped=[False]*3, beta_clipped=[False]*3,
        training_rms=[{'free_rms': 1., 'relative_rms': .1}]*3)) for update in range(1, 1408)]
    path = out/'metrics.jsonl'
    path.write_text('\n'.join(rows)+'\n')
    assert runner.validate('1', 'beta_1')['validated']
    path.write_text('\n'.join(rows[:100]+rows[101:])+'\n')
    with pytest.raises(AssertionError):
        runner.validate('1', 'beta_1')


def test_current_batch_collection_checks_start_selected_and_pair_counters(root, monkeypatch):
    from experiments.cifar_conv_eqprop import initial_training_beta_state, next_training_beta_state, select_current_beta
    out = prepare_stage(root, '1')
    monkeypatch.setattr(runner.reporting, 'validate_run', lambda _: [])
    spec = dict(targets=[1., 1., 1.], relative_tolerance=.25, max_evaluations=5,
                beta_bounds=[1e-8, 1e4], factor_bounds=[.001, 1000.], rms_floor=1e-12)
    case = dict(method='hybrid_eqprop', betas=[1., 1., 1.], beta_search=spec)
    write_json(root/'inputs/cases.json', {'beta_1': case})
    state = initial_training_beta_state(case)
    initial = torch.load(out/'checkpoints/initial_model.pt', weights_only=False)
    initial.update(case=case, conv_beta_state=copy.deepcopy(state))
    torch.save(initial, out/'checkpoints/initial_model.pt')
    def measure(beta):
        return [torch.tensor(beta)], dict(relative_rms=beta/2, absolute_rms=beta/2, free_rms=1., elements=1)
    rows = []
    for update in range(1, 1408):
        measurements = [select_current_beta(measure, beta, 1., spec)[1] for beta in state['values']]
        state, telemetry = next_training_beta_state(state, case, measurements)
        rows.append(dict(stage='training', cursor={'total_updates': update}, **telemetry))
    final = torch.load(out/'checkpoints/final_model.pt', weights_only=False)
    final.update(case=case, conv_beta_state=state)
    torch.save(final, out/'checkpoints/final_model.pt')
    metrics = out/'metrics.jsonl'
    metrics.write_text('\n'.join(json.dumps(row) for row in rows)+'\n')
    assert runner.validate('1', 'beta_1')['validated']
    assert rows[0]['starting_betas'] != rows[0]['betas_used']
    assert state['search_counts']['pairs'] == [1408] * 3
    rows[0]['beta_search'][0]['chosen_index'] = 0  # Would associate the selected gradient with the wrong pair.
    metrics.write_text('\n'.join(json.dumps(row) for row in rows)+'\n')
    with pytest.raises(ValueError, match='selection'):
        runner.validate('1', 'beta_1')


@pytest.mark.parametrize('rejected', [False, True])
def test_wrong_environment_blocks_success_and_numerical_rejection(root, monkeypatch, rejected):
    out = prepare_stage(root, '20')
    write_json(root/'inputs/study.json', {'expected_environment': {'torch': '2.5', 'cuda': '12.4'}})
    if rejected:
        write_json(out/'artifacts/scientific_rejection.json', {'kind': 'nonfinite'})
    with pytest.raises(ValueError, match='Runtime differs'):
        runner.validate('20', 'beta_1')


def test_rejection_receipt_also_preserves_environment(root):
    out = prepare_stage(root, '20')
    (out/'result.json').unlink()
    write_json(out/'status.json', {'state': 'failed'})
    write_json(out/'artifacts/scientific_rejection.json',
               {'kind': 'nonfinite', 'case': 'beta_1', 'stage': '20', 'reason': 'loss'})
    receipt = runner.validate('20', 'beta_1')
    assert receipt['rejected'] and receipt['environment'] == ENVIRONMENT


def test_optional_study_settings_keep_existing_ten_epoch_default(root, monkeypatch):
    seen = []
    monkeypatch.setattr(runner, 'decide', lambda records, final_stage, start_branch: (
        seen.append((records, final_stage, start_branch)) or dict(requests=[], state='pending')))
    runner.advance()
    assert seen[0][1:] == ('10', 'initial') and tuple(seen[0][0]) == tuple(runner.PARENTS)
    write_json(root/'inputs/study.json', {'final_stage': '50', 'start_branch': 'fallback'})
    runner.advance()
    assert seen[1][1:] == ('50', 'fallback')


def test_advance_blocks_stale_mismatched_receipt_before_decision(root, monkeypatch):
    write_json(root/'inputs/study.json', {'expected_environment': ENVIRONMENT})
    write_json(root/'analysis/beta_1-20-collection.json',
               {'case': 'beta_1', 'environment': {'torch': '2.5', 'cuda': '12.4'}})
    monkeypatch.setattr(runner, 'decide', lambda *a, **kw: pytest.fail('must not decide'))
    with pytest.raises(ValueError, match='Runtime differs'):
        runner.advance()


def test_fifty_epoch_submission_stages_exact_thirty_epoch_parent_once(root, monkeypatch):
    parent = root/'cells/30/beta_1/checkpoints/final_model.pt'
    parent.parent.mkdir(parents=True)
    parent.write_bytes(b'exact parent checkpoint')
    spec_path = root/'launch/beta_1-50.json'
    write_json(spec_path, {'id': 'fresh-beta1-50', 'placements': {'local': {
        'workdir': str(root/'source'), 'prepared_files': []}}})
    calls = []

    def command(argv, **kwargs):
        calls.append(argv)
        return SimpleNamespace(returncode=1, stderr='Unknown job')

    monkeypatch.setattr(runner.subprocess, 'run', command)
    assert runner.submit('50', 'beta_1') == 'submitted'
    spec = json.loads(spec_path.read_text())
    assert spec['placements']['local']['prepared_files'] == [
        {'path': str(parent), 'sha256': runner.reporting.sha256_file(parent)}]
    original = copy.deepcopy(spec)
    calls.clear()
    monkeypatch.setattr(runner.subprocess, 'run', lambda argv, **kwargs: (
        calls.append(argv) or SimpleNamespace(returncode=0, stdout='{"job":{"state":"running"}}')))
    assert runner.submit('50', 'beta_1') == 'running'
    assert len(calls) == 1 and json.loads(spec_path.read_text()) == original


@pytest.mark.parametrize('receipts,expected', [
    ({}, 'pending'),
    ({'free_t': {'validated': True}}, 'pending'),
    ({'free_t': {'validated': False}}, 'blocked'),
    ({'free_t': {'validated': True}, 'native': {'validated': True}}, 'complete'),
    ({'free_t': {'validated': True, 'rejected': True}, 'native': {'validated': True}}, 'complete'),
])
def test_comparison_only_never_decides_promotions_or_mutates_queue(root, monkeypatch, receipts, expected):
    write_json(root/'inputs/study.json', {'comparison_only': True})
    write_json(root/'inputs/cases.json', {'free_t': {'method': 'free_t_bptt'},
                                         'native': {'method': 'native_bptt'}})
    for case, receipt in receipts.items():
        write_json(root/f'analysis/{case}-1-collection.json', dict(receipt, case=case))
    monkeypatch.setattr(runner, 'decide', lambda *a, **kw: pytest.fail('no promotion decisions'))
    monkeypatch.setattr(runner, 'submit', lambda *a, **kw: pytest.fail('no queue submissions'))
    monkeypatch.setattr(runner.subprocess, 'run', lambda *a, **kw: pytest.fail('no commands'))
    runner.advance()
    decision = json.loads((root/'analysis/decision.json').read_text())
    assert decision['state'] == expected and decision['requests'] == []
    assert decision['phase'] == 'comparison_1'


def test_comparison_only_rejects_longer_runs_before_queue_access(root, monkeypatch):
    write_json(root/'inputs/study.json', {'comparison_only': True})
    monkeypatch.setattr(runner.subprocess, 'run', lambda *a, **kw: pytest.fail('no commands'))
    with pytest.raises(ValueError, match='bounded to one epoch'):
        runner.main('run', '5', 'free_t')


@pytest.mark.parametrize('mode', ['collect-only', 'collect'])
@pytest.mark.parametrize('target', ['local', 'nom-cool-1'])
def test_collect_only_validates_and_records_without_advancing(root, monkeypatch, mode, target):
    prepare_stage(root, '5')
    monkeypatch.setattr(runner.reporting, 'validate_run', lambda _: [])
    monkeypatch.setenv('RUN_WATCH_JOB_ID', 'matched-stage5')
    monkeypatch.setattr(runner.subprocess, 'check_output', lambda *a, **kw:
                        json.dumps({'attempts': [{'target': target}]}))
    copied=[]
    def copy(argv, **kwargs):
        assert argv[0]=='rsync'
        copied.append(argv)
    monkeypatch.setattr(runner.subprocess, 'run', copy)
    advanced=[]
    monkeypatch.setattr(runner, 'advance', lambda: advanced.append(True))
    monkeypatch.setattr(runner, 'submit', lambda *a: pytest.fail('collection must not submit'))
    runner.main(mode, '5', 'beta_1')
    receipt=json.loads((root/'analysis/beta_1-5-collection.json').read_text())
    assert receipt['validated'] and receipt['cursor']['total_updates']==7035
    assert receipt['job_id']=='matched-stage5' and receipt['target']==target
    assert len(copied)==(target!='local')
    assert advanced==([True] if mode=='collect' else [])


def test_collect_only_refuses_bad_parent_before_writing_receipt(root, monkeypatch):
    out=prepare_stage(root, '5')
    wrong=torch.load(out/'checkpoints/initial_model.pt',weights_only=False)
    wrong['model']['module']['weight']+=1
    torch.save(wrong,out/'checkpoints/initial_model.pt')
    monkeypatch.setattr(runner.reporting,'validate_run',lambda _: [])
    monkeypatch.setenv('RUN_WATCH_JOB_ID','bad-parent')
    monkeypatch.setattr(runner.subprocess,'check_output',lambda *a,**kw:
                        json.dumps({'attempts':[{'target':'local'}]}))
    monkeypatch.setattr(runner,'advance',lambda: pytest.fail('no advance'))
    with pytest.raises(AssertionError):runner.main('collect-only','5','beta_1')
    assert not (root/'analysis/beta_1-5-collection.json').exists()
