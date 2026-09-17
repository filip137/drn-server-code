"""Verify declared coverage before handing off the complete recovery evidence."""
import hashlib
import json
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parent


def read(path):
    return json.loads(Path(path).read_text())


def main():
    plan = read(ROOT / 'plan.with_lr_boundary.json')
    phases = ['drn_screen', 'drn_lr_boundary', 'drn_confirmation', 'drn_readouts',
              'crossbar_collected', 'crossbar_confirmation_collected', 'crossbar_readouts_collected']
    for phase in phases:
        assert read(ROOT / phase / 'status.json')['status'] == 'complete', phase
    arms = {arm['arm_id']: arm for arm in plan['drn']['arms']}
    screen = {path.parent.name: read(path) for path in (ROOT / 'drn_screen/runs').glob('*/result.json')}
    assert set(screen) == set(arms)
    for name, result in screen.items():
        assert result['arm'] == arms[name] and result['replica'] == 'array-1-write-1', name
    frozen = read(ROOT / 'drn_confirmation/frozen_selection.json')['winners']
    assert {(w['condition'], w['schedule_kind']) for w in frozen} == {
        (condition, kind) for condition in ['healthy', 'faulted'] for kind in ['constant', 'exponential']}
    expected = {}
    for winner in frozen:
        for replica in plan['drn']['replicas'][1:]:
            case = winner['arm_id'] + '-' + replica['id']
            expected[case] = (dict(arms[winner['arm_id']], epochs=winner['confirmation_epochs']), replica['id'])
        if winner.get('extension_result_root'):
            result = read(Path(winner['extension_result_root']) / 'result.json')
            assert result['arm'] == dict(arms[winner['arm_id']], epochs=60)
            assert result['epochs_completed'] == 60 and result['selected_replay_passed']
    confirmation = {path.parent.name: read(path) for path in (ROOT / 'drn_confirmation/runs').glob('*/result.json')}
    assert set(confirmation) == set(expected)
    for case, result in confirmation.items():
        assert (result['arm'], result['replica']) == expected[case], case
    combined = read(ROOT / 'analysis/paired_test_readouts.json')
    assert combined['status'] == 'complete' and not combined['selection_used_test']
    methods = {'p0', 'previous_one_epoch', 'tuned_within_one_epoch', 'constant', 'exponential'}
    for architecture in ['drn', 'crossbar']:
        rows = [r for r in combined['rows'] if r['architecture'] == architecture]
        replicas = {r['replica'] for r in rows}
        assert len(rows) == 30 and len(replicas) == 3
        assert {(r['replica'], r['condition'], r['method']) for r in rows} == {
            (replica, condition, method) for replica in replicas
            for condition in ['healthy', 'faulted'] for method in methods}
    audit_counts = {}
    for architecture in ['drn', 'crossbar']:
        audit = read(ROOT / 'analysis' / (architecture + '_artifact_audit.json'))
        paths = [Path(run['result']) for run in audit['runs']]
        actual = list((ROOT / 'drn_screen').glob('runs/*/result.json')) + list((ROOT / 'drn_confirmation').glob('runs/*/result.json'))
        actual += list((ROOT / 'drn_extensions').glob('runs/*/result.json')) + list((ROOT / 'drn_readouts').glob('one_epoch_runs/*/result.json'))
        if architecture == 'crossbar':
            actual = [path for folder in phases[-3:] for path in (ROOT / folder).glob('**/result.json')]
        assert set(paths) == set(actual), architecture
        assert audit['status'] == 'passed_for_collected_completed_runs'
        audit_counts[architecture] = audit['completed_runs']
    native_summaries = [path for folder in phases[-3:] for path in (ROOT / folder).glob('**/analysis/summary.json')]
    assert len(native_summaries) == 3
    for path in native_summaries:
        assert read(path)['ready_for_review'], path
    files = ['plan.json', 'plan.with_lr_boundary.json', 'run_drn.py', 'confirm.py', 'finish_readouts.py',
             'readout_drn.py', 'analyze.py', 'audit_completed.py', 'summarize_readouts.py', 'plot_confirmations.py']
    receipt = dict(numerical_work_complete=True, verified_at=time.time(),
                   screen_arms=sum(len(plan[a]['arms']) for a in ['drn', 'crossbar']),
                   completed_runs=audit_counts, common_test_cases=len(combined['rows']),
                   native_studies_ready_for_review=[str(path) for path in native_summaries],
                   managed_scientific_review='pending_user_interpretation_and_next_tests',
                   source_sha256={name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in files})
    target = ROOT / 'analysis/completion_receipt.json'
    target.write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
