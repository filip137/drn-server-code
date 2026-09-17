"""Check completed experimental coverage, selection, schedules, and artifacts."""
import argparse
import hashlib
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def learning_rate(base, schedule, epoch):
    factor = 1 if schedule['kind'] == 'constant' else schedule['final_factor'] ** min((epoch - 1) / (schedule['decay_epochs'] - 1), 1)
    return base * factor


def main(args):
    audit = dict(architecture=args.architecture, status='passed_for_collected_completed_runs', runs=[],
                 scope='Checks completed outputs only; phase status and expected coverage must also be complete before final handoff.')
    if args.architecture == 'drn':
        import torch
        first_epochs = {}
        trajectories = {}
        paths = list((HERE / 'drn_screen').glob('runs/*/result.json'))
        paths += list((HERE / 'drn_confirmation').glob('runs/*/result.json'))
        paths += list((HERE / 'drn_extensions').glob('runs/*/result.json'))
        paths += list((HERE / 'drn_readouts').glob('one_epoch_runs/*/result.json'))
        for path in sorted(paths):
            result = read(path)
            assert result['status'] == 'complete' and result['selected_replay_passed'], path
            arm = result['arm']; epochs = arm['epochs']
            metrics = [json.loads(line) for line in (path.parent / 'metrics.jsonl').read_text().splitlines()]
            full = [m for m in metrics if m['kind'] == 'epoch']
            assert len(full) == result['epochs_completed'] == epochs, path
            for epoch, record in enumerate(full, 1):
                assert record['epoch'] == epoch and record['examples'] == 55000 and record['batches'] == 3438, path
                assert math.isclose(record['learning_rate'], learning_rate(arm['learning_rate'], arm['schedule'], epoch), rel_tol=1e-12), path
                assert math.isfinite(record['validation']['apparent']['kl_teacher_student']), path
            key = lambda r: (r['validation']['apparent']['kl_teacher_student'], -r['validation']['apparent']['student_accuracy'], r['epoch'])
            assert min(result['candidates'], key=key) == result['selected'], path
            prefix_key = (result['replica'], arm['condition'], arm['learning_rate'])
            prefix = next(r for r in result['candidates'] if r['epoch'] == 1)
            assert first_epochs.setdefault(prefix_key, prefix) == prefix, path
            trajectory_key = (result['replica'], arm['arm_id'])
            if trajectory_key in trajectories:
                old = trajectories[trajectory_key]
                common = min(len(old), len(result['candidates']))
                assert old[:common] == result['candidates'][:common], path
            trajectories[trajectory_key] = result['candidates']
            checkpoint = torch.load(path.parent / 'final_state.pt', map_location='cpu', weights_only=False)
            pulse_counts = checkpoint['state']['plant']['pulse_count']
            optimizer = checkpoint['state']['optimizer']
            assert optimizer['step'] == epochs * 3438, path
            assert int(pulse_counts.sum()) == result['total_pulses'], path
            audit['runs'].append(dict(result=str(path), epochs=epochs, selected_epoch=result['selected']['epoch'],
                                      maximum_pulses_per_cell=int(pulse_counts.max()), cells_at_cap=int((pulse_counts >= optimizer['pulse_cap']).sum()),
                                      pulse_cap=optimizer['pulse_cap'], probability_clipped=optimizer['total_probability_clipped_cells'],
                                      selected_state_sha256=digest(path.parent / 'selected_state.pt')))
    else:
        roots = [HERE / 'crossbar_collected', HERE / 'crossbar_confirmation_collected', HERE / 'crossbar_readouts_collected']
        paths = [path for root in roots for path in root.glob('**/result.json')]
        data_order = {}
        first_epochs = {}
        for path in sorted(paths):
            result = read(path)
            assert result['status'] == 'complete', path
            m = result['metrics']; full = m['epochs']
            assert len(full) == m['training_final_epoch'], path
            candidates = [(0, m['initial']['validation']['apparent_forward'])]
            for epoch, record in enumerate(full, 1):
                assert record['epoch'] == epoch and record['examples'] == 55000 and record['batches'] == 3438, path
                expected = learning_rate(m['learning_rate'], record['learning_rate_schedule'], epoch)
                assert math.isclose(record['effective_learning_rate'], expected, rel_tol=1e-12), path
                primary = record['validation']['apparent_forward']
                assert primary['examples'] == 5000 and math.isfinite(primary['kl_teacher_student']), path
                assert record['test'] is None, path
                candidates.append((epoch, primary))
                order = (record['ordered_labels_sha256'], record['ordered_model_inputs_sha256'])
                assert data_order.setdefault(epoch, order) == order, path
            selected = min(candidates, key=lambda v: (v[1]['kl_teacher_student'], -v[1]['student_accuracy'], v[0]))
            assert selected[0] == m['selected_epoch'] and selected[1] == m['selected_validation']['apparent_forward'], path
            prefix_key = (m['assignment_seed'], m['endpoint_seed'], m['input_role'], m['learning_rate'])
            prefix = (full[0]['apparent_sha256'], full[0]['persistent_sha256'])
            assert first_epochs.setdefault(prefix_key, prefix) == prefix, path
            artifacts = []
            for artifact in result['artifacts']:
                target = path.parent / artifact['path']
                assert target.is_file() and target.stat().st_size == artifact['size_bytes'], target
                assert digest(target) == artifact['sha256'], target
                artifacts.append(artifact['path'])
            optimizer = m['training_final_optimizer']
            audit['runs'].append(dict(result=str(path), epochs=len(full), selected_epoch=m['selected_epoch'],
                                      maximum_pulses_per_cell=optimizer['maximum_pulses_per_cell'], cells_at_cap=optimizer['cells_at_cap'],
                                      pulse_cap=optimizer['pulse_cap_per_cell'], probability_clipped=optimizer['probability_clipped'],
                                      blocked_at_cap=optimizer['blocked_at_cap'], verified_artifacts=artifacts))
    audit['completed_runs'] = len(audit['runs'])
    audit['any_cells_at_cap'] = any(r['cells_at_cap'] for r in audit['runs'])
    out = HERE / 'analysis' / (args.architecture + '_artifact_audit.json')
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(audit, indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(output=str(out), completed_runs=audit['completed_runs'], any_cells_at_cap=audit['any_cells_at_cap'])))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('architecture', choices=['drn', 'crossbar'])
    main(parser.parse_args())
