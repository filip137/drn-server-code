"""Run the predeclared one-epoch references and common test replay after selection."""
import argparse
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from analyze import screen_results

HERE = Path(__file__).resolve().parent


def read(path):
    return json.loads(Path(path).read_text())


def save(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(payload, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def native_result(root):
    paths = list(Path(root).glob('*/result.json'))
    if len(paths) != 1 or read(paths[0])['status'] != 'complete':
        raise RuntimeError(f'Expected one complete native result in {root}: {paths}')
    return paths[0]


def within_one_epoch(rows):
    choices = []
    for condition in ['healthy', 'faulted']:
        candidates = []
        for row in rows:
            if row['condition'] != condition or row['schedule_kind'] != 'constant':
                continue
            for point in row['trajectory']:
                if point['epoch'] <= 1:
                    candidates.append(dict(arm_id=row['arm_id'], condition=condition,
                                           learning_rate=row['learning_rate'], **point))
        choices.append(min(candidates, key=lambda v: (v['kl'], v['epoch'], v['learning_rate'])))
    return choices


def main(args):
    arch = args.architecture
    args.plan = args.plan.resolve()
    plan = read(args.plan)
    out = HERE / (arch + '_readouts')
    out.mkdir(exist_ok=False)
    (out / 'logs').mkdir()
    status = dict(status='waiting_for_confirmation', architecture=arch, pid=os.getpid(),
                  completed=[], heartbeat=time.time())
    save(out / 'status.json', status)
    env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
               PYTHONUNBUFFERED='1', EBL_DEFER_CURRENT_SIMULATIONS='1', EBL_MNIST_ROOT=args.mnist_root)
    code = HERE / 'crossbar_code' if arch == 'crossbar' else HERE.parent / 'ibm-om-cell-aware-quantized'

    def execute(label, command, cwd=code):
        status.update(status='running', current_case=label, command=command, heartbeat=time.time())
        save(out / 'status.json', status)
        print('START', label, flush=True)
        with (out / 'logs' / (label + '.log')).open('w') as log:
            child = subprocess.Popen(command, cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT)
            status['child_pid'] = child.pid
            while child.poll() is None:
                status['heartbeat'] = time.time()
                save(out / 'status.json', status)
                time.sleep(5)
        if child.returncode:
            status.update(status='failed', exit_code=child.returncode, heartbeat=time.time())
            save(out / 'status.json', status)
            raise RuntimeError(f'{label} failed; inspect the retained log.')
        status['completed'].append(label)
        status.update(child_pid=None, heartbeat=time.time())
        save(out / 'status.json', status)
        print('COMPLETE', label, flush=True)

    confirmation = HERE / (arch + '_confirmation')
    while True:
        if (confirmation / 'status.json').exists():
            current = read(confirmation / 'status.json')
            if current['status'] == 'complete':
                break
            if current['status'] in {'failed', 'blocked_by_failed_screen'}:
                status.update(status='blocked_by_failed_confirmation', heartbeat=time.time())
                save(out / 'status.json', status)
                raise RuntimeError('Confirmation did not complete; investigate before test replay.')
        status['heartbeat'] = time.time()
        save(out / 'status.json', status)
        time.sleep(30)

    rows = screen_results(HERE / (arch + '_screen'), arch)
    if len(rows) != len(plan[arch]['arms']):
        raise RuntimeError('Incomplete screening coverage.')
    frozen = read(confirmation / 'frozen_selection.json')['winners']
    short = within_one_epoch(rows)
    save(out / 'one_epoch_selection.json', dict(
        rule='Minimum development validation KL within epoch one among constant rates, then earlier checkpoint and lower rate. Includes P0 and the declared DRN quarter-epoch checks.',
        frozen_at=time.time(), choices=short))
    entries = []

    def drn_entry(condition, replica, kind, result_path=None, state=None, rate=None):
        entry = dict(architecture=arch, condition=condition, replica=replica, kind=kind,
                     state_path=str(state) if state else None, learning_rate=rate,
                     result_path=str(result_path) if result_path else None)
        if result_path:
            result = read(result_path)
            entry.update(selected_epoch=result['selected']['epoch'],
                         validation=result['selected']['validation'],
                         selected_pulses=result['selected']['pulses'])
        entries.append(entry)

    if arch == 'drn':
        short_plan = copy.deepcopy(plan)
        short_plan['drn']['arms'] = [dict(next(a for a in plan['drn']['arms'] if a['arm_id'] == s['arm_id']), epochs=1) for s in short]
        short_plan_path = out / 'one_epoch_plan.json'
        save(short_plan_path, short_plan)
        for replica in plan['drn']['replicas']:
            rid = replica['id']
            for condition in ['healthy', 'faulted']:
                drn_entry(condition, rid, 'p0')
                matched_results = []
                for winner in [w for w in frozen if w['condition'] == condition]:
                    if rid == 'array-1-write-1':
                        base = Path(winner.get('extension_result_root', HERE / 'drn_screen/runs' / winner['arm_id']))
                    else:
                        base = confirmation / 'runs' / (winner['arm_id'] + '-' + rid)
                    result_path = base / 'result.json'
                    result = read(result_path)
                    if result['status'] != 'complete':
                        raise RuntimeError(result_path)
                    drn_entry(condition, rid, winner['schedule_kind'], result_path,
                              base / 'selected_state.pt', winner['learning_rate'])
                    matched_results.append((base, result))
                # The prior Figure-6 selected state is retained exactly as reported.
                label = 'clean' if condition == 'healthy' else 'reset_stuck'
                legacy_root = Path(replica['deployment']).parents[4] / 'adam-arm'
                legacy = list((legacy_root / ('hwa__' + label + '__' + rid.replace('-', '_').replace('_write_', '__write_'))).glob('*/checkpoints/selected_adam_state.pt'))
                if len(legacy) != 1:
                    raise RuntimeError(f'Expected one exact historical state: {legacy_root}, {rid}, {label}, {legacy}')
                drn_entry(condition, rid, 'previous_one_epoch', state=legacy[0], rate=1e-4)
                choice = next(s for s in short if s['condition'] == condition)
                if rid == 'array-1-write-1':
                    base = HERE / 'drn_screen/runs' / choice['arm_id']
                    matched_results.append((base, read(base / 'result.json')))
                reusable = [base for base, result in matched_results
                            if result['arm']['arm_id'] == choice['arm_id'] and result['selected']['epoch'] <= 1]
                if reusable:
                    base = reusable[0]
                else:
                    case = choice['arm_id'] + '-' + rid
                    base = out / 'one_epoch_runs' / case
                    execute('reference-' + case, [sys.executable, str(HERE / 'run_drn.py'), '--plan', str(short_plan_path),
                                                '--arm', choice['arm_id'], '--replica', rid, '--output', str(base)])
                drn_entry(condition, rid, 'tuned_within_one_epoch', base / 'result.json',
                          base / 'selected_state.pt', choice['learning_rate'])
    else:
        base_inputs = Path(args.crossbar_inputs)
        inputs = read(HERE / 'confirmation_inputs/receipt.json')
        inputs += [dict(condition=condition, assignment_seed=plan['crossbar']['screen_assignment'],
                        endpoint_seed=plan['crossbar']['screen_endpoint'],
                        local_path=str(base_inputs / ('hwa_healthy_p0.pt' if condition == 'healthy' else 'hwa_faulted_p0.pt')))
                   for condition in ['healthy', 'faulted']]
        confirmation_cases = read(confirmation / 'cases.json')
        confirmation_study = 'mnist-ibm-om-crossbar-long-kl-confirmation-20260906-v1'
        reference_study = 'mnist-ibm-om-crossbar-one-epoch-references-20260906-v1'
        study = copy.deepcopy(read(code / 'studies' / (plan['crossbar']['study_id'] + '.json')))
        study.update(study_id=reference_study, title='Frozen one-epoch recovery references',
                     hypothesis='A tuned one-epoch reference tests whether a longer horizon offers further KL improvement.', arms=[],
                     completion_criteria=['All declared reference cases complete from their exact paired HWA P0 states.',
                                          'One-epoch rates are frozen from development validation before any test replay.'])
        pending = []

        def cb_entry(inp, kind, config, result=None):
            state = Path(inp['local_path']) if result is None else Path(result).parent / 'checkpoints/crossbar_device_state.pt'
            entry = dict(architecture=arch, condition=inp['condition'], replica='array-' + str(inp['assignment_seed']),
                         kind=kind, config=str(config), state_path=str(state), result_path=str(result) if result else None)
            if result:
                metrics = read(result)['metrics']
                entry.update(selected_epoch=metrics['selected_epoch'], validation=metrics['selected_validation'],
                             learning_rate=metrics['learning_rate'], selected_pulses=metrics['selected_optimizer']['applied_pulses'])
            entries.append(entry)
            return entry

        def new_reference(inp, base_id, kind):
            case = inp['condition'] + '-' + kind + '-array-' + str(inp['assignment_seed'])
            cfg = read(code / 'examples/mnist_analog_relu/long_kl_schedules_20260906' / (base_id + '.json'))
            cfg['device'].update(assignment_seed=inp['assignment_seed'], endpoint_seed=inp['endpoint_seed'])
            cfg['stage']['epochs'] = 1
            config = code / 'examples/mnist_analog_relu/one_epoch_references_20260906' / (case + '.json')
            save(config, cfg)
            study['arms'].append(dict(arm_id=case, configs=['../' + str(config.relative_to(code))],
                                     description='Frozen ' + kind + ' recovery from exact paired P0',
                                     experiment_id='mnist_ibm_om_crossbar_relu.v2', mode='train'))
            pending.append(dict(case=case, inp=inp, config=str(config), kind=kind))
            return config, case

        for inp in inputs:
            condition = inp['condition']
            development = inp['assignment_seed'] == plan['crossbar']['screen_assignment']
            previous_id = condition + ('-constant-1e-5' if condition == 'healthy' else '-constant-3e-5')
            if development:
                previous_config, previous_case = new_reference(inp, previous_id, 'previous_one_epoch')
                previous_entry = None
            else:
                job = next(j for j in confirmation_cases if j['reference'] and j['input']['condition'] == condition
                           and j['input']['assignment_seed'] == inp['assignment_seed'])
                previous_config = Path(job['config'])
                previous_result = native_result(confirmation / 'results' / confirmation_study / 'runs' / job['case'])
                previous_entry = cb_entry(inp, 'previous_one_epoch', previous_config, previous_result)
            cb_entry(inp, 'p0', previous_config)
            for winner in [w for w in frozen if w['condition'] == condition]:
                if development:
                    config = code / 'examples/mnist_analog_relu/long_kl_schedules_20260906' / (winner['arm_id'] + '.json')
                    result_root = Path(winner.get('extension_result_root', HERE / 'crossbar_screen/results' / plan['crossbar']['study_id'] / 'runs' / winner['arm_id']))
                else:
                    job = next(j for j in confirmation_cases if not j['reference'] and j['schedule_source'] == winner['arm_id']
                               and j['input']['assignment_seed'] == inp['assignment_seed'])
                    config = Path(job['config'])
                    result_root = confirmation / 'results' / confirmation_study / 'runs' / job['case']
                cb_entry(inp, winner['schedule_kind'], config, native_result(result_root))
            short_id = next(s['arm_id'] for s in short if s['condition'] == condition)
            if short_id == previous_id:
                if previous_entry:
                    entries.append(dict(previous_entry, kind='tuned_within_one_epoch'))
                else:
                    next(v for v in pending if v['case'] == previous_case)['alias'] = 'tuned_within_one_epoch'
            else:
                new_reference(inp, short_id, 'tuned_within_one_epoch')
        save(out / 'one_epoch_cases.json', pending)
        study_path = code / 'studies' / (reference_study + '.json')
        save(study_path, study)
        execute('prepare-references', [sys.executable, '-m', 'ebl', 'study', 'prepare', '--plan', str(study_path), '--results-root', str(out / 'reference_results')])
        for job in pending:
            root = out / 'reference_results' / reference_study / 'runs' / job['case']
            execute('reference-' + job['case'], [sys.executable, '-m', 'ebl', 'train', '--config', job['config'],
                                               '--output-dir', str(root), '--teacher-weights', str(base_inputs / 'teacher_weights.pt'),
                                               '--device-state', job['inp']['local_path']])
            entry = cb_entry(job['inp'], job['kind'], job['config'], native_result(root))
            if job.get('alias'):
                entries.append(dict(entry, kind=job['alias']))
        execute('verify-references', [sys.executable, '-m', 'ebl', 'study', 'summarize', '--study-dir',
                                     str(out / 'reference_results' / reference_study), '--verify-artifacts'])

    if len(entries) != 30:
        raise RuntimeError(f'Expected 30 paired readout entries, got {len(entries)}.')
    save(out / 'frozen_readout_cases.json', dict(frozen_at=time.time(), test_selection=False, entries=entries))
    seen = {}
    for entry in entries:
        label = '-'.join([entry['condition'], entry['replica'], entry['kind']])
        key = (entry['state_path'], entry['condition'], entry['replica'])
        if key in seen:
            entry['test_readout'] = seen[key]
            continue
        output = out / 'test' / (label + '.json')
        if arch == 'drn':
            command = [sys.executable, str(HERE / 'readout_drn.py'), '--plan', str(args.plan),
                       '--replica', entry['replica'], '--condition', entry['condition'], '--split', 'test', '--output', str(output)]
            if entry['state_path']:
                command += ['--state', entry['state_path']]
            execute('readout-' + label, command)
        else:
            execute('readout-' + label, [sys.executable, '-m', 'ebl', 'recovery-readout', '--config', entry['config'],
                                       '--teacher-weights', str(Path(args.crossbar_inputs) / 'teacher_weights.pt'),
                                       '--device-state', entry['state_path'], '--split', 'test', '--output', str(output)],
                    cwd=HERE / 'readout_code')
        replay = read(output)
        if replay['status'] != 'complete' or replay['selection_performed'] or replay['writes'] != 0 or not replay['unchanged_state_check']['passed']:
            raise RuntimeError(output)
        if arch == 'drn' and entry['kind'] == 'previous_one_epoch' and replay['selected_epoch'] != 1:
            raise RuntimeError('The declared historical one-epoch DRN state was selected at a different epoch.')
        entry['test_readout'] = str(output)
        seen[key] = str(output)
    save(out / 'index.json', dict(status='complete', entries=entries, test_selection=False))
    status.update(status='complete', completed_at=time.time(), heartbeat=time.time())
    save(out / 'status.json', status)
    print('READOUT_COMPLETE', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('architecture', choices=['drn', 'crossbar'])
    parser.add_argument('--mnist-root', required=True)
    parser.add_argument('--crossbar-inputs', default='')
    parser.add_argument('--plan', type=Path, default=HERE / 'plan.json')
    main(parser.parse_args())
