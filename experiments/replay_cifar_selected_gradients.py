"""Replay two exact 101-update CIFAR prefixes and inspect Adam's selected gradients."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import shlex
import signal
import sys
import time

import torch
from torch.nn import functional as F

from experiments import reporting
from experiments.cifar_conv_eqprop import (
    initial_training_beta_state, replay_without_training_effects,
    validate_beta_search_record, validate_training_beta_state,
)
from experiments.compare_cifar_block_beta_gradients import agreement
from experiments.screen_cifar_beta_half_epoch import verify_hashes
from experiments.train_cifar_beta_stages import (
    cursor, epoch_loader, epoch_order, save_state, training_update, validate_config,
)
from experiments.train_cifar_l8_analog import (
    Cohort, check_gpu_resident_execution, deterministic_setup, make_optimizer, read_training_data,
)
from labs.cifar_l8_analog import CifarL8Analog, batchnorm_minibatch_stats_only

PROBES = (0, 1, 20, 50, 100)
UPDATES = 101


def reference_gradients(model, x, y, iterations, method):
    """Mean-CE reference at current parameters, with no BN/RNG/gradient side effects."""
    if method not in ('native_bptt', 'free_t_bptt'):
        raise ValueError('Unknown reference method')
    parameters = model.trainable_tensors()
    with replay_without_training_effects(model), batchnorm_minibatch_stats_only(model):
        model.train()
        logits = model.bptt(x, iterations)[1] if method == 'native_bptt' else model(x, iterations, reset=True)
        loss = F.cross_entropy(logits, y, reduction='mean')
        gradients = torch.autograd.grad(loss, tuple(parameters.values()))
        result = {name: value.detach().cpu().double().clone() for name, value in zip(parameters, gradients)}
        if not torch.isfinite(loss) or not all(torch.isfinite(g).all() for g in result.values()):
            raise FloatingPointError('Nonfinite reference gradient/loss')
        return result, float(loss.detach())


def measured_update(model, optimizer, x, y, iterations, case):
    """Capture the actual Adam inputs; perform the production update unchanged."""
    parameters = model.trainable_tensors()
    before = {name: p.detach().cpu().double().clone() for name, p in parameters.items()}
    captured = {}
    calls = []

    def capture(optim, args, kwargs):
        if optim is not optimizer or calls:
            raise RuntimeError('Expected exactly one Adam step')
        calls.append(True)
        for name, parameter in parameters.items():
            if parameter.grad is None:
                raise RuntimeError('Missing Adam input: ' + name)
            captured[name] = parameter.grad.detach().cpu().double().clone()

    handle = optimizer.register_step_pre_hook(capture)
    try:
        prediction, loss, telemetry = training_update(model, optimizer, x, y, iterations, case)
    finally:
        handle.remove()
    if len(calls) != 1:
        raise RuntimeError('Adam input was not captured')
    deltas = {name: p.detach().cpu().double() - before[name] for name, p in parameters.items()}
    return prediction, loss, telemetry, captured, before, deltas


def parameter_family(name):
    if name.startswith('blocks.'):
        return 'conv'
    if '_input_gain_raw' in name:
        return 'gain'
    return 'head' if name.startswith('head.') else 'bn'


def gradient_summary(actual, references, before, deltas):
    def summarize(names):
        flatten = lambda values: torch.cat([values[name].reshape(-1) for name in names])
        gradient, parameter, delta = flatten(actual), flatten(before), flatten(deltas)
        pn, dn = float(parameter.norm()), float(delta.norm())
        return dict(references={method: agreement(gradient, flatten(values)) for method, values in references.items()},
                    parameter_norm=pn, update_norm=dn, relative_update_norm=dn / pn if pn else None)

    layers = {name: dict(family=parameter_family(name), **summarize([name])) for name in actual}
    groups = {family: [name for name in actual if parameter_family(name) == family]
              for family in ('conv', 'bn', 'gain', 'head')}
    groups['digital'] = [name for name in actual if parameter_family(name) != 'conv']
    for i in range(3):
        groups[f'block{i+1}'] = [name for name in actual if name.startswith(f'blocks.{i}.')]
    return dict(layers=layers, pooled={group: summarize(names) for group, names in groups.items() if names})


def original_prefix(path):
    losses = {}
    for row in reporting.iter_metrics(path):
        if row.get('stage') == 'training':
            update = row['cursor']['total_updates']
            if update > UPDATES:
                break
            if update in losses or not math.isfinite(row['loss']):
                raise ValueError('Invalid original prefix loss coverage')
            losses[update] = row['loss']
    if set(losses) != set(range(1, UPDATES + 1)):
        raise ValueError('Original metrics must cover all101 prefix losses')
    return losses


def check_loss(actual, expected, label, update):
    if not math.isfinite(actual) or actual != expected:
        raise RuntimeError(f'Exact prefix loss mismatch: {label} update{update}: {actual!r} != {expected!r}')


def validate_replay(run):
    """CPU-only declared coverage, loss identity and controller/checkpoint validation."""
    run = Path(run)
    errors = reporting.validate_run(run)
    if errors:
        raise ValueError(errors)
    manifest = json.loads((run/'manifest.json').read_text())
    expected = json.loads((run/'artifacts/original_prefix.json').read_text())
    initial_bn = json.loads((run/'artifacts/initial_bn_counts.json').read_text())
    rows = list(reporting.iter_metrics(run/'metrics.jsonl'))
    for label, spec in manifest['replays'].items():
        training = [r for r in rows if r.get('replay') == label and r['stage'] == 'training']
        probes = [r for r in rows if r.get('replay') == label and r['stage'] == 'probe']
        if [r['update'] for r in training] != list(range(1, UPDATES + 1)) or [r['completed_updates'] for r in probes] != list(PROBES):
            raise ValueError('Incomplete replay coverage: ' + label)
        case = spec['case']
        state = initial_training_beta_state(case)
        for row in training:
            check_loss(row['loss'], expected[label][str(row['update'])], label, row['update'])
            if (row['loss_match'] is not True or row['starting_betas'] != state['values']
                    or row['betas_next'] != row['betas_used'] or row['beta_successful_updates'] != row['update']
                    or len(row['beta_search']) != 3 or len(row['training_rms']) != 3):
                raise ValueError('Replay loss/beta continuity mismatch')
            for i, search in enumerate(row['beta_search']):
                validate_beta_search_record(search, state['values'][i], case['beta_search']['targets'][i], case['beta_search'])
                if search['selected_beta'] != row['betas_used'][i]:
                    raise ValueError('Selected beta mismatch')
                chosen = search['trials'][search['chosen_index']]
                if any(chosen[key] != value for key, value in row['training_rms'][i].items()):
                    raise ValueError('Selected displacement mismatch')
                state['search_counts']['pairs'][i] += search['pair_count']
                state['search_counts']['tolerance_misses'][i] += int(search['tolerance_miss'])
            state['values'] = row['betas_next']
            state['successful_updates'] += 1
        for probe in probes:
            update = training[probe['completed_updates']]
            if probe['beta_search'] != update['beta_search'] or probe['betas_used'] != update['betas_used']:
                raise ValueError('Probe differs from actual training selection')
            if not {'conv', 'digital', 'block1', 'block2', 'block3'} <= probe['pooled'].keys():
                raise ValueError('Incomplete gradient groups')
            if not probe['layers'] or any(set(row['references']) != {'native_bptt', 'free_t_bptt'}
                    for row in (*probe['layers'].values(), *probe['pooled'].values())):
                raise ValueError('Incomplete reference gradients')
        checkpoint = torch.load(run/'checkpoints'/f'{label}_prefix.pt', map_location='cpu', weights_only=False)
        if (checkpoint['staged_cursor'] != cursor(0, UPDATES) or checkpoint['scheduler']['last_epoch'] != 0
                or checkpoint['epoch'] != 0 or checkpoint['native_epoch_resume_safe']
                or checkpoint['config'] != spec['config'] or checkpoint['case'] != case
                or checkpoint['case_id'] != spec['case_id'] or checkpoint['input_identity'] != spec['input_identity']
                or {int(value['step']) for value in checkpoint['optimizer']['state'].values()} != {UPDATES}
                or checkpoint['conv_beta_state'] != state):
            raise ValueError('Replay checkpoint/counter mismatch')
        validate_training_beta_state(checkpoint['conv_beta_state'], case, UPDATES)
        for name, count in initial_bn[label].items():
            if int(checkpoint['model']['module'][name]) != count + UPDATES:
                raise ValueError('Replay changed BatchNorm update count')
    if set(manifest['replays']) != {'exp015', 'exp017'} or len(rows) != 2 * (UPDATES + len(PROBES)):
        raise ValueError('Unexpected replay coverage')
    return dict(validated=True, trajectories=2, updates_per_trajectory=UPDATES, probes=10, exact_prefix_losses=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--validate-run', type=Path)
    parser.add_argument('--root', type=Path)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--dataset-root', type=Path)
    parser.add_argument('--target')
    parser.add_argument('--maximum-seconds', type=float, default=1200.)
    args = parser.parse_args()
    if args.validate_run:
        print(json.dumps(validate_replay(args.validate_run))); return
    if any(getattr(args, name) is None for name in ('root', 'output_dir', 'dataset_root', 'target')):
        parser.error('--root, --output-dir, --dataset-root and --target are required')
    if not 60 < args.maximum_seconds <= 1200:
        raise ValueError('Replay allowance must exceed60s and cannot exceed1200s')
    started = time.monotonic()
    deadline = started + args.maximum_seconds - 30
    root, run = args.root.resolve(), args.output_dir.resolve()
    if run.exists():
        raise FileExistsError(run)
    source = Path(__file__).resolve().parents[1]
    source_identity = json.loads((source/'source_identity.json').read_text())
    verify_hashes(source, source_identity['source_sha256'], ['experiments/replay_cifar_selected_gradients.py'])
    declarations = json.loads((root/'inputs/replays.json').read_text())
    if set(declarations) != {'exp015', 'exp017'}:
        raise ValueError('The declared comparison is exp015 versus exp017')
    specifications, prefixes = {}, {}
    environment = dict(torch=str(torch.__version__), cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version())
    for label, entry in declarations.items():
        original = Path(entry['root']).resolve()
        inputs = original/'inputs'
        identity = json.loads((inputs/'identity.json').read_text())
        verify_hashes(inputs, identity['sha256'], ['initializer.pt', 'config.json', 'split.json', 'cohort_indices.json', 'cases.json'])
        config = json.loads((inputs/'config.json').read_text())
        validate_config(config)
        case = json.loads((inputs/'cases.json').read_text())[entry['case']]
        initial_training_beta_state(case)
        if case['beta_search']['beta_bounds'] != [1e-8, 10000. if label == 'exp015' else 30000.]:
            raise ValueError('Unexpected declared beta ceiling')
        original_run = original/'cells/1'/entry['case']
        original_manifest = json.loads((original_run/'manifest.json').read_text())
        if any(environment[k] != original_manifest['environment'][k] for k in ('torch', 'cuda')):
            raise ValueError('Replay runtime differs from original training')
        # Preserve original numerical modules; operational wrappers/tests may differ.
        for name, digest in original_manifest['source_identity']['source_sha256'].items():
            if name.endswith('.py') and not name.startswith('labs/tests/') and name not in (
                    'experiments/run_cifar_beta_stages.py', 'experiments/cifar_beta_stage_decision.py'):
                if source_identity['source_sha256'].get(name) != digest:
                    raise ValueError('Frozen numerical source differs: ' + name)
        metric_path = original_run/'metrics.jsonl'
        prefixes[label] = original_prefix(metric_path)
        specifications[label] = dict(original_root=str(original), case_id=entry['case'], case=case, config=config,
            input_identity=identity, original_metrics_sha256=reporting.sha256_file(metric_path),
            original_source_identity=original_manifest['source_identity'])
    left, right = specifications.values()
    for name in ('initializer.pt', 'config.json', 'split.json', 'cohort_indices.json'):
        if left['input_identity']['sha256'][name] != right['input_identity']['sha256'][name]:
            raise ValueError('Replay inputs are not matched: ' + name)
    a, b = copy.deepcopy(left['case']), copy.deepcopy(right['case'])
    a['beta_search']['beta_bounds'][1] = b['beta_search']['beta_bounds'][1]
    if a != b:
        raise ValueError('Replay cases differ beyond the beta ceiling')
    data_hashes = json.loads((source/'training-data-hashes.json').read_text())
    names = [f'data_batch_{i}' for i in range(1, 6)]
    verify_hashes(args.dataset_root, {name: data_hashes[name] for name in names}, names)
    reporting.start_run(run, dict(study_id=root.name, run_id=run.name, arm_id='prefix_replay', smoke=False,
        evidence_class='cifar_actual_selected_gradient_replay', replays=specifications, source_identity=source_identity,
        declaration_sha256=reporting.sha256_file(root/'inputs/replays.json'), environment=environment,
        runtime=reporting.runtime_context(target=args.target), maximum_seconds=args.maximum_seconds,
        probes=list(PROBES), updates_per_trajectory=UPDATES, dataset=dict(official_test_read=False),
        command=shlex.join([sys.executable, *sys.argv])))
    reporting.atomic_write_json(run/'artifacts/original_prefix.json', prefixes)
    stop, summaries, initial_bn = [], {}, {}
    handlers = {sig: signal.signal(sig, lambda number, frame: stop.append(number)) for sig in (signal.SIGTERM, signal.SIGINT)}
    model = optimizer = scheduler = None
    completed = 0
    label = None

    def guard():
        if stop or time.monotonic() >= deadline:
            raise TimeoutError('Replay signal/deadline; coverage is incomplete')

    try:
        check_gpu_resident_execution('cuda', False)
        torch.cuda.reset_peak_memory_stats()
        images, labels = read_training_data(args.dataset_root)
        for label, spec in specifications.items():
            guard()
            deterministic_setup(spec['config']['seed'])
            model = CifarL8Analog(spec['config'], 'cuda')
            optimizer, scheduler = make_optimizer(model, spec['config'])
            model.conv_beta_state = initial_training_beta_state(spec['case'])
            original = Path(spec['original_root'])
            initial = torch.load(original/'inputs/initializer.pt', map_location='cpu', weights_only=False)
            if initial['epoch'] != 0:
                raise ValueError('Prefix replay requires the original epoch0 initializer')
            model.restore(initial['model'])
            initial_bn[label] = {name: int(value) for name, value in initial['model']['module'].items()
                                 if name.endswith('num_batches_tracked')}
            reporting.atomic_write_json(run/'artifacts/initial_bn_counts.json', initial_bn)
            del initial
            completed = 0
            split = json.loads((original/'inputs/split.json').read_text())
            order = epoch_order(split['train'], spec['config']['seed'], 1)
            order_hash = hashlib.sha256(json.dumps(order).encode()).hexdigest()
            batches = epoch_loader(Cohort(images, labels, order, spec['config'], 1, True), 32,
                spec['config']['num_workers'], True, spec['config']['seed'], 1)
            for x, y in batches:
                guard()
                model.train(); optimizer.zero_grad(set_to_none=True)
                x, y = x.cuda(), y.cuda()
                probe = None
                if completed in PROBES:
                    references, losses = {}, {}
                    for method in ('native_bptt', 'free_t_bptt'):
                        guard()
                        references[method], losses[method] = reference_gradients(model, x, y, spec['config']['iterations'], method)
                    prediction, loss, telemetry, actual, before, deltas = measured_update(model, optimizer, x, y,
                        spec['config']['iterations'], spec['case'])
                    probe = dict(stage='probe', replay=label, completed_updates=completed,
                        batch_indices=order[completed*32:(completed+1)*32], reference_losses=losses,
                        **gradient_summary(actual, references, before, deltas), **telemetry)
                    del references, actual, before, deltas
                else:
                    prediction, loss, telemetry = training_update(model, optimizer, x, y, spec['config']['iterations'], spec['case'])
                completed += 1
                value = float(loss)
                row = dict(stage='training', replay=label, update=completed, loss=value,
                    original_loss=prefixes[label][completed], loss_match=value == prefixes[label][completed],
                    training_order_sha256=order_hash, elapsed_seconds=time.monotonic()-started, **telemetry)
                reporting.append_metric(run/'metrics.jsonl', row)
                if probe is not None:
                    reporting.append_metric(run/'metrics.jsonl', probe)
                reporting.update_status_progress(run, dict(stage='replay', replay=label, completed_updates=completed,
                    elapsed_seconds=time.monotonic()-started))
                check_loss(value, prefixes[label][completed], label, completed)
                del x, y, prediction, loss
                if completed == UPDATES:
                    break
            if completed != UPDATES:
                raise RuntimeError('Incomplete training prefix')
            save_state(run/'checkpoints'/f'{label}_prefix.pt', model, optimizer, scheduler, spec['config'],
                spec['case_id'], spec['case'], cursor(0, completed), spec['input_identity'])
            summaries[label] = dict(updates=completed, probes=list(PROBES), exact_prefix_losses=True,
                                   final_beta_state=copy.deepcopy(model.conv_beta_state))
            del model, optimizer, scheduler
            model = optimizer = scheduler = None
        summary = dict(replays=summaries, probes=10, elapsed_seconds=time.monotonic()-started,
                       peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                       peak_reserved_bytes=torch.cuda.max_memory_reserved())
        reporting.atomic_write_json(run/'artifacts/summary.json', summary)
        reporting.complete_run(run, terminal_metrics=summary, completion=dict(criteria_met=True, official_test_read=False,
            full_prefix_coverage=True, exact_prefix_losses=True))
    except BaseException as error:
        if model is not None and label is not None:
            spec = specifications[label]
            try:
                save_state(run/'checkpoints'/f'{label}_partial.pt', model, optimizer, scheduler, spec['config'],
                    spec['case_id'], spec['case'], cursor(0, completed), spec['input_identity'])
            except Exception as save_error:
                reporting.atomic_write_json(run/'artifacts/checkpoint_error.json', dict(error=str(save_error)))
        reporting.atomic_write_json(run/'artifacts/partial_coverage.json', dict(completed_replays=summaries,
            current_replay=label, current_updates=completed, error=str(error)))
        reporting.fail_run(run, error=error)
        raise
    finally:
        for sig, handler in handlers.items():
            signal.signal(sig, handler)


if __name__ == '__main__':
    main()
