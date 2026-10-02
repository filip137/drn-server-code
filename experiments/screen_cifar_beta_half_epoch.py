"""Independent, matched epoch-10 CIFAR branches over exactly 22,500 examples.

Native BPTT uses free T plus tracked warm-start K. The free-T control and
existing hybrid differentiate the reset free-T pass; hybrid replaces Conv only.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shlex
import sys
import time

import numpy as np
import torch
from torch.nn import functional as F

from experiments import reporting
from experiments.cifar_conv_eqprop import (
    EpochBeta, METHOD, hybrid_backward, replay_without_training_effects,
)
from experiments.train_cifar_l8_analog import (
    Cohort, check_gpu_resident_execution, deterministic_setup, evaluate, loader,
    make_optimizer, read_training_data, restore_checkpoint,
)
from labs.cifar_l8_analog import CifarL8Analog


EXAMPLES = 22500
BATCH_SIZE = 32
START_EPOCH = 10
TRAINING_EPOCH = 11
UPDATES = math.ceil(EXAMPLES / BATCH_SIZE)


def selected_order(train_indices, seed):
    if len(train_indices) != 45000 or len(set(train_indices)) != 45000:
        raise ValueError('Expected 45,000 distinct training examples')
    permutation = torch.randperm(len(train_indices), generator=torch.Generator().manual_seed(
        seed + 1000000 * TRAINING_EPOCH)).tolist()
    return [train_indices[i] for i in permutation[:EXAMPLES]]


def validate_case(case):
    if case['method'] not in ('native_bptt', 'free_t_bptt', 'hybrid_eqprop'):
        raise ValueError('Unknown gradient method')
    if case['method'] == 'hybrid_eqprop':
        values = case.get('betas', [])
        if len(values) != 3 or not all(math.isfinite(v) and v > 0 for v in values):
            raise ValueError('Hybrid requires three positive finite betas')
    elif case.get('betas'):
        raise ValueError('Control arms do not use beta')


def verify_hashes(root, hashes, required=()):
    if not hashes or not set(required) <= set(hashes):
        raise ValueError('Incomplete identity manifest')
    root = Path(root).resolve()
    for name, expected in hashes.items():
        path = (root / name).resolve()
        if not path.is_relative_to(root) or reporting.sha256_file(path) != expected:
            raise ValueError('Identity mismatch: ' + name)


def restore_parent(checkpoint, model, optimizer, scheduler, config, parent_steps):
    if checkpoint.get('partial_epoch') or checkpoint['epoch'] != START_EPOCH:
        raise ValueError('Requires a complete epoch-10 parent')
    restored = restore_checkpoint(checkpoint, model, optimizer, scheduler, config)
    if restored != START_EPOCH or {int(v['step']) for v in optimizer.state.values()} != {parent_steps}:
        raise ValueError('Parent Adam steps do not match the completed epoch')
    actual = model.snapshot()
    for group, values in checkpoint['model'].items():
        for name, value in values.items():
            if not torch.equal(value.cpu(), actual[group][name].cpu()):
                raise ValueError('Restore changed tensor: ' + name)


def partial_metadata(updates, examples, parent_steps, order_hash):
    if not 0 <= updates <= UPDATES or examples != min(updates * BATCH_SIZE, EXAMPLES):
        raise ValueError('Inconsistent partial-epoch progress')
    return dict(completed_epoch=START_EPOCH, partial_epoch=TRAINING_EPOCH,
                updates_in_epoch=updates, examples_in_epoch=examples,
                adam_steps=parent_steps + updates, scheduler_last_epoch=START_EPOCH,
                native_epoch_resume_safe=False, training_order_sha256=order_hash)


def save_partial(path, model, optimizer, scheduler, config, progress):
    if scheduler.last_epoch != START_EPOCH:
        raise ValueError('Half-epoch branch advanced the scheduler')
    if {int(v['step']) for v in optimizer.state.values()} != {progress['adam_steps']}:
        raise ValueError('Checkpoint Adam steps disagree with progress')
    payload = dict(model=model.snapshot(), optimizer=optimizer.state_dict(),
                   scheduler=scheduler.state_dict(), epoch=START_EPOCH, config=config,
                   torch_rng=torch.get_rng_state(), numpy_rng=np.random.get_state(),
                   python_rng=random.getstate(),
                   cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None,
                   partial_epoch=copy.deepcopy(progress), native_epoch_resume_safe=False)
    if hasattr(model, 'conv_beta_state'):
        payload['conv_beta_state'] = copy.deepcopy(model.conv_beta_state)
    temporary = path.with_suffix('.tmp')
    torch.save(payload, temporary)
    os.replace(temporary, path)


def finite_parameters(model, *, gradients=False):
    for name, parameter in model.trainable_tensors().items():
        value = parameter.grad if gradients else parameter
        if value is None or not torch.isfinite(value).all():
            raise FloatingPointError(('Missing/nonfinite gradient: ' if gradients else 'Nonfinite parameter: ') + name)


def backward(model, x, y, iterations, case):
    if case['method'] == 'hybrid_eqprop':
        return hybrid_backward(model, x, y, iterations, case['betas'])
    if case['method'] == 'native_bptt':
        prediction, tracked = model.bptt(x, iterations)
    else:
        tracked = model(x, iterations, reset=True)
        prediction = tracked.detach()
    loss = F.cross_entropy(tracked, y)
    if not torch.isfinite(loss):
        raise FloatingPointError('Nonfinite training loss')
    loss.backward()
    return prediction, loss.detach()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--case', required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--dataset-root', type=Path, required=True)
    parser.add_argument('--target', required=True)
    parser.add_argument('--maximum-seconds', type=float, required=True)
    args = parser.parse_args()
    started = time.monotonic()
    if not math.isfinite(args.maximum_seconds) or args.maximum_seconds <= 0:
        raise ValueError('Positive finite runtime budget required')
    deadline = started + args.maximum_seconds
    root, run = args.root.resolve(), args.output_dir.resolve()
    if run.exists():
        raise FileExistsError(run)
    source = Path(__file__).resolve().parents[1]
    identity = json.loads((source / 'source_identity.json').read_text())
    verify_hashes(source, identity['source_sha256'], ['experiments/screen_cifar_beta_half_epoch.py'])
    inputs = json.loads((root / 'inputs/identity.json').read_text())
    verify_hashes(root / 'inputs', inputs['sha256'],
                  ['parent.pt', 'split.json', 'cohort_indices.json', 'cases.json'])
    data_hashes = json.loads((source / 'training-data-hashes.json').read_text())
    train_files = [f'data_batch_{i}' for i in range(1, 6)]
    # Even integrity checks must not read the official test split.
    if not set(train_files) <= set(data_hashes):
        raise ValueError('Missing CIFAR training data hashes')
    verify_hashes(args.dataset_root, {name:data_hashes[name] for name in train_files}, train_files)
    case = json.loads((root / 'inputs/cases.json').read_text())[args.case]
    validate_case(case)
    checkpoint = torch.load(root / 'inputs/parent.pt', map_location='cpu', weights_only=False)
    config = copy.deepcopy(checkpoint['config'])
    if config.get('scheme') != 'ours' or config['batch_size'] != BATCH_SIZE or config['label_smoothing'] != 0:
        raise ValueError('Requires ours, batch32, unsmoothed-CE parent')
    if config.get('gradient_method') == METHOD:
        raise ValueError('Requires a native BPTT parent')
    config['epochs'] = max(config['epochs'], TRAINING_EPOCH)
    split = json.loads((root / 'inputs/split.json').read_text())
    cohort = json.loads((root / 'inputs/cohort_indices.json').read_text())
    if (len(split['validation']) != 5000 or len(set(split['validation'])) != 5000
            or set(split['train']) & set(split['validation'])
            or set(split['train']) | set(split['validation']) != set(range(50000))):
        raise ValueError('Invalid disjoint 45k/5k CIFAR split')
    if len(cohort) != 256 or len(set(cohort)) != 256 or not set(cohort) <= set(split['train']):
        raise ValueError('Requires 256 distinct training diagnostic indices')
    indices = selected_order(split['train'], config['seed'])
    order_hash = hashlib.sha256(json.dumps(indices).encode()).hexdigest()
    parent_steps = START_EPOCH * math.ceil(len(split['train']) / BATCH_SIZE)
    intervention = dict(case, starting_epoch=START_EPOCH, training_epoch=TRAINING_EPOCH,
                        examples=EXAMPLES, updates=UPDATES, scheduler_advanced=False,
                        native_epoch_resume_safe=False)
    reporting.start_run(run, dict(study_id=root.name, run_id=run.name, arm_id=args.case,
        seed=config['seed'], smoke=False, evidence_class='cifar_half_epoch_beta_training_diagnostic',
        resolved_config=config, intervention=intervention, source_identity=identity, inputs=inputs,
        training_order_sha256=order_hash, diagnostic_cohort_indices=cohort,
        dataset=dict(official_test_read=False, training_examples=EXAMPLES, validation_examples=5000,
                     evaluation_batch_size=config['evaluation_batch_size']),
        runtime=dict(reporting.runtime_context(target=args.target), pid=os.getpid()),
        environment=dict(torch=str(torch.__version__), cuda=torch.version.cuda),
        maximum_seconds=args.maximum_seconds, command=shlex.join([sys.executable, *sys.argv])))
    try:
        check_gpu_resident_execution('cuda', False)
        deterministic_setup(config['seed'])
        torch.cuda.reset_peak_memory_stats()
        reporting.atomic_write_json(run / 'artifacts/device.json',
            dict(name=torch.cuda.get_device_name(), index=torch.cuda.current_device()))
        images, labels = read_training_data(args.dataset_root)
        model = CifarL8Analog(config, 'cuda')
        optimizer, scheduler = make_optimizer(model, config)
        restore_parent(checkpoint, model, optimizer, scheduler, config, parent_steps)
        rates = scheduler.get_last_lr()
        config = copy.deepcopy(config)
        config['half_epoch_intervention'] = intervention
        controller = None
        if case['method'] == 'hybrid_eqprop':
            config['gradient_method'] = METHOD
            config['conv_eqprop'] = dict(policy='fixed', initial_betas=case['betas'],
                rms_targets=[.02, .035, .001], cohort_indices=cohort, relative_tolerance=.03,
                max_evaluations=1, cosine_epochs=[0, 1], read_noise=0.)
            controller = EpochBeta(model, config,
                loader(Cohort(images, labels, cohort, config), BATCH_SIZE),
                torch.device('cuda'), run, deadline)
            controller.state.update(initialization_reused=False, source_checkpoint_epoch=START_EPOCH,
                nominal_rms_anchor_targets=[.02, .035, .001], recalibrated=False)
        elif case['method'] == 'free_t_bptt':
            config['gradient_method'] = 'free_t_bptt'
        eval_batch_size = config['evaluation_batch_size']
        cohorts = dict(validation=loader(Cohort(images, labels, split['validation'], config), eval_batch_size),
                       train_cohort=loader(Cohort(images, labels, cohort, config), eval_batch_size))

        def observation(stage, updates):
            reporting.update_status_progress(run, dict(stage=stage + '_evaluation', updates=updates))
            with replay_without_training_effects(model):
                values = {name:evaluate(model, batches, config['iterations'], 'cuda', deadline)
                          for name, batches in cohorts.items()}
            beta = controller.observe(0 if stage == 'initial' else 1) if controller else None
            row = dict(stage=stage, updates=updates, adam_steps=parent_steps + updates,
                       measurements=values, learning_rates=rates,
                       elapsed_seconds=time.monotonic() - started,
                       training_order_sha256=order_hash)
            if beta is not None:
                row['beta_observation'] = dict(blocks=beta['blocks'], layers=beta['layers'],
                    diagnostic_index=beta['epoch'], checkpoint_epoch=START_EPOCH,
                    updates_in_epoch=updates, elapsed_seconds=beta['elapsed_seconds'])
            reporting.append_metric(run / 'metrics.jsonl', row)
            print('EVALUATION', stage, json.dumps(values), flush=True)
            return values

        initial = observation('initial', 0)
        save_partial(run / 'checkpoints/initial_model.pt', model, optimizer, scheduler, config,
                     partial_metadata(0, 0, parent_steps, order_hash))
        del checkpoint
        totals = dict(loss=0., correct=0, examples=0)
        batches = loader(Cohort(images, labels, indices, config, TRAINING_EPOCH, True),
                         BATCH_SIZE, config['num_workers'], True)
        for step, (x, y) in enumerate(batches, 1):
            if time.monotonic() > deadline:
                raise TimeoutError('Half-epoch training budget exhausted')
            model.train()
            x, y = x.cuda(), y.cuda()
            optimizer.zero_grad(set_to_none=True)
            prediction, loss = backward(model, x, y, config['iterations'], case)
            finite_parameters(model, gradients=True)
            optimizer.step()
            model.project_()
            model.detach_state_()
            finite_parameters(model)
            totals['loss'] += float(loss) * len(y)
            totals['correct'] += int((prediction.argmax(1) == y).sum())
            totals['examples'] += len(y)
            if step == 1 or step % 20 == 0 or step == UPDATES:
                row = dict(stage='training', updates=step, total_updates=UPDATES,
                           examples=totals['examples'], batch_loss=float(loss),
                           elapsed_seconds=time.monotonic() - started)
                reporting.append_metric(run / 'metrics.jsonl', row)
                reporting.update_status_progress(run, row)
                print('TRAINING', step, totals['examples'], float(loss), flush=True)
            del x, y, prediction, loss
        if step != UPDATES or totals['examples'] != EXAMPLES or scheduler.get_last_lr() != rates:
            raise RuntimeError('Half-epoch coverage or learning-rate mismatch')
        final = observation('final', step)
        progress = partial_metadata(step, totals['examples'], parent_steps, order_hash)
        save_partial(run / 'checkpoints/final_model.pt', model, optimizer, scheduler, config, progress)
        reporting.atomic_write_json(run / 'artifacts/partial_epoch.json', progress)
        verify_hashes(root / 'inputs', inputs['sha256'])
        reporting.complete_run(run, terminal_metrics=dict(initial=initial, final=final,
            train_loss=totals['loss'] / EXAMPLES, train_accuracy=totals['correct'] / EXAMPLES,
            updates=step, training_examples=EXAMPLES, adam_steps=parent_steps + step,
            learning_rates=rates, elapsed_seconds=time.monotonic() - started,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_reserved_bytes=torch.cuda.max_memory_reserved()),
            completion=dict(criteria_met=True, official_test_read=False, parent_unchanged=True,
                            full_epoch=False, native_epoch_resume_safe=False))
    except BaseException as exc:
        reporting.fail_run(run, error=exc)
        raise


if __name__ == '__main__':
    main()
