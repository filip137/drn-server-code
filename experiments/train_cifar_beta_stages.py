"""Fresh fixed-beta CIFAR training with exact staged continuations through epoch50."""
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
import signal
import sys
import time

import numpy as np
import torch

from experiments import reporting
from experiments.cifar_conv_eqprop import (
    EpochBeta, METHOD, hybrid_backward, initial_training_beta_state, next_training_beta_state,
    replay_without_training_effects, validate_training_beta_case, validate_training_beta_state,
)
from experiments.screen_cifar_beta_half_epoch import backward, finite_parameters, validate_case, verify_hashes
from experiments.train_cifar_l8_analog import (
    Cohort, check_gpu_resident_execution, deterministic_setup, evaluate, loader,
    make_optimizer, read_training_data,
)
from labs.cifar_l8_analog import CifarL8Analog

TRAIN_EXAMPLES = 45000
BATCH_SIZE = 32
EPOCH_UPDATES = math.ceil(TRAIN_EXAMPLES / BATCH_SIZE)
STAGES = ('1', '5', '10', '20', '30', '50')
DIAGNOSTIC_EPOCHS = tuple(map(int, STAGES))


def epoch_order(indices, seed, epoch):
    permutation = torch.randperm(len(indices), generator=torch.Generator().manual_seed(
        seed + 1000000 * epoch)).tolist()
    return [indices[i] for i in permutation]


def epoch_loader(dataset, batch_size, workers, cuda, seed, epoch):
    batches = loader(dataset, batch_size, workers, cuda)
    # DataLoader otherwise consumes global RNG when a resumed iterator starts.
    batches.generator = torch.Generator().manual_seed(seed + 2000000 * epoch)
    return batches


def cursor(completed_epoch=0, next_batch=0, *, train_examples=TRAIN_EXAMPLES, batch_size=BATCH_SIZE):
    per_epoch = math.ceil(train_examples / batch_size)
    if completed_epoch < 0 or not 0 <= next_batch < per_epoch:
        raise ValueError('Invalid staged cursor')
    return dict(completed_epoch=completed_epoch, next_epoch=completed_epoch + 1,
                next_batch=next_batch, examples_in_epoch=next_batch * batch_size,
                total_updates=completed_epoch * per_epoch + next_batch)


def advance_cursor(progress, *, train_examples=TRAIN_EXAMPLES, batch_size=BATCH_SIZE):
    next_batch = progress['next_batch'] + 1
    full = next_batch == math.ceil(train_examples / batch_size)
    return cursor(progress['completed_epoch'] + int(full), 0 if full else next_batch,
                  train_examples=train_examples, batch_size=batch_size), full


def save_state(path, model, optimizer, scheduler, config, case_id, case, progress,
               input_identity, last_observation=None):
    if scheduler.last_epoch != progress['completed_epoch']:
        raise ValueError('Checkpoint scheduler and cursor disagree')
    steps = {int(value['step']) for value in optimizer.state.values()}
    if steps != ({progress['total_updates']} if progress['total_updates'] else set()):
        raise ValueError('Checkpoint Adam steps and cursor disagree')
    if case['method'] == 'hybrid_eqprop':
        validate_training_beta_state(getattr(model, 'conv_beta_state', None), case, progress['total_updates'])
    payload = dict(model=model.snapshot(), optimizer=optimizer.state_dict(),
        scheduler=scheduler.state_dict(), config=copy.deepcopy(config), epoch=progress['completed_epoch'],
        staged_cursor=copy.deepcopy(progress), case_id=case_id, case=copy.deepcopy(case),
        input_identity=copy.deepcopy(input_identity), last_observation=copy.deepcopy(last_observation),
        native_epoch_resume_safe=progress['next_batch'] == 0,
        torch_rng=torch.get_rng_state(), numpy_rng=np.random.get_state(), python_rng=random.getstate(),
        cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None)
    if hasattr(model, 'conv_beta_state'):
        payload['conv_beta_state'] = copy.deepcopy(model.conv_beta_state)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    torch.save(payload, temporary)
    os.replace(temporary, path)


def restore_state(checkpoint, model, optimizer, scheduler, config, case_id, case,
                  input_identity, *, train_examples=TRAIN_EXAMPLES, batch_size=BATCH_SIZE):
    for key, expected in [('config', config), ('case_id', case_id), ('case', case),
                          ('input_identity', input_identity)]:
        if checkpoint[key] != expected:
            raise ValueError('Resume mismatch: ' + key)
    progress = checkpoint['staged_cursor']
    if progress != cursor(progress['completed_epoch'], progress['next_batch'],
                          train_examples=train_examples, batch_size=batch_size):
        raise ValueError('Resume cursor mismatch')
    if checkpoint['epoch'] != progress['completed_epoch'] or checkpoint['scheduler']['last_epoch'] != checkpoint['epoch']:
        raise ValueError('Resume scheduler boundary mismatch')
    model.restore(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    scheduler.load_state_dict(checkpoint['scheduler'])
    steps = {int(value['step']) for value in optimizer.state.values()}
    if steps != ({progress['total_updates']} if progress['total_updates'] else set()):
        raise ValueError('Resume Adam steps mismatch')
    if [g['lr'] for g in optimizer.param_groups] != scheduler.get_last_lr():
        raise ValueError('Resume learning rates mismatch')
    if case['method'] == 'hybrid_eqprop':
        state = checkpoint.get('conv_beta_state')
        validate_training_beta_state(state, case, progress['total_updates'])
        model.conv_beta_state = copy.deepcopy(state)
    torch.set_rng_state(checkpoint['torch_rng'].cpu())
    np.random.set_state(checkpoint['numpy_rng'])
    random.setstate(checkpoint['python_rng'])
    if checkpoint.get('cuda_rng') is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([state.cpu() for state in checkpoint['cuda_rng']])
    return copy.deepcopy(progress)


def stage_target(stage):
    if stage not in STAGES:
        raise ValueError('Stages are 1, 5, 10, 20, 30 and 50 complete epochs')
    return int(stage) * EPOCH_UPDATES


def stage_bounds(stage):
    target = stage_target(stage)
    previous = {'1': 0, '5': 1, '10': 5, '20': 10, '30': 20, '50': 30}[stage]
    return previous * EPOCH_UPDATES, target


def validate_config(config):
    if (config['scheme'] not in ('baseline', 'ours', 'legacy')
            or config['batch_size'] != BATCH_SIZE or config['seed'] != 0
            or config['evaluation_batch_size'] != 16 or config['iterations'] != [6, 6, 4]
            or config['label_smoothing'] != 0 or config['scheduler']['horizon_epochs'] != 50):
        raise ValueError('Config differs from the assigned fresh CIFAR contract')


def diagnostic_config(config, case, cohort):
    resolved = copy.deepcopy(config)
    resolved.update(gradient_method=METHOD, conv_eqprop=dict(policy='fixed',
        initial_betas=copy.deepcopy(case['betas']),
        rms_targets=copy.deepcopy(case.get('beta_search', case.get('beta_feedback', {})).get('targets',
            case.get('rms_targets', [.02, .035, .001]))), cohort_indices=cohort,
        relative_tolerance=.03, max_evaluations=1, cosine_epochs=[0, *DIAGNOSTIC_EPOCHS], read_noise=0.))
    return resolved


def training_update(model, optimizer, x, y, iterations, case):
    """Compute declared beta selection, then commit controller state after the update."""
    proposal = telemetry = None
    if validate_training_beta_case(case):
        prediction, loss, measured = hybrid_backward(model, x, y, iterations,
            model.conv_beta_state['values'], measure_displacement=True, beta_search=case.get('beta_search'))
        proposal, telemetry = next_training_beta_state(model.conv_beta_state, case, measured)
    else:
        prediction, loss = backward(model, x, y, iterations, case)
    finite_parameters(model, gradients=True)
    optimizer.step()
    model.project_()
    model.detach_state_()
    finite_parameters(model)
    if proposal is not None:
        model.conv_beta_state = proposal
    return prediction, loss, telemetry


def verified_nonfinite(error, model):
    """Missing gradients and dead forces are operational errors, not beta failures."""
    if not isinstance(error, FloatingPointError):
        return False
    if str(error) in {'Nonfinite training loss', 'Nonfinite hybrid loss',
                      'Nonfinite evaluation', 'Nonfinite parameter gradient', 'Nonfinite training displacement'}:
        return True
    if model is not None:
        for parameter in model.trainable_tensors().values():
            if not torch.isfinite(parameter).all():
                return True
            if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                return True
    return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--case', required=True)
    parser.add_argument('--stage', choices=STAGES, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--dataset-root', type=Path, required=True)
    parser.add_argument('--target', required=True)
    parser.add_argument('--maximum-seconds', type=float, required=True)
    parser.add_argument('--resume-checkpoint', type=Path)
    args = parser.parse_args()
    started = time.monotonic()
    if not math.isfinite(args.maximum_seconds) or args.maximum_seconds <= 60:
        raise ValueError('Runtime budget must exceed checkpoint reserve of60 seconds')
    deadline = started + args.maximum_seconds - 60
    root, run = args.root.resolve(), args.output_dir.resolve()
    if run.exists():
        raise FileExistsError(run)
    source = Path(__file__).resolve().parents[1]
    identity = json.loads((source / 'source_identity.json').read_text())
    verify_hashes(source, identity['source_sha256'], ['experiments/train_cifar_beta_stages.py'])
    inputs = json.loads((root / 'inputs/identity.json').read_text())
    verify_hashes(root / 'inputs', inputs['sha256'],
        ['initializer.pt', 'config.json', 'split.json', 'cohort_indices.json', 'cases.json'])
    data_hashes = json.loads((source / 'training-data-hashes.json').read_text())
    names = [f'data_batch_{i}' for i in range(1, 6)]
    verify_hashes(args.dataset_root, {n:data_hashes[n] for n in names}, names)
    config = json.loads((root / 'inputs/config.json').read_text())
    case = json.loads((root / 'inputs/cases.json').read_text())[args.case]
    validate_case(case)
    measure_training_rms = validate_training_beta_case(case)
    validate_config(config)
    split = json.loads((root / 'inputs/split.json').read_text())
    cohort = json.loads((root / 'inputs/cohort_indices.json').read_text())
    if (len(split['train']) != TRAIN_EXAMPLES or len(set(split['train'])) != TRAIN_EXAMPLES
            or len(split['validation']) != 5000 or len(set(split['validation'])) != 5000
            or set(split['train']) & set(split['validation'])
            or set(split['train']) | set(split['validation']) != set(range(50000))):
        raise ValueError('Invalid45k/5k split')
    if len(cohort) != 256 or len(set(cohort)) != 256 or not set(cohort) <= set(split['train']):
        raise ValueError('Invalid diagnostic training cohort')
    if args.stage != '1' and args.resume_checkpoint is None:
        raise ValueError('Extensions require the exact preceding stage checkpoint')
    resume_sha = reporting.sha256_file(args.resume_checkpoint) if args.resume_checkpoint else None
    reporting.start_run(run, dict(study_id=root.name, run_id=run.name, arm_id=args.case,
        stage=args.stage, seed=0, smoke=False, evidence_class='cifar_fresh_staged_beta_training',
        resolved_config=config, case=case, source_identity=identity, inputs=inputs,
        resume_checkpoint=str(args.resume_checkpoint) if args.resume_checkpoint else None,
        resume_checkpoint_sha256=resume_sha, maximum_seconds=args.maximum_seconds,
        dataset=dict(official_test_read=False, train_examples=45000, validation_examples=5000,
                     batch_size=32, evaluation_batch_size=16),
        runtime=dict(reporting.runtime_context(target=args.target), pid=os.getpid()),
        environment=dict(torch=str(torch.__version__), cuda=torch.version.cuda),
        command=shlex.join([sys.executable, *sys.argv])))
    stop = []
    old_handlers = {sig:signal.signal(sig, lambda number, frame: stop.append(number))
                    for sig in (signal.SIGTERM, signal.SIGINT)}
    progress = None
    model = None
    last_observation = None
    try:
        check_gpu_resident_execution('cuda', False)
        deterministic_setup(config['seed'])
        torch.cuda.reset_peak_memory_stats()
        images, labels = read_training_data(args.dataset_root)
        model = CifarL8Analog(config, 'cuda')
        optimizer, scheduler = make_optimizer(model, config)
        controller = None
        if case['method'] == 'hybrid_eqprop':
            controller = EpochBeta(model, diagnostic_config(config, case, cohort),
                loader(Cohort(images, labels, cohort, config), BATCH_SIZE), torch.device('cuda'), run, deadline)
            if measure_training_rms:
                model.conv_beta_state = initial_training_beta_state(case)
            controller.state.update(initialization_reused=False, beta_source_checkpoint_epoch=10, recalibrated=False)
        if args.resume_checkpoint:
            checkpoint = torch.load(args.resume_checkpoint, map_location='cpu', weights_only=False)
            progress = restore_state(checkpoint, model, optimizer, scheduler, config, args.case, case, inputs)
            last_observation = checkpoint.get('last_observation')
            del checkpoint
        else:
            initial = torch.load(root / 'inputs/initializer.pt', map_location='cpu', weights_only=False)
            if initial['epoch'] != 0:
                raise ValueError('Fresh initialization requires epoch0')
            model.restore(initial['model'])
            restored_model = model.snapshot()
            for group, values in initial['model'].items():
                restored = restored_model[group]
                if any(not torch.equal(value.cpu(), restored[name].cpu()) for name, value in values.items()):
                    raise ValueError('Initializer restore mismatch')
            del initial, restored_model, restored
            progress = cursor()
        minimum, target = stage_bounds(args.stage)
        if not minimum <= progress['total_updates'] <= target:
            raise ValueError('Checkpoint outside the requested stage')
        start_cursor = copy.deepcopy(progress)
        evaluation = dict(validation=loader(Cohort(images, labels, split['validation'], config), 16),
                          train_cohort=loader(Cohort(images, labels, cohort, config), 16))

        def save(name):
            save_state(run / 'checkpoints' / name, model, optimizer, scheduler, config,
                       args.case, case, progress, inputs, last_observation)

        def guard():
            if stop or time.monotonic() >= deadline:
                raise TimeoutError('Signal requested pause' if stop else 'Stage runtime reserve reached')

        def observe(label, diagnostic=None):
            nonlocal last_observation
            guard()
            reporting.update_status_progress(run, dict(stage='validation', label=label, **progress))
            with replay_without_training_effects(model):
                measured = {name:evaluate(model, batches, config['iterations'], 'cuda', deadline)
                            for name, batches in evaluation.items()}
            beta = controller.observe(diagnostic) if controller is not None and diagnostic is not None else None
            row = dict(stage=label, cursor=copy.deepcopy(progress), measurements=measured,
                       learning_rates=scheduler.get_last_lr(), elapsed_seconds=time.monotonic() - started)
            if beta is not None:
                row['beta_observation'] = dict(blocks=beta['blocks'], layers=beta['layers'],
                    diagnostic_index=diagnostic, examples=beta['examples'])
            reporting.append_metric(run / 'metrics.jsonl', row)
            last_observation = row
            print('EVALUATION', label, progress['total_updates'], json.dumps(measured), flush=True)

        save('initial_model.pt')
        if progress['total_updates'] == 0:
            observe('initial', 0)
        elif progress['next_batch'] == 0 and (last_observation is None or last_observation['cursor'] != progress):
            epoch = progress['completed_epoch']
            observe('epoch', epoch if epoch in DIAGNOSTIC_EPOCHS else None)
            save(f'epoch_{epoch:03}.pt')
        while progress['total_updates'] < target:
            guard()
            epoch = progress['next_epoch']
            order = epoch_order(split['train'], config['seed'], epoch)
            order_hash = hashlib.sha256(json.dumps(order).encode()).hexdigest()
            indices = order[progress['examples_in_epoch']:]
            batches = epoch_loader(Cohort(images, labels, indices, config, epoch, True), BATCH_SIZE,
                                   config['num_workers'], True, config['seed'], epoch)
            for x, y in batches:
                guard()
                model.train()
                optimizer.zero_grad(set_to_none=True)
                x, y = x.cuda(), y.cuda()
                prediction, loss, beta_telemetry = training_update(model, optimizer, x, y, config['iterations'], case)
                progress, full = advance_cursor(progress)
                if full:
                    scheduler.step()
                report_progress = progress['total_updates'] % 20 == 0 or full or progress['total_updates'] == target
                if beta_telemetry is not None or report_progress:
                    row = dict(stage='training', cursor=copy.deepcopy(progress), loss=float(loss),
                               training_order_sha256=order_hash, elapsed_seconds=time.monotonic() - started)
                    if beta_telemetry is not None:
                        row.update(beta_telemetry)
                    reporting.append_metric(run / 'metrics.jsonl', row)
                if report_progress:
                    reporting.update_status_progress(run, row)
                    print('TRAINING', progress['total_updates'], float(loss), flush=True)
                del x, y, prediction, loss
                if full:
                    # Advance schedule/cursor before saving; validation can be recovered separately.
                    save(f'epoch_{epoch:03}.pt')
                    observe('epoch', epoch if epoch in DIAGNOSTIC_EPOCHS else None)
                    save(f'epoch_{epoch:03}.pt')
                if progress['total_updates'] == target:
                    break
        if last_observation is None or last_observation['cursor'] != progress:
            observe('epoch', int(args.stage))
        save('final_model.pt')
        reporting.atomic_write_json(run / 'artifacts/staged_cursor.json', progress)
        reporting.append_metric(run / 'metrics.jsonl', dict(stage='final', cursor=progress,
            measurements=last_observation['measurements'], learning_rates=scheduler.get_last_lr()))
        verify_hashes(root / 'inputs', inputs['sha256'])
        reporting.complete_run(run, terminal_metrics=dict(stage=args.stage, starting_cursor=start_cursor,
            cursor=progress, final=last_observation['measurements'], updates=progress['total_updates'],
            stage_updates=progress['total_updates'] - start_cursor['total_updates'],
            learning_rates=scheduler.get_last_lr(), elapsed_seconds=time.monotonic() - started,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(), peak_reserved_bytes=torch.cuda.max_memory_reserved()),
            completion=dict(criteria_met=True, official_test_read=False, initializer_unchanged=True,
                            native_epoch_resume_safe=progress['next_batch'] == 0))
    except TimeoutError as exc:
        if progress is not None:
            path = run / 'checkpoints/paused_model.pt'
            save_state(path, model, optimizer, scheduler, config, args.case, case, progress, inputs, last_observation)
            reporting.pause_run(run, checkpoint=path, progress=progress, reason=str(exc))
        else:
            reporting.fail_run(run, error=exc)
        raise SystemExit(75) from exc
    except FloatingPointError as exc:
        reporting.fail_run(run, error=exc)
        if verified_nonfinite(exc, model):
            reporting.atomic_write_json(run / 'artifacts/scientific_rejection.json',
                dict(case=args.case, stage=args.stage, kind='nonfinite', reason=str(exc),
                     cursor=progress, training_successful=False))
            print('SCIENTIFIC_REJECTION', args.case, str(exc), flush=True)
            return
        raise
    except BaseException as exc:
        reporting.fail_run(run, error=exc)
        raise
    finally:
        for sig, handler in old_handlers.items():
            signal.signal(sig, handler)


if __name__ == '__main__':
    main()
