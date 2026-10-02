"""Frozen-checkpoint nonlinear readout control: calibrated, matched local replay."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import torch
from torch.nn import functional as F

from experiments import reporting
from experiments.calibrate_cifar_block_beta import (
    FrozenOutputCost, capture_batches, measure_block, nudged_minimizer, pooled_rms, search_beta,
)
from experiments.compare_cifar_block_beta_gradients import (
    accumulate_displacement, agreement, check_environment, eqprop_gradients,
    local_bptt_gradients, state_measurement_fields,
)
from experiments.replay_cifar_precision_response import check_snapshot, convert_model, verify_dtype
from experiments.train_cifar_l8_analog import Cohort, deterministic_setup, loader, make_split, read_training_data
from labs.cifar_l8_analog import CifarL8Analog
from training.sgd import AugmentedFunction

TARGETS = [[.0005, .0015, .005, .015, .05, .15],
           [.02, .03, .05, .075, .1, .15],
           [.0005, .0015, .005, .015, .05, .15, .3]]
TRAINED_TARGETS = [[.001, .0025, .005, .01, .025, .05],
                   [.001, .0025, .005, .01, .025, .05],
                   [.00001, .00005, .00015, .0005, .0015, .005, .015]]
DEPTHS = (3, 3, 2)
PRECISIONS = ('float64', 'float32')
INITIAL_BETAS = (0.0005857333362799141, 5.431465723731432e-5, 2.204854001994856e-6)
INITIAL_TARGETS = (.01, .005, 1e-5)


class Deadline:
    def __init__(self, seconds):
        if not math.isfinite(seconds) or seconds <= 0:
            raise ValueError('Budget must be positive and finite')
        self.start = time.monotonic()
        self.seconds = seconds

    def check(self):
        if self.elapsed >= self.seconds:
            raise TimeoutError('Checkpoint replay budget exhausted')

    @property
    def elapsed(self):
        return time.monotonic() - self.start


def tensor_hash(value):
    tensor = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(json.dumps({'dtype': str(tensor.dtype), 'shape': list(tensor.shape)}, sort_keys=True).encode())
    digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def snapshot_hashes(snapshot):
    hashes = {group: {name: tensor_hash(value) for name, value in values.items()}
              for group, values in snapshot.items()}
    trunk = {group: {name: digest for name, digest in values.items() if not name.startswith('head.')}
             for group, values in hashes.items()}
    return dict(tensors=hashes, all_sha256=hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest(),
                trunk_sha256=hashlib.sha256(json.dumps(trunk, sort_keys=True).encode()).hexdigest())


def capture_hashes(saved):
    return dict(input=tensor_hash(saved['input']), states=[tensor_hash(s) for s in saved['states']])


def source_hashes():
    root = Path(__file__).resolve().parents[1]
    return {str(path.relative_to(root)): reporting.sha256_file(path)
            for directory in ('experiments', 'labs', 'model', 'training')
            for path in sorted((root / directory).rglob('*.py')) if '__pycache__' not in path.parts}


def checkpoint_epoch(config):
    epoch = config.get('checkpoint_epoch', 0)
    if type(epoch) is not int or epoch not in (0, 1, 5):
        raise ValueError('Replay checkpoint_epoch must be 0, 1 or 5')
    return epoch


def validate_target_grid(config):
    expected = TRAINED_TARGETS if checkpoint_epoch(config) else TARGETS
    if config['relative_rms_targets'] != expected:
        raise ValueError('Replay requires the exact target grids declared for its checkpoint epoch')


def validate_training_provenance(config, checkpoint_hash):
    """Bind trained diagnostics to a declared float32 BPTT training protocol."""
    if checkpoint_epoch(config) == 0:
        return
    provenance = config.get('training_provenance', {})
    if provenance.get('training_checkpoint_sha256') != checkpoint_hash:
        raise ValueError('Training provenance checkpoint hash mismatch')
    digest = provenance.get('initial_trunk_sha256', '')
    if not isinstance(digest, str) or len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
        raise ValueError('Training provenance requires initializer trunk SHA-256')
    if not provenance.get('training_run') or not provenance.get('training_source_sha256'):
        raise ValueError('Training provenance requires run and source identity')
    training_config = provenance.get('training_config')
    if not isinstance(training_config, dict) or training_config != config['model_config']:
        raise ValueError('Replay model config differs from exact training config')
    if (training_config.get('gradient_method') != 'bptt' or
            training_config.get('precision') != 'float32' or training_config.get('epochs') != 5):
        raise ValueError('Trained replay requires five-epoch float32 BPTT training')


def validate_checkpoint(checkpoint, config, checkpoint_hash):
    epoch = checkpoint_epoch(config)
    if type(checkpoint.get('epoch')) is not int or checkpoint['epoch'] != epoch:
        raise ValueError('Actual checkpoint epoch differs from requested replay epoch')
    validate_training_provenance(config, checkpoint_hash)
    if epoch:
        if checkpoint.get('config') != config['training_provenance']['training_config']:
            raise ValueError('Checkpoint training config differs from declared provenance')
        if checkpoint.get('precision', 'float32') != 'float32':
            raise ValueError('Training checkpoint precision differs from float32')
        for values in checkpoint['model'].values():
            for tensor in values.values():
                if tensor.is_floating_point() and tensor.dtype != torch.float32:
                    raise ValueError('Training checkpoint contains non-float32 model tensors')


def weighted_mean(tensors, sizes):
    if not tensors or len(tensors) != len(sizes) or any(size <= 0 for size in sizes):
        raise ValueError('Gradient pooling requires positive matching batch sizes')
    result = tensors[0] * sizes[0]
    for value, size in zip(tensors[1:], sizes[1:]):
        result.add_(value, alpha=size)
    return result / sum(sizes)


def calibration_search(measure, target, initial_beta):
    """Retain every measurement even when the bounded search misses a target."""
    history = []

    def recorded(beta):
        measured = measure(beta)
        history.append(dict(beta_injected_B=beta, relative_target_error=abs(measured['relative_rms'] / target - 1),
                            **measured))
        return measured

    try:
        beta, _ = search_beta(recorded, target, metric='relative_rms', tolerance=.03,
                              max_evaluations=12, initial_beta=initial_beta)
        return dict(status='matched', beta_injected_B=beta, achieved=history[-1], history=history)
    except RuntimeError as error:
        # OOM, identity, nonfinite and other unexpected failures must propagate.
        message = str(error)
        if not (message == 'Beta search exceeded declared bounds' or
                message.startswith('No target match within 12 measurements; last=')):
            raise
        return dict(status='missing', beta_injected_B=None, reason=message, history=history)


def output_displacement(displacement):
    return dict(pooled_rms(displacement['centered_squares'], displacement['free_squares'],
                           displacement['elements']),
                centered_squares=displacement['centered_squares'], free_squares=displacement['free_squares'],
                unchanged_elements=displacement['unchanged_elements'],
                unchanged_fraction=displacement['unchanged_elements'] / displacement['elements'])


def validate_evidence(config, calibration, comparison, identity):
    """Operational completeness allows explicit calibration misses, never absent rows."""
    validate_target_grid(config)
    expected = {(block + 1, index): target for block, targets in enumerate(config['relative_rms_targets'])
                for index, target in enumerate(targets)}
    actual = {(row['block'], row['target_index']): row for row in calibration}
    if len(actual) != len(calibration) or set(actual) != set(expected):
        raise ValueError('Not every declared target was attempted exactly once')
    missing = []
    matched = set()
    for key, row in actual.items():
        if row['target_relative_rms'] != expected[key] or not 1 <= len(row['history']) <= 12:
            raise ValueError('Calibration target/history mismatch')
        if row['status'] == 'missing':
            if row['beta_injected_B'] is not None or not row.get('reason'):
                raise ValueError('Malformed missing calibration')
            missing.append(dict(block=key[0], target_index=key[1], target_relative_rms=expected[key]))
        elif row['status'] == 'matched':
            if row['beta_injected_B'] <= 0 or row['history'][-1]['relative_target_error'] > .03:
                raise ValueError('Calibration match outside declared tolerance')
            matched.add(key)
        else:
            raise ValueError('Unknown calibration outcome')
    for field, batches in (('layers', [None]), ('batches', range(4))):
        expected_rows = {(precision, block, index, layer, batch)
                         for block, index in matched for precision in PRECISIONS
                         for layer in range(1, DEPTHS[block-1]+1) for batch in batches}
        actual_rows = set()
        for row in comparison[field]:
            key = (row['precision'], row['block'], row['target_index'], row['layer'], row.get('batch'))
            if key in actual_rows:
                raise ValueError('Duplicate gradient comparison')
            actual_rows.add(key)
            cal = actual[row['block'], row['target_index']]
            if row['beta_injected_B'] != cal['beta_injected_B'] or row['target_relative_rms'] != cal['target_relative_rms']:
                raise ValueError('Comparison does not match calibration')
            if row['examples'] != (128 if field == 'layers' else 32):
                raise ValueError('Incomplete batch or pooled cohort')
            for name in ('common_full64_local_bptt', 'own_precision_local_bptt', 'state_displacement',
                         'output_displacement', 'output_force_rms', 'injected_force_rms'):
                if name not in row:
                    raise ValueError(f'Missing comparison measurement: {name}')
        if actual_rows != expected_rows:
            raise ValueError(f'Incomplete successful-target {field} coverage')
    if comparison['cohort_indices'] != config['cohort_indices']:
        raise ValueError('Ordered cohort changed')
    for field in ('parameters_and_bn_unchanged', 'checkpoint_bytes_unchanged', 'precision_checks_passed',
                  'config_bytes_unchanged', 'source_bytes_unchanged'):
        if identity.get(field) is not True:
            raise ValueError(f'Immutable identity verification missing: {field}')
    captures = identity['captures']
    if len(captures) != 24 or {(r['precision'], r['batch'], r['block']) for r in captures} != {
            (p, b, k) for p in PRECISIONS for b in range(4) for k in range(1, 4)}:
        raise ValueError('Capture identity coverage incomplete')
    return dict(targets_attempted=len(actual), targets_matched=len(matched), coverage_missing=missing,
                scientific_coverage_complete=not missing, pooled_rows=len(comparison['layers']),
                batch_rows=len(comparison['batches']))


def validate_run(run_dir):
    errors = reporting.validate_run(run_dir)
    if errors:
        raise ValueError(errors)
    manifest = json.loads((run_dir / 'manifest.json').read_text())
    result = json.loads((run_dir / 'result.json').read_text())
    epoch = checkpoint_epoch(manifest['resolved_config'])
    validate_training_provenance(manifest['resolved_config'], manifest['checkpoint_sha256'])
    if manifest.get('checkpoint_epoch', 0) != epoch:
        raise ValueError('Manifest checkpoint epoch differs from config')
    artifacts = run_dir / 'artifacts'
    identity = json.loads((artifacts / 'identity.json').read_text())
    if identity.get('checkpoint_epoch', 0) != epoch or (epoch and not identity.get('training_checkpoint_verified')):
        raise ValueError('Checkpoint training identity verification missing')
    coverage = validate_evidence(manifest['resolved_config'],
        json.loads((artifacts / 'calibration.json').read_text()),
        json.loads((artifacts / 'comparison.json').read_text()),
        identity)
    if coverage != result['terminal_metrics']['coverage']:
        raise ValueError('Sealed coverage does not match evidence')
    return coverage


def run(args):
    deadline = Deadline(args.budget_seconds)
    out = args.run_dir
    config = json.loads(args.config.read_text())
    model_config = config['model_config']
    epoch = checkpoint_epoch(config)
    indices = config['cohort_indices']
    validate_target_grid(config)
    if len(indices) != 128 or len(set(indices)) != 128:
        raise ValueError('Pilot requires its exact 128-image cohort')
    if model_config['iterations'] != [6, 6, 4] or model_config.get('head_hidden_units', 0) not in (0, 512):
        raise ValueError('Pilot architecture/TK mismatch')
    if model_config.get('head_iterations', 4) != 4:
        raise ValueError('Pilot nonlinear head requires four iterations')
    if (model_config['solver']['voltage_amp'] != 4 or model_config['solver']['current_amp'] != 1
            or model_config['input_gain_init'] != 100 or model_config['seed'] != 0):
        raise ValueError('Pilot requires ours V4/C1, gain100 and seed0')
    hidden = model_config.get('head_hidden_units', 0)
    if config['arm_id'] != ('hidden512' if hidden else 'single'):
        raise ValueError('Arm label does not match head architecture')
    device = torch.device(args.device)
    environment = check_environment(config.get('expected_environment'))
    checkpoint_hash = reporting.sha256_file(args.checkpoint)
    if config.get('checkpoint_sha256', checkpoint_hash) != checkpoint_hash:
        raise ValueError('Checkpoint hash mismatch')
    validate_training_provenance(config, checkpoint_hash)
    config_hash = reporting.sha256_file(args.config)
    sources = source_hashes()
    for path, digest in config.get('source_sha256', {}).items():
        if sources.get(path) != digest:
            raise ValueError(f'Source hash mismatch: {path}')
    data_hashes = {f'data_batch_{i}': reporting.sha256_file(args.dataset_root / f'data_batch_{i}') for i in range(1, 6)}
    reporting.start_run(out, dict(study_id=config['study_id'], run_id=out.name, arm_id=config['arm_id'],
        seed=model_config['seed'], smoke=False, checkpoint_epoch=epoch,
        evidence_class=('cifar_trained_nonlinear_readout_diagnostic' if epoch else
                        'cifar_initialization_nonlinear_readout_diagnostic'),
        resolved_config=config, command=sys.argv, target=args.target, budget_seconds=args.budget_seconds,
        config_sha256=config_hash, checkpoint_sha256=checkpoint_hash, source_sha256=sources,
        training_data_sha256=data_hashes, environment=environment, runtime=reporting.runtime_context(),
        dataset='CIFAR10 fixed original training cohort', official_test_read=False,
        precision_order=list(PRECISIONS), head_hidden_units=hidden, head_iterations=4,
        common_reference='float64 matched block-local finite-iteration BPTT',
        immutable_bn_policy='minibatch statistics; unchanged running buffers', device=str(device)))
    calibration, comparison = [], dict(layers=[], batches=[], cohort_indices=indices)
    identity = dict(captures=[], models={})
    diagnostics = []

    def progress(stage, **fields):
        deadline.check()
        value = dict(stage=stage, elapsed_seconds=deadline.elapsed, **fields)
        reporting.update_status_progress(out, value)
        print(json.dumps(value), flush=True)

    def save():
        reporting.atomic_write_json(out / 'artifacts/calibration.json', calibration)
        reporting.atomic_write_json(out / 'artifacts/comparison.json', comparison)
        reporting.atomic_write_json(out / 'artifacts/identity.json', identity)
        reporting.atomic_write_json(out / 'artifacts/head_diagnostics.json', diagnostics)

    try:
        deterministic_setup(model_config['seed'])
        checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
        validate_checkpoint(checkpoint, config, checkpoint_hash)
        identity.update(checkpoint_epoch=epoch, training_checkpoint_verified=bool(epoch))
        images, labels = read_training_data(args.dataset_root)
        if not set(indices) <= set(make_split(labels, model_config['split']['seed'])['train']):
            raise ValueError('Cohort contains non-training examples')
        batches = list(loader(Cohort(images, labels, indices, model_config, augment=False), 32))
        del images, labels
        models, snapshots, captures, references = {}, {}, {}, {}
        for precision in PRECISIONS:
            progress('construct', precision=precision)
            dtype = getattr(torch, precision)
            model = CifarL8Analog(model_config, device)
            # Restore only AFTER conversion: double initializers must not pass through float32.
            convert_model(model, dtype)
            model.restore(checkpoint['model'])
            snapshot = model.snapshot()
            check_snapshot(checkpoint['model'], snapshot)
            verify_dtype(model, dtype)
            snapshots[precision], models[precision] = snapshot, model
            identity['models'][precision] = snapshot_hashes(snapshot)
            captures[precision], references[precision] = [], []
            for batch, (x, y) in enumerate(batches):
                progress('capture', precision=precision, batch=batch)
                def head_hook(_module, _inputs, logits):
                    value = logits.detach()
                    if not torch.isfinite(value).all():
                        raise FloatingPointError('Nonfinite head logits')
                    diagnostics.append(dict(precision=precision, batch=batch, examples=len(y),
                        logits=value.cpu().double().tolist(),
                        cross_entropy=float(F.cross_entropy(value, y.to(device))),
                        logits_rms=float(value.double().square().mean().sqrt())))
                hook = model.head.register_forward_hook(head_hook)
                try:
                    saved = capture_batches(model, [(x.to(dtype=dtype), y)], model_config['iterations'], device)[0]
                finally:
                    hook.remove()
                if not all(torch.isfinite(value).all() for record in saved
                           for value in (record['input'], *record['states'], record['gradient'])):
                    raise FloatingPointError('Nonfinite captured free state or loss force')
                captures[precision].append(saved)
                local = []
                for block_index, block in enumerate(model.analog_blocks):
                    progress('local_reference', precision=precision, batch=batch, block=block_index + 1)
                    identity['captures'].append(dict(precision=precision, batch=batch, block=block_index + 1,
                                                    **capture_hashes(saved[block_index])))
                    local.append(local_bptt_gradients(block, saved[block_index],
                                                     model_config['iterations'][block_index], device))
                    if not all(torch.isfinite(value).all() for value in local[-1]):
                        raise FloatingPointError('Nonfinite local BPTT reference')
                references[precision].append(local)
                save()
            check_snapshot(snapshot, model.snapshot())
        del checkpoint
        names = [[f'blocks.{i}.weights.{j}' for j in range(depth)] for i, depth in enumerate(DEPTHS)]
        actual_names = [name for name, _ in models['float64'].named_conductances() if name.startswith('blocks.')]
        if actual_names != [name for group in names for name in group]:
            raise ValueError('Unexpected convolutional conductance layout')
        sizes = [len(y) for _, y in batches]
        for block_index, targets in enumerate(config['relative_rms_targets']):
            block64 = models['float64'].analog_blocks[block_index]
            cost = FrozenOutputCost(block64.free_layers()[-1])
            augmented = AugmentedFunction(block64.energy, cost, nudging_mode='current', current_scale=1.)
            minimizer = nudged_minimizer(block64, augmented, model_config['iterations'][block_index])
            previous_beta, previous_target = INITIAL_BETAS[block_index], INITIAL_TARGETS[block_index]
            for target_index, target in enumerate(targets):
                progress('calibrate_target', block=block_index + 1, target_index=target_index, target_relative_rms=target)
                evaluations = 0
                def measure(beta):
                    nonlocal evaluations
                    evaluations += 1
                    centered = free = 0.
                    elements = 0
                    for batch, record in enumerate(captures['float64']):
                        progress('calibration_measure', block=block_index + 1, target_index=target_index,
                                 evaluation=evaluations, batch=batch, beta_injected_B=beta)
                        value = measure_block(block64, minimizer, augmented, cost, [record], block_index, beta, device)
                        centered += value['absolute_rms'] ** 2 * value['elements']
                        free += value['free_rms'] ** 2 * value['elements']
                        elements += value['elements']
                        deadline.check()
                    value = pooled_rms(centered, free, elements)
                    reporting.append_metric(out / 'metrics.jsonl', dict(stage='calibration', dataset_split='training',
                        block=block_index + 1, target_index=target_index, target_relative_rms=target,
                        evaluation=evaluations, beta_injected_B=beta, **value))
                    return value
                match = calibration_search(measure, target, previous_beta * target / previous_target)
                match.update(block=block_index + 1, target_index=target_index, target_relative_rms=target)
                calibration.append(match)
                save()
                if match['status'] == 'missing':
                    continue
                beta = match['beta_injected_B']
                previous_beta, previous_target = beta, match['achieved']['relative_rms']
                for precision in PRECISIONS:
                    model = models[precision]
                    block = model.analog_blocks[block_index]
                    estimates, displacement_sum = [], {}
                    for batch, saved in enumerate(captures[precision]):
                        progress('replay', precision=precision, block=block_index + 1, target_index=target_index, batch=batch)
                        estimate, measured = eqprop_gradients(block, saved[block_index], beta,
                            model_config['iterations'][block_index], device, measure_layer_displacement=True)
                        deadline.check()
                        estimates.append(estimate)
                        accumulate_displacement(displacement_sum, measured)
                        for layer_index, (name, tensor, layer) in enumerate(zip(names[block_index], estimate, block.free_layers())):
                            row = dict(precision=precision, block=block_index + 1, layer=layer_index + 1, name=name,
                                target_index=target_index, target_relative_rms=target, beta_injected_B=beta,
                                batch=batch, examples=sizes[batch],
                                common_full64_local_bptt=agreement(tensor, references['float64'][batch][block_index][layer_index]),
                                own_precision_local_bptt=agreement(tensor, references[precision][batch][block_index][layer_index]),
                                output_displacement=output_displacement(measured), **state_measurement_fields(measured, layer.name))
                            comparison['batches'].append(row)
                            reporting.append_metric(out / 'metrics.jsonl', dict(stage='batch_comparison', dataset_split='training', **row))
                        save()
                    for layer_index, (name, layer) in enumerate(zip(names[block_index], block.free_layers())):
                        tensor = weighted_mean([value[layer_index] for value in estimates], sizes)
                        common = weighted_mean([value[block_index][layer_index] for value in references['float64']], sizes)
                        own = weighted_mean([value[block_index][layer_index] for value in references[precision]], sizes)
                        row = dict(precision=precision, block=block_index + 1, layer=layer_index + 1, name=name,
                            target_index=target_index, target_relative_rms=target, beta_injected_B=beta, examples=sum(sizes),
                            common_full64_local_bptt=agreement(tensor, common), own_precision_local_bptt=agreement(tensor, own),
                            output_displacement=output_displacement(displacement_sum),
                            **state_measurement_fields(displacement_sum, layer.name))
                        comparison['layers'].append(row)
                        reporting.append_metric(out / 'metrics.jsonl', dict(stage='pooled_comparison', dataset_split='training', **row))
                    del estimates
                    save()
        progress('verify_immutable')
        for precision, model in models.items():
            check_snapshot(snapshots[precision], model.snapshot())
            verify_dtype(model, getattr(torch, precision))
        if reporting.sha256_file(args.checkpoint) != checkpoint_hash or reporting.sha256_file(args.config) != config_hash:
            raise ValueError('Checkpoint or config bytes changed')
        if source_hashes() != sources:
            raise ValueError('Source bytes changed')
        identity.update(parameters_and_bn_unchanged=True, checkpoint_bytes_unchanged=True,
                        config_bytes_unchanged=True, source_bytes_unchanged=True, precision_checks_passed=True)
        save()
        coverage = validate_evidence(config, calibration, comparison, identity)
        deadline.check()
        reporting.complete_run(out, terminal_metrics=dict(coverage=coverage, elapsed_seconds=deadline.elapsed,
            peak_memory_reserved_bytes=torch.cuda.max_memory_reserved(device) if device.type == 'cuda' else 0),
            completion=dict(criteria_met=True, all_targets_attempted=True, official_test_read=False,
                            scientific_coverage_complete=coverage['scientific_coverage_complete'],
                            protocol_deviations=[]))
    except BaseException as error:
        save()
        reporting.fail_run(out, error=error)
        raise
    return validate_run(out)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    for name in ('config', 'checkpoint', 'dataset-root'):
        parser.add_argument('--' + name, type=Path)
    parser.add_argument('--target', default='local')
    parser.add_argument('--budget-seconds', type=float, default=1740)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--validate', action='store_true')
    args = parser.parse_args()
    if not args.validate and any(getattr(args, name) is None for name in ('config', 'checkpoint', 'dataset_root')):
        parser.error('--config, --checkpoint and --dataset-root are required for execution')
    print(json.dumps(validate_run(args.run_dir) if args.validate else run(args), indent=2))


if __name__ == '__main__':
    main()
