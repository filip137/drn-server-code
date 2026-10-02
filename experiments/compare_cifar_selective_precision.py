"""Read-only CIFAR replay with calibrated, block-selective nudged precision.

The fixed reference is full-float64 block-local BPTT at the original K. It is
neither native full-network BPTT nor a qualified equilibrium gradient.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack, contextmanager
import json
import math
from pathlib import Path
import sys
import time

import torch

from experiments import reporting
from experiments.calibrate_cifar_block_beta import (
    FrozenOutputCost, capture_batches, measure_block, nudged_minimizer, pooled_rms, search_beta,
)
from experiments.compare_cifar_block_beta_gradients import (
    accumulate_displacement, agreement, eqprop_gradients, local_bptt_gradients,
    state_measurement_fields,
)
from experiments.compare_cifar_split_precision import add_tensor, promote_saved
from experiments.replay_cifar_precision_response import check_snapshot, convert_model
from experiments.train_cifar_l8_analog import (
    Cohort, deterministic_setup, loader, make_split, read_training_data,
)
from labs.cifar_l8_analog import CifarL8Analog
from training.sgd import AugmentedFunction

MODES = ('candidate', 'all64_nudged_control', 'full64_eqprop_native_k')
REFERENCE = 'full64_local_bptt'
CALIBRATION_TOLERANCE = .03
CALIBRATION_EVALUATIONS = 12


def audit_block(block, dtype):
    variables = [block.boundary_layer, *block.free_layers(), *block.weights, *block.biases]
    if not all(v.state.dtype == dtype for v in variables):
        raise ValueError(f'Unexpected block precision: expected {dtype}')


@contextmanager
def block_scope(block, dtype, *, cache=True):
    """Keep caller state objects intact; coefficient caches are solve-scoped."""
    audit_block(block, dtype)
    states = [(v, v.state) for v in [block.boundary_layer, *block.free_layers()]]
    try:
        with ExitStack() as contexts:
            if cache:
                for interaction in block.interactions:
                    contexts.enter_context(interaction.cache_a_col2im())
            yield
            audit_block(block, dtype)
    finally:
        for variable, value in states:
            variable.state = value


def selected_block(models, index, float64_blocks):
    promoted = index + 1 in float64_blocks
    return models['donor64' if promoted else 'full32'].analog_blocks[index], promoted


def candidate_gradients(models, saved32, index, beta, iterations, float64_blocks, device):
    block, promoted = selected_block(models, index, float64_blocks)
    dtype = torch.float64 if promoted else torch.float32
    saved = promote_saved(saved32) if promoted else saved32
    with block_scope(block, dtype):
        return eqprop_gradients(block, saved, beta, iterations, device,
                                measure_layer_displacement=True)


def output_displacement(displacement):
    return dict(**pooled_rms(displacement['centered_squares'], displacement['free_squares'],
                            displacement['elements']),
                unchanged_fraction=displacement['unchanged_elements']/displacement['elements'])


def numerical_pass(value):
    return (value['cosine'] is not None and value['cosine'] > .95
            and value['norm_ratio'] is not None and .9 <= value['norm_ratio'] <= 1.1)


def calibrate(models, records32, args, out, check_budget):
    """Use actual candidate dtype/K, endpoint measurements only, all 256 images."""
    matches = []
    for i, target in enumerate(args.rms_targets):
        block, promoted = selected_block(models, i, args.float64_blocks)
        dtype = torch.float64 if promoted else torch.float32
        cost = FrozenOutputCost(block.free_layers()[-1])
        augmented = AugmentedFunction(block.energy, cost, nudging_mode='current', current_scale=1.)
        minimizer = nudged_minimizer(block, augmented, args.nudged_iterations[i])
        record = dict(block=i+1, dtype=str(dtype), nudged_iterations=args.nudged_iterations[i],
            target_relative_rms=target, initial_beta=args.betas[i], status='running', history=[])
        matches.append(record)

        def measure(beta):
            check_budget()
            # Generate just this block's promoted capture; keep the full cohort on CPU.
            def records():
                for batch in records32:
                    check_budget()
                    yield {i: promote_saved(batch[i]) if promoted else batch[i]}
            with block_scope(block, dtype):
                value = measure_block(block, minimizer, augmented, cost, records(), i, beta, args.device)
            row = dict(beta_injected_B=beta, relative_target_error=abs(value['relative_rms']/target-1), **value)
            record['history'].append(row)
            reporting.atomic_write_json(out/'artifacts/calibration.json', matches)
            progress = dict(stage='beta_calibration', block=i+1, evaluation=len(record['history']),
                            target_relative_rms=target, dtype=str(dtype), **row)
            reporting.append_metric(out/'metrics.jsonl', dict(dataset_split='training', **progress))
            reporting.update_status_progress(out, progress)
            print(json.dumps(progress), flush=True)
            return value
        try:
            beta, history = search_beta(measure, target, metric='relative_rms',
                tolerance=CALIBRATION_TOLERANCE, max_evaluations=CALIBRATION_EVALUATIONS,
                initial_beta=args.betas[i])
            record.update(status='matched', beta_injected_B=beta, achieved=history[-1])
        except BaseException as exc:
            record.update(status='failed', error=str(exc))
            raise
        finally:
            reporting.atomic_write_json(out/'artifacts/calibration.json', matches)
    return matches


def run(args):
    config = json.loads(args.config.read_text())
    assert config['scheme'] == 'ours' and config['iterations'] == [6, 6, 4]
    assert config['batch_size'] == 32
    indices = config['conv_eqprop']['cohort_indices']
    assert len(indices) == len(set(indices)) == 256
    assert len(args.betas) == len(args.rms_targets) == len(args.nudged_iterations) == 3
    assert all(math.isfinite(v) and v > 0 for v in [*args.betas, *args.rms_targets, args.budget_seconds])
    assert all(isinstance(k, int) and k > 0 for k in args.nudged_iterations)
    assert set(args.float64_blocks) <= {1, 2, 3}
    deterministic_setup(config['seed'])
    device, out = torch.device(args.device), args.run_dir
    source = Path(__file__).resolve().parents[1]
    files = [p for folder in ['experiments', 'labs', 'model', 'training'] for p in (source/folder).rglob('*.py')]
    inputs = [args.config, args.checkpoint, *[args.dataset_root/f'data_batch_{i}' for i in range(1, 6)]]
    hashes = {str(p.resolve()): reporting.sha256_file(p) for p in inputs}
    source_hashes = {str(p.relative_to(source)): reporting.sha256_file(p) for p in files}
    reporting.start_run(out, dict(study_id=config.get('study_id', 'cifar-selective-precision-20261001-v1'), run_id=out.name,
        arm_id='selective_precision_replay', seed=config['seed'], smoke=False, mode='diagnostic',
        evidence_class='cifar_finite_iteration_selective_precision_replay', resolved_config=config,
        command=sys.argv, input_sha256=hashes, source_sha256=source_hashes,
        source_identity_kind='per-file SHA256 of executed snapshot',
        runtime=reporting.runtime_context(target=args.target),
        environment=dict(torch=str(torch.__version__), cuda=torch.version.cuda,
                         gpu=torch.cuda.get_device_name(device) if device.type == 'cuda' else None),
        initial_beta_guesses=args.betas, rms_targets=args.rms_targets,
        calibration=dict(relative_tolerance=CALIBRATION_TOLERANCE,
                         max_evaluations_per_block=CALIBRATION_EVALUATIONS, examples=256),
        budget_seconds=args.budget_seconds, free_iterations=config['iterations'],
        nudged_iterations=args.nudged_iterations, reference_iterations=config['iterations'],
        modes=list(MODES), float64_blocks=args.float64_blocks,
        dtype_contract=dict(candidate='float32 free; selected blocks float64 phases/formula',
            all64_nudged_control='same float32 free/forcing and candidate beta/K; float64 phases/formula',
            full64_eqprop_native_k='float64 free/forcing/phases/formula; original K; candidate beta'),
        coefficient_cache='solve-scoped in all EP modes; backend activates only where valid',
        dataset=dict(name='CIFAR10', source_split='training', cohort_indices=indices,
                     augmentation=False, official_test_read=False),
        updates=0, read_noise=0., batchnorm='minibatch statistics with frozen running buffers',
        reference='full64 block-local finite-K BPTT, original K=[6,6,4]; not native full-network BPTT',
        numerical_pass=dict(cosine_strictly_above=.95, norm_ratio_inclusive=[.9, 1.1],
                            scope='every layer and minibatch against fixed full64 local BPTT')))
    start = time.monotonic()
    def check_budget():
        if time.monotonic()-start > args.budget_seconds:
            raise TimeoutError('Selective precision replay budget exhausted')
    def progress(**values):
        values.update(elapsed_seconds=time.monotonic()-start)
        reporting.update_status_progress(out, values)
        print(json.dumps(values), flush=True)
    try:
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
        models, snapshots = {}, {}
        for name, dtype in [('full64', torch.float64), ('full32', torch.float32), ('donor64', torch.float64)]:
            model = CifarL8Analog(config, device)
            convert_model(model, dtype)  # Preserve checkpoint doubles in the fixed reference.
            initial = snapshots['full32'] if name == 'donor64' else checkpoint['model']
            model.restore(initial)
            snapshots[name] = model.snapshot()
            check_snapshot(initial, snapshots[name])
            models[name] = model
        epoch = checkpoint['epoch']
        del checkpoint
        images, labels = read_training_data(args.dataset_root)
        assert set(indices) <= set(make_split(labels, config['split']['seed'])['train'])
        data = Cohort(images, labels, indices, config, augment=False)
        names = [n for n, _ in models['full64'].named_conductances() if n.startswith('blocks.')]
        assert len(names) == 8
        compute_start = time.monotonic()
        captures32 = []
        for batch, (x, y) in enumerate(loader(data, 32)):
            check_budget()
            captures32.extend(capture_batches(models['full32'], [(x.float(), y)], config['iterations'], device))
            progress(stage='float32_free_capture', batch=batch+1, batches=8)
        matches = calibrate(models, captures32, args, out, check_budget)
        betas = [row['beta_injected_B'] for row in matches]
        pooled, state_pools, batches = {}, {}, []
        count = 0
        for batch, (x, y) in enumerate(loader(data, 32)):
            check_budget()
            saved64 = capture_batches(models['full64'], [(x.double(), y)], config['iterations'], device)[0]
            offset = 0
            for i, reference_k in enumerate(config['iterations']):
                check_budget()
                block64 = models['full64'].analog_blocks[i]
                with block_scope(block64, torch.float64, cache=False):
                    reference = local_bptt_gradients(block64, saved64[i], reference_k, device)
                estimates, displacements = {}, {}
                estimates['candidate'], displacements['candidate'] = candidate_gradients(
                    models, captures32[batch][i], i, betas[i], args.nudged_iterations[i], args.float64_blocks, device)
                check_budget()
                donor = models['donor64'].analog_blocks[i]
                if i+1 in args.float64_blocks:
                    # Same donor, captures, beta, K and formula: this control is
                    # identically the candidate on an already-double block.
                    estimates[MODES[1]] = estimates['candidate']
                    displacements[MODES[1]] = displacements['candidate']
                else:
                    with block_scope(donor, torch.float64):
                        estimates[MODES[1]], displacements[MODES[1]] = eqprop_gradients(donor,
                            promote_saved(captures32[batch][i]), betas[i], args.nudged_iterations[i], device,
                            measure_layer_displacement=True)
                check_budget()
                with block_scope(block64, torch.float64):
                    estimates[MODES[2]], displacements[MODES[2]] = eqprop_gradients(block64,
                        saved64[i], betas[i], reference_k, device, measure_layer_displacement=True)
                for mode in MODES:
                    accumulate_displacement(state_pools.setdefault((mode, i+1), {}), displacements[mode])
                for j, name in enumerate(names[offset:offset+len(reference)]):
                    add_tensor(pooled, (REFERENCE, name), reference[j], len(y))
                    for mode in MODES:
                        estimate = estimates[mode][j]
                        add_tensor(pooled, (mode, name), estimate, len(y))
                        fixed = agreement(estimate, reference[j])
                        row = dict(batch=batch, examples=len(y), mode=mode, block=i+1, layer=j+1, name=name,
                            beta_injected_B=betas[i], target_relative_rms=args.rms_targets[i],
                            nudged_iterations=reference_k if mode == MODES[2] else args.nudged_iterations[i],
                            reference_iterations=reference_k,
                            nudged_dtype='torch.float32' if mode == 'candidate' and i+1 not in args.float64_blocks else 'torch.float64',
                            common_full64_local_bptt=fixed, numerical_pass=numerical_pass(fixed),
                            same_k_all64_nudged_control=agreement(estimate, estimates[MODES[1]][j]),
                            full64_eqprop_native_k=agreement(estimate, estimates[MODES[2]][j]),
                            output_displacement=output_displacement(displacements[mode]),
                            **state_measurement_fields(displacements[mode], block64.free_layers()[j].name))
                        batches.append(row)
                        reporting.append_metric(out/'metrics.jsonl', dict(stage='gradient_comparison', dataset_split='training', **row))
                offset += len(reference)
                progress(stage='selective_precision_replay', batch=batch+1, batches=8, block=i+1)
            count += len(y)
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        compute_elapsed = time.monotonic()-compute_start
        rows = []
        for mode in MODES:
            for name in names:
                _, block_index, _, layer_index = name.split('.')
                i, j = int(block_index), int(layer_index)
                fixed = agreement(pooled[mode, name]/count, pooled[REFERENCE, name]/count)
                rows.append(dict(mode=mode, name=name, block=i+1, layer=j+1, examples=count,
                    beta_injected_B=betas[i], target_relative_rms=args.rms_targets[i],
                    nudged_iterations=config['iterations'][i] if mode == MODES[2] else args.nudged_iterations[i],
                    reference_iterations=config['iterations'][i], common_full64_local_bptt=fixed,
                    numerical_pass=numerical_pass(fixed),
                    same_k_all64_nudged_control=agreement(pooled[mode, name]/count, pooled[MODES[1], name]/count),
                    full64_eqprop_native_k=agreement(pooled[mode, name]/count, pooled[MODES[2], name]/count),
                    output_displacement=output_displacement(state_pools[mode, i+1]),
                    **state_measurement_fields(state_pools[mode, i+1], models['full64'].analog_blocks[i].free_layers()[j].name)))
        assert count == 256 and len(batches) == 192 and len(rows) == 24
        calibration_errors = [abs(state_pools['candidate', i+1]['centered_squares'] /
            state_pools['candidate', i+1]['free_squares'])**.5 / target - 1
            for i, target in enumerate(args.rms_targets)]
        assert all(abs(error) <= CALIBRATION_TOLERANCE for error in calibration_errors)
        actual_dtypes = {}
        for name, model in models.items():
            check_snapshot(snapshots[name], model.snapshot())
            dtype = torch.float32 if name == 'full32' else torch.float64
            for block in model.analog_blocks:
                audit_block(block, dtype)
            actual_dtypes[name] = [[str(w.state.dtype) for w in b.weights] for b in model.analog_blocks]
        for path, digest in hashes.items(): assert reporting.sha256_file(Path(path)) == digest, path
        for path, digest in source_hashes.items(): assert reporting.sha256_file(source/path) == digest, path
        torch.save(dict(example_weighted_gradient_sums=pooled, examples=count), out/'artifacts/pooled_gradients.pt')
        passes = {mode: all(row['numerical_pass'] for row in batches if row['mode'] == mode) for mode in MODES}
        reporting.atomic_write_json(out/'artifacts/comparison.json', dict(epoch=epoch, layers=rows,
            batches=batches, calibration=matches, actual_betas=betas, cohort_indices=indices,
            all_layer_batch_numerical_pass=passes, reference=REFERENCE,
            reference_iterations=config['iterations'], candidate_nudged_iterations=args.nudged_iterations))
        check_budget()
        reporting.complete_run(out, terminal_metrics=dict(epoch=epoch, examples=count, rows=len(rows),
            batch_rows=len(batches), actual_betas=betas, calibration_relative_errors=calibration_errors,
            all_layer_batch_numerical_pass=passes, parameters_and_bn_unchanged=True,
            checkpoint_bytes_unchanged=True, source_bytes_unchanged=True,
            actual_block_parameter_dtypes=actual_dtypes, precision_checks_passed=True,
            elapsed_seconds=time.monotonic()-start, compute_elapsed_seconds=compute_elapsed,
            timing_scope='diagnostic free captures, calibration and gradient replay; not training epoch timing',
            peak_memory_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else 0,
            peak_memory_reserved_bytes=torch.cuda.max_memory_reserved(device) if device.type == 'cuda' else 0),
            completion=dict(criteria_met=True, official_test_read=False, all_three_modes=True, updates=0,
                            all_calibration_targets_met=True, compared_all_conv_weights=True))
    except BaseException as exc:
        reporting.fail_run(out, error=exc)
        raise
    validate(out)


def validate(out):
    errors = reporting.validate_run(out)
    if errors:
        raise RuntimeError(errors)
    result = json.loads((out/'result.json').read_text())
    assert result['completion']['criteria_met'] and result['completion']['all_calibration_targets_met']
    comparison = json.loads((out/'artifacts/comparison.json').read_text())
    assert len(comparison['batches']) == 192 and len(comparison['layers']) == 24
    assert comparison['reference_iterations'] == [6, 6, 4]
    assert all(row['status'] == 'matched' and len(row['history']) <= CALIBRATION_EVALUATIONS
               and row['achieved']['relative_target_error'] <= CALIBRATION_TOLERANCE
               for row in comparison['calibration'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['config', 'checkpoint', 'dataset-root', 'run-dir']:
        parser.add_argument('--'+name, type=Path, required=name == 'run-dir')
    parser.add_argument('--betas', type=float, nargs=3, help='Initial guesses; recalibrated on the fixed cohort')
    parser.add_argument('--nudged-iterations', type=int, nargs=3, default=[4, 4, 3])
    parser.add_argument('--float64-blocks', type=int, nargs='*', default=[3])
    parser.add_argument('--rms-targets', type=float, nargs=3, default=[.01, .005, .00001])
    parser.add_argument('--target', default='trex')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--budget-seconds', type=float, default=600.)
    parser.add_argument('--validate', action='store_true')
    args = parser.parse_args()
    if args.validate:
        validate(args.run_dir)
        print('VALIDATED', args.run_dir)
        return
    if any(getattr(args, name) is None for name in ['config', 'checkpoint', 'dataset_root', 'betas']):
        parser.error('--config --checkpoint --dataset-root --betas are required for replay')
    run(args)


if __name__ == '__main__':
    main()
