"""Matched CIFAR state-precision and gradient-arithmetic controls; no updates."""
from __future__ import annotations
import argparse
import copy
import json
import math
import time
from pathlib import Path

import torch
from torch.nn import functional as F

from experiments import reporting
from experiments.calibrate_cifar_block_beta import FrozenOutputCost, capture_batches, nudged_minimizer
from experiments.compare_cifar_block_beta_gradients import (
    agreement, check_environment, local_bptt_gradients, restore_block, stable_centered_conv_gradient,
)
from experiments.train_cifar_l8_analog import Cohort, deterministic_setup, loader, make_split, read_training_data
from labs.cifar_l8_analog import CifarL8Analog, batchnorm_minibatch_stats_only
from model.resistive.interaction import ConvResistive
from training.sgd import AugmentedFunction, EquilibriumProp


def convert_model(model, dtype):
    # Blocks implement _apply for their unregistered analog variables; the dense
    # readouts do not, so explicitly convert every head state and conductance.
    model.to(dtype=dtype)
    for variable in model.head.analog_variables():
        requires_grad = variable.state.requires_grad
        variable.state = variable.state.detach().to(dtype=dtype).requires_grad_(requires_grad)
    verify_dtype(model, dtype)


def verify_dtype(model, dtype):
    tensors = list(model.parameters()) + list(model.buffers())
    for block in model.analog_blocks:
        tensors.extend(v.state for v in (block.boundary_layer, *block.free_layers(), *block.weights, *block.biases))
    tensors.extend(v.state for v in model.head.analog_variables())
    assert all(t.dtype == dtype for t in tensors if t.is_floating_point()), 'Mixed runtime precision'


def check_snapshot(expected, observed):
    assert expected.keys() == observed.keys()
    for group in expected:
        assert expected[group].keys() == observed[group].keys()
        for name, value in expected[group].items():
            target = observed[group][name]
            assert torch.equal(value.to(target.dtype), target), (group, name, 'state changed')


def state_measurement(minus, plus, free):
    delta = (plus.double() - minus.double()) * .5
    initial = free.to(delta.device).double()
    assert torch.isfinite(delta).all() and torch.isfinite(initial).all()
    active = initial != 0
    base32 = initial.abs().float()
    spacing = (torch.nextafter(base32, torch.full_like(base32, float('inf'))) - base32).double()
    return dict(elements=delta.numel(), centered_squares=float(delta.square().sum()),
        free_squares=float(initial.square().sum()), unchanged_elements=int((delta == 0).sum()),
        active_elements=int(active.sum()), active_unchanged_elements=int((active & (delta == 0)).sum()),
        active_sub_ulp_elements=int((active & (delta != 0) & (2 * delta.abs() < spacing)).sum()))


def finish_measurement(row):
    out = dict(row)
    out.update(absolute_rms=math.sqrt(out['centered_squares'] / out['elements']),
        free_rms=math.sqrt(out['free_squares'] / out['elements']),
        relative_rms=math.sqrt(out['centered_squares'] / out['free_squares']) if out['free_squares'] else None,
        unchanged_fraction=out['unchanged_elements'] / out['elements'],
        active_unchanged_fraction=out['active_unchanged_elements'] / out['active_elements'] if out['active_elements'] else None,
        active_sub_ulp_fraction=out['active_sub_ulp_elements'] / out['active_elements'] if out['active_elements'] else None)
    return out


def promoted_interaction(interaction):
    # The estimator derives scalar amplification/normalization tensor dtypes
    # from its layer objects, not from the supplied phase dictionaries. Cast
    # shallow layer views too, without touching either physical endpoint.
    result = copy.copy(interaction)
    for name in ('_layer_pre', '_layer_post'):
        layer = copy.copy(getattr(interaction, name))
        layer.state = layer.state.double()
        setattr(result, name, layer)
    return result


def precision_pair(block, saved, beta, iterations, device, mixed):
    """One phase pair; optional second arithmetic pass sees exactly its tensors."""
    restore_block(block, saved, device)
    dtype = block.weights[0].state.dtype
    assert all(x.state.dtype == dtype for x in (block.boundary_layer, *block.free_layers()))
    cost = FrozenOutputCost(block.free_layers()[-1]); cost.gradient = saved['gradient'].to(device)
    augmented = AugmentedFunction(block.energy, cost, nudging_mode='current', current_scale=1.)
    estimator = EquilibriumProp(block.weights, block.free_layers(), augmented, cost,
        nudged_minimizer(block, augmented, iterations), variant='centered', nudging=beta)
    ordered = []
    for weight in block.weights:
        interaction, = [fn for fn in block.energy._interactions
                         if isinstance(fn, ConvResistive) and fn._weight is weight]
        ordered.append(interaction)
    measurements, promoted = [], []

    def parameter_gradients(minus, plus):
        assert all(t.dtype == dtype for t in [*minus.values(), *plus.values()])
        measurements.extend(state_measurement(minus[layer.name], plus[layer.name], free)
                            for layer, free in zip(block.free_layers(), saved['states']))
        original = [stable_centered_conv_gradient(fn, minus, plus, beta) for fn in ordered]
        assert all(g.dtype == dtype for g in original)
        if mixed:
            first = {k: v.double() for k, v in minus.items()}
            second = {k: v.double() for k, v in plus.items()}
            first[block.boundary_layer.name] = block.boundary_layer.state.double()
            second[block.boundary_layer.name] = first[block.boundary_layer.name]
            promoted.extend(stable_centered_conv_gradient(promoted_interaction(fn), first, second, beta).cpu()
                            for fn in ordered)
            assert all(g.dtype == torch.float64 for g in promoted)
        return original

    estimator._standard_param_grads = parameter_gradients
    with torch.no_grad():
        original = [g.detach().cpu().double() for g in estimator.compute_gradient()]
    block.detach_state_()
    assert all(torch.isfinite(g).all() for g in original + promoted)
    return original, promoted, measurements


def run(args):
    cfg = json.loads(args.config.read_text())
    source = Path(__file__).resolve().parents[1]
    identity = json.loads(args.source_identity.read_text())
    for name, digest in identity['source_sha256'].items():
        assert reporting.sha256_file(source / name) == digest, name
    for name, digest in cfg['input_sha256'].items():
        assert reporting.sha256_file(args.input_root / name) == digest, name
    environment = check_environment(None if args.smoke else cfg['expected_environment'])
    if not args.smoke:
        assert args.device == 'cuda' and torch.cuda.is_available()
    config = json.loads((args.input_root / 'inputs/model_config.json').read_text())
    deterministic_setup(config['seed'])
    epoch = cfg['checkpoint_epoch']
    out = args.output_dir
    reporting.start_run(out, dict(study_id=cfg['study_id'], run_id=out.name, arm_id='precision_controls',
        evidence_class='cifar_finite_iteration_precision_replay', smoke=args.smoke,
        seed=config['seed'], checkpoint_epoch=epoch, resolved_config=cfg, model_config=config,
        input_root=str(args.input_root), source_identity=identity, environment=environment,
        runtime=reporting.runtime_context(), dataset='CIFAR10 original training-probe cohort'))
    start = time.monotonic()
    try:
        checkpoint = torch.load(args.input_root / cfg['checkpoint_file'], map_location='cpu', weights_only=False)
        assert checkpoint['epoch'] == epoch
        models, snapshots = {}, {}
        for precision, dtype in [('float32', torch.float32), ('float64', torch.float64)]:
            model = CifarL8Analog(config, args.device)
            model.restore(checkpoint['model'])
            convert_model(model, dtype)
            snapshots[precision] = model.snapshot()
            check_snapshot(checkpoint['model'], snapshots[precision])
            model.train(); models[precision] = model
        del checkpoint
        cohort = json.loads((args.input_root / 'inputs/cohort.json').read_text())
        indices = cohort['indices'][:2] if args.smoke else cohort['indices']
        assert len(indices) == len(set(indices)) == (2 if args.smoke else 256)
        images, labels = read_training_data(args.dataset_root)
        assert set(indices) <= set(make_split(labels, config['split']['seed'])['train'])
        data = Cohort(images, labels, indices, config, augment=False)
        multipliers = [1e-4] if args.smoke else cfg['beta_multipliers']
        names = [name for name, _ in models['float32'].named_conductances() if name.startswith('blocks.')]
        assert len(names) == 8
        pooled, reference_pools, state_pools = {}, {}, {}
        batch_rows, reference_batches, forcing, baseline_checks = [], [], [], []
        baseline = {}
        if not args.smoke:
            old = json.loads((args.input_root / f'cells/e{epoch}/artifacts/comparison.json').read_text())
            baseline = {(r['batch'], r['name'], r['beta_multiplier']): r for r in old['batches']}
        count = 0
        for batch, (x, y) in enumerate(loader(data, 2 if args.smoke else cfg['batch_size'])):
            references_this_batch = {}
            for precision, dtype in [('float32', torch.float32), ('float64', torch.float64)]:
                model = models[precision]
                typed_x = x.to(dtype=dtype)
                saved = capture_batches(model, [(typed_x, y)], config['iterations'], args.device)[0]
                with batchnorm_minibatch_stats_only(model):
                    _, tracked = model.bptt(typed_x.to(args.device), config['iterations'])
                    native = torch.autograd.grad(F.cross_entropy(tracked, y.to(args.device)),
                        [p for name, p in model.named_conductances() if name.startswith('blocks.')])
                assert all(g.dtype == dtype for g in native)
                native = [g.detach().cpu().double() for g in native]
                model.detach_state_(); del tracked
                verify_dtype(model, dtype)
                offset = 0
                for block_index, block in enumerate(model.analog_blocks):
                    local = local_bptt_gradients(block, saved[block_index], config['iterations'][block_index], args.device)
                    force = saved[block_index]['gradient'].double()
                    forcing.append(dict(batch=batch, precision=precision, block=block_index+1,
                        elements=force.numel(), force_squares=float(force.square().sum())))
                    for j, reference in enumerate(local):
                        name = names[offset+j]
                        references_this_batch[precision, name] = {'native': native[offset+j], 'local': reference}
                        for kind, tensor in references_this_batch[precision, name].items():
                            key = (precision, name, kind)
                            if key not in reference_pools: reference_pools[key] = tensor * len(y)
                            else: reference_pools[key].add_(tensor, alpha=len(y))
                    for case, multiplier in enumerate(multipliers):
                        if time.monotonic() - start > (300 if args.smoke else cfg['budget_seconds']):
                            raise TimeoutError('Precision replay budget exhausted')
                        beta = cfg['base_betas'][block_index] * multiplier
                        original, promoted, measurements = precision_pair(block, saved[block_index], beta,
                            config['iterations'][block_index], args.device, mixed=precision == 'float32')
                        arms = [(precision, original)]
                        if precision == 'float32': arms.append(('float32_states_float64_arithmetic', promoted))
                        for j, measured in enumerate(measurements):
                            name = names[offset+j]
                            state_key = (precision, case, name)
                            if state_key not in state_pools: state_pools[state_key] = dict(measured)
                            else:
                                for key, value in measured.items(): state_pools[state_key][key] += value
                            for arm, estimates in arms:
                                estimate = estimates[j]
                                key = (arm, case, name)
                                if key not in pooled: pooled[key] = estimate * len(y)
                                else: pooled[key].add_(estimate, alpha=len(y))
                                row = dict(epoch=epoch, batch=batch, examples=len(y), arm=arm,
                                    block=block_index+1, layer=j+1, name=name, beta_multiplier=multiplier,
                                    beta_injected_B=beta, case_index=case,
                                    state_displacement=finish_measurement(measured),
                                    native_bptt=agreement(estimate, native[offset+j]),
                                    local_bptt=agreement(estimate, local[j]))
                                batch_rows.append(row)
                                reporting.append_metric(out / 'metrics.jsonl', row)
                                prior = baseline.get((batch, name, multiplier)) if arm == 'float32' else None
                                if prior:
                                    ds = row['state_displacement']; old_ds = prior['state_displacement']
                                    assert ds['elements'] == old_ds['elements']
                                    for field in ['free_rms', 'absolute_rms', 'relative_rms']:
                                        assert math.isclose(ds[field], old_ds[field], rel_tol=2e-5, abs_tol=1e-15), (name, field)
                                    for field in ['cosine', 'norm_ratio']:
                                        actual, previous = row['native_bptt'][field], prior['native_bptt'][field]
                                        if actual is None or previous is None: assert actual == previous
                                        else: assert math.isclose(actual, previous, rel_tol=2e-5, abs_tol=2e-6), (name, field, actual, previous)
                                    baseline_checks.append((batch, name, multiplier))
                        progress = dict(stage='precision_sweep', epoch=epoch, batch=batch, precision=precision,
                            block=block_index+1, case=case+1, cases=len(multipliers), elapsed_seconds=time.monotonic()-start)
                        reporting.update_status_progress(out, progress)
                    offset += len(block.weights)
                    print(json.dumps(progress), flush=True)
                del saved, native, local, original, promoted
            for name in names:
                reference_batches.append(dict(batch=batch, name=name, **{kind: agreement(
                    references_this_batch['float64', name][kind], references_this_batch['float32', name][kind])
                    for kind in ['native', 'local']}))
            count += len(y)
            del references_this_batch
        rows = []
        for (arm, case, name), value in pooled.items():
            precision = 'float64' if arm == 'float64' else 'float32'
            _, block_index, _, layer_index = name.split('.')
            references = {kind: reference_pools[precision, name, kind] / count for kind in ['native', 'local']}
            rows.append(dict(epoch=epoch, arm=arm, block=int(block_index)+1, layer=int(layer_index)+1,
                name=name, case_index=case, beta_multiplier=multipliers[case],
                beta_injected_B=cfg['base_betas'][int(block_index)]*multipliers[case], examples=count,
                state_displacement=finish_measurement(state_pools[precision, case, name]),
                native_bptt=agreement(value/count, references['native']),
                local_bptt=agreement(value/count, references['local']),
                local_vs_native_bptt=agreement(references['local'], references['native'])))
        reference_rows = [dict(name=name, **{kind: agreement(reference_pools['float64', name, kind]/count,
            reference_pools['float32', name, kind]/count) for kind in ['native', 'local']}) for name in names]
        for precision, model in models.items():
            check_snapshot(snapshots[precision], model.snapshot())
            verify_dtype(model, torch.float64 if precision == 'float64' else torch.float32)
        for name, digest in cfg['input_sha256'].items():
            assert reporting.sha256_file(args.input_root / name) == digest, name
        expected = 3*len(multipliers)*8
        assert len(rows) == expected and len(batch_rows) == expected*(1 if args.smoke else 8)
        assert len(baseline_checks) == (0 if args.smoke else 8*17*8)
        reporting.atomic_write_json(out / 'artifacts/comparison.json', dict(layers=rows, batches=batch_rows,
            references=reference_rows, reference_batches=reference_batches, forcing=forcing, cohort_indices=indices))
        reporting.complete_run(out, terminal_metrics=dict(epoch=epoch, examples=count, rows=len(rows),
            batch_rows=len(batch_rows), baseline_rows_checked=len(baseline_checks),
            checkpoint_bytes_unchanged=True, parameters_and_bn_unchanged=True, precision_checks_passed=True,
            elapsed_seconds=time.monotonic()-start,
            peak_memory_reserved_bytes=torch.cuda.max_memory_reserved() if args.device == 'cuda' else 0),
            completion=dict(criteria_met=True, official_test_read=False, all_three_arms=True))
    except BaseException as exc:
        reporting.fail_run(out, error=exc)
        raise
    assert not reporting.validate_run(out)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('config', 'input-root', 'output-dir', 'dataset-root', 'source-identity'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--smoke', action='store_true')
    run(parser.parse_args())
