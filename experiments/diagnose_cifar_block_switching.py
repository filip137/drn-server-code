"""Float64 block3 finite-K replay: diode switching, response and causal mask control."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import math
from pathlib import Path
import sys
import time

import torch

from experiments import reporting
from experiments.calibrate_cifar_block_beta import (
    FrozenOutputCost, capture_batches, measure_block, nudged_minimizer, search_beta,
)
from experiments.compare_cifar_block_beta_gradients import (
    agreement, local_bptt_gradients, restore_block, stable_centered_conv_gradient,
)
from experiments.compare_cifar_selective_precision import block_scope, numerical_pass
from experiments.replay_cifar_precision_response import check_snapshot, convert_model, verify_dtype
from experiments.train_cifar_l8_analog import (
    Cohort, deterministic_setup, loader, make_split, read_training_data,
)
from labs.cifar_l8_analog import CifarL8Analog
from model.resistive.interaction import ConvResistive
from training.sgd import AugmentedFunction

TARGETS = [1e-6, 1e-5, 2e-5, 3e-5, 1e-4]
PRIOR_BETA = 1.6455980388298777
K = 4


def finite(tensor):
    if not torch.isfinite(tensor).all():
        raise FloatingPointError('Nonfinite state, candidate, or response')


def margin_summary(margin):
    """Signed distance into the allowed half-space, before activate clamps it."""
    value = margin.detach().cpu().double().flatten()
    finite(value)
    absolute = value.abs()
    return dict(elements=value.numel(), signed_min=float(value.min()), signed_max=float(value.max()),
        signed_mean=float(value.mean()), absolute_min=float(absolute.min()),
        absolute_mean=float(absolute.mean()),
        absolute_quantiles=dict(zip(['q01', 'q10', 'q50', 'q90', 'q99'],
            torch.quantile(absolute, torch.tensor([.01, .1, .5, .9, .99], dtype=torch.float64)).tolist())),
        exact_boundary_count=int((value == 0).sum()),
        near_boundary_counts={str(t): int((absolute <= t).sum()) for t in [1e-12, 1e-9, 1e-6, 1e-3]})


@contextmanager
def trace_activations(minimizer, *, zero_trace=None, force_zero_masks=False):
    """Hook actual activate calls; leave grouped pre_activate computation untouched.

    At exactly zero the unconstrained branch is retained (torch.clamp's derivative
    is one there). Only strictly forbidden candidates count as clamped. A forced
    zero mask pins clamped units to zero and passes free candidates unchanged,
    including candidates that cross the physical boundary under this intervention.
    """
    assert not minimizer._adaptive_equilibrium
    assert not minimizer._overrelaxation_reject_steps
    expected = [u._layer for _ in range(minimizer.num_iterations)
                for group in minimizer._equilibrium_sweep_groups() for u in group]
    layers = list(dict.fromkeys(expected))
    assert all(layer.non_linearity == 'perfect_diode' for layer in layers)
    if zero_trace is not None:
        assert len(zero_trace) == len(expected), 'Trace count mismatch'
    if force_zero_masks:
        assert zero_trace is not None
    trace, originals = [], []
    for layer in layers:
        original = layer.activate
        existed = 'activate' in layer.__dict__
        originals.append((layer, original, existed))

        def activate(layer=layer, original=original):
            position = len(trace)
            assert position < len(expected) and expected[position] is layer, 'Trace order mismatch'
            candidate = layer.state
            finite(candidate)
            assert candidate.dtype == torch.float64
            half = layer._shape[0] // 2
            margin = torch.cat((candidate[:, :half], -candidate[:, half:]), dim=1)
            clamped = margin < 0
            identity = dict(position=position, update=position // len(layers) + 1,
                            state_name=layer.name, shape=list(candidate.shape))
            row = dict(**identity, clamped=clamped.detach().cpu(), margins=margin_summary(margin))
            ref = None
            if zero_trace is not None:
                refrow = zero_trace[position]
                assert all(refrow[key] == value for key, value in identity.items()), 'Trace identity mismatch'
                ref = refrow['clamped'].to(candidate.device)
                changed = clamped != ref
                row.update(switch_count=int(changed.sum()),
                    clamp_to_free_count=int((ref & ~clamped).sum()),
                    free_to_clamp_count=int((~ref & clamped).sum()),
                    switch_fraction=float(changed.double().mean()))
                row['margins']['at_switches'] = margin_summary(margin[changed]) if changed.any() else None
            output = torch.where(ref, torch.zeros_like(candidate), candidate) if force_zero_masks else original()
            finite(output)
            trace.append(row)
            return output
        layer.activate = activate
    try:
        yield trace
        assert len(trace) == len(expected), 'Trace count mismatch'
    finally:
        for layer, original, existed in originals:
            if existed:
                layer.activate = original
            else:
                del layer.activate


def phase(block, saved, beta, device, *, zero_trace=None, fixed=False, instrument=True):
    with block_scope(block, torch.float64), torch.no_grad():
        restore_block(block, saved, device)
        cost = FrozenOutputCost(block.free_layers()[-1])
        cost.gradient = saved['gradient'].to(device)
        augmented = AugmentedFunction(block.energy, cost, nudging_mode='current', current_scale=1.)
        augmented.prepare_nudging()
        augmented.nudging = beta
        minimizer = nudged_minimizer(block, augmented, K)
        force = augmented._nudging._force.clone()
        if instrument:
            with trace_activations(minimizer, zero_trace=zero_trace, force_zero_masks=fixed) as trace:
                minimizer.compute_equilibrium()
        else:
            assert not fixed
            minimizer.compute_equilibrium()
            trace = []
        assert torch.equal(force, augmented._nudging._force)
        endpoints = {layer.name: layer.state.detach().cpu().clone() for layer in block.free_layers()}
        for value in endpoints.values():
            finite(value)
    return endpoints, trace


def centered(block, saved, beta, device, zero_trace, *, fixed=False):
    minus, mt = phase(block, saved, -beta, device, zero_trace=zero_trace, fixed=fixed)
    plus, pt = phase(block, saved, beta, device, zero_trace=zero_trace, fixed=fixed)
    with block_scope(block, torch.float64), torch.no_grad():
        restore_block(block, saved, device)
        minus_device = {k: v.to(device) for k, v in minus.items()}
        plus_device = {k: v.to(device) for k, v in plus.items()}
        gradients = []
        for weight in block.weights:
            interaction, = [fn for fn in block.energy._interactions
                            if isinstance(fn, ConvResistive) and fn._weight is weight]
            value = stable_centered_conv_gradient(interaction, minus_device, plus_device, beta)
            finite(value)
            gradients.append(value.cpu())
    return gradients, minus, plus, mt, pt


def state_sums(minus, plus, zero, free, beta, reference_response):
    d = (plus - minus) * .5
    response = d / beta
    symmetry = (plus + minus) * .5 - zero
    drift = zero - free
    for value in [d, response, symmetry, drift, reference_response]:
        finite(value)
    return dict(elements=free.numel(), centered_squares=float(d.square().sum()),
        free_squares=float(free.square().sum()), drift_squares=float(drift.square().sum()),
        asymmetry_squares=float(symmetry.square().sum()),
        linearity_error_squares=float((response-reference_response).square().sum()),
        reference_response_squares=float(reference_response.square().sum()))


def state_metrics(sums):
    def ratio(numerator, denominator):
        return math.sqrt(sums[numerator]/sums[denominator]) if sums[denominator] else None
    return dict(**sums, centered_rms=math.sqrt(sums['centered_squares']/sums['elements']),
        relative_rms=ratio('centered_squares', 'free_squares'),
        zero_nudge_relative_drift=ratio('drift_squares', 'free_squares'),
        phase_asymmetry=ratio('asymmetry_squares', 'centered_squares'),
        relative_response_linearity=ratio('linearity_error_squares', 'reference_response_squares'))


def trace_rows(trace, phase_name, batch):
    return [dict(batch=batch, phase=phase_name, **{k: v for k, v in row.items() if k != 'clamped'},
                 clamped_count=int(row['clamped'].sum())) for row in trace]


def run(args):
    config = json.loads(args.config.read_text())
    assert config['scheme'] == 'ours' and config['iterations'] == [6, 6, 4]
    assert config['batch_size'] == 32 and config['solver']['non_linearity'] == 'perfect_diode'
    indices = config['conv_eqprop']['cohort_indices']
    assert len(indices) == len(set(indices)) == 256
    assert math.isfinite(args.budget_seconds) and args.budget_seconds > 0
    deterministic_setup(config['seed'])
    device, out = torch.device(args.device), args.run_dir
    source = Path(__file__).resolve().parents[1]
    files = [p for folder in ['experiments', 'labs', 'model', 'training'] for p in (source/folder).rglob('*.py')]
    inputs = [args.config, args.checkpoint, *[args.dataset_root/f'data_batch_{i}' for i in range(1, 6)]]
    hashes = {str(p.resolve()): reporting.sha256_file(p) for p in inputs}
    source_hashes = {str(p.relative_to(source)): reporting.sha256_file(p) for p in files}
    reporting.start_run(out, dict(study_id='cifar-block3-switching-20261001-v1', run_id=out.name,
        arm_id='float64_block3_switching', seed=config['seed'], smoke=False, mode='diagnostic',
        evidence_class='cifar_finite_iteration_switching_diagnostic', resolved_config=config,
        command=sys.argv, input_sha256=hashes, source_sha256=source_hashes,
        runtime=reporting.runtime_context(target=args.target),
        environment=dict(torch=str(torch.__version__), cuda=torch.version.cuda,
                         gpu=torch.cuda.get_device_name(device) if device.type == 'cuda' else None),
        budget_seconds=args.budget_seconds, precision='float64 including free capture, forcing and formulas',
        free_iterations=[6, 6, 4], block=3, local_bptt_iterations=K, nudged_iterations=K,
        rms_targets=TARGETS, calibration=dict(relative_tolerance=.03, max_evaluations_per_target=12,
            initial_beta_at_1e_minus_5=PRIOR_BETA, initial_scaling='target / 1e-5'),
        dataset=dict(name='CIFAR10', source_split='training', cohort_indices=indices,
            augmentation=False, official_test_read=False, batches=8, batch_size=32),
        batchnorm='training minibatch statistics, running buffers frozen', updates=0, read_noise=0., tf32=False,
        clamp_contract='zero boundary; first half channels >=0; remaining channels <=0; exact zero is free',
        intervention='first ascending target failing any batch/layer: replay zero-nudge per-update decisions',
        intervention_warning='fixed-mask linearity alone does not establish switching as cause',
        reference='fixed block-local finite-K BPTT; not native full-network or equilibrium gradient'))
    start = time.monotonic()
    def check_budget():
        if time.monotonic()-start > args.budget_seconds:
            raise TimeoutError('Switching diagnostic budget exhausted')
    def progress(**values):
        check_budget()
        values.update(elapsed_seconds=time.monotonic()-start)
        reporting.update_status_progress(out, values)
        reporting.append_metric(out/'metrics.jsonl', dict(dataset_split='training', **values))
        print(json.dumps(values), flush=True)
    try:
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
        assert checkpoint['epoch'] == 1
        model = CifarL8Analog(config, device)
        convert_model(model, torch.float64)
        model.restore(checkpoint['model'])
        before = model.snapshot()
        check_snapshot(checkpoint['model'], before)
        del checkpoint
        images, labels = read_training_data(args.dataset_root)
        assert set(indices) <= set(make_split(labels, config['split']['seed'])['train'])
        data = Cohort(images, labels, indices, config, augment=False)
        block = model.analog_blocks[2]
        assert len(block.weights) == len(block.free_layers()) == 2
        captures, references, zeros, traces = [], [], [], []
        for batch, (x, y) in enumerate(loader(data, 32)):
            check_budget()
            saved = capture_batches(model, [(x.double(), y)], config['iterations'], device)[0][2]
            assert all(t.dtype == torch.float64 for t in [saved['input'], saved['gradient'], *saved['states']])
            captures.append(saved)
            with block_scope(block, torch.float64, cache=False):
                references.append(local_bptt_gradients(block, saved, K, device))
            zero, trace = phase(block, saved, 0., device)
            zeros.append(zero)
            traces.append(trace)
            progress(stage='free_reference_zero_capture', batch=batch+1, batches=8)
        assert len(captures) == 8
        torch.save(dict(zero_endpoints=zeros, zero_traces=traces, captures=captures,
                        fixed_bptt_gradients=references), out/'artifacts/reference.pt')
        matches, native, interventions, low_response, fixed_low_response = [], [], [], {}, {}
        pooled_gradients = {}
        comparison = dict(epoch=1, block=3, free_iterations=[6, 6, 4], local_bptt_iterations=K,
            nudged_iterations=K, cohort_indices=indices, rms_targets=TARGETS, calibration=matches,
            native=native, interventions=interventions, first_failing_target=None,
            zero_trace_rows=[row for b, trace in enumerate(traces) for row in trace_rows(trace, 'zero', b)])
        names = [name for name, _ in model.named_conductances() if name.startswith('blocks.2.')]
        assert len(names) == 2

        def evaluate(target, beta, mode, low, initialize_low=False):
            batches, updates, state_pool = [], [], {}
            sums = [torch.zeros_like(g) for g in references[0]]
            for b, saved in enumerate(captures):
                check_budget()
                gs, minus, plus, mt, pt = centered(block, saved, beta, device, traces[b], fixed=mode == 'fixed_mask')
                updates.extend(trace_rows(mt, 'minus', b) + trace_rows(pt, 'plus', b))
                for j, (layer, name) in enumerate(zip(block.free_layers(), names)):
                    key = (b, layer.name)
                    if initialize_low:
                        low[key] = (plus[layer.name]-minus[layer.name]) * (.5/beta)
                    raw = state_sums(minus[layer.name], plus[layer.name], zeros[b][layer.name],
                                     saved['states'][j], beta, low[key])
                    pooled = state_pool.setdefault(j, {k: 0 for k in raw})
                    for k, v in raw.items():
                        pooled[k] += v
                    sums[j] += gs[j]*32
                    gradient = agreement(gs[j], references[b][j])
                    batches.append(dict(batch=b, layer=j+1, name=name, gradient=gradient,
                        numerical_pass=numerical_pass(gradient), state=state_metrics(raw)))
                progress(stage='gradient_and_switch_replay', mode=mode, target_relative_rms=target, batch=b+1)
            pooled = []
            for j, name in enumerate(names):
                ref = sum(r[j]*32 for r in references)/256
                estimate = sums[j]/256
                gradient = agreement(estimate, ref)
                pooled.append(dict(layer=j+1, name=name, gradient=gradient,
                    numerical_pass=numerical_pass(gradient), state=state_metrics(state_pool[j])))
                pooled_gradients[mode, target, name] = estimate
            # Counts pool corresponding updates over all eight equal-size batches.
            switches = []
            for sign in ['minus', 'plus']:
                for position in range(len(traces[0])):
                    subset = [r for r in updates if r['phase'] == sign and r['position'] == position]
                    assert len(subset) == 8
                    elements = sum(r['margins']['elements'] for r in subset)
                    counts = {k: sum(r[k] for r in subset) for k in
                              ['switch_count', 'clamp_to_free_count', 'free_to_clamp_count', 'clamped_count']}
                    switches.append(dict(phase=sign, position=position, update=subset[0]['update'],
                        state_name=subset[0]['state_name'], elements=elements, **counts,
                        switch_fraction=counts['switch_count']/elements))
            result = dict(mode=mode, target_relative_rms=target, beta_injected_B=beta,
                switch_count_semantics='unconstrained candidate decision versus zero trajectory; forced mode overrides it',
                batches=batches, layers=pooled, updates=updates, pooled_switches=switches,
                all_layer_batch_numerical_pass=all(r['numerical_pass'] for r in batches),
                all_pooled_layer_numerical_pass=all(r['numerical_pass'] for r in pooled))
            return result

        for target in TARGETS:
            cost = FrozenOutputCost(block.free_layers()[-1])
            augmented = AugmentedFunction(block.energy, cost, nudging_mode='current', current_scale=1.)
            minimizer = nudged_minimizer(block, augmented, K)
            match = dict(target_relative_rms=target, status='running', history=[])
            matches.append(match)
            def measure(beta):
                def records():
                    for saved in captures:
                        check_budget()
                        yield {2: saved}
                with block_scope(block, torch.float64):
                    value = measure_block(block, minimizer, augmented, cost, records(), 2, beta, device)
                match['history'].append(dict(beta_injected_B=beta,
                    relative_target_error=abs(value['relative_rms']/target-1), **value))
                reporting.atomic_write_json(out/'artifacts/calibration.json', matches)
                progress(stage='beta_calibration', target_relative_rms=target,
                    evaluation=len(match['history']), beta_injected_B=beta, **value)
                return value
            try:
                beta, history = search_beta(measure, target, metric='relative_rms', tolerance=.03,
                    max_evaluations=12, initial_beta=PRIOR_BETA*target/1e-5)
                match.update(status='matched', beta_injected_B=beta, achieved=history[-1])
            except BaseException as exc:
                match.update(status='failed', error=str(exc))
                raise
            finally:
                reporting.atomic_write_json(out/'artifacts/calibration.json', matches)
            value = evaluate(target, beta, 'native', low_response, initialize_low=target == TARGETS[0])
            native.append(value)
            if target == TARGETS[0]:
                comparison['low_target_reference_trusted'] = value['all_layer_batch_numerical_pass']
                comparison['causal_interpretation_eligible'] = value['all_layer_batch_numerical_pass']
            assert abs(value['layers'][-1]['state']['relative_rms']/target-1) <= .03
            reporting.atomic_write_json(out/'artifacts/comparison.json', comparison)
            if not value['all_layer_batch_numerical_pass'] and comparison['first_failing_target'] is None:
                comparison['first_failing_target'] = target
                low_beta = matches[0]['beta_injected_B']
                interventions.append(evaluate(TARGETS[0], low_beta, 'fixed_mask', fixed_low_response, initialize_low=True))
                if target != TARGETS[0]:
                    interventions.append(evaluate(target, beta, 'fixed_mask', fixed_low_response))
                reporting.atomic_write_json(out/'artifacts/comparison.json', comparison)
        verify_dtype(model, torch.float64)
        check_snapshot(before, model.snapshot())
        for path, digest in hashes.items():
            assert reporting.sha256_file(Path(path)) == digest, path
        for path, digest in source_hashes.items():
            assert reporting.sha256_file(source/path) == digest, path
        torch.save(pooled_gradients, out/'artifacts/pooled_gradients.pt')
        reporting.atomic_write_json(out/'artifacts/comparison.json', comparison)
        check_budget()
        reporting.complete_run(out, terminal_metrics=dict(examples=256, targets=len(native),
            first_failing_target=comparison['first_failing_target'], parameters_and_bn_unchanged=True,
            input_bytes_unchanged=True, source_bytes_unchanged=True, dtype='torch.float64',
            elapsed_seconds=time.monotonic()-start,
            peak_memory_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else 0,
            peak_memory_reserved_bytes=torch.cuda.max_memory_reserved(device) if device.type == 'cuda' else 0),
            completion=dict(criteria_met=True, all_calibration_targets_met=True,
                official_test_read=False, updates=0, fixed_bptt_reference=True,
                intervention_completed_if_triggered=True, exact_trace_order_and_count=True))
    except BaseException as exc:
        reporting.fail_run(out, error=exc)
        raise
    validate(out)


def validate(out):
    errors = reporting.validate_run(out)
    if errors:
        raise RuntimeError(errors)
    value = json.loads((out/'artifacts/comparison.json').read_text())
    result = json.loads((out/'result.json').read_text())
    assert result['completion']['criteria_met']
    assert value['rms_targets'] == TARGETS and len(value['native']) == len(TARGETS)
    assert value['epoch'] == 1 and value['local_bptt_iterations'] == value['nudged_iterations'] == K
    for match in value['calibration']:
        assert match['status'] == 'matched' and 1 <= len(match['history']) <= 12
        assert match['achieved']['relative_target_error'] <= .03
    failures = [r['target_relative_rms'] for r in value['native'] if not r['all_layer_batch_numerical_pass']]
    assert value['first_failing_target'] == (failures[0] if failures else None)
    if failures:
        expected = [TARGETS[0]] if failures[0] == TARGETS[0] else [TARGETS[0], failures[0]]
        assert [r['target_relative_rms'] for r in value['interventions']] == expected
    for cell in value['native'] + value['interventions']:
        assert len(cell['batches']) == 16 and len(cell['layers']) == 2
        assert len(cell['updates']) == 128 and len(cell['pooled_switches']) == 16
        assert cell['all_layer_batch_numerical_pass'] == all(numerical_pass(r['gradient']) for r in cell['batches'])
    assert len(value['zero_trace_rows']) == 64
    assert value['low_target_reference_trusted'] == value['native'][0]['all_layer_batch_numerical_pass']
    assert value['causal_interpretation_eligible'] == value['low_target_reference_trusted']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['config', 'checkpoint', 'dataset-root', 'run-dir']:
        parser.add_argument('--'+name, type=Path, required=name == 'run-dir')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--target', default='local')
    parser.add_argument('--budget-seconds', type=float, default=900.)
    parser.add_argument('--validate', action='store_true')
    args = parser.parse_args()
    if args.validate:
        validate(args.run_dir)
        print('VALIDATED', args.run_dir)
    else:
        if any(getattr(args, key) is None for key in ['config', 'checkpoint', 'dataset_root']):
            parser.error('--config --checkpoint --dataset-root are required')
        run(args)


if __name__ == '__main__':
    main()
