"""Bounded phase timing and numerical A/B checks for the CIFAR hybrid update."""
from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import contextmanager, ExitStack
import inspect
import json
from pathlib import Path
import statistics
import time
from unittest.mock import patch

import torch

from experiments import reporting
from experiments import cifar_conv_eqprop as hybrid
from experiments import compare_cifar_block_beta_gradients as comparison
from experiments.train_cifar_l8_analog import (Cohort, configure_training_precision,
    deterministic_setup, loader, make_optimizer, make_split, read_training_data,
    verify_training_precision)
from labs.cifar_l8_analog import CifarL8Analog


class Timings:
    def __init__(self):
        self.events = []

    @contextmanager
    def section(self, name):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        try:
            yield
        finally:
            end.record()
            self.events.append((name, start, end))

    def wrap(self, name, fn):
        def call(*args, **kwargs):
            with self.section(name):
                return fn(*args, **kwargs)
        return call

    def collect(self):
        torch.cuda.synchronize()
        values = defaultdict(float)
        for name, start, end in self.events:
            values[name] += start.elapsed_time(end) / 1000
        self.events.clear()
        return dict(values)


def cpu_tree(value):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: cpu_tree(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [cpu_tree(v) for v in value]
    return value


def compare_tree(actual, expected, path='', rows=None):
    rows = [] if rows is None else rows
    if torch.is_tensor(expected):
        assert actual.dtype == expected.dtype and actual.shape == expected.shape, path
        a, b = actual.double(), expected.double()
        error = (a - b).abs()
        rows.append(dict(name=path, exact=torch.equal(actual, expected),
                         max_absolute=float(error.max()) if error.numel() else 0.,
                         relative_l2=float(error.norm() / b.norm().clamp_min(1e-300))))
        torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-12, msg=path)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys(), path
        for k in expected:
            compare_tree(actual[k], expected[k], f'{path}/{k}', rows)
    elif isinstance(expected, (list, tuple)):
        assert len(actual) == len(expected), path
        for k, (a, b) in enumerate(zip(actual, expected)):
            compare_tree(a, b, f'{path}/{k}', rows)
    else:
        assert actual == expected, path
    return rows


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--dataset-root', type=Path, required=True)
    p.add_argument('--run-dir', type=Path, required=True)
    p.add_argument('--variant', choices=['reference', 'optimized'], default='reference')
    p.add_argument('--compare-to', type=Path)
    p.add_argument('--steps', type=int, default=4)
    p.add_argument('--warmup-steps', type=int, default=1)
    p.add_argument('--budget-seconds', type=float, default=0.,
                   help='Bound the diagnostic runtime; zero retains the original unbounded behavior')
    p.add_argument('--target', default='trex')
    p.add_argument('--validate', action='store_true')
    p.add_argument('--operators', action='store_true', help='Also time convolution helpers (nested intervals)')
    p.add_argument('--no-coefficient-cache', action='store_true')
    p.add_argument('--nudged-precision', choices=['same', 'float64'], default='same',
                   help='Diagnostic split: float32 model/optimizer with float64 nudged solves')
    p.add_argument('--phase-mode', choices=['sequential', 'packed'], default='sequential')
    p.add_argument('--float64-blocks', type=int, nargs='+', default=[1, 2, 3],
                   help='One-based blocks promoted when --nudged-precision=float64')
    p.add_argument('--nudged-iterations', type=int, nargs=3,
                   help='Override only nudged K for the three blocks; free T is unchanged')
    args = p.parse_args()
    if args.validate:
        errors = reporting.validate_run(args.run_dir)
        if errors:
            raise ValueError(errors)
        print('VALIDATED', args.run_dir)
        return
    if not 1 <= args.warmup_steps < args.steps or args.budget_seconds < 0:
        raise ValueError('Need warm-up and measured updates, and a nonnegative time budget')
    started = time.monotonic()
    c = json.loads(args.config.read_text())
    nudged_iterations = args.nudged_iterations or c['iterations']
    if len(nudged_iterations) != 3 or any(k < 1 for k in nudged_iterations):
        raise ValueError('Need three positive nudged iteration counts')
    dtype = getattr(torch, c['precision'])
    if dtype not in (torch.float32, torch.float64):
        raise ValueError('Expected float32 or float64')
    split_precision = args.nudged_precision == 'float64'
    if not args.float64_blocks or len(set(args.float64_blocks)) != len(args.float64_blocks) or any(
            block not in (1, 2, 3) for block in args.float64_blocks):
        raise ValueError('Float64 blocks must be unique indices from 1, 2, 3')
    if split_precision and dtype != torch.float32:
        raise ValueError('Split precision requires a float32 main model')
    deterministic_setup(c['seed'])
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required')
    source = Path(__file__).resolve().parents[1]
    files = [p for folder in ['experiments', 'labs', 'model', 'training']
             for p in (source / folder).rglob('*.py')]
    reporting.start_run(args.run_dir, dict(study_id=c.get('study_id', 'cifar-float64-performance-20261001'),
        run_id=args.run_dir.name, arm_id=args.variant, mode='diagnostic', smoke=False,
        evidence_class='performance_profile', resolved_config=c,
        dataset=dict(name='CIFAR10', official_test_read=False),
        runtime=reporting.runtime_context(target=args.target),
        environment=dict(torch=str(torch.__version__), cuda=torch.version.cuda,
                         gpu=torch.cuda.get_device_name()),
        checkpoint_sha256=reporting.sha256_file(args.checkpoint),
        source_sha256={str(p.relative_to(source)): reporting.sha256_file(p) for p in files},
        steps=args.steps, warmup_steps=args.warmup_steps, operators=args.operators,
        budget_seconds=args.budget_seconds,
        coefficient_cache=not args.no_coefficient_cache and args.variant == 'optimized',
        nudged_precision=args.nudged_precision,
        float64_blocks=args.float64_blocks if split_precision else [],
        phase_mode=args.phase_mode, free_iterations=c['iterations'],
        nudged_iterations=nudged_iterations,
        compare_to=str(args.compare_to)))
    try:
        images, labels = read_training_data(args.dataset_root)
        split = make_split(labels, c['split']['seed'])
        order = torch.randperm(len(split['train']), generator=torch.Generator().manual_seed(c['seed']+1000000)).tolist()
        indices = [split['train'][i] for i in order[:args.steps*c['batch_size']]]
        # A short, deterministic CPU cohort avoids worker-startup distortion.
        batches = list(loader(Cohort(images, labels, indices, c, 1, True), c['batch_size']))
        if len(batches) != args.steps:
            raise ValueError('Requested steps exceed the available training cohort')
        model = CifarL8Analog(c, 'cuda')
        configure_training_precision(model, c)
        initial = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
        model.restore(initial['model'])
        del initial
        optimizer, _ = make_optimizer(model, c)
        model.train()
        donor = None
        if split_precision:
            from experiments.replay_cifar_precision_response import convert_model
            donor = CifarL8Analog(c, 'cuda')
            convert_model(donor, torch.float64)
        timing = Timings()
        block_ids = {id(block): i+1 for i, block in enumerate(model.analog_blocks)}
        original_ep, original_minimizer = hybrid.eqprop_gradients, comparison.nudged_minimizer
        original_gradient = comparison.stable_centered_conv_gradient
        active_block = [0]
        forces = {}

        def ep(block, saved, beta, iterations, device, **kw):
            active_block[0] = block_ids[id(block)]
            count = nudged_iterations[active_block[0]-1]
            kw['phase_mode'] = args.phase_mode
            # Only retain GPU references; copies occur after the timed update.
            forces[active_block[0]] = saved['gradient']
            with timing.section(f'block{active_block[0]}/eqprop_total'):
                if split_precision and active_block[0] in args.float64_blocks:
                    from experiments.compare_cifar_split_precision import eqprop_from_float32
                    target = donor.analog_blocks[active_block[0]-1]
                    with torch.no_grad():
                        for dst, src in zip([*target.weights, *target.biases], [*block.weights, *block.biases]):
                            dst.state.copy_(src.state)
                    with ExitStack() as cache:
                        if not args.no_coefficient_cache:
                            for interaction in target.interactions:
                                cache.enter_context(interaction.cache_a_col2im())
                        return eqprop_from_float32(target, saved, beta, count, device, **kw)
                return original_ep(block, saved, beta, count, device, **kw)

        def minimizer(*a, **kw):
            result = original_minimizer(*a, **kw)
            result.compute_equilibrium = timing.wrap(
                f'block{active_block[0]}/nudged_solves', result.compute_equilibrium)
            return result

        def gradient(*a, **kw):
            with timing.section(f'block{active_block[0]}/gradient_formula'):
                return original_gradient(*a, **kw)

        options = {}
        if 'skip_conv_autograd' in inspect.signature(hybrid.hybrid_backward).parameters:
            options['skip_conv_autograd'] = args.variant == 'optimized'
        elif args.variant == 'optimized':
            raise RuntimeError('Optimization flag unavailable in this source')
        if 'cache_conv_coefficients' in inspect.signature(hybrid.hybrid_backward).parameters:
            options['cache_conv_coefficients'] = args.variant == 'optimized' and not args.no_coefficient_cache
        rows = []
        torch.cuda.reset_peak_memory_stats()
        with ExitStack() as operator_patches, \
             patch.object(model, 'forward', timing.wrap('free_forward', model.forward)), \
             patch.object(torch.autograd, 'backward', timing.wrap('autograd_backward', torch.autograd.backward)), \
             patch.object(hybrid, 'eqprop_gradients', ep), \
             patch.object(comparison, 'nudged_minimizer', minimizer), \
             patch.object(comparison, 'stable_centered_conv_gradient', gradient):
            if args.operators:
                for name in ['im2col', 'col2im', 'a_im2col', 'a_col2im']:
                    operator_patches.enter_context(patch.object(comparison.ConvResistive, name,
                        timing.wrap('conv_helper/'+name, getattr(comparison.ConvResistive, name))))
            for step, (x, y) in enumerate(batches):
                if args.budget_seconds and time.monotonic() - started >= args.budget_seconds:
                    raise TimeoutError('Diagnostic profiling budget exhausted')
                torch.cuda.synchronize()
                start = time.perf_counter()
                with timing.section('transfer_zero_grad'):
                    x, y = x.to(device='cuda', dtype=dtype), y.cuda()
                    optimizer.zero_grad(set_to_none=True)
                with timing.section('hybrid_total'):
                    prediction, loss, responses = hybrid.hybrid_backward(model, x, y,
                        c['iterations'], c['conv_eqprop']['initial_betas'],
                        measure_displacement=True, **options)
                with timing.section('optimizer_project'):
                    optimizer.step()
                    model.project_()
                    model.detach_state_()
                with timing.section('metrics'):
                    loss_value = float(loss)
                    accuracy = float((prediction.argmax(1) == y).double().mean())
                phases = timing.collect()
                row = dict(step=step+1, warmup=step < args.warmup_steps, wall_seconds=time.perf_counter()-start,
                           phases=phases, loss=loss_value, accuracy=accuracy)
                rows.append(row)
                reporting.append_metric(args.run_dir/'metrics.jsonl', row)
                reporting.update_status_progress(args.run_dir, dict(stage='profile', step=step+1))
                print(json.dumps(row), flush=True)
                if step == 0 or step == len(batches)-1:
                    label = 'first_update' if step == 0 else 'final_update'
                    first = cpu_tree(dict(prediction=prediction, loss=loss, forces=forces,
                        gradients={k: v.grad for k,v in model.trainable_tensors().items()},
                        model=model.snapshot(), optimizer=optimizer.state_dict()))
                    torch.save(first, args.run_dir/'checkpoints'/f'{label}.pt')
                    if args.compare_to:
                        expected = torch.load(args.compare_to/f'checkpoints/{label}.pt', weights_only=False)
                        comparison_rows = compare_tree(first, expected)
                        reporting.atomic_write_json(args.run_dir/'artifacts'/f'equivalence_{label}.json',
                            dict(passed=True, tolerance=dict(rtol=1e-10, atol=1e-12), tensors=comparison_rows))
                        del expected
                    del first
        verify_training_precision(model, c, optimizer)
        measured = rows[args.warmup_steps:]
        summary = dict(variant=args.variant, measured_updates=len(measured),
            precision=c['precision'], nudged_precision=args.nudged_precision,
            float64_blocks=args.float64_blocks if split_precision else [],
            phase_mode=args.phase_mode, free_iterations=c['iterations'],
            nudged_iterations=nudged_iterations, batch_size=c['batch_size'],
            median_seconds=statistics.median(r['wall_seconds'] for r in measured),
            mean_seconds=statistics.mean(r['wall_seconds'] for r in measured),
            measured_update_seconds=sum(r['wall_seconds'] for r in measured),
            projected_epoch_updates=(len(split['train']) + c['batch_size'] - 1) // c['batch_size'],
            mean_phases={k: statistics.mean(r['phases'].get(k,0.) for r in measured)
                         for k in rows[-1]['phases']},
            peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            note='Nested CUDA event intervals; EqProp totals include solves/formula. Declared warm-up updates are excluded. Diagnostic updates are discarded; no training continuation.')
        reporting.atomic_write_json(args.run_dir/'artifacts/timing.json', summary)
        reporting.complete_run(args.run_dir, terminal_metrics=summary,
                               completion=dict(criteria_met=True, official_test_read=False))
        print(json.dumps(summary), flush=True)
    except BaseException as exc:
        reporting.fail_run(args.run_dir, error=exc)
        raise


if __name__ == '__main__':
    main()
