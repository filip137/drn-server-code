"""Matched CIFAR circuit diagnostics and a bounded Adam equivalence control."""
from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import sys
import time

import numpy as np
import torch
from torch.nn import functional as F

from experiments import reporting
from experiments.train_cifar_l8_analog import (
    Cohort, audit, check_gpu_resident_execution, deterministic_setup, loader,
    make_optimizer, read_training_data,
)
from labs.cifar_l8_analog import CifarL8Analog


def relative_error(a, b):
    a, b = a.detach(), b.detach()
    if not (a.is_floating_point() or b.is_floating_point()):
        return 0. if torch.equal(a, b) else 1.
    return float((a - b).norm() / torch.maximum(a.norm(), b.norm()).clamp_min(1e-12))


def mapping_error(a, b):
    if a.keys() != b.keys():
        raise ValueError('Compared tensor identities differ')
    errors = {name: relative_error(value, b[name]) for name, value in a.items()}
    name = max(errors, key=errors.get)
    return {'maximum_relative_error': errors[name], 'worst_tensor': name}


def check_unchanged(model, state, *, buffers=True):
    current = model.snapshot()
    for category, values in state.items():
        for name, value in values.items():
            if not buffers and (name.endswith('running_mean') or name.endswith('running_var')
                                or name.endswith('num_batches_tracked')):
                continue
            if not torch.equal(value.cpu(), current[category][name]):
                raise RuntimeError(f'Replay mutated saved tensor {category}/{name}')


def equivalence_step(models, optimizers, x, y, iterations):
    observations = []
    for model, optimizer in zip(models, optimizers):
        model.train(); optimizer.zero_grad(set_to_none=True)
        params = model.trainable_tensors()
        before = {n: p.detach().clone() for n, p in params.items()}
        free, tracked = model.bptt(x, iterations)
        loss = F.cross_entropy(tracked, y)
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite equivalence loss')
        loss.backward()
        gradients = {n: p.grad.detach().clone() for n, p in params.items()}
        optimizer.step(); model.project_(); model.detach_state_()
        observations.append({
            'free': free.detach(), 'tracked': tracked.detach(), 'loss': loss.detach(),
            'gradients': gradients,
            'parameters': {n: p.detach().clone() for n, p in params.items()},
            'updates': {n: p.detach() - before[n] for n, p in params.items()},
            'bn': {n: p.detach().clone() for n, p in model.named_buffers()},
        })
    a, b = observations
    errors = {name: mapping_error(a[name], b[name])
              for name in ('gradients', 'parameters', 'updates', 'bn')}
    errors.update({name: {'maximum_relative_error': relative_error(a[name], b[name])}
                   for name in ('free', 'tracked', 'loss')})
    maximum = max(row['maximum_relative_error'] for row in errors.values())
    return {'errors': errors, 'maximum_relative_error': maximum,
            'passed': maximum <= 1e-5, 'baseline_loss': float(a['loss']),
            'legacy_loss': float(b['loss'])}


def run_equivalence(args, images, labels, split, config):
    original = args.study_root / 'inputs/initial_model.pt'
    before_hash = reporting.sha256_file(original)
    saved = torch.load(original, map_location='cpu', weights_only=False)
    c = copy.deepcopy(config)
    legacy = copy.deepcopy(c)
    legacy['solver'].update(voltage_amp=4., current_amp=.25)
    legacy['block_output_normalization'] = 'voltage'
    models = [CifarL8Analog(cfg, args.device) for cfg in (c, legacy)]
    for model in models:
        model.restore(saved['model'])
    optimizers = [make_optimizer(model, c)[0] for model in models]
    permutation = torch.randperm(len(split['train']), generator=torch.Generator().manual_seed(
        c['seed'] + 1000000)).tolist()
    indices = [split['train'][i] for i in permutation[:args.steps * c['batch_size']]]
    maximum = 0.
    for step, (x, y) in enumerate(loader(Cohort(images, labels, indices, c, 1, True),
                                       c['batch_size']), 1):
        args.check_deadline()
        row = equivalence_step(models, optimizers, x.to(args.device), y.to(args.device), c['iterations'])
        row.update(stage='equivalence', step=step, expected_steps=args.steps)
        reporting.append_metric(args.output_dir / 'metrics.jsonl', row)
        reporting.update_status_progress(args.output_dir, {'stage': 'equivalence', 'step': step,
                                                          'expected_steps': args.steps})
        maximum = max(maximum, row['maximum_relative_error'])
        if step == 1 or step % 20 == 0:
            print('EQUIVALENCE', step, 'max_relative_error', maximum, flush=True)
        if not row['passed']:
            raise RuntimeError(f'Equivalence rejected at minibatch {step}: {row}')
    if step != args.steps:
        raise RuntimeError('Equivalence coverage incomplete')
    if reporting.sha256_file(original) != before_hash:
        raise RuntimeError('Initializer bytes changed')
    return {'steps': step, 'maximum_relative_error': maximum, 'threshold': 1e-5,
            'initializer_sha256': before_hash, 'passed': True}


def moments(value):
    v = value.detach().float()
    return {'count': v.numel(), 'mean': float(v.mean()), 'rms': float(v.square().mean().sqrt()),
            'std': float(v.std(unbiased=False)), 'min': float(v.min()), 'max': float(v.max())}


@torch.no_grad()
def describe_states(model, mode):
    records = []
    for block_index, block in enumerate(model.analog_blocks):
        for index, layer in enumerate(block.free_layers()):
            v = layer.state.detach()
            a = block.energy.a_coef_fn(layer)().detach()
            b = block.energy.b_coef_fn(layer)().detach()
            gradient = 2 * a * v + b
            half = v.shape[1] // 2
            projected = torch.cat((
                torch.where(v[:, :half] > 0, gradient[:, :half], gradient[:, :half].clamp(max=0)),
                torch.where(v[:, half:] < 0, gradient[:, half:], gradient[:, half:].clamp(min=0)),
            ), dim=1)
            denominator = (2 * a * v).norm() + b.norm()
            incoming = block.weights[index].state.detach().sum((1, 2, 3))[None, :, None, None]
            if index + 1 < len(block.weights):
                following = block.free_layers()[index + 1].state
                downstream = (block.solver.voltage_amp * block.solver.current_amp) * F.conv_transpose2d(
                    torch.ones_like(following[:1]), block.weights[index + 1].state.detach(), padding=1)
                loading = downstream / (incoming + downstream)
                loading_median = float(loading.median())
            else:
                loading_median = 0.
            records.append({'mode': mode, 'block': block_index, 'layer': index + 1,
                'voltage': moments(v),
                'normalized_voltage': moments(v / block.solver.voltage_amp ** index),
                'clamped_fraction': float((v == 0).float().mean()),
                'projected_kkt_relative_l2': float(projected.norm() / denominator.clamp_min(1e-30)),
                'normalized_downstream_loading_median': loading_median})
    return records


def capture_bn(model, destination):
    handles = []
    for i, bridge in enumerate(model.bridges):
        def capture(module, inputs, index=i):
            x = inputs[0].detach()
            variance = x.var(dim=(0, 2, 3), unbiased=False)
            used_variance = variance if module.training else module.running_var.detach()
            share = module.eps / (used_variance + module.eps)
            destination.append({'block': index, 'training': module.training,
                'phase': ('free' if len(destination) < len(model.bridges) else 'tracked') if module.training else 'evaluation',
                'batch_variance': variance.cpu().tolist(), 'used_variance': used_variance.cpu().tolist(),
                'running_variance': module.running_var.detach().cpu().tolist(),
                'epsilon': module.eps, 'epsilon_share': share.cpu().tolist(),
                'epsilon_share_median': float(share.median())})
        handles.append(bridge[1].register_forward_pre_hook(capture))
    return handles


def adam_proposal_stats(model, checkpoint, config):
    rows = []
    groups = checkpoint['optimizer']['param_groups']
    tensors = model.trainable_tensors()
    for group in groups:
        name = group['name']; p = tensors[name]; g = p.grad.detach()
        state = checkpoint['optimizer']['state'].get(group['params'][0], {})
        step = int(state.get('step', 0)) + 1
        beta1, beta2 = group['betas']
        first = state.get('exp_avg', torch.zeros_like(p)).to(p.device)
        second = state.get('exp_avg_sq', torch.zeros_like(p)).to(p.device)
        first = (beta1 * first + (1 - beta1) * g) / (1 - beta1 ** step)
        second = (beta2 * second + (1 - beta2) * g.square()) / (1 - beta2 ** step)
        proposal = -group['lr'] * first / (second.sqrt() + group['eps'])
        if name.startswith('blocks.') or name == 'head.weight':
            projected = (p.detach() + proposal).clamp(config['weight_min'], config['weight_max']) - p.detach()
        else:
            projected = (p.detach() + proposal) - p.detach()
        parameter_norm = float(p.detach().norm())
        rows.append({'parameter': name, 'parameter_l2': parameter_norm, 'gradient_l2': float(g.norm()),
            'gradient_rms': float(g.square().mean().sqrt()),
            'gradient_zero_fraction': float((g.abs() <= 1e-12).float().mean()),
            'next_step_lr': group['lr'], 'adam_next_step': step,
            'adam_proposal_l2': float(proposal.norm()), 'projected_adam_proposal_l2': float(projected.norm()),
            'relative_adam_proposal_l2': float(proposal.norm()) / parameter_norm if parameter_norm else None,
            'relative_projected_adam_proposal_l2': float(projected.norm()) / parameter_norm if parameter_norm else None})
    return rows


def measure_checkpoint(args, record, images, labels, indices):
    path = args.study_root / record['checkpoint']
    checksum = reporting.sha256_file(path)
    if checksum != record['sha256']:
        raise RuntimeError(f'Checkpoint hash mismatch: {path}')
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    config = json.loads((args.study_root / record['config']).read_text())
    model = CifarL8Analog(config, args.device)
    saved = checkpoint['model']; model.restore(saved)
    dataset = Cohort(images, labels, indices, config)
    batches = list(loader(dataset, 32))
    rows = []
    for mode in ('evaluation', 'training'):
        for batch, (x, y) in enumerate(batches):
            args.check_deadline(); model.restore(saved); model.train(mode == 'training')
            for p in model.trainable_tensors().values():
                p.grad = None
            bn, state_passes = [], []
            handles = capture_bn(model, bn)
            def capture_states(module, inputs, output):
                phase = ('free' if not state_passes else 'tracked') if mode == 'training' else 'evaluation'
                state_passes.append(describe_states(model, phase))
            handles.append(model.head.register_forward_hook(capture_states))
            x, y = x.to(args.device), y.to(args.device)
            try:
                if mode == 'training':
                    free, logits = model.bptt(x, config['iterations'])
                    loss = F.cross_entropy(logits, y); loss.backward()
                    gradients = adam_proposal_stats(model, checkpoint, config)
                else:
                    with torch.no_grad():
                        logits = model(x, config['iterations']); loss = F.cross_entropy(logits, y)
                    gradients = []
                row = {'stage': 'checkpoint', 'source': record['id'], 'epoch': record['epoch'],
                    'mode': mode, 'batch': batch, 'examples': len(y), 'loss': float(loss.detach()),
                    'accuracy': float((logits.argmax(1) == y).float().mean()),
                    'states': [value for group in state_passes for value in group], 'bn': bn, 'gradients': gradients}
                if not torch.isfinite(loss):
                    raise FloatingPointError('Nonfinite replay loss')
                check_unchanged(model, saved, buffers=False)
                rows.append(row); reporting.append_metric(args.output_dir / 'metrics.jsonl', row)
                reporting.update_status_progress(args.output_dir, {'stage': 'replay', 'checkpoint': record['id'],
                    'mode': mode, 'batch': batch + 1, 'batches': len(batches)})
            finally:
                for handle in handles:
                    handle.remove()
                model.detach_state_()
                model.restore(saved)
    # State convergence uses the full batch32 cohort without building a long
    # autograd graph. The separately labelled gradient gate retains its native
    # 8-example/microbatch2 protocol to fit long unrolls without CPU offload.
    for schedule_name, iterations in [('reference', config['reference_iterations']),
                                      ('sentinel', config['reference_check_iterations'])]:
        for batch, (x, _) in enumerate(batches):
            args.check_deadline(); model.restore(saved); model.train()
            with torch.no_grad():
                logits = model(x.to(args.device), iterations)
                row = {'stage': 'long_state', 'source': record['id'], 'epoch': record['epoch'],
                       'schedule': schedule_name, 'iterations': iterations, 'batch': batch,
                       'states': describe_states(model, 'training'), 'logit_rms': float(logits.square().mean().sqrt())}
                reporting.append_metric(args.output_dir / 'metrics.jsonl', row)
            model.detach_state_(); model.restore(saved)
    args.check_deadline()
    gate = audit(model, Cohort(images, labels, indices[:config['gate']['cohort_size']], config),
                 config, torch.device(args.device), [config['iterations']], args.output_dir)
    reporting.atomic_write_json(args.output_dir / 'artifacts' / (record['id'] + '_solver_audit.json'), gate)
    check_unchanged(model, saved)
    if reporting.sha256_file(path) != checksum:
        raise RuntimeError('Replay changed checkpoint bytes')
    reporting.append_metric(args.output_dir / 'metrics.jsonl', {'stage': 'checkpoint_guard',
        'source': record['id'], 'epoch': record['epoch'], 'checkpoint_sha256': checksum,
        'parameters_unchanged': True, 'bn_buffers_restored': True,
        'gradient_gate_microbatch_size': config['gate']['microbatch_size'],
        'gradient_gate_examples': config['gate']['cohort_size'], 'solver_audit_passed': gate['passed']})
    del rows, checkpoint, saved, model, batches
    gc.collect(); torch.cuda.empty_cache()
    return {'id': record['id'], 'epoch': record['epoch'], 'sha256': checksum, 'solver_audit_passed': gate['passed']}


def matched_weight_replay(args, record, images, labels, indices):
    path = args.study_root / record['checkpoint']
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    config = json.loads((args.study_root / record['config']).read_text())
    results = {}
    for variant in ('baseline', 'legacy', 'legacy_normalized'):
        c = copy.deepcopy(config)
        if variant != 'baseline':
            c['solver'].update(voltage_amp=4., current_amp=.25)
        c['block_output_normalization'] = 'voltage' if variant == 'legacy_normalized' else 'none'
        model = CifarL8Analog(c, args.device); model.restore(checkpoint['model']); model.train()
        variant_rows = []
        for x, y in loader(Cohort(images, labels, indices, c), 32):
            args.check_deadline(); model.restore(checkpoint['model'])
            for p in model.trainable_tensors().values():
                p.grad = None
            _, logits = model.bptt(x.to(args.device), c['iterations'])
            F.cross_entropy(logits, y.to(args.device)).backward()
            variant_rows.append({'logits': logits.detach().cpu(),
                'gradients': {n: p.grad.detach().cpu().clone() for n, p in model.trainable_tensors().items()}})
            check_unchanged(model, checkpoint['model'], buffers=False); model.detach_state_()
        model.restore(checkpoint['model']); check_unchanged(model, checkpoint['model'])
        results[variant] = variant_rows
        del model; gc.collect(); torch.cuda.empty_cache()
    for variant in ('legacy', 'legacy_normalized'):
        for batch, (a, b) in enumerate(zip(results['baseline'], results[variant])):
            row = {'stage': 'matched_weights', 'source': record['id'], 'epoch': record['epoch'],
                   'variant': variant, 'mode': 'training', 'batch': batch,
                   'relative_logit_error': relative_error(a['logits'], b['logits']),
                   'gradient_comparison': mapping_error(a['gradients'], b['gradients'])}
            reporting.append_metric(args.output_dir / 'metrics.jsonl', row)
    if reporting.sha256_file(path) != record['sha256']:
        raise RuntimeError('Matched replay changed checkpoint bytes')
    return {'source': record['id'], 'compared_variants': list(results), 'checkpoint_unchanged': True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['equivalence', 'replay'], required=True)
    parser.add_argument('--study-root', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--dataset-root', type=Path, required=True)
    parser.add_argument('--source-identity', type=Path, required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--target', choices=['loulou', 'fifi'], default='loulou')
    parser.add_argument('--steps', type=int, default=200)
    parser.add_argument('--maximum-hours', type=float, default=1.)
    parser.add_argument('--checkpoint-id', action='append', help='Optional bounded replay subset')
    args = parser.parse_args(); args.study_root = args.study_root.resolve(); args.output_dir = args.output_dir.resolve()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    check_gpu_resident_execution(args.device, False)
    deterministic_setup(0)
    identity = json.loads(args.source_identity.read_text())
    repo = Path(__file__).resolve().parents[1]
    for name, digest in identity['source_sha256'].items():
        if reporting.sha256_file(repo / name) != digest:
            raise ValueError('Frozen source mismatch: ' + name)
    config = json.loads((args.study_root / 'configs/baseline_reference.json').read_text())
    split = json.loads((args.study_root / 'inputs/split.json').read_text())
    indices = split['train'][:128]
    reporting.start_run(args.output_dir, {'study_id': config['study_id'], 'run_id': args.output_dir.name,
        'arm_id': args.mode, 'smoke': args.mode == 'equivalence',
        'evidence_class': 'cifar10_amplification_mechanism_diagnostic', 'source_identity': identity,
        'command': shlex.join([sys.executable, *sys.argv]), 'resolved_config': config,
        'runtime': dict(reporting.runtime_context(target=args.target), pid=os.getpid()),
        'dataset': {'name': 'CIFAR10', 'official_test_read': False},
        'cohort_indices': indices, 'cohort_sha256': hashlib.sha256(json.dumps(indices).encode()).hexdigest(),
        'environment': {'torch': str(torch.__version__), 'cuda': torch.version.cuda,
                        'cudnn': torch.backends.cudnn.version(), 'gpu': torch.cuda.get_device_name()},
        'maximum_hours': args.maximum_hours, 'steps': args.steps})
    start = time.monotonic()
    def check_deadline():
        if time.monotonic() - start > args.maximum_hours * 3600:
            raise TimeoutError('Diagnostic runtime cap exhausted')
    args.check_deadline = check_deadline
    try:
        images, labels = read_training_data(args.dataset_root)
        if args.mode == 'equivalence':
            measured = run_equivalence(args, images, labels, split, config)
        else:
            records = json.loads((args.study_root / 'inputs/checkpoints.json').read_text())
            if args.checkpoint_id:
                requested = set(args.checkpoint_id)
                records = [r for r in records if r['id'] in requested]
                if {r['id'] for r in records} != requested:
                    raise ValueError('Requested checkpoint missing')
            completed, matched = [], []
            for record in records:
                print('REPLAY', record['id'], flush=True)
                completed.append(measure_checkpoint(args, record, images, labels, indices))
                if record['scheme'] == 'baseline':
                    matched.append(matched_weight_replay(args, record, images, labels, indices))
            measured = {'checkpoints': completed, 'matched_weight_checks': matched,
                        'all_checkpoint_guards_passed': True, 'cohort_examples': len(indices)}
        measured['elapsed_seconds'] = time.monotonic() - start
        reporting.atomic_write_json(args.output_dir / 'artifacts/summary.json', measured)
        reporting.complete_run(args.output_dir, terminal_metrics=measured,
                               completion={'criteria_met': True, 'official_test_read': False})
        errors = reporting.validate_run(args.output_dir)
        if errors:
            raise RuntimeError(errors)
        print('COMPLETE', args.mode, measured, flush=True)
    except BaseException as error:
        if not (args.output_dir / 'result.json').exists():
            reporting.fail_run(args.output_dir, error=error)
        raise


if __name__ == '__main__':
    main()
