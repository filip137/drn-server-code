"""Evaluate a fixed digital L8 checkpoint after a direct legacy sign-split transfer."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import pickle
import shlex
import subprocess
import sys
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from experiments import reporting
from experiments.train_cifar_digital_l12 import DigitalL12
from experiments.train_cifar_l8_analog import (
    Cohort, deterministic_setup, loader, make_split, read_training_data,
)
from labs.cifar_l8_analog import CifarL8Analog


def split_convolution(weight, floor, ceiling):
    """Symmetric pairs encode [h,-h]; equal floors cancel in the signed drive."""
    positive = weight.clamp_min(0) + floor
    negative = (-weight).clamp_min(0) + floor
    conductance = torch.cat((torch.cat((positive, negative), 1),
                             torch.cat((negative, positive), 1)), 0)
    if not torch.isfinite(conductance).all() or conductance.max() > ceiling:
        raise ValueError('Transfer requires invalid/out-of-bounds conductances')
    return conductance


@torch.no_grad()
def transfer(digital, analog, config):
    convolutions = [m for m in digital.modules() if isinstance(m, nn.Conv2d)]
    norms = [m for m in digital.modules() if isinstance(m, nn.BatchNorm2d)]
    targets = [w.state for block in analog.analog_blocks for w in block.weights]
    if len(convolutions) != len(targets) or len(norms) != len(analog.bridges):
        raise ValueError('Incompatible block counts')
    floor, ceiling = config['weight_min'], config['weight_max']
    errors = []
    for source, target in zip(convolutions, targets):
        if source.bias is not None or source.stride != (1, 1) or source.padding != (1, 1):
            raise ValueError('Incompatible convolution geometry')
        mapped = split_convolution(source.weight, floor, ceiling)
        if mapped.shape != target.shape:
            raise ValueError('Incompatible convolution dimensions')
        target.copy_(mapped)
        out_channels, in_channels = source.weight.shape[:2]
        recovered = target[:out_channels, :in_channels] - target[:out_channels, in_channels:]
        torch.testing.assert_close(recovered, source.weight, atol=2e-8, rtol=1e-6)
        errors.append(float((recovered - source.weight).abs().max()))
    for source, bridge in zip(norms, analog.bridges):
        if source.eps != bridge[1].eps:
            raise ValueError('BN epsilon mismatch')
        bridge[1].load_state_dict(source.state_dict(), strict=True)
    weight = digital.head.weight.T
    mapped_head = torch.cat((weight.clamp_min(0) + floor,
                             (-weight).clamp_min(0) + floor), 0)
    if mapped_head.shape != analog.head.weight.state.shape or mapped_head.max() > ceiling:
        raise ValueError('Incompatible analog classifier')
    analog.head.weight.state.copy_(mapped_head)
    return {'convolution_reconstruction_max_errors': errors,
            'digital_head_bias': digital.head.bias.tolist(),
            'native_head_bias_representable': False,
            'native_head_policy': 'omit_bias_explicitly; no fitting or calibration'}


@torch.no_grad()
def analog_features(model, images, iterations):
    for block, bridge, count, scale in zip(model.analog_blocks, model.bridges,
                                          iterations, model.block_output_scales):
        images = bridge(block(images, num_iterations=count, reset=True) / scale)
    return images.flatten(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True, type=Path)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--dataset-root', required=True, type=Path,
                        help='Directory containing CIFAR data_batch_1, ..., test_batch')
    parser.add_argument('--split', choices=['original-test', 'training-overlap-validation'], required=True)
    parser.add_argument('--head', choices=['digital', 'native-analog-omit-bias', 'both'], required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--limit', type=int, default=0, help='Nonzero means smoke only')
    parser.add_argument('--maximum-seconds', type=float, default=3600)
    parser.add_argument('--source-identity', type=Path)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if args.limit < 0 or args.batch_size < 1 or args.maximum_seconds <= 0:
        raise ValueError('Invalid execution bounds')
    deterministic_setup(0)
    device = torch.device(args.device)
    checkpoint_hash = reporting.sha256_file(args.checkpoint)
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    config = json.loads(args.config.read_text())
    if config['blocks'] != checkpoint['config']['blocks']:
        raise ValueError('Block architecture mismatch')
    if config['normalization'] != {k: checkpoint['config']['dataset'][k] for k in ('mean', 'std')}:
        raise ValueError('Input normalization mismatch')
    digital = DigitalL12(checkpoint['config']).to(device).eval()
    digital.load_state_dict(checkpoint['model'], strict=True)
    analog = CifarL8Analog(config, device).eval()
    audit = transfer(digital, analog, config)
    before = analog.snapshot()
    source_files = [Path('experiments/replay_cifar_digital_to_legacy.py'), Path('labs/cifar_l8_analog.py'),
                    Path('experiments/train_cifar_digital_l12.py'),
                    Path('experiments/train_cifar_l8_analog.py'), Path('experiments/reporting.py')]
    source_files += sorted(Path('model').rglob('*.py'))
    source_files += [Path('labs/custom_minimizer.py')]
    source_hashes = {str(p): reporting.sha256_file(p) for p in source_files}
    if args.source_identity:
        identity = json.loads(args.source_identity.read_text())
        for name, digest in identity['source_sha256'].items():
            if reporting.sha256_file(Path(name)) != digest:
                raise ValueError('Staged source mismatch: ' + name)
    else:
        identity = {'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                    'source_sha256': source_hashes}
    reporting.start_run(args.output_dir, {
        'study_id': 'cifar10-digital-l8-legacy-transfer-20260928-v1',
        'run_id': args.output_dir.name, 'arm_id': 'direct-sign-split-transfer',
        'seed': 0, 'evidence_class': 'cifar_fixed_checkpoint_transfer_diagnostic',
        'smoke': bool(args.limit), 'resolved_config': config,
        'transfer': {'encoding': 'symmetric_sign_split_plus_common_floor', 'head': args.head,
                     'bn': 'copy_all_affine_and_running_state_eval_only', 'optimizer_steps': 0},
        'checkpoint': {'path': str(args.checkpoint), 'sha256': checkpoint_hash,
                       'epoch': checkpoint['epoch']},
        'dataset': {'split': args.split, 'official_test_read': args.split == 'original-test',
                    'paper_facing': False, 'root': str(args.dataset_root)},
        'command': shlex.join([sys.executable, *sys.argv]),
        'source_sha256': source_hashes,
        'source_identity': identity,
        'runtime': dict(reporting.runtime_context(target=str(device)), pid=os.getpid()),
        'environment': {'torch': torch.__version__, 'cuda': torch.version.cuda},
        'maximum_seconds': args.maximum_seconds, 'batch_size': args.batch_size,
    })
    started = time.monotonic()
    try:
        reporting.atomic_write_json(args.output_dir / 'artifacts/transfer_audit.json', audit)
        if args.split == 'original-test':
            path = args.dataset_root / 'test_batch'
            with path.open('rb') as handle:
                batch = pickle.load(handle, encoding='latin1')
            images = batch['data'].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
            labels = np.asarray(batch['labels'], dtype=np.int64)
            indices = list(range(10000))
            if len(images) != 10000:
                raise ValueError('Unexpected original test size')
            data_hashes = {'test_batch': reporting.sha256_file(path)}
        else:
            images, labels = read_training_data(args.dataset_root)
            indices = make_split(labels, config['split']['seed'])['validation']
            data_hashes = {f'data_batch_{i}': reporting.sha256_file(args.dataset_root / f'data_batch_{i}')
                           for i in range(1, 6)}
        if args.limit:
            indices = indices[:args.limit]
        reporting.atomic_write_json(args.output_dir / 'artifacts/data_identity.json',
                                    {'sha256': data_hashes, 'indices': indices})
        batches = loader(Cohort(images, labels, indices, config), args.batch_size)
        names = ['digital_reference']
        if args.head in ('digital', 'both'):
            names.append('legacy_convolutions_digital_head')
        if args.head in ('native-analog-omit-bias', 'both'):
            names.append('legacy_native_head_bias_omitted')
        totals = {name: dict(loss=0., correct=0, agreement=0, squared_error=0.) for name in names}
        saved = {name: [] for name in names}
        truth, count, reference_squared = [], 0, 0.
        with torch.no_grad():
            for step, (x, y) in enumerate(batches, 1):
                if time.monotonic() - started > args.maximum_seconds:
                    raise TimeoutError('Replay budget exhausted')
                x, y = x.to(device), y.to(device)
                reference = digital(x)
                features = analog_features(analog, x, config['iterations'])
                outputs = {'digital_reference': reference}
                if 'legacy_convolutions_digital_head' in names:
                    outputs['legacy_convolutions_digital_head'] = digital.head(features)
                if 'legacy_native_head_bias_omitted' in names:
                    outputs['legacy_native_head_bias_omitted'] = analog.head(features)
                for name, logits in outputs.items():
                    if not torch.isfinite(logits).all():
                        raise FloatingPointError('Nonfinite logits: ' + name)
                    totals[name]['loss'] += float(F.cross_entropy(logits, y, reduction='sum'))
                    totals[name]['correct'] += int((logits.argmax(1) == y).sum())
                    totals[name]['agreement'] += int((logits.argmax(1) == reference.argmax(1)).sum())
                    totals[name]['squared_error'] += float((logits.double() - reference.double()).square().sum())
                    saved[name].append(logits.cpu())
                reference_squared += float(reference.double().square().sum())
                truth.append(y.cpu())
                count += len(y)
                analog.detach_state_()
                if step == 1 or step % 10 == 0 or count == len(indices):
                    progress = {'stage': 'evaluation', 'step': step, 'examples': count,
                                'expected_examples': len(indices), 'elapsed_seconds': time.monotonic() - started}
                    reporting.update_status_progress(args.output_dir, progress)
                    reporting.append_metric(args.output_dir / 'metrics.jsonl', progress)
                    print(json.dumps(progress), flush=True)
        after = analog.snapshot()
        for family in before:
            for name in before[family]:
                if not torch.equal(before[family][name], after[family][name]):
                    raise RuntimeError('Parameter/BN state mutated during replay: ' + name)
        if reporting.sha256_file(args.checkpoint) != checkpoint_hash:
            raise RuntimeError('Source checkpoint changed during replay')
        metrics = {name: {'accuracy': values['correct'] / count,
                          'cross_entropy': values['loss'] / count,
                          'prediction_agreement': values['agreement'] / count,
                          'logit_relative_l2': (values['squared_error'] / reference_squared) ** .5,
                          'correct': values['correct'], 'examples': count}
                   for name, values in totals.items()}
        for name, values in metrics.items():
            values['accuracy_delta_pp'] = 100 * (values['accuracy'] - metrics['digital_reference']['accuracy'])
            values['retains_within_0_2pp'] = values['accuracy_delta_pp'] >= -.2
            reporting.append_metric(args.output_dir / 'metrics.jsonl',
                                    dict(stage='summary', arm=name, evaluation_split=args.split, **values))
        np.savez_compressed(args.output_dir / 'artifacts/predictions.npz',
                            labels=torch.cat(truth).numpy(), indices=np.asarray(indices),
                            **{name: torch.cat(values).numpy() for name, values in saved.items()})
        reporting.complete_run(args.output_dir,
            terminal_metrics={'arms': metrics, 'elapsed_seconds': time.monotonic() - started,
                              'peak_memory_reserved_bytes': torch.cuda.max_memory_reserved(device) if device.type == 'cuda' else 0},
            completion={'criteria_met': count == len(indices), 'parameters_and_bn_unchanged': True,
                        'checkpoint_unchanged': True, 'paper_facing': False, 'optimizer_steps': 0})
        errors = reporting.validate_run(args.output_dir)
        if errors:
            raise RuntimeError(errors)
        print(json.dumps(metrics), flush=True)
    except BaseException as error:
        if not (args.output_dir / 'result.json').exists():
            reporting.fail_run(args.output_dir, error=error)
        raise


if __name__ == '__main__':
    main()
