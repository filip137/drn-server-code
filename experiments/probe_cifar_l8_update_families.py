"""Read-only CIFAR BPTT diagnostic isolating convolution, affine-BN and gain Adam updates."""
from __future__ import annotations
import argparse
import copy
import gc
import json
import os
from pathlib import Path
import shlex
import sys
import time

import torch
from torch.nn import functional as F
from experiments import reporting
from experiments.analyze_cifar_l8_mechanism import check_unchanged
from experiments.train_cifar_l8_analog import (
    Cohort, check_gpu_resident_execution, deterministic_setup, loader, make_optimizer,
    read_training_data,
)
from labs.cifar_l8_analog import CifarL8Analog, batchnorm_minibatch_stats_only

from itertools import combinations
from experiments.probe_cifar_l8_classifier_update import (
    HEAD, apply_adam_step, evaluate, assert_buffers_unchanged,
)

FAMILIES = ('conv', 'bn', 'gain')
VARIANTS = [('before', None)] + [
    ('head_' + ('_'.join(subset) if subset else 'only'), subset)
    for size in range(4) for subset in combinations(FAMILIES, size)
] + [('repeat_before', None)]
CONV_VARIANTS = [('before', None), ('conv_zero', 0.), ('conv_1_32', 1/32),
                 ('conv_1_8', 1/8), ('conv_1_4', 1/4), ('conv_half', .5),
                 ('conv_full', 1.), ('repeat_before', None)]


def apply_conv_adam_step(model, checkpoint, config, gradients, families, multiplier):
    model.restore(checkpoint['model'])
    optimizer, _ = make_optimizer(model, config)
    optimizer.load_state_dict(copy.deepcopy(checkpoint['optimizer']))
    params = model.trainable_tensors()
    if {g['name'] for g in optimizer.param_groups} != set(params):
        raise RuntimeError('Optimizer parameter coverage mismatch')
    for group in optimizer.param_groups:
        if families[group['name']] == 'conv':
            group['lr'] *= multiplier
    for name, parameter in params.items():
        parameter.grad = gradients[name].clone()
    optimizer.step()
    model.project_()
    model.detach_state_()
    if not all(torch.isfinite(p).all() for p in params.values()):
        raise FloatingPointError('Nonfinite convolution-LR counterfactual')


def partition(model):
    params = model.trainable_tensors()
    conv = set(dict(model.named_conductances())) - {HEAD}
    bn = {f'{name}.{pn}' for name, module in model.named_modules()
          if isinstance(module, torch.nn.modules.batchnorm._BatchNorm)
          for pn, p in module.named_parameters(recurse=False) if p.requires_grad}
    gain = {n for n in params if n.endswith('_input_gain_raw')}
    groups = {'conv': conv, 'bn': bn, 'gain': gain, 'head': {HEAD}}
    flattened = [n for names in groups.values() for n in names]
    if len(flattened) != len(set(flattened)) or set(flattened) != set(params):
        raise RuntimeError('Parameter families are not an exhaustive disjoint partition')
    if any(not names for names in groups.values()):
        raise RuntimeError('Empty parameter family')
    return {n: group for group, names in groups.items() for n in names}


def probe_record(args, record, images, labels, split, check):
    path = args.root / record['checkpoint']
    if reporting.sha256_file(path) != record['sha256']:
        raise RuntimeError('Checkpoint hash mismatch')
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    if checkpoint['epoch'] != 30 or record['epoch'] != 30:
        raise RuntimeError('Expected original epoch-30 checkpoint')
    config = checkpoint['config']
    model = CifarL8Analog(config, 'cuda')
    families = partition(model)
    gradient_indices = split['train'][:args.batch_size * (1 if args.smoke else 4)]
    holdout_count = args.batch_size if args.smoke else args.holdout_count
    holdout_indices = split['train'][args.holdout_start:args.holdout_start + holdout_count]
    if len(holdout_indices) != holdout_count:
        raise RuntimeError('Incomplete separate training cohort')
    if set(gradient_indices) & set(holdout_indices):
        raise RuntimeError('Overlapping gradient and probe cohorts')
    batches = list(loader(Cohort(images, labels, gradient_indices, config), args.batch_size))
    holdout = list(loader(Cohort(images, labels, holdout_indices, config), args.batch_size))
    rows = []
    for batch, (x, y) in enumerate(batches):
        check()
        model.restore(checkpoint['model'])
        original = {n: p.detach().clone() for n, p in model.trainable_tensors().items()}
        model.train()
        optimizer, _ = make_optimizer(model, config)
        optimizer.load_state_dict(copy.deepcopy(checkpoint['optimizer']))
        optimizer.zero_grad(set_to_none=True)
        free, tracked = model.bptt(x.cuda(), config['iterations'])
        loss = F.cross_entropy(tracked, y.cuda(), label_smoothing=config['label_smoothing'])
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite gradient loss')
        loss.backward()
        gradients = {n: p.grad.detach().clone() for n, p in model.trainable_tensors().items()}
        if not all(torch.isfinite(g).all() for g in gradients.values()):
            raise FloatingPointError('Nonfinite gradient')
        gradient_loss = float(loss.detach())
        rates = {g['name']: g['lr'] for g in optimizer.param_groups}
        del free, tracked, loss, optimizer
        model.detach_state_()
        full = apply_adam_step(model, checkpoint, config, gradients, 1.)
        if args.probe_kind == 'families':
            del gradients
        reference = {}
        variants = VARIANTS if args.probe_kind == 'families' else CONV_VARIANTS
        for variant, enabled in variants:
            check()
            model.restore(checkpoint['model'])
            if args.probe_kind == 'conv-lr':
                if enabled is not None:
                    apply_conv_adam_step(model, checkpoint, config, gradients, families, enabled)
                for name, parameter in model.trainable_tensors().items():
                    expected = (original[name] if enabled is None or (enabled == 0 and families[name] == 'conv')
                                else full[name] if families[name] != 'conv' or enabled == 1 else None)
                    if expected is not None and not torch.equal(parameter, expected):
                        raise RuntimeError('Convolution-LR tensor identity mismatch: ' + name)
                params = model.trainable_tensors()
                names = [n for n in params if families[n] == 'conv']
                relative_update = float(torch.sqrt(sum((params[n].detach() - original[n]).double().square().sum() for n in names)
                                       / sum(original[n].double().square().sum() for n in names)))
                details = {'conv_lr_multiplier': enabled, 'conv_relative_update': relative_update}
            else:
                enabled_groups = set() if enabled is None else set(enabled) | {'head'}
                with torch.no_grad():
                    for name, parameter in model.trainable_tensors().items():
                        expected = full[name] if families[name] in enabled_groups else original[name]
                        parameter.copy_(expected)
                        if not torch.equal(parameter, expected):
                            raise RuntimeError('Counterfactual tensor identity mismatch: ' + name)
                details = {'enabled_families': sorted(enabled_groups)}
            row = {'stage': 'update_family', 'source': record['id'], 'scheme': record['scheme'],
                   'epoch': 30, 'batch': batch, 'batch_size': args.batch_size, 'variant': variant,
                   **details, 'gradient_loss': gradient_loss,
                   'learning_rates': rates, 'tensor_identity_passed': True, 'measurements': {}}
            for mode in ['minibatch', 'saved']:
                row['measurements'][mode] = {}
                for cohort, samples in [('gradient', [(x, y)]), ('holdout', holdout)]:
                    if cohort == 'holdout' and args.holdout_count > 256:
                        block_batches = 256 // args.batch_size
                        blocks, outputs = [], []
                        for start in range(0, len(samples), block_batches):
                            block, output = evaluate(model, samples[start:start + block_batches],
                                                     config['iterations'], mode, check)
                            blocks.append(block)
                            outputs.append(output)
                        count = sum(block['examples'] for block in blocks)
                        measured = {key: sum(block[key] * block['examples'] for block in blocks) / count
                                    for key in ['loss', 'accuracy']}
                        measured.update(examples=count, blocks=blocks)
                        logits = torch.cat(outputs)
                    else:
                        measured, logits = evaluate(model, samples, config['iterations'], mode, check)
                    key = (mode, cohort)
                    if variant == 'before':
                        reference[key] = (measured['loss'], logits)
                    before_loss, before_logits = reference[key]
                    measured['delta_loss'] = measured['loss'] - before_loss
                    measured['logit_relative_l2'] = float((logits.double() - before_logits.double()).norm()
                                                        / before_logits.double().norm().clamp_min(1e-30))
                    row['measurements'][mode][cohort] = measured
                assert_buffers_unchanged(model, checkpoint)
            reporting.append_metric(args.output_dir / 'metrics.jsonl', row)
            reporting.update_status_progress(args.output_dir, {'source': record['id'],
                'batch': batch + 1, 'batches': len(batches), 'variant': variant})
            rows.append(row)
        del original, full
        if args.probe_kind == 'conv-lr':
            del gradients
        print('DIRECTION_COMPLETE', record['id'], batch + 1, flush=True)
    model.restore(checkpoint['model'])
    check_unchanged(model, checkpoint['model'])
    if reporting.sha256_file(path) != record['sha256']:
        raise RuntimeError('Checkpoint file changed')
    result = {'source': record['id'], 'checkpoint_sha256': record['sha256'],
              'original_checkpoint_unchanged': True, 'parameter_families': families,
              'gradient_indices': gradient_indices, 'holdout_indices': holdout_indices, 'probes': rows}
    del model, checkpoint
    gc.collect(); torch.cuda.empty_cache()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--dataset-root', type=Path, required=True)
    parser.add_argument('--target', required=True)
    parser.add_argument('--maximum-seconds', type=float, default=1500)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--memory-fraction', type=float, default=0.70)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--probe-kind', choices=['families', 'conv-lr'], default='families')
    parser.add_argument('--holdout-start', type=int, default=256)
    parser.add_argument('--holdout-count', type=int, default=256,
                        help='Separate training examples; multiples of 256 retain per-cohort metrics')
    args = parser.parse_args()
    if args.batch_size <= 0 or args.batch_size > 32 or 256 % args.batch_size:
        parser.error('batch-size must be a positive divisor of 256 no larger than 32')
    if args.holdout_start < 0:
        parser.error('holdout-start must be nonnegative')
    if args.holdout_count <= 0 or args.holdout_count % 256:
        parser.error('holdout-count must be a positive multiple of 256')
    args.root = args.root.resolve(); args.output_dir = args.output_dir.resolve()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    check_gpu_resident_execution('cuda', False)
    deterministic_setup(0)
    torch.cuda.set_per_process_memory_fraction(args.memory_fraction)
    torch.cuda.reset_peak_memory_stats()
    source = Path(__file__).resolve().parents[1]
    identity = json.loads((source / 'source_identity.json').read_text())
    for name, digest in identity['source_sha256'].items():
        if reporting.sha256_file(source / name) != digest:
            raise RuntimeError('Source mismatch: ' + name)
    for name, digest in json.loads((source / 'training-data-hashes.json').read_text()).items():
        if reporting.sha256_file(args.dataset_root / name) != digest:
            raise RuntimeError('Training data mismatch: ' + name)
    split = json.loads((args.root / 'inputs/split.json').read_text())
    records = json.loads((args.root / 'inputs/checkpoints.json').read_text())
    if args.smoke:
        records = records[:1]
    reporting.start_run(args.output_dir, {'study_id': args.root.name, 'run_id': args.output_dir.name,
        'arm_id': args.probe_kind, 'seed': 0, 'smoke': args.smoke,
        'evidence_class': 'cifar10_update_family_readonly_diagnostic',
        'resolved_config': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        'source_identity': identity, 'inputs': records, 'split_sha256': reporting.sha256_file(args.root / 'inputs/split.json'),
        'dataset': {'official_test_read': False, 'gradient_split': 'training', 'probe_split': 'training', 'augmentation': False},
        'command': shlex.join([sys.executable, *sys.argv]),
        'runtime': dict(reporting.runtime_context(target=args.target), pid=os.getpid()),
        'environment': {'torch': str(torch.__version__), 'cuda': torch.version.cuda,
                        'gpu': torch.cuda.get_device_name()}})
    start = time.monotonic()
    def check():
        if time.monotonic() - start > args.maximum_seconds:
            raise TimeoutError('Diagnostic allowance exhausted')
    try:
        images, labels = read_training_data(args.dataset_root)
        complete = []
        for record in records:
            check()
            measured = probe_record(args, record, images, labels, split, check)
            reporting.atomic_write_json(args.output_dir / 'artifacts' / (record['id'] + '.json'), measured)
            complete.append(record['id'])
        reporting.complete_run(args.output_dir, terminal_metrics={'completed': complete,
                               'directions': len(complete) * (1 if args.smoke else 4),
                               'elapsed_seconds': time.monotonic() - start,
                               'peak_allocated_mib': torch.cuda.max_memory_allocated() / 2**20,
                               'peak_reserved_mib': torch.cuda.max_memory_reserved() / 2**20},
                               completion={'criteria_met': True, 'official_test_read': False,
                                           'original_checkpoints_unchanged': True})
    except BaseException as exc:
        reporting.fail_run(args.output_dir, error=exc)
        raise


if __name__ == '__main__':
    main()
