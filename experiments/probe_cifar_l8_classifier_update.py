"""Read-only CIFAR BPTT diagnostic isolating the classifier conductance Adam step."""
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

HEAD = 'head.weight'


def apply_adam_step(model, checkpoint, config, gradients, multiplier):
    model.restore(checkpoint['model'])
    optimizer, _ = make_optimizer(model, config)
    optimizer.load_state_dict(copy.deepcopy(checkpoint['optimizer']))
    params = model.trainable_tensors()
    if {g['name'] for g in optimizer.param_groups} != set(params):
        raise RuntimeError('Optimizer parameter coverage mismatch')
    matched = 0
    for group in optimizer.param_groups:
        if group['name'] == HEAD:
            group['lr'] *= multiplier
            matched += 1
    if matched != 1:
        raise RuntimeError('Expected exactly one classifier weight group')
    for name, parameter in params.items():
        parameter.grad = gradients[name].clone()
    optimizer.step()
    model.project_()
    model.detach_state_()
    if not all(torch.isfinite(p).all() for p in params.values()):
        raise FloatingPointError('Nonfinite counterfactual parameters')
    return {n: p.detach().clone() for n, p in params.items()}


@torch.no_grad()
def evaluate(model, batches, iterations, mode, check):
    model.train(mode == 'minibatch')
    total_loss = correct = count = 0
    outputs = []
    with batchnorm_minibatch_stats_only(model):
        for x, y in batches:
            check()
            x, y = x.cuda(), y.cuda()
            logits = model(x, iterations)
            loss = F.cross_entropy(logits, y, reduction='sum')
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite evaluation loss')
            total_loss += float(loss)
            correct += int((logits.argmax(1) == y).sum())
            count += len(y)
            outputs.append(logits.cpu())
            model.detach_state_()
    return {'loss': total_loss / count, 'accuracy': correct / count,
            'examples': count}, torch.cat(outputs)


def assert_buffers_unchanged(model, checkpoint):
    for name, value in model.named_buffers():
        if not torch.equal(value.cpu(), checkpoint['model']['module'][name]):
            raise RuntimeError('BN buffer mutated: ' + name)


def probe_record(args, record, images, labels, split, check):
    checkpoint_path = args.root / record['checkpoint']
    digest = reporting.sha256_file(checkpoint_path)
    if digest != record['sha256']:
        raise RuntimeError('Checkpoint hash mismatch: ' + record['id'])
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    config = checkpoint['config']
    if checkpoint['epoch'] != record['epoch']:
        raise RuntimeError('Checkpoint epoch mismatch')
    model = CifarL8Analog(config, 'cuda')
    gradient_indices = split['train'][:32 if args.smoke else 128]
    holdout_indices = split['train'][256:288 if args.smoke else 512]
    if set(gradient_indices) & set(holdout_indices):
        raise RuntimeError('Gradient and probe cohorts overlap')
    batches = list(loader(Cohort(images, labels, gradient_indices, config), 32))
    holdout = list(loader(Cohort(images, labels, holdout_indices, config), 32))
    rows = []
    for batch, (x, y) in enumerate(batches):
        check()
        model.restore(checkpoint['model'])
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
        reference = {}
        nominal = None
        for variant, multiplier in [('before', None), ('full', 1.), ('head_zero', 0.),
                                    ('head_half', .5), ('head_double', 2.), ('repeat_before', None)]:
            check()
            if multiplier is None:
                model.restore(checkpoint['model'])
                values = {n: p.detach() for n, p in model.trainable_tensors().items()}
            else:
                values = apply_adam_step(model, checkpoint, config, gradients, multiplier)
                if variant == 'full':
                    nominal = values
                elif not all(torch.equal(p, nominal[n]) for n, p in values.items() if n != HEAD):
                    raise RuntimeError('Non-head update differs from nominal step')
                if multiplier == 0 and not torch.equal(values[HEAD].cpu(), checkpoint['model']['conductances'][HEAD]):
                    raise RuntimeError('Zero head LR moved classifier weights')
            head_before = checkpoint['model']['conductances'][HEAD].to('cuda')
            row = {'stage': 'classifier_update', 'source': record['id'], 'scheme': record['scheme'],
                   'epoch': record['epoch'], 'batch': batch, 'variant': variant,
                   'head_lr_multiplier': multiplier, 'gradient_loss': gradient_loss,
                   'learning_rates': rates,
                   'head_relative_update': float((values[HEAD] - head_before).double().norm()
                                                / head_before.double().norm()),
                   'nonhead_update_matches_full': multiplier is not None,
                   'measurements': {}}
            for mode in ['minibatch', 'saved']:
                row['measurements'][mode] = {}
                for cohort, samples in [('gradient', [(x, y)]), ('holdout', holdout)]:
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
            del head_before, values
        del gradients, nominal
        print('DIRECTION_COMPLETE', record['id'], batch + 1, flush=True)
    model.restore(checkpoint['model'])
    check_unchanged(model, checkpoint['model'])
    if reporting.sha256_file(checkpoint_path) != digest:
        raise RuntimeError('Checkpoint file changed')
    result = {'source': record['id'], 'checkpoint_sha256': digest,
              'original_checkpoint_unchanged': True, 'gradient_indices': gradient_indices,
              'holdout_indices': holdout_indices, 'probes': rows}
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
    args = parser.parse_args()
    args.root = args.root.resolve(); args.output_dir = args.output_dir.resolve()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    check_gpu_resident_execution('cuda', False)
    deterministic_setup(0)
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
        'arm_id': 'classifier_update', 'seed': 0, 'smoke': args.smoke,
        'evidence_class': 'cifar10_classifier_update_readonly_diagnostic',
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
                               'elapsed_seconds': time.monotonic() - start},
                               completion={'criteria_met': True, 'official_test_read': False,
                                           'original_checkpoints_unchanged': True})
    except BaseException as exc:
        reporting.fail_run(args.output_dir, error=exc)
        raise


if __name__ == '__main__':
    main()
