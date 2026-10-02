"""Matched CIFAR branches differing only in convolution or dense-readout LR."""
from __future__ import annotations
import argparse
import copy
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import shlex
import sys
import time

import numpy as np
import torch
from torch.nn import functional as F
from experiments import reporting
from experiments.analyze_cifar_l8_mechanism import check_unchanged
from experiments.train_cifar_l8_analog import (
    Cohort, check_gpu_resident_execution, deterministic_setup, evaluate, loader,
    make_optimizer, read_training_data, restore_checkpoint, save_checkpoint,
)
from labs.cifar_l8_analog import CifarL8Analog


def scale_parameter_rates(optimizer, scheduler, conv_names, multiplier):
    """Change LR coordinates only, retaining Adam moments and schedule phase."""
    found = set()
    for i, group in enumerate(optimizer.param_groups):
        if group['name'] in conv_names:
            found.add(group['name'])
            group['lr'] *= multiplier
            group['initial_lr'] *= multiplier
            scheduler.base_lrs[i] *= multiplier
            scheduler._last_lr[i] *= multiplier
    if found != set(conv_names):
        raise ValueError('Incomplete selected optimizer-group coverage')
    if [g['lr'] for g in optimizer.param_groups] != scheduler.get_last_lr():
        raise ValueError('Optimizer/scheduler LR mismatch')


# Preserve the original convolution-screen API.
scale_convolution_rates = scale_parameter_rates

def measure(model, cohorts, config, deadline):
    """Saved-BN evaluation that does not advance training RNG or BN buffers."""
    rng = (torch.get_rng_state(), np.random.get_state(), random.getstate(),
           torch.cuda.get_rng_state_all())
    buffers = {n: b.detach().clone() for n, b in model.named_buffers()}
    was_training = model.training
    try:
        out = {name: evaluate(model, batches, config['iterations'], 'cuda', deadline)
               for name, batches in cohorts.items()}
        for name, value in model.named_buffers():
            if not torch.equal(value, buffers[name]):
                raise RuntimeError('Evaluation mutated BN buffer: ' + name)
        return out
    finally:
        model.detach_state_()
        model.train(was_training)
        torch.set_rng_state(rng[0]); np.random.set_state(rng[1]); random.setstate(rng[2])
        torch.cuda.set_rng_state_all(rng[3])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--dataset-root', type=Path, required=True)
    p.add_argument('--target', required=True)
    p.add_argument('--maximum-seconds', type=float, default=5100)
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--parameter-family', choices=['conv', 'head'], default='conv')
    p.add_argument('--checkpoint', default='legacy_e30.pt')
    p.add_argument('--multipliers', type=float, nargs='+', default=[1., 3.])
    p.add_argument('--updates', type=int, default=1407)
    args = p.parse_args()
    root, run = args.root.resolve(), args.output_dir.resolve()
    assert args.multipliers[0] == 1. and len(set(args.multipliers)) == len(args.multipliers)
    assert all(np.isfinite(m) and m > 0 for m in args.multipliers)
    assert 1 <= args.updates <= 1407
    arms = [('control' if m == 1 else f'{args.parameter_family}{m:.8g}x', m) for m in args.multipliers]
    if run.exists():
        raise FileExistsError(run)
    check_gpu_resident_execution('cuda', False)
    deterministic_setup(0)
    torch.cuda.set_per_process_memory_fraction(.50)
    torch.cuda.reset_peak_memory_stats()
    source = Path(__file__).resolve().parents[1]
    identity = json.loads((source / 'source_identity.json').read_text())
    for name, sha in identity['source_sha256'].items():
        if reporting.sha256_file(source / name) != sha:
            raise RuntimeError('Source identity mismatch: ' + name)
    inputs = json.loads((root / 'inputs/identity.json').read_text())
    for name, sha in inputs['sha256'].items():
        if reporting.sha256_file(root / 'inputs' / name) != sha:
            raise RuntimeError('Input identity mismatch: ' + name)
    for name, sha in json.loads((source / 'training-data-hashes.json').read_text()).items():
        if reporting.sha256_file(args.dataset_root / name) != sha:
            raise RuntimeError('Training data identity mismatch: ' + name)
    checkpoint = torch.load(root / 'inputs' / args.checkpoint, map_location='cpu', weights_only=False)
    config = copy.deepcopy(checkpoint['config'])
    start_epoch = checkpoint['epoch']; epoch = start_epoch + 1
    config['epochs'] = epoch
    assert config['batch_size'] == 32
    parent_steps = start_epoch * 1407
    split = json.loads((root / 'inputs/split.json').read_text())
    cohort_indices = json.loads((root / 'inputs/cohorts.json').read_text())
    assert set(cohort_indices['train']) <= set(split['train'])
    assert set(cohort_indices['validation']) <= set(split['validation'])
    assert not set(split['train']) & set(split['validation'])
    permutation = torch.randperm(len(split['train']), generator=torch.Generator().manual_seed(config['seed'] + 1000000 * epoch)).tolist()
    indices = [split['train'][i] for i in permutation][:args.updates * 32]
    if args.smoke:
        indices = indices[:64]
        cohort_indices = {k:v[:16] for k,v in cohort_indices.items()}
    markers = [0, 2] if args.smoke else [0] + sorted({min(100,args.updates), min(400,args.updates), args.updates})
    full_epoch = not args.smoke and len(indices) == len(split['train'])
    order_hash = hashlib.sha256(json.dumps(indices).encode()).hexdigest()
    reporting.start_run(run, {'study_id': root.name, 'run_id': run.name, 'arm_id': 'paired_' + args.parameter_family + '_lr',
        'seed': 0, 'smoke': args.smoke, 'evidence_class': 'cifar_one_epoch_optimizer_screen',
        'resolved_config': config, 'intervention': {args.parameter_family + '_lr_multipliers': args.multipliers, 'epoch': epoch,
            'parameter_family': args.parameter_family, 'arms': dict(arms),
            'evaluation_updates': markers, 'evaluation_bn': 'saved', 'evaluation_batch_size': 16},
        'source_identity': identity, 'inputs': inputs, 'cohort_indices': cohort_indices,
        'training_order_sha256': order_hash,
        'dataset': {'official_test_read': False, 'training_examples': len(indices),
                    'evaluation_examples': {k:len(v) for k,v in cohort_indices.items()}},
        'runtime': dict(reporting.runtime_context(target=args.target), pid=os.getpid()),
        'environment': {'torch':str(torch.__version__), 'cuda':torch.version.cuda, 'gpu':torch.cuda.get_device_name()},
        'command': shlex.join([sys.executable, *sys.argv])})
    started = time.monotonic(); deadline = started + args.maximum_seconds
    try:
        images, labels = read_training_data(args.dataset_root)
        cohorts = {name:list(loader(Cohort(images, labels, ids, config), 16)) for name, ids in cohort_indices.items()}
        initial_measurement = None
        for arm, multiplier in arms:
            c = copy.deepcopy(config)
            model = CifarL8Analog(c, 'cuda')
            optimizer, scheduler = make_optimizer(model, c)
            restored = restore_checkpoint(copy.deepcopy(checkpoint), model, optimizer, scheduler, c)
            assert restored == start_epoch and {int(v['step']) for v in optimizer.state.values()} == {parent_steps}
            check_unchanged(model, checkpoint['model'])
            conv_names = (set(dict(model.named_conductances())) - {'head.weight'}
                          if args.parameter_family == 'conv' else {'head.weight'})
            old_rates = {g['name']:g['lr'] for g in optimizer.param_groups}
            scale_parameter_rates(optimizer, scheduler, conv_names, multiplier)
            if args.parameter_family == 'conv':
                c['optimizer']['conv_learning_rates'] = [[v * multiplier for v in block] for block in c['optimizer']['conv_learning_rates']]
            else:
                c['optimizer']['head_learning_rate'] *= multiplier
            assert all(g['lr'] == old_rates[g['name']] * (multiplier if g['name'] in conv_names else 1.) for g in optimizer.param_groups)
            model.train()
            initial_path = run / 'checkpoints' / f'{arm}_initial_model.pt'
            initial_path.parent.mkdir(exist_ok=True)
            save_checkpoint(initial_path, model, optimizer, scheduler, start_epoch, c)
            totals = {'loss': 0., 'correct': 0, 'examples': 0}
            batches = loader(Cohort(images, labels, indices, c, epoch, True), 32, c['num_workers'], True)

            def observation(step):
                reporting.update_status_progress(run, {'arm':arm, 'stage':'evaluation', 'updates':step, 'total_updates':markers[-1]})
                values = measure(model, cohorts, c, deadline)
                for parameter in model.trainable_tensors().values():
                    if not torch.isfinite(parameter).all():
                        raise FloatingPointError('Nonfinite parameter')
                row = {'stage':'evaluation', 'arm':arm, args.parameter_family + '_lr_multiplier':multiplier,
                       'updates':step, 'adam_steps':parent_steps+step, 'measurements':values,
                       'learning_rates':{g['name']:g['lr'] for g in optimizer.param_groups},
                       'elapsed_seconds':time.monotonic()-started, 'training_order_sha256':order_hash}
                reporting.append_metric(run/'metrics.jsonl', row)
                print('EVALUATION', arm, step, json.dumps(values), flush=True)
                return values

            before = observation(0)
            if initial_measurement is None:
                initial_measurement = before
            elif before != initial_measurement:
                raise RuntimeError('Starting cohort measurements differ between matched branches')
            for step, (x, y) in enumerate(batches, 1):
                if time.monotonic() > deadline:
                    raise TimeoutError('One-epoch screen allowance exhausted')
                x, y = x.cuda(), y.cuda()
                model.train(); optimizer.zero_grad(set_to_none=True)
                free, tracked = model.bptt(x, c['iterations'])
                loss = F.cross_entropy(tracked, y, label_smoothing=c['label_smoothing'])
                if not torch.isfinite(loss):
                    raise FloatingPointError('Nonfinite training loss')
                loss.backward()
                for name, parameter in model.trainable_tensors().items():
                    if parameter.grad is None or not torch.isfinite(parameter.grad).all():
                        raise FloatingPointError('Nonfinite/missing gradient: ' + name)
                optimizer.step(); model.project_(); model.detach_state_()
                totals['loss'] += float(loss.detach()) * len(y)
                totals['correct'] += int((free.argmax(1) == y).sum())
                totals['examples'] += len(y)
                del free, tracked, loss, x, y
                if step == 1 or step % 100 == 0:
                    reporting.update_status_progress(run, {'arm':arm, 'stage':'training', 'updates':step, 'total_updates':markers[-1]})
                    print('TRAINING', arm, step, flush=True)
                if step in markers:
                    observation(step)
                    if step != markers[-1]:
                        # These milestones preserve partial progress, but only the final
                        # complete-epoch checkpoint is accepted by native epoch resume.
                        save_checkpoint(run/'checkpoints'/f'{arm}_update{step}.pt', model, optimizer, scheduler, start_epoch, c)
                        reporting.atomic_write_json(run/'artifacts'/f'{arm}_update{step}.json',
                            {'completed_epoch':start_epoch,'partial_epoch':epoch,'updates_in_epoch':step,'native_epoch_resume_safe':False})
            assert step == markers[-1] and totals['examples'] == len(indices)
            assert {int(v['step']) for v in optimizer.state.values()} == {parent_steps + step}
            if full_epoch:
                scheduler.step()
            save_checkpoint(run/'checkpoints'/f'{arm}_final_model.pt', model, optimizer, scheduler, epoch if full_epoch else start_epoch, c)
            reporting.atomic_write_json(run/'artifacts'/f'{arm}_training.json',
                {'updates':step, 'train_loss':totals['loss']/totals['examples'],
                 'train_accuracy':totals['correct']/totals['examples'], 'train_examples':totals['examples'],
                 args.parameter_family + '_lr_multiplier':multiplier, 'final_checkpoint_epoch':epoch if full_epoch else start_epoch,
                 'native_epoch_resume_safe':full_epoch, 'training_order_sha256':order_hash})
            del batches, model, optimizer, scheduler
            gc.collect(); torch.cuda.empty_cache()
        assert reporting.sha256_file(root/'inputs'/args.checkpoint) == inputs['sha256'][args.checkpoint]
        reporting.complete_run(run, terminal_metrics={'completed_arms':[arm for arm, _ in arms],
            'updates_per_arm':markers[-1], 'elapsed_seconds':time.monotonic()-started,
            'peak_allocated_mib':torch.cuda.max_memory_allocated()/2**20,
            'peak_reserved_mib':torch.cuda.max_memory_reserved()/2**20},
            completion={'criteria_met':True,'official_test_read':False,'original_checkpoint_unchanged':True,
                        'full_epoch_per_arm':full_epoch})
    except BaseException as exc:
        reporting.fail_run(run, error=exc)
        raise


if __name__ == '__main__':
    main()
