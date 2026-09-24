"""Read-only sequential BN recalibration and saved-Adam counterfactual probes."""
from __future__ import annotations
import argparse
import copy
import gc
import hashlib
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
from experiments.train_cifar_l8_analog import (Cohort, check_gpu_resident_execution,
    deterministic_setup, loader, make_optimizer, read_training_data)
from labs.cifar_l8_analog import CifarL8Analog


class ChannelMoments:
    """Parallel Welford channel moments over examples and spatial positions."""
    def __init__(self):
        self.count = 0
        self.mean = self.m2 = None

    def add(self, x):
        x = x.detach().double().movedim(1, 0).flatten(1)
        n = x.shape[1]
        mean = x.mean(1)
        m2 = (x - mean[:, None]).square().sum(1)
        if self.count == 0:
            self.mean, self.m2, self.count = mean, m2, n
        else:
            delta = mean - self.mean
            total = self.count + n
            self.m2 += m2 + delta.square() * (self.count * n / total)
            self.mean += delta * (n / total)
            self.count = total

    def finish(self):
        if self.count < 2:
            raise ValueError('At least two scalar observations per channel required')
        return self.mean, self.m2 / (self.count - 1)


@torch.no_grad()
def set_probe_parameters(model, before, nominal, multiplier):
    # Special-case endpoints to preserve exact float32 nominal Adam step bytes.
    for name, p in model.trainable_tensors().items():
        p.copy_(before[name] if multiplier == 0 else nominal[name] if multiplier == 1
                else before[name] + multiplier * (nominal[name] - before[name]))
    unprojected = {n: p.clone() for n, p in model.trainable_tensors().items()}
    model.project_()
    return unprojected


def vector_summary(before, after, gradients, unprojected):
    rows = {}
    for name, p in before.items():
        d = (after[name] - p).double()
        g = gradients[name].double()
        dn, gn = float(d.norm()), float(g.norm())
        dot = float((d * g).sum())
        rows[name] = {'gradient_dot_update': dot,
            'gradient_update_cosine': dot / (dn * gn) if dn * gn else None,
            'update_l2': dn, 'parameter_l2': float(p.double().norm()),
            'relative_update_l2': dn / max(float(p.double().norm()), 1e-300),
            'projected_coordinate_fraction': float((after[name] != unprojected[name]).float().mean())}
    dot = sum(r['gradient_dot_update'] for r in rows.values())
    dn = sum(r['update_l2'] ** 2 for r in rows.values()) ** .5
    gn = sum(float(g.double().square().sum()) for g in gradients.values()) ** .5
    pn = sum(r['parameter_l2'] ** 2 for r in rows.values()) ** .5
    return {'parameters': rows, 'gradient_dot_update': dot,
        'gradient_update_cosine': dot / (dn * gn) if dn * gn else None,
        'relative_update_l2': dn / max(pn, 1e-300)}


@torch.no_grad()
def evaluate(model, batches, iterations, check, reference=None):
    model.eval(); ce = 0.; correct = count = 0; outputs = []
    for x, y in batches:
        check(); x, y = x.cuda(), y.cuda()
        logits = model(x, iterations)
        loss = F.cross_entropy(logits, y, reduction='sum')
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite probe loss')
        ce += float(loss); correct += int((logits.argmax(1) == y).sum()); count += len(y)
        outputs.append(logits.cpu()); model.detach_state_()
    logits = torch.cat(outputs)
    row = {'loss': ce / count, 'accuracy': correct / count, 'examples': count}
    if reference is not None:
        row['logit_relative_l2'] = float((logits.double() - reference.double()).norm()
                                         / reference.double().norm().clamp_min(1e-300))
    return row, logits


def recalibrate(model, batches, iterations, check, progress):
    model.eval(); records = []
    with torch.no_grad():
        for target, bridge in enumerate(model.bridges):
            stats = ChannelMoments()
            for batch, (x, _) in enumerate(batches):
                check(); x = x.cuda()
                for index in range(target + 1):
                    x = model.analog_blocks[index](x, num_iterations=iterations[index], reset=True)
                    scale = model.block_output_scales[index]
                    if scale != 1.:
                        x = x / scale
                    x = model.bridges[index][0](x)
                    if index == target:
                        stats.add(x)
                    else:
                        x = model.bridges[index][1](x)
                model.detach_state_()
                if batch % 16 == 0:
                    progress({'stage': 'bn_fit', 'block': target + 1, 'batch': batch + 1})
            mean, variance = stats.finish(); bn = bridge[1]
            records.append({'block': target + 1, 'scalar_count_per_channel': stats.count,
                'old_mean': bn.running_mean.cpu().tolist(), 'old_variance': bn.running_var.cpu().tolist(),
                'new_mean': mean.cpu().tolist(), 'new_variance': variance.cpu().tolist()})
            bn.running_mean.copy_(mean); bn.running_var.copy_(variance)
    return records


def bn_probe(args, model, checkpoint, config, images, labels, split):
    fit_indices = split['train'][:32 if args.smoke else 4096]
    val_indices = split['validation'][:32] if args.smoke else split['validation']
    batches = lambda ix: loader(Cohort(images, labels, ix, config), 16)
    before, _ = evaluate(model, batches(val_indices), config['iterations'], args.check)
    stats = recalibrate(model, batches(fit_indices), config['iterations'], args.check, args.progress)
    check_unchanged(model, checkpoint['model'], buffers=False)
    after, _ = evaluate(model, batches(val_indices), config['iterations'], args.check)
    result = {'before': before, 'after': after, 'accuracy_delta': after['accuracy']-before['accuracy'],
        'loss_delta': after['loss']-before['loss'], 'statistics': stats,
        'fit_indices': fit_indices, 'validation_indices': val_indices,
        'variance_estimator': 'pooled float64 Welford M2/(N-1), N=examples*height*width per channel',
        'fitting_labels_used': False, 'sequential_order': [1, 2, 3]}
    return result


def adam_probe(args, model, checkpoint, config, images, labels, split):
    indices = split['train'][:32 if args.smoke else 128]
    holdout_indices = split['train'][128:160 if args.smoke else 256]
    batches = list(loader(Cohort(images, labels, indices, config), 32))
    holdout = list(loader(Cohort(images, labels, holdout_indices, config), 32))
    rows = []
    for batch, (x, y) in enumerate(batches):
        args.check(); model.restore(checkpoint['model']); model.train()
        optimizer, _ = make_optimizer(model, config)
        optimizer.load_state_dict(copy.deepcopy(checkpoint['optimizer']))
        optimizer.zero_grad(set_to_none=True)
        if {g['name'] for g in optimizer.param_groups} != set(model.trainable_tensors()):
            raise RuntimeError('Optimizer does not cover every trainable parameter')
        before = {n: p.detach().clone() for n, p in model.trainable_tensors().items()}
        _, logits = model.bptt(x.cuda(), config['iterations'])
        loss = F.cross_entropy(logits, y.cuda(), label_smoothing=config['label_smoothing'])
        loss.backward()
        gradients = {n: p.grad.detach().clone() for n, p in model.trainable_tensors().items()}
        if not all(torch.isfinite(g).all() for g in gradients.values()):
            raise FloatingPointError('Nonfinite Adam gradient')
        optimizer.step()
        raw = {n: p.detach().clone() for n, p in model.trainable_tensors().items()}
        model.project_(); model.detach_state_()
        nominal = {n: p.detach().clone() for n, p in model.trainable_tensors().items()}
        nominal_summary = vector_summary(before, nominal, gradients, raw)
        references = {}
        for multiplier in (0., .5, 1., 2.):
            args.check(); model.restore(checkpoint['model'])
            raw_probe = set_probe_parameters(model, before, nominal, multiplier)
            current = {n: p.detach() for n, p in model.trainable_tensors().items()}
            if multiplier == 1 and not all(torch.equal(current[n], v) for n,v in nominal.items()):
                raise RuntimeError('Lambda1 differs from exact projected Adam update')
            row = {'stage': 'adam_probe', 'source': args.current_id, 'batch': batch,
                'multiplier': multiplier, 'training_gradient_loss': float(loss.detach()),
                'nominal_update': nominal_summary, 'update': vector_summary(before, current, gradients, raw_probe),
                'learning_rates': {g['name']: g['lr'] for g in optimizer.param_groups}}
            for name, probe in [('same_batch', [(x,y)]), ('holdout', holdout)]:
                measured, output = evaluate(model, probe, config['iterations'], args.check, references.get(name))
                if multiplier == 0:
                    references[name] = output; measured['logit_relative_l2'] = 0.
                row[name] = measured
            # Evaluation must leave original saved running buffers byte-identical.
            now = model.snapshot()['module']
            for name, value in checkpoint['model']['module'].items():
                if 'running_' in name or 'num_batches_tracked' in name:
                    if not torch.equal(now[name], value):
                        raise RuntimeError('Paired probe changed BN buffer: ' + name)
            reporting.append_metric(args.output_dir/'metrics.jsonl',row); rows.append(row)
            args.progress({'stage':'adam_probe','batch':batch+1,'multiplier':multiplier})
        del optimizer, gradients, before, nominal, raw, raw_probe, logits, loss
    return {'probes': rows, 'gradient_indices': indices, 'holdout_indices': holdout_indices,
        'direction_convention': 'exact saved-state Adam step, then bound projection; lambda interpolates from original toward projected endpoint, followed by bound reprojection; lambda1 exact endpoint',
        'bn_convention': 'training BN for gradient; original saved buffers fixed in eval mode for paired loss/logit evaluation'}


def select_checkpoint_records(records, checkpoint_ids=None):
    """Select an explicit recovery subset from the epoch/mode-eligible records."""
    available = {record['id'] for record in records}
    if len(available) != len(records):
        raise ValueError('Duplicate checkpoint IDs in input records')
    if checkpoint_ids is not None:
        requested = set(checkpoint_ids)
        if len(requested) != len(checkpoint_ids):
            raise ValueError('Duplicate requested checkpoint IDs')
        unavailable = requested - available
        if unavailable:
            raise ValueError('Checkpoint IDs unavailable for selected mode/epochs: '
                             + ', '.join(sorted(unavailable)))
        records = [record for record in records if record['id'] in requested]
        if {record['id'] for record in records} != requested:
            raise RuntimeError('Checkpoint selection coverage mismatch')
    if not records:
        raise ValueError('Checkpoint selection is empty')
    return records


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode', choices=['adam','bn'], required=True)
    p.add_argument('--study-root', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--dataset-root', type=Path, required=True)
    p.add_argument('--source-identity', type=Path, required=True)
    p.add_argument('--target', default='riri')
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--epochs', type=int, nargs='+', choices=[10, 30, 50], default=[30],
                   help='Saved checkpoint epochs for Adam probes (default: 30)')
    p.add_argument('--checkpoint-ids', nargs='+',
                   help='Only these checkpoint IDs within the selected mode/epochs')
    p.add_argument('--maximum-seconds', type=float, default=3000)
    args=p.parse_args(); start=time.monotonic()
    check_gpu_resident_execution('cuda',False); deterministic_setup(0)
    root=args.study_root.resolve(); args.output_dir=args.output_dir.resolve()
    if args.output_dir.exists(): raise FileExistsError(args.output_dir)
    identity=json.loads(args.source_identity.read_text()); repo=Path(__file__).resolve().parents[1]
    for name,digest in identity['source_sha256'].items():
        if reporting.sha256_file(repo/name)!=digest: raise RuntimeError('Source mismatch '+name)
    records=json.loads((root/'inputs/checkpoints.json').read_text())
    records=[r for r in records if r['epoch'] in (args.epochs if args.mode=='adam' else [10,50])]
    if args.mode=='bn':
        path=root/'cells/legacy_voltage_normalized/checkpoints/final_model.pt'
        records.append({'id':'legacy_voltage_normalized_e10','epoch':10,
            'checkpoint':str(path),'sha256':reporting.sha256_file(path)})
    records=select_checkpoint_records(records,args.checkpoint_ids)
    if args.smoke: records=records[:1]
    split=json.loads((root/'inputs/split.json').read_text())
    cohort_hash=hashlib.sha256(json.dumps(split['train'][:128]).encode()).hexdigest()
    if cohort_hash!='82962c61d8dff52bdc730ec1b5bc1ae0222d792d663da455b6fa8babf5778b00':
        raise RuntimeError('Unexpected replay cohort')
    reporting.start_run(args.output_dir, {'study_id':'cifar10-l8-bn-gain-followup-seed0-20260923-v1',
        'run_id':args.output_dir.name,'arm_id':args.mode,'seed':0,'smoke':args.smoke,
        'evidence_class':'cifar10_bn_adam_readonly_diagnostic','resolved_config':vars(args)|{'study_root':str(root),
            'output_dir':str(args.output_dir),'dataset_root':str(args.dataset_root),'source_identity':str(args.source_identity)},
        'source_identity':identity,'inputs':records,'cohort_sha256':cohort_hash,
        'command':shlex.join([sys.executable,*sys.argv]),
        'runtime':dict(reporting.runtime_context(target=args.target),pid=os.getpid()),
        'dataset':{'official_test_read':False,'fitting_split':'training','augmentation':False},
        'environment':{'torch':str(torch.__version__),'cuda':torch.version.cuda,
                       'gpu':torch.cuda.get_device_name(),'sharing':'Ben mumax workloads; timings include sharing'}})
    def check():
        if time.monotonic()-start>args.maximum_seconds: raise TimeoutError('Declared diagnostic budget exhausted')
    args.check=check
    args.progress=lambda x:reporting.update_status_progress(args.output_dir,{'checkpoint':args.current_id,**x})
    try:
        images,labels=read_training_data(args.dataset_root); completed=[]
        for record in records:
            check(); args.current_id=record['id']; print('START',record['id'],flush=True)
            path=root/record['checkpoint']; digest=reporting.sha256_file(path)
            if digest!=record['sha256']:raise RuntimeError('Checkpoint hash mismatch')
            checkpoint=torch.load(path,map_location='cpu',weights_only=False)
            config=checkpoint['config']; model=CifarL8Analog(config,'cuda'); model.restore(checkpoint['model'])
            measured=(adam_probe if args.mode=='adam' else bn_probe)(args,model,checkpoint,config,images,labels,split)
            model.restore(checkpoint['model']);check_unchanged(model,checkpoint['model'])
            if reporting.sha256_file(path)!=digest:raise RuntimeError('Checkpoint bytes changed')
            measured.update(source=record['id'],sha256=digest,original_checkpoint_unchanged=True,
                            elapsed_seconds=time.monotonic()-start)
            reporting.atomic_write_json(args.output_dir/'artifacts'/f"{record['id']}.json",measured)
            reporting.append_metric(args.output_dir/'metrics.jsonl',{'stage':'checkpoint_complete','source':record['id'],
                'elapsed_seconds':time.monotonic()-start,'original_checkpoint_unchanged':True,
                **({k:measured[k] for k in ('before','after','accuracy_delta','loss_delta')} if args.mode=='bn' else {})})
            completed.append(record['id']);print('COMPLETE_CHECKPOINT',record['id'],time.monotonic()-start,flush=True)
            del model,checkpoint;gc.collect();torch.cuda.empty_cache()
        reporting.complete_run(args.output_dir,terminal_metrics={'completed':completed,
            'elapsed_seconds':time.monotonic()-start,'peak_memory_reserved_bytes':torch.cuda.max_memory_reserved()},
            completion={'criteria_met':len(completed)==len(records),'official_test_read':False})
        errors=reporting.validate_run(args.output_dir)
        if errors:raise RuntimeError(errors)
        print('COMPLETE',args.mode,flush=True)
    except BaseException as error:
        if not (args.output_dir/'result.json').exists():reporting.fail_run(args.output_dir,error=error)
        raise

if __name__=='__main__':main()
