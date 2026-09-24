"""Direct CIFAR L8 analog training, convergence checks, and batch-size comparison."""
from __future__ import annotations
import argparse
import copy
from contextlib import contextmanager, nullcontext
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import pickle
import random
import shlex
import shutil
import signal
import sys
import time

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as TF

from experiments import reporting
from labs.cifar_l8_analog import CifarL8Analog, boundary_normalization_settings


class PlannedPause(RuntimeError):
    """The overnight segment ended; the scientific epoch target is unchanged."""


def parse_stop_before(value):
    timestamp = datetime.fromisoformat(value.replace('Z', '+00:00'))
    if timestamp.tzinfo is None:
        raise argparse.ArgumentTypeError('stop-before requires an explicit timezone')
    return timestamp.astimezone(timezone.utc)


def epoch_admission_seconds(durations, initial_seconds, reserve_seconds):
    """Keep checkpoint/shutdown headroom and tolerate recent throughput variation."""
    estimate = max(durations[-3:]) if durations else initial_seconds
    return 1.25 * estimate + reserve_seconds


def check_boundary_compatibility(previous, current):
    if boundary_normalization_settings(previous) != boundary_normalization_settings(current):
        raise ValueError('Boundary normalization config mismatch')
    if previous.get('conv_input_gain_mode','softplus') != current.get('conv_input_gain_mode','softplus'):
        raise ValueError('Convolutional input gain mode mismatch')
    if previous['input_gain_init'] != current['input_gain_init']:
        raise ValueError('Input gain initialization mismatch')


def tensor_statistics(values):
    values = values.detach()
    return {'mean': float(values.mean()), 'rms': float(values.square().mean().sqrt()),
            'std': float(values.std(unbiased=False)), 'abs_max': float(values.abs().max())}


@contextmanager
def capture_boundary_statistics(model):
    """Last (tracked) pass statistics; hooks never change activations or BN state."""
    records, handles = {}, []
    for i, bridge in enumerate(model.bridges):
        for label, module in [('pooled_voltage', bridge[0]), ('boundary_output', bridge)]:
            def hook(module, inputs, output, key=f'block_{i}/{label}'):
                records[key] = tensor_statistics(output)
            handles.append(module.register_forward_hook(hook))
    try:
        yield records
    finally:
        for handle in handles:
            handle.remove()


def deterministic_setup(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False


def check_gpu_resident_execution(device, offload):
    if torch.device(device).type != 'cuda' or offload:
        raise ValueError('GPU-resident training requires CUDA and forbids CPU offloading')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; CPU fallback is forbidden')


def read_training_data(root):
    # Deliberately do not instantiate torchvision CIFAR10: its integrity check
    # hashes the test file even when train=True. Only the five training files
    # enter this experiment, including all splits and audits.
    arrays, labels = [], []
    for i in range(1, 6):
        with (root / f'data_batch_{i}').open('rb') as f:
            batch = pickle.load(f, encoding='latin1')
        arrays.append(batch['data']); labels.extend(batch['labels'])
    images = np.concatenate(arrays).reshape(-1,3,32,32).transpose(0,2,3,1)
    labels = np.asarray(labels, dtype=np.int64)
    if images.shape != (50000,32,32,3) or labels.shape != (50000,):
        raise ValueError('Unexpected training data geometry')
    return images, labels


def make_split(labels, seed):
    rng = np.random.default_rng(seed)
    train, validation = [], []
    for label in range(10):
        indices = np.flatnonzero(labels == label)
        if len(indices) != 5000:
            raise ValueError('Expected 5000 training examples per class')
        rng.shuffle(indices)
        validation.extend(indices[:500].tolist()); train.extend(indices[500:].tolist())
    return {'train':sorted(train), 'validation':sorted(validation)}


class Cohort(Dataset):
    def __init__(self, images, labels, indices, config, epoch=0, augment=False):
        self.images,self.labels,self.indices=images,labels,indices
        self.config,self.epoch,self.augment=config,epoch,augment

    def __len__(self): return len(self.indices)

    def __getitem__(self, position):
        index=int(self.indices[position]); image=Image.fromarray(self.images[index])
        if self.augment:
            rng=np.random.default_rng(np.random.SeedSequence([self.config['seed'],self.epoch,index]))
            if rng.random()<self.config['augmentation']['flip_probability']:
                image=TF.hflip(image)
            pad=self.config['augmentation']['crop_padding']
            image=TF.pad(image,pad,padding_mode=self.config['augmentation']['padding_mode'])
            top,left=rng.integers(0,2*pad+1,size=2)
            image=TF.crop(image,int(top),int(left),32,32)
        norm=self.config['normalization']
        return TF.normalize(TF.to_tensor(image),norm['mean'],norm['std']),int(self.labels[index])


def loader(dataset, batch, workers=0, cuda=False):
    return DataLoader(dataset,batch_size=batch,shuffle=False,drop_last=False,
                      num_workers=workers,pin_memory=cuda)


def make_optimizer(model,c):
    opt=c['optimizer']
    optimizer=torch.optim.Adam(model.optimizer_groups(opt),betas=tuple(opt['betas']),eps=opt['eps'],
                               weight_decay=opt['weight_decay'],foreach=False,fused=False)
    sched=c['scheduler']
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda epoch:
        sched['final_lr_ratio']+(1-sched['final_lr_ratio'])*.5*
        (1+math.cos(math.pi*min(epoch/sched['horizon_epochs'],1))))
    return optimizer,scheduler


def save_checkpoint(path,model,optimizer,scheduler,epoch,config):
    temp=path.with_suffix('.tmp')
    torch.save({'model':model.snapshot(),'optimizer':optimizer.state_dict(),
                'scheduler':scheduler.state_dict(),'epoch':epoch,'config':config,
                'torch_rng':torch.get_rng_state(),'numpy_rng':np.random.get_state(),
                'python_rng':random.getstate(),
                'cuda_rng':torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None},temp)
    os.replace(temp,path)


def restore_checkpoint(checkpoint, model, optimizer, scheduler, config):
    """Resume an epoch boundary without changing the scientific contract."""
    allowed = {'study_id', 'arm_id', 'evidence_class', 'epochs',
               'maximum_runtime_hours', 'continuation'}
    previous = checkpoint['config']
    check_boundary_compatibility(previous, config)
    for key in set(previous) | set(config):
        if key not in allowed and previous.get(key) != config.get(key):
            raise ValueError(f'Resume config mismatch: {key}')
    epoch = checkpoint['epoch']
    if not isinstance(epoch, int) or not 0 < epoch < config['epochs']:
        raise ValueError('Resume requires a completed epoch below the target epoch')
    if checkpoint['scheduler']['last_epoch'] != epoch:
        raise ValueError('Checkpoint scheduler is not at the completed epoch boundary')
    model.restore(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    scheduler.load_state_dict(checkpoint['scheduler'])
    if [g['lr'] for g in optimizer.param_groups] != scheduler.get_last_lr():
        raise ValueError('Restored optimizer and scheduler learning rates disagree')
    torch.set_rng_state(checkpoint['torch_rng'].cpu())
    np.random.set_state(checkpoint['numpy_rng'])
    random.setstate(checkpoint['python_rng'])
    if checkpoint.get('cuda_rng') is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([state.cpu() for state in checkpoint['cuda_rng']])
    return epoch


def gradient_norms(model):
    result={}
    for name,p in model.trainable_tensors().items():
        if p.grad is None or not torch.isfinite(p.grad).all():
            raise FloatingPointError(f'Missing/nonfinite gradient: {name}')
        result[name]=float(p.grad.norm())
        if result[name]<=0:
            raise FloatingPointError(f'Zero gradient: {name}')
    return result


def evaluate(model,batches,iterations,device,deadline):
    model.eval(); totals=torch.zeros(2,device=device,dtype=torch.float64);count=0
    with torch.no_grad():
        for x,y in batches:
            if time.monotonic()>deadline: raise TimeoutError('Runtime cap exhausted')
            x,y=x.to(device),y.to(device)
            logits=model(x,iterations); loss=F.cross_entropy(logits,y)
            if not torch.isfinite(loss): raise FloatingPointError('Nonfinite evaluation')
            totals[0]+=loss.double()*len(y);totals[1]+=(logits.argmax(1)==y).sum();count+=len(y)
    loss,accuracy=(totals/count).cpu().tolist()
    return {'loss':loss,'accuracy':accuracy,'examples':count}


def audit(model,dataset,config,device,candidates,run_dir):
    """Read-only replay with BN-buffer restoration and a fixed train cohort."""
    saved=model.snapshot(); was_training=model.training
    gate=config['gate']; examples=[]
    for x,y in loader(dataset,gate['microbatch_size']): examples.append((x,y))
    def replay(schedule):
        records=[]
        for x,y in examples:
            model.restore(saved);model.train()
            for p in model.trainable_tensors().values():p.grad=None
            free,tracked=model.bptt(x.to(device),schedule)
            loss=F.cross_entropy(tracked,y.to(device))
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite audit loss')
            loss.backward()
            grads={n:p.grad.detach().cpu().clone() for n,p in model.named_conductances()
                   if p.grad is not None}
            if len(grads)!=9:raise RuntimeError('Audit requires all nine conductance tensors')
            records.append({'free':free.cpu(),'tracked':tracked.detach().cpu(),'gradients':grads})
            model.detach_state_()
        return records
    def compare(candidate,reference):
        layers={}; passed=True
        for name in reference[0]['gradients']:
            norms,refnorms,cosines,zeros,refzeros,rms=[],[],[],[],[],[]
            for cand,ref in zip(candidate,reference):
                a=cand['gradients'][name].flatten().double();b=ref['gradients'][name].flatten().double()
                if not torch.isfinite(a).all() or not torch.isfinite(b).all():
                    raise FloatingPointError(f'Nonfinite audit gradient {name}')
                na,nb=float(a.norm()),float(b.norm());norms.append(na);refnorms.append(nb)
                cosines.append(float(torch.dot(a,b)/(a.norm()*b.norm()).clamp_min(1e-300)))
                zeros.append(float((a.abs()<=gate['zero_epsilon']).double().mean()))
                refzeros.append(float((b.abs()<=gate['zero_epsilon']).double().mean()))
                rms.append(nb/math.sqrt(b.numel()))
            norm_delta=abs(np.mean(norms)-np.mean(refnorms))/max(float(np.mean(refnorms)),1e-300)
            zero_delta=abs(np.mean(zeros)-np.mean(refzeros))
            viable=bool(np.median(rms)>gate['minimum_reference_gradient_rms'] and
                        np.quantile(refzeros,.9)<gate['maximum_reference_zero_q90'])
            ok=bool(viable and np.mean(cosines)>=gate['minimum_cosine'] and
                    norm_delta<=gate['maximum_relative_norm_delta'] and
                    zero_delta<=gate['maximum_zero_fraction_delta'])
            layers[name]={'mean_cosine':float(np.mean(cosines)),'minimum_cosine':min(cosines),
                          'relative_norm_delta':float(norm_delta),'zero_fraction_delta':float(zero_delta),
                          'reference_viable':viable,'passed':ok}
            passed &= ok
        logits={}
        for key in ['free','tracked']:
            a=torch.cat([r[key] for r in candidate]).double()
            b=torch.cat([r[key] for r in reference]).double()
            delta=float((a-b).norm()/b.norm().clamp_min(1e-300));logits[key]=delta
            passed &= delta<=gate['maximum_logit_relative_l2']
        return {'passed':bool(passed),'layers':layers,'logit_relative_l2':logits}
    try:
        reporting.update_status_progress(run_dir,{'stage':'solver_audit','schedule':'reference_check'})
        reference=replay(config['reference_iterations'])
        sentinel=replay(config['reference_check_iterations'])
        stability=compare(reference,sentinel);del sentinel
        output={'reference_stability':stability,'candidates':[], 'selected_iterations':None,
                'official_test_read':False,'cohort_indices':dataset.indices,'thresholds':gate}
        if stability['passed']:
            for candidate in candidates:
                reporting.update_status_progress(run_dir,{'stage':'solver_audit','schedule':candidate})
                measured=replay(candidate);comparison=compare(measured,reference);del measured
                output['candidates'].append({'iterations':candidate,**comparison})
                print('AUDIT',candidate,'passed=',comparison['passed'],flush=True)
                if comparison['passed']:
                    output['selected_iterations']=candidate;break
        output['passed']=output['selected_iterations'] is not None
        return output
    finally:
        model.restore(saved);model.detach_state_();model.train(was_training)
        for p in model.trainable_tensors().values():p.grad=None


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config',type=Path)
    parser.add_argument('--mode',choices=['prepare','smoke','train'],required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--assets',type=Path)
    parser.add_argument('--dataset-root',type=Path,required=True)
    parser.add_argument('--source-identity',type=Path,required=True)
    parser.add_argument('--device',default='cuda')
    parser.add_argument('--target',default='fifi')
    parser.add_argument('--initial-checkpoint',type=Path,
                        help='Prepare from an existing exact shared initializer')
    parser.add_argument('--resume-run',type=Path,
                        help='Validated completed parent bundle; resume its final epoch checkpoint')
    parser.add_argument('--resume-checkpoint',type=Path,
                        help='Recover an interrupted run from a saved epoch boundary; retain strict scientific config checks')
    parser.add_argument('--save-tensors-on-cpu',action='store_true',
                        help='Offload backward saved tensors without changing batch size')
    parser.add_argument('--require-gpu-resident',action='store_true',
                        help='Fail closed on CPU execution or saved-tensor CPU offloading')
    parser.add_argument('--smoke-batches',type=int,default=2)
    parser.add_argument('--stop-before',type=parse_stop_before,
                        help='Operational timezone-qualified deadline; pause with full epoch-boundary state')
    parser.add_argument('--initial-epoch-seconds',type=float,default=900.,
                        help='Conservative first-epoch duration for deadline admission')
    parser.add_argument('--deadline-reserve-seconds',type=float,default=120.)
    args=parser.parse_args();c=json.loads(args.config.read_text());run=args.output_dir
    if run.exists():raise FileExistsError(run)
    if args.resume_run and args.mode=='prepare':raise ValueError('Cannot resume preparation')
    if args.resume_checkpoint and (args.resume_run or args.mode=='prepare'):
        raise ValueError('Checkpoint recovery cannot be combined with resume-run or preparation')
    if args.require_gpu_resident:check_gpu_resident_execution(args.device,args.save_tensors_on_cpu)
    identity=json.loads(args.source_identity.read_text());repo=Path(__file__).resolve().parents[1]
    for name,sha in identity['source_sha256'].items():
        if reporting.sha256_file(repo/name)!=sha:raise ValueError(f'Staged source mismatch: {name}')
    deterministic_setup(c['seed']);device=torch.device(args.device)
    if args.smoke_batches<1:raise ValueError('Smoke requires at least one training batch')
    if args.initial_epoch_seconds<=0 or args.deadline_reserve_seconds<30:
        raise ValueError('Positive epoch estimate and at least 30 seconds deadline reserve required')
    reporting.start_run(run,{'study_id':c['study_id'],'run_id':run.name,'arm_id':c.get('arm_id',f"bs{c['batch_size']}"),
        'seed':c['seed'],'smoke':args.mode!='train','evidence_class':c['evidence_class'],
        'resolved_config':c,'mode':args.mode,'source_identity':identity,
        'command':shlex.join([sys.executable,*sys.argv]),
        'dataset':{'name':'CIFAR10','official_test_read':False,'train_examples':45000,'validation_examples':5000},
        'runtime':dict(reporting.runtime_context(target=args.target),pid=os.getpid()),
        'execution':{'save_tensors_on_cpu':args.save_tensors_on_cpu,'smoke_batches':args.smoke_batches,
                     'require_gpu_resident':args.require_gpu_resident,
                     'stop_before':args.stop_before.isoformat() if args.stop_before else None,
                     'initial_epoch_seconds':args.initial_epoch_seconds,
                     'deadline_reserve_seconds':args.deadline_reserve_seconds},
        'environment':{'torch':str(torch.__version__),'cuda':torch.version.cuda,'cudnn':torch.backends.cudnn.version(),
                       'device':str(device),'gpu':torch.cuda.get_device_name(device) if device.type=='cuda' else None}})
    start=time.monotonic();deadline=start+c['maximum_runtime_hours']*3600
    if args.stop_before:
        deadline=min(deadline,start+(args.stop_before-datetime.now(timezone.utc)).total_seconds()-30)
    try:
        reporting.atomic_write_json(run/'artifacts/config.json',c)
        images,labels=read_training_data(args.dataset_root)
        split=make_split(labels,c['split']['seed'])
        model=CifarL8Analog(c,device)
        if args.require_gpu_resident and any(p.device.type!='cuda' for p in model.trainable_tensors().values()):
            raise RuntimeError('Trainable model tensor is on CPU; refusing training')
        optimizer,scheduler=make_optimizer(model,c)
        if args.mode=='prepare':
            reporting.atomic_write_json(run/'artifacts/split.json',split)
            if args.initial_checkpoint:
                initial=torch.load(args.initial_checkpoint,map_location='cpu',weights_only=False)
                if initial['epoch']!=0:raise ValueError('Shared checkpoint must be at initialization')
                omitted=model.restore_initializer(initial['model']);del initial
                reporting.atomic_write_json(run/'artifacts/initializer_transfer.json',{
                    'source':str(args.initial_checkpoint),
                    'source_sha256':reporting.sha256_file(args.initial_checkpoint),
                    'omitted_bn_keys':omitted})
                save_checkpoint(run/'checkpoints/initial_model.pt',model,optimizer,scheduler,0,c)
            else:
                save_checkpoint(run/'checkpoints/initial_model.pt',model,optimizer,scheduler,0,c)
            cohort=split['train'][:c['gate']['cohort_size']]
            dataset=Cohort(images,labels,cohort,c,augment=False)
            candidates=([c['iterations']] if c.get('fixed_iterations',False)
                        else [c['iterations'],c['alternative_iterations']])
            qualification=audit(model,dataset,c,device,candidates,run)
            reporting.atomic_write_json(run/'artifacts/qualification.json',qualification)
            if not qualification['passed']:raise RuntimeError('No candidate passed T/K gate')
            reporting.append_metric(run/'metrics.jsonl',{'stage':'qualification','passed':True,
                                     'iterations':qualification['selected_iterations']})
            reporting.complete_run(run,terminal_metrics={'elapsed_seconds':time.monotonic()-start},
                completion={'criteria_met':True,'official_test_read':False})
            print('PREPARED',qualification['selected_iterations'],flush=True);return
        if args.assets is None:raise ValueError('Prepared assets are required')
        errors=reporting.validate_run(args.assets)
        if errors:raise ValueError(errors)
        qualified=json.loads((args.assets/'manifest.json').read_text())['resolved_config']
        check_boundary_compatibility(qualified,c)
        for field in ['solver','blocks','input_size','num_classes','gate',
                      'reference_iterations','reference_check_iterations']:
            if qualified[field]!=c[field]:raise ValueError(f'Qualification config mismatch: {field}')
        if qualified.get('block_output_normalization', 'none') != c.get('block_output_normalization', 'none'):
            raise ValueError('Qualification config mismatch: block_output_normalization')
        qualification=json.loads((args.assets/'artifacts/qualification.json').read_text())
        if not qualification['passed']:raise ValueError('Preparation did not qualify')
        if json.loads((args.assets/'artifacts/split.json').read_text())!=split:raise ValueError('Split changed')
        initial=torch.load(args.assets/'checkpoints/initial_model.pt',map_location='cpu',weights_only=False)
        model.restore(initial['model']);del initial
        iterations=qualification['selected_iterations']
        reporting.atomic_write_json(run/'artifacts/assets.json',{'initial_sha256':reporting.sha256_file(args.assets/'checkpoints/initial_model.pt'),
            'split_sha256':reporting.sha256_file(args.assets/'artifacts/split.json'),
            'qualification_sha256':reporting.sha256_file(args.assets/'artifacts/qualification.json'),
            'iterations':iterations})
        start_epoch=0;steps=0
        if args.resume_run:
            errors=reporting.validate_run(args.resume_run)
            if errors:raise ValueError(errors)
            parent_manifest=json.loads((args.resume_run/'manifest.json').read_text())
            if parent_manifest['mode']!='train':raise ValueError('Cannot resume a smoke')
            parent_assets=json.loads((args.resume_run/'artifacts/assets.json').read_text())
            if parent_assets!=json.loads((run/'artifacts/assets.json').read_text()):
                raise ValueError('Resume assets mismatch')
            path=args.resume_run/'checkpoints/final_model.pt'
            checkpoint=torch.load(path,map_location='cpu',weights_only=False)
            if reporting.sha256_file(path)!=c['continuation']['checkpoint_sha256']:
                raise ValueError('Parent checkpoint SHA mismatch')
            start_epoch=restore_checkpoint(checkpoint,model,optimizer,scheduler,c)
            if start_epoch!=c['continuation']['from_epoch']:raise ValueError('Parent epoch mismatch')
            rows=[json.loads(line) for line in (args.resume_run/'metrics.jsonl').read_text().splitlines()]
            if rows[-1]['epoch']!=start_epoch:raise ValueError('Parent metrics epoch mismatch')
            steps=rows[-1]['steps']
            if {int(v['step']) for v in optimizer.state.values()}!={steps}:
                raise ValueError('Adam step count does not match parent metrics')
            reporting.atomic_write_json(run/'artifacts/continuation.json',{
                'parent_run':str(args.resume_run),'checkpoint_sha256':reporting.sha256_file(path),
                'parent_manifest_sha256':reporting.sha256_file(args.resume_run/'manifest.json'),
                'from_epoch':start_epoch,'parent_steps':steps,
                'restored_learning_rates':scheduler.get_last_lr(),
                'scheduler_last_epoch':scheduler.last_epoch,
                'best_checkpoint_scope':'new continuation epochs only',
                'rng_note':'CPU torch, NumPy and Python restored; model uses no stochastic CUDA operations'})
            del checkpoint
        elif args.resume_checkpoint:
            checkpoint=torch.load(args.resume_checkpoint,map_location='cpu',weights_only=False)
            start_epoch=restore_checkpoint(checkpoint,model,optimizer,scheduler,c)
            saved_steps={int(v['step']) for v in optimizer.state.values()}
            steps=start_epoch*math.ceil(len(split['train'])/c['batch_size'])
            if saved_steps!={steps}:
                raise ValueError('Recovery Adam steps do not match the completed epoch boundary')
            reporting.atomic_write_json(run/'artifacts/recovery.json',{
                'checkpoint':str(args.resume_checkpoint),
                'checkpoint_sha256':reporting.sha256_file(args.resume_checkpoint),
                'from_epoch':start_epoch,'parent_steps':steps,
                'restored_learning_rates':scheduler.get_last_lr(),
                'scheduler_last_epoch':scheduler.last_epoch,
                'reason':'Explicit recovery from interrupted epoch-boundary checkpoint',
                'best_checkpoint_scope':'recovery epochs only'})
            del checkpoint
        save_checkpoint(run/'checkpoints/initial_model.pt',model,optimizer,scheduler,start_epoch,c)
        if args.stop_before:
            def request_pause(signum, frame):
                raise PlannedPause(f'Signal {signum}: recover the last complete epoch boundary')
            signal.signal(signal.SIGTERM, request_pause)
        epochs=start_epoch+1 if args.mode=='smoke' else c['epochs'];best=(-1.,-float('inf'));best_epoch=0
        validation=split['validation'][:c['evaluation_batch_size']] if args.mode=='smoke' else split['validation']
        val_loader=loader(Cohort(images,labels,validation,c),c['evaluation_batch_size'],c['num_workers'],device.type=='cuda')
        epoch_durations=[]
        for epoch in range(start_epoch+1,epochs+1):
            if args.stop_before and deadline-time.monotonic()<epoch_admission_seconds(
                    epoch_durations,args.initial_epoch_seconds,args.deadline_reserve_seconds):
                raise PlannedPause(f'Insufficient time to admit epoch {epoch}/{epochs} before {args.stop_before.isoformat()}')
            tick=time.monotonic()
            permutation=torch.randperm(len(split['train']),generator=torch.Generator().manual_seed(c['seed']+1000000*epoch)).tolist()
            indices=[split['train'][i] for i in permutation]
            if args.mode=='smoke':indices=indices[:args.smoke_batches*c['batch_size']]
            batches=loader(Cohort(images,labels,indices,c,epoch,True),c['batch_size'],c['num_workers'],device.type=='cuda')
            model.train();total_loss=0.;correct=0;count=0;norms={}
            for batch,(x,y) in enumerate(batches,1):
                if time.monotonic()>deadline:
                    if args.stop_before:raise PlannedPause('Deadline fallback: recover the last complete epoch boundary')
                    raise TimeoutError('Declared runtime cap exhausted')
                x,y=x.to(device),y.to(device);optimizer.zero_grad(set_to_none=True)
                # The unpinned path preserves dense tensor strides. Pinned
                # packing makes saved views contiguous and can change the
                # floating-point gradient accumulation order.
                context=torch.autograd.graph.save_on_cpu(pin_memory=False) if args.save_tensors_on_cpu else nullcontext()
                capture = (capture_boundary_statistics(model)
                           if batch==1 and c.get('record_boundary_statistics',False) else nullcontext({}))
                with context, capture as boundary_stats:
                    free,tracked=model.bptt(x,iterations)
                    loss=F.cross_entropy(tracked,y,label_smoothing=c['label_smoothing'])
                if batch==1:
                    first_boundary_stats=dict(boundary_stats)
                    first_logit_stats=tensor_statistics(tracked)
                if not torch.isfinite(loss):raise FloatingPointError('Nonfinite training loss')
                loss.backward()
                if batch==1:norms=gradient_norms(model)
                optimizer.step();model.project_();model.detach_state_()
                total_loss+=float(loss.detach())*len(y);correct+=int((free.argmax(1)==y).sum());count+=len(y);steps+=1
                if batch==1 or batch%100==0:
                    reporting.update_status_progress(run,{'stage':'training','epoch':epoch,'epochs':epochs,
                        'batch':batch,'batches':len(batches),'steps':steps})
                    print(f'EPOCH {epoch}/{epochs} BATCH {batch}/{len(batches)} loss={float(loss):.5f}',flush=True)
            reporting.update_status_progress(run,{'stage':'validation','epoch':epoch})
            measured=evaluate(model,val_loader,iterations,device,deadline)
            clipping={n:{'lower':float((p<=c['weight_min']).float().mean()),
                         'upper':float((p>=c['weight_max']).float().mean())} for n,p in model.named_conductances()}
            row={'epoch':epoch,'evaluation_split':'validation','train_examples':count,
                 'validation_examples':measured['examples'],'train_loss':total_loss/count,'train_accuracy':correct/count,
                 'validation_loss':measured['loss'],'validation_accuracy':measured['accuracy'],
                 'first_batch_gradient_norms':norms,'clipping':clipping,'steps':steps,
                 'input_gains':[float(block.input_gain.detach()) for block in model.analog_blocks],
                 'head_input_gain':float(F.softplus(model.head._input_gain_raw.detach())),
                 'epoch_seconds':time.monotonic()-tick,'learning_rates':[g['lr'] for g in optimizer.param_groups],
                 'peak_memory_reserved_bytes':torch.cuda.max_memory_reserved(device) if device.type=='cuda' else 0}
            if c.get('record_boundary_statistics',False):
                row.update(first_batch_boundary_statistics=first_boundary_stats,
                           first_batch_logit_statistics=first_logit_stats)
            reporting.append_metric(run/'metrics.jsonl',row);scheduler.step()
            key=(measured['accuracy'],-measured['loss'])
            if key>best:
                best=key;best_epoch=epoch;save_checkpoint(run/'checkpoints/best_model.pt',model,optimizer,scheduler,epoch,c)
            save_checkpoint(run/'checkpoints/final_model.pt',model,optimizer,scheduler,epoch,c)
            epoch_durations.append(time.monotonic()-tick)
            print(f"METRIC epoch={epoch} val_acc={measured['accuracy']:.6f} val_ce={measured['loss']:.6f} seconds={row['epoch_seconds']:.2f}",flush=True)
        final_gate=None
        if args.mode=='train':
            final_gate=audit(model,Cohort(images,labels,split['train'][:c['gate']['cohort_size']],c),c,device,[iterations],run)
            reporting.atomic_write_json(run/'artifacts/final_solver_audit.json',final_gate)
        reporting.complete_run(run,terminal_metrics={'epochs_completed':epochs,'steps':steps,
            'starting_epoch':start_epoch,'epochs_trained':epochs-start_epoch,
            'final_validation_accuracy':measured['accuracy'],'final_validation_loss':measured['loss'],
            'best_validation_accuracy':best[0],'best_epoch':best_epoch,
            'elapsed_seconds':time.monotonic()-start,'last_epoch_seconds':row['epoch_seconds'],
            'peak_memory_reserved_bytes':row['peak_memory_reserved_bytes'],
            'final_solver_audit_passed':None if final_gate is None else final_gate['passed']},
            completion={'criteria_met':True,'official_test_read':False,'full_epoch_budget':epochs==c['epochs']})
        errors=reporting.validate_run(run)
        if errors:raise RuntimeError(errors)
        print('COMPLETE',run,flush=True)
    except (PlannedPause, TimeoutError) as exc:
        if not args.stop_before:
            reporting.fail_run(run,error=exc)
            raise
        checkpoint_path=run/'checkpoints/final_model.pt'
        if not checkpoint_path.exists():checkpoint_path=run/'checkpoints/initial_model.pt'
        checkpoint=torch.load(checkpoint_path,map_location='cpu',weights_only=False)
        reporting.atomic_write_json(run/'artifacts/pause.json',{
            'reason':str(exc),'planned_pause':True,'target_epochs':c['epochs'],
            'completed_epoch':checkpoint['epoch'],'resume_checkpoint':str(checkpoint_path),
            'checkpoint_sha256':reporting.sha256_file(checkpoint_path),
            'scheduler_last_epoch':checkpoint['scheduler']['last_epoch'],
            'stop_before':args.stop_before.isoformat(),
            'resume_note':'Resume this full epoch-boundary checkpoint in a new bundle; any incomplete epoch is replayed.'})
        reporting.fail_run(run,error=PlannedPause(str(exc)),returncode=75)
        print('PLANNED_PAUSE',checkpoint_path,flush=True)
        raise SystemExit(75)
    except BaseException as exc:
        if not (run/'result.json').exists():reporting.fail_run(run,error=exc)
        raise

if __name__=='__main__':main()
