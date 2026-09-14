"""Read-only matched-checkpoint comparison of measured feedback directions."""

import argparse
import csv
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from labs.fmnist_homeostasis import (PaperNet,MeasuredFeedback,free_state,forward_tangent,
    advance_error,measured_responses,slope)
from labs.directed_eqprop import local_gradient
from labs.tools.train_directed_eqprop_mnist import write_json,utc


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def cosines(a,b):
    na,nb=a.norm(dim=-1),b.norm(dim=-1)
    valid=(na>1e-12)&(nb>1e-12)
    return torch.where(valid,(a*b).sum(-1)/(na*nb).clamp_min(1e-300),torch.nan)


def mean(values):
    a=np.asarray(values,dtype=float)
    a=a[np.isfinite(a)]
    return None if len(a)==0 else float(a.mean())


def load_checkpoint(path):
    with np.load(path) as f:
        cfg=json.loads(str(f['config_json']))
        model=PaperNet(hidden=cfg['hidden']).double()
        model.load_state_dict({k:torch.tensor(f[k]) for k in model.state_dict()})
        learner=None
        if 'feedback_matrix' in f:
            learner=MeasuredFeedback(model.n,model.outputs)
            learner.matrix=f['feedback_matrix'].copy()
    return model,learner


@torch.no_grad()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--trials',type=int,default=8)
    p.add_argument('--budgets',type=int,nargs='+',default=[1,4,16])
    args=p.parse_args()
    if args.output.exists():raise ValueError(f'Expected fresh output path; got {args.output}')
    if args.trials<2 or any(m<1 or m>64 for m in args.budgets):
        raise ValueError('Expected at least two trials and probe budgets in 1..64')
    args.output.mkdir(parents=True)
    torch.set_num_threads(1)
    root=args.root
    checkpoints=[]
    for seed in (0,1):
        for role,name,filename,epoch in [('init','vf_ad','initial.npz',0),('vf','vf_ad','final.npz',10),
                                         ('homeostasis','vf_ad_homeo','final.npz',10)]:
            path=root/f'seed{seed}_{name}'/filename
            checkpoints.append(dict(name=f'seed{seed}_{role}',seed=seed,role=role,epoch=epoch,path=str(path),sha256=digest(path)))
    path=root/'seed0_zo_learned4/final.npz'
    checkpoints.append(dict(name='seed0_learned',seed=0,role='learned',epoch=10,path=str(path),sha256=digest(path)))
    test_x=np.load(root/'data/test_x.npy',mmap_mode='r')
    test_y=np.load(root/'data/test_y.npy',mmap_mode='r')
    rng=np.random.default_rng(20260915)
    candidates=np.concatenate([rng.choice(np.flatnonzero(test_y==label),4,replace=False) for label in range(10)])
    rng.shuffle(candidates)
    x=torch.tensor(test_x[candidates]).double();y=torch.tensor(test_y[candidates])
    # One shared valid cohort across all checkpoints, not a favorable subset for
    # each method. Record all exclusions and require true free equilibria.
    common=torch.ones(len(x),dtype=torch.bool)
    screening={}
    for cp in checkpoints:
        model,_=load_checkpoint(cp['path'])
        u,drive=free_state(model,x,150)
        norms=model.force(u,drive).norm(dim=-1)
        valid=norms<1e-7
        common &= valid
        screening[cp['name']]=dict(valid_examples=int(valid.sum()),force_residuals=norms.tolist())
    selected=candidates[common.numpy()]
    if len(selected)<8:raise RuntimeError(f'Only {len(selected)} common equilibrated examples; see candidate cohort')
    x,y=x[common],y[common]
    np.savez(args.output/'cohort.npz',candidate_indices=candidates,selected_indices=selected,labels=y.numpy())
    cohort_hash=hashlib.sha256(selected.tobytes()).hexdigest()
    manifest=dict(status='running',started_utc=utc(),checkpoints=checkpoints,candidate_examples=40,
        selected_examples=len(selected),cohort_sha256=cohort_hash,screening=screening,
        settings=dict(free_steps=150,response_steps=20,beta=.01,dtype='float64',device='cpu',
                      trials=args.trials,probe_budgets=args.budgets,batch_size=10,free_force_l2_threshold=1e-7),
        selection='Four examples per class from official test set, fixed seed; intersection of free-equilibrium-valid examples across every checkpoint',
        changes='No training, optimizer updates, predictor fitting, or test accuracy measurements',
        learned_predictor='Saved predictor is evaluated only on its own seed-0 checkpoint; never transferred to unrelated weights')
    write_json(args.output/'run.json',manifest)
    rows,param_rows=[],[]
    initial_norms={};initial_gradients={}

    for cp in checkpoints:
        model,learner=load_checkpoint(cp['path'])
        saved={k:v.clone() for k,v in model.state_dict().items()}
        u,drive=free_state(model,x,150)
        c=model.cost(u,y)[2]
        d=slope(u)
        ju=model.dense_w()*d.unsqueeze(-2)-torch.eye(model.n,dtype=u.dtype)
        truth=torch.linalg.solve(ju.transpose(-1,-2),(d*c).unsqueeze(-1)).squeeze(-1)
        finite=torch.zeros_like(u)
        for _ in range(20):finite=d*model.recurrent_transpose(finite)-d*c
        target=torch.nn.functional.one_hot(y,10).double()
        q_ad=forward_tangent(u,drive,target,*model.weights(),model.readout,model.readout_bias,20)
        eargs=(u,drive,target,*model.weights(),model.readout,model.readout_bias)
        q=(torch.sigmoid(4*advance_error(*eargs,.01,20)-2)-torch.sigmoid(4*advance_error(*eargs,-.01,20)-2))/.02
        true_grads={}
        for batch,start in enumerate(range(0,len(x),10)):
            sl=slice(start,start+10)
            true_grads[batch]=local_gradient(model,x[sl],u[sl],y[sl],truth[sl])
            if cp['role']=='init':
                for name,g in true_grads[batch].items():
                    initial_norms[(cp['seed'],batch,name)]=float(g.square().mean().sqrt())
                    initial_gradients[(cp['seed'],batch,name)]=g.clone()

        def measure(method,budget,trial,ell):
            for layer,a,b in [('all',ell,truth),*[(name,a,b) for name,a,b in zip(
                    ('hidden1','hidden2','output'),ell.split(model.sizes,dim=-1),truth.split(model.sizes,dim=-1))]]:
                cs=cosines(a,b)
                rows.append(dict(checkpoint=cp['name'],role=cp['role'],seed=cp['seed'],epoch=cp['epoch'],
                    method=method,probes=budget,trial=trial,layer=layer,cohort_sha256=cohort_hash,
                    cosine=mean(cs.tolist()),nonzero_examples=int(torch.isfinite(cs).sum()),
                    relative_error=float((a-b).norm()/b.norm().clamp_min(1e-300)),
                    norm_ratio=float(a.norm()/b.norm().clamp_min(1e-300))))
            for batch,start in enumerate(range(0,len(x),10)):
                sl=slice(start,start+10)
                grads=local_gradient(model,x[sl],u[sl],y[sl],ell[sl])
                for name,g in grads.items():
                    ref=true_grads[batch][name]
                    rms=float(g.square().mean().sqrt())
                    initial=initial_gradients[(cp['seed'],batch,name)]
                    param_rows.append(dict(checkpoint=cp['name'],role=cp['role'],seed=cp['seed'],epoch=cp['epoch'],
                        method=method,probes=budget,trial=trial,batch=batch,parameter=name,cohort_sha256=cohort_hash,
                        cosine=mean(cosines(g.flatten()[None],ref.flatten()[None]).tolist()),
                        gradient_rms=rms,reference_rms=float(ref.square().mean().sqrt()),
                        fraction_abs_le_1e12=float((g.abs()<=1e-12).double().mean()),
                        relative_to_initial_reference_rms=rms/max(initial_norms[(cp['seed'],batch,name)],1e-300),
                        relative_error=float((g-ref).norm()/ref.norm().clamp_min(1e-300)),
                        reference_cosine_to_initial=mean(cosines(ref.flatten()[None],initial.flatten()[None]).tolist()),
                        reference_change_from_initial=float((ref-initial).norm()/initial.norm().clamp_min(1e-300))))

        measure('true_adjoint',0,0,truth)
        measure('finite_time_adjoint',0,0,finite)
        measure('vf_ad',0,0,q_ad)
        measure('vf_central',0,0,q)
        if learner is not None:
            baseline=learner.predict(u,c)
            measure('learned_baseline',0,0,baseline)
        for trial in range(args.trials):
            gen=torch.Generator().manual_seed(92000+trial)
            z=(2*torch.randint(2,(max(args.budgets),len(u),model.n),generator=gen)-1).double()/math.sqrt(model.n)
            response=measured_responses(model,drive,u,z,.01,20)
            values=(response*c).sum(-1)
            for m in args.budgets:
                zz=z[:m]
                # Gram solve depends only on chosen injection directions; it is
                # not an inverse network Jacobian or a reference-gradient call.
                design=zz.permute(1,0,2)
                gram=design@design.transpose(-1,-2)
                baselines=[('ep',q)]
                if learner is not None:baselines.append(('learned',baseline))
                for prefix,base in baselines:
                    residual=values[:m]-(zz*base).sum(-1)
                    mc=base+model.n*(zz*residual.unsqueeze(-1)).mean(0)
                    coefficients=torch.linalg.solve(gram,residual.T.unsqueeze(-1))
                    projection=base+(design.transpose(-1,-2)@coefficients).squeeze(-1)
                    measure(prefix+'_mc',m,trial,mc)
                    measure(prefix+'_projection',m,trial,projection)
        assert all(torch.equal(v,saved[k]) for k,v in model.state_dict().items())
        assert digest(cp['path'])==cp['sha256']
        print(cp['name'],'complete on',len(x),'shared examples',flush=True)
        write_json(args.output/'status.json',dict(status='running',last_checkpoint=cp['name'],utc=utc()))

    for filename,values in [('feedback_cosines.csv',rows),('parameter_gradients.csv',param_rows)]:
        with (args.output/filename).open('w') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(values[0]));writer.writeheader();writer.writerows(values)
    summaries={}
    for cp in checkpoints:
        subset=[r for r in rows if r['checkpoint']==cp['name'] and r['layer']=='hidden1']
        keys=sorted({(r['method'],r['probes']) for r in subset})
        summaries[cp['name']]=[]
        for method,m in keys:
            items=[r for r in subset if r['method']==method and r['probes']==m]
            scores=[r['cosine'] for r in items if r['cosine'] is not None]
            pars=[r['cosine'] for r in param_rows if r['checkpoint']==cp['name'] and r['method']==method and r['probes']==m and r['parameter']=='input' and r['cosine'] is not None]
            summaries[cp['name']].append(dict(method=method,probes=m,hidden1_cosine=mean(scores),
                trial_sd=float(np.std(scores,ddof=1)) if len(scores)>1 else 0.,
                input_gradient_cosine=mean(pars),relative_error=mean([r['relative_error'] for r in items])))
    write_json(args.output/'summary.json',summaries)
    fig,axes=plt.subplots(2,3,figsize=(12,7),sharex=True,sharey=True)
    for ax,cp in zip(axes.flat,[c for c in checkpoints if c['role']!='learned']):
        data=summaries[cp['name']]
        vf=next(r['hidden1_cosine'] for r in data if r['method']=='vf_ad')
        ax.axhline(vf,color='#5b6470',label='VF, exact response')
        for method,label,color in [('ep_mc','MC correction','#bd5126'),('ep_projection','Projection correction','#19794f')]:
            ds=sorted([r for r in data if r['method']==method],key=lambda r:r['probes'])
            ax.errorbar([r['probes'] for r in ds],[r['hidden1_cosine'] for r in ds],
                yerr=[r['trial_sd'] for r in ds],marker='o',label=label,color=color,capsize=3)
        ax.set_title(cp['name']+f" (epoch {cp['epoch']})")
        ax.set_xscale('log',base=2);ax.set_xticks(args.budgets,[str(m) for m in args.budgets]);ax.grid(alpha=.2)
        ax.set_ylim(-.1,1.05)
    axes[0,0].legend(fontsize=8)
    for ax in axes[:,0]:ax.set_ylabel('Hidden-1 cosine to true adjoint')
    for ax in axes[-1]:ax.set_xlabel('Fresh measured probe directions')
    fig.suptitle(f'Fixed Fashion-MNIST checkpoints; {len(x)} shared equilibrated examples')
    fig.tight_layout();fig.savefig(args.output/'cosine_comparison.png',dpi=180);fig.savefig(args.output/'cosine_comparison.pdf');plt.close(fig)
    lines=['# Feedback cosine similarity on fixed Fashion-MNIST checkpoints','',
        f'{len(checkpoints)} checkpoints; {len(x)} shared examples retained from 40 stratified candidates; {args.trials} probe trials per budget.',
        'No training or accuracy measurement. Same weights, examples and physical probes for estimator comparisons. Homeostasis is assessed through its saved weights, not by adding its regularizer gradient to the task-gradient reference.','',
        '| Checkpoint | VF cosine, hidden 1 | MC, 4 probes | Projection, 4 probes |','|---|---:|---:|---:|']
    for cp in checkpoints:
        data=summaries[cp['name']]
        find=lambda name,m:next(r['hidden1_cosine'] for r in data if r['method']==name and r['probes']==m)
        lines.append(f"| {cp['name']} | {find('vf_ad',0):.3f} | {find('ep_mc',4):.3f} | {find('ep_projection',4):.3f} |")
    lines += ['', 'MC and projection use the same response measurements. MC is unbiased only in the ideal linear-response limit and has high directional variance. Projection keeps the unmeasured EqProp component: it is biased and can improve Euclidean error without necessarily improving cosine or training.',
        '', 'Finite 20-step response bias is retained. The CSV includes a finite-time adjoint control, full-state and layerwise cosines, norm ratios and relative errors. Parameter-gradient CSV includes cosine, RMS, near-zero fraction and scale/change relative to initialization on the same minibatches.',
        '', 'The cohort is an intersection selected by the declared force-residual criterion. Conclusions apply to these equilibrated examples; excluded examples and residuals are recorded in run.json. A few saved seeds and eight probe trials are diagnostic evidence, not an accuracy result or a hardware speedup.',
        '', 'The saved learned predictor is evaluated only at seed0_learned; it receives no new fitting during replay. Checkpoint hashes before/after and parameter equality were verified. Only epochs 0 and 10 are compared; intermediate gradients were not reconstructed from logs.','']
    (args.output/'report.md').write_text('\n'.join(lines))
    for cp in checkpoints:assert digest(cp['path'])==cp['sha256']
    write_json(args.output/'completion.json',dict(status='complete',checkpoints=len(checkpoints),selected_examples=len(x),
        source_sha256=digest(__file__),checkpoint_hashes_unchanged=True,parameter_tensors_unchanged=True,
        predictor_updates=0,optimizer_steps=0,finished_utc=utc()))
    write_json(args.output/'status.json',dict(status='complete',utc=utc()))


if __name__=='__main__':main()
