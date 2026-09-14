"""Read-only solver diagnostic on a fixed training cohort; no retraining."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from labs.fmnist_homeostasis import PaperNet,free_state,advance,recur
from labs.tools.train_directed_eqprop_mnist import write_json


@torch.jit.script
def damped(u:torch.Tensor,drive:torch.Tensor,f1:torch.Tensor,f2:torch.Tensor,
           b1:torch.Tensor,b2:torch.Tensor,steps:int,rate:float):
    for _ in range(steps):
        force=recur(torch.sigmoid(4*u-2),f1,f2,b1,b2)+drive-u
        u=u+rate*force
    return u


@torch.no_grad()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--seed',type=int,default=0)
    args=p.parse_args()
    torch.set_num_threads(1)
    root=args.root
    x=torch.tensor(np.load(root/'data/train_x.npy',mmap_mode='r')[:500]).double()
    run=json.loads((root/'run.json').read_text())
    results={}
    for name,cfg in run['cases'].items():
        if cfg['seed']!=args.seed:continue
        path=root/name/'final.npz'
        with np.load(path) as f:
            model=PaperNet(hidden=cfg['hidden']).double()
            model.load_state_dict({k:torch.tensor(f[k]) for k in model.state_dict()})
        u,drive=free_state(model,x,150)
        u1500=advance(u,drive,torch.zeros_like(u),*model.weights(),1350)
        u_next=advance(u1500,drive,torch.zeros_like(u),*model.weights(),1)
        u_second=advance(u_next,drive,torch.zeros_like(u),*model.weights(),1)
        alternative=damped(torch.zeros_like(u),drive,*model.weights(),1500,.5)
        summarize=lambda s:dict(nonconverged_fraction=float((model.force(s,drive).norm(dim=-1)>1e-7).double().mean()),
                               max_force_residual=float(model.force(s,drive).norm(dim=-1).max()))
        n1=(u_next-u1500).norm(dim=-1);n2=(u_second-u1500).norm(dim=-1)
        results[name]=dict(synchronous_150=summarize(u),synchronous_1500=summarize(u1500),
            damped_rate_half_1500=summarize(alternative),
            persistent_period_two_fraction=float(((n1>1e-7)&(n2<1e-7)).double().mean()))
        print(name,results[name],flush=True)
    write_json(root.parent/f'settling_seed{args.seed}.json',dict(status='complete',cohort='first 500 training images',
        purpose='posthoc solver validity diagnostic; no changed weights, training, hyperparameters, or reported test accuracy',
        caveat='Damping preserves force zeros but can change trajectories and selected basins. It was not used in training.',
        residual_threshold_l2=1e-7,results=results))


if __name__=='__main__':main()
