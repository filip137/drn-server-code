"""Separate predictor error, finite-response bias and Monte Carlo variance."""

import argparse
import math
from pathlib import Path

import numpy as np
import torch

from labs.fmnist_homeostasis import PaperNet,MeasuredFeedback,free_state,measured_responses,slope
from labs.tools.train_directed_eqprop_mnist import write_json


@torch.no_grad()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--seed',type=int,default=0)
    args=p.parse_args()
    torch.set_num_threads(1)
    root=args.root
    path=root/f'seed{args.seed}_zo_learned4'/'final.npz'
    with np.load(path) as f:
        model=PaperNet(hidden=f['forward1'].shape[0]).double()
        model.load_state_dict({k:torch.tensor(f[k]) for k in model.state_dict()})
        predictor=MeasuredFeedback(model.n,10)
        predictor.matrix=f['feedback_matrix'].copy()
    x=torch.tensor(np.load(root/'data/train_x.npy',mmap_mode='r')[:64]).double()
    y=torch.tensor(np.load(root/'data/train_y.npy',mmap_mode='r')[:64])
    u,drive=free_state(model,x)
    valid=model.force(u,drive).norm(dim=-1)<1e-7
    selected=valid.nonzero().flatten()[:16]
    if len(selected)==0:raise RuntimeError('No equilibrated examples in the declared cohort')
    x,y,u,drive=(v[selected] for v in (x,y,u,drive))
    c=model.cost(u,y)[2]
    d=slope(u)
    ju=model.dense_w()*d.unsqueeze(-2)-torch.eye(model.n,dtype=u.dtype)
    truth=torch.linalg.solve(ju.transpose(-1,-2),(d*c).unsqueeze(-1)).squeeze(-1)
    finite=torch.zeros_like(u)
    for _ in range(20):
        finite=d*model.recurrent_transpose(finite)-d*c
    baseline=predictor.predict(u,c)
    ratio=lambda a,b:float((a-b).norm()/b.norm().clamp_min(1e-30))
    generator=torch.Generator().manual_seed(91000)
    z=(2*torch.randint(2,(32,len(u),model.n),generator=generator)-1).double()/math.sqrt(model.n)
    exact_values=(z*finite).sum(-1)
    amplitudes={}
    for beta in (.1,.01,.001):
        measured=(measured_responses(model,drive,u,z,beta,20)*c).sum(-1)
        amplitudes[str(beta)]=dict(relative_projection_error_to_finite_time=ratio(measured,exact_values),
            relative_projection_error_to_equilibrium=ratio(measured,(z*truth).sum(-1)))
    # This variance experiment deliberately uses exact finite-time projections,
    # isolating the Monte Carlo estimator from measurement/settling bias.
    residual=finite-baseline
    baseline_mse=float(residual.square().sum(-1).mean())
    variance={}
    for probes in (1,4,16,64):
        errors=[]
        for _ in range(128):
            z=(2*torch.randint(2,(probes,len(u),model.n),generator=generator)-1).double()/math.sqrt(model.n)
            correction=model.n*(z*(z*residual).sum(-1,keepdim=True)).mean(0)
            errors.append(float((baseline+correction-finite).square().sum(-1).mean()))
        variance[str(probes)]=dict(trials=128,empirical_mse_over_baseline=float(np.mean(errors))/baseline_mse,
                                   predicted_mse_over_baseline=(model.n-1)/probes)
    result=dict(status='complete',checkpoint=str(path),seed=args.seed,states=model.n,
        cohort='first 16 equilibrated examples among first 64 training images',cohort_indices=selected.tolist(),
        invalid_examples_in_first_64=int((~valid).sum()),
        predictor_relative_error_to_equilibrium=ratio(baseline,truth),
        predictor_relative_error_to_finite_time=ratio(baseline,finite),
        finite_time_relative_error_to_equilibrium=ratio(finite,truth),
        measured_amplitude_sweep=amplitudes,ideal_projection_monte_carlo_variance=variance,
        limitations='Frozen checkpoint, noiseless float64 read-only diagnostic; no additional training or predictor updates. Variance trials use exact oracle projections, not physical measurements. Does not establish why test accuracies differ.')
    write_json(root.parent/f'probe_error_seed{args.seed}.json',result)
    print(result,flush=True)


if __name__=='__main__':main()
