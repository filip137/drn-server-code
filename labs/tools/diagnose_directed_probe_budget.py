"""Read-only probe-count, finite-amplitude and read-noise checks on trained networks.

The cohort is training-only, and saved model/baseline parameters never update.
This is post-training diagnosis, not hyperparameter or checkpoint selection.
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from labs.directed_eqprop import (Counts, DirectedNet, LearnedFeedback, activation,
                                  adjoint, paired_response, settle, slope)
from labs.tools.train_directed_eqprop_mnist import write_json


def diagnose(root, trials=32, examples=8):
    torch.set_num_threads(1)
    root=Path(root)
    x=torch.tensor(np.load(root/"data/train_x.npy",mmap_mode="r")[:examples])
    labels=torch.tensor(np.load(root/"data/train_y.npy",mmap_mode="r")[:examples])
    cases=["alpha90_seed0_"+m for m in ("ep_activity","ep_homeo","learned_probe4","learned_probe4_homeo")]
    output=[]
    for name in cases:
        weights=np.load(root/name/"final.npz")
        cfg=json.loads(str(weights["config_json"]))
        model=DirectedNet(hidden=cfg["hidden"],seed=cfg["seed"],alpha=cfg["alpha"],cap=cfg["cap"])
        model.load_state_dict({k:torch.from_numpy(weights[k]) for k in model.state_dict()})
        learner=None
        if cfg["method"].startswith("learned"):
            learner=LearnedFeedback(model.n,model.outputs)
            learner.matrix=weights["feedback_matrix"].copy()
        with torch.no_grad():
            drive=model.drive(x)
            anchor_counts=Counts()
            free=settle(model,drive,tolerance=1e-12,counts=anchor_counts)
            anchor_counts.state_reads+=examples
            anchor_counts.scalar_reads+=examples*model.n
            c=model.cost(free,labels)[2]
            truth=adjoint(model,free,c,tolerance=1e-13)
            j=model.jacobian(free,"membrane")
            q_linear=slope(free)*torch.linalg.solve(j,c.unsqueeze(-1)).squeeze(-1)
            maximum_m=16
            gen=torch.Generator().manual_seed(8981)
            shape=(trials,maximum_m,examples,model.n)
            z=(2*torch.randint(2,shape,generator=gen)-1).double()/math.sqrt(model.n)
            for beta in (.1,.01,.001):
                counts=Counts()
                plus=settle(model,drive,initial=free,beta=beta,labels=labels,tolerance=1e-12,counts=counts)
                minus=settle(model,drive,initial=free,beta=-beta,labels=labels,tolerance=1e-12,counts=counts)
                counts.state_reads+=2*examples
                counts.scalar_reads+=2*examples*model.n
                counts.excitation_sq+=beta**2*float(model.cost(plus,labels)[2].square().sum()+model.cost(minus,labels)[2].square().sum())
                q=(activation(plus)-activation(minus))/(2*beta)
                baseline=q if learner is None else learner.predict(free,c)
                baseline_mse=float((baseline-truth).square().sum(-1).mean())
                rz=paired_response(model,drive.repeat(trials*maximum_m,1),free.repeat(trials*maximum_m,1),
                    z.reshape(-1,model.n),beta=beta,tolerance=1e-12,counts=counts).reshape(shape)
                # For each sign read, independent per-state noise of variance sigma^2.
                # Its central difference has variance sigma^2/(2 beta^2).
                for sigma in (0.,1e-5):
                    response=rz.clone()
                    if sigma:
                        response+=sigma/(math.sqrt(2)*beta)*torch.randn(shape,generator=gen,dtype=torch.float64)
                    y=(response*c).sum(-1)
                    residual=y-(z*baseline).sum(-1)
                    for m in (1,4,16):
                        estimate=baseline+model.n*(z[:,:m]*residual[:,:m].unsqueeze(-1)).mean(1)
                        squared=(estimate-truth).square().sum(-1).mean(-1)
                        predicted=(model.n-1)/m*baseline_mse
                        output.append(dict(case=name,beta=beta,read_noise=sigma,probes=m,trials=trials,
                            examples=examples,states=model.n,baseline_mse=baseline_mse,
                            correction_mse=float(squared.mean()),
                            correction_mse_standard_error=float(squared.std()/math.sqrt(trials)),
                            ideal_linear_noiseless_mse=predicted,
                            empirical_to_ideal_ratio=float(squared.mean())/max(predicted,1e-30),
                            finite_eqprop_response_relative_error=float((q-q_linear).norm()/q_linear.norm()),
                            anchor_counts=anchor_counts.__dict__.copy(),
                            paired_measurement_counts=counts.__dict__.copy()))
                print(f"diagnosed {name} beta={beta:g}",flush=True)
    write_json(root/"probe_budget_diagnostic.json",dict(status="complete",cohort="first eight training examples",
        scope="read-only final checkpoints, seed 0, mixing angle 90; no parameter/baseline updates or selection",
        notes="Directions are shared across beta/noise comparisons. Smaller budgets use prefixes of the 16 directions. "
              "Each row reports MSE across independent probe sets; ideal prediction excludes finite-nudge/read noise. "
              "Read noise affects fresh probe responses only; baselines and free states are held fixed and noiseless. "
              "Measurement counts include the shared 16-probe acquisition and the EqProp pair used to check finite-beta bias, "
              "not independent cost per table row. Anchor counts are shared across all acquisitions for a checkpoint.",
        rows=output))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
    for name in cases:
        rows=[r for r in output if r["case"]==name and r["beta"]==.01 and r["read_noise"]==0]
        axes[0].loglog([r["probes"] for r in rows],
            [r["correction_mse"]/r["baseline_mse"] for r in rows],marker="o",label=name.removeprefix("alpha90_seed0_"))
        rows=[r for r in output if r["case"]==name and r["probes"]==4 and r["read_noise"]==1e-5]
        axes[1].loglog([r["beta"] for r in rows],[r["correction_mse"] for r in rows],marker="o")
    n=output[0]["states"]
    axes[0].loglog([1,4,16],[(n-1)/m for m in (1,4,16)],"k--",label="Ideal (n−1)/m")
    axes[0].axhline(1,color="grey",linewidth=.8)
    axes[0].set_xlabel("Probes per example")
    axes[0].set_ylabel("Correction MSE / baseline MSE")
    axes[0].set_title("No read noise; nudge amplitude 0.01")
    axes[0].legend(fontsize=8)
    axes[1].set_xlabel("Nudge amplitude")
    axes[1].set_ylabel("Correction MSE (four probes)")
    axes[1].set_title("Read noise standard deviation 1e−5")
    for ax in axes:
        ax.grid(alpha=.2,which="both")
    fig.savefig(root/"probe_budget_diagnostic.png",dpi=180)
    plt.close(fig)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root",type=Path)
    parser.add_argument("--trials",type=int,default=32)
    args=parser.parse_args()
    diagnose(args.root,args.trials)
