#!/usr/bin/env python3
"""Separate Monte Carlo sampling/noise scaling from nonlinear training results."""

import argparse
import json
import os
from pathlib import Path
import platform
import shlex
import subprocess
import sys
import time
from datetime import datetime, timezone

import numpy as np

from labs.adjoint_estimators import estimate_adjoint, make_probes
from labs.recurrent_eqprop import cost_gradient, flatten_gradient, make_classifier, parameter_gradient, settle
from labs.tools.run_random_nudge_hopfield import cosine, write_csv, write_json
from labs.tools.train_recurrent_eqprop_digits import dataset, oracle_adjoint


def expected_mc_mse(size, count, discrepancy_sq, baseline_variance=0.0, reading_variance=0.0):
    """IID-sign result including the single noisy baseline shared by all probes.

    Baseline noise is independent/isotropic with variance per state coordinate;
    scalar reading noise is independent between probe pairs and of the baseline.
    The discrepancy is relative to the clean baseline.
    """
    if size < 1 or count < 1 or min(discrepancy_sq, baseline_variance, reading_variance) < 0:
        raise ValueError("Expected positive dimensions and nonnegative squared discrepancy/variances")
    return ((size-1)/count)*(discrepancy_sq+size*baseline_variance) + size**2/count*reading_variance


def statistical_check(sizes, trials, rng, progress, output):
    """Use synthetic exact projection equations; no simulated hardware claimed."""
    rows = []
    amplitude = 0.01
    for size in sizes:
        truth = rng.normal(size=size)
        truth /= np.linalg.norm(truth)  # unit discrepancy, zero clean q; ||c||=1
        for count in sorted({8, size//4}):
            samples = {sigma: [] for sigma in (0.0, 1e-5, 1e-4)}
            for begin in range(0, trials, 4):
                chunk = min(4, trials-begin)
                probes = make_probes(rng, size, chunk*count).reshape(chunk,count,size)
                exact_y = probes @ truth
                for sigma, errors in samples.items():
                    sd = sigma/(np.sqrt(2)*amplitude)
                    q = rng.normal(scale=sd,size=(chunk,size))
                    y = exact_y+rng.normal(scale=sd,size=(chunk,count))
                    estimate = estimate_adjoint(q,probes,y,"mc")
                    errors.extend(np.sum((estimate-truth)**2,axis=-1).tolist())
            for sigma, errors in samples.items():
                variance = sigma**2/(2*amplitude**2)
                predicted = expected_mc_mse(size,count,1.0,variance,variance)
                rows.append(dict(size=size,probes=count,read_noise=sigma,amplitude=amplitude,
                                 trials=trials,empirical_mse=float(np.mean(errors)),
                                 mse_standard_error=float(np.std(errors,ddof=1)/np.sqrt(trials)),
                                 predicted_mse=predicted,empirical_over_predicted=float(np.mean(errors)/predicted),
                                 equilibrations=3+2*count,measurement_model="synthetic exact projections"))
            write_csv(output/"statistical.csv",rows)
            progress(f"statistical n={size}, m={count}, {trials} draws",True)
    return rows


def frozen_gradients(sizes, seeds, trials, rng, progress, output):
    """Oracle local responses isolate sampling variance after parameter mapping.

    Each draw uses independent designs for every example. Prefixes share their
    measured estimates. Reference states/adjoints stay fixed across all draws.
    No noisy voltage or finite-amplitude error is added to this diagnostic.
    """
    data = dataset()
    batch_sizes = (1,8,32,96)
    rows, references = [], []
    for size in sizes:
        for seed in seeds:
            model = make_classifier(seed,size=size)
            order = np.random.default_rng(10000+seed).permutation(len(data["train_x"]))[:96]
            x,y = data["train_x"][order],data["train_y"][order]
            net = model.network()
            free = settle(net,model.drive(x)).state
            c = cost_gradient(free,y,model.hidden,model.logit_scale)
            truth = oracle_adjoint(net,free,c)
            baseline = np.linalg.solve(net.jacobian(free),c[...,None])[...,0]
            delta_sq = np.sum((truth-baseline)**2)
            true_gradients = {b:flatten_gradient(parameter_gradient(model,x[:b],free[:b],truth[:b])) for b in batch_sizes}
            references.append(dict(size=size,seed=seed,baseline_adjoint_relative_error=float(np.sqrt(delta_sq/np.sum(truth**2))),
                                   mean_state_rms=float(np.sqrt(np.mean(free**2))),
                                   parameter_count=size*(size-1)//2+(size-10)*64+size))
            designs = [("random_sign",8),("random_sign",size//4),("hadamard",8)]
            for design,count in dict.fromkeys(designs):
                records = {b:[] for b in batch_sizes}
                for draw in range(trials):
                    probes = np.stack([make_probes(rng,size,count,design) for _ in x])
                    readings = np.einsum("bmn,bn->bm",probes,truth)
                    estimate = estimate_adjoint(baseline,probes,readings,"mc")
                    for b in batch_sizes:
                        gradient = flatten_gradient(parameter_gradient(model,x[:b],free[:b],estimate[:b]))
                        reference = true_gradients[b]
                        records[b].append((np.sum((estimate[:b]-truth[:b])**2)/np.sum((baseline[:b]-truth[:b])**2),
                                           np.sum((gradient-reference)**2),cosine(gradient,reference)))
                for b,values in records.items():
                    values = np.asarray(values)
                    reference_sq = np.sum(true_gradients[b]**2)
                    rows.append(dict(size=size,seed=seed,design=design,probes=count,batch_size=b,trials=trials,
                                     adjoint_mse_over_baseline=float(values[:,0].mean()),
                                     gradient_mse=float(values[:,1].mean()),
                                     gradient_relative_rmse=float(np.sqrt(values[:,1].mean()/reference_sq)),
                                     gradient_cosine_mean=float(values[:,2].mean()),
                                     gradient_cosine_std=float(values[:,2].std()),
                                     reference_gradient_sq=float(reference_sq),
                                     response_source="oracle exact local linear response"))
                write_csv(output/"frozen_gradients.csv",rows)
                write_csv(output/"frozen_references.csv",references)
                progress(f"frozen network n={size}, seed={seed}, {design} m={count}",True)
    return rows


def plots(output,statistical,frozen):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,axes = plt.subplots(1,3,figsize=(16,4.5),constrained_layout=True)
    for sigma in (0.0,1e-5,1e-4):
        rows = [r for r in statistical if r["probes"]==8 and r["read_noise"]==sigma]
        line, = axes[0].loglog([r["size"] for r in rows],[r["empirical_mse"] for r in rows],"o",label=f"read σ={sigma:g}")
        axes[0].loglog([r["size"] for r in rows],[r["predicted_mse"] for r in rows],color=line.get_color())
    axes[0].set(xlabel="State dimension n",ylabel="Squared adjoint error / clean discrepancy²",title="8 probes: points measured, lines predicted")
    axes[0].legend(fontsize=8)
    policies = (("random_sign","fixed", "8 random-sign probes"),("random_sign","proportional","n/4 random-sign probes"),("hadamard","fixed","8 orthogonal MC probes"))
    sizes = sorted({r["size"] for r in frozen})
    for design,policy,label in policies:
        selected = [r for r in frozen if r["design"]==design and r["probes"]==(8 if policy=="fixed" else r["size"]//4) and r["batch_size"]==96]
        means = [np.mean([r["gradient_cosine_mean"] for r in selected if r["size"]==size]) for size in sizes]
        axes[1].plot(sizes,means,"o-",label=label)
    axes[1].set(xlabel="State dimension n",ylabel="Mean batch gradient cosine",title="96 examples, ideal local responses",ylim=(-0.05,1.05),xscale="log",xticks=sizes,xticklabels=sizes)
    axes[1].legend(fontsize=8)
    for size in sizes:
        selected = [r for r in frozen if r["size"]==size and r["design"]=="random_sign" and r["probes"]==8]
        batches = sorted({r["batch_size"] for r in selected})
        values = [np.sqrt(np.mean([r["gradient_relative_rmse"]**2 for r in selected if r["batch_size"]==b])) for b in batches]
        axes[2].loglog(batches,values,"o-",label=f"n={size}")
    axes[2].set(xlabel="Minibatch size",ylabel="Relative batch gradient RMSE",title="8 probes: independent designs per example")
    axes[2].legend(fontsize=8)
    for ax in axes:
        ax.spines[["top","right"]].set_visible(False)
    fig.savefig(output/"diagnostics.png",dpi=180)
    fig.savefig(output/"diagnostics.svg")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--statistical-sizes",type=int,nargs="+",default=[32,64,128,256,512,1024,2048,4096])
    parser.add_argument("--network-sizes",type=int,nargs="+",default=[32,64,128,256,512])
    parser.add_argument("--seeds",type=int,nargs="+",default=[0,1,2])
    parser.add_argument("--statistical-trials",type=int,default=128)
    parser.add_argument("--gradient-trials",type=int,default=32)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"Expected a new output directory; exists: {args.output}")
    if min(args.statistical_trials,args.gradient_trials)<2 or any(n<16 or n&(n-1) for n in args.statistical_sizes+args.network_sizes):
        parser.error("Expected power-of-two sizes >= 16 and at least two trials")
    args.output.mkdir(parents=True)
    expected = sum(len({8,n//4}) for n in args.statistical_sizes)+len(args.seeds)*sum(len({("random_sign",8),("random_sign",n//4),("hadamard",8)}) for n in args.network_sizes)
    start = time.time()
    config = {key:str(value) if isinstance(value,Path) else value for key,value in vars(args).items()}
    run = dict(config=config,pid=os.getpid(),expected_cases=expected,evidence_tier="exploratory",
               command=shlex.join([sys.executable,"-m","labs.tools.measure_mc_scaling",*sys.argv[1:]]),
               commit=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
               source_status=subprocess.check_output(["git","status","--short"],text=True),
               python=sys.version,numpy=np.__version__,platform=platform.platform(),
               started_utc=datetime.now(timezone.utc).isoformat())
    write_json(args.output/"run.json",run)
    state = dict(status="running",pid=os.getpid(),expected_cases=expected,completed_cases=0)

    def progress(detail,advance):
        state["completed_cases"] += int(advance)
        state.update(detail=detail,elapsed_seconds=time.time()-start,heartbeat_utc=datetime.now(timezone.utc).isoformat())
        write_json(args.output/"status.json",state)
        line = f"[{time.time()-start:.1f}s] {state['completed_cases']}/{expected} {detail}"
        print(line,flush=True)
        with (args.output/"run.log").open("a") as stream:
            stream.write(line+"\n")

    progress("starting ideal statistical diagnostics",False)
    try:
        statistical = statistical_check(args.statistical_sizes,args.statistical_trials,np.random.default_rng(20260912),progress,args.output)
        frozen = frozen_gradients(args.network_sizes,args.seeds,args.gradient_trials,np.random.default_rng(20260913),progress,args.output)
        if state["completed_cases"] != expected:
            raise RuntimeError(f"Expected {expected} completed cases; got {state['completed_cases']}")
        for row in statistical+frozen:
            if any(not np.isfinite(v) for v in row.values() if isinstance(v,(int,float))):
                raise RuntimeError(f"Expected finite metrics; got {row}")
        plots(args.output,statistical,frozen)
        state.update(status="complete",elapsed_seconds=time.time()-start,statistical_rows=len(statistical),gradient_rows=len(frozen))
        write_json(args.output/"status.json",state)
    except BaseException as error:
        state.update(status="failed",error=f"{type(error).__name__}: {error}")
        write_json(args.output/"status.json",state)
        raise


if __name__ == "__main__":
    main()
