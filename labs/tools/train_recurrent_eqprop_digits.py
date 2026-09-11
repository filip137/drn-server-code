#!/usr/bin/env python3
"""Compare physical adjoint estimators on recurrent handwritten-digit learning.

Uses the bundled sklearn 8x8 digits dataset. Outputs, checkpoints and provenance
live under a fresh ignored run directory. No dataset download or GPU is needed.
"""

import argparse
import copy
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
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import sklearn

from labs.adjoint_estimators import estimate_adjoint, make_probes
from labs.recurrent_eqprop import (
    Classifier, contrastive_gradient, cost_gradient, error_pair, flatten_gradient,
    loss_accuracy, make_classifier, parameter_gradient, probe_responses, settle,
)
from labs.tools.run_random_nudge_hopfield import cosine, write_csv, write_json


METHODS = ["adjoint", "contrastive_ep", "known_skew_asymep", "mc8", "kaczmarz8",
           "lstsq8", "ridge8", "orthogonal8", "orthogonal32"]
# Separate exploratory follow-up; preserve the original nine-method default.
METHOD_CHOICES = METHODS + ["orthogonal_mc8"]


def dataset(quick=False):
    x, y = load_digits(return_X_y=True)
    indices = np.arange(len(y))
    remaining, test = train_test_split(indices, test_size=0.2, stratify=y, random_state=20260911)
    train, validation = train_test_split(remaining, test_size=0.2, stratify=y[remaining], random_state=20260912)
    if quick:
        train, _ = train_test_split(train, train_size=192, stratify=y[train], random_state=1)
        validation, _ = train_test_split(validation, train_size=64, stratify=y[validation], random_state=1)
        test, _ = train_test_split(test, train_size=64, stratify=y[test], random_state=1)
    x = x / 16.0
    mean = x[train].mean(axis=0)
    std = np.maximum(x[train].std(axis=0), 0.1)
    x = (x - mean)/std
    return dict(train_x=x[train], train_y=y[train], val_x=x[validation], val_y=y[validation],
                test_x=x[test], test_y=y[test], train_indices=train, val_indices=validation,
                test_indices=test, feature_mean=mean, feature_std=std)


def oracle_adjoint(network, state, c):
    j = network.jacobian(state)
    return np.linalg.solve(np.swapaxes(j, -1, -2), c[..., None])[..., 0]


def gradient_step(model, x, labels, method, rng, *, beta, sigma, tolerance=1e-9, audit=False):
    """Produce a gradient; corrected methods receive measured q and y only.

    Any dense derivative below is either the named oracle or a post-estimation
    audit. Disabling audit removes it completely from physical-method steps.
    """
    net = model.network()
    drive = model.drive(x)
    free = settle(net, drive, tolerance=tolerance)
    c = cost_gradient(free.state, labels, model.hidden, model.logit_scale)
    metadata = dict(equilibrations=free.equilibrations, state_reads=free.equilibrations,
                    relaxation_iterations=free.iterations, max_residual=free.residual,
                    probe_count=0, total_probe_excitation_sq=0.0,
                    mean_free_error_force_norm=float(np.linalg.norm(c, axis=-1).mean()))
    if method == "adjoint":
        feedback = oracle_adjoint(net, free.state, c)
        gradient = parameter_gradient(model, x, free.state, feedback)
    else:
        pair = error_pair(model, net, x, labels, free.state, beta,
                          known_skew=method == "known_skew_asymep", sigma=sigma, rng=rng, tolerance=tolerance)
        feedback = pair.response
        metadata["equilibrations"] += pair.equilibrations
        metadata["state_reads"] += pair.equilibrations
        metadata["relaxation_iterations"] += pair.iterations
        metadata["max_residual"] = max(metadata["max_residual"], pair.residual)
        # Initial +/- error current has the same norm cap as one probe current.
        metadata["total_error_excitation_sq"] = float(2*np.sum((pair.effective_beta*np.linalg.norm(c, axis=-1))**2))
        if method in ("contrastive_ep", "known_skew_asymep"):
            gradient = contrastive_gradient(model, x, pair)
        else:
            if method == "orthogonal_mc8":
                estimator, design, m = "mc", "hadamard", 8
            elif method.startswith("orthogonal"):
                estimator, design = "orthogonal", "hadamard"
                m = int(method.removeprefix("orthogonal"))
            else:
                estimator = method.rstrip("0123456789")
                design, m = "random_sign", int(method[len(estimator):])
            probes = np.stack([make_probes(rng, model.size, m, design) for _ in range(len(x))])
            responses, equilibrations, iterations, residual = probe_responses(
                net, drive, free.state, probes, beta, sigma=sigma, rng=rng, tolerance=tolerance)
            readings = np.einsum("bmn,bn->bm", responses, c)
            ridge = 0.1*m/model.size if estimator == "ridge" else 0.0
            feedback = estimate_adjoint(feedback, probes, readings, estimator, ridge=ridge)
            gradient = parameter_gradient(model, x, free.state, feedback)
            metadata["equilibrations"] += equilibrations
            metadata["state_reads"] += equilibrations
            metadata["relaxation_iterations"] += iterations
            metadata["max_residual"] = max(metadata["max_residual"], residual)
            metadata["probe_count"] = m
            metadata["total_probe_excitation_sq"] = 2*len(x)*m*beta**2
    if audit:
        exact = oracle_adjoint(net, free.state, c)
        exact_gradient = flatten_gradient(parameter_gradient(model, x, free.state, exact))
        actual_gradient = flatten_gradient(gradient)
        j = net.jacobian(free.state)
        metadata.update(
            gradient_cosine=cosine(actual_gradient, exact_gradient),
            gradient_relative_error=float(np.linalg.norm(actual_gradient-exact_gradient)/max(np.linalg.norm(exact_gradient),1e-20)),
            adjoint_relative_error=float(np.linalg.norm(feedback-exact)/max(np.linalg.norm(exact),1e-20)),
            realized_jacobian_asymmetry=float(np.mean(np.linalg.norm(j-np.swapaxes(j,-1,-2),axis=(-2,-1))/np.linalg.norm(j,axis=(-2,-1)))),
        )
    return gradient, metadata


def evaluate(model, x, y):
    settled = settle(model.network(), model.drive(x))
    loss, accuracy = loss_accuracy(settled.state, y, model.hidden, model.logit_scale)
    return dict(loss=loss, accuracy=accuracy, residual=settled.residual)


def train_one(data, method, seed, config, output, progress, *, calibration=False):
    model = make_classifier(seed, asymmetry=config["asymmetry"])
    shuffle = np.random.default_rng(10000+seed)
    probes = np.random.default_rng(20000+seed)
    cumulative = dict(equilibrations=0, state_reads=0, relaxation_iterations=0,
                      total_probe_excitation_sq=0.0, total_error_excitation_sq=0.0)
    rows = []
    start = time.time()
    projected = 0
    initial_model = copy.deepcopy(model)
    for epoch in range(config["epochs"]+1):
        train = evaluate(model, data["train_x"], data["train_y"])
        validation = evaluate(model, data["val_x"], data["val_y"])
        row = dict(method=method, seed=seed, epoch=epoch, learning_rate=config["learning_rate"],
                   train_loss=train["loss"], train_accuracy=train["accuracy"],
                   validation_loss=validation["loss"], validation_accuracy=validation["accuracy"],
                   max_force_residual=max(train["residual"],validation["residual"]),
                   projected_updates=projected, elapsed_seconds=time.time()-start, **cumulative)
        if epoch == config["epochs"]:
            if not calibration:
                test = evaluate(model, data["test_x"], data["test_y"])
                row.update(test_loss=test["loss"], test_accuracy=test["accuracy"])
            rows.append(row)
            break
        order = shuffle.permutation(len(data["train_x"]))
        epoch_residual = 0.0
        for batch_index, begin in enumerate(range(0,len(order),config["batch_size"])):
            indices = order[begin:begin+config["batch_size"]]
            gradient, meta = gradient_step(model, data["train_x"][indices],data["train_y"][indices],
                                            method, probes, beta=config["beta"], sigma=config["read_noise"],
                                            audit=batch_index==0 and config["audit"])
            projected += model.update(gradient, config["learning_rate"])
            for key in cumulative:
                cumulative[key] += meta.get(key,0)
            epoch_residual = max(epoch_residual,meta["max_residual"])
            if batch_index == 0:
                for key in ("gradient_cosine","gradient_relative_error","adjoint_relative_error","realized_jacobian_asymmetry"):
                    if key in meta:
                        row[key] = meta[key]
        row["max_force_residual"] = max(row["max_force_residual"], epoch_residual)
        rows.append(row)
        write_csv(output / "current.csv",rows)
        progress(f"{method} seed={seed}, epoch={epoch+1}/{config['epochs']}, val={validation['accuracy']:.3f}", advance=False)
    np.savez_compressed(output / f"{method}_seed{seed}_lr{config['learning_rate']:g}.npz",
                        symmetric=model.symmetric, skew=model.skew, inputs=model.inputs,bias=model.bias,
                        initial_symmetric=initial_model.symmetric,initial_inputs=initial_model.inputs,
                        outputs=model.outputs,cubic=model.cubic,logit_scale=model.logit_scale,
                        symmetric_cap=model.symmetric_cap,**data)
    progress(f"{method} seed={seed} finished: validation={row['validation_accuracy']:.3f}", advance=True)
    return rows


def report(output, rows, config, calibration):
    last = [row for row in rows if row["epoch"] == config["epochs"]]
    if calibration:
        best = max(last,key=lambda r:(r["validation_accuracy"],-r["validation_loss"]))
        summary = dict(kind="validation-only learning-rate calibration", selected_learning_rate=best["learning_rate"], trials=last)
    else:
        summary = dict(kind="exploratory recurrent digits classification", methods=[])
        for method in dict.fromkeys(row["method"] for row in last):
            selected = [row for row in last if row["method"] == method]
            summary["methods"].append(dict(method=method,seeds=len(selected),
                test_accuracy_mean=float(np.mean([r["test_accuracy"] for r in selected])),
                test_accuracy_std=float(np.std([r["test_accuracy"] for r in selected])),
                test_loss_mean=float(np.mean([r["test_loss"] for r in selected])),
                validation_accuracy_mean=float(np.mean([r["validation_accuracy"] for r in selected])),
                training_equilibrations_mean=float(np.mean([r["equilibrations"] for r in selected])),
                elapsed_seconds_mean=float(np.mean([r["elapsed_seconds"] for r in selected]))))
    write_json(output / "summary.json", summary)
    if calibration:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1,3,figsize=(16,4.5),constrained_layout=True)
    for method in dict.fromkeys(row["method"] for row in rows):
        selected = [row for row in rows if row["method"] == method]
        epochs = sorted({r["epoch"] for r in selected})
        values = np.array([[r["validation_accuracy"] for r in selected if r["epoch"] == ep] for ep in epochs])
        means, stds = values.mean(axis=1),values.std(axis=1)
        line, = axes[0].plot(epochs,means,label=method)
        axes[0].fill_between(epochs,means-stds,means+stds,color=line.get_color(),alpha=0.12)
        costs = [np.mean([r["equilibrations"] for r in selected if r["epoch"] == ep]) for ep in epochs]
        axes[1].plot(costs,means,label=method)
        audited = [r for r in selected if "gradient_cosine" in r]
        if audited:
            steps = sorted({r["epoch"] for r in audited})
            axes[2].plot(steps,[np.mean([r["gradient_cosine"] for r in audited if r["epoch"] == ep]) for ep in steps],label=method)
    axes[0].set(xlabel="Training epoch",ylabel="Validation accuracy",title="Digits: mean ± SD across seeds",ylim=(0,1.02))
    axes[1].set(xlabel="Training equilibrations",ylabel="Validation accuracy",title="Physical work (oracle solves excluded)",ylim=(0,1.02))
    axes[2].set(xlabel="Training epoch",ylabel="Batch parameter-gradient cosine",title="First minibatch audit each epoch",ylim=(-1.05,1.05))
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=5, fontsize=8)
    for ax in axes:
        ax.spines[["top","right"]].set_visible(False)
    fig.savefig(output / "learning.png",dpi=180)
    fig.savefig(output / "learning.svg")
    plt.close(fig)
    lines = ["# Recurrent EqProp digits comparison", "", "Exploratory simulation. Test set evaluated only at the final epoch.", "",
             "All methods train symmetric recurrent weights, hidden input weights, and biases, with fixed skew coupling.", "",
             "| Feedback/update | Test accuracy, mean ± SD | Test cross entropy | Training equilibrations |",
             "|---|---:|---:|---:|"]
    for row in summary["methods"]:
        lines.append(f"| {row['method']} | {100*row['test_accuracy_mean']:.2f}% ± {100*row['test_accuracy_std']:.2f}% | {row['test_loss_mean']:.4f} | {row['training_equilibrations_mean']:,.0f} |")
    lines += ["", "Equilibration counts exclude validation/test evaluation, Jacobian audits, and dense adjoint solve work.",
              "Known-skew AsymEP assumes access to the model's skew couplings; generic probe estimators do not.",
              "Probe methods use a common learning rate and predeclared ridge, not individually optimized hyperparameters."]
    (output / "report.md").write_text("\n".join(lines)+"\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--quick",action="store_true")
    parser.add_argument("--calibrate",action="store_true",help="Oracle validation-only LR screen; never evaluate held-out test")
    parser.add_argument("--epochs",type=int)
    parser.add_argument("--learning-rate",type=float,default=0.1)
    parser.add_argument("--seeds",type=int,nargs="+",default=[0,1,2])
    parser.add_argument("--methods",nargs="+",choices=METHOD_CHOICES,default=METHODS)
    parser.add_argument("--beta",type=float,default=0.01)
    parser.add_argument("--read-noise",type=float,default=1e-5)
    parser.add_argument("--asymmetry",type=float,default=1.0)
    parser.add_argument("--no-audit",action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"Expected a new output directory; exists: {args.output}")
    epochs = args.epochs if args.epochs is not None else (2 if args.quick else (8 if args.calibrate else 15))
    if epochs < 1 or not np.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error("Expected positive epochs and finite positive learning rate")
    data = dataset(args.quick)
    config = dict(epochs=epochs,learning_rate=args.learning_rate,batch_size=96,
                  beta=args.beta,read_noise=args.read_noise,asymmetry=args.asymmetry,audit=not args.no_audit,
                  seeds=[0] if args.quick or args.calibrate else args.seeds,methods=args.methods,
                  dataset="sklearn bundled digits, train-only standardization",size=32,hidden=22,outputs=10,
                  train_samples=len(data["train_y"]),validation_samples=len(data["val_y"]),test_samples=len(data["test_y"]),
                  evidence_tier="exploratory, non-canonical",ridge_rule="0.1*m/n",probe_norm="L2=1")
    if args.calibrate:
        cases = [("adjoint",0,lr) for lr in (0.01,0.03,0.1,0.3)]
    else:
        cases = [(method,seed,args.learning_rate) for seed in config["seeds"] for method in config["methods"]]
    args.output.mkdir(parents=True)
    root = Path(__file__).resolve().parents[2]
    start = time.time()
    run = dict(config=config,expected_trajectories=len(cases),pid=os.getpid(),launcher="foreground CPU Python",
               command=shlex.join([sys.executable,"-m","labs.tools.train_recurrent_eqprop_digits",*sys.argv[1:]]),
               commit=subprocess.check_output(["git","rev-parse","HEAD"],cwd=root,text=True).strip(),
               source_status=subprocess.check_output(["git","status","--short"],cwd=root,text=True),
               python=sys.version,numpy=np.__version__,sklearn=sklearn.__version__,platform=platform.platform(),
               started_utc=datetime.now(timezone.utc).isoformat(),heartbeat="each epoch and trajectory",
               recovery="preserve failed output, diagnose, rerun same scientific config in new directory")
    write_json(args.output / "run.json",run)
    state = dict(status="running",expected_trajectories=len(cases),completed_trajectories=0,pid=os.getpid())

    def progress(detail,advance):
        state["completed_trajectories"] += int(advance)
        state.update(detail=detail,elapsed_seconds=time.time()-start,heartbeat_utc=datetime.now(timezone.utc).isoformat())
        write_json(args.output / "status.json",state)
        line=f"[{time.time()-start:.1f}s] {state['completed_trajectories']}/{len(cases)} {detail}"
        print(line,flush=True)
        with (args.output / "run.log").open("a") as stream:
            stream.write(line+"\n")

    write_json(args.output / "status.json",state)
    rows = []
    try:
        for method,seed,lr in cases:
            case_config=dict(config,learning_rate=lr)
            rows.extend(train_one(data,method,seed,case_config,args.output,progress,calibration=args.calibrate))
            write_csv(args.output / "metrics.csv",rows)
        assert state["completed_trajectories"] == len(cases)
        assert len(rows) == len(cases)*(epochs+1)
        report(args.output,rows,config,args.calibrate)
        state.update(status="complete",elapsed_seconds=time.time()-start,terminal_result="summary.json")
        write_json(args.output / "status.json",state)
        print(f"Complete: {args.output.resolve()} ({time.time()-start:.1f}s)",flush=True)
    except BaseException as error:
        state.update(status="failed",error=f"{type(error).__name__}: {error}",elapsed_seconds=time.time()-start)
        write_json(args.output / "status.json",state)
        raise


if __name__ == "__main__":
    main()
