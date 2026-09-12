#!/usr/bin/env python3
"""Replay matched width/probe scaling runs and report accuracy versus cost."""

import argparse
import json
from pathlib import Path

import numpy as np

from labs.recurrent_eqprop import Classifier
from labs.tools.run_random_nudge_hopfield import write_csv, write_json
from labs.tools.summarize_recurrent_eqprop_digits import read_metrics
from labs.tools.train_recurrent_eqprop_digits import evaluate, probe_specification


def collect(paths):
    rows, checks, sources = [], [], []
    shared = None
    starts = {}
    shared_data = None
    comparable = ("epochs","learning_rate","batch_size","beta","read_noise","asymmetry",
                  "outputs","train_samples","validation_samples","test_samples")
    for path in paths:
        run = json.loads((path/"run.json").read_text())
        status = json.loads((path/"status.json").read_text())
        config = run["config"]
        if status["status"] != "complete" or status["completed_trajectories"] != status["expected_trajectories"]:
            raise ValueError(f"Expected completed declared coverage; got {path}: {status}")
        if shared is None:
            shared = config
        if any(shared[key] != config[key] for key in comparable):
            raise ValueError(f"Expected matched training configuration apart from width/probes; got {path}")
        size = config["size"]
        selected_methods = {"adjoint","mc8",f"mc{size//4}"}
        measured = read_metrics(path/"metrics.csv")
        if len(measured) != status["expected_trajectories"]*(config["epochs"]+1):
            raise ValueError(f"Expected complete epoch rows at {path}; got {len(measured)}")
        selected = [r for r in measured if r["method"] in selected_methods]
        for row in selected:
            row.update(size=size,source_run=str(path.resolve()))
            if row["max_force_residual"] > 1e-9:
                raise ValueError(f"Expected converged states; got {row}")
            method,seed = row["method"],int(row["seed"])
            spec = probe_specification(method,size)
            m = spec[2] if spec else 0
            phases = 1 if method=="adjoint" else 3+2*m
            expected_cost = row["epoch"]*config["train_samples"]*phases
            if row["equilibrations"] != expected_cost or row["state_reads"] != expected_cost:
                raise ValueError(f"Expected {expected_cost} training equilibrations; got {row}")
            if row["epoch"] != config["epochs"]:
                if "test_accuracy" in row or "test_loss" in row:
                    raise ValueError(f"Expected test metrics only at final epoch; got {row}")
                continue
            checkpoint = path/f"{method}_seed{seed}_lr{config['learning_rate']:g}.npz"
            with np.load(checkpoint) as v:
                model = Classifier(v["symmetric"],v["skew"],v["inputs"],v["bias"],
                                   outputs=int(v["outputs"]),cubic=float(v["cubic"]),
                                   logit_scale=float(v["logit_scale"]),symmetric_cap=float(v["symmetric_cap"]))
                if model.size != size or model.inputs.shape != (size-10,64):
                    raise ValueError(f"Expected configured network width; got {checkpoint}")
                if (not np.allclose(model.symmetric,model.symmetric.T) or np.any(np.diag(model.symmetric))
                        or np.linalg.norm(model.symmetric,2)>model.symmetric_cap+1e-12):
                    raise ValueError(f"Expected symmetric, zero-diagonal, bounded recurrent weights; got {checkpoint}")
                indices = [set(v[k].tolist()) for k in ("train_indices","val_indices","test_indices")]
                if any(indices[i]&indices[j] for i,j in ((0,1),(0,2),(1,2))):
                    raise ValueError(f"Expected disjoint data splits; got {checkpoint}")
                data = {k:v[k].copy() for k in ("train_indices","val_indices","test_indices","feature_mean","feature_std","train_x","train_y","val_x","val_y","test_x","test_y")}
                if shared_data is None:
                    shared_data = data
                elif any(not np.array_equal(data[k],shared_data[k]) for k in data):
                    raise ValueError(f"Expected the same dataset/preprocessing at every width; got {checkpoint}")
                initial = {k:v[k].copy() for k in ("initial_symmetric","initial_inputs","skew")}
                key = size,seed
                if key in starts and any(not np.array_equal(initial[k],starts[key][k]) for k in initial):
                    raise ValueError(f"Expected matched per-width/seed initialization; got {checkpoint}")
                starts[key] = initial
                changed = float(np.linalg.norm(model.symmetric-v["initial_symmetric"]))
                result = evaluate(model,v["test_x"],v["test_y"])
                error = abs(result["loss"]-row["test_loss"])
                if error>1e-12 or result["accuracy"]!=row["test_accuracy"] or changed<1e-6:
                    raise ValueError(f"Expected reproducible trained checkpoint; got {checkpoint}: {result}")
                checks.append(dict(size=size,method=method,seed=seed,test_accuracy=result["accuracy"],
                                   replay_loss_error=error,recurrent_change_norm=changed,checkpoint=str(checkpoint.resolve())))
        rows.extend(selected)
        sources.append(dict(path=str(path.resolve()),run=run,status=status,selected_methods=sorted({r["method"] for r in selected})))
    identities = [(r["size"],r["method"],r["seed"],r["epoch"]) for r in rows]
    if len(identities) != len(set(identities)):
        raise ValueError("Expected unique width/method/seed/epoch rows")
    for size in sorted({r["size"] for r in rows}):
        for method in {"adjoint","mc8",f"mc{size//4}"}:
            for seed in (0,1,2):
                epochs = {int(r["epoch"]) for r in rows if r["size"]==size and r["method"]==method and r["seed"]==seed}
                if epochs != set(range(shared["epochs"]+1)):
                    raise ValueError(f"Expected three-seed full coverage for n={size} {method} seed={seed}; got {epochs}")
    return rows,checks,sources,shared


def summarize(rows,config):
    last = [r for r in rows if r["epoch"]==config["epochs"]]
    results = []
    for size,method in dict.fromkeys((r["size"],r["method"]) for r in last):
        selected = [r for r in last if r["size"]==size and r["method"]==method]
        spec = probe_specification(method,size)
        m = spec[2] if spec else 0
        entry = dict(size=size,method=method,probes=m,parameter_count=size*(size-1)//2+(size-10)*64+size,
                     test_accuracy_mean=float(np.mean([r["test_accuracy"] for r in selected])),
                     test_accuracy_std=float(np.std([r["test_accuracy"] for r in selected])),
                     test_loss_mean=float(np.mean([r["test_loss"] for r in selected])),
                     validation_accuracy_mean=float(np.mean([r["validation_accuracy"] for r in selected])),
                     equilibrations_per_example=1 if method=="adjoint" else 3+2*m,
                     training_equilibrations=selected[0]["equilibrations"],seeds=len(selected),
                     elapsed_seconds_mean=float(np.mean([r["elapsed_seconds"] for r in selected])))
        for name in ("gradient_seconds","update_seconds"):
            if all(name in r for r in selected):
                entry[name+"_mean"] = float(np.mean([r[name] for r in selected]))
        for threshold in (0.9,0.95):
            hits = []
            for seed in (0,1,2):
                series = sorted((r for r in rows if r["size"]==size and r["method"]==method and r["seed"]==seed),key=lambda r:r["epoch"])
                first = next((r for r in series if r["validation_accuracy"]>=threshold),None)
                hits.append(dict(seed=seed,epoch=None if first is None else first["epoch"],
                                 equilibrations=None if first is None else first["equilibrations"]))
            entry[f"first_validation_{int(threshold*100)}percent"] = hits
        results.append(entry)
    return results


def verify_noise_control(path,primary_checks,primary_config):
    """Replay the separately declared, matched-probe effectively noiseless control."""
    run = json.loads((path/"run.json").read_text())
    status = json.loads((path/"status.json").read_text())
    config = run["config"]
    if status["status"]!="complete" or status["completed_trajectories"]!=3 or config["read_noise"]!=1e-30:
        raise ValueError(f"Expected completed three-seed 1e-30-noise control; got {path}")
    comparable = ("epochs","learning_rate","batch_size","beta","asymmetry","outputs",
                  "train_samples","validation_samples","test_samples")
    if any(config[k]!=primary_config[k] for k in comparable) or config["size"]!=256 or config["methods"]!=["mc8"]:
        raise ValueError(f"Expected matched n=256 mc8 control configuration; got {config}")
    rows = read_metrics(path/"metrics.csv")
    identities = {(int(r["seed"]),int(r["epoch"])) for r in rows}
    expected = {(seed,epoch) for seed in (0,1,2) for epoch in range(config["epochs"]+1)}
    if identities!=expected or len(rows)!=len(expected):
        raise ValueError(f"Expected complete unique noise-control rows; got {path}")
    checks = []
    for row in rows:
        cost = row["epoch"]*config["train_samples"]*19
        if row["equilibrations"]!=cost or row["state_reads"]!=cost or row["max_force_residual"]>1e-9:
            raise ValueError(f"Expected correct physical counts and converged states; got {row}")
        if row["epoch"]!=config["epochs"]:
            if "test_accuracy" in row or "test_loss" in row:
                raise ValueError("Expected no intermediate test evaluation in noise control")
            continue
        seed = int(row["seed"])
        original = next(r for r in primary_checks if r["size"]==256 and r["method"]=="mc8" and r["seed"]==seed)
        checkpoint = path/f"mc8_seed{seed}_lr{config['learning_rate']:g}.npz"
        with np.load(checkpoint) as v,np.load(original["checkpoint"]) as reference:
            for key in ("initial_symmetric","initial_inputs","skew","train_indices","val_indices","test_indices",
                        "feature_mean","feature_std","train_x","train_y","val_x","val_y","test_x","test_y"):
                if not np.array_equal(v[key],reference[key]):
                    raise ValueError(f"Expected matched initial state/data for noise control; got {key} at {checkpoint}")
            model = Classifier(v["symmetric"],v["skew"],v["inputs"],v["bias"],
                               outputs=int(v["outputs"]),cubic=float(v["cubic"]),
                               logit_scale=float(v["logit_scale"]),symmetric_cap=float(v["symmetric_cap"]))
            if (not np.allclose(model.symmetric,model.symmetric.T) or np.any(np.diag(model.symmetric))
                    or np.linalg.norm(model.symmetric,2)>model.symmetric_cap+1e-12):
                raise ValueError(f"Expected bounded symmetric recurrent weights; got {checkpoint}")
            result = evaluate(model,v["test_x"],v["test_y"])
            error = abs(result["loss"]-row["test_loss"])
            if error>1e-12 or result["accuracy"]!=row["test_accuracy"]:
                raise ValueError(f"Expected exact noise-control checkpoint replay; got {checkpoint}")
            checks.append(dict(seed=seed,noisy_test_accuracy=original["test_accuracy"],
                               effectively_clean_test_accuracy=result["accuracy"],replay_loss_error=error,
                               checkpoint=str(checkpoint.resolve())))
    clean = [r["effectively_clean_test_accuracy"] for r in checks]
    return dict(status="complete",comparison_group="separate exploratory noise control",run=run,
                checks=checks,test_accuracy_mean=float(np.mean(clean)),test_accuracy_std=float(np.std(clean)),
                paired_accuracy_change_mean=float(np.mean([r["effectively_clean_test_accuracy"]-r["noisy_test_accuracy"] for r in checks])))


def plots(output,rows,summary):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import NullLocator
    policies = (("adjoint","Exact adjoint"),("fixed","8 random-sign probes"),("proportional","n/4 random-sign probes"))
    colors = {"adjoint":"#1f77b4","fixed":"#ff7f0e","proportional":"#2ca02c"}
    sizes = sorted({int(r["size"]) for r in rows})
    fig,axes = plt.subplots(1,3,figsize=(16,4.5),constrained_layout=True)
    for policy,label in policies:
        selected = [next(r for r in summary if r["size"]==n and r["method"]==("adjoint" if policy=="adjoint" else "mc8" if policy=="fixed" else f"mc{n//4}")) for n in sizes]
        axes[0].errorbar(sizes,[100*r["test_accuracy_mean"] for r in selected],yerr=[100*r["test_accuracy_std"] for r in selected],fmt="o-",capsize=3,label=label,color=colors[policy])
        if policy!="adjoint":
            axes[1].plot(sizes,[r["equilibrations_per_example"] for r in selected],"o-",label=label,color=colors[policy])
    axes[0].set(xlabel="State dimension n",ylabel="Test accuracy (%)",title="15 epochs: mean ± SD, three seeds",ylim=(80,100))
    axes[0].legend(fontsize=8)
    axes[1].set(xlabel="State dimension n",ylabel="Equilibrations per example/update",title="One free + two error + 2m probe states")
    axes[1].legend(fontsize=8)
    for size in sizes:
        selected = [r for r in rows if r["size"]==size and r["method"]=="mc8"]
        epochs = sorted({r["epoch"] for r in selected})
        axes[2].plot(epochs,[100*np.mean([r["validation_accuracy"] for r in selected if r["epoch"]==e]) for e in epochs],label=f"n={size}")
    axes[2].set(xlabel="Epoch",ylabel="Validation accuracy (%)",title="Fixed 8 probes: learning curves",ylim=(0,100))
    axes[2].legend(fontsize=8)
    for ax in axes[:2]:
        ax.set(xscale="log",xticks=sizes,xticklabels=sizes)
        ax.xaxis.set_minor_locator(NullLocator())
    for ax in axes:
        ax.spines[["top","right"]].set_visible(False)
    fig.savefig(output/"scaling.png",dpi=180)
    fig.savefig(output/"scaling.svg")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs",type=Path,nargs="+",required=True)
    parser.add_argument("--noise-control-run",type=Path)
    parser.add_argument("--output",type=Path,required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"Expected a new output directory; exists: {args.output}")
    rows,checks,sources,config = collect(args.runs)
    summary = summarize(rows,config)
    noise_control = None if args.noise_control_run is None else verify_noise_control(args.noise_control_run,checks,config)
    args.output.mkdir(parents=True)
    write_csv(args.output/"metrics.csv",rows)
    write_json(args.output/"sources.json",sources)
    write_json(args.output/"summary.json",summary)
    write_json(args.output/"verification.json",dict(status="complete",rows=len(rows),checkpoints=checks,
               max_force_residual=max(r["max_force_residual"] for r in rows)))
    if noise_control is not None:
        write_json(args.output/"noise_control_verification.json",noise_control)
    plots(args.output,rows,summary)
    lines = ["# Random-sign Monte Carlo scaling", "", "Exploratory width comparison. Mean ± population SD across three seeds; common learning rate, 15 epochs.", "",
             "| States | Parameters | Method | Test accuracy | Equilibrations/example/update |", "|---:|---:|---|---:|---:|"]
    for row in sorted(summary,key=lambda r:(r["size"],r["method"])):
        lines.append(f"| {row['size']} | {row['parameter_count']:,} | {row['method']} | {100*row['test_accuracy_mean']:.2f}% ± {100*row['test_accuracy_std']:.2f} pp | {row['equilibrations_per_example']} |")
    lines += ["", "The oracle also requires a dense adjoint solve; operation counts exclude that work and all evaluation/audits.",
              "The 32-state controls are reused from the earlier completed experiment. At that size mc8 also represents n/4 probes.",
              "Wall time includes CPU implementation overhead and concurrent host load; it is not hardware timing.",
              "summary.json records first validation-threshold hits; unreached targets are null, not excluded from the seed count."]
    if noise_control is not None:
        lines += ["",f"Separate n=256 eight-probe effectively zero-noise control: {100*noise_control['test_accuracy_mean']:.2f}% ± {100*noise_control['test_accuracy_std']:.2f} pp, three matched seeds.",
                  "Read noise 1e-30 keeps RNG consumption identical to the 1e-5 runs while removing its numerical effect; this follow-up was added after the primary fixed-eight-probe trajectories completed."]
    (args.output/"report.md").write_text("\n".join(lines)+"\n")
    print(f"Validated {len(checks)} primary checkpoints and {0 if noise_control is None else len(noise_control['checks'])} noise controls, {len(rows)} primary epoch rows at {args.output.resolve()}")


if __name__=="__main__":
    main()
