#!/usr/bin/env python3
"""Replay completed improvement runs, check budgets, and compare saved controls."""

import argparse
import json
from pathlib import Path

import numpy as np

from labs.adjoint_baselines import MeasuredBaseline, local_slope_baseline
from labs.recurrent_eqprop import Classifier, cost_gradient, settle
from labs.tools.run_random_nudge_hopfield import write_csv, write_json
from labs.tools.summarize_recurrent_eqprop_digits import read_metrics
from labs.tools.test_nudge_improvements import METHODS, select_configurations
from labs.tools.train_recurrent_eqprop_digits import evaluate, probe_specification


def phase_count(method, size=256):
    if method == "adjoint":
        return 1
    probes = probe_specification(method, size)[2]
    return (1 if method.startswith(("local_mc", "learned_mc")) else 3) + 2*probes


class Collector:
    def __init__(self):
        self.sources = {}
        self.checks = {}
        self.shared_data = None
        self.initial = {}

    def read(self, path, *, screen=False):
        run = json.loads((path / "run.json").read_text())
        status = json.loads((path / "status.json").read_text())
        config = run["config"]
        expected = status["expected_trajectories"]
        if status["status"] != "complete" or status["completed_trajectories"] != expected:
            raise ValueError(f"Expected completed trajectory coverage; got {path}: {status}")
        conditions = dict(size=256, batch_size=96, beta=0.01, read_noise=1e-5,
                          asymmetry=1.0, outputs=10, train_samples=1149,
                          validation_samples=288, test_samples=360, epochs=8 if screen else 15)
        if any(config.get(key) != value for key, value in conditions.items()):
            raise ValueError(f"Expected the declared matched experiment config; got {path}: {config}")
        rows = read_metrics(path / "metrics.csv")
        if len(rows) != expected*(config["epochs"]+1):
            raise ValueError(f"Expected all epoch rows; got {path}: {len(rows)}")
        cases = {}
        for row in rows:
            row.setdefault("momentum", 0.0)
            identity = row["method"], int(row["seed"]), row["learning_rate"], row["momentum"]
            cases.setdefault(identity, []).append(row)
            cost = row["epoch"]*1149*phase_count(row["method"])
            if row["equilibrations"] != cost or row["state_reads"] != cost or row["max_force_residual"] > 1e-9:
                raise ValueError(f"Expected correct counts and converged forces; got {row}")
            if (screen or row["epoch"] != config["epochs"]) and any(k.startswith("test_") for k in row):
                raise ValueError(f"Expected test metrics only for final confirmations; got {row}")
            row["source_run"] = str(path.resolve())
        if len(cases) != expected:
            raise ValueError(f"Expected {expected} distinct trajectories; got {len(cases)}")
        for identity, series in cases.items():
            if len(series) != config["epochs"]+1 or {r["epoch"] for r in series} != set(range(config["epochs"]+1)):
                raise ValueError(f"Expected unique complete epoch coverage; got {identity}")
            final = next(r for r in series if r["epoch"] == config["epochs"])
            method, seed, lr, _ = identity
            checkpoint = Path(final.get("checkpoint", path / f"{method}_seed{seed}_lr{lr:g}.npz"))
            if not checkpoint.is_absolute():
                checkpoint = Path(__file__).resolve().parents[2] / checkpoint
            self.replay(checkpoint, final, screen=screen)
        self.sources[str(path.resolve())] = dict(run=run, status=status)
        return rows

    def replay(self, checkpoint, row, *, screen):
        if str(checkpoint) in self.checks:
            return
        with np.load(checkpoint) as v:
            model = Classifier(v["symmetric"], v["skew"], v["inputs"], v["bias"],
                               outputs=int(v["outputs"]), cubic=float(v["cubic"]),
                               logit_scale=float(v["logit_scale"]), symmetric_cap=float(v["symmetric_cap"]))
            if (model.size != 256 or model.inputs.shape != (246, 64)
                    or not np.allclose(model.symmetric, model.symmetric.T, atol=1e-13)
                    or np.any(np.diag(model.symmetric))
                    or np.linalg.norm(model.symmetric, 2) > model.symmetric_cap+1e-12):
                raise ValueError(f"Expected configured stable symmetric parameters; got {checkpoint}")
            keys = ("train_x", "train_y", "val_x", "val_y", "test_x", "test_y",
                    "train_indices", "val_indices", "test_indices", "feature_mean", "feature_std")
            data = {k: v[k].copy() for k in keys}
            if self.shared_data is None:
                self.shared_data = data
            elif any(not np.array_equal(data[k], self.shared_data[k]) for k in keys):
                raise ValueError(f"Expected matched data/preprocessing; got {checkpoint}")
            indices = [set(data[k].tolist()) for k in ("train_indices", "val_indices", "test_indices")]
            if any(indices[i] & indices[j] for i, j in ((0,1), (0,2), (1,2))):
                raise ValueError(f"Expected disjoint dataset splits; got {checkpoint}")
            start = {k: v[k].copy() for k in ("initial_symmetric", "initial_inputs", "skew")}
            seed = int(row["seed"])
            if seed in self.initial and any(not np.array_equal(start[k], self.initial[seed][k]) for k in start):
                raise ValueError(f"Expected matched seed initialization; got {checkpoint}")
            self.initial[seed] = start
            if np.linalg.norm(model.symmetric-v["initial_symmetric"]) < 1e-6:
                raise ValueError(f"Expected trained recurrent parameters; got {checkpoint}")
            if row["method"].startswith("learned_mc"):
                m = probe_specification(row["method"], 256)[2]
                if (v["predictor_matrix"].shape != (256, 10) or not np.isfinite(v["predictor_matrix"]).all()
                        or v["predictor_observations"] != row["epoch"]*1149*m):
                    raise ValueError(f"Expected complete measured-predictor updates; got {checkpoint}")
            results = {}
            for split, label in (("val", "validation"),) if screen else (("val", "validation"), ("test", "test")):
                result = evaluate(model, data[f"{split}_x"], data[f"{split}_y"])
                error = abs(result["loss"] - row[f"{label}_loss"])
                if error > 1e-12 or result["accuracy"] != row[f"{label}_accuracy"]:
                    raise ValueError(f"Expected exact saved-metric replay; got {checkpoint}: {label}, {result}")
                results[f"{label}_accuracy"] = result["accuracy"]
                results[f"{label}_loss_replay_error"] = error
            self.checks[str(checkpoint)] = dict(method=row["method"], seed=seed, screen=screen, **results)


def aggregate(rows, group, method):
    selected = [r for r in rows if r["group"] == group and r["method"] == method]
    final = [r for r in selected if r["epoch"] == 15]
    if {r["seed"] for r in final} != {0,1,2} or len(final) != 3:
        raise ValueError(f"Expected three-seed coverage for {group}/{method}")
    result = dict(group=group, method=method, seeds=3, phases_per_example=phase_count(method),
                  learning_rate=final[0]["learning_rate"], momentum=final[0]["momentum"])
    for key in ("test_accuracy", "test_loss", "validation_accuracy", "equilibrations",
                "elapsed_seconds", "projected_updates", "predictor_update_seconds"):
        values = [r.get(key, 0.0) for r in final]
        result[key+"_mean"] = float(np.mean(values))
        result[key+"_std"] = float(np.std(values))
    for metric in ("baseline_relative_error", "gradient_cosine", "gradient_relative_error", "applied_update_cosine"):
        for epoch in (0, 7, 14):
            values = [r[metric] for r in selected if r["epoch"] == epoch and metric in r]
            if values:
                result[f"{metric}_epoch{epoch}"] = float(np.mean(values))
    result["first_validation_95percent"] = []
    for seed in (0,1,2):
        series = sorted((r for r in selected if r["seed"] == seed), key=lambda r: r["epoch"])
        hit = next((r for r in series if r["validation_accuracy"] >= .95), None)
        result["first_validation_95percent"].append(dict(seed=seed, epoch=None if hit is None else hit["epoch"],
                                                          equilibrations=None if hit is None else hit["equilibrations"]))
    return result


def read_family(collector, root, name):
    """Accept one three-seed process or three independent one-seed shards."""
    path = root / name
    paths = [path] if path.is_dir() else [root / f"{name}_seed{seed}" for seed in (0,1,2)]
    return [row for path in paths for row in collector.read(path)]


def held_out_baseline_audit(rows):
    """Post-run, oracle-only explanation on identical states for each baseline.

    Use the first 96 validation inputs; H never trains on these readings. Compare
    H, local slopes and ideal ordinary EqProp at the SAME final trained model.
    Variance ratios below exclude voltage noise and finite-nudge bias.
    """
    results = []
    for row in rows:
        if row["group"] != "selected" or row["epoch"] != 15 or not row["method"].startswith("learned_mc"):
            continue
        checkpoint = Path(row["checkpoint"])
        if not checkpoint.is_absolute():
            checkpoint = Path(__file__).resolve().parents[2] / checkpoint
        with np.load(checkpoint) as v:
            model = Classifier(v["symmetric"],v["skew"],v["inputs"],v["bias"],
                               outputs=int(v["outputs"]),cubic=float(v["cubic"]),
                               logit_scale=float(v["logit_scale"]),symmetric_cap=float(v["symmetric_cap"]))
            x, labels = v["val_x"][:96], v["val_y"][:96]
            net = model.network()
            state = settle(net, model.drive(x)).state
            c = cost_gradient(state,labels,model.hidden,model.logit_scale)
            jacobian = net.jacobian(state)
            exact = np.linalg.solve(np.swapaxes(jacobian,-1,-2), c[...,None])[...,0]
            q = np.linalg.solve(jacobian,c[...,None])[...,0]
            local = local_slope_baseline(state,c,model.cubic)
            learned = MeasuredBaseline(v["predictor_matrix"]).predict(state,c,model.cubic)
            errors = {name:float(np.sum((b-exact)**2)) for name,b in
                      (("ideal_eqprop",q),("local",local),("learned",learned))}
            total = float(np.sum(exact**2))
            m = probe_specification(row["method"],256)[2]
            results.append(dict(method=row["method"],seed=int(row["seed"]),examples=len(x),
                                relative_errors={k:float(np.sqrt(value/total)) for k,value in errors.items()},
                                ideal_mc_mse_ratio_vs_q_mc8=(8/m)*errors["learned"]/errors["ideal_eqprop"],
                                ideal_mc_mse_ratio_vs_local_same_probes=errors["learned"]/errors["local"],
                                checkpoint=str(checkpoint)))
    return results


def plots(output, rows, screens):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    curves = [("original", "adjoint"), ("original", "mc64"), ("original", "mc8")]
    curves += [("selected", m) for m in METHODS]
    for group, method in curves:
        selected = [r for r in rows if r["group"] == group and r["method"] == method]
        label = method + (" tuned" if group == "selected" else " control")
        means = np.array([[r["validation_accuracy"] for r in selected if r["epoch"] == e] for e in range(16)])
        mean, sd = 100*means.mean(axis=1), 100*means.std(axis=1)
        line, = axes[0,0].plot(range(16), mean, label=label)
        color = line.get_color()
        axes[0,0].fill_between(range(16), mean-sd, mean+sd, color=color, alpha=.08)
        axes[0,1].plot(np.arange(1,16)*1149*phase_count(method), mean[1:], label=label, color=color)
        for ax, metric in ((axes[1,0], "baseline_relative_error"), (axes[1,1], "gradient_cosine")):
            values = [np.mean([r[metric] for r in selected if r["epoch"] == e and metric in r])
                      for e in range(15) if any(r["epoch"] == e and metric in r for r in selected)]
            if values:
                ax.plot(range(len(values)), values, label=label, color=color)
    axes[0,0].set(xlabel="Training epoch", ylabel="Validation accuracy (%)", ylim=(82,100), title="Mean ± SD over three seeds")
    axes[0,1].set(xlabel="Training equilibrations, log scale\n(evaluation and screening excluded)",
                  xscale="log", ylabel="Validation accuracy (%)", ylim=(82,100), title="Work to reach a validation target")
    axes[1,0].set(xlabel="First minibatch of epoch (zero-based)", ylabel="Baseline / true-adjoint relative error")
    axes[1,1].set(xlabel="First minibatch of epoch (zero-based)", ylabel="Raw minibatch gradient cosine")
    for ax in axes.flat:
        ax.spines[["top", "right"]].set_visible(False)
    handles, labels = axes[0,0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=4, fontsize=8)
    fig.savefig(output / "improvements.png", dpi=180)
    fig.savefig(output / "improvements.svg")
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 4), constrained_layout=True)
    settings = [(lr, mu) for lr in (.03, .1, .3) for mu in (0., .9)]
    matrix = np.array([[next(r["validation_accuracy"] for r in screens
                            if r["method"] == m and r["epoch"] == 8 and
                            r["learning_rate"] == lr and r["momentum"] == mu)
                        for lr, mu in settings] for m in METHODS])*100
    im = ax.imshow(matrix, cmap="viridis", vmin=85, vmax=98)
    for i in range(len(METHODS)):
        for j in range(len(settings)):
            ax.text(j, i, f"{matrix[i,j]:.1f}", ha="center", va="center", color="black" if matrix[i,j]>94 else "white")
    ax.set_xticks(range(6), [f"lr={lr:g}\nEMA={mu:g}" for lr,mu in settings])
    ax.set_yticks(range(5), METHODS)
    ax.set_title("Eight-epoch, seed-0 validation screen (%)")
    fig.colorbar(im, ax=ax, label="Validation accuracy (%)")
    fig.savefig(output / "screen.png", dpi=180)
    fig.savefig(output / "screen.svg")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"Expected a fresh output directory; got {args.output}")
    collector = Collector()
    screens, rows, selections = [], [], []
    confirms = {}
    for method in METHODS:
        path = args.root / f"screen_{method}"
        screen = collector.read(path, screen=True)
        selected = select_configurations(screen, 8, [method])
        if selected != json.loads((path / "selection.json").read_text()):
            raise ValueError(f"Expected reproducible validation-only selection; got {path}")
        selections.extend(selected["selections"])
        screens.extend(screen)
        confirmation = read_family(collector, args.root, f"confirm_{method}")
        choice = selected["selections"][0]
        if any(r["learning_rate"] != choice["learning_rate"] or r["momentum"] != choice["momentum"] for r in confirmation):
            raise ValueError(f"Expected confirmation of selected settings; got {method}")
        confirms[method] = confirmation
        rows.extend(dict(r, group="selected") for r in confirmation)
    for seed in (0,1,2):
        original = collector.read(args.root.parent / f"mc_scaling_digits_n256_seed{seed}")
        rows.extend(dict(r, group="original") for r in original)
    for method in METHODS[1:]:
        choice = next(r for r in selections if r["method"] == method)
        reference = (confirms[method] if choice["learning_rate"] == .3 and choice["momentum"] == 0
                     else read_family(collector, args.root, f"reference_{method}"))
        if any(r["learning_rate"] != .3 or r["momentum"] != 0 for r in reference):
            raise ValueError(f"Expected original optimizer for baseline ablation; got {method}")
        rows.extend(dict(r, group="original") for r in reference)
    summary = [aggregate(rows, group, method) for group, method in dict.fromkeys((r["group"],r["method"]) for r in rows)]
    screening_cost = sum(r["equilibrations"] for r in screens if r["epoch"] == 8)
    args.output.mkdir(parents=True)
    write_csv(args.output / "metrics.csv", rows)
    write_csv(args.output / "screen.csv", screens)
    write_json(args.output / "summary.json", dict(results=summary, selections=selections,
                                                  screening_equilibrations=screening_cost,
                                                  verified_checkpoints=len(collector.checks)))
    write_json(args.output / "verification.json", collector.checks)
    write_json(args.output / "sources.json", collector.sources)
    write_json(args.output / "held_out_baseline_audit.json", held_out_baseline_audit(rows))
    plots(args.output, rows, screens)
    lines = ["# Cheaper measured-adjoint corrections at 256 states", "",
             "Exploratory digits simulation: 15 epochs, three seeds, mean ± population SD. Screening uses validation only.", "",
             "| Settings | Method | LR | EMA | Test accuracy | Phases/example | First validation 95% epoch, seeds 0/1/2 |",
             "|---|---|---:|---:|---:|---:|---|"]
    for r in summary:
        hits = ", ".join("never" if h["epoch"] is None else str(int(h["epoch"])) for h in r["first_validation_95percent"])
        lines.append(f"| {r['group']} | {r['method']} | {r['learning_rate']:g} | {r['momentum']:g} | "
                     f"{100*r['test_accuracy_mean']:.2f}% ± {100*r['test_accuracy_std']:.2f}% | {r['phases_per_example']} | {hits} |")
    lines += ["", f"Validation screening consumed {screening_cost:,.0f} training equilibrations across 30 eight-epoch trajectories.",
              f"Verified {len(collector.checks)} unique checkpoints against saved validation metrics and final test metrics where permitted.",
              "Phase counts exclude evaluation, diagnostic Jacobians, predictor digital arithmetic and the oracle's dense solves.",
              "The exact adjoint and mc8/mc64 original controls are reused from the previous matched experiment.",
              "The screen reuses seed 0; seeds 1 and 2 are fresh initialization checks on the same data split.",
              "A first crossing is not a sustained target or an early-stopping policy. No claim of improved asymptotic scaling follows from one width."]
    (args.output / "report.md").write_text("\n".join(lines)+"\n")
    print(f"Verified {len(collector.checks)} checkpoints; report: {args.output / 'report.md'}")


if __name__ == "__main__":
    main()
