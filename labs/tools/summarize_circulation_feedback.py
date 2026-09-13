#!/usr/bin/env python3
"""Verify completed circulation-calibration runs, replay checkpoints, and report.

Read-only with respect to source runs. Calibration observation time and state
reads remain separate from training equilibrium counts; neither is converted to
the other. A small smoke run can pass declared coverage without claiming the
complete 32/64/256-state, three-seed, five-method research grid.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from labs.recurrent_eqprop import Classifier, make_classifier, settle
from labs.tools.run_random_nudge_hopfield import write_csv, write_json
from labs.tools.summarize_recurrent_eqprop_digits import read_metrics
from labs.tools.train_recurrent_eqprop_digits import dataset, evaluate


PHASES = dict(contrastive_ep=3, known_skew_asymep=3, circulation_asymep=3,
              adjoint=1, learned_mc4=9)
AUDIT_METHODS = ("ordinary_ep", "known_skew", "calibrated_double_gain", "calibrated_single_gain")
LABELS = dict(contrastive_ep="Ordinary EP", known_skew_asymep="Known-skew AsymEP",
              circulation_asymep="Calibrated circulation", adjoint="Exact adjoint",
              learned_mc4="Learned baseline + MC4")
AUDIT_LABELS = dict(ordinary_ep="Ordinary EP", known_skew="Known skew",
                    calibrated_double_gain="Calibrated, doubled gain",
                    calibrated_single_gain="Calibrated, single gain")
DATA_KEYS = ("train_x", "train_y", "val_x", "val_y", "test_x", "test_y",
             "train_indices", "val_indices", "test_indices", "feature_mean", "feature_std")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def close(actual, expected, label, *, atol=1e-12):
    require(np.allclose(actual, expected, atol=atol, rtol=1e-12),
            f"Expected {label} to match its independent reference; got {actual} vs {expected}")


def mean_sd(values):
    values = np.asarray(values, dtype=float)
    return dict(mean=float(values.mean()), std=float(values.std()))


class Collector:
    def __init__(self):
        self.sources = []
        self.calibrations = []
        self.histories = []
        self.rows = []
        self.replays = []
        self.data = {}
        self.initial_free = {}
        self.initial_metrics = {}

    def read(self, path):
        path = path.resolve()
        run = json.loads((path/"run.json").read_text())
        status = json.loads((path/"status.json").read_text())
        summary = json.loads((path/"summary.json").read_text())
        config = run["config"]
        sizes, seeds, methods = config["sizes"], config["seeds"], config["methods"]
        expected_cases = {(int(size), int(seed)) for size in sizes for seed in seeds}
        require(len(expected_cases) == len(sizes)*len(seeds), f"Expected unique size/seed declarations; got {path}")
        require(len(set(methods)) == len(methods) and set(methods) <= PHASES.keys(),
                f"Expected unique supported methods; got {path}: {methods}")
        expected_training = len(expected_cases)*len(methods) if config["epochs"] else 0
        require(run["expected_calibrations"] == len(expected_cases)
                and run["expected_training_trajectories"] == expected_training
                and status["status"] == "complete"
                and status["completed_calibrations"] == len(expected_cases)
                and status["completed_training"] == expected_training,
                f"Expected complete declared calibration and training coverage; got {path}: {status}")
        cases = {(int(c["size"]), int(c["seed"])): c for c in summary["cases"]}
        require(set(cases) == expected_cases and len(summary["cases"]) == len(cases),
                f"Expected exactly the declared calibration summaries; got {path}")
        quick = bool(config["quick"])
        if quick not in self.data:
            self.data[quick] = dataset(quick)
        data = self.data[quick]
        matrices = {}
        for (size, seed), case in cases.items():
            matrices[size, seed] = self.calibration(path, config, case, data)

        rows = read_metrics(path/"metrics.csv") if expected_training else []
        require(len(rows) == expected_training*(config["epochs"]+1),
                f"Expected one row per epoch and trajectory; got {path}: {len(rows)}")
        grouped = {}
        for row in rows:
            identity = int(row["size"]), int(row["seed"]), row["method"]
            require(identity[:2] in expected_cases and identity[2] in methods,
                    f"Expected declared trajectory identity; got {path}: {identity}")
            grouped.setdefault(identity, []).append(row)
            epoch, method = int(row["epoch"]), row["method"]
            require(row["epoch"] == epoch and 0 <= epoch <= config["epochs"],
                    f"Expected an integer epoch in the declared interval; got {row}")
            count = epoch*len(data["train_y"])*PHASES[method]
            require(row["equilibrations"] == count and row["state_reads"] == count
                    and row["max_force_residual"] <= 1e-9,
                    f"Expected measured training counts and converged forces; got {row}")
            close(row["learning_rate"], config["learning_rate"], "learning rate")
            close(row["momentum"], config["momentum"], "weight-gradient momentum")
            probes = 4 if method == "learned_mc4" else 0
            close(row["total_probe_excitation_sq"], 2*epoch*len(data["train_y"])*probes*0.01**2,
                  "probe excitation budget", atol=1e-10)
            require((epoch == config["epochs"]) == ("test_accuracy" in row and "test_loss" in row),
                    f"Expected test evaluation only on final rows; got {row}")
            stats = cases[identity[:2]]["calibration"]
            charged = method == "circulation_asymep"
            close(row["calibration_total_record_time"], stats["total_record_time"] if charged else 0,
                  "separately charged calibration time")
            close(row["calibration_state_reads"], stats["state_reads"] if charged else 0,
                  "separately charged calibration reads")
            row.update(source_run=str(path), declared_epochs=int(config["epochs"]), quick=quick,
                       calibration_dt=float(config["dt"]), asymmetry=float(config["asymmetry"]),
                       read_noise=float(config["read_noise"]))
        require(len(grouped) == expected_training, f"Expected all declared trajectories; got {path}")
        finals = {(int(r["size"]), int(r["seed"]), r["method"]): r for r in summary["training_final"]}
        require(len(finals) == len(summary["training_final"]) == expected_training and set(finals) == set(grouped),
                f"Expected complete final summary coverage; got {path}")
        for identity, series in grouped.items():
            series.sort(key=lambda r: r["epoch"])
            require([r["epoch"] for r in series] == list(range(config["epochs"]+1)),
                    f"Expected unique complete epoch coverage; got {path}: {identity}")
            for key, value in finals[identity].items():
                require(key in series[-1], f"Expected final summary field in metrics; got {key}")
                if isinstance(value, (int, float)):
                    close(series[-1][key], value, f"final summary {key}")
                else:
                    require(series[-1][key] == value, f"Expected final summary {key} equality")
            self.replay(path, config, identity, series, data, matrices[identity[:2]])
        self.rows.extend(rows)
        self.sources.append(dict(path=str(path), run=run, status=status,
                                 verified_calibrations=len(cases), verified_training=len(grouped)))

    def calibration(self, path, config, case, data):
        size, seed = int(case["size"]), int(case["seed"])
        directory = path/f"n{size}_seed{seed}"
        require(json.loads((directory/"calibration_summary.json").read_text()) == case,
                f"Expected matching per-case and run calibration summaries; got {directory}")
        require(case["controller_frozen_after_calibration"]
                and not case["reference_skew_used_for_calibration"],
                f"Expected a frozen controller fit without reference skew; got {directory}")
        initial = make_classifier(seed, size=size, asymmetry=config["asymmetry"])
        with np.load(directory/"controller.npz") as saved:
            controller = saved["controller"].copy()
            final = saved["final_controller"]
            require(controller.shape == final.shape == (size, size)
                    and np.isfinite(controller).all() and np.isfinite(final).all(),
                    f"Expected finite square controller matrices; got {directory}")
            close(controller, -controller.T, "skew controller", atol=1e-14)
            close(final, -final.T, "skew final controller", atol=1e-14)
            center = settle(initial.network(), initial.drive(data["train_x"][:1])[0]).state
            close(saved["calibration_center"], center, "calibration free-state anchor")
            error = np.linalg.norm(controller+initial.skew)/max(np.linalg.norm(initial.skew), 1e-20)
            close(error, case["controller_relative_error"], "controller error audit")
        stats = case["calibration"]
        warm = int(np.ceil(config["burn_in"]/config["dt"]))
        steps = int(np.ceil(config["duration"]/config["dt"]))
        reads = config["replicas"]*(warm+steps+1)
        expected = dict(states=size, replicas=config["replicas"], dt=config["dt"],
                        temperature=config["temperature"], learning_rate=config["calibration_rate"],
                        update_interval=config["update_interval"], average_after=config["average_after"],
                        read_noise=config["calibration_read_noise"], seed=64000+seed,
                        adaptive_time=steps*config["dt"], burn_in_time=warm*config["dt"],
                        state_reads=reads, scalar_state_reads=size*reads,
                        physical_time_per_record=(warm+steps)*config["dt"],
                        total_record_time=config["replicas"]*(warm+steps)*config["dt"],
                        force_evaluations=2*config["replicas"]*(warm+steps),
                        updates=int(np.ceil(steps/config["update_interval"])),
                        independent_controller_coefficients=size*(size-1)//2)
        for key, value in expected.items():
            close(stats[key], value, f"calibration accounting {key}")
        require(not stats["averaged_controller_used_during_calibration"],
                f"Expected reported averaging to leave adaptive dynamics unchanged; got {directory}")
        history = read_metrics(directory/"calibration.csv")
        times = [r["adaptive_time"] for r in history]
        require(times and all(a < b for a, b in zip(times, times[1:])),
                f"Expected increasing calibration history; got {directory}")
        close(times[-1], stats["adaptive_time"], "terminal calibration time")
        close(history[-1]["averaged_controller_relative_error"], error, "terminal averaged controller error")
        for name in ("initial_validation_gradient_audit", "changed_symmetric_model_gradient_audit"):
            audit = case[name]
            require(len(audit) == len(AUDIT_METHODS) and {r["method"] for r in audit} == set(AUDIT_METHODS),
                    f"Expected complete four-method gradient audit; got {directory}: {name}")
            for row in audit:
                require(row["max_residual"] <= 1e-9 and row["probe_count"] == 0,
                        f"Expected converged zero-probe audit phases; got {row}")
                for metric in ("gradient_cosine", "input_gradient_cosine", "gradient_relative_error",
                               "input_gradient_relative_error", "adjoint_relative_error"):
                    require(np.isfinite(row[metric]), f"Expected finite {metric}; got {row}")
        record = dict(case, source_run=str(path), config=config, training_epochs=int(config["epochs"]))
        self.calibrations.append(record)
        self.histories.append(dict(size=size, seed=seed, dt=config["dt"], duration=config["duration"],
                                   source_run=str(path), history=history))
        return controller

    def replay(self, path, config, identity, series, data, controller):
        size, seed, method = identity
        checkpoint = path/f"n{size}_seed{seed}"/f"{method}_seed{seed}_lr{config['learning_rate']:g}.npz"
        initial = make_classifier(seed, size=size, asymmetry=config["asymmetry"])
        with np.load(checkpoint) as saved:
            model = Classifier(saved["symmetric"], saved["skew"], saved["inputs"], saved["bias"],
                               outputs=int(saved["outputs"]), cubic=float(saved["cubic"]),
                               logit_scale=float(saved["logit_scale"]), symmetric_cap=float(saved["symmetric_cap"]))
            require(model.size == size and model.outputs == 10 and model.inputs.shape == (size-10, 64),
                    f"Expected declared model shape; got {checkpoint}")
            for array in (model.symmetric, model.skew, model.inputs, model.bias):
                require(np.isfinite(array).all(), f"Expected finite trained parameters; got {checkpoint}")
            close(model.symmetric, model.symmetric.T, "trained symmetric recurrence", atol=1e-14)
            require(not np.any(np.diag(model.symmetric))
                    and np.max(np.abs(np.linalg.eigvalsh(model.symmetric))) <= model.symmetric_cap+1e-12,
                    f"Expected projected stable recurrent parameters; got {checkpoint}")
            for key, expected in (("skew", initial.skew), ("initial_symmetric", initial.symmetric),
                                  ("initial_inputs", initial.inputs)):
                require(np.array_equal(saved[key], expected),
                        f"Expected exact fixed-skew and seed initialization for {key}; got {checkpoint}")
            close(float(saved["momentum"]), config["momentum"], "saved optimizer momentum")
            for key in DATA_KEYS:
                require(np.array_equal(saved[key], data[key]),
                        f"Expected identical deterministic data and preprocessing for {key}; got {checkpoint}")
            indices = [set(saved[key].tolist()) for key in ("train_indices", "val_indices", "test_indices")]
            require(not any(indices[a] & indices[b] for a, b in ((0,1), (0,2), (1,2))),
                    f"Expected disjoint training, validation, and test inputs; got {checkpoint}")
            if method == "circulation_asymep":
                require(np.array_equal(saved["feedback_controller"], controller),
                        f"Expected the same frozen calibrated controller in the training checkpoint; got {checkpoint}")
            if method == "learned_mc4":
                require(saved["predictor_matrix"].shape == (size, 10)
                        and np.isfinite(saved["predictor_matrix"]).all()
                        and int(saved["predictor_observations"]) == 4*len(data["train_y"])*config["epochs"],
                        f"Expected complete learned-predictor observation accounting; got {checkpoint}")
            key = size, seed, bool(config["quick"]), float(config["asymmetry"])
            reconstructed = Classifier(saved["initial_symmetric"], saved["skew"], saved["initial_inputs"],
                                       np.zeros(size), outputs=model.outputs, cubic=model.cubic,
                                       logit_scale=model.logit_scale, symmetric_cap=model.symmetric_cap)
            free = settle(reconstructed.network(), reconstructed.drive(data["val_x"][:8])).state
            if key in self.initial_free:
                require(np.array_equal(free, self.initial_free[key]),
                        f"Expected identical reconstructed initial ordinary free states across methods; got {checkpoint}")
            else:
                self.initial_free[key] = free
                self.initial_metrics[key] = {label: evaluate(initial, data[f"{split}_x"], data[f"{split}_y"])
                                             for split, label in (("train", "train"), ("val", "validation"))}
            for label, result in self.initial_metrics[key].items():
                close(series[0][label+"_loss"], result["loss"], f"initial {label} loss")
                close(series[0][label+"_accuracy"], result["accuracy"], f"initial {label} accuracy")
            replay = {}
            for split, label in (("val", "validation"), ("test", "test")):
                result = evaluate(model, data[f"{split}_x"], data[f"{split}_y"])
                close(series[-1][label+"_loss"], result["loss"], f"checkpoint {label} loss")
                require(series[-1][label+"_accuracy"] == result["accuracy"] and result["residual"] <= 1e-9,
                        f"Expected exact checkpoint {label} accuracy and converged forces; got {checkpoint}")
                replay[label] = result
            self.replays.append(dict(checkpoint=str(checkpoint), size=size, seed=seed, method=method,
                                     initial_free_states_replayed=8, **replay))


def aggregate(collector):
    groups = {}
    for row in collector.rows:
        if row["epoch"] != row["declared_epochs"]:
            continue
        key = (int(row["size"]), row["method"], int(row["epoch"]), row["learning_rate"],
               row["momentum"], row["quick"], row["asymmetry"], row["read_noise"], row["calibration_dt"])
        groups.setdefault(key, []).append(row)
    training = []
    for key, rows in sorted(groups.items()):
        require(len({r["seed"] for r in rows}) == len(rows),
                f"Expected at most one trajectory per seed and experimental condition; got {key}")
        record = dict(size=key[0], method=key[1], epochs=key[2], learning_rate=key[3], momentum=key[4],
                      quick=key[5], asymmetry=key[6], read_noise=key[7], calibration_dt=key[8],
                      seeds=sorted(int(r["seed"]) for r in rows), equilibrations_per_example=PHASES[key[1]])
        for metric in ("test_accuracy", "test_loss", "validation_accuracy", "validation_loss", "equilibrations",
                       "state_reads", "elapsed_seconds", "calibration_total_record_time", "calibration_state_reads"):
            record[metric] = mean_sd([r[metric] for r in rows])
        series = [r for r in collector.rows if r["source_run"] in {s["source_run"] for s in rows}
                  and r["method"] == key[1] and int(r["size"]) == key[0]]
        for epoch in (0, key[2]-1):
            selected = [r for r in series if r["epoch"] == epoch]
            for metric in ("gradient_cosine", "gradient_relative_error", "applied_update_cosine", "adjoint_relative_error"):
                values = [r[metric] for r in selected if metric in r]
                if values:
                    record[f"{metric}_epoch{epoch}"] = mean_sd(values)
        training.append(record)
    calibration_groups = {}
    for case in collector.calibrations:
        config = case["config"]
        key = (case["size"], config["dt"], config["duration"], config["temperature"], config["replicas"],
               config["calibration_rate"], config["average_after"], config["quick"], config["calibration_read_noise"])
        calibration_groups.setdefault(key, []).append(case)
    calibration = []
    for key, cases in sorted(calibration_groups.items()):
        record = dict(size=key[0], dt=key[1], duration=key[2], temperature=key[3], replicas=key[4],
                      calibration_rate=key[5], average_after=key[6], quick=key[7], read_noise=key[8],
                      seeds=sorted(c["seed"] for c in cases), controller_error=mean_sd([c["controller_relative_error"] for c in cases]))
        for metric in ("total_record_time", "state_reads", "scalar_state_reads", "physical_time_per_record",
                       "independent_controller_coefficients"):
            record[metric] = mean_sd([c["calibration"][metric] for c in cases])
        for source, destination in (("initial_validation_gradient_audit", "initial_audit"),
                                    ("changed_symmetric_model_gradient_audit", "changed_model_audit")):
            record[destination] = {}
            for method in AUDIT_METHODS:
                selected = [next(a for a in c[source] if a["method"] == method) for c in cases]
                record[destination][method] = {metric: mean_sd([r[metric] for r in selected]) for metric in
                                               ("gradient_cosine", "gradient_relative_error", "input_gradient_cosine",
                                                "input_gradient_relative_error", "adjoint_relative_error")}
        calibration.append(record)
    expected = {(size, seed, method) for size in (32, 64, 256) for seed in (0,1,2) for method in PHASES}
    actual = {(int(r["size"]), int(r["seed"]), r["method"]) for r in collector.rows
              if r["epoch"] == r["declared_epochs"] == 15 and not r["quick"] and r["calibration_dt"] == .02}
    refinement = {c["seed"] for c in collector.calibrations
                  if c["size"] == 64 and c["config"]["dt"] == .01 and c["training_epochs"] == 0}
    conditions = {(r["learning_rate"], r["momentum"], r["asymmetry"], r["read_noise"])
                  for r in collector.rows if r["epoch"] == r["declared_epochs"] == 15
                  and not r["quick"] and r["calibration_dt"] == .02}
    coverage = dict(declared_sources_complete=True, checkpoint_replays=len(collector.replays),
                    main_grid_complete=expected <= actual and len(conditions) == 1,
                    matched_main_training_conditions=len(conditions) == 1,
                    main_training_conditions=[dict(learning_rate=lr, momentum=mu, asymmetry=a, read_noise=sigma)
                                              for lr,mu,a,sigma in sorted(conditions)],
                    missing_main_trajectories=[dict(size=n, seed=s, method=m) for n,s,m in sorted(expected-actual)],
                    refinement_n64_dt001_complete={0,1,2} <= refinement,
                    refinement_seeds=sorted(refinement))
    return dict(evidence_tier="exploratory, non-canonical", coverage=coverage, training=training,
                calibration=calibration,
                cost_interpretation="Calibration record time and voltage reads are separate from training equilibrium counts; no conversion assumed.",
                limitations=["Shared learning rate and momentum; methods are not individually retuned.",
                             "Three seeds on one small dataset do not establish superiority from small accuracy differences.",
                             "Dense skew controller has n(n-1)/2 independent coefficients and full state-read access.",
                             "Fixed skew permits a frozen calibration; arbitrary state-dependent asymmetry is not tested.",
                             "Adjoint solves and diagnostic replay work are excluded from physical training equilibrium counts."])


def plots(output, collector, summary):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), constrained_layout=True)
    histories = {}
    for case in collector.histories:
        histories.setdefault((case["size"], case["dt"], case["duration"]), []).append(case["history"])
    for (size, dt, duration), cases in sorted(histories.items()):
        times = np.array([r["adaptive_time"] for r in cases[0]])
        curves = []
        for history in cases:
            close([r["adaptive_time"] for r in history], times, "matched calibration plotting grid")
            curves.append([r["averaged_controller_relative_error"] for r in history])
        curves = np.array(curves)
        line, = axes[0].plot(times, curves.mean(axis=0), label=f"n={size}, dt={dt:g}")
        axes[0].fill_between(times, curves.mean(axis=0)-curves.std(axis=0),
                             curves.mean(axis=0)+curves.std(axis=0), color=line.get_color(), alpha=.12)
    for method in PHASES:
        rows = sorted([r for r in summary["training"] if r["method"] == method], key=lambda r: r["size"])
        if rows:
            axes[1].errorbar([r["size"] for r in rows], [100*r["test_accuracy"]["mean"] for r in rows],
                             yerr=[100*r["test_accuracy"]["std"] for r in rows], marker="o", capsize=3, label=LABELS[method])
    for method in AUDIT_METHODS:
        rows = sorted([r for r in summary["calibration"] if r["dt"] == .02], key=lambda r: r["size"])
        if rows:
            axes[2].errorbar([r["size"] for r in rows],
                             [r["initial_audit"][method]["gradient_cosine"]["mean"] for r in rows],
                             yerr=[r["initial_audit"][method]["gradient_cosine"]["std"] for r in rows],
                             marker="o", capsize=3, label=AUDIT_LABELS[method])
    axes[0].set(xlabel="Adaptive physical time per record", ylabel="Relative controller error",
                title="Calibration: mean ± SD over seeds")
    axes[1].set(xlabel="State count", ylabel="Final test accuracy (%)", title="Matched training controls", ylim=(0, 101))
    axes[2].set(xlabel="State count", ylabel="Parameter-gradient cosine", title="Same initial validation states", ylim=(-1.05, 1.05))
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
        if axis.get_legend_handles_labels()[0]:
            axis.legend(fontsize=7)
    widths = sorted({r["size"] for r in summary["calibration"]})
    axes[1].set_xticks(widths)
    axes[2].set_xticks(widths)
    fig.savefig(output/"circulation_feedback.png", dpi=180)
    fig.savefig(output/"circulation_feedback.svg")
    plt.close(fig)


def report(output, summary):
    coverage = summary["coverage"]
    lines = ["# Circulation-calibrated asymmetric EqProp", "",
             "Exploratory simulation. All source declarations, calibration accounting, saved-controller equality, "
             "fixed skew, shared initialization/data, reconstructed initial free states, and final validation/test replay were checked.", "",
             f"Verified checkpoints: {coverage['checkpoint_replays']}. Full 32/64/256 × three-seed × five-method grid: "
             f"{coverage['main_grid_complete']}. Three-seed n=64, dt=0.01 refinement: {coverage['refinement_n64_dt001_complete']}.", "",
             "Means ± population SD across the listed seeds. Gradient audits use first validation inputs and no read noise; "
             "training uses the recorded read noise. Epoch gradient audits use the first training minibatch before that epoch's updates.", "",
             "## Final training", "",
             "| States | Method | Seeds | Epochs | Test accuracy | Validation accuracy | Training equilibrations | Last audited gradient cosine |",
             "|---:|---|---|---:|---:|---:|---:|---:|"]
    for row in summary["training"]:
        test, val = row["test_accuracy"], row["validation_accuracy"]
        gradient = row.get(f"gradient_cosine_epoch{row['epochs']-1}")
        gradient_text = "unavailable" if gradient is None else f"{gradient['mean']:.4f} ± {gradient['std']:.4f}"
        lines.append(f"| {row['size']} | {row['method']} | {row['seeds']} | {row['epochs']} | "
                     f"{100*test['mean']:.2f} ± {100*test['std']:.2f}% | {100*val['mean']:.2f} ± {100*val['std']:.2f}% | "
                     f"{row['equilibrations']['mean']:,.0f} | {gradient_text} |")
    lines += ["", "Calibration costs below are paid once for each circulation-trained model, in addition to its training equilibrations. "
              "No conversion between record time, voltage reads, and equilibrium phases is assumed. The adjoint baseline also requires digital solves.", "",
              "## Calibration and step-size refinement", "",
              "| States | dt | Seeds | Controller error | Total record time | State-vector reads | Scalar voltage reads | Controller coefficients |",
              "|---:|---:|---|---:|---:|---:|---:|---:|"]
    for row in summary["calibration"]:
        err = row["controller_error"]
        lines.append(f"| {row['size']} | {row['dt']:g} | {row['seeds']} | {err['mean']:.4f} ± {err['std']:.4f} | "
                     f"{row['total_record_time']['mean']:,.0f} | {row['state_reads']['mean']:,.0f} | "
                     f"{row['scalar_state_reads']['mean']:,.0f} | {row['independent_controller_coefficients']['mean']:,.0f} |")
    lines += ["", "Replicas count as independent records and their times add; warm-up is included. "
              "The dt=0.01 runs test numerical-step sensitivity and do not provide additional training trajectories.", "",
              "## Initial and changed-model gradient audits", "",
              "The changed model rescales the symmetric recurrence by 0.4, input weights by 1.8, and adds a bias ramp. "
              "It retains the same skew and controller; this is a transfer diagnostic, not a trained checkpoint.", "",
              "| States | dt | Feedback | Initial gradient cosine | Initial input-weight cosine | Changed-model gradient cosine |",
              "|---:|---:|---|---:|---:|---:|"]
    for row in summary["calibration"]:
        for method in AUDIT_METHODS:
            initial, changed = row["initial_audit"][method], row["changed_model_audit"][method]
            lines.append(f"| {row['size']} | {row['dt']:g} | {method} | "
                         f"{initial['gradient_cosine']['mean']:.4f} ± {initial['gradient_cosine']['std']:.4f} | "
                         f"{initial['input_gradient_cosine']['mean']:.4f} ± {initial['input_gradient_cosine']['std']:.4f} | "
                         f"{changed['gradient_cosine']['mean']:.4f} ± {changed['gradient_cosine']['std']:.4f} |")
    lines += ["", "## Limits", ""]+[f"- {text}" for text in summary["limitations"]]
    (output/"report.md").write_text("\n".join(lines)+"\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"Expected a fresh output directory; got {args.output}")
    if len({p.resolve() for p in args.runs}) != len(args.runs):
        parser.error("Expected unique source run paths")
    collector = Collector()
    for path in args.runs:
        print(f"Verifying and replaying {path}", flush=True)
        collector.read(path)
    summary = aggregate(collector)
    args.output.mkdir(parents=True)
    write_json(args.output/"summary.json", summary)
    write_json(args.output/"verification.json", dict(coverage=summary["coverage"], checkpoint_replays=collector.replays))
    write_json(args.output/"sources.json", collector.sources)
    if collector.rows:
        write_csv(args.output/"metrics.csv", collector.rows)
    report(args.output, summary)
    plots(args.output, collector, summary)
    coverage = dict(summary["coverage"])
    coverage["missing_main_trajectory_count"] = len(coverage.pop("missing_main_trajectories"))
    print(json.dumps(coverage, indent=2), flush=True)


if __name__ == "__main__":
    main()
