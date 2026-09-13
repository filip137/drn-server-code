#!/usr/bin/env python3
"""Verify declared few-loop calibration/training coverage and replay checkpoints.

Source runs are read-only. Calibration time, projected readouts, process-noise
injections, and task equilibrations remain distinct resources. Passing this
verifier establishes only the grids explicitly declared by its input runs.
"""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from labs.recurrent_eqprop import Classifier, settle
from labs.tools.run_random_nudge_hopfield import write_csv, write_json
from labs.tools.summarize_circulation_feedback import (
    AUDIT_METHODS, DATA_KEYS, LABELS, PHASES, close, mean_sd, require,
)
from labs.tools.summarize_recurrent_eqprop_digits import read_metrics
from labs.tools.test_circulation_feedback import audit_controller
from labs.tools.test_structured_circulation import make_loop_case, recording_cost
from labs.tools.train_recurrent_eqprop_digits import dataset, evaluate


PROTOCOL = ("loops", "duration", "dt", "temperature", "replicas", "calibration_rate",
            "update_interval", "burn_in", "average_after", "calibration_read_noise",
            "quick", "epochs", "learning_rate", "momentum", "read_noise")


def finite(value, label):
    require(np.isfinite(value).all(), f"Expected finite {label}; got {value}")


def verify_audit(saved, model, data, matrix):
    require(len(saved) == len(AUDIT_METHODS) and {r["method"] for r in saved} == set(AUDIT_METHODS),
            "Expected all four ordinary/known-skew/single/doubled gradient audits")
    fresh = {r["method"]: r for r in audit_controller(model, data, matrix)}
    for row in saved:
        expected = fresh[row["method"]]
        for key, value in row.items():
            if isinstance(value, (float, int)):
                finite(value, f"gradient audit {key}")
                close(value, expected[key], f"replayed gradient audit {key}", atol=1e-9)
        require(row["max_residual"] <= 1e-9 and row["probe_count"] == 0,
                "Expected converged zero-random-probe gradient audit")


def verify_calibration(directory, config, case, model, left, right, reference, data, *, dense=False):
    record = case["dense_comparison"] if dense else case
    filename = "dense_controller.npz" if dense else "controller.npz"
    size, seed, loops = case["size"], case["seed"], config["loops"]
    with np.load(directory/filename) as saved:
        matrix = saved["controller"].copy()
        require(matrix.shape == (size, size), f"Expected {size}x{size} controller; got {matrix.shape}")
        finite(matrix, filename)
        close(matrix, -matrix.T, "skew controller", atol=1e-14)
        if dense:
            final = saved["final_controller"]
            require(final.shape == matrix.shape, "Expected matching final dense controller shape")
            finite(final, "final dense controller")
            close(final, -final.T, "final skew controller", atol=1e-14)
            final_error = np.linalg.norm(final+model.skew)/np.linalg.norm(model.skew)
        else:
            coefficients, final = saved["coefficients"], saved["final_coefficients"]
            require(coefficients.shape == final.shape == (loops,), "Expected one learned coefficient per loop")
            finite(coefficients, "averaged loop coefficients")
            finite(final, "final loop coefficients")
            product = (left*coefficients) @ right.T
            close(matrix, (product-product.T)/np.sqrt(2), "controller reconstructed from known wiring")
            final_error = np.linalg.norm(final+reference["normalized_gains"])/np.linalg.norm(reference["normalized_gains"])
        center = settle(model.network(), model.drive(data["train_x"][:1])[0]).state
        close(saved["calibration_center"], center, "calibration free-state anchor")
    error = float(np.linalg.norm(matrix+model.skew)/np.linalg.norm(model.skew))
    close(record["controller_relative_error"], error, "independent controller error")
    stats = record["calibration"]
    cost = recording_cost(SimpleNamespace(**config), size, structured=not dense)
    recorded_cost = record["extra_study_cost"] if dense else record["measurement_cost"]
    steps = int(np.ceil(config["duration"]/config["dt"]))
    warm = int(np.ceil(config["burn_in"]/config["dt"]))
    snapshots, channels = cost["measurement_snapshots"], cost["observed_scalar_channels"]
    expected = dict(states=size, replicas=config["replicas"], dt=config["dt"],
                    temperature=config["temperature"], learning_rate=config["calibration_rate"],
                    update_interval=config["update_interval"], average_after=config["average_after"],
                    read_noise=config["calibration_read_noise"], seed=64000+seed,
                    adaptive_time=steps*config["dt"], burn_in_time=warm*config["dt"],
                    state_reads=snapshots, scalar_state_reads=channels*snapshots,
                    physical_time_per_record=(warm+steps)*config["dt"],
                    total_record_time=cost["total_recording_time"],
                    force_evaluations=2*config["replicas"]*(warm+steps),
                    updates=int(np.ceil(steps/config["update_interval"])))
    if dense:
        expected["independent_controller_coefficients"] = size*(size-1)//2
    else:
        expected.update(loops=loops, readout_channels=2*loops, learned_parameters=loops,
                        fixed_wiring_coefficients=2*size*loops,
                        scalar_measurement_reads=2*loops*snapshots)
    for key, value in expected.items():
        close(stats[key], value, f"calibration accounting {key}")
    for key, value in recorded_cost.items():
        close(value, cost[key], f"recorded measurement cost {key}")
    # Older smoke fixtures lack these keys; reconstruction still charges them.
    noise = dict(process_noise_channels=size,
                 process_noise_scalar_increments=size*config["replicas"]*(warm+steps))
    for key, value in noise.items():
        if key in stats:
            close(stats[key], value, f"physical injection accounting {key}")
    cost.update(noise, injection_accounting_reconstructed=True)
    require(not stats["averaged_controller_used_during_calibration"],
            "Expected controller averaging to be a readout, not a changed adaptation")
    history = read_metrics(directory/("dense_calibration.csv" if dense else "structured_calibration.csv"))
    times = [r["adaptive_time"] for r in history]
    require(times and all(a < b for a, b in zip(times, times[1:])), "Expected increasing calibration history")
    close(times[-1], stats["adaptive_time"], "completed calibration history")
    prefix = "controller" if dense else "coefficient"
    close(history[-1][f"averaged_{prefix}_relative_error"], error, "terminal averaged controller audit")
    close(history[-1][f"current_{prefix}_relative_error"], final_error, "terminal unaveraged controller audit")
    verify_audit(record["gradient_audit"], model, data, matrix)
    return matrix, cost


def replay_checkpoint(checkpoint, model, controller, data, config, method, series):
    with np.load(checkpoint) as saved:
        for name in DATA_KEYS:
            require(np.array_equal(saved[name], data[name]), f"Expected deterministic dataset/preprocessing for {name}: {checkpoint}")
        for name, expected in (("skew", model.skew), ("initial_symmetric", model.symmetric),
                               ("initial_inputs", model.inputs)):
            require(np.array_equal(saved[name], expected), f"Expected identical fixed K and initial {name}: {checkpoint}")
        for name in ("outputs", "cubic", "logit_scale", "symmetric_cap"):
            close(saved[name], getattr(model, name), f"saved {name}")
        close(saved["momentum"], config["momentum"], "saved optimizer momentum")
        trained = Classifier(saved["symmetric"], saved["skew"], saved["inputs"], saved["bias"],
                             outputs=int(saved["outputs"]), cubic=float(saved["cubic"]),
                             logit_scale=float(saved["logit_scale"]), symmetric_cap=float(saved["symmetric_cap"]))
        for name in ("symmetric", "skew", "inputs", "bias"):
            value = getattr(trained, name)
            require(value.shape == getattr(model, name).shape, f"Expected unchanged model shape for {name}")
            finite(value, f"trained {name}")
        close(trained.symmetric, trained.symmetric.T, "symmetric trained recurrence", atol=1e-14)
        require(not np.any(np.diag(trained.symmetric))
                and np.linalg.norm(trained.symmetric, 2) <= trained.symmetric_cap+1e-12,
                "Expected zero-diagonal stable recurrent parameters")
        if method == "circulation_asymep":
            require(np.array_equal(saved["feedback_controller"], controller), "Expected fixed calibrated controller in checkpoint")
        if method == "learned_mc4":
            require(saved["predictor_matrix"].shape == (model.size, 10)
                    and np.isfinite(saved["predictor_matrix"]).all()
                    and int(saved["predictor_observations"]) == 4*len(data["train_y"])*config["epochs"],
                    "Expected complete MC4 predictor learning and measurement accounting")
        results = {}
        for split, label in (("val", "validation"), ("test", "test")):
            result = evaluate(trained, data[f"{split}_x"], data[f"{split}_y"])
            close(series[-1][label+"_loss"], result["loss"], f"replayed {label} loss")
            require(series[-1][label+"_accuracy"] == result["accuracy"] and result["residual"] <= 1e-9,
                    f"Expected exact {label} accuracy and converged checkpoint replay")
            results[label] = result
    return dict(checkpoint=str(checkpoint), method=method, **results)


def read_run(path):
    path = path.resolve()
    run = json.loads((path/"run.json").read_text())
    config = run["config"]
    require(json.loads((path/"config.json").read_text()) == config, "Expected matching config/run provenance")
    status = json.loads((path/"status.json").read_text())
    summary = json.loads((path/"summary.json").read_text())
    expected_cases = {(n, seed) for n in config["sizes"] for seed in config["seeds"]}
    require(len(expected_cases) == len(config["sizes"])*len(config["seeds"]), "Expected unique declared cases")
    methods, epochs = config["methods"], config["epochs"]
    require(len(set(methods)) == len(methods) and set(methods) <= PHASES.keys(), "Expected unique supported methods")
    counts = dict(structured_calibrations=len(expected_cases),
                  dense_calibrations=len(expected_cases)*int(config["dense_control"]),
                  training=len(expected_cases)*len(methods) if epochs else 0)
    require(status["status"] == "complete", f"Expected a completed source run; got {status}")
    for name, count in counts.items():
        declared = "expected_training_trajectories" if name == "training" else "expected_"+name
        require(run[declared] == count and status["completed_"+name] == count,
                f"Expected complete declared {name}; got {path}")
    cases = {(c["size"], c["seed"]): c for c in summary["cases"]}
    require(set(cases) == expected_cases and len(cases) == len(summary["cases"]), "Expected exact calibration case coverage")
    data = dataset(config["quick"])
    rows = read_metrics(path/"metrics.csv") if epochs else []
    require(len(rows) == counts["training"]*(epochs+1), "Expected every declared epoch row")
    grouped = {}
    for row in rows:
        identity = row["size"], row["seed"], row["method"]
        require(identity[:2] in cases and identity[2] in methods, "Expected declared training identity")
        grouped.setdefault(identity, []).append(row)
    require(len(grouped) == counts["training"], "Expected every declared training trajectory")
    final_rows = {(r["size"], r["seed"], r["method"]): r for r in summary["training_final"]}
    require(len(final_rows) == len(summary["training_final"]) == len(grouped)
            and set(final_rows) == set(grouped), "Expected complete final-row summary coverage")
    records, replays, checkpoint_paths = [], [], set()
    total_recording_time = 0.0
    for (size, seed), case in cases.items():
        directory = path/f"n{size}_seed{seed}"
        require(json.loads((directory/"calibration_summary.json").read_text()) == case,
                "Expected matching per-case calibration summary")
        require(case["loops"] == config["loops"] and case["controller_frozen_after_calibration"]
                and not case["known_gains_used_for_calibration"]
                and case["known_basis_is_an_exact_structural_prior"]
                and case["training_uses_identical_initial_physical_model_for_all_methods"],
                "Expected declared fixed-controller/known-wiring experiment")
        model, left, right, reference = make_loop_case(seed, size, config["loops"], input_size=64)
        with np.load(directory/"wiring.npz") as wiring, np.load(directory/"reference.npz") as saved:
            for name, expected in (("left", left), ("right", right)):
                require(np.array_equal(wiring[name], expected), "Expected independently regenerated known wiring")
            for name, expected in {**reference, "symmetric": model.symmetric, "inputs": model.inputs, "bias": model.bias}.items():
                require(np.array_equal(saved[name], expected), f"Expected independently regenerated reference {name}")
        controller, cost = verify_calibration(directory, config, case, model, left, right, reference, data)
        total_recording_time += cost["total_recording_time"]
        record = dict(source_run=str(path), size=size, seed=seed, config=config,
                      structured_error=case["controller_relative_error"], measurement_cost=cost,
                      gradient_audit=case["gradient_audit"])
        require(("dense_comparison" in case) == bool(config["dense_control"]), "Expected declared dense-control coverage")
        if config["dense_control"]:
            dense = case["dense_comparison"]
            require(dense["matched_duration_replicas_physical_model_and_noise_seed"] and not dense["used_for_training"],
                    "Expected a matched extra dense comparison, unused by training")
            _, dense_cost = verify_calibration(directory, config, case, model, left, right, reference, data, dense=True)
            total_recording_time += dense_cost["total_recording_time"]
            record.update(dense_error=dense["controller_relative_error"],
                          dense_extra_study_cost=dense_cost, dense_gradient_audit=dense["gradient_audit"])
        records.append(record)
        initial = {label: evaluate(model, data[f"{split}_x"], data[f"{split}_y"])
                   for split, label in (("train", "train"), ("val", "validation"))} if epochs else {}
        for method in methods if epochs else []:
            identity = size, seed, method
            series = sorted(grouped[identity], key=lambda r: r["epoch"])
            require([r["epoch"] for r in series] == list(range(epochs+1)), "Expected unique complete epoch coverage")
            for row in series:
                count = row["epoch"]*len(data["train_y"])*PHASES[method]
                require(row["equilibrations"] == row["state_reads"] == count and row["max_force_residual"] <= 1e-9,
                        "Expected correct training measurement counts and convergence")
                close(row["learning_rate"], config["learning_rate"], "paired learning rate")
                close(row["momentum"], config["momentum"], "paired optimizer momentum")
                probes = 4 if method == "learned_mc4" else 0
                close(row["total_probe_excitation_sq"], 2*row["epoch"]*len(data["train_y"])*probes*.01**2,
                      "training probe excitation", atol=1e-10)
                require((row["epoch"] == epochs) == ("test_loss" in row and "test_accuracy" in row),
                        "Expected test evaluation only at the final epoch")
                charged = method == "circulation_asymep"
                for name, value in (("calibration_total_record_time", cost["total_recording_time"]),
                                    ("calibration_scalar_measurement_reads", cost["scalar_measurement_reads"]),
                                    ("calibration_projection_readout_channels", 2*config["loops"]),
                                    ("learned_controller_coefficients", config["loops"])):
                    close(row[name], value if charged else 0, f"training calibration charge {name}")
            for label, value in initial.items():
                close(series[0][label+"_loss"], value["loss"], f"matched initial {label} loss")
                close(series[0][label+"_accuracy"], value["accuracy"], f"matched initial {label} accuracy")
            for name, value in final_rows[identity].items():
                require(name in series[-1], f"Expected final metric {name}")
                if isinstance(value, (int, float)):
                    close(series[-1][name], value, f"final summary {name}")
                else:
                    require(series[-1][name] == value, "Expected matching final summary text")
            checkpoint = directory/"training"/method/f"{method}_seed{seed}_lr{config['learning_rate']:g}.npz"
            checkpoint_paths.add(checkpoint)
            replay = replay_checkpoint(checkpoint, model, controller, data, config, method, series)
            replay.update(size=size, seed=seed, source_run=str(path))
            replays.append(replay)
            for row in series:
                row.update(source_run=str(path), declared_epochs=epochs,
                           calibration_process_noise_channels=size if charged else 0)
    require(set(path.glob("n*_seed*/training/*/*.npz")) == checkpoint_paths,
            "Expected exactly the declared training checkpoints")
    close(summary["total_calibration_recording_time"], total_recording_time, "total study calibration time including dense controls")
    source = dict(path=str(path), run=run, verified_counts=counts,
                  coverage_scope="only this declared run", quick=config["quick"], epochs=epochs)
    return source, records, rows, replays


def aggregate(records, rows):
    groups = {}
    for case in records:
        key = tuple(case["config"][name] for name in PROTOCOL)+(case["size"],)
        groups.setdefault(key, []).append(case)
    calibrations, training = [], []
    for key, cases in groups.items():
        require(len({c["seed"] for c in cases}) == len(cases),
                "Expected one case per seed/protocol; overlapping runs would double-count evidence")
        protocol = dict(zip(PROTOCOL, key[:-1]))
        group = dict(protocol=protocol, size=key[-1], seeds=sorted(c["seed"] for c in cases),
                     structured_error=mean_sd([c["structured_error"] for c in cases]),
                     measurement_cost=cases[0]["measurement_cost"])
        paired = [c for c in cases if "dense_error" in c]
        if paired:
            group.update(dense_error=mean_sd([c["dense_error"] for c in paired]),
                         dense_seeds=sorted(c["seed"] for c in paired),
                         dense_extra_study_cost=paired[0]["dense_extra_study_cost"])
        group["gradient_audit"] = {}
        for method in AUDIT_METHODS:
            selected = [next(r for r in c["gradient_audit"] if r["method"] == method) for c in cases]
            group["gradient_audit"][method] = {metric: mean_sd([r[metric] for r in selected])
                for metric in ("gradient_cosine", "input_gradient_cosine", "gradient_relative_error")}
        calibrations.append(group)
        sources = {c["source_run"] for c in cases}
        selected = [r for r in rows if r["source_run"] in sources and r["size"] == key[-1]
                    and r["epoch"] == r["declared_epochs"]]
        for method in sorted({r["method"] for r in selected}):
            finals = [r for r in selected if r["method"] == method]
            record = dict(protocol=protocol, size=key[-1], method=method,
                          seeds=sorted(int(r["seed"]) for r in finals),
                          equilibrations_per_example=PHASES[method])
            for metric in ("test_accuracy", "validation_accuracy", "test_loss", "equilibrations", "state_reads"):
                record[metric] = mean_sd([r[metric] for r in finals])
            training.append(record)
    return calibrations, training


def report(output, calibration, training, verification):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2 if training else 1, figsize=(11 if training else 6, 4.5), squeeze=False,
                                constrained_layout=True)
    axis = axes[0, 0]
    protocols = {}
    for row in calibration:
        key = json.dumps(row["protocol"], sort_keys=True)
        protocols.setdefault(key, []).append(row)
    for index, selected in enumerate(protocols.values(), 1):
        selected.sort(key=lambda r: r["size"])
        for method, label in (("structured_error", "Few-loop"), ("dense_error", "Dense same-K control")):
            values = [r for r in selected if method in r]
            if values:
                axis.errorbar([r["size"] for r in values], [r[method]["mean"] for r in values],
                              yerr=[r[method]["std"] for r in values], marker="o", capsize=3,
                              label=f"{label}, protocol {index}")
    axis.set(xlabel="State count n", ylabel="Controller relative Frobenius error", title="Matched calibration budgets")
    axis.legend(fontsize=8)
    if training:
        axis = axes[0, 1]
        by_method = {}
        for row in training:
            key = row["method"], json.dumps(row["protocol"], sort_keys=True)
            by_method.setdefault(key, []).append(row)
        for (method, _), selected in by_method.items():
            selected.sort(key=lambda r: r["size"])
            axis.errorbar([r["size"] for r in selected], [100*r["test_accuracy"]["mean"] for r in selected],
                          yerr=[100*r["test_accuracy"]["std"] for r in selected], marker="o", capsize=3,
                          label=LABELS[method])
        axis.set(xlabel="State count n", ylabel="Final test accuracy (%)", title="Training from matched initialization")
        axis.legend(fontsize=8)
    figure.savefig(output/"structured_results.png", dpi=180)
    figure.savefig(output/"structured_results.svg")
    plt.close(figure)
    text = ["# Exploratory few-loop circulation results", "",
            f"Verified {verification['calibrations']} structured calibrations and {verification['checkpoint_replays']} checkpoints.",
            "Coverage refers only to the grids declared below; calibration-only and quick smoke runs do not establish a larger study.", "",
            "| n | Seeds | Duration / replicas | Quick / epochs | Structured error | Dense error (paired seeds) |", 
            "|---:|---|---|---|---:|---:|"]
    for row in calibration:
        p = row["protocol"]
        dense = row.get("dense_error")
        dense_text = f"{dense['mean']:.4f} ± {dense['std']:.4f} ({row['dense_seeds']})" if dense else "not run"
        err = row["structured_error"]
        text.append(f"| {row['size']} | {row['seeds']} | {p['duration']} / {p['replicas']} | {p['quick']} / {p['epochs']} | {err['mean']:.4f} ± {err['std']:.4f} | {dense_text} |")
    text += ["", "Each structured case learns L gains with 2nL supplied wiring coefficients. It reads 2L projections but injects independent process noise into all n states. Read counts, injection counts, recording time, and training equilibrations are reported separately in summary.json; dense comparisons add study cost.", "",
             "| n | Protocol | Audit | Gradient cosine | Input-gradient cosine |", "|---:|---:|---|---:|---:|"]
    for row in calibration:
        for method, metrics in row["gradient_audit"].items():
            text.append(f"| {row['size']} | {list(protocols).index(json.dumps(row['protocol'], sort_keys=True))+1} | {method} | {metrics['gradient_cosine']['mean']:.4f} | {metrics['input_gradient_cosine']['mean']:.4f} |")
    if training:
        text += ["", "| n | Epochs / quick | Method | Test accuracy (%) | Training equilibrations |", "|---:|---|---|---:|---:|"]
        for row in training:
            accuracy = row["test_accuracy"]
            text.append(f"| {row['size']} | {row['protocol']['epochs']} / {row['protocol']['quick']} | {LABELS[row['method']]} | {100*accuracy['mean']:.2f} ± {100*accuracy['std']:.2f} | {row['equilibrations']['mean']:.0f} |")
    text += ["", "All validation/test checkpoints and initial-condition metrics were replayed. Gradient audits were independently recomputed. This verifies artifact consistency and declared experimental coverage; it does not establish novelty, hardware efficiency, or a general low-dimensional model of arbitrary non-reciprocity.", ""]
    (output/"report.md").write_text("\n".join(text))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"Expected a new output directory; got {args.output}")
    if len({p.resolve() for p in args.runs}) != len(args.runs):
        parser.error("Expected distinct source run paths")
    sources, records, rows, replays = [], [], [], []
    for path in args.runs:
        source, cases, metrics, checks = read_run(path)
        sources.append(source)
        records.extend(cases)
        rows.extend(metrics)
        replays.extend(checks)
    calibration, training = aggregate(records, rows)
    verification = dict(declared_coverage_verified=True,
                        coverage_scope="each input run's declared grid only; no inferred research-grid completion",
                        calibrations=len(records), dense_comparisons=sum("dense_error" in r for r in records),
                        checkpoint_replays=len(replays), gradient_audits_recomputed=True,
                        source_runs_unchanged=True)
    args.output.mkdir(parents=True)
    write_json(args.output/"summary.json", dict(calibration=calibration, training=training,
                                                verification=verification, cases=records))
    write_json(args.output/"verification.json", dict(**verification, checkpoint_replays_details=replays))
    write_json(args.output/"sources.json", sources)
    if rows:
        write_csv(args.output/"metrics.csv", rows)
    report(args.output, calibration, training, verification)
    print(json.dumps(verification, indent=2))


if __name__ == "__main__":
    main()
