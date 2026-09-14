"""Verify full-MNIST experiment coverage, replay checkpoints, and plot results.

Replay uses the independent original NumPy relaxation. Only completed run
directories are accepted. No estimator or optimizer state is modified.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import hashlib
import json
import multiprocessing
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from labs.mnist_eqprop_data import load_mnist
from labs.recurrent_eqprop import Classifier, loss_accuracy, settle
from labs.tools.run_random_nudge_hopfield import write_csv, write_json


LABELS = {"contrastive_ep":"Ordinary EqProp", "known_skew_asymep":"Known-skew reference",
          "dc_asymep":"DC calibrated + EqProp", "noise_asymep":"Noise calibrated + EqProp",
          "learned_mc4":"Learned baseline + MC4"}
COLORS = {"contrastive_ep":"#777777", "known_skew_asymep":"#222222", "dc_asymep":"#168c83",
          "noise_asymep":"#3675c1", "learned_mc4":"#d57b27"}


def read_json(path):
    return json.loads(Path(path).read_text())


def replay_checkpoint(payload):
    root, result_path = map(Path, payload)
    result = read_json(result_path)
    config = read_json(root/"config.json")
    path = result_path.with_name("final.npz")
    data, _ = load_mnist(config["raw_dir"], train_limit=config["train_limit"], val_limit=config["val_limit"])
    checkpoint = np.load(path)
    model = Classifier(**{key:checkpoint[key].copy() for key in ("symmetric", "skew", "inputs", "bias")},
                       outputs=int(checkpoint["outputs"]), cubic=float(checkpoint["cubic"]),
                       logit_scale=float(checkpoint["logit_scale"]), symmetric_cap=float(checkpoint["symmetric_cap"]))
    assert int(checkpoint["epoch"]) == config["epochs"]
    assert model.inputs.shape == (result["size"]-10, 784)
    assert all(np.isfinite(getattr(model, key)).all() for key in ("symmetric", "skew", "inputs", "bias"))
    np.testing.assert_allclose(model.symmetric, model.symmetric.T, atol=1e-13, rtol=0)
    np.testing.assert_array_equal(np.diag(model.symmetric), 0)
    assert np.linalg.norm(model.symmetric, 2) <= model.symmetric_cap+1e-12
    initial = np.load(path.parent.parent/"initial_model.npz")
    np.testing.assert_array_equal(model.skew, initial["skew"])
    if result["method"] in ("dc_asymep", "noise_asymep"):
        controllers = np.load(path.parent.parent/"controllers.npz")
        key = result["method"].split("_")[0]+"_controller"
        np.testing.assert_array_equal(checkpoint["feedback_controller"], controllers[key])
    saved_split = np.load(root/"dataset_split.npz")
    for key in saved_split.files:
        np.testing.assert_array_equal(data[key], saved_split[key])
    network = model.network()
    checks = {}
    for name, prefix in (("validation", "val"), ("test", "test")):
        if result[name] is None:
            continue
        x, y = data[prefix+"_x"], data[prefix+"_y"]
        loss, correct, max_residual = 0., 0, 0.
        for begin in range(0, len(x), 500):
            xb, yb = x[begin:begin+500], y[begin:begin+500]
            free = settle(network, model.drive(xb), tolerance=1e-10)
            batch_loss, _ = loss_accuracy(free.state, yb, model.hidden, model.logit_scale)
            loss += len(xb)*batch_loss
            correct += int(np.sum(free.state[:, model.hidden:].argmax(axis=-1) == yb))
            max_residual = max(max_residual, free.residual)
        loss /= len(x)
        assert correct == result[name]["correct"], (path, name, correct, result[name]["correct"])
        assert abs(loss-result[name]["loss"]) < 1e-7, (path, name, loss, result[name]["loss"])
        checks[name] = dict(examples=len(x), correct=correct, accuracy=correct/len(x),
                            loss=loss, max_force_residual=max_residual)
    counts = result["train_examples"]*config["epochs"]*(9 if result["method"] == "learned_mc4" else 3)
    assert result["training_equilibrations"] == result["training_state_reads"] == counts
    assert result["training_max_residual"] <= 1e-9
    assert int(checkpoint["optimizer_steps"]) == config["epochs"]*int(np.ceil(result["train_examples"]/config["batch_size"]))
    if result["method"] == "learned_mc4":
        assert 0 < int(checkpoint["predictor_observations"]) <= 4*result["train_examples"]*config["epochs"]
        assert np.isfinite(checkpoint["predictor_matrix"]).all()
    return dict(size=result["size"], seed=result["seed"], method=result["method"],
                checkpoint=str(path.relative_to(root)), checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                independent_numpy_replay=checks, training_equilibrations=counts, status="passed")


def aggregate(root, output, results):
    config = read_json(root/"config.json")
    calibrations = {(r["size"], r["seed"]):r for r in read_json(root/"calibration.json")}
    rows = []
    for size in config["sizes"]:
        for method in config["methods"]:
            runs = [r for r in results if r["size"] == size and r["method"] == method]
            test = np.array([r["test"]["accuracy"] for r in runs])
            val = np.array([r["validation"]["accuracy"] for r in runs])
            phases = runs[0]["training_equilibrations"]
            calibration = calibrations[(size, config["seeds"][0])]
            dc_counts, noise_counts = calibration["dc"]["counts"], calibration["noise"]["counts"]
            calibration_equilibrations = (dc_counts["equilibrations"]+dc_counts["free_anchor_equilibrations"]
                if method == "dc_asymep" else (calibration["noise"]["free_anchor_equilibrations"] if method == "noise_asymep" else 0))
            row = dict(size=size, hidden=size-10, method=method, seeds=len(runs),
                test_accuracy_mean=float(test.mean()), test_accuracy_sd=float(test.std()),
                validation_accuracy_mean=float(val.mean()), validation_accuracy_sd=float(val.std()),
                final_gradient_cosine_mean=float(np.mean([r["final_audit"]["gradient_cosine"] for r in runs])),
                initial_gradient_cosine_mean=float(np.mean([r["initial_audit"]["gradient_cosine"] for r in runs])),
                final_gradient_relative_error_mean=float(np.mean([r["final_audit"]["gradient_relative_error"] for r in runs])),
                final_baseline_relative_error_mean=float(np.mean([r["final_audit"]["baseline_relative_error"] for r in runs])),
                training_equilibrations_per_seed=phases,
                calibration_equilibrations_per_seed=calibration_equilibrations,
                calibration_record_time_per_seed=noise_counts["total_record_time"] if method == "noise_asymep" else 0,
                calibration_scalar_reads_per_seed=dc_counts["scalar_reads"]+dc_counts["anchor_projection_reads"]
                    if method == "dc_asymep" else (noise_counts["scalar_measurement_reads"] if method == "noise_asymep" else 0),
                runtime_seconds_mean=float(np.mean([r["elapsed_seconds"] for r in runs])))
            rows.append(row)
    write_csv(output/"summary.csv", rows)
    write_json(output/"summary.json", rows)
    methods, sizes = config["methods"], config["sizes"]
    fig, axes = plt.subplots(1, len(sizes), figsize=(6*len(sizes), 4.6), squeeze=False)
    for ax, size in zip(axes[0], sizes):
        for i, method in enumerate(methods):
            row = next(r for r in rows if r["size"] == size and r["method"] == method)
            ax.bar(i, 100*row["test_accuracy_mean"], yerr=100*row["test_accuracy_sd"],
                   color=COLORS[method], capsize=4, width=.72)
            ax.text(i, 100*row["test_accuracy_mean"]+1.5, f"{100*row['test_accuracy_mean']:.2f}", ha="center", fontsize=9)
        ax.set_xticks(range(len(methods)), [LABELS[m].replace(" + ", "\n+ ") for m in methods], rotation=23, ha="right")
        ax.set(title=f"MNIST: {size} states ({size-10} hidden)", ylabel="Final test accuracy (%)", ylim=(0, 104))
        ax.grid(axis="y", alpha=.2)
    fig.suptitle(f"Full 784-pixel MNIST · {config['epochs']} epochs · mean ± SD over {len(config['seeds'])} seeds")
    fig.tight_layout()
    fig.savefig(output/"test_accuracy.png", dpi=180)
    fig.savefig(output/"test_accuracy.pdf")
    plt.close(fig)
    fig, axes = plt.subplots(2, len(sizes), figsize=(6*len(sizes), 7.8), squeeze=False)
    for column, size in enumerate(sizes):
        for method in methods:
            curves = []
            costs = []
            for seed in config["seeds"]:
                with (root/f"n{size}_seed{seed}"/method/"metrics.csv").open() as stream:
                    curve = list(csv.DictReader(stream))
                curves.append([float(r["validation_accuracy"])*100 for r in curve])
                costs = [int(r["training_equilibrations"]) for r in curve]
            values = np.array(curves)
            mean, sd = values.mean(axis=0), values.std(axis=0)
            epochs = np.arange(len(mean))
            axes[0, column].plot(epochs, mean, color=COLORS[method], label=LABELS[method])
            axes[0, column].fill_between(epochs, mean-sd, mean+sd, color=COLORS[method], alpha=.13)
            axes[1, column].plot(np.array(costs[1:])/1e6, mean[1:], color=COLORS[method], label=LABELS[method])
        axes[0, column].set(title=f"{size} states", xlabel="Epoch", ylabel="Validation accuracy (%)", ylim=(0,100))
        axes[1, column].set(xlabel="Training equilibrations (millions)", ylabel="Validation accuracy (%)", ylim=(0,100))
        axes[0, column].legend(fontsize=8)
        for ax in axes[:, column]:
            ax.grid(alpha=.2)
    fig.suptitle("Identical optimizer settings; fixed four-gain structure\nOne-time calibration costs are listed separately in summary.csv")
    fig.tight_layout()
    fig.savefig(output/"learning_curves.png", dpi=180)
    fig.savefig(output/"learning_curves.pdf")
    plt.close(fig)
    lines = ["# Full-MNIST physical feedback comparison", "", "Exploratory; final epoch fixed in advance. Mean ± population SD across seeds.", "",
             "| States | Method | Test accuracy (%) | Final audit gradient cosine | Training equilibria |", "|---:|---|---:|---:|---:|"]
    for row in rows:
        lines.append(f"| {row['size']} | {LABELS[row['method']]} | {100*row['test_accuracy_mean']:.2f} ± {100*row['test_accuracy_sd']:.2f} | {row['final_gradient_cosine_mean']:.4f} | {row['training_equilibrations_per_seed']:,} |")
    dc_row = next((r for r in rows if r["method"] == "dc_asymep"), None)
    noise_row = next((r for r in rows if r["method"] == "noise_asymep"), None)
    if dc_row is not None:
        lines += ["", f"DC calibration adds {dc_row['calibration_equilibrations_per_seed']:,} equilibria and {dc_row['calibration_scalar_reads_per_seed']:,} scalar reads per seed."]
    if noise_row is not None:
        lines += [f"Noise calibration adds {noise_row['calibration_equilibrations_per_seed']} anchor and {noise_row['calibration_record_time_per_seed']:g} total record-time units, with {noise_row['calibration_scalar_reads_per_seed']:,} scalar projection reads."]
    lines += ["These costs are separate from training, validation, and audit costs.", "",
              "Both calibrated controllers exploit the same fixed four-gain wiring prior; no unrestricted Jacobian is estimated. MC4 is the prior measured-baseline algorithm with four fresh unit-norm random-sign probe pairs per training example. The other methods use centered contrastive EqProp.", "",
              "All methods use common LR 0.1 and bias-corrected gradient EMA 0.9, with 250-image batches. These settings were not separately tuned for each method. Error bars describe seed variation and do not establish statistical equivalence.", "",
              "Gradient audits use the same 16 training images, extra measured phases, and post-estimation exact adjoints. Audits never update the live predictor, model or optimizer. Independent NumPy replay verifies final validation and test metrics from every checkpoint.", ""]
    (output/"report.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"Expected a fresh report directory; got {args.output}")
    config, run = read_json(args.run/"config.json"), read_json(args.run/"run.json")
    assert read_json(args.run/"status.json")["status"] == "complete"
    assert not config["no_test"]
    results = read_json(args.run/"results.json")
    expected = {(n, seed, m) for n in config["sizes"] for seed in config["seeds"] for m in config["methods"]}
    actual = {(r["size"], r["seed"], r["method"]) for r in results}
    assert actual == expected and len(results) == len(expected) == run["expected_training"]
    calibration = read_json(args.run/"calibration.json")
    assert len(calibration)*2 == run["expected_calibrations"]
    for row in calibration:
        count = row["dc"]["counts"]
        assert count["equilibrations"] == count["scalar_reads"] == 4*config["loops"]*config["dc_steps"]
        assert row["noise"]["counts"]["total_record_time"] == 880
    args.output.mkdir(parents=True)
    write_json(args.output/"status.json", dict(status="replaying", completed=0, expected=len(expected)))
    payloads = [(str(args.run.resolve()), str((args.run/f"n{n}_seed{seed}"/m/"result.json").resolve()))
                for n, seed, m in sorted(expected)]
    checks = []
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        for future in as_completed([pool.submit(replay_checkpoint, payload) for payload in payloads]):
            check = future.result()
            checks.append(check)
            write_json(args.output/"checkpoint_replays.json", checks)
            write_json(args.output/"status.json", dict(status="replaying", completed=len(checks), expected=len(expected)))
            print(f"Verified {len(checks)}/{len(expected)}: n={check['size']} seed={check['seed']} {check['method']}", flush=True)
    aggregate(args.run, args.output, results)
    verification = dict(status="passed", training_trajectories=len(checks), calibrations=len(calibration)*2,
        source_commit=run["commit"], dtype="float64", independent_solver="original NumPy IMEX, residual <=1e-10",
        full_validation_and_test_replayed=True, exact_training_counts_checked=True,
        frozen_physical_skew_and_controller_checked=True, dataset_split_and_preprocessing_checked=True)
    write_json(args.output/"verification.json", verification)
    write_json(args.output/"status.json", dict(status="complete", completed=len(checks), expected=len(expected)))


if __name__ == "__main__":
    main()
