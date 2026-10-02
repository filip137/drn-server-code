"""Validate the six-cell Conv3 target-scale pilot and plot completed evidence."""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path

import torch

from experiments.analyze_conv2_target_scale import COLORS, LABELS, ROOT, SCHEMES, read, sha
from experiments.exact_run import _training_artifact_errors
from experiments.reporting import validate_run

SCALES = (1 / 16, 1 / 64)
AMPLIFICATION = {"baseline": (1, 1), "ours": (4, 1), "legacy": (4, 0.25)}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_lines(path):
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    json.dumps(rows, allow_nan=False)
    return rows


def finite_state(value, label):
    if isinstance(value, torch.Tensor):
        require(bool(torch.isfinite(value).all()), f"nonfinite tensor in {label}")
    elif isinstance(value, dict):
        for key, child in value.items():
            finite_state(child, f"{label}.{key}")
    elif isinstance(value, (tuple, list)):
        for child in value:
            finite_state(child, label)
    elif isinstance(value, float):
        require(math.isfinite(value), f"nonfinite scalar in {label}")


def model_state(path):
    """Validate saved model tensors and reproduce the runner's ordered-state hash."""
    payload = torch.load(path, map_location="cpu", weights_only=False)
    digest = hashlib.sha256(b"mnist-train-ordered-parameter-state/v1\0")
    biases = []
    for spec, tensor in zip(payload["schema"], payload["states"], strict=True):
        name = spec["name"].strip()
        finite_state(tensor, str(path))
        require(tensor.dtype == torch.float32, f"non-float32 parameter in {path}")
        if name.startswith("Bias_"):
            biases.append(name)
            require(torch.count_nonzero(tensor) == 0, f"nonzero bias in {path}")
        if name.startswith(("ConvWeight_", "DenseWeight_")):
            require(bool(((tensor >= 0) & (tensor <= 100)).all()), f"weight bounds in {path}")
        for item in (name.encode(), str(tensor.dtype).encode(),
                     json.dumps(list(tensor.shape)).encode()):
            digest.update(item)
            digest.update(b"\0")
        digest.update(tensor.contiguous().numpy().tobytes(order="C"))
    require(sorted(biases) == ["Bias_0", "Bias_1", "Bias_2"], f"expected three biases in {path}")
    return digest.hexdigest()


def validate_checkpoints(run, epochs, initial_hash, rates):
    records = read_lines(run / "epoch_checkpoint_index.jsonl")
    require([r["epoch"] for r in records] == list(range(epochs + 1)),
            "epoch checkpoint coverage must include initialization and every epoch")
    for record in records:
        for kind in ("model", "optimizer"):
            path = (run / record[f"{kind}_path"]).resolve()
            require(path.is_relative_to(run.resolve()), "checkpoint path escapes run bundle")
            require(sha(path) == record[f"{kind}_sha256"], f"checkpoint hash mismatch: {path}")
            if kind == "model":
                state_hash = model_state(path)
                if record["epoch"] == 0:
                    require(state_hash == initial_hash, "initial checkpoint parameter-state mismatch")
            else:
                state = torch.load(path, map_location="cpu", weights_only=False)
                finite_state(state, str(path))
                require(state["epoch"] == record["epoch"], "optimizer checkpoint epoch mismatch")
                require(state["model_checkpoint_sha256"] == record["model_sha256"],
                        "optimizer/model checkpoint pair mismatch")
                require([g["lr"] for g in state["optimizer_state_dict"]["param_groups"]] == rates,
                        "checkpoint learning rates changed")
                require("continuation_state" in state, "missing full-state continuation checkpoint")
    for name in ("best_model.pt", "final_model.pt"):
        model_state(run / name)


def validate_cell(root, case, index, *, smoke):
    epochs = 3 if smoke else 30
    parent = root / ("local-canary/smoke" if smoke else "runs")
    matches = sorted(parent.glob(f"{index:03d}_*"))
    require(len(matches) == 1, f"expected one bundle, found {len(matches)}")
    run = matches[0]
    errors = validate_run(run)
    require(not errors, "; ".join(errors))
    manifest, status, result, metrics, resolved = (
        read(run / name) for name in
        ("manifest.json", "status.json", "result.json", "metrics.json", "config.json")
    )
    source = root / "source" / case["path"]
    expected = read(source)
    require(sha(source) == case["sha256"], "source config hash mismatch")
    require(manifest["configuration"]["sha256"] == case["sha256"]
            and manifest["configuration"]["resolved"] == expected, "manifest config mismatch")
    require(manifest["arm_id"] == case["arm_id"], "arm identity mismatch")
    require(status["state"] == "complete" and result["completion"]["criteria_met"], "run incomplete")
    require(manifest["smoke"] is smoke, "smoke/production mismatch")
    require(manifest["dataset"]["official_test_read"] is False
            and metrics["official_test_evaluations"] == 0
            and metrics["official_test_examples"] == 0, "official-test access")
    evaluation = resolved["evaluation"]
    require(evaluation["epoch_split"] == "validation"
            and evaluation["checkpoint_selection"] == "maximum_validation_accuracy"
            and evaluation["official_test_policy"] == "disabled", "evaluation contract mismatch")
    require(resolved["target_scale"] == case["target_scale"] == expected["target_scale"],
            "target scale mismatch")
    require(resolved["training_algorithm"] == "BP" and resolved["seed"] == 0, "algorithm/seed mismatch")
    training = resolved["training"]
    for key, value in dict(runtime_dtype="float32", epochs=epochs, batch_size=16,
                           num_iterations_training=8, num_iterations_inference=8,
                           batch_state_policy="reset_each_batch", optimizer_steps_applied=True,
                           max_batches=2 if smoke else None,
                           max_test_batches=1 if smoke else None).items():
        require(training[key] == value, f"training.{key} mismatch")
    noise = training.get("eqprop", {})
    require(noise.get("endpoint_read_noise_std", 0) == 0
            and noise.get("endpoint_read_noise_eta", 0) == 0
            and not noise.get("input_read_noise", False), "clean training noise contract mismatch")
    params = resolved["dataset"]["params"]
    for key, value in dict(batch_size=16, validation_batch_size=64, split_seed=0, shuffle_seed=0,
                           affine_config=None, normalize=True, normalize_mean=0.1307,
                           normalize_std=0.3081, normalize_scale=0.3).items():
        require(params[key] == value, f"dataset.{key} mismatch")
    architecture = resolved["architecture"]
    require(architecture["layer_shapes"] == [[2, 28, 28], [64, 14, 14], [128, 7, 7],
                                             [256, 7, 7], [20]], "Conv3 architecture mismatch")
    require(architecture["input_gain"] == 360 and architecture["non_linearity"] == "perfect_diode",
            "input gain or diode mismatch")
    require(resolved["bounds"] == dict(weight_min=0.0, weight_max=100.0), "weight contract mismatch")
    require(tuple(resolved["amplification"][k] for k in ("voltage_amp", "current_amp"))
            == AMPLIFICATION[case["scheme"]], "amplification scheme mismatch")
    require(resolved["minimizer"]["adaptive_equilibrium"] is False, "adaptive equilibrium enabled")
    rates = expected["lr"]
    require(len(rates) == 7 and rates[-3:] == [0.0] * 3, "bias rates must be zero")
    require(resolved["optimizer"]["name"] == "Adam" and resolved["optimizer"]["lr_decay"] == 1
            and resolved["optimizer"]["learning_rate"] == rates
            and metrics["final_learning_rate"] == rates, "optimizer/rates mismatch")
    historical_config = ROOT / "paper_ready_results/bundles/table1_wide_bptt/conv3" / case["scheme"] / "seed0/config.json"
    require(rates == read(historical_config)["optimizer"]["learning_rate"], "original scheme rates changed")
    require(bool(expected["initialization"]["checkpoint_sha256"])
            and bool(resolved["init_checkpoint_path"]), "missing hash-verified initializer")
    initializer = root / "source" / expected["init_checkpoint_path"]
    require(sha(initializer) == expected["initialization"]["checkpoint_sha256"],
            "initializer file hash mismatch")
    errors = _training_artifact_errors(run, metrics, epochs=epochs, evaluation_split="validation")
    require(not errors, "; ".join(errors))
    provenance = resolved["dataset"]["provenance"]
    require(provenance == metrics["dataset_provenance"], "dataset provenance mismatch")
    require(provenance["source_split"] == "train"
            and provenance["split_seed"] == provenance["shuffle_seed"] == 0,
            "dataset split/seed provenance mismatch")
    for name in ("train_indices_sha256", "validation_indices_sha256", "first_epoch_batch_order_sha256"):
        require(isinstance(provenance[name], str) and len(provenance[name]) == 64,
                f"missing dataset hash: {name}")
    require(len(provenance["train_batch_order_sha256"]) == epochs, "training-order coverage mismatch")
    require(resolved["diagnostics"]["checkpoint_every_epoch"]
            and metrics["epoch_checkpoint_count"] == epochs + 1, "epoch checkpoints missing")
    initial_hash = architecture["initial_parameter_state_sha256"]
    require(model_state(initializer) == initial_hash, "initializer differs from actual initial parameters")
    validate_checkpoints(run, epochs, initial_hash, rates)
    history = read_lines(run / "metrics.jsonl")
    best = max(h["metrics"]["validation_accuracy"] for h in history)
    require(best == metrics["best_validation_accuracy"] == metrics["best_test_accuracy"],
            "best accuracy differs from epoch history")
    require(history[metrics["best_epoch"] - 1]["metrics"]["validation_accuracy"] == best,
            "selected epoch is not a validation maximum")
    require(history[-1]["metrics"]["validation_loss"] == metrics["final_test_loss"],
            "final loss differs from history")
    alpha = case["target_scale"]
    row = metric_row(metrics, scheme=case["scheme"], alpha=alpha, historical=False, run=run)
    common = (initial_hash, expected["initialization"]["checkpoint_sha256"],
              json.dumps(provenance, sort_keys=True))
    return row, history, common


def metric_row(metrics, *, scheme, alpha, historical, run):
    return dict(scheme=scheme, target_scale=alpha, historical=historical,
                best_accuracy_percent=100 * metrics["best_test_accuracy"],
                final_accuracy_percent=100 * metrics["final_test_accuracy"],
                best_epoch=metrics["best_epoch"], final_mse=metrics["final_test_loss"],
                normalized_final_mse=metrics["final_test_loss"] / alpha ** 2,
                path=str(run), result_sha256=sha(run / "result.json"))


def historical_rows():
    rows = []
    for scheme in SCHEMES:
        run = ROOT / "paper_ready_results/bundles/table1_wide_bptt/conv3" / scheme / "seed0"
        metrics = read(run / "metrics.json")
        require(metrics["official_test_evaluations"] == 0, "historical reference must be validation evidence")
        rows.append(metric_row(metrics, scheme=scheme, alpha=1.0, historical=True, run=run))
    return rows


def reversals(rows):
    lookup = {(r["scheme"], r["target_scale"]): r for r in rows}
    changes = []
    for first, second in itertools.combinations(SCHEMES, 2):
        for a, b in itertools.combinations(sorted((*SCALES, 1.0)), 2):
            gaps = [lookup[first, x]["best_accuracy_percent"]
                    - lookup[second, x]["best_accuracy_percent"] for x in (a, b)]
            if gaps[0] * gaps[1] < 0:
                changes.append(dict(schemes=[first, second], scales=[a, b],
                                    gaps_pp=gaps, uses_historical_reference=b == 1.0))
    return changes


def plot(rows, histories, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    for scheme in SCHEMES:
        group = sorted((r for r in rows if r["scheme"] == scheme), key=lambda r: r["target_scale"])
        x, y = [r["target_scale"] for r in group], [r["best_accuracy_percent"] for r in group]
        ax.plot(x[:2], y[:2], "o-", color=COLORS[scheme], label=LABELS[scheme])
        ax.plot(x[1:], y[1:], ":", color=COLORS[scheme])
        ax.plot(x[2], y[2], "o", mfc="white", mec=COLORS[scheme], mew=1.7)
    ax.set_xscale("log", base=2)
    ax.set_xticks([1 / 64, 1 / 16, 1], ["1/64", "1/16", "1 (historical)"])
    ax.set(xlabel="One-hot target amplitude", ylabel="Best validation accuracy (%)",
           title="Conv3 · fixed per-scheme Adam recipes · seed 0 · 30 epochs")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.text(0.5, 0.015, "Filled: new runs. Open: historical target-1 runs. Single-seed pilot.",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(out / "accuracy_vs_target.jpg", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
    for ax, alpha in zip(axes, SCALES):
        for scheme in SCHEMES:
            history = histories[scheme, alpha]
            ax.plot([h["epoch"] for h in history],
                    [100 * h["metrics"]["validation_accuracy"] for h in history],
                    color=COLORS[scheme], label=LABELS[scheme])
        ax.set(title=f"Target amplitude 1/{round(1 / alpha)}", xlabel="Epoch")
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("Validation accuracy (%)")
    axes[1].legend()
    fig.suptitle("Conv3 · seed 0 · inherited learning rates")
    fig.tight_layout()
    fig.savefig(out / "validation_learning_curves.jpg", dpi=180)
    plt.close(fig)


def collect(root, *, smoke):
    rows, histories, matching, errors = [], {}, [], []
    try:
        cases = read(root / "preparation.json")["cases"]
        require([(c["scheme"], c["target_scale"]) for c in cases]
                == list(itertools.product(SCHEMES, SCALES)), "six ordered cells required")
        require([c["index"] for c in cases] == list(range(6)), "case indices must be 0..5")
    except (ValueError, KeyError, OSError) as exc:
        cases = []
        errors.append(f"preparation: {exc}")
    for case in cases:
        try:
            row, history, common = validate_cell(root, case, case["index"], smoke=smoke)
            rows.append(row)
            histories[case["scheme"], case["target_scale"]] = history
            matching.append(common)
        except (ValueError, KeyError, OSError, RuntimeError, TypeError) as exc:
            errors.append(f"{case['arm_id']}: {type(exc).__name__}: {exc}")
    if matching and len(set(matching)) != 1:
        errors.append("Initializer or training/validation cohort/order differs between cells")
    summary = dict(complete=len(rows) == 6 and not errors, expected=6, validated=len(rows),
                   smoke=smoke, errors=errors, rows=rows,
                   limitations="Single seed and fixed per-scheme rates; target-1 references are historical. "
                   "Accuracy is validation evidence, not official-test or causal-equivalence evidence. "
                   "Native target-scaled MSE is divided by alpha squared only for reporting.")
    out = root / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    if summary["complete"] and not smoke:
        try:
            rows.extend(historical_rows())
            summary["ranking_reversals"] = reversals(rows)
            plot(rows, histories, out)
            with (out / "accuracy.csv").open("w") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
        except (ValueError, KeyError, OSError, RuntimeError) as exc:
            errors.append(f"historical comparison/plot: {exc}")
            summary["complete"] = False
    (out / ("smoke_validation.json" if smoke else "summary.json")).write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-root", type=Path, default=ROOT / "results/conv3-target-scale-20260929-v1")
    parser.add_argument("--smoke", action="store_true", help="Validate 3-epoch continuation smoke; no plots.")
    args = parser.parse_args()
    summary = collect(args.study_root.resolve(), smoke=args.smoke)
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}, indent=2))
    return 0 if summary["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
