"""Validate the collected six-cell pilot and plot target-amplitude dependence."""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
from pathlib import Path

import torch

from experiments.exact_run import _training_artifact_errors
from experiments.reporting import validate_run

ROOT = Path(__file__).resolve().parents[1]
SCHEMES = ("baseline", "ours", "legacy")
LABELS = {"baseline": "Baseline", "ours": "Balanced (ours)", "legacy": "Legacy"}
COLORS = {"baseline": "#4477AA", "ours": "#228833", "legacy": "#CC6677"}


def read(path):
    value = json.loads(path.read_text())
    json.dumps(value, allow_nan=False)
    return value


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_cell(root, case, index, *, smoke):
    epochs = 1 if smoke else 30
    parent = root / "local-canary/smoke" if smoke else root / "runs"
    matches = sorted(parent.glob(f"{index:03d}_*"))
    if len(matches) != 1:
        raise ValueError(f"expected one bundle, found {len(matches)}")
    run = matches[0]
    errors = validate_run(run)
    if errors:
        raise ValueError("; ".join(errors))
    manifest, status, result, metrics, resolved = (
        read(run / name) for name in
        ("manifest.json", "status.json", "result.json", "metrics.json", "config.json")
    )
    expected = read(root / "source" / case["path"])
    assert sha(root / "source" / case["path"]) == case["sha256"]
    assert manifest["configuration"]["sha256"] == case["sha256"]
    assert manifest["configuration"]["resolved"] == expected
    assert manifest["arm_id"] == case["arm_id"]
    assert status["state"] == "complete" and result["completion"]["criteria_met"]
    assert manifest["smoke"] is smoke
    assert manifest["dataset"]["official_test_read"] is False
    assert metrics["official_test_evaluations"] == 0
    assert resolved["target_scale"] == case["target_scale"]
    assert resolved["training_algorithm"] == "BP" and resolved["seed"] == 0
    assert resolved["training"]["runtime_dtype"] == "float32"
    assert resolved["training"]["epochs"] == epochs
    assert resolved["training"]["num_iterations_training"] == 6
    assert resolved["training"]["num_iterations_inference"] == 6
    assert resolved["optimizer"]["learning_rate"] == expected["lr"]
    assert metrics["final_learning_rate"] == expected["lr"]
    assert resolved["training"]["optimizer_steps_applied"]
    assert resolved["training"]["max_batches"] == (1 if smoke else None)
    assert resolved["training"]["max_test_batches"] == (1 if smoke else None)
    errors = _training_artifact_errors(run, metrics, epochs=epochs, evaluation_split="validation")
    if errors:
        raise ValueError("; ".join(errors))
    for name in ("best_model.pt", "final_model.pt"):
        payload = torch.load(run / name, map_location="cpu", weights_only=False)
        biases = 0
        for spec, tensor in zip(payload["schema"], payload["states"], strict=True):
            assert torch.isfinite(tensor).all(), f"nonfinite tensor in {name}"
            if spec["name"].strip().startswith("Bias_"):
                biases += 1
                assert torch.count_nonzero(tensor) == 0, f"nonzero bias in {name}"
        assert biases == 2
    history = [read_metric for line in (run / "metrics.jsonl").read_text().splitlines()
               if line.strip() for read_metric in [json.loads(line)]]
    alpha = case["target_scale"]
    row = dict(scheme=case["scheme"], target_scale=alpha, historical=False,
               best_accuracy_percent=100 * metrics["best_test_accuracy"],
               final_accuracy_percent=100 * metrics["final_test_accuracy"],
               best_epoch=metrics["best_epoch"], final_mse=metrics["final_test_loss"],
               normalized_final_mse=metrics["final_test_loss"] / alpha ** 2,
               path=str(run.relative_to(ROOT)), result_sha256=sha(run / "result.json"))
    common = (resolved["architecture"]["initial_parameter_state_sha256"],
              json.dumps(resolved["dataset"]["provenance"], sort_keys=True))
    return row, history, common


def historical_rows():
    rows = []
    for scheme in SCHEMES:
        run = ROOT / "paper_ready_results/bundles/table1_wide_bptt/conv2" / scheme / "seed0"
        metrics = read(run / "metrics.json")
        rows.append(dict(scheme=scheme, target_scale=1.0, historical=True,
                         best_accuracy_percent=100 * metrics["best_test_accuracy"],
                         final_accuracy_percent=100 * metrics["final_test_accuracy"],
                         best_epoch=metrics["best_epoch"], final_mse=metrics["final_test_loss"],
                         normalized_final_mse=metrics["final_test_loss"],
                         path=str(run.relative_to(ROOT)), result_sha256=sha(run / "result.json")))
    return rows


def reversals(rows):
    lookup = {(r["scheme"], r["target_scale"]): r for r in rows}
    changes = []
    for first, second in itertools.combinations(SCHEMES, 2):
        for a, b in itertools.combinations((1 / 16, 1 / 4, 1.0), 2):
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
        x = [r["target_scale"] for r in group]
        y = [r["best_accuracy_percent"] for r in group]
        ax.plot(x[:2], y[:2], "o-", color=COLORS[scheme], label=LABELS[scheme])
        ax.plot(x[1:], y[1:], ":", color=COLORS[scheme])
        ax.plot(x[2], y[2], "o", mfc="white", mec=COLORS[scheme], mew=1.7)
    ax.set_xscale("log", base=2)
    ax.set_xticks([1 / 16, 1 / 4, 1], ["1/16", "1/4", "1 (historical)"])
    ax.set(xlabel="One-hot target amplitude", ylabel="Best validation accuracy (%)",
           title="Conv2 · fixed per-scheme Adam recipes · seed 0 · 30 epochs")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.text(0.5, 0.015, "Filled: new Jean Zay runs. Open: historical target-1 runs. Single-seed pilot.",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(out / "accuracy_vs_target.jpg", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4), sharey=True)
    for ax, alpha in zip(axes, (1 / 16, 1 / 4)):
        for scheme in SCHEMES:
            history = histories[scheme, alpha]
            ax.plot([h["epoch"] for h in history],
                    [100 * h["metrics"]["validation_accuracy"] for h in history],
                    color=COLORS[scheme], label=LABELS[scheme])
        ax.set(title=f"Target amplitude {alpha:g}", xlabel="Epoch")
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("Validation accuracy (%)")
    axes[1].legend()
    fig.suptitle("Conv2 · seed 0 · inherited learning rates")
    fig.tight_layout()
    fig.savefig(out / "validation_learning_curves.jpg", dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-root", type=Path, default=ROOT / "results/conv2-target-scale-20260928-v1")
    parser.add_argument("--smoke", action="store_true", help="Validate local smoke only; do not plot it.")
    args = parser.parse_args()
    root = args.study_root.resolve()
    cases = read(root / "preparation.json")["cases"]
    assert len(cases) == 6
    rows, histories, matching, errors = [], {}, [], []
    for index, case in enumerate(cases):
        try:
            row, history, common = validate_cell(root, case, index, smoke=args.smoke)
            rows.append(row)
            histories[case["scheme"], case["target_scale"]] = history
            matching.append(common)
        except (AssertionError, ValueError, KeyError, OSError, RuntimeError) as exc:
            errors.append(f"{case['arm_id']}: {type(exc).__name__}: {exc}")
    if matching and len(set(matching)) != 1:
        errors.append("Initializer or training/validation cohort/order differs between cells")
    complete = len(rows) == 6 and not errors
    summary = dict(complete=complete, expected=6, validated=len(rows), smoke=args.smoke, errors=errors,
                   limitations="Single seed and fixed per-scheme rates; target-1 references are historical. "
                   "Accuracy is validation evidence, not an official-test or causal-equivalence result.")
    out = root / "analysis"
    out.mkdir(exist_ok=True)
    if complete and not args.smoke:
        rows.extend(historical_rows())
        summary["ranking_reversals"] = reversals(rows)
        plot(rows, histories, out)
        with (out / "accuracy.csv").open("w") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    summary["rows"] = rows
    (out / ("smoke_validation.json" if args.smoke else "summary.json")).write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}, indent=2))
    return 0 if complete else 1


if __name__ == "__main__":
    raise SystemExit(main())
