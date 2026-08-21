#!/usr/bin/env python3
"""Validate and summarize the matched Conv1 positive-only/zero-bias diagnostic."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from experiments.reporting import validate_run
from model.function.interaction import load_function_checkpoint_artifact


REPO_ROOT = Path(__file__).resolve().parents[1]
STUDY_ID = (
    "perfectdiode-conv1-positive-vs-zero-bias-ordinary-mnist-"
    "seed0-20260815-v1"
)
EVIDENCE_CLASS = "ordinary_mnist_bias_ablation"
CONFIG_ROOT = REPO_ROOT / "configs/conv" / STUDY_ID
DEFAULT_STUDY_ROOT = REPO_ROOT / "results" / STUDY_ID
EXPECTED_EPOCHS = 10
PRACTICAL_SIMILARITY_THRESHOLD_PP = 0.5
EXPECTED_BASE_COMMIT = "68d2f8552c703f06756fea48f8a3f7c2d885d6d7"
EXPECTED_SOURCE_ARCHIVE_SHA256 = (
    "f10761eedb120ded72a9c6e728ebe2ed79e158cffa7d48ebf6c5a6460a74a9b3"
)
ROLE_ORDER = ("positive_only_bias", "zero_bias")
DATASET_HASH_KEYS = (
    "train_indices_sha256",
    "validation_indices_sha256",
    "first_epoch_batch_order_sha256",
)


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"Expected finite {label}, got {value!r}.")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"Expected finite {label}, got {value!r}.")
    return number


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write an empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _epoch_rows(run_dir: Path, role: str) -> list[dict[str, Any]]:
    rows = []
    for line in (run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        if record.get("kind") != "epoch":
            continue
        metrics = record["metrics"]
        rows.append(
            {
                "role": role,
                "epoch": int(record["epoch"]),
                "train_loss": _finite(metrics["train_loss"], "train loss"),
                "train_accuracy_percent": 100.0
                * _finite(metrics["train_accuracy"], "train accuracy"),
                "validation_loss": _finite(
                    metrics["validation_loss"], "validation loss"
                ),
                "validation_accuracy_percent": 100.0
                * _finite(metrics["validation_accuracy"], "validation accuracy"),
                "best_validation_accuracy_percent": 100.0
                * _finite(
                    metrics["best_validation_accuracy"],
                    "best validation accuracy",
                ),
                "is_best": bool(metrics["is_best"]),
            }
        )
    expected = list(range(1, EXPECTED_EPOCHS + 1))
    if [row["epoch"] for row in rows] != expected:
        raise ValueError(
            f"Expected epoch records {expected} in {run_dir}, "
            f"got {[row['epoch'] for row in rows]}."
        )
    return rows


def _dataset_hashes(metrics: Mapping[str, Any]) -> dict[str, str]:
    provenance = metrics.get("dataset_provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError("Missing ordinary-MNIST train/validation provenance.")
    if provenance.get("schema") != "mnist-train-validation-split/v1":
        raise ValueError(f"Unexpected dataset provenance schema: {provenance!r}")
    batch_order = provenance.get("train_batch_order_sha256")
    if (
        not isinstance(batch_order, list)
        or len(batch_order) != EXPECTED_EPOCHS
        or not all(isinstance(value, str) for value in batch_order)
    ):
        raise ValueError("Missing complete ten-epoch minibatch-order provenance.")
    batch_schedule_sha256 = hashlib.sha256(
        ("\n".join(batch_order) + "\n").encode("ascii")
    ).hexdigest()
    return {
        **{key: str(provenance[key]) for key in DATASET_HASH_KEYS},
        "ten_epoch_batch_order_schedule_sha256": batch_schedule_sha256,
    }


def _checkpoint_payload(path: Path) -> tuple[dict[str, torch.Tensor], list[str]]:
    states, schema, source_format = load_function_checkpoint_artifact(
        path, map_location="cpu"
    )
    if source_format != "versioned" or not isinstance(schema, list):
        raise ValueError(f"Expected a versioned checkpoint with schema: {path}")
    if len(states) != len(schema):
        raise ValueError(f"Checkpoint state/schema length mismatch: {path}")
    by_name: dict[str, torch.Tensor] = {}
    bias_names = []
    for state, entry in zip(states, schema):
        name = str(entry.get("name", "")).strip()
        type_name = str(entry.get("type", ""))
        if not name or name in by_name:
            raise ValueError(f"Invalid or duplicate checkpoint parameter name: {path}")
        tensor = state.detach().cpu().contiguous()
        if not bool(torch.isfinite(tensor).all()):
            raise ValueError(f"Nonfinite checkpoint parameter {name!r}: {path}")
        by_name[name] = tensor
        if type_name.endswith(".Bias") or name.startswith("Bias_"):
            bias_names.append(name)
    if bias_names != ["Bias_0"]:
        raise ValueError(f"Expected exactly Bias_0 in {path}, got {bias_names!r}.")
    return by_name, bias_names


def _bias_stats(path: Path, *, role: str, label: str) -> dict[str, Any]:
    params, _ = _checkpoint_payload(path)
    values = params["Bias_0"].numpy()
    negative = int(np.count_nonzero(values < 0.0))
    positive = int(np.count_nonzero(values > 0.0))
    zero = int(np.count_nonzero(values == 0.0))
    return {
        "role": role,
        "checkpoint": label,
        "path": str(path),
        "count": int(values.size),
        "negative_count": negative,
        "zero_count": zero,
        "positive_count": positive,
        "minimum": float(np.min(values)),
        "maximum": float(np.max(values)),
        "mean": float(np.mean(values, dtype=np.float64)),
        "rms": float(np.sqrt(np.mean(np.square(values, dtype=np.float64)))),
        "nonzero_percent": 100.0 * (negative + positive) / int(values.size),
    }


def _checkpoint_rows(run_dir: Path, role: str) -> list[dict[str, Any]]:
    index_rows = [
        json.loads(line)
        for line in (run_dir / "epoch_checkpoint_index.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    if [int(row["epoch"]) for row in index_rows] != list(
        range(EXPECTED_EPOCHS + 1)
    ):
        raise ValueError(f"Unexpected epoch checkpoint coverage in {run_dir}.")
    rows = []
    for record in index_rows:
        path = run_dir / record["model_path"]
        if _sha256(path) != record["model_sha256"]:
            raise ValueError(f"Epoch checkpoint hash mismatch: {path}")
        rows.append(
            _bias_stats(path, role=role, label=f"epoch_{int(record['epoch']):03d}")
        )
    rows.append(_bias_stats(run_dir / "best_model.pt", role=role, label="best"))
    rows.append(_bias_stats(run_dir / "final_model.pt", role=role, label="final"))
    return rows


def _assert_matched_initialization(run_dirs: Mapping[str, Path]) -> dict[str, Any]:
    initial = {}
    for role, run_dir in run_dirs.items():
        index = json.loads(
            (run_dir / "epoch_checkpoint_index.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()[0]
        )
        if int(index["epoch"]) != 0:
            raise ValueError(f"First checkpoint is not initialization: {run_dir}")
        initial[role], _ = _checkpoint_payload(run_dir / index["model_path"])

    reference = initial[ROLE_ORDER[0]]
    comparison = initial[ROLE_ORDER[1]]
    if set(reference) != set(comparison):
        raise ValueError("The two initialization checkpoints name different parameters.")
    mismatched = [
        name for name in reference if not torch.equal(reference[name], comparison[name])
    ]
    if mismatched:
        raise ValueError(f"Initialization tensors differ across arms: {mismatched!r}")
    if bool(torch.count_nonzero(reference["Bias_0"])):
        raise ValueError("Bias_0 must initialize to exact zero in both arms.")
    return {
        "exact_all_parameter_tensors_matched": True,
        "parameter_names": sorted(reference),
        "bias_initialization_exact_zero": True,
    }


def _classify(
    *,
    delta_best_validation_accuracy_pp: float,
    positive_bias_active: bool,
) -> str:
    if not positive_bias_active:
        return "inconclusive_no_positive_bias_learned"
    if abs(delta_best_validation_accuracy_pp) <= PRACTICAL_SIMILARITY_THRESHOLD_PP:
        return "no_material_effect_observed"
    return "material_difference_observed"


def _plot(epoch_rows: list[dict[str, Any]], path: Path) -> None:
    labels = {
        "positive_only_bias": "positive-only learned bias",
        "zero_bias": "bias LR = 0",
    }
    colors = {"positive_only_bias": "#0072B2", "zero_bias": "#D55E00"}
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.1), constrained_layout=True)
    for role in ROLE_ORDER:
        rows = [row for row in epoch_rows if row["role"] == role]
        epochs = [row["epoch"] for row in rows]
        axes[0].plot(
            epochs,
            [row["validation_accuracy_percent"] for row in rows],
            marker="o",
            linewidth=1.8,
            markersize=4,
            label=labels[role],
            color=colors[role],
        )
        axes[1].plot(
            epochs,
            [row["validation_loss"] for row in rows],
            marker="o",
            linewidth=1.8,
            markersize=4,
            label=labels[role],
            color=colors[role],
        )
    axes[0].set_ylabel("Validation accuracy (%)")
    axes[1].set_ylabel("Validation loss")
    for axis in axes:
        axis.set_xlabel("Epoch")
        axis.set_xticks(range(1, EXPECTED_EPOCHS + 1))
        axis.grid(alpha=0.25)
    axes[0].legend(frameon=False)
    fig.suptitle("Perfect-diode Conv1 ordinary-MNIST bias diagnostic")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def analyze(study_root: Path, output_dir: Path | None = None) -> dict[str, Any]:
    study_root = Path(study_root).expanduser().resolve()
    output_dir = (
        Path(output_dir).expanduser().resolve()
        if output_dir is not None
        else study_root / "analysis"
    )
    study_manifest = _load_json(CONFIG_ROOT / "study_manifest.json")
    if (
        study_manifest.get("study_id") != STUDY_ID
        or study_manifest.get("expected_run_count") != 2
    ):
        raise ValueError("Unexpected source study manifest.")
    expected_by_sha = {row["config_sha256"]: row for row in study_manifest["runs"]}
    for digest, expected in expected_by_sha.items():
        config_path = REPO_ROOT / expected["config"]
        if _sha256(config_path) != digest:
            raise ValueError(f"Config authority hash mismatch: {config_path}")

    run_dirs_found = sorted(
        path.parent for path in (study_root / "production").glob("*/manifest.json")
    )
    if len(run_dirs_found) != 2:
        raise ValueError(f"Expected exactly two production runs, found {len(run_dirs_found)}.")

    run_dirs: dict[str, Path] = {}
    summaries: dict[str, dict[str, Any]] = {}
    epoch_rows: list[dict[str, Any]] = []
    checkpoint_rows: list[dict[str, Any]] = []
    dataset_hashes: dict[str, set[str]] = defaultdict(set)
    initial_hashes: set[str] = set()
    seen_configs: set[str] = set()
    for run_dir in run_dirs_found:
        errors = validate_run(run_dir)
        if errors:
            raise ValueError(f"Invalid canonical bundle {run_dir}: {errors!r}")
        manifest = _load_json(run_dir / "manifest.json")
        metrics = _load_json(run_dir / "metrics.json")
        result = _load_json(run_dir / "result.json")
        if manifest.get("study_id") != STUDY_ID:
            raise ValueError(f"Unexpected study identity: {run_dir}")
        if manifest.get("evidence_class") != EVIDENCE_CLASS:
            raise ValueError(f"Unexpected evidence class: {run_dir}")
        if manifest.get("smoke") is not False:
            raise ValueError(f"Production run is marked smoke: {run_dir}")
        git_state = manifest.get("git", {})
        if (
            git_state.get("commit") != EXPECTED_BASE_COMMIT
            or git_state.get("source_archive_sha256")
            != EXPECTED_SOURCE_ARCHIVE_SHA256
        ):
            raise ValueError(f"Unexpected frozen-source identity: {run_dir}")
        runtime = manifest.get("runtime", {})
        if runtime.get("target") != "akib" or manifest.get("runtime", {}).get(
            "device"
        ) != "cuda":
            raise ValueError(f"Unexpected production target/device: {run_dir}")
        config_sha = str(manifest["configuration"]["sha256"])
        if config_sha not in expected_by_sha or config_sha in seen_configs:
            raise ValueError(f"Unexpected or duplicate config authority: {run_dir}")
        seen_configs.add(config_sha)
        expected = expected_by_sha[config_sha]
        role = str(expected["role"])
        run_dirs[role] = run_dir
        config = manifest["configuration"]["resolved"]
        bias_lr = float(config["learning_rates_by_parameter"]["Bias_0"])
        expected_bias_lr = float(expected["bias_learning_rate"])
        if bias_lr != expected_bias_lr:
            raise ValueError(f"Bias learning-rate mismatch for {role}.")
        if config["evaluation"]["official_test"]["policy"] != "disabled":
            raise ValueError(f"Official-test policy is not disabled: {run_dir}")
        if metrics.get("official_test_evaluations") != 0:
            raise ValueError(f"Official test was evaluated: {run_dir}")
        if result.get("dataset", {}).get("official_test_read") is not False:
            raise ValueError(f"Result does not prove official_test_read=false: {run_dir}")
        if metrics.get("epoch_checkpoint_count") != EXPECTED_EPOCHS + 1:
            raise ValueError(f"Incomplete epoch-checkpoint coverage: {run_dir}")

        role_epochs = _epoch_rows(run_dir, role)
        epoch_rows.extend(role_epochs)
        role_checkpoint_rows = _checkpoint_rows(run_dir, role)
        checkpoint_rows.extend(role_checkpoint_rows)
        hashes = _dataset_hashes(metrics)
        for key, value in hashes.items():
            dataset_hashes[key].add(value)
        initial_hash = str(metrics.get("initial_parameter_state_sha256"))
        if not initial_hash or initial_hash == "None":
            raise ValueError(f"Missing initial state hash: {run_dir}")
        initial_hashes.add(initial_hash)
        last_three = role_epochs[-3:]
        summaries[role] = {
            "role": role,
            "arm_id": manifest["arm_id"],
            "run_dir": str(run_dir),
            "config_sha256": config_sha,
            "bias_learning_rate": bias_lr,
            "best_epoch": int(metrics["best_epoch"]),
            "best_validation_accuracy_percent": 100.0
            * _finite(metrics["best_validation_accuracy"], "best validation accuracy"),
            "final_validation_accuracy_percent": 100.0
            * _finite(metrics["final_validation_accuracy"], "final validation accuracy"),
            "final_validation_loss": _finite(
                metrics["final_validation_loss"], "final validation loss"
            ),
            "last_three_validation_accuracy_mean_percent": float(
                np.mean([row["validation_accuracy_percent"] for row in last_three])
            ),
            "last_three_validation_loss_mean": float(
                np.mean([row["validation_loss"] for row in last_three])
            ),
            "initial_parameter_state_sha256": initial_hash,
            **hashes,
        }

    if set(run_dirs) != set(ROLE_ORDER):
        raise ValueError(f"Missing expected roles: found {sorted(run_dirs)}")
    mismatched_hashes = {
        key: sorted(values) for key, values in dataset_hashes.items() if len(values) != 1
    }
    if mismatched_hashes:
        raise ValueError(f"Dataset/order hashes differ: {mismatched_hashes!r}")
    if len(initial_hashes) != 1:
        raise ValueError(f"Initial parameter hashes differ: {sorted(initial_hashes)!r}")
    initialization = _assert_matched_initialization(run_dirs)

    positive_rows = [
        row for row in checkpoint_rows if row["role"] == "positive_only_bias"
    ]
    zero_rows = [row for row in checkpoint_rows if row["role"] == "zero_bias"]
    if any(row["negative_count"] for row in positive_rows):
        raise ValueError("Positive-only Bias_0 contains a negative checkpoint value.")
    if any(row["negative_count"] or row["positive_count"] for row in zero_rows):
        raise ValueError("Zero-bias Bias_0 changed from exact zero.")
    selected_positive = {
        row["checkpoint"]: row["positive_count"]
        for row in positive_rows
        if row["checkpoint"] in {"best", "final"}
    }
    positive_bias_active = (
        set(selected_positive) == {"best", "final"}
        and all(count > 0 for count in selected_positive.values())
    )
    positive_final = next(
        row
        for row in positive_rows
        if row["checkpoint"] == "final"
    )
    zero_final = next(row for row in zero_rows if row["checkpoint"] == "final")
    for role, row in (
        ("positive_only_bias", positive_final),
        ("zero_bias", zero_final),
    ):
        summaries[role].update(
            {
                "final_bias_minimum": row["minimum"],
                "final_bias_maximum": row["maximum"],
                "final_bias_mean": row["mean"],
                "final_bias_rms": row["rms"],
                "final_bias_positive_count": row["positive_count"],
                "final_bias_nonzero_percent": row["nonzero_percent"],
            }
        )

    positive = summaries["positive_only_bias"]
    zero = summaries["zero_bias"]
    delta_best = (
        positive["best_validation_accuracy_percent"]
        - zero["best_validation_accuracy_percent"]
    )
    delta_final = (
        positive["final_validation_accuracy_percent"]
        - zero["final_validation_accuracy_percent"]
    )
    positive_epochs = {
        row["epoch"]: row
        for row in epoch_rows
        if row["role"] == "positive_only_bias"
    }
    zero_epochs = {
        row["epoch"]: row for row in epoch_rows if row["role"] == "zero_bias"
    }
    epoch_accuracy_deltas = [
        positive_epochs[epoch]["validation_accuracy_percent"]
        - zero_epochs[epoch]["validation_accuracy_percent"]
        for epoch in range(1, EXPECTED_EPOCHS + 1)
    ]
    epoch_loss_deltas = [
        positive_epochs[epoch]["validation_loss"]
        - zero_epochs[epoch]["validation_loss"]
        for epoch in range(1, EXPECTED_EPOCHS + 1)
    ]
    classification = _classify(
        delta_best_validation_accuracy_pp=delta_best,
        positive_bias_active=positive_bias_active,
    )
    criterion_met = classification == "no_material_effect_observed"
    comparison = {
        "positive_minus_zero_best_validation_accuracy_pp": delta_best,
        "absolute_best_validation_accuracy_difference_pp": abs(delta_best),
        "positive_minus_zero_final_validation_accuracy_pp": delta_final,
        "maximum_absolute_epoch_validation_accuracy_difference_pp": max(
            abs(value) for value in epoch_accuracy_deltas
        ),
        "maximum_absolute_epoch_validation_loss_difference": max(
            abs(value) for value in epoch_loss_deltas
        ),
        "epochs_with_identical_validation_accuracy": sum(
            value == 0.0 for value in epoch_accuracy_deltas
        ),
        "practical_similarity_threshold_pp": PRACTICAL_SIMILARITY_THRESHOLD_PP,
        "positive_bias_active_at_best_and_final": positive_bias_active,
        "practical_similarity_criterion_met": criterion_met,
        "classification": classification,
    }
    summary = {
        "schema_version": "conv1-positive-vs-zero-bias-analysis/v1",
        "study_id": STUDY_ID,
        "evidence_class": EVIDENCE_CLASS,
        "paper_facing": False,
        "scope": "single-seed ordinary-MNIST diagnostic",
        "runs": [summaries[role] for role in ROLE_ORDER],
        "comparison": comparison,
        "guards": {
            **initialization,
            "initial_parameter_state_sha256": next(iter(initial_hashes)),
            "dataset_and_minibatch_hashes_matched": True,
            "zero_bias_exact_at_every_saved_checkpoint": True,
            "positive_only_bias_nonnegative_at_every_saved_checkpoint": True,
            "official_test_evaluations": 0,
            "canonical_bundles_valid": True,
        },
    }

    _write_csv(output_dir / "summary.csv", [summaries[role] for role in ROLE_ORDER])
    _write_csv(output_dir / "epoch_metrics.csv", epoch_rows)
    _write_csv(output_dir / "bias_checkpoint_stats.csv", checkpoint_rows)
    _write_json(output_dir / "summary.json", summary)
    _plot(epoch_rows, output_dir / "accuracy_loss_trajectories.png")
    conclusion = {
        "no_material_effect_observed": (
            "The learned positive-only bias changed Bias_0, while its absolute "
            "best-validation accuracy difference from the zero-bias arm stayed "
            "within the predeclared 0.5 percentage-point threshold."
        ),
        "material_difference_observed": (
            "The absolute best-validation accuracy difference exceeded the "
            "predeclared 0.5 percentage-point threshold."
        ),
        "inconclusive_no_positive_bias_learned": (
            "The positive-only arm did not contain positive Bias_0 values at both "
            "best and final checkpoints, so this run cannot test active bias learning."
        ),
    }[classification]
    report = f"""# Perfect-diode Conv1 bias diagnostic

{conclusion}

The positive-only arm reached {positive['best_validation_accuracy_percent']:.3f}% best validation accuracy; the fixed-zero arm reached {zero['best_validation_accuracy_percent']:.3f}%. The signed difference (positive-only minus zero) is {delta_best:+.3f} percentage points, and the absolute difference is {abs(delta_best):.3f} percentage points. Validation accuracy is identical in {comparison['epochs_with_identical_validation_accuracy']}/10 epochs, and the maximum absolute epoch-level difference is {comparison['maximum_absolute_epoch_validation_accuracy_difference_pp']:.3f} percentage points.

Final validation accuracy was {positive['final_validation_accuracy_percent']:.3f}% versus {zero['final_validation_accuracy_percent']:.3f}% ({delta_final:+.3f} percentage points). The learned arm's final Bias_0 range was [{positive_final['minimum']:.6g}, {positive_final['maximum']:.6g}], with {positive_final['nonzero_percent']:.3f}% nonzero entries; Bias_0 remained exact zero in the control arm at every saved checkpoint.

This is a matched, single-seed ordinary-MNIST diagnostic. It supports only the stated diagnostic conclusion; it is neither paper-facing accuracy evidence nor a statistical equivalence test. The official test split was not evaluated.
"""
    (output_dir / "report.md").write_text(report, encoding="utf-8")
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-root", type=Path, default=DEFAULT_STUDY_ROOT)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    summary = analyze(args.study_root, args.output_dir)
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
