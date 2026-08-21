#!/usr/bin/env python3
"""Validate and summarize the Conv1 bounded-uniform absolute-scale diagnostic."""

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

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

try:
    from experiments.reporting import validate_run
except ModuleNotFoundError:  # Support direct execution from the repository root.
    from reporting import validate_run  # type: ignore[no-redef]


REPO_ROOT = Path(__file__).resolve().parents[1]
STUDY_ID = "perfectdiode-conv1-bounded-uniform-scale1e4-adam-3ep-seed0-20260813-v1"
EVIDENCE_CLASS = "ordinary_mnist_absolute_weight_scale_exploratory"
DEFAULT_STUDY_ROOT = REPO_ROOT / "results" / STUDY_ID
SCHEME_ORDER = ("baseline", "ours", "legacy")
SCALE_FACTOR = 1.0e4


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
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"Expected finite {label}, got {value!r}.")
    return result


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


def _weight_audit(path: Path, *, weight_min: float, weight_max: float) -> dict[str, Any]:
    total = lower = upper = outside = 0
    observed_min = math.inf
    observed_max = -math.inf
    with np.load(path, allow_pickle=False) as checkpoint:
        names = [str(value) for value in checkpoint["param_names"].tolist()]
        for name in names:
            values = checkpoint[name]
            if name.startswith(("ConvWeight_", "DenseWeight_")):
                lo = np.asarray(weight_min, dtype=values.dtype)[()]
                hi = np.asarray(weight_max, dtype=values.dtype)[()]
                total += int(values.size)
                lower += int(np.count_nonzero(values == lo))
                upper += int(np.count_nonzero(values == hi))
                outside += int(np.count_nonzero((values < lo) | (values > hi)))
                observed_min = min(observed_min, float(np.min(values)))
                observed_max = max(observed_max, float(np.max(values)))
            elif name.startswith("Bias_") and np.count_nonzero(values) != 0:
                raise ValueError(f"Expected exact-zero final bias in {path}: {name}")
    if total <= 0 or outside:
        raise ValueError(f"Invalid final bounded weights in {path}: total={total}, outside={outside}")
    return {
        "weight_count": total,
        "lower_endpoint_count": lower,
        "upper_endpoint_count": upper,
        "either_endpoint_count": lower + upper,
        "lower_endpoint_percent": 100.0 * lower / total,
        "upper_endpoint_percent": 100.0 * upper / total,
        "either_endpoint_percent": 100.0 * (lower + upper) / total,
        "weight_min_observed": observed_min,
        "weight_max_observed": observed_max,
    }


def _epoch_rows(
    run_dir: Path, *, source_run_dir: Path, scheme: str
) -> list[dict[str, Any]]:
    source_losses = np.load(source_run_dir / "loss_test.npy")
    source_accuracies = np.load(source_run_dir / "accuracy_test.npy")
    if source_losses.shape != (3,) or source_accuracies.shape != (3,):
        raise ValueError(f"Expected three source epochs in {source_run_dir}.")
    rows = []
    for line in (run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        if record.get("kind") != "epoch":
            continue
        metrics = record["metrics"]
        epoch_index = int(record["epoch"])
        scaled_loss = _finite(metrics["validation_loss"], "validation loss")
        scaled_accuracy = 100.0 * _finite(
            metrics["validation_accuracy"], "validation accuracy"
        )
        source_loss = _finite(source_losses[epoch_index - 1].item(), "source validation loss")
        source_accuracy = 100.0 * _finite(
            source_accuracies[epoch_index - 1].item(), "source validation accuracy"
        )
        rows.append(
            {
                "scheme": scheme,
                "epoch": epoch_index,
                "train_loss": _finite(metrics["train_loss"], "train loss"),
                "train_accuracy_percent": 100.0
                * _finite(metrics["train_accuracy"], "train accuracy"),
                "source_validation_loss": source_loss,
                "scaled_validation_loss": scaled_loss,
                "scaled_minus_source_validation_loss": scaled_loss - source_loss,
                "source_validation_accuracy_percent": source_accuracy,
                "scaled_validation_accuracy_percent": scaled_accuracy,
                "scaled_minus_source_validation_accuracy_pp": (
                    scaled_accuracy - source_accuracy
                ),
                "best_validation_accuracy_percent": 100.0
                * _finite(metrics["best_validation_accuracy"], "best validation accuracy"),
                "is_best": bool(metrics["is_best"]),
            }
        )
    if [row["epoch"] for row in rows] != [1, 2, 3]:
        raise ValueError(f"Expected epoch rows 1,2,3 in {run_dir}, got {rows!r}.")
    return rows


def _normalized_weight_delta(
    source_path: Path, scaled_path: Path, *, scale_factor: float
) -> dict[str, float]:
    with np.load(source_path, allow_pickle=False) as source, np.load(
        scaled_path, allow_pickle=False
    ) as scaled:
        source_names = [
            str(value)
            for value in source["param_names"].tolist()
            if str(value).startswith(("ConvWeight_", "DenseWeight_"))
        ]
        scaled_names = [
            str(value)
            for value in scaled["param_names"].tolist()
            if str(value).startswith(("ConvWeight_", "DenseWeight_"))
        ]
        if source_names != scaled_names:
            raise ValueError("Source and scaled weight parameter orders differ.")
        source_values = np.concatenate(
            [source[name].reshape(-1) for name in source_names]
        ).astype(np.float64)
        normalized_scaled_values = np.concatenate(
            [scaled[name].reshape(-1) for name in scaled_names]
        ).astype(np.float64) / scale_factor
    delta = normalized_scaled_values - source_values
    source_rms = float(np.sqrt(np.mean(np.square(source_values))))
    return {
        "normalized_final_weight_max_absolute_delta": float(np.max(np.abs(delta))),
        "normalized_final_weight_rms_delta": float(np.sqrt(np.mean(np.square(delta)))),
        "normalized_final_weight_relative_rms_delta": float(
            np.sqrt(np.mean(np.square(delta))) / source_rms
        ),
    }


def _dataset_hashes(metrics: Mapping[str, Any]) -> dict[str, str]:
    provenance = metrics.get("dataset_provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError("Missing dataset provenance.")
    return {
        key: str(provenance[key])
        for key in (
            "train_indices_sha256",
            "validation_indices_sha256",
            "first_epoch_batch_order_sha256",
        )
    }


def analyze(study_root: Path, output_dir: Path | None = None) -> dict[str, Any]:
    study_root = Path(study_root).expanduser().resolve()
    output_dir = (
        Path(output_dir).expanduser().resolve()
        if output_dir is not None
        else study_root / "analysis"
    )
    manifest = _load_json(study_root / "study_manifest.json")
    if manifest.get("study_id") != STUDY_ID or manifest.get("run_count") != 3:
        raise ValueError(f"Unexpected study manifest in {study_root}.")
    if manifest.get("canonical_lr_handoff") is not False:
        raise ValueError("Absolute-scale diagnostic must remain noncanonical.")
    expected_by_sha = {row["config_sha256"]: row for row in manifest["runs"]}
    run_dirs = sorted(path.parent for path in (study_root / "runs").glob("*/manifest.json"))
    if len(run_dirs) != 3:
        raise ValueError(f"Expected exactly three production runs, found {len(run_dirs)}.")

    rows: list[dict[str, Any]] = []
    epochs: list[dict[str, Any]] = []
    seen_config_sha: set[str] = set()
    target_dataset_hashes: dict[str, set[str]] = defaultdict(set)
    target_initial_hashes: set[str] = set()
    for run_dir in run_dirs:
        errors = validate_run(run_dir)
        if errors:
            raise ValueError(f"Invalid canonical bundle {run_dir}: {errors!r}")
        run_manifest = _load_json(run_dir / "manifest.json")
        if run_manifest.get("study_id") != STUDY_ID:
            raise ValueError(f"Unexpected run study identity: {run_dir}")
        if run_manifest.get("evidence_class") != EVIDENCE_CLASS:
            raise ValueError(f"Unexpected evidence class: {run_dir}")
        config_sha = run_manifest["configuration"]["sha256"]
        if config_sha not in expected_by_sha or config_sha in seen_config_sha:
            raise ValueError(f"Unexpected or duplicate config authority: {run_dir}")
        seen_config_sha.add(config_sha)
        expected = expected_by_sha[config_sha]
        scheme = str(expected["scheme"])
        config = run_manifest["configuration"]["resolved"]
        transfer = config["absolute_scale_transfer"]
        metrics = _load_json(run_dir / "metrics.json")
        result = _load_json(run_dir / "result.json")

        if metrics.get("official_test_evaluations") != 0:
            raise ValueError(f"Official test was evaluated in {run_dir}.")
        if metrics.get("epoch_checkpoint_count") != 4:
            raise ValueError(f"Expected initialization plus three epoch checkpoints: {run_dir}")
        if result.get("dataset", {}).get("official_test_read") is not False:
            raise ValueError(f"Result does not prove official_test_read=false: {run_dir}")
        if config["model_base"]["weight_min"] != 0.1 or config["model_base"]["weight_max"] != 1.0:
            raise ValueError(f"Unexpected scaled bounds in {run_dir}.")
        source_rates = transfer["source_learning_rates_by_parameter"]
        scaled_rates = transfer["scaled_learning_rates_by_parameter"]
        for name, source_rate in source_rates.items():
            if not math.isclose(
                float(scaled_rates[name]),
                float(source_rate) * SCALE_FACTOR,
                rel_tol=0.0,
                abs_tol=0.0,
            ):
                raise ValueError(f"LR scale mismatch for {scheme}/{name}.")

        hashes = _dataset_hashes(metrics)
        for key, value in hashes.items():
            target_dataset_hashes[key].add(value)
        initial_hash = str(metrics.get("initial_parameter_state_sha256"))
        if not initial_hash or initial_hash == "None":
            raise ValueError(f"Missing initial parameter state hash: {run_dir}")
        target_initial_hashes.add(initial_hash)

        source_config_path = REPO_ROOT / transfer["source_config"]
        if _sha256(source_config_path) != transfer["source_config_sha256"]:
            raise ValueError(f"Source config digest mismatch: {source_config_path}")
        source_dir = source_config_path.parent
        source_errors = validate_run(source_dir)
        if source_errors:
            raise ValueError(f"Invalid source run {source_dir}: {source_errors!r}")
        source_metrics = _load_json(source_dir / "metrics.json")
        if source_metrics.get("official_test_evaluations") != 0:
            raise ValueError(f"Official test was evaluated in source {source_dir}.")
        if _dataset_hashes(source_metrics) != hashes:
            raise ValueError(f"Dataset/order mismatch between source and scaled run for {scheme}.")

        source_accuracy = 100.0 * _finite(
            source_metrics["final_validation_accuracy"], "source validation accuracy"
        )
        expected_source_accuracy = _finite(
            expected["source_final_validation_accuracy_percent"], "expected source accuracy"
        )
        if not math.isclose(source_accuracy, expected_source_accuracy, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(f"Source accuracy mismatch for {scheme}.")
        scaled_final_accuracy = 100.0 * _finite(
            metrics["final_validation_accuracy"], "scaled final validation accuracy"
        )
        scaled_best_accuracy = 100.0 * _finite(
            metrics["best_validation_accuracy"], "scaled best validation accuracy"
        )
        source_audit = _weight_audit(
            source_dir / "weights_final.npz", weight_min=1.0e-5, weight_max=1.0e-4
        )
        scaled_audit = _weight_audit(
            run_dir / "weights_final.npz", weight_min=0.1, weight_max=1.0
        )
        epoch_comparison = _epoch_rows(run_dir, source_run_dir=source_dir, scheme=scheme)
        weight_delta = _normalized_weight_delta(
            source_dir / "weights_final.npz",
            run_dir / "weights_final.npz",
            scale_factor=SCALE_FACTOR,
        )
        rows.append(
            {
                "scheme": scheme,
                "optimizer": "Adam",
                "source_weight_min": 1.0e-5,
                "source_weight_max": 1.0e-4,
                "scaled_weight_min": 0.1,
                "scaled_weight_max": 1.0,
                "weight_and_lr_scale_factor": SCALE_FACTOR,
                "source_final_validation_accuracy_percent": source_accuracy,
                "scaled_final_validation_accuracy_percent": scaled_final_accuracy,
                "scaled_best_validation_accuracy_percent": scaled_best_accuracy,
                "scaled_minus_source_final_accuracy_pp": (
                    scaled_final_accuracy - source_accuracy
                ),
                "scaled_final_validation_loss": _finite(
                    metrics["final_validation_loss"], "scaled final validation loss"
                ),
                "scaled_best_epoch": int(metrics["best_epoch"]),
                "source_final_bound_occupancy_percent": source_audit[
                    "either_endpoint_percent"
                ],
                "scaled_final_bound_occupancy_percent": scaled_audit[
                    "either_endpoint_percent"
                ],
                "scaled_final_lower_occupancy_percent": scaled_audit[
                    "lower_endpoint_percent"
                ],
                "scaled_final_upper_occupancy_percent": scaled_audit[
                    "upper_endpoint_percent"
                ],
                "scaled_final_weight_min_observed": scaled_audit["weight_min_observed"],
                "scaled_final_weight_max_observed": scaled_audit["weight_max_observed"],
                "max_absolute_epoch_validation_accuracy_delta_pp": max(
                    abs(float(epoch["scaled_minus_source_validation_accuracy_pp"]))
                    for epoch in epoch_comparison
                ),
                "max_absolute_epoch_validation_loss_delta": max(
                    abs(float(epoch["scaled_minus_source_validation_loss"]))
                    for epoch in epoch_comparison
                ),
                **weight_delta,
                "initial_parameter_state_sha256": initial_hash,
                "config_sha256": config_sha,
                "result_sha256": _sha256(run_dir / "result.json"),
                "source_result_sha256": _sha256(source_dir / "result.json"),
                "run_dir": str(run_dir.relative_to(study_root)),
                "source_run_dir": str(source_dir.relative_to(REPO_ROOT)),
            }
        )
        epochs.extend(epoch_comparison)

    if seen_config_sha != set(expected_by_sha):
        raise ValueError("Production coverage does not match the frozen config manifest.")
    if any(len(values) != 1 for values in target_dataset_hashes.values()):
        raise ValueError(f"Dataset/cohort hashes differ across runs: {target_dataset_hashes!r}")
    if len(target_initial_hashes) != 1:
        raise ValueError(f"Initial parameter states differ across runs: {target_initial_hashes!r}")
    rows.sort(key=lambda row: SCHEME_ORDER.index(str(row["scheme"])))
    epochs.sort(key=lambda row: (SCHEME_ORDER.index(str(row["scheme"])), int(row["epoch"])))

    by_scheme = {str(row["scheme"]): row for row in rows}
    margins = []
    for comparator in ("baseline", "ours"):
        source_margin = (
            float(by_scheme["legacy"]["source_final_validation_accuracy_percent"])
            - float(by_scheme[comparator]["source_final_validation_accuracy_percent"])
        )
        scaled_margin = (
            float(by_scheme["legacy"]["scaled_final_validation_accuracy_percent"])
            - float(by_scheme[comparator]["scaled_final_validation_accuracy_percent"])
        )
        margins.append(
            {
                "comparison": f"legacy_minus_{comparator}",
                "source_margin_pp": source_margin,
                "scaled_margin_pp": scaled_margin,
                "margin_change_pp": scaled_margin - source_margin,
            }
        )

    deltas = [float(row["scaled_minus_source_final_accuracy_pp"]) for row in rows]
    summary = {
        "study_id": STUDY_ID,
        "evidence_class": EVIDENCE_CLASS,
        "run_count": len(rows),
        "all_bundles_valid": True,
        "all_official_test_reads_zero": True,
        "matched_dataset_and_first_epoch_order": True,
        "shared_scaled_initial_parameter_state_sha256": next(iter(target_initial_hashes)),
        "final_accuracy_delta_range_pp": [min(deltas), max(deltas)],
        "median_final_accuracy_delta_pp": float(np.median(deltas)),
        "maximum_absolute_epoch_accuracy_delta_pp": max(
            float(row["max_absolute_epoch_validation_accuracy_delta_pp"])
            for row in rows
        ),
        "normalized_final_weight_relative_rms_delta_range": [
            min(float(row["normalized_final_weight_relative_rms_delta"]) for row in rows),
            max(float(row["normalized_final_weight_relative_rms_delta"]) for row in rows),
        ],
        "all_scaled_best_epochs_are_terminal": all(
            int(row["scaled_best_epoch"]) == 3 for row in rows
        ),
        "rows": rows,
        "legacy_margins": margins,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "summary.csv", rows)
    _write_csv(output_dir / "epochs.csv", epochs)
    _write_csv(output_dir / "legacy_margins.csv", margins)
    _write_json(output_dir / "summary.json", summary)

    x = np.arange(len(rows))
    width = 0.36
    fig, axis = plt.subplots(figsize=(7.2, 4.4))
    axis.bar(
        x - width / 2,
        [float(row["source_final_validation_accuracy_percent"]) for row in rows],
        width,
        label="[1e-5, 1e-4] source",
        color="#7a8ea3",
    )
    axis.bar(
        x + width / 2,
        [float(row["scaled_final_validation_accuracy_percent"]) for row in rows],
        width,
        label="[0.1, 1] + LR x1e4",
        color="#d97843",
    )
    axis.set_xticks(x, [str(row["scheme"]) for row in rows])
    axis.set_ylabel("Final validation accuracy (%)")
    axis.set_title("Conv1 bounded-uniform Adam: absolute-scale repeat")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_dir / "accuracy_comparison.png", dpi=180)
    plt.close(fig)

    report_lines = [
        "# Conv1 bounded-uniform absolute-scale repeat",
        "",
        "Exploratory ordinary-MNIST diagnostic; not a canonical LR handoff or paper-facing result.",
        "",
        "| Scheme | Source final | Scaled final | Delta | Scaled best (epoch) | Source occupancy | Scaled occupancy |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        report_lines.append(
            "| {scheme} | {source:.2f}% | {scaled:.2f}% | {delta:+.2f} pp | "
            "{best:.2f}% ({epoch}) | {source_occ:.2f}% | {scaled_occ:.2f}% |".format(
                scheme=row["scheme"],
                source=float(row["source_final_validation_accuracy_percent"]),
                scaled=float(row["scaled_final_validation_accuracy_percent"]),
                delta=float(row["scaled_minus_source_final_accuracy_pp"]),
                best=float(row["scaled_best_validation_accuracy_percent"]),
                epoch=int(row["scaled_best_epoch"]),
                source_occ=float(row["source_final_bound_occupancy_percent"]),
                scaled_occ=float(row["scaled_final_bound_occupancy_percent"]),
            )
        )
    report_lines.extend(
        [
            "",
            "The initialization uses the exact source checkpoint quantiles multiplied by `1e4`; "
            "all nonzero raw Adam learning rates are multiplied by the same factor. The dynamic "
            "range remains 10:1, biases remain exactly zero, and the dataset split plus first-epoch "
            "batch order match each source run.",
            "",
            "Across all nine matched epoch accuracies, the largest absolute difference is "
            f"`{summary['maximum_absolute_epoch_accuracy_delta_pp']:.2f} pp`. After dividing the "
            "scaled final weights by `1e4`, their relative RMS difference from the source final "
            "weights ranges from "
            f"`{summary['normalized_final_weight_relative_rms_delta_range'][0]:.4%}` to "
            f"`{summary['normalized_final_weight_relative_rms_delta_range'][1]:.4%}`.",
            "",
            "All three canonical bundles validate, stay within `[0.1,1]`, complete three epochs, "
            "and record no official-test read.",
        ]
    )
    (output_dir / "report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-root", type=Path, default=DEFAULT_STUDY_ROOT)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    print(json.dumps(analyze(args.study_root, args.output_dir), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
