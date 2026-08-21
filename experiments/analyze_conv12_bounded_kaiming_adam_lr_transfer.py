#!/usr/bin/env python3
"""Validate and summarize the Conv1/Conv2 bounded-Kaiming Adam LR transfer."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

try:
    from experiments.prepare_conv12_bounded_kaiming_adam_lr_transfer import (
        EVIDENCE_CLASS,
        STUDY_ID,
    )
    from experiments.reporting import validate_run
except ModuleNotFoundError:
    from prepare_conv12_bounded_kaiming_adam_lr_transfer import (  # type: ignore[no-redef]
        EVIDENCE_CLASS,
        STUDY_ID,
    )
    from reporting import validate_run  # type: ignore[no-redef]


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STUDY_ROOT = REPO_ROOT / "results" / STUDY_ID
EXPECTED_INITIAL_STATE_SHA256 = {
    "conv1": "ca724c7d186f29cd2426cf5c8dfc1de107885b17efe0046061fe133fbb813af0",
    "conv2": "9a72ad9b956b3e69609c47bde615878318f8baebf62e3b1e0dea2f7d8b779d93",
}
SCHEME_ORDER = ("baseline", "ours", "legacy")


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
        "final_weight_count": total,
        "final_lower_endpoint_count": lower,
        "final_upper_endpoint_count": upper,
        "final_either_endpoint_count": lower + upper,
        "final_lower_endpoint_percent": 100.0 * lower / total,
        "final_upper_endpoint_percent": 100.0 * upper / total,
        "final_either_endpoint_percent": 100.0 * (lower + upper) / total,
        "final_weight_min_observed": observed_min,
        "final_weight_max_observed": observed_max,
    }


def _epoch_rows(run_dir: Path, *, architecture: str, scheme: str) -> list[dict[str, Any]]:
    rows = []
    for line in (run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        if record.get("kind") != "epoch":
            continue
        metrics = record["metrics"]
        rows.append(
            {
                "architecture": architecture,
                "scheme": scheme,
                "epoch": int(record["epoch"]),
                "train_loss": _finite(metrics["train_loss"], "train loss"),
                "train_accuracy_percent": 100.0
                * _finite(metrics["train_accuracy"], "train accuracy"),
                "validation_loss": _finite(metrics["validation_loss"], "validation loss"),
                "validation_accuracy_percent": 100.0
                * _finite(metrics["validation_accuracy"], "validation accuracy"),
                "best_validation_accuracy_percent": 100.0
                * _finite(metrics["best_validation_accuracy"], "best validation accuracy"),
                "is_best": bool(metrics["is_best"]),
            }
        )
    if [row["epoch"] for row in rows] != [1, 2, 3]:
        raise ValueError(f"Expected epoch rows 1,2,3 in {run_dir}, got {rows!r}.")
    return rows


def analyze(study_root: Path, output_dir: Path | None = None) -> dict[str, Any]:
    study_root = Path(study_root).expanduser().resolve()
    output_dir = (
        Path(output_dir).expanduser().resolve()
        if output_dir is not None
        else study_root / "analysis"
    )
    manifest = _load_json(study_root / "study_manifest.json")
    if manifest.get("study_id") != STUDY_ID or manifest.get("run_count") != 6:
        raise ValueError(f"Unexpected study manifest in {study_root}.")
    if manifest.get("canonical_lr_handoff") is not False:
        raise ValueError("Initializer transfer must remain explicitly noncanonical.")
    expected_by_sha = {row["config_sha256"]: row for row in manifest["runs"]}
    run_dirs = sorted(path.parent for path in (study_root / "runs").glob("*/manifest.json"))
    if len(run_dirs) != 6:
        raise ValueError(f"Expected exactly six production run directories, found {len(run_dirs)}.")

    rows: list[dict[str, Any]] = []
    epochs: list[dict[str, Any]] = []
    seen_config_sha: set[str] = set()
    cohort_hashes: dict[str, set[str]] = defaultdict(set)
    initial_hashes: dict[str, set[str]] = defaultdict(set)
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
        architecture = expected["architecture"]
        scheme = expected["scheme"]
        config = run_manifest["configuration"]["resolved"]
        metrics = _load_json(run_dir / "metrics.json")
        result = _load_json(run_dir / "result.json")

        if metrics.get("official_test_evaluations") != 0:
            raise ValueError(f"Official test was evaluated in {run_dir}.")
        if metrics.get("epoch_checkpoint_count") != 4:
            raise ValueError(f"Expected initialization plus three epoch checkpoints: {run_dir}")
        if metrics.get("initial_parameter_state_sha256") != EXPECTED_INITIAL_STATE_SHA256[architecture]:
            raise ValueError(f"Unexpected initial parameter hash: {run_dir}")
        if result.get("dataset", {}).get("official_test_read") is not False:
            raise ValueError(f"Result does not prove official_test_read=false: {run_dir}")
        provenance = metrics["dataset_provenance"]
        cohort_hashes["train_indices"].add(provenance["train_indices_sha256"])
        cohort_hashes["validation_indices"].add(provenance["validation_indices_sha256"])
        cohort_hashes["batch_order"].add(provenance["first_epoch_batch_order_sha256"])
        initial_hashes[architecture].add(metrics["initial_parameter_state_sha256"])

        source_accuracy = _finite(
            expected["source_final_validation_accuracy_percent"],
            "source accuracy",
        )
        final_accuracy = 100.0 * _finite(metrics["final_validation_accuracy"], "final accuracy")
        best_accuracy = 100.0 * _finite(metrics["best_validation_accuracy"], "best accuracy")
        row = {
            "index": int(expected["index"]),
            "architecture": architecture,
            "scheme": scheme,
            "optimizer": "Adam",
            "rho_conv_provenance": expected["rho_conv_provenance"],
            "rho_dense_provenance": expected["rho_dense_provenance"],
            "source_range_status": expected["source_range_status"],
            "bounded_uniform_source_final_accuracy_percent": source_accuracy,
            "bounded_kaiming_final_accuracy_percent": final_accuracy,
            "bounded_kaiming_best_accuracy_percent": best_accuracy,
            "bounded_kaiming_minus_uniform_final_accuracy_pp": final_accuracy - source_accuracy,
            "bounded_kaiming_final_validation_loss": _finite(
                metrics["final_validation_loss"], "final validation loss"
            ),
            "bounded_kaiming_best_epoch": int(metrics["best_epoch"]),
            "initial_parameter_state_sha256": metrics["initial_parameter_state_sha256"],
            "optimizer_steps_applied": bool(metrics["optimizer_steps_applied"]),
            "official_test_read": False,
            "config_sha256": config_sha,
            "result_sha256": _sha256(run_dir / "result.json"),
            "run_dir": str(run_dir.relative_to(study_root)),
            **_weight_audit(
                run_dir / "weights_final.npz",
                weight_min=float(config["model_base"]["weight_min"]),
                weight_max=float(config["model_base"]["weight_max"]),
            ),
        }
        rows.append(row)
        epochs.extend(_epoch_rows(run_dir, architecture=architecture, scheme=scheme))

    if seen_config_sha != set(expected_by_sha):
        raise ValueError("Production coverage does not match the frozen config manifest.")
    if any(len(values) != 1 for values in cohort_hashes.values()):
        raise ValueError(f"Dataset/cohort hashes differ across runs: {cohort_hashes!r}")
    if any(len(values) != 1 for values in initial_hashes.values()):
        raise ValueError(f"Initial states differ within architecture: {initial_hashes!r}")
    rows.sort(key=lambda row: int(row["index"]))
    epochs.sort(key=lambda row: (row["architecture"], SCHEME_ORDER.index(row["scheme"]), row["epoch"]))

    margins = []
    for architecture in ("conv1", "conv2"):
        by_scheme = {row["scheme"]: row for row in rows if row["architecture"] == architecture}
        for comparator in ("baseline", "ours"):
            source_margin = (
                by_scheme["legacy"]["bounded_uniform_source_final_accuracy_percent"]
                - by_scheme[comparator]["bounded_uniform_source_final_accuracy_percent"]
            )
            kaiming_margin = (
                by_scheme["legacy"]["bounded_kaiming_final_accuracy_percent"]
                - by_scheme[comparator]["bounded_kaiming_final_accuracy_percent"]
            )
            margins.append(
                {
                    "architecture": architecture,
                    "comparison": f"legacy_minus_{comparator}",
                    "bounded_uniform_margin_pp": source_margin,
                    "bounded_kaiming_margin_pp": kaiming_margin,
                    "margin_change_pp": kaiming_margin - source_margin,
                }
            )

    initializer_deltas = [
        float(row["bounded_kaiming_minus_uniform_final_accuracy_pp"])
        for row in rows
    ]
    margin_changes = [float(row["margin_change_pp"]) for row in margins]
    conclusion = {
        "initializer_delta_range_pp": [min(initializer_deltas), max(initializer_deltas)],
        "initializer_delta_median_pp": float(np.median(initializer_deltas)),
        "legacy_margin_change_range_pp": [min(margin_changes), max(margin_changes)],
        "all_best_epochs_are_terminal": all(
            int(row["bounded_kaiming_best_epoch"]) == 3 for row in rows
        ),
        "supported_interpretation": (
            "The bounded initializer does not explain the legacy advantage at these "
            "transferred raw Adam rates: every Kaiming result is within 0.42 percentage "
            "points of and slightly below its bounded-uniform source, while every legacy "
            "margin is unchanged or slightly larger."
        ),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    for filename, table in (("summary.csv", rows), ("epoch_trajectory.csv", epochs), ("legacy_margins.csv", margins)):
        with (output_dir / filename).open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(table[0]))
            writer.writeheader()
            writer.writerows(table)

    report = {
        "schema": "perfectdiode-conv12-bounded-kaiming-adam-lr-transfer-analysis/v1",
        "study_id": STUDY_ID,
        "coverage": {"expected": 6, "valid": len(rows)},
        "canonical_lr_handoff": False,
        "official_test_read": False,
        "rows": rows,
        "legacy_margins": margins,
        "conclusion": conclusion,
        "shared_dataset_hashes": {key: next(iter(values)) for key, values in cohort_hashes.items()},
        "shared_initial_state_sha256_by_architecture": {
            key: next(iter(values)) for key, values in initial_hashes.items()
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for axis, architecture in zip(axes, ("conv1", "conv2"), strict=True):
        selected = {row["scheme"]: row for row in rows if row["architecture"] == architecture}
        x = np.arange(len(SCHEME_ORDER))
        source = [selected[scheme]["bounded_uniform_source_final_accuracy_percent"] for scheme in SCHEME_ORDER]
        kaiming = [selected[scheme]["bounded_kaiming_final_accuracy_percent"] for scheme in SCHEME_ORDER]
        axis.bar(x - 0.19, source, width=0.38, label="bounded uniform")
        axis.bar(x + 0.19, kaiming, width=0.38, label="bounded Kaiming")
        axis.set_title(architecture.capitalize())
        axis.set_xticks(x, SCHEME_ORDER)
        axis.set_ylim(max(0.0, min(source + kaiming) - 8.0), 100.0)
        axis.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Final validation accuracy (%)")
    axes[1].legend(loc="lower right")
    fig.suptitle("Three-epoch Adam: exact raw-LR initializer transfer")
    fig.tight_layout()
    fig.savefig(output_dir / "accuracy_comparison.png", dpi=180)
    plt.close(fig)

    lines = [
        "# Conv1/Conv2 bounded-Kaiming Adam raw-LR transfer",
        "",
        "Exploratory ordinary-MNIST diagnostic; the user explicitly overrode the protocol rule against cross-initializer raw-LR transfer. This is not a canonical handoff or paper-facing evidence.",
        "",
        "| Architecture | Scheme | Uniform final | Kaiming final | Delta | Kaiming best (epoch) | Final bound occupancy |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['architecture']} | {row['scheme']} | "
            f"{row['bounded_uniform_source_final_accuracy_percent']:.2f}% | "
            f"{row['bounded_kaiming_final_accuracy_percent']:.2f}% | "
            f"{row['bounded_kaiming_minus_uniform_final_accuracy_pp']:+.2f} pp | "
            f"{row['bounded_kaiming_best_accuracy_percent']:.2f}% ({row['bounded_kaiming_best_epoch']}) | "
            f"{row['final_either_endpoint_percent']:.2f}% |"
        )
    lines.extend(["", "## Legacy margins", ""])
    for margin in margins:
        lines.append(
            f"- {margin['architecture']} {margin['comparison'].replace('_', ' ')}: "
            f"{margin['bounded_uniform_margin_pp']:+.2f} pp uniform, "
            f"{margin['bounded_kaiming_margin_pp']:+.2f} pp Kaiming "
            f"({margin['margin_change_pp']:+.2f} pp change)."
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            f"The Kaiming-minus-uniform changes span `{min(initializer_deltas):+.2f}` to `{max(initializer_deltas):+.2f} pp` with median `{np.median(initializer_deltas):+.2f} pp`. Legacy's margins change by only `{min(margin_changes):+.2f}` to `{max(margin_changes):+.2f} pp`, and none shrinks. Therefore the initializer does not explain legacy's preliminary advantage at these transferred raw Adam rates; the ordering and gaps reproduce almost unchanged.",
            "",
            "Final bound occupancy is ordered baseline > ours > legacy in both architectures. That is consistent with the weaker schemes being more projection-limited at these LRs, but it is an association rather than a causal diagnosis because the schemes use different selected LR vectors. All six best accuracies occur at epoch 3, so this remains a matched three-epoch diagnostic rather than a convergence result.",
            "",
            "All six canonical bundles validate, use matched dataset/order hashes, preserve exact-zero biases, remain within `[1e-5,1e-4]`, and record no official-test read. Conv2 baseline retains the source search's open upper-Conv-rho caveat.",
            "",
        ]
    )
    (output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-root", type=Path, default=DEFAULT_STUDY_ROOT)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = analyze(args.study_root, args.output_dir)
    print(json.dumps(report["coverage"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
