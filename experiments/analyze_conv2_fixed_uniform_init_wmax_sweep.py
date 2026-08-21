#!/usr/bin/env python3
"""Validate and analyze the Conv2 fixed-initializer upper-weight sweep."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np

try:
    from experiments.prepare_conv2_fixed_uniform_init_wmax_sweep import (
        EVIDENCE_CLASS,
        INITIALIZER_SHA256,
        STUDY_ID,
        WMAX_CASES,
    )
    from experiments.reporting import validate_run
except ModuleNotFoundError:
    from prepare_conv2_fixed_uniform_init_wmax_sweep import (  # type: ignore[no-redef]
        EVIDENCE_CLASS,
        INITIALIZER_SHA256,
        STUDY_ID,
        WMAX_CASES,
    )
    from reporting import validate_run  # type: ignore[no-redef]


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STUDY_ROOT = REPO_ROOT / "results" / STUDY_ID
SCHEME_ORDER = ("baseline", "ours", "legacy")
OPTIMIZER_ORDER = ("SGD", "Adam")
WEIGHT_MAX_VALUES = tuple(case.value for case in WMAX_CASES)


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


def _weight_audit(
    path: Path,
    *,
    weight_min: float,
    weight_max: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    pooled = {"count": 0, "lower": 0, "upper": 0, "outside": 0}
    observed_min = math.inf
    observed_max = -math.inf
    layers: list[dict[str, Any]] = []
    with np.load(path, allow_pickle=False) as checkpoint:
        names = [str(value) for value in checkpoint["param_names"].tolist()]
        for name in names:
            values = checkpoint[name]
            if name.startswith(("ConvWeight_", "DenseWeight_")):
                lo = np.asarray(weight_min, dtype=values.dtype)[()]
                hi = np.asarray(weight_max, dtype=values.dtype)[()]
                count = int(values.size)
                lower = int(np.count_nonzero(values == lo))
                upper = int(np.count_nonzero(values == hi))
                outside = int(np.count_nonzero((values < lo) | (values > hi)))
                layer_min = float(np.min(values))
                layer_max = float(np.max(values))
                layers.append(
                    {
                        "parameter": name,
                        "weight_count": count,
                        "lower_endpoint_count": lower,
                        "upper_endpoint_count": upper,
                        "either_endpoint_count": lower + upper,
                        "lower_endpoint_percent": 100.0 * lower / count,
                        "upper_endpoint_percent": 100.0 * upper / count,
                        "either_endpoint_percent": 100.0 * (lower + upper) / count,
                        "outside_count": outside,
                        "weight_min_observed": layer_min,
                        "weight_max_observed": layer_max,
                    }
                )
                pooled["count"] += count
                pooled["lower"] += lower
                pooled["upper"] += upper
                pooled["outside"] += outside
                observed_min = min(observed_min, layer_min)
                observed_max = max(observed_max, layer_max)
            elif name.startswith("Bias_") and np.count_nonzero(values) != 0:
                raise ValueError(f"Expected exact-zero final bias in {path}: {name}")
    if pooled["count"] <= 0 or pooled["outside"]:
        raise ValueError(f"Invalid bounded weights in {path}: {pooled!r}")
    count = pooled["count"]
    summary = {
        "final_weight_count": count,
        "final_lower_endpoint_count": pooled["lower"],
        "final_upper_endpoint_count": pooled["upper"],
        "final_either_endpoint_count": pooled["lower"] + pooled["upper"],
        "final_lower_endpoint_percent": 100.0 * pooled["lower"] / count,
        "final_upper_endpoint_percent": 100.0 * pooled["upper"] / count,
        "final_either_endpoint_percent": 100.0
        * (pooled["lower"] + pooled["upper"])
        / count,
        "final_weight_min_observed": observed_min,
        "final_weight_max_observed": observed_max,
    }
    return summary, layers


def _epoch_rows(
    run_dir: Path,
    *,
    index: int,
    scheme: str,
    optimizer: str,
    weight_max: float,
) -> list[dict[str, Any]]:
    rows = []
    for line in (run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        if record.get("kind") != "epoch":
            continue
        metrics = record["metrics"]
        rows.append(
            {
                "index": index,
                "scheme": scheme,
                "optimizer": optimizer,
                "weight_max": weight_max,
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
    if [row["epoch"] for row in rows] != list(range(1, 11)):
        raise ValueError(f"Expected exact epoch rows 1..10 in {run_dir}.")
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write an empty table: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def analyze(study_root: Path, output_dir: Path | None = None) -> dict[str, Any]:
    study_root = Path(study_root).expanduser().resolve()
    output_dir = (
        Path(output_dir).expanduser().resolve()
        if output_dir is not None
        else study_root / "analysis"
    )
    manifest = _load_json(study_root / "study_manifest.json")
    if manifest.get("study_id") != STUDY_ID or manifest.get("run_count") != 24:
        raise ValueError(f"Unexpected study manifest in {study_root}.")
    if manifest.get("canonical_lr_handoff") is not False:
        raise ValueError("The exploratory sweep must remain explicitly noncanonical.")
    initializer = study_root / "assets/bounded_uniform/conv2/final_model.pt"
    if _sha256(initializer) != INITIALIZER_SHA256:
        raise ValueError("Authoritative local initializer digest mismatch.")

    expected_by_sha = {row["config_sha256"]: row for row in manifest["runs"]}
    if len(expected_by_sha) != 24:
        raise ValueError("Frozen manifest contains duplicate config authorities.")
    run_dirs = sorted(path.parent for path in (study_root / "runs").glob("*/manifest.json"))
    if len(run_dirs) != 24:
        raise ValueError(f"Expected 24 production run directories, found {len(run_dirs)}.")

    rows: list[dict[str, Any]] = []
    epoch_rows: list[dict[str, Any]] = []
    occupancy_rows: list[dict[str, Any]] = []
    seen_config_sha: set[str] = set()
    cohort_hashes: dict[str, set[str]] = defaultdict(set)
    initial_hashes: set[str] = set()
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
        config = run_manifest["configuration"]["resolved"]
        metrics = _load_json(run_dir / "metrics.json")
        result = _load_json(run_dir / "result.json")
        scheme = expected["scheme"]
        optimizer = expected["optimizer"]
        index = int(expected["index"])
        weight_min = float(expected["weight_min"])
        weight_max = float(expected["weight_max"])

        if config["optimizer"]["name"] != optimizer:
            raise ValueError(f"Optimizer disagrees with manifest: {run_dir}")
        if config["learning_rates_by_parameter"] != expected["learning_rates_by_parameter"]:
            raise ValueError(f"Learning-rate mapping disagrees with manifest: {run_dir}")
        if any(config["learning_rates_by_parameter"][name] != 0.0 for name in ("Bias_0", "Bias_1")):
            raise ValueError(f"Bias learning rate is nonzero: {run_dir}")
        if float(config["model_base"]["weight_min"]) != weight_min or float(
            config["model_base"]["weight_max"]
        ) != weight_max:
            raise ValueError(f"Configured projection interval disagrees: {run_dir}")
        if int(config["lab"]["epochs"]) != 10:
            raise ValueError(f"Unexpected epoch budget: {run_dir}")
        if config["weight_ceiling_sweep"]["initializer_checkpoint_sha256"] != INITIALIZER_SHA256:
            raise ValueError(f"Initializer authority disagrees: {run_dir}")
        if metrics.get("official_test_evaluations") != 0:
            raise ValueError(f"Official test was evaluated: {run_dir}")
        if metrics.get("epoch_checkpoint_count") != 11:
            raise ValueError(f"Expected initialization plus ten epoch checkpoints: {run_dir}")
        initial_hash = metrics.get("initial_parameter_state_sha256")
        if not isinstance(initial_hash, str) or len(initial_hash) != 64:
            raise ValueError(f"Missing initial parameter-state digest: {run_dir}")
        initial_hashes.add(initial_hash)
        if result.get("dataset", {}).get("official_test_read") is not False:
            raise ValueError(f"Result does not prove official_test_read=false: {run_dir}")
        provenance = metrics["dataset_provenance"]
        cohort_hashes["train_indices"].add(provenance["train_indices_sha256"])
        cohort_hashes["validation_indices"].add(provenance["validation_indices_sha256"])
        cohort_hashes["batch_order"].add(provenance["first_epoch_batch_order_sha256"])

        pooled, layers = _weight_audit(
            run_dir / "weights_final.npz",
            weight_min=weight_min,
            weight_max=weight_max,
        )
        final_accuracy = 100.0 * _finite(
            metrics["final_validation_accuracy"], "final validation accuracy"
        )
        best_accuracy = 100.0 * _finite(
            metrics["best_validation_accuracy"], "best validation accuracy"
        )
        row = {
            "index": index,
            "scheme": scheme,
            "optimizer": optimizer,
            "weight_min": weight_min,
            "weight_max": weight_max,
            "rho_conv_provenance": expected["rho_conv_provenance"],
            "rho_dense_provenance": expected["rho_dense_provenance"],
            "source_range_status": expected["source_range_status"],
            "source_three_epoch_final_accuracy_percent": expected[
                "source_final_validation_accuracy_percent"
            ],
            "final_validation_accuracy_percent": final_accuracy,
            "best_validation_accuracy_percent": best_accuracy,
            "best_epoch": int(metrics["best_epoch"]),
            "final_validation_loss": _finite(
                metrics["final_validation_loss"], "final validation loss"
            ),
            "optimizer_steps_applied": bool(metrics["optimizer_steps_applied"]),
            "initial_parameter_state_sha256": initial_hash,
            "official_test_read": False,
            "config_sha256": config_sha,
            "result_sha256": _sha256(run_dir / "result.json"),
            "run_dir": str(run_dir.relative_to(study_root)),
            **pooled,
        }
        rows.append(row)
        epoch_rows.extend(
            _epoch_rows(
                run_dir,
                index=index,
                scheme=scheme,
                optimizer=optimizer,
                weight_max=weight_max,
            )
        )
        for layer in layers:
            occupancy_rows.append(
                {
                    "index": index,
                    "scheme": scheme,
                    "optimizer": optimizer,
                    "weight_min": weight_min,
                    "weight_max": weight_max,
                    **layer,
                }
            )

    if seen_config_sha != set(expected_by_sha):
        raise ValueError("Production coverage does not match the frozen manifest.")
    if len(initial_hashes) != 1:
        raise ValueError(f"Initial parameter states differ across runs: {initial_hashes!r}")
    if any(len(values) != 1 for values in cohort_hashes.values()):
        raise ValueError(f"Dataset/cohort hashes differ across runs: {cohort_hashes!r}")

    rows.sort(key=lambda row: int(row["index"]))
    epoch_rows.sort(key=lambda row: (int(row["index"]), int(row["epoch"])))
    occupancy_rows.sort(key=lambda row: (int(row["index"]), str(row["parameter"])))
    by_surface: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_surface[(row["scheme"], row["optimizer"])].append(row)
    surface_summaries = []
    for scheme in SCHEME_ORDER:
        for optimizer in OPTIMIZER_ORDER:
            surface_rows = sorted(
                by_surface[(scheme, optimizer)], key=lambda row: float(row["weight_max"])
            )
            if [float(row["weight_max"]) for row in surface_rows] != list(WEIGHT_MAX_VALUES):
                raise ValueError(f"Incomplete wmax coverage for {scheme}/{optimizer}.")
            reference = surface_rows[0]
            for row in surface_rows:
                row["final_accuracy_delta_vs_1e4_pp"] = (
                    float(row["final_validation_accuracy_percent"])
                    - float(reference["final_validation_accuracy_percent"])
                )
                row["best_accuracy_delta_vs_1e4_pp"] = (
                    float(row["best_validation_accuracy_percent"])
                    - float(reference["best_validation_accuracy_percent"])
                )
                row["final_loss_delta_vs_1e4"] = (
                    float(row["final_validation_loss"])
                    - float(reference["final_validation_loss"])
                )
            best_final = max(
                surface_rows,
                key=lambda row: (
                    float(row["final_validation_accuracy_percent"]),
                    -float(row["weight_max"]),
                ),
            )
            best_checkpoint = max(
                surface_rows,
                key=lambda row: (
                    float(row["best_validation_accuracy_percent"]),
                    -float(row["weight_max"]),
                ),
            )
            surface_summaries.append(
                {
                    "scheme": scheme,
                    "optimizer": optimizer,
                    "reference_final_accuracy_percent": reference[
                        "final_validation_accuracy_percent"
                    ],
                    "best_final_weight_max": best_final["weight_max"],
                    "best_final_accuracy_percent": best_final[
                        "final_validation_accuracy_percent"
                    ],
                    "best_final_delta_vs_1e4_pp": best_final[
                        "final_accuracy_delta_vs_1e4_pp"
                    ],
                    "best_checkpoint_weight_max": best_checkpoint["weight_max"],
                    "best_checkpoint_accuracy_percent": best_checkpoint[
                        "best_validation_accuracy_percent"
                    ],
                    "best_checkpoint_epoch": best_checkpoint["best_epoch"],
                    "best_checkpoint_delta_vs_1e4_pp": best_checkpoint[
                        "best_accuracy_delta_vs_1e4_pp"
                    ],
                }
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "summary.csv", rows)
    _write_csv(output_dir / "surface_summary.csv", surface_summaries)
    _write_csv(output_dir / "epoch_trajectory.csv", epoch_rows)
    _write_csv(output_dir / "endpoint_occupancy.csv", occupancy_rows)

    report = {
        "schema": "perfectdiode-conv2-fixed-uniform-init-wmax-sweep-analysis/v1",
        "study_id": STUDY_ID,
        "coverage": {"expected": 24, "valid": len(rows)},
        "canonical_lr_handoff": False,
        "paper_facing": False,
        "official_test_read": False,
        "initializer_checkpoint_sha256": INITIALIZER_SHA256,
        "shared_initial_parameter_state_sha256": next(iter(initial_hashes)),
        "shared_dataset_hashes": {
            key: next(iter(values)) for key, values in cohort_hashes.items()
        },
        "rows": rows,
        "surface_summaries": surface_summaries,
        "limitations": [
            "single seed",
            "ordinary MNIST exploratory evidence",
            "fixed raw learning rates selected at the 1e-4 ceiling",
            "Conv2 baseline-Adam source search remains open toward higher Conv rho",
            "cross-scheme comparisons are confounded by different raw LR vectors",
        ],
    }
    (output_dir / "summary.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {1e-4: "#1f77b4", 2e-4: "#ff7f0e", 5e-4: "#2ca02c", 1e-3: "#d62728"}
    fig, axes = plt.subplots(2, 3, figsize=(13, 8), sharex=True)
    for row_index, optimizer in enumerate(OPTIMIZER_ORDER):
        for column, scheme in enumerate(SCHEME_ORDER):
            axis = axes[row_index, column]
            for weight_max in WEIGHT_MAX_VALUES:
                selected = [
                    row
                    for row in epoch_rows
                    if row["scheme"] == scheme
                    and row["optimizer"] == optimizer
                    and float(row["weight_max"]) == weight_max
                ]
                axis.plot(
                    [row["epoch"] for row in selected],
                    [row["validation_accuracy_percent"] for row in selected],
                    marker="o",
                    markersize=2.5,
                    color=colors[weight_max],
                    label=f"{weight_max:.0e}",
                )
            axis.set_title(f"{scheme} / {optimizer}")
            axis.grid(alpha=0.25)
            if row_index == 1:
                axis.set_xlabel("Epoch")
            if column == 0:
                axis.set_ylabel("Validation accuracy (%)")
    axes[0, 2].legend(title="weight max", fontsize=8)
    fig.suptitle("Conv2 fixed-initialization accuracy trajectories")
    fig.tight_layout()
    fig.savefig(output_dir / "accuracy_trajectories.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(13, 8), sharex=True)
    for row_index, optimizer in enumerate(OPTIMIZER_ORDER):
        for column, scheme in enumerate(SCHEME_ORDER):
            axis = axes[row_index, column]
            selected = sorted(
                by_surface[(scheme, optimizer)], key=lambda row: float(row["weight_max"])
            )
            x = [float(row["weight_max"]) for row in selected]
            axis.plot(
                x,
                [row["final_validation_accuracy_percent"] for row in selected],
                marker="o",
                label="epoch 10",
            )
            axis.plot(
                x,
                [row["best_validation_accuracy_percent"] for row in selected],
                marker="s",
                linestyle="--",
                label="best of 10",
            )
            axis.set_xscale("log")
            axis.set_title(f"{scheme} / {optimizer}")
            axis.grid(alpha=0.25)
            if row_index == 1:
                axis.set_xlabel("Upper weight limit")
            if column == 0:
                axis.set_ylabel("Validation accuracy (%)")
    axes[0, 2].legend(fontsize=8)
    fig.suptitle("Conv2 accuracy versus upper weight limit")
    fig.tight_layout()
    fig.savefig(output_dir / "accuracy_vs_weight_max.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(13, 8), sharex=True)
    for row_index, optimizer in enumerate(OPTIMIZER_ORDER):
        for column, scheme in enumerate(SCHEME_ORDER):
            axis = axes[row_index, column]
            selected = sorted(
                by_surface[(scheme, optimizer)], key=lambda row: float(row["weight_max"])
            )
            x = [float(row["weight_max"]) for row in selected]
            axis.plot(
                x,
                [row["final_lower_endpoint_percent"] for row in selected],
                marker="o",
                label="lower",
            )
            axis.plot(
                x,
                [row["final_upper_endpoint_percent"] for row in selected],
                marker="s",
                label="upper",
            )
            axis.set_xscale("log")
            axis.set_yscale("symlog", linthresh=0.01)
            axis.set_title(f"{scheme} / {optimizer}")
            axis.grid(alpha=0.25)
            if row_index == 1:
                axis.set_xlabel("Upper weight limit")
            if column == 0:
                axis.set_ylabel("Final endpoint occupancy (%)")
    axes[0, 2].legend(fontsize=8)
    fig.suptitle("Conv2 final endpoint occupancy")
    fig.tight_layout()
    fig.savefig(output_dir / "endpoint_occupancy_vs_weight_max.png", dpi=180)
    plt.close(fig)

    lines = [
        "# Conv2 fixed-initialization upper-weight sweep",
        "",
        "Exploratory seed-0 ordinary-MNIST evidence. All cases start from one exact `Uniform[1e-5,1e-4)` checkpoint, keep their selected raw LR vectors fixed, and vary only the training projection ceiling within each scheme/optimizer surface. This is not paper-facing evidence or a new LR handoff.",
        "",
        "| Scheme | Optimizer | Weight max | Epoch-10 accuracy | Delta vs 1e-4 | Best accuracy (epoch) | Final lower / upper occupancy |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['scheme']} | {row['optimizer']} | {float(row['weight_max']):.0e} | "
            f"{float(row['final_validation_accuracy_percent']):.2f}% | "
            f"{float(row['final_accuracy_delta_vs_1e4_pp']):+.2f} pp | "
            f"{float(row['best_validation_accuracy_percent']):.2f}% ({int(row['best_epoch'])}) | "
            f"{float(row['final_lower_endpoint_percent']):.2f}% / "
            f"{float(row['final_upper_endpoint_percent']):.2f}% |"
        )
    lines.extend(["", "## Best observed ceiling by surface", ""])
    for surface in surface_summaries:
        lines.append(
            f"- {surface['scheme']} / {surface['optimizer']}: highest epoch-10 accuracy "
            f"{float(surface['best_final_accuracy_percent']):.2f}% at "
            f"`wmax={float(surface['best_final_weight_max']):.0e}` "
            f"({float(surface['best_final_delta_vs_1e4_pp']):+.2f} pp versus `1e-4`)."
        )
    row_by_case = {
        (str(row["scheme"]), str(row["optimizer"]), float(row["weight_max"])): row
        for row in rows
    }

    def _accuracy(scheme: str, optimizer: str, weight_max: float) -> float:
        return float(
            row_by_case[(scheme, optimizer, weight_max)][
                "final_validation_accuracy_percent"
            ]
        )

    upper_at_1e4 = [
        float(row["final_upper_endpoint_percent"])
        for row in rows
        if float(row["weight_max"]) == 1e-4
    ]
    upper_at_1e3 = [
        float(row["final_upper_endpoint_percent"])
        for row in rows
        if float(row["weight_max"]) == 1e-3
    ]
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "Relaxing the ceiling from `1e-4` to `1e-3` improves epoch-10 "
            "accuracy for every surface: baseline by "
            f"{_accuracy('baseline', 'SGD', 1e-3) - _accuracy('baseline', 'SGD', 1e-4):+.2f}/"
            f"{_accuracy('baseline', 'Adam', 1e-3) - _accuracy('baseline', 'Adam', 1e-4):+.2f} pp, "
            "ours by "
            f"{_accuracy('ours', 'SGD', 1e-3) - _accuracy('ours', 'SGD', 1e-4):+.2f}/"
            f"{_accuracy('ours', 'Adam', 1e-3) - _accuracy('ours', 'Adam', 1e-4):+.2f} pp, "
            "and legacy by "
            f"{_accuracy('legacy', 'SGD', 1e-3) - _accuracy('legacy', 'SGD', 1e-4):+.2f}/"
            f"{_accuracy('legacy', 'Adam', 1e-3) - _accuracy('legacy', 'Adam', 1e-4):+.2f} pp "
            "for SGD/Adam, respectively.",
            "",
            "The epoch-10 legacy-minus-ours gap contracts from "
            f"{_accuracy('legacy', 'SGD', 1e-4) - _accuracy('ours', 'SGD', 1e-4):.2f} to "
            f"{_accuracy('legacy', 'SGD', 1e-3) - _accuracy('ours', 'SGD', 1e-3):.2f} pp with SGD, "
            "and from "
            f"{_accuracy('legacy', 'Adam', 1e-4) - _accuracy('ours', 'Adam', 1e-4):.2f} to "
            f"{_accuracy('legacy', 'Adam', 1e-3) - _accuracy('ours', 'Adam', 1e-3):.2f} pp with Adam. "
            "This shows that the tight `1e-4` projection ceiling explains a material part of legacy's earlier advantage, although it does not explain all of it.",
            "",
            f"Across the six surfaces, final upper-endpoint occupancy falls from {min(upper_at_1e4):.2f}%--{max(upper_at_1e4):.2f}% at `wmax=1e-4` to {min(upper_at_1e3):.2f}%--{max(upper_at_1e3):.2f}% at `wmax=1e-3`. Together with the accuracy gains, this supports projection throttling as the mechanism. Baseline-SGD and legacy-SGD are already flat between `5e-4` and `1e-3`; ours and both non-legacy Adam surfaces are still improving modestly at `1e-3`.",
        ]
    )
    lines.extend(
        [
            "",
            "## Validation and limitations",
            "",
            "All 24 canonical bundles validate, complete exactly ten epochs, share the same initial parameter state and MNIST split/minibatch-order hashes, preserve exact-zero biases, remain within their configured `[1e-5,wmax]` intervals, and record no official-test read.",
            "",
            "The within-surface deltas isolate relaxing the upper projection limit conditional on the fixed initialization and transferred raw LR vector. Cross-scheme accuracy differences are not a weight-limit causal comparison because each scheme/optimizer uses a different LR vector. Conv2 baseline-Adam also retains its source search's open upper-Conv-rho caveat.",
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
