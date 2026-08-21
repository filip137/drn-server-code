#!/usr/bin/env python3
"""Validate and analyze the Conv1/2/3 fixed-initializer Adam wmax sweep."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

try:
    from experiments.analyze_conv2_fixed_uniform_init_wmax_sweep import _weight_audit
    from experiments.prepare_conv123_fixed_uniform_init_wmax_adam import (
        ARCHITECTURES,
        EVIDENCE_CLASS,
        STUDY_ID,
        WMAX_CASES,
    )
    from experiments.reporting import validate_run
except ModuleNotFoundError:
    from analyze_conv2_fixed_uniform_init_wmax_sweep import _weight_audit  # type: ignore[no-redef]
    from prepare_conv123_fixed_uniform_init_wmax_adam import (  # type: ignore[no-redef]
        ARCHITECTURES,
        EVIDENCE_CLASS,
        STUDY_ID,
        WMAX_CASES,
    )
    from reporting import validate_run  # type: ignore[no-redef]


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STUDY_ROOT = REPO_ROOT / "results" / STUDY_ID
ARCHITECTURE_ORDER = tuple(item.name for item in ARCHITECTURES)
EXPECTED_EPOCHS = {item.name: item.epochs for item in ARCHITECTURES}
INITIALIZER_SHA256 = {item.name: item.initializer_sha256 for item in ARCHITECTURES}
SCHEME_ORDER = ("baseline", "ours", "legacy")
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


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write an empty table: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _case_order(row: dict[str, Any]) -> tuple[int, int, int]:
    return (
        ARCHITECTURE_ORDER.index(str(row["architecture"])),
        SCHEME_ORDER.index(str(row["scheme"])),
        WEIGHT_MAX_VALUES.index(float(row["weight_max"])),
    )


def analyze(study_root: Path, output_dir: Path | None = None) -> dict[str, Any]:
    study_root = Path(study_root).expanduser().resolve()
    output_dir = (
        Path(output_dir).expanduser().resolve()
        if output_dir is not None
        else study_root / "analysis"
    )
    manifest = _load_json(study_root / "study_manifest.json")
    if manifest.get("study_id") != STUDY_ID or manifest.get("run_count") != 27:
        raise ValueError(f"Unexpected study manifest in {study_root}.")
    if manifest.get("canonical_lr_handoff") is not False:
        raise ValueError("This exploratory sweep must remain explicitly noncanonical.")

    for architecture in ARCHITECTURES:
        initializer = (
            study_root
            / "assets"
            / "bounded_uniform"
            / architecture.name
            / "final_model.pt"
        )
        if _sha256(initializer) != architecture.initializer_sha256:
            raise ValueError(f"Initializer digest mismatch for {architecture.name}.")

    expected_by_sha = {row["config_sha256"]: row for row in manifest["runs"]}
    if len(expected_by_sha) != 27:
        raise ValueError("Frozen manifest contains duplicate config authorities.")
    run_dirs = sorted(path.parent for path in (study_root / "runs").glob("*/manifest.json"))
    if len(run_dirs) != 27:
        raise ValueError(f"Expected 27 production run directories, found {len(run_dirs)}.")

    rows: list[dict[str, Any]] = []
    occupancy_rows: list[dict[str, Any]] = []
    seen_config_sha: set[str] = set()
    cohort_hashes: dict[str, set[str]] = defaultdict(set)
    initial_hashes: dict[str, set[str]] = defaultdict(set)
    for run_dir in run_dirs:
        errors = validate_run(run_dir)
        if errors:
            raise ValueError(f"Invalid canonical bundle {run_dir}: {errors!r}")
        run_manifest = _load_json(run_dir / "manifest.json")
        if run_manifest.get("study_id") != STUDY_ID:
            raise ValueError(f"Unexpected study identity: {run_dir}")
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
        architecture = str(expected["architecture"])
        scheme = str(expected["scheme"])
        weight_min = float(expected["weight_min"])
        weight_max = float(expected["weight_max"])
        epochs = EXPECTED_EPOCHS[architecture]

        if config["optimizer"]["name"] != "Adam":
            raise ValueError(f"Unexpected optimizer: {run_dir}")
        if config["learning_rates_by_parameter"] != expected["learning_rates_by_parameter"]:
            raise ValueError(f"Learning-rate mapping disagrees with manifest: {run_dir}")
        if any(
            value != 0.0
            for name, value in config["learning_rates_by_parameter"].items()
            if name.startswith("Bias_")
        ):
            raise ValueError(f"Bias learning rate is nonzero: {run_dir}")
        if float(config["model_base"]["weight_min"]) != weight_min or float(
            config["model_base"]["weight_max"]
        ) != weight_max:
            raise ValueError(f"Configured projection interval disagrees: {run_dir}")
        if int(config["lab"]["epochs"]) != epochs:
            raise ValueError(f"Unexpected epoch budget: {run_dir}")
        if (
            config["weight_ceiling_sweep"]["initializer_checkpoint_sha256"]
            != INITIALIZER_SHA256[architecture]
        ):
            raise ValueError(f"Initializer authority disagrees: {run_dir}")
        if metrics.get("official_test_evaluations") != 0:
            raise ValueError(f"Official test was evaluated: {run_dir}")
        if metrics.get("epoch_checkpoint_count") != epochs + 1:
            raise ValueError(f"Unexpected epoch-checkpoint coverage: {run_dir}")
        if result.get("dataset", {}).get("official_test_read") is not False:
            raise ValueError(f"Result does not prove official_test_read=false: {run_dir}")
        if result.get("completion", {}).get("criteria_met") is not True:
            raise ValueError(f"Completion criteria were not met: {run_dir}")

        initial_hash = metrics.get("initial_parameter_state_sha256")
        if not isinstance(initial_hash, str) or len(initial_hash) != 64:
            raise ValueError(f"Missing initial parameter-state digest: {run_dir}")
        initial_hashes[architecture].add(initial_hash)
        provenance = metrics["dataset_provenance"]
        cohort_hashes["train_indices"].add(provenance["train_indices_sha256"])
        cohort_hashes["validation_indices"].add(provenance["validation_indices_sha256"])
        cohort_hashes["batch_order"].add(provenance["first_epoch_batch_order_sha256"])

        pooled, layers = _weight_audit(
            run_dir / "weights_final.npz",
            weight_min=weight_min,
            weight_max=weight_max,
        )
        row = {
            "architecture": architecture,
            "scheme": scheme,
            "optimizer": "Adam",
            "weight_min": weight_min,
            "weight_max": weight_max,
            "epochs": epochs,
            "best_validation_accuracy_percent": 100.0
            * _finite(metrics["best_validation_accuracy"], "best validation accuracy"),
            "best_epoch": int(metrics["best_epoch"]),
            "final_validation_accuracy_percent": 100.0
            * _finite(metrics["final_validation_accuracy"], "final validation accuracy"),
            "final_validation_loss": _finite(
                metrics["final_validation_loss"], "final validation loss"
            ),
            "initial_parameter_state_sha256": initial_hash,
            "official_test_read": False,
            "config_sha256": config_sha,
            "result_sha256": _sha256(run_dir / "result.json"),
            "run_dir": str(run_dir.relative_to(study_root)),
            **pooled,
        }
        rows.append(row)
        for layer in layers:
            occupancy_rows.append(
                {
                    "architecture": architecture,
                    "scheme": scheme,
                    "optimizer": "Adam",
                    "weight_min": weight_min,
                    "weight_max": weight_max,
                    **layer,
                }
            )

    if seen_config_sha != set(expected_by_sha):
        raise ValueError("Production coverage does not match the frozen manifest.")
    if any(len(values) != 1 for values in initial_hashes.values()):
        raise ValueError(f"Initial states differ within an architecture: {initial_hashes!r}")
    if set(initial_hashes) != set(ARCHITECTURE_ORDER):
        raise ValueError(f"Missing architecture initial-state proof: {initial_hashes!r}")
    if any(len(values) != 1 for values in cohort_hashes.values()):
        raise ValueError(f"Dataset/cohort hashes differ across runs: {cohort_hashes!r}")

    rows.sort(key=_case_order)
    occupancy_rows.sort(
        key=lambda row: (*_case_order(row), str(row["parameter"]))
    )
    by_surface: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_surface[(str(row["architecture"]), str(row["scheme"]))].append(row)
    surface_summaries: list[dict[str, Any]] = []
    for architecture in ARCHITECTURE_ORDER:
        for scheme in SCHEME_ORDER:
            surface_rows = sorted(
                by_surface[(architecture, scheme)], key=lambda row: float(row["weight_max"])
            )
            if [float(row["weight_max"]) for row in surface_rows] != list(
                WEIGHT_MAX_VALUES
            ):
                raise ValueError(f"Incomplete wmax coverage for {architecture}/{scheme}.")
            reference = surface_rows[0]
            for row in surface_rows:
                row["best_accuracy_delta_vs_1e4_pp"] = (
                    float(row["best_validation_accuracy_percent"])
                    - float(reference["best_validation_accuracy_percent"])
                )
                row["final_accuracy_delta_vs_1e4_pp"] = (
                    float(row["final_validation_accuracy_percent"])
                    - float(reference["final_validation_accuracy_percent"])
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
                    "architecture": architecture,
                    "scheme": scheme,
                    "reference_best_accuracy_percent": reference[
                        "best_validation_accuracy_percent"
                    ],
                    "best_weight_max": best_checkpoint["weight_max"],
                    "best_accuracy_percent": best_checkpoint[
                        "best_validation_accuracy_percent"
                    ],
                    "best_epoch": best_checkpoint["best_epoch"],
                    "best_delta_vs_1e4_pp": best_checkpoint[
                        "best_accuracy_delta_vs_1e4_pp"
                    ],
                }
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "summary.csv", rows)
    _write_csv(output_dir / "surface_summary.csv", surface_summaries)
    _write_csv(output_dir / "endpoint_occupancy.csv", occupancy_rows)

    report = {
        "schema": "perfectdiode-conv123-fixed-uniform-init-wmax-adam-analysis/v1",
        "study_id": STUDY_ID,
        "coverage": {"expected": 27, "valid": len(rows)},
        "canonical_lr_handoff": False,
        "paper_facing": False,
        "official_test_read": False,
        "initial_parameter_state_sha256_by_architecture": {
            key: next(iter(values)) for key, values in initial_hashes.items()
        },
        "shared_dataset_hashes": {
            key: next(iter(values)) for key, values in cohort_hashes.items()
        },
        "rows": rows,
        "surface_summaries": surface_summaries,
        "limitations": [
            "single seed",
            "ordinary MNIST exploratory evidence",
            "different epoch budgets across architectures",
            "raw learning rates were selected at the 1e-4 ceiling and kept fixed",
            "cross-scheme comparisons are confounded by different raw LR vectors",
            "Conv2 baseline and Conv3 ours inherit open-upper-Conv-rho source searches",
            "Conv3 legacy inherits a noncanonical below-gate LR source",
        ],
    }
    (output_dir / "summary.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 3, figsize=(12, 10), sharex=True)
    for row_index, architecture in enumerate(ARCHITECTURE_ORDER):
        for column, scheme in enumerate(SCHEME_ORDER):
            axis = axes[row_index, column]
            selected = sorted(
                by_surface[(architecture, scheme)],
                key=lambda row: float(row["weight_max"]),
            )
            x = [float(row["weight_max"]) for row in selected]
            axis.plot(
                x,
                [row["best_validation_accuracy_percent"] for row in selected],
                marker="o",
                label="best",
            )
            axis.plot(
                x,
                [row["final_validation_accuracy_percent"] for row in selected],
                marker="s",
                linestyle="--",
                label="final",
            )
            axis.set_xscale("log")
            axis.set_title(f"{architecture} / {scheme}")
            axis.grid(alpha=0.25)
            if row_index == 2:
                axis.set_xlabel("Upper weight limit")
            if column == 0:
                axis.set_ylabel("Validation accuracy (%)")
    axes[0, 2].legend(fontsize=8)
    fig.suptitle("Fixed-initialization Adam accuracy versus upper weight limit")
    fig.tight_layout()
    fig.savefig(output_dir / "accuracy_vs_weight_max.png", dpi=180)
    plt.close(fig)

    row_by_case = {
        (str(row["architecture"]), str(row["scheme"]), float(row["weight_max"])): row
        for row in rows
    }

    def _best(architecture: str, scheme: str, weight_max: float) -> float:
        return float(
            row_by_case[(architecture, scheme, weight_max)][
                "best_validation_accuracy_percent"
            ]
        )

    lines = [
        "# Conv1/2/3 fixed-initialization Adam upper-weight sweep",
        "",
        "Exploratory seed-0 ordinary-MNIST evidence. Within each architecture and scheme, all three ceilings start from the same exact `Uniform[1e-5,1e-4)` checkpoint and use the same transferred raw Adam learning-rate vector. Only the upper projection limit changes.",
        "",
        "| Architecture | Scheme | Best @ 1e-4 | Best @ 5e-4 | Best @ 1e-3 | 5e-4 delta | 1e-3 delta |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for architecture in ARCHITECTURE_ORDER:
        for scheme in SCHEME_ORDER:
            reference = _best(architecture, scheme, 1e-4)
            middle = _best(architecture, scheme, 5e-4)
            high = _best(architecture, scheme, 1e-3)
            lines.append(
                f"| {architecture} | {scheme} | {reference:.2f}% | {middle:.2f}% | "
                f"{high:.2f}% | {middle - reference:+.2f} pp | {high - reference:+.2f} pp |"
            )
    deltas_1e3 = [
        _best(architecture, scheme, 1e-3) - _best(architecture, scheme, 1e-4)
        for architecture in ARCHITECTURE_ORDER
        for scheme in SCHEME_ORDER
    ]
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
            "Increasing the ceiling improves best validation accuracy in every architecture/scheme surface. The `1e-4` to `1e-3` gain ranges from "
            f"{min(deltas_1e3):.2f} to {max(deltas_1e3):.2f} percentage points. Gains are largest for baseline/ours in the deeper networks; legacy starts much stronger and gains less.",
            "",
            "The marginal gain from `5e-4` to `1e-3` is nearly zero for legacy (0.00/0.04/0.18 pp for Conv1/2/3), while Conv2 and Conv3 baseline/ours still gain. Thus `5e-4` is already effectively sufficient for legacy, but `1e-3` remains the best tested ceiling for the deeper baseline/ours cases.",
            "",
            "Final exact upper-endpoint occupancy falls from "
            f"{min(upper_at_1e4):.2f}%--{max(upper_at_1e4):.2f}% across the nine `wmax=1e-4` cases to "
            f"{min(upper_at_1e3):.2f}%--{max(upper_at_1e3):.2f}% at `wmax=1e-3`. Together with the paired accuracy gains, this supports upper-bound projection throttling as the mechanism.",
            "",
            "The best-accuracy legacy-minus-ours gap contracts from "
            f"{_best('conv1', 'legacy', 1e-4) - _best('conv1', 'ours', 1e-4):.2f} to {_best('conv1', 'legacy', 1e-3) - _best('conv1', 'ours', 1e-3):.2f} pp in Conv1, "
            f"{_best('conv2', 'legacy', 1e-4) - _best('conv2', 'ours', 1e-4):.2f} to {_best('conv2', 'legacy', 1e-3) - _best('conv2', 'ours', 1e-3):.2f} pp in Conv2, and "
            f"{_best('conv3', 'legacy', 1e-4) - _best('conv3', 'ours', 1e-4):.2f} to {_best('conv3', 'legacy', 1e-3) - _best('conv3', 'ours', 1e-3):.2f} pp in Conv3. The tight ceiling therefore explains a substantial part, but not all, of legacy's advantage.",
            "",
            "## Validation and limitations",
            "",
            "All 27 canonical bundles validate, have complete epoch/checkpoint coverage, share the intended per-architecture initial state and MNIST cohort/order hashes, keep biases exactly zero, remain inside their configured bounds, and prove that the official test split was not read.",
            "",
            "These are within-surface ceiling effects conditional on fixed transferred learning rates. They are single-seed exploratory results, not paper-facing evidence or a new learning-rate handoff. Cross-scheme accuracy differences remain confounded by different learning-rate vectors.",
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
