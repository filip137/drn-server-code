#!/usr/bin/env python3
"""Analyze the complete Conv1/2/3 EqProp beta-collapse study."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from experiments.reporting import atomic_write_json, validate_run
from experiments.run_conv123_centered_eqprop_beta_collapse import (
    ARCHITECTURES,
    BETA_TIERS,
    CASE_SPECS,
    OPTIMIZERS,
    SCHEMES,
    STUDY_ID,
    load_and_validate_study,
)
from experiments.validate_conv123_eqprop_beta_collapse_pack import (
    _epoch_rows,
    _scientific_outcome,
    _sha256,
)


DEFAULT_STUDY_ROOT = Path("results") / STUDY_ID


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def classify_pair(
    boundary: Mapping[str, Any], control: Mapping[str, Any]
) -> str:
    """Classify a matched boundary/control pair using the frozen 5 pp rule."""
    if bool(boundary["collapsed"]):
        return (
            "inconclusive_shared_instability"
            if bool(control["collapsed"])
            else "confirmed_beta_collapse"
        )
    if bool(control["collapsed"]):
        return "inconclusive_control_only_instability"
    boundary_final = boundary.get("final_validation_accuracy")
    control_final = control.get("final_validation_accuracy")
    if boundary_final is not None and control_final is not None:
        relative_gap_pp = 100.0 * (float(control_final) - float(boundary_final))
        if relative_gap_pp >= 5.0:
            return "degradation_without_internal_collapse"
    return "not_confirmed"


def _discover_runs(production_root: Path) -> dict[int, Path]:
    rows: dict[int, Path] = {}
    for manifest_path in sorted(production_root.glob("*/manifest.json")):
        run_dir = manifest_path.parent
        exact = _read_json(run_dir / "exact_run.json")
        index = int(exact.get("index", -1))
        if index in rows:
            raise ValueError(f"Duplicate logical index {index}: {rows[index]} and {run_dir}.")
        rows[index] = run_dir
    expected = set(range(len(CASE_SPECS)))
    if set(rows) != expected:
        raise ValueError(
            f"Production coverage mismatch: expected {sorted(expected)}, got {sorted(rows)}."
        )
    return rows


def _case_rows(study_root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    production_root = study_root / "production"
    config_root = study_root / "resolved_configs"
    summary_root = study_root / "task_summaries/production/runs"
    log_root = study_root / "packed_logs/production"
    receipt_root = study_root / "transport_receipts/production/runs"
    pack_receipt_root = study_root / "transport_receipts/production/packs"
    configs = sorted(config_root.glob("*.json"))
    if len(configs) != len(CASE_SPECS):
        raise ValueError(f"Expected 36 collected resolved configs; found {len(configs)}.")
    pack_receipts = sorted(pack_receipt_root.glob("*.json"))
    if len(pack_receipts) != 18:
        raise ValueError(f"Expected 18 pack receipts; found {len(pack_receipts)}.")
    run_dirs = _discover_runs(production_root)

    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    initialization_by_architecture: dict[str, set[str]] = {
        architecture: set() for architecture in ARCHITECTURES
    }
    provenance_values: set[tuple[str, str, tuple[str, ...]]] = set()
    for spec in CASE_SPECS:
        run_dir = run_dirs[spec.index]
        errors.extend(f"{run_dir.name}: {error}" for error in validate_run(run_dir))
        manifest = _read_json(run_dir / "manifest.json")
        status = _read_json(run_dir / "status.json")
        exact = _read_json(run_dir / "exact_run.json")
        run_config = _read_json(run_dir / "config.json")
        summary = _read_json(summary_root / f"{spec.index}.json")
        receipt_path = receipt_root / f"{spec.index}.json"
        receipt = _read_json(receipt_path)
        log_path = log_root / f"{spec.index}.log"
        config_path = configs[spec.index]
        if not isinstance(summary, list) or len(summary) != 1:
            errors.append(f"logical {spec.index}: invalid exact-run summary")
            continue
        epoch_rows = _epoch_rows(run_dir / "metrics.jsonl")
        outcome = _scientific_outcome(
            state=str(status.get("state")),
            summary_status=str(summary[0].get("status")),
            exact_returncode=exact.get("returncode"),
            log_path=log_path,
            epoch_rows=epoch_rows,
            expected_epochs=10,
        )
        if receipt.get("outcome") != outcome:
            errors.append(f"logical {spec.index}: receipt outcome changed")
        expected_hashes = {
            "selected_config_sha256": _sha256(config_path),
            "manifest_sha256": _sha256(run_dir / "manifest.json"),
            "status_sha256": _sha256(run_dir / "status.json"),
            "metrics_jsonl_sha256": _sha256(run_dir / "metrics.jsonl"),
            "result_sha256": (
                _sha256(run_dir / "result.json")
                if (run_dir / "result.json").is_file()
                else None
            ),
        }
        for key, value in expected_hashes.items():
            if receipt.get(key) != value:
                errors.append(f"logical {spec.index}: receipt {key} mismatch")
        if manifest.get("arm_id") != spec.arm_id:
            errors.append(f"logical {spec.index}: arm mismatch")
        if manifest.get("dataset", {}).get("official_test_read") is not False:
            errors.append(f"logical {spec.index}: official-test guard failed")

        architecture = run_config.get("architecture", {})
        provenance = run_config.get("dataset", {}).get("provenance", {})
        initial_hash = str(architecture.get("initial_parameter_state_sha256", ""))
        initialization_by_architecture[spec.architecture].add(initial_hash)
        provenance_values.add(
            (
                str(provenance.get("train_indices_sha256", "")),
                str(provenance.get("validation_indices_sha256", "")),
                tuple(
                    str(value)
                    for value in provenance.get("train_batch_order_sha256", [])
                ),
            )
        )
        config = _read_json(config_path)
        row = {
            "index": spec.index,
            "architecture": spec.architecture,
            "scheme": spec.scheme,
            "optimizer": spec.optimizer,
            "beta_tier": spec.beta_tier,
            "injected_beta_B": float(config["eqprop"]["injected_beta_B"]),
            "base_beta": float(config["beta"]),
            "terminal_kind": outcome["terminal_kind"],
            "collapsed": bool(outcome["collapsed"]),
            "completed_epochs": int(outcome["completed_epochs"]),
            "best_validation_accuracy": outcome["best_validation_accuracy"],
            "final_validation_accuracy": outcome["final_validation_accuracy"],
            "final_drop_from_best_pp": outcome["final_drop_from_best_pp"],
            "failure_epoch": (
                outcome["failure"]["epoch"] if outcome["failure"] else None
            ),
            "failure_batch": (
                outcome["failure"]["batch"] if outcome["failure"] else None
            ),
            "initial_parameter_state_sha256": initial_hash,
            "run_dir": str(run_dir),
            "log_path": str(log_path),
            "receipt_path": str(receipt_path),
            "epochs": epoch_rows,
        }
        rows.append(row)

    for architecture, values in initialization_by_architecture.items():
        if len(values) != 1:
            errors.append(f"{architecture}: initial states differ across matched cases")
    if len(provenance_values) != 1:
        errors.append("dataset split or minibatch order differs across cases")
    if errors:
        raise ValueError("Study validation failed:\n- " + "\n- ".join(errors))

    verification = {
        "schema_version": "conv123-eqprop-beta-collapse-verification/v1",
        "study_id": STUDY_ID,
        "status": "pass",
        "case_count": len(rows),
        "pack_count": len(pack_receipts),
        "complete_run_count": sum(
            row["terminal_kind"] == "finite_complete" for row in rows
        ),
        "scientific_nonfinite_count": sum(
            row["terminal_kind"] == "nonfinite_training" for row in rows
        ),
        "collapsed_case_count": sum(row["collapsed"] for row in rows),
        "initial_parameter_state_sha256_by_architecture": {
            architecture: next(iter(values))
            for architecture, values in initialization_by_architecture.items()
        },
        "runtime_dtype": "float64",
        "T": 8,
        "K": 8,
        "official_test_read": False,
    }
    return rows, verification


def _pair_rows(cases: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    by_identity = {
        (
            row["architecture"],
            row["scheme"],
            row["optimizer"],
            row["beta_tier"],
        ): row
        for row in cases
    }
    rows: list[dict[str, Any]] = []
    for architecture in ARCHITECTURES:
        for scheme in SCHEMES:
            for optimizer in OPTIMIZERS:
                boundary = by_identity[(architecture, scheme, optimizer, "boundary")]
                control = by_identity[
                    (architecture, scheme, optimizer, "one_decade_lower")
                ]
                boundary_final = boundary.get("final_validation_accuracy")
                control_final = control.get("final_validation_accuracy")
                final_gap_pp = (
                    100.0 * (float(boundary_final) - float(control_final))
                    if boundary_final is not None and control_final is not None
                    else None
                )
                rows.append(
                    {
                        "architecture": architecture,
                        "scheme": scheme,
                        "optimizer": optimizer,
                        "classification": classify_pair(boundary, control),
                        "boundary_injected_beta_B": boundary["injected_beta_B"],
                        "control_injected_beta_B": control["injected_beta_B"],
                        "boundary_terminal_kind": boundary["terminal_kind"],
                        "control_terminal_kind": control["terminal_kind"],
                        "boundary_collapsed": boundary["collapsed"],
                        "control_collapsed": control["collapsed"],
                        "boundary_best_validation_accuracy": boundary[
                            "best_validation_accuracy"
                        ],
                        "boundary_final_validation_accuracy": boundary[
                            "final_validation_accuracy"
                        ],
                        "control_best_validation_accuracy": control[
                            "best_validation_accuracy"
                        ],
                        "control_final_validation_accuracy": control[
                            "final_validation_accuracy"
                        ],
                        "boundary_minus_control_final_pp": final_gap_pp,
                    }
                )
    return rows


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _plot(cases: Sequence[Mapping[str, Any]], path: Path) -> None:
    by_identity = {
        (
            row["architecture"],
            row["scheme"],
            row["optimizer"],
            row["beta_tier"],
        ): row
        for row in cases
    }
    fig, axes = plt.subplots(3, 6, figsize=(20, 10.5), sharex=True, sharey=True)
    for row_index, architecture in enumerate(ARCHITECTURES):
        for scheme_index, scheme in enumerate(SCHEMES):
            for optimizer_index, optimizer in enumerate(OPTIMIZERS):
                column = 2 * scheme_index + optimizer_index
                axis = axes[row_index, column]
                for tier, color, marker, label in (
                    ("boundary", "#d62728", "o", "boundary"),
                    ("one_decade_lower", "#1f77b4", "^", "1 decade lower"),
                ):
                    case = by_identity[(architecture, scheme, optimizer, tier)]
                    epochs = case["epochs"]
                    axis.plot(
                        [row["epoch"] for row in epochs],
                        [100.0 * row["validation_accuracy"] for row in epochs],
                        color=color,
                        marker=marker,
                        linewidth=1.5,
                        markersize=3.5,
                        label=label,
                    )
                    if case["terminal_kind"] == "nonfinite_training" and epochs:
                        axis.scatter(
                            epochs[-1]["epoch"],
                            100.0 * epochs[-1]["validation_accuracy"],
                            color=color,
                            marker="x",
                            s=50,
                            zorder=4,
                        )
                axis.set_title(f"{architecture.upper()} · {scheme} · {optimizer}")
                axis.grid(True, alpha=0.25)
                if row_index == 2:
                    axis.set_xlabel("Epoch")
                if column == 0:
                    axis.set_ylabel("Validation accuracy (%)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.972),
        ncol=2,
        frameon=False,
    )
    fig.suptitle("Centered-float64 EqProp: boundary vs one-decade-lower beta", y=0.998)
    fig.tight_layout(rect=(0, 0, 1, 0.935))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def analyze(study_root: Path, output_dir: Path) -> dict[str, Any]:
    study_root = study_root.expanduser().resolve()
    load_and_validate_study(
        study_root
        / "source/configs/conv/perfectdiode_conv123_centered_float64_eqprop_"
        "beta_collapse_boundary_vs_1decade_sgd_adam_10ep_seed0_20260815_v1.json"
        if (study_root / "source").is_dir()
        else Path(
            "configs/conv/perfectdiode_conv123_centered_float64_eqprop_"
            "beta_collapse_boundary_vs_1decade_sgd_adam_10ep_seed0_20260815_v1.json"
        )
    )
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    cases, verification = _case_rows(study_root)
    pairs = _pair_rows(cases)
    counts = Counter(row["classification"] for row in pairs)
    summary = {
        "schema_version": "conv123-eqprop-beta-collapse-analysis/v1",
        "study_id": STUDY_ID,
        "verification": verification,
        "collapse_definition": {
            "nonfinite_training": True,
            "final_validation_drop_from_best_pp_at_least": 5.0,
        },
        "pair_classification_counts": dict(sorted(counts.items())),
        "cases": [{key: value for key, value in row.items() if key != "epochs"} for row in cases],
        "pairs": pairs,
        "limitations": [
            "Exploratory seed-0 ordinary-MNIST evidence; it is not paper-facing.",
            "Conv3 baseline at T=K=8 retains the predeclared free-state residual caveat.",
            "A one-decade-lower collapse is reported as inconclusive and does not trigger an automatic two-decade extension.",
        ],
    }
    atomic_write_json(output_dir / "summary.json", summary)
    _write_csv(
        output_dir / "cases.csv",
        cases,
        (
            "index",
            "architecture",
            "scheme",
            "optimizer",
            "beta_tier",
            "injected_beta_B",
            "base_beta",
            "terminal_kind",
            "collapsed",
            "completed_epochs",
            "best_validation_accuracy",
            "final_validation_accuracy",
            "final_drop_from_best_pp",
            "failure_epoch",
            "failure_batch",
            "initial_parameter_state_sha256",
            "run_dir",
            "log_path",
            "receipt_path",
        ),
    )
    _write_csv(
        output_dir / "pairs.csv",
        pairs,
        (
            "architecture",
            "scheme",
            "optimizer",
            "classification",
            "boundary_injected_beta_B",
            "control_injected_beta_B",
            "boundary_terminal_kind",
            "control_terminal_kind",
            "boundary_collapsed",
            "control_collapsed",
            "boundary_best_validation_accuracy",
            "boundary_final_validation_accuracy",
            "control_best_validation_accuracy",
            "control_final_validation_accuracy",
            "boundary_minus_control_final_pp",
        ),
    )
    _plot(cases, output_dir / "validation_accuracy_trajectories.png")

    lines = [
        "# Conv1/Conv2/Conv3 EqProp beta-collapse study",
        "",
        "Exploratory seed-0 ordinary-MNIST evidence; no official-test examples were read.",
        "Collapse means an explicit non-finite training endpoint or a final validation accuracy at least 5.00 percentage points below the run's best epoch.",
        "",
        "## Coverage",
        "",
        f"All {verification['case_count']} cases and {verification['pack_count']} two-run GPU packs validated locally.",
        f"Finite completions: {verification['complete_run_count']}; explicit scientific non-finite endpoints: {verification['scientific_nonfinite_count']}; collapsed cases: {verification['collapsed_case_count']}.",
        "",
        "## Pair conclusions",
        "",
        "| Architecture | Scheme | Optimizer | Classification | Boundary collapsed | Control collapsed | Boundary-control final (pp) |",
        "|---|---|---|---|---:|---:|---:|",
    ]
    for row in pairs:
        gap = row["boundary_minus_control_final_pp"]
        gap_text = "n/a" if gap is None else f"{float(gap):+.2f}"
        lines.append(
            f"| {row['architecture']} | {row['scheme']} | {row['optimizer']} | "
            f"{row['classification']} | {row['boundary_collapsed']} | "
            f"{row['control_collapsed']} | {gap_text} |"
        )
    lines.extend(
        [
            "",
            "## Limitations",
            "",
            "- This is exploratory seed-0 selection evidence, not a paper result.",
            "- Conv3 baseline retains the predeclared `T=K=8` residual caveat.",
            "- A collapsing one-decade control is left inconclusive; no two-decade run is added automatically.",
            "",
            "See `cases.csv`, `pairs.csv`, `summary.json`, and `validation_accuracy_trajectories.png` for the exact evidence.",
        ]
    )
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-root", type=Path, default=DEFAULT_STUDY_ROOT)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir = args.output_dir or args.study_root / "analysis"
    summary = analyze(args.study_root, output_dir)
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
