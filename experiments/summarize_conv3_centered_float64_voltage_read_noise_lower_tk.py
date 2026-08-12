#!/usr/bin/env python3
"""Summarize the matched Conv3 endpoint-voltage read-noise T/K replay."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STUDY_ROOT = (
    ROOT
    / "results/perfectdiode-conv3-centered-float64-voltage-read-noise-tk8-tk12-20260812-v1"
)
DEFAULT_T64_RUN = (
    ROOT
    / "results/perfectdiode-conv3-centered-float64-voltage-read-noise-20260811-v1/production"
)
SCHEMES = ("baseline", "ours", "legacy")
ROLES = ("reconstructed_initialization", "best_validation")
LAYERS = ("C0", "C1", "C2", "Dense")
COLORS = {"baseline": "#4C78A8", "ours": "#F58518", "legacy": "#E45756"}
ROLE_STYLES = {"reconstructed_initialization": "--", "best_validation": "-"}


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return [dict(row) for row in csv.DictReader(stream)]


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Expected rows for {path}.")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected an object in {path}.")
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"true", "1"}:
        return True
    if normalized in {"false", "0", ""}:
        return False
    raise ValueError(f"Expected boolean, got {value!r}.")


def _float(value: Any) -> float | None:
    if value is None or str(value).strip() == "":
        return None
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"Expected finite float, got {value!r}.")
    return result


def _validate_bundle(run_dir: Path) -> dict[str, Any]:
    status = _read_json(run_dir / "status.json")
    result = _read_json(run_dir / "result.json")
    manifest = _read_json(run_dir / "manifest.json")
    if status.get("state") != "complete":
        raise ValueError(f"Run is not complete: {run_dir}.")
    if result.get("completion", {}).get("criteria_met") is not True:
        raise ValueError(f"Completion criteria failed: {run_dir}.")
    if result.get("manifest_sha256") != _sha256(run_dir / "manifest.json"):
        raise ValueError(f"Manifest hash mismatch: {run_dir}.")
    if result.get("metrics_sha256") != _sha256(run_dir / "metrics.jsonl"):
        raise ValueError(f"Metrics hash mismatch: {run_dir}.")
    return {
        "path": str(run_dir),
        "result_sha256": _sha256(run_dir / "result.json"),
        "manifest_sha256": _sha256(run_dir / "manifest.json"),
        "run_id": manifest["run_id"],
    }


def _case_guards(run_dir: Path) -> dict[tuple[str, str], dict[str, Any]]:
    payload = _read_json(run_dir / "read_only_guards.json")
    return {
        (str(row["scheme"]), str(row["checkpoint_role"])): dict(row)
        for row in payload["case_guards"]
    }


def _normalize_thresholds(run_dir: Path, tk: int) -> list[dict[str, Any]]:
    guards = _case_guards(run_dir)
    rows: list[dict[str, Any]] = []
    for source in _read_csv(run_dir / "noise_thresholds.csv"):
        key = (source["scheme"], source["checkpoint_role"])
        guard = guards[key]
        residual_valid = _bool(guard["all_residual_gates_passed"])
        clean_valid = _bool(
            source.get(
                "clean_eqprop_vs_same_tk_bptt_gate_passed",
                guard.get(
                    "clean_all_layer_eqprop_vs_same_tk_bptt_gate_passed",
                    guard.get("clean_all_layer_eqprop_vs_bptt_gate_passed", True),
                ),
            )
        )
        diagnostic = _float(source["usable_maximum_sustained_sigma"])
        rows.append(
            {
                "T": tk,
                "K": tk,
                "scheme": source["scheme"],
                "checkpoint_role": source["checkpoint_role"],
                "acquisition_only_maximum_sustained_sigma": _float(
                    source["acquisition_only_maximum_sustained_sigma"]
                ),
                "diagnostic_gradient_usable_maximum_sustained_sigma": diagnostic,
                "equilibrium_valid_usable_maximum_sustained_sigma": (
                    diagnostic if residual_valid and clean_valid else None
                ),
                "usable_first_failing_sigma": _float(
                    source["usable_first_failing_sigma"]
                ),
                "minimum_clean_phase_delta_rms": float(
                    source["minimum_clean_positive_minus_negative_state_delta_rms"]
                ),
                "usable_sigma_over_minimum_clean_phase_delta_rms": _float(
                    source["usable_sigma_over_minimum_state_delta_rms"]
                ),
                "phase_difference_noise_rms_over_minimum_clean_phase_delta_rms": _float(
                    source[
                        "usable_phase_difference_noise_rms_over_minimum_state_delta_rms"
                    ]
                ),
                "clean_eqprop_vs_same_tk_bptt_gate_passed": clean_valid,
                "all_residual_gates_passed": residual_valid,
                "under_relaxed_diagnostic": not residual_valid,
                "clean_beta_selection_status": source["clean_beta_selection_status"],
            }
        )
    return rows


def _clean_fidelity(run_dir: Path, tk: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for source in _read_csv(run_dir / "layer_quantiles.csv"):
        if float(source["sigma_voltage"]) != 0.0:
            continue
        rows.append(
            {
                "T": tk,
                "K": tk,
                "scheme": source["scheme"],
                "checkpoint_role": source["checkpoint_role"],
                "parameter_name": source["parameter_name"],
                "layer": source["layer"],
                "cosine_vs_same_tk_bptt": _float(source["task_cosine_p05"]),
                "symmetric_norm_delta_vs_same_tk_bptt": float(
                    source["task_symmetric_norm_delta_p95"]
                ),
                "gate_passed": _bool(source["task_gate_passed"]),
            }
        )
    return rows


def _residual_summary(run_dir: Path, tk: int) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in _read_csv(run_dir / "phase_residuals.csv"):
        grouped.setdefault((row["scheme"], row["checkpoint_role"]), []).append(row)
    result: list[dict[str, Any]] = []
    for (scheme, role), values in sorted(grouped.items()):
        limiting = max(values, key=lambda row: float(row["selected_max_p90"]))
        result.append(
            {
                "T": tk,
                "K": tk,
                "scheme": scheme,
                "checkpoint_role": role,
                "maximum_phase_layer_selected_max_p90": float(
                    limiting["selected_max_p90"]
                ),
                "limiting_phase": limiting["phase"],
                "limiting_state_layer_name": limiting["state_layer_name"],
                "maximum_selected_maximum": max(
                    float(row["selected_maximum"]) for row in values
                ),
                "all_residual_gates_passed": all(
                    _bool(row["gate_passed"]) for row in values
                ),
                "failed_phase_layer_count": sum(
                    not _bool(row["gate_passed"]) for row in values
                ),
                "any_hard_failure": any(_bool(row["hard_failure"]) for row in values),
                "gate_threshold": float(limiting["gate_threshold"]),
            }
        )
    return result


def _displacement(run_dir: Path, tk: int) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for source in _read_csv(run_dir / "clean_state_displacement.csv"):
        result.append(
            {
                "T": tk,
                "K": tk,
                "scheme": source["scheme"],
                "checkpoint_role": source["checkpoint_role"],
                "state_layer_name": source["state_layer_name"],
                "state_layer_type": source["state_layer_type"],
                "delta_rms": float(source["delta_rms"]),
                "displacement_l2": float(source["displacement_l2"]),
                "relative_displacement": float(source["relative_displacement"]),
                "max_abs_delta": float(source["max_abs_delta"]),
            }
        )
    return result


def _drift(lower_run: Path, tk: int) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for source in _read_csv(lower_run / "tk64_gradient_drift.csv"):
        result.append(
            {
                "T": tk,
                "K": tk,
                "reference_T": 64,
                "reference_K": 64,
                "scheme": source["scheme"],
                "checkpoint_role": source["checkpoint_role"],
                "parameter_name": source["parameter_name"],
                "layer": source["layer"],
                "quantity": source["quantity"],
                "cosine_vs_tk64": _float(source["cosine"]),
                "symmetric_norm_delta_vs_tk64": float(source["symmetric_norm_delta"]),
                "relative_l2_vs_tk64": _float(source["relative_l2"]),
                "norm_ratio_vs_tk64": _float(source["norm_ratio"]),
                "candidate_zero_fraction": float(source["candidate_zero_fraction"]),
                "reference_zero_fraction": float(source["reference_zero_fraction"]),
                "drift_gate_passed": _bool(source["gate_passed"]),
            }
        )
    return result


def _noise_configuration(run_dir: Path, tk: int) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for source in _read_csv(run_dir / "configuration_summary.csv"):
        result.append(
            {
                "T": tk,
                "K": tk,
                "scheme": source["scheme"],
                "checkpoint_role": source["checkpoint_role"],
                "sigma_voltage": float(source["sigma_voltage"]),
                "minimum_layer_acquisition_cosine_p05": _float(
                    source["minimum_layer_acquisition_cosine_p05"]
                ),
                "maximum_layer_acquisition_symmetric_norm_delta_p95": float(
                    source["maximum_layer_acquisition_symmetric_norm_delta_p95"]
                ),
                "minimum_layer_task_cosine_p05": _float(
                    source["minimum_layer_task_cosine_p05"]
                ),
                "maximum_layer_task_symmetric_norm_delta_p95": float(
                    source["maximum_layer_task_symmetric_norm_delta_p95"]
                ),
                "all_layers_acquisition_gate_passed": _bool(
                    source["all_layers_acquisition_gate_passed"]
                ),
                "all_layers_task_gate_passed": _bool(
                    source["all_layers_task_gate_passed"]
                ),
                "all_layers_usable_gate_passed": _bool(
                    source["all_layers_usable_gate_passed"]
                ),
                "all_layers_all_trials_usable_gate_passed": _bool(
                    source["all_layers_all_trials_usable_gate_passed"]
                ),
            }
        )
    return result


def _classification_consistency(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    fields = (
        "all_layers_acquisition_gate_passed",
        "all_layers_task_gate_passed",
        "all_layers_usable_gate_passed",
        "all_layers_all_trials_usable_gate_passed",
    )
    lookup = {
        (
            int(row["T"]),
            str(row["scheme"]),
            str(row["checkpoint_role"]),
            float(row["sigma_voltage"]),
        ): row
        for row in rows
    }
    result: list[dict[str, Any]] = []
    for tk in (8, 12):
        for scheme in SCHEMES:
            for role in ROLES:
                sigma_values = sorted(
                    float(row["sigma_voltage"])
                    for row in rows
                    if int(row["T"]) == tk
                    and row["scheme"] == scheme
                    and row["checkpoint_role"] == role
                )
                for sigma in sigma_values:
                    lower = lookup[(tk, scheme, role, sigma)]
                    reference = lookup[(64, scheme, role, sigma)]
                    comparisons = {
                        f"{field}_matches_tk64": bool(lower[field] == reference[field])
                        for field in fields
                    }
                    result.append(
                        {
                            "T": tk,
                            "K": tk,
                            "reference_T": 64,
                            "reference_K": 64,
                            "scheme": scheme,
                            "checkpoint_role": role,
                            "sigma_voltage": sigma,
                            **{field: lower[field] for field in fields},
                            **comparisons,
                            "all_gate_classifications_match_tk64": all(
                                comparisons.values()
                            ),
                        }
                    )
    return result


def _plot_overview(
    path: Path,
    thresholds: Sequence[Mapping[str, Any]],
    fidelity: Sequence[Mapping[str, Any]],
    residuals: Sequence[Mapping[str, Any]],
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    for scheme in SCHEMES:
        for role in ROLES:
            label = f"{scheme}, {'init' if role.startswith('reconstructed') else 'best'}"
            selected_thresholds = sorted(
                [row for row in thresholds if row["scheme"] == scheme and row["checkpoint_role"] == role],
                key=lambda row: int(row["T"]),
            )
            x = [int(row["T"]) for row in selected_thresholds]
            y = [float(row["diagnostic_gradient_usable_maximum_sustained_sigma"]) for row in selected_thresholds]
            axes[0].plot(x, y, color=COLORS[scheme], linestyle=ROLE_STYLES[role], marker="o", label=label)
            for row in selected_thresholds:
                if row["under_relaxed_diagnostic"]:
                    axes[0].scatter(int(row["T"]), float(row["diagnostic_gradient_usable_maximum_sustained_sigma"]), marker="x", s=80, linewidth=2, color="black", zorder=4)
            selected_fidelity = [row for row in fidelity if row["scheme"] == scheme and row["checkpoint_role"] == role]
            worst = {
                tk: min(float(row["cosine_vs_same_tk_bptt"]) for row in selected_fidelity if int(row["T"]) == tk)
                for tk in (8, 12, 64)
            }
            axes[1].plot(list(worst), list(worst.values()), color=COLORS[scheme], linestyle=ROLE_STYLES[role], marker="o")
            selected_residuals = sorted(
                [row for row in residuals if row["scheme"] == scheme and row["checkpoint_role"] == role],
                key=lambda row: int(row["T"]),
            )
            axes[2].plot(
                [int(row["T"]) for row in selected_residuals],
                [float(row["maximum_phase_layer_selected_max_p90"]) for row in selected_residuals],
                color=COLORS[scheme], linestyle=ROLE_STYLES[role], marker="o",
            )
    for axis in axes:
        axis.set_xticks((8, 12, 64))
        axis.set_xlabel("Equal iteration budget T=K")
        axis.grid(True, which="both", alpha=0.25)
    axes[0].set_yscale("log")
    axes[0].set_ylabel(r"Usable endpoint-noise $\sigma_v$")
    axes[0].set_title("Gradient-only noise tolerance\n(x = residual-invalid)")
    axes[0].legend(fontsize=7, ncol=2)
    axes[1].axhline(0.99, color="black", linestyle=":", linewidth=1)
    axes[1].set_ylim(0.985, 1.0005)
    axes[1].set_ylabel("Worst-layer clean cosine")
    axes[1].set_title("Clean EqProp vs same-T/K BPTT")
    axes[2].axhline(1.0e-2, color="black", linestyle=":", linewidth=1)
    axes[2].set_yscale("log")
    axes[2].set_ylabel("Worst phase/layer residual p90")
    axes[2].set_title("Equilibrium residual")
    fig.suptitle("Conv3 centered float64: effect of reducing T=K from 64")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_layerwise(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    *,
    value_field: str,
    ylabel: str,
    log_y: bool,
    gate: float | None = None,
) -> None:
    layer_colors = {"C0": "#4C78A8", "C1": "#F58518", "C2": "#54A24B", "Dense": "#B279A2"}
    state_colors = {"Layer_1": "#4C78A8", "Layer_2": "#F58518", "Layer_3": "#54A24B", "Layer_4": "#B279A2"}
    fig, axes = plt.subplots(3, 2, figsize=(10.5, 10), sharex=True, constrained_layout=True)
    for row_index, scheme in enumerate(SCHEMES):
        for column_index, role in enumerate(ROLES):
            axis = axes[row_index, column_index]
            selected = [row for row in rows if row["scheme"] == scheme and row["checkpoint_role"] == role]
            key_field = "layer" if selected and "layer" in selected[0] else "state_layer_name"
            names = LAYERS if key_field == "layer" else ("Layer_1", "Layer_2", "Layer_3", "Layer_4")
            colors = layer_colors if key_field == "layer" else state_colors
            for name in names:
                values = sorted([row for row in selected if row[key_field] == name], key=lambda row: int(row["T"]))
                axis.plot([int(row["T"]) for row in values], [float(row[value_field]) for row in values], marker="o", color=colors[name], label=name)
            if gate is not None:
                axis.axhline(gate, color="black", linestyle=":", linewidth=1)
            if log_y:
                axis.set_yscale("log")
            axis.grid(True, which="both", alpha=0.25)
            axis.set_title(f"{scheme} — {'initialization' if role.startswith('reconstructed') else 'best epoch'}")
            axis.set_xticks((8, 12, 64))
            if column_index == 0:
                axis.set_ylabel(ylabel)
            if row_index == 2:
                axis.set_xlabel("Equal iteration budget T=K")
            if row_index == 0 and column_index == 1:
                axis.legend(fontsize=8)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_noise_overlay(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    *,
    field: str,
    ylabel: str,
    gate: float,
    log_y: bool,
) -> None:
    colors = {8: "#E45756", 12: "#F2CF5B", 64: "#4C78A8"}
    fig, axes = plt.subplots(
        3, 2, figsize=(10.8, 10), sharex=True, constrained_layout=True
    )
    for row_index, scheme in enumerate(SCHEMES):
        for column_index, role in enumerate(ROLES):
            axis = axes[row_index, column_index]
            for tk in (8, 12, 64):
                values = sorted(
                    [
                        row
                        for row in rows
                        if int(row["T"]) == tk
                        and row["scheme"] == scheme
                        and row["checkpoint_role"] == role
                        and float(row["sigma_voltage"]) > 0.0
                    ],
                    key=lambda row: float(row["sigma_voltage"]),
                )
                axis.plot(
                    [float(row["sigma_voltage"]) for row in values],
                    [float(row[field]) for row in values],
                    marker="o",
                    markersize=2.8,
                    linewidth=1.35,
                    color=colors[tk],
                    label=f"T=K={tk}",
                )
            axis.axhline(gate, color="black", linestyle=":", linewidth=1)
            axis.set_xscale("log")
            if log_y:
                axis.set_yscale("log")
            else:
                axis.set_ylim(-0.05, 1.02)
            axis.grid(True, which="both", alpha=0.25)
            axis.set_title(
                f"{scheme} — "
                f"{'initialization' if role.startswith('reconstructed') else 'best epoch'}"
            )
            if column_index == 0:
                axis.set_ylabel(ylabel)
            if row_index == 2:
                axis.set_xlabel(r"Endpoint read-noise $\sigma_v$")
            if row_index == 0 and column_index == 1:
                axis.legend(fontsize=8)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_drift(
    path: Path, drift: Sequence[Mapping[str, Any]]
) -> None:
    layer_colors = {
        "C0": "#4C78A8",
        "C1": "#F58518",
        "C2": "#54A24B",
        "Dense": "#B279A2",
    }
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.5), constrained_layout=True)
    for row_index, quantity in enumerate(("eqprop", "bptt")):
        for column_index, field in enumerate(("cosine_error", "relative_l2_vs_tk64")):
            axis = axes[row_index, column_index]
            for layer in LAYERS:
                for scheme in SCHEMES:
                    values = sorted(
                        [
                            row
                            for row in drift
                            if row["quantity"] == quantity
                            and row["layer"] == layer
                            and row["scheme"] == scheme
                        ],
                        key=lambda row: (
                            int(row["T"]),
                            str(row["checkpoint_role"]),
                        ),
                    )
                    # Show the worst checkpoint for each scheme/layer/T.
                    by_t = {
                        tk: max(
                            [row for row in values if int(row["T"]) == tk],
                            key=lambda row: (
                                1.0 - float(row["cosine_vs_tk64"])
                                if field == "cosine_error"
                                else float(row["relative_l2_vs_tk64"])
                            ),
                        )
                        for tk in (8, 12)
                    }
                    y = [
                        (
                            max(1.0 - float(by_t[tk]["cosine_vs_tk64"]), 1.0e-17)
                            if field == "cosine_error"
                            else max(float(by_t[tk]["relative_l2_vs_tk64"]), 1.0e-17)
                        )
                        for tk in (8, 12)
                    ]
                    axis.plot(
                        (8, 12),
                        y,
                        color=layer_colors[layer],
                        linestyle=ROLE_STYLES[
                            "reconstructed_initialization"
                            if scheme == "baseline"
                            else "best_validation"
                        ],
                        marker={"baseline": "o", "ours": "s", "legacy": "^"}[scheme],
                        linewidth=1,
                        markersize=4,
                        alpha=0.85,
                        label=f"{scheme}, {layer}",
                    )
            axis.set_yscale("log")
            axis.set_xticks((8, 12))
            axis.grid(True, which="both", alpha=0.25)
            axis.set_xlabel("Equal iteration budget T=K")
            axis.set_ylabel(
                r"$1-\cos(g_{T/K},g_{64})$"
                if field == "cosine_error"
                else r"$\|g_{T/K}-g_{64}\|/\|g_{64}\|$"
            )
            axis.set_title(f"{quantity.upper()} drift vs T=K=64")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    fig.legend(unique.values(), unique.keys(), loc="outside lower center", ncol=6, fontsize=7)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def run(args: argparse.Namespace) -> None:
    study_root = args.study_root.expanduser().resolve()
    t64_run = args.t64_run.expanduser().resolve()
    runs = {8: study_root / "tk8-v2", 12: study_root / "tk12-v2", 64: t64_run}
    sources = {str(tk): _validate_bundle(path) for tk, path in runs.items()}
    thresholds = [row for tk, path in runs.items() for row in _normalize_thresholds(path, tk)]
    fidelity = [row for tk, path in runs.items() for row in _clean_fidelity(path, tk)]
    residuals = [row for tk, path in runs.items() for row in _residual_summary(path, tk)]
    displacement = [row for tk, path in runs.items() for row in _displacement(path, tk)]
    drift = [row for tk in (8, 12) for row in _drift(runs[tk], tk)]
    noise_configuration = [
        row for tk, path in runs.items() for row in _noise_configuration(path, tk)
    ]
    classification = _classification_consistency(noise_configuration)
    if not all(row["all_gate_classifications_match_tk64"] for row in classification):
        raise ValueError("At least one lower-T/K noise gate classification differs from T64.")
    displacement_lookup = {
        (
            int(row["T"]),
            str(row["scheme"]),
            str(row["checkpoint_role"]),
            str(row["state_layer_name"]),
        ): row
        for row in displacement
    }
    for row in displacement:
        reference = displacement_lookup[
            (
                64,
                str(row["scheme"]),
                str(row["checkpoint_role"]),
                str(row["state_layer_name"]),
            )
        ]
        row["delta_rms_over_tk64"] = float(row["delta_rms"]) / float(
            reference["delta_rms"]
        )
        row["relative_displacement_over_tk64"] = float(
            row["relative_displacement"]
        ) / float(reference["relative_displacement"])
    expected_threshold_keys = {
        (tk, scheme, role) for tk in (8, 12, 64) for scheme in SCHEMES for role in ROLES
    }
    observed_threshold_keys = {
        (int(row["T"]), row["scheme"], row["checkpoint_role"]) for row in thresholds
    }
    if observed_threshold_keys != expected_threshold_keys:
        raise ValueError("Threshold comparison coverage is incomplete.")
    output = study_root / "analysis"
    output.mkdir(parents=True, exist_ok=True)
    _write_csv(output / "threshold_comparison.csv", thresholds)
    _write_csv(output / "layerwise_clean_fidelity.csv", fidelity)
    _write_csv(output / "residual_summary.csv", residuals)
    _write_csv(output / "layerwise_phase_displacement.csv", displacement)
    _write_csv(output / "tk64_gradient_drift.csv", drift)
    _write_csv(output / "noise_configuration_summary.csv", noise_configuration)
    _write_csv(output / "gate_classification_consistency.csv", classification)
    _plot_overview(output / "tk_overview.png", thresholds, fidelity, residuals)
    _plot_layerwise(output / "layerwise_clean_cosine_vs_tk.png", fidelity, value_field="cosine_vs_same_tk_bptt", ylabel="Clean cosine vs same-T/K BPTT", log_y=False, gate=0.99)
    _plot_layerwise(output / "layerwise_phase_delta_rms_vs_tk.png", displacement, value_field="delta_rms", ylabel="Positive-negative phase delta RMS", log_y=True)
    _plot_noise_overlay(
        output / "noise_task_cosine_vs_sigma_by_tk.png",
        noise_configuration,
        field="minimum_layer_task_cosine_p05",
        ylabel="Worst-layer p05 cosine vs BPTT",
        gate=0.99,
        log_y=False,
    )
    _plot_noise_overlay(
        output / "noise_task_norm_delta_vs_sigma_by_tk.png",
        noise_configuration,
        field="maximum_layer_task_symmetric_norm_delta_p95",
        ylabel="Worst-layer p95 symmetric norm delta",
        gate=0.1,
        log_y=True,
    )
    _plot_drift(output / "gradient_drift_vs_tk64.png", drift)
    by_case: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for row in thresholds:
        by_case.setdefault((str(row["scheme"]), str(row["checkpoint_role"])), []).append(row)
    lines = [
        "# Conv3 centered float64 endpoint-voltage read noise: T/K comparison",
        "",
        "One fixed ordinary-MNIST batch, the same six checkpoints, the T=K=64-selected beta values, centered current nudging, float64 endpoints/subtraction, and 32 matched trials per nonzero sigma are held fixed.",
        "",
        "| scheme | checkpoint | usable sigma at 8 / 12 / 64 | residual-valid at 8 / 12 / 64 |",
        "|---|---|---:|---:|",
    ]
    for scheme in SCHEMES:
        for role in ROLES:
            values = {int(row["T"]): row for row in by_case[(scheme, role)]}
            lines.append(
                f"| {scheme} | {role} | "
                + " / ".join(str(values[tk]["diagnostic_gradient_usable_maximum_sustained_sigma"]) for tk in (8, 12, 64))
                + " | "
                + " / ".join(str(values[tk]["all_residual_gates_passed"]) for tk in (8, 12, 64))
                + " |"
            )
    worst_clean = min(
        fidelity,
        key=lambda row: float(row["cosine_vs_same_tk_bptt"]),
    )
    worst_t8_drift = max(
        [row for row in drift if int(row["T"]) == 8 and row["quantity"] == "eqprop"],
        key=lambda row: float(row["relative_l2_vs_tk64"]),
    )
    displacement_deviation = max(
        abs(float(row["delta_rms_over_tk64"]) - 1.0)
        for row in displacement
        if int(row["T"]) in (8, 12)
    )
    lines.extend(
        [
            "",
            f"All {len(classification)} lower-T/K configuration rows exactly match the corresponding T=K=64 acquisition, task, usable, and all-trials usable gate classifications.",
            "",
            f"The worst clean same-T/K EqProp/BPTT cosine is {float(worst_clean['cosine_vs_same_tk_bptt']):.9f} ({worst_clean['scheme']}, {worst_clean['checkpoint_role']}, {worst_clean['layer']}, T=K={worst_clean['T']}). The largest T=K=8 EqProp relative-L2 drift versus T=K=64 is {float(worst_t8_drift['relative_l2_vs_tk64']):.6g} ({worst_t8_drift['scheme']}, {worst_t8_drift['checkpoint_role']}, {worst_t8_drift['layer']}). The maximum absolute fractional change in layerwise phase-delta RMS at lower T/K is {displacement_deviation:.6g}.",
            "",
            "A numeric gradient-only threshold from an under-relaxed case is retained as a diagnostic but is not equilibrium-valid evidence. Cross-T/K gradient drift uses accepted T=K=64 tensors only as references; the task comparison at every T/K uses BPTT recomputed from that same post-T state with the same K.",
        ]
    )
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    _write_json(
        output / "analysis_manifest.json",
        {
            "schema_version": "perfectdiode-conv3-centered-float64-voltage-read-noise-tk-summary/v1",
            "sources": sources,
            "fixed_contract": {
                "cohort": "same frozen 16-example ordinary-MNIST batch",
                "beta": "same six T64-selected injected B and base beta values",
                "precision": "float64",
                "eqprop_variant": "centered",
                "nudging_mode": "current",
                "trial_count_per_nonzero_sigma": 32,
                "bias_gradients_excluded": True,
                "optimizer_steps_applied": False,
                "official_test_read": False,
            },
            "outputs": {
                path.name: _sha256(path)
                for path in sorted(output.iterdir())
                if path.is_file() and path.name != "analysis_manifest.json"
            },
        },
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-root", type=Path, default=DEFAULT_STUDY_ROOT)
    parser.add_argument("--t64-run", type=Path, default=DEFAULT_T64_RUN)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    run(parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
