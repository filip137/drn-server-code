#!/usr/bin/env python3
"""Analyze the matched Conv1/2/3 centered true-float64 beta sweep at T=K=8."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping, Sequence

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt

from experiments import reporting


DEFAULT_STUDY_ROOT = (
    REPOSITORY_ROOT
    / "results/perfectdiode-conv123-baseline-ours-centered-float64-beta-1em3-to-1e3-tk8-seed0-20260812-v1"
)
SCHEMA = "conv123-centered-float64-beta-tk8-analysis/v1"
ARCHITECTURES = ("conv1", "conv2", "conv3")
SCHEMES = ("baseline", "ours", "legacy")
DEFAULT_SCHEMES = ("baseline", "ours")
BETAS = (0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0)
THRESHOLDS = (0.90, 0.95, 0.99)
FACTORS = {
    ("conv1", "baseline"): 1.0,
    ("conv1", "ours"): 4.0,
    ("conv2", "baseline"): 1.0,
    ("conv2", "ours"): 16.0,
    ("conv3", "baseline"): 1.0,
    ("conv3", "ours"): 64.0,
    ("conv1", "legacy"): 16.0,
    ("conv2", "legacy"): 256.0,
    ("conv3", "legacy"): 4096.0,
}
PARAMETERS = {
    "conv1": ("ConvWeight_0", "DenseWeight_0"),
    "conv2": ("ConvWeight_0", "ConvWeight_1", "DenseWeight_0"),
    "conv3": ("ConvWeight_0", "ConvWeight_1", "ConvWeight_2", "DenseWeight_0"),
}


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return [dict(row) for row in csv.DictReader(stream)]


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty table {path}.")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _boolean(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no", ""}:
        return False
    raise ValueError(f"Expected boolean, got {value!r}.")


def _close(left: Any, right: Any) -> bool:
    return math.isclose(float(left), float(right), rel_tol=1.0e-9, abs_tol=1.0e-14)


def _beta_label(beta: float) -> str:
    text = f"{beta:g}"
    if beta < 1.0:
        text = text[1:]
    return text.replace(".", "p")


def _run_id(architecture: str, scheme: str, beta: float) -> str:
    return f"production-centered-{architecture}-{scheme}-B{_beta_label(beta)}-tk8"


def _state_layer_index(name: str) -> int:
    if not name.startswith("Layer_"):
        raise ValueError(f"Unexpected state layer name {name!r}.")
    return int(name.split("_")[-1])


def _single(values: Iterable[Any], *, label: str) -> Any:
    unique = set(values)
    if len(unique) != 1:
        raise ValueError(f"Expected one {label}, got {sorted(unique, key=str)}.")
    return next(iter(unique))


def _validate_guards(run_id: str, guards: Mapping[str, Any]) -> None:
    required_true = (
        "all_dtype_proofs_passed",
        "all_iteration_contracts_passed",
        "all_parameter_and_arithmetic_guards_passed",
        "cohort_exact",
        "source_bundle_files_unchanged",
        "source_checkpoint_bytes_unchanged",
        "source_config_semantically_exact",
    )
    if not all(guards.get(key) is True for key in required_true):
        raise ValueError(f"Source/dtype/arithmetic guard failed in {run_id}.")
    required_false = ("optimizer_constructed", "optimizer_steps_applied", "official_test_read")
    if not all(guards.get(key) is False for key in required_false):
        raise ValueError(f"Forbidden optimizer/test action in {run_id}.")
    replay = guards.get("bptt_replay", {})
    if replay.get("all_bptt_and_configured_eqprop_phases_started_from_common_post_T") is not True:
        raise ValueError(f"Common phase start guard failed in {run_id}.")


def _residual_summary(run_dir: Path) -> dict[str, Any]:
    rows = [
        row
        for row in _read_csv(run_dir / "equilibrium_residual_summary.csv")
        if row["precision"] == "float64"
    ]
    if not rows:
        raise ValueError(f"Missing float64 residual rows in {run_dir}.")
    return {
        "float64_all_residual_gates_passed": all(_boolean(row["gate_passed"]) for row in rows),
        "float64_residual_hard_failure": any(_boolean(row["hard_failure"]) for row in rows),
        "float64_maximum_residual_selected_p90": max(float(row["selected_max_p90"]) for row in rows),
        "float64_maximum_residual_selected_maximum": max(float(row["selected_maximum"]) for row in rows),
    }


def analyze(
    study_root: Path,
    *,
    schemes: Sequence[str] = DEFAULT_SCHEMES,
    launcher_exitcode_name: str = "production-launch.log.exitcode",
    authoritative_smoke_run_id: str = "smoke-gpu-centered-conv3-ours-B0p1-tk8",
    excluded_smoke_run_id: str | None = "smoke-cpu-centered-conv3-ours-B0p1-tk8",
    excluded_smoke_reason: str = "CPU process externally interrupted at starting with no scientific artifacts or terminal result",
) -> dict[str, Any]:
    study_root = study_root.expanduser().resolve()
    schemes = tuple(schemes)
    if not schemes or len(set(schemes)) != len(schemes):
        raise ValueError("--schemes must contain one or more unique schemes.")
    unsupported = set(schemes) - set(SCHEMES)
    if unsupported:
        raise ValueError(f"Unsupported schemes: {sorted(unsupported)}.")
    expected = {
        _run_id(architecture, scheme, beta)
        for architecture in ARCHITECTURES
        for scheme in schemes
        for beta in BETAS
    }
    discovered = {path.parent.name for path in study_root.glob("production-*/status.json")}
    if discovered != expected:
        raise ValueError(
            f"Production coverage mismatch: missing={sorted(expected - discovered)}, "
            f"extra={sorted(discovered - expected)}."
        )

    curve_rows: list[dict[str, Any]] = []
    layer_rows: list[dict[str, Any]] = []
    displacement_rows: list[dict[str, Any]] = []
    included_runs: list[dict[str, Any]] = []
    batch_payloads: set[str] = set()
    batch_sources: set[str] = set()
    checkpoint_hashes: dict[tuple[str, str], set[str]] = {}
    bundle_study_ids: set[str] = set()

    for architecture in ARCHITECTURES:
        for scheme in schemes:
            factor = FACTORS[(architecture, scheme)]
            for beta in BETAS:
                run_id = _run_id(architecture, scheme, beta)
                run_dir = study_root / run_id
                errors = reporting.validate_run(run_dir)
                if errors:
                    raise ValueError(f"Invalid canonical bundle {run_id}: {'; '.join(errors)}")
                status = _read_json(run_dir / "status.json")
                result = _read_json(run_dir / "result.json")
                manifest = _read_json(run_dir / "manifest.json")
                guards = _read_json(run_dir / "read_only_guards.json")
                if status.get("state") != "complete" or result["completion"].get("criteria_met") is not True:
                    raise ValueError(f"Non-complete result in {run_id}.")
                _validate_guards(run_id, guards)
                bundle_study_ids.add(str(result["study_id"]))

                parameters = _read_csv(run_dir / "parameter_precision_comparison.csv")
                if tuple(row["parameter_name"] for row in parameters) != PARAMETERS[architecture]:
                    raise ValueError(f"Unexpected parameter order in {run_id}.")
                if _single((row["architecture"] for row in parameters), label="architecture") != architecture:
                    raise ValueError(f"Architecture mismatch in {run_id}.")
                if _single((row["scheme"] for row in parameters), label="scheme") != scheme:
                    raise ValueError(f"Scheme mismatch in {run_id}.")
                if _single((row["eqprop_variant"] for row in parameters), label="estimator") != "centered":
                    raise ValueError(f"Estimator mismatch in {run_id}.")
                if {(int(row["T"]), int(row["K"])) for row in parameters} != {(8, 8)}:
                    raise ValueError(f"T/K mismatch in {run_id}.")
                if not all(_close(row["effective_beta"], beta) for row in parameters):
                    raise ValueError(f"Injected beta mismatch in {run_id}.")
                if not all(_close(row["actual_beta"], beta / factor) for row in parameters):
                    raise ValueError(f"Base beta mismatch in {run_id}.")
                if not all(_close(row["amplification_factor"], factor) for row in parameters):
                    raise ValueError(f"Amplification-factor mismatch in {run_id}.")
                if not all(row["float64_eqprop_gradient_dtype"] == "torch.float64" for row in parameters):
                    raise ValueError(f"Non-float64 EqProp gradient in {run_id}.")
                if not all(row["float64_bptt_gradient_dtype"] == "torch.float64" for row in parameters):
                    raise ValueError(f"Non-float64 BPTT gradient in {run_id}.")

                batch_payload = _single((row["batch_payload_sha256"] for row in parameters), label="batch payload")
                batch_source = manifest["replay"]["fixed_batch"]["source_indices_sha256"]
                batch_payloads.add(batch_payload)
                batch_sources.add(batch_source)
                checkpoint_hash = guards["source_checkpoint_hashes_before"][f"{scheme}/{architecture}"]["best_model.pt"]
                checkpoint_hashes.setdefault((architecture, scheme), set()).add(checkpoint_hash)

                cosines: list[float] = []
                for parameter in parameters:
                    cosine_text = parameter["float64_eqprop_vs_bptt_cosine"]
                    if not cosine_text:
                        raise ValueError(f"Undefined float64 cosine in {run_id}/{parameter['parameter_name']}.")
                    cosine = float(cosine_text)
                    cosines.append(cosine)
                    layer_rows.append(
                        {
                            "architecture": architecture,
                            "scheme": scheme,
                            "checkpoint_role": "best_validation",
                            "T": 8,
                            "K": 8,
                            "injected_beta": beta,
                            "actual_base_beta": beta / factor,
                            "amplification_factor": factor,
                            "parameter_name": parameter["parameter_name"],
                            "float64_eqprop_vs_bptt_cosine": cosine,
                            "float64_eqprop_over_bptt_norm_ratio": float(parameter["float64_eqprop_over_bptt_norm_ratio"]),
                            "float64_eqprop_vs_bptt_symmetric_norm_delta": float(parameter["float64_eqprop_vs_bptt_symmetric_norm_delta"]),
                            "float64_eqprop_exact_zero_fraction": float(parameter["float64_exact_zero_fraction"]),
                            "run_id": run_id,
                            "input_result_sha256": reporting.sha256_file(run_dir / "result.json"),
                        }
                    )

                states = [
                    row
                    for row in _read_csv(run_dir / "state_displacement.csv")
                    if row["precision"] == "float64"
                    and row["reference_kind"] == "matched_zero_K"
                    and row["state_layer_type"] != "aggregate"
                ]
                by_phase = {
                    phase: sorted(
                        (row for row in states if row["phase"] == phase),
                        key=lambda row: _state_layer_index(row["state_layer_name"]),
                    )
                    for phase in ("negative", "positive")
                }
                expected_layers = len(PARAMETERS[architecture])
                if any(len(rows) != expected_layers for rows in by_phase.values()):
                    raise ValueError(f"Incomplete float64 displacement in {run_id}.")
                for parameter_name, negative, positive in zip(
                    PARAMETERS[architecture], by_phase["negative"], by_phase["positive"]
                ):
                    if negative["state_layer_name"] != positive["state_layer_name"]:
                        raise ValueError(f"Signed state-layer mismatch in {run_id}.")
                    displacement_rows.append(
                        {
                            "architecture": architecture,
                            "scheme": scheme,
                            "checkpoint_role": "best_validation",
                            "T": 8,
                            "K": 8,
                            "injected_beta": beta,
                            "actual_base_beta": beta / factor,
                            "amplification_factor": factor,
                            "parameter_name": parameter_name,
                            "state_layer_name": positive["state_layer_name"],
                            "state_layer_type": positive["state_layer_type"],
                            "float64_positive_relative_displacement": float(positive["relative_displacement"]),
                            "float64_positive_delta_rms": float(positive["delta_rms"]),
                            "float64_positive_active_set_transition_fraction": float(positive["active_set_transition_fraction"]),
                            "float64_negative_relative_displacement": float(negative["relative_displacement"]),
                            "float64_negative_delta_rms": float(negative["delta_rms"]),
                            "float64_negative_active_set_transition_fraction": float(negative["active_set_transition_fraction"]),
                            "run_id": run_id,
                            "input_result_sha256": reporting.sha256_file(run_dir / "result.json"),
                        }
                    )

                minimum = min(cosines)
                limiting_index = cosines.index(minimum)
                residual = _residual_summary(run_dir)
                curve_rows.append(
                    {
                        "architecture": architecture,
                        "scheme": scheme,
                        "checkpoint_role": "best_validation",
                        "T": 8,
                        "K": 8,
                        "injected_beta": beta,
                        "actual_base_beta": beta / factor,
                        "amplification_factor": factor,
                        "float64_minimum_eqprop_vs_bptt_cosine": minimum,
                        "float64_limiting_parameter": PARAMETERS[architecture][limiting_index],
                        "float64_maximum_symmetric_norm_delta": max(
                            float(row["float64_eqprop_vs_bptt_symmetric_norm_delta"])
                            for row in parameters
                        ),
                        "float64_first_hidden_positive_relative_displacement": float(by_phase["positive"][0]["relative_displacement"]),
                        "float64_output_positive_relative_displacement": float(by_phase["positive"][-1]["relative_displacement"]),
                        "float64_output_negative_relative_displacement": float(by_phase["negative"][-1]["relative_displacement"]),
                        "batch_payload_sha256": batch_payload,
                        "batch_source_indices_sha256": batch_source,
                        "best_checkpoint_sha256": checkpoint_hash,
                        **residual,
                        "run_id": run_id,
                        "input_result_sha256": reporting.sha256_file(run_dir / "result.json"),
                    }
                )
                included_runs.append(
                    {
                        "run_id": run_id,
                        "path": str(run_dir),
                        "result_sha256": reporting.sha256_file(run_dir / "result.json"),
                        "manifest_study_id": result["study_id"],
                    }
                )

    if len(batch_payloads) != 1 or len(batch_sources) != 1:
        raise ValueError("The surface does not share one fixed minibatch.")
    if any(len(hashes) != 1 for hashes in checkpoint_hashes.values()):
        raise ValueError("Checkpoint identity changed within an architecture/scheme curve.")

    architecture_order = {value: index for index, value in enumerate(ARCHITECTURES)}
    scheme_order = {value: index for index, value in enumerate(schemes)}
    curve_rows.sort(key=lambda row: (architecture_order[row["architecture"]], scheme_order[row["scheme"]], row["injected_beta"]))
    layer_rows.sort(key=lambda row: (architecture_order[row["architecture"]], scheme_order[row["scheme"]], row["injected_beta"], PARAMETERS[row["architecture"]].index(row["parameter_name"])))
    displacement_rows.sort(key=lambda row: (architecture_order[row["architecture"]], scheme_order[row["scheme"]], row["injected_beta"], PARAMETERS[row["architecture"]].index(row["parameter_name"])))

    threshold_rows: list[dict[str, Any]] = []
    for architecture in ARCHITECTURES:
        for scheme in schemes:
            curve = [
                row
                for row in curve_rows
                if row["architecture"] == architecture and row["scheme"] == scheme
            ]
            for threshold in THRESHOLDS:
                passing = [
                    row
                    for row in curve
                    if row["float64_minimum_eqprop_vs_bptt_cosine"] >= threshold
                ]
                qualified = [row for row in passing if row["float64_all_residual_gates_passed"]]
                maximum = max(passing, key=lambda row: row["injected_beta"]) if passing else None
                maximum_qualified = max(qualified, key=lambda row: row["injected_beta"]) if qualified else None
                threshold_rows.append(
                    {
                        "architecture": architecture,
                        "scheme": scheme,
                        "cosine_threshold": threshold,
                        "maximum_tested_injected_beta": None if maximum is None else maximum["injected_beta"],
                        "actual_base_beta_at_maximum": None if maximum is None else maximum["actual_base_beta"],
                        "minimum_cosine_at_maximum_tested_beta": None if maximum is None else maximum["float64_minimum_eqprop_vs_bptt_cosine"],
                        "limiting_parameter_at_maximum_tested_beta": None if maximum is None else maximum["float64_limiting_parameter"],
                        "maximum_is_tested_upper_edge": bool(maximum is not None and _close(maximum["injected_beta"], BETAS[-1])),
                        "passing_tested_betas": ";".join(f"{row['injected_beta']:g}" for row in passing),
                        "residual_qualified_maximum_tested_injected_beta": None if maximum_qualified is None else maximum_qualified["injected_beta"],
                        "residual_qualified_actual_base_beta": None if maximum_qualified is None else maximum_qualified["actual_base_beta"],
                        "residual_qualified_minimum_cosine": None if maximum_qualified is None else maximum_qualified["float64_minimum_eqprop_vs_bptt_cosine"],
                        "residual_qualification_note": (
                            "qualified"
                            if maximum_qualified is not None
                            else "no cosine-passing beta also passes all float64 residual gates"
                        ),
                    }
                )

    analysis_dir = study_root / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(analysis_dir / "curve_summary.csv", curve_rows)
    _write_csv(analysis_dir / "layer_cosine.csv", layer_rows)
    _write_csv(analysis_dir / "layer_displacement.csv", displacement_rows)
    _write_csv(analysis_dir / "max_beta_by_cosine_threshold.csv", threshold_rows)
    _write_plot(
        analysis_dir / "float64_cosine_and_displacement_vs_beta.png",
        curve_rows,
        displacement_rows,
        schemes=schemes,
    )
    _write_report(
        analysis_dir / "report.md",
        threshold_rows,
        curve_rows,
        schemes=schemes,
    )

    authoritative_smoke = study_root / authoritative_smoke_run_id
    smoke_summary: dict[str, Any] = {
        "authoritative": {
            "path": str(authoritative_smoke),
            "result_sha256": reporting.sha256_file(authoritative_smoke / "result.json"),
        }
    }
    if excluded_smoke_run_id is not None:
        incomplete_smoke = study_root / excluded_smoke_run_id
        smoke_summary["excluded"] = {
            "path": str(incomplete_smoke),
            "reason": excluded_smoke_reason,
        }
    summary = {
        "schema": SCHEMA,
        "study_root": str(study_root),
        "coverage": {
            "production_bundles": len(included_runs),
            "curve_rows": len(curve_rows),
            "layer_cosine_rows": len(layer_rows),
            "layer_displacement_rows": len(displacement_rows),
            "threshold_rows": len(threshold_rows),
            "architectures": list(ARCHITECTURES),
            "schemes": list(schemes),
            "injected_betas": list(BETAS),
            "T": 8,
            "K": 8,
        },
        "guards": {
            "all_production_bundles_canonical_and_complete": True,
            "all_source_checkpoint_parameter_phase_dtype_and_arithmetic_guards_passed": True,
            "one_fixed_batch": True,
            "optimizer_steps_applied": False,
            "official_test_read": False,
            "launcher_exit_code": int((study_root / launcher_exitcode_name).read_text().strip()),
        },
        "smokes": smoke_summary,
        "reporting_deviation": {
            "bundle_manifest_study_ids": sorted(bundle_study_ids),
            "enclosing_study_id": study_root.name,
            "note": "The reusable audit runner retains its source-compatible study ID; the enclosing directory and complete inventory bind this T=K=8 surface.",
        },
        "included_runs": included_runs,
        "artifacts": {
            "curve_summary": "analysis/curve_summary.csv",
            "layer_cosine": "analysis/layer_cosine.csv",
            "layer_displacement": "analysis/layer_displacement.csv",
            "thresholds": "analysis/max_beta_by_cosine_threshold.csv",
            "figure": "analysis/float64_cosine_and_displacement_vs_beta.png",
            "report": "analysis/report.md",
        },
    }
    reporting.atomic_write_json(analysis_dir / "summary.json", summary)
    return summary


def _write_plot(
    path: Path,
    curve_rows: Sequence[Mapping[str, Any]],
    displacement_rows: Sequence[Mapping[str, Any]],
    *,
    schemes: Sequence[str],
) -> None:
    colors = {"baseline": "#1f77b4", "ours": "#d62728", "legacy": "#2ca02c"}
    markers = {"baseline": "o", "ours": "s", "legacy": "^"}
    fig, axes = plt.subplots(3, 2, figsize=(11.5, 11.0), sharex="col")
    for row_index, architecture in enumerate(ARCHITECTURES):
        cosine_axis, displacement_axis = axes[row_index]
        for scheme in schemes:
            curve = [
                row
                for row in curve_rows
                if row["architecture"] == architecture and row["scheme"] == scheme
            ]
            cosine_axis.plot(
                [row["injected_beta"] for row in curve],
                [row["float64_minimum_eqprop_vs_bptt_cosine"] for row in curve],
                color=colors[scheme], marker=markers[scheme], linewidth=1.7, label=scheme,
            )
            for parameter_index, parameter_name in enumerate(PARAMETERS[architecture]):
                states = [
                    row
                    for row in displacement_rows
                    if row["architecture"] == architecture
                    and row["scheme"] == scheme
                    and row["parameter_name"] == parameter_name
                ]
                displacement_axis.plot(
                    [row["injected_beta"] for row in states],
                    [row["float64_positive_relative_displacement"] for row in states],
                    color=colors[scheme],
                    linestyle=("-", "--", ":", "-.")[parameter_index],
                    linewidth=1.5,
                    label=f"{scheme} {parameter_name}",
                )
        for threshold in THRESHOLDS:
            cosine_axis.axhline(threshold, color="0.45", linestyle=":", linewidth=0.8)
        cosine_axis.set_xscale("log")
        cosine_axis.set_ylim(-1.02, 1.02)
        cosine_axis.set_ylabel(f"{architecture.title()}\nworst-layer cosine")
        cosine_axis.grid(True, which="both", alpha=0.24)
        displacement_axis.set_xscale("log")
        displacement_axis.set_yscale("log")
        displacement_axis.set_ylabel("positive relative displacement")
        displacement_axis.grid(True, which="both", alpha=0.24)
        if row_index == 0:
            cosine_axis.set_title("Centered float64 EqProp vs same-T/K BPTT")
            displacement_axis.set_title("Layerwise matched-zero displacement")
        if row_index == 2:
            cosine_axis.set_xlabel("injected beta B")
            displacement_axis.set_xlabel("injected beta B")
        cosine_axis.legend(fontsize=8, loc="lower left")
        displacement_axis.legend(fontsize=6.5, ncol=2, loc="best")
    fig.suptitle("Matched Conv1/Conv2/Conv3 centered float64 beta sweep, T=K=8")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _display(value: Any) -> str:
    if value is None:
        return "none"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _write_report(
    path: Path,
    threshold_rows: Sequence[Mapping[str, Any]],
    curve_rows: Sequence[Mapping[str, Any]],
    *,
    schemes: Sequence[str],
) -> None:
    scheme_label = "/".join(schemes)
    lines = [
        "# Matched centered float64 beta sweep at T=K=8",
        "",
        f"Exploratory ordinary-MNIST checkpoint-gradient diagnostic. Trained {scheme_label} best-validation Conv1/2/3 checkpoints use one fixed 16-example batch, centered frozen-current nudging, and a common 13-point injected-beta grid from `.001` through `1000`. The reported cosine is the minimum across all weight layers against same-T/K BPTT.",
        "",
        "## Maximum tested beta by minimum-cosine threshold",
        "",
        "| architecture | scheme | threshold | max tested B | actual base beta | min cosine there | limiting layer | upper tested edge | residual-qualified max B |",
        "|---|---|---:|---:|---:|---:|---|---|---:|",
    ]
    for row in threshold_rows:
        lines.append(
            "| "
            + " | ".join(
                (
                    str(row["architecture"]),
                    str(row["scheme"]),
                    _display(row["cosine_threshold"]),
                    _display(row["maximum_tested_injected_beta"]),
                    _display(row["actual_base_beta_at_maximum"]),
                    _display(row["minimum_cosine_at_maximum_tested_beta"]),
                    _display(row["limiting_parameter_at_maximum_tested_beta"]),
                    "yes" if row["maximum_is_tested_upper_edge"] else "no",
                    _display(row["residual_qualified_maximum_tested_injected_beta"]),
                )
            )
            + " |"
        )
    residual_limited = sorted(
        {
            (row["architecture"], row["scheme"])
            for row in curve_rows
            if not row["float64_all_residual_gates_passed"]
        }
    )
    lines.extend(
        (
            "",
            "`max tested B` is the largest sampled beta satisfying the cosine threshold; it is not an interpolated crossing. A value at `1000` is an open tested edge and therefore a lower bound on the unknown true maximum.",
            "",
            "Residual qualification is reported separately because the user requested cosine thresholds. Residual-limited architecture/scheme surfaces: "
            + (", ".join(f"{a}/{s}" for a, s in residual_limited) if residual_limited else "none")
            + ".",
            "",
            "All production bundles pass canonical validation and source/checkpoint/parameter/common-start/frozen-force/iteration/dtype/native-arithmetic guards. No optimizer was constructed or stepped and the official test split was not read.",
            "",
        )
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-root", type=Path, default=DEFAULT_STUDY_ROOT)
    parser.add_argument(
        "--schemes",
        nargs="+",
        choices=SCHEMES,
        default=list(DEFAULT_SCHEMES),
        help="Schemes expected in the study root (default: baseline ours).",
    )
    parser.add_argument("--launcher-exitcode-name", default="production-launch.log.exitcode")
    parser.add_argument(
        "--authoritative-smoke-run-id",
        default="smoke-gpu-centered-conv3-ours-B0p1-tk8",
    )
    parser.add_argument(
        "--excluded-smoke-run-id",
        default="smoke-cpu-centered-conv3-ours-B0p1-tk8",
        help="Use 'none' when there is no excluded smoke bundle.",
    )
    parser.add_argument(
        "--excluded-smoke-reason",
        default="CPU process externally interrupted at starting with no scientific artifacts or terminal result",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    summary = analyze(
        args.study_root,
        schemes=args.schemes,
        launcher_exitcode_name=args.launcher_exitcode_name,
        authoritative_smoke_run_id=args.authoritative_smoke_run_id,
        excluded_smoke_run_id=(
            None if args.excluded_smoke_run_id.lower() == "none" else args.excluded_smoke_run_id
        ),
        excluded_smoke_reason=args.excluded_smoke_reason,
    )
    print(json.dumps(summary["coverage"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
