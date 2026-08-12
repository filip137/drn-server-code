#!/usr/bin/env python3
"""Analyze the matched Conv1/Conv2 higher-beta true-dtype EqProp replay."""

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
    / "results/perfectdiode-conv12-baseline-ours-eqprop-higher-beta-seed0-20260812-v1"
)
SCHEMA = "conv12-eqprop-higher-beta-analysis/v1"
ARCHITECTURES = ("conv1", "conv2")
SCHEMES = ("baseline", "ours")
VARIANTS = ("centered", "positive_one_sided")
PRECISIONS = ("float32", "float64")
BETAS = (100.0, 300.0, 1000.0)
FACTORS = {
    ("conv1", "baseline"): 1.0,
    ("conv1", "ours"): 4.0,
    ("conv2", "baseline"): 1.0,
    ("conv2", "ours"): 16.0,
}
LAYER_NAMES = {
    "conv1": ("ConvWeight_0", "DenseWeight_0"),
    "conv2": ("ConvWeight_0", "ConvWeight_1", "DenseWeight_0"),
}
COSINE_MINIMUM = 0.99
SYMMETRIC_NORM_DELTA_MAXIMUM = 0.10


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return [dict(row) for row in csv.DictReader(stream)]


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write an empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _close(left: float, right: float) -> bool:
    return math.isclose(float(left), float(right), rel_tol=1.0e-9, abs_tol=1.0e-12)


def _boolean(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no", ""}:
        return False
    raise ValueError(f"Expected boolean, got {value!r}.")


def _optional_float(value: Any) -> float | None:
    if value is None or str(value).strip().lower() in {"", "none", "null", "nan"}:
        return None
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"Expected finite value, got {value!r}.")
    return result


def _run_id(variant: str, architecture: str, scheme: str, beta: float) -> str:
    variant_label = "one-sided" if variant == "positive_one_sided" else variant
    return f"production-{variant_label}-{architecture}-{scheme}-B{int(beta)}-tk64"


def _single(values: Iterable[Any], *, label: str) -> Any:
    unique = set(values)
    if len(unique) != 1:
        raise ValueError(f"Expected one {label}, got {sorted(unique, key=str)}.")
    return next(iter(unique))


def _matched_displacement(
    run_dir: Path, *, architecture: str, variant: str
) -> dict[str, float | None]:
    rows = _read_csv(run_dir / "state_displacement.csv")
    matched = [
        row
        for row in rows
        if row["precision"] == "float64"
        and row["reference_kind"] == "matched_zero_K"
        and row["state_layer_type"] != "aggregate"
    ]
    positive = [row for row in matched if row["phase"] == "positive"]
    expected_count = len(LAYER_NAMES[architecture])
    if len(positive) != expected_count:
        raise ValueError(f"Unexpected positive displacement coverage in {run_dir}.")
    positive.sort(key=lambda row: int(row["state_layer_name"].split("_")[-1]))
    negative = [row for row in matched if row["phase"] == "negative"]
    if variant == "centered" and len(negative) != expected_count:
        raise ValueError(f"Unexpected negative displacement coverage in {run_dir}.")
    negative.sort(key=lambda row: int(row["state_layer_name"].split("_")[-1]))
    return {
        "float64_positive_first_hidden_relative_displacement": float(
            positive[0]["relative_displacement"]
        ),
        "float64_positive_output_relative_displacement": float(
            positive[-1]["relative_displacement"]
        ),
        "float64_negative_output_relative_displacement": (
            float(negative[-1]["relative_displacement"]) if negative else None
        ),
    }


def _residual_summary(run_dir: Path) -> dict[str, Any]:
    rows = _read_csv(run_dir / "equilibrium_residual_summary.csv")
    if not rows:
        raise ValueError(f"No residual rows in {run_dir}.")
    return {
        "maximum_residual_selected_p90": max(float(row["selected_max_p90"]) for row in rows),
        "all_residual_gates_passed": all(_boolean(row["gate_passed"]) for row in rows),
        "residual_hard_failure": any(_boolean(row["hard_failure"]) for row in rows),
    }


def analyze(study_root: Path) -> dict[str, Any]:
    study_root = study_root.expanduser().resolve()
    if not study_root.is_dir():
        raise FileNotFoundError(study_root)
    expected = {
        _run_id(variant, architecture, scheme, beta)
        for architecture in ARCHITECTURES
        for scheme in SCHEMES
        for beta in BETAS
        for variant in VARIANTS
    }
    discovered = {path.parent.name for path in study_root.glob("production-*/status.json")}
    if discovered != expected:
        raise ValueError(
            f"Production coverage mismatch; missing={sorted(expected - discovered)}, "
            f"extra={sorted(discovered - expected)}."
        )

    layer_rows: list[dict[str, Any]] = []
    context_rows: list[dict[str, Any]] = []
    included_runs: list[dict[str, Any]] = []
    cohort_payloads: set[str] = set()
    source_indices: set[str] = set()
    checkpoint_hashes: dict[tuple[str, str], set[str]] = {}
    bundle_study_ids: set[str] = set()

    for architecture in ARCHITECTURES:
        for scheme in SCHEMES:
            factor = FACTORS[(architecture, scheme)]
            for beta in BETAS:
                for variant in VARIANTS:
                    run_id = _run_id(variant, architecture, scheme, beta)
                    run_dir = study_root / run_id
                    errors = reporting.validate_run(run_dir)
                    if errors:
                        raise ValueError(f"Invalid bundle {run_id}: {'; '.join(errors)}")
                    status = _read_json(run_dir / "status.json")
                    result = _read_json(run_dir / "result.json")
                    manifest = _read_json(run_dir / "manifest.json")
                    guards = _read_json(run_dir / "read_only_guards.json")
                    if status.get("state") != "complete" or not result["completion"].get("criteria_met"):
                        raise ValueError(f"Non-complete bundle {run_id}.")
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
                        raise ValueError(f"Read-only or arithmetic guard failed in {run_id}.")
                    if any(
                        guards.get(key) is not False
                        for key in ("optimizer_constructed", "optimizer_steps_applied", "official_test_read")
                    ):
                        raise ValueError(f"Forbidden mutation or test read in {run_id}.")
                    bundle_study_ids.add(str(result["study_id"]))

                    parameters = _read_csv(run_dir / "parameter_precision_comparison.csv")
                    if tuple(row["parameter_name"] for row in parameters) != LAYER_NAMES[architecture]:
                        raise ValueError(f"Unexpected layer order in {run_id}.")
                    if _single((row["architecture"] for row in parameters), label="architecture") != architecture:
                        raise ValueError(f"Architecture mismatch in {run_id}.")
                    if _single((row["scheme"] for row in parameters), label="scheme") != scheme:
                        raise ValueError(f"Scheme mismatch in {run_id}.")
                    if _single((row["eqprop_variant"] for row in parameters), label="variant") != variant:
                        raise ValueError(f"Estimator mismatch in {run_id}.")
                    if {(int(row["T"]), int(row["K"])) for row in parameters} != {(64, 64)}:
                        raise ValueError(f"T/K mismatch in {run_id}.")
                    if not all(_close(row["effective_beta"], beta) for row in parameters):
                        raise ValueError(f"Injected beta mismatch in {run_id}.")
                    expected_base = beta / factor
                    if not all(_close(row["actual_beta"], expected_base) for row in parameters):
                        raise ValueError(f"Base beta mismatch in {run_id}.")
                    if not all(_close(row["amplification_factor"], factor) for row in parameters):
                        raise ValueError(f"Amplification factor mismatch in {run_id}.")
                    if not all(_boolean(row["source_grid_cap_bypassed"]) for row in parameters):
                        raise ValueError(f"Missing source-cap-bypass provenance in {run_id}.")

                    cohort_payloads.add(parameters[0]["batch_payload_sha256"])
                    selection_hash = manifest["configuration"]["resolved"]["selected_cases"][0][
                        "source_selection_row_sha256"
                    ]
                    batch_source_indices_sha256 = manifest["replay"]["fixed_batch"][
                        "source_indices_sha256"
                    ]
                    source_indices.add(batch_source_indices_sha256)
                    checkpoint_hash = guards["source_checkpoint_hashes_before"][
                        f"{scheme}/{architecture}"
                    ]["best_model.pt"]
                    checkpoint_hashes.setdefault((architecture, scheme), set()).add(checkpoint_hash)
                    displacement = _matched_displacement(
                        run_dir, architecture=architecture, variant=variant
                    )
                    residual = _residual_summary(run_dir)

                    context: dict[str, Any] = {
                        "architecture": architecture,
                        "scheme": scheme,
                        "eqprop_variant": variant,
                        "injected_beta": beta,
                        "actual_base_beta": expected_base,
                        "amplification_factor": factor,
                        "T": 64,
                        "K": 64,
                        "batch_payload_sha256": parameters[0]["batch_payload_sha256"],
                        "batch_source_indices_sha256": batch_source_indices_sha256,
                        "best_checkpoint_sha256": checkpoint_hash,
                        "source_selection_row_sha256": selection_hash,
                        **displacement,
                        **residual,
                    }
                    for precision in PRECISIONS:
                        cosines = [
                            _optional_float(row[f"{precision}_eqprop_vs_bptt_cosine"])
                            for row in parameters
                        ]
                        norm_deltas = [
                            float(row[f"{precision}_eqprop_vs_bptt_symmetric_norm_delta"])
                            for row in parameters
                        ]
                        numeric = [value for value in cosines if value is not None]
                        minimum = min(numeric) if len(numeric) == len(cosines) else None
                        maximum_norm_delta = max(norm_deltas)
                        context[f"minimum_{precision}_eqprop_vs_bptt_cosine"] = minimum
                        context[f"maximum_{precision}_eqprop_vs_bptt_symmetric_norm_delta"] = maximum_norm_delta
                        context[f"all_layer_{precision}_gate_passed"] = bool(
                            minimum is not None
                            and minimum >= COSINE_MINIMUM
                            and maximum_norm_delta <= SYMMETRIC_NORM_DELTA_MAXIMUM
                            and residual["all_residual_gates_passed"]
                        )
                        for row, cosine, norm_delta in zip(parameters, cosines, norm_deltas):
                            layer_rows.append(
                                {
                                    "architecture": architecture,
                                    "scheme": scheme,
                                    "eqprop_variant": variant,
                                    "precision": precision,
                                    "injected_beta": beta,
                                    "actual_base_beta": expected_base,
                                    "amplification_factor": factor,
                                    "parameter_name": row["parameter_name"],
                                    "eqprop_vs_bptt_cosine": cosine,
                                    "eqprop_over_bptt_norm_ratio": float(
                                        row[f"{precision}_eqprop_over_bptt_norm_ratio"]
                                    ),
                                    "eqprop_vs_bptt_symmetric_norm_delta": norm_delta,
                                    "eqprop_exact_zero_fraction": float(
                                        row[f"{precision}_exact_zero_fraction"]
                                    ),
                                    "residual_limited": _boolean(row["residual_limited"]),
                                    "run_id": run_id,
                                    "input_result_sha256": reporting.sha256_file(run_dir / "result.json"),
                                }
                            )
                    context_rows.append(context)
                    included_runs.append(
                        {
                            "run_id": run_id,
                            "path": str(run_dir),
                            "result_sha256": reporting.sha256_file(run_dir / "result.json"),
                            "manifest_study_id": result["study_id"],
                        }
                    )

    if len(cohort_payloads) != 1 or len(source_indices) != 1:
        raise ValueError("Production bundles do not share one fixed source cohort.")
    if any(len(values) != 1 for values in checkpoint_hashes.values()):
        raise ValueError("Checkpoint source identity changed within a scheme surface.")

    context_rows.sort(
        key=lambda row: (
            ARCHITECTURES.index(str(row["architecture"])),
            SCHEMES.index(str(row["scheme"])),
            VARIANTS.index(str(row["eqprop_variant"])),
            float(row["injected_beta"]),
        )
    )
    layer_rows.sort(
        key=lambda row: (
            ARCHITECTURES.index(str(row["architecture"])),
            SCHEMES.index(str(row["scheme"])),
            VARIANTS.index(str(row["eqprop_variant"])),
            PRECISIONS.index(str(row["precision"])),
            float(row["injected_beta"]),
            LAYER_NAMES[str(row["architecture"])].index(str(row["parameter_name"])),
        )
    )

    analysis_dir = study_root / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(analysis_dir / "summary.csv", context_rows)
    _write_csv(analysis_dir / "layer_metrics.csv", layer_rows)
    _write_plot(analysis_dir / "cosine_vs_injected_beta.png", context_rows)
    _write_report(analysis_dir / "report.md", context_rows)

    summary = {
        "schema": SCHEMA,
        "study_root": str(study_root),
        "coverage": {
            "production_bundles": len(included_runs),
            "context_rows": len(context_rows),
            "layer_precision_rows": len(layer_rows),
            "architectures": list(ARCHITECTURES),
            "schemes": list(SCHEMES),
            "estimators": list(VARIANTS),
            "precisions": list(PRECISIONS),
            "injected_betas": list(BETAS),
        },
        "guards": {
            "all_bundles_canonical": True,
            "all_source_and_parameter_guards_passed": True,
            "all_cases_tk64": True,
            "one_fixed_batch": True,
            "source_cap_bypass_recorded": True,
            "optimizer_steps_applied": False,
            "official_test_read": False,
            "launcher_exit_code": int(
                (study_root / "production-launch.log.exitcode").read_text().strip()
            ),
        },
        "reporting_deviation": {
            "bundle_manifest_study_ids": sorted(bundle_study_ids),
            "enclosing_study_id": study_root.name,
            "note": "The reusable audit runner retains its source-compatible study ID; the enclosing directory and complete run inventory bind this Conv1/Conv2 extension.",
        },
        "included_runs": included_runs,
        "artifacts": {
            "summary_csv": "analysis/summary.csv",
            "layer_metrics_csv": "analysis/layer_metrics.csv",
            "report": "analysis/report.md",
            "plot": "analysis/cosine_vs_injected_beta.png",
        },
    }
    reporting.atomic_write_json(analysis_dir / "summary.json", summary)
    return summary


def _write_plot(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.2), sharex=True, sharey=True)
    colors = {"baseline": "#1f77b4", "ours": "#d62728"}
    markers = {"centered": "o", "positive_one_sided": "s"}
    linestyles = {"centered": "-", "positive_one_sided": "--"}
    for row_index, architecture in enumerate(ARCHITECTURES):
        for column_index, precision in enumerate(PRECISIONS):
            axis = axes[row_index, column_index]
            for scheme in SCHEMES:
                for variant in VARIANTS:
                    selected = sorted(
                        (
                            row
                            for row in rows
                            if row["architecture"] == architecture
                            and row["scheme"] == scheme
                            and row["eqprop_variant"] == variant
                        ),
                        key=lambda row: float(row["injected_beta"]),
                    )
                    axis.plot(
                        [row["injected_beta"] for row in selected],
                        [row[f"minimum_{precision}_eqprop_vs_bptt_cosine"] for row in selected],
                        color=colors[scheme],
                        marker=markers[variant],
                        linestyle=linestyles[variant],
                        linewidth=1.6,
                        label=f"{scheme}, {'centered' if variant == 'centered' else 'one-sided'}",
                    )
            axis.axhline(COSINE_MINIMUM, color="0.45", linewidth=1.0, linestyle=":")
            axis.set_xscale("log")
            axis.set_ylim(-1.02, 1.03)
            axis.grid(True, which="both", alpha=0.24)
            axis.set_title(f"{architecture.title()} — {precision}")
            if row_index == 1:
                axis.set_xlabel("injected beta B")
            if column_index == 0:
                axis.set_ylabel("worst-layer EqProp/BPTT cosine")
    axes[0, 0].legend(fontsize=8, loc="lower left")
    fig.suptitle("Conv1/Conv2 higher-beta true-dtype EqProp replay")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _format(value: Any, digits: int = 6) -> str:
    if value is None:
        return "undefined"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.{digits}g}"
    return str(value)


def _write_report(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    lines = [
        "# Conv1/Conv2 higher-beta EqProp replay",
        "",
        "Exploratory ordinary-MNIST gradient diagnostic. These are read-only replays of the accepted BPTT-trained best-validation checkpoints on one fixed 16-example validation batch; they are not EqProp training or paper-facing accuracy evidence.",
        "",
        "All cases use `T=K=64`, frozen-current nudging, native float32 and float64 finite differences, and injected beta `B={100,300,1000}`. Baseline uses base beta `B`; ours uses `B/4` for Conv1 and `B/16` for Conv2.",
        "",
        "| architecture | scheme | estimator | injected B | min f32 cosine | min f64 cosine | max f32 norm delta | max f64 norm delta | f64 output displacement | residual gates |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                (
                    str(row["architecture"]),
                    str(row["scheme"]),
                    "centered" if row["eqprop_variant"] == "centered" else "one-sided",
                    _format(row["injected_beta"]),
                    _format(row["minimum_float32_eqprop_vs_bptt_cosine"]),
                    _format(row["minimum_float64_eqprop_vs_bptt_cosine"]),
                    _format(row["maximum_float32_eqprop_vs_bptt_symmetric_norm_delta"]),
                    _format(row["maximum_float64_eqprop_vs_bptt_symmetric_norm_delta"]),
                    _format(row["float64_positive_output_relative_displacement"]),
                    "pass" if row["all_residual_gates_passed"] else "limited",
                )
            )
            + " |"
        )
    lines.extend(
        (
            "",
            "The all-layer diagnostic gate is cosine `>=.99`, symmetric norm delta `<=.1`, and passing equilibrium residuals. Scientific gate failure is a measured outcome, not an operational bundle failure.",
            "",
            "All 24 production bundles validate canonically. Source files, checkpoints, in-memory parameters, cohort, phase starts, frozen force, iteration counts, dtypes, and native arithmetic guards pass. No optimizer was constructed or stepped and the official test split was not read. Betas above the source `.01` injected-current cap are deliberate diagnostic extrapolations, not proposed training settings.",
            "",
        )
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-root", type=Path, default=DEFAULT_STUDY_ROOT)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    summary = analyze(args.study_root)
    print(json.dumps(summary["coverage"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
