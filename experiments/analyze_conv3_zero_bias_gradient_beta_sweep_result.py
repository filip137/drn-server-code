#!/usr/bin/env python3
"""Analyze the Conv3 all-layer EqProp/BPTT beta sweep with read noise."""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
import math
from pathlib import Path
from statistics import median
import sys
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.reporting import sha256_file, validate_run  # noqa: E402


DEFAULT_CONFIG = (
    ROOT
    / "configs/conv/perfectdiode_conv3_zero_bias_adam_eqprop_bptt_gradient_beta_sweep_one_decade_to_max_sigma_5em4_seed0_20260817_v1.json"
)
DEFAULT_STUDY_ROOT = (
    ROOT
    / "results/perfectdiode-conv3-zero-bias-adam-centered-float64-eqprop-bptt-gradient-beta-sweep-one-decade-to-max-sigma-5em4-seed0-20260817-v1"
)
SCHEMES = ("baseline", "ours", "legacy")
ROLES = ("reconstructed_initialization", "best_validation")
PARAMETERS = ("ConvWeight_0", "ConvWeight_1", "ConvWeight_2", "DenseWeight_0")
LAYER_LABELS = {
    "ConvWeight_0": "C0",
    "ConvWeight_1": "C1",
    "ConvWeight_2": "C2",
    "DenseWeight_0": "Dense",
}
LAYER_COLORS = {
    "ConvWeight_0": "#1f77b4",
    "ConvWeight_1": "#ff7f0e",
    "ConvWeight_2": "#2ca02c",
    "DenseWeight_0": "#9467bd",
}
ROLE_LABELS = {
    "reconstructed_initialization": "initialization",
    "best_validation": "best checkpoint",
}


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    return value


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return [dict(row) for row in csv.DictReader(stream)]


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty CSV {path}.")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _bool(value: Any) -> bool:
    normalized = str(value).strip().lower()
    if normalized in {"true", "1"}:
        return True
    if normalized in {"false", "0"}:
        return False
    raise ValueError(f"Expected boolean, got {value!r}.")


def _sweep_points(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    points = [dict(row) for row in config["sweep_contract"]["beta_scale_points"]]
    if len(points) != 9:
        raise ValueError("The declared Conv3 sweep must contain nine beta points.")
    factors = [float(row["beta_scale"]) for row in points]
    if factors != sorted(factors) or not math.isclose(factors[0], 1.0):
        raise ValueError("Beta-scale factors must be ordered and begin at one.")
    if not math.isclose(factors[-1], 10.0):
        raise ValueError("The beta sweep must end at the original maximum (10x).")
    log_steps = [math.log10(right) - math.log10(left) for left, right in zip(factors, factors[1:])]
    if max(log_steps) - min(log_steps) > 1.0e-12:
        raise ValueError("Beta-scale factors are not uniformly log-spaced.")
    if len({str(row["run_id"]) for row in points}) != len(points):
        raise ValueError("Sweep run IDs must be unique.")
    return points


def _expected_layer_keys() -> set[tuple[str, str, int, str]]:
    return {
        (scheme, role, batch_index, parameter)
        for scheme in SCHEMES
        for role in ROLES
        for batch_index in range(4)
        for parameter in PARAMETERS
    }


def _noise_signature(rows: Sequence[Mapping[str, str]]) -> list[tuple[Any, ...]]:
    keys = (
        "scheme",
        "checkpoint_role",
        "batch_index",
        "phase",
        "state_layer_index",
        "seed",
        "standard_normal_sha256",
    )
    return sorted(tuple(row[key] for key in keys) for row in rows)


def _validate_point(
    *,
    config: Mapping[str, Any],
    config_path: Path,
    study_root: Path,
    point: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any], list[tuple[Any, ...]]]:
    run_dir = study_root / str(point["run_id"])
    errors = validate_run(run_dir)
    if errors:
        raise ValueError(f"Invalid run bundle {run_dir}: {'; '.join(errors)}")
    manifest = _read_json(run_dir / "manifest.json")
    result = _read_json(run_dir / "result.json")
    status = _read_json(run_dir / "status.json")
    if status.get("state") != "complete" or bool(manifest.get("smoke")):
        raise ValueError(f"Expected completed production bundle at {run_dir}.")
    if manifest.get("study_id") != config["study_id"]:
        raise ValueError(f"Study ID mismatch in {run_dir}.")
    declared_scale = float(point["beta_scale"])
    observed_scale = float(
        manifest["configuration"]["execution_overrides"]["beta_scale"]
    )
    if not math.isclose(observed_scale, declared_scale, rel_tol=0.0, abs_tol=1.0e-15):
        raise ValueError(f"Beta-scale mismatch in {run_dir}.")
    if manifest["configuration"]["sha256"] != sha256_file(config_path):
        raise ValueError(f"Config hash mismatch in {run_dir}.")
    completion = result["completion"]
    required_completion = (
        "criteria_met",
        "completed_all_expected_replays",
        "completed_all_expected_layer_comparisons",
        "source_bundles_validated",
        "source_bytes_unchanged",
        "all_bias_learning_rates_exact_zero",
        "all_bias_tensors_exact_zero",
        "all_parameter_and_frozen_force_guards_passed",
        "all_float64_dtype_proofs_passed",
        "all_endpoint_read_noise_guards_passed",
        "matched_endpoint_read_noise_draws_across_schemes",
        "completed_expected_endpoint_read_noise_draws",
        "fixed_cohort_reproduced",
    )
    if not all(bool(completion.get(key)) for key in required_completion):
        raise ValueError(f"Completion guard failure in {run_dir}.")
    terminal = result["terminal_metrics"]
    expected_counts = {
        "replay_count": 24,
        "layer_comparison_count": 96,
        "endpoint_read_noise_draw_count": 192,
    }
    for key, expected in expected_counts.items():
        if int(terminal[key]) != expected:
            raise ValueError(f"Unexpected {key} in {run_dir}: {terminal[key]}.")

    raw_rows = _read_csv(run_dir / "layer_metrics.csv")
    if len(raw_rows) != 96:
        raise ValueError(f"Expected 96 layer rows in {run_dir}, got {len(raw_rows)}.")
    anchors = config["sweep_contract"]["injected_beta_anchors"]
    observed_keys: set[tuple[str, str, int, str]] = set()
    normalized: list[dict[str, Any]] = []
    for row in raw_rows:
        if row["architecture"] != "conv3":
            raise ValueError(f"Non-Conv3 row in {run_dir}.")
        key = (
            row["scheme"],
            row["checkpoint_role"],
            int(row["batch_index"]),
            row["parameter_name"],
        )
        if key in observed_keys:
            raise ValueError(f"Duplicate layer key {key} in {run_dir}.")
        observed_keys.add(key)
        expected_beta = float(anchors[row["scheme"]]) * declared_scale
        if not math.isclose(
            float(row["injected_beta"]),
            expected_beta,
            rel_tol=1.0e-12,
            abs_tol=1.0e-15,
        ):
            raise ValueError(f"Injected beta mismatch for {key} in {run_dir}.")
        if not math.isclose(float(row["endpoint_read_noise_std"]), 5.0e-4):
            raise ValueError(f"Read-noise mismatch for {key} in {run_dir}.")
        numeric_fields = (
            "bptt_l2",
            "cosine",
            "clean_eqprop_bptt_cosine",
            "noisy_clean_eqprop_cosine",
            "symmetric_norm_delta",
        )
        if not all(math.isfinite(float(row[name])) for name in numeric_fields):
            raise ValueError(f"Non-finite gradient metric for {key} in {run_dir}.")
        normalized.append(
            {
                "point_index": int(point["index"]),
                "run_id": str(point["run_id"]),
                "beta_scale": declared_scale,
                **row,
            }
        )
    if observed_keys != _expected_layer_keys():
        raise ValueError(f"Layer coverage mismatch in {run_dir}.")
    noise_rows = _read_csv(run_dir / "endpoint_read_noise.csv")
    if len(noise_rows) != 192:
        raise ValueError(f"Expected 192 endpoint-noise rows in {run_dir}.")
    receipt = {
        "point_index": int(point["index"]),
        "run_id": str(point["run_id"]),
        "beta_scale": declared_scale,
        "result_sha256": sha256_file(run_dir / "result.json"),
        "layer_metrics_sha256": sha256_file(run_dir / "layer_metrics.csv"),
        "endpoint_read_noise_sha256": sha256_file(
            run_dir / "endpoint_read_noise.csv"
        ),
        "replay_count": int(terminal["replay_count"]),
        "layer_comparison_count": int(terminal["layer_comparison_count"]),
        "endpoint_read_noise_draw_count": int(
            terminal["endpoint_read_noise_draw_count"]
        ),
    }
    return normalized, receipt, _noise_signature(noise_rows)


def _bptt_invariance_guard(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    grouped: dict[tuple[str, str, int, str], list[float]] = defaultdict(list)
    for row in rows:
        grouped[
            (
                str(row["scheme"]),
                str(row["checkpoint_role"]),
                int(row["batch_index"]),
                str(row["parameter_name"]),
            )
        ].append(float(row["bptt_l2"]))
    maximum_delta = max(max(values) - min(values) for values in grouped.values())
    if maximum_delta > 1.0e-15:
        raise RuntimeError("BPTT reference norms changed across beta points.")
    return {
        "coordinate_count": len(grouped),
        "maximum_absolute_bptt_l2_delta": maximum_delta,
        "passed": True,
    }


def _aggregate(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[
            (
                int(row["point_index"]),
                str(row["scheme"]),
                str(row["checkpoint_role"]),
                str(row["parameter_name"]),
            )
        ].append(row)
    output: list[dict[str, Any]] = []
    for point_index in sorted({key[0] for key in grouped}):
        for scheme in SCHEMES:
            for role in ROLES:
                for parameter in PARAMETERS:
                    values = grouped[(point_index, scheme, role, parameter)]
                    if len(values) != 4:
                        raise ValueError(
                            f"Expected four batches for {point_index}/{scheme}/{role}/{parameter}."
                        )
                    noisy = [float(row["cosine"]) for row in values]
                    clean = [float(row["clean_eqprop_bptt_cosine"]) for row in values]
                    acquisition = [
                        float(row["noisy_clean_eqprop_cosine"]) for row in values
                    ]
                    output.append(
                        {
                            "point_index": point_index,
                            "run_id": values[0]["run_id"],
                            "beta_scale": float(values[0]["beta_scale"]),
                            "scheme": scheme,
                            "checkpoint_role": role,
                            "parameter_name": parameter,
                            "layer": LAYER_LABELS[parameter],
                            "base_beta": float(values[0]["base_beta"]),
                            "injected_beta": float(values[0]["injected_beta"]),
                            "batch_count": len(values),
                            "noisy_cosine_minimum": min(noisy),
                            "noisy_cosine_median": median(noisy),
                            "noisy_cosine_maximum": max(noisy),
                            "clean_cosine_minimum": min(clean),
                            "clean_cosine_median": median(clean),
                            "clean_cosine_maximum": max(clean),
                            "noisy_clean_cosine_minimum": min(acquisition),
                            "noisy_clean_cosine_median": median(acquisition),
                            "maximum_symmetric_norm_delta": max(
                                float(row["symmetric_norm_delta"]) for row in values
                            ),
                            "cosine_passed_batches": sum(value >= 0.99 for value in noisy),
                            "full_gradient_gate_passed_batches": sum(
                                _bool(row["gradient_fidelity_passed"])
                                for row in values
                            ),
                        }
                    )
    return output


def _threshold_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for scheme in SCHEMES:
        for role in ROLES:
            for parameter in PARAMETERS:
                values = sorted(
                    (
                        row
                        for row in rows
                        if row["scheme"] == scheme
                        and row["checkpoint_role"] == role
                        and row["parameter_name"] == parameter
                    ),
                    key=lambda row: float(row["injected_beta"]),
                )
                passing = [row for row in values if float(row["noisy_cosine_minimum"]) >= 0.99]
                best = max(values, key=lambda row: float(row["noisy_cosine_minimum"]))
                output.append(
                    {
                        "scheme": scheme,
                        "checkpoint_role": role,
                        "parameter_name": parameter,
                        "layer": LAYER_LABELS[parameter],
                        "first_all_batch_cosine_ge_0p99_beta": (
                            float(passing[0]["injected_beta"]) if passing else None
                        ),
                        "best_observed_beta": float(best["injected_beta"]),
                        "best_observed_minimum_noisy_cosine": float(
                            best["noisy_cosine_minimum"]
                        ),
                        "best_observed_median_noisy_cosine": float(
                            best["noisy_cosine_median"]
                        ),
                        "clean_minimum_at_best_observed_beta": float(
                            best["clean_cosine_minimum"]
                        ),
                    }
                )
    return output


def _context_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for scheme in SCHEMES:
        for role in ROLES:
            points = sorted(
                {float(row["injected_beta"]) for row in rows if row["scheme"] == scheme}
            )
            values = []
            for beta in points:
                selected = [
                    row
                    for row in rows
                    if row["scheme"] == scheme
                    and row["checkpoint_role"] == role
                    and math.isclose(float(row["injected_beta"]), beta)
                ]
                values.append(
                    {
                        "injected_beta": beta,
                        "minimum_noisy_cosine": min(
                            float(row["noisy_cosine_minimum"]) for row in selected
                        ),
                        "minimum_clean_cosine": min(
                            float(row["clean_cosine_minimum"]) for row in selected
                        ),
                    }
                )
            passing = [row for row in values if row["minimum_noisy_cosine"] >= 0.99]
            best = max(values, key=lambda row: row["minimum_noisy_cosine"])
            output.append(
                {
                    "scheme": scheme,
                    "checkpoint_role": role,
                    "first_all_layer_all_batch_cosine_ge_0p99_beta": (
                        passing[0]["injected_beta"] if passing else None
                    ),
                    "best_observed_beta": best["injected_beta"],
                    "best_observed_worst_noisy_cosine": best["minimum_noisy_cosine"],
                    "clean_worst_cosine_at_best_observed_beta": best[
                        "minimum_clean_cosine"
                    ],
                }
            )
    return output


def _plot_layerwise(
    rows: Sequence[Mapping[str, Any]], *, path: Path, worst_only: bool
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(16.2, 9.2), sharey=True)
    all_noisy = [float(row["noisy_cosine_minimum"]) for row in rows]
    lower = min(-0.05, math.floor(min(all_noisy) * 10.0) / 10.0)
    lower = max(lower, -1.0)
    for row_index, role in enumerate(ROLES):
        for column_index, scheme in enumerate(SCHEMES):
            axis = axes[row_index][column_index]
            for parameter in PARAMETERS:
                values = sorted(
                    (
                        row
                        for row in rows
                        if row["scheme"] == scheme
                        and row["checkpoint_role"] == role
                        and row["parameter_name"] == parameter
                    ),
                    key=lambda row: float(row["injected_beta"]),
                )
                x = [float(row["injected_beta"]) for row in values]
                color = LAYER_COLORS[parameter]
                if worst_only:
                    noisy = [float(row["noisy_cosine_minimum"]) for row in values]
                    clean = [float(row["clean_cosine_minimum"]) for row in values]
                else:
                    noisy = [float(row["noisy_cosine_median"]) for row in values]
                    clean = [float(row["clean_cosine_median"]) for row in values]
                    axis.fill_between(
                        x,
                        [float(row["noisy_cosine_minimum"]) for row in values],
                        [float(row["noisy_cosine_maximum"]) for row in values],
                        color=color,
                        alpha=0.10,
                        linewidth=0,
                    )
                axis.plot(x, noisy, color=color, marker="o", linewidth=1.8)
                axis.plot(x, clean, color=color, linestyle="--", linewidth=1.0, alpha=0.75)
            axis.axhline(0.99, color="black", linestyle=":", linewidth=1.0)
            axis.set_xscale("log")
            axis.set_ylim(lower, 1.015)
            axis.grid(True, which="both", alpha=0.22)
            axis.set_title(f"{scheme} — {ROLE_LABELS[role]}")
            axis.set_xlabel("injected beta magnitude B")
            if column_index == 0:
                label = "worst-batch cosine" if worst_only else "median cosine"
                axis.set_ylabel(f"EqProp vs BPTT {label}")
    layer_handles = [
        Line2D([0], [0], color=LAYER_COLORS[name], linewidth=2, label=LAYER_LABELS[name])
        for name in PARAMETERS
    ]
    style_handles = [
        Line2D([0], [0], color="black", marker="o", linewidth=1.8, label="noisy EqProp"),
        Line2D([0], [0], color="black", linestyle="--", linewidth=1.0, label="clean EqProp"),
        Line2D([0], [0], color="black", linestyle=":", linewidth=1.0, label="cosine = 0.99"),
    ]
    fig.legend(
        handles=layer_handles + style_handles,
        loc="lower center",
        ncol=7,
        frameon=False,
        bbox_to_anchor=(0.5, 0.01),
    )
    statistic = "worst of four batches" if worst_only else "median with noisy min–max band"
    fig.suptitle(
        "Conv3 all-layer centered EqProp–BPTT cosine vs beta\n"
        f"endpoint read noise sigma = 5e-4; {statistic}; T=K=8",
        fontsize=14,
    )
    fig.tight_layout(rect=(0, 0.07, 1, 0.93))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _format_beta(value: Any) -> str:
    if value is None:
        return "none"
    return f"{float(value):.6g}"


def _write_report(
    *,
    path: Path,
    config: Mapping[str, Any],
    context_rows: Sequence[Mapping[str, Any]],
    threshold_rows: Sequence[Mapping[str, Any]],
    receipts: Sequence[Mapping[str, Any]],
    bptt_guard: Mapping[str, Any],
    noise_guard: Mapping[str, Any],
) -> None:
    lines = [
        "# Conv3 all-layer cosine versus beta under endpoint read noise",
        "",
        "## Outcome",
        "",
        "This exploratory read-only replay sweeps nine logarithmic beta points from the accepted one-decade tier through the original maximum for baseline, ours, and legacy. Both reconstructed initialization and best-validation checkpoints are measured on the same four validation minibatches. Solid curves are noisy centered EqProp versus exact same-T/K BPTT; dashed curves are the embedded clean control.",
        "",
        "### All-layer worst-batch result",
        "",
        "| Scheme | Checkpoint | First beta with all layers/batches cosine >= .99 | Best observed beta | Best observed worst cosine | Clean worst cosine there |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in context_rows:
        lines.append(
            "| {scheme} | {role} | {first} | {best_beta} | {best_cos:.6f} | {clean:.6f} |".format(
                scheme=row["scheme"],
                role=ROLE_LABELS[str(row["checkpoint_role"])],
                first=_format_beta(row["first_all_layer_all_batch_cosine_ge_0p99_beta"]),
                best_beta=_format_beta(row["best_observed_beta"]),
                best_cos=float(row["best_observed_worst_noisy_cosine"]),
                clean=float(row["clean_worst_cosine_at_best_observed_beta"]),
            )
        )
    lines.extend(
        [
            "",
            "### Per-layer threshold",
            "",
            "| Scheme | Checkpoint | Layer | First all-batch beta >= .99 | Best beta | Best minimum cosine |",
            "|---|---|---|---:|---:|---:|",
        ]
    )
    for row in threshold_rows:
        lines.append(
            "| {scheme} | {role} | {layer} | {first} | {best_beta} | {best_cos:.6f} |".format(
                scheme=row["scheme"],
                role=ROLE_LABELS[str(row["checkpoint_role"])],
                layer=row["layer"],
                first=_format_beta(row["first_all_batch_cosine_ge_0p99_beta"]),
                best_beta=_format_beta(row["best_observed_beta"]),
                best_cos=float(row["best_observed_minimum_noisy_cosine"]),
            )
        )
    lines.extend(
        [
            "",
            "## Contract and integrity",
            "",
            f"- Coverage: `{len(receipts)}/9` valid beta bundles, `{len(receipts) * 24}` checkpoint-batch replays, `{len(receipts) * 96}` layer-batch comparisons, and `{len(receipts) * 192}` endpoint-noise records.",
            "- Injected beta ranges: baseline `100--1000`, ours `3--30`, legacy `.001--.01`.",
            "- Shared contract: Conv3, exact-zero biases, `T=K=8`, centered frozen-current true-float64 EqProp, exact same-T/K BPTT reference, four fixed 16-example validation batches, endpoint read-noise sigma `5e-4`, and no optimizer step or official-test read.",
            f"- BPTT invariance guard: passed over `{bptt_guard['coordinate_count']}` coordinates; maximum L2-norm delta `{bptt_guard['maximum_absolute_bptt_l2_delta']:.3g}`.",
            f"- Cross-beta noise-draw guard: `{noise_guard['matched_signature_count']}/9` point signatures match exactly.",
            "- The trained Conv3-baseline free-state residual caveat at `T=8` remains separate from the cosine curves.",
            "",
            "## Artifacts",
            "",
            "- `layerwise_cosine_vs_beta.csv`: four-batch layer summaries at every beta.",
            "- `per_layer_thresholds.csv`: first `.99` crossing and best observed point for every layer.",
            "- `context_thresholds.csv`: all-layer worst-batch result by scheme/checkpoint.",
            "- `plots/median_cosine_vs_injected_beta_all_layers.png`: median curves with noisy batch spread.",
            "- `plots/worst_batch_cosine_vs_injected_beta_all_layers.png`: conservative worst-batch curves.",
            "- `summary.json`: run hashes and all integrity guards.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    config_path = args.config.expanduser().resolve()
    study_root = args.study_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    config = _read_json(config_path)
    if config.get("study_id") != study_root.name:
        raise ValueError("Config study ID and result-directory name differ.")
    points = _sweep_points(config)
    all_rows: list[dict[str, Any]] = []
    receipts: list[dict[str, Any]] = []
    signatures: list[list[tuple[Any, ...]]] = []
    for point in points:
        rows, receipt, signature = _validate_point(
            config=config,
            config_path=config_path,
            study_root=study_root,
            point=point,
        )
        all_rows.extend(rows)
        receipts.append(receipt)
        signatures.append(signature)
    matched_signatures = sum(signature == signatures[0] for signature in signatures)
    if matched_signatures != len(signatures):
        raise RuntimeError("Endpoint-noise draws changed across beta points.")
    noise_guard = {
        "point_count": len(signatures),
        "matched_signature_count": matched_signatures,
        "signature_row_count": len(signatures[0]),
        "passed": True,
    }
    bptt_guard = _bptt_invariance_guard(all_rows)
    layer_rows = _aggregate(all_rows)
    threshold_rows = _threshold_rows(layer_rows)
    context_rows = _context_rows(layer_rows)

    _write_csv(output_dir / "layer_metrics_all_points.csv", all_rows)
    _write_csv(output_dir / "layerwise_cosine_vs_beta.csv", layer_rows)
    _write_csv(output_dir / "per_layer_thresholds.csv", threshold_rows)
    _write_csv(output_dir / "context_thresholds.csv", context_rows)
    _plot_layerwise(
        layer_rows,
        path=output_dir / "plots/median_cosine_vs_injected_beta_all_layers.png",
        worst_only=False,
    )
    _plot_layerwise(
        layer_rows,
        path=output_dir / "plots/worst_batch_cosine_vs_injected_beta_all_layers.png",
        worst_only=True,
    )
    summary = {
        "schema_version": "conv3-zero-bias-gradient-beta-sweep-analysis/v1",
        "study_id": config["study_id"],
        "config_path": str(config_path),
        "config_sha256": sha256_file(config_path),
        "point_count": len(points),
        "replay_count": len(points) * 24,
        "layer_comparison_count": len(all_rows),
        "endpoint_noise_record_count": len(points) * 192,
        "run_receipts": receipts,
        "bptt_invariance_guard": bptt_guard,
        "cross_beta_noise_draw_guard": noise_guard,
        "context_thresholds": context_rows,
        "per_layer_thresholds": threshold_rows,
        "official_test_read": False,
        "optimizer_steps_applied": False,
    }
    _write_json(output_dir / "summary.json", summary)
    _write_report(
        path=output_dir / "report.md",
        config=config,
        context_rows=context_rows,
        threshold_rows=threshold_rows,
        receipts=receipts,
        bptt_guard=bptt_guard,
        noise_guard=noise_guard,
    )
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--study-root", type=Path, default=DEFAULT_STUDY_ROOT)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    if args.output_dir is None:
        args.output_dir = args.study_root / "analysis"
    return args


def main(argv: Sequence[str] | None = None) -> int:
    summary = run(parse_args(argv))
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
