#!/usr/bin/env python3
"""Collate Conv3 float64 state and local-learning observables over beta.

The input arms are completed ``audit_eqprop_float64_shadow`` bundles.  The
production convention is one ``beta-NN`` bundle per effective/injected beta in
the study config; each bundle contains matched baseline and ours replays.  A
``--run-glob`` subset is accepted for smoke analysis, while the default path
fails closed unless the complete configured beta grid is present.

This is a read-only exploratory ordinary-MNIST diagnostic.  It reports the
matched-K displacement from the beta=0 endpoint to the positive endpoint and
the exact local energy statistic q=dE/dw at both endpoints.  The archived
float64 EqProp gradient must be bitwise identical to ``(q_plus-q_zero)/B``,
where B is the amplification-scaled effective beta.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from experiments import reporting  # noqa: E402


DEFAULT_CONFIG = (
    REPOSITORY_ROOT
    / "configs/conv/perfectdiode_conv3_baseline_ours_float64_beta_observables_20260811_v1.json"
)
CONFIG_SCHEMA = "perfectdiode-conv3-float64-beta-observables/v1"
OUTPUT_SCHEMA = "perfectdiode-conv3-float64-beta-observables-analysis/v1"
SCHEMES = ("baseline", "ours")
PARAMETERS = ("ConvWeight_0", "ConvWeight_1", "ConvWeight_2", "DenseWeight_0")
LAYER_BY_PARAMETER = {
    "ConvWeight_0": "C0",
    "ConvWeight_1": "C1",
    "ConvWeight_2": "C2",
    "DenseWeight_0": "D",
}
STATE_LAYER_BY_PARAMETER = {
    "ConvWeight_0": "Layer_1",
    "ConvWeight_1": "Layer_2",
    "ConvWeight_2": "Layer_3",
    "DenseWeight_0": "Layer_4",
}
PARAMETER_BY_STATE_LAYER = {
    state_layer: parameter
    for parameter, state_layer in STATE_LAYER_BY_PARAMETER.items()
}

# q is the exact code-path endpoint statistic.  For these layers,
# q = alpha/2 * sum_spatial(drop**2), after the runner's minibatch mean.
ALPHA_BY_SCHEME_LAYER = {
    "baseline": {"C0": 1.0, "C1": 1.0, "C2": 1.0, "D": 1.0},
    "ours": {"C0": 1.0, "C1": 0.25, "C2": 0.0625, "D": 0.015625},
}
SPATIAL_COUNT_BY_LAYER = {"C0": 196, "C1": 49, "C2": 49, "D": 1}
LAYER_ORDER = {layer: index for index, layer in enumerate(("C0", "C1", "C2", "D"))}

SCALING_REQUIRED = {
    "architecture",
    "scheme",
    "checkpoint_role",
    "eqprop_variant",
    "actual_beta",
    "voltage_amplification",
    "current_amplification",
    "amplification_factor",
    "effective_beta",
}
DISPLACEMENT_REQUIRED = {
    "architecture",
    "scheme",
    "checkpoint_role",
    "precision",
    "eqprop_variant",
    "phase",
    "actual_beta",
    "reference_kind",
    "outcome",
    "state_layer_name",
    "element_count",
    "displacement_l2",
    "reference_l2",
    "relative_displacement",
    "delta_rms",
    "reference_rms",
    "max_abs_delta",
    "active_set_transition_fraction",
}
PRECISION_REQUIRED = {
    "architecture",
    "scheme",
    "checkpoint_role",
    "eqprop_variant",
    "parameter_name",
    "actual_beta",
    "effective_beta",
    "float64_eqprop_vs_bptt_cosine",
    "float64_eqprop_vs_bptt_symmetric_norm_delta",
}

LONG_FIELDS = (
    "schema",
    "beta_index",
    "run_id",
    "input_result_sha256",
    "scheme",
    "layer",
    "parameter_name",
    "state_layer_name",
    "effective_beta_B",
    "base_beta",
    "amplification_factor",
    "alpha",
    "spatial_count",
    "state_element_count",
    "state_reference_rms",
    "state_displacement_rms",
    "state_relative_displacement",
    "state_displacement_l2",
    "state_reference_l2",
    "state_max_abs_displacement",
    "state_active_set_transition_fraction",
    "q_element_count",
    "q_zero_mean",
    "q_zero_rms",
    "q_plus_mean",
    "q_plus_rms",
    "delta_q_mean",
    "delta_q_rms",
    "delta_q_rms_over_q_zero_rms",
    "elementwise_delta_q_over_q_zero_rms",
    "eqprop_g_mean",
    "eqprop_g_rms",
    "float64_eqprop_vs_bptt_cosine",
    "float64_eqprop_vs_bptt_symmetric_norm_delta",
    "raw_drop2_zero_mean",
    "raw_drop2_zero_rms",
    "raw_drop2_plus_mean",
    "raw_drop2_plus_rms",
    "raw_delta_drop2_mean",
    "raw_delta_drop2_rms",
    "archive_eqprop_exact",
)

STATE_WIDE_METRICS = (
    "state_reference_rms",
    "state_displacement_rms",
    "state_relative_displacement",
    "state_max_abs_displacement",
)
LEARNING_WIDE_METRICS = (
    "q_zero_rms",
    "q_plus_rms",
    "delta_q_rms",
    "elementwise_delta_q_over_q_zero_rms",
    "eqprop_g_rms",
    "raw_drop2_zero_mean",
    "raw_drop2_plus_mean",
    "raw_delta_drop2_rms",
)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path, required: set[str]) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        missing = required.difference(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")
        rows = [dict(row) for row in reader]
    if not rows:
        raise ValueError(f"Scientific artifact is empty: {path}")
    return rows


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finite_float(value: object, *, label: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite, got {value!r}")
    return result


def _positive_float(value: object, *, label: str) -> float:
    result = _finite_float(value, label=label)
    if result <= 0.0:
        raise ValueError(f"{label} must be positive, got {value!r}")
    return result


def _optional_finite_float(value: object, *, label: str) -> float | None:
    if value is None or str(value).strip().lower() in {"", "none", "null", "nan"}:
        return None
    return _finite_float(value, label=label)


def _close(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=1.0e-12, abs_tol=1.0e-18)


def _rms(values: np.ndarray) -> float:
    flat = np.asarray(values, dtype=np.float64).reshape(-1)
    if flat.size == 0 or not np.isfinite(flat).all():
        raise ValueError("Observable tensor must be non-empty and finite.")
    return float(np.sqrt(np.mean(np.square(flat))))


def _mean(values: np.ndarray) -> float:
    flat = np.asarray(values, dtype=np.float64).reshape(-1)
    if flat.size == 0 or not np.isfinite(flat).all():
        raise ValueError("Observable tensor must be non-empty and finite.")
    return float(np.mean(flat))


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(fields), extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def _markdown_table(headers: Sequence[str], rows: Iterable[Sequence[str]]) -> str:
    def clean(value: object) -> str:
        return str(value).replace("|", "\\|").replace("\n", " ")

    lines = [
        "| " + " | ".join(clean(value) for value in headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend(
        "| " + " | ".join(clean(value) for value in row) + " |" for row in rows
    )
    return "\n".join(lines)


def _study_root(config: Mapping[str, Any], explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit.expanduser().resolve()
    configured = Path(str(config["output_root"]))
    if not configured.is_absolute():
        configured = REPOSITORY_ROOT / configured
    return configured.resolve()


def _configured_betas(config: Mapping[str, Any]) -> list[float]:
    values = [
        _positive_float(value, label="configured common effective beta")
        for value in config["beta_contract"]["common_effective_betas"]
    ]
    if not values or any(right <= left for left, right in zip(values, values[1:])):
        raise ValueError("Configured common effective betas must be non-empty and increasing.")
    return values


def _case_contract(config: Mapping[str, Any]) -> dict[str, dict[str, float]]:
    cases: dict[str, dict[str, float]] = {}
    for raw in config["source"]["cases"]:
        scheme = str(raw["scheme"])
        if scheme in cases or scheme not in SCHEMES:
            raise ValueError(f"Unexpected or duplicate scheme in config: {scheme!r}")
        voltage = _positive_float(raw["voltage_amplification"], label=f"{scheme} voltage amplification")
        current = _positive_float(raw["current_amplification"], label=f"{scheme} current amplification")
        cases[scheme] = {
            "voltage_amplification": voltage,
            "current_amplification": current,
            "amplification_factor": (voltage / current) ** 3,
        }
    if set(cases) != set(SCHEMES):
        raise ValueError(f"Config must contain exactly {SCHEMES}; found {sorted(cases)}")
    return cases


def _discover_runs(
    study_root: Path,
    betas: Sequence[float],
    run_globs: Sequence[str],
) -> list[tuple[Path, int | None]]:
    if not run_globs:
        expected = [(study_root / f"beta-{index:02d}", index) for index in range(len(betas))]
        missing = [str(path) for path, _ in expected if not path.is_dir()]
        if missing:
            raise FileNotFoundError(
                "Production analysis requires every configured beta bundle; missing: "
                + ", ".join(missing)
            )
        return expected

    discovered: dict[Path, None] = {}
    for pattern in run_globs:
        matches = sorted(path.resolve() for path in study_root.glob(pattern) if path.is_dir())
        if not matches:
            raise FileNotFoundError(f"--run-glob {pattern!r} matched no directories in {study_root}")
        discovered.update((path, None) for path in matches)
    return [(path, None) for path in discovered]


def _beta_index(value: float, betas: Sequence[float]) -> int:
    matches = [index for index, expected in enumerate(betas) if _close(value, expected)]
    if len(matches) != 1:
        raise ValueError(
            f"Observed effective beta {value:.17g} does not uniquely match the configured grid."
        )
    return matches[0]


def _validate_bundle(run_dir: Path, *, production: bool) -> tuple[dict[str, Any], str]:
    errors = reporting.validate_run(run_dir)
    if errors:
        raise ValueError(f"Invalid completed bundle {run_dir}: {'; '.join(errors)}")
    status = _read_json(run_dir / "status.json")
    result = _read_json(run_dir / "result.json")
    manifest = _read_json(run_dir / "manifest.json")
    if status.get("state") != "complete" or not bool(result.get("completion", {}).get("criteria_met")):
        raise ValueError(f"Bundle is not scientifically complete: {run_dir}")
    if production and (bool(result.get("smoke")) or bool(manifest.get("smoke"))):
        raise ValueError(f"Production beta grid cannot include a smoke bundle: {run_dir}")
    if result.get("terminal_metrics", {}).get("eqprop_variant") != "positive_one_sided":
        raise ValueError(f"Expected positive one-sided EqProp in {run_dir}")
    if int(result.get("terminal_metrics", {}).get("checkpoint_cases", -1)) != len(SCHEMES):
        raise ValueError(f"Expected matched baseline/ours cases in {run_dir}")
    return result, _sha256(run_dir / "result.json")


def _scaling_by_scheme(
    run_dir: Path,
    *,
    cases: Mapping[str, Mapping[str, float]],
) -> dict[str, dict[str, Any]]:
    rows = _read_csv(run_dir / "beta_scaling.csv", SCALING_REQUIRED)
    selected = [
        row
        for row in rows
        if row["architecture"] == "conv3"
        and row["checkpoint_role"] == "best_validation"
        and row["eqprop_variant"] == "positive_one_sided"
    ]
    if len(selected) != len(SCHEMES) or {row["scheme"] for row in selected} != set(SCHEMES):
        raise ValueError(f"Expected exactly one Conv3 scaling row per scheme in {run_dir}")
    output: dict[str, dict[str, Any]] = {}
    for row in selected:
        scheme = row["scheme"]
        contract = cases[scheme]
        base_beta = _positive_float(row["actual_beta"], label=f"{run_dir} {scheme} base beta")
        effective_beta = _positive_float(
            row["effective_beta"], label=f"{run_dir} {scheme} effective beta"
        )
        voltage = _positive_float(row["voltage_amplification"], label="voltage amplification")
        current = _positive_float(row["current_amplification"], label="current amplification")
        factor = _positive_float(row["amplification_factor"], label="amplification factor")
        expected_factor = float(contract["amplification_factor"])
        if not all(
            (
                _close(voltage, float(contract["voltage_amplification"])),
                _close(current, float(contract["current_amplification"])),
                _close(factor, expected_factor),
                _close(effective_beta, base_beta * expected_factor),
            )
        ):
            raise ValueError(f"Amplification-scaled beta contract failed for {scheme} in {run_dir}")
        output[scheme] = {
            **row,
            "base_beta_value": base_beta,
            "effective_beta_value": effective_beta,
            "amplification_factor_value": expected_factor,
        }
    if not _close(output["baseline"]["effective_beta_value"], output["ours"]["effective_beta_value"]):
        raise ValueError(f"Baseline and ours do not share one effective beta in {run_dir}")
    return output


def _displacement_by_scheme_layer(run_dir: Path) -> dict[tuple[str, str], dict[str, str]]:
    rows = _read_csv(run_dir / "state_displacement.csv", DISPLACEMENT_REQUIRED)
    selected = [
        row
        for row in rows
        if row["architecture"] == "conv3"
        and row["checkpoint_role"] == "best_validation"
        and row["precision"] == "float64"
        and row["eqprop_variant"] == "positive_one_sided"
        and row["phase"] == "positive"
        and row["reference_kind"] == "matched_zero_K"
        and row["state_layer_name"] in PARAMETER_BY_STATE_LAYER
    ]
    expected = {
        (scheme, state_layer)
        for scheme in SCHEMES
        for state_layer in PARAMETER_BY_STATE_LAYER
    }
    keyed = {(row["scheme"], row["state_layer_name"]): row for row in selected}
    if len(selected) != len(expected) or set(keyed) != expected:
        raise ValueError(f"Incomplete or duplicate float64 matched-zero displacement rows in {run_dir}")
    if any(row["outcome"] != "ok" for row in selected):
        raise ValueError(f"Non-ok float64 displacement outcome in {run_dir}")
    return keyed


def _precision_by_scheme_parameter(run_dir: Path) -> dict[tuple[str, str], dict[str, str]]:
    rows = _read_csv(run_dir / "parameter_precision_comparison.csv", PRECISION_REQUIRED)
    selected = [
        row
        for row in rows
        if row["architecture"] == "conv3"
        and row["checkpoint_role"] == "best_validation"
        and row["eqprop_variant"] == "positive_one_sided"
        and row["parameter_name"] in PARAMETERS
    ]
    expected = {(scheme, parameter) for scheme in SCHEMES for parameter in PARAMETERS}
    keyed = {(row["scheme"], row["parameter_name"]): row for row in selected}
    if len(selected) != len(expected) or set(keyed) != expected:
        raise ValueError(f"Incomplete or duplicate float64 EqProp/BPTT rows in {run_dir}")
    return keyed


def _endpoint_archive(run_dir: Path, scheme: str) -> Path:
    path = run_dir / "artifacts/eqprop_gradients" / f"conv3__{scheme}__best_validation.npz"
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _endpoint_rows(
    run_dir: Path,
    *,
    scheme: str,
    scaling: Mapping[str, Any],
    displacement: Mapping[tuple[str, str], Mapping[str, str]],
    precision: Mapping[tuple[str, str], Mapping[str, str]],
    beta_index: int,
    result_sha256: str,
) -> list[dict[str, Any]]:
    effective_beta = float(scaling["effective_beta_value"])
    base_beta = float(scaling["base_beta_value"])
    amplification_factor = float(scaling["amplification_factor_value"])
    archive = _endpoint_archive(run_dir, scheme)
    output: list[dict[str, Any]] = []
    with np.load(archive, allow_pickle=False) as values:
        metadata = json.loads(str(values["metadata_json"].item()))
        if (
            metadata.get("architecture") != "conv3"
            or metadata.get("scheme") != scheme
            or metadata.get("checkpoint_role") != "best_validation"
            or metadata.get("eqprop_variant") != "positive_one_sided"
            or not _close(float(metadata["actual_beta"]), base_beta)
            or not _close(float(metadata["effective_beta"]), effective_beta)
        ):
            raise ValueError(f"Endpoint archive metadata disagrees with scaling CSV: {archive}")
        for parameter in PARAMETERS:
            layer = LAYER_BY_PARAMETER[parameter]
            state_layer = STATE_LAYER_BY_PARAMETER[parameter]
            q_zero = values[f"endpoint_zero_float64__{parameter}"]
            q_plus = values[f"endpoint_positive_float64__{parameter}"]
            archived_g = values[f"eqprop_float64__{parameter}"]
            if q_zero.dtype != np.float64 or q_plus.dtype != np.float64 or archived_g.dtype != np.float64:
                raise ValueError(f"Endpoint and EqProp arrays must be float64 in {archive}: {parameter}")
            if q_zero.shape != q_plus.shape or q_zero.shape != archived_g.shape:
                raise ValueError(f"Endpoint and EqProp shapes differ in {archive}: {parameter}")
            if not (np.isfinite(q_zero).all() and np.isfinite(q_plus).all() and np.isfinite(archived_g).all()):
                raise ValueError(f"Non-finite endpoint observable in {archive}: {parameter}")

            delta_q = q_plus - q_zero
            computed_g = delta_q / effective_beta
            archive_exact = bool(np.array_equal(computed_g, archived_g))
            if not archive_exact:
                max_error = float(np.max(np.abs(computed_g - archived_g)))
                raise ValueError(
                    f"Archived g is not exactly (q_plus-q_zero)/B for {archive} {parameter}; "
                    f"maximum absolute error={max_error:.17g}"
                )
            q_zero_rms = _rms(q_zero)
            delta_q_rms = _rms(delta_q)
            if q_zero_rms == 0.0:
                raise ValueError(f"q_zero RMS is zero in {archive}: {parameter}")
            if np.any(q_zero == 0.0):
                raise ValueError(
                    f"Elementwise delta_q/q_zero RMS is undefined because q_zero contains zero: "
                    f"{archive} {parameter}"
                )
            elementwise_relative = delta_q / q_zero
            alpha = ALPHA_BY_SCHEME_LAYER[scheme][layer]
            spatial_count = SPATIAL_COUNT_BY_LAYER[layer]
            raw_scale = 2.0 / (alpha * spatial_count)
            drop2_zero = raw_scale * q_zero
            drop2_plus = raw_scale * q_plus
            delta_drop2 = drop2_plus - drop2_zero
            state = displacement[(scheme, state_layer)]
            gradient_comparison = precision[(scheme, parameter)]
            if not _close(float(state["actual_beta"]), base_beta):
                raise ValueError(f"State displacement base beta disagrees in {run_dir}: {scheme} {layer}")
            if not _close(float(gradient_comparison["actual_beta"]), base_beta) or not _close(
                float(gradient_comparison["effective_beta"]), effective_beta
            ):
                raise ValueError(f"EqProp/BPTT beta metadata disagrees in {run_dir}: {scheme} {layer}")
            output.append(
                {
                    "schema": OUTPUT_SCHEMA,
                    "beta_index": beta_index,
                    "run_id": run_dir.name,
                    "input_result_sha256": result_sha256,
                    "scheme": scheme,
                    "layer": layer,
                    "parameter_name": parameter,
                    "state_layer_name": state_layer,
                    "effective_beta_B": effective_beta,
                    "base_beta": base_beta,
                    "amplification_factor": amplification_factor,
                    "alpha": alpha,
                    "spatial_count": spatial_count,
                    "state_element_count": int(state["element_count"]),
                    "state_reference_rms": _positive_float(state["reference_rms"], label="state reference RMS"),
                    "state_displacement_rms": _finite_float(state["delta_rms"], label="state displacement RMS"),
                    "state_relative_displacement": _finite_float(state["relative_displacement"], label="relative displacement"),
                    "state_displacement_l2": _finite_float(state["displacement_l2"], label="state displacement L2"),
                    "state_reference_l2": _positive_float(state["reference_l2"], label="state reference L2"),
                    "state_max_abs_displacement": _finite_float(state["max_abs_delta"], label="maximum state displacement"),
                    "state_active_set_transition_fraction": _finite_float(
                        state["active_set_transition_fraction"], label="active-set transition fraction"
                    ),
                    "q_element_count": int(q_zero.size),
                    "q_zero_mean": _mean(q_zero),
                    "q_zero_rms": q_zero_rms,
                    "q_plus_mean": _mean(q_plus),
                    "q_plus_rms": _rms(q_plus),
                    "delta_q_mean": _mean(delta_q),
                    "delta_q_rms": delta_q_rms,
                    "delta_q_rms_over_q_zero_rms": delta_q_rms / q_zero_rms,
                    "elementwise_delta_q_over_q_zero_rms": _rms(elementwise_relative),
                    "eqprop_g_mean": _mean(archived_g),
                    "eqprop_g_rms": _rms(archived_g),
                    "float64_eqprop_vs_bptt_cosine": _optional_finite_float(
                        gradient_comparison["float64_eqprop_vs_bptt_cosine"],
                        label="float64 EqProp/BPTT cosine",
                    ),
                    "float64_eqprop_vs_bptt_symmetric_norm_delta": _finite_float(
                        gradient_comparison[
                            "float64_eqprop_vs_bptt_symmetric_norm_delta"
                        ],
                        label="float64 EqProp/BPTT symmetric norm delta",
                    ),
                    "raw_drop2_zero_mean": _mean(drop2_zero),
                    "raw_drop2_zero_rms": _rms(drop2_zero),
                    "raw_drop2_plus_mean": _mean(drop2_plus),
                    "raw_drop2_plus_rms": _rms(drop2_plus),
                    "raw_delta_drop2_mean": _mean(delta_drop2),
                    "raw_delta_drop2_rms": _rms(delta_drop2),
                    "archive_eqprop_exact": archive_exact,
                }
            )
    return output


def _wide_rows(
    rows: Sequence[Mapping[str, Any]], metrics: Sequence[str]
) -> tuple[list[dict[str, Any]], list[str]]:
    grouped: dict[tuple[int, str], list[Mapping[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((int(row["beta_index"]), str(row["scheme"])), []).append(row)
    fields = ["beta_index", "effective_beta_B", "scheme", "base_beta"] + [
        f"{layer}_{metric}"
        for layer in ("C0", "C1", "C2", "D")
        for metric in metrics
    ]
    output: list[dict[str, Any]] = []
    for key in sorted(grouped, key=lambda item: (item[0], SCHEMES.index(item[1]))):
        members = grouped[key]
        by_layer = {str(row["layer"]): row for row in members}
        if set(by_layer) != {"C0", "C1", "C2", "D"}:
            raise ValueError(f"Wide-table layer coverage failed for {key}")
        first = members[0]
        wide: dict[str, Any] = {
            "beta_index": key[0],
            "effective_beta_B": first["effective_beta_B"],
            "scheme": key[1],
            "base_beta": first["base_beta"],
        }
        for layer in ("C0", "C1", "C2", "D"):
            for metric in metrics:
                wide[f"{layer}_{metric}"] = by_layer[layer][metric]
        output.append(wide)
    return output, fields


def _format_scientific(value: object) -> str:
    return f"{float(value):.6e}"


def _write_observable_table(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    headers = (
        "B effective",
        "base beta",
        "scheme",
        "layer",
        "RMS delta s",
        "RMS delta s / RMS s0",
        "mean q0",
        "RMS q0",
        "mean q+",
        "RMS q+",
        "RMS delta q",
        "RMS(delta q / q0)",
        "RMS g",
        "mean drop^2 zero",
        "mean drop^2 plus",
    )
    table_rows = [
        (
            _format_scientific(row["effective_beta_B"]),
            _format_scientific(row["base_beta"]),
            str(row["scheme"]),
            str(row["layer"]),
            _format_scientific(row["state_displacement_rms"]),
            _format_scientific(row["state_relative_displacement"]),
            _format_scientific(row["q_zero_mean"]),
            _format_scientific(row["q_zero_rms"]),
            _format_scientific(row["q_plus_mean"]),
            _format_scientific(row["q_plus_rms"]),
            _format_scientific(row["delta_q_rms"]),
            _format_scientific(row["elementwise_delta_q_over_q_zero_rms"]),
            _format_scientific(row["eqprop_g_rms"]),
            _format_scientific(row["raw_drop2_zero_mean"]),
            _format_scientific(row["raw_drop2_plus_mean"]),
        )
        for row in rows
    ]
    lines = [
        "# Conv3 float64 beta-observable table",
        "",
        "`s0` is the matched beta=0 K-step endpoint and `s+` is the positive-nudged K-step endpoint. "
        "`q=dE/dw`, `g=(q+ - q0)/B`, and `drop^2=2q/(alpha*spatial_count)`.",
        "",
        _markdown_table(headers, table_rows),
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _plot(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    colors = {"C0": "#1f77b4", "C1": "#ff7f0e", "C2": "#2ca02c", "D": "#d62728"}
    styles = {"baseline": "-", "ours": "--"}
    panels = (
        ("state_displacement_rms", r"RMS $|s_+-s_0|$"),
        ("state_relative_displacement", r"RMS displacement / RMS $s_0$"),
        ("elementwise_delta_q_over_q_zero_rms", r"RMS $[(q_+-q_0)/q_0]$"),
        ("eqprop_g_rms", r"RMS $|(q_+-q_0)/B|$"),
    )
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.5), constrained_layout=True)
    for ax, (metric, ylabel) in zip(axes.flat, panels, strict=True):
        for scheme in SCHEMES:
            for layer in ("C0", "C1", "C2", "D"):
                selected = sorted(
                    (
                        row
                        for row in rows
                        if row["scheme"] == scheme and row["layer"] == layer
                    ),
                    key=lambda row: int(row["beta_index"]),
                )
                x = np.asarray([float(row["effective_beta_B"]) for row in selected])
                y = np.asarray([float(row[metric]) for row in selected])
                positive = np.isfinite(x) & np.isfinite(y) & (x > 0.0) & (y > 0.0)
                if np.any(positive):
                    ax.loglog(
                        x[positive],
                        y[positive],
                        color=colors[layer],
                        linestyle=styles[scheme],
                        marker="o" if scheme == "baseline" else "s",
                        markersize=3.0,
                        linewidth=1.2,
                        label=f"{scheme} {layer}",
                    )
        ax.set_xlabel(r"effective / injected beta $B$")
        ax.set_ylabel(ylabel)
        ax.grid(True, which="both", alpha=0.25)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=4,
        fontsize=8,
    )
    fig.suptitle("Conv3 float64 free-to-nudged observables", y=1.055)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _write_report(
    path: Path,
    *,
    config_path: Path,
    study_root: Path,
    rows: Sequence[Mapping[str, Any]],
    included_runs: Sequence[Mapping[str, Any]],
    production_complete: bool,
) -> None:
    beta_by_index: dict[int, float] = {}
    for row in rows:
        index = int(row["beta_index"])
        value = float(row["effective_beta_B"])
        if index in beta_by_index and not _close(beta_by_index[index], value):
            raise ValueError(f"Schemes disagree on effective beta at index {index}.")
        beta_by_index[index] = value
    beta_values = [beta_by_index[index] for index in sorted(beta_by_index)]
    lines = [
        "# Conv3 float64 beta-observable sweep",
        "",
        "Evidence tier: exploratory ordinary-MNIST checkpoint replay; not paper-facing training evidence.",
        "",
        f"Included `{len(included_runs)}` completed bundles, `{len(beta_values)}` effective beta values, "
        f"and `{len(rows)}` scheme/layer rows. Full configured production coverage: "
        f"`{str(production_complete).lower()}`.",
        "",
        f"Effective beta range: `{beta_values[0]:.6e}` to `{beta_values[-1]:.6e}`. "
        "Baseline uses `base_beta=B`; ours uses `base_beta=B/64`, because "
        "`B=base_beta*(voltage_amplification/current_amplification)^3` and ours has voltage amplification 4.",
        "",
        "The state displacement is `s_plus-s_zero` against the matched beta=0 K-step endpoint. "
        "The endpoint statistic is the exact code-path `q=dE/dw`. Every archived float64 EqProp tensor "
        "was required to be bitwise equal to `(q_plus-q_zero)/B`.",
        "",
        "For a per-position physical scale, the analysis reports "
        "`drop^2 = 2q/(alpha*spatial_count)`, using spatial counts `196,49,49,1` for `C0,C1,C2,D`; "
        "baseline alpha is one throughout and ours uses `1,1/4,1/16,1/64`.",
        "",
        "Two relative q metrics are retained in the long CSV: "
        "`RMS(delta_q)/RMS(q_zero)` and the requested elementwise `RMS(delta_q/q_zero)`.",
        "",
        "Outputs: `observables_long.csv`, `state_displacement_wide.csv`, `learning_rule_wide.csv`, "
        "`observables_table.md`, `beta_observables_loglog.png`, and `summary.json`.",
        "",
        f"Config: `{config_path}`. Input root: `{study_root}`.",
        "",
        "Limitations: one fixed 16-example ordinary-MNIST minibatch and two trained Conv3 checkpoints are "
        "replayed; no EqProp optimizer step or training-accuracy sweep is performed. Endpoint q includes the "
        "code-path minibatch mean, Conv spatial sum, and amplification prefactor.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    config_path = args.config.expanduser().resolve()
    config = _read_json(config_path)
    if config.get("schema_version") != CONFIG_SCHEMA:
        raise ValueError(f"Unexpected config schema: {config.get('schema_version')!r}")
    if config.get("replay", {}).get("dtype") != "float64":
        raise ValueError("Observable analysis requires the declared float64 replay.")
    if config.get("replay", {}).get("primary_displacement_reference") != "matched_zero_K_endpoint":
        raise ValueError("Config displacement reference must be matched_zero_K_endpoint.")

    betas = _configured_betas(config)
    cases = _case_contract(config)
    if not _close(cases["baseline"]["amplification_factor"], 1.0) or not _close(
        cases["ours"]["amplification_factor"], 64.0
    ):
        raise ValueError("Expected baseline factor 1 and ours factor 64.")
    study_root = _study_root(config, args.study_root)
    production = not bool(args.run_glob)
    run_specs = _discover_runs(study_root, betas, args.run_glob)
    long_rows: list[dict[str, Any]] = []
    included_runs: list[dict[str, Any]] = []
    seen_indices: set[int] = set()
    for run_dir, declared_index in run_specs:
        result, result_sha256 = _validate_bundle(run_dir, production=production)
        scaling = _scaling_by_scheme(run_dir, cases=cases)
        effective_beta = float(scaling["baseline"]["effective_beta_value"])
        index = _beta_index(effective_beta, betas)
        if declared_index is not None and index != declared_index:
            raise ValueError(
                f"{run_dir.name} contains beta index {index}, expected {declared_index}."
            )
        if index in seen_indices:
            raise ValueError(f"Duplicate configured beta index {index} among selected runs.")
        seen_indices.add(index)
        displacement = _displacement_by_scheme_layer(run_dir)
        precision = _precision_by_scheme_parameter(run_dir)
        for scheme in SCHEMES:
            expected_base = betas[index] / float(cases[scheme]["amplification_factor"])
            if not _close(float(scaling[scheme]["base_beta_value"]), expected_base):
                raise ValueError(
                    f"Scaled base beta mismatch for {run_dir.name} {scheme}: expected {expected_base:.17g}"
                )
            long_rows.extend(
                _endpoint_rows(
                    run_dir,
                    scheme=scheme,
                    scaling=scaling[scheme],
                    displacement=displacement,
                    precision=precision,
                    beta_index=index,
                    result_sha256=result_sha256,
                )
            )
        included_runs.append(
            {
                "beta_index": index,
                "effective_beta_B": effective_beta,
                "run_id": run_dir.name,
                "path": str(run_dir),
                "result_sha256": result_sha256,
                "smoke": bool(result.get("smoke")),
            }
        )

    long_rows.sort(
        key=lambda row: (
            int(row["beta_index"]),
            SCHEMES.index(str(row["scheme"])),
            LAYER_ORDER[str(row["layer"])],
        )
    )
    included_runs.sort(key=lambda row: int(row["beta_index"]))
    expected_rows = len(seen_indices) * len(SCHEMES) * len(PARAMETERS)
    if len(long_rows) != expected_rows:
        raise ValueError(f"Expected {expected_rows} long rows, got {len(long_rows)}")
    production_complete = production and seen_indices == set(range(len(betas)))
    if production and not production_complete:
        raise ValueError("Production analysis did not cover every configured beta.")

    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else (study_root / "analysis").resolve()
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    state_wide, state_fields = _wide_rows(long_rows, STATE_WIDE_METRICS)
    learning_wide, learning_fields = _wide_rows(long_rows, LEARNING_WIDE_METRICS)
    _write_csv(output_dir / "observables_long.csv", long_rows, LONG_FIELDS)
    _write_csv(output_dir / "state_displacement_wide.csv", state_wide, state_fields)
    _write_csv(output_dir / "learning_rule_wide.csv", learning_wide, learning_fields)
    _write_observable_table(output_dir / "observables_table.md", long_rows)
    _plot(output_dir / "beta_observables_loglog.png", long_rows)
    _write_report(
        output_dir / "report.md",
        config_path=config_path,
        study_root=study_root,
        rows=long_rows,
        included_runs=included_runs,
        production_complete=production_complete,
    )

    summary = {
        "schema": OUTPUT_SCHEMA,
        "study_id": config["study_id"],
        "evidence_tier": config["evidence_tier"],
        "evidence_class": config["evidence_class"],
        "config": {"path": str(config_path), "sha256": _sha256(config_path)},
        "input_root": str(study_root),
        "production_default": production,
        "production_complete": production_complete,
        "expected_beta_count": len(betas),
        "included_beta_count": len(seen_indices),
        "included_effective_betas": [betas[index] for index in sorted(seen_indices)],
        "included_runs": included_runs,
        "counts": {
            "schemes": len(SCHEMES),
            "layers_per_scheme": len(PARAMETERS),
            "long_rows": len(long_rows),
            "state_wide_rows": len(state_wide),
            "learning_wide_rows": len(learning_wide),
        },
        "beta_scaling": {
            "formula": "B=base_beta*(voltage_amplification/current_amplification)^3",
            "baseline": "base_beta=B",
            "ours": "base_beta=B/64",
        },
        "observable_definitions": {
            "state_displacement": "s_plus-s_zero at matched beta=0 and positive K-step endpoints",
            "q": "exact code-path q=dE/dw endpoint statistic",
            "g": "(q_plus-q_zero)/B",
            "raw_drop_squared": "2q/(alpha*spatial_count)",
            "delta_q_rms_over_q_zero_rms": "RMS(delta_q)/RMS(q_zero)",
            "elementwise_delta_q_over_q_zero_rms": "RMS(delta_q/q_zero)",
        },
        "guards": {
            "all_input_bundles_valid": True,
            "all_float64_positive_matched_zero_rows_present": True,
            "all_scaled_betas_exact_within_1e-12_relative": True,
            "all_archived_eqprop_arrays_bitwise_exact": all(
                bool(row["archive_eqprop_exact"]) for row in long_rows
            ),
        },
        "outputs": {
            "long_csv": "observables_long.csv",
            "state_wide_csv": "state_displacement_wide.csv",
            "learning_wide_csv": "learning_rule_wide.csv",
            "markdown_table": "observables_table.md",
            "report": "report.md",
            "plot": "beta_observables_loglog.png",
        },
        "limitations": [
            "Exploratory ordinary-MNIST checkpoint replay; not paper-facing training evidence.",
            "One fixed 16-example minibatch and one trained checkpoint per scheme.",
            "No EqProp optimizer step or accuracy training sweep is performed.",
            "q includes the code-path minibatch mean, Conv spatial sum, and amplification prefactor.",
        ],
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--study-root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--run-glob",
        action="append",
        default=[],
        help=(
            "Analyze only directories matching this study-root-relative glob. "
            "Repeatable and intended for smoke/subset validation. With no glob, "
            "beta-00 through beta-NN are all required and smoke bundles are rejected."
        ),
    )
    return parser


def main() -> int:
    summary = analyze(_parser().parse_args())
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
