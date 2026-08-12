#!/usr/bin/env python3
"""Derive post-square noise tolerance from the completed Conv3 EqProp sweep.

This read-only ordinary-MNIST diagnostic selects the largest centered-float64
beta whose four weight layers satisfy cosine >= .99 and symmetric norm delta
<= .1.  It then consumes the exact archived q+/q-, clean EqProp, and BPTT
arrays.  Two ideal nominal-bit quantizers and one fixed-seed Gaussian q-read
noise model are evaluated.  Values are aggregate per-weight endpoint energy
statistics, not node-voltage ADC samples and not ENOB measurements.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STUDY_ROOT = REPOSITORY_ROOT / (
    "results/perfectdiode-conv3-eqprop-beta-1em3-to-1e2-init-best-"
    "true-dtype-seed0-20260811-v1"
)
SCHEMA = "conv3-centered-float64-post-square-noise/v1"
SCHEMES = ("baseline", "ours", "legacy")
ROLES = ("reconstructed_initialization", "best_validation")
PARAMETERS = ("ConvWeight_0", "ConvWeight_1", "ConvWeight_2", "DenseWeight_0")
LAYERS = {"ConvWeight_0": "C0", "ConvWeight_1": "C1", "ConvWeight_2": "C2", "DenseWeight_0": "Dense"}
STATE_LAYERS = {"ConvWeight_0": "Layer_1", "ConvWeight_1": "Layer_2", "ConvWeight_2": "Layer_3", "DenseWeight_0": "Layer_4"}
SPATIAL_COUNT = {"C0": 196, "C1": 49, "C2": 49, "Dense": 1}
ALPHA = {
    "baseline": {"C0": 1.0, "C1": 1.0, "C2": 1.0, "Dense": 1.0},
    "ours": {"C0": 1.0, "C1": 1.0 / 4.0, "C2": 1.0 / 16.0, "Dense": 1.0 / 64.0},
    "legacy": {"C0": 1.0, "C1": 1.0 / 16.0, "C2": 1.0 / 256.0, "Dense": 1.0 / 4096.0},
}
BITS = tuple(range(2, 53))
QUANTIZATION_MODES = ("separate_unsigned_q_endpoints", "signed_analog_q_difference")
COSINE_MINIMUM = 0.99
SYMMETRIC_NORM_MAXIMUM = 0.10
RANGE_HEADROOM = 1.05
GAUSSIAN_TRIALS = 4096
GAUSSIAN_BASE_SEED = 20260811
GAUSSIAN_SIGMA_RATIOS = tuple(float(10.0 ** (exponent / 8.0)) for exponent in range(-128, -7))
NORM_EPSILON = 1.0e-30


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        rows = [dict(row) for row in reader]
    if not rows:
        raise ValueError(f"Empty CSV: {path}.")
    return rows


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty CSV: {path}.")
    fields = list(rows[0])
    fields.extend(sorted(set().union(*(set(row) for row in rows)).difference(fields)))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row.get(field) for field in fields} for row in rows)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _boolean(value: object) -> bool:
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no"}:
        return False
    raise ValueError(f"Expected boolean, got {value!r}.")


def _rms(array: np.ndarray) -> float:
    values = np.asarray(array, dtype=np.float64).reshape(-1)
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError("RMS input must be finite and nonempty.")
    return float(np.sqrt(np.mean(np.square(values))))


def _gradient_metrics(candidate: np.ndarray, reference: np.ndarray) -> dict[str, Any]:
    left = np.asarray(candidate, dtype=np.float64).reshape(-1)
    right = np.asarray(reference, dtype=np.float64).reshape(-1)
    if left.shape != right.shape or not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError("Gradient metric inputs must be matching finite arrays.")
    left_norm = float(np.linalg.norm(left))
    right_norm = float(np.linalg.norm(right))
    dot = float(np.dot(left, right))
    denominator = left_norm * right_norm
    cosine = dot / denominator if denominator > NORM_EPSILON else None
    symmetric = 2.0 * abs(left_norm - right_norm) / max(left_norm + right_norm, NORM_EPSILON)
    return {
        "cosine": cosine,
        "symmetric_norm_delta": symmetric,
        "candidate_l2": left_norm,
        "reference_l2": right_norm,
        "gate_passed": bool(cosine is not None and cosine >= COSINE_MINIMUM and symmetric <= SYMMETRIC_NORM_MAXIMUM),
    }


def _quantize_unsigned(array: np.ndarray, bits: int, full_scale: float) -> tuple[np.ndarray, float]:
    if bits not in BITS or not math.isfinite(full_scale) or full_scale <= 0.0:
        raise ValueError("Invalid unsigned quantizer contract.")
    values = np.asarray(array, dtype=np.float64)
    if not np.isfinite(values).all() or np.min(values) < 0.0:
        raise ValueError("Unsigned q endpoint must be finite and nonnegative.")
    qmax = (1 << bits) - 1
    step = full_scale / float(qmax)
    return np.rint(np.clip(values, 0.0, full_scale) / step) * step, step


def _quantize_signed_midtread(array: np.ndarray, bits: int, full_scale: float) -> tuple[np.ndarray, float]:
    if bits not in BITS or not math.isfinite(full_scale) or full_scale <= 0.0:
        raise ValueError("Invalid signed quantizer contract.")
    values = np.asarray(array, dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("Signed q difference must be finite.")
    qmax = (1 << (bits - 1)) - 1
    step = full_scale / float(qmax)
    return np.rint(np.clip(values, -full_scale, full_scale) / step) * step, step


def _sustained_minimum_bits(bit_flags: Mapping[int, bool]) -> dict[str, Any]:
    if set(bit_flags) != set(BITS):
        raise ValueError("Sustained-bit audit requires every bit from 2 through 52.")
    flags = [bool(bit_flags[bit]) for bit in BITS]
    sustained = [bit for index, bit in enumerate(BITS) if all(flags[index:])]
    threshold = min(sustained) if sustained else None
    lower_failure = max((bit for bit in BITS if threshold is not None and bit < threshold and not bit_flags[bit]), default=None)
    reversals = sum(bit_flags[BITS[index - 1]] and not bit_flags[BITS[index]] for index in range(1, len(BITS)))
    return {
        "sustained_minimum_bits": threshold,
        "lower_failing_bits": lower_failure,
        "pass_to_fail_reversal_count": int(reversals),
        "threshold_resolved": threshold is not None,
    }


def select_clean_contexts(summary_csv: Path) -> tuple[list[dict[str, Any]], dict[tuple[str, str, str], dict[str, str]]]:
    rows = [
        row for row in _read_csv(summary_csv)
        if row["eqprop_variant"] == "centered" and row["precision"] == "float64"
    ]
    expected = len(SCHEMES) * len(ROLES) * len(PARAMETERS) * 11
    if len(rows) != expected:
        raise ValueError(f"Expected {expected} centered-float64 summary rows, got {len(rows)}.")
    lookup: dict[tuple[str, str, str], dict[str, str]] = {}
    selected: list[dict[str, Any]] = []
    for scheme in SCHEMES:
        for role in ROLES:
            candidates: list[tuple[float, list[dict[str, str]]]] = []
            betas = sorted({float(row["injected_beta"]) for row in rows})
            for beta in betas:
                layer_rows = [row for row in rows if row["scheme"] == scheme and row["checkpoint_role"] == role and float(row["injected_beta"]) == beta]
                if len(layer_rows) != len(PARAMETERS):
                    raise ValueError(f"Incomplete layer coverage for {scheme}/{role}/B={beta}.")
                for row in layer_rows:
                    key = (scheme, role, row["parameter_name"])
                    if key in lookup and float(lookup[key]["injected_beta"]) == beta:
                        raise ValueError(f"Duplicate summary coordinate: {key}/B={beta}.")
                if all(
                    _boolean(row["cosine_defined"])
                    and float(row["eqprop_vs_bptt_cosine"]) >= COSINE_MINIMUM
                    and float(row["eqprop_vs_bptt_symmetric_norm_delta"]) <= SYMMETRIC_NORM_MAXIMUM
                    for row in layer_rows
                ):
                    candidates.append((beta, layer_rows))
            if not candidates:
                raise ValueError(f"No clean centered-float64 beta for {scheme}/{role}.")
            beta, layer_rows = max(candidates, key=lambda item: item[0])
            next_tested_beta = next((value for value in betas if value > beta), None)
            run_ids = {row["run_id"] for row in layer_rows}
            if len(run_ids) != 1:
                raise ValueError(f"Selected layers span multiple bundles: {scheme}/{role}.")
            selected.append(
                {
                    "schema": SCHEMA,
                    "scheme": scheme,
                    "checkpoint_role": role,
                    "selected_injected_beta": beta,
                    "selected_base_beta": float(layer_rows[0]["actual_base_beta"]),
                    "amplification_factor": float(layer_rows[0]["amplification_factor"]),
                    "run_id": next(iter(run_ids)),
                    "next_tested_beta": next_tested_beta,
                    "selection_boundary": (
                        "open_upper_tested_edge"
                        if next_tested_beta is None
                        else "bracketed_by_next_failing_beta"
                    ),
                    "minimum_clean_cosine": min(float(row["eqprop_vs_bptt_cosine"]) for row in layer_rows),
                    "maximum_clean_symmetric_norm_delta": max(float(row["eqprop_vs_bptt_symmetric_norm_delta"]) for row in layer_rows),
                    "all_float64_residual_gates_passed": True,
                }
            )
            for row in layer_rows:
                lookup[(scheme, role, row["parameter_name"])] = row
    return selected, lookup


def _validate_residual_and_phase_rows(run_dir: Path, context: Mapping[str, Any]) -> list[dict[str, Any]]:
    scheme = str(context["scheme"])
    role = str(context["checkpoint_role"])
    residual = [
        row for row in _read_csv(run_dir / "equilibrium_residual_summary.csv")
        if row["scheme"] == scheme and row["checkpoint_role"] == role and row["precision"] == "float64"
    ]
    if len(residual) != 16 or not all(_boolean(row["gate_passed"]) and not _boolean(row["hard_failure"]) and row["outcome"] == "ok" for row in residual):
        raise ValueError(f"Float64 residual contract failed for {scheme}/{role} in {run_dir}.")
    phase_rows = [
        row for row in _read_csv(run_dir / "state_displacement.csv")
        if row["scheme"] == scheme and row["checkpoint_role"] == role
        and row["precision"] == "float64" and row["phase"] == "positive_minus_negative"
        and row["state_layer_name"] in set(STATE_LAYERS.values())
    ]
    if len(phase_rows) != 4:
        raise ValueError(f"Expected four centered phase-separation rows for {scheme}/{role}.")
    by_state = {row["state_layer_name"]: row for row in phase_rows}
    output = []
    for parameter in PARAMETERS:
        raw = by_state[STATE_LAYERS[parameter]]
        if raw["reference_kind"] != "negative_phase" or raw["outcome"] != "ok":
            raise ValueError(f"Invalid phase-separation row for {scheme}/{role}/{parameter}.")
        output.append(
            {
                "schema": SCHEMA,
                "scheme": scheme,
                "checkpoint_role": role,
                "selected_injected_beta": float(context["selected_injected_beta"]),
                "gradient_layer": LAYERS[parameter],
                "state_layer_name": STATE_LAYERS[parameter],
                "state_element_count": int(raw["element_count"]),
                "phase_separation_rms": float(raw["delta_rms"]),
                "phase_separation_relative_to_negative_state": float(raw["relative_displacement"]),
                "negative_state_rms": float(raw["reference_rms"]),
                "phase_separation_max_abs": float(raw["max_abs_delta"]),
            }
        )
    return output


def _stable_seed(scheme: str, role: str, parameter: str) -> int:
    token = f"{GAUSSIAN_BASE_SEED}|{scheme}|{role}|{parameter}".encode("utf-8")
    return (GAUSSIAN_BASE_SEED + int(hashlib.sha256(token).hexdigest()[:8], 16)) % (2**32)


def _gaussian_quantiles(
    clean: np.ndarray,
    bptt: np.ndarray,
    *,
    common_q_rms: float,
    injected_beta: float,
    seed: int,
) -> list[dict[str, Any]]:
    g = np.asarray(clean, dtype=np.float64).reshape(-1)
    b = np.asarray(bptt, dtype=np.float64).reshape(-1)
    n = g.size
    gnorm = float(np.linalg.norm(g))
    bnorm = float(np.linalg.norm(b))
    e1_dot_b = float(np.dot(g, b) / gnorm)
    b_residual = math.sqrt(max(bnorm * bnorm - e1_dot_b * e1_dot_b, 0.0))
    rank = 2 if b_residual > bnorm * 1.0e-14 else 1
    rng = np.random.default_rng(seed)
    z1 = rng.standard_normal(GAUSSIAN_TRIALS)
    z2 = rng.standard_normal(GAUSSIAN_TRIALS) if rank == 2 else np.zeros(GAUSSIAN_TRIALS)
    chi = rng.chisquare(n - rank, GAUSSIAN_TRIALS) if n > rank else np.zeros(GAUSSIAN_TRIALS)
    output: list[dict[str, Any]] = []
    clean_dot_bptt = float(np.dot(g, b))
    for ratio in GAUSSIAN_SIGMA_RATIOS:
        sigma_q = ratio * common_q_rms
        sigma_gradient = sigma_q / (math.sqrt(2.0) * injected_beta)
        eta_dot_g = sigma_gradient * gnorm * z1
        eta_dot_b = sigma_gradient * (e1_dot_b * z1 + b_residual * z2)
        eta_norm2 = sigma_gradient * sigma_gradient * (z1 * z1 + z2 * z2 + chi)
        candidate_norm = np.sqrt(np.maximum(gnorm * gnorm + 2.0 * eta_dot_g + eta_norm2, 0.0))
        acquisition_cosine = (gnorm * gnorm + eta_dot_g) / (candidate_norm * gnorm)
        training_cosine = (clean_dot_bptt + eta_dot_b) / (candidate_norm * bnorm)
        acquisition_symmetric = 2.0 * np.abs(candidate_norm - gnorm) / np.maximum(candidate_norm + gnorm, NORM_EPSILON)
        training_symmetric = 2.0 * np.abs(candidate_norm - bnorm) / np.maximum(candidate_norm + bnorm, NORM_EPSILON)
        row = {
            "sigma_over_q_common_rms": ratio,
            "endpoint_noise_sigma_q": sigma_q,
            "difference_noise_sigma_q": math.sqrt(2.0) * sigma_q,
            "p05_candidate_vs_clean_eqprop_cosine": float(np.quantile(acquisition_cosine, 0.05)),
            "p95_candidate_vs_clean_eqprop_symmetric_norm_delta": float(np.quantile(acquisition_symmetric, 0.95)),
            "p05_candidate_vs_bptt_cosine": float(np.quantile(training_cosine, 0.05)),
            "p95_candidate_vs_bptt_symmetric_norm_delta": float(np.quantile(training_symmetric, 0.95)),
        }
        row["combined_gate_passed"] = bool(
            row["p05_candidate_vs_clean_eqprop_cosine"] >= COSINE_MINIMUM
            and row["p95_candidate_vs_clean_eqprop_symmetric_norm_delta"] <= SYMMETRIC_NORM_MAXIMUM
            and row["p05_candidate_vs_bptt_cosine"] >= COSINE_MINIMUM
            and row["p95_candidate_vs_bptt_symmetric_norm_delta"] <= SYMMETRIC_NORM_MAXIMUM
        )
        output.append(row)
    return output


def _leading_noise_threshold(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    ordered = sorted(rows, key=lambda row: float(row["sigma_over_q_common_rms"]))
    passing: list[Mapping[str, Any]] = []
    for row in ordered:
        if not bool(row["combined_gate_passed"]):
            break
        passing.append(row)
    first_failure = ordered[len(passing)] if len(passing) < len(ordered) else None
    return {
        "maximum_sustained_sigma_over_q_common_rms": (
            None if not passing else float(passing[-1]["sigma_over_q_common_rms"])
        ),
        "next_failing_sigma_over_q_common_rms": (
            None if first_failure is None else float(first_failure["sigma_over_q_common_rms"])
        ),
        "threshold_resolved": bool(passing and first_failure is not None),
        "pass_after_first_failure_count": sum(bool(row["combined_gate_passed"]) for row in ordered[len(passing) + 1 :]),
    }


def analyze_context(
    study_root: Path,
    context: Mapping[str, Any],
    summary_lookup: Mapping[tuple[str, str, str], Mapping[str, str]],
) -> tuple[
    list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]],
    list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any],
]:
    scheme = str(context["scheme"])
    role = str(context["checkpoint_role"])
    injected_beta = float(context["selected_injected_beta"])
    run_dir = study_root / str(context["run_id"])
    if not run_dir.is_dir():
        raise FileNotFoundError(run_dir)
    phase_rows = _validate_residual_and_phase_rows(run_dir, context)
    archive = run_dir / "artifacts/eqprop_gradients" / f"conv3__{scheme}__{role}.npz"
    if not archive.is_file():
        raise FileNotFoundError(archive)
    q_rows: list[dict[str, Any]] = []
    quantization_rows: list[dict[str, Any]] = []
    quantization_thresholds: list[dict[str, Any]] = []
    gaussian_rows: list[dict[str, Any]] = []
    gaussian_thresholds: list[dict[str, Any]] = []
    with np.load(archive, allow_pickle=False) as values:
        metadata = json.loads(str(values["metadata_json"]))
        expected_metadata = {
            "architecture": "conv3",
            "scheme": scheme,
            "checkpoint_role": role,
            "eqprop_variant": "centered",
            "T": 64,
            "K": 64,
            "effective_beta": injected_beta,
            "scored_gradients": "weights_only_biases_excluded",
        }
        for key, expected in expected_metadata.items():
            observed = metadata.get(key)
            if isinstance(expected, float):
                valid = math.isclose(float(observed), expected, rel_tol=1.0e-12, abs_tol=1.0e-18)
            else:
                valid = observed == expected
            if not valid:
                raise ValueError(f"Archive metadata mismatch for {key} in {archive}: {observed!r}.")
        if any("Bias" in key for key in values.files):
            raise ValueError(f"Bias array unexpectedly archived in {archive}.")
        for parameter in PARAMETERS:
            layer = LAYERS[parameter]
            required = {
                name: values[f"{name}_float64__{parameter}"].copy()
                for name in ("endpoint_positive", "endpoint_negative", "numerator", "eqprop", "bptt")
            }
            q_plus = required["endpoint_positive"]
            q_minus = required["endpoint_negative"]
            numerator = required["numerator"]
            clean = required["eqprop"]
            bptt = required["bptt"]
            arrays = (q_plus, q_minus, numerator, clean, bptt)
            if any(array.dtype != np.float64 for array in arrays) or len({array.shape for array in arrays}) != 1:
                raise ValueError(f"Archive dtype/shape mismatch for {archive}/{parameter}.")
            if any(not np.isfinite(array).all() for array in arrays) or min(float(q_plus.min()), float(q_minus.min())) < 0.0:
                raise ValueError(f"Nonfinite or negative endpoint q for {archive}/{parameter}.")
            difference = q_plus - q_minus
            if not np.array_equal(difference, numerator) or not np.array_equal(difference / (2.0 * injected_beta), clean):
                raise ValueError(f"Centered archive identity failed for {archive}/{parameter}.")
            stored = summary_lookup[(scheme, role, parameter)]
            clean_training = _gradient_metrics(clean, bptt)
            if not (
                math.isclose(float(stored["eqprop_vs_bptt_cosine"]), float(clean_training["cosine"]), rel_tol=1.0e-11, abs_tol=1.0e-12)
                and math.isclose(float(stored["eqprop_vs_bptt_symmetric_norm_delta"]), float(clean_training["symmetric_norm_delta"]), rel_tol=1.0e-10, abs_tol=1.0e-12)
                and bool(clean_training["gate_passed"])
            ):
                raise ValueError(f"Archived clean gradient disagrees with selected summary for {scheme}/{role}/{parameter}.")

            common = 0.5 * (q_plus + q_minus)
            common_rms = _rms(common)
            difference_rms = _rms(difference)
            alpha = ALPHA[scheme][layer]
            spatial = SPATIAL_COUNT[layer]
            raw_scale = 2.0 / (alpha * spatial)
            drop2_common_rms = raw_scale * common_rms
            delta_drop2_rms = raw_scale * difference_rms
            unsigned_full_scale = RANGE_HEADROOM * max(float(q_plus.max()), float(q_minus.max()))
            signed_full_scale = RANGE_HEADROOM * float(np.max(np.abs(difference)))
            q_rows.append(
                {
                    "schema": SCHEMA,
                    "scheme": scheme,
                    "checkpoint_role": role,
                    "selected_injected_beta": injected_beta,
                    "selected_base_beta": float(context["selected_base_beta"]),
                    "run_id": str(context["run_id"]),
                    "archive_path": str(archive),
                    "archive_sha256": _sha256(archive),
                    "parameter_name": parameter,
                    "gradient_layer": layer,
                    "element_count": int(difference.size),
                    "alpha": alpha,
                    "spatial_count": spatial,
                    "q_plus_rms": _rms(q_plus),
                    "q_minus_rms": _rms(q_minus),
                    "q_common_rms": common_rms,
                    "delta_q_rms": difference_rms,
                    "delta_q_rms_over_q_common_rms": difference_rms / common_rms,
                    "q_common_rms_over_delta_q_rms": common_rms / difference_rms,
                    "drop2_common_rms": drop2_common_rms,
                    "delta_drop2_rms": delta_drop2_rms,
                    "drop2_conversion": raw_scale,
                    "unsigned_endpoint_full_scale_q": unsigned_full_scale,
                    "unsigned_endpoint_full_scale_drop2": raw_scale * unsigned_full_scale,
                    "signed_difference_full_scale_q": signed_full_scale,
                    "signed_difference_full_scale_delta_drop2": raw_scale * signed_full_scale,
                    "clean_eqprop_vs_bptt_cosine": clean_training["cosine"],
                    "clean_eqprop_vs_bptt_symmetric_norm_delta": clean_training["symmetric_norm_delta"],
                    "centered_archive_identity_exact": True,
                }
            )

            layer_quantization: list[dict[str, Any]] = []
            for bits in BITS:
                for mode in QUANTIZATION_MODES:
                    if mode == "separate_unsigned_q_endpoints":
                        quantized_plus, step = _quantize_unsigned(q_plus, bits, unsigned_full_scale)
                        quantized_minus, other_step = _quantize_unsigned(q_minus, bits, unsigned_full_scale)
                        if step != other_step:
                            raise RuntimeError("Shared endpoint quantizer step differs.")
                        candidate = (quantized_plus - quantized_minus) / (2.0 * injected_beta)
                        full_scale = unsigned_full_scale
                    else:
                        quantized_difference, step = _quantize_signed_midtread(difference, bits, signed_full_scale)
                        candidate = quantized_difference / (2.0 * injected_beta)
                        full_scale = signed_full_scale
                    acquisition = _gradient_metrics(candidate, clean)
                    training = _gradient_metrics(candidate, bptt)
                    combined = bool(acquisition["gate_passed"] and training["gate_passed"])
                    row = {
                        "schema": SCHEMA,
                        "scheme": scheme,
                        "checkpoint_role": role,
                        "selected_injected_beta": injected_beta,
                        "parameter_name": parameter,
                        "gradient_layer": layer,
                        "quantization_mode": mode,
                        "nominal_bits": bits,
                        "range_headroom": RANGE_HEADROOM,
                        "full_scale_q_units": full_scale,
                        "full_scale_drop2_units": raw_scale * full_scale,
                        "quantization_step_q_units": step,
                        "quantization_step_drop2_units": raw_scale * step,
                        "candidate_vs_clean_eqprop_cosine": acquisition["cosine"],
                        "candidate_vs_clean_eqprop_symmetric_norm_delta": acquisition["symmetric_norm_delta"],
                        "acquisition_gate_passed": acquisition["gate_passed"],
                        "candidate_vs_bptt_cosine": training["cosine"],
                        "candidate_vs_bptt_symmetric_norm_delta": training["symmetric_norm_delta"],
                        "training_gate_passed": training["gate_passed"],
                        "combined_gate_passed": combined,
                    }
                    layer_quantization.append(row)
                    quantization_rows.append(row)
            for mode in QUANTIZATION_MODES:
                mode_rows = [row for row in layer_quantization if row["quantization_mode"] == mode]
                threshold = _sustained_minimum_bits({int(row["nominal_bits"]): bool(row["combined_gate_passed"]) for row in mode_rows})
                threshold_bits = threshold["sustained_minimum_bits"]
                threshold_row = next((row for row in mode_rows if int(row["nominal_bits"]) == threshold_bits), None)
                quantization_thresholds.append(
                    {
                        "schema": SCHEMA,
                        "row_scope": "layer",
                        "scheme": scheme,
                        "checkpoint_role": role,
                        "selected_injected_beta": injected_beta,
                        "parameter_name": parameter,
                        "gradient_layer": layer,
                        "quantization_mode": mode,
                        **threshold,
                        "threshold_step_q_units": None if threshold_row is None else threshold_row["quantization_step_q_units"],
                        "threshold_step_drop2_units": None if threshold_row is None else threshold_row["quantization_step_drop2_units"],
                        "ideal_nominal_bits_not_enob": True,
                    }
                )

            seed = _stable_seed(scheme, role, parameter)
            layer_gaussian = _gaussian_quantiles(
                clean,
                bptt,
                common_q_rms=common_rms,
                injected_beta=injected_beta,
                seed=seed,
            )
            for row in layer_gaussian:
                row.update(
                    {
                        "schema": SCHEMA,
                        "scheme": scheme,
                        "checkpoint_role": role,
                        "selected_injected_beta": injected_beta,
                        "parameter_name": parameter,
                        "gradient_layer": layer,
                        "fixed_seed": seed,
                        "trial_count": GAUSSIAN_TRIALS,
                        "endpoint_noise_sigma_drop2": raw_scale * float(row["endpoint_noise_sigma_q"]),
                        "difference_noise_sigma_delta_drop2": raw_scale * float(row["difference_noise_sigma_q"]),
                    }
                )
                gaussian_rows.append(row)
            noise_threshold = _leading_noise_threshold(layer_gaussian)
            ratio = noise_threshold["maximum_sustained_sigma_over_q_common_rms"]
            sigma_q = None if ratio is None else ratio * common_rms
            gaussian_thresholds.append(
                {
                    "schema": SCHEMA,
                    "row_scope": "layer",
                    "scheme": scheme,
                    "checkpoint_role": role,
                    "selected_injected_beta": injected_beta,
                    "parameter_name": parameter,
                    "gradient_layer": layer,
                    **noise_threshold,
                    "maximum_sustained_endpoint_noise_sigma_q": sigma_q,
                    "maximum_sustained_endpoint_noise_sigma_drop2": None if sigma_q is None else raw_scale * sigma_q,
                    "maximum_sustained_difference_noise_sigma_q": None if sigma_q is None else math.sqrt(2.0) * sigma_q,
                    "maximum_sustained_difference_noise_sigma_delta_drop2": None if sigma_q is None else math.sqrt(2.0) * raw_scale * sigma_q,
                    "maximum_endpoint_noise_sigma_over_delta_q_rms": None if sigma_q is None else sigma_q / difference_rms,
                    "fixed_seed": seed,
                    "trial_count": GAUSSIAN_TRIALS,
                    "quantile_gate": "p05_cosine>=.99_and_p95_symmetric_norm_delta<=.1_vs_clean_and_bptt",
                }
            )
    provenance = {
        "run_dir": str(run_dir),
        "run_result_sha256": _sha256(run_dir / "result.json"),
        "archive": str(archive),
        "archive_sha256": _sha256(archive),
    }
    return q_rows, quantization_rows, quantization_thresholds, gaussian_rows, gaussian_thresholds, phase_rows, provenance


def _aggregate_quantization_thresholds(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for scheme in SCHEMES:
        for role in ROLES:
            for mode in QUANTIZATION_MODES:
                selected = [row for row in rows if row["scheme"] == scheme and row["checkpoint_role"] == role and row["quantization_mode"] == mode]
                bit_flags = {
                    bit: all(bool(row["combined_gate_passed"]) for row in selected if int(row["nominal_bits"]) == bit)
                    for bit in BITS
                }
                threshold = _sustained_minimum_bits(bit_flags)
                layer_thresholds = {
                    layer: _sustained_minimum_bits(
                        {
                            bit: next(
                                bool(row["combined_gate_passed"])
                                for row in selected
                                if row["gradient_layer"] == layer and int(row["nominal_bits"]) == bit
                            )
                            for bit in BITS
                        }
                    )["sustained_minimum_bits"]
                    for layer in ("C0", "C1", "C2", "Dense")
                }
                limiting = [layer for layer, value in layer_thresholds.items() if value == threshold["sustained_minimum_bits"]]
                output.append(
                    {
                        "schema": SCHEMA,
                        "row_scope": "all_layers",
                        "scheme": scheme,
                        "checkpoint_role": role,
                        "selected_injected_beta": float(selected[0]["selected_injected_beta"]),
                        "parameter_name": "__all_weights__",
                        "gradient_layer": "all",
                        "quantization_mode": mode,
                        **threshold,
                        "threshold_step_q_units": None,
                        "threshold_step_drop2_units": None,
                        "ideal_nominal_bits_not_enob": True,
                        "limiting_layers": ";".join(sorted(set(limiting))),
                    }
                )
    return output


def _aggregate_gaussian_thresholds(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for scheme in SCHEMES:
        for role in ROLES:
            selected = [row for row in rows if row["scheme"] == scheme and row["checkpoint_role"] == role and row["row_scope"] == "layer"]
            limiting = min(selected, key=lambda row: float(row["maximum_sustained_sigma_over_q_common_rms"]))
            output.append(
                {
                    "schema": SCHEMA,
                    "row_scope": "all_layers",
                    "scheme": scheme,
                    "checkpoint_role": role,
                    "selected_injected_beta": float(limiting["selected_injected_beta"]),
                    "parameter_name": "__all_weights__",
                    "gradient_layer": "all",
                    "maximum_sustained_sigma_over_q_common_rms": limiting["maximum_sustained_sigma_over_q_common_rms"],
                    "next_failing_sigma_over_q_common_rms": limiting["next_failing_sigma_over_q_common_rms"],
                    "threshold_resolved": limiting["threshold_resolved"],
                    "pass_after_first_failure_count": None,
                    "maximum_sustained_endpoint_noise_sigma_q": None,
                    "maximum_sustained_endpoint_noise_sigma_drop2": None,
                    "maximum_sustained_difference_noise_sigma_q": None,
                    "maximum_sustained_difference_noise_sigma_delta_drop2": None,
                    "maximum_endpoint_noise_sigma_over_delta_q_rms": None,
                    "fixed_seed": "per_layer_fixed_seed",
                    "trial_count": GAUSSIAN_TRIALS,
                    "quantile_gate": limiting["quantile_gate"],
                    "limiting_layer": limiting["gradient_layer"],
                }
            )
    return output


def _plot_thresholds(
    quantization: Sequence[Mapping[str, Any]],
    gaussian: Sequence[Mapping[str, Any]],
    output_dir: Path,
) -> list[Path]:
    colors = {QUANTIZATION_MODES[0]: "#0072B2", QUANTIZATION_MODES[1]: "#D55E00"}
    labels = {QUANTIZATION_MODES[0]: "separate unsigned q±", QUANTIZATION_MODES[1]: "signed analog Δq"}
    x = np.arange(len(PARAMETERS))
    layer_labels = [LAYERS[item] for item in PARAMETERS]
    paths: list[Path] = []
    fig, axes = plt.subplots(3, 2, figsize=(12.5, 10.5), sharex=True, sharey=True)
    for row_index, scheme in enumerate(SCHEMES):
        for column_index, role in enumerate(ROLES):
            axis = axes[row_index, column_index]
            for mode in QUANTIZATION_MODES:
                selected = [row for row in quantization if row["row_scope"] == "layer" and row["scheme"] == scheme and row["checkpoint_role"] == role and row["quantization_mode"] == mode]
                by_layer = {row["gradient_layer"]: row for row in selected}
                y = [float(by_layer[label]["sustained_minimum_bits"]) for label in layer_labels]
                axis.plot(x, y, marker="o" if mode == QUANTIZATION_MODES[0] else "s", color=colors[mode], label=labels[mode])
                for xpos, value in zip(x, y, strict=True):
                    axis.text(xpos, value + 0.7, f"{int(value)}", ha="center", fontsize=8, color=colors[mode])
            axis.set_title(f"{scheme} · {'initialization' if role == ROLES[0] else 'best checkpoint'}")
            axis.grid(True, axis="y", alpha=0.25)
            axis.set_ylim(0, 46)
            axis.set_xticks(x, layer_labels)
            if column_index == 0:
                axis.set_ylabel("Sustained ideal nominal bits")
    axes[0, 0].legend(frameon=False)
    fig.suptitle("Conv3 centered float64 post-square quantization thresholds\nBoth acquisition-vs-clean and training-vs-BPTT gates must pass")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    path = output_dir / "quantization_sustained_bits.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    paths.append(path)

    fig, axes = plt.subplots(3, 2, figsize=(12.5, 10.5), sharex=True, sharey=True)
    for row_index, scheme in enumerate(SCHEMES):
        for column_index, role in enumerate(ROLES):
            axis = axes[row_index, column_index]
            selected = [row for row in gaussian if row["row_scope"] == "layer" and row["scheme"] == scheme and row["checkpoint_role"] == role]
            by_layer = {row["gradient_layer"]: row for row in selected}
            y = [float(by_layer[label]["maximum_sustained_sigma_over_q_common_rms"]) for label in layer_labels]
            axis.plot(x, y, marker="o", color="#009E73")
            axis.set_yscale("log")
            axis.grid(True, which="both", axis="y", alpha=0.25)
            axis.set_title(f"{scheme} · {'initialization' if role == ROLES[0] else 'best checkpoint'}")
            axis.set_xticks(x, layer_labels)
            if column_index == 0:
                axis.set_ylabel(r"Max endpoint $\sigma_q/\mathrm{RMS}(q_{common})$")
    fig.suptitle(f"Independent Gaussian q± read-noise tolerance ({GAUSSIAN_TRIALS} fixed-seed trials)\np05 cosine and p95 norm gates vs clean EqProp and BPTT")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    path = output_dir / "gaussian_post_square_tolerance.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    paths.append(path)
    return paths


def _markdown_table(headers: Sequence[str], rows: Iterable[Sequence[object]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    lines.extend("| " + " | ".join(str(value) for value in row) + " |" for row in rows)
    return "\n".join(lines)


def _write_report(
    path: Path,
    selected: Sequence[Mapping[str, Any]],
    quantization: Sequence[Mapping[str, Any]],
    gaussian: Sequence[Mapping[str, Any]],
) -> None:
    q_aggregate = [row for row in quantization if row["row_scope"] == "all_layers"]
    g_aggregate = [row for row in gaussian if row["row_scope"] == "all_layers"]
    g_layers = [row for row in gaussian if row["row_scope"] == "layer"]

    def limiting_gaussian_row(aggregate: Mapping[str, Any]) -> Mapping[str, Any]:
        return next(
            row
            for row in g_layers
            if row["scheme"] == aggregate["scheme"]
            and row["checkpoint_role"] == aggregate["checkpoint_role"]
            and row["gradient_layer"] == aggregate["limiting_layer"]
        )

    ours_initial = limiting_gaussian_row(next(row for row in g_aggregate if row["scheme"] == "ours" and row["checkpoint_role"] == ROLES[0]))
    legacy_initial = limiting_gaussian_row(next(row for row in g_aggregate if row["scheme"] == "legacy" and row["checkpoint_role"] == ROLES[0]))
    ours_trained = limiting_gaussian_row(next(row for row in g_aggregate if row["scheme"] == "ours" and row["checkpoint_role"] == ROLES[1]))
    legacy_trained = limiting_gaussian_row(next(row for row in g_aggregate if row["scheme"] == "legacy" and row["checkpoint_role"] == ROLES[1]))
    initial_drop2_ratio = float(ours_initial["maximum_sustained_endpoint_noise_sigma_drop2"]) / float(legacy_initial["maximum_sustained_endpoint_noise_sigma_drop2"])
    trained_drop2_ratio = float(ours_trained["maximum_sustained_endpoint_noise_sigma_drop2"]) / float(legacy_trained["maximum_sustained_endpoint_noise_sigma_drop2"])
    lines = [
        "# Conv3 centered-float64 post-square noise tolerance",
        "",
        "Evidence class: **ordinary-MNIST read-only learning-algorithm diagnostic**. This is one fixed 16-example minibatch at `T=K=64`; it is not paper-facing accuracy or an EqProp training run.",
        "",
        "## Selected clean points",
        "",
        _markdown_table(
            ("scheme", "checkpoint", "injected B", "base beta", "boundary", "min cosine", "max norm delta"),
            ((row["scheme"], row["checkpoint_role"], f"{float(row['selected_injected_beta']):.6g}", f"{float(row['selected_base_beta']):.6g}", row["selection_boundary"], f"{float(row['minimum_clean_cosine']):.6f}", f"{float(row['maximum_clean_symmetric_norm_delta']):.6g}") for row in selected),
        ),
        "",
        "Baseline `B=100` is the largest tested clean point and remains open at the upper grid edge. Ours and legacy are bracketed by the next tested beta, which fails at least one all-layer clean-fidelity gate.",
        "",
        "## All-layer ideal quantization threshold",
        "",
        _markdown_table(
            ("scheme", "checkpoint", "separate q± bits", "analog Δq bits"),
            (
                (
                    scheme,
                    role,
                    next(row["sustained_minimum_bits"] for row in q_aggregate if row["scheme"] == scheme and row["checkpoint_role"] == role and row["quantization_mode"] == QUANTIZATION_MODES[0]),
                    next(row["sustained_minimum_bits"] for row in q_aggregate if row["scheme"] == scheme and row["checkpoint_role"] == role and row["quantization_mode"] == QUANTIZATION_MODES[1]),
                )
                for scheme in SCHEMES for role in ROLES
            ),
        ),
        "",
        "Bits are ideal nominal transfer-function bits, not measured ENOB. Endpoint mode quantizes nonnegative aggregate per-weight `q+` and `q-` independently with one shared per-layer range `1.05*max(q+,q-)`, then subtracts digitally. Analog-difference mode quantizes `q+-q-` with a symmetric zero-preserving range. The sustained threshold requires every higher tested bit through 52 to pass both candidate-vs-clean-EqProp and candidate-vs-BPTT gates (`cosine>=.99`, symmetric norm delta `<=.1`).",
        "",
        "## Independent post-square Gaussian q-read noise",
        "",
        _markdown_table(
            ("scheme", "checkpoint", "limiting layer", "max endpoint sigma/common RMS", "endpoint sigma_drop2", "difference sigma_delta_d2", "next failing"),
            (
                (
                    row["scheme"],
                    row["checkpoint_role"],
                    row["limiting_layer"],
                    f"{float(row['maximum_sustained_sigma_over_q_common_rms']):.6g}",
                    f"{float(limiting_gaussian_row(row)['maximum_sustained_endpoint_noise_sigma_drop2']):.6g}",
                    f"{float(limiting_gaussian_row(row)['maximum_sustained_difference_noise_sigma_delta_drop2']):.6g}",
                    f"{float(row['next_failing_sigma_over_q_common_rms']):.6g}",
                )
                for row in g_aggregate
            ),
        ),
        "",
        f"Each aggregate per-weight q entry receives independent zero-mean Gaussian noise in q+ and q- before subtraction. Results use {GAUSSIAN_TRIALS} fixed-seed trials per layer with common random numbers over a 1/8-decade grid. Passing requires p05 cosine and p95 symmetric-norm gates against both clean EqProp and BPTT. This is post-square aggregate-statistic noise, not voltage-read noise before squaring, and it omits device correlations and non-Gaussian errors.",
        "",
        f"At the C0 all-layer bottleneck, ours tolerates about `{initial_drop2_ratio:.0f}x` the legacy interaction-normalized endpoint `sigma_drop2` at initialization and `{trained_drop2_ratio:.0f}x` after training. These are ratios of the largest passing points on the discrete 1/8-decade grid, not continuous or confidence-bounded threshold estimates.",
        "",
        "## Cross-scheme units",
        "",
        "Every q/common/difference and noise threshold is reported both in stored endpoint-statistic units and interaction-normalized squared-drop units using `delta_d2 = 2*(q+ - q-)/(alpha*spatial_count)`. Spatial counts are `196/49/49/1`; alpha is `1,1,1,1` baseline, `1,1/4,1/16,1/64` ours, and `1,1/16,1/256,1/4096` legacy. This conversion must be used before treating an absolute q-noise number as cross-scheme physical evidence.",
        "",
        "Initialization and best-validation checkpoints are reported separately. Biases are active in dynamics but excluded from every scored archive.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-root", type=Path, default=DEFAULT_STUDY_ROOT)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    study_root = args.study_root.expanduser().resolve()
    output_dir = (args.output_dir.expanduser().resolve() if args.output_dir else study_root / "analysis/noise_tolerance")
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = study_root / "analysis/summary.csv"
    selected, lookup = select_clean_contexts(summary_csv)
    q_rows: list[dict[str, Any]] = []
    quantization_rows: list[dict[str, Any]] = []
    quantization_thresholds: list[dict[str, Any]] = []
    gaussian_rows: list[dict[str, Any]] = []
    gaussian_thresholds: list[dict[str, Any]] = []
    phase_rows: list[dict[str, Any]] = []
    provenance: list[dict[str, Any]] = []
    for context in selected:
        q, qc, qt, gc, gt, phases, source = analyze_context(study_root, context, lookup)
        q_rows.extend(q)
        quantization_rows.extend(qc)
        quantization_thresholds.extend(qt)
        gaussian_rows.extend(gc)
        gaussian_thresholds.extend(gt)
        phase_rows.extend(phases)
        provenance.append(source)
    quantization_thresholds.extend(_aggregate_quantization_thresholds(quantization_rows))
    gaussian_thresholds.extend(_aggregate_gaussian_thresholds(gaussian_thresholds))
    artifacts = {
        "selected_clean_betas.csv": selected,
        "q_common_difference.csv": q_rows,
        "quantization_curves.csv": quantization_rows,
        "quantization_thresholds.csv": quantization_thresholds,
        "gaussian_noise_curves.csv": gaussian_rows,
        "gaussian_noise_thresholds.csv": gaussian_thresholds,
        "phase_separation.csv": phase_rows,
    }
    for name, rows in artifacts.items():
        _write_csv(output_dir / name, rows)
    plots = _plot_thresholds(quantization_thresholds, gaussian_thresholds, output_dir)
    report = output_dir / "report.md"
    _write_report(report, selected, quantization_thresholds, gaussian_thresholds)
    outputs = [*(output_dir / name for name in artifacts), *plots, report]
    summary = {
        "schema": SCHEMA,
        "evidence_class": "ordinary_mnist_read_only_learning_algorithm_diagnostic",
        "paper_facing": False,
        "official_test_read": False,
        "optimizer_steps_applied": False,
        "bias_gradients_excluded": True,
        "settings": {
            "T": 64,
            "K": 64,
            "precision": "float64",
            "eqprop_variant": "centered",
            "cosine_minimum": COSINE_MINIMUM,
            "symmetric_norm_delta_maximum": SYMMETRIC_NORM_MAXIMUM,
            "nominal_bit_range": [min(BITS), max(BITS)],
            "range_headroom": RANGE_HEADROOM,
            "gaussian_trials": GAUSSIAN_TRIALS,
            "gaussian_base_seed": GAUSSIAN_BASE_SEED,
            "gaussian_sigma_ratio_grid": list(GAUSSIAN_SIGMA_RATIOS),
            "q_statistic_scope": "aggregate_per_weight_endpoint_energy_gradient",
            "bits_are_enob": False,
        },
        "coverage": {
            "selected_contexts": len(selected),
            "q_layer_rows": len(q_rows),
            "quantization_curve_rows": len(quantization_rows),
            "quantization_threshold_rows": len(quantization_thresholds),
            "gaussian_curve_rows": len(gaussian_rows),
            "gaussian_threshold_rows": len(gaussian_thresholds),
            "phase_separation_rows": len(phase_rows),
        },
        "selected_contexts": selected,
        "provenance": provenance,
        "input_summary": {"path": str(summary_csv), "sha256": _sha256(summary_csv)},
        "outputs": {path.name: {"path": str(path), "sha256": _sha256(path)} for path in outputs},
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), "summary": str(summary_path), "coverage": summary["coverage"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
