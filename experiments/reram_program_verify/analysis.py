"""Held-out endpoint fitting, Wan comparison, and dependency-free plots."""

from __future__ import annotations

from collections import Counter
from hashlib import sha256
from html import escape
import json
import math
from pathlib import Path
import sqlite3
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch

from experiments.artifacts import atomic_write_json


_QUANTILES = (0.01, 0.05, 0.5, 0.95, 0.99)
_EXCEEDANCE_THRESHOLDS = (0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2)


def _condition_key(
    controller: str,
    start_protocol: str,
    tolerance_ratio: float,
) -> str:
    return f"{controller}__{start_protocol}__tau_step_{tolerance_ratio:g}"


def _moments(values: Sequence[float] | np.ndarray) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return {
            "count": 0,
            "bias": None,
            "standard_deviation": None,
            "mae": None,
            "rmse": None,
            "quantiles": {f"q{int(q * 100):02d}": None for q in _QUANTILES},
            "skewness": None,
            "excess_kurtosis": None,
        }
    mean = float(np.mean(array))
    std = float(np.std(array, ddof=0))
    centered = array - mean
    if std > 0.0:
        skewness = float(np.mean(centered**3) / std**3)
        kurtosis = float(np.mean(centered**4) / std**4 - 3.0)
    else:
        skewness = None
        kurtosis = None
    return {
        "count": int(array.size),
        "bias": mean,
        "standard_deviation": std,
        "mae": float(np.mean(np.abs(array))),
        "rmse": float(np.sqrt(np.mean(array**2))),
        "quantiles": {
            f"q{int(q * 100):02d}": float(np.quantile(array, q))
            for q in _QUANTILES
        },
        "skewness": skewness,
        "excess_kurtosis": kurtosis,
    }


def _cost(values: Sequence[int] | np.ndarray) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return {"count": 0, "mean": None, "median": None, "p95": None, "maximum": None}
    return {
        "count": int(array.size),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p95": float(np.quantile(array, 0.95)),
        "maximum": int(np.max(array)),
    }


def _exceedance(values: Sequence[float] | np.ndarray) -> dict[str, float | None]:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    return {
        f"abs_error_gt_{threshold:g}": (
            float(np.mean(np.abs(array) > threshold)) if array.size else None
        )
        for threshold in _EXCEEDANCE_THRESHOLDS
    }


def _polyval(coefficients: Sequence[float], values: np.ndarray) -> np.ndarray:
    return np.polynomial.polynomial.polyval(values, np.asarray(coefficients))


def _wasserstein_equal_weight(first: np.ndarray, second: np.ndarray) -> float:
    """One-dimensional Wasserstein distance without a SciPy dependency."""

    if first.size == 0 or second.size == 0:
        return float("nan")
    count = max(first.size, second.size)
    probabilities = (np.arange(count, dtype=np.float64) + 0.5) / count
    first_q = np.quantile(first, probabilities)
    second_q = np.quantile(second, probabilities)
    return float(np.mean(np.abs(first_q - second_q)))


def _seed(base_seed: int, key: str, target_index: int) -> int:
    digest = sha256(f"{base_seed}\x1f{key}\x1f{target_index}".encode()).digest()
    return int.from_bytes(digest[:8], "little")


def _distinct_conditions(connection: sqlite3.Connection) -> list[tuple[str, str, float]]:
    return [
        (str(row[0]), str(row[1]), float(row[2]))
        for row in connection.execute(
            """
            SELECT DISTINCT controller, start_protocol, tolerance_ratio
            FROM trajectories
            ORDER BY controller, start_protocol, tolerance_ratio
            """
        )
    ]


def _target_rows(
    connection: sqlite3.Connection,
    condition: tuple[str, str, float],
    target_index: int,
    *,
    partition: str | None = None,
) -> list[sqlite3.Row]:
    controller, start, ratio = condition
    query = """
        SELECT * FROM trajectories
        WHERE controller=? AND start_protocol=? AND tolerance_ratio=?
          AND target_index=?
    """
    values: list[object] = [controller, start, ratio, target_index]
    if partition is not None:
        query += " AND partition_name=?"
        values.append(partition)
    query += " ORDER BY device_id, repeat_id"
    return list(connection.execute(query, values))


def _failure_class(row: sqlite3.Row) -> str | None:
    if bool(row["accepted"]):
        return None
    if bool(row["initialization_failed"]):
        return "initialization_failed"
    if bool(row["nonfinite"]):
        return "nonfinite"
    if bool(row["saturated"]):
        return "saturated_without_acceptance"
    if bool(row["budget_exhausted"]):
        return "budget_exhausted"
    return "other"


def _outcome_summary(rows: Sequence[sqlite3.Row]) -> dict[str, Any]:
    failures = Counter(
        failure for row in rows if (failure := _failure_class(row)) is not None
    )
    behavior = Counter(
        "accepted_at_stuck_state"
        if bool(row["accepted"]) and bool(row["corrupt"])
        else "accepted"
        if bool(row["accepted"])
        else str(_failure_class(row))
        for row in rows
    )
    successes = sum(bool(row["accepted"]) for row in rows)
    return {
        "trajectory_count": len(rows),
        "success_count": successes,
        "success_probability": successes / len(rows) if rows else None,
        "failure_probability": 1.0 - successes / len(rows) if rows else None,
        "failure_classes": dict(sorted(failures.items())),
        "behavior_classes": dict(sorted(behavior.items())),
        "saturated_fraction": (
            sum(bool(row["saturated"]) for row in rows) / len(rows) if rows else None
        ),
        "cost": {
            "set_pulses": _cost([int(row["set_count"] or 0) for row in rows]),
            "reset_pulses": _cost([int(row["reset_count"] or 0) for row in rows]),
            "total_pulses": _cost([int(row["total_pulses"] or 0) for row in rows]),
            "verify_reads": _cost([int(row["verify_count"] or 0) for row in rows]),
            "reversals": _cost([int(row["reversals"] or 0) for row in rows]),
        },
    }


def _kernel_bin(rows: Sequence[sqlite3.Row]) -> dict[str, Any]:
    accepted_rows = [
        row
        for row in rows
        if bool(row["accepted"])
        and not bool(row["corrupt"])
        and row["residual_apparent"] is not None
        and row["residual_persistent"] is not None
        and row["endpoint_apparent"] is not None
        and row["endpoint_persistent"] is not None
    ]
    apparent = [float(row["residual_apparent"]) for row in accepted_rows]
    persistent = [float(row["residual_persistent"]) for row in accepted_rows]
    apparent_minus_persistent = [
        float(row["endpoint_apparent"] - row["endpoint_persistent"])
        for row in accepted_rows
    ]
    all_outcomes = _outcome_summary(rows)
    noncorrupt_rows = [row for row in rows if not bool(row["corrupt"])]
    corrupt_rows = [row for row in rows if bool(row["corrupt"])]
    return {
        "apparent_residual": _moments(apparent),
        "persistent_residual": _moments(persistent),
        "apparent_minus_persistent": _moments(apparent_minus_persistent),
        "trajectory_count": all_outcomes["trajectory_count"],
        "success_count": all_outcomes["success_count"],
        "noncorrupt_success_count": len(accepted_rows),
        "failure_probability": all_outcomes["failure_probability"],
        "failure_classes": all_outcomes["failure_classes"],
        "absolute_error_exceedance": _exceedance(apparent),
        "endpoint_outside_0_1": {
            "below_zero_fraction": (
                sum(float(row["endpoint_apparent"]) < 0.0 for row in accepted_rows)
                / len(accepted_rows)
                if accepted_rows
                else None
            ),
            "above_one_fraction": (
                sum(float(row["endpoint_apparent"]) > 1.0 for row in accepted_rows)
                / len(accepted_rows)
                if accepted_rows
                else None
            ),
        },
        "saturated_fraction": all_outcomes["saturated_fraction"],
        "cost": all_outcomes["cost"],
        "population_breakdown": {
            "all": all_outcomes,
            "noncorrupt": _outcome_summary(noncorrupt_rows),
            "corrupt": _outcome_summary(corrupt_rows),
        },
    }


def build_empirical_kernel(
    database_path: Path,
    *,
    output_path: Path,
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    connection = sqlite3.connect(database_path)
    connection.row_factory = sqlite3.Row
    try:
        target_records = list(
            connection.execute(
                "SELECT DISTINCT target_index, target FROM trajectories ORDER BY target_index"
            )
        )
        conditions: dict[str, Any] = {}
        for condition in _distinct_conditions(connection):
            key = _condition_key(*condition)
            bins = []
            for target_row in target_records:
                rows = _target_rows(connection, condition, int(target_row[0]))
                summary = _kernel_bin(rows)
                summary.update(
                    {
                        "target_index": int(target_row[0]),
                        "target": float(target_row[1]),
                        "partitions": {
                            partition: _kernel_bin(
                                [
                                    row
                                    for row in rows
                                    if str(row["partition_name"]) == partition
                                ]
                            )
                            for partition in ("calibration", "fit", "validation")
                        },
                    }
                )
                bins.append(summary)
            conditions[key] = {
                "controller": condition[0],
                "start_protocol": condition[1],
                "tolerance_ratio": condition[2],
                "bins": bins,
                "sample_storage": {
                    "artifact": database_path.name,
                    "table": "trajectories",
                    "residual_column": "residual_apparent",
                    "success_condition": "accepted=1 AND corrupt=0",
                },
            }
        artifact = {
            "schema": "ebl.ibm_reram.empirical_endpoint_kernel",
            "schema_version": 1,
            "conditioning": "success",
            "coordinate": "x=(w+1)/2",
            "claim_class": metadata.get(
                "evidence_class", "model_based_aihwkit_preset"
            ),
            "continuous_kernel_population": (
                "accepted non-corrupt trajectories; failures and corrupt outcomes "
                "are modeled separately"
            ),
            "identity_partitions": {
                "calibration": "controller-estimator calibration only",
                "fit": "Gaussian surrogate fitting",
                "validation": "held-out adequacy and comparison",
            },
            "metadata": dict(metadata),
            "conditions": conditions,
        }
        atomic_write_json(output_path, artifact)
        return artifact
    finally:
        connection.close()


def fit_gaussian_surrogates(
    database_path: Path,
    *,
    output_path: Path,
    polynomial_order: int,
    standard_deviation_floor: float,
    analysis_seed: int,
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    connection = sqlite3.connect(database_path)
    connection.row_factory = sqlite3.Row
    try:
        targets = [
            (int(row[0]), float(row[1]))
            for row in connection.execute(
                "SELECT DISTINCT target_index, target FROM trajectories ORDER BY target_index"
            )
        ]
        condition_models: dict[str, Any] = {}
        for condition in _distinct_conditions(connection):
            key = _condition_key(*condition)
            fit_points = []
            for target_index, target in targets:
                rows = _target_rows(connection, condition, target_index, partition="fit")
                residuals = np.asarray(
                    [
                        float(row["residual_apparent"])
                        for row in rows
                        if bool(row["accepted"])
                        and not bool(row["corrupt"])
                        and row["residual_apparent"] is not None
                    ],
                    dtype=np.float64,
                )
                if residuals.size:
                    fit_points.append(
                        (target, float(np.mean(residuals)), max(float(np.std(residuals)), standard_deviation_floor), int(residuals.size))
                    )

            if len(fit_points) < polynomial_order + 1:
                condition_models[key] = {
                    "controller": condition[0],
                    "start_protocol": condition[1],
                    "tolerance_ratio": condition[2],
                    "fit_status": "insufficient_accepted_target_bins",
                    "fit_target_bins": len(fit_points),
                    "required_target_bins": polynomial_order + 1,
                    "adequate": False,
                    "validation": None,
                }
                continue

            target_values = np.asarray([point[0] for point in fit_points])
            mean_values = np.asarray([point[1] for point in fit_points])
            log_std_values = np.log(np.asarray([point[2] for point in fit_points]))
            mu_coeff = np.polynomial.polynomial.polyfit(target_values, mean_values, polynomial_order)
            log_sigma_coeff = np.polynomial.polynomial.polyfit(
                target_values, log_std_values, polynomial_order
            )

            all_validation_error: list[float] = []
            coverage90_hits = 0
            coverage95_hits = 0
            validation_count = 0
            wasserstein: list[float] = []
            per_target = []
            for target_index, target in targets:
                rows = _target_rows(connection, condition, target_index, partition="validation")
                accepted_rows = [
                    row
                    for row in rows
                    if bool(row["accepted"])
                    and not bool(row["corrupt"])
                    and row["residual_apparent"] is not None
                ]
                residuals = np.asarray(
                    [float(row["residual_apparent"]) for row in accepted_rows],
                    dtype=np.float64,
                )
                outcomes = _outcome_summary(rows)
                noncorrupt_outcomes = _outcome_summary(
                    [row for row in rows if not bool(row["corrupt"])]
                )
                corrupt_outcomes = _outcome_summary(
                    [row for row in rows if bool(row["corrupt"])]
                )
                mu = float(_polyval(mu_coeff, np.asarray([target]))[0])
                sigma = max(
                    float(np.exp(_polyval(log_sigma_coeff, np.asarray([target]))[0])),
                    standard_deviation_floor,
                )
                if residuals.size:
                    centered = residuals - mu
                    covered90 = np.abs(centered) <= 1.6448536269514722 * sigma
                    covered95 = np.abs(centered) <= 1.959963984540054 * sigma
                    target_coverage90 = float(
                        np.mean(covered90)
                    )
                    target_coverage95 = float(
                        np.mean(covered95)
                    )
                    coverage90_hits += int(np.count_nonzero(covered90))
                    coverage95_hits += int(np.count_nonzero(covered95))
                    validation_count += int(residuals.size)
                    all_validation_error.extend(residuals.tolist())
                    generator = np.random.default_rng(_seed(analysis_seed, key, target_index))
                    gaussian = mu + sigma * generator.standard_normal(residuals.size)
                    distance = _wasserstein_equal_weight(residuals, gaussian)
                    wasserstein.append(distance)
                else:
                    distance = None
                    target_coverage90 = None
                    target_coverage95 = None
                per_target.append(
                    {
                        "target_index": target_index,
                        "target": target,
                        "predicted_mu": mu,
                        "predicted_sigma": sigma,
                        "held_out_residual": _moments(residuals),
                        "absolute_error_exceedance": _exceedance(residuals),
                        "coverage_90": target_coverage90,
                        "coverage_95": target_coverage95,
                        "wasserstein_distance": distance,
                        "trajectory_count": len(rows),
                        "success_count": outcomes["success_count"],
                        "noncorrupt_success_count": len(accepted_rows),
                        "failure_probability": outcomes["failure_probability"],
                        "failure_classes": outcomes["failure_classes"],
                        "saturated_fraction": outcomes["saturated_fraction"],
                        "endpoint_outside_0_1": {
                            "below_zero_fraction": (
                                sum(float(row["endpoint_apparent"]) < 0.0 for row in accepted_rows)
                                / len(accepted_rows)
                                if accepted_rows
                                else None
                            ),
                            "above_one_fraction": (
                                sum(float(row["endpoint_apparent"]) > 1.0 for row in accepted_rows)
                                / len(accepted_rows)
                                if accepted_rows
                                else None
                            ),
                        },
                        "cost": outcomes["cost"],
                        "population_breakdown": {
                            "all": outcomes,
                            "noncorrupt": noncorrupt_outcomes,
                            "corrupt": corrupt_outcomes,
                        },
                    }
                )

            coverage90 = coverage90_hits / validation_count if validation_count else None
            coverage95 = coverage95_hits / validation_count if validation_count else None
            residual_std = float(np.std(np.asarray(all_validation_error))) if all_validation_error else None
            median_wasserstein = float(np.median(wasserstein)) if wasserstein else None
            normalized_wasserstein = (
                median_wasserstein / residual_std
                if median_wasserstein is not None and residual_std is not None and residual_std > 0.0
                else None
            )
            coverage_ok = (
                coverage90 is not None and coverage95 is not None
                and abs(coverage90 - 0.90) <= 0.03
                and abs(coverage95 - 0.95) <= 0.03
            )
            wasserstein_ok = normalized_wasserstein is not None and normalized_wasserstein <= 0.1
            condition_models[key] = {
                "controller": condition[0],
                "start_protocol": condition[1],
                "tolerance_ratio": condition[2],
                "fit_status": "fit",
                "polynomial_basis": "ascending_power_x",
                "mu_coefficients": mu_coeff.tolist(),
                "log_sigma_coefficients": log_sigma_coeff.tolist(),
                "standard_deviation_floor": standard_deviation_floor,
                "fit_points": [
                    {"target": point[0], "mu": point[1], "sigma": point[2], "count": point[3]}
                    for point in fit_points
                ],
                "validation": {
                    "accepted_noncorrupt_count": validation_count,
                    "residual": _moments(all_validation_error),
                    "coverage_90": coverage90,
                    "coverage_95": coverage95,
                    "coverage_90_error": abs(coverage90 - 0.90) if coverage90 is not None else None,
                    "coverage_95_error": abs(coverage95 - 0.95) if coverage95 is not None else None,
                    "median_target_binned_wasserstein": median_wasserstein,
                    "median_wasserstein_over_residual_std": normalized_wasserstein,
                    "per_target": per_target,
                },
                "adequacy_thresholds": {
                    "maximum_absolute_coverage_error": 0.03,
                    "maximum_median_wasserstein_over_residual_std": 0.1,
                },
                "adequate": bool(coverage_ok and wasserstein_ok),
            }

        artifact = {
            "schema": "ebl.ibm_reram.gaussian_endpoint_model",
            "schema_version": 1,
            "coordinate": "x=(w+1)/2",
            "raw_equation": "x_programmed=x_target+mu(x_target)+sigma(x_target)*Normal(0,1)",
            "deployment_view": "clip raw endpoint to [0,1] only when explicitly requested",
            "fit_partition": "fit device identities; accepted non-corrupt trajectories only",
            "validation_partition": "held-out validation device identities",
            "metadata": dict(metadata),
            "conditions": condition_models,
        }
        atomic_write_json(output_path, artifact)
        return artifact
    finally:
        connection.close()


def sample_wan_reference(
    *,
    targets: Sequence[float],
    samples_per_target: int,
    g_max_us: float,
    noise_scale: float,
    wan_seed: int,
) -> dict[str, Any]:
    """Sample the pinned AIHWKit Wan endpoint model without IBM fit inputs."""

    try:
        import aihwkit
        from aihwkit.inference.noise.reram import ReRamWan2022NoiseModel
    except ImportError as error:  # pragma: no cover - environment dependent
        raise RuntimeError(f"Expected AIHWKit Wan-2022 noise model. Provided value: {error}.") from error

    model = ReRamWan2022NoiseModel(g_max=g_max_us, noise_scale=noise_scale)
    programming_time = min(model.coeff_dic)
    coefficients = model.coeff_dic[programming_time]
    coefficient_reference = float(model.coeff_g_max_reference)
    records = []
    for target_index, target in enumerate(targets):
        target_tensor = torch.full((samples_per_target,), float(g_max_us * target), dtype=torch.float32)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(_seed(wan_seed, "wan2022", target_index))
            endpoint_us = model.apply_programming_noise_to_conductance(target_tensor)
        raw_endpoint = endpoint_us.double().cpu().numpy() / g_max_us
        raw_error = raw_endpoint - target
        clipped_endpoint = np.clip(raw_endpoint, 0.0, 1.0)
        clipped_error = clipped_endpoint - target
        polynomial_sigma_us = sum(
            float(coefficient) * float(target) ** power
            for power, coefficient in enumerate(coefficients)
        ) * (g_max_us / coefficient_reference) * noise_scale
        records.append(
            {
                "target_index": target_index,
                "target": float(target),
                "samples": int(samples_per_target),
                "pre_native_lower_clamp_gaussian": {
                    "mu_normalized": 0.0,
                    "sigma_normalized": abs(polynomial_sigma_us) / g_max_us,
                },
                "raw": {
                    "endpoint_below_zero_fraction": float(np.mean(raw_endpoint < 0.0)),
                    "endpoint_above_one_fraction": float(np.mean(raw_endpoint > 1.0)),
                    "endpoint_at_zero_fraction": float(np.mean(raw_endpoint == 0.0)),
                    "endpoint_at_one_fraction": float(np.mean(raw_endpoint == 1.0)),
                    "error": _moments(raw_error),
                    "absolute_error_exceedance": _exceedance(raw_error),
                },
                "explicit_clipped_0_1": {
                    "clipped_low_fraction": float(np.mean(raw_endpoint < 0.0)),
                    "clipped_high_fraction": float(np.mean(raw_endpoint > 1.0)),
                    "endpoint_at_zero_fraction": float(np.mean(clipped_endpoint == 0.0)),
                    "endpoint_at_one_fraction": float(np.mean(clipped_endpoint == 1.0)),
                    "error": _moments(clipped_error),
                    "absolute_error_exceedance": _exceedance(clipped_error),
                },
            }
        )

    return {
        "schema": "ebl.ibm_reram.wan2022_reference_samples",
        "schema_version": 1,
        "aihwkit_version": str(getattr(aihwkit, "__version__", "unknown")),
        "wan_model": {
            "class": "ReRamWan2022NoiseModel",
            "g_max_us": g_max_us,
            "noise_scale": noise_scale,
            "programming_time_seconds": float(programming_time),
            "coefficients_at_programming_time": [
                float(value) for value in coefficients
            ],
            "coefficient_g_max_reference_us": coefficient_reference,
            "native_lower_clamp": True,
            "native_upper_clamp": False,
        },
        "sampling": {
            "base_seed": int(wan_seed),
            "target_seed_derivation": (
                "first 64 bits of SHA256(base_seed, 'wan2022', target_index), "
                "interpreted little-endian"
            ),
            "samples_per_target": int(samples_per_target),
        },
        "targets": records,
    }


def build_wan_comparison(
    *,
    targets: Sequence[float],
    samples_per_target: int,
    g_max_us: float,
    noise_scale: float,
    wan_seed: int,
    fit_artifact: Mapping[str, Any],
    output_path: Path,
    wan_reference: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    reference = (
        sample_wan_reference(
            targets=targets,
            samples_per_target=samples_per_target,
            g_max_us=g_max_us,
            noise_scale=noise_scale,
            wan_seed=wan_seed,
        )
        if wan_reference is None
        else dict(wan_reference)
    )
    if (
        reference.get("schema") != "ebl.ibm_reram.wan2022_reference_samples"
        or reference.get("schema_version") != 1
        or reference.get("sampling", {}).get("base_seed") != int(wan_seed)
        or reference.get("sampling", {}).get("samples_per_target")
        != int(samples_per_target)
        or len(reference.get("targets", [])) != len(targets)
    ):
        raise RuntimeError(
            "Expected the Wan reference samples to match the requested comparison."
        )
    model_metadata = reference["wan_model"]
    if (
        float(model_metadata.get("g_max_us", -1.0)) != float(g_max_us)
        or float(model_metadata.get("noise_scale", -1.0)) != float(noise_scale)
    ):
        raise RuntimeError(
            "Expected the Wan reference model settings to match the requested comparison."
        )
    records = list(reference["targets"])

    ibm_conditions: dict[str, Any] = {}
    for key, condition in fit_artifact["conditions"].items():
        validation = condition.get("validation")
        if not validation:
            continue
        comparisons = []
        for ibm, wan in zip(validation["per_target"], records):
            ibm_stats = ibm["held_out_residual"]
            wan_raw_stats = wan["raw"]["error"]
            wan_clipped_stats = wan["explicit_clipped_0_1"]["error"]

            def difference(field: str, comparator: Mapping[str, Any]) -> float | None:
                ibm_value = ibm_stats[field]
                wan_value = comparator[field]
                if ibm_value is None or wan_value is None:
                    return None
                return float(ibm_value) - float(wan_value)

            comparisons.append(
                {
                    "target_index": ibm["target_index"],
                    "target": ibm["target"],
                    "ibm_success_conditioned_noncorrupt": {
                        "error": ibm_stats,
                        "absolute_error_exceedance": ibm[
                            "absolute_error_exceedance"
                        ],
                        "endpoint_outside_0_1": ibm["endpoint_outside_0_1"],
                        "sample_count": ibm["noncorrupt_success_count"],
                    },
                    "ibm_population_outcomes": {
                        "trajectory_count": ibm["trajectory_count"],
                        "success_count": ibm["success_count"],
                        "failure_probability": ibm["failure_probability"],
                        "failure_classes": ibm["failure_classes"],
                        "saturated_fraction": ibm["saturated_fraction"],
                        "cost": ibm["cost"],
                        "population_breakdown": ibm["population_breakdown"],
                    },
                    "wan_raw_one_second": wan["raw"],
                    "wan_explicit_clipped_0_1_one_second": wan[
                        "explicit_clipped_0_1"
                    ],
                    "ibm_minus_wan_raw": {
                        field: difference(field, wan_raw_stats)
                        for field in ("bias", "standard_deviation", "mae", "rmse")
                    },
                    "ibm_minus_wan_explicit_clipped_0_1": {
                        field: difference(field, wan_clipped_stats)
                        for field in ("bias", "standard_deviation", "mae", "rmse")
                    },
                }
            )
        ibm_conditions[key] = comparisons

    artifact = {
        "schema": "ebl.ibm_reram.wan2022_comparison",
        "schema_version": 1,
        "aihwkit_version": reference["aihwkit_version"],
        "wan_model": model_metadata,
        "sampling": reference["sampling"],
        "normalization": (
            f"G_target={g_max_us:g}*x microSiemens; endpoint and error divided "
            f"by {g_max_us:g} microSiemens"
        ),
        "ibm_fit_provenance": {
            "schema": fit_artifact.get("schema"),
            "schema_version": fit_artifact.get("schema_version"),
            "metadata": fit_artifact.get("metadata", {}),
        },
        "comparison_population": {
            "ibm": (
                "held-out validation identities; continuous error statistics are "
                "conditioned on accepted non-corrupt trajectories"
            ),
            "ibm_failures": (
                "unconditional convergence, saturation, corruption, and cost remain "
                "separate target-conditioned outcomes"
            ),
            "wan": (
                "unconditional one-second endpoint samples; the AIHWKit inference "
                "model has no pulse, verify, reversal, or P&V failure model"
            ),
        },
        "wan_cost_and_convergence": {
            "endpoint_sample_probability": 1.0,
            "program_and_verify_cost_available": False,
            "reason": (
                "ReRamWan2022NoiseModel samples a fitted programming endpoint and "
                "does not expose a physical pulse/verify trajectory"
            ),
        },
        "limitations": [
            f"The IBM preset coordinate [-1,1] has no unique calibration to 0-{g_max_us:g} microSiemens.",
            "IBM endpoints are immediate and time-unspecified; Wan endpoints use the one-second fit.",
            "This is an operational normalized comparison, not a physical equivalence claim.",
        ],
        "targets": records,
        "ibm_condition_comparisons": ibm_conditions,
    }
    atomic_write_json(output_path, artifact)
    return artifact


_COLORS = ("#2563eb", "#dc2626", "#059669", "#9333ea", "#ea580c", "#0891b2", "#4f46e5", "#be123c")


def _write_line_svg(
    path: Path,
    *,
    title: str,
    x_label: str,
    y_label: str,
    series: Mapping[str, Sequence[tuple[float, float | None]]],
    note: str | None = None,
) -> None:
    width, height = 900, 520
    left, right, top, bottom = 85, 25, 55, 75
    points = [(x, y) for values in series.values() for x, y in values if y is not None and math.isfinite(y)]
    if not points:
        points = [(0.0, 0.0), (1.0, 1.0)]
    x_values = [point[0] for point in points]
    y_values = [point[1] for point in points]
    x_min, x_max = min(x_values), max(x_values)
    y_min, y_max = min(y_values), max(y_values)
    if x_max == x_min:
        x_max = x_min + 1.0
    if y_max == y_min:
        padding = max(abs(y_min) * 0.1, 1e-6)
        y_min -= padding
        y_max += padding
    else:
        padding = 0.08 * (y_max - y_min)
        y_min -= padding
        y_max += padding
    plot_width = width - left - right
    plot_height = height - top - bottom
    sx = lambda value: left + (value - x_min) / (x_max - x_min) * plot_width
    sy = lambda value: top + (y_max - value) / (y_max - y_min) * plot_height
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width/2}" y="28" text-anchor="middle" font-family="sans-serif" font-size="18">{escape(title)}</text>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" stroke="#111827"/>',
        f'<line x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}" stroke="#111827"/>',
        f'<text x="{left + plot_width/2}" y="{height-20}" text-anchor="middle" font-family="sans-serif" font-size="13">{escape(x_label)}</text>',
        f'<text transform="translate(20 {top + plot_height/2}) rotate(-90)" text-anchor="middle" font-family="sans-serif" font-size="13">{escape(y_label)}</text>',
    ]
    if note:
        lines.append(
            f'<text x="{width/2}" y="{height-4}" text-anchor="middle" '
            f'font-family="sans-serif" font-size="9" fill="#4b5563">{escape(note)}</text>'
        )
    for tick in range(6):
        fraction = tick / 5
        x_value = x_min + fraction * (x_max - x_min)
        y_value = y_min + fraction * (y_max - y_min)
        lines.append(f'<text x="{sx(x_value):.2f}" y="{height-bottom+20}" text-anchor="middle" font-family="monospace" font-size="10">{x_value:.2g}</text>')
        lines.append(f'<text x="{left-8}" y="{sy(y_value)+4:.2f}" text-anchor="end" font-family="monospace" font-size="10">{y_value:.2g}</text>')
    legend_y = top + 5
    for index, (name, values) in enumerate(series.items()):
        color = _COLORS[index % len(_COLORS)]
        finite = [(x, y) for x, y in values if y is not None and math.isfinite(y)]
        if finite:
            coordinates = " ".join(f"{sx(x):.2f},{sy(float(y)):.2f}" for x, y in finite)
            lines.append(f'<polyline points="{coordinates}" fill="none" stroke="{color}" stroke-width="2"/>')
        legend_x = left + 8 + (index % 3) * 265
        if index and index % 3 == 0:
            legend_y += 17
        lines.append(f'<line x1="{legend_x}" y1="{legend_y}" x2="{legend_x+18}" y2="{legend_y}" stroke="{color}" stroke-width="3"/>')
        lines.append(f'<text x="{legend_x+23}" y="{legend_y+4}" font-family="sans-serif" font-size="10">{escape(name[:38])}</text>')
    lines.append("</svg>")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_plots(
    *,
    fit_artifact: Mapping[str, Any],
    wan_artifact: Mapping[str, Any],
    output_dir: Path,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    comparison_note = (
        "IBM normalized immediate/time-unspecified endpoint vs Wan "
        f"0-{wan_artifact['wan_model']['g_max_us']:g} uS at 1 s; "
        "operational normalization, not physical equivalence"
    )
    metrics = {
        "residual_bias": ("Held-out residual bias", "normalized error", "bias"),
        "residual_sigma": ("Held-out residual standard deviation", "normalized error", "standard_deviation"),
        "fitted_mu": ("Fitted Gaussian mu(x)", "normalized error", "predicted_mu"),
        "fitted_sigma": ("Fitted Gaussian sigma(x)", "normalized error", "predicted_sigma"),
        "convergence": ("Held-out P&V success probability", "probability", "success"),
        "pulse_cost": ("Held-out target-programming pulse cost", "mean pulses", "total_pulses"),
        "verify_cost": ("Held-out verify cost", "mean reads", "verify_reads"),
        "reversals": ("Held-out polarity reversals", "mean reversals", "reversals"),
    }
    created = []
    for filename, (title, ylabel, metric) in metrics.items():
        series: dict[str, list[tuple[float, float | None]]] = {}
        for key, condition in fit_artifact["conditions"].items():
            validation = condition.get("validation")
            if not validation:
                continue
            values = []
            for record in validation["per_target"]:
                if metric == "success":
                    total = record["trajectory_count"]
                    value = record["success_count"] / total if total else None
                elif metric in {"total_pulses", "verify_reads", "reversals"}:
                    value = record["cost"][metric]["mean"]
                elif metric in {"predicted_mu", "predicted_sigma"}:
                    value = record[metric]
                else:
                    value = record["held_out_residual"][metric]
                values.append((record["target"], value))
            series[key] = values
        if metric in {"residual_bias", "residual_sigma", "predicted_mu", "predicted_sigma"}:
            field = (
                "bias"
                if metric in {"residual_bias", "predicted_mu"}
                else "standard_deviation"
            )
            series["Wan2022 raw (1 s)"] = [
                (record["target"], record["raw"]["error"][field])
                for record in wan_artifact["targets"]
            ]
        path = output_dir / f"{filename}.svg"
        _write_line_svg(
            path,
            title=title,
            x_label="target x",
            y_label=ylabel,
            series=series,
            note=(
                comparison_note
                if metric
                in {"residual_bias", "residual_sigma", "predicted_mu", "predicted_sigma"}
                else None
            ),
        )
        created.append(path)

    quantile_series: dict[str, list[tuple[float, float | None]]] = {}
    for key, condition in fit_artifact["conditions"].items():
        validation = condition.get("validation")
        if not validation:
            continue
        for quantile in ("q05", "q50", "q95"):
            quantile_series[f"{key} {quantile}"] = [
                (record["target"], record["held_out_residual"]["quantiles"][quantile])
                for record in validation["per_target"]
            ]
    for quantile in ("q05", "q50", "q95"):
        quantile_series[f"Wan2022 raw (1 s) {quantile}"] = [
            (record["target"], record["raw"]["error"]["quantiles"][quantile])
            for record in wan_artifact["targets"]
        ]
    residual_path = output_dir / "residual_distributions.svg"
    _write_line_svg(
        residual_path,
        title="Held-out residual quantiles (5%, 50%, 95%)",
        x_label="target x",
        y_label="normalized error",
        series=quantile_series,
        note=comparison_note,
    )
    created.append(residual_path)

    exceedance_series: dict[str, list[tuple[float, float | None]]] = {}
    for key, condition in fit_artifact["conditions"].items():
        validation = condition.get("validation")
        if not validation or float(condition["tolerance_ratio"]) != 0.5:
            continue
        values = []
        for threshold in _EXCEEDANCE_THRESHOLDS:
            field = f"abs_error_gt_{threshold:g}"
            numerator = 0.0
            denominator = 0
            for record in validation["per_target"]:
                count = int(record["held_out_residual"]["count"])
                probability = record["absolute_error_exceedance"][field]
                if probability is not None:
                    numerator += float(probability) * count
                    denominator += count
            values.append((threshold, numerator / denominator if denominator else None))
        exceedance_series[key] = values
    exceedance_series["Wan2022 raw (1 s)"] = [
        (
            threshold,
            float(
                np.mean(
                    [
                        record["raw"]["absolute_error_exceedance"][
                            f"abs_error_gt_{threshold:g}"
                        ]
                        for record in wan_artifact["targets"]
                    ]
                )
            ),
        )
        for threshold in _EXCEEDANCE_THRESHOLDS
    ]
    exceedance_path = output_dir / "absolute_error_exceedance_primary_tolerance.svg"
    _write_line_svg(
        exceedance_path,
        title="Absolute-error exceedance (primary tolerance tau/step=0.5)",
        x_label="absolute-error threshold",
        y_label="exceedance probability",
        series=exceedance_series,
        note=comparison_note,
    )
    created.append(exceedance_path)

    boundary_series: dict[str, list[tuple[float, float | None]]] = {}
    for key, condition in fit_artifact["conditions"].items():
        validation = condition.get("validation")
        if not validation or float(condition["tolerance_ratio"]) != 0.5:
            continue
        boundary_series[f"{key} physical saturation"] = [
            (record["target"], record["saturated_fraction"])
            for record in validation["per_target"]
        ]
    boundary_series["Wan2022 native zero or >1"] = [
        (
            record["target"],
            record["raw"]["endpoint_at_zero_fraction"]
            + record["raw"]["endpoint_above_one_fraction"],
        )
        for record in wan_artifact["targets"]
    ]
    boundary_series["Wan2022 explicit [0,1] boundary mass"] = [
        (
            record["target"],
            record["explicit_clipped_0_1"]["endpoint_at_zero_fraction"]
            + record["explicit_clipped_0_1"]["endpoint_at_one_fraction"],
        )
        for record in wan_artifact["targets"]
    ]
    boundary_path = output_dir / "saturation_and_clamp_mass_primary_tolerance.svg"
    _write_line_svg(
        boundary_path,
        title="Saturation and clamp mass (primary tolerance tau/step=0.5)",
        x_label="target x",
        y_label="probability mass",
        series=boundary_series,
        note=comparison_note,
    )
    created.append(boundary_path)
    return created


def write_report(
    *,
    path: Path,
    preset: str,
    corrupt_population: bool,
    execution_profile: str,
    trajectory_count: int,
    fit_artifact: Mapping[str, Any],
    wan_artifact: Mapping[str, Any],
) -> None:
    def formatted(value: Any, digits: int = 6) -> str:
        if value is None:
            return "n/a"
        return f"{float(value):.{digits}g}"

    def pooled_cost(records: Sequence[Mapping[str, Any]], field: str) -> float | None:
        numerator = 0.0
        denominator = 0
        for record in records:
            summary = record["cost"][field]
            if summary["mean"] is not None:
                numerator += float(summary["mean"]) * int(summary["count"])
                denominator += int(summary["count"])
        return numerator / denominator if denominator else None

    rows = []
    for key, condition in fit_artifact["conditions"].items():
        validation = condition.get("validation")
        if validation:
            per_target = validation["per_target"]
            total = sum(item["trajectory_count"] for item in per_target)
            successes = sum(item["success_count"] for item in per_target)
            success = successes / total if total else float("nan")
            rows.append(
                f"| `{key}` | {success:.4f} | "
                f"{formatted(validation['residual']['rmse'])} | "
                f"{formatted(pooled_cost(per_target, 'total_pulses'), 5)} | "
                f"{formatted(pooled_cost(per_target, 'verify_reads'), 5)} | "
                f"{formatted(validation['coverage_90'], 4)} | "
                f"{formatted(validation['coverage_95'], 4)} | "
                f"{'yes' if condition['adequate'] else 'no'} |"
            )
        else:
            rows.append(
                f"| `{key}` | n/a | n/a | n/a | n/a | n/a | n/a | "
                "no (insufficient fit bins) |"
            )

    corrupt_rows = []
    if corrupt_population:
        for key, condition in fit_artifact["conditions"].items():
            validation = condition.get("validation")
            if not validation or float(condition["tolerance_ratio"]) != 0.5:
                continue
            outcomes = [
                record["population_breakdown"]["corrupt"]
                for record in validation["per_target"]
            ]
            total = sum(int(record["trajectory_count"]) for record in outcomes)
            successes = sum(int(record["success_count"]) for record in outcomes)
            saturated = sum(
                float(record["saturated_fraction"] or 0.0)
                * int(record["trajectory_count"])
                for record in outcomes
            )
            corrupt_rows.append(
                f"| `{key}` | {total} | "
                f"{formatted(successes / total if total else None, 5)} | "
                f"{formatted(saturated / total if total else None, 5)} | "
                f"{formatted(pooled_cost(outcomes, 'total_pulses'), 5)} |"
            )

    comparison_rows = []
    for key, condition in fit_artifact["conditions"].items():
        validation = condition.get("validation")
        if not validation or float(condition["tolerance_ratio"]) != 0.5:
            continue
        per_target = validation["per_target"]
        population_count = sum(int(record["trajectory_count"]) for record in per_target)
        saturation_mass = sum(
            float(record["saturated_fraction"] or 0.0)
            * int(record["trajectory_count"])
            for record in per_target
        )
        comparison_rows.append(
            f"| IBM `{key}` | {formatted(validation['residual']['bias'])} | "
            f"{formatted(validation['residual']['standard_deviation'])} | "
            f"{formatted(validation['residual']['rmse'])} | "
            f"{formatted(saturation_mass / population_count if population_count else None, 5)} |"
        )
    for label, field in (
        ("native raw (lower-clamped only)", "raw"),
        ("explicitly clipped [0,1]", "explicit_clipped_0_1"),
    ):
        records = wan_artifact["targets"]
        stats = [record[field]["error"] for record in records]
        if field == "raw":
            boundary = [
                record[field]["endpoint_at_zero_fraction"]
                + record[field]["endpoint_above_one_fraction"]
                for record in records
            ]
        else:
            boundary = [
                record[field]["endpoint_at_zero_fraction"]
                + record[field]["endpoint_at_one_fraction"]
                for record in records
            ]
        comparison_rows.append(
            f"| Wan {label} | {formatted(np.median([item['bias'] for item in stats]))} | "
            f"{formatted(np.median([item['standard_deviation'] for item in stats]))} | "
            f"{formatted(np.median([item['rmse'] for item in stats]))} | "
            f"{formatted(np.median(boundary), 5)} |"
        )

    corrupt_section = ""
    if corrupt_rows:
        corrupt_section = f"""
## Corrupt-device outcomes at the primary tolerance

Corrupt identities remain discrete outcomes and are not absorbed into the
continuous Gaussian variance.

| Condition | corrupt trajectories | success | saturated | mean target pulses |
| --- | ---: | ---: | ---: | ---: |
{chr(10).join(corrupt_rows)}
"""
    text = f"""# IBM ReRAM program-and-verify characterization

## Result boundary

This `{execution_profile}` run characterizes the AIHWKit fitted preset `{preset}` with published
corruption {'enabled' if corrupt_population else 'disabled'}. It is model-based
evidence, not raw IBM pulse-trace replay. It contains {trajectory_count:,}
programming trajectories. No HWA, Tiki-Taka, LoRA, or network-accuracy result
is produced by this milestone.

{'This smoke profile is an operational validation only and cannot satisfy the declared production study.' if execution_profile == 'smoke' else 'This production profile uses the predeclared characterization contract.'}

The P&V controller observed only apparent normalized verify values. Persistent
state and hidden device parameters were retained for diagnostics and were not
available to controller decisions. The persisted SQLite trajectory/event
ledger passed the machine-checked integrity contract before this report and
any endpoint fit were emitted.

## Held-out model results

| Condition | success | residual RMSE | mean pulses | mean verifies | 90% coverage | 95% coverage | Gaussian adequate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
{chr(10).join(rows)}

An inadequate Gaussian fit remains a documented approximation; the empirical
kernel and trajectory database are authoritative.

{corrupt_section}

## Wan-2022 comparison boundary

The comparator is `ReRamWan2022NoiseModel(g_max={wan_artifact['wan_model']['g_max_us']},
noise_scale={wan_artifact['wan_model']['noise_scale']})` at one second. Both its
native raw view (lower-clamped by AIHWKit, not upper-clamped) and an explicitly
clipped `[0,1]` deployment view are recorded.

- The IBM normalized `[-1,1]` coordinate has no unique physical mapping to
  Wan's `0-{wan_artifact['wan_model']['g_max_us']:g} µS` range.
- The IBM endpoint is immediate and time-unspecified; Wan is a one-second fit.
- The comparison is operational and normalized, not a physical-equivalence claim.

These limitations apply directly to the following headline summary. IBM
residual moments pool successful non-corrupt held-out trajectories and its
saturation mass uses the full held-out population; Wan entries are medians
across the common target grid. Full target-conditioned quantiles, skewness,
excess kurtosis, exceedance tails, and clamp masses are retained in the
machine-readable comparison artifact.

| Endpoint view | bias | standard deviation | RMSE | saturation/boundary mass |
| --- | ---: | ---: | ---: | ---: |
{chr(10).join(comparison_rows)}

IBM convergence and pulse/verify/reversal costs are reported in the held-out
condition table and JSON artifacts. Wan has no pulse trajectory or P&V failure
model, so no corresponding cost or convergence number is invented.

## Interpretation

Use the success-conditioned empirical endpoint kernel together with its
separate failure and cost models. A later matched HWA/on-chip study must begin
from preserved programmed device states; it must not redraw these endpoints.
"""
    path.write_text(text, encoding="utf-8")


__all__ = [
    "build_empirical_kernel",
    "build_wan_comparison",
    "fit_gaussian_surrogates",
    "sample_wan_reference",
    "write_plots",
    "write_report",
]
