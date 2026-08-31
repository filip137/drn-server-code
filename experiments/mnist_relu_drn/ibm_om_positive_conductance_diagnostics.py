"""Read-only diagnostics for calibrated positive-conductance populations."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping

import torch

from experiments.mnist_relu_drn.ibm_om_baseline_selection import quad_stack
from experiments.mnist_relu_drn.ibm_om_positive_conductance import (
    DESTINATION_PAIR_BASELINE,
    QUAD_BASELINE,
    PositiveConductancePopulation,
    build_common_conductance_baseline,
    load_positive_conductance_calibration,
)


DIAGNOSTIC_SCHEMA = (
    "ebl.mnist_relu_drn.ibm_om_positive_conductance_population_diagnostic"
)
DIAGNOSTIC_SCHEMA_VERSION = 1


def _summary(value: torch.Tensor, *, scale: float = 1.0) -> Mapping[str, float]:
    flat = value.detach().to(device="cpu", dtype=torch.float64).reshape(-1)
    if flat.numel() < 1 or not bool(torch.isfinite(flat).all()):
        raise ValueError("Expected a non-empty finite diagnostic tensor.")
    scaled = flat * float(scale)
    return {
        "minimum": float(scaled.min().item()),
        "p05": float(torch.quantile(scaled, 0.05).item()),
        "median": float(torch.quantile(scaled, 0.5).item()),
        "mean": float(scaled.mean().item()),
        "p95": float(torch.quantile(scaled, 0.95).item()),
        "maximum": float(scaled.max().item()),
    }


def _endpoint_report(
    population: PositiveConductancePopulation,
) -> Mapping[str, Any]:
    reset = population.reset_conductance_s
    set_state = population.set_conductance_s
    headroom = torch.abs(set_state - reset)
    low = torch.minimum(reset, set_state)
    high = torch.maximum(reset, set_state)
    dynamic_range = high / low
    return {
        "conductance_unit": "uS",
        "reset": _summary(reset, scale=1e6),
        "set": _summary(set_state, scale=1e6),
        "absolute_RESET_to_SET_headroom": _summary(headroom, scale=1e6),
        "dynamic_range_G_high_over_G_low": _summary(dynamic_range),
    }


def _baseline_report(
    population: PositiveConductancePopulation,
    *,
    policy: str,
    layout: str,
) -> Mapping[str, Any]:
    devices = population.devices
    reset = population.reset_conductance_s.reshape(2, devices // 2)
    set_state = population.set_conductance_s.reshape(2, devices // 2)
    group_count = devices // (2 if policy == DESTINATION_PAIR_BASELINE else 4)
    try:
        baseline = build_common_conductance_baseline(
            reset,
            set_state,
            direction=population.direction,
            policy=policy,
            layout=layout,
            baseline_position_fraction=0.0,
            calibration_sha256=population.calibration_sha256,
            source_sha256=population.source_sha256,
        )
    except ValueError as error:
        return {
            "policy": policy,
            "all_groups_feasible": False,
            "group_count": group_count,
            "feasible_group_count": None,
            "failure": str(error),
            "target_projection": "none",
            "post_handoff_clipping": False,
        }

    baseline_quad = quad_stack(baseline.baseline_s, layout=layout)
    destination_loading = torch.stack(
        (
            baseline_quad[..., 0] + baseline_quad[..., 2],
            baseline_quad[..., 1] + baseline_quad[..., 3],
        ),
        dim=-1,
    )
    full_quad_loading = baseline_quad.sum(dim=-1)
    return {
        "policy": policy,
        "all_groups_feasible": True,
        "group_count": group_count,
        "feasible_group_count": group_count,
        "baseline_position_fraction": 0.0,
        "conductance_unit": "uS",
        "group_RESET_frontier": _summary(
            baseline.group_reset_frontier_s, scale=1e6
        ),
        "group_SET_frontier": _summary(
            baseline.group_set_frontier_s, scale=1e6
        ),
        "group_SET_side_headroom_from_baseline": _summary(
            baseline.group_headroom_s, scale=1e6
        ),
        "baseline_cell_conductance": _summary(
            baseline.baseline_s, scale=1e6
        ),
        "destination_pair_zero_state_loading": _summary(
            destination_loading, scale=1e6
        ),
        "full_quad_zero_state_loading": _summary(
            full_quad_loading, scale=1e6
        ),
        "target_projection": "none",
        "post_handoff_clipping": False,
    }


def diagnose_positive_conductance_population(
    calibration_path: Path | str,
    *,
    devices: int,
    assignment_seed: int,
    layout: str = "halves",
) -> Mapping[str, Any]:
    """Sample and summarize one declared physical-G control population.

    The one-dimensional sampled assignment is deterministically reshaped to a
    ``(2, devices/2)`` rail matrix.  ``devices`` must therefore be a multiple
    of four.  The function is read-only: it performs no programming, network
    mapping, training, fitting, clipping, or artifact writes.
    """

    if (
        isinstance(devices, bool)
        or not isinstance(devices, int)
        or devices < 4
        or devices % 4
    ):
        raise ValueError("Expected devices to be a positive multiple of four.")
    if isinstance(assignment_seed, bool) or not isinstance(assignment_seed, int):
        raise ValueError("Expected assignment_seed to be an integer.")
    calibration = load_positive_conductance_calibration(calibration_path)
    population = calibration.sample_population(
        devices=devices,
        assignment_seed=assignment_seed,
    )
    result = {
        "schema": DIAGNOSTIC_SCHEMA,
        "schema_version": DIAGNOSTIC_SCHEMA_VERSION,
        "analysis_class": "exploratory_read_only_model_based_control",
        "calibration": dict(calibration.report()),
        "population": {
            "devices": devices,
            "assignment_seed": assignment_seed,
            "population_sha256": population.population_sha256,
            "pairing_policy": calibration.pairing_policy,
            "joint_model_assumption": population.report[
                "joint_model_assumption"
            ],
            "identity_semantics": population.report["identity_semantics"],
            "rail_layout": layout,
            "flat_assignment_to_rail_matrix": f"reshape(2,{devices // 2})",
        },
        "endpoints": _endpoint_report(population),
        "baselines": {
            policy: _baseline_report(population, policy=policy, layout=layout)
            for policy in (DESTINATION_PAIR_BASELINE, QUAD_BASELINE)
        },
        "limitations": (
            "The Figure-6 RESET and SET CDFs are unpaired marginals. Per-cell "
            "windows, dynamic ranges, and group feasibility depend on the "
            "declared synthetic copula and are not raw measured identities."
        ),
    }
    for policy, report in result["baselines"].items():
        if report["all_groups_feasible"] and not math.isfinite(
            report["full_quad_zero_state_loading"]["mean"]
        ):
            raise RuntimeError(f"Non-finite baseline diagnostic for {policy}.")
    return result


__all__ = [
    "DIAGNOSTIC_SCHEMA",
    "DIAGNOSTIC_SCHEMA_VERSION",
    "diagnose_positive_conductance_population",
]
