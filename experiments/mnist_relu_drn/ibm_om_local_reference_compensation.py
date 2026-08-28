"""Pure ideal controls for local four-cell reference compensation.

This module never reassigns a sampled device identity and never changes its
intrinsic ``reference`` parameter.  It instead compares two stored baseline
states at the original four-cell quad address:

``local_nearest_symmetry``
    ``B0 = clip((r + 1) / 2, [l, u])`` independently in every cell.

``local_min_l2_exact_zero``
    the nearest in-bound baseline ``B`` to the mapped intrinsic reference,
    subject to ``B++ - B+- - B-+ + B-- = 0`` in every quad.

Continuous logical offsets are then shared exactly between the two policies.
Every branch passed to the circuit is the full physical conductance
``G = B + d``; no baseline is subtracted from transfer or loading.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from experiments.artifacts import atomic_write_json, sha256_file
from experiments.mnist_relu_drn.components import signed_dual_rail_lift
from experiments.mnist_relu_drn.ibm_om_bounded_codebook_scheme_screen import (
    _binding_slices,
    _expanded_quad_minimum,
    _float_summary,
    _native_to_unit,
    _quad_columns,
    _tensor_sha256,
)
from experiments.mnist_relu_drn.ibm_om_four_reference_balance import (
    _inward_float32_bounds,
)
from experiments.mnist_relu_drn.ibm_om_ideal_mapping_scheme_screen import (
    _normalized,
    _quad_contrast,
)
from training.ibm_reram_hwa import IbmReramArrayPopulation


PLAN_SCHEMA = "ebl.mnist_relu_drn.ibm_om_local_reference_baseline_plan"
PLAN_SCHEMA_VERSION = 1
MAPPING_SCHEMA = (
    "ebl.mnist_relu_drn.ibm_om_local_reference_matched_continuous_mapping"
)
MAPPING_SCHEMA_VERSION = 1
CONTROL_POLICY = "local_nearest_symmetry"
COMPENSATED_POLICY = "local_min_l2_exact_zero"
TREATMENT_POLICY = COMPENSATED_POLICY
POLICIES = (CONTROL_POLICY, COMPENSATED_POLICY)
LAYOUTS = ("halves", "paired")
_QUAD_SIGN = torch.tensor((1.0, -1.0, -1.0, 1.0), dtype=torch.float64)


def _tensor_digest(value: torch.Tensor) -> str:
    return _tensor_sha256(value.detach().contiguous().cpu())


def _summary_with_abs(value: torch.Tensor) -> dict[str, Any]:
    result = _float_summary(value)
    result["absolute"] = _float_summary(value.abs())
    return result


def _quad_stack(value: torch.Tensor, *, layout: str) -> torch.Tensor:
    rows, plus, minus = _quad_columns(
        tuple(value.shape), layout, device=value.device
    )
    return torch.stack(
        (
            value[:rows][:, plus],
            value[:rows][:, minus],
            value[rows:][:, plus],
            value[rows:][:, minus],
        ),
        dim=-1,
    )


def _scatter_quads(
    quads: torch.Tensor,
    *,
    shape: tuple[int, int],
    layout: str,
) -> torch.Tensor:
    if quads.ndim != 3 or quads.shape[-1] != 4:
        raise ValueError("Expected logical-row by logical-column by four values.")
    rows, plus, minus = _quad_columns(shape, layout, device=quads.device)
    if quads.shape[:2] != (rows, int(plus.numel())):
        raise ValueError("Expected quad values to match the declared rail matrix.")
    result = torch.empty(shape, dtype=quads.dtype, device=quads.device)
    result[:rows][:, plus] = quads[..., 0]
    result[:rows][:, minus] = quads[..., 1]
    result[rows:][:, plus] = quads[..., 2]
    result[rows:][:, minus] = quads[..., 3]
    return result


def _quad_loading_components(quads: torch.Tensor) -> Mapping[str, torch.Tensor]:
    return {
        "row": torch.stack(
            (quads[..., 0] + quads[..., 1], quads[..., 2] + quads[..., 3]),
            dim=-1,
        ),
        "column": torch.stack(
            (quads[..., 0] + quads[..., 2], quads[..., 1] + quads[..., 3]),
            dim=-1,
        ),
        "total": quads.sum(dim=-1),
    }


def _validate_quad_inputs(
    reference: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    values = tuple(value.detach().cpu().to(torch.float64) for value in (
        reference,
        lower,
        upper,
    ))
    reference64, lower64, upper64 = values
    if (
        reference64.ndim < 1
        or reference64.shape[-1] != 4
        or lower64.shape != reference64.shape
        or upper64.shape != reference64.shape
        or not all(bool(torch.isfinite(value).all()) for value in values)
        or bool(torch.any(lower64 > upper64))
    ):
        raise ValueError("Expected finite ordered (...,4) reference and bounds.")
    return reference64, lower64, upper64


def _feasible_zero_interval(
    lower: torch.Tensor,
    upper: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    # Minimum/maximum attainable s.B for s=(+,-,-,+).
    minimum = lower[..., 0] - upper[..., 1] - upper[..., 2] + lower[..., 3]
    maximum = upper[..., 0] - lower[..., 1] - lower[..., 2] + upper[..., 3]
    return minimum, maximum


def project_local_exact_zero(
    reference: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
    *,
    tolerance: float = 1e-12,
) -> tuple[torch.Tensor, Mapping[str, torch.Tensor]]:
    """Project each local quad onto its in-bound exact-zero hyperplane.

    The KKT solution is ``B_i(lambda)=clip(rho_i-lambda*s_i,[l_i,u_i])``.
    A fixed-iteration vectorized bisection makes the solver deterministic.
    The final sub-ulp equality repair is distributed over free coordinates;
    it affects only floating-point roundoff and preserves the KKT solution.
    """

    reference64, lower64, upper64 = _validate_quad_inputs(
        reference, lower, upper
    )
    if not math.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("Expected a finite positive projection tolerance.")
    sign = _QUAD_SIGN.to(reference64.device)
    minimum, maximum = _feasible_zero_interval(lower64, upper64)
    feasible = (minimum <= tolerance) & (maximum >= -tolerance)
    if not bool(feasible.all()):
        count = int((~feasible).sum().item())
        raise ValueError(
            "Exact-zero local baseline is infeasible for "
            f"{count} quad(s) under the supplied cell bounds."
        )

    raw_contrast = (reference64 * sign).sum(dim=-1)
    unconstrained = reference64 - (raw_contrast / 4.0).unsqueeze(-1) * sign
    unconstrained_in_bounds = (
        (unconstrained >= lower64) & (unconstrained <= upper64)
    ).all(dim=-1)

    # At lambda_low every positive-sign coordinate is at its upper bound and
    # every negative-sign coordinate is at its lower bound.  lambda_high has
    # the opposite saturation pattern.
    lower_breaks = torch.where(
        sign > 0.0, reference64 - upper64, lower64 - reference64
    )
    upper_breaks = torch.where(
        sign > 0.0, reference64 - lower64, upper64 - reference64
    )
    lambda_low = lower_breaks.min(dim=-1).values
    lambda_high = upper_breaks.max(dim=-1).values
    for _ in range(128):
        midpoint = (lambda_low + lambda_high) / 2.0
        candidate = torch.minimum(
            torch.maximum(
                reference64 - midpoint.unsqueeze(-1) * sign,
                lower64,
            ),
            upper64,
        )
        contrast = (candidate * sign).sum(dim=-1)
        lambda_low = torch.where(contrast > 0.0, midpoint, lambda_low)
        lambda_high = torch.where(contrast > 0.0, lambda_high, midpoint)

    multiplier = (lambda_low + lambda_high) / 2.0
    projected = torch.minimum(
        torch.maximum(
            reference64 - multiplier.unsqueeze(-1) * sign,
            lower64,
        ),
        upper64,
    )

    # Remove the last arithmetic residual while retaining the same free-set
    # direction.  Any subsequent capacity fallback is only for a degenerate
    # all-bound feasible point and is deterministic in (++,+-,-+,--) order.
    residual = (projected * sign).sum(dim=-1)
    free = (projected > lower64) & (projected < upper64)
    free_count = free.sum(dim=-1)
    distributed = torch.where(
        free_count > 0,
        residual / free_count.clamp_min(1).to(torch.float64),
        torch.zeros_like(residual),
    )
    projected = projected - distributed.unsqueeze(-1) * sign * free
    projected = torch.minimum(torch.maximum(projected, lower64), upper64)

    for cell in range(4):
        residual = (projected * sign).sum(dim=-1)
        residual_sign = torch.sign(residual)
        if cell in (0, 3):
            positive_capacity = projected[..., cell] - lower64[..., cell]
            negative_capacity = upper64[..., cell] - projected[..., cell]
        else:
            positive_capacity = upper64[..., cell] - projected[..., cell]
            negative_capacity = projected[..., cell] - lower64[..., cell]
        capacity = torch.where(
            residual_sign >= 0.0, positive_capacity, negative_capacity
        ).clamp_min(0.0)
        amount = torch.minimum(residual.abs(), capacity)
        projected[..., cell] = projected[..., cell] - (
            residual_sign * sign[cell] * amount
        )

    residual = (projected * sign).sum(dim=-1)
    violation = (projected < lower64) | (projected > upper64)
    if bool(violation.any()) or float(residual.abs().max().item()) > tolerance:
        raise RuntimeError(
            "Exact-zero box projection failed its bounds/equality invariant."
        )

    at_lower = torch.isclose(projected, lower64, rtol=0.0, atol=tolerance)
    at_upper = torch.isclose(projected, upper64, rtol=0.0, atol=tolerance)
    diagnostics = {
        "feasible": feasible,
        "feasible_minimum_contrast": minimum,
        "feasible_maximum_contrast": maximum,
        "raw_contrast": raw_contrast,
        "unconstrained": unconstrained,
        "unconstrained_in_bounds": unconstrained_in_bounds,
        "bound_active_quad": ~unconstrained_in_bounds,
        "at_lower": at_lower,
        "at_upper": at_upper,
        "multiplier": multiplier,
        "zero_residual": residual,
    }
    return projected, diagnostics


@dataclass(frozen=True)
class LocalBaselinePlan:
    """One immutable-identity local baseline policy."""

    policy: str
    assignment_seed: int
    population_fingerprint: str
    baseline_unit: torch.Tensor
    report: Mapping[str, Any]
    components: tuple[Mapping[str, torch.Tensor], ...]

    def __post_init__(self) -> None:
        value = self.baseline_unit
        if (
            self.policy not in POLICIES
            or value.ndim != 1
            or value.dtype != torch.float64
            or value.device.type != "cpu"
            or not bool(torch.isfinite(value).all())
        ):
            raise ValueError("Expected a canonical CPU float64 local baseline plan.")


def _loading_change_report(
    baseline_quads: torch.Tensor,
    origin_quads: torch.Tensor,
) -> Mapping[str, Any]:
    baseline_loading = _quad_loading_components(baseline_quads)
    origin_loading = _quad_loading_components(origin_quads)
    return {
        name: _summary_with_abs(baseline_loading[name] - origin_loading[name])
        for name in ("row", "column", "total")
    }


def build_local_baseline_plan(
    population: IbmReramArrayPopulation,
    *,
    policy: str,
    layouts: Sequence[str] = LAYOUTS,
    zero_tolerance: float = 1e-12,
) -> LocalBaselinePlan:
    """Build one local baseline plan without changing identity binding or ``r``."""

    if policy not in POLICIES:
        raise ValueError(f"Expected baseline policy to be one of {POLICIES!r}.")
    if tuple(layouts) != LAYOUTS or len(layouts) != len(population.binding_keys):
        raise ValueError("Expected the canonical W1-halves/W2-paired layout.")
    lower = _native_to_unit(population.min_bound.detach().cpu().to(torch.float64))
    upper = _native_to_unit(population.max_bound.detach().cpu().to(torch.float64))
    raw_reference = population.reference.detach().cpu().to(torch.float64)
    reference = _native_to_unit(population.reference.detach().cpu().to(torch.float64))
    reachable_control = torch.minimum(torch.maximum(reference, lower), upper)
    slices = _binding_slices(population)
    baseline_flat = torch.empty_like(reference)
    layer_reports: list[Mapping[str, Any]] = []
    components: list[Mapping[str, torch.Tensor]] = []
    offset = 0
    for layer_index, (key, shape, layout) in enumerate(
        zip(population.binding_keys, population.binding_shapes, layouts)
    ):
        binding_slice = slices[key]
        cell_raw_reference = raw_reference[binding_slice].reshape(shape)
        cell_reference = reference[binding_slice].reshape(shape)
        cell_lower = lower[binding_slice].reshape(shape)
        cell_upper = upper[binding_slice].reshape(shape)
        cell_control = reachable_control[binding_slice].reshape(shape)
        reference_quads = _quad_stack(cell_reference, layout=layout)
        lower_quads = _quad_stack(cell_lower, layout=layout)
        upper_quads = _quad_stack(cell_upper, layout=layout)
        control_quads = _quad_stack(cell_control, layout=layout)
        minimum, maximum = _feasible_zero_interval(lower_quads, upper_quads)
        feasible = (minimum <= zero_tolerance) & (maximum >= -zero_tolerance)
        if not bool(feasible.all()):
            count = int((~feasible).sum().item())
            raise ValueError(
                f"Layer {key!r} has {count} exact-zero-infeasible local quad(s)."
            )
        raw_contrast = (reference_quads * _QUAD_SIGN).sum(dim=-1)
        unconstrained = reference_quads - (
            raw_contrast / 4.0
        ).unsqueeze(-1) * _QUAD_SIGN
        unconstrained_in_bounds = (
            (unconstrained >= lower_quads) & (unconstrained <= upper_quads)
        ).all(dim=-1)
        if policy == CONTROL_POLICY:
            baseline_quads = control_quads
            projection = {
                "feasible_minimum_contrast": minimum,
                "feasible_maximum_contrast": maximum,
                "unconstrained": unconstrained,
                "unconstrained_in_bounds": unconstrained_in_bounds,
                "bound_active_quad": torch.zeros_like(
                    unconstrained_in_bounds, dtype=torch.bool
                ),
                "at_lower": baseline_quads == lower_quads,
                "at_upper": baseline_quads == upper_quads,
                "multiplier": torch.zeros_like(raw_contrast),
                "zero_residual": (baseline_quads * _QUAD_SIGN).sum(dim=-1),
            }
        else:
            baseline_quads, projection = project_local_exact_zero(
                reference_quads,
                lower_quads,
                upper_quads,
                tolerance=zero_tolerance,
            )
        cell_baseline = _scatter_quads(
            baseline_quads, shape=tuple(shape), layout=layout
        )
        baseline_flat[binding_slice] = cell_baseline.reshape(-1)
        correction = cell_baseline - cell_reference
        correction_from_control = cell_baseline - cell_control
        baseline_contrast = (baseline_quads * _QUAD_SIGN).sum(dim=-1)
        identity_indices = torch.arange(
            offset, offset + math.prod(shape), dtype=torch.int64
        )
        layer_reports.append(
            {
                "layer": layer_index,
                "binding_key": key,
                "shape": list(shape),
                "layout": layout,
                "identity_binding": "sampled_order_no_permutation",
                "identity_indices_sha256": _tensor_digest(identity_indices),
                "intrinsic_reference_is_immutable": True,
                "baseline_definition": (
                    "clip(mapped_intrinsic_reference,cell_bounds)"
                    if policy == CONTROL_POLICY
                    else "box_constrained_min_l2_exact_zero_projection"
                ),
                "projection_objective": (
                    None
                    if policy == CONTROL_POLICY
                    else "minimize_0.5_l2_squared_from_mapped_intrinsic_reference"
                ),
                "quad_sign_order": [1, -1, -1, 1],
                "quad_count": int(baseline_quads.numel() // 4),
                "exact_zero_feasible_count": int(feasible.sum().item()),
                "exact_zero_infeasible_count": int((~feasible).sum().item()),
                "feasible_minimum_contrast": _float_summary(minimum),
                "feasible_maximum_contrast": _float_summary(maximum),
                "reference_outside_active_bounds_count": int(
                    ((cell_reference < cell_lower) | (cell_reference > cell_upper))
                    .sum()
                    .item()
                ),
                "analytic_unconstrained_in_bounds_count": int(
                    unconstrained_in_bounds.sum().item()
                ),
                "bound_active_projection_quad_count": int(
                    projection["bound_active_quad"].sum().item()
                ),
                "baseline_at_lower_count": int(projection["at_lower"].sum().item()),
                "baseline_at_upper_count": int(projection["at_upper"].sum().item()),
                "intrinsic_reference_contrast": _summary_with_abs(raw_contrast),
                "reachable_control_contrast": _summary_with_abs(
                    (control_quads * _QUAD_SIGN).sum(dim=-1)
                ),
                "baseline_contrast": _summary_with_abs(baseline_contrast),
                "exact_zero_max_abs_residual": float(
                    baseline_contrast.abs().max().item()
                ),
                "projection_multiplier_applies": policy == COMPENSATED_POLICY,
                "projection_multiplier": (
                    None
                    if policy == CONTROL_POLICY
                    else _float_summary(projection["multiplier"])
                ),
                "correction_from_intrinsic_reference": _summary_with_abs(
                    correction
                ),
                "correction_from_reachable_control": _summary_with_abs(
                    correction_from_control
                ),
                "loading_change_from_intrinsic_reference": _loading_change_report(
                    baseline_quads, reference_quads
                ),
                "loading_change_from_reachable_control": _loading_change_report(
                    baseline_quads, control_quads
                ),
                "hashes": {
                    "raw_intrinsic_reference": _tensor_digest(cell_raw_reference),
                    "intrinsic_reference": _tensor_digest(cell_reference),
                    "lower_bound": _tensor_digest(cell_lower),
                    "upper_bound": _tensor_digest(cell_upper),
                    "reachable_control_baseline": _tensor_digest(cell_control),
                    "baseline": _tensor_digest(cell_baseline),
                    "correction": _tensor_digest(correction),
                    "bound_active_quad": _tensor_digest(
                        projection["bound_active_quad"]
                    ),
                },
            }
        )
        components.append(
            {
                "raw_intrinsic_reference": cell_raw_reference,
                "intrinsic_reference": cell_reference,
                "lower_bound": cell_lower,
                "upper_bound": cell_upper,
                "reachable_control_baseline": cell_control,
                "baseline": cell_baseline,
                "correction": correction,
                "correction_from_reachable_control": correction_from_control,
                "baseline_contrast": baseline_contrast,
                "feasible_minimum_contrast": minimum,
                "feasible_maximum_contrast": maximum,
                "projection_multiplier": projection["multiplier"],
                "bound_active_quad": projection["bound_active_quad"],
                "at_lower": projection["at_lower"],
                "at_upper": projection["at_upper"],
            }
        )
        offset += math.prod(shape)
    if offset != population.size:
        raise RuntimeError("Expected local baseline plan to cover every sampled cell.")
    binding_sha = _tensor_digest(torch.arange(population.size, dtype=torch.int64))
    report = {
        "schema": PLAN_SCHEMA,
        "schema_version": PLAN_SCHEMA_VERSION,
        "policy": policy,
        "assignment_seed": population.assignment_seed,
        "population_fingerprint": population.fingerprint,
        "conductance_coordinate": "clip((r+1)/2,0,1)",
        "identity_binding": "sampled_order_no_permutation",
        "destination_to_source_sha256": binding_sha,
        "intrinsic_reference_is_immutable": True,
        "uses_weights": False,
        "uses_labels": False,
        "uses_accuracy": False,
        "layers": layer_reports,
        "baseline_sha256": _tensor_digest(baseline_flat),
    }
    return LocalBaselinePlan(
        policy=policy,
        assignment_seed=population.assignment_seed,
        population_fingerprint=population.fingerprint,
        baseline_unit=baseline_flat,
        report=report,
        components=tuple(components),
    )


def save_local_baseline_plan(
    directory: Path,
    population: IbmReramArrayPopulation,
    plan: LocalBaselinePlan,
) -> Mapping[str, Any]:
    """Persist the immutable binding and all local baseline ingredients."""

    _validate_plan(population, plan, expected_policy=plan.policy)
    destination = directory.expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    stem = f"{plan.policy}-assignment-{population.assignment_seed}"
    artifact = destination / f"{stem}.npz"
    receipt_path = destination / f"{stem}.receipt.json"
    arrays: dict[str, np.ndarray] = {
        "schema": np.asarray(PLAN_SCHEMA),
        "schema_version": np.asarray(PLAN_SCHEMA_VERSION, dtype=np.int64),
        "policy": np.asarray(plan.policy),
        "assignment_seed": np.asarray(plan.assignment_seed, dtype=np.int64),
        "population_fingerprint": np.asarray(plan.population_fingerprint),
        "identity_order": np.arange(population.size, dtype=np.int64),
        "baseline_unit": plan.baseline_unit.numpy(),
    }
    expected_fields = (
        "raw_intrinsic_reference",
        "intrinsic_reference",
        "lower_bound",
        "upper_bound",
        "reachable_control_baseline",
        "baseline",
        "correction",
        "correction_from_reachable_control",
        "baseline_contrast",
        "feasible_minimum_contrast",
        "feasible_maximum_contrast",
        "projection_multiplier",
        "bound_active_quad",
        "at_lower",
        "at_upper",
    )
    for layer_index, layer in enumerate(plan.components):
        if set(layer) != set(expected_fields):
            raise ValueError("Expected exact local baseline component fields.")
        for name in expected_fields:
            arrays[f"layer_{layer_index}_{name}"] = (
                layer[name].detach().contiguous().cpu().numpy()
            )
    with tempfile.NamedTemporaryFile(
        dir=destination, prefix=f".{stem}.", suffix=".npz", delete=False
    ) as handle:
        temporary = Path(handle.name)
        np.savez_compressed(handle, **arrays)
        handle.flush()
    temporary.replace(artifact)
    receipt = {
        **dict(plan.report),
        "artifact": artifact.name,
        "artifact_sha256": sha256_file(artifact),
        "binding_keys": list(population.binding_keys),
        "binding_shapes": [list(shape) for shape in population.binding_shapes],
    }
    atomic_write_json(receipt_path, receipt)
    return {
        "path": str(artifact),
        "sha256": sha256_file(artifact),
        "receipt": str(receipt_path),
        "receipt_sha256": sha256_file(receipt_path),
        "policy": plan.policy,
        "identity_order_sha256": plan.report["destination_to_source_sha256"],
        "baseline_sha256": plan.report["baseline_sha256"],
    }


def _validate_plan(
    population: IbmReramArrayPopulation,
    plan: LocalBaselinePlan,
    *,
    expected_policy: str,
) -> None:
    if (
        plan.policy != expected_policy
        or plan.assignment_seed != population.assignment_seed
        or plan.population_fingerprint != population.fingerprint
        or plan.baseline_unit.numel() != population.size
    ):
        raise ValueError(
            "Expected the local baseline plan policy and population provenance "
            "to match."
        )


def _round_nonnegative_float32_down(value: torch.Tensor) -> torch.Tensor:
    if value.dtype != torch.float64 or bool(torch.any(value < 0.0)):
        raise ValueError("Expected a nonnegative float64 offset request.")
    rounded = value.to(torch.float32)
    negative_infinity = torch.full_like(rounded, float("-inf"))
    return torch.where(
        rounded.to(torch.float64) > value,
        torch.nextafter(rounded, negative_infinity),
        rounded,
    )


@dataclass(frozen=True)
class MatchedContinuousTargets:
    """Two baseline arms mapped with one bit-identical offset tensor."""

    targets_by_policy: Mapping[str, tuple[torch.Tensor, ...]]
    reports_by_policy: Mapping[str, Mapping[str, Any]]
    components_by_policy: Mapping[
        str, tuple[Mapping[str, torch.Tensor], ...]
    ]
    shared_report: Mapping[str, Any]

    def __post_init__(self) -> None:
        expected = set(POLICIES)
        if (
            set(self.targets_by_policy) != expected
            or set(self.reports_by_policy) != expected
            or set(self.components_by_policy) != expected
        ):
            raise ValueError("Expected exact control and compensated mapping arms.")


def _build_matched_continuous_targets(
    logical_weights: Sequence[torch.Tensor],
    population: IbmReramArrayPopulation,
    control_plan: LocalBaselinePlan,
    compensated_plan: LocalBaselinePlan,
    *,
    scale_fractions: tuple[float, float],
    conductance_min: float,
    conductance_max: float,
) -> MatchedContinuousTargets:
    """Map both policies with identical ``d`` inside their common headroom."""

    _validate_plan(population, control_plan, expected_policy=CONTROL_POLICY)
    _validate_plan(
        population, compensated_plan, expected_policy=COMPENSATED_POLICY
    )
    if len(logical_weights) != 2 or len(scale_fractions) != 2:
        raise ValueError("Expected exactly two logical layers and scale fractions.")
    if not (
        math.isfinite(conductance_min)
        and math.isfinite(conductance_max)
        and 0.0 <= conductance_min < conductance_max
    ):
        raise ValueError("Expected finite nonnegative increasing conductance bounds.")
    span = float(conductance_max - conductance_min)
    lower_unit = _native_to_unit(
        population.min_bound.detach().cpu().to(torch.float64)
    )
    upper_unit = _native_to_unit(
        population.max_bound.detach().cpu().to(torch.float64)
    )
    slices = _binding_slices(population)
    targets: dict[str, list[torch.Tensor]] = {policy: [] for policy in POLICIES}
    reports: dict[str, list[Mapping[str, Any]]] = {
        policy: [] for policy in POLICIES
    }
    components: dict[str, list[Mapping[str, torch.Tensor]]] = {
        policy: [] for policy in POLICIES
    }
    shared_layers: list[Mapping[str, Any]] = []
    for layer_index, (logical, fraction, layout, key, shape) in enumerate(
        zip(
            logical_weights,
            scale_fractions,
            LAYOUTS,
            population.binding_keys,
            population.binding_shapes,
        )
    ):
        fraction = float(fraction)
        if not math.isfinite(fraction) or not 0.0 < fraction <= 1.0:
            raise ValueError("Expected every scale fraction in (0,1].")
        normalized = _normalized(logical.detach().cpu()).to(torch.float64)
        lifted = signed_dual_rail_lift(normalized, target_layout=layout)
        binding_slice = slices[key]
        cell_lower_unit = lower_unit[binding_slice].reshape(shape)
        cell_upper_unit = upper_unit[binding_slice].reshape(shape)
        physical_lower = conductance_min + span * cell_lower_unit
        physical_upper = conductance_min + span * cell_upper_unit
        lower_inward, upper_inward = _inward_float32_bounds(
            physical_lower, physical_upper
        )
        physical_baselines: dict[str, torch.Tensor] = {}
        for policy, plan in (
            (CONTROL_POLICY, control_plan),
            (COMPENSATED_POLICY, compensated_plan),
        ):
            unit_baseline = plan.baseline_unit[binding_slice].reshape(shape)
            raw = (conductance_min + span * unit_baseline).to(torch.float32)
            physical_baselines[policy] = torch.minimum(
                torch.maximum(raw, lower_inward), upper_inward
            )
        cell_common_headroom = torch.minimum(
            upper_inward.to(torch.float64)
            - physical_baselines[CONTROL_POLICY].to(torch.float64),
            upper_inward.to(torch.float64)
            - physical_baselines[COMPENSATED_POLICY].to(torch.float64),
        ).clamp_min(0.0)
        common_headroom = _expanded_quad_minimum(
            cell_common_headroom, layout=layout
        )
        desired_offset = fraction * common_headroom * lifted
        physical_offset = _round_nonnegative_float32_down(desired_offset)
        inward_offset_adjustment_count = 0
        inward_offset_adjustment_max = 0.0
        # ``d`` is one shared physical tensor.  If float32 B+d rounds above a
        # bound in either arm, step every branch of that affected local quad
        # one representable value toward zero and recheck both arms.  This is
        # deterministic, preserves the rail-pair symmetry of d, and cannot
        # silently turn a matched comparison into arm-specific clipping.
        for _ in range(32):
            violation_by_policy = []
            for baseline in physical_baselines.values():
                trial = baseline + physical_offset
                violation_by_policy.append(
                    (trial.to(torch.float64) < physical_lower)
                    | (trial.to(torch.float64) > physical_upper)
                )
            union_violation = torch.stack(violation_by_policy).any(dim=0)
            if not bool(union_violation.any()):
                break
            bad_quads = _quad_stack(
                union_violation, layout=layout
            ).any(dim=-1)
            offset_quads = _quad_stack(physical_offset, layout=layout)
            stepped_quads = torch.nextafter(
                offset_quads, torch.zeros_like(offset_quads)
            )
            adjusted_quads = torch.where(
                bad_quads.unsqueeze(-1), stepped_quads, offset_quads
            )
            adjustment = (adjusted_quads - offset_quads).abs()
            inward_offset_adjustment_count += int(
                (adjustment > 0.0).sum().item()
            )
            inward_offset_adjustment_max = max(
                inward_offset_adjustment_max,
                float(adjustment.max().item()),
            )
            physical_offset = _scatter_quads(
                adjusted_quads, shape=tuple(shape), layout=layout
            )
        else:
            raise RuntimeError(
                "Could not construct one float32 matched offset inside both arms."
            )
        offset_hash = _tensor_digest(physical_offset)
        common_quad = _quad_stack(common_headroom, layout=layout)[..., 0]
        shared_layers.append(
            {
                "layer": layer_index,
                "binding_key": key,
                "layout": layout,
                "scale_fraction": fraction,
                "headroom_rule": (
                    "quad_minimum_over_all_four_cells_and_both_baseline_policies"
                ),
                "common_matched_headroom": _float_summary(common_quad),
                "zero_common_headroom_count": int((common_quad <= 0.0).sum().item()),
                "float32_inward_offset_adjustment_count": (
                    inward_offset_adjustment_count
                ),
                "float32_inward_offset_adjustment_max_abs": (
                    inward_offset_adjustment_max
                ),
                "offset_sha256": offset_hash,
            }
        )
        for policy in POLICIES:
            baseline = physical_baselines[policy]
            conductance = baseline + physical_offset
            residual = conductance - (baseline + physical_offset)
            violation = (
                (conductance.to(torch.float64) < physical_lower)
                | (conductance.to(torch.float64) > physical_upper)
            )
            if (
                not bool(torch.isfinite(conductance).all())
                or bool(torch.any(conductance < 0.0))
                or bool(violation.any())
                or not torch.equal(conductance, baseline + physical_offset)
            ):
                raise RuntimeError(
                    "Matched continuous mapping violated G=B+d or cell bounds."
                )
            baseline_contrast = _quad_contrast(baseline, layout=layout) / 2.0
            offset_contrast = _quad_contrast(
                physical_offset, layout=layout
            ) / 2.0
            realized_contrast = _quad_contrast(conductance, layout=layout) / 2.0
            loading = _quad_stack(conductance, layout=layout).sum(dim=-1)
            baseline_loading = _quad_stack(baseline, layout=layout).sum(dim=-1)
            nonzero = normalized != 0.0
            sign_flips = nonzero & (
                torch.sign(realized_contrast) != torch.sign(normalized)
            )
            reports[policy].append(
                {
                    "layer": layer_index,
                    "binding_key": key,
                    "shape": list(shape),
                    "layout": layout,
                    "scale_fraction": fraction,
                    "identity_binding": "sampled_order_no_permutation",
                    "mapping": "continuous_positive_only_common_matched_headroom",
                    "baseline_contrast": _summary_with_abs(baseline_contrast),
                    "offset_contrast": _summary_with_abs(offset_contrast),
                    "realized_contrast": _summary_with_abs(realized_contrast),
                    "baseline_loading": _float_summary(baseline_loading),
                    "quad_loading": _float_summary(loading),
                    "logical_sign_flip_count": int(sign_flips.sum().item()),
                    "logical_sign_flip_fraction_nonzero": (
                        float(sign_flips.sum().item()) / int(nonzero.sum().item())
                    ),
                    "full_g_decomposition_max_abs_residual": float(
                        residual.abs().max().item()
                    ),
                    "bound_violation_count": int(violation.sum().item()),
                    "hashes": {
                        "baseline": _tensor_digest(baseline),
                        "offset": offset_hash,
                        "conductance": _tensor_digest(conductance),
                        "baseline_contrast": _tensor_digest(baseline_contrast),
                        "realized_contrast": _tensor_digest(realized_contrast),
                        "loading": _tensor_digest(loading),
                        "common_headroom": _tensor_digest(
                            common_headroom.to(torch.float32)
                        ),
                    },
                }
            )
            targets[policy].append(conductance)
            components[policy].append(
                {
                    "baseline": baseline,
                    "offset": physical_offset,
                    "conductance": conductance,
                    "baseline_contrast": baseline_contrast.to(torch.float32),
                    "offset_contrast": offset_contrast.to(torch.float32),
                    "realized_contrast": realized_contrast.to(torch.float32),
                    "loading": loading.to(torch.float32),
                    "active_mask": lifted > 0.0,
                    "common_headroom": common_headroom.to(torch.float32),
                }
            )
    reports_by_policy = {
        policy: {
            "schema": MAPPING_SCHEMA,
            "schema_version": MAPPING_SCHEMA_VERSION,
            "baseline_policy": policy,
            "population_fingerprint": population.fingerprint,
            "assignment_seed": population.assignment_seed,
            "identity_binding": "sampled_order_no_permutation",
            "intrinsic_reference_is_immutable": True,
            "decomposition": "every physical branch G=B+d",
            "circuit_accounting": "full G enters numerator and denominator",
            "offset_matching": "bit_identical_between_baseline_policies",
            "scale_fractions": [float(value) for value in scale_fractions],
            "layers": reports[policy],
            "target_hashes": [_tensor_digest(value) for value in targets[policy]],
        }
        for policy in POLICIES
    }
    shared_report = {
        "schema": MAPPING_SCHEMA,
        "schema_version": MAPPING_SCHEMA_VERSION,
        "population_fingerprint": population.fingerprint,
        "assignment_seed": population.assignment_seed,
        "policies": list(POLICIES),
        "matched_variable": "offset_d",
        "identity_binding": "sampled_order_no_permutation",
        "layers": shared_layers,
        "offset_hashes_equal": all(
            reports_by_policy[CONTROL_POLICY]["layers"][index]["hashes"]["offset"]
            == reports_by_policy[COMPENSATED_POLICY]["layers"][index]["hashes"]["offset"]
            for index in range(2)
        ),
    }
    if not shared_report["offset_hashes_equal"]:
        raise RuntimeError("Expected the two mapping arms to use identical offsets.")
    return MatchedContinuousTargets(
        targets_by_policy={
            policy: tuple(values) for policy, values in targets.items()
        },
        reports_by_policy=reports_by_policy,
        components_by_policy={
            policy: tuple(values) for policy, values in components.items()
        },
        shared_report=shared_report,
    )


def build_matched_continuous_targets(
    logical_weights: Sequence[torch.Tensor],
    population: IbmReramArrayPopulation,
    plans_or_control: Mapping[str, LocalBaselinePlan] | LocalBaselinePlan,
    compensated_plan: LocalBaselinePlan | None = None,
    *,
    policy: str | None = None,
    scale_fractions: tuple[float, float],
    conductance_min: float,
    conductance_max: float,
) -> (
    MatchedContinuousTargets
    | tuple[
        tuple[torch.Tensor, ...],
        Mapping[str, Any],
        tuple[Mapping[str, torch.Tensor], ...],
    ]
):
    """Build both arms, or select one arm for prior-style runtime use.

    Passing a policy-keyed mapping is the preferred runtime interface.  The
    explicit two-plan form remains convenient for mathematical unit tests.
    When ``policy`` is supplied, the selected report embeds the shared
    matching receipt and the return shape is ``(targets, report, components)``.
    """

    if isinstance(plans_or_control, Mapping):
        if compensated_plan is not None:
            raise ValueError("Do not mix a plan mapping with an explicit plan.")
        if set(plans_or_control) != set(POLICIES):
            raise ValueError("Expected exact control and compensated plans.")
        control = plans_or_control[CONTROL_POLICY]
        treatment = plans_or_control[COMPENSATED_POLICY]
    else:
        if compensated_plan is None:
            raise ValueError("Expected both explicit local baseline plans.")
        control = plans_or_control
        treatment = compensated_plan
    matched = _build_matched_continuous_targets(
        logical_weights,
        population,
        control,
        treatment,
        scale_fractions=scale_fractions,
        conductance_min=conductance_min,
        conductance_max=conductance_max,
    )
    if policy is None:
        return matched
    if policy not in POLICIES:
        raise ValueError(f"Expected mapping policy to be one of {POLICIES!r}.")
    selected_report = {
        **dict(matched.reports_by_policy[policy]),
        "shared_matching": dict(matched.shared_report),
    }
    return (
        matched.targets_by_policy[policy],
        selected_report,
        matched.components_by_policy[policy],
    )


def save_continuous_mapping(
    directory: Path,
    *,
    assignment_seed: int,
    baseline_policy: str,
    calibration_policy: str,
    mapping_report: Mapping[str, Any],
    components: Sequence[Mapping[str, torch.Tensor]],
) -> Mapping[str, Any]:
    """Persist full physical ``B``, matched ``d``, and deployed ``G``."""

    if baseline_policy not in POLICIES or calibration_policy not in POLICIES:
        raise ValueError("Expected canonical baseline and calibration policies.")
    if len(components) != 2:
        raise ValueError("Expected two mapped layer component records.")
    destination = directory.expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    stem = (
        f"assignment-{assignment_seed}-{baseline_policy}"
        f"-calibrated-by-{calibration_policy}"
    )
    artifact = destination / f"{stem}.npz"
    receipt_path = destination / f"{stem}.receipt.json"
    arrays: dict[str, np.ndarray] = {
        "schema": np.asarray(MAPPING_SCHEMA),
        "schema_version": np.asarray(MAPPING_SCHEMA_VERSION, dtype=np.int64),
        "assignment_seed": np.asarray(assignment_seed, dtype=np.int64),
        "baseline_policy": np.asarray(baseline_policy),
        "calibration_policy": np.asarray(calibration_policy),
    }
    expected_fields = (
        "baseline",
        "offset",
        "conductance",
        "baseline_contrast",
        "offset_contrast",
        "realized_contrast",
        "loading",
        "active_mask",
        "common_headroom",
    )
    for layer_index, layer in enumerate(components):
        if set(layer) != set(expected_fields):
            raise ValueError("Expected exact matched continuous mapping fields.")
        for name in expected_fields:
            arrays[f"layer_{layer_index}_{name}"] = (
                layer[name].detach().contiguous().cpu().numpy()
            )
    with tempfile.NamedTemporaryFile(
        dir=destination, prefix=f".{stem}.", suffix=".npz", delete=False
    ) as handle:
        temporary = Path(handle.name)
        np.savez_compressed(handle, **arrays)
        handle.flush()
    temporary.replace(artifact)
    receipt = {
        "schema": MAPPING_SCHEMA,
        "schema_version": MAPPING_SCHEMA_VERSION,
        "assignment_seed": assignment_seed,
        "baseline_policy": baseline_policy,
        "calibration_policy": calibration_policy,
        "artifact": artifact.name,
        "artifact_sha256": sha256_file(artifact),
        "mapping": dict(mapping_report),
    }
    atomic_write_json(receipt_path, receipt)
    return {
        "path": str(artifact),
        "sha256": sha256_file(artifact),
        "receipt": str(receipt_path),
        "receipt_sha256": sha256_file(receipt_path),
        "assignment_seed": assignment_seed,
        "baseline_policy": baseline_policy,
        "calibration_policy": calibration_policy,
    }


__all__ = [
    "COMPENSATED_POLICY",
    "CONTROL_POLICY",
    "TREATMENT_POLICY",
    "LAYOUTS",
    "MAPPING_SCHEMA",
    "MAPPING_SCHEMA_VERSION",
    "MatchedContinuousTargets",
    "PLAN_SCHEMA",
    "PLAN_SCHEMA_VERSION",
    "POLICIES",
    "LocalBaselinePlan",
    "build_local_baseline_plan",
    "build_matched_continuous_targets",
    "project_local_exact_zero",
    "save_continuous_mapping",
    "save_local_baseline_plan",
]
