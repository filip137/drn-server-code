"""Pure mapping helpers for four-device intrinsic-reference balancing.

The intervention in this module changes only the binding between sampled OM
device identities and four-cell rail positions.  A complete identity (bounds,
reference, pulse parameters, and corruption metadata) always moves together.
The mapper then forms every physical conductance explicitly as ``G = B + d``;
the full ``G`` tensor is passed to the resistive solver without subtracting a
baseline from either transfer or denominator loading.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
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
from experiments.mnist_relu_drn.ibm_om_ideal_mapping_scheme_screen import (
    _normalized,
    _quad_contrast,
)
from training.ibm_reram_hwa import IbmReramArrayPopulation
from training.ibm_reram_program_verify import derive_seed


PLAN_SCHEMA = "ebl.mnist_relu_drn.ibm_om_four_reference_binding_plan"
PLAN_SCHEMA_VERSION = 1
MAPPING_SCHEMA = "ebl.mnist_relu_drn.ibm_om_four_reference_continuous_mapping"
MAPPING_SCHEMA_VERSION = 1
BALANCE_ALGORITHM = "reference_balanced_binding_v1"
RANDOM_POLICY = "random_binding"
BALANCED_POLICY = "reference_balanced_binding_v1"
POLICIES = (RANDOM_POLICY, BALANCED_POLICY)
LAYOUTS = ("halves", "paired")


@dataclass(frozen=True)
class ReferenceBindingPlan:
    """Destination-position to sampled-source-identity permutation."""

    policy: str
    assignment_seed: int
    population_fingerprint: str
    destination_to_source: torch.Tensor
    report: Mapping[str, Any]

    def __post_init__(self) -> None:
        value = self.destination_to_source
        if (
            self.policy not in POLICIES
            or value.ndim != 1
            or value.dtype != torch.int64
            or value.device.type != "cpu"
        ):
            raise ValueError("Expected a canonical CPU int64 reference binding plan.")
        expected = torch.arange(value.numel(), dtype=torch.int64)
        if not torch.equal(torch.sort(value).values, expected):
            raise ValueError("Expected a bijective destination-to-source permutation.")


def _tensor_digest(value: torch.Tensor) -> str:
    return _tensor_sha256(value.detach().contiguous().cpu())


def _quad_stack(value: torch.Tensor, *, layout: str) -> torch.Tensor:
    rows, plus, minus = _quad_columns(tuple(value.shape), layout, device=value.device)
    return torch.stack(
        (
            value[:rows][:, plus],
            value[:rows][:, minus],
            value[rows:][:, plus],
            value[rows:][:, minus],
        ),
        dim=-1,
    )


def _scatter_quads(quads: torch.Tensor, *, shape: tuple[int, int], layout: str) -> torch.Tensor:
    if quads.ndim != 3 or quads.shape[-1] != 4:
        raise ValueError("Expected logical-row by logical-column by four quad values.")
    result = torch.empty(shape, dtype=quads.dtype, device=quads.device)
    rows, plus, minus = _quad_columns(shape, layout, device=quads.device)
    if quads.shape[:2] != (rows, int(plus.numel())):
        raise ValueError("Expected quad values to match the declared rail matrix shape.")
    result[:rows][:, plus] = quads[..., 0]
    result[:rows][:, minus] = quads[..., 1]
    result[rows:][:, plus] = quads[..., 2]
    result[rows:][:, minus] = quads[..., 3]
    return result


def _best_partition(group_ids: torch.Tensor, group_reference: torch.Tensor) -> torch.Tensor:
    """Choose the deterministic minimum-error two-versus-two partition."""

    if group_ids.shape != (4,) or group_reference.shape != (4,):
        raise ValueError("Expected exactly four sorted identities per balance group.")
    # Destination order is (++,+-,-+,--).  The three rows enumerate the
    # unique pair partitions; order inside each pair follows source ID.
    partitions = ((0, 2, 3, 1), (0, 1, 3, 2), (0, 1, 2, 3))
    candidates: list[tuple[float, tuple[int, ...], torch.Tensor]] = []
    for pattern in partitions:
        destination = group_ids[list(pattern)].clone()
        # Canonicalize within the positive and negative rail pairs before the
        # lexicographic tie break.
        positive = sorted((int(destination[0]), int(destination[3])))
        negative = sorted((int(destination[1]), int(destination[2])))
        destination = torch.tensor(
            (positive[0], negative[0], negative[1], positive[1]),
            dtype=torch.int64,
        )
        ref_by_id = {
            int(identity): float(reference)
            for identity, reference in zip(group_ids.tolist(), group_reference.tolist())
        }
        error = abs(
            ref_by_id[int(destination[0])]
            - ref_by_id[int(destination[1])]
            - ref_by_id[int(destination[2])]
            + ref_by_id[int(destination[3])]
        )
        key = tuple(int(item) for item in destination.tolist())
        candidates.append((error, key, destination))
    return min(candidates, key=lambda item: (item[0], item[1]))[2]


def _balanced_local_permutation(
    reference: torch.Tensor,
    *,
    assignment_seed: int,
    binding_key: str,
    shape: tuple[int, int],
    layout: str,
) -> tuple[torch.Tensor, int]:
    """Globally regroup one layer by nearby references, then shuffle quads."""

    if reference.ndim != 1 or reference.numel() % 4:
        raise ValueError("Expected one finite reference per cell and a multiple of four.")
    if not bool(torch.isfinite(reference).all()):
        raise ValueError("Expected finite mapped references before balancing.")
    identities = torch.arange(reference.numel(), dtype=torch.int64)
    # Python tuple sorting freezes the exact secondary original-index tie rule
    # independently of torch sort implementation details.
    ordered = torch.tensor(
        sorted(
            identities.tolist(),
            key=lambda index: (float(reference[index].item()), int(index)),
        ),
        dtype=torch.int64,
    )
    groups = ordered.reshape(-1, 4)
    balanced = torch.stack(
        tuple(_best_partition(group, reference[group]) for group in groups),
        dim=0,
    )
    shuffle_seed = derive_seed(
        assignment_seed,
        binding_key,
        BALANCE_ALGORITHM,
    )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(shuffle_seed)
    balanced = balanced[torch.randperm(balanced.shape[0], generator=generator)]
    logical_rows, plus, _minus = _quad_columns(
        shape, layout, device=balanced.device
    )
    quad_grid = balanced.reshape(logical_rows, int(plus.numel()), 4)
    return _scatter_quads(quad_grid, shape=shape, layout=layout).reshape(-1), shuffle_seed


def build_reference_binding_plan(
    population: IbmReramArrayPopulation,
    *,
    policy: str,
    layouts: Sequence[str] = LAYOUTS,
) -> ReferenceBindingPlan:
    """Build a weight/label-blind, within-layer identity binding plan."""

    if policy not in POLICIES:
        raise ValueError(f"Expected binding policy to be one of {POLICIES!r}.")
    if len(layouts) != len(population.binding_keys) or tuple(layouts) != LAYOUTS:
        raise ValueError("Expected the canonical W1-halves/W2-paired four-device layout.")
    slices = _binding_slices(population)
    mapped_reference = _native_to_unit(population.reference.detach().cpu())
    global_permutation = torch.empty(population.size, dtype=torch.int64)
    layer_reports = []
    offset = 0
    for key, shape, layout in zip(
        population.binding_keys, population.binding_shapes, layouts
    ):
        binding_slice = slices[key]
        count = math.prod(shape)
        local_reference = mapped_reference[binding_slice]
        raw_reference = population.reference.detach().cpu()[binding_slice]
        if policy == RANDOM_POLICY:
            local_permutation = torch.arange(count, dtype=torch.int64)
            shuffle_seed = None
        else:
            local_permutation, shuffle_seed = _balanced_local_permutation(
                local_reference,
                assignment_seed=population.assignment_seed,
                binding_key=key,
                shape=tuple(shape),
                layout=layout,
            )
        source_global = local_permutation + offset
        global_permutation[binding_slice] = source_global
        before_matrix = local_reference.reshape(shape)
        after_matrix = local_reference[local_permutation].reshape(shape)
        before = _quad_contrast(before_matrix, layout=layout) / 2.0
        after = _quad_contrast(after_matrix, layout=layout) / 2.0
        layer_reports.append(
            {
                "binding_key": key,
                "shape": list(shape),
                "layout": layout,
                "source_identity_count": count,
                "quad_count": count // 4,
                "quad_shuffle_seed": shuffle_seed,
                "raw_sampled_reference": _float_summary(raw_reference),
                "mapped_reference": _float_summary(local_reference),
                "reference_clipped_by_global_coordinate_count": int(
                    ((raw_reference < -1.0) | (raw_reference > 1.0)).sum().item()
                ),
                "baseline_contrast_before": _float_summary(before),
                "baseline_contrast_after": _float_summary(after),
                "baseline_contrast_rms_ratio_after_over_before": (
                    float(after.double().square().mean().sqrt().item())
                    / float(before.double().square().mean().sqrt().item())
                ),
                "destination_to_source_sha256": _tensor_digest(local_permutation),
                "reference_multiset_preserved": bool(
                    torch.equal(
                        torch.sort(local_reference).values,
                        torch.sort(local_reference[local_permutation]).values,
                    )
                ),
            }
        )
        offset += count
    permutation_sha256 = _tensor_digest(global_permutation)
    hardware_instance_id = sha256(
        "\x1f".join(
            (population.fingerprint, policy, permutation_sha256)
        ).encode("utf-8")
    ).hexdigest()
    plan_report = {
        "schema": PLAN_SCHEMA,
        "schema_version": PLAN_SCHEMA_VERSION,
        "policy": policy,
        "algorithm": (
            "identity_order_unchanged" if policy == RANDOM_POLICY else BALANCE_ALGORITHM
        ),
        "balance_coordinate": "clip((r+1)/2,0,1)",
        "balance_scope": "within_layer_global_sorted_consecutive_quartets",
        "partition_objective": "minimum_absolute_signed_reference_sum",
        "partition_tie_break": "lexicographic_original_source_identity_tuple",
        "quad_address_assignment": (
            "identity_order" if policy == RANDOM_POLICY else "deterministic_seeded_shuffle"
        ),
        "uses_weights": False,
        "uses_labels": False,
        "uses_bounds": False,
        "uses_accuracy": False,
        "population_fingerprint": population.fingerprint,
        "assignment_seed": population.assignment_seed,
        "destination_to_source_sha256": permutation_sha256,
        "hardware_instance_id": hardware_instance_id,
        "layers": layer_reports,
    }
    return ReferenceBindingPlan(
        policy=policy,
        assignment_seed=population.assignment_seed,
        population_fingerprint=population.fingerprint,
        destination_to_source=global_permutation,
        report=plan_report,
    )


def _permuted(population: IbmReramArrayPopulation, plan: ReferenceBindingPlan, name: str) -> torch.Tensor:
    if (
        plan.population_fingerprint != population.fingerprint
        or plan.assignment_seed != population.assignment_seed
    ):
        raise ValueError("Expected binding plan and OM population provenance to match.")
    return getattr(population, name).detach().cpu()[plan.destination_to_source]


def _summary_with_abs(value: torch.Tensor) -> dict[str, Any]:
    result = _float_summary(value)
    result["absolute"] = _float_summary(value.abs())
    return result


def _inward_float32_bounds(
    lower: torch.Tensor,
    upper: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return float32 bounds that remain inside the supplied float64 box."""

    if (
        lower.dtype != torch.float64
        or upper.dtype != torch.float64
        or lower.shape != upper.shape
        or not bool(torch.isfinite(lower).all())
        or not bool(torch.isfinite(upper).all())
        or bool(torch.any(lower > upper))
    ):
        raise ValueError("Expected finite ordered float64 conductance bounds.")
    negative_infinity = torch.full_like(lower.to(torch.float32), float("-inf"))
    positive_infinity = torch.full_like(lower.to(torch.float32), float("inf"))
    lower_inward = lower.to(torch.float32)
    lower_inward = torch.where(
        lower_inward.to(torch.float64) < lower,
        torch.nextafter(lower_inward, positive_infinity),
        lower_inward,
    )
    upper_inward = upper.to(torch.float32)
    upper_inward = torch.where(
        upper_inward.to(torch.float64) > upper,
        torch.nextafter(upper_inward, negative_infinity),
        upper_inward,
    )
    if bool(torch.any(lower_inward > upper_inward)):
        raise RuntimeError("No float32 conductance exists inside an active bound interval.")
    return lower_inward, upper_inward


def build_continuous_reference_targets(
    logical_weights: Sequence[torch.Tensor],
    population: IbmReramArrayPopulation,
    plan: ReferenceBindingPlan,
    *,
    scale_fractions: tuple[float, float],
    conductance_min: float,
    conductance_max: float,
) -> tuple[
    tuple[torch.Tensor, ...],
    Mapping[str, Any],
    tuple[Mapping[str, torch.Tensor], ...],
]:
    """Map continuous positive-only lifts while retaining every full baseline."""

    if len(logical_weights) != 2 or len(scale_fractions) != 2:
        raise ValueError("Expected exactly two logical layers and scale fractions.")
    if not (
        math.isfinite(conductance_min)
        and math.isfinite(conductance_max)
        and 0.0 <= conductance_min < conductance_max
    ):
        raise ValueError("Expected finite nonnegative increasing conductance bounds.")
    if len(population.binding_keys) != 2:
        raise ValueError("Expected exactly two four-device physical bindings.")
    span = float(conductance_max - conductance_min)
    lower = _native_to_unit(_permuted(population, plan, "min_bound").to(torch.float64))
    upper = _native_to_unit(_permuted(population, plan, "max_bound").to(torch.float64))
    reference = _native_to_unit(_permuted(population, plan, "reference").to(torch.float64))
    active_zero = torch.minimum(torch.maximum(reference, lower), upper)
    slices = _binding_slices(population)
    targets: list[torch.Tensor] = []
    layers = []
    components: list[Mapping[str, torch.Tensor]] = []
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
        if not math.isfinite(fraction) or fraction <= 0.0 or fraction > 1.0:
            raise ValueError("Expected every scale fraction in (0,1].")
        binding_slice = slices[key]
        normalized = _normalized(logical.detach().cpu()).to(torch.float64)
        lifted = signed_dual_rail_lift(normalized, target_layout=layout)
        cell_reference = reference[binding_slice].reshape(shape)
        cell_zero = active_zero[binding_slice].reshape(shape)
        cell_upper = upper[binding_slice].reshape(shape)
        cell_lower = lower[binding_slice].reshape(shape)
        headroom = cell_upper - cell_zero
        common_headroom = _expanded_quad_minimum(headroom, layout=layout)
        offset_unit = fraction * common_headroom * lifted
        active_mask = lifted > 0.0
        baseline_unit = torch.where(active_mask, cell_zero, cell_reference)
        physical_lower = conductance_min + span * cell_lower
        physical_upper = conductance_min + span * cell_upper
        lower_inward, upper_inward = _inward_float32_bounds(
            physical_lower,
            physical_upper,
        )
        raw_physical_baseline = (
            conductance_min + span * baseline_unit
        ).to(torch.float32)
        active_physical_baseline = torch.minimum(
            torch.maximum(raw_physical_baseline, lower_inward),
            upper_inward,
        )
        physical_baseline = torch.where(
            active_mask,
            active_physical_baseline,
            raw_physical_baseline,
        )
        desired_physical_conductance = physical_baseline + (
            span * offset_unit
        ).to(torch.float32)
        bounded_physical_conductance = torch.where(
            active_mask,
            torch.minimum(
                torch.maximum(desired_physical_conductance, lower_inward),
                upper_inward,
            ),
            desired_physical_conductance,
        )
        baseline_inward_adjustment = active_mask & (
            active_physical_baseline != raw_physical_baseline
        )
        conductance_inward_adjustment = active_mask & (
            bounded_physical_conductance != desired_physical_conductance
        )
        physical_offset = bounded_physical_conductance - physical_baseline
        # This float32 addition is the canonical deployed state.  Persisting
        # and applying this exact tensor makes the branch-level invariant
        # bit-exact rather than merely true before dtype conversion.
        physical_conductance = physical_baseline + physical_offset
        decomposition_residual = physical_conductance - (
            physical_baseline + physical_offset
        )
        if not bool(torch.isfinite(physical_conductance).all()):
            raise RuntimeError("Expected every mapped physical conductance to be finite.")
        if bool(torch.any(physical_conductance < -1e-12)):
            raise RuntimeError("Expected every mapped physical conductance to be nonnegative.")
        active_bound_violation = active_mask & (
            (physical_conductance.to(torch.float64) < physical_lower)
            | (physical_conductance.to(torch.float64) > physical_upper)
        )
        if bool(torch.any(active_bound_violation)):
            raise RuntimeError("Expected every active conductance to stay in cell bounds.")
        baseline_contrast = _quad_contrast(physical_baseline, layout=layout) / 2.0
        realized_contrast = _quad_contrast(physical_conductance, layout=layout) / 2.0
        offset_contrast = _quad_contrast(physical_offset, layout=layout) / 2.0
        quad_loading = _quad_stack(physical_conductance, layout=layout).sum(dim=-1)
        baseline_loading = _quad_stack(physical_baseline, layout=layout).sum(dim=-1)
        normalized_nonzero = normalized != 0.0
        sign_flips = normalized_nonzero & (
            torch.sign(realized_contrast) != torch.sign(normalized)
        )
        reference_outside = (cell_reference < cell_lower) | (cell_reference > cell_upper)
        logical_headroom = _quad_stack(common_headroom, layout=layout)[..., 0]
        layers.append(
            {
                "layer": layer_index,
                "binding_key": key,
                "shape": list(shape),
                "layout": layout,
                "scale_fraction": fraction,
                "mapping": "continuous_positive_only_local_common_headroom",
                "level_rounding": False,
                "level_spacing": None,
                "level_count_cap": None,
                "baseline_contrast": _summary_with_abs(baseline_contrast),
                "offset_contrast": _summary_with_abs(offset_contrast),
                "realized_contrast": _summary_with_abs(realized_contrast),
                "quad_loading": _float_summary(quad_loading),
                "baseline_loading": _float_summary(baseline_loading),
                "common_positive_headroom": _float_summary(logical_headroom),
                "zero_positive_headroom_count": int((logical_headroom <= 0.0).sum().item()),
                "zero_positive_headroom_fraction": float(
                    (logical_headroom <= 0.0).to(torch.float64).mean().item()
                ),
                "reference_outside_active_bounds_count": int(reference_outside.sum().item()),
                "active_zero_minus_reference": _summary_with_abs(cell_zero - cell_reference),
                "continuous_logical_sign_flip_count": int(sign_flips.sum().item()),
                "continuous_logical_sign_flip_fraction_nonzero": (
                    float(sign_flips.sum().item()) / int(normalized_nonzero.sum().item())
                ),
                "full_g_decomposition_max_abs_residual": float(
                    decomposition_residual.abs().max().item()
                ),
                "finite_conductance": True,
                "nonnegative_conductance": True,
                "active_bound_violation_count": int(active_bound_violation.sum().item()),
                "float32_inward_baseline_adjustment_count": int(
                    baseline_inward_adjustment.sum().item()
                ),
                "float32_inward_conductance_adjustment_count": int(
                    conductance_inward_adjustment.sum().item()
                ),
                "float32_inward_conductance_max_abs_adjustment": float(
                    (
                        bounded_physical_conductance
                        - desired_physical_conductance
                    )
                    .abs()
                    .max()
                    .item()
                ),
                "hashes": {
                    "baseline": _tensor_digest(physical_baseline.to(torch.float32)),
                    "offset": _tensor_digest(physical_offset.to(torch.float32)),
                    "conductance": _tensor_digest(physical_conductance.to(torch.float32)),
                    "baseline_contrast": _tensor_digest(baseline_contrast.to(torch.float32)),
                    "realized_contrast": _tensor_digest(realized_contrast.to(torch.float32)),
                    "loading": _tensor_digest(quad_loading.to(torch.float32)),
                    "active_mask": _tensor_digest(active_mask),
                    "common_headroom": _tensor_digest(common_headroom.to(torch.float32)),
                },
            }
        )
        targets.append(physical_conductance)
        components.append(
            {
                "baseline": physical_baseline,
                "offset": physical_offset,
                "conductance": physical_conductance,
                "baseline_contrast": baseline_contrast.to(torch.float32),
                "offset_contrast": offset_contrast.to(torch.float32),
                "realized_contrast": realized_contrast.to(torch.float32),
                "loading": quad_loading.to(torch.float32),
                "active_mask": active_mask,
                "common_headroom": common_headroom.to(torch.float32),
            }
        )
    return tuple(targets), {
        "policy": plan.policy,
        "population_fingerprint": population.fingerprint,
        "assignment_seed": population.assignment_seed,
        "conductance_coordinate": "G=conductance_min+span*clip((a+1)/2,0,1)",
        "decomposition": "every physical branch G=B+d",
        "circuit_accounting": "full G enters numerator and denominator",
        "scale_fractions": [float(value) for value in scale_fractions],
        "layers": layers,
        "target_hashes": [_tensor_digest(value) for value in targets],
    }, tuple(components)


def save_reference_binding_plan(
    directory: Path,
    population: IbmReramArrayPopulation,
    plan: ReferenceBindingPlan,
) -> Mapping[str, Any]:
    """Persist a pickle-free plan and exact receipt below a run artifact root."""

    destination = directory.expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    stem = f"{plan.policy}-assignment-{population.assignment_seed}"
    artifact = destination / f"{stem}.npz"
    receipt_path = destination / f"{stem}.receipt.json"
    with tempfile.NamedTemporaryFile(
        dir=destination, prefix=f".{stem}.", suffix=".npz", delete=False
    ) as handle:
        temporary = Path(handle.name)
        np.savez_compressed(
            handle,
            schema=np.asarray(PLAN_SCHEMA),
            schema_version=np.asarray(PLAN_SCHEMA_VERSION, dtype=np.int64),
            policy=np.asarray(plan.policy),
            assignment_seed=np.asarray(plan.assignment_seed, dtype=np.int64),
            population_fingerprint=np.asarray(plan.population_fingerprint),
            binding_keys_json=np.asarray(
                json.dumps(list(population.binding_keys), separators=(",", ":"))
            ),
            binding_shapes_json=np.asarray(
                json.dumps([list(shape) for shape in population.binding_shapes], separators=(",", ":"))
            ),
            destination_to_source=plan.destination_to_source.numpy(),
        )
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
        "destination_to_source_sha256": plan.report["destination_to_source_sha256"],
    }


def save_continuous_mapping(
    directory: Path,
    *,
    assignment_seed: int,
    binding_policy: str,
    calibration_policy: str,
    mapping_report: Mapping[str, Any],
    components: Sequence[Mapping[str, torch.Tensor]],
) -> Mapping[str, Any]:
    """Persist full ``B``, ``d``, and ``G`` tensors for one evaluated map."""

    if binding_policy not in POLICIES or calibration_policy not in POLICIES:
        raise ValueError("Expected canonical binding and calibration policy names.")
    if len(components) != 2:
        raise ValueError("Expected two mapped layer component records.")
    destination = directory.expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    stem = (
        f"assignment-{assignment_seed}-{binding_policy}"
        f"-calibrated-by-{calibration_policy}"
    )
    artifact = destination / f"{stem}.npz"
    receipt_path = destination / f"{stem}.receipt.json"
    arrays: dict[str, np.ndarray] = {
        "schema": np.asarray(MAPPING_SCHEMA),
        "schema_version": np.asarray(MAPPING_SCHEMA_VERSION, dtype=np.int64),
        "assignment_seed": np.asarray(assignment_seed, dtype=np.int64),
        "binding_policy": np.asarray(binding_policy),
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
            raise ValueError("Expected exact continuous mapping component fields.")
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
        "binding_policy": binding_policy,
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
        "binding_policy": binding_policy,
        "calibration_policy": calibration_policy,
    }


__all__ = [
    "BALANCED_POLICY",
    "BALANCE_ALGORITHM",
    "PLAN_SCHEMA",
    "PLAN_SCHEMA_VERSION",
    "MAPPING_SCHEMA",
    "MAPPING_SCHEMA_VERSION",
    "POLICIES",
    "RANDOM_POLICY",
    "ReferenceBindingPlan",
    "build_continuous_reference_targets",
    "build_reference_binding_plan",
    "save_reference_binding_plan",
    "save_continuous_mapping",
]
