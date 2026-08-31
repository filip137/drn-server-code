"""Core primitives for the corrupt-source IBM OM deployment ladder.

The experiment deliberately keeps two views of every physical assignment:

* ``published`` is the literal AIHWKit OM population with the published
  corrupt-device probability enabled; and
* ``repaired`` replaces only those same corrupt coordinates with healthy
  donors and is used solely to define the logical mapping/code capacities.

Both views are Winsorized in AIHWKit's native raw-``a`` coordinate before the
positive circuit embedding ``G = a + 1 = 2*x``.  Native corrupt singletons are
never repaired, moved, or assigned a non-zero pulse step in the published
plant.  Consequently a corrupt cell returns its immutable stuck ``x`` for
every logical code in the persistent-endpoint HWA table.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from experiments.mnist_relu_drn.ibm_om_baseline_selection import (
    quad_stack,
)
from experiments.mnist_relu_drn.ibm_om_baseline_spacing_pv_truncated_nominal import (
    project_shared_destination_reset_baselines,
)
from experiments.mnist_relu_drn.ibm_om_winsorized_pv_ensemble_qat import (
    ENSEMBLE_SIZE,
    PersistentEndpointCodebookEnsemble,
    PersistentEndpointLevelDiagnostics,
)
from experiments.mnist_relu_drn.ibm_om_winsorized_qat import (
    LAYOUTS,
    WinsorizedQatLayerTemplate,
    WinsorizedQatView,
    quantize_winsorized_logical_master,
)
from training.ibm_reram_hwa import (
    IbmReramArrayPopulation,
    sample_om_array_population,
    sample_om_array_population_external,
)
from training.ibm_reram_program_verify import (
    ControllerSettings,
    IbmReramRawActivePlant,
    make_buffered_normal_draws,
    run_program_verify,
)
from training.ibm_reram_raw_active_program_verify import (
    RAW_ACTIVE_SUPPORT_TOLERANCE_UNIT,
    array_population_as_pulse_population,
    matched_trajectory_seeds,
    required_cuda_random_draws,
)
from model.resistive.builders import ParameterBinding


EXPERIMENT_ID = "mnist_ibm_om_corrupt_source_hwa_cross_array_pulse_adam.v1"
EVIDENCE_TIER = "exploratory_noncanonical"
NOMINAL_RAW_A_MINIMUM = -1.0
NOMINAL_RAW_A_MAXIMUM = 1.0
FULL_CONDUCTANCE_FORMULA = "G=raw_a_winsorized+1=2*x"


def _population_fingerprint(
    source: IbmReramArrayPopulation,
    *,
    max_bound: torch.Tensor,
    min_bound: torch.Tensor,
) -> str:
    """Recompute the canonical HWA-population fingerprint after Winsorizing.

    This intentionally matches ``training.ibm_reram_hwa`` byte-for-byte so a
    Winsorized population saved with its public NPZ writer remains loadable by
    the strict public loader.  Intervention provenance lives in the paired
    report instead of being smuggled into the population hash algorithm.
    """

    digest = sha256()
    digest.update(str(source.assignment_seed).encode())
    digest.update(source.corruption_policy.encode())
    for key, shape, binding_seed, donor_seed in zip(
        source.binding_keys,
        source.binding_shapes,
        source.binding_sampling_seeds,
        source.donor_sampling_seeds,
    ):
        digest.update(key.encode())
        digest.update(repr(tuple(shape)).encode())
        digest.update(str(int(binding_seed)).encode())
        digest.update(str(int(donor_seed)).encode())
    scalars: Mapping[str, float | str] = {
        "aihwkit_version": source.aihwkit_version,
        "nominal_dw_min": float(source.nominal_dw_min),
        "dw_min_std": float(source.dw_min_std),
        "write_noise_std": float(source.write_noise_std),
    }
    for name in sorted(scalars):
        value = scalars[name]
        digest.update(name.encode())
        digest.update(
            (float(value).hex() if isinstance(value, float) else str(value)).encode()
        )
    tensors = {
        "max_bound": max_bound,
        "min_bound": min_bound,
        "dwmin_up": source.dwmin_up,
        "dwmin_down": source.dwmin_down,
        "reference": source.reference,
        "corrupt": source.corrupt,
        "published_corrupt": source.published_corrupt,
    }
    for name in sorted(tensors):
        value = tensors[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _winsorize_population(
    population: IbmReramArrayPopulation,
) -> IbmReramArrayPopulation:
    """Winsorize one assignment without changing native corrupt cells."""

    if not isinstance(population, IbmReramArrayPopulation):
        raise TypeError("Expected one IbmReramArrayPopulation.")
    corrupt = population.corrupt.detach().cpu()
    original_lower = population.min_bound.detach().cpu().to(torch.float32)
    original_upper = population.max_bound.detach().cpu().to(torch.float32)
    corrupt_outside = corrupt & (
        (original_lower < NOMINAL_RAW_A_MINIMUM)
        | (original_upper > NOMINAL_RAW_A_MAXIMUM)
    )
    if bool(torch.any(corrupt_outside)):
        raise ValueError(
            "A native corrupt singleton lies outside nominal raw-a support; "
            "refusing to move or clip the stuck value."
        )
    lower = torch.where(
        corrupt,
        original_lower,
        original_lower.clamp(min=NOMINAL_RAW_A_MINIMUM),
    )
    upper = torch.where(
        corrupt,
        original_upper,
        original_upper.clamp(max=NOMINAL_RAW_A_MAXIMUM),
    )
    if bool(torch.any((~corrupt) & (upper <= lower))):
        raise ValueError("Nominal Winsorization collapsed a healthy OM cell support.")
    if bool(torch.any(lower < -1.0)) or bool(torch.any(upper > 1.0)):
        raise RuntimeError("Winsorized raw-a support left [-1,1].")
    fingerprint = _population_fingerprint(
        population,
        max_bound=upper,
        min_bound=lower,
    )
    return IbmReramArrayPopulation(
        assignment_seed=population.assignment_seed,
        corruption_policy=population.corruption_policy,
        binding_keys=population.binding_keys,
        binding_shapes=population.binding_shapes,
        binding_sampling_seeds=population.binding_sampling_seeds,
        donor_sampling_seeds=population.donor_sampling_seeds,
        nominal_dw_min=population.nominal_dw_min,
        dw_min_std=population.dw_min_std,
        write_noise_std=population.write_noise_std,
        max_bound=upper.clone(),
        min_bound=lower.clone(),
        dwmin_up=population.dwmin_up.detach().cpu().clone(),
        dwmin_down=population.dwmin_down.detach().cpu().clone(),
        reference=population.reference.detach().cpu().clone(),
        corrupt=corrupt.clone(),
        published_corrupt=population.published_corrupt.detach().cpu().clone(),
        fingerprint=fingerprint,
        aihwkit_version=population.aihwkit_version,
    )


def _paired_population_invariants(
    published: IbmReramArrayPopulation,
    repaired: IbmReramArrayPopulation,
) -> None:
    scalar_pairs = (
        (published.assignment_seed, repaired.assignment_seed),
        (published.binding_keys, repaired.binding_keys),
        (published.binding_shapes, repaired.binding_shapes),
        (published.binding_sampling_seeds, repaired.binding_sampling_seeds),
        (published.donor_sampling_seeds, repaired.donor_sampling_seeds),
        (published.aihwkit_version, repaired.aihwkit_version),
        (published.nominal_dw_min, repaired.nominal_dw_min),
        (published.dw_min_std, repaired.dw_min_std),
        (published.write_noise_std, repaired.write_noise_std),
    )
    if any(left != right for left, right in scalar_pairs):
        raise ValueError("Published and repaired views do not name one paired assignment.")
    if (
        published.corruption_policy != "published"
        or repaired.corruption_policy != "counterfactual_repaired"
        or not torch.equal(published.corrupt, published.published_corrupt)
        or bool(torch.any(repaired.corrupt))
        or not torch.equal(published.published_corrupt, repaired.published_corrupt)
    ):
        raise ValueError("Expected literal published faults plus a paired repaired view.")
    healthy = ~published.corrupt
    for name in ("max_bound", "min_bound", "dwmin_up", "dwmin_down", "reference"):
        if not torch.equal(getattr(published, name)[healthy], getattr(repaired, name)[healthy]):
            raise ValueError(f"Paired OM healthy parameters differ for {name!r}.")
    corrupt = published.corrupt
    if bool(torch.any(corrupt)):
        if (
            not torch.equal(published.min_bound[corrupt], published.max_bound[corrupt])
            or bool(torch.any(published.dwmin_up[corrupt] != 0.0))
            or bool(torch.any(published.dwmin_down[corrupt] != 0.0))
        ):
            raise ValueError("Published corrupt cells must remain collapsed and zero-step.")


@dataclass(frozen=True)
class PairedWinsorizedOmPopulation:
    """Paired mapping template and physical plant for one OM assignment."""

    published: IbmReramArrayPopulation
    repaired: IbmReramArrayPopulation
    report: Mapping[str, Any]

    def __post_init__(self) -> None:
        _paired_population_invariants(self.published, self.repaired)
        for population in (self.published, self.repaired):
            if (
                bool(torch.any(population.min_bound < -1.0))
                or bool(torch.any(population.max_bound > 1.0))
                or bool(torch.any(population.min_bound + 1.0 < 0.0))
            ):
                raise ValueError("Paired Winsorized populations must produce G in [0,2].")

    @property
    def assignment_seed(self) -> int:
        return self.published.assignment_seed

    @property
    def corrupt_mask(self) -> torch.Tensor:
        return self.published.corrupt.detach().clone()


def pair_and_winsorize_om_populations(
    published: IbmReramArrayPopulation,
    repaired: IbmReramArrayPopulation,
) -> PairedWinsorizedOmPopulation:
    """Validate paired native draws and apply the nominal support intervention."""

    _paired_population_invariants(published, repaired)
    published_w = _winsorize_population(published)
    repaired_w = _winsorize_population(repaired)
    _paired_population_invariants(published_w, repaired_w)
    corrupt = published_w.corrupt
    stuck = published_w.min_bound[corrupt]
    return PairedWinsorizedOmPopulation(
        published=published_w,
        repaired=repaired_w,
        report={
            "evidence_tier": EVIDENCE_TIER,
            "assignment_seed": int(published.assignment_seed),
            "preset": "reram_array_om",
            "required_aihwkit_version": "1.1.0",
            "pairing": "same_literal_published_draw_repair_only_native_corrupt_cells",
            "winsorization": "healthy_raw_a_bounds_intersect_[-1,1]",
            "native_corrupt_policy": "preserve_singleton_and_zero_both_pulse_steps",
            "full_conductance_formula": FULL_CONDUCTANCE_FORMULA,
            "published_native_fingerprint": published.fingerprint,
            "repaired_native_fingerprint": repaired.fingerprint,
            "published_winsorized_fingerprint": published_w.fingerprint,
            "repaired_winsorized_fingerprint": repaired_w.fingerprint,
            "cells": published.size,
            "corrupt_cells": int(corrupt.sum().item()),
            "corrupt_fraction": float(corrupt.to(torch.float64).mean().item()),
            "corrupt_raw_a_minimum": float(stuck.min().item()) if stuck.numel() else None,
            "corrupt_raw_a_maximum": float(stuck.max().item()) if stuck.numel() else None,
            "negative_full_conductance_count": 0,
        },
    )


def sample_paired_winsorized_om_populations(
    bindings: Sequence[ParameterBinding],
    *,
    assignment_seed: int,
) -> PairedWinsorizedOmPopulation:
    """Sample both paired views in the current pinned AIHWKit environment."""

    published = sample_om_array_population(
        bindings, assignment_seed=assignment_seed, corruption_policy="published"
    )
    repaired = sample_om_array_population(
        bindings,
        assignment_seed=assignment_seed,
        corruption_policy="counterfactual_repaired",
    )
    return pair_and_winsorize_om_populations(published, repaired)


def sample_paired_winsorized_om_populations_external(
    bindings: Sequence[ParameterBinding],
    *,
    assignment_seed: int,
    aihwkit_python: Path,
    published_population_path: Path,
    published_receipt_path: Path,
    repaired_population_path: Path,
    repaired_receipt_path: Path,
) -> tuple[PairedWinsorizedOmPopulation, Mapping[str, Mapping[str, Any]]]:
    """Sample two hash-receipted NPZ views through pinned external AIHWKit."""

    published, published_receipt = sample_om_array_population_external(
        bindings,
        assignment_seed=assignment_seed,
        corruption_policy="published",
        aihwkit_python=aihwkit_python,
        population_path=published_population_path,
        receipt_path=published_receipt_path,
    )
    repaired, repaired_receipt = sample_om_array_population_external(
        bindings,
        assignment_seed=assignment_seed,
        corruption_policy="counterfactual_repaired",
        aihwkit_python=aihwkit_python,
        population_path=repaired_population_path,
        receipt_path=repaired_receipt_path,
    )
    pair = pair_and_winsorize_om_populations(published, repaired)
    return pair, {"published": published_receipt, "repaired": repaired_receipt}


def _split_flat(
    value: torch.Tensor, shapes: Sequence[Sequence[int]]
) -> tuple[torch.Tensor, ...]:
    flat = torch.as_tensor(value).detach().cpu().reshape(-1)
    result: list[torch.Tensor] = []
    offset = 0
    for raw_shape in shapes:
        shape = tuple(int(item) for item in raw_shape)
        count = math.prod(shape)
        result.append(flat[offset : offset + count].reshape(shape))
        offset += count
    if offset != flat.numel():
        raise ValueError("Binding shapes do not cover the flattened population.")
    return tuple(result)


def build_repaired_mapping_templates(
    logical_masters: Sequence[torch.Tensor],
    pair: PairedWinsorizedOmPopulation,
    *,
    level_spacing_raw_x: float,
    layouts: Sequence[str] = LAYOUTS,
) -> tuple[WinsorizedQatLayerTemplate, ...]:
    """Build shared-destination mapping templates from repaired support only."""

    masters = tuple(logical_masters)
    selected_layouts = tuple(layouts)
    if len(masters) != 2 or selected_layouts != LAYOUTS:
        raise ValueError("Expected two logical masters with canonical DRN layouts.")
    spacing = float(level_spacing_raw_x)
    if not math.isfinite(spacing) or spacing <= 0.0:
        raise ValueError("Expected positive finite raw-x level spacing.")
    repaired = pair.repaired
    lower_layers = _split_flat((repaired.min_bound + 1.0) / 2.0, repaired.binding_shapes)
    upper_layers = _split_flat((repaired.max_bound + 1.0) / 2.0, repaired.binding_shapes)
    baselines, _baseline_report = project_shared_destination_reset_baselines(
        lower_layers,
        upper_layers,
        lower_layers,
        layouts=selected_layouts,
    )
    templates: list[WinsorizedQatLayerTemplate] = []
    tolerance = RAW_ACTIVE_SUPPORT_TOLERANCE_UNIT
    for layer_index, (master_value, baseline, upper, shape, layout) in enumerate(
        zip(masters, baselines, upper_layers, repaired.binding_shapes, selected_layouts)
    ):
        master = torch.as_tensor(master_value).detach().cpu().to(torch.float32)
        expected_logical_shape = (shape[0] // 2, shape[1] // 2)
        if (
            tuple(master.shape) != expected_logical_shape
            or not bool(torch.all(torch.isfinite(master)))
            or bool(torch.any(master < -1.0))
            or bool(torch.any(master > 1.0))
        ):
            raise ValueError("Logical master does not match the paired physical layout.")
        baseline32 = baseline.detach().cpu().to(torch.float32)
        upper32 = upper.detach().cpu().to(torch.float32)
        headroom_quad = quad_stack(upper32 - baseline32, layout=layout)
        positive_headroom = torch.minimum(headroom_quad[..., 0], headroom_quad[..., 3])
        negative_headroom = torch.minimum(headroom_quad[..., 1], headroom_quad[..., 2])
        positive_capacity = torch.floor(
            (positive_headroom.to(torch.float64) + tolerance) / spacing + 1e-12
        ).to(torch.int64)
        negative_capacity = torch.floor(
            (negative_headroom.to(torch.float64) + tolerance) / spacing + 1e-12
        ).to(torch.int64)
        templates.append(
            WinsorizedQatLayerTemplate(
                layer_index=layer_index,
                layout=layout,
                baseline_full_g=2.0 * baseline32,
                baseline_raw_x=baseline32,
                cell_upper_raw_x=upper32,
                positive_headroom_raw_x=positive_headroom,
                negative_headroom_raw_x=negative_headroom,
                positive_capacity=positive_capacity,
                negative_capacity=negative_capacity,
                initial_normalized_weight=master.clone(),
                level_spacing_raw_x=spacing,
            )
        )
    return tuple(templates)


@dataclass(frozen=True)
class PublishedPersistentTargets:
    """Logical mapping request evaluated against the literal published plant."""

    quantized_view: WinsorizedQatView
    target_raw_x: tuple[torch.Tensor, ...]
    target_raw_a: tuple[torch.Tensor, ...]
    target_full_conductance: tuple[torch.Tensor, ...]
    corrupt_mask: tuple[torch.Tensor, ...]
    unsupported_healthy_mask: tuple[torch.Tensor, ...]
    report: Mapping[str, Any]


@dataclass(frozen=True)
class PublishedFaultClampedConductance:
    """Deterministic published-fault realization before stochastic writing."""

    requested_full_conductance: tuple[torch.Tensor, ...]
    realized_full_conductance: tuple[torch.Tensor, ...]
    corrupt_mask: tuple[torch.Tensor, ...]
    report: Mapping[str, Any]


def build_published_persistent_targets(
    logical_masters: Sequence[torch.Tensor],
    templates: Sequence[WinsorizedQatLayerTemplate],
    pair: PairedWinsorizedOmPopulation,
) -> PublishedPersistentTargets:
    """Map through repaired templates, then audit requests on published cells."""

    view = quantize_winsorized_logical_master(logical_masters, templates)
    lower = _split_flat((pair.published.min_bound + 1.0) / 2.0, pair.published.binding_shapes)
    upper = _split_flat((pair.published.max_bound + 1.0) / 2.0, pair.published.binding_shapes)
    corrupt = _split_flat(pair.published.corrupt, pair.published.binding_shapes)
    raw_x = tuple(value.detach().cpu().to(torch.float32) for value in view.raw_x_targets)
    unsupported: list[torch.Tensor] = []
    for target, low, high, mask in zip(raw_x, lower, upper, corrupt):
        if bool(torch.any(target < 0.0)) or bool(torch.any(target > 1.0)):
            raise RuntimeError("Repaired mapping produced raw x outside [0,1].")
        unsupported.append((~mask) & ((target < low - 1e-6) | (target > high + 1e-6)))
    if any(bool(torch.any(value)) for value in unsupported):
        raise RuntimeError("Paired healthy published support drifted from repaired mapping.")
    corrupt_flat = torch.cat(tuple(value.reshape(-1) for value in corrupt))
    target_flat = torch.cat(tuple(value.reshape(-1) for value in raw_x))
    stuck_flat = (pair.published.min_bound + 1.0) / 2.0
    corrupt_mismatch = corrupt_flat & (torch.abs(target_flat - stuck_flat) > 1e-6)
    return PublishedPersistentTargets(
        quantized_view=view,
        target_raw_x=raw_x,
        target_raw_a=tuple(value * 2.0 - 1.0 for value in raw_x),
        target_full_conductance=tuple(2.0 * value for value in raw_x),
        corrupt_mask=tuple(value.to(torch.bool) for value in corrupt),
        unsupported_healthy_mask=tuple(unsupported),
        report={
            "mapping_population": "paired_counterfactual_repaired",
            "programming_population": "literal_published",
            "target_clipping": False,
            "full_conductance_formula": FULL_CONDUCTANCE_FORMULA,
            "cells": pair.published.size,
            "corrupt_cells": int(corrupt_flat.sum().item()),
            "corrupt_target_differs_from_stuck": int(corrupt_mismatch.sum().item()),
            "unsupported_healthy_cells": 0,
        },
    )


def clamp_targets_to_published_corrupt_singletons(
    requested_full_conductance: Sequence[torch.Tensor],
    pair: PairedWinsorizedOmPopulation,
) -> PublishedFaultClampedConductance:
    """Apply only native permanent faults, without programming noise.

    Healthy cells retain the repaired-template request bit-for-bit. Each
    published corrupt coordinate is replaced by its immutable native
    singleton ``G_stuck = raw_a_stuck + 1``. This is an analysis rung, not a
    writer, remapper, or repair operation, and makes no pulse or verify calls.
    """

    requested = tuple(
        torch.as_tensor(value).detach().cpu().to(torch.float32).clone()
        for value in requested_full_conductance
    )
    if len(requested) != len(pair.published.binding_shapes):
        raise ValueError("Expected one requested full-G tensor per binding.")
    masks = _split_flat(pair.published.corrupt, pair.published.binding_shapes)
    stuck = _split_flat(
        pair.published.min_bound + 1.0, pair.published.binding_shapes
    )
    realized: list[torch.Tensor] = []
    changed = 0
    maximum = 0.0
    for value, mask, stuck_value, shape in zip(
        requested, masks, stuck, pair.published.binding_shapes
    ):
        if (
            tuple(value.shape) != tuple(shape)
            or not bool(torch.all(torch.isfinite(value)))
            or bool(torch.any(value < 0.0))
            or bool(torch.any(value > 2.0))
        ):
            raise ValueError("Expected finite requested full G in [0,2].")
        output = torch.where(mask, stuck_value.to(torch.float32), value)
        difference = (output - value).abs()
        if bool(torch.any(difference[~mask] != 0.0)):
            raise RuntimeError("Fault clamping changed a healthy cell.")
        changed += int((difference[mask] != 0.0).sum().item())
        if bool(torch.any(mask)):
            maximum = max(maximum, float(difference[mask].max().item()))
        if bool(torch.any(output < 0.0)) or bool(torch.any(output > 2.0)):
            raise RuntimeError("A native corrupt singleton produced invalid full G.")
        realized.append(output)
    corrupt_cells = int(pair.published.corrupt.sum().item())
    return PublishedFaultClampedConductance(
        requested_full_conductance=requested,
        realized_full_conductance=tuple(realized),
        corrupt_mask=tuple(value.to(torch.bool) for value in masks),
        report={
            "policy": "healthy_request_plus_native_published_corrupt_singleton",
            "role": "deterministic_fault_only_no_write_analysis_rung",
            "program_verify": False,
            "verify_reads": 0,
            "programming_noise": False,
            "target_clipping": False,
            "healthy_requested_values_preserved": True,
            "full_conductance_formula": FULL_CONDUCTANCE_FORMULA,
            "cells": pair.published.size,
            "corrupt_cells": corrupt_cells,
            "corrupt_cells_differing_from_request": changed,
            "maximum_absolute_corrupt_delta_full_G": maximum,
            "negative_conductance_count": 0,
        },
    )


def build_corrupt_persistent_endpoint_codebook_ensemble(
    pair: PairedWinsorizedOmPopulation,
    templates: Sequence[WinsorizedQatLayerTemplate],
    *,
    endpoint_seeds: Sequence[int],
    tolerance_raw_x: float,
    maximum_program_pulses: int,
    device: torch.device | str,
) -> PersistentEndpointCodebookEnsemble:
    """Build four noisy persistent tables while retaining immutable faults.

    Code capacities and targets come from the repaired view.  The exact pulse
    plant is the published view.  Healthy cells run ordinary noisy one-pulse
    P&V; corrupt cells are excluded from controller writes and their native
    singleton is copied into every repaired-supported code entry.
    """

    seeds = tuple(endpoint_seeds)
    if (
        len(seeds) != ENSEMBLE_SIZE
        or len(set(seeds)) != ENSEMBLE_SIZE
        or any(isinstance(seed, bool) or not isinstance(seed, int) for seed in seeds)
    ):
        raise ValueError("Expected exactly four unique integer endpoint seeds.")
    tolerance = float(tolerance_raw_x)
    if not math.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("Expected positive finite tolerance_raw_x.")
    if (
        isinstance(maximum_program_pulses, bool)
        or not isinstance(maximum_program_pulses, int)
        or maximum_program_pulses < 1
    ):
        raise ValueError("Expected a positive maximum_program_pulses.")
    execution_device = torch.device(device)
    if execution_device.type not in {"cpu", "cuda"}:
        raise ValueError("Expected CPU or CUDA endpoint construction.")
    if execution_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Expected CUDA to be available.")

    selected_templates = tuple(templates)
    if len(selected_templates) != len(pair.repaired.binding_shapes):
        raise ValueError("Expected one repaired template per binding.")
    spacings = tuple(float(item.level_spacing_raw_x) for item in selected_templates)
    if any(not math.isclose(value, spacings[0], rel_tol=0.0, abs_tol=1e-15) for value in spacings):
        raise ValueError("Expected one common raw-x spacing.")
    spacing = spacings[0]
    baseline = torch.cat(
        tuple(item.baseline_raw_x.detach().cpu().reshape(-1) for item in selected_templates)
    ).to(torch.float32)
    baseline_g = torch.cat(
        tuple(item.baseline_full_g.detach().cpu().reshape(-1) for item in selected_templates)
    ).to(torch.float32)
    template_upper = torch.cat(
        tuple(item.cell_upper_raw_x.detach().cpu().reshape(-1) for item in selected_templates)
    ).to(torch.float32)
    repaired_lower = (pair.repaired.min_bound + 1.0) / 2.0
    repaired_upper = (pair.repaired.max_bound + 1.0) / 2.0
    if (
        baseline.shape != repaired_lower.shape
        or not torch.equal(baseline, baseline_g / 2.0)
        or bool(torch.any(baseline < repaired_lower - RAW_ACTIVE_SUPPORT_TOLERANCE_UNIT))
        or bool(torch.any(baseline > repaired_upper + RAW_ACTIVE_SUPPORT_TOLERANCE_UNIT))
        or float((template_upper - repaired_upper).abs().max().item())
        > RAW_ACTIVE_SUPPORT_TOLERANCE_UNIT
    ):
        raise ValueError("Repaired templates do not match paired repaired support.")
    maximum = torch.floor(
        (
            template_upper.to(torch.float64)
            - baseline.to(torch.float64)
            + RAW_ACTIVE_SUPPORT_TOLERANCE_UNIT
        )
        / spacing
        + 1e-12
    ).to(torch.int64)
    if bool(torch.any(maximum < 0)):
        raise ValueError("Repaired baseline exceeds repaired cell support.")
    levels = int(maximum.max().item()) + 1
    table = torch.full(
        (ENSEMBLE_SIZE, pair.published.size, levels),
        float("nan"),
        dtype=torch.float32,
    )
    diagnostics: list[PersistentEndpointLevelDiagnostics] = []
    pulse_cpu = array_population_as_pulse_population(pair.published)
    pulse = pulse_cpu.to(execution_device)
    corrupt = pair.published.corrupt.to(execution_device)
    published_lower = ((pulse.min_bound + 1.0) / 2.0).to(torch.float32)
    published_upper = ((pulse.max_bound + 1.0) / 2.0).to(torch.float32)
    healthy = ~corrupt
    maximum_device = maximum.to(execution_device)
    baseline_device = baseline.to(execution_device)
    baseline_g_device = baseline_g.to(execution_device)
    maximum_draws = (
        required_cuda_random_draws(
            pair.published, maximum_program_pulses=maximum_program_pulses
        )
        if execution_device.type == "cuda"
        else None
    )

    for sample_index, endpoint_seed in enumerate(seeds):
        trajectory_seeds = matched_trajectory_seeds(
            pair.published, endpoint_seed=endpoint_seed
        )
        normal_draws = (
            make_buffered_normal_draws(
                trajectory_seeds,
                maximum_random_draws=maximum_draws,
                device=execution_device,
            )
            if maximum_draws is not None
            else None
        )
        for level in range(levels):
            supported = maximum_device >= level
            controller_eligible = supported & healthy
            offset_g = torch.tensor(
                2.0 * float(level) * spacing,
                dtype=torch.float64,
                device=execution_device,
            ).to(torch.float32)
            level_target = (baseline_g_device + offset_g) / 2.0
            # Cells with a smaller repaired capacity stay at their baseline;
            # only the supported subset participates in this level table.
            target = torch.where(supported, level_target, baseline_device)
            if (
                bool(torch.any(target[healthy] < published_lower[healthy] - 1e-6))
                or bool(torch.any(target[healthy] > published_upper[healthy] + 1e-6))
            ):
                raise RuntimeError("A healthy codebook target left published support.")
            plant = IbmReramRawActivePlant(
                pulse,
                seeds=trajectory_seeds,
                device=execution_device,
                maximum_random_draws=maximum_draws,
                normal_draws=normal_draws,
            )
            plant.initialize_at_sampled_lower()
            result = run_program_verify(
                plant.controller_port(),
                targets=target,
                tolerance=tolerance,
                maximum_pulses=maximum_program_pulses,
                settings=ControllerSettings(kind="one_pulse"),
                eligible=controller_eligible,
            )
            persistent = (plant.persistent + 1.0) / 2.0
            if (
                not bool(torch.all(torch.isfinite(persistent[supported])))
                or bool(torch.any(persistent[supported] < 0.0))
                or bool(torch.any(persistent[supported] > 1.0))
                or not torch.equal(
                    plant.persistent[corrupt], pulse.min_bound[corrupt]
                )
            ):
                raise RuntimeError("Published endpoint table violated a physical invariant.")
            selected = persistent[supported]
            row = table[sample_index, :, level]
            row[supported.detach().cpu()] = selected.detach().cpu().to(torch.float32)
            persistent_inside = controller_eligible & (
                torch.abs(persistent - target) <= tolerance
            )
            eligible_count = int(supported.sum().item())
            total_pulses = result.total_pulses[supported].to(torch.float64)
            diagnostics.append(
                PersistentEndpointLevelDiagnostics(
                    sample_index=sample_index,
                    endpoint_seed=endpoint_seed,
                    level=level,
                    eligible_cells=eligible_count,
                    apparent_accepted=int(result.accepted[supported].sum().item()),
                    persistent_inside_tolerance=int(
                        persistent_inside[supported].sum().item()
                    ),
                    budget_exhausted=int(
                        result.budget_exhausted[supported].sum().item()
                    ),
                    mean_total_pulses=float(total_pulses.mean().item()),
                )
            )

    ensemble = PersistentEndpointCodebookEnsemble(
        assignment_seed=pair.assignment_seed,
        population_fingerprint=pair.published.fingerprint,
        binding_shapes=pair.published.binding_shapes,
        endpoint_seeds=seeds,  # type: ignore[arg-type]
        spacing_raw_x=spacing,
        tolerance_raw_x=tolerance,
        maximum_program_pulses=maximum_program_pulses,
        baseline_raw_x=baseline,
        maximum_index=maximum,
        persistent_raw_x=table,
        level_diagnostics=tuple(diagnostics),
    )
    corrupt_cpu = pair.published.corrupt.cpu()
    stuck_x = ((pair.published.min_bound + 1.0) / 2.0).cpu()
    if bool(torch.any(corrupt_cpu)):
        expected = stuck_x[corrupt_cpu].reshape(1, -1, 1).expand(
            ENSEMBLE_SIZE, -1, ensemble.num_levels
        )
        observed = ensemble.persistent_raw_x[:, corrupt_cpu, :]
        valid = (
            torch.arange(ensemble.num_levels).reshape(1, 1, -1)
            <= ensemble.maximum_index[corrupt_cpu].reshape(1, -1, 1)
        ).expand_as(observed)
        if not torch.equal(observed[valid], expected[valid]):
            raise RuntimeError("Corrupt codebook rows are not immutable stuck values.")
    return ensemble


# Re-export the already strict, weights-only-safe persistent-state helpers.
# Their contract is agnostic to whether the physical population contains
# immutable corrupt cells; the continuation state itself remains authoritative.
from experiments.mnist_relu_drn.ibm_om_winsorized_onchip_adam import (  # noqa: E402
    clone_p0_bundle,
    load_p0_bundle,
    plant_from_p0,
    save_p0_bundle,
    validate_p0_bundle,
)


__all__ = [
    "EVIDENCE_TIER",
    "EXPERIMENT_ID",
    "FULL_CONDUCTANCE_FORMULA",
    "PairedWinsorizedOmPopulation",
    "PublishedFaultClampedConductance",
    "PublishedPersistentTargets",
    "build_corrupt_persistent_endpoint_codebook_ensemble",
    "build_published_persistent_targets",
    "build_repaired_mapping_templates",
    "clamp_targets_to_published_corrupt_singletons",
    "clone_p0_bundle",
    "load_p0_bundle",
    "pair_and_winsorize_om_populations",
    "plant_from_p0",
    "sample_paired_winsorized_om_populations",
    "sample_paired_winsorized_om_populations_external",
    "save_p0_bundle",
    "validate_p0_bundle",
]
