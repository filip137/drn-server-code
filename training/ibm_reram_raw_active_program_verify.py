"""Raw-active IBM OM network-scale program-and-verify primitives.

The established IBM pulse plant normally stores AIHWKit's tile-facing
``a-r`` state.  Four-device/no-fixed-reference deployment instead represents
each DRN branch with the complete raw active conductance ``a``.  This module
provides the narrow conversion and execution surface needed by that scheme:

* the pulse plant stores/evolves persistent raw ``a``;
* the controller observes only noisy apparent ``x=(a+1)/2``;
* exact bounds and persistent state remain analysis-side facts; and
* noiseless inference consumes the final persistent raw endpoint, converted
  affinely to the DRN's full nonnegative conductance ``G``.

No baseline is subtracted here.  A caller that supplied ``G=B+n*h`` receives
the programmed complete ``G`` back for both transfer and KCL loading.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence

import torch

from training.ibm_reram_hwa import IbmReramArrayPopulation
from training.ibm_reram_program_verify import (
    ControllerSettings,
    IbmReramPopulation,
    IbmReramRawActivePlant,
    OM_PRESET,
    PopulationStepEstimator,
    ProgramVerifyResult,
    derive_seed,
    run_program_verify,
)


RAW_ACTIVE_COORDINATE = "x=(native_raw_active_a+1)/2"
RAW_ACTIVE_SUPPORT_TOLERANCE_UNIT = 1e-6
TRAJECTORY_SEED_DERIVATION = (
    "derive_seed(endpoint_seed,assignment_seed,binding_sampling_seeds,"
    "raw_active_pv_v1) "
    "followed by unique contiguous positive 63-bit trajectory seeds"
)


def array_population_as_pulse_population(
    population: IbmReramArrayPopulation,
) -> IbmReramPopulation:
    """Adapt a frozen network assignment without changing sampled parameters.

    AIHWKit samples each named array binding from one recorded construction
    seed, rather than assigning a separately recorded construction seed to
    every tile coordinate.  The returned vector therefore repeats each
    binding's literal sampling seed at every cell in that binding.  The
    binding key, local coordinate, array fingerprint, and sampled parameter
    values remain the complete physical identity in the source artifact.
    """

    if not isinstance(population, IbmReramArrayPopulation):
        raise TypeError("Expected one frozen IbmReramArrayPopulation.")
    if population.aihwkit_version != "1.1.0":
        raise ValueError("Expected the frozen IBM OM assignment to use AIHWKit 1.1.0.")

    construction_seeds = torch.cat(
        tuple(
            torch.full(
                (math.prod(shape),),
                int(seed),
                dtype=torch.int64,
                device="cpu",
            )
            for shape, seed in zip(
                population.binding_shapes,
                population.binding_sampling_seeds,
            )
        )
    )
    return IbmReramPopulation(
        preset=OM_PRESET,
        aihwkit_version=population.aihwkit_version,
        nominal_dw_min=float(population.nominal_dw_min),
        dw_min_std=float(population.dw_min_std),
        write_noise_std=float(population.write_noise_std),
        mult_noise=False,
        construction_seeds=construction_seeds,
        max_bound=population.max_bound.detach().to(device="cpu", dtype=torch.float32).clone(),
        min_bound=population.min_bound.detach().to(device="cpu", dtype=torch.float32).clone(),
        dwmin_up=population.dwmin_up.detach().to(device="cpu", dtype=torch.float32).clone(),
        dwmin_down=population.dwmin_down.detach().to(device="cpu", dtype=torch.float32).clone(),
        # Retained only as immutable identity provenance.  The raw-active
        # plant never reads this field during initialization or execution.
        reference=population.reference.detach().to(device="cpu", dtype=torch.float32).clone(),
        corrupt=population.corrupt.detach().to(device="cpu", dtype=torch.bool).clone(),
        preset_parameters={
            "source": "IbmReramArrayPopulation",
            "population_fingerprint": population.fingerprint,
            "assignment_seed": int(population.assignment_seed),
            "binding_keys": list(population.binding_keys),
            "binding_shapes": [list(shape) for shape in population.binding_shapes],
            "binding_sampling_seeds": list(population.binding_sampling_seeds),
            "state_coordinate": "native_raw_active_a",
            "reference_role": "unused_identity_provenance",
        },
    )


def matched_trajectory_seeds(
    population: IbmReramArrayPopulation,
    *,
    endpoint_seed: int,
) -> tuple[int, ...]:
    """Return one explicit, order-stable pulse seed per physical cell.

    The seed stream intentionally excludes target, baseline position, and
    spacing.  Reusing an endpoint seed on the same frozen assignment therefore
    gives matched latent random streams across all ``(B,h)`` arms.
    """

    if not isinstance(population, IbmReramArrayPopulation):
        raise TypeError("Expected one frozen IbmReramArrayPopulation.")
    if isinstance(endpoint_seed, bool) or not isinstance(endpoint_seed, int):
        raise TypeError("Expected endpoint_seed to be an integer.")
    modulus = 2**63 - 1
    first = derive_seed(
        endpoint_seed,
        population.assignment_seed,
        *population.binding_sampling_seeds,
        "raw_active_pv_v1",
    )
    return tuple((first - 1 + index) % modulus + 1 for index in range(population.size))


def required_cuda_random_draws(
    population: IbmReramArrayPopulation | IbmReramPopulation,
    *,
    maximum_program_pulses: int,
) -> int:
    """Exact worst-case per-trajectory draw budget for lower-start P&V."""

    if (
        isinstance(maximum_program_pulses, bool)
        or not isinstance(maximum_program_pulses, int)
        or maximum_program_pulses < 1
    ):
        raise ValueError("Expected a positive maximum_program_pulses value.")
    noisy = float(population.write_noise_std) > 0.0
    # One apparent sample after RESET conditioning, then one cycle draw and,
    # when enabled, one apparent write draw for every target pulse.
    return int(noisy) + maximum_program_pulses * (1 + int(noisy))


def raw_active_unit_to_full_conductance(
    endpoint_unit: torch.Tensor,
    *,
    conductance_min: float,
    conductance_max: float,
) -> torch.Tensor:
    """Map raw-active ``x`` to full nonnegative DRN conductance ``G``."""

    lower = float(conductance_min)
    upper = float(conductance_max)
    value = torch.as_tensor(endpoint_unit)
    if (
        not math.isfinite(lower)
        or not math.isfinite(upper)
        or lower < 0.0
        or upper <= lower
        or not bool(torch.all(torch.isfinite(value)))
    ):
        raise ValueError("Expected finite raw endpoints and increasing nonnegative G bounds.")
    if bool(torch.any(value < 0.0)) or bool(torch.any(value > 1.0)):
        raise ValueError(
            "Raw-active endpoint leaves the public x=[0,1] conductance "
            "coordinate; use explicit public-coordinate projection and "
            "retain its masks before forming full G."
        )
    return lower + (upper - lower) * value


def raw_active_x_to_full_conductance(
    endpoint_unit: torch.Tensor,
    *,
    raw_x_origin: float,
    conductance_per_raw_x: float,
    conductance_ceiling: float,
) -> torch.Tensor:
    """Apply a positive affine raw-``x`` conductance embedding without clipping.

    ``raw_x_origin`` is the raw device coordinate that would represent zero
    conductance, while ``conductance_per_raw_x`` is the literal slope in
    conductance per raw-``x`` unit. Every supplied endpoint must lie strictly
    above the origin and at or below the declared circuit ceiling. Unlike
    :func:`project_raw_active_unit_to_full_conductance`, this conversion never
    projects, clamps, or otherwise changes a device endpoint.

    The calculation is performed in float64 and returned in the input floating
    dtype. This keeps the positivity/support checks independent of float32
    subtraction round-off while preserving the circuit tensor dtype expected
    by callers.
    """

    origin = float(raw_x_origin)
    slope = float(conductance_per_raw_x)
    ceiling = float(conductance_ceiling)
    value = torch.as_tensor(endpoint_unit)
    if (
        value.numel() < 1
        or not value.is_floating_point()
        or not bool(torch.all(torch.isfinite(value)))
        or not math.isfinite(origin)
        or not math.isfinite(slope)
        or not math.isfinite(ceiling)
        or slope <= 0.0
        or ceiling <= 0.0
    ):
        raise ValueError(
            "Expected finite floating raw-x endpoints, a finite origin, and "
            "positive finite affine conductance parameters."
        )

    raw64 = value.to(dtype=torch.float64)
    full64 = (raw64 - origin) * slope
    tolerance = max(1e-15, ceiling * 1e-12)
    if bool(torch.any(raw64 <= origin)) or bool(torch.any(full64 <= 0.0)):
        raise ValueError(
            "Raw-active endpoint does not map to strictly positive "
            "conductance under the declared no-clipping affine embedding."
        )
    if bool(torch.any(full64 > ceiling + tolerance)):
        raise ValueError(
            "Raw-active endpoint exceeds the declared no-clipping affine "
            "conductance ceiling."
        )

    full = full64.to(dtype=value.dtype)
    if bool(torch.any(full <= 0.0)) or not bool(torch.all(torch.isfinite(full))):
        raise ValueError(
            "Affine raw-x conductance became non-positive or non-finite in "
            "the requested circuit dtype."
        )
    return full


@dataclass(frozen=True)
class PublicConductanceProjection:
    """Auditable handoff from native raw ``x`` to passive public conductance.

    ``raw_endpoint_unit`` remains the unmodified device/controller endpoint.
    ``applied_endpoint_unit`` is a separate circuit-facing tensor.  This
    object does not mutate a plant or its exact continuation state.
    """

    raw_endpoint_unit: torch.Tensor
    applied_endpoint_unit: torch.Tensor
    below_public_minimum: torch.Tensor
    above_public_maximum: torch.Tensor
    projection_delta_unit: torch.Tensor
    full_conductance: torch.Tensor
    conductance_min: float
    conductance_max: float
    policy: str = "hard_clip_raw_x_to_public_0_1_at_circuit_handoff"

    @property
    def projected(self) -> torch.Tensor:
        return self.below_public_minimum | self.above_public_maximum

    @property
    def projected_count(self) -> int:
        return int(self.projected.sum().item())

    def report(self) -> Mapping[str, object]:
        displacement = self.projection_delta_unit.to(torch.float64)
        absolute = displacement.abs()
        return {
            "policy": self.policy,
            "native_controller_state_changed": False,
            "persistent_continuation_state_changed": False,
            "coordinate_before": RAW_ACTIVE_COORDINATE,
            "coordinate_applied": "public_x_clipped_to_[0,1]",
            "devices": int(displacement.numel()),
            "below_public_minimum": int(self.below_public_minimum.sum().item()),
            "above_public_maximum": int(self.above_public_maximum.sum().item()),
            "projected": self.projected_count,
            "raw_minimum": float(self.raw_endpoint_unit.min().item()),
            "raw_maximum": float(self.raw_endpoint_unit.max().item()),
            "applied_minimum": float(self.applied_endpoint_unit.min().item()),
            "applied_maximum": float(self.applied_endpoint_unit.max().item()),
            "projection_signed_mean": float(displacement.mean().item()),
            "projection_mae": float(absolute.mean().item()),
            "projection_rmse": float(displacement.square().mean().sqrt().item()),
            "projection_maximum_absolute": float(absolute.max().item()),
            "conductance_min": self.conductance_min,
            "conductance_max": self.conductance_max,
        }


def project_raw_active_unit_to_full_conductance(
    endpoint_unit: torch.Tensor,
    *,
    conductance_min: float,
    conductance_max: float,
) -> PublicConductanceProjection:
    """Project only the circuit handoff into the public passive ``x`` range.

    IBM OM identities can have sampled native bounds beyond ``a in [-1,1]``.
    The repository's public conductance coordinate nevertheless assigns
    ``x=(a+1)/2`` only on ``[0,1]``.  A noisy apparent admission can therefore
    leave the persistent or apparent endpoint outside the passive range even
    when its requested target was in range.  Hard projection is the declared
    public-coordinate mapping for evaluation; it is not a controller verify,
    programming pulse, or modification of the saved persistent native state.
    """

    lower = float(conductance_min)
    upper = float(conductance_max)
    raw = torch.as_tensor(endpoint_unit)
    if (
        raw.numel() < 1
        or not bool(torch.all(torch.isfinite(raw)))
        or not math.isfinite(lower)
        or not math.isfinite(upper)
        or lower < 0.0
        or upper <= lower
    ):
        raise ValueError(
            "Expected finite raw endpoints and increasing nonnegative G bounds."
        )
    below = raw < 0.0
    above = raw > 1.0
    applied = raw.clamp(0.0, 1.0)
    full = lower + (upper - lower) * applied
    return PublicConductanceProjection(
        raw_endpoint_unit=_cpu(raw),
        applied_endpoint_unit=_cpu(applied),
        below_public_minimum=_cpu(below),
        above_public_maximum=_cpu(above),
        projection_delta_unit=_cpu(applied - raw),
        full_conductance=_cpu(full),
        conductance_min=lower,
        conductance_max=upper,
    )


def _cpu(value: torch.Tensor) -> torch.Tensor:
    return value.detach().to(device="cpu").clone()


def _cpu_program_verify_result(value: ProgramVerifyResult) -> ProgramVerifyResult:
    return ProgramVerifyResult(
        accepted=_cpu(value.accepted),
        nonfinite=_cpu(value.nonfinite),
        budget_exhausted=_cpu(value.budget_exhausted),
        apparent_endpoint=_cpu(value.apparent_endpoint),
        set_count=_cpu(value.set_count),
        reset_count=_cpu(value.reset_count),
        total_pulses=_cpu(value.total_pulses),
        verify_count=_cpu(value.verify_count),
        reversals=_cpu(value.reversals),
    )


def _cpu_continuation_state(state: Mapping[str, object]) -> dict[str, object]:
    result: dict[str, object] = {}
    for name, value in state.items():
        result[name] = _cpu(value) if isinstance(value, torch.Tensor) else value
    return result


@dataclass(frozen=True)
class RawActiveProgramVerifyOutcome:
    """Persistent and apparent endpoints kept as explicitly different facts."""

    endpoint_seed: int
    trajectory_seed_derivation: str
    coordinate: str
    tolerance_unit: float
    maximum_program_pulses: int
    controller: str
    target_unit: torch.Tensor
    raw_lower_unit: torch.Tensor
    raw_upper_unit: torch.Tensor
    exact_target_in_support: torch.Tensor
    initial_persistent_unit: torch.Tensor
    initial_apparent_unit: torch.Tensor
    programming: ProgramVerifyResult
    persistent_endpoint_unit: torch.Tensor
    apparent_endpoint_unit: torch.Tensor
    persistent_inside_acceptance_window: torch.Tensor
    saturated_lower: torch.Tensor
    saturated_upper: torch.Tensor
    continuation_state: Mapping[str, object]
    inference_read_noise_enabled: bool = False


@dataclass(frozen=True)
class UniformCodeClassification:
    """Nearest persistent code is independent of apparent P&V acceptance."""

    requested_index: torch.Tensor
    nearest_persistent_index: torch.Tensor
    requested_code_correct: torch.Tensor
    persistent_residual_to_requested: torch.Tensor
    persistent_residual_to_nearest: torch.Tensor


def classify_persistent_uniform_codes(
    persistent_endpoint_unit: torch.Tensor,
    *,
    baseline_unit: torch.Tensor,
    spacing_unit: float,
    requested_index: torch.Tensor,
    maximum_index: torch.Tensor,
) -> UniformCodeClassification:
    """Classify persistent endpoints on a bounded one-sided ``B+n*h`` grid."""

    endpoint = torch.as_tensor(persistent_endpoint_unit, dtype=torch.float64)
    baseline = torch.as_tensor(baseline_unit, dtype=torch.float64)
    requested = torch.as_tensor(requested_index, dtype=torch.int64)
    maximum = torch.as_tensor(maximum_index, dtype=torch.int64)
    spacing = float(spacing_unit)
    if (
        endpoint.shape != baseline.shape
        or endpoint.shape != requested.shape
        or endpoint.shape != maximum.shape
        or endpoint.numel() < 1
        or not bool(torch.all(torch.isfinite(endpoint)))
        or not bool(torch.all(torch.isfinite(baseline)))
        or not math.isfinite(spacing)
        or spacing <= 0.0
        or bool(torch.any(maximum < 0))
        or bool(torch.any(requested < 0))
        or bool(torch.any(requested > maximum))
    ):
        raise ValueError("Expected a finite bounded one-sided uniform codebook.")
    continuous_index = (endpoint - baseline) / spacing
    # All valid code indices are nonnegative, so floor(x+0.5) is the declared
    # half-up tie rule.  Clamp only the candidate index, never the endpoint.
    nearest = torch.floor(continuous_index + 0.5).to(torch.int64)
    nearest = torch.maximum(nearest, torch.zeros_like(nearest))
    nearest = torch.minimum(nearest, maximum)
    requested_code = baseline + spacing * requested.to(torch.float64)
    nearest_code = baseline + spacing * nearest.to(torch.float64)
    return UniformCodeClassification(
        requested_index=requested.detach().to(device="cpu").clone(),
        nearest_persistent_index=nearest.detach().to(device="cpu").clone(),
        requested_code_correct=(nearest == requested).detach().to(device="cpu").clone(),
        persistent_residual_to_requested=(endpoint - requested_code)
        .detach()
        .to(device="cpu")
        .clone(),
        persistent_residual_to_nearest=(endpoint - nearest_code)
        .detach()
        .to(device="cpu")
        .clone(),
    )


def run_raw_active_program_verify(
    population: IbmReramArrayPopulation,
    *,
    targets_unit: torch.Tensor,
    endpoint_seed: int,
    tolerance_unit: float,
    maximum_program_pulses: int,
    device: torch.device | str,
    settings: ControllerSettings | None = None,
    estimator: PopulationStepEstimator | None = None,
) -> RawActiveProgramVerifyOutcome:
    """Program full raw-active targets from sampled RESET on explicit streams.

    Target-support validation occurs before the capability-limited port is
    created.  The controller sees only the target, noisy apparent verifies,
    and its own history through :func:`run_program_verify`.  Accuracy callers
    must use ``persistent_endpoint_unit``; ``apparent_endpoint_unit`` and
    ``programming.accepted`` are diagnostics, not an inference state.

    ``settings`` defaults to the frozen one-pulse controller.  Supplying an
    adaptive setting and estimator is supported for a separately declared
    future control without changing the raw-state coordinate.
    """

    if not isinstance(population, IbmReramArrayPopulation):
        raise TypeError("Expected one frozen IbmReramArrayPopulation.")
    execution_device = torch.device(device)
    if execution_device.type not in {"cpu", "cuda"}:
        raise ValueError("Expected raw-active P&V device to be cpu or cuda.")
    if execution_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Expected CUDA for raw-active P&V execution.")
    tolerance = float(tolerance_unit)
    if not math.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("Expected a positive finite raw-active tolerance.")

    pulse_population = array_population_as_pulse_population(population)
    target = torch.as_tensor(
        targets_unit, dtype=torch.float32, device=execution_device
    ).reshape(-1)
    if target.shape != (population.size,) or not bool(torch.all(torch.isfinite(target))):
        raise ValueError(
            "Expected one finite raw-active target per frozen physical cell."
        )
    lower = ((pulse_population.min_bound + 1.0) / 2.0).to(execution_device)
    upper = ((pulse_population.max_bound + 1.0) / 2.0).to(execution_device)
    support_tolerance = RAW_ACTIVE_SUPPORT_TOLERANCE_UNIT
    exact_support = (
        ~pulse_population.corrupt.to(execution_device)
        & (target >= lower - support_tolerance)
        & (target <= upper + support_tolerance)
    )
    if not bool(torch.all(exact_support)):
        raise ValueError(
            "Expected every raw-active P&V target to be in exact persistent "
            f"support; unsupported={int((~exact_support).sum().item())}."
        )

    selected_settings = settings or ControllerSettings(kind="one_pulse")
    seeds = matched_trajectory_seeds(population, endpoint_seed=endpoint_seed)
    maximum_draws = (
        required_cuda_random_draws(
            population,
            maximum_program_pulses=maximum_program_pulses,
        )
        if execution_device.type == "cuda"
        else None
    )
    plant = IbmReramRawActivePlant(
        pulse_population,
        seeds=seeds,
        device=execution_device,
        maximum_random_draws=maximum_draws,
    )
    plant.initialize_at_sampled_lower()
    initial_persistent = (plant.persistent + 1.0) / 2.0
    initial_apparent = (plant.apparent + 1.0) / 2.0
    programmed = run_program_verify(
        plant.controller_port(),
        targets=target,
        tolerance=tolerance,
        maximum_pulses=maximum_program_pulses,
        settings=selected_settings,
        estimator=estimator,
    )
    persistent = (plant.persistent + 1.0) / 2.0
    apparent = (plant.apparent + 1.0) / 2.0
    if not torch.equal(apparent, programmed.apparent_endpoint):
        raise RuntimeError("Expected raw-active controller and plant apparent endpoints to match.")
    persistent_inside = torch.abs(persistent - target) <= tolerance
    native_tolerance = 2e-6
    saturated_lower = torch.abs(plant.persistent - plant.population.min_bound) <= native_tolerance
    saturated_upper = torch.abs(plant.persistent - plant.population.max_bound) <= native_tolerance

    return RawActiveProgramVerifyOutcome(
        endpoint_seed=int(endpoint_seed),
        trajectory_seed_derivation=TRAJECTORY_SEED_DERIVATION,
        coordinate=RAW_ACTIVE_COORDINATE,
        tolerance_unit=tolerance,
        maximum_program_pulses=int(maximum_program_pulses),
        controller=selected_settings.kind,
        target_unit=_cpu(target),
        raw_lower_unit=_cpu(lower),
        raw_upper_unit=_cpu(upper),
        exact_target_in_support=_cpu(exact_support),
        initial_persistent_unit=_cpu(initial_persistent),
        initial_apparent_unit=_cpu(initial_apparent),
        programming=_cpu_program_verify_result(programmed),
        persistent_endpoint_unit=_cpu(persistent),
        apparent_endpoint_unit=_cpu(apparent),
        persistent_inside_acceptance_window=_cpu(persistent_inside),
        saturated_lower=_cpu(saturated_lower),
        saturated_upper=_cpu(saturated_upper),
        continuation_state=_cpu_continuation_state(plant.state_dict()),
        inference_read_noise_enabled=False,
    )


__all__ = [
    "RAW_ACTIVE_COORDINATE",
    "RAW_ACTIVE_SUPPORT_TOLERANCE_UNIT",
    "PublicConductanceProjection",
    "RawActiveProgramVerifyOutcome",
    "TRAJECTORY_SEED_DERIVATION",
    "UniformCodeClassification",
    "array_population_as_pulse_population",
    "classify_persistent_uniform_codes",
    "matched_trajectory_seeds",
    "project_raw_active_unit_to_full_conductance",
    "raw_active_x_to_full_conductance",
    "raw_active_unit_to_full_conductance",
    "required_cuda_random_draws",
    "run_raw_active_program_verify",
]
