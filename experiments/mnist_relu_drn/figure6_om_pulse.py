"""Figure-6 endpoint model driven by literal IBM-OM pulse identities.

The synthetic endpoint field remains the conductance model:

``G = G_RESET + p * (G_SET - G_RESET)``.

Only the transition dynamics are supplied by AIHWKit 1.1.0's IBM optimized-
material (OM) preset.  Healthy devices use the sampled per-cell ``dwmin_up``
and ``dwmin_down`` values with the OM additive soft-bounds, cycle-noise, and
write-noise law in ``a=2*p-1``.  Thus the synthetic Figure-6 endpoints replace
the OM bound locations, but not its pulse dynamics.

Published corrupt cells retain their literal sampled singleton raw-``a``
coordinate and zero pulse steps.  The paired counterfactual-repaired view
replaces only those coordinates with healthy donors; every originally healthy
identity remains bitwise identical between the two views.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence

import torch

from experiments.mnist_relu_drn.hfo2_figure6_drn import (
    EndpointField,
    PulseStepReport,
    flatten_physical,
    master_to_progress,
    progress_to_conductance,
)
from training.ibm_reram_hwa import IbmReramArrayPopulation
from training.ibm_reram_program_verify import (
    ControllerSettings,
    ProgramVerifyResult,
    run_program_verify,
)


OM_NOMINAL_DW_MIN_RAW_A = 0.0949
OM_NOMINAL_DW_MIN_PROGRESS = OM_NOMINAL_DW_MIN_RAW_A / 2.0
OM_CYCLE_NOISE_STD = 0.4158
OM_WRITE_NOISE_STD = 1.4113
OM_PROGRAM_VERIFY_TOLERANCE_PROGRESS = OM_NOMINAL_DW_MIN_PROGRESS / 2.0
OM_PROGRAM_VERIFY_MAXIMUM_PULSES = 128

CORRUPTION_POLICIES = ("published", "counterfactual_repaired")
PULSE_MODEL_LABEL = (
    "AIHWKit_1.1.0_ReRamArrayOM_additive_soft_bounds_in_"
    "Figure6_normalized_progress"
)
RECOVERY_CLAIM_LABEL = "hardware-in-loop OM-pulse-mediated recovery"


def _split_flat(
    value: torch.Tensor,
    shapes: Sequence[Sequence[int]],
) -> tuple[torch.Tensor, ...]:
    result = []
    offset = 0
    flat = value.reshape(-1)
    for raw_shape in shapes:
        shape = tuple(int(item) for item in raw_shape)
        count = math.prod(shape)
        result.append(flat[offset : offset + count].reshape(shape))
        offset += count
    if offset != flat.numel():
        raise ValueError("Physical shapes do not cover the flattened OM state.")
    return tuple(result)


def validate_paired_om_populations(
    published: IbmReramArrayPopulation,
    repaired: IbmReramArrayPopulation,
) -> Mapping[str, Any]:
    """Prove that repair changes only the literal published corrupt cells."""

    scalar_names = (
        "assignment_seed",
        "binding_keys",
        "binding_shapes",
        "binding_sampling_seeds",
        "donor_sampling_seeds",
        "nominal_dw_min",
        "dw_min_std",
        "write_noise_std",
        "aihwkit_version",
    )
    if any(getattr(published, name) != getattr(repaired, name) for name in scalar_names):
        raise ValueError("OM published/repaired populations do not name one assignment.")
    if (
        published.corruption_policy != "published"
        or repaired.corruption_policy != "counterfactual_repaired"
        or not torch.equal(published.corrupt, published.published_corrupt)
        or bool(torch.any(repaired.corrupt))
        or not torch.equal(published.published_corrupt, repaired.published_corrupt)
    ):
        raise ValueError("Expected a literal published population and paired repair.")
    healthy = ~published.corrupt
    parameters = ("max_bound", "min_bound", "dwmin_up", "dwmin_down", "reference")
    for name in parameters:
        if not torch.equal(getattr(published, name)[healthy], getattr(repaired, name)[healthy]):
            raise ValueError(f"Healthy OM identities changed during repair: {name}.")
    corrupt = published.corrupt
    if bool(torch.any(corrupt)) and (
        not torch.equal(published.min_bound[corrupt], published.max_bound[corrupt])
        or bool(torch.any(published.dwmin_up[corrupt] != 0.0))
        or bool(torch.any(published.dwmin_down[corrupt] != 0.0))
    ):
        raise ValueError("Published corrupt OM cells are not zero-step singletons.")
    stuck = published.min_bound[corrupt]
    if bool(torch.any(stuck < -1.0)) or bool(torch.any(stuck > 1.0)):
        raise ValueError("A published corrupt OM singleton left raw-a [-1,1].")
    return {
        "pairing": "same_literal_draw_repair_only_published_corrupt_cells",
        "assignment_seed": int(published.assignment_seed),
        "cells": published.size,
        "published_corrupt_cells": int(corrupt.sum().item()),
        "published_corrupt_fraction": float(corrupt.to(torch.float64).mean().item()),
        "healthy_cells_bitwise_identical": True,
        "published_population_fingerprint": published.fingerprint,
        "counterfactual_repaired_population_fingerprint": repaired.fingerprint,
        "published_stuck_raw_a_minimum": float(stuck.min().item()) if stuck.numel() else None,
        "published_stuck_raw_a_maximum": float(stuck.max().item()) if stuck.numel() else None,
        "published_stuck_progress_minimum": (
            float(((stuck + 1.0) / 2.0).min().item()) if stuck.numel() else None
        ),
        "published_stuck_progress_maximum": (
            float(((stuck + 1.0) / 2.0).max().item()) if stuck.numel() else None
        ),
    }


class Figure6OmPulsePlant:
    """Persistent Figure-6 cells updated by sampled IBM-OM pulse dynamics."""

    def __init__(
        self,
        field: EndpointField,
        initial_progress: Sequence[torch.Tensor],
        population: IbmReramArrayPopulation,
        *,
        pulse_noise_seed: int,
    ) -> None:
        selected = tuple(initial_progress)
        progress_to_conductance(selected, field)
        if (
            population.corruption_policy not in CORRUPTION_POLICIES
            or population.binding_shapes != field.shapes
            or population.size != field.devices
            or population.aihwkit_version != "1.1.0"
        ):
            raise ValueError("OM population does not match the Figure-6 endpoint field.")
        expected_scalars = (
            float(population.nominal_dw_min) == OM_NOMINAL_DW_MIN_RAW_A
            and float(population.dw_min_std) == OM_CYCLE_NOISE_STD
            and float(population.write_noise_std) == OM_WRITE_NOISE_STD
        )
        if not expected_scalars:
            raise ValueError("Sampled OM pulse constants differ from the pinned model.")

        self.field = field
        self.source_population = population
        self.device = field.reset[0].device
        self.dtype = field.reset[0].dtype
        if self.dtype != torch.float32:
            raise ValueError("The OM pulse plant requires float32 endpoint tensors.")
        self.corruption_policy = population.corruption_policy
        self.population_fingerprint = population.fingerprint
        self.corrupt = population.corrupt.to(self.device)
        self.published_corrupt = population.published_corrupt.to(self.device)
        self.dwmin_up_raw_a = population.dwmin_up.to(self.device, dtype=self.dtype)
        self.dwmin_down_raw_a = population.dwmin_down.to(self.device, dtype=self.dtype)
        self.native_stuck_raw_a = population.min_bound.to(
            self.device, dtype=self.dtype
        )
        self.stuck_raw_a = self.native_stuck_raw_a.clone()
        self.fault_intervention = "none"
        self.fault_observation_seed: int | None = None
        flat_progress = flatten_physical(selected).to(self.device, dtype=self.dtype)
        raw = 2.0 * flat_progress - 1.0
        self.raw_a = torch.where(self.corrupt, self.native_stuck_raw_a, raw)
        if bool(torch.any(self.raw_a < -1.0)) or bool(torch.any(self.raw_a > 1.0)):
            raise ValueError("Initial OM raw-a state left normalized [-1,1].")
        self.apparent_raw_a = self.raw_a.clone()
        self.noise_generator = torch.Generator(device=self.device)
        self.noise_generator.manual_seed(int(pulse_noise_seed))
        self.pulse_noise_seed = int(pulse_noise_seed)
        self.pulse_count = torch.zeros(
            population.size, dtype=torch.int64, device=self.device
        )

    def inject_post_pv_reset_stuck_faults(
        self,
        mask: torch.Tensor,
        *,
        observation_seed: int,
    ) -> Mapping[str, Any]:
        """Force the paired published-OM mask to RESET after clean P&V.

        The persistent coordinate of each selected cell becomes raw ``a=-1``
        (progress zero) and remains immutable.  The fault event makes exactly
        one noisy apparent observation for those cells only; all unselected
        persistent and apparent coordinates remain bitwise unchanged.
        """

        selected = torch.as_tensor(mask, device=self.device, dtype=torch.bool)
        if selected.shape != self.raw_a.shape:
            raise ValueError("Post-P&V fault mask does not match the OM plant.")
        if self.corruption_policy != "counterfactual_repaired" or bool(
            torch.any(self.corrupt)
        ):
            raise ValueError(
                "Post-P&V reset-stuck injection requires a clean "
                "counterfactual-repaired plant."
            )
        if not torch.equal(selected, self.published_corrupt):
            raise ValueError(
                "Post-P&V fault mask must equal the paired published-OM "
                "corruption mask."
            )
        if self.fault_intervention != "none":
            raise ValueError("Post-P&V faults have already been injected.")
        if isinstance(observation_seed, bool) or not isinstance(observation_seed, int):
            raise ValueError("Expected an integer fault observation seed.")

        persistent_before = self.raw_a.clone()
        apparent_before = self.apparent_raw_a.clone()
        self.corrupt.copy_(selected)
        self.stuck_raw_a[selected] = -1.0
        self.raw_a[selected] = -1.0
        observation_generator = torch.Generator(device=self.device)
        observation_generator.manual_seed(int(observation_seed))
        write = torch.randn(
            (int(selected.sum().item()),),
            generator=observation_generator,
            device=self.device,
            dtype=self.dtype,
        )
        self.apparent_raw_a[selected] = -1.0 + (
            OM_WRITE_NOISE_STD * OM_NOMINAL_DW_MIN_RAW_A * write
        )
        self.fault_intervention = "post_pv_reset_stuck_fault"
        self.fault_observation_seed = int(observation_seed)
        return {
            "intervention": self.fault_intervention,
            "mask_source": "paired_published_OM_population",
            "fault_cells": int(selected.sum().item()),
            "fault_fraction": float(selected.to(torch.float64).mean().item()),
            "persistent_target_progress": 0.0,
            "apparent_observation": "one_OM_write_noise_draw_at_fault_event",
            "observation_seed": self.fault_observation_seed,
            "nonfault_persistent_bitwise_unchanged": bool(
                torch.equal(self.raw_a[~selected], persistent_before[~selected])
            ),
            "nonfault_apparent_bitwise_unchanged": bool(
                torch.equal(
                    self.apparent_raw_a[~selected], apparent_before[~selected]
                )
            ),
        }

    @property
    def size(self) -> int:
        return int(self.raw_a.numel())

    @property
    def progress(self) -> tuple[torch.Tensor, torch.Tensor]:
        return _split_flat((self.raw_a + 1.0) / 2.0, self.field.shapes)  # type: ignore[return-value]

    @property
    def apparent_progress_unprojected(self) -> tuple[torch.Tensor, torch.Tensor]:
        return _split_flat(
            (self.apparent_raw_a + 1.0) / 2.0,
            self.field.shapes,
        )  # type: ignore[return-value]

    @property
    def apparent_progress_projected(self) -> tuple[torch.Tensor, torch.Tensor]:
        return tuple(
            value.clamp(0.0, 1.0) for value in self.apparent_progress_unprojected
        )  # type: ignore[return-value]

    @property
    def full_conductance(self) -> tuple[torch.Tensor, torch.Tensor]:
        return progress_to_conductance(self.progress, self.field)

    @property
    def apparent_full_conductance(self) -> tuple[torch.Tensor, torch.Tensor]:
        return progress_to_conductance(self.apparent_progress_projected, self.field)

    def sample_apparent(self) -> None:
        write = torch.randn(
            self.raw_a.shape,
            generator=self.noise_generator,
            device=self.device,
            dtype=self.dtype,
        )
        self.apparent_raw_a.copy_(
            self.raw_a + OM_WRITE_NOISE_STD * OM_NOMINAL_DW_MIN_RAW_A * write
        )

    def clone_for_new_pulse_phase(
        self,
        *,
        pulse_noise_seed: int,
        retain_apparent_observation: bool = True,
    ) -> "Figure6OmPulsePlant":
        clone = Figure6OmPulsePlant(
            self.field,
            tuple(torch.zeros_like(value) for value in self.progress),
            self.source_population,
            pulse_noise_seed=int(pulse_noise_seed),
        )
        if (
            not torch.equal(clone.dwmin_up_raw_a, self.dwmin_up_raw_a)
            or not torch.equal(clone.dwmin_down_raw_a, self.dwmin_down_raw_a)
        ):
            raise RuntimeError("OM pulse identities did not reproduce exactly.")
        clone.corrupt.copy_(self.corrupt)
        clone.stuck_raw_a.copy_(self.stuck_raw_a)
        clone.fault_intervention = self.fault_intervention
        clone.fault_observation_seed = self.fault_observation_seed
        clone.raw_a.copy_(self.raw_a)
        clone.apparent_raw_a.copy_(
            self.apparent_raw_a if retain_apparent_observation else self.raw_a
        )
        return clone

    def pulse(self, directions: torch.Tensor, *, pulse_cap: int) -> tuple[int, int]:
        """Apply at most one OM SET/RESET pulse per requested eligible cell."""

        direction = torch.as_tensor(directions, device=self.device, dtype=torch.int8)
        if (
            direction.shape != self.raw_a.shape
            or bool(torch.any((direction < -1) | (direction > 1)))
            or isinstance(pulse_cap, bool)
            or not isinstance(pulse_cap, int)
            or pulse_cap < 1
        ):
            raise ValueError("Invalid OM pulse direction or pulse cap.")
        selected = (direction != 0) & (self.pulse_count < pulse_cap)
        capped = (direction != 0) & ~selected
        if not bool(torch.any(selected)):
            return 0, int(capped.sum().item())

        before = self.raw_a.clone()
        cycle = torch.randn(
            self.raw_a.shape,
            generator=self.noise_generator,
            device=self.device,
            dtype=self.dtype,
        )
        candidate = self.raw_a.clone()
        up = selected & (direction > 0)
        down = selected & (direction < 0)
        if bool(torch.any(up)):
            response = self.dwmin_up_raw_a * (
                1.0 - self.raw_a + OM_CYCLE_NOISE_STD * cycle
            )
            candidate[up] = self.raw_a[up] + response[up]
        if bool(torch.any(down)):
            response = self.dwmin_down_raw_a * (
                1.0 + self.raw_a + OM_CYCLE_NOISE_STD * cycle
            )
            candidate[down] = self.raw_a[down] - response[down]
        candidate.clamp_(-1.0, 1.0)
        candidate[self.corrupt] = self.stuck_raw_a[self.corrupt]
        self.raw_a[selected] = candidate[selected]

        write = torch.randn(
            self.raw_a.shape,
            generator=self.noise_generator,
            device=self.device,
            dtype=self.dtype,
        )
        apparent = self.raw_a + (
            OM_WRITE_NOISE_STD * OM_NOMINAL_DW_MIN_RAW_A * write
        )
        self.apparent_raw_a[selected] = apparent[selected]
        self.pulse_count[selected] += 1
        changed = int((self.raw_a[selected] != before[selected]).sum().item())
        return changed, int(capped.sum().item())

    def state_dict(self) -> dict[str, Any]:
        return {
            "schema": "ebl.figure6_om_pulse_plant",
            "schema_version": 2,
            "endpoint_model": self.field.report(),
            "pulse_model": PULSE_MODEL_LABEL,
            "corruption_policy": self.corruption_policy,
            "population_fingerprint": self.population_fingerprint,
            "persistent_state_coordinate": "raw_a=2*progress-1",
            "published_corrupt_state": (
                "paired_mask_for_post_pv_reset_stuck_progress_zero"
                if self.fault_intervention == "post_pv_reset_stuck_fault"
                else "immutable_native_singleton_raw_a"
            ),
            "active_corrupt": self.corrupt.detach().cpu().clone(),
            "active_stuck_raw_a": self.stuck_raw_a.detach().cpu().clone(),
            "fault_intervention": self.fault_intervention,
            "fault_observation_seed": self.fault_observation_seed,
            "persistent_raw_a": self.raw_a.detach().cpu().clone(),
            "apparent_raw_a": self.apparent_raw_a.detach().cpu().clone(),
            "pulse_count": self.pulse_count.detach().cpu().clone(),
            "pulse_noise_seed": self.pulse_noise_seed,
            "pulse_noise_rng_state": self.noise_generator.get_state().cpu().clone(),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if (
            state.get("schema") != "ebl.figure6_om_pulse_plant"
            or state.get("schema_version") not in {1, 2}
            or state.get("pulse_model") != PULSE_MODEL_LABEL
            or state.get("corruption_policy") != self.corruption_policy
            or state.get("population_fingerprint") != self.population_fingerprint
            or state.get("pulse_noise_seed") != self.pulse_noise_seed
        ):
            raise ValueError("Figure-6 OM pulse-plant continuation mismatch.")
        if state.get("schema_version") == 2:
            active_corrupt = state.get("active_corrupt")
            active_stuck = state.get("active_stuck_raw_a")
            if (
                not isinstance(active_corrupt, torch.Tensor)
                or active_corrupt.shape != self.corrupt.shape
                or not isinstance(active_stuck, torch.Tensor)
                or active_stuck.shape != self.stuck_raw_a.shape
            ):
                raise ValueError("Invalid saved post-P&V fault state.")
            self.corrupt.copy_(active_corrupt.to(self.device, dtype=torch.bool))
            self.stuck_raw_a.copy_(
                active_stuck.to(self.device, dtype=self.dtype)
            )
            fault_intervention = state.get("fault_intervention", "none")
            if fault_intervention not in {"none", "post_pv_reset_stuck_fault"}:
                raise ValueError("Invalid saved OM fault intervention.")
            self.fault_intervention = fault_intervention
            fault_seed = state.get("fault_observation_seed")
            if fault_seed is not None and (
                isinstance(fault_seed, bool) or not isinstance(fault_seed, int)
            ):
                raise ValueError("Invalid saved fault observation seed.")
            self.fault_observation_seed = fault_seed
            if self.fault_intervention == "post_pv_reset_stuck_fault" and (
                self.corruption_policy != "counterfactual_repaired"
                or not torch.equal(self.corrupt, self.published_corrupt)
                or not torch.equal(
                    self.stuck_raw_a[self.corrupt],
                    torch.full_like(self.stuck_raw_a[self.corrupt], -1.0),
                )
            ):
                raise ValueError("Saved post-P&V fault mask or stuck state is invalid.")
        for name, target in (
            ("persistent_raw_a", self.raw_a),
            ("apparent_raw_a", self.apparent_raw_a),
            ("pulse_count", self.pulse_count),
        ):
            value = state.get(name)
            if not isinstance(value, torch.Tensor) or value.shape != target.shape:
                raise ValueError(f"Invalid saved OM pulse-plant field {name!r}.")
            target.copy_(value.to(device=self.device, dtype=target.dtype))
        if not torch.equal(
            self.raw_a[self.corrupt], self.stuck_raw_a[self.corrupt]
        ):
            raise ValueError("Saved OM state moved a published corrupt singleton.")
        rng = state.get("pulse_noise_rng_state")
        if not isinstance(rng, torch.Tensor):
            raise ValueError("Expected a saved OM pulse-noise RNG state.")
        self.noise_generator.set_state(rng.detach().cpu())

    def report(self) -> Mapping[str, Any]:
        progress = (self.raw_a + 1.0) / 2.0
        apparent = (self.apparent_raw_a + 1.0) / 2.0
        corrupt_pulses = self.pulse_count[self.corrupt]
        return {
            "claim_label": RECOVERY_CLAIM_LABEL,
            "pulse_model": PULSE_MODEL_LABEL,
            "corruption_policy": self.corruption_policy,
            "population_fingerprint": self.population_fingerprint,
            "cells": self.size,
            "corrupt_cells": int(self.corrupt.sum().item()),
            "published_corrupt_cells": int(self.published_corrupt.sum().item()),
            "fault_intervention": self.fault_intervention,
            "fault_observation_seed": self.fault_observation_seed,
            "persistent_progress_minimum": float(progress.min().item()),
            "persistent_progress_mean": float(progress.mean().item()),
            "persistent_progress_maximum": float(progress.max().item()),
            "apparent_below_RESET": int((apparent < 0.0).sum().item()),
            "apparent_above_SET": int((apparent > 1.0).sum().item()),
            "cells_at_reset": int((self.raw_a == -1.0).sum().item()),
            "cells_at_set": int((self.raw_a == 1.0).sum().item()),
            "total_pulses": int(self.pulse_count.sum().item()),
            "corrupt_pulse_attempts": int(corrupt_pulses.sum().item()),
            "pulsed_cells": int((self.pulse_count > 0).sum().item()),
            "maximum_pulses_per_cell": int(self.pulse_count.max().item()),
            "endpoint_bounds_replaced_by_figure6_model": True,
            "om_native_bound_dtod_used_for_healthy_endpoint_locations": False,
            "om_reference_used": False,
            "published_corrupt_singletons_preserved": (
                self.fault_intervention == "none"
                and self.corruption_policy == "published"
            ),
            "post_pv_fault_mask_matches_published_om": (
                self.fault_intervention != "post_pv_reset_stuck_fault"
                or torch.equal(self.corrupt, self.published_corrupt)
            ),
            "apparent_state_projection": "clamp_progress_to_[0,1]_before_positive_G",
        }


class _Figure6OmProgramVerifyPort:
    def __init__(self, plant: Figure6OmPulsePlant, *, maximum_pulses: int) -> None:
        self.__plant = plant
        self.__maximum_pulses = int(maximum_pulses)

    @property
    def size(self) -> int:
        return self.__plant.size

    def verify(self) -> torch.Tensor:
        return (self.__plant.apparent_raw_a.clone() + 1.0) / 2.0

    def apply_identical_pulses(
        self,
        directions: torch.Tensor,
        counts: torch.Tensor,
    ) -> None:
        direction = torch.as_tensor(directions, device=self.__plant.device, dtype=torch.int8)
        count = torch.as_tensor(counts, device=self.__plant.device, dtype=torch.int64)
        if direction.shape != (self.size,) or count.shape != (self.size,):
            raise ValueError("P&V directions and counts must match the OM plant.")
        if bool(torch.any(count < 0)):
            raise ValueError("P&V pulse counts must be non-negative.")
        for pulse_index in range(int(count.max().item()) if count.numel() else 0):
            active = torch.where(
                count > pulse_index,
                direction,
                torch.zeros_like(direction),
            )
            _, capped = self.__plant.pulse(
                active,
                pulse_cap=self.__maximum_pulses,
            )
            if capped:
                raise RuntimeError("The P&V controller exceeded its OM pulse budget.")


def program_progress_with_verify(
    field: EndpointField,
    target_progress: Sequence[torch.Tensor],
    population: IbmReramArrayPopulation,
    *,
    pulse_noise_seed: int,
    tolerance_progress: float = OM_PROGRAM_VERIFY_TOLERANCE_PROGRESS,
    maximum_pulses: int = OM_PROGRAM_VERIFY_MAXIMUM_PULSES,
    noisy_initial_reset_verify: bool = True,
    preaccept_exact_healthy_reset_targets: bool = True,
) -> tuple[Figure6OmPulsePlant, ProgramVerifyResult]:
    """Program Figure-6 targets from RESET using OM apparent-feedback pulses."""

    selected = tuple(target_progress)
    progress_to_conductance(selected, field)
    if not math.isfinite(float(tolerance_progress)) or tolerance_progress <= 0.0:
        raise ValueError("Expected a positive finite P&V tolerance.")
    if isinstance(maximum_pulses, bool) or maximum_pulses < 1:
        raise ValueError("Expected a positive P&V pulse budget.")
    plant = Figure6OmPulsePlant(
        field,
        tuple(torch.zeros_like(value) for value in selected),
        population,
        pulse_noise_seed=int(pulse_noise_seed),
    )
    if noisy_initial_reset_verify:
        plant.sample_apparent()
    target = flatten_physical(selected)
    preaccepted = (
        (target == 0.0) & ~plant.corrupt
        if preaccept_exact_healthy_reset_targets
        else torch.zeros_like(target, dtype=torch.bool)
    )
    raw = run_program_verify(
        _Figure6OmProgramVerifyPort(plant, maximum_pulses=maximum_pulses),
        targets=target,
        tolerance=float(tolerance_progress),
        maximum_pulses=int(maximum_pulses),
        settings=ControllerSettings(kind="one_pulse"),
        eligible=~preaccepted,
    )
    result = ProgramVerifyResult(
        accepted=raw.accepted | preaccepted,
        nonfinite=raw.nonfinite,
        budget_exhausted=raw.budget_exhausted,
        apparent_endpoint=raw.apparent_endpoint,
        set_count=raw.set_count,
        reset_count=raw.reset_count,
        total_pulses=raw.total_pulses,
        verify_count=raw.verify_count + preaccepted.to(torch.int64),
        reversals=raw.reversals,
    )
    if not torch.equal(plant.pulse_count, result.total_pulses):
        raise RuntimeError("OM plant and P&V pulse accounting diverged.")
    if not torch.equal(
        flatten_physical(plant.apparent_progress_unprojected),
        result.apparent_endpoint,
    ):
        raise RuntimeError("OM P&V result does not match its apparent endpoint.")
    return plant, result


def program_masters_with_verify(
    masters: Sequence[torch.Tensor],
    field: EndpointField,
    population: IbmReramArrayPopulation,
    **settings: Any,
) -> tuple[Figure6OmPulsePlant, ProgramVerifyResult]:
    return program_progress_with_verify(
        field,
        master_to_progress(masters, field),
        population,
        **settings,
    )


def corruption_constrained_progress(
    target_progress: Sequence[torch.Tensor],
    field: EndpointField,
    population: IbmReramArrayPopulation,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply only immutable corrupt singleton constraints to an ideal target."""

    selected = tuple(target_progress)
    progress_to_conductance(selected, field)
    flat = flatten_physical(selected)
    corrupt = population.corrupt.to(device=flat.device)
    stuck = ((population.min_bound + 1.0) / 2.0).to(
        device=flat.device, dtype=flat.dtype
    )
    constrained = torch.where(corrupt, stuck, flat)
    return _split_flat(constrained, field.shapes)  # type: ignore[return-value]


class PersistentFigure6OmPulseAdam:
    """Digital Adam commands realized solely as persistent IBM-OM pulses."""

    def __init__(
        self,
        plant: Figure6OmPulsePlant,
        *,
        learning_rate_progress: float,
        pulse_cap: int,
        pulse_selection_seed: int,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
    ) -> None:
        if any(
            not math.isfinite(float(value))
            for value in (learning_rate_progress, beta1, beta2, epsilon)
        ):
            raise ValueError("Expected finite OM pulse-Adam settings.")
        if learning_rate_progress <= 0.0 or epsilon <= 0.0:
            raise ValueError("Expected positive OM pulse-Adam learning settings.")
        if not (0.0 <= beta1 < 1.0 and 0.0 <= beta2 < 1.0):
            raise ValueError("Expected Adam beta values in [0,1).")
        if isinstance(pulse_cap, bool) or not isinstance(pulse_cap, int) or pulse_cap < 1:
            raise ValueError("Expected a positive OM recovery pulse cap.")
        self.plant = plant
        self.learning_rate_progress = float(learning_rate_progress)
        self.pulse_cap = int(pulse_cap)
        self.beta1 = float(beta1)
        self.beta2 = float(beta2)
        self.epsilon = float(epsilon)
        self.step_index = 0
        self.first_moment = torch.zeros_like(plant.raw_a)
        self.second_moment = torch.zeros_like(plant.raw_a)
        self.selection_generator = torch.Generator(device=plant.device)
        self.selection_generator.manual_seed(int(pulse_selection_seed))
        self.pulse_selection_seed = int(pulse_selection_seed)
        self.total_probability_clipped_cells = 0

    def step(
        self,
        physical_conductance_gradients: Sequence[torch.Tensor],
    ) -> PulseStepReport:
        gradients = tuple(physical_conductance_gradients)
        if len(gradients) != 2:
            raise ValueError("Expected two physical conductance gradients.")
        for gradient, shape in zip(gradients, self.plant.field.shapes, strict=True):
            if tuple(gradient.shape) != shape or not bool(torch.isfinite(gradient).all()):
                raise ValueError("Expected finite physical gradients of fixed shape.")
        gradient_progress = flatten_physical(
            tuple(
                gradient.detach() * window
                for gradient, window in zip(
                    gradients,
                    self.plant.field.window,
                    strict=True,
                )
            )
        )
        self.step_index += 1
        self.first_moment.mul_(self.beta1).add_(
            gradient_progress,
            alpha=1.0 - self.beta1,
        )
        self.second_moment.mul_(self.beta2).addcmul_(
            gradient_progress,
            gradient_progress,
            value=1.0 - self.beta2,
        )
        first_hat = self.first_moment / (1.0 - self.beta1**self.step_index)
        second_hat = self.second_moment / (1.0 - self.beta2**self.step_index)
        command = -self.learning_rate_progress * first_hat / (
            second_hat.sqrt() + self.epsilon
        )
        raw_probability = command.abs() / OM_NOMINAL_DW_MIN_PROGRESS
        probability = raw_probability.clamp(max=1.0)
        random_value = torch.rand(
            probability.shape,
            generator=self.selection_generator,
            device=self.plant.device,
            dtype=probability.dtype,
        )
        requested = (random_value < probability) & (command != 0.0)
        eligible = self.plant.pulse_count < self.pulse_cap
        selected = requested & eligible
        direction = torch.zeros_like(self.plant.raw_a, dtype=torch.int8)
        direction[selected & (command > 0.0)] = 1
        direction[selected & (command < 0.0)] = -1
        changed, capped_by_plant = self.plant.pulse(
            direction,
            pulse_cap=self.pulse_cap,
        )
        capped = int((requested & ~eligible).sum().item()) + capped_by_plant
        self.total_probability_clipped_cells += int(
            (raw_probability > 1.0).sum().item()
        )
        return PulseStepReport(
            step=self.step_index,
            requested_cells=int(requested.sum().item()),
            pulsed_cells=int(selected.sum().item()),
            capped_cells=capped,
            upward_pulses=int((direction > 0).sum().item()),
            downward_pulses=int((direction < 0).sum().item()),
            mean_probability=float(probability.mean().item()),
            maximum_probability=float(probability.max().item()),
            effective_state_changes=changed,
        )

    def state_dict(self) -> dict[str, Any]:
        return {
            "schema": "ebl.figure6_om_persistent_pulse_adam",
            "schema_version": 1,
            "claim_label": RECOVERY_CLAIM_LABEL,
            "population_fingerprint": self.plant.population_fingerprint,
            "learning_rate_progress": self.learning_rate_progress,
            "nominal_delta_progress": OM_NOMINAL_DW_MIN_PROGRESS,
            "pulse_cap": self.pulse_cap,
            "beta1": self.beta1,
            "beta2": self.beta2,
            "epsilon": self.epsilon,
            "step": self.step_index,
            "first_moment": self.first_moment.detach().cpu().clone(),
            "second_moment": self.second_moment.detach().cpu().clone(),
            "pulse_selection_seed": self.pulse_selection_seed,
            "pulse_selection_rng_state": self.selection_generator.get_state().cpu().clone(),
            "total_probability_clipped_cells": (
                self.total_probability_clipped_cells
            ),
            "authoritative_weight_shadow": None,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if (
            state.get("schema") != "ebl.figure6_om_persistent_pulse_adam"
            or state.get("schema_version") != 1
            or state.get("claim_label") != RECOVERY_CLAIM_LABEL
            or state.get("population_fingerprint") != self.plant.population_fingerprint
            or state.get("pulse_selection_seed") != self.pulse_selection_seed
            or float(state.get("learning_rate_progress", -1.0))
            != self.learning_rate_progress
            or int(state.get("pulse_cap", -1)) != self.pulse_cap
        ):
            raise ValueError("Figure-6 OM pulse-Adam continuation mismatch.")
        for name, target in (
            ("first_moment", self.first_moment),
            ("second_moment", self.second_moment),
        ):
            value = state.get(name)
            if not isinstance(value, torch.Tensor) or value.shape != target.shape:
                raise ValueError(f"Invalid saved OM pulse-Adam field {name!r}.")
            target.copy_(value.to(device=target.device, dtype=target.dtype))
        step = state.get("step")
        if isinstance(step, bool) or not isinstance(step, int) or step < 0:
            raise ValueError("Invalid saved OM pulse-Adam step.")
        self.step_index = step
        clipped = state.get("total_probability_clipped_cells", 0)
        if isinstance(clipped, bool) or not isinstance(clipped, int) or clipped < 0:
            raise ValueError("Invalid saved OM pulse-probability clipping count.")
        self.total_probability_clipped_cells = clipped
        rng = state.get("pulse_selection_rng_state")
        if not isinstance(rng, torch.Tensor):
            raise ValueError("Expected a saved OM pulse-selection RNG state.")
        self.selection_generator.set_state(rng.detach().cpu())


__all__ = [
    "CORRUPTION_POLICIES",
    "Figure6OmPulsePlant",
    "OM_CYCLE_NOISE_STD",
    "OM_NOMINAL_DW_MIN_PROGRESS",
    "OM_NOMINAL_DW_MIN_RAW_A",
    "OM_PROGRAM_VERIFY_MAXIMUM_PULSES",
    "OM_PROGRAM_VERIFY_TOLERANCE_PROGRESS",
    "OM_WRITE_NOISE_STD",
    "PULSE_MODEL_LABEL",
    "PersistentFigure6OmPulseAdam",
    "RECOVERY_CLAIM_LABEL",
    "corruption_constrained_progress",
    "program_masters_with_verify",
    "program_progress_with_verify",
    "validate_paired_om_populations",
]
