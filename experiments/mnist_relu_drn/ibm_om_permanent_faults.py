"""AIHWKit-compatible permanent-fault overlays for IBM OM raw-active cells.

The source array for this experiment is a counterfactually repaired IBM OM
population.  This module overlays a fresh, independently sampled population
of permanent corrupt devices without exposing their locations to the learner.
Each selected physical cell is collapsed at a value drawn uniformly from the
intersection of its sampled raw-active support and AIHWKit's published
``[-corrupt_devices_range, +corrupt_devices_range]`` interval.

Coordinates are deliberately explicit throughout this module.  ``raw_a`` is
AIHWKit's signed internal device coordinate; it is *not* a conductance.  The
DRN sees the complete non-negative conductance ``G = raw_a + 1 = 2*x``.  Thus
the default corrupt interval maps to approximately ``G in [0.99, 1.01]``.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import math
from typing import Any, Mapping, Protocol, Sequence

import torch

from training.ibm_reram_hwa import IbmReramArrayPopulation
from training.ibm_reram_program_verify import (
    IbmReramPopulation,
    PUBLISHED_CORRUPT_PROBABILITY,
    VerifyPort,
    derive_seed,
)
from training.ibm_reram_raw_active_program_verify import (
    array_population_as_pulse_population,
)


FAULT_OVERLAY_SCHEMA = "ebl.mnist_relu_drn.ibm_om_permanent_fault_overlay"
FAULT_OVERLAY_SCHEMA_VERSION = 1
MASK_RNG_DOMAIN = "ibm_om_permanent_fault_overlay.v1.mask"
RAW_A_STUCK_RNG_DOMAIN = "ibm_om_permanent_fault_overlay.v1.raw_a_stuck"
DEFAULT_CORRUPT_PROBABILITY = PUBLISHED_CORRUPT_PROBABILITY["reram_array_om"]
DEFAULT_CORRUPT_RAW_A_RANGE = 0.01
RAW_A_COORDINATE = "AIHWKit native signed raw_a (not conductance)"
FULL_CONDUCTANCE_COORDINATE = "G=raw_a+1=2*x"


def _tensor_sha256(value: torch.Tensor) -> str:
    cpu = value.detach().contiguous().cpu()
    digest = sha256()
    digest.update(str(cpu.dtype).encode("utf-8"))
    digest.update(json.dumps(list(cpu.shape), separators=(",", ":")).encode("utf-8"))
    digest.update(cpu.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _validate_seed(value: int, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < 2**63:
        raise ValueError(f"Expected {name} to be an integer in [0, 2**63).")
    return int(value)


def _binding_slices(
    shapes: Sequence[Sequence[int]],
) -> tuple[slice, ...]:
    result = []
    offset = 0
    for raw_shape in shapes:
        shape = tuple(int(item) for item in raw_shape)
        if not shape or any(item < 1 for item in shape):
            raise ValueError("Expected non-empty positive binding shapes.")
        count = math.prod(shape)
        result.append(slice(offset, offset + count))
        offset += count
    return tuple(result)


class WriteOnlyPulsePort(Protocol):
    """Learner-facing capability: size plus blind pulse requests only."""

    @property
    def size(self) -> int: ...

    def apply_identical_pulses(
        self,
        directions: torch.Tensor,
        counts: torch.Tensor,
    ) -> None: ...


@dataclass(frozen=True)
class PermanentFaultReceipt:
    """Serializable provenance and realized-count receipt for one overlay."""

    mask_seed: int
    raw_a_stuck_seed: int
    corrupt_probability: float
    corrupt_raw_a_range: float
    source_dw_min_std: float
    source_write_noise_std: float
    source_population_fingerprint: str
    binding_keys: tuple[str, ...]
    binding_shapes: tuple[tuple[int, ...], ...]
    realized_fault_count_by_binding: tuple[int, ...]
    realized_fault_count: int
    population_size: int
    mask_sha256: str
    selected_raw_a_stuck_sha256: str
    selected_full_conductance_stuck_sha256: str
    empty_support_intersection_count: int
    device_truncated_intersection_count: int
    raw_a_stuck_minimum: float | None
    raw_a_stuck_maximum: float | None
    full_conductance_stuck_minimum: float | None
    full_conductance_stuck_maximum: float | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": FAULT_OVERLAY_SCHEMA,
            "schema_version": FAULT_OVERLAY_SCHEMA_VERSION,
            "mechanism": (
                "independent_Bernoulli_crosspoint_selection_then_uniform_"
                "raw_a_stuck_value_in_device_support_intersection"
            ),
            "mask_rng": {
                "domain": MASK_RNG_DOMAIN,
                "seed": self.mask_seed,
                "backend": "torch.Generator(cpu)",
            },
            "raw_a_stuck_rng": {
                "domain": RAW_A_STUCK_RNG_DOMAIN,
                "seed": self.raw_a_stuck_seed,
                "backend": "torch.Generator(cpu)",
                "draw_policy": (
                    "one_uniform_draw_per_flat_crosspoint_then_retain_masked_draws"
                ),
            },
            "corrupt_probability": self.corrupt_probability,
            "corrupt_raw_a_range": self.corrupt_raw_a_range,
            "pulse_response_policy": {
                "heterogeneous_sampled_bounds_and_steps_retained": True,
                "source_dw_min_std": self.source_dw_min_std,
                "source_write_noise_std": self.source_write_noise_std,
                "overlay_dw_min_std": 0.0,
                "overlay_write_noise_std": 0.0,
                "cycle_to_cycle_noise_enabled": False,
                "apparent_write_noise_enabled": False,
                "only_pulse_selection_is_stochastic_during_recovery": True,
            },
            "source_population_fingerprint": self.source_population_fingerprint,
            "source_population_policy": "counterfactual_repaired",
            "binding_keys": list(self.binding_keys),
            "binding_shapes": [list(shape) for shape in self.binding_shapes],
            "realized_fault_count_by_binding": dict(
                zip(self.binding_keys, self.realized_fault_count_by_binding)
            ),
            "realized_fault_count": self.realized_fault_count,
            "realized_fault_fraction": self.realized_fault_count / self.population_size,
            "population_size": self.population_size,
            "mask_sha256": self.mask_sha256,
            "selected_raw_a_stuck_sha256": self.selected_raw_a_stuck_sha256,
            "selected_full_conductance_stuck_sha256": (
                self.selected_full_conductance_stuck_sha256
            ),
            "support_intersection_audit": {
                "empty_count": self.empty_support_intersection_count,
                "device_truncated_count": self.device_truncated_intersection_count,
            },
            "coordinates": {
                "device_state": RAW_A_COORDINATE,
                "circuit_facing_conductance": FULL_CONDUCTANCE_COORDINATE,
                "raw_a_stuck_minimum": self.raw_a_stuck_minimum,
                "raw_a_stuck_maximum": self.raw_a_stuck_maximum,
                "full_conductance_stuck_minimum": (
                    self.full_conductance_stuck_minimum
                ),
                "full_conductance_stuck_maximum": (
                    self.full_conductance_stuck_maximum
                ),
                "negative_full_conductance_stuck_count": 0,
            },
            "fault_cell_transition": {
                "min_bound": "raw_a_stuck",
                "max_bound": "raw_a_stuck",
                "dwmin_up": 0.0,
                "dwmin_down": 0.0,
                "both_pulse_directions_disabled": True,
            },
        }


class PermanentFaultWriteLedger:
    """Analysis-side pulse-request accounting, kept separate from the port."""

    def __init__(
        self,
        *,
        binding_keys: Sequence[str],
        binding_shapes: Sequence[Sequence[int]],
    ) -> None:
        self._binding_keys = tuple(str(key) for key in binding_keys)
        self._binding_slices = _binding_slices(binding_shapes)
        if len(self._binding_keys) != len(self._binding_slices):
            raise ValueError("Expected one binding key per shape.")
        self._attempted = torch.zeros(len(self._binding_keys), dtype=torch.int64)
        self._suppressed = torch.zeros_like(self._attempted)
        self._forwarded = torch.zeros_like(self._attempted)
        self._attempted_up = 0
        self._attempted_down = 0
        self._suppressed_up = 0
        self._suppressed_down = 0

    def _record(
        self,
        direction: torch.Tensor,
        attempted: torch.Tensor,
        suppressed: torch.Tensor,
    ) -> None:
        cpu_direction = direction.detach().to(device="cpu", dtype=torch.int8)
        cpu_attempted = attempted.detach().to(device="cpu", dtype=torch.int64)
        cpu_suppressed = suppressed.detach().to(device="cpu", dtype=torch.int64)
        for index, binding_slice in enumerate(self._binding_slices):
            attempted_count = cpu_attempted[binding_slice].sum()
            suppressed_count = cpu_suppressed[binding_slice].sum()
            self._attempted[index] += attempted_count
            self._suppressed[index] += suppressed_count
            self._forwarded[index] += attempted_count - suppressed_count
        self._attempted_up += int(cpu_attempted[cpu_direction > 0].sum().item())
        self._attempted_down += int(cpu_attempted[cpu_direction < 0].sum().item())
        self._suppressed_up += int(cpu_suppressed[cpu_direction > 0].sum().item())
        self._suppressed_down += int(cpu_suppressed[cpu_direction < 0].sum().item())

    def report(self) -> dict[str, Any]:
        attempted = int(self._attempted.sum().item())
        suppressed = int(self._suppressed.sum().item())
        forwarded = int(self._forwarded.sum().item())
        if attempted != suppressed + forwarded:
            raise RuntimeError("Permanent-fault write accounting is inconsistent.")
        return {
            "attempted_pulse_requests": attempted,
            "attempted_set_pulse_requests": self._attempted_up,
            "attempted_reset_pulse_requests": self._attempted_down,
            "suppressed_fault_pulse_requests": suppressed,
            "suppressed_fault_set_pulse_requests": self._suppressed_up,
            "suppressed_fault_reset_pulse_requests": self._suppressed_down,
            # Forwarded requests are intentionally not called applied pulses:
            # a healthy cell can already be at a sampled bound.
            "forwarded_healthy_pulse_requests": forwarded,
            "attempted_pulse_requests_by_binding": dict(
                zip(self._binding_keys, self._attempted.tolist())
            ),
            "suppressed_fault_pulse_requests_by_binding": dict(
                zip(self._binding_keys, self._suppressed.tolist())
            ),
            "forwarded_healthy_pulse_requests_by_binding": dict(
                zip(self._binding_keys, self._forwarded.tolist())
            ),
            "verify_reads": 0,
        }


class _PermanentFaultWritePort:
    """Blind write wrapper that suppresses both directions at fault sites."""

    __slots__ = ("__delegate", "__fault_mask", "__ledger")

    def __init__(
        self,
        delegate: VerifyPort,
        *,
        fault_mask: torch.Tensor,
        ledger: PermanentFaultWriteLedger,
    ) -> None:
        if fault_mask.shape != (delegate.size,) or fault_mask.dtype != torch.bool:
            raise ValueError("Expected a boolean fault mask matching the pulse port.")
        self.__delegate = delegate
        self.__fault_mask = fault_mask.detach().to(device="cpu", dtype=torch.bool).clone()
        self.__ledger = ledger

    @property
    def size(self) -> int:
        return self.__delegate.size

    def apply_identical_pulses(
        self,
        directions: torch.Tensor,
        counts: torch.Tensor,
    ) -> None:
        direction = torch.as_tensor(directions, dtype=torch.int8)
        count = torch.as_tensor(counts, dtype=torch.int64, device=direction.device)
        if direction.shape != (self.size,) or count.shape != (self.size,):
            raise ValueError("Expected directions and counts to match write-port size.")
        if bool(torch.any((direction < -1) | (direction > 1))):
            raise ValueError("Expected pulse directions in {-1,0,+1}.")
        if bool(torch.any(count < 0)):
            raise ValueError("Expected non-negative pulse counts.")
        attempted = torch.where(direction != 0, count, torch.zeros_like(count))
        fault_mask = self.__fault_mask.to(direction.device)
        suppressed = torch.where(fault_mask, attempted, torch.zeros_like(attempted))
        forwarded = attempted - suppressed
        forwarded_direction = torch.where(
            forwarded > 0,
            direction,
            torch.zeros_like(direction),
        )
        self.__ledger._record(direction, attempted, suppressed)
        self.__delegate.apply_identical_pulses(forwarded_direction, forwarded)


@dataclass(frozen=True)
class PermanentFaultOverlay:
    """Analysis-side overlay state and the collapsed pulse population."""

    pulse_population: IbmReramPopulation
    fault_mask: torch.Tensor
    raw_a_stuck: torch.Tensor
    binding_keys: tuple[str, ...]
    binding_shapes: tuple[tuple[int, ...], ...]
    receipt: PermanentFaultReceipt

    def __post_init__(self) -> None:
        size = self.pulse_population.size
        if (
            self.fault_mask.shape != (size,)
            or self.fault_mask.dtype != torch.bool
            or self.raw_a_stuck.shape != (size,)
            or self.raw_a_stuck.dtype != torch.float32
            or not bool(torch.all(torch.isfinite(self.raw_a_stuck)))
            or sum(math.prod(shape) for shape in self.binding_shapes) != size
            or len(self.binding_keys) != len(self.binding_shapes)
        ):
            raise ValueError("Invalid permanent-fault overlay tensors or bindings.")
        population = self.pulse_population
        if not torch.equal(population.corrupt.cpu(), self.fault_mask.cpu()):
            raise ValueError("Overlay fault mask and pulse-population corruption differ.")
        mask = self.fault_mask.to(population.min_bound.device)
        stuck = self.raw_a_stuck.to(population.min_bound.device)
        if not torch.equal(population.min_bound[mask], stuck[mask]) or not torch.equal(
            population.max_bound[mask], stuck[mask]
        ):
            raise ValueError("Fault-cell bounds must collapse exactly at raw_a_stuck.")
        if bool(torch.any(population.dwmin_up[mask] != 0.0)) or bool(
            torch.any(population.dwmin_down[mask] != 0.0)
        ):
            raise ValueError("Fault cells must disable both pulse directions.")
        full_lower = population.min_bound + 1.0
        if bool(torch.any(full_lower < 0.0)):
            raise ValueError(
                "Pulse population would expose a negative circuit conductance G=raw_a+1."
            )
        if self.receipt.realized_fault_count != int(self.fault_mask.sum().item()):
            raise ValueError("Receipt fault count does not match overlay mask.")

    @property
    def size(self) -> int:
        return self.pulse_population.size

    @property
    def full_conductance_stuck(self) -> torch.Tensor:
        """Return circuit-facing ``G=raw_a+1`` (healthy entries are placeholders)."""

        return self.raw_a_stuck + 1.0

    def clamp_raw_a_(self, raw_a: torch.Tensor) -> torch.Tensor:
        """Project continuous internal raw-a state and re-clamp every fault."""

        if raw_a.shape != (self.size,) or not bool(torch.all(torch.isfinite(raw_a))):
            raise ValueError("Expected one finite flat raw_a state vector.")
        lower = self.pulse_population.min_bound.to(raw_a.device)
        upper = self.pulse_population.max_bound.to(raw_a.device)
        raw_a.copy_(torch.maximum(torch.minimum(raw_a, upper), lower))
        return raw_a

    def clamp_full_conductance_(self, conductance_g: torch.Tensor) -> torch.Tensor:
        """Project continuous circuit state in the explicit ``G=raw_a+1`` coordinate."""

        if conductance_g.shape != (self.size,) or not bool(
            torch.all(torch.isfinite(conductance_g))
        ):
            raise ValueError("Expected one finite flat full-conductance G vector.")
        lower_g = self.pulse_population.min_bound.to(conductance_g.device) + 1.0
        upper_g = self.pulse_population.max_bound.to(conductance_g.device) + 1.0
        if bool(torch.any(lower_g < 0.0)):
            raise RuntimeError("Full-conductance projection would admit negative G.")
        conductance_g.copy_(
            torch.maximum(torch.minimum(conductance_g, upper_g), lower_g)
        )
        return conductance_g

    def make_write_port(
        self,
        delegate: VerifyPort,
    ) -> tuple[WriteOnlyPulsePort, PermanentFaultWriteLedger]:
        """Return a learner-facing blind write port and separate analysis ledger."""

        if delegate.size != self.size:
            raise ValueError("Pulse port and permanent-fault overlay sizes differ.")
        ledger = PermanentFaultWriteLedger(
            binding_keys=self.binding_keys,
            binding_shapes=self.binding_shapes,
        )
        port = _PermanentFaultWritePort(
            delegate,
            fault_mask=self.fault_mask,
            ledger=ledger,
        )
        return port, ledger

    def transition_report(
        self,
        before: torch.Tensor,
        after: torch.Tensor,
        *,
        coordinate: str,
        movement_tolerance: float = 0.0,
    ) -> dict[str, Any]:
        """Report fault immobility and healthy movement in raw-a or full-G."""

        if coordinate not in {"raw_a", "full_conductance_g"}:
            raise ValueError("Expected coordinate to be 'raw_a' or 'full_conductance_g'.")
        if (
            before.shape != (self.size,)
            or after.shape != (self.size,)
            or not bool(torch.all(torch.isfinite(before)))
            or not bool(torch.all(torch.isfinite(after)))
            or not math.isfinite(float(movement_tolerance))
            or movement_tolerance < 0.0
        ):
            raise ValueError("Expected finite matched transition states and tolerance.")
        if coordinate == "full_conductance_g" and (
            bool(torch.any(before < 0.0)) or bool(torch.any(after < 0.0))
        ):
            raise ValueError("Circuit-facing full conductance G must be non-negative.")
        delta = (after - before).abs()
        mask = self.fault_mask.to(delta.device)
        fault_delta = delta[mask]
        healthy_delta = delta[~mask]
        tolerance = float(movement_tolerance)
        return {
            "coordinate": coordinate,
            "fault_cell_count": int(mask.sum().item()),
            "fault_moved_count": int((fault_delta > tolerance).sum().item()),
            "fault_maximum_absolute_movement": (
                float(fault_delta.max().item()) if fault_delta.numel() else 0.0
            ),
            "healthy_cell_count": int((~mask).sum().item()),
            "healthy_moved_count": int((healthy_delta > tolerance).sum().item()),
            "healthy_maximum_absolute_movement": (
                float(healthy_delta.max().item()) if healthy_delta.numel() else 0.0
            ),
        }


def sample_aihwkit_compatible_permanent_fault_overlay(
    population: IbmReramArrayPopulation,
    *,
    mask_seed: int,
    raw_a_stuck_seed: int | None = None,
    corrupt_probability: float = DEFAULT_CORRUPT_PROBABILITY,
    corrupt_raw_a_range: float = DEFAULT_CORRUPT_RAW_A_RANGE,
) -> PermanentFaultOverlay:
    """Overlay independently sampled permanent faults on a repaired OM array.

    ``mask_seed`` is used directly for the named mask RNG.  If no separate
    stuck-value seed is supplied, it is deterministically derived in the
    named ``RAW_A_STUCK_RNG_DOMAIN`` so mask and stuck draws never share state.
    """

    if not isinstance(population, IbmReramArrayPopulation):
        raise TypeError("Expected one IbmReramArrayPopulation.")
    if population.corruption_policy != "counterfactual_repaired" or bool(
        torch.any(population.corrupt)
    ):
        raise ValueError("Permanent-fault overlay requires a fully repaired source array.")
    mask_seed = _validate_seed(mask_seed, name="mask_seed")
    if raw_a_stuck_seed is None:
        raw_a_stuck_seed = derive_seed(mask_seed, RAW_A_STUCK_RNG_DOMAIN)
    raw_a_stuck_seed = _validate_seed(raw_a_stuck_seed, name="raw_a_stuck_seed")
    if (
        not math.isfinite(float(corrupt_probability))
        or not 0.0 <= float(corrupt_probability) <= 1.0
    ):
        raise ValueError("Expected corrupt_probability in [0,1].")
    if not math.isfinite(float(corrupt_raw_a_range)) or corrupt_raw_a_range <= 0.0:
        raise ValueError("Expected a positive finite corrupt_raw_a_range.")

    base = array_population_as_pulse_population(population)
    if bool(torch.any(base.min_bound + 1.0 < 0.0)):
        raise ValueError(
            "Repaired source support leaves non-negative circuit coordinate G=raw_a+1."
        )
    size = base.size
    mask_generator = torch.Generator(device="cpu")
    mask_generator.manual_seed(mask_seed)
    fault_mask = torch.rand(
        size,
        dtype=torch.float32,
        generator=mask_generator,
        device="cpu",
    ) < float(corrupt_probability)

    corrupt_range = float(corrupt_raw_a_range)
    intersection_lower = torch.maximum(
        base.min_bound.cpu(),
        torch.full((size,), -corrupt_range, dtype=torch.float32),
    )
    intersection_upper = torch.minimum(
        base.max_bound.cpu(),
        torch.full((size,), corrupt_range, dtype=torch.float32),
    )
    invalid = fault_mask & (intersection_upper < intersection_lower)
    invalid_count = int(invalid.sum().item())
    if invalid_count:
        raise ValueError(
            "AIHWKit corrupt raw-a interval has an empty intersection with "
            f"{invalid_count} selected device supports."
        )

    stuck_generator = torch.Generator(device="cpu")
    stuck_generator.manual_seed(raw_a_stuck_seed)
    uniform = torch.rand(
        size,
        dtype=torch.float32,
        generator=stuck_generator,
        device="cpu",
    )
    sampled_stuck = intersection_lower + uniform * (
        intersection_upper - intersection_lower
    )
    # Healthy entries are finite placeholders and are never interpreted as
    # stuck values; this makes device transfer and hashing unambiguous.
    raw_a_stuck = torch.where(fault_mask, sampled_stuck, torch.zeros_like(sampled_stuck))
    selected_raw_a = raw_a_stuck[fault_mask].contiguous()
    selected_full_g = (selected_raw_a + 1.0).contiguous()
    if bool(torch.any(selected_full_g < 0.0)):
        raise RuntimeError("Sampled permanent fault produced negative conductance G.")

    min_bound = base.min_bound.cpu().clone()
    max_bound = base.max_bound.cpu().clone()
    dwmin_up = base.dwmin_up.cpu().clone()
    dwmin_down = base.dwmin_down.cpu().clone()
    min_bound[fault_mask] = selected_raw_a
    max_bound[fault_mask] = selected_raw_a
    dwmin_up[fault_mask] = 0.0
    dwmin_down[fault_mask] = 0.0
    slices = _binding_slices(population.binding_shapes)
    fault_count_by_binding = tuple(
        int(fault_mask[binding_slice].sum().item()) for binding_slice in slices
    )
    device_truncated = fault_mask & (
        (intersection_lower > -corrupt_range)
        | (intersection_upper < corrupt_range)
    )
    count = int(fault_mask.sum().item())
    receipt = PermanentFaultReceipt(
        mask_seed=mask_seed,
        raw_a_stuck_seed=raw_a_stuck_seed,
        corrupt_probability=float(corrupt_probability),
        corrupt_raw_a_range=corrupt_range,
        source_dw_min_std=float(base.dw_min_std),
        source_write_noise_std=float(base.write_noise_std),
        source_population_fingerprint=population.fingerprint,
        binding_keys=tuple(population.binding_keys),
        binding_shapes=tuple(tuple(shape) for shape in population.binding_shapes),
        realized_fault_count_by_binding=fault_count_by_binding,
        realized_fault_count=count,
        population_size=size,
        mask_sha256=_tensor_sha256(fault_mask),
        selected_raw_a_stuck_sha256=_tensor_sha256(selected_raw_a),
        selected_full_conductance_stuck_sha256=_tensor_sha256(selected_full_g),
        empty_support_intersection_count=invalid_count,
        device_truncated_intersection_count=int(device_truncated.sum().item()),
        raw_a_stuck_minimum=(float(selected_raw_a.min().item()) if count else None),
        raw_a_stuck_maximum=(float(selected_raw_a.max().item()) if count else None),
        full_conductance_stuck_minimum=(
            float(selected_full_g.min().item()) if count else None
        ),
        full_conductance_stuck_maximum=(
            float(selected_full_g.max().item()) if count else None
        ),
    )
    pulse_population = IbmReramPopulation(
        preset=base.preset,
        aihwkit_version=base.aihwkit_version,
        nominal_dw_min=base.nominal_dw_min,
        # The first recovery rung isolates permanent corruption.  It retains
        # every sampled bound and state-dependent step, while removing cycle
        # and apparent write noise.  Bernoulli pulse selection remains in the
        # optimizer/controller and is the only stochastic write mechanism.
        dw_min_std=0.0,
        write_noise_std=0.0,
        mult_noise=base.mult_noise,
        construction_seeds=base.construction_seeds.cpu().clone(),
        max_bound=max_bound,
        min_bound=min_bound,
        dwmin_up=dwmin_up,
        dwmin_down=dwmin_down,
        reference=base.reference.cpu().clone(),
        corrupt=fault_mask.clone(),
        preset_parameters={
            **dict(base.preset_parameters),
            "permanent_fault_overlay": receipt.as_dict(),
        },
    )
    return PermanentFaultOverlay(
        pulse_population=pulse_population,
        fault_mask=fault_mask,
        raw_a_stuck=raw_a_stuck,
        binding_keys=tuple(population.binding_keys),
        binding_shapes=tuple(tuple(shape) for shape in population.binding_shapes),
        receipt=receipt,
    )


__all__ = [
    "DEFAULT_CORRUPT_PROBABILITY",
    "DEFAULT_CORRUPT_RAW_A_RANGE",
    "FAULT_OVERLAY_SCHEMA",
    "FAULT_OVERLAY_SCHEMA_VERSION",
    "FULL_CONDUCTANCE_COORDINATE",
    "MASK_RNG_DOMAIN",
    "PermanentFaultOverlay",
    "PermanentFaultReceipt",
    "PermanentFaultWriteLedger",
    "RAW_A_COORDINATE",
    "RAW_A_STUCK_RNG_DOMAIN",
    "WriteOnlyPulsePort",
    "sample_aihwkit_compatible_permanent_fault_overlay",
]
