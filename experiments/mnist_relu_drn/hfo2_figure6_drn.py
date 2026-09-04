"""Direct DRN use of the Figure-6-informed HfO2 endpoint model.

The endpoint model defines a positive physical interval for every cell.  A
normalized programming coordinate ``p in [0, 1]`` is embedded as

``G = G_RESET + p * (G_SET - G_RESET)``.

This module keeps three scientifically distinct operations separate:

* direct initialization maps a signed logical DRN master into one fixed
  endpoint population;
* endpoint-noise HWA maps the same master through sampled endpoint
  populations and lifts the resulting physical gradients exactly; and
* hardware-in-loop recovery mutates an authoritative persistent cell state
  only through a re-parameterized AIHWKit HfO2 soft-bounds pulse law.

The endpoint fit itself contains no pulse dynamics.  Recovery therefore uses
the pulse parameters of AIHWKit 1.1.0's ``ReRamArrayHfO2PresetDevice`` while
replacing only its endpoint coordinate with the synthetic Figure-6 interval.
Digital BPTT and Adam select pulses, so this is not autonomous on-chip Adam.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import math
from typing import Any, Mapping, Sequence

import torch

from experiments.mnist_relu_drn.hfo2_figure6_endpoint_regimes import (
    HfO2Figure6EndpointPopulation,
)
from experiments.mnist_relu_drn.ibm_om_baseline_selection import (
    quad_stack,
    scatter_quads,
)
from training.ibm_reram_program_verify import (
    ControllerSettings,
    ProgramVerifyResult,
    run_program_verify,
)


HFO2_NOMINAL_DW_MIN_RAW_A = 0.4622
HFO2_NOMINAL_DW_MIN_PROGRESS = HFO2_NOMINAL_DW_MIN_RAW_A / 2.0
HFO2_DW_MIN_DTOD_LOG_STD = 0.7125
HFO2_UP_DOWN_DTOD = 0.01
HFO2_CYCLE_NOISE_STD = 0.2174
HFO2_WRITE_NOISE_STD = 0.5841
HFO2_PROGRAM_VERIFY_TOLERANCE_PROGRESS = HFO2_NOMINAL_DW_MIN_PROGRESS / 2.0
HFO2_PROGRAM_VERIFY_MAXIMUM_PULSES = 128

ENDPOINT_LAYOUTS = ("halves", "paired")
PULSE_MODEL_LABEL = (
    "AIHWKit_1.1.0_ReRamArrayHfO2_soft_bounds_in_normalized_progress"
)
RECOVERY_CLAIM_LABEL = "hardware-in-loop pulse-mediated recovery"


def _tensor_sha256(value: torch.Tensor) -> str:
    tensor = value.detach().contiguous().cpu()
    digest = sha256()
    digest.update(str(tensor.dtype).encode("utf-8"))
    digest.update(str(tuple(tensor.shape)).encode("utf-8"))
    digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _validate_shapes(
    shapes: Sequence[Sequence[int]], layouts: Sequence[str]
) -> tuple[tuple[tuple[int, int], ...], tuple[str, ...]]:
    selected_shapes = tuple(tuple(int(item) for item in shape) for shape in shapes)
    selected_layouts = tuple(layouts)
    if len(selected_shapes) != 2 or selected_layouts != ENDPOINT_LAYOUTS:
        raise ValueError("Expected two physical DRN shapes with halves/paired layouts.")
    if any(len(shape) != 2 or shape[0] % 2 or shape[1] % 2 for shape in selected_shapes):
        raise ValueError("Expected even-by-even rank-two physical DRN shapes.")
    return selected_shapes, selected_layouts


def _split_flat(
    value: torch.Tensor, shapes: Sequence[Sequence[int]]
) -> tuple[torch.Tensor, ...]:
    flat = torch.as_tensor(value).reshape(-1)
    result = []
    offset = 0
    for shape_value in shapes:
        shape = tuple(int(item) for item in shape_value)
        count = math.prod(shape)
        result.append(flat[offset : offset + count].reshape(shape))
        offset += count
    if offset != flat.numel():
        raise ValueError("Endpoint vector length does not match the physical DRN shapes.")
    return tuple(result)


def flatten_physical(values: Sequence[torch.Tensor]) -> torch.Tensor:
    selected = tuple(values)
    if not selected:
        raise ValueError("Expected at least one physical tensor.")
    return torch.cat(tuple(value.reshape(-1) for value in selected))


@dataclass(frozen=True)
class EndpointField:
    """One fixed Figure-6 endpoint assignment in DRN binding order."""

    reset: tuple[torch.Tensor, torch.Tensor]
    set: tuple[torch.Tensor, torch.Tensor]
    shapes: tuple[tuple[int, int], tuple[int, int]]
    layouts: tuple[str, str]
    population_report: Mapping[str, Any]

    def __post_init__(self) -> None:
        shapes, layouts = _validate_shapes(self.shapes, self.layouts)
        if self.shapes != shapes or self.layouts != layouts:
            raise ValueError("Endpoint field shapes/layouts must use canonical tuples.")
        for name in ("reset", "set"):
            values = getattr(self, name)
            if len(values) != 2:
                raise ValueError(f"Expected two endpoint tensors in {name}.")
            for value, shape in zip(values, shapes, strict=True):
                if (
                    not isinstance(value, torch.Tensor)
                    or tuple(value.shape) != shape
                    or not value.is_floating_point()
                    or not bool(torch.isfinite(value).all())
                ):
                    raise ValueError(f"Expected finite floating {name} endpoint tensors.")
        if any(
            bool(torch.any(upper <= lower))
            for lower, upper in zip(self.reset, self.set, strict=True)
        ):
            raise ValueError("Every SET endpoint must exceed its RESET endpoint.")
        if any(bool(torch.any(value < 0.0)) for value in self.reset):
            raise ValueError("Every RESET endpoint must be non-negative.")

    @property
    def devices(self) -> int:
        return sum(math.prod(shape) for shape in self.shapes)

    @property
    def window(self) -> tuple[torch.Tensor, torch.Tensor]:
        return tuple(
            upper - lower
            for lower, upper in zip(self.reset, self.set, strict=True)
        )

    def to(
        self, device: torch.device | str, *, dtype: torch.dtype = torch.float32
    ) -> "EndpointField":
        target = torch.device(device)
        return EndpointField(
            reset=tuple(value.to(device=target, dtype=dtype) for value in self.reset),
            set=tuple(value.to(device=target, dtype=dtype) for value in self.set),
            shapes=self.shapes,
            layouts=self.layouts,
            population_report=self.population_report,
        )

    def report(self) -> Mapping[str, Any]:
        return {
            "devices": self.devices,
            "layouts": list(self.layouts),
            "shapes": [list(shape) for shape in self.shapes],
            "reset_sha256": [_tensor_sha256(value) for value in self.reset],
            "set_sha256": [_tensor_sha256(value) for value in self.set],
            "population": dict(self.population_report),
        }


def endpoint_field_from_population(
    population: HfO2Figure6EndpointPopulation,
    *,
    shapes: Sequence[Sequence[int]],
    layouts: Sequence[str] = ENDPOINT_LAYOUTS,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> EndpointField:
    """Place a flat endpoint population into canonical DRN binding order."""

    selected_shapes, selected_layouts = _validate_shapes(shapes, layouts)
    expected = sum(math.prod(shape) for shape in selected_shapes)
    if population.devices != expected:
        raise ValueError(
            "Endpoint population size does not cover the physical DRN: "
            f"population={population.devices}, expected={expected}."
        )
    target = torch.device(device)
    reset = tuple(
        value.to(device=target, dtype=dtype)
        for value in _split_flat(population.reset_state, selected_shapes)
    )
    set_state = tuple(
        value.to(device=target, dtype=dtype)
        for value in _split_flat(population.set_state, selected_shapes)
    )
    return EndpointField(
        reset=reset,
        set=set_state,
        shapes=selected_shapes,
        layouts=selected_layouts,
        population_report=population.report(),
    )


def rotate_endpoint_field(field: EndpointField, offset: int) -> EndpointField:
    """Rotate cell identities while preserving every endpoint pair and marginal.

    This is an assignment augmentation for array-agnostic HWA, not a new
    endpoint draw.  RESET and SET move together, so the selected copula and
    every finite marginal value remain unchanged exactly.
    """

    if isinstance(offset, bool) or not isinstance(offset, int):
        raise ValueError("Expected an integer endpoint-assignment rotation.")
    normalized = offset % field.devices
    reset = _split_flat(
        torch.roll(flatten_physical(field.reset), shifts=normalized), field.shapes
    )
    set_state = _split_flat(
        torch.roll(flatten_physical(field.set), shifts=normalized), field.shapes
    )
    return EndpointField(
        reset=reset,  # type: ignore[arg-type]
        set=set_state,  # type: ignore[arg-type]
        shapes=field.shapes,
        layouts=field.layouts,
        population_report={
            "assignment_augmentation": "joint_circular_cell_identity_rotation",
            "rotation_offset": normalized,
            "base_population": dict(field.population_report),
        },
    )


def _validate_masters(
    masters: Sequence[torch.Tensor], field: EndpointField
) -> tuple[torch.Tensor, torch.Tensor]:
    selected = tuple(masters)
    if len(selected) != 2:
        raise ValueError("Expected exactly two signed logical DRN masters.")
    for master, shape in zip(selected, field.shapes, strict=True):
        expected = (shape[0] // 2, shape[1] // 2)
        if (
            not isinstance(master, torch.Tensor)
            or tuple(master.shape) != expected
            or not master.is_floating_point()
            or not bool(torch.isfinite(master).all())
            or bool(torch.any(master < -1.0))
            or bool(torch.any(master > 1.0))
        ):
            raise ValueError(
                "Expected finite signed masters in [-1,1] matching the endpoint field."
            )
    return selected  # type: ignore[return-value]


def master_to_progress(
    masters: Sequence[torch.Tensor], field: EndpointField
) -> tuple[torch.Tensor, torch.Tensor]:
    """Lift signed masters into four non-negative per-cell levels in [0,1]."""

    selected = _validate_masters(masters, field)
    result = []
    for master, shape, layout in zip(
        selected, field.shapes, field.layouts, strict=True
    ):
        positive = torch.relu(master)
        negative = torch.relu(-master)
        quads = torch.stack((positive, negative, negative, positive), dim=-1)
        result.append(scatter_quads(quads, shape=shape, layout=layout))
    return tuple(result)  # type: ignore[return-value]


def progress_to_conductance(
    progress: Sequence[torch.Tensor], field: EndpointField
) -> tuple[torch.Tensor, torch.Tensor]:
    """Embed normalized persistent levels between each cell's endpoints."""

    selected = tuple(progress)
    if len(selected) != 2:
        raise ValueError("Expected two normalized progress tensors.")
    result = []
    for value, lower, upper, shape in zip(
        selected, field.reset, field.set, field.shapes, strict=True
    ):
        if (
            tuple(value.shape) != shape
            or not bool(torch.isfinite(value).all())
            or bool(torch.any(value < 0.0))
            or bool(torch.any(value > 1.0))
        ):
            raise ValueError("Expected finite normalized progress in [0,1].")
        result.append(lower + value.to(lower) * (upper - lower))
    return tuple(result)  # type: ignore[return-value]


def map_masters_to_conductance(
    masters: Sequence[torch.Tensor], field: EndpointField
) -> tuple[torch.Tensor, torch.Tensor]:
    """Directly initialize full physical DRN conductances from signed masters."""

    return progress_to_conductance(master_to_progress(masters, field), field)


def lift_endpoint_gradients(
    masters: Sequence[torch.Tensor],
    physical_gradients: Sequence[torch.Tensor],
    field: EndpointField,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply the exact chain rule through the endpoint-affine four-cell map."""

    selected = _validate_masters(masters, field)
    gradients = tuple(physical_gradients)
    if len(gradients) != 2:
        raise ValueError("Expected one physical gradient tensor per DRN layer.")
    result = []
    for master, physical, window, shape, layout in zip(
        selected,
        gradients,
        field.window,
        field.shapes,
        field.layouts,
        strict=True,
    ):
        if tuple(physical.shape) != shape or not bool(torch.isfinite(physical).all()):
            raise ValueError("Physical endpoint gradient shape or values are invalid.")
        contribution = quad_stack(
            physical.to(device=master.device, dtype=master.dtype)
            * window.to(device=master.device, dtype=master.dtype),
            layout=layout,
        )
        positive = contribution[..., 0] + contribution[..., 3]
        negative = -(contribution[..., 1] + contribution[..., 2])
        gradient = torch.where(master >= 0.0, positive, negative)
        if not bool(torch.isfinite(gradient).all()):
            raise FloatingPointError("Endpoint-aware logical gradient is non-finite.")
        result.append(gradient)
    return tuple(result)  # type: ignore[return-value]


@dataclass(frozen=True)
class PulseStepReport:
    step: int
    requested_cells: int
    pulsed_cells: int
    capped_cells: int
    upward_pulses: int
    downward_pulses: int
    mean_probability: float
    maximum_probability: float
    effective_state_changes: int


class HfO2Figure6PulsePlant:
    """Persistent Figure-6 cells updated by the stock HfO2 soft-bounds law.

    ``raw_a`` is authoritative and lies in ``[-1,1]``.  Its normalized
    progress is ``p=(a+1)/2``; the circuit always receives the full positive
    conductance obtained from :func:`progress_to_conductance`.
    """

    def __init__(
        self,
        field: EndpointField,
        initial_progress: Sequence[torch.Tensor],
        *,
        pulse_parameter_seed: int,
        pulse_noise_seed: int,
    ) -> None:
        selected = tuple(initial_progress)
        progress_to_conductance(selected, field)
        self.field = field
        self.device = field.reset[0].device
        self.dtype = field.reset[0].dtype
        self.raw_a = 2.0 * flatten_physical(selected).to(
            device=self.device, dtype=self.dtype
        ) - 1.0
        self.apparent_raw_a = self.raw_a.clone()
        parameter_generator = torch.Generator(device="cpu")
        parameter_generator.manual_seed(int(pulse_parameter_seed))
        size = self.raw_a.numel()
        gain_z = torch.randn(size, generator=parameter_generator, dtype=torch.float32)
        asymmetry_z = torch.randn(
            size, generator=parameter_generator, dtype=torch.float32
        )
        gain = torch.exp(HFO2_DW_MIN_DTOD_LOG_STD * gain_z)
        asymmetry = HFO2_UP_DOWN_DTOD * asymmetry_z
        self.dwmin_up_raw_a = (
            HFO2_NOMINAL_DW_MIN_RAW_A * (gain + asymmetry)
        ).abs().to(device=self.device, dtype=self.dtype)
        self.dwmin_down_raw_a = (
            HFO2_NOMINAL_DW_MIN_RAW_A * (gain - asymmetry)
        ).abs().to(device=self.device, dtype=self.dtype)
        self.noise_generator = torch.Generator(device=self.device)
        self.noise_generator.manual_seed(int(pulse_noise_seed))
        self.pulse_parameter_seed = int(pulse_parameter_seed)
        self.pulse_noise_seed = int(pulse_noise_seed)
        self.pulse_count = torch.zeros(
            size, device=self.device, dtype=torch.int64
        )

    @property
    def size(self) -> int:
        return int(self.raw_a.numel())

    @property
    def progress(self) -> tuple[torch.Tensor, torch.Tensor]:
        flat = (self.raw_a + 1.0) / 2.0
        return _split_flat(flat, self.field.shapes)  # type: ignore[return-value]

    @property
    def apparent_progress_unprojected(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the literal noisy apparent state in normalized progress."""

        flat = (self.apparent_raw_a + 1.0) / 2.0
        return _split_flat(flat, self.field.shapes)  # type: ignore[return-value]

    @property
    def apparent_progress_projected(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Project an apparent observation to the modeled physical interval.

        AIHWKit apparent write noise can place a read outside the persistent
        bounds. A negative branch conductance is invalid for this DRN, so an
        apparent-state network diagnostic explicitly projects normalized
        progress to ``[0, 1]`` before applying the positive endpoint map.
        The literal unprojected observation remains available above.
        """

        return tuple(
            value.clamp(0.0, 1.0)
            for value in self.apparent_progress_unprojected
        )  # type: ignore[return-value]

    @property
    def full_conductance(self) -> tuple[torch.Tensor, torch.Tensor]:
        return progress_to_conductance(self.progress, self.field)

    @property
    def apparent_full_conductance(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the passivity-projected apparent-state conductance."""

        return progress_to_conductance(
            self.apparent_progress_projected,
            self.field,
        )

    def sample_apparent(self) -> None:
        """Draw one noisy apparent observation for every persistent cell."""

        write = torch.randn(
            self.raw_a.shape,
            generator=self.noise_generator,
            device=self.device,
            dtype=self.dtype,
        )
        write_scale_raw_a = HFO2_WRITE_NOISE_STD * HFO2_NOMINAL_DW_MIN_RAW_A
        self.apparent_raw_a.copy_(self.raw_a + write_scale_raw_a * write)

    def clone_for_new_pulse_phase(
        self,
        *,
        pulse_noise_seed: int,
        retain_apparent_observation: bool = True,
    ) -> "HfO2Figure6PulsePlant":
        """Clone the exact persistent state with fresh phase-local accounting.

        Device-to-device pulse parameters are regenerated from their original
        seed and checked for bitwise equality. The new stochastic pulse phase
        receives its own declared noise stream and starts with zero pulse
        counts, while the persistent P&V endpoint is copied exactly.
        """

        clone = HfO2Figure6PulsePlant(
            self.field,
            tuple(torch.zeros_like(value) for value in self.progress),
            pulse_parameter_seed=self.pulse_parameter_seed,
            pulse_noise_seed=int(pulse_noise_seed),
        )
        if not torch.equal(clone.dwmin_up_raw_a, self.dwmin_up_raw_a) or not torch.equal(
            clone.dwmin_down_raw_a, self.dwmin_down_raw_a
        ):
            raise RuntimeError("HfO2 pulse parameters did not reproduce exactly.")
        clone.raw_a.copy_(self.raw_a)
        if retain_apparent_observation:
            clone.apparent_raw_a.copy_(self.apparent_raw_a)
        else:
            clone.apparent_raw_a.copy_(self.raw_a)
        return clone

    def pulse(self, directions: torch.Tensor, *, pulse_cap: int) -> tuple[int, int]:
        """Apply at most one SET/RESET pulse to each requested eligible cell."""

        direction = torch.as_tensor(
            directions, device=self.device, dtype=torch.int8
        )
        if (
            direction.shape != self.raw_a.shape
            or bool(torch.any((direction < -1) | (direction > 1)))
            or isinstance(pulse_cap, bool)
            or not isinstance(pulse_cap, int)
            or pulse_cap < 1
        ):
            raise ValueError("Invalid HfO2 pulse direction or pulse cap.")
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
        up = selected & (direction > 0)
        down = selected & (direction < 0)
        candidate = self.raw_a.clone()
        if bool(torch.any(up)):
            response = self.dwmin_up_raw_a * (
                1.0 - self.raw_a + HFO2_CYCLE_NOISE_STD * cycle
            )
            candidate[up] = self.raw_a[up] + response[up]
        if bool(torch.any(down)):
            response = self.dwmin_down_raw_a * (
                1.0 + self.raw_a + HFO2_CYCLE_NOISE_STD * cycle
            )
            candidate[down] = self.raw_a[down] - response[down]
        candidate.clamp_(-1.0, 1.0)
        self.raw_a[selected] = candidate[selected]
        write = torch.randn(
            self.raw_a.shape,
            generator=self.noise_generator,
            device=self.device,
            dtype=self.dtype,
        )
        write_scale_raw_a = (
            HFO2_WRITE_NOISE_STD * HFO2_NOMINAL_DW_MIN_RAW_A
        )
        self.apparent_raw_a[selected] = (
            self.raw_a + write_scale_raw_a * write
        )[selected]
        self.pulse_count[selected] += 1
        changed = int((self.raw_a[selected] != before[selected]).sum().item())
        return changed, int(capped.sum().item())

    def state_dict(self) -> dict[str, Any]:
        return {
            "schema": "ebl.hfo2_figure6_pulse_plant",
            "schema_version": 1,
            "endpoint_model": self.field.report(),
            "pulse_model": PULSE_MODEL_LABEL,
            "persistent_state_coordinate": "raw_a=2*progress-1",
            "circuit_conductance_formula": "G=RESET+progress*(SET-RESET)",
            "persistent_raw_a": self.raw_a.detach().cpu().clone(),
            "apparent_raw_a": self.apparent_raw_a.detach().cpu().clone(),
            "dwmin_up_raw_a": self.dwmin_up_raw_a.detach().cpu().clone(),
            "dwmin_down_raw_a": self.dwmin_down_raw_a.detach().cpu().clone(),
            "pulse_count": self.pulse_count.detach().cpu().clone(),
            "pulse_parameter_seed": self.pulse_parameter_seed,
            "pulse_noise_seed": self.pulse_noise_seed,
            "pulse_noise_rng_state": self.noise_generator.get_state().cpu().clone(),
            "inference_uses_persistent_not_apparent_state": True,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if (
            state.get("schema") != "ebl.hfo2_figure6_pulse_plant"
            or state.get("schema_version") != 1
            or state.get("pulse_model") != PULSE_MODEL_LABEL
            or state.get("pulse_parameter_seed") != self.pulse_parameter_seed
            or state.get("pulse_noise_seed") != self.pulse_noise_seed
        ):
            raise ValueError("HfO2 Figure-6 pulse-plant continuation mismatch.")
        expected = (self.size,)
        for name, target in (
            ("persistent_raw_a", self.raw_a),
            ("apparent_raw_a", self.apparent_raw_a),
            ("dwmin_up_raw_a", self.dwmin_up_raw_a),
            ("dwmin_down_raw_a", self.dwmin_down_raw_a),
            ("pulse_count", self.pulse_count),
        ):
            value = state.get(name)
            if not isinstance(value, torch.Tensor) or value.shape != expected:
                raise ValueError(f"Invalid saved HfO2 pulse-plant field {name!r}.")
            target.copy_(value.to(device=self.device, dtype=target.dtype))
        rng = state.get("pulse_noise_rng_state")
        if not isinstance(rng, torch.Tensor):
            raise ValueError("Expected a saved HfO2 pulse-noise RNG state.")
        self.noise_generator.set_state(rng.detach().cpu())

    def report(self) -> Mapping[str, Any]:
        progress = (self.raw_a + 1.0) / 2.0
        pulses = self.pulse_count
        return {
            "claim_label": RECOVERY_CLAIM_LABEL,
            "pulse_model": PULSE_MODEL_LABEL,
            "cells": self.size,
            "persistent_progress_minimum": float(progress.min().item()),
            "persistent_progress_mean": float(progress.mean().item()),
            "persistent_progress_maximum": float(progress.max().item()),
            "cells_at_reset": int((self.raw_a == -1.0).sum().item()),
            "cells_at_set": int((self.raw_a == 1.0).sum().item()),
            "total_pulses": int(pulses.sum().item()),
            "pulsed_cells": int((pulses > 0).sum().item()),
            "maximum_pulses_per_cell": int(pulses.max().item()),
            "persistent_state_is_sole_weight_authority": True,
            "inference_uses_full_positive_conductance": True,
            "inference_uses_apparent_write_noise": False,
            "apparent_state_projection": "clamp_progress_to_[0,1]_before_positive_G",
        }


class _HfO2Figure6ProgramVerifyPort:
    """Capability-limited apparent-feedback port for the synthetic plant."""

    def __init__(
        self,
        plant: HfO2Figure6PulsePlant,
        *,
        maximum_pulses: int,
    ) -> None:
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
        direction = torch.as_tensor(
            directions,
            device=self.__plant.device,
            dtype=torch.int8,
        )
        count = torch.as_tensor(
            counts,
            device=self.__plant.device,
            dtype=torch.int64,
        )
        if direction.shape != (self.size,) or count.shape != (self.size,):
            raise ValueError("P&V directions and counts must match the plant size.")
        if bool(torch.any(count < 0)):
            raise ValueError("P&V pulse counts must be non-negative.")
        maximum = int(count.max().item()) if count.numel() else 0
        for pulse_index in range(maximum):
            active_direction = torch.where(
                count > pulse_index,
                direction,
                torch.zeros_like(direction),
            )
            _, capped = self.__plant.pulse(
                active_direction,
                pulse_cap=self.__maximum_pulses,
            )
            if capped:
                raise RuntimeError(
                    "The controller exceeded the declared P&V pulse budget."
                )


def program_progress_with_verify(
    field: EndpointField,
    target_progress: Sequence[torch.Tensor],
    *,
    pulse_parameter_seed: int,
    pulse_noise_seed: int,
    tolerance_progress: float = HFO2_PROGRAM_VERIFY_TOLERANCE_PROGRESS,
    maximum_pulses: int = HFO2_PROGRAM_VERIFY_MAXIMUM_PULSES,
    noisy_initial_reset_verify: bool = True,
    preaccept_exact_reset_targets: bool = True,
) -> tuple[HfO2Figure6PulsePlant, ProgramVerifyResult]:
    """Program continuous targets from full RESET with apparent P&V feedback.

    Every persistent cell starts at ``p=0``, which maps to that identity's
    sampled ``full_tile_RESET`` endpoint. The one-pulse controller sees only
    apparent normalized progress, verifies after every stochastic pulse, and
    stops on tolerance or budget. Exact ``p=0`` targets are already realized
    by the declared full-RESET initialization and are therefore preaccepted
    without target-programming pulses by default. The returned plant retains
    both the persistent endpoint used by the circuit and the last apparent
    observation.
    """

    selected = tuple(target_progress)
    # Validate target shape and interval through the physical endpoint map.
    progress_to_conductance(selected, field)
    if (
        not math.isfinite(float(tolerance_progress))
        or float(tolerance_progress) <= 0.0
    ):
        raise ValueError("Expected a positive finite P&V progress tolerance.")
    if (
        isinstance(maximum_pulses, bool)
        or not isinstance(maximum_pulses, int)
        or maximum_pulses < 1
    ):
        raise ValueError("Expected a positive integer P&V pulse budget.")
    plant = HfO2Figure6PulsePlant(
        field,
        tuple(torch.zeros_like(value) for value in selected),
        pulse_parameter_seed=int(pulse_parameter_seed),
        pulse_noise_seed=int(pulse_noise_seed),
    )
    if noisy_initial_reset_verify:
        plant.sample_apparent()
    flat_target = flatten_physical(selected)
    reset_targets = flat_target == 0.0
    eligible = (
        ~reset_targets
        if preaccept_exact_reset_targets
        else torch.ones_like(reset_targets)
    )
    raw_result = run_program_verify(
        _HfO2Figure6ProgramVerifyPort(
            plant,
            maximum_pulses=maximum_pulses,
        ),
        targets=flat_target,
        tolerance=float(tolerance_progress),
        maximum_pulses=maximum_pulses,
        settings=ControllerSettings(kind="one_pulse"),
        eligible=eligible,
    )
    if preaccept_exact_reset_targets:
        result = ProgramVerifyResult(
            accepted=raw_result.accepted | reset_targets,
            nonfinite=raw_result.nonfinite,
            budget_exhausted=raw_result.budget_exhausted,
            apparent_endpoint=raw_result.apparent_endpoint,
            set_count=raw_result.set_count,
            reset_count=raw_result.reset_count,
            total_pulses=raw_result.total_pulses,
            verify_count=raw_result.verify_count + reset_targets.to(torch.int64),
            reversals=raw_result.reversals,
        )
    else:
        result = raw_result
    if not torch.equal(plant.pulse_count, result.total_pulses):
        raise RuntimeError("Plant and P&V controller pulse accounting diverged.")
    observed_apparent = flatten_physical(plant.apparent_progress_unprojected)
    if not torch.equal(observed_apparent, result.apparent_endpoint):
        raise RuntimeError("P&V result does not match the plant apparent endpoint.")
    return plant, result


def program_masters_with_verify(
    masters: Sequence[torch.Tensor],
    field: EndpointField,
    **settings: Any,
) -> tuple[HfO2Figure6PulsePlant, ProgramVerifyResult]:
    """Map signed masters to cell targets and program them from full RESET."""

    return program_progress_with_verify(
        field,
        master_to_progress(masters, field),
        **settings,
    )


class PersistentHfO2PulseAdam:
    """Digital Adam commands realized solely as persistent HfO2 cell pulses."""

    def __init__(
        self,
        plant: HfO2Figure6PulsePlant,
        *,
        learning_rate_progress: float,
        pulse_cap: int,
        pulse_selection_seed: int,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
    ) -> None:
        scalars = (learning_rate_progress, beta1, beta2, epsilon)
        if any(not math.isfinite(float(value)) for value in scalars):
            raise ValueError("Expected finite pulse-Adam settings.")
        if learning_rate_progress <= 0.0 or epsilon <= 0.0:
            raise ValueError("Expected positive pulse-Adam learning rate and epsilon.")
        if not (0.0 <= beta1 < 1.0 and 0.0 <= beta2 < 1.0):
            raise ValueError("Expected Adam beta values in [0,1).")
        if isinstance(pulse_cap, bool) or not isinstance(pulse_cap, int) or pulse_cap < 1:
            raise ValueError("Expected a positive recovery pulse cap.")
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

    def step(
        self, physical_conductance_gradients: Sequence[torch.Tensor]
    ) -> PulseStepReport:
        gradients_g = tuple(physical_conductance_gradients)
        if len(gradients_g) != 2:
            raise ValueError("Expected two physical conductance gradients.")
        for gradient, shape in zip(gradients_g, self.plant.field.shapes, strict=True):
            if tuple(gradient.shape) != shape or not bool(torch.isfinite(gradient).all()):
                raise ValueError("Expected finite physical gradients of fixed shape.")
        # G = RESET + p*(SET-RESET), hence dL/dp = window*dL/dG.
        gradient_progress = flatten_physical(
            tuple(
                gradient.detach() * window
                for gradient, window in zip(
                    gradients_g, self.plant.field.window, strict=True
                )
            )
        )
        self.step_index += 1
        self.first_moment.mul_(self.beta1).add_(
            gradient_progress, alpha=1.0 - self.beta1
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
        probability = (
            command.abs() / HFO2_NOMINAL_DW_MIN_PROGRESS
        ).clamp(max=1.0)
        random_value = torch.rand(
            probability.shape,
            generator=self.selection_generator,
            device=self.plant.device,
            dtype=probability.dtype,
        )
        requested = (random_value < probability) & (command != 0.0)
        eligible = self.plant.pulse_count < self.pulse_cap
        selected = requested & eligible
        directions = torch.zeros_like(self.plant.raw_a, dtype=torch.int8)
        directions[selected & (command > 0.0)] = 1
        directions[selected & (command < 0.0)] = -1
        changed, capped_by_plant = self.plant.pulse(
            directions, pulse_cap=self.pulse_cap
        )
        capped = int((requested & ~eligible).sum().item()) + capped_by_plant
        return PulseStepReport(
            step=self.step_index,
            requested_cells=int(requested.sum().item()),
            pulsed_cells=int(selected.sum().item()),
            capped_cells=capped,
            upward_pulses=int((directions > 0).sum().item()),
            downward_pulses=int((directions < 0).sum().item()),
            mean_probability=float(probability.mean().item()),
            maximum_probability=float(probability.max().item()),
            effective_state_changes=changed,
        )

    def state_dict(self) -> dict[str, Any]:
        return {
            "schema": "ebl.hfo2_figure6_persistent_pulse_adam",
            "schema_version": 1,
            "claim_label": RECOVERY_CLAIM_LABEL,
            "learning_rate_progress": self.learning_rate_progress,
            "nominal_delta_progress": HFO2_NOMINAL_DW_MIN_PROGRESS,
            "pulse_cap": self.pulse_cap,
            "beta1": self.beta1,
            "beta2": self.beta2,
            "epsilon": self.epsilon,
            "step": self.step_index,
            "first_moment": self.first_moment.detach().cpu().clone(),
            "second_moment": self.second_moment.detach().cpu().clone(),
            "pulse_selection_seed": self.pulse_selection_seed,
            "pulse_selection_rng_state": self.selection_generator.get_state().cpu().clone(),
            "authoritative_weight_shadow": None,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if (
            state.get("schema") != "ebl.hfo2_figure6_persistent_pulse_adam"
            or state.get("schema_version") != 1
            or state.get("claim_label") != RECOVERY_CLAIM_LABEL
            or state.get("pulse_selection_seed") != self.pulse_selection_seed
            or float(state.get("learning_rate_progress", -1.0))
            != self.learning_rate_progress
            or int(state.get("pulse_cap", -1)) != self.pulse_cap
        ):
            raise ValueError("HfO2 persistent pulse-Adam continuation mismatch.")
        for name, target in (
            ("first_moment", self.first_moment),
            ("second_moment", self.second_moment),
        ):
            value = state.get(name)
            if not isinstance(value, torch.Tensor) or value.shape != target.shape:
                raise ValueError(f"Invalid saved pulse-Adam field {name!r}.")
            target.copy_(value.to(device=target.device, dtype=target.dtype))
        step = state.get("step")
        if isinstance(step, bool) or not isinstance(step, int) or step < 0:
            raise ValueError("Invalid saved pulse-Adam step.")
        self.step_index = step
        rng = state.get("pulse_selection_rng_state")
        if not isinstance(rng, torch.Tensor):
            raise ValueError("Expected saved pulse-selection RNG state.")
        self.selection_generator.set_state(rng.detach().cpu())


__all__ = [
    "ENDPOINT_LAYOUTS",
    "EndpointField",
    "HFO2_CYCLE_NOISE_STD",
    "HFO2_DW_MIN_DTOD_LOG_STD",
    "HFO2_NOMINAL_DW_MIN_PROGRESS",
    "HFO2_NOMINAL_DW_MIN_RAW_A",
    "HFO2_PROGRAM_VERIFY_MAXIMUM_PULSES",
    "HFO2_PROGRAM_VERIFY_TOLERANCE_PROGRESS",
    "HFO2_UP_DOWN_DTOD",
    "HFO2_WRITE_NOISE_STD",
    "HfO2Figure6PulsePlant",
    "PULSE_MODEL_LABEL",
    "PersistentHfO2PulseAdam",
    "PulseStepReport",
    "RECOVERY_CLAIM_LABEL",
    "endpoint_field_from_population",
    "flatten_physical",
    "lift_endpoint_gradients",
    "map_masters_to_conductance",
    "master_to_progress",
    "progress_to_conductance",
    "program_masters_with_verify",
    "program_progress_with_verify",
    "rotate_endpoint_field",
]
