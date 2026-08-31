"""Open-loop optimizer primitives for matched IBM-OM endpoint recovery.

The direct SGD/Adam arms use the existing column-serial independent-cell
Bernoulli coincidence surface.  The Tiki-Taka arms are deliberately narrower
than native AIHWKit tiles: they are qualified minibatch-equation emulators
whose fast accumulator is a separate physical IBM-OM pulse plant and whose
slow array is the exact persistent deployment P0.

For both TT variants, ``gamma=0`` and only slow C is visible in the DRN
forward path.  A is initialized by a hidden-model oracle at each cell's
bounded intrinsic symmetry point.  Its held apparent post-write sample is
read in raw-x units; this includes modeled write noise but is not a fresh MVM
read-noise sample.  Transfers scan canonical logical ``[out,in]`` input
columns: one physical source-rail row pair across every destination rail.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Literal, Mapping, Sequence

import torch

from experiments.mnist_relu_drn.ibm_om_winsorized_onchip_adam import (
    flatten_physical,
)
from training.ibm_reram_program_verify import (
    IbmReramPopulation,
    IbmReramRawActivePlant,
)


EVIDENCE_TIER = "exploratory_noncanonical"
TT_VARIANTS = ("tt_v1", "tt_v2")


@dataclass(frozen=True)
class SymmetryInitialization:
    raw_a: torch.Tensor
    report: Mapping[str, Any]


def intrinsic_symmetry_raw_a(
    population: IbmReramPopulation,
    *,
    numerical_clamp_tolerance: float = 2e-6,
) -> SymmetryInitialization:
    """Solve equality of raw-active up/down pulse response per cell.

    For ``du*(1-a/U) == dd*(1-a/L)``, the unique solution is
    ``a*=(du-dd)/(du/U-dd/L)``.  This function requires a genuinely
    bidirectional interval and rejects any material clamp instead of silently
    turning an out-of-support solution into an initialization.
    """

    if not isinstance(population, IbmReramPopulation):
        raise TypeError("Expected one IBM ReRAM pulse population.")
    if not math.isfinite(numerical_clamp_tolerance) or numerical_clamp_tolerance < 0.0:
        raise ValueError("Expected a finite nonnegative symmetry clamp tolerance.")
    lower = population.min_bound.detach().to(device="cpu", dtype=torch.float64)
    upper = population.max_bound.detach().to(device="cpu", dtype=torch.float64)
    up = population.dwmin_up.detach().to(device="cpu", dtype=torch.float64)
    down = population.dwmin_down.detach().to(device="cpu", dtype=torch.float64)
    if not (
        bool(torch.all(torch.isfinite(lower)))
        and bool(torch.all(torch.isfinite(upper)))
        and bool(torch.all(torch.isfinite(up)))
        and bool(torch.all(torch.isfinite(down)))
        and bool(torch.all(lower < 0.0))
        and bool(torch.all(upper > 0.0))
        and bool(torch.all(up > 0.0))
        and bool(torch.all(down > 0.0))
    ):
        raise ValueError(
            "Intrinsic-symmetry initialization requires finite L<0<U and du,dd>0."
        )
    denominator = up / upper - down / lower
    symmetry = (up - down) / denominator
    if not bool(torch.all(torch.isfinite(symmetry))):
        raise ValueError("Intrinsic-symmetry solution became non-finite.")
    below = lower - symmetry
    above = symmetry - upper
    material = (below > numerical_clamp_tolerance) | (
        above > numerical_clamp_tolerance
    )
    if bool(torch.any(material)):
        raise ValueError(
            "Intrinsic-symmetry solution materially left the physical cell support."
        )
    numerical_clamp = (symmetry < lower) | (symmetry > upper)
    symmetry = torch.maximum(torch.minimum(symmetry, upper), lower)
    response_up = up * (1.0 - symmetry / upper)
    response_down = down * (1.0 - symmetry / lower)
    residual = response_up - response_down
    return SymmetryInitialization(
        raw_a=symmetry.to(torch.float32),
        report={
            "policy": "bounded_hidden_model_oracle_intrinsic_symmetry",
            "formula": "(dwmin_up-dwmin_down)/(dwmin_up/max-dwmin_down/min)",
            "cells": int(symmetry.numel()),
            "numerically_clamped_cells": int(numerical_clamp.sum().item()),
            "materially_out_of_support_cells": 0,
            "raw_a_minimum": float(symmetry.min().item()),
            "raw_a_mean": float(symmetry.mean().item()),
            "raw_a_maximum": float(symmetry.max().item()),
            "maximum_pulse_response_mismatch": float(residual.abs().max().item()),
            "mean_pulse_response_mismatch": float(residual.abs().mean().item()),
        },
    )


def initialize_auxiliary_at_symmetry(
    plant: IbmReramRawActivePlant,
    symmetry_raw_a: torch.Tensor,
) -> None:
    """Initialize a fresh auxiliary replica exactly at its declared A=0."""

    value = torch.as_tensor(
        symmetry_raw_a, dtype=torch.float32, device=plant.device
    )
    if value.shape != (plant.size,) or not bool(torch.all(torch.isfinite(value))):
        raise ValueError("Expected one finite intrinsic-symmetry value per A cell.")
    lower = plant.population.min_bound.to(plant.device)
    upper = plant.population.max_bound.to(plant.device)
    if bool(torch.any(value < lower)) or bool(torch.any(value > upper)):
        raise ValueError("Auxiliary symmetry initialization left physical support.")
    plant.persistent.copy_(value)
    # No programming/read operation is claimed here.  Apparent and persistent
    # begin equal, then apparent becomes the held post-write sample after a
    # physical A pulse.
    plant.apparent.copy_(value)


class PulseOnlyPort:
    """Pulse-only structural protocol; a verify method is intentionally absent."""

    @property
    def size(self) -> int:  # pragma: no cover - structural protocol
        raise NotImplementedError

    def apply_identical_pulses(
        self, directions: torch.Tensor, counts: torch.Tensor
    ) -> None:  # pragma: no cover - structural protocol
        raise NotImplementedError


@dataclass(frozen=True)
class PulseApplication:
    requested: int
    applied: int
    capped: int
    upward: int
    downward: int
    probability_clipped: int
    mean_probability: float
    maximum_probability: float
    applied_mask: torch.Tensor


class CappedBernoulliPulseWriter:
    """Translate a raw-x command to at most one open-loop pulse per call."""

    def __init__(
        self,
        port: PulseOnlyPort,
        *,
        device: torch.device | str,
        nominal_delta_x: float,
        pulse_cap: int,
        selection_seed: int,
    ) -> None:
        if not math.isfinite(nominal_delta_x) or nominal_delta_x <= 0.0:
            raise ValueError("Expected a positive finite nominal raw-x step.")
        if isinstance(pulse_cap, bool) or not isinstance(pulse_cap, int) or pulse_cap < 1:
            raise ValueError("Expected a positive integer pulse cap.")
        self.port = port
        self.device = torch.device(device)
        self.nominal_delta_x = float(nominal_delta_x)
        self.pulse_cap = int(pulse_cap)
        self.pulse_count = torch.zeros(port.size, dtype=torch.int64, device=self.device)
        self.generator = torch.Generator(device=self.device)
        self.generator.manual_seed(int(selection_seed))
        self.selection_seed = int(selection_seed)
        self.calls = 0

    def apply(self, command_raw_x: torch.Tensor) -> PulseApplication:
        command = torch.as_tensor(
            command_raw_x, dtype=torch.float32, device=self.device
        )
        if command.shape != (self.port.size,) or not bool(torch.all(torch.isfinite(command))):
            raise ValueError("Expected one finite raw-x command per pulse-port cell.")
        raw_probability = command.abs() / self.nominal_delta_x
        probability = raw_probability.clamp(max=1.0)
        random_value = torch.rand(
            probability.shape,
            dtype=probability.dtype,
            device=probability.device,
            generator=self.generator,
        )
        requested_mask = (random_value < probability) & (command != 0.0)
        eligible = self.pulse_count < self.pulse_cap
        applied_mask = requested_mask & eligible
        directions = torch.zeros(self.port.size, dtype=torch.int8, device=self.device)
        directions[applied_mask & (command > 0.0)] = 1
        directions[applied_mask & (command < 0.0)] = -1
        counts = applied_mask.to(torch.int64)
        self.port.apply_identical_pulses(directions, counts)
        self.pulse_count.add_(counts)
        self.calls += 1
        return PulseApplication(
            requested=int(requested_mask.sum().item()),
            applied=int(applied_mask.sum().item()),
            capped=int((requested_mask & ~eligible).sum().item()),
            upward=int((directions > 0).sum().item()),
            downward=int((directions < 0).sum().item()),
            probability_clipped=int((raw_probability > 1.0).sum().item()),
            mean_probability=float(probability.mean().item()),
            maximum_probability=float(probability.max().item()),
            applied_mask=applied_mask,
        )

    def state_dict(self) -> dict[str, Any]:
        return {
            "schema": "ebl.mnist_relu_drn.open_loop_bernoulli_writer",
            "schema_version": 1,
            "nominal_delta_x": self.nominal_delta_x,
            "pulse_cap": self.pulse_cap,
            "selection_seed": self.selection_seed,
            "pulse_count": self.pulse_count.detach().cpu().clone(),
            "selection_rng_state": self.generator.get_state().cpu().clone(),
            "calls": self.calls,
            "verify_reads": 0,
        }


class CappedDeterministicPulseWriter:
    """Dispatch thresholded signed one-pulse requests without verification."""

    def __init__(
        self,
        port: PulseOnlyPort,
        *,
        device: torch.device | str,
        pulse_cap: int,
    ) -> None:
        if isinstance(pulse_cap, bool) or not isinstance(pulse_cap, int) or pulse_cap < 1:
            raise ValueError("Expected a positive integer pulse cap.")
        self.port = port
        self.device = torch.device(device)
        self.pulse_cap = int(pulse_cap)
        self.pulse_count = torch.zeros(port.size, dtype=torch.int64, device=self.device)
        self.calls = 0

    def apply(self, directions: torch.Tensor) -> PulseApplication:
        requested_direction = torch.as_tensor(
            directions, dtype=torch.int8, device=self.device
        )
        if requested_direction.shape != (self.port.size,) or bool(
            torch.any((requested_direction < -1) | (requested_direction > 1))
        ):
            raise ValueError("Expected one signed ternary request per pulse-port cell.")
        requested_mask = requested_direction != 0
        eligible = self.pulse_count < self.pulse_cap
        applied_mask = requested_mask & eligible
        applied_direction = torch.where(
            applied_mask, requested_direction, torch.zeros_like(requested_direction)
        )
        counts = applied_mask.to(torch.int64)
        self.port.apply_identical_pulses(applied_direction, counts)
        self.pulse_count.add_(counts)
        self.calls += 1
        return PulseApplication(
            requested=int(requested_mask.sum().item()),
            applied=int(applied_mask.sum().item()),
            capped=int((requested_mask & ~eligible).sum().item()),
            upward=int((applied_direction > 0).sum().item()),
            downward=int((applied_direction < 0).sum().item()),
            probability_clipped=0,
            mean_probability=0.0,
            maximum_probability=0.0,
            applied_mask=applied_mask,
        )

    def state_dict(self) -> dict[str, Any]:
        return {
            "schema": "ebl.mnist_relu_drn.open_loop_deterministic_writer",
            "schema_version": 1,
            "pulse_cap": self.pulse_cap,
            "pulse_count": self.pulse_count.detach().cpu().clone(),
            "calls": self.calls,
            "verify_reads": 0,
        }


@dataclass(frozen=True)
class DirectOptimizerStep:
    step: int
    pulse: PulseApplication
    conceptual_column_phases: int


class ColumnSerialOpenLoopSgd:
    """Layerwise digital SGD command with open-loop Bernoulli C writes."""

    def __init__(
        self,
        port: PulseOnlyPort,
        *,
        device: torch.device | str,
        binding_shapes: Sequence[Sequence[int]],
        layer_learning_rates_raw_x: Sequence[float],
        nominal_delta_x: float,
        pulse_cap: int,
        pulse_selection_seed: int,
    ) -> None:
        self.device = torch.device(device)
        self.binding_shapes = tuple(tuple(int(v) for v in shape) for shape in binding_shapes)
        self.layer_learning_rates_raw_x = tuple(float(v) for v in layer_learning_rates_raw_x)
        if len(self.binding_shapes) != 2 or len(self.layer_learning_rates_raw_x) != 2:
            raise ValueError("Expected two SGD bindings and layer learning rates.")
        if sum(math.prod(shape) for shape in self.binding_shapes) != port.size:
            raise ValueError("SGD bindings do not cover the slow pulse port.")
        if any(len(shape) != 2 for shape in self.binding_shapes) or any(
            not math.isfinite(rate) or rate <= 0.0
            for rate in self.layer_learning_rates_raw_x
        ):
            raise ValueError("Expected rank-2 bindings and positive finite SGD rates.")
        self.writer = CappedBernoulliPulseWriter(
            port,
            device=self.device,
            nominal_delta_x=nominal_delta_x,
            pulse_cap=pulse_cap,
            selection_seed=pulse_selection_seed,
        )
        self.step_index = 0

    def step(self, physical_conductance_gradients: Sequence[torch.Tensor]) -> DirectOptimizerStep:
        gradients = tuple(physical_conductance_gradients)
        if len(gradients) != 2:
            raise ValueError("Expected two physical conductance gradients.")
        commands = []
        for gradient, shape, rate in zip(
            gradients, self.binding_shapes, self.layer_learning_rates_raw_x
        ):
            if tuple(gradient.shape) != shape or not bool(torch.all(torch.isfinite(gradient))):
                raise ValueError("Expected finite SGD gradients with frozen shapes.")
            # G=2*x, so dL/dx=2*dL/dG.
            commands.append(-rate * 2.0 * gradient.detach())
        self.step_index += 1
        return DirectOptimizerStep(
            step=self.step_index,
            pulse=self.writer.apply(flatten_physical(tuple(commands))),
            conceptual_column_phases=sum(shape[1] for shape in self.binding_shapes),
        )

    def state_dict(self) -> dict[str, Any]:
        return {
            "schema": "ebl.mnist_relu_drn.ibm_om_open_loop_sgd_state",
            "schema_version": 1,
            "evidence_tier": EVIDENCE_TIER,
            "layer_learning_rates_raw_x": list(self.layer_learning_rates_raw_x),
            "binding_shapes": [list(shape) for shape in self.binding_shapes],
            "step": self.step_index,
            "writer": self.writer.state_dict(),
            "authoritative_weight_shadow": None,
            "verify_reads_during_updates": 0,
        }


def canonical_input_group_indices(
    shape: Sequence[int],
    group_index: int,
    *,
    device: torch.device | str,
) -> torch.Tensor:
    """Indices for one logical input column's two physical source rails."""

    if (
        len(shape) != 2
        or int(shape[0]) % 2
        or int(shape[1]) % 2
        or int(shape[0]) < 2
        or int(shape[1]) < 2
    ):
        raise ValueError("Expected one even-by-even physical rail matrix.")
    rows, columns = int(shape[0]), int(shape[1])
    logical_inputs = rows // 2
    if isinstance(group_index, bool) or not isinstance(group_index, int) or not (
        0 <= group_index < logical_inputs
    ):
        raise ValueError("Canonical input-column group index is out of range.")
    column_offsets = torch.arange(columns, dtype=torch.int64, device=device)
    plus = group_index * columns + column_offsets
    minus = (group_index + logical_inputs) * columns + column_offsets
    return torch.cat((plus, minus))


@dataclass(frozen=True)
class TikiTakaStep:
    step: int
    fast_a_pulse: PulseApplication
    slow_c_pulse: PulseApplication
    a_read_events: int
    a_values_read: int
    h_threshold_crossings: tuple[int, int]
    h_cap_debt_retained: tuple[int, int]
    cursors_after: tuple[int, int]


class OmPlantTikiTaka:
    """Qualified physical-OM A / persistent-C TT-v1 or unchopped TTv2.

    This emulates the algorithms' minibatch equations and scan schedule.  It
    is not a native fully parallel IBM/AIHWKit tile implementation.
    """

    def __init__(
        self,
        slow_port: PulseOnlyPort,
        fast_plant: IbmReramRawActivePlant,
        *,
        variant: Literal["tt_v1", "tt_v2"],
        symmetry_raw_a: torch.Tensor,
        device: torch.device | str,
        binding_shapes: Sequence[Sequence[int]],
        effective_transfer_lambdas: Sequence[float],
        fast_learning_rate_raw_x: float,
        nominal_delta_x: float,
        pulse_cap: int,
        fast_pulse_selection_seed: int,
        slow_pulse_selection_seed: int,
    ) -> None:
        if variant not in TT_VARIANTS:
            raise ValueError(f"Expected TT variant in {TT_VARIANTS!r}.")
        self.variant = variant
        self.device = torch.device(device)
        self.binding_shapes = tuple(tuple(int(v) for v in shape) for shape in binding_shapes)
        self.effective_transfer_lambdas = tuple(float(v) for v in effective_transfer_lambdas)
        if len(self.binding_shapes) != 2 or len(self.effective_transfer_lambdas) != 2:
            raise ValueError("Expected two TT bindings and effective transfer lambdas.")
        if any(len(shape) != 2 or shape[0] % 2 or shape[1] % 2 for shape in self.binding_shapes):
            raise ValueError("TT bindings must be even-by-even rank-2 rail matrices.")
        size = sum(math.prod(shape) for shape in self.binding_shapes)
        if slow_port.size != size or fast_plant.size != size:
            raise ValueError("Fast A and slow C must cover the same physical bindings.")
        if any(not math.isfinite(v) or v <= 0.0 for v in self.effective_transfer_lambdas):
            raise ValueError("Expected positive finite effective transfer lambdas.")
        if not math.isfinite(fast_learning_rate_raw_x) or fast_learning_rate_raw_x <= 0.0:
            raise ValueError("Expected a positive finite fast-A learning rate.")
        symmetry = torch.as_tensor(symmetry_raw_a, dtype=torch.float32, device=self.device)
        if symmetry.shape != (size,) or not bool(torch.all(torch.isfinite(symmetry))):
            raise ValueError("Expected one finite symmetry value per fast-A cell.")
        self.fast_plant = fast_plant
        self.symmetry_raw_a = symmetry
        self.fast_learning_rate_raw_x = float(fast_learning_rate_raw_x)
        self.nominal_delta_x = float(nominal_delta_x)
        self.fast_writer = CappedBernoulliPulseWriter(
            _PlantPulsePort(fast_plant),
            device=self.device,
            nominal_delta_x=nominal_delta_x,
            pulse_cap=pulse_cap,
            selection_seed=fast_pulse_selection_seed,
        )
        if variant == "tt_v1":
            self.slow_writer: CappedBernoulliPulseWriter | CappedDeterministicPulseWriter = (
                CappedBernoulliPulseWriter(
                    slow_port,
                    device=self.device,
                    nominal_delta_x=nominal_delta_x,
                    pulse_cap=pulse_cap,
                    selection_seed=slow_pulse_selection_seed,
                )
            )
        else:
            self.slow_writer = CappedDeterministicPulseWriter(
                slow_port, device=self.device, pulse_cap=pulse_cap
            )
        self.h = (
            torch.zeros(size, dtype=torch.float32, device=self.device)
            if variant == "tt_v2"
            else None
        )
        self.cursors = [0, 0]
        self.step_index = 0
        self.binding_offsets = []
        offset = 0
        for shape in self.binding_shapes:
            self.binding_offsets.append(offset)
            offset += math.prod(shape)

    def _fast_command(self, gradients: Sequence[torch.Tensor]) -> torch.Tensor:
        if len(gradients) != 2:
            raise ValueError("Expected two TT physical conductance gradients.")
        commands = []
        for gradient, shape in zip(gradients, self.binding_shapes):
            if tuple(gradient.shape) != shape or not bool(torch.all(torch.isfinite(gradient))):
                raise ValueError("Expected finite TT gradients with frozen shapes.")
            # A is updated in raw-x coordinates: dL/dx=2*dL/dG.
            commands.append(-self.fast_learning_rate_raw_x * 2.0 * gradient.detach())
        return flatten_physical(tuple(commands))

    def _read_indices(self, layer_index: int) -> torch.Tensor:
        local = canonical_input_group_indices(
            self.binding_shapes[layer_index],
            self.cursors[layer_index],
            device=self.device,
        )
        return local + self.binding_offsets[layer_index]

    def step(self, physical_conductance_gradients: Sequence[torch.Tensor]) -> TikiTakaStep:
        fast_pulse = self.fast_writer.apply(self._fast_command(tuple(physical_conductance_gradients)))
        command = torch.zeros(
            self.fast_plant.size, dtype=torch.float32, device=self.device
        )
        directions = torch.zeros(
            self.fast_plant.size, dtype=torch.int8, device=self.device
        )
        crossings = [0, 0]
        indices_by_layer = []
        candidate_by_layer = []
        # Held apparent post-write A sample in raw-x units.  Persistent A is
        # retained separately as the physical mutation truth and diagnostic.
        apparent_read_x = (self.fast_plant.apparent - self.symmetry_raw_a) / 2.0
        for layer_index, transfer_lambda in enumerate(self.effective_transfer_lambdas):
            indices = self._read_indices(layer_index)
            indices_by_layer.append(indices)
            read_x = apparent_read_x[indices]
            if self.variant == "tt_v1":
                command[indices] = transfer_lambda * read_x
                candidate_by_layer.append(None)
            else:
                assert self.h is not None
                candidate = self.h[indices] + (
                    transfer_lambda / self.nominal_delta_x
                ) * read_x
                self.h[indices] = candidate
                candidate_by_layer.append(candidate)
                quantized = torch.trunc(candidate).clamp(min=-1.0, max=1.0)
                active = quantized != 0.0
                crossings[layer_index] = int(active.sum().item())
                directions[indices] = quantized.to(torch.int8)
        if self.variant == "tt_v1":
            assert isinstance(self.slow_writer, CappedBernoulliPulseWriter)
            slow_pulse = self.slow_writer.apply(command)
            debt = [0, 0]
        else:
            assert isinstance(self.slow_writer, CappedDeterministicPulseWriter)
            slow_pulse = self.slow_writer.apply(directions)
            assert self.h is not None
            # Forget-buffer=true, momentum=0: a dispatched pulse clears H.
            # If the C cap blocks dispatch, debt is deliberately retained.
            self.h[slow_pulse.applied_mask] = 0.0
            debt = []
            for indices, candidate in zip(indices_by_layer, candidate_by_layer):
                assert candidate is not None
                debt.append(
                    int(
                        (
                            (candidate.abs() >= 1.0)
                            & ~slow_pulse.applied_mask[indices]
                        ).sum().item()
                    )
                )
        self.step_index += 1
        for layer_index, shape in enumerate(self.binding_shapes):
            self.cursors[layer_index] = (
                self.cursors[layer_index] + 1
            ) % (shape[0] // 2)
        return TikiTakaStep(
            step=self.step_index,
            fast_a_pulse=fast_pulse,
            slow_c_pulse=slow_pulse,
            a_read_events=2,
            a_values_read=sum(2 * shape[1] for shape in self.binding_shapes),
            h_threshold_crossings=(crossings[0], crossings[1]),
            h_cap_debt_retained=(debt[0], debt[1]),
            cursors_after=(self.cursors[0], self.cursors[1]),
        )

    def state_dict(self) -> dict[str, Any]:
        return {
            "schema": "ebl.mnist_relu_drn.ibm_om_plant_tiki_taka_state",
            "schema_version": 1,
            "evidence_tier": EVIDENCE_TIER,
            "variant": self.variant,
            "qualification": (
                "OM_plant_minibatch_equation_emulator_not_native_parallel_tile"
            ),
            "binding_shapes": [list(shape) for shape in self.binding_shapes],
            "effective_transfer_lambdas": list(self.effective_transfer_lambdas),
            "scale_transfer_lr": False,
            "fast_learning_rate_raw_x": self.fast_learning_rate_raw_x,
            "nominal_delta_x": self.nominal_delta_x,
            "gamma": 0.0,
            "transfer_every_minibatches": 1,
            "units_in_mbatch": True,
            "n_reads_per_transfer": 1,
            "transfer_columns": True,
            "transfer_axis": "canonical_logical_[out,in]_input_columns",
            "random_selection": False,
            "with_reset_probability": 0.0,
            "in_chop_probability": 0.0,
            "out_chop_probability": 0.0,
            "desired_bl": 1,
            "auto_scale": False,
            "correct_gradient_magnitudes": True,
            "h_threshold": 1.0,
            "h_momentum": 0.0,
            "forget_buffer": True,
            "a_read_state": (
                "held_apparent_post_write_raw_x_displacement;"
                "not_fresh_independent_read_noise"
            ),
            "a_mutation_state": "persistent_native_raw_a",
            "a_symmetry_raw_a": self.symmetry_raw_a.detach().cpu().clone(),
            "a_plant_continuation_state": self.fast_plant.state_dict(),
            "a_writer": self.fast_writer.state_dict(),
            "c_writer": self.slow_writer.state_dict(),
            "h": None if self.h is None else self.h.detach().cpu().clone(),
            "cursors": list(self.cursors),
            "step": self.step_index,
            "algorithmic_a_reads": 2 * self.step_index,
            "verify_reads_during_updates": 0,
            "authoritative_weight_shadow": None,
        }


class _PlantPulsePort(PulseOnlyPort):
    """Pulse-only A view, structurally identical to the slow-C port."""

    def __init__(self, plant: IbmReramRawActivePlant) -> None:
        self.plant = plant

    @property
    def size(self) -> int:
        return self.plant.size

    def apply_identical_pulses(
        self, directions: torch.Tensor, counts: torch.Tensor
    ) -> None:
        direction = torch.as_tensor(directions, dtype=torch.int8, device=self.plant.device)
        count = torch.as_tensor(counts, dtype=torch.int64, device=self.plant.device)
        if direction.shape != (self.size,) or count.shape != (self.size,):
            raise ValueError("Open-loop A pulse request shape mismatch.")
        if bool(torch.any(count < 0)):
            raise ValueError("Open-loop A pulse counts must be nonnegative.")
        maximum = int(count.max().item()) if count.numel() else 0
        for pulse_index in range(maximum):
            self.plant.pulse(
                torch.where(
                    count > pulse_index, direction, torch.zeros_like(direction)
                )
            )


__all__ = [
    "CappedBernoulliPulseWriter",
    "CappedDeterministicPulseWriter",
    "ColumnSerialOpenLoopSgd",
    "DirectOptimizerStep",
    "EVIDENCE_TIER",
    "OmPlantTikiTaka",
    "PulseApplication",
    "SymmetryInitialization",
    "TikiTakaStep",
    "canonical_input_group_indices",
    "initialize_auxiliary_at_symmetry",
    "intrinsic_symmetry_raw_a",
]
