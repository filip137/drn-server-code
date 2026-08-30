"""Checkpointable IBM-OM primitives for a standard tiled crossbar MLP.

The DRN uses full non-negative conductances in a passive equilibrium solve.
This module implements the deliberately different AIHWKit-style comparator:
each programmable crosspoint exposes the effective signed state ``q=a-r`` to
an ordinary matrix-vector multiply, the hidden non-linearity is digital ReLU,
and a second ordinary MVM produces the logits.

AIHWKit 1.1.0 is still authoritative for sampled OM identity parameters.  The
pulse equation is evaluated here with explicit Torch generators because the
native CPU tile does not expose a serializable cycle-to-cycle RNG state.  This
keeps persistent deployments and recovery forks exactly replayable.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import math
from typing import Any, Mapping, Sequence

import torch

from training.ibm_reram_hwa import IbmReramArrayPopulation


@dataclass(frozen=True)
class CrossbarTileSpec:
    """One physical input-sliced crossbar tile in canonical order."""

    key: str
    layer_index: int
    tile_index: int
    input_start: int
    input_stop: int
    out_features: int

    @property
    def shape(self) -> tuple[int, int]:
        return (self.input_stop - self.input_start, self.out_features)

    @property
    def cells(self) -> int:
        return math.prod(self.shape)


def build_crossbar_layout(
    dims: Sequence[int],
    *,
    maximum_input_size: int,
) -> tuple[CrossbarTileSpec, ...]:
    """Split each dense input dimension as AIHWKit's balanced tile array does."""

    dimensions = tuple(int(value) for value in dims)
    if (
        len(dimensions) != 3
        or any(value < 1 for value in dimensions)
        or isinstance(maximum_input_size, bool)
        or maximum_input_size < 1
    ):
        raise ValueError(
            "Expected three positive dimensions and a positive maximum input size."
        )
    result: list[CrossbarTileSpec] = []
    for layer_index, (inputs, outputs) in enumerate(
        zip(dimensions[:-1], dimensions[1:], strict=True)
    ):
        tile_count = math.ceil(inputs / maximum_input_size)
        base, remainder = divmod(inputs, tile_count)
        start = 0
        for tile_index in range(tile_count):
            width = base + (1 if tile_index < remainder else 0)
            stop = start + width
            result.append(
                CrossbarTileSpec(
                    key=f"crossbar.layer{layer_index}.tile{tile_index}",
                    layer_index=layer_index,
                    tile_index=tile_index,
                    input_start=start,
                    input_stop=stop,
                    out_features=outputs,
                )
            )
            start = stop
        if start != inputs:  # pragma: no cover - arithmetic invariant
            raise RuntimeError("Expected the balanced tile split to cover its input.")
    return tuple(result)


def validate_population_layout(
    population: IbmReramArrayPopulation,
    layout: Sequence[CrossbarTileSpec],
) -> None:
    expected_keys = tuple(tile.key for tile in layout)
    expected_shapes = tuple(tile.shape for tile in layout)
    if (
        population.binding_keys != expected_keys
        or population.binding_shapes != expected_shapes
        or population.size != sum(tile.cells for tile in layout)
    ):
        raise ValueError(
            "Expected the IBM OM population to match the standard-crossbar "
            f"layout exactly. Provided value: keys={population.binding_keys!r}, "
            f"shapes={population.binding_shapes!r}."
        )


def apply_population_bound_policy(
    population: IbmReramArrayPopulation,
    *,
    policy: str,
) -> tuple[IbmReramArrayPopulation, dict[str, Any]]:
    """Apply the declared native or DRN-matched bound treatment.

    The matched policy reproduces the recent four-device DRN control's only
    bound intervention: sample and repair the complete identity first, then
    intersect each raw active-state support with ``a in [-1, 1]``.  It never
    resamples a cell and it leaves pulse sizes and intrinsic references fixed.
    """

    if policy not in {
        "native_sampled_bounds",
        "winsorize_raw_active_a_to_unit_interval",
    }:
        raise ValueError("Expected a supported IBM OM population-bound policy.")
    original_minimum = population.min_bound.detach().cpu().to(torch.float32)
    original_maximum = population.max_bound.detach().cpu().to(torch.float32)
    if policy == "native_sampled_bounds":
        minimum = original_minimum.clone()
        maximum = original_maximum.clone()
    else:
        minimum = torch.maximum(original_minimum, torch.full_like(original_minimum, -1.0))
        maximum = torch.minimum(original_maximum, torch.full_like(original_maximum, 1.0))
    empty = maximum < minimum
    if bool(torch.any(empty)):
        raise RuntimeError(
            "An IBM OM identity has no support after applying the declared "
            "bound policy; refusing to resample it."
        )
    reference = population.reference.detach().cpu().to(torch.float32)
    reference_outside = (reference < minimum) | (reference > maximum)

    lower_changed = minimum != original_minimum
    upper_changed = maximum != original_maximum
    if policy == "native_sampled_bounds":
        fingerprint = population.fingerprint
    else:
        digest = sha256()
        digest.update(population.fingerprint.encode("utf-8"))
        digest.update(policy.encode("utf-8"))
        digest.update(minimum.contiguous().numpy().tobytes())
        digest.update(maximum.contiguous().numpy().tobytes())
        fingerprint = digest.hexdigest()
    result = IbmReramArrayPopulation(
        assignment_seed=population.assignment_seed,
        corruption_policy=population.corruption_policy,
        binding_keys=population.binding_keys,
        binding_shapes=population.binding_shapes,
        binding_sampling_seeds=population.binding_sampling_seeds,
        donor_sampling_seeds=population.donor_sampling_seeds,
        nominal_dw_min=population.nominal_dw_min,
        dw_min_std=population.dw_min_std,
        write_noise_std=population.write_noise_std,
        max_bound=maximum.clone(),
        min_bound=minimum.clone(),
        dwmin_up=population.dwmin_up.detach().cpu().clone(),
        dwmin_down=population.dwmin_down.detach().cpu().clone(),
        reference=reference.clone(),
        corrupt=population.corrupt.detach().cpu().clone(),
        published_corrupt=population.published_corrupt.detach().cpu().clone(),
        fingerprint=fingerprint,
        aihwkit_version=population.aihwkit_version,
    )
    return result, {
        "policy": policy,
        "source_population_fingerprint": population.fingerprint,
        "effective_population_fingerprint": result.fingerprint,
        "identity_resampled": False,
        "raw_active_interval": None if policy == "native_sampled_bounds" else [-1.0, 1.0],
        "lower_bound_changed": int(lower_changed.sum().item()),
        "upper_bound_changed": int(upper_changed.sum().item()),
        "any_bound_changed": int((lower_changed | upper_changed).sum().item()),
        "reference_changed": False,
        "reference_outside_effective_support": int(reference_outside.sum().item()),
    }


def layer_cell_slices(
    layout: Sequence[CrossbarTileSpec],
) -> tuple[slice, slice]:
    """Return the two contiguous flattened cell slices."""

    counts = [0, 0]
    seen = []
    for tile in layout:
        if tile.layer_index not in {0, 1}:
            raise ValueError("Expected exactly two crossbar layers.")
        counts[tile.layer_index] += tile.cells
        seen.append(tile.layer_index)
    if set(seen) != {0, 1}:
        raise ValueError("Expected the layout to contain both crossbar layers.")
    return (slice(0, counts[0]), slice(counts[0], counts[0] + counts[1]))


def map_logical_weights(
    logical_weights: Sequence[torch.Tensor],
    layout: Sequence[CrossbarTileSpec],
    *,
    weight_scaling_omega: Sequence[float],
) -> tuple[torch.Tensor, tuple[float, float]]:
    """Map logical weights to AIHWKit-style effective device targets ``q``.

    One positive digital output scale is shared by all tiles in a logical
    layer.  This avoids the tile-wise calibration confound that would arise if
    each input slice independently used its own maximum.
    """

    weights = tuple(
        value.detach().to(device="cpu", dtype=torch.float32).contiguous()
        for value in logical_weights
    )
    if len(weights) != 2:
        raise ValueError("Expected exactly two logical weight tensors.")
    omega = tuple(float(value) for value in weight_scaling_omega)
    if len(omega) != 2 or any(
        not math.isfinite(value) or value <= 0.0 or value > 1.0
        for value in omega
    ):
        raise ValueError("Expected two finite weight-scaling omega values in (0, 1].")

    expected_shapes: list[tuple[int, int]] = []
    for layer_index in (0, 1):
        selected = [tile for tile in layout if tile.layer_index == layer_index]
        expected_shapes.append(
            (sum(tile.shape[0] for tile in selected), selected[0].out_features)
        )
    if tuple(value.shape for value in weights) != tuple(expected_shapes):
        raise ValueError(
            "Expected logical weight shapes to match the tiled layout. "
            f"Provided value: {tuple(tuple(value.shape) for value in weights)!r}."
        )

    scales = []
    for layer_index, value in enumerate(weights):
        absolute_maximum = float(value.abs().max().item())
        if not math.isfinite(absolute_maximum) or absolute_maximum <= 0.0:
            raise ValueError("Expected every logical layer to have non-zero finite weights.")
        scales.append(absolute_maximum / omega[layer_index])

    pieces = []
    for tile in layout:
        logical = weights[tile.layer_index]
        target = (
            logical[tile.input_start : tile.input_stop]
            / scales[tile.layer_index]
        )
        pieces.append(target.reshape(-1))
    flattened = torch.cat(pieces)
    if not bool(torch.all(torch.isfinite(flattened))):
        raise RuntimeError("Expected finite effective crossbar targets.")
    return flattened, (float(scales[0]), float(scales[1]))


def effective_state_to_logical_weights(
    effective_state: torch.Tensor,
    layout: Sequence[CrossbarTileSpec],
    *,
    digital_scales: Sequence[float],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reassemble flattened device states into two logical dense matrices."""

    flat = effective_state.reshape(-1)
    expected = sum(tile.cells for tile in layout)
    scales = tuple(float(value) for value in digital_scales)
    if flat.numel() != expected or len(scales) != 2 or any(
        not math.isfinite(value) or value <= 0.0 for value in scales
    ):
        raise ValueError("Expected a complete effective state and two positive scales.")
    offsets = 0
    layers: list[list[torch.Tensor]] = [[], []]
    for tile in layout:
        stop = offsets + tile.cells
        value = flat[offsets:stop].reshape(tile.shape)
        layers[tile.layer_index].append(value)
        offsets = stop
    logical = []
    for layer_index in (0, 1):
        logical.append(torch.cat(layers[layer_index], dim=0) * scales[layer_index])
    return logical[0], logical[1]


def standard_crossbar_logits(
    inputs: torch.Tensor,
    effective_state: torch.Tensor,
    layout: Sequence[CrossbarTileSpec],
    *,
    digital_scales: Sequence[float],
) -> torch.Tensor:
    """Analog MVM -- digital ReLU -- analog MVM forward equation."""

    weight_0, weight_1 = effective_state_to_logical_weights(
        effective_state,
        layout,
        digital_scales=digital_scales,
    )
    weight_0 = weight_0.to(device=inputs.device, dtype=inputs.dtype)
    weight_1 = weight_1.to(device=inputs.device, dtype=inputs.dtype)
    return torch.relu(inputs @ weight_0) @ weight_1


@dataclass(frozen=True)
class DeterministicEffectiveCodebook:
    """Per-cell lower-to-SET no-noise effective-state codebook."""

    values: torch.Tensor
    effective_level_counts: torch.Tensor

    @property
    def maximum_pulses(self) -> int:
        return int(self.values.shape[0]) - 1


def build_deterministic_effective_codebook(
    population: IbmReramArrayPopulation,
    *,
    maximum_pulses: int,
) -> DeterministicEffectiveCodebook:
    """Enumerate exact ``xi=0`` OM states and retain the sampled reference."""

    if isinstance(maximum_pulses, bool) or maximum_pulses < 1:
        raise ValueError("Expected a positive deterministic pulse cap.")
    minimum = population.min_bound.detach().cpu().to(torch.float32)
    active = minimum.clone()
    maximum = population.max_bound.detach().cpu().to(torch.float32)
    step = population.dwmin_up.detach().cpu().to(torch.float32)
    reference = population.reference.detach().cpu().to(torch.float32)
    levels = [active - reference]
    for _ in range(maximum_pulses):
        normalized = torch.where(maximum > 0.0, active / maximum, torch.zeros_like(active))
        active = torch.minimum(
            torch.maximum(active + step * (1.0 - normalized), minimum),
            maximum,
        )
        levels.append(active - reference)
    values = torch.stack(levels)
    if not bool(torch.all(torch.isfinite(values))) or bool(
        torch.any(values[1:] < values[:-1])
    ):
        raise RuntimeError("Expected finite monotone deterministic OM codebooks.")
    counts = 1 + (values[1:] != values[:-1]).sum(dim=0).to(torch.int64)
    return DeterministicEffectiveCodebook(values=values, effective_level_counts=counts)


def project_to_nearest_effective_code(
    codebook: DeterministicEffectiveCodebook,
    targets: torch.Tensor,
    *,
    chunk_size: int = 8192,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Project targets with the lowest-pulse index winning exact ties."""

    target = targets.detach().cpu().to(torch.float32).reshape(-1)
    values = codebook.values
    if (
        values.ndim != 2
        or values.shape[1] != target.numel()
        or chunk_size < 1
        or not bool(torch.all(torch.isfinite(target)))
    ):
        raise ValueError("Expected one finite target for every deterministic codebook.")
    realized = torch.empty_like(target)
    indices = torch.empty(target.shape, dtype=torch.int64)
    for start in range(0, target.numel(), chunk_size):
        stop = min(start + chunk_size, target.numel())
        local = values[:, start:stop]
        selected = torch.abs(local - target[start:stop].unsqueeze(0)).argmin(dim=0)
        realized[start:stop] = local.gather(0, selected.unsqueeze(0)).squeeze(0)
        indices[start:stop] = selected
    return realized, indices


def project_to_nearest_effective_code_device(
    codebook_values: torch.Tensor,
    targets: torch.Tensor,
    *,
    rowwise: bool = False,
    validate: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Project row-wise monotone codebooks without leaving the target device.

    This is the training-time counterpart of
    :func:`project_to_nearest_effective_code`. ``torch.searchsorted`` avoids a
    dense levels-by-cells distance tensor on every QAT minibatch. Exact ties
    select the lower pulse index. Callers may cache a contiguous
    cells-by-levels tensor with ``rowwise=True`` and, after one validated call,
    use ``validate=False`` inside a fixed-codebook training loop.
    """

    if not isinstance(rowwise, bool) or not isinstance(validate, bool):
        raise TypeError("Expected boolean rowwise and validate flags.")
    values = codebook_values
    target = targets.reshape(-1)
    level_count = values.shape[1] if rowwise and values.ndim == 2 else (
        values.shape[0] if values.ndim == 2 else 0
    )
    expected_shape = (
        (target.numel(), values.shape[1])
        if rowwise and values.ndim == 2
        else (values.shape[0], target.numel())
        if values.ndim == 2
        else ()
    )
    if (
        values.ndim != 2
        or values.shape != expected_shape
        or level_count < 2
        or target.numel() < 1
        or values.device != target.device
        or values.dtype != target.dtype
        or (validate and not bool(torch.all(torch.isfinite(target))))
    ):
        raise ValueError("Expected finite monotone device-local codebooks and targets.")
    table = values if rowwise else values.transpose(0, 1).contiguous()
    if validate and (
        not bool(torch.all(torch.isfinite(table)))
        or bool(torch.any(table[:, 1:] < table[:, :-1]))
    ):
        raise ValueError("Expected finite monotone device-local codebooks and targets.")
    insertion = torch.searchsorted(
        table,
        target.unsqueeze(1),
        right=False,
    ).squeeze(1)
    high_index = insertion.clamp(min=1, max=table.shape[1] - 1)
    low_index = high_index - 1
    rows = torch.arange(target.numel(), device=target.device)
    low = table[rows, low_index]
    high = table[rows, high_index]
    choose_high = torch.abs(high - target) < torch.abs(target - low)
    indices = torch.where(choose_high, high_index, low_index)
    realized = table[rows, indices]
    return realized, indices


class IbmOmEffectiveCrossbarPlant:
    """Persistent ``q=a-r`` OM plant with one serializable RNG stream."""

    STATE_SCHEMA = "ebl.ibm_om_effective_crossbar_plant"

    def __init__(
        self,
        population: IbmReramArrayPopulation,
        *,
        generator: torch.Generator,
        device: torch.device | str,
    ) -> None:
        self.device = torch.device(device)
        self.population = population.to(self.device)
        self.generator = generator
        self.persistent = self.population.logical_min.clone()
        self.apparent = self.persistent + self._write_scale * self._normal_all()
        self.upward_pulses = torch.zeros(
            self.population.size, dtype=torch.int64, device=self.device
        )
        self.downward_pulses = torch.zeros_like(self.upward_pulses)

    @property
    def size(self) -> int:
        return self.population.size

    @property
    def _write_scale(self) -> float:
        return float(self.population.write_noise_std * self.population.nominal_dw_min)

    def _normal_all(self) -> torch.Tensor:
        return torch.randn(
            (self.size,),
            dtype=torch.float32,
            device=self.device,
            generator=self.generator,
        )

    def pulse(self, directions: torch.Tensor) -> None:
        direction = torch.as_tensor(directions, dtype=torch.int8, device=self.device)
        if direction.shape != (self.size,) or bool(
            torch.any((direction < -1) | (direction > 1))
        ):
            raise ValueError("Expected one {-1,0,1} pulse direction per crosspoint.")
        active = direction != 0
        if not bool(torch.any(active)):
            return
        population = self.population
        active_state = self.persistent + population.reference
        cycle = self._normal_all()
        candidate = active_state.clone()
        upward = direction > 0
        downward = direction < 0
        if bool(torch.any(upward)):
            normalized = torch.where(
                population.max_bound > 0.0,
                active_state / population.max_bound,
                torch.zeros_like(active_state),
            )
            response = population.dwmin_up * (
                1.0 - normalized + population.dw_min_std * cycle
            )
            candidate[upward] = active_state[upward] + response[upward]
        if bool(torch.any(downward)):
            normalized = torch.where(
                population.min_bound < 0.0,
                active_state / population.min_bound,
                torch.zeros_like(active_state),
            )
            response = population.dwmin_down * (
                1.0 - normalized + population.dw_min_std * cycle
            )
            candidate[downward] = active_state[downward] - response[downward]
        candidate = torch.maximum(candidate, population.min_bound)
        candidate = torch.minimum(candidate, population.max_bound)
        self.persistent[active] = (candidate - population.reference)[active]
        apparent = self.persistent + self._write_scale * self._normal_all()
        self.apparent[active] = apparent[active]
        self.upward_pulses += upward.to(torch.int64)
        self.downward_pulses += downward.to(torch.int64)

    def controller_port(self) -> "_EffectiveControllerPort":
        return _EffectiveControllerPort(self)

    def state_dict(self) -> dict[str, Any]:
        return {
            "schema": self.STATE_SCHEMA,
            "schema_version": 1,
            "population_fingerprint": self.population.fingerprint,
            "persistent": self.persistent.detach().cpu().clone(),
            "apparent": self.apparent.detach().cpu().clone(),
            "upward_pulses": self.upward_pulses.detach().cpu().clone(),
            "downward_pulses": self.downward_pulses.detach().cpu().clone(),
            "generator_state": self.generator.get_state().detach().cpu().clone(),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        expected = {
            "schema",
            "schema_version",
            "population_fingerprint",
            "persistent",
            "apparent",
            "upward_pulses",
            "downward_pulses",
            "generator_state",
        }
        if (
            set(state) != expected
            or state["schema"] != self.STATE_SCHEMA
            or state["schema_version"] != 1
            or state["population_fingerprint"] != self.population.fingerprint
        ):
            raise ValueError("Expected a matching effective-crossbar plant state.")
        for name, target, dtype in (
            ("persistent", self.persistent, torch.float32),
            ("apparent", self.apparent, torch.float32),
            ("upward_pulses", self.upward_pulses, torch.int64),
            ("downward_pulses", self.downward_pulses, torch.int64),
        ):
            value = state[name]
            if not isinstance(value, torch.Tensor) or value.shape != target.shape or value.dtype != dtype:
                raise ValueError(f"Expected saved crossbar field {name!r} to match.")
            target.copy_(value.to(self.device))
        generator_state = state["generator_state"]
        if not isinstance(generator_state, torch.Tensor):
            raise ValueError("Expected a saved crossbar generator state tensor.")
        self.generator.set_state(generator_state.detach().cpu())

    def pulse_statistics(self) -> dict[str, Any]:
        total = self.upward_pulses + self.downward_pulses
        active = self.persistent + self.population.reference
        return {
            "total": int(total.sum().item()),
            "upward": int(self.upward_pulses.sum().item()),
            "downward": int(self.downward_pulses.sum().item()),
            "changed_cells": int((total > 0).sum().item()),
            "maximum_per_cell": int(total.max().item()) if total.numel() else 0,
            "saturated_lower": int(
                torch.isclose(active, self.population.min_bound, atol=1e-7, rtol=0.0).sum().item()
            ),
            "saturated_upper": int(
                torch.isclose(active, self.population.max_bound, atol=1e-7, rtol=0.0).sum().item()
            ),
        }


class _EffectiveControllerPort:
    """P&V port exposing only apparent normalized effective state."""

    def __init__(self, plant: IbmOmEffectiveCrossbarPlant) -> None:
        self._plant = plant

    @property
    def size(self) -> int:
        return self._plant.size

    def verify(self) -> torch.Tensor:
        return (self._plant.apparent.clone() + 1.0) / 2.0

    def apply_identical_pulses(
        self,
        directions: torch.Tensor,
        counts: torch.Tensor,
    ) -> None:
        direction = torch.as_tensor(directions, dtype=torch.int8, device=self._plant.device)
        count = torch.as_tensor(counts, dtype=torch.int64, device=self._plant.device)
        if direction.shape != (self.size,) or count.shape != (self.size,) or bool(
            torch.any(count < 0)
        ):
            raise ValueError("Expected valid P&V directions and counts per crosspoint.")
        maximum = int(count.max().item()) if count.numel() else 0
        for index in range(maximum):
            self._plant.pulse(
                torch.where(count > index, direction, torch.zeros_like(direction))
            )


class PulseAdam:
    """Digital Adam moments followed by one open-loop OM pulse opportunity."""

    def __init__(
        self,
        *,
        size: int,
        layout: Sequence[CrossbarTileSpec],
        learning_rates: Sequence[float],
        betas: Sequence[float],
        epsilon: float,
        layer_scope: str,
        nominal_dw_min: float,
        pulse_cap_per_cell: int,
        generator: torch.Generator,
        device: torch.device | str,
    ) -> None:
        rates = tuple(float(value) for value in learning_rates)
        beta = tuple(float(value) for value in betas)
        if (
            size < 1
            or len(rates) != 2
            or any(not math.isfinite(value) or value <= 0.0 for value in rates)
            or len(beta) != 2
            or not 0.0 <= beta[0] < 1.0
            or not 0.0 <= beta[1] < 1.0
            or not math.isfinite(epsilon)
            or epsilon <= 0.0
            or not math.isfinite(nominal_dw_min)
            or nominal_dw_min <= 0.0
            or isinstance(pulse_cap_per_cell, bool)
            or pulse_cap_per_cell < 1
            or layer_scope not in {"all", "input_only", "output_only"}
        ):
            raise ValueError("Expected a valid two-layer pulse-Adam configuration.")
        self.device = torch.device(device)
        self.generator = generator
        self.beta_1, self.beta_2 = beta
        self.epsilon = float(epsilon)
        self.nominal_dw_min = float(nominal_dw_min)
        self.pulse_cap_per_cell = int(pulse_cap_per_cell)
        self.step_index = 0
        self.first_moment = torch.zeros(size, dtype=torch.float32, device=self.device)
        self.second_moment = torch.zeros_like(self.first_moment)
        slices = layer_cell_slices(layout)
        self.learning_rate = torch.empty_like(self.first_moment)
        self.learning_rate[slices[0]] = rates[0]
        self.learning_rate[slices[1]] = rates[1]
        self.enabled = torch.zeros(size, dtype=torch.bool, device=self.device)
        if layer_scope in {"all", "input_only"}:
            self.enabled[slices[0]] = True
        if layer_scope in {"all", "output_only"}:
            self.enabled[slices[1]] = True
        self.requested = 0
        self.applied = 0
        self.probability_clipped = 0
        self.blocked_at_cap = 0
        self.pulse_count = torch.zeros(size, dtype=torch.int64, device=self.device)
        self._layer_slices = slices

    def step(
        self,
        gradient: torch.Tensor,
        plant: IbmOmEffectiveCrossbarPlant,
    ) -> dict[str, int | float]:
        value = gradient.detach().to(device=self.device, dtype=torch.float32).reshape(-1)
        if value.shape != self.first_moment.shape or not bool(torch.all(torch.isfinite(value))):
            raise ValueError("Expected one finite gradient per crosspoint.")
        self.step_index += 1
        self.first_moment.mul_(self.beta_1).add_(value, alpha=1.0 - self.beta_1)
        self.second_moment.mul_(self.beta_2).addcmul_(
            value, value, value=1.0 - self.beta_2
        )
        first_hat = self.first_moment / (1.0 - self.beta_1**self.step_index)
        second_hat = self.second_moment / (1.0 - self.beta_2**self.step_index)
        command = -self.learning_rate * first_hat / (second_hat.sqrt() + self.epsilon)
        command = torch.where(self.enabled, command, torch.zeros_like(command))
        raw_probability = command.abs() / self.nominal_dw_min
        clipped = raw_probability > 1.0
        probability = raw_probability.clamp(max=1.0)
        candidate = torch.rand(
            probability.shape,
            dtype=torch.float32,
            device=self.device,
            generator=self.generator,
        ) < probability
        at_cap = self.pulse_count >= self.pulse_cap_per_cell
        blocked = candidate & self.enabled & at_cap
        selected = candidate & self.enabled & ~at_cap
        direction = torch.sign(command).to(torch.int8) * selected.to(torch.int8)
        requested = int((raw_probability > 0.0).sum().item())
        applied = int(selected.sum().item())
        clipped_count = int(clipped.sum().item())
        blocked_count = int(blocked.sum().item())
        self.requested += requested
        self.applied += applied
        self.probability_clipped += clipped_count
        self.blocked_at_cap += blocked_count
        self.pulse_count += selected.to(torch.int64)
        plant.pulse(direction)
        return {
            "requested_nonzero_commands": requested,
            "applied_pulses": applied,
            "probability_clipped": clipped_count,
            "blocked_at_cap": blocked_count,
            "maximum_probability_before_clip": float(raw_probability.max().item()),
        }

    def report(self) -> dict[str, Any]:
        layer_counts = [
            int(self.pulse_count[layer_slice].sum().item())
            for layer_slice in self._layer_slices
        ]
        return {
            "optimizer_steps": self.step_index,
            "requested_nonzero_commands": self.requested,
            "applied_pulses": self.applied,
            "applied_pulses_by_layer": layer_counts,
            "probability_clipped": self.probability_clipped,
            "blocked_at_cap": self.blocked_at_cap,
            "pulse_cap_per_cell": self.pulse_cap_per_cell,
            "cells_at_cap": int(
                (self.pulse_count >= self.pulse_cap_per_cell).sum().item()
            ),
            "changed_cells": int((self.pulse_count > 0).sum().item()),
            "maximum_pulses_per_cell": int(self.pulse_count.max().item()),
            "enabled_cells": int(self.enabled.sum().item()),
        }


def tensor_sha256(value: torch.Tensor) -> str:
    tensor = value.detach().cpu().contiguous()
    digest = sha256()
    digest.update(str(tensor.dtype).encode("utf-8"))
    digest.update(repr(tuple(tensor.shape)).encode("utf-8"))
    digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


__all__ = [
    "CrossbarTileSpec",
    "DeterministicEffectiveCodebook",
    "IbmOmEffectiveCrossbarPlant",
    "PulseAdam",
    "apply_population_bound_policy",
    "build_crossbar_layout",
    "build_deterministic_effective_codebook",
    "effective_state_to_logical_weights",
    "layer_cell_slices",
    "map_logical_weights",
    "project_to_nearest_effective_code",
    "project_to_nearest_effective_code_device",
    "standard_crossbar_logits",
    "tensor_sha256",
    "validate_population_layout",
]
