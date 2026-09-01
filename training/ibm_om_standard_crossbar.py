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

from training.ibm_reram_hwa import (
    IbmReramArrayPopulation,
    om_array_population_fingerprint,
)
from training.ibm_reram_program_verify import (
    OM_PRESET,
    PUBLISHED_CORRUPT_RANGE,
)


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
    fingerprint = population.fingerprint
    if policy != "native_sampled_bounds":
        fingerprint = om_array_population_fingerprint(
            assignment_seed=population.assignment_seed,
            corruption_policy=population.corruption_policy,
            keys=population.binding_keys,
            shapes=population.binding_shapes,
            binding_sampling_seeds=population.binding_sampling_seeds,
            donor_sampling_seeds=population.donor_sampling_seeds,
            scalar_parameters={
                "aihwkit_version": population.aihwkit_version,
                "nominal_dw_min": population.nominal_dw_min,
                "dw_min_std": population.dw_min_std,
                "write_noise_std": population.write_noise_std,
            },
            tensors={
                "max_bound": maximum,
                "min_bound": minimum,
                "dwmin_up": population.dwmin_up,
                "dwmin_down": population.dwmin_down,
                "reference": reference,
                "corrupt": population.corrupt,
                "published_corrupt": population.published_corrupt,
            },
        )
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

    return standard_crossbar_forward_states(
        inputs,
        effective_state,
        layout,
        digital_scales=digital_scales,
    ).output_logits


@dataclass(frozen=True)
class StandardCrossbarForwardStates:
    """Locally observable states of the two-crossbar digital-ReLU path."""

    hidden_preactivation: torch.Tensor
    hidden_post_relu: torch.Tensor
    output_logits: torch.Tensor


def standard_crossbar_forward_states(
    inputs: torch.Tensor,
    effective_state: torch.Tensor,
    layout: Sequence[CrossbarTileSpec],
    *,
    digital_scales: Sequence[float],
) -> StandardCrossbarForwardStates:
    """Return ``u``, ``h=ReLU(u)``, and ``z`` without hiding local states."""

    weight_0, weight_1 = effective_state_to_logical_weights(
        effective_state,
        layout,
        digital_scales=digital_scales,
    )
    weight_0 = weight_0.to(device=inputs.device, dtype=inputs.dtype)
    weight_1 = weight_1.to(device=inputs.device, dtype=inputs.dtype)
    hidden_preactivation = inputs @ weight_0
    hidden_post_relu = torch.relu(hidden_preactivation)
    return StandardCrossbarForwardStates(
        hidden_preactivation=hidden_preactivation,
        hidden_post_relu=hidden_post_relu,
        output_logits=hidden_post_relu @ weight_1,
    )


def flatten_logical_crossbar_matrices(
    matrices: Sequence[torch.Tensor],
    layout: Sequence[CrossbarTileSpec],
) -> torch.Tensor:
    """Flatten two logical matrices in the physical tile order."""

    values = tuple(matrices)
    if len(values) != 2:
        raise ValueError("Expected exactly two logical crossbar matrices.")
    pieces: list[torch.Tensor] = []
    for tile in layout:
        value = values[tile.layer_index]
        expected_rows = sum(
            item.shape[0]
            for item in layout
            if item.layer_index == tile.layer_index
        )
        if value.ndim != 2 or value.shape != (expected_rows, tile.out_features):
            raise ValueError("Expected logical matrices to match the tiled layout.")
        pieces.append(value[tile.input_start : tile.input_stop].reshape(-1))
    if not pieces:
        raise ValueError("Expected a non-empty two-layer crossbar layout.")
    return torch.cat(pieces)


@dataclass(frozen=True)
class LocalStarCrossbarStep:
    """One label-addressed STAR step made only from layer-local errors."""

    gradient_q: torch.Tensor
    hidden_error: torch.Tensor
    output_error: torch.Tensor
    hidden_loss: torch.Tensor
    output_loss: torch.Tensor
    states: StandardCrossbarForwardStates


@torch.no_grad()
def local_star_crossbar_step(
    inputs: torch.Tensor,
    labels: torch.Tensor,
    effective_state: torch.Tensor,
    layout: Sequence[CrossbarTileSpec],
    *,
    digital_scales: Sequence[float],
    hidden_targets: torch.Tensor,
    output_targets: torch.Tensor,
    hidden_gain: float,
    output_gain: float,
) -> LocalStarCrossbarStep:
    """Compute stopped, layer-local STAR errors and physical-``q`` gradients.

    The class label only addresses the stored target row.  There is no
    softmax/cross-entropy term, teacher query, transpose MVM, or cross-layer
    error propagation.  The hidden and output outer products are gradients of
    two separate local state-matching objectives.
    """

    if inputs.ndim != 2 or labels.ndim != 1 or labels.shape[0] != inputs.shape[0]:
        raise ValueError("Expected rank-2 inputs and one label per example.")
    gains = (float(hidden_gain), float(output_gain))
    if any(not math.isfinite(value) or value < 0.0 for value in gains):
        raise ValueError("Expected finite non-negative STAR gains.")
    scales = tuple(float(value) for value in digital_scales)
    if len(scales) != 2 or any(
        not math.isfinite(value) or value <= 0.0 for value in scales
    ):
        raise ValueError("Expected two finite positive digital scales.")
    labels_device = labels.to(device=inputs.device, dtype=torch.int64)
    hidden_table = hidden_targets.to(device=inputs.device, dtype=inputs.dtype)
    output_table = output_targets.to(device=inputs.device, dtype=inputs.dtype)
    if (
        hidden_table.ndim != 2
        or output_table.ndim != 2
        or hidden_table.shape[0] != output_table.shape[0]
        or hidden_table.shape[0] < 1
        or bool(torch.any(labels_device < 0))
        or bool(torch.any(labels_device >= hidden_table.shape[0]))
        or not bool(torch.all(torch.isfinite(hidden_table)))
        or not bool(torch.all(torch.isfinite(output_table)))
    ):
        raise ValueError("Expected finite class-indexed STAR target tables.")

    states = standard_crossbar_forward_states(
        inputs,
        effective_state,
        layout,
        digital_scales=scales,
    )
    if (
        hidden_table.shape[1] != states.hidden_post_relu.shape[1]
        or output_table.shape[1] != states.output_logits.shape[1]
    ):
        raise ValueError("Expected STAR target widths to match local network states.")
    batch_size = inputs.shape[0]
    hidden_delta = states.hidden_post_relu - hidden_table[labels_device]
    output_delta = states.output_logits - output_table[labels_device]
    hidden_error = (
        gains[0]
        * hidden_delta
        * (states.hidden_preactivation > 0).to(inputs.dtype)
        / batch_size
    )
    output_error = gains[1] * output_delta / batch_size

    # Stop the inter-layer path: W1 sees only its hidden target error, while W2
    # sees the current hidden state solely as its locally available row signal.
    gradient_w0 = inputs.transpose(0, 1) @ hidden_error
    gradient_w1 = states.hidden_post_relu.detach().transpose(0, 1) @ output_error
    gradient_q = flatten_logical_crossbar_matrices(
        (scales[0] * gradient_w0, scales[1] * gradient_w1),
        layout,
    )
    return LocalStarCrossbarStep(
        gradient_q=gradient_q,
        hidden_error=hidden_error,
        output_error=output_error,
        hidden_loss=0.5 * gains[0] * hidden_delta.square().sum(dim=1).mean(),
        output_loss=0.5 * gains[1] * output_delta.square().sum(dim=1).mean(),
        states=states,
    )


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
    corrupt = population.corrupt.detach().cpu()
    if bool(torch.any(counts[corrupt] != 1)) or bool(
        torch.any(values[:, corrupt] != values[0, corrupt].unsqueeze(0))
    ):
        raise RuntimeError(
            "Expected every collapsed zero-step IBM OM cell to retain one "
            "immutable deterministic code."
        )
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
        self.post_deployment_fault_mask = torch.zeros(
            self.population.size, dtype=torch.bool, device=self.device
        )
        self.post_deployment_stuck_persistent_q = torch.zeros(
            self.population.size, dtype=torch.float32, device=self.device
        )
        self.post_deployment_fault_pulse_baseline = torch.zeros(
            self.population.size, dtype=torch.int64, device=self.device
        )
        self.fault_transition: dict[str, Any] | None = None

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
        movable = active & ~self.post_deployment_fault_mask
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
        self.persistent[movable] = (candidate - population.reference)[movable]
        apparent = self.persistent + self._write_scale * self._normal_all()
        # AIHWKit corrupt cells have zero persistent increments, but write
        # noise still changes the apparent state whenever a write is attempted.
        self.apparent[active] = apparent[active]
        self.upward_pulses += upward.to(torch.int64)
        self.downward_pulses += downward.to(torch.int64)

    def apply_grouped_pulse_counts(
        self,
        positive_counts: torch.Tensor,
        negative_counts: torch.Tensor,
    ) -> None:
        """Apply counted coincidence pulses with an explicit grouped ordering.

        Stochastic-compressed bit lines can command both update directions for
        one cell over a minibatch.  The caller retains the exact positive and
        negative coincidence counts; this plant applies all positive pulse
        rounds first and all negative pulse rounds second.  The grouping is a
        declared distribution-level approximation to a native tile's hidden
        bit-line ordering, not a claim of bitwise AIHWKit replay.
        """

        positive = torch.as_tensor(
            positive_counts, dtype=torch.int64, device=self.device
        )
        negative = torch.as_tensor(
            negative_counts, dtype=torch.int64, device=self.device
        )
        if (
            positive.shape != (self.size,)
            or negative.shape != (self.size,)
            or bool(torch.any(positive < 0))
            or bool(torch.any(negative < 0))
        ):
            raise ValueError(
                "Expected non-negative positive/negative pulse counts per crosspoint."
            )
        positive_maximum = int(positive.max().item()) if positive.numel() else 0
        negative_maximum = int(negative.max().item()) if negative.numel() else 0
        for pulse_index in range(positive_maximum):
            self.pulse((positive > pulse_index).to(torch.int8))
        for pulse_index in range(negative_maximum):
            self.pulse(-(negative > pulse_index).to(torch.int8))

    def apply_stuck_at_fault_transition(
        self,
        *,
        mask: torch.Tensor,
        stuck_persistent_q: torch.Tensor,
        transition_id: str,
        source_population_fingerprint: str,
    ) -> dict[str, Any]:
        """Apply one immutable post-deployment stuck-state transition.

        The mask is transition-authority data: it is stored inside the plant
        for pulse blocking and reporting, but is never required by the STAR
        gradient or pulse-writer interfaces.
        """

        fault_mask = torch.as_tensor(mask, dtype=torch.bool, device=self.device)
        values = torch.as_tensor(
            stuck_persistent_q,
            dtype=torch.float32,
            device=self.device,
        )
        if self.fault_transition is not None:
            raise RuntimeError("Expected exactly one post-deployment fault transition.")
        if (
            not isinstance(transition_id, str)
            or not transition_id.strip()
            or not isinstance(source_population_fingerprint, str)
            or not source_population_fingerprint.strip()
            or fault_mask.shape != (self.size,)
            or values.shape != (self.size,)
            or not bool(torch.any(fault_mask))
            or not bool(torch.all(torch.isfinite(values[fault_mask])))
        ):
            raise ValueError("Expected a non-empty finite stuck-at transition.")
        pre_persistent_state = self.persistent.detach().clone()
        pre_apparent_state = self.apparent.detach().clone()
        pre_persistent = tensor_sha256(pre_persistent_state)
        pre_apparent = tensor_sha256(pre_apparent_state)
        self.post_deployment_fault_mask.copy_(fault_mask)
        self.post_deployment_stuck_persistent_q[fault_mask] = values[fault_mask]
        self.post_deployment_fault_pulse_baseline.copy_(
            self.upward_pulses + self.downward_pulses
        )
        self.persistent[fault_mask] = values[fault_mask]
        transition_apparent = self.persistent + self._write_scale * self._normal_all()
        self.apparent[fault_mask] = transition_apparent[fault_mask]
        non_fault_persistent_unchanged = bool(
            torch.equal(
                self.persistent[~fault_mask],
                pre_persistent_state[~fault_mask],
            )
        )
        non_fault_apparent_unchanged = bool(
            torch.equal(
                self.apparent[~fault_mask],
                pre_apparent_state[~fault_mask],
            )
        )
        if not non_fault_persistent_unchanged or not non_fault_apparent_unchanged:
            raise RuntimeError(
                "Expected the post-deployment fault transition to leave every "
                "non-fault cell bit-exact."
            )
        transition = {
            "schema": "ebl.ibm_om_crossbar_post_deployment_fault_transition",
            "schema_version": 1,
            "transition_id": transition_id.strip(),
            "policy": "aihwkit_sampled_corrupt_device_overlay",
            "source_population_fingerprint": source_population_fingerprint,
            "healthy_population_fingerprint": self.population.fingerprint,
            "pre_fault_persistent_sha256": pre_persistent,
            "pre_fault_apparent_sha256": pre_apparent,
            "fault_mask_sha256": tensor_sha256(fault_mask),
            "stuck_persistent_q_sha256": tensor_sha256(values[fault_mask]),
            "persistent_fault_state": "sampled_collapsed_bound_with_zero_pulse_increments",
            "apparent_fault_state": "write_noise_resampled_at_transition_and_each_write_attempt",
            "faulted_cells": int(fault_mask.sum().item()),
            "cells": self.size,
            "fault_fraction": float(fault_mask.float().mean().item()),
            "post_fault_persistent_sha256": tensor_sha256(self.persistent),
            "post_fault_apparent_sha256": tensor_sha256(self.apparent),
            "non_fault_persistent_unchanged": non_fault_persistent_unchanged,
            "non_fault_apparent_unchanged": non_fault_apparent_unchanged,
            "fault_mask_exposed_to_local_update_kernel": False,
        }
        self.fault_transition = transition
        return dict(transition)

    def controller_port(self) -> "_EffectiveControllerPort":
        return _EffectiveControllerPort(self)

    def restricted_recovery_update_port(self) -> "_LocalStarUpdatePort":
        """Expose apparent state and pulse writes without fault or persistent state."""

        if self.fault_transition is None:
            raise RuntimeError(
                "Expected the post-deployment fault before opening the recovery update port."
            )
        return _LocalStarUpdatePort(self)

    def local_star_update_port(self) -> "_LocalStarUpdatePort":
        """Backward-compatible name for the restricted recovery update port."""

        return self.restricted_recovery_update_port()

    def on_chip_recovery_port(
        self,
        *,
        layout: Sequence[CrossbarTileSpec],
        digital_scales: Sequence[float],
    ) -> "_OnChipCrossbarRecoveryPort":
        """Open a forward/BP/pulse capability after the immutable fault event.

        The returned learner view never exposes the full apparent or
        persistent tensors, the population, RNG, or fault metadata.  Forward
        MVMs and the one required transpose MVM remain capabilities of the
        physical plant rather than weight-tensor handoffs.
        """

        if self.fault_transition is None:
            raise RuntimeError(
                "Expected the post-deployment fault before opening the on-chip "
                "recovery port."
            )
        validate_population_layout(self.population, layout)
        return _OnChipCrossbarRecoveryPort(
            self,
            layout=layout,
            digital_scales=digital_scales,
        )

    def tiki_taka_fast_port(
        self,
        *,
        layout: Sequence[CrossbarTileSpec],
    ) -> "_TikiTakaFastArrayPort":
        """Open a restricted auxiliary-array pulse/read capability.

        Tiki-Taka transfer may read only the selected physical input column.
        It cannot obtain a full apparent fast matrix or any hidden plant
        authority.
        """

        validate_population_layout(self.population, layout)
        return _TikiTakaFastArrayPort(self, layout=layout)

    def state_dict(self) -> dict[str, Any]:
        return {
            "schema": self.STATE_SCHEMA,
            "schema_version": 2,
            "population_fingerprint": self.population.fingerprint,
            "persistent": self.persistent.detach().cpu().clone(),
            "apparent": self.apparent.detach().cpu().clone(),
            "upward_pulses": self.upward_pulses.detach().cpu().clone(),
            "downward_pulses": self.downward_pulses.detach().cpu().clone(),
            "post_deployment_fault_mask": self.post_deployment_fault_mask.detach().cpu().clone(),
            "post_deployment_stuck_persistent_q": (
                self.post_deployment_stuck_persistent_q.detach().cpu().clone()
            ),
            "post_deployment_fault_pulse_baseline": (
                self.post_deployment_fault_pulse_baseline.detach().cpu().clone()
            ),
            "fault_transition": (
                None if self.fault_transition is None else dict(self.fault_transition)
            ),
            "generator_state": self.generator.get_state().detach().cpu().clone(),
        }

    def load_state_dict(
        self,
        state: Mapping[str, Any],
        *,
        expected_fault_source_population: IbmReramArrayPopulation | None = None,
        expected_pre_fault_state: Mapping[str, Any] | None = None,
    ) -> None:
        """Restore one plant state, requiring authority for faulted replays.

        Healthy checkpoints remain self-contained.  A checkpoint carrying a
        post-deployment fault overlay is different: its stuck values and
        chronology are meaningful only relative to the published AIHWKit
        companion population and the exact healthy P0 state that preceded the
        transition.  Requiring both objects here prevents a coherently edited
        checkpoint from authenticating its own fault values or historical
        hashes.
        """
        expected_v1 = {
            "schema",
            "schema_version",
            "population_fingerprint",
            "persistent",
            "apparent",
            "upward_pulses",
            "downward_pulses",
            "generator_state",
        }
        expected_v2 = expected_v1 | {
            "post_deployment_fault_mask",
            "post_deployment_stuck_persistent_q",
            "post_deployment_fault_pulse_baseline",
            "fault_transition",
        }
        if (
            not isinstance(state, Mapping)
            or "schema_version" not in state
        ):
            raise ValueError("Expected a matching effective-crossbar plant state.")
        schema_version = state.get("schema_version")
        if (
            set(state) != (expected_v1 if schema_version == 1 else expected_v2)
            or state["schema"] != self.STATE_SCHEMA
            or isinstance(schema_version, bool)
            or not isinstance(schema_version, int)
            or schema_version not in {1, 2}
            or state["population_fingerprint"] != self.population.fingerprint
        ):
            raise ValueError("Expected a matching effective-crossbar plant state.")
        pending: dict[str, torch.Tensor] = {}
        for name, target, dtype in (
            ("persistent", self.persistent, torch.float32),
            ("apparent", self.apparent, torch.float32),
            ("upward_pulses", self.upward_pulses, torch.int64),
            ("downward_pulses", self.downward_pulses, torch.int64),
        ):
            value = state[name]
            if not isinstance(value, torch.Tensor) or value.shape != target.shape or value.dtype != dtype:
                raise ValueError(f"Expected saved crossbar field {name!r} to match.")
            pending[name] = value.detach().to(self.device).clone()
        if (
            not bool(torch.all(torch.isfinite(pending["persistent"])))
            or not bool(torch.all(torch.isfinite(pending["apparent"])))
            or bool(torch.any(pending["upward_pulses"] < 0))
            or bool(torch.any(pending["downward_pulses"] < 0))
        ):
            raise ValueError("Expected finite states and non-negative pulse counters.")

        pending_fault_mask = torch.zeros_like(self.post_deployment_fault_mask)
        pending_fault_q = torch.zeros_like(
            self.post_deployment_stuck_persistent_q
        )
        pending_fault_baseline = torch.zeros_like(
            self.post_deployment_fault_pulse_baseline
        )
        pending_transition: dict[str, Any] | None = None
        if schema_version == 1:
            pass
        else:
            fault_mask = state["post_deployment_fault_mask"]
            fault_q = state["post_deployment_stuck_persistent_q"]
            fault_pulse_baseline = state["post_deployment_fault_pulse_baseline"]
            transition = state["fault_transition"]
            if (
                not isinstance(fault_mask, torch.Tensor)
                or fault_mask.shape != self.post_deployment_fault_mask.shape
                or fault_mask.dtype != torch.bool
                or not isinstance(fault_q, torch.Tensor)
                or fault_q.shape != self.post_deployment_stuck_persistent_q.shape
                or fault_q.dtype != torch.float32
                or not isinstance(fault_pulse_baseline, torch.Tensor)
                or fault_pulse_baseline.shape
                != self.post_deployment_fault_pulse_baseline.shape
                or fault_pulse_baseline.dtype != torch.int64
                or (transition is not None and not isinstance(transition, Mapping))
                or (transition is None and bool(torch.any(fault_mask)))
                or (transition is not None and not bool(torch.any(fault_mask)))
            ):
                raise ValueError("Expected a valid saved stuck-at fault overlay.")
            pending_fault_mask = fault_mask.detach().to(self.device).clone()
            pending_fault_q = fault_q.detach().to(self.device).clone()
            pending_fault_baseline = (
                fault_pulse_baseline.detach().to(self.device).clone()
            )
            total_pulses = pending["upward_pulses"] + pending["downward_pulses"]
            if (
                not bool(torch.all(torch.isfinite(pending_fault_q)))
                or bool(torch.any(pending_fault_baseline < 0))
                or bool(torch.any(pending_fault_baseline > total_pulses))
            ):
                raise ValueError(
                    "Expected finite fault states and an ordered non-negative pulse baseline."
                )
            if transition is None:
                if (
                    expected_fault_source_population is not None
                    or expected_pre_fault_state is not None
                ):
                    raise ValueError(
                        "Expected fault replay bindings only for a saved fault transition."
                    )
                if (
                    bool(torch.any(pending_fault_q != 0.0))
                    or bool(torch.any(pending_fault_baseline != 0))
                ):
                    raise ValueError("Expected an empty saved fault overlay without a transition.")
            else:
                if not isinstance(
                    expected_fault_source_population,
                    IbmReramArrayPopulation,
                ) or not isinstance(expected_pre_fault_state, Mapping):
                    raise ValueError(
                        "Expected the published companion population and exact "
                        "healthy P0 state for faulted checkpoint replay."
                    )
                pending_transition = dict(transition)
                transition_fields = {
                    "schema",
                    "schema_version",
                    "transition_id",
                    "policy",
                    "source_population_fingerprint",
                    "healthy_population_fingerprint",
                    "pre_fault_persistent_sha256",
                    "pre_fault_apparent_sha256",
                    "fault_mask_sha256",
                    "stuck_persistent_q_sha256",
                    "persistent_fault_state",
                    "apparent_fault_state",
                    "faulted_cells",
                    "cells",
                    "fault_fraction",
                    "post_fault_persistent_sha256",
                    "post_fault_apparent_sha256",
                    "non_fault_persistent_unchanged",
                    "non_fault_apparent_unchanged",
                    "fault_mask_exposed_to_local_update_kernel",
                }

                def valid_sha256(value: Any) -> bool:
                    return (
                        isinstance(value, str)
                        and len(value) == 64
                        and all(character in "0123456789abcdef" for character in value)
                    )

                if set(pending_transition) != transition_fields:
                    raise ValueError(
                        "Expected exact saved fault-transition receipt fields."
                    )
                expected_fraction = float(
                    pending_fault_mask.to(torch.float32).mean().item()
                )
                reported_fraction = pending_transition["fault_fraction"]
                transition_schema_version = pending_transition["schema_version"]
                if (
                    pending_transition["schema"]
                    != "ebl.ibm_om_crossbar_post_deployment_fault_transition"
                    or isinstance(transition_schema_version, bool)
                    or not isinstance(transition_schema_version, int)
                    or transition_schema_version != 1
                    or pending_transition["policy"]
                    != "aihwkit_sampled_corrupt_device_overlay"
                    or pending_transition["persistent_fault_state"]
                    != "sampled_collapsed_bound_with_zero_pulse_increments"
                    or pending_transition["apparent_fault_state"]
                    != "write_noise_resampled_at_transition_and_each_write_attempt"
                    or not isinstance(pending_transition["transition_id"], str)
                    or not pending_transition["transition_id"].strip()
                    or not isinstance(
                        pending_transition["source_population_fingerprint"], str
                    )
                    or not pending_transition["source_population_fingerprint"].strip()
                    or pending_transition["healthy_population_fingerprint"]
                    != self.population.fingerprint
                    or any(
                        not valid_sha256(pending_transition[name])
                        for name in (
                            "pre_fault_persistent_sha256",
                            "pre_fault_apparent_sha256",
                            "fault_mask_sha256",
                            "stuck_persistent_q_sha256",
                            "post_fault_persistent_sha256",
                            "post_fault_apparent_sha256",
                        )
                    )
                    or pending_transition["fault_mask_sha256"]
                    != tensor_sha256(pending_fault_mask)
                    or pending_transition["stuck_persistent_q_sha256"]
                    != tensor_sha256(pending_fault_q[pending_fault_mask])
                    or pending_transition["faulted_cells"]
                    != int(pending_fault_mask.sum().item())
                    or pending_transition["cells"] != self.size
                    or isinstance(reported_fraction, bool)
                    or not isinstance(reported_fraction, (int, float))
                    or not math.isfinite(float(reported_fraction))
                    or not math.isclose(
                        float(reported_fraction),
                        expected_fraction,
                        rel_tol=0.0,
                        abs_tol=1e-12,
                    )
                    or pending_transition["non_fault_persistent_unchanged"] is not True
                    or pending_transition["non_fault_apparent_unchanged"] is not True
                    or pending_transition[
                        "fault_mask_exposed_to_local_update_kernel"
                    ] is not False
                    or bool(torch.any(pending_fault_q[~pending_fault_mask] != 0.0))
                ):
                    raise ValueError("Expected a self-consistent saved fault-transition receipt.")
                if not torch.equal(
                    pending["persistent"][pending_fault_mask],
                    pending_fault_q[pending_fault_mask],
                ):
                    raise ValueError("Expected saved persistent stuck cells to be immutable.")

                fault_source = expected_fault_source_population.to(self.device)
                source_mask = fault_source.corrupt
                if (
                    fault_source.corruption_policy != "published"
                    or fault_source.assignment_seed
                    != self.population.assignment_seed
                    or fault_source.binding_keys != self.population.binding_keys
                    or fault_source.binding_shapes != self.population.binding_shapes
                    or fault_source.binding_sampling_seeds
                    != self.population.binding_sampling_seeds
                    or fault_source.donor_sampling_seeds
                    != self.population.donor_sampling_seeds
                    or fault_source.nominal_dw_min
                    != self.population.nominal_dw_min
                    or fault_source.dw_min_std != self.population.dw_min_std
                    or fault_source.write_noise_std
                    != self.population.write_noise_std
                    or fault_source.aihwkit_version
                    != self.population.aihwkit_version
                    or pending_transition["source_population_fingerprint"]
                    != fault_source.fingerprint
                    or not torch.equal(source_mask, fault_source.published_corrupt)
                    or not torch.equal(source_mask, pending_fault_mask)
                    or not torch.equal(
                        self.population.published_corrupt,
                        pending_fault_mask,
                    )
                ):
                    raise ValueError(
                        "Expected the saved fault overlay to match its published "
                        "AIHWKit companion population exactly."
                    )
                healthy_mask = ~pending_fault_mask
                for name in (
                    "min_bound",
                    "max_bound",
                    "dwmin_up",
                    "dwmin_down",
                    "reference",
                ):
                    if not torch.equal(
                        getattr(fault_source, name)[healthy_mask],
                        getattr(self.population, name)[healthy_mask],
                    ):
                        raise ValueError(
                            "Expected every non-fault companion identity field "
                            "to match the healthy population."
                        )
                corrupt_range = float(PUBLISHED_CORRUPT_RANGE[OM_PRESET])
                collapsed_q = fault_source.logical_min
                if (
                    not torch.equal(
                        fault_source.min_bound[pending_fault_mask],
                        fault_source.max_bound[pending_fault_mask],
                    )
                    or not torch.equal(
                        collapsed_q[pending_fault_mask],
                        fault_source.logical_max[pending_fault_mask],
                    )
                    or not torch.equal(
                        fault_source.dwmin_up[pending_fault_mask],
                        torch.zeros_like(
                            fault_source.dwmin_up[pending_fault_mask]
                        ),
                    )
                    or not torch.equal(
                        fault_source.dwmin_down[pending_fault_mask],
                        torch.zeros_like(
                            fault_source.dwmin_down[pending_fault_mask]
                        ),
                    )
                    or bool(
                        torch.any(
                            torch.abs(
                                fault_source.min_bound[pending_fault_mask]
                            )
                            > corrupt_range + 1e-7
                        )
                    )
                    or not torch.equal(
                        pending_fault_q[pending_fault_mask],
                        collapsed_q[pending_fault_mask],
                    )
                ):
                    raise ValueError(
                        "Expected exact AIHWKit sampled collapsed stuck values, "
                        "zero pulse increments, and configured corrupt range."
                    )

                pre_fault = expected_pre_fault_state
                pre_fault_v1 = {
                    "schema",
                    "schema_version",
                    "population_fingerprint",
                    "persistent",
                    "apparent",
                    "upward_pulses",
                    "downward_pulses",
                    "generator_state",
                }
                pre_fault_v2 = pre_fault_v1 | {
                    "post_deployment_fault_mask",
                    "post_deployment_stuck_persistent_q",
                    "post_deployment_fault_pulse_baseline",
                    "fault_transition",
                }
                pre_schema_version = pre_fault.get("schema_version")
                pre_persistent = pre_fault.get("persistent")
                pre_apparent = pre_fault.get("apparent")
                pre_upward = pre_fault.get("upward_pulses")
                pre_downward = pre_fault.get("downward_pulses")
                if (
                    set(pre_fault)
                    != (pre_fault_v1 if pre_schema_version == 1 else pre_fault_v2)
                    or pre_fault.get("schema") != self.STATE_SCHEMA
                    or isinstance(pre_schema_version, bool)
                    or not isinstance(pre_schema_version, int)
                    or pre_schema_version not in {1, 2}
                    or pre_fault.get("population_fingerprint")
                    != self.population.fingerprint
                    or not isinstance(pre_persistent, torch.Tensor)
                    or pre_persistent.shape != self.persistent.shape
                    or pre_persistent.dtype != torch.float32
                    or not bool(torch.all(torch.isfinite(pre_persistent)))
                    or not isinstance(pre_apparent, torch.Tensor)
                    or pre_apparent.shape != self.apparent.shape
                    or pre_apparent.dtype != torch.float32
                    or not bool(torch.all(torch.isfinite(pre_apparent)))
                    or not isinstance(pre_upward, torch.Tensor)
                    or pre_upward.shape != self.upward_pulses.shape
                    or pre_upward.dtype != torch.int64
                    or bool(torch.any(pre_upward < 0))
                    or not isinstance(pre_downward, torch.Tensor)
                    or pre_downward.shape != self.downward_pulses.shape
                    or pre_downward.dtype != torch.int64
                    or bool(torch.any(pre_downward < 0))
                ):
                    raise ValueError(
                        "Expected a complete healthy P0 state for fault chronology."
                    )
                if pre_schema_version == 2:
                    pre_mask = pre_fault["post_deployment_fault_mask"]
                    pre_stuck = pre_fault[
                        "post_deployment_stuck_persistent_q"
                    ]
                    pre_baseline = pre_fault[
                        "post_deployment_fault_pulse_baseline"
                    ]
                    if (
                        not isinstance(pre_mask, torch.Tensor)
                        or pre_mask.shape != self.post_deployment_fault_mask.shape
                        or pre_mask.dtype != torch.bool
                        or not isinstance(pre_stuck, torch.Tensor)
                        or pre_stuck.shape
                        != self.post_deployment_stuck_persistent_q.shape
                        or pre_stuck.dtype != torch.float32
                        or not bool(torch.all(torch.isfinite(pre_stuck)))
                        or not isinstance(pre_baseline, torch.Tensor)
                        or pre_baseline.shape
                        != self.post_deployment_fault_pulse_baseline.shape
                        or pre_baseline.dtype != torch.int64
                        or pre_fault["fault_transition"] is not None
                        or bool(torch.any(pre_mask))
                        or bool(torch.any(pre_stuck != 0.0))
                        or bool(torch.any(pre_baseline != 0))
                    ):
                        raise ValueError(
                            "Expected the replay P0 state to precede every "
                            "fault transition."
                        )
                pre_upward_device = pre_upward.detach().to(self.device)
                pre_downward_device = pre_downward.detach().to(self.device)
                if (
                    pending_transition["pre_fault_persistent_sha256"]
                    != tensor_sha256(pre_persistent)
                    or pending_transition["pre_fault_apparent_sha256"]
                    != tensor_sha256(pre_apparent)
                    or not torch.equal(
                        pending_fault_baseline,
                        pre_upward_device + pre_downward_device,
                    )
                    or bool(
                        torch.any(
                            pending["upward_pulses"] < pre_upward_device
                        )
                    )
                    or bool(
                        torch.any(
                            pending["downward_pulses"] < pre_downward_device
                        )
                    )
                ):
                    raise ValueError(
                        "Expected saved fault chronology to match the exact healthy P0 hashes."
                    )
                if torch.equal(total_pulses, pending_fault_baseline) and (
                    pending_transition["post_fault_persistent_sha256"]
                    != tensor_sha256(pending["persistent"])
                    or pending_transition["post_fault_apparent_sha256"]
                    != tensor_sha256(pending["apparent"])
                ):
                    raise ValueError(
                        "Expected the immediate post-fault state hashes to match."
                    )
                if torch.equal(total_pulses, pending_fault_baseline) and (
                    not torch.equal(
                        pending["persistent"][healthy_mask],
                        pre_persistent.detach().to(self.device)[healthy_mask],
                    )
                    or not torch.equal(
                        pending["apparent"][healthy_mask],
                        pre_apparent.detach().to(self.device)[healthy_mask],
                    )
                ):
                    raise ValueError(
                        "Expected the immediate fault transition to leave P0 "
                        "non-fault states bit-exact."
                    )
        healthy = ~pending_fault_mask
        if bool(
            torch.any(
                pending["persistent"][healthy]
                < self.population.logical_min[healthy]
            )
        ) or bool(
            torch.any(
                pending["persistent"][healthy]
                > self.population.logical_max[healthy]
            )
        ):
            raise ValueError("Expected saved programmable states to remain in support.")
        generator_state = state["generator_state"]
        if (
            not isinstance(generator_state, torch.Tensor)
            or generator_state.dtype != torch.uint8
            or generator_state.device.type != "cpu"
            or generator_state.ndim != 1
        ):
            raise ValueError("Expected a saved crossbar generator state tensor.")
        probe = torch.Generator(device=self.device.type)
        try:
            probe.set_state(generator_state.detach().cpu())
        except RuntimeError as error:
            raise ValueError("Expected a valid saved crossbar generator state.") from error

        self.persistent.copy_(pending["persistent"])
        self.apparent.copy_(pending["apparent"])
        self.upward_pulses.copy_(pending["upward_pulses"])
        self.downward_pulses.copy_(pending["downward_pulses"])
        self.post_deployment_fault_mask.copy_(pending_fault_mask)
        self.post_deployment_stuck_persistent_q.copy_(pending_fault_q)
        self.post_deployment_fault_pulse_baseline.copy_(pending_fault_baseline)
        self.fault_transition = pending_transition
        self.generator.set_state(generator_state.detach().cpu())

    def pulse_statistics(self) -> dict[str, Any]:
        total = self.upward_pulses + self.downward_pulses
        active = self.persistent + self.population.reference
        commanded = total > 0
        corrupt = self.population.corrupt
        post_fault = self.post_deployment_fault_mask
        post_transition_pulses = total - self.post_deployment_fault_pulse_baseline
        diagnostic_minimum = self.population.min_bound.clone()
        diagnostic_maximum = self.population.max_bound.clone()
        if bool(torch.any(post_fault)):
            collapsed_active = (
                self.post_deployment_stuck_persistent_q
                + self.population.reference
            )
            diagnostic_minimum[post_fault] = collapsed_active[post_fault]
            diagnostic_maximum[post_fault] = collapsed_active[post_fault]
        return {
            "total": int(total.sum().item()),
            "upward": int(self.upward_pulses.sum().item()),
            "downward": int(self.downward_pulses.sum().item()),
            # Retain the historical key, but make its command semantics explicit.
            "changed_cells": int(commanded.sum().item()),
            "commanded_cells": int(commanded.sum().item()),
            "persistent_moved_from_reset_cells": int(
                (~torch.isclose(
                    active,
                    self.population.min_bound,
                    atol=1e-7,
                    rtol=0.0,
                )).sum().item()
            ),
            "final_corrupt_cells": int(corrupt.sum().item()),
            "published_corrupt_cells": int(
                self.population.published_corrupt.sum().item()
            ),
            "commanded_final_corrupt_cells": int((commanded & corrupt).sum().item()),
            "pulses_to_final_corrupt_cells": int(total[corrupt].sum().item()),
            "post_deployment_fault_cells": int(post_fault.sum().item()),
            "commanded_post_deployment_fault_cells": int(
                ((post_transition_pulses > 0) & post_fault).sum().item()
            ),
            "pulses_to_post_deployment_fault_cells": int(
                post_transition_pulses[post_fault].sum().item()
            ),
            "maximum_per_cell": int(total.max().item()) if total.numel() else 0,
            "saturated_lower": int(
                torch.isclose(active, diagnostic_minimum, atol=1e-7, rtol=0.0).sum().item()
            ),
            "saturated_upper": int(
                torch.isclose(active, diagnostic_maximum, atol=1e-7, rtol=0.0).sum().item()
            ),
            "post_deployment_fault_bound_diagnostics": (
                "sampled_collapsed_active_bound_overlay"
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


class _LocalStarUpdatePort:
    """Restricted local-recovery view with no fault map or persistent state."""

    __slots__ = (
        "__size",
        "__nominal_dw_min",
        "__apparent_state",
        "__persistent_state",
        "__pulse_writer",
    )

    def __init__(self, plant: IbmOmEffectiveCrossbarPlant) -> None:
        self.__size = plant.size
        self.__nominal_dw_min = float(plant.population.nominal_dw_min)
        self.__apparent_state = plant.apparent
        self.__persistent_state = plant.persistent
        self.__pulse_writer = plant.pulse

    @property
    def size(self) -> int:
        return self.__size

    @property
    def nominal_dw_min(self) -> float:
        return self.__nominal_dw_min

    @property
    def apparent(self) -> torch.Tensor:
        return self.__apparent_state.detach().clone()

    def pulse(self, directions: torch.Tensor) -> None:
        self.__pulse_writer(directions)

    def state_hash_receipt(self) -> dict[str, str]:
        return {
            "apparent_sha256": tensor_sha256(self.__apparent_state),
            "persistent_sha256": tensor_sha256(self.__persistent_state),
        }


def _layout_receipt(
    layout: Sequence[CrossbarTileSpec],
) -> tuple[tuple[str, int, int, int, int, int], ...]:
    """Return the immutable scalar layout identity used by restricted ports."""

    return tuple(
        (
            tile.key,
            tile.layer_index,
            tile.tile_index,
            tile.input_start,
            tile.input_stop,
            tile.out_features,
        )
        for tile in layout
    )


class _OnChipCrossbarRecoveryPort:
    """Capability-limited slow-array view for label-supervised recovery."""

    __slots__ = (
        "__size",
        "__nominal_dw_min",
        "__layout",
        "__layout_receipt_value",
        "__digital_scales",
        "__device",
        "__apparent_state",
        "__persistent_state",
        "__grouped_pulse_writer",
    )

    def __init__(
        self,
        plant: IbmOmEffectiveCrossbarPlant,
        *,
        layout: Sequence[CrossbarTileSpec],
        digital_scales: Sequence[float],
    ) -> None:
        scales = tuple(float(value) for value in digital_scales)
        if len(scales) != 2 or any(
            not math.isfinite(value) or value <= 0.0 for value in scales
        ):
            raise ValueError("Expected two finite positive physical-q scales.")
        self.__size = plant.size
        self.__nominal_dw_min = float(plant.population.nominal_dw_min)
        self.__layout = tuple(layout)
        self.__layout_receipt_value = _layout_receipt(self.__layout)
        self.__digital_scales = scales
        self.__device = plant.device
        self.__apparent_state = plant.apparent
        self.__persistent_state = plant.persistent
        self.__grouped_pulse_writer = plant.apply_grouped_pulse_counts

    @property
    def size(self) -> int:
        return self.__size

    @property
    def nominal_dw_min(self) -> float:
        return self.__nominal_dw_min

    @property
    def q_scales(self) -> tuple[float, float]:
        return self.__digital_scales

    def hardware_contract(self) -> dict[str, Any]:
        return {
            "size": self.__size,
            "nominal_dw_min": self.__nominal_dw_min,
            "layout": self.__layout_receipt_value,
            "q_scales": self.__digital_scales,
            "device": str(self.__device),
            "forward_state": "apparent_q",
            "full_weight_tensor_exposed": False,
            "fault_mask_exposed": False,
        }

    @torch.no_grad()
    def forward(self, inputs: torch.Tensor) -> StandardCrossbarForwardStates:
        if inputs.ndim != 2 or inputs.device != self.__device or not bool(
            torch.all(torch.isfinite(inputs))
        ):
            raise ValueError(
                "Expected finite rank-2 inputs on the recovery-port device."
            )
        return standard_crossbar_forward_states(
            inputs,
            self.__apparent_state,
            self.__layout,
            digital_scales=self.__digital_scales,
        )

    @torch.no_grad()
    def output_transpose_mvm(self, output_errors: torch.Tensor) -> torch.Tensor:
        """Apply the apparent second-layer transpose without exporting W2."""

        if output_errors.ndim != 2 or output_errors.device != self.__device or not bool(
            torch.all(torch.isfinite(output_errors))
        ):
            raise ValueError(
                "Expected finite rank-2 output errors on the recovery-port device."
            )
        _, output_weight = effective_state_to_logical_weights(
            self.__apparent_state,
            self.__layout,
            digital_scales=self.__digital_scales,
        )
        if output_errors.shape[1] != output_weight.shape[1]:
            raise ValueError("Expected output errors to match the output crossbar.")
        return output_errors @ output_weight.transpose(0, 1)

    def apply_grouped_pulse_counts(
        self,
        positive_counts: torch.Tensor,
        negative_counts: torch.Tensor,
    ) -> None:
        self.__grouped_pulse_writer(positive_counts, negative_counts)

    def state_hash_receipt(self) -> dict[str, str]:
        return {
            "apparent_sha256": tensor_sha256(self.__apparent_state),
            "persistent_sha256": tensor_sha256(self.__persistent_state),
        }


class _TikiTakaFastArrayPort:
    """Restricted fast-array port exposing only selected transfer columns."""

    __slots__ = (
        "__size",
        "__nominal_dw_min",
        "__layout_receipt_value",
        "__tile_offsets",
        "__tile_shapes",
        "__device",
        "__apparent_state",
        "__persistent_state",
        "__grouped_pulse_writer",
    )

    def __init__(
        self,
        plant: IbmOmEffectiveCrossbarPlant,
        *,
        layout: Sequence[CrossbarTileSpec],
    ) -> None:
        tiles = tuple(layout)
        offsets = []
        offset = 0
        for tile in tiles:
            offsets.append(offset)
            offset += tile.cells
        self.__size = plant.size
        self.__nominal_dw_min = float(plant.population.nominal_dw_min)
        self.__layout_receipt_value = _layout_receipt(tiles)
        self.__tile_offsets = tuple(offsets)
        self.__tile_shapes = tuple(tile.shape for tile in tiles)
        self.__device = plant.device
        self.__apparent_state = plant.apparent
        self.__persistent_state = plant.persistent
        self.__grouped_pulse_writer = plant.apply_grouped_pulse_counts

    @property
    def size(self) -> int:
        return self.__size

    @property
    def nominal_dw_min(self) -> float:
        return self.__nominal_dw_min

    def hardware_contract(self) -> dict[str, Any]:
        return {
            "size": self.__size,
            "nominal_dw_min": self.__nominal_dw_min,
            "layout": self.__layout_receipt_value,
            "device": str(self.__device),
            "selected_slice_only": True,
            "full_weight_tensor_exposed": False,
            "fault_mask_exposed": False,
        }

    @torch.no_grad()
    def read_apparent_column(
        self,
        *,
        tile_index: int,
        column_index: int,
    ) -> torch.Tensor:
        """Read one physical input column in AIHWKit's ``[out,in]`` sense."""

        if (
            isinstance(tile_index, bool)
            or not isinstance(tile_index, int)
            or tile_index < 0
            or tile_index >= len(self.__tile_shapes)
        ):
            raise ValueError("Expected a valid physical Tiki-Taka tile index.")
        inputs, outputs = self.__tile_shapes[tile_index]
        if (
            isinstance(column_index, bool)
            or not isinstance(column_index, int)
            or column_index < 0
            or column_index >= inputs
        ):
            raise ValueError("Expected a valid physical input-column index.")
        start = self.__tile_offsets[tile_index] + column_index * outputs
        return self.__apparent_state[start : start + outputs].detach().clone()

    def apply_grouped_pulse_counts(
        self,
        positive_counts: torch.Tensor,
        negative_counts: torch.Tensor,
    ) -> None:
        self.__grouped_pulse_writer(positive_counts, negative_counts)

    def state_hash_receipt(self) -> dict[str, str]:
        return {
            "apparent_sha256": tensor_sha256(self.__apparent_state),
            "persistent_sha256": tensor_sha256(self.__persistent_state),
        }


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
            "commanded_pulses": applied,
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
            "commanded_pulses": self.applied,
            "applied_pulses": self.applied,
            "applied_pulses_by_layer": layer_counts,
            "probability_clipped": self.probability_clipped,
            "blocked_at_cap": self.blocked_at_cap,
            "pulse_cap_per_cell": self.pulse_cap_per_cell,
            "cells_at_cap": int(
                (self.pulse_count >= self.pulse_cap_per_cell).sum().item()
            ),
            "commanded_cells": int((self.pulse_count > 0).sum().item()),
            "changed_cells": int((self.pulse_count > 0).sum().item()),
            "maximum_pulses_per_cell": int(self.pulse_count.max().item()),
            "enabled_cells": int(self.enabled.sum().item()),
        }


class PulseSGD:
    """Moment-free local gradient to stochastic one-pulse conversion."""

    def __init__(
        self,
        *,
        size: int,
        layout: Sequence[CrossbarTileSpec],
        learning_rates: Sequence[float],
        layer_scope: str,
        nominal_dw_min: float,
        pulse_cap_per_cell: int,
        pulse_rule: str,
        generator: torch.Generator,
        device: torch.device | str,
    ) -> None:
        rates = tuple(float(value) for value in learning_rates)
        if (
            size < 1
            or len(rates) != 2
            or any(not math.isfinite(value) or value <= 0.0 for value in rates)
            or not math.isfinite(nominal_dw_min)
            or nominal_dw_min <= 0.0
            or isinstance(pulse_cap_per_cell, bool)
            or pulse_cap_per_cell < 1
            or layer_scope not in {"all", "input_only", "output_only"}
            or pulse_rule not in {"stochastic_pulse_sgd", "stochastic_pulse_sign_sgd"}
        ):
            raise ValueError("Expected a valid two-layer moment-free pulse-SGD configuration.")
        self.device = torch.device(device)
        self.generator = generator
        self.nominal_dw_min = float(nominal_dw_min)
        self.pulse_cap_per_cell = int(pulse_cap_per_cell)
        self.pulse_rule = pulse_rule
        self.step_index = 0
        slices = layer_cell_slices(layout)
        self._learning_rates = rates
        self.enabled = torch.zeros(size, dtype=torch.bool, device=self.device)
        if layer_scope in {"all", "input_only"}:
            self.enabled[slices[0]] = True
        if layer_scope in {"all", "output_only"}:
            self.enabled[slices[1]] = True
        self.requested = 0
        self.commanded = 0
        self.probability_clipped = 0
        self.blocked_at_cap = 0
        self.pulse_count = torch.zeros(size, dtype=torch.int64, device=self.device)
        self._layer_slices = slices

    def step(
        self,
        gradient: torch.Tensor,
        plant: IbmOmEffectiveCrossbarPlant | _LocalStarUpdatePort,
    ) -> dict[str, int | float]:
        value = gradient.detach().to(device=self.device, dtype=torch.float32).reshape(-1)
        if value.shape != self.pulse_count.shape or not bool(torch.all(torch.isfinite(value))):
            raise ValueError("Expected one finite local gradient per crosspoint.")
        self.step_index += 1
        magnitude = (
            value.abs()
            if self.pulse_rule == "stochastic_pulse_sgd"
            else (value != 0).to(torch.float32)
        )
        raw_probability = torch.zeros_like(magnitude)
        for layer_index, layer_slice in enumerate(self._layer_slices):
            raw_probability[layer_slice] = (
                self._learning_rates[layer_index]
                * magnitude[layer_slice]
                / self.nominal_dw_min
            )
        raw_probability = torch.where(
            self.enabled, raw_probability, torch.zeros_like(raw_probability)
        )
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
        direction = -torch.sign(value).to(torch.int8) * selected.to(torch.int8)
        requested = int((raw_probability > 0.0).sum().item())
        commanded = int(selected.sum().item())
        clipped_count = int(clipped.sum().item())
        blocked_count = int(blocked.sum().item())
        self.requested += requested
        self.commanded += commanded
        self.probability_clipped += clipped_count
        self.blocked_at_cap += blocked_count
        self.pulse_count += selected.to(torch.int64)
        plant.pulse(direction)
        return {
            "requested_nonzero_commands": requested,
            "commanded_pulses": commanded,
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
            "optimizer": "moment_free_pulse_sgd",
            "pulse_rule": self.pulse_rule,
            "learning_rates_q": list(self._learning_rates),
            "nominal_dw_min": self.nominal_dw_min,
            "command_direction": "-sign(gradient_q)",
            "selection_probability": (
                "min(1,learning_rate_q*abs(gradient_q)/nominal_dw_min)"
            ),
            "optimizer_steps": self.step_index,
            "requested_nonzero_commands": self.requested,
            "commanded_pulses": self.commanded,
            "commanded_pulses_by_layer": layer_counts,
            "probability_clipped": self.probability_clipped,
            "blocked_at_cap": self.blocked_at_cap,
            "pulse_cap_per_cell": self.pulse_cap_per_cell,
            "cells_at_cap": int(
                (self.pulse_count >= self.pulse_cap_per_cell).sum().item()
            ),
            "commanded_cells": int((self.pulse_count > 0).sum().item()),
            "maximum_pulses_per_cell": int(self.pulse_count.max().item()),
            "enabled_cells": int(self.enabled.sum().item()),
            "digital_first_moment_values": 0,
            "digital_second_moment_values": 0,
            "fp32_shadow_weight_values": 0,
            "persistent_fp32_update_state_values": 0,
            "persistent_per_cell_state": "integer_endurance_counter_only",
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
    "LocalStarCrossbarStep",
    "PulseAdam",
    "PulseSGD",
    "StandardCrossbarForwardStates",
    "apply_population_bound_policy",
    "build_crossbar_layout",
    "build_deterministic_effective_codebook",
    "effective_state_to_logical_weights",
    "flatten_logical_crossbar_matrices",
    "layer_cell_slices",
    "local_star_crossbar_step",
    "map_logical_weights",
    "project_to_nearest_effective_code",
    "project_to_nearest_effective_code_device",
    "standard_crossbar_logits",
    "standard_crossbar_forward_states",
    "tensor_sha256",
    "validate_population_layout",
]
