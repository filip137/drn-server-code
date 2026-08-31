"""Persistent pulse-mediated Adam primitives for Winsorized IBM OM arrays.

This module deliberately gives the optimizer no authoritative floating-point
weight.  The persistent raw-active plant is the only weight state.  A DRN
tensor is an ephemeral differentiable mirror, refreshed from persistent
``G=2*x`` before each forward, and every update is a physical pulse request
through the capability-limited controller port.

The Adam moments and pulse-selection RNG remain digital controller state.
Consequently the scientifically accurate label is *hardware-in-loop
pulse-mediated Adam*, not fully on-chip Adam.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Literal

import torch

from experiments.mnist_relu_drn.ibm_om_baseline_selection import (
    quad_stack,
    scatter_quads,
)
from experiments.mnist_relu_drn.ibm_om_baseline_spacing_pv import (
    apply_full_conductance_targets,
)
from training.checkpoint import atomic_torch_save
from training.ibm_reram_program_verify import (
    IbmReramPopulation,
    IbmReramRawActivePlant,
    VerifyPort,
)


P0_SCHEMA = "ebl.mnist_relu_drn.ibm_om_winsorized_onchip_adam_p0"
P0_SCHEMA_VERSION = 1
RECOVERY_SCHEMA = (
    "ebl.mnist_relu_drn.ibm_om_winsorized_onchip_adam_recovery_state"
)
RECOVERY_SCHEMA_VERSION = 1
EVIDENCE_TIER = "exploratory_noncanonical"
CLAIM_LABEL = "hardware-in-loop pulse-mediated Adam"
STATE_AUTHORITY = "persistent_raw_active_plant_only"
TECHNIQUES = ("direct_physical_rail", "contrast_constrained")
LAYOUTS = ("halves", "paired")

_P0_FIELDS = {
    "schema",
    "schema_version",
    "evidence_tier",
    "claim_label",
    "state_authority",
    "assignment_seed",
    "endpoint_seed",
    "spacing_delta_x_multiplier",
    "qat_checkpoint_sha256",
    "teacher_sha256",
    "population_sha256",
    "population_fingerprint",
    "mapping_sha256",
    "joint_assignment_sha256",
    "commissioning_sha256",
    "binding_keys",
    "binding_shapes",
    "maximum_program_pulses",
    "recovery_pulse_cap",
    "maximum_random_draws",
    "pulse_selection_seed",
    "target_raw_x",
    "requested_index",
    "logical_master_sha256",
    "programming",
    "continuation_state",
    "validation",
    "validation_prediction_sha256",
    "no_remap_after_p0",
    "apparent_endpoint_applied_to_drn",
}


def _clone_tree(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    if isinstance(value, Mapping):
        return {str(key): _clone_tree(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_clone_tree(item) for item in value)
    if isinstance(value, list):
        return [_clone_tree(item) for item in value]
    return deepcopy(value)


def _finite_float_tensor(
    value: Any,
    *,
    name: str,
    shape: tuple[int, ...] | None = None,
) -> torch.Tensor:
    if (
        not isinstance(value, torch.Tensor)
        or value.dtype != torch.float32
        or (shape is not None and tuple(value.shape) != shape)
        or not bool(torch.all(torch.isfinite(value)))
    ):
        raise ValueError(
            f"Expected {name} to be a finite float32 tensor"
            + (f" of shape {shape!r}." if shape is not None else ".")
        )
    return value


def validate_p0_bundle(payload: Mapping[str, Any]) -> None:
    """Validate the exact, weights-only-safe P0 checkpoint contract."""

    if not isinstance(payload, Mapping) or set(payload) != _P0_FIELDS:
        raise ValueError(
            "Expected an exact Winsorized on-chip Adam P0 bundle; "
            f"fields={sorted(payload) if isinstance(payload, Mapping) else None!r}."
        )
    if (
        payload["schema"] != P0_SCHEMA
        or payload["schema_version"] != P0_SCHEMA_VERSION
        or payload["evidence_tier"] != EVIDENCE_TIER
        or payload["claim_label"] != CLAIM_LABEL
        or payload["state_authority"] != STATE_AUTHORITY
        or payload["no_remap_after_p0"] is not True
        or payload["apparent_endpoint_applied_to_drn"] is not False
    ):
        raise ValueError("P0 bundle semantic contract mismatch.")
    for name in (
        "assignment_seed",
        "endpoint_seed",
        "spacing_delta_x_multiplier",
        "maximum_program_pulses",
        "recovery_pulse_cap",
        "maximum_random_draws",
        "pulse_selection_seed",
    ):
        value = payload[name]
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"Expected positive integer P0 field {name!r}.")
    for name in (
        "qat_checkpoint_sha256",
        "teacher_sha256",
        "population_sha256",
        "population_fingerprint",
        "mapping_sha256",
        "joint_assignment_sha256",
        "commissioning_sha256",
        "validation_prediction_sha256",
    ):
        value = payload[name]
        if not isinstance(value, str) or len(value) != 64:
            raise ValueError(f"Expected a SHA-256 string in P0 field {name!r}.")
    keys = payload["binding_keys"]
    shapes = payload["binding_shapes"]
    if (
        not isinstance(keys, (tuple, list))
        or not isinstance(shapes, (tuple, list))
        or len(keys) != len(shapes)
        or len(keys) != 2
        or any(not isinstance(key, str) or not key for key in keys)
        or any(
            not isinstance(shape, (tuple, list))
            or len(shape) != 2
            or any(
                isinstance(item, bool) or not isinstance(item, int) or item < 1
                for item in shape
            )
            for shape in shapes
        )
    ):
        raise ValueError("Expected two named rank-2 physical P0 bindings.")
    size = sum(math.prod(tuple(shape)) for shape in shapes)
    _finite_float_tensor(payload["target_raw_x"], name="P0 target_raw_x", shape=(size,))
    requested = payload["requested_index"]
    if (
        not isinstance(requested, torch.Tensor)
        or requested.dtype != torch.int64
        or tuple(requested.shape) != (size,)
        or bool(torch.any(requested < 0))
    ):
        raise ValueError("Expected nonnegative int64 P0 requested_index.")
    logical_hashes = payload["logical_master_sha256"]
    if (
        not isinstance(logical_hashes, (tuple, list))
        or len(logical_hashes) != 2
        or any(not isinstance(item, str) or len(item) != 64 for item in logical_hashes)
    ):
        raise ValueError("Expected two logical-master SHA-256 values in P0.")
    if not isinstance(payload["programming"], Mapping):
        raise ValueError("Expected P0 programming diagnostics.")
    if not isinstance(payload["validation"], Mapping):
        raise ValueError("Expected P0 validation diagnostics.")
    state = payload["continuation_state"]
    if not isinstance(state, Mapping):
        raise ValueError("Expected one exact raw-active continuation state in P0.")
    expected_state_fields = {
        "schema_version",
        "preset",
        "construction_seeds",
        "persistent",
        "apparent",
        "rng_backend",
        "seeds",
        "maximum_random_draws",
        "draw_indices",
        "state_coordinate",
    }
    if set(state) != expected_state_fields:
        raise ValueError("Expected exact CUDA raw-active continuation-state fields.")
    if (
        state["schema_version"] != 2
        or state["state_coordinate"]
        != IbmReramRawActivePlant.STATE_COORDINATE
        or state["maximum_random_draws"] != payload["maximum_random_draws"]
        or state["rng_backend"] != "per_trajectory_buffered_torch_cpu"
    ):
        raise ValueError("P0 raw-active continuation-state contract mismatch.")
    construction = state["construction_seeds"]
    draw_indices = state["draw_indices"]
    for name in ("persistent", "apparent"):
        _finite_float_tensor(state[name], name=f"P0 continuation {name}", shape=(size,))
    if (
        not isinstance(construction, torch.Tensor)
        or construction.dtype != torch.int64
        or tuple(construction.shape) != (size,)
        or not isinstance(draw_indices, torch.Tensor)
        or draw_indices.dtype != torch.int64
        or tuple(draw_indices.shape) != (size,)
        or bool(torch.any(draw_indices < 0))
        or bool(torch.any(draw_indices > payload["maximum_random_draws"]))
        or not isinstance(state["seeds"], (tuple, list))
        or len(state["seeds"]) != size
    ):
        raise ValueError("Invalid P0 continuation identity or RNG tensors.")


def save_p0_bundle(payload: Mapping[str, Any], path: Path | str) -> Path:
    """Atomically save an exact P0 bundle after validation."""

    clean = _clone_tree(payload)
    validate_p0_bundle(clean)
    return atomic_torch_save(clean, Path(path))


def load_p0_bundle(path: Path | str) -> dict[str, Any]:
    """Load a P0 without arbitrary pickle objects and return an owned clone."""

    source = Path(path).expanduser().resolve()
    try:
        payload = torch.load(source, map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        raise ValueError(f"Expected a readable weights-only P0 bundle: {source}.") from error
    if not isinstance(payload, Mapping):
        raise ValueError("Expected a mapping-valued P0 bundle.")
    validate_p0_bundle(payload)
    return _clone_tree(payload)


def clone_p0_bundle(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Create a byte-value-identical, storage-independent arm seed."""

    validate_p0_bundle(payload)
    clone = _clone_tree(payload)
    validate_p0_bundle(clone)
    return clone


def plant_from_p0(
    population: IbmReramPopulation,
    payload: Mapping[str, Any],
    *,
    device: torch.device | str,
) -> IbmReramRawActivePlant:
    """Restore one exact persistent plant; never remap an arm from targets."""

    validate_p0_bundle(payload)
    state = payload["continuation_state"]
    assert isinstance(state, Mapping)
    if population.size != int(payload["target_raw_x"].numel()):
        raise ValueError("P0 and pulse population sizes differ.")
    construction = state["construction_seeds"]
    if not isinstance(construction, torch.Tensor) or not torch.equal(
        construction.detach().cpu(), population.construction_seeds.detach().cpu()
    ):
        raise ValueError("P0 names a different physical-cell population.")
    seeds = state["seeds"]
    assert isinstance(seeds, (tuple, list))
    plant = IbmReramRawActivePlant(
        population,
        seeds=tuple(int(seed) for seed in seeds),
        device=device,
        maximum_random_draws=int(payload["maximum_random_draws"]),
    )
    plant.load_state_dict(state)
    if not torch.equal(
        plant.persistent.detach().cpu(), state["persistent"].detach().cpu()
    ) or not torch.equal(
        plant.apparent.detach().cpu(), state["apparent"].detach().cpu()
    ):
        raise RuntimeError("Exact P0 plant restoration changed endpoint state.")
    return plant


def split_flat_physical(
    value: torch.Tensor, shapes: Sequence[Sequence[int]]
) -> tuple[torch.Tensor, ...]:
    """Split one canonical binding-order cell vector into physical matrices."""

    flat = torch.as_tensor(value).reshape(-1)
    result = []
    offset = 0
    for shape_value in shapes:
        shape = tuple(int(item) for item in shape_value)
        count = math.prod(shape)
        result.append(flat[offset : offset + count].reshape(shape))
        offset += count
    if offset != flat.numel():
        raise ValueError("Flat physical state length does not match binding shapes.")
    return tuple(result)


def flatten_physical(values: Sequence[torch.Tensor]) -> torch.Tensor:
    selected = tuple(values)
    if not selected:
        raise ValueError("Expected at least one physical tensor.")
    return torch.cat(tuple(value.reshape(-1) for value in selected))


def sync_catalog_from_persistent(
    catalog: Any,
    plant: IbmReramRawActivePlant,
    *,
    binding_shapes: Sequence[Sequence[int]],
) -> None:
    """Refresh the ephemeral DRN mirror from persistent ``G=2*x`` only."""

    raw_x = (plant.persistent + 1.0) / 2.0
    if bool(torch.any(raw_x < 0.0)) or bool(torch.any(raw_x > 1.0)):
        raise RuntimeError("Persistent Winsorized raw x left [0,1].")
    full_g = split_flat_physical(2.0 * raw_x, binding_shapes)
    apply_full_conductance_targets(
        catalog,
        full_g,
        conductance_min=0.0,
        conductance_max=2.0,
    )


def contrast_project_physical_gradients(
    gradients: Sequence[torch.Tensor],
    *,
    layouts: Sequence[str] = LAYOUTS,
) -> tuple[torch.Tensor, ...]:
    """Project physical gradients onto ``(+,-,-,+)`` quad contrast."""

    selected = tuple(gradients)
    chosen_layouts = tuple(layouts)
    if len(selected) != len(chosen_layouts):
        raise ValueError("Expected one quad layout per physical gradient.")
    sign = None
    projected = []
    for gradient, layout in zip(selected, chosen_layouts):
        quads = quad_stack(gradient, layout=layout)
        if sign is None or sign.device != quads.device or sign.dtype != quads.dtype:
            sign = torch.tensor(
                (1.0, -1.0, -1.0, 1.0),
                dtype=quads.dtype,
                device=quads.device,
            )
        coefficient = (quads * sign).sum(dim=-1, keepdim=True) / 4.0
        projected.append(
            scatter_quads(
                coefficient * sign,
                shape=gradient.shape,
                layout=layout,
            )
        )
    return tuple(projected)


def contrast_coefficients(
    gradients: Sequence[torch.Tensor],
    *,
    layouts: Sequence[str] = LAYOUTS,
) -> tuple[torch.Tensor, ...]:
    """Reduce each quad to its mean signed contrast-gradient coefficient."""

    selected = tuple(gradients)
    if len(selected) != len(tuple(layouts)):
        raise ValueError("Expected one quad layout per physical gradient.")
    result = []
    for gradient, layout in zip(selected, tuple(layouts)):
        quads = quad_stack(gradient, layout=layout)
        sign = torch.tensor(
            (1.0, -1.0, -1.0, 1.0),
            dtype=quads.dtype,
            device=quads.device,
        )
        result.append((quads * sign).sum(dim=-1) / 4.0)
    return tuple(result)


def lift_contrast_coefficients(
    coefficients: Sequence[torch.Tensor],
    *,
    shapes: Sequence[Sequence[int]],
    layouts: Sequence[str] = LAYOUTS,
) -> tuple[torch.Tensor, ...]:
    """Lift scalar quad commands onto exactly signed four-cell rails."""

    if not (len(coefficients) == len(shapes) == len(tuple(layouts))):
        raise ValueError("Expected matched coefficient, shape, and layout sequences.")
    result = []
    for coefficient, shape, layout in zip(coefficients, shapes, tuple(layouts)):
        sign = torch.tensor(
            (1.0, -1.0, -1.0, 1.0),
            dtype=coefficient.dtype,
            device=coefficient.device,
        )
        result.append(
            scatter_quads(
                coefficient.unsqueeze(-1) * sign,
                shape=tuple(shape),
                layout=layout,
            )
        )
    return tuple(result)


@dataclass(frozen=True)
class PulseAdamStep:
    step: int
    requested_cells: int
    pulsed_cells: int
    capped_cells: int
    upward_pulses: int
    downward_pulses: int
    mean_probability: float
    maximum_probability: float


class PersistentPulseAdam:
    """Digital Adam moments translated directly into persistent cell pulses."""

    def __init__(
        self,
        port: VerifyPort,
        *,
        device: torch.device | str,
        binding_shapes: Sequence[Sequence[int]],
        technique: Literal["direct_physical_rail", "contrast_constrained"],
        learning_rate_raw_x: float,
        nominal_delta_x: float,
        pulse_cap: int,
        pulse_selection_seed: int,
        layouts: Sequence[str] = LAYOUTS,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
    ) -> None:
        if technique not in TECHNIQUES:
            raise ValueError(f"Expected technique in {TECHNIQUES!r}.")
        shapes = tuple(tuple(int(item) for item in shape) for shape in binding_shapes)
        chosen_layouts = tuple(layouts)
        if len(shapes) != 2 or len(chosen_layouts) != len(shapes):
            raise ValueError("Expected two physical binding shapes and layouts.")
        if sum(math.prod(shape) for shape in shapes) != port.size:
            raise ValueError("Adam binding shapes do not cover the controller port.")
        scalars = (learning_rate_raw_x, nominal_delta_x, beta1, beta2, epsilon)
        if any(not math.isfinite(float(value)) for value in scalars):
            raise ValueError("Expected finite pulse-Adam scalar settings.")
        if learning_rate_raw_x <= 0.0 or nominal_delta_x <= 0.0 or epsilon <= 0.0:
            raise ValueError("Expected positive pulse-Adam scale settings.")
        if not (0.0 <= beta1 < 1.0 and 0.0 <= beta2 < 1.0):
            raise ValueError("Expected Adam beta values in [0,1).")
        if isinstance(pulse_cap, bool) or not isinstance(pulse_cap, int) or pulse_cap < 1:
            raise ValueError("Expected a positive per-cell recovery pulse cap.")
        self._port = port
        self.device = torch.device(device)
        if self.device.type not in {"cpu", "cuda"}:
            raise ValueError("Expected a CPU or CUDA pulse-controller device.")
        self.binding_shapes = shapes
        self.layouts = chosen_layouts
        self.technique = technique
        self.learning_rate_raw_x = float(learning_rate_raw_x)
        self.nominal_delta_x = float(nominal_delta_x)
        self.pulse_cap = int(pulse_cap)
        self.beta1 = float(beta1)
        self.beta2 = float(beta2)
        self.epsilon = float(epsilon)
        self.step_index = 0
        moment_size = (
            port.size
            if technique == "direct_physical_rail"
            else sum(math.prod(shape) // 4 for shape in shapes)
        )
        self.first_moment = torch.zeros(
            moment_size, dtype=torch.float32, device=self.device
        )
        self.second_moment = torch.zeros_like(self.first_moment)
        self.pulse_count = torch.zeros(
            port.size, dtype=torch.int64, device=self.device
        )
        self.selection_generator = torch.Generator(device=self.device)
        self.selection_generator.manual_seed(int(pulse_selection_seed))

    def _adam_command(self, gradient: torch.Tensor) -> torch.Tensor:
        if gradient.shape != self.first_moment.shape or not bool(
            torch.all(torch.isfinite(gradient))
        ):
            raise ValueError("Expected finite pulse-Adam gradients of fixed shape.")
        self.step_index += 1
        self.first_moment.mul_(self.beta1).add_(gradient, alpha=1.0 - self.beta1)
        self.second_moment.mul_(self.beta2).addcmul_(
            gradient, gradient, value=1.0 - self.beta2
        )
        first_hat = self.first_moment / (1.0 - self.beta1**self.step_index)
        second_hat = self.second_moment / (1.0 - self.beta2**self.step_index)
        return -self.learning_rate_raw_x * first_hat / (
            second_hat.sqrt() + self.epsilon
        )

    def step(self, physical_conductance_gradients: Sequence[torch.Tensor]) -> PulseAdamStep:
        """Apply at most one physical pulse per eligible cell for one minibatch."""

        gradients_g = tuple(physical_conductance_gradients)
        if len(gradients_g) != len(self.binding_shapes):
            raise ValueError("Expected one physical conductance gradient per binding.")
        for gradient, shape in zip(gradients_g, self.binding_shapes):
            if tuple(gradient.shape) != shape or not bool(torch.all(torch.isfinite(gradient))):
                raise ValueError("Expected finite physical conductance gradients.")
        # Winsorized full conductance is G=2*x, hence dL/dx=2*dL/dG.
        gradients_x = tuple(2.0 * value.detach() for value in gradients_g)
        coefficient_command = None
        if self.technique == "direct_physical_rail":
            command = self._adam_command(flatten_physical(gradients_x))
        else:
            coefficients = contrast_coefficients(gradients_x, layouts=self.layouts)
            coefficient_command = self._adam_command(flatten_physical(coefficients))
            split_coefficients = []
            offset = 0
            for shape in self.binding_shapes:
                logical_shape = (shape[0] // 2, shape[1] // 2)
                count = math.prod(logical_shape)
                split_coefficients.append(
                    coefficient_command[offset : offset + count].reshape(logical_shape)
                )
                offset += count
            command = flatten_physical(
                lift_contrast_coefficients(
                    split_coefficients,
                    shapes=self.binding_shapes,
                    layouts=self.layouts,
                )
            )
        if self.technique == "direct_physical_rail":
            probability = (command.abs() / self.nominal_delta_x).clamp(max=1.0)
            random_value = torch.rand(
                probability.shape,
                dtype=probability.dtype,
                device=probability.device,
                generator=self.selection_generator,
            )
            requested = (random_value < probability) & (command != 0.0)
            eligible = self.pulse_count < self.pulse_cap
            selected = requested & eligible
            directions = torch.zeros(
                self._port.size, dtype=torch.int8, device=self.device
            )
            directions[selected & (command > 0.0)] = 1
            directions[selected & (command < 0.0)] = -1
            counts = selected.to(torch.int64)
            requested_cells = int(requested.sum().item())
            capped_cells = int((requested & ~eligible).sum().item())
        else:
            assert coefficient_command is not None
            probability_quad = (
                coefficient_command.abs() / self.nominal_delta_x
            ).clamp(max=1.0)
            random_value = torch.rand(
                probability_quad.shape,
                dtype=probability_quad.dtype,
                device=probability_quad.device,
                generator=self.selection_generator,
            )
            requested_quad = (
                (random_value < probability_quad) & (coefficient_command != 0.0)
            )
            eligible_quad_parts = []
            offset = 0
            for shape, layout in zip(self.binding_shapes, self.layouts):
                count = math.prod(shape)
                cell_eligible = (self.pulse_count[offset : offset + count] < self.pulse_cap)
                cell_eligible = cell_eligible.reshape(shape)
                eligible_quad_parts.append(
                    quad_stack(cell_eligible, layout=layout).all(dim=-1).reshape(-1)
                )
                offset += count
            eligible_quad = torch.cat(eligible_quad_parts)
            selected_quad = requested_quad & eligible_quad
            selected_coefficients = []
            selected_counts = []
            offset = 0
            for shape in self.binding_shapes:
                logical_shape = (shape[0] // 2, shape[1] // 2)
                count = math.prod(logical_shape)
                selected_coefficients.append(
                    torch.sign(coefficient_command[offset : offset + count])
                    .reshape(logical_shape)
                    * selected_quad[offset : offset + count].reshape(logical_shape)
                )
                selected_counts.append(
                    selected_quad[offset : offset + count]
                    .reshape(logical_shape)
                    .to(torch.float32)
                )
                offset += count
            directions = flatten_physical(
                lift_contrast_coefficients(
                    selected_coefficients,
                    shapes=self.binding_shapes,
                    layouts=self.layouts,
                )
            ).to(torch.int8)
            counts = flatten_physical(
                lift_contrast_coefficients(
                    selected_counts,
                    shapes=self.binding_shapes,
                    layouts=self.layouts,
                )
            ).abs().to(torch.int64)
            probability = probability_quad
            requested_cells = 4 * int(requested_quad.sum().item())
            capped_cells = 4 * int((requested_quad & ~eligible_quad).sum().item())
        # This capability-limited call is the sole weight mutation.
        self._port.apply_identical_pulses(directions, counts)
        self.pulse_count.add_(counts)
        return PulseAdamStep(
            step=self.step_index,
            requested_cells=requested_cells,
            pulsed_cells=int(counts.sum().item()),
            capped_cells=capped_cells,
            upward_pulses=int((directions > 0).sum().item()),
            downward_pulses=int((directions < 0).sum().item()),
            mean_probability=float(probability.mean().item()),
            maximum_probability=float(probability.max().item()),
        )

    def state_dict(self) -> dict[str, Any]:
        return {
            "schema": RECOVERY_SCHEMA,
            "schema_version": RECOVERY_SCHEMA_VERSION,
            "claim_label": CLAIM_LABEL,
            "state_authority": STATE_AUTHORITY,
            "technique": self.technique,
            "binding_shapes": [list(shape) for shape in self.binding_shapes],
            "layouts": list(self.layouts),
            "learning_rate_raw_x": self.learning_rate_raw_x,
            "nominal_delta_x": self.nominal_delta_x,
            "pulse_cap": self.pulse_cap,
            "beta1": self.beta1,
            "beta2": self.beta2,
            "epsilon": self.epsilon,
            "step": self.step_index,
            "first_moment": self.first_moment.detach().cpu().clone(),
            "second_moment": self.second_moment.detach().cpu().clone(),
            "pulse_count": self.pulse_count.detach().cpu().clone(),
            "pulse_selection_rng_state": self.selection_generator.get_state().cpu().clone(),
            "authoritative_weight_shadow": None,
        }


__all__ = [
    "CLAIM_LABEL",
    "EVIDENCE_TIER",
    "LAYOUTS",
    "P0_SCHEMA",
    "P0_SCHEMA_VERSION",
    "PersistentPulseAdam",
    "PulseAdamStep",
    "RECOVERY_SCHEMA",
    "RECOVERY_SCHEMA_VERSION",
    "STATE_AUTHORITY",
    "TECHNIQUES",
    "clone_p0_bundle",
    "contrast_coefficients",
    "contrast_project_physical_gradients",
    "flatten_physical",
    "lift_contrast_coefficients",
    "load_p0_bundle",
    "plant_from_p0",
    "save_p0_bundle",
    "split_flat_physical",
    "sync_catalog_from_persistent",
    "validate_p0_bundle",
]
