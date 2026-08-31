"""Fresh-array transfer and column-serial open-loop pulse primitives.

The transfer object is the signed four-rail contrast learned in the exact
assignment-87003 persistent recovery state.  A target array supplies its own
shared-destination baselines.  Requested contrast is never clipped to the
target cell support; unsupported requests remain visible to P&V as saturation
or pulse-budget exhaustion.

Fine-tuning uses digital Adam moments.  Each dense per-cell Bernoulli update
is expressed as one deterministic physical-column line plus independently
sampled signed row lines for every column.  The 100+20 disjoint column phases
are evaluated in one vectorized plant call.  This is distribution- and
state-equivalent for the independent cell plant, but it is an emulator rather
than IBM's fully parallel outer-product/Tiki-Taka update circuit.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math
from typing import Any, Mapping, Protocol, Sequence

import torch

from experiments.artifacts import sha256_file
from experiments.mnist_relu_drn.ibm_om_baseline_selection import (
    quad_stack,
    scatter_quads,
)
from experiments.mnist_relu_drn.ibm_om_winsorized_onchip_adam import (
    RECOVERY_SCHEMA,
    RECOVERY_SCHEMA_VERSION,
    flatten_physical,
    split_flat_physical,
)
from training.ibm_reram_program_verify import IbmReramRawActivePlant


EVIDENCE_TIER = "exploratory_noncanonical"
CLAIM_LABEL = (
    "digital-Adam, column-serial open-loop stochastic-coincidence emulator"
)
UPDATE_SURFACE = "column_serial_open_loop_stochastic_coincidence_emulator"
LAYOUTS = ("halves", "paired")


@dataclass(frozen=True)
class CrossArrayTransfer:
    source_full_conductance: tuple[torch.Tensor, ...]
    source_signed_contrast_full_g: tuple[torch.Tensor, ...]
    target_raw_x: tuple[torch.Tensor, ...]
    target_full_conductance: tuple[torch.Tensor, ...]
    unsupported_logical_weight: tuple[torch.Tensor, ...]
    unsupported_cell: tuple[torch.Tensor, ...]
    report: Mapping[str, Any]


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
            + (f" with shape {shape!r}." if shape is not None else ".")
        )
    return value


def load_selected_source_recovery(
    path: Path | str,
    *,
    expected_sha256: str,
    expected_learning_rate_raw_x: float,
) -> dict[str, Any]:
    """Load the one hash-pinned 87003 recovery state used for transfer."""

    source = Path(path).expanduser().resolve()
    if sha256_file(source) != expected_sha256:
        raise ValueError("Selected source recovery checkpoint SHA-256 mismatch.")
    try:
        payload = torch.load(source, map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        raise ValueError("Expected a readable weights-only recovery checkpoint.") from error
    required = {
        "schema",
        "schema_version",
        "technique",
        "learning_rate_raw_x",
        "binding_shapes",
        "layouts",
        "plant_continuation_state",
        "state_authority",
        "authoritative_weight_shadow",
        "no_remap_after_p0",
        "apparent_endpoint_applied_to_drn",
    }
    if not isinstance(payload, Mapping) or not required.issubset(payload):
        raise ValueError("Selected source recovery checkpoint fields are incomplete.")
    if (
        payload["schema"] != RECOVERY_SCHEMA
        or payload["schema_version"] != RECOVERY_SCHEMA_VERSION
        or payload["technique"] != "direct_physical_rail"
        or float(payload["learning_rate_raw_x"])
        != float(expected_learning_rate_raw_x)
        or payload["layouts"] != list(LAYOUTS)
        or payload["authoritative_weight_shadow"] is not None
        or payload["no_remap_after_p0"] is not True
        or payload["apparent_endpoint_applied_to_drn"] is not False
    ):
        raise ValueError("Selected source recovery semantic contract mismatch.")
    shapes = payload["binding_shapes"]
    if (
        not isinstance(shapes, list)
        or len(shapes) != 2
        or any(
            not isinstance(shape, list)
            or len(shape) != 2
            or any(isinstance(item, bool) or not isinstance(item, int) or item < 1 for item in shape)
            for shape in shapes
        )
    ):
        raise ValueError("Expected two rank-2 source physical bindings.")
    state = payload["plant_continuation_state"]
    if not isinstance(state, Mapping) or state.get("state_coordinate") != (
        IbmReramRawActivePlant.STATE_COORDINATE
    ):
        raise ValueError("Expected a raw-active source continuation state.")
    size = sum(math.prod(shape) for shape in shapes)
    _finite_float_tensor(
        state.get("persistent"), name="source persistent state", shape=(size,)
    )
    return {str(key): value for key, value in payload.items()}


def source_full_g_and_signed_contrast(
    source_payload: Mapping[str, Any],
) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
    """Extract full ``G=a+1`` and ``(G++-G+--G-++G--)/2`` per quad."""

    shapes = tuple(tuple(int(item) for item in shape) for shape in source_payload["binding_shapes"])
    persistent = source_payload["plant_continuation_state"]["persistent"]
    _finite_float_tensor(
        persistent,
        name="source persistent state",
        shape=(sum(math.prod(shape) for shape in shapes),),
    )
    full = split_flat_physical(persistent.detach().cpu() + 1.0, shapes)
    contrasts = []
    for value, layout in zip(full, LAYOUTS):
        quads = quad_stack(value, layout=layout)
        contrasts.append(
            (quads[..., 0] - quads[..., 1] - quads[..., 2] + quads[..., 3])
            / 2.0
        )
    return tuple(full), tuple(contrasts)


def _summary(value: torch.Tensor) -> Mapping[str, float]:
    flat = value.detach().to(device="cpu", dtype=torch.float64).reshape(-1)
    return {
        "minimum": float(flat.min().item()),
        "mean": float(flat.mean().item()),
        "rms": float(flat.square().mean().sqrt().item()),
        "maximum": float(flat.max().item()),
    }


def build_target_baseline_contrast_transfer(
    source_payload: Mapping[str, Any],
    *,
    target_baseline_raw_x: Sequence[torch.Tensor],
    target_lower_raw_x: Sequence[torch.Tensor],
    target_upper_raw_x: Sequence[torch.Tensor],
) -> CrossArrayTransfer:
    """Place the unmodified source contrast over target-owned baselines.

    No requested target is clipped.  The unsupported masks are diagnostics
    consumed later by the stochastic P&V deployment.
    """

    baselines = tuple(target_baseline_raw_x)
    lowers = tuple(target_lower_raw_x)
    uppers = tuple(target_upper_raw_x)
    if not (len(baselines) == len(lowers) == len(uppers) == 2):
        raise ValueError("Expected two target physical layers.")
    source_full, contrasts = source_full_g_and_signed_contrast(source_payload)
    target_x = []
    target_g = []
    unsupported_weights = []
    unsupported_cells = []
    layers = []
    for layer_index, (baseline_value, lower_value, upper_value, contrast, layout) in enumerate(
        zip(baselines, lowers, uppers, contrasts, LAYOUTS)
    ):
        baseline = torch.as_tensor(baseline_value).detach().cpu().to(torch.float32)
        lower = torch.as_tensor(lower_value).detach().cpu().to(torch.float32)
        upper = torch.as_tensor(upper_value).detach().cpu().to(torch.float32)
        if (
            not (baseline.shape == lower.shape == upper.shape)
            or baseline.numel() < 1
            or not all(bool(torch.all(torch.isfinite(value))) for value in (baseline, lower, upper))
            or bool(torch.any(upper < lower))
            or bool(torch.any(baseline < lower - 1e-6))
            or bool(torch.any(baseline > upper + 1e-6))
        ):
            raise ValueError("Expected bounded finite target baseline tensors.")
        if tuple(contrast.shape) != (baseline.shape[0] // 2, baseline.shape[1] // 2):
            raise ValueError("Source contrast and target physical shape mismatch.")
        positive = contrast >= 0.0
        offset_quad = torch.zeros(
            (*contrast.shape, 4), dtype=torch.float32
        )
        raw_offset = contrast.abs().to(torch.float32) / 2.0
        offset_quad[..., 0] = torch.where(positive, raw_offset, 0.0)
        offset_quad[..., 3] = torch.where(positive, raw_offset, 0.0)
        offset_quad[..., 1] = torch.where(positive, 0.0, raw_offset)
        offset_quad[..., 2] = torch.where(positive, 0.0, raw_offset)
        offset = scatter_quads(offset_quad, shape=baseline.shape, layout=layout)
        requested_x = baseline + offset
        cell_bad = (requested_x < lower - 1e-6) | (requested_x > upper + 1e-6)
        logical_bad = quad_stack(cell_bad, layout=layout).any(dim=-1)
        requested_g = 2.0 * requested_x
        requested_quad = quad_stack(requested_g, layout=layout)
        reconstructed = (
            requested_quad[..., 0]
            - requested_quad[..., 1]
            - requested_quad[..., 2]
            + requested_quad[..., 3]
        ) / 2.0
        residual = reconstructed.to(torch.float64) - contrast.to(torch.float64)
        if float(residual.abs().max().item()) > 4e-7:
            raise RuntimeError("Target baseline transfer changed source signed contrast.")
        target_x.append(requested_x)
        target_g.append(requested_g)
        unsupported_weights.append(logical_bad)
        unsupported_cells.append(cell_bad)
        layers.append(
            {
                "layer": layer_index,
                "layout": layout,
                "source_signed_contrast_full_G": _summary(contrast),
                "target_requested_raw_x": _summary(requested_x),
                "unsupported_logical_weights": int(logical_bad.sum().item()),
                "logical_weights": int(logical_bad.numel()),
                "unsupported_cells": int(cell_bad.sum().item()),
                "cells": int(cell_bad.numel()),
                "maximum_contrast_residual_full_G": float(residual.abs().max().item()),
                "requested_raw_x_below_zero": int((requested_x < 0.0).sum().item()),
                "requested_raw_x_above_one": int((requested_x > 1.0).sum().item()),
            }
        )
    return CrossArrayTransfer(
        source_full_conductance=source_full,
        source_signed_contrast_full_g=contrasts,
        target_raw_x=tuple(target_x),
        target_full_conductance=tuple(target_g),
        unsupported_logical_weight=tuple(unsupported_weights),
        unsupported_cell=tuple(unsupported_cells),
        report={
            "policy": (
                "target_shared_destination_baseline_plus_unmodified_source_"
                "signed_contrast"
            ),
            "source_contrast_formula": "(Gpp-Gpm-Gmp+Gmm)/2",
            "target_clipping": False,
            "unsupported_target_policy": (
                "retain_request_for_PV_saturation_or_budget_exhaustion"
            ),
            "unsupported_logical_weights": sum(
                int(value.sum().item()) for value in unsupported_weights
            ),
            "logical_weights": sum(value.numel() for value in unsupported_weights),
            "unsupported_cells": sum(int(value.sum().item()) for value in unsupported_cells),
            "cells": sum(value.numel() for value in unsupported_cells),
            "layers": layers,
        },
    )


def literal_full_g_transfer_diagnostic(
    source_payload: Mapping[str, Any],
    *,
    target_lower_raw_x: Sequence[torch.Tensor],
    target_upper_raw_x: Sequence[torch.Tensor],
) -> Mapping[str, Any]:
    """Report literal source-full-G requests on target identities, unclipped."""

    full, contrast = source_full_g_and_signed_contrast(source_payload)
    lowers = tuple(target_lower_raw_x)
    uppers = tuple(target_upper_raw_x)
    layers = []
    raw_targets = []
    for layer_index, (source_g, source_c, lower, upper, layout) in enumerate(
        zip(full, contrast, lowers, uppers, LAYOUTS)
    ):
        target = source_g / 2.0
        lower_cpu = torch.as_tensor(lower).detach().cpu().to(torch.float32)
        upper_cpu = torch.as_tensor(upper).detach().cpu().to(torch.float32)
        cell_bad = (target < lower_cpu - 1e-6) | (target > upper_cpu + 1e-6)
        logical_bad = quad_stack(cell_bad, layout=layout).any(dim=-1)
        raw_targets.append(target)
        layers.append(
            {
                "layer": layer_index,
                "layout": layout,
                "requested_raw_x": _summary(target),
                "source_signed_contrast_full_G": _summary(source_c),
                "unsupported_logical_weights": int(logical_bad.sum().item()),
                "logical_weights": int(logical_bad.numel()),
                "unsupported_cells": int(cell_bad.sum().item()),
                "cells": int(cell_bad.numel()),
            }
        )
    return {
        "policy": "literal_source_full_G_to_target_cells_without_clipping",
        "target_clipping": False,
        "raw_x_targets": tuple(raw_targets),
        "unsupported_logical_weights": sum(
            int(layer["unsupported_logical_weights"]) for layer in layers
        ),
        "unsupported_cells": sum(int(layer["unsupported_cells"]) for layer in layers),
        "layers": layers,
    }


class OpenLoopPulsePort(Protocol):
    @property
    def size(self) -> int: ...

    def apply_identical_pulses(
        self, directions: torch.Tensor, counts: torch.Tensor
    ) -> None: ...


class _RawActiveOpenLoopPulsePort:
    """Pulse-only capability view.  Deliberately has no ``verify`` method."""

    def __init__(self, plant: IbmReramRawActivePlant) -> None:
        self.__plant = plant

    @property
    def size(self) -> int:
        return self.__plant.size

    def apply_identical_pulses(
        self, directions: torch.Tensor, counts: torch.Tensor
    ) -> None:
        direction = torch.as_tensor(
            directions, dtype=torch.int8, device=self.__plant.device
        )
        count = torch.as_tensor(counts, dtype=torch.int64, device=self.__plant.device)
        if direction.shape != (self.size,) or count.shape != (self.size,):
            raise ValueError("Open-loop pulse request shape mismatch.")
        if bool(torch.any(count < 0)):
            raise ValueError("Open-loop pulse counts must be nonnegative.")
        maximum = int(count.max().item()) if count.numel() else 0
        for pulse_index in range(maximum):
            self.__plant.pulse(
                torch.where(
                    count > pulse_index,
                    direction,
                    torch.zeros_like(direction),
                )
            )


def open_loop_pulse_port(plant: IbmReramRawActivePlant) -> OpenLoopPulsePort:
    return _RawActiveOpenLoopPulsePort(plant)


@dataclass(frozen=True)
class CoincidenceAdamStep:
    step: int
    conceptual_column_phases: int
    stochastic_row_line_assertions: int
    requested_cell_coincidences: int
    applied_cell_pulses: int
    capped_cell_requests: int
    upward_pulses: int
    downward_pulses: int
    mean_probability: float
    maximum_probability: float
    probability_clipped_cells: int


class ColumnSerialOpenLoopAdam:
    """Digital Adam with exact column-serial stochastic coincidence writes."""

    def __init__(
        self,
        port: OpenLoopPulsePort,
        *,
        device: torch.device | str,
        binding_shapes: Sequence[Sequence[int]],
        learning_rate_raw_x: float,
        nominal_delta_x: float,
        pulse_cap: int,
        pulse_selection_seed: int,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
    ) -> None:
        shapes = tuple(tuple(int(item) for item in shape) for shape in binding_shapes)
        if len(shapes) != 2 or sum(math.prod(shape) for shape in shapes) != port.size:
            raise ValueError("Expected two physical bindings covering the pulse port.")
        if any(len(shape) != 2 for shape in shapes):
            raise ValueError("Column-serial updates require rank-2 bindings.")
        scalars = (learning_rate_raw_x, nominal_delta_x, beta1, beta2, epsilon)
        if any(not math.isfinite(float(value)) for value in scalars):
            raise ValueError("Expected finite Adam settings.")
        if learning_rate_raw_x <= 0.0 or nominal_delta_x <= 0.0 or epsilon <= 0.0:
            raise ValueError("Expected positive Adam scale settings.")
        if not (0.0 <= beta1 < 1.0 and 0.0 <= beta2 < 1.0):
            raise ValueError("Expected Adam beta values in [0,1).")
        if isinstance(pulse_cap, bool) or not isinstance(pulse_cap, int) or pulse_cap < 1:
            raise ValueError("Expected a positive recovery pulse cap.")
        self._port = port
        self.device = torch.device(device)
        self.binding_shapes = shapes
        self.learning_rate_raw_x = float(learning_rate_raw_x)
        self.nominal_delta_x = float(nominal_delta_x)
        self.pulse_cap = int(pulse_cap)
        self.beta1 = float(beta1)
        self.beta2 = float(beta2)
        self.epsilon = float(epsilon)
        self.step_index = 0
        self.first_moment = torch.zeros(port.size, dtype=torch.float32, device=self.device)
        self.second_moment = torch.zeros_like(self.first_moment)
        self.pulse_count = torch.zeros(port.size, dtype=torch.int64, device=self.device)
        self.selection_generator = torch.Generator(device=self.device)
        self.selection_generator.manual_seed(int(pulse_selection_seed))
        self.total_conceptual_column_phases = 0
        self.total_probability_clipped_cells = 0

    def _adam_command(self, gradient: torch.Tensor) -> torch.Tensor:
        if gradient.shape != self.first_moment.shape or not bool(
            torch.all(torch.isfinite(gradient))
        ):
            raise ValueError("Expected finite fixed-shape physical gradients.")
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

    def step(self, physical_conductance_gradients: Sequence[torch.Tensor]) -> CoincidenceAdamStep:
        gradients = tuple(physical_conductance_gradients)
        if len(gradients) != len(self.binding_shapes):
            raise ValueError("Expected one physical conductance gradient per binding.")
        for gradient, shape in zip(gradients, self.binding_shapes):
            if tuple(gradient.shape) != shape or not bool(torch.all(torch.isfinite(gradient))):
                raise ValueError("Expected finite physical conductance gradients.")
        # G=2*x, hence dL/dx=2*dL/dG.
        command = self._adam_command(flatten_physical(tuple(2.0 * value.detach() for value in gradients)))
        raw_probability = command.abs() / self.nominal_delta_x
        probability = raw_probability.clamp(max=1.0)
        random_value = torch.rand(
            probability.shape,
            dtype=probability.dtype,
            device=probability.device,
            generator=self.selection_generator,
        )
        # Flattened Bernoulli draws are exactly one independent signed row-line
        # draw for each deterministic column-line phase.  Since columns are
        # disjoint, applying the selected one-pulse requests together is state-
        # and per-cell-RNG-equivalent to serial application.
        requested = (random_value < probability) & (command != 0.0)
        eligible = self.pulse_count < self.pulse_cap
        selected = requested & eligible
        directions = torch.zeros(self._port.size, dtype=torch.int8, device=self.device)
        directions[selected & (command > 0.0)] = 1
        directions[selected & (command < 0.0)] = -1
        counts = selected.to(torch.int64)
        self._port.apply_identical_pulses(directions, counts)
        self.pulse_count.add_(counts)
        phases = sum(shape[1] for shape in self.binding_shapes)
        clipped = int((raw_probability > 1.0).sum().item())
        self.total_conceptual_column_phases += phases
        self.total_probability_clipped_cells += clipped
        return CoincidenceAdamStep(
            step=self.step_index,
            conceptual_column_phases=phases,
            stochastic_row_line_assertions=int(requested.sum().item()),
            requested_cell_coincidences=int(requested.sum().item()),
            applied_cell_pulses=int(counts.sum().item()),
            capped_cell_requests=int((requested & ~eligible).sum().item()),
            upward_pulses=int((directions > 0).sum().item()),
            downward_pulses=int((directions < 0).sum().item()),
            mean_probability=float(probability.mean().item()),
            maximum_probability=float(probability.max().item()),
            probability_clipped_cells=clipped,
        )

    def state_dict(self) -> dict[str, Any]:
        return {
            "schema": "ebl.mnist_relu_drn.ibm_om_cross_array_open_loop_adam_state",
            "schema_version": 1,
            "evidence_tier": EVIDENCE_TIER,
            "claim_label": CLAIM_LABEL,
            "update_surface": UPDATE_SURFACE,
            "binding_shapes": [list(shape) for shape in self.binding_shapes],
            "conceptual_column_phases_per_minibatch": [
                shape[1] for shape in self.binding_shapes
            ],
            "column_phase_vectorization": (
                "disjoint_cell_state_and_per_cell_RNG_equivalent"
            ),
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
            "total_conceptual_column_phases": self.total_conceptual_column_phases,
            "total_probability_clipped_cells": self.total_probability_clipped_cells,
            "verify_reads_during_updates": 0,
            "authoritative_weight_shadow": None,
            "fully_parallel_ibm_outer_product_equivalent": False,
        }


__all__ = [
    "CLAIM_LABEL",
    "ColumnSerialOpenLoopAdam",
    "CoincidenceAdamStep",
    "CrossArrayTransfer",
    "EVIDENCE_TIER",
    "UPDATE_SURFACE",
    "build_target_baseline_contrast_transfer",
    "literal_full_g_transfer_diagnostic",
    "load_selected_source_recovery",
    "open_loop_pulse_port",
    "source_full_g_and_signed_contrast",
]
