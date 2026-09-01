"""Matched digital-Adam command realization for exact-P0 recovery.

The closed-loop arm owns only a bounded desired-``x`` accumulator and a
capability-limited :class:`VerifyPort`.  Persistent device state remains
inaccessible to the controller and remains the sole forward-pass authority.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import math
from typing import Any, Mapping, Sequence

import torch

from experiments.mnist_relu_drn.ibm_om_winsorized_onchip_adam import (
    flatten_physical,
)
from training.ibm_reram_program_verify import VerifyPort


def _mask_sha256(value: torch.Tensor) -> str:
    cpu = value.detach().contiguous().cpu()
    digest = sha256()
    digest.update(str(cpu.dtype).encode("utf-8"))
    digest.update(json.dumps(list(cpu.shape), separators=(",", ":")).encode("utf-8"))
    digest.update(cpu.numpy().tobytes(order="C"))
    return digest.hexdigest()


@dataclass(frozen=True)
class TargetProjection:
    cells: int
    signed_sum: float
    absolute_sum: float
    maximum_absolute: float


@dataclass(frozen=True)
class IncrementalProgramVerifyAdamStep:
    step: int
    issued_cell_pulses: int
    upward_pulses: int
    downward_pulses: int
    api_full_port_verify_calls: int
    conceptual_post_pulse_cell_observations: int
    accepted_cells: int
    per_call_one_pulse_budget_reached_cells: int
    cap_blocked_cells: int
    target_debt_cells: int
    frozen_target_debt_cells: int
    trainable_accepted_cells: int
    maximum_absolute_target_debt: float
    reversals: int
    target_projection: TargetProjection
    target_projection_by_layer: tuple[TargetProjection, ...]


def _projection(unprojected: torch.Tensor, projected: torch.Tensor) -> TargetProjection:
    # Signed loss is request minus projection: positive at the upper rail,
    # negative at the lower rail.  It is distinct from the opposite-signed
    # correction applied by the projection operator.
    lost = unprojected - projected
    absolute = lost.abs()
    return TargetProjection(
        cells=int((lost != 0.0).sum().item()),
        signed_sum=float(lost.sum().item()),
        absolute_sum=float(absolute.sum().item()),
        maximum_absolute=float(absolute.max().item()) if absolute.numel() else 0.0,
    )


def _projection_by_layer(
    unprojected: torch.Tensor,
    projected: torch.Tensor,
    binding_shapes: Sequence[Sequence[int]],
) -> tuple[TargetProjection, ...]:
    result = []
    offset = 0
    for shape in binding_shapes:
        size = math.prod(tuple(shape))
        result.append(
            _projection(
                unprojected[offset : offset + size],
                projected[offset : offset + size],
            )
        )
        offset += size
    return tuple(result)


class IncrementalOnePulseProgramVerifyAdam:
    """Digital Adam with an incremental, max-one-pulse P&V realization.

    ``desired_raw_x`` is controller state.  It is initialized from the saved
    P0 apparent observation and projected into the nonnegative full-
    conductance domain.  Each Adam increment is accumulated then projected to
    ``[0, 1]`` before one capability-limited P&V iteration.  The class never
    reads persistent state or hidden device parameters.
    """

    def __init__(
        self,
        port: VerifyPort,
        *,
        initial_apparent_raw_a: torch.Tensor,
        device: torch.device | str,
        binding_shapes: Sequence[Sequence[int]],
        learning_rate_raw_x: float,
        nominal_delta_x: float,
        verify_tolerance_raw_x: float,
        pulse_cap: int,
        maximum_pulses_per_cell_per_minibatch: int = 1,
        trainable_cell_mask: torch.Tensor | None = None,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
    ) -> None:
        shapes = tuple(tuple(int(item) for item in shape) for shape in binding_shapes)
        if len(shapes) != 2 or any(len(shape) != 2 for shape in shapes):
            raise ValueError("Expected two rank-2 physical bindings.")
        if sum(math.prod(shape) for shape in shapes) != port.size:
            raise ValueError("Physical bindings do not cover the P&V port.")
        scalars = (
            learning_rate_raw_x,
            nominal_delta_x,
            verify_tolerance_raw_x,
            beta1,
            beta2,
            epsilon,
        )
        if any(not math.isfinite(float(value)) for value in scalars):
            raise ValueError("Expected finite incremental P&V Adam settings.")
        if (
            learning_rate_raw_x <= 0.0
            or nominal_delta_x <= 0.0
            or verify_tolerance_raw_x <= 0.0
            or epsilon <= 0.0
        ):
            raise ValueError("Expected positive Adam, spacing and tolerance scales.")
        if not (0.0 <= beta1 < 1.0 and 0.0 <= beta2 < 1.0):
            raise ValueError("Expected Adam beta values in [0,1).")
        if isinstance(pulse_cap, bool) or not isinstance(pulse_cap, int) or pulse_cap < 1:
            raise ValueError("Expected a positive cumulative pulse cap.")
        if (
            isinstance(maximum_pulses_per_cell_per_minibatch, bool)
            or not isinstance(maximum_pulses_per_cell_per_minibatch, int)
            or maximum_pulses_per_cell_per_minibatch < 1
            or maximum_pulses_per_cell_per_minibatch > pulse_cap
        ):
            raise ValueError("Expected a positive per-minibatch pulse ceiling within cap.")
        initial_a = torch.as_tensor(
            initial_apparent_raw_a, dtype=torch.float32, device=device
        )
        if initial_a.shape != (port.size,) or not bool(torch.all(torch.isfinite(initial_a))):
            raise ValueError("Expected a finite P0 apparent raw-a vector.")

        self._port = port
        self.device = torch.device(device)
        self.binding_shapes = shapes
        self.learning_rate_raw_x = float(learning_rate_raw_x)
        self.nominal_delta_x = float(nominal_delta_x)
        self.verify_tolerance_raw_x = float(verify_tolerance_raw_x)
        self.pulse_cap = int(pulse_cap)
        self.maximum_pulses_per_cell_per_minibatch = int(
            maximum_pulses_per_cell_per_minibatch
        )
        if trainable_cell_mask is None:
            writable = torch.ones(port.size, dtype=torch.bool, device=self.device)
        else:
            writable = torch.as_tensor(
                trainable_cell_mask, dtype=torch.bool, device=self.device
            ).clone()
        if writable.shape != (port.size,) or not bool(torch.any(writable)):
            raise ValueError("Expected a nonempty fixed trainable-cell mask.")
        self.trainable_cell_mask = writable
        self.trainable_cell_mask_sha256 = _mask_sha256(writable)
        self.beta1 = float(beta1)
        self.beta2 = float(beta2)
        self.epsilon = float(epsilon)
        self.step_index = 0
        self.first_moment = torch.zeros(port.size, dtype=torch.float32, device=self.device)
        self.second_moment = torch.zeros_like(self.first_moment)
        self.pulse_count = torch.zeros(port.size, dtype=torch.int64, device=self.device)
        raw_initial = (initial_a + 1.0) / 2.0
        self.cached_apparent_raw_x = raw_initial.clone()
        self.desired_raw_x = raw_initial.clamp(0.0, 1.0)
        self.initial_target_projection = _projection(raw_initial, self.desired_raw_x)
        self.initial_target_projection_by_layer = _projection_by_layer(
            raw_initial, self.desired_raw_x, self.binding_shapes
        )
        self.previous_pulse_direction = torch.zeros(
            port.size, dtype=torch.int8, device=self.device
        )
        self.total_api_full_port_verify_calls = 0
        self.total_conceptual_post_pulse_cell_observations = 0
        self.total_reversals = 0
        self.total_upward_pulses = 0
        self.total_downward_pulses = 0
        self.total_per_call_pulse_ceiling_reached_events = 0
        self.total_cap_blocked_events = 0
        self.total_target_projection_cells = 0
        self.total_target_projection_signed_sum = 0.0
        self.total_target_projection_absolute_sum = 0.0
        self.maximum_target_projection_absolute = 0.0
        self.ever_update_target_projected = torch.zeros(
            port.size, dtype=torch.bool, device=self.device
        )
        self.ever_cap_blocked = torch.zeros_like(self.ever_update_target_projected)
        self.total_target_projection_by_layer = [
            {
                "cell_events": 0,
                "signed_sum": 0.0,
                "absolute_sum": 0.0,
                "maximum_absolute": 0.0,
            }
            for _ in self.binding_shapes
        ]

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

    def step(
        self, physical_conductance_gradients: Sequence[torch.Tensor]
    ) -> IncrementalProgramVerifyAdamStep:
        gradients = tuple(physical_conductance_gradients)
        if len(gradients) != len(self.binding_shapes):
            raise ValueError("Expected one physical conductance gradient per binding.")
        for gradient, shape in zip(gradients, self.binding_shapes, strict=True):
            if tuple(gradient.shape) != shape or not bool(
                torch.all(torch.isfinite(gradient))
            ):
                raise ValueError("Expected finite physical conductance gradients.")

        # G=2*x, hence dL/dx=2*dL/dG.  This equation intentionally matches
        # ColumnSerialOpenLoopAdam exactly before controller realization.
        gradient_raw_x = flatten_physical(
            tuple(2.0 * value.detach() for value in gradients)
        )
        gradient_raw_x.masked_fill_(~self.trainable_cell_mask, 0.0)
        command = self._adam_command(gradient_raw_x)
        command.masked_fill_(~self.trainable_cell_mask, 0.0)
        unprojected = self.desired_raw_x + command
        projected = unprojected.clamp(0.0, 1.0)
        target_projection = _projection(unprojected, projected)
        target_projection_by_layer = _projection_by_layer(
            unprojected, projected, self.binding_shapes
        )
        self.desired_raw_x.copy_(projected)
        self.total_target_projection_cells += target_projection.cells
        self.ever_update_target_projected |= projected != unprojected
        self.total_target_projection_signed_sum += target_projection.signed_sum
        self.total_target_projection_absolute_sum += target_projection.absolute_sum
        self.maximum_target_projection_absolute = max(
            self.maximum_target_projection_absolute,
            target_projection.maximum_absolute,
        )
        for cumulative, layer_projection in zip(
            self.total_target_projection_by_layer,
            target_projection_by_layer,
            strict=True,
        ):
            cumulative["cell_events"] += layer_projection.cells
            cumulative["signed_sum"] += layer_projection.signed_sum
            cumulative["absolute_sum"] += layer_projection.absolute_sum
            cumulative["maximum_absolute"] = max(
                cumulative["maximum_absolute"],
                layer_projection.maximum_absolute,
            )

        # The P0 apparent observation is a sufficient cache because this model
        # has no fresh read noise or retention.  Therefore no redundant
        # pre-pulse full-port verify is issued.  A full-port API verify occurs
        # only after an actual vectorized pulse round; only pulsed coordinates
        # are conceptually observed and written back to the controller cache.
        total_this_step = torch.zeros_like(self.pulse_count)
        upward = 0
        downward = 0
        api_verify_calls = 0
        conceptual_observations = 0
        reversals = 0
        for _ in range(self.maximum_pulses_per_cell_per_minibatch):
            residual = self.cached_apparent_raw_x - self.desired_raw_x
            finite = torch.isfinite(residual)
            debt = (
                finite
                & self.trainable_cell_mask
                & (residual.abs() > self.verify_tolerance_raw_x)
            )
            eligible = self.pulse_count + total_this_step < self.pulse_cap
            selected = debt & eligible
            if not bool(torch.any(selected)):
                break
            direction = torch.zeros(self._port.size, dtype=torch.int8, device=self.device)
            direction[selected & (residual < 0.0)] = 1
            direction[selected & (residual > 0.0)] = -1
            changed = (
                selected
                & (self.previous_pulse_direction != 0)
                & (direction != self.previous_pulse_direction)
            )
            reversals += int(changed.sum().item())
            self.previous_pulse_direction[selected] = direction[selected]
            counts = selected.to(torch.int64)
            self._port.apply_identical_pulses(direction, counts)
            observed = self._port.verify()
            if observed.shape != self.cached_apparent_raw_x.shape or not bool(
                torch.all(torch.isfinite(observed[selected]))
            ):
                raise RuntimeError("Post-pulse P&V observation is invalid.")
            self.cached_apparent_raw_x[selected] = observed[selected]
            total_this_step.add_(counts)
            upward += int((direction > 0).sum().item())
            downward += int((direction < 0).sum().item())
            api_verify_calls += 1
            conceptual_observations += int(selected.sum().item())

        self.pulse_count.add_(total_this_step)
        if bool(torch.any(self.pulse_count > self.pulse_cap)):
            raise RuntimeError("Incremental P&V exceeded its cumulative pulse cap.")
        residual = self.cached_apparent_raw_x - self.desired_raw_x
        raw_debt = torch.isfinite(residual) & (
            residual.abs() > self.verify_tolerance_raw_x
        )
        debt = raw_debt & self.trainable_cell_mask
        frozen_debt = raw_debt & ~self.trainable_cell_mask
        cap_blocked = (self.pulse_count >= self.pulse_cap) & debt
        issued = int(total_this_step.sum().item())
        per_call_ceiling = int(
            (debt & (total_this_step >= self.maximum_pulses_per_cell_per_minibatch))
            .sum()
            .item()
        )
        cap_blocked_count = int(cap_blocked.sum().item())
        self.ever_cap_blocked |= cap_blocked
        self.total_api_full_port_verify_calls += api_verify_calls
        self.total_conceptual_post_pulse_cell_observations += conceptual_observations
        self.total_reversals += reversals
        self.total_upward_pulses += upward
        self.total_downward_pulses += downward
        self.total_per_call_pulse_ceiling_reached_events += per_call_ceiling
        self.total_cap_blocked_events += cap_blocked_count
        return IncrementalProgramVerifyAdamStep(
            step=self.step_index,
            issued_cell_pulses=issued,
            upward_pulses=upward,
            downward_pulses=downward,
            api_full_port_verify_calls=api_verify_calls,
            conceptual_post_pulse_cell_observations=conceptual_observations,
            accepted_cells=int((~debt & torch.isfinite(residual)).sum().item()),
            per_call_one_pulse_budget_reached_cells=per_call_ceiling,
            cap_blocked_cells=cap_blocked_count,
            target_debt_cells=int(debt.sum().item()),
            frozen_target_debt_cells=int(frozen_debt.sum().item()),
            trainable_accepted_cells=int(
                (self.trainable_cell_mask & ~debt & torch.isfinite(residual)).sum().item()
            ),
            maximum_absolute_target_debt=(
                float(residual[debt].abs().max().item()) if bool(torch.any(debt)) else 0.0
            ),
            reversals=reversals,
            target_projection=target_projection,
            target_projection_by_layer=target_projection_by_layer,
        )

    def state_dict(self) -> dict[str, Any]:
        return {
            "schema": "ebl.mnist_relu_drn.ibm_om_incremental_one_pulse_program_verify_adam_state",
            "schema_version": 2,
            "binding_shapes": [list(shape) for shape in self.binding_shapes],
            "learning_rate_raw_x": self.learning_rate_raw_x,
            "nominal_delta_x": self.nominal_delta_x,
            "verify_tolerance_raw_x": self.verify_tolerance_raw_x,
            "pulse_cap": self.pulse_cap,
            "maximum_pulses_per_cell_per_minibatch": (
                self.maximum_pulses_per_cell_per_minibatch
            ),
            "beta1": self.beta1,
            "beta2": self.beta2,
            "epsilon": self.epsilon,
            "step": self.step_index,
            "first_moment": self.first_moment.detach().cpu().clone(),
            "second_moment": self.second_moment.detach().cpu().clone(),
            "pulse_count": self.pulse_count.detach().cpu().clone(),
            "trainable_cell_mask": self.trainable_cell_mask.detach().cpu().clone(),
            "trainable_cell_mask_sha256": self.trainable_cell_mask_sha256,
            "trainable_cells": int(self.trainable_cell_mask.sum().item()),
            "frozen_cells": int((~self.trainable_cell_mask).sum().item()),
            "desired_raw_x": self.desired_raw_x.detach().cpu().clone(),
            "initial_target_projection": asdict(self.initial_target_projection),
            "initial_target_projection_by_layer": [
                asdict(value) for value in self.initial_target_projection_by_layer
            ],
            "target_projection_domain": [0.0, 1.0],
            "cached_apparent_raw_x": self.cached_apparent_raw_x.detach().cpu().clone(),
            "previous_pulse_direction": self.previous_pulse_direction.detach().cpu().clone(),
            "total_api_full_port_verify_calls": self.total_api_full_port_verify_calls,
            "total_conceptual_post_pulse_cell_observations": (
                self.total_conceptual_post_pulse_cell_observations
            ),
            "total_reversals": self.total_reversals,
            "total_upward_pulses": self.total_upward_pulses,
            "total_downward_pulses": self.total_downward_pulses,
            "total_per_call_pulse_ceiling_reached_events": (
                self.total_per_call_pulse_ceiling_reached_events
            ),
            "total_cap_blocked_cell_events": self.total_cap_blocked_events,
            "total_target_projection_cell_events": self.total_target_projection_cells,
            "total_target_projection_signed_sum": self.total_target_projection_signed_sum,
            "total_target_projection_absolute_sum": self.total_target_projection_absolute_sum,
            "maximum_target_projection_absolute": self.maximum_target_projection_absolute,
            "ever_update_target_projected": (
                self.ever_update_target_projected.detach().cpu().clone()
            ),
            "ever_cap_blocked": self.ever_cap_blocked.detach().cpu().clone(),
            "total_target_projection_by_layer": [
                dict(value) for value in self.total_target_projection_by_layer
            ],
            "controller_visible_state": "desired_raw_x_plus_apparent_verify_history",
            "pre_pulse_verify_policy": "cached_P0_apparent_no_redundant_preverify",
            "authoritative_weight_shadow": None,
            "persistent_state_visible_to_controller": False,
            "pulse_cap_semantic": "recovery_only_additional_pulses_after_exact_P0",
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if (
            state.get("schema")
            != "ebl.mnist_relu_drn.ibm_om_incremental_one_pulse_program_verify_adam_state"
            or state.get("schema_version") != 2
            or state.get("binding_shapes")
            != [list(shape) for shape in self.binding_shapes]
            or state.get("pulse_cap") != self.pulse_cap
            or state.get("maximum_pulses_per_cell_per_minibatch")
            != self.maximum_pulses_per_cell_per_minibatch
            or not math.isclose(
                float(state.get("learning_rate_raw_x", float("nan"))),
                self.learning_rate_raw_x,
                abs_tol=0.0,
            )
            or not math.isclose(
                float(state.get("beta1", float("nan"))), self.beta1, abs_tol=0.0
            )
            or not math.isclose(
                float(state.get("beta2", float("nan"))), self.beta2, abs_tol=0.0
            )
            or not math.isclose(
                float(state.get("epsilon", float("nan"))), self.epsilon, abs_tol=0.0
            )
            or state.get("target_projection_domain") != [0.0, 1.0]
            or state.get("pre_pulse_verify_policy")
            != "cached_P0_apparent_no_redundant_preverify"
            or state.get("controller_visible_state")
            != "desired_raw_x_plus_apparent_verify_history"
            or state.get("pulse_cap_semantic")
            != "recovery_only_additional_pulses_after_exact_P0"
            or state.get("initial_target_projection")
            != asdict(self.initial_target_projection)
            or state.get("initial_target_projection_by_layer")
            != [asdict(value) for value in self.initial_target_projection_by_layer]
            or state.get("trainable_cell_mask_sha256")
            != self.trainable_cell_mask_sha256
            or not math.isclose(
                float(state.get("nominal_delta_x", float("nan"))),
                self.nominal_delta_x,
                abs_tol=0.0,
            )
            or not math.isclose(
                float(state.get("verify_tolerance_raw_x", float("nan"))),
                self.verify_tolerance_raw_x,
                abs_tol=0.0,
            )
        ):
            raise ValueError("Incremental P&V Adam checkpoint contract mismatch.")
        tensors = {
            "first_moment": self.first_moment,
            "second_moment": self.second_moment,
            "pulse_count": self.pulse_count,
            "trainable_cell_mask": self.trainable_cell_mask,
            "desired_raw_x": self.desired_raw_x,
            "cached_apparent_raw_x": self.cached_apparent_raw_x,
            "previous_pulse_direction": self.previous_pulse_direction,
            "ever_update_target_projected": self.ever_update_target_projected,
            "ever_cap_blocked": self.ever_cap_blocked,
        }
        for name, destination in tensors.items():
            source = state.get(name)
            if (
                not isinstance(source, torch.Tensor)
                or source.shape != destination.shape
                or source.dtype != destination.dtype
            ):
                raise ValueError(f"Invalid incremental P&V Adam tensor {name!r}.")
            if source.is_floating_point() and not bool(torch.all(torch.isfinite(source))):
                raise ValueError(f"Non-finite incremental P&V Adam tensor {name!r}.")
        if (
            bool(torch.any(state["pulse_count"] < 0))
            or bool(torch.any(state["pulse_count"] > self.pulse_cap))
            or bool(torch.any(state["desired_raw_x"] < 0.0))
            or bool(torch.any(state["desired_raw_x"] > 1.0))
            or not torch.equal(
                state["trainable_cell_mask"].to(self.device),
                self.trainable_cell_mask,
            )
            or int(state.get("trainable_cells", -1))
            != int(self.trainable_cell_mask.sum().item())
            or int(state.get("frozen_cells", -1))
            != int((~self.trainable_cell_mask).sum().item())
            or bool(
                torch.any(
                    (state["previous_pulse_direction"] < -1)
                    | (state["previous_pulse_direction"] > 1)
                )
            )
            or int(state.get("step", -1)) < 0
        ):
            raise ValueError("Incremental P&V Adam checkpoint values are invalid.")
        for name, destination in tensors.items():
            source = state[name]
            destination.copy_(source.to(self.device))
        if (
            bool(torch.any(self.first_moment[~self.trainable_cell_mask] != 0.0))
            or bool(torch.any(self.second_moment[~self.trainable_cell_mask] != 0.0))
            or bool(torch.any(self.pulse_count[~self.trainable_cell_mask] != 0))
        ):
            raise ValueError("Frozen cells changed in the incremental P&V Adam state.")
        self.step_index = int(state["step"])
        self.total_api_full_port_verify_calls = int(
            state["total_api_full_port_verify_calls"]
        )
        self.total_conceptual_post_pulse_cell_observations = int(
            state["total_conceptual_post_pulse_cell_observations"]
        )
        self.total_reversals = int(state["total_reversals"])
        self.total_upward_pulses = int(state["total_upward_pulses"])
        self.total_downward_pulses = int(state["total_downward_pulses"])
        self.total_per_call_pulse_ceiling_reached_events = int(
            state["total_per_call_pulse_ceiling_reached_events"]
        )
        self.total_cap_blocked_events = int(state["total_cap_blocked_cell_events"])
        self.total_target_projection_cells = int(
            state["total_target_projection_cell_events"]
        )
        self.total_target_projection_signed_sum = float(
            state["total_target_projection_signed_sum"]
        )
        self.total_target_projection_absolute_sum = float(
            state["total_target_projection_absolute_sum"]
        )
        self.maximum_target_projection_absolute = float(
            state["maximum_target_projection_absolute"]
        )
        raw_by_layer = state.get("total_target_projection_by_layer")
        if not isinstance(raw_by_layer, list) or len(raw_by_layer) != len(
            self.binding_shapes
        ):
            raise ValueError("Invalid per-layer target-projection state.")
        self.total_target_projection_by_layer = [dict(value) for value in raw_by_layer]


__all__ = [
    "IncrementalOnePulseProgramVerifyAdam",
    "IncrementalProgramVerifyAdamStep",
    "TargetProjection",
]
