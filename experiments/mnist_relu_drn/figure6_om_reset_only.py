"""RESET-only Figure-6 mapping and absolute-conductance programming.

The mapper may consume each cell's fixed RESET endpoint and one frozen nominal
SET value.  The true per-cell SET endpoint is plant-only state: it limits the
realized conductance, but it is never used to resize a requested target or to
normalize a verify observation for the controller.

For each four-cell dual-rail quad the requested baseline is the largest RESET
endpoint in that quad.  This makes the requested physical zero common to all
four cells.  The two sign-selected cells receive one common active target, so
``G++ == G--`` and ``G+- == G-+`` hold for requested targets.  They need not
hold after the hidden SET limits or stochastic program-and-verify act.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence

import torch

from experiments.mnist_relu_drn.figure6_om_pulse import Figure6OmPulsePlant
from experiments.mnist_relu_drn.hfo2_figure6_drn import (
    EndpointField,
    flatten_physical,
)
from experiments.mnist_relu_drn.ibm_om_baseline_selection import (
    quad_stack,
    scatter_quads,
)
from training.ibm_reram_hwa import IbmReramArrayPopulation
from training.ibm_reram_program_verify import (
    ControllerSettings,
    ProgramVerifyResult,
    run_program_verify,
)


RESET_ONLY_MAPPING = "shared_quad_exact_RESET_nominal_SET"
RESET_ONLY_GRADIENT = "nominal_span_STE_without_true_SET"
RESET_ONLY_VERIFY = "absolute_apparent_conductance_without_SET_normalization"


def _validate_mapping_settings(
    field: EndpointField,
    *,
    nominal_set: float,
    scale_fractions: Sequence[float],
) -> tuple[float, tuple[float, float]]:
    nominal = float(nominal_set)
    fractions = tuple(float(value) for value in scale_fractions)
    if not math.isfinite(nominal) or nominal <= 0.0:
        raise ValueError("Expected a positive finite nominal SET value.")
    if len(fractions) != 2 or any(
        not math.isfinite(value) or value <= 0.0 or value > 1.0
        for value in fractions
    ):
        raise ValueError("Expected two scale fractions in (0, 1].")
    if any(bool(torch.any(reset >= nominal)) for reset in field.reset):
        raise ValueError("Nominal SET must exceed every known RESET endpoint.")
    return nominal, fractions  # type: ignore[return-value]


def _validate_masters(
    masters: Sequence[torch.Tensor], field: EndpointField
) -> tuple[torch.Tensor, torch.Tensor]:
    selected = tuple(masters)
    if len(selected) != 2:
        raise ValueError("Expected exactly two signed logical masters.")
    for master, shape in zip(selected, field.shapes, strict=True):
        if (
            tuple(master.shape) != (shape[0] // 2, shape[1] // 2)
            or not master.is_floating_point()
            or not bool(torch.isfinite(master).all())
            or bool(torch.any(master < -1.0))
            or bool(torch.any(master > 1.0))
        ):
            raise ValueError(
                "Expected finite signed masters in [-1,1] matching the endpoint field."
            )
    return selected  # type: ignore[return-value]


@dataclass(frozen=True)
class ResetOnlyMapping:
    """Requested targets and the realization limited by hidden true SETs."""

    requested_g: tuple[torch.Tensor, torch.Tensor]
    plant_limited_g: tuple[torch.Tensor, torch.Tensor]
    baselines: tuple[torch.Tensor, torch.Tensor]
    nominal_spans: tuple[torch.Tensor, torch.Tensor]
    requested_above_true_set: tuple[torch.Tensor, torch.Tensor]


def map_masters_from_reset_only(
    masters: Sequence[torch.Tensor],
    field: EndpointField,
    *,
    nominal_set: float,
    scale_fractions: Sequence[float] = (1.0, 1.0),
) -> ResetOnlyMapping:
    """Map logical masters without consuming the true SET in target creation.

    The true SET is used only in ``plant_limited_g`` to model physical
    saturation.  ``requested_g`` and every baseline/span depend only on the
    logical master, the known RESET endpoints, and the frozen nominal SET.
    """

    nominal, fractions = _validate_mapping_settings(
        field, nominal_set=nominal_set, scale_fractions=scale_fractions
    )
    selected = _validate_masters(masters, field)
    requested_layers = []
    realized_layers = []
    baseline_layers = []
    span_layers = []
    unreachable_layers = []
    for master, reset, true_set, shape, layout, fraction in zip(
        selected,
        field.reset,
        field.set,
        field.shapes,
        field.layouts,
        fractions,
        strict=True,
    ):
        reset_quad = quad_stack(reset, layout=layout)
        baseline = reset_quad.max(dim=-1).values
        span = (nominal - baseline) * fraction
        positive = torch.relu(master)
        negative = torch.relu(-master)
        offsets = torch.stack(
            (positive, negative, negative, positive), dim=-1
        ) * span.unsqueeze(-1)
        requested_quad = baseline.unsqueeze(-1) + offsets
        requested = scatter_quads(requested_quad, shape=shape, layout=layout)
        # The true SET remains on the plant side.  It limits what the ideal
        # continuous plant can realize but never changes the requested target.
        realized = torch.minimum(requested, true_set)
        if bool(torch.any(realized < reset)):
            raise RuntimeError("RESET-only realization fell below a RESET endpoint.")
        requested_layers.append(requested)
        realized_layers.append(realized)
        baseline_layers.append(baseline)
        span_layers.append(span)
        unreachable_layers.append(requested > true_set)
    return ResetOnlyMapping(
        requested_g=tuple(requested_layers),  # type: ignore[arg-type]
        plant_limited_g=tuple(realized_layers),  # type: ignore[arg-type]
        baselines=tuple(baseline_layers),  # type: ignore[arg-type]
        nominal_spans=tuple(span_layers),  # type: ignore[arg-type]
        requested_above_true_set=tuple(unreachable_layers),  # type: ignore[arg-type]
    )


def lift_reset_only_gradients(
    masters: Sequence[torch.Tensor],
    physical_gradients: Sequence[torch.Tensor],
    field: EndpointField,
    *,
    nominal_set: float,
    scale_fractions: Sequence[float] = (1.0, 1.0),
) -> tuple[torch.Tensor, torch.Tensor]:
    """Lift physical gradients with a RESET/nominal-span straight-through rule.

    No true SET value or saturation mask enters this gradient port.  This is
    deliberately less informed than differentiating through the hidden
    per-cell upper endpoint.
    """

    nominal, fractions = _validate_mapping_settings(
        field, nominal_set=nominal_set, scale_fractions=scale_fractions
    )
    selected = _validate_masters(masters, field)
    gradients = tuple(physical_gradients)
    if len(gradients) != 2:
        raise ValueError("Expected one physical gradient tensor per DRN layer.")
    result = []
    for master, physical, reset, shape, layout, fraction in zip(
        selected,
        gradients,
        field.reset,
        field.shapes,
        field.layouts,
        fractions,
        strict=True,
    ):
        if tuple(physical.shape) != shape or not bool(torch.isfinite(physical).all()):
            raise ValueError("Physical RESET-only gradient shape or values are invalid.")
        baseline = quad_stack(reset, layout=layout).max(dim=-1).values
        nominal_span = (nominal - baseline).to(master) * fraction
        contribution = quad_stack(
            physical.to(device=master.device, dtype=master.dtype), layout=layout
        )
        positive = contribution[..., 0] + contribution[..., 3]
        negative = -(contribution[..., 1] + contribution[..., 2])
        gradient = nominal_span * torch.where(master >= 0.0, positive, negative)
        if not bool(torch.isfinite(gradient).all()):
            raise FloatingPointError("RESET-only logical gradient is non-finite.")
        result.append(gradient)
    return tuple(result)  # type: ignore[return-value]


def apparent_conductance_unprojected(plant: Figure6OmPulsePlant) -> torch.Tensor:
    """Return the literal apparent conductance observed by an absolute-G verifier."""

    reset = flatten_physical(plant.field.reset)
    window = flatten_physical(plant.field.window)
    apparent_progress = (plant.apparent_raw_a + 1.0) / 2.0
    return reset + apparent_progress * window


class _AbsoluteConductanceProgramVerifyPort:
    """Controller port exposing only apparent absolute conductance and pulses."""

    def __init__(self, plant: Figure6OmPulsePlant, *, maximum_pulses: int) -> None:
        self.__plant = plant
        self.__maximum_pulses = int(maximum_pulses)

    @property
    def size(self) -> int:
        return self.__plant.size

    def verify(self) -> torch.Tensor:
        return apparent_conductance_unprojected(self.__plant).clone()

    def apply_identical_pulses(
        self, directions: torch.Tensor, counts: torch.Tensor
    ) -> None:
        direction = torch.as_tensor(
            directions, device=self.__plant.device, dtype=torch.int8
        )
        count = torch.as_tensor(counts, device=self.__plant.device, dtype=torch.int64)
        if direction.shape != (self.size,) or count.shape != (self.size,):
            raise ValueError("P&V directions and counts must match the OM plant.")
        if bool(torch.any(count < 0)):
            raise ValueError("P&V pulse counts must be non-negative.")
        for pulse_index in range(int(count.max().item()) if count.numel() else 0):
            active = torch.where(
                count > pulse_index, direction, torch.zeros_like(direction)
            )
            _, capped = self.__plant.pulse(
                active, pulse_cap=self.__maximum_pulses
            )
            if capped:
                raise RuntimeError("The P&V controller exceeded its OM pulse budget.")


def program_conductance_with_verify(
    field: EndpointField,
    requested_g: Sequence[torch.Tensor],
    population: IbmReramArrayPopulation,
    *,
    pulse_noise_seed: int,
    tolerance_conductance: float,
    maximum_pulses: int,
    noisy_initial_reset_verify: bool = True,
    preaccept_exact_healthy_reset_targets: bool = True,
) -> tuple[Figure6OmPulsePlant, ProgramVerifyResult]:
    """Program absolute-G requests without exposing true SET to the controller."""

    selected = tuple(requested_g)
    if len(selected) != 2:
        raise ValueError("Expected two requested conductance tensors.")
    for target, reset, shape in zip(
        selected, field.reset, field.shapes, strict=True
    ):
        if (
            tuple(target.shape) != shape
            or not bool(torch.isfinite(target).all())
            or bool(torch.any(target < reset))
        ):
            raise ValueError(
                "Requested conductances must be finite, shape-matched, and no lower than RESET."
            )
    tolerance = float(tolerance_conductance)
    if not math.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("Expected a positive finite conductance tolerance.")
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
    reset = flatten_physical(field.reset)
    preaccepted = (
        (target == reset) & ~plant.corrupt
        if preaccept_exact_healthy_reset_targets
        else torch.zeros_like(target, dtype=torch.bool)
    )
    raw = run_program_verify(
        _AbsoluteConductanceProgramVerifyPort(
            plant, maximum_pulses=maximum_pulses
        ),
        targets=target,
        tolerance=tolerance,
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
    observed = apparent_conductance_unprojected(plant)
    if not torch.equal(observed, result.apparent_endpoint):
        raise RuntimeError("Absolute-G P&V result does not match the held apparent state.")
    if not torch.equal(plant.pulse_count, result.total_pulses):
        raise RuntimeError("OM plant and absolute-G P&V pulse accounting diverged.")
    return plant, result


def mapping_contract() -> Mapping[str, Any]:
    return {
        "mapping": RESET_ONLY_MAPPING,
        "gradient": RESET_ONLY_GRADIENT,
        "verify": RESET_ONLY_VERIFY,
        "controller_knows": [
            "continuous requested absolute conductance",
            "current held apparent absolute conductance",
            "its own history",
        ],
        "controller_does_not_know": [
            "true per-cell SET endpoint",
            "persistent state",
            "OM pulse parameters",
            "corruption mask",
        ],
    }


__all__ = [
    "RESET_ONLY_GRADIENT",
    "RESET_ONLY_MAPPING",
    "RESET_ONLY_VERIFY",
    "ResetOnlyMapping",
    "apparent_conductance_unprojected",
    "lift_reset_only_gradients",
    "map_masters_from_reset_only",
    "mapping_contract",
    "program_conductance_with_verify",
]
