"""Shared contract helpers for apparent-state Adam recovery from P&V states."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import fmean
from typing import Any, Mapping, Sequence

import torch

from experiments.mnist_relu_drn.ibm_om_winsorized_onchip_adam import (
    split_flat_physical,
)


EXPERIMENT_ID = "mnist_ibm_om_apparent_pv_adam_recovery.v1"
EXTENDED_FIGURE6_EXPERIMENT_ID = (
    "mnist_ibm_om_apparent_pv_adam_recovery.figure6_corrupt_3ep_lr3e5_cap192.v1"
)
SCHEMA_VERSION = 1

MODEL_WINSORIZED = "winsorized_0_2"
MODEL_FIGURE6 = "figure6_independent"
MODELS = (MODEL_WINSORIZED, MODEL_FIGURE6)

TARGET_DIRECT = "direct"
TARGET_HWA = "hwa"
TARGETS = (TARGET_DIRECT, TARGET_HWA)

CORRUPTION_REPAIRED = "repaired"
CORRUPTION_CORRUPT = "corrupt"
CORRUPTIONS = (CORRUPTION_REPAIRED, CORRUPTION_CORRUPT)

LEARNING_RATE_GRID = (3.0e-6, 1.0e-5, 3.0e-5, 1.0e-4, 3.0e-4)
SCREEN_BATCHES = 512
MAIN_TRAINING_EXAMPLES = 55_000
PULSE_CAP_PER_CELL = 64
NOMINAL_DELTA_PROGRESS = 0.04745

# Frozen exploratory follow-up requested after the eight-arm one-epoch run.
# The cap is scaled with the number of epochs so it remains a safety budget,
# rather than silently tightening the permitted lifetime pulse count.
EXTENDED_FIGURE6_LEARNING_RATE = 3.0e-5
EXTENDED_FIGURE6_TRAINING_EPOCHS = 3
EXTENDED_FIGURE6_PULSE_CAP_PER_CELL = (
    PULSE_CAP_PER_CELL * EXTENDED_FIGURE6_TRAINING_EPOCHS
)
EXTENDED_FIGURE6_TRAINING_EXAMPLES = (
    MAIN_TRAINING_EXAMPLES * EXTENDED_FIGURE6_TRAINING_EPOCHS
)

FORWARD_STATE = "held_apparent_post_write_state"
PERSISTENT_ROLE = "secondary_diagnostic_only"
UPDATE_RULE = "open_loop_stochastic_OM_pulses_only"


@dataclass(frozen=True, order=True)
class ArmKey:
    """One of the eight frozen main recovery arms."""

    model: str
    target: str
    corruption: str

    def __post_init__(self) -> None:
        if self.model not in MODELS:
            raise ValueError(f"Unknown device model: {self.model!r}.")
        if self.target not in TARGETS:
            raise ValueError(f"Unknown P&V target: {self.target!r}.")
        if self.corruption not in CORRUPTIONS:
            raise ValueError(f"Unknown corruption state: {self.corruption!r}.")

    @property
    def slug(self) -> str:
        return f"{self.model}__{self.target}__{self.corruption}"


def main_arms() -> tuple[ArmKey, ...]:
    """Return the declared 2 x 2 x 2 arm matrix in stable order."""

    return tuple(
        ArmKey(model=model, target=target, corruption=corruption)
        for model in MODELS
        for target in TARGETS
        for corruption in CORRUPTIONS
    )


def winsorized_full_conductance(
    raw_a: torch.Tensor,
    binding_shapes: Sequence[Sequence[int]],
    *,
    apparent: bool,
) -> tuple[torch.Tensor, ...]:
    """Map old OM raw ``a`` to positive G, projecting only apparent reads."""

    value = torch.as_tensor(raw_a)
    if value.ndim != 1 or not value.is_floating_point() or not bool(
        torch.isfinite(value).all()
    ):
        raise ValueError("Expected one finite floating raw-a vector.")
    full_g = value + 1.0
    if apparent:
        # Write noise may put the held observation outside the physical
        # Winsorized interval.  The DRN receives the realizable G in [0,2].
        full_g = full_g.clamp(0.0, 2.0)
    elif bool(torch.any((full_g < 0.0) | (full_g > 2.0))):
        raise RuntimeError("Persistent Winsorized OM state left G=[0,2].")
    return split_flat_physical(full_g, binding_shapes)


def winsorized_apparent_projection(raw_a: torch.Tensor) -> dict[str, Any]:
    """Describe the lower/upper projection used by old apparent forwards."""

    literal = torch.as_tensor(raw_a).detach().reshape(-1) + 1.0
    below = literal < 0.0
    above = literal > 2.0
    projected = below | above
    return {
        "coordinate": "literal_held_apparent_G=apparent_raw_a+1",
        "forward_projection": "clamp_to_[0,2]",
        "below_Gmin": int(below.sum().item()),
        "above_Gmax": int(above.sum().item()),
        "projected_cells": int(projected.sum().item()),
        "literal_minimum": float(literal.min().item()),
        "literal_mean": float(literal.mean().item()),
        "literal_maximum": float(literal.max().item()),
    }


def counts_by_layer(
    values: torch.Tensor,
    binding_shapes: Sequence[Sequence[int]],
) -> list[dict[str, int]]:
    """Summarize one flattened counter in physical binding order."""

    result = []
    offset = 0
    flat = torch.as_tensor(values).detach().reshape(-1)
    for layer, shape_value in enumerate(binding_shapes):
        shape = tuple(int(item) for item in shape_value)
        count = 1
        for item in shape:
            count *= item
        selected = flat[offset : offset + count]
        if selected.numel() != count:
            raise ValueError("Counter vector does not cover every binding.")
        result.append(
            {
                "layer": layer,
                "cells": count,
                "total": int(selected.to(torch.int64).sum().item()),
                "nonzero_cells": int((selected != 0).sum().item()),
                "maximum_per_cell": int(selected.max().item()),
            }
        )
        offset += count
    if offset != flat.numel():
        raise ValueError("Counter vector exceeds the declared bindings.")
    return result


def _candidate_summary(candidate: Mapping[str, Any]) -> dict[str, float | int]:
    anchors = candidate.get("anchors")
    if not isinstance(anchors, Mapping) or set(anchors) != set(MODELS):
        raise ValueError("Each learning-rate candidate needs both model anchors.")
    accuracies = []
    divergences = []
    pulses = 0
    for model in MODELS:
        anchor = anchors[model]
        if not isinstance(anchor, Mapping):
            raise ValueError("Invalid model anchor report.")
        validation = anchor.get("after_validation")
        pulse = anchor.get("pulses")
        if not isinstance(validation, Mapping) or not isinstance(pulse, Mapping):
            raise ValueError("Anchor report lacks validation or pulse metrics.")
        apparent = validation.get("apparent")
        if not isinstance(apparent, Mapping):
            raise ValueError("Anchor validation lacks its apparent-state primary.")
        accuracies.append(float(apparent["student_accuracy"]))
        divergences.append(float(apparent["kl_teacher_student"]))
        pulses += int(pulse["applied"])
    return {
        "macro_apparent_validation_accuracy": fmean(accuracies),
        "macro_apparent_validation_kl": fmean(divergences),
        "total_applied_pulses": pulses,
    }


def select_common_learning_rate(
    candidates: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Select one common rate by the frozen apparent-state tie-break rules."""

    selected = tuple(candidates)
    if len(selected) != len(LEARNING_RATE_GRID):
        raise ValueError("Expected exactly the frozen five-rate screen.")
    observed = tuple(float(item.get("learning_rate", -1.0)) for item in selected)
    if observed != LEARNING_RATE_GRID:
        raise ValueError("Learning-rate candidates are not in the frozen grid order.")

    scored = []
    for candidate in selected:
        summary = _candidate_summary(candidate)
        rate = float(candidate["learning_rate"])
        key = (
            float(summary["macro_apparent_validation_accuracy"]),
            -float(summary["macro_apparent_validation_kl"]),
            -int(summary["total_applied_pulses"]),
            -rate,
        )
        scored.append({"learning_rate": rate, **summary, "selection_key": key})
    winner = max(scored, key=lambda item: item["selection_key"])
    return {
        "learning_rate": float(winner["learning_rate"]),
        "criterion": (
            "maximum macro apparent validation accuracy across repaired/direct "
            "old+Figure6 anchors; then lower macro apparent KL; then fewer "
            "applied pulses; then lower learning rate"
        ),
        "scored_candidates": [
            {
                key: (list(value) if key == "selection_key" else value)
                for key, value in item.items()
            }
            for item in scored
        ],
    }


__all__ = [
    "ArmKey",
    "CORRUPTIONS",
    "CORRUPTION_CORRUPT",
    "CORRUPTION_REPAIRED",
    "EXPERIMENT_ID",
    "EXTENDED_FIGURE6_EXPERIMENT_ID",
    "EXTENDED_FIGURE6_LEARNING_RATE",
    "EXTENDED_FIGURE6_PULSE_CAP_PER_CELL",
    "EXTENDED_FIGURE6_TRAINING_EPOCHS",
    "EXTENDED_FIGURE6_TRAINING_EXAMPLES",
    "FORWARD_STATE",
    "LEARNING_RATE_GRID",
    "MAIN_TRAINING_EXAMPLES",
    "MODELS",
    "MODEL_FIGURE6",
    "MODEL_WINSORIZED",
    "NOMINAL_DELTA_PROGRESS",
    "PERSISTENT_ROLE",
    "PULSE_CAP_PER_CELL",
    "SCHEMA_VERSION",
    "SCREEN_BATCHES",
    "TARGETS",
    "TARGET_DIRECT",
    "TARGET_HWA",
    "UPDATE_RULE",
    "counts_by_layer",
    "main_arms",
    "select_common_learning_rate",
    "winsorized_apparent_projection",
    "winsorized_full_conductance",
]
