"""Matched IBM-OM corrupt-array HWA, transfer, and pulse-Adam runtime.

This exploratory runtime deliberately keeps four state transitions in one
hash-tracked :class:`~experiments.artifacts.RunStore`:

1. initialize a positive-conductance perfect-diode DRN from frozen ReLU
   weights (one normalized logical master per layer);
2. perform three epochs of mean-two persistent-endpoint HWA on one published
   corrupt IBM-OM assignment, selecting only on a disjoint endpoint bank;
3. after that checkpoint is frozen, open a different published corrupt array,
   map both epoch zero and the selected master through that array's repaired
   twin, and program five exact persistent endpoints; and
4. continue exactly endpoint 93301 with one epoch of column-serial, open-loop
   stochastic-pulse Adam.  No verify capability is exposed during recovery.

The repaired twins are mapping oracles only.  Every persistent HWA view,
deployment, and recovery pulse is executed on the corresponding *published*
population, including its immutable AIHWKit corrupt cells.  Circuit-facing
conductances are always the complete non-negative ``G=2*x=a+1`` values;
neither a signed conductance nor a baseline-subtracted surrogate is applied to
the DRN.
"""

from __future__ import annotations

import gc
import math
from pathlib import Path
from statistics import fmean, pstdev
from types import SimpleNamespace
from typing import Any, Iterable, Mapping, Sequence, TYPE_CHECKING

import torch

from experiments.artifacts import RunStore, atomic_write_json, sha256_file
from experiments.mnist_relu_drn.components import build_student_stack
from experiments.mnist_relu_drn.ibm_om_baseline_selection_runtime import (
    _artifact_records,
    _evaluate_detailed,
    _input,
    _sampler_python,
)
from experiments.mnist_relu_drn.ibm_om_baseline_spacing_pv import (
    apply_full_conductance_targets,
)
from experiments.mnist_relu_drn.ibm_om_bounded_codebook_scheme_screen import (
    _tensor_sha256,
)
from experiments.mnist_relu_drn.ibm_om_winsorized_cross_array_open_loop_adam_runtime import (
    _deploy_once,
    _fine_tune_open_loop,
)
from experiments.mnist_relu_drn.ibm_om_winsorized_pv_ensemble_qat import (
    gradient_mixture_weights,
    mix_logical_gradients,
)
from experiments.mnist_relu_drn.ibm_om_winsorized_pv_ensemble_qat_runtime import (
    _sample_indices,
)
from experiments.mnist_relu_drn.ibm_om_winsorized_qat import (
    WinsorizedQatLayerTemplate,
    logical_master_gradients,
    quantize_winsorized_logical_master,
)
from experiments.mnist_relu_drn.ibm_om_winsorized_qat_runtime import (
    NOMINAL_DELTA_X,
)
from experiments.mnist_relu_drn.runtime import _load_teacher
from experiments.mnist_shared import build_mnist_loaders, limited
from experiments.schema import to_plain_data
from training.checkpoint import atomic_torch_save
from training.ibm_reram_raw_active_program_verify import (
    array_population_as_pulse_population,
)
from training.ibm_reram_hwa import save_om_array_population


if TYPE_CHECKING:
    from ebl.cli import TrainRequest


_ROOT = Path(__file__).resolve().parents[2]
SUMMARY_SCHEMA = (
    "ebl.mnist_relu_drn.ibm_om_corrupt_source_hwa_cross_array_pulse_adam"
)
SUMMARY_SCHEMA_VERSION = 1
HWA_CHECKPOINT_SCHEMA = (
    "ebl.mnist_relu_drn.ibm_om_corrupt_source_hwa_selected_logical_master"
)
HWA_CHECKPOINT_SCHEMA_VERSION = 1
ENDPOINT_BANK_SCHEMA = (
    "ebl.mnist_relu_drn.ibm_om_corrupt_capable_persistent_endpoint_bank"
)
ENDPOINT_BANK_SCHEMA_VERSION = 1
EVIDENCE_TIER = "exploratory_noncanonical"
LAYOUTS = ("halves", "paired")
_MISSING = object()


def _plain_section(value: Any) -> Mapping[str, Any]:
    """Return a dataclass or mapping protocol section as plain data."""

    result = to_plain_data(value)
    if not isinstance(result, Mapping):
        raise TypeError("Expected a mapping-like protocol section.")
    return result


def _field(value: Any, name: str, default: Any = _MISSING) -> Any:
    """Read a strict config field from either a dataclass or mapping."""

    if isinstance(value, Mapping):
        if name in value:
            return value[name]
    elif hasattr(value, name):
        return getattr(value, name)
    if default is not _MISSING:
        return default
    raise AttributeError(f"Protocol section lacks required field {name!r}.")


def _positive_full_g(values: Sequence[torch.Tensor]) -> None:
    """Fail closed if a circuit view is not complete positive conductance."""

    selected = tuple(values)
    if len(selected) != 2:
        raise ValueError("Expected two complete conductance tensors.")
    for value in selected:
        if (
            not isinstance(value, torch.Tensor)
            or not value.is_floating_point()
            or not bool(torch.all(torch.isfinite(value)))
            or bool(torch.any(value < 0.0))
            or bool(torch.any(value > 2.0))
        ):
            raise ValueError("Expected every circuit conductance in G=[0,2].")


def _apply_full_g(stack: Any, values: Sequence[torch.Tensor]) -> None:
    _positive_full_g(values)
    apply_full_conductance_targets(
        stack.bundle.catalog,
        tuple(values),
        conductance_min=0.0,
        conductance_max=2.0,
    )


def _initial_logical_masters(
    teacher: Any,
    *,
    device: torch.device,
) -> tuple[tuple[torch.nn.Parameter, ...], tuple[float, ...], tuple[str, ...]]:
    """Create the sole clean master as ``W / absmax(W)`` in float32."""

    source = tuple(
        value.detach().cpu().to(torch.float32).clone()
        for value in teacher.parameters()
    )
    if len(source) != 2 or any(value.ndim != 2 for value in source):
        raise RuntimeError("Expected exactly two rank-two frozen ReLU weights.")
    maxima = tuple(float(value.abs().max().item()) for value in source)
    if any(not math.isfinite(value) or value <= 0.0 for value in maxima):
        raise RuntimeError("Frozen ReLU weights have invalid layer absmax.")
    normalized = tuple(
        (value.to(torch.float64) / maximum).to(torch.float32)
        for value, maximum in zip(source, maxima)
    )
    if any(
        not bool(torch.all(torch.isfinite(value)))
        or bool(torch.any(value < -1.0))
        or bool(torch.any(value > 1.0))
        for value in normalized
    ):
        raise RuntimeError("Normalized ReLU initializer left [-1,1].")
    return (
        tuple(torch.nn.Parameter(value.to(device)) for value in normalized),
        maxima,
        tuple(_tensor_sha256(value) for value in source),
    )


def _one_forward_gradient(
    *,
    stack: Any,
    inputs: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    masters: Sequence[torch.Tensor],
    templates: Sequence[WinsorizedQatLayerTemplate],
    full_conductance: Sequence[torch.Tensor],
) -> tuple[float, tuple[torch.Tensor, ...], Mapping[str, float | int]]:
    """Evaluate one full-G state and lift its BPTT gradient through the STE."""

    _apply_full_g(stack, full_conductance)
    stack.network.set_input(inputs, reset=True)
    stack.minimizer.compute_equilibrium()
    stack.cost.set_teacher(teacher_logits, labels)
    with torch.no_grad():
        per_example_kl = stack.cost.eval()
        prediction = stack.cost.student_logits().argmax(dim=1)
        teacher_prediction = teacher_logits.argmax(dim=1)
        loss = float(per_example_kl.to(torch.float64).mean().item())
        metrics: Mapping[str, float | int] = {
            "examples": int(labels.numel()),
            "kl_teacher_student": loss,
            "correct": int(prediction.eq(labels).sum().item()),
            "teacher_agreement": int(
                prediction.eq(teacher_prediction).sum().item()
            ),
        }
    physical = tuple(stack.differentiator.compute_gradient())
    if any(not bool(torch.all(torch.isfinite(value))) for value in physical):
        raise FloatingPointError("A source-HWA physical gradient is non-finite.")
    logical = logical_master_gradients(masters, physical, templates)
    return loss, logical, metrics


def _mean2_objective(ideal_loss: float, persistent_losses: Sequence[float]) -> float:
    weights = gradient_mixture_weights(
        "mean2", persistent_losses, ideal_loss=ideal_loss
    )
    result = weights.ideal_weight * float(ideal_loss) + sum(
        weight * float(loss)
        for weight, loss in zip(
            weights.persistent_weights, persistent_losses, strict=True
        )
    )
    if not math.isfinite(result):
        raise FloatingPointError("Mean-two HWA objective became non-finite.")
    return result


def _train_source_epoch(
    *,
    stack: Any,
    teacher: Any,
    loader: Iterable,
    masters: Sequence[torch.nn.Parameter],
    templates: Sequence[WinsorizedQatLayerTemplate],
    endpoint_bank: Any,
    optimizer: torch.optim.Optimizer,
    global_minibatch_ordinal: int,
    maximum_batches: int | None,
) -> tuple[Mapping[str, Any], int]:
    """Run one exact mean-two endpoint-ensemble source-HWA epoch."""

    totals = {
        "examples": 0,
        "batches": 0,
        "objective_kl_sum": 0.0,
        "ideal_kl_sum": 0.0,
        "ideal_correct": 0,
        "persistent_kl_sum": 0.0,
        "persistent_correct": 0,
    }
    gradient_squared = [0.0, 0.0]
    gradient_values = [0, 0]
    start = int(global_minibatch_ordinal)
    for inputs, labels in limited(loader, maximum_batches):
        inputs = inputs.to(stack.device, dtype=torch.float32)
        labels = labels.to(stack.device, dtype=torch.long)
        with torch.no_grad():
            teacher_logits = teacher.logits(inputs)
            ideal_view = quantize_winsorized_logical_master(
                masters, templates, include_report=False
            )
        ideal_loss, ideal_gradients, ideal_metrics = _one_forward_gradient(
            stack=stack,
            inputs=inputs,
            teacher_logits=teacher_logits,
            labels=labels,
            masters=masters,
            templates=templates,
            full_conductance=ideal_view.full_conductance_targets,
        )
        sample_indices = _sample_indices("mean2", global_minibatch_ordinal)
        persistent_losses: list[float] = []
        persistent_gradients: list[tuple[torch.Tensor, ...]] = []
        persistent_metrics: list[Mapping[str, float | int]] = []
        for full_g in endpoint_bank.full_conductance_views(
            ideal_view.requested_index, sample_indices=sample_indices
        ):
            loss, gradients, metrics = _one_forward_gradient(
                stack=stack,
                inputs=inputs,
                teacher_logits=teacher_logits,
                labels=labels,
                masters=masters,
                templates=templates,
                full_conductance=full_g,
            )
            persistent_losses.append(loss)
            persistent_gradients.append(gradients)
            persistent_metrics.append(metrics)
        if len(persistent_losses) != 2:
            raise RuntimeError("Mean-two HWA did not materialize two endpoints.")
        mixed = mix_logical_gradients(
            "mean2",
            persistent_losses,
            persistent_gradients,
            ideal_loss=ideal_loss,
            ideal_gradients=ideal_gradients,
        )
        objective = _mean2_objective(ideal_loss, persistent_losses)
        optimizer.zero_grad(set_to_none=True)
        for layer, (master, gradient) in enumerate(zip(masters, mixed)):
            master.grad = gradient
            gradient_squared[layer] += float(
                gradient.detach().to(torch.float64).square().sum().item()
            )
            gradient_values[layer] += int(gradient.numel())
        optimizer.step()
        with torch.no_grad():
            for master in masters:
                master.clamp_(-1.0, 1.0)

        count = int(labels.numel())
        totals["examples"] += count
        totals["batches"] += 1
        totals["objective_kl_sum"] += objective * count
        totals["ideal_kl_sum"] += ideal_loss * count
        totals["ideal_correct"] += int(ideal_metrics["correct"])
        totals["persistent_kl_sum"] += sum(persistent_losses) * count
        totals["persistent_correct"] += sum(
            int(value["correct"]) for value in persistent_metrics
        )
        global_minibatch_ordinal += 1

    examples = int(totals["examples"])
    batches = int(totals["batches"])
    if examples < 1 or batches < 1:
        raise RuntimeError("Expected source HWA to process at least one minibatch.")
    return {
        "arm": "mean2",
        "examples": examples,
        "batches": batches,
        "global_minibatch_ordinal_start": start,
        "global_minibatch_ordinal_end": int(global_minibatch_ordinal),
        "objective_weights": {"ideal": 0.25, "persistent": [0.375, 0.375]},
        "objective_kl_teacher_student": totals["objective_kl_sum"] / examples,
        "ideal_kl_teacher_student": totals["ideal_kl_sum"] / examples,
        "ideal_student_accuracy": totals["ideal_correct"] / examples,
        "persistent_mean_kl_teacher_student": (
            totals["persistent_kl_sum"] / (2 * examples)
        ),
        "persistent_mean_student_accuracy": (
            totals["persistent_correct"] / (2 * examples)
        ),
        "logical_gradient_rms": [
            math.sqrt(total / count)
            for total, count in zip(gradient_squared, gradient_values)
        ],
    }, int(global_minibatch_ordinal)


def _evaluate_endpoint_bank(
    *,
    stack: Any,
    teacher: Any,
    loader: Iterable,
    masters: Sequence[torch.Tensor],
    templates: Sequence[WinsorizedQatLayerTemplate],
    endpoint_bank: Any,
) -> Mapping[str, Any]:
    """Evaluate ideal code and every persistent view in a frozen endpoint bank."""

    # Saved epoch snapshots are deliberately CPU tensors.  Materialize them
    # on the bank/stack device before lookup so initial and selected replay use
    # exactly the same CUDA table path as training.
    selected_masters = tuple(
        value.to(device=stack.device, dtype=torch.float32) for value in masters
    )
    view = quantize_winsorized_logical_master(selected_masters, templates)
    _apply_full_g(stack, view.full_conductance_targets)
    ideal, ideal_prediction = _evaluate_detailed(
        stack, teacher, loader, sample_limit=None
    )
    repeats = []
    for endpoint_seed, full_g in zip(
        tuple(endpoint_bank.endpoint_seeds),
        endpoint_bank.full_conductance_views(view.requested_index),
        strict=True,
    ):
        _apply_full_g(stack, full_g)
        metrics, prediction = _evaluate_detailed(
            stack, teacher, loader, sample_limit=None
        )
        repeats.append(
            {
                "endpoint_seed": int(endpoint_seed),
                "metrics": dict(metrics),
                "prediction_sha256": _tensor_sha256(prediction.to(torch.int64)),
            }
        )
    accuracies = [float(item["metrics"]["student_accuracy"]) for item in repeats]
    divergences = [
        float(item["metrics"]["kl_teacher_student"]) for item in repeats
    ]
    return {
        "assignment_seed": int(endpoint_bank.assignment_seed),
        "population_fingerprint": str(endpoint_bank.population_fingerprint),
        "ideal": dict(ideal),
        "ideal_prediction_sha256": _tensor_sha256(
            ideal_prediction.to(torch.int64)
        ),
        "persistent_repeats": repeats,
        "persistent_mean_accuracy": fmean(accuracies),
        "persistent_minimum_accuracy": min(accuracies),
        "persistent_population_std_accuracy": pstdev(accuracies),
        "persistent_mean_kl_teacher_student": fmean(divergences),
        "persistent_maximum_kl_teacher_student": max(divergences),
        "persistent_full_conductance_formula": "G=2*x_persistent=a_persistent+1",
        "negative_conductance_count": 0,
        "apparent_endpoint_applied_to_drn": False,
    }


def _development_key(report: Mapping[str, Any], epoch: int) -> tuple[float, float, float, int]:
    """Select by persistent mean, minimum, KL, then earlier epoch."""

    return (
        float(report["persistent_mean_accuracy"]),
        float(report["persistent_minimum_accuracy"]),
        -float(report["persistent_mean_kl_teacher_student"]),
        -int(epoch),
    )


def _select_trained_hwa_epoch(
    candidates: Sequence[tuple[int, Mapping[str, Any]]],
) -> int:
    """Select among trained epochs only; epoch zero is a matched control."""

    selected = tuple(candidates)
    if (
        not selected
        or len({int(epoch) for epoch, _report in selected}) != len(selected)
        or any(int(epoch) < 1 for epoch, _report in selected)
    ):
        raise ValueError("HWA checkpoint candidates must be unique trained epochs >=1.")
    return max(
        selected,
        key=lambda item: _development_key(item[1], int(item[0])),
    )[0]


def _snapshot_masters(
    masters: Sequence[torch.Tensor],
) -> tuple[torch.Tensor, ...]:
    result = tuple(
        value.detach().cpu().to(torch.float32).clone() for value in masters
    )
    if len(result) != 2:
        raise ValueError("Expected two logical masters.")
    return result


def _copy_masters_(
    masters: Sequence[torch.Tensor], values: Sequence[torch.Tensor]
) -> None:
    if len(tuple(masters)) != 2 or len(tuple(values)) != 2:
        raise ValueError("Expected two logical masters and two selected values.")
    with torch.no_grad():
        for master, value in zip(masters, values):
            if master.shape != value.shape:
                raise ValueError("Selected logical-master shape changed.")
            master.copy_(value.to(device=master.device, dtype=master.dtype))


def _hwa_checkpoint_payload(
    *,
    epoch: int,
    masters: Sequence[torch.Tensor],
    source_absmax: Sequence[float],
    development: Mapping[str, Any],
    teacher_sha256: str,
    source_assignment_seed: int,
    source_published_fingerprint: str,
    source_repaired_fingerprint: str,
    global_minibatch_ordinal: int,
) -> Mapping[str, Any]:
    values = _snapshot_masters(masters)
    return {
        "schema": HWA_CHECKPOINT_SCHEMA,
        "schema_version": HWA_CHECKPOINT_SCHEMA_VERSION,
        "evidence_tier": EVIDENCE_TIER,
        "selected_epoch": int(epoch),
        "teacher_sha256": teacher_sha256,
        "source_absmax": list(map(float, source_absmax)),
        "source_assignment_seed": int(source_assignment_seed),
        "source_published_population_fingerprint": source_published_fingerprint,
        "source_repaired_mapping_twin_fingerprint": source_repaired_fingerprint,
        "logical_master": values,
        "logical_master_sha256": tuple(_tensor_sha256(value) for value in values),
        "checkpoint_role": "selected_logical_master_only_not_optimizer_resume",
        "optimizer_continuation_state": None,
        "selection_development": dict(development),
        "global_minibatch_ordinal": int(global_minibatch_ordinal),
        "target_assignment_opened": False,
    }


def _bank_state(bank: Any) -> Mapping[str, Any]:
    """Serialize the minimum exact tensor state needed to audit an endpoint bank."""

    if hasattr(bank, "state_dict"):
        state = bank.state_dict()
        if isinstance(state, Mapping):
            return dict(state)
    result: dict[str, Any] = {
        "schema": ENDPOINT_BANK_SCHEMA,
        "schema_version": ENDPOINT_BANK_SCHEMA_VERSION,
        "assignment_seed": int(bank.assignment_seed),
        "population_fingerprint": str(bank.population_fingerprint),
        "endpoint_seeds": list(map(int, bank.endpoint_seeds)),
    }
    for name in (
        "binding_shapes",
        "baseline_raw_x",
        "maximum_index",
        "persistent_raw_x",
        "requested_raw_x",
        "persistent_full_g",
        "report",
    ):
        if hasattr(bank, name):
            value = getattr(bank, name)
            if isinstance(value, torch.Tensor):
                value = value.detach().cpu().clone()
            result[name] = value
    if hasattr(bank, "build_report"):
        result["build_report"] = dict(bank.build_report)
    return result


def _save_endpoint_bank(path: Path, bank: Any) -> Path:
    return atomic_torch_save(_bank_state(bank), path)


def _assert_corrupt_immobile(
    before_native: torch.Tensor,
    after_native: torch.Tensor,
    corrupt_mask: torch.Tensor,
    *,
    context: str,
) -> Mapping[str, Any]:
    """Prove that every AIHWKit corrupt coordinate stayed at its stuck value."""

    before = before_native.detach().cpu().to(torch.float32).reshape(-1)
    after = after_native.detach().cpu().to(torch.float32).reshape(-1)
    mask = corrupt_mask.detach().cpu().to(torch.bool).reshape(-1)
    if before.shape != after.shape or before.shape != mask.shape:
        raise ValueError("Corrupt immobility tensors do not cover the same cells.")
    difference = (after[mask] - before[mask]).abs()
    maximum = float(difference.max().item()) if difference.numel() else 0.0
    changed = int((difference != 0.0).sum().item())
    if changed:
        raise RuntimeError(
            f"Published corrupt cells moved during {context}: changed={changed}, "
            f"maximum={maximum}."
        )
    return {
        "context": context,
        "corrupt_cells": int(mask.sum().item()),
        "changed_corrupt_cells": changed,
        "maximum_absolute_corrupt_drift_raw_a": maximum,
        "stuck_cells_immobile": True,
    }


def _assert_p0_roundtrip(
    expected: Mapping[str, Any], observed: Mapping[str, Any]
) -> Mapping[str, Any]:
    """Verify that recovery will consume the exact serialized P0 state."""

    scalar_fields = (
        "schema",
        "schema_version",
        "assignment_seed",
        "endpoint_seed",
        "target_kind",
        "maximum_program_pulses",
        "maximum_random_draws",
    )
    if any(expected.get(name) != observed.get(name) for name in scalar_fields):
        raise RuntimeError("Saved P0 scalar identity changed during round trip.")
    expected_state = expected.get("continuation_state")
    observed_state = observed.get("continuation_state")
    if not isinstance(expected_state, Mapping) or not isinstance(observed_state, Mapping):
        raise RuntimeError("Saved P0 lacks a continuation state after round trip.")
    state_scalars = (
        "schema",
        "schema_version",
        "state_coordinate",
        "preset",
        "rng_backend",
        "maximum_random_draws",
    )
    if any(expected_state.get(name) != observed_state.get(name) for name in state_scalars):
        raise RuntimeError("Saved P0 continuation metadata changed during round trip.")
    for name in (
        "construction_seeds",
        "persistent",
        "apparent",
        "draw_indices",
    ):
        left = expected_state.get(name)
        right = observed_state.get(name)
        if isinstance(left, torch.Tensor) or isinstance(right, torch.Tensor):
            if not isinstance(left, torch.Tensor) or not isinstance(right, torch.Tensor):
                raise RuntimeError(f"Saved P0 field {name!r} changed type.")
            if not torch.equal(left.detach().cpu(), right.detach().cpu()):
                raise RuntimeError(f"Saved P0 field {name!r} changed value.")
        elif left != right:
            raise RuntimeError(f"Saved P0 field {name!r} changed value.")
    if expected_state.get("seeds") != observed_state.get("seeds"):
        raise RuntimeError("Saved P0 trajectory seeds changed during round trip.")
    expected_target = expected.get("target_raw_x")
    observed_target = observed.get("target_raw_x")
    if (
        not isinstance(expected_target, torch.Tensor)
        or not isinstance(observed_target, torch.Tensor)
        or not torch.equal(expected_target.detach().cpu(), observed_target.detach().cpu())
    ):
        raise RuntimeError("Saved P0 requested target changed during round trip.")
    return {
        "scalar_identity_equal": True,
        "target_raw_x_equal": True,
        "persistent_equal": True,
        "apparent_equal": True,
        "rng_continuation_equal": True,
        "recovery_consumes_reloaded_artifact": True,
    }


def _population_report(population: Any) -> Mapping[str, Any]:
    return {
        "assignment_seed": int(population.assignment_seed),
        "corruption_policy": str(population.corruption_policy),
        "population_fingerprint": str(population.fingerprint),
        "aihwkit_version": str(population.aihwkit_version),
        "cells": int(population.size),
        "published_corrupt_cells": int(population.published_corrupt.sum().item()),
        "active_corrupt_cells": int(population.corrupt.sum().item()),
        "corrupt_fraction": float(population.corrupt.float().mean().item()),
        "nominal_dw_min": float(population.nominal_dw_min),
        "dw_min_std": float(population.dw_min_std),
        "write_noise_std": float(population.write_noise_std),
        "circuit_coordinate": "G=2*x=a+1",
    }


def _assert_distinct_array_pair(source_pair: Any, target_pair: Any) -> Mapping[str, Any]:
    """Fail closed unless source and target are distinct physical draws."""

    source_published = source_pair.published
    target_published = target_pair.published
    source_repaired = source_pair.repaired
    target_repaired = target_pair.repaired
    source_seeds = tuple(map(int, source_published.binding_sampling_seeds))
    target_seeds = tuple(map(int, target_published.binding_sampling_seeds))
    source_donors = tuple(map(int, source_published.donor_sampling_seeds))
    target_donors = tuple(map(int, target_published.donor_sampling_seeds))
    same_layout = (
        source_published.binding_keys == target_published.binding_keys
        and source_published.binding_shapes == target_published.binding_shapes
    )
    if not same_layout:
        raise RuntimeError("Source and target arrays do not share the DRN binding layout.")
    if (
        int(source_pair.assignment_seed) == int(target_pair.assignment_seed)
        or source_published.fingerprint == target_published.fingerprint
        or source_repaired.fingerprint == target_repaired.fingerprint
        or len(source_seeds) != len(target_seeds)
        or any(left == right for left, right in zip(source_seeds, target_seeds))
        or len(source_donors) != len(target_donors)
        or any(left == right for left, right in zip(source_donors, target_donors))
    ):
        raise RuntimeError("Target assignment is not a distinct physical OM array draw.")
    return {
        "source_assignment_seed": int(source_pair.assignment_seed),
        "target_assignment_seed": int(target_pair.assignment_seed),
        "assignment_seeds_differ": True,
        "published_fingerprints_differ": True,
        "repaired_fingerprints_differ": True,
        "binding_construction_seeds_differ_coordinatewise": True,
        "donor_construction_seeds_differ_coordinatewise": True,
        "binding_layout_matches": True,
        "source_binding_construction_seeds": list(source_seeds),
        "target_binding_construction_seeds": list(target_seeds),
    }


def _deployment_protocol_adapter(protocol: Any) -> Any:
    """Adapt the new strict protocol to the proven deployment/recovery helpers."""

    target = protocol.target
    deployment = protocol.deployment
    recovery = protocol.recovery
    target_seed = int(_field(target, "assignment_seed"))
    p0_seed = int(
        _field(
            target,
            "fine_tune_endpoint_seed",
            _field(target, "p0_endpoint_seed", 93301),
        )
    )
    return SimpleNamespace(
        target=SimpleNamespace(
            assignment_seed=target_seed,
            fine_tune_endpoint_seed=p0_seed,
        ),
        deployment=SimpleNamespace(
            maximum_random_draws=int(_field(deployment, "maximum_random_draws")),
            maximum_program_pulses=int(
                _field(deployment, "maximum_program_pulses")
            ),
            verify_tolerance_raw_x=float(
                _field(deployment, "verify_tolerance_raw_x")
            ),
        ),
        recovery=SimpleNamespace(
            learning_rate_raw_x=float(_field(recovery, "learning_rate_raw_x")),
            beta1=float(_field(recovery, "beta1", 0.9)),
            beta2=float(_field(recovery, "beta2", 0.999)),
            epsilon=float(_field(recovery, "epsilon", 1e-8)),
            epochs=int(_field(recovery, "epochs", 1)),
            maximum_batches=_field(recovery, "maximum_batches", None),
            pulse_cap=int(_field(recovery, "pulse_cap")),
            pulse_selection_seed=int(_field(recovery, "pulse_selection_seed")),
            conceptual_column_phases_per_minibatch=tuple(
                _field(
                    recovery,
                    "conceptual_column_phases_per_minibatch",
                    (100, 20),
                )
            ),
            vectorization_equivalence=str(
                _field(
                    recovery,
                    "vectorization_equivalence",
                    "disjoint_cell_state_and_per_cell_RNG_equivalent",
                )
            ),
        ),
    )


def _endpoint_accuracies(reports: Sequence[Mapping[str, Any]]) -> Mapping[str, float]:
    values = tuple(float(item["test"]["student_accuracy"]) for item in reports)
    if not values:
        raise ValueError("Expected at least one deployment report.")
    return {
        "mean": fmean(values),
        "minimum": min(values),
        "maximum": max(values),
        "population_standard_deviation": pstdev(values),
    }


def _paired_target_hwa_gate(
    epoch0: Sequence[Mapping[str, Any]],
    selected_hwa: Sequence[Mapping[str, Any]],
    *,
    minimum_mean_gain: float = 0.02,
    minimum_wins: int = 4,
) -> Mapping[str, Any]:
    """Evaluate the predeclared paired target endpoint HWA retention gate."""

    before = tuple(epoch0)
    after = tuple(selected_hwa)
    if len(before) != len(after) or not before:
        raise ValueError("Expected matched nonempty epoch0 and HWA deployments.")
    rows = []
    for initial, trained in zip(before, after, strict=True):
        initial_seed = int(initial["endpoint_seed"])
        trained_seed = int(trained["endpoint_seed"])
        if initial_seed != trained_seed:
            raise ValueError("Target HWA gate endpoint seeds are not paired.")
        initial_accuracy = float(initial["test"]["student_accuracy"])
        trained_accuracy = float(trained["test"]["student_accuracy"])
        rows.append(
            {
                "endpoint_seed": initial_seed,
                "epoch0_accuracy": initial_accuracy,
                "selected_hwa_accuracy": trained_accuracy,
                "selected_minus_epoch0_accuracy": trained_accuracy - initial_accuracy,
                "win": trained_accuracy > initial_accuracy,
            }
        )
    deltas = [float(row["selected_minus_epoch0_accuracy"]) for row in rows]
    wins = sum(bool(row["win"]) for row in rows)
    mean_delta = fmean(deltas)
    return {
        "per_endpoint": rows,
        "mean_selected_minus_epoch0_accuracy": mean_delta,
        "wins": wins,
        "endpoints": len(rows),
        "minimum_mean_gain": float(minimum_mean_gain),
        "minimum_wins": int(minimum_wins),
        "passed": mean_delta >= float(minimum_mean_gain) and wins >= int(minimum_wins),
    }


def _recovery_gate(
    recovery: Mapping[str, Any],
    *,
    selected_source_persistent_mean_accuracy: float,
    minimum_gain: float = 0.10,
    maximum_source_gap: float = 0.02,
) -> Mapping[str, Any]:
    """Evaluate recovery accuracy and physical-safety requirements together."""

    before = float(recovery["before_test"]["student_accuracy"])
    after = float(recovery["after_test"]["student_accuracy"])
    source = float(selected_source_persistent_mean_accuracy)
    gain = after - before
    source_gap = after - source
    immobile = bool(recovery["corrupt_immobility"]["stuck_cells_immobile"])
    zero_verify = int(recovery["pulse"]["verify_reads_during_updates"]) == 0
    gain_passed = gain >= float(minimum_gain)
    source_gap_passed = abs(source_gap) <= float(maximum_source_gap)
    return {
        "pre_recovery_accuracy": before,
        "post_recovery_accuracy": after,
        "post_minus_pre_accuracy": gain,
        "minimum_gain": float(minimum_gain),
        "gain_passed": gain_passed,
        "selected_source_persistent_mean_accuracy": source,
        "post_minus_selected_source_accuracy": source_gap,
        "maximum_absolute_source_gap": float(maximum_source_gap),
        "source_gap_passed": source_gap_passed,
        "corrupt_cells_immobile": immobile,
        "zero_verify_reads": zero_verify,
        "physical_safety_passed": immobile and zero_verify,
        "passed": gain_passed and source_gap_passed and immobile and zero_verify,
    }


def _logical_master_hashes(values: Sequence[torch.Tensor]) -> tuple[str, ...]:
    return tuple(_tensor_sha256(value.detach().cpu().to(torch.float32)) for value in values)


def _sample_and_save_pair(
    *,
    bindings: Sequence[Any],
    assignment_seed: int,
    role: str,
    sampler: Path,
    artifact_root: Path,
) -> tuple[Any, list[Mapping[str, Any]]]:
    """Externally sample and persist both native and Winsorized paired views."""

    from experiments.mnist_relu_drn.ibm_om_corrupt_source_hwa_cross_array_pulse_adam import (
        sample_paired_winsorized_om_populations_external,
    )

    root = artifact_root / role / "identity"
    native = root / "native"
    native_published = native / "published_population.npz"
    native_published_receipt = native / "published_population.receipt.json"
    native_repaired = native / "repaired_population.npz"
    native_repaired_receipt = native / "repaired_population.receipt.json"
    pair, external_receipts = sample_paired_winsorized_om_populations_external(
        bindings,
        assignment_seed=int(assignment_seed),
        aihwkit_python=sampler,
        published_population_path=native_published,
        published_receipt_path=native_published_receipt,
        repaired_population_path=native_repaired,
        repaired_receipt_path=native_repaired_receipt,
    )
    winsorized = root / "winsorized"
    winsorized.mkdir(parents=True, exist_ok=True)
    winsorized_published = winsorized / "published_population.npz"
    winsorized_repaired = winsorized / "repaired_population.npz"
    save_om_array_population(winsorized_published, pair.published)
    save_om_array_population(winsorized_repaired, pair.repaired)
    pair_report = root / "paired_winsorized_assignment.json"
    report = {
        "schema": (
            "ebl.mnist_relu_drn.ibm_om_paired_winsorized_assignment_receipt"
        ),
        "schema_version": 1,
        "role": role,
        "assignment_seed": int(assignment_seed),
        "pair": dict(pair.report),
        "external_native": {
            "published": {
                "path": str(native_published),
                "sha256": sha256_file(native_published),
                "receipt_path": str(native_published_receipt),
                "receipt_sha256": sha256_file(native_published_receipt),
                "receipt": dict(external_receipts["published"]),
            },
            "repaired": {
                "path": str(native_repaired),
                "sha256": sha256_file(native_repaired),
                "receipt_path": str(native_repaired_receipt),
                "receipt_sha256": sha256_file(native_repaired_receipt),
                "receipt": dict(external_receipts["repaired"]),
            },
        },
        "winsorized": {
            "published": {
                "path": str(winsorized_published),
                "sha256": sha256_file(winsorized_published),
                "population_fingerprint": pair.published.fingerprint,
            },
            "repaired": {
                "path": str(winsorized_repaired),
                "sha256": sha256_file(winsorized_repaired),
                "population_fingerprint": pair.repaired.fingerprint,
            },
        },
    }
    atomic_write_json(pair_report, report)
    artifacts = [
        {
            "path": str(native_published),
            "receipt": str(native_published_receipt),
            "kind": f"{role}_native_published_population",
        },
        {
            "path": str(native_repaired),
            "receipt": str(native_repaired_receipt),
            "kind": f"{role}_native_repaired_population",
        },
        {
            "path": str(winsorized_published),
            "kind": f"{role}_winsorized_published_population",
        },
        {
            "path": str(winsorized_repaired),
            "kind": f"{role}_winsorized_repaired_population",
        },
        {"path": str(pair_report), "kind": f"{role}_paired_assignment_receipt"},
    ]
    return pair, artifacts


def _evaluate_full_g(
    *,
    stack: Any,
    teacher: Any,
    loader: Iterable,
    full_g: Sequence[torch.Tensor],
) -> Mapping[str, Any]:
    _apply_full_g(stack, full_g)
    metrics, prediction = _evaluate_detailed(
        stack, teacher, loader, sample_limit=None
    )
    return {
        "metrics": dict(metrics),
        "prediction_sha256": _tensor_sha256(prediction.to(torch.int64)),
        "full_conductance_formula": "G=2*x=a+1",
        "negative_conductance_count": 0,
    }


def _require_protocol_contract(protocol: Any) -> Mapping[str, Any]:
    """Resolve names shared by the strict config and assert the fixed ladder."""

    source = protocol.source
    target = protocol.target
    mapping = protocol.mapping
    hwa = getattr(protocol, "hwa", getattr(protocol, "training", None))
    if hwa is None:
        raise AttributeError("Protocol lacks an HWA/training section.")
    source_training = tuple(
        map(
            int,
            _field(
                source,
                "training_endpoint_seeds",
                _field(
                    source,
                    "train_endpoint_seeds",
                    _field(
                        hwa,
                        "training_endpoint_seeds",
                        (93201, 93202, 93203, 93204),
                    ),
                ),
            ),
        )
    )
    source_selection = tuple(
        map(
            int,
            _field(
                source,
                "selection_endpoint_seeds",
                _field(
                    hwa,
                    "selection_endpoint_seeds",
                    (93201, 93202, 93203, 93204),
                ),
            ),
        )
    )
    target_endpoints = tuple(map(int, _field(target, "endpoint_seeds")))
    p0 = int(
        _field(
            target,
            "fine_tune_endpoint_seed",
            _field(target, "p0_endpoint_seed", 93301),
        )
    )
    if isinstance(protocol, Mapping):
        teacher_sha256 = protocol.get("expected_teacher_weights_sha256")
    else:
        teacher_sha256 = getattr(protocol, "expected_teacher_weights_sha256", None)
    if teacher_sha256 is None:
        teacher_sha256 = _field(source, "expected_teacher_weights_sha256")
    result = {
        "teacher_sha256": str(teacher_sha256),
        "source_assignment_seed": int(_field(source, "assignment_seed")),
        "source_training_endpoint_seeds": source_training,
        "source_selection_endpoint_seeds": source_selection,
        "target_assignment_seed": int(_field(target, "assignment_seed")),
        "target_endpoint_seeds": target_endpoints,
        "p0_endpoint_seed": p0,
        "level_spacing_raw_x": float(
            _field(
                mapping,
                "level_spacing_raw_x",
                int(_field(mapping, "spacing_delta_x_multiplier", 1))
                * NOMINAL_DELTA_X,
            )
        ),
        "fixed_logit_gain": float(_field(mapping, "fixed_logit_gain")),
        "hwa_epochs": int(_field(hwa, "epochs")),
        "hwa_learning_rates": tuple(map(float, _field(hwa, "learning_rates"))),
        "hwa_momentum": float(_field(hwa, "momentum", 0.0)),
        "hwa_weight_decay": float(_field(hwa, "weight_decay", 0.0)),
        "hwa_maximum_batches": _field(hwa, "maximum_batches", None),
        "hwa_maximum_program_pulses": int(
            _field(
                hwa,
                "maximum_program_pulses",
                _field(protocol.deployment, "maximum_program_pulses"),
            )
        ),
        "hwa_verify_tolerance_raw_x": float(
            _field(
                hwa,
                "verify_tolerance_raw_x",
                _field(protocol.deployment, "verify_tolerance_raw_x"),
            )
        ),
    }
    if (
        result["source_assignment_seed"] != 93001
        or result["target_assignment_seed"] != 93002
        or len(source_training) != 4
        or len(source_selection) != 4
        or len(set(source_training)) != 4
        or len(set(source_selection)) != 4
        or set(source_training) & set(source_selection)
        or target_endpoints != (93301, 93302, 93303, 93304, 93305)
        or p0 != 93301
        or result["hwa_epochs"] != 3
        or len(result["hwa_learning_rates"]) != 2
        or not math.isclose(
            result["level_spacing_raw_x"], NOMINAL_DELTA_X, abs_tol=1e-15
        )
    ):
        raise ValueError("Resolved protocol does not match the fixed standard ladder.")
    return result


def run_train(request: "TrainRequest") -> int:
    """Execute the complete source-HWA / target-transfer / pulse-recovery ladder."""

    from experiments.mnist_relu_drn.ibm_om_corrupt_source_hwa_cross_array_pulse_adam import (
        build_corrupt_persistent_endpoint_codebook_ensemble,
        build_published_persistent_targets,
        build_repaired_mapping_templates,
        clamp_targets_to_published_corrupt_singletons,
    )
    from experiments.mnist_relu_drn.ibm_om_corrupt_source_hwa_cross_array_pulse_adam_config import (
        CorruptSourceHwaCrossArrayPulseAdamTrainSpec,
    )

    spec = request.spec
    if not isinstance(spec, CorruptSourceHwaCrossArrayPulseAdamTrainSpec):
        raise TypeError("Expected the dedicated corrupt-source ladder train spec.")
    if request.teacher_weights is None:
        raise ValueError("Expected explicit --teacher-weights for the ReLU initializer.")
    for name in ("weights", "base_weights", "resume", "device_data", "device_model"):
        if getattr(request, name) is not None:
            raise ValueError(
                "The corrupt-source ladder accepts only --teacher-weights; "
                f"unexpected {name}."
            )
    protocol = spec.protocol
    contract = _require_protocol_contract(protocol)
    teacher_path = request.teacher_weights.expanduser().resolve()
    if sha256_file(teacher_path) != contract["teacher_sha256"]:
        raise ValueError("Frozen ReLU teacher/source checkpoint SHA-256 mismatch.")
    sampler = _sampler_python()
    store = RunStore.create(
        output_root=request.output_dir,
        experiment_id=spec.experiment_id,
        resolved_config=to_plain_data(spec),
        command=request.command,
        repo_root=_ROOT,
        input_artifacts=(
            _input("teacher_weights", teacher_path),
            _input("aihwkit_python", sampler),
        ),
        resume_capability="unsupported",
    )
    try:
        student = spec.student
        if student.runtime.device != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("The corrupt-source standard ladder requires local CUDA.")
        torch.manual_seed(student.runtime.seed)
        torch.cuda.manual_seed_all(student.runtime.seed)
        device = torch.device("cuda")
        loaders = build_mnist_loaders(
            student.data,
            data_seed=student.runtime.data_seed,
            calibration_examples=student.mapping.calibration_examples,
            calibration_batch_size=student.mapping.calibration_batch_size,
        )
        teacher, teacher_metadata = _load_teacher(
            teacher_path, device=device, spec=student
        )
        stack = build_student_stack(student, enable_measured=False)
        stack.cost.gain = float(contract["fixed_logit_gain"])
        masters, source_absmax, source_weight_hashes = _initial_logical_masters(
            teacher, device=device
        )
        initial_master = _snapshot_masters(masters)
        artifact_root = store.run_dir / "artifacts"
        generated_artifacts: list[Mapping[str, Any]] = []

        # Source sampling and all HWA/selection complete before target 93002 is
        # even sampled.  This ordering is an experimental isolation gate.
        source_pair, source_pair_artifacts = _sample_and_save_pair(
            bindings=stack.bundle.catalog.trainable,
            assignment_seed=int(contract["source_assignment_seed"]),
            role="source_93001",
            sampler=sampler,
            artifact_root=artifact_root,
        )
        generated_artifacts.extend(source_pair_artifacts)
        source_templates_cpu = build_repaired_mapping_templates(
            initial_master,
            source_pair,
            level_spacing_raw_x=float(contract["level_spacing_raw_x"]),
        )
        source_templates = tuple(value.to(device) for value in source_templates_cpu)
        print(
            "building source-93001 corrupt persistent HWA bank "
            f"seeds={list(contract['source_training_endpoint_seeds'])}",
            flush=True,
        )
        source_training_bank_cpu = (
            build_corrupt_persistent_endpoint_codebook_ensemble(
                source_pair,
                source_templates_cpu,
                endpoint_seeds=contract["source_training_endpoint_seeds"],
                tolerance_raw_x=float(contract["hwa_verify_tolerance_raw_x"]),
                maximum_program_pulses=int(
                    contract["hwa_maximum_program_pulses"]
                ),
                device=device,
            )
        )
        source_selection_bank_cpu = (
            build_corrupt_persistent_endpoint_codebook_ensemble(
                source_pair,
                source_templates_cpu,
                endpoint_seeds=contract["source_selection_endpoint_seeds"],
                tolerance_raw_x=float(contract["hwa_verify_tolerance_raw_x"]),
                maximum_program_pulses=int(
                    contract["hwa_maximum_program_pulses"]
                ),
                device=device,
            )
        )
        training_bank_path = _save_endpoint_bank(
            artifact_root / "source_93001/endpoint_banks/training.pt",
            source_training_bank_cpu,
        )
        selection_bank_path = _save_endpoint_bank(
            artifact_root / "source_93001/endpoint_banks/selection.pt",
            source_selection_bank_cpu,
        )
        generated_artifacts.extend(
            (
                {"path": str(training_bank_path), "kind": "source_hwa_endpoint_bank"},
                {"path": str(selection_bank_path), "kind": "source_selection_endpoint_bank"},
            )
        )
        source_training_bank = source_training_bank_cpu.to(device)
        source_selection_bank = source_selection_bank_cpu.to(device)
        optimizer = torch.optim.SGD(
            [
                {"params": [master], "lr": float(rate)}
                for master, rate in zip(
                    masters, contract["hwa_learning_rates"], strict=True
                )
            ],
            momentum=float(contract["hwa_momentum"]),
            weight_decay=float(contract["hwa_weight_decay"]),
        )
        initial_development = _evaluate_endpoint_bank(
            stack=stack,
            teacher=teacher,
            loader=loaders.validation,
            masters=masters,
            templates=source_templates,
            endpoint_bank=source_selection_bank,
        )
        store.append_metric(
            {"mode": "source_hwa", "epoch": 0, "development": initial_development}
        )
        history: list[Mapping[str, Any]] = []
        master_by_epoch: dict[int, tuple[torch.Tensor, ...]] = {}
        ordinal_by_epoch: dict[int, int] = {}
        global_ordinal = 0
        for epoch in range(1, int(contract["hwa_epochs"]) + 1):
            train_report, global_ordinal = _train_source_epoch(
                stack=stack,
                teacher=teacher,
                loader=loaders.train,
                masters=masters,
                templates=source_templates,
                endpoint_bank=source_training_bank,
                optimizer=optimizer,
                global_minibatch_ordinal=global_ordinal,
                maximum_batches=(
                    contract["hwa_maximum_batches"]
                    if contract["hwa_maximum_batches"] is not None
                    else student.settings.max_batches
                ),
            )
            development = _evaluate_endpoint_bank(
                stack=stack,
                teacher=teacher,
                loader=loaders.validation,
                masters=masters,
                templates=source_templates,
                endpoint_bank=source_selection_bank,
            )
            record = {
                "mode": "source_hwa",
                "epoch": epoch,
                "train": train_report,
                "development": development,
            }
            history.append(record)
            store.append_metric(record)
            master_by_epoch[epoch] = _snapshot_masters(masters)
            ordinal_by_epoch[epoch] = int(global_ordinal)
            print(
                f"source HWA epoch={epoch}/{contract['hwa_epochs']} "
                f"dev_persistent_mean={100*float(development['persistent_mean_accuracy']):.2f}%",
                flush=True,
            )
        best_epoch = _select_trained_hwa_epoch(
            tuple(
                (int(record["epoch"]), record["development"])
                for record in history
            )
        )
        best_record = next(
            record for record in history if int(record["epoch"]) == best_epoch
        )
        best_development = dict(best_record["development"])
        best_master = master_by_epoch[best_epoch]
        best_ordinal = ordinal_by_epoch[best_epoch]
        hwa_underperformed_epoch0 = (
            _development_key(best_development, best_epoch)[:3]
            < _development_key(initial_development, 0)[:3]
        )
        _copy_masters_(masters, best_master)
        selected_checkpoint = atomic_torch_save(
            _hwa_checkpoint_payload(
                epoch=best_epoch,
                masters=masters,
                source_absmax=source_absmax,
                development=best_development,
                teacher_sha256=sha256_file(teacher_path),
                source_assignment_seed=int(contract["source_assignment_seed"]),
                source_published_fingerprint=source_pair.published.fingerprint,
                source_repaired_fingerprint=source_pair.repaired.fingerprint,
                global_minibatch_ordinal=best_ordinal,
            ),
            store.run_dir / "checkpoints/source_93001_selected_hwa_master.pt",
        )
        generated_artifacts.append(
            {"path": str(selected_checkpoint), "kind": "selected_source_hwa_master"}
        )
        # Test is opened only after source validation selection is frozen.
        source_test = {
            "epoch_0": _evaluate_endpoint_bank(
                stack=stack,
                teacher=teacher,
                loader=loaders.test,
                masters=initial_master,
                templates=source_templates,
                endpoint_bank=source_selection_bank,
            ),
            "selected_hwa": _evaluate_endpoint_bank(
                stack=stack,
                teacher=teacher,
                loader=loaders.test,
                masters=masters,
                templates=source_templates,
                endpoint_bank=source_selection_bank,
            ),
        }

        # The first target-array action occurs only after the immutable source
        # checkpoint above exists and its hash is known.
        frozen_source_checkpoint_sha256 = sha256_file(selected_checkpoint)
        target_pair, target_pair_artifacts = _sample_and_save_pair(
            bindings=stack.bundle.catalog.trainable,
            assignment_seed=int(contract["target_assignment_seed"]),
            role="target_93002",
            sampler=sampler,
            artifact_root=artifact_root,
        )
        generated_artifacts.extend(target_pair_artifacts)
        distinct_array_gate = _assert_distinct_array_pair(source_pair, target_pair)
        target_templates_cpu = build_repaired_mapping_templates(
            initial_master,
            target_pair,
            level_spacing_raw_x=float(contract["level_spacing_raw_x"]),
        )
        initial_target = build_published_persistent_targets(
            initial_master, target_templates_cpu, target_pair
        )
        selected_target = build_published_persistent_targets(
            best_master, target_templates_cpu, target_pair
        )
        initial_fault_clamped = clamp_targets_to_published_corrupt_singletons(
            initial_target.target_full_conductance, target_pair
        )
        selected_fault_clamped = clamp_targets_to_published_corrupt_singletons(
            selected_target.target_full_conductance, target_pair
        )
        target_no_write = {
            "epoch_0": _evaluate_full_g(
                stack=stack,
                teacher=teacher,
                loader=loaders.test,
                full_g=initial_target.target_full_conductance,
            ),
            "selected_hwa": _evaluate_full_g(
                stack=stack,
                teacher=teacher,
                loader=loaders.test,
                full_g=selected_target.target_full_conductance,
            ),
        }
        target_fault_clamped_no_write = {
            "epoch_0": {
                "fault_realization": dict(initial_fault_clamped.report),
                "evaluation": _evaluate_full_g(
                    stack=stack,
                    teacher=teacher,
                    loader=loaders.test,
                    full_g=initial_fault_clamped.realized_full_conductance,
                ),
            },
            "selected_hwa": {
                "fault_realization": dict(selected_fault_clamped.report),
                "evaluation": _evaluate_full_g(
                    stack=stack,
                    teacher=teacher,
                    loader=loaders.test,
                    full_g=selected_fault_clamped.realized_full_conductance,
                ),
            },
        }
        runtime_protocol = _deployment_protocol_adapter(protocol)
        target_population = target_pair.published
        pulse_population = array_population_as_pulse_population(target_population)
        if (
            float(target_population.dw_min_std) <= 0.0
            or float(target_population.write_noise_std) <= 0.0
        ):
            raise RuntimeError("Target pulse plant unexpectedly disabled OM stochasticity.")
        target_initial_deployments: list[Mapping[str, Any]] = []
        target_selected_deployments: list[Mapping[str, Any]] = []
        p0_payload: dict[str, Any] | None = None
        for label, target, destination in (
            ("epoch_0", initial_target, target_initial_deployments),
            ("selected_hwa", selected_target, target_selected_deployments),
        ):
            for endpoint_seed in contract["target_endpoint_seeds"]:
                report, payload = _deploy_once(
                    name=f"{label}_published_seed_{endpoint_seed}",
                    endpoint_seed=int(endpoint_seed),
                    raw_x_targets=target.target_raw_x,
                    target_kind=(
                        f"target_93002_repaired_template_{label}_to_published_cells"
                    ),
                    population=target_population,
                    pulse_population=pulse_population,
                    stack=stack,
                    teacher=teacher,
                    test_loader=loaders.test,
                    protocol=runtime_protocol,
                    store=store,
                )
                immobility = _assert_corrupt_immobile(
                    target_population.min_bound,
                    payload["continuation_state"]["persistent"],
                    target_population.corrupt,
                    context=f"target_PV_{label}_seed_{endpoint_seed}",
                )
                report = {**report, "corrupt_immobility": immobility}
                destination.append(report)
                generated_artifacts.append(
                    {"path": str(report["path"]), "kind": f"target_{label}_deployment"}
                )
                if label == "selected_hwa" and int(endpoint_seed) == int(
                    contract["p0_endpoint_seed"]
                ):
                    p0_payload = dict(payload)
        if p0_payload is None:
            raise RuntimeError("The selected-HWA target P0 endpoint was not produced.")
        p0_checkpoint = atomic_torch_save(
            {
                **p0_payload,
                "p0_role": "exact_selected_hwa_target_93002_seed_93301_continuation",
                "source_hwa_checkpoint_sha256": frozen_source_checkpoint_sha256,
                "no_remap_after_p0": True,
            },
            store.run_dir / "checkpoints/target_93002_seed_93301_exact_p0.pt",
        )
        generated_artifacts.append(
            {"path": str(p0_checkpoint), "kind": "exact_target_p0_continuation"}
        )
        try:
            reloaded_p0 = torch.load(
                p0_checkpoint, map_location="cpu", weights_only=True
            )
        except (OSError, RuntimeError, TypeError, ValueError) as error:
            raise RuntimeError("Could not reload the exact saved target P0.") from error
        if not isinstance(reloaded_p0, Mapping):
            raise RuntimeError("Reloaded target P0 is not a mapping.")
        p0_roundtrip = _assert_p0_roundtrip(p0_payload, reloaded_p0)
        p0_payload = dict(reloaded_p0)
        p0_payload["artifact_path"] = str(p0_checkpoint)
        recovery = dict(
            _fine_tune_open_loop(
                p0=p0_payload,
                population=target_population,
                pulse_population=pulse_population,
                stack=stack,
                teacher=teacher,
                loaders=loaders,
                protocol=runtime_protocol,
                store=store,
            )
        )
        recovery_checkpoint = Path(str(recovery["checkpoint"]))
        try:
            recovery_state = torch.load(
                recovery_checkpoint, map_location="cpu", weights_only=True
            )
        except (OSError, RuntimeError, TypeError, ValueError) as error:
            raise RuntimeError("Could not audit the saved pulse-Adam continuation.") from error
        recovery_immobility = _assert_corrupt_immobile(
            p0_payload["continuation_state"]["persistent"],
            recovery_state["plant_continuation_state"]["persistent"],
            target_population.corrupt,
            context="target_open_loop_pulse_adam",
        )
        corrupt_mask = target_population.corrupt.detach().cpu()
        pulse_count = recovery_state["pulse_count"].detach().cpu()
        recovery["corrupt_immobility"] = recovery_immobility
        recovery["pulse_attempts_by_fault_status"] = {
            "corrupt": int(pulse_count[corrupt_mask].sum().item()),
            "healthy": int(pulse_count[~corrupt_mask].sum().item()),
            "semantic": (
                "selected pulse commands may address stuck cells, but AIHWKit "
                "zero-step collapsed devices cannot move"
            ),
        }
        generated_artifacts.append(
            {"path": str(recovery_checkpoint), "kind": "open_loop_pulse_adam_continuation"}
        )

        initial_stats = _endpoint_accuracies(target_initial_deployments)
        selected_stats = _endpoint_accuracies(target_selected_deployments)
        target_hwa_gate = _paired_target_hwa_gate(
            target_initial_deployments, target_selected_deployments
        )
        recovery_gate = _recovery_gate(
            recovery,
            selected_source_persistent_mean_accuracy=float(
                source_test["selected_hwa"]["persistent_mean_accuracy"]
            ),
        )
        summary = {
            "schema": SUMMARY_SCHEMA,
            "schema_version": SUMMARY_SCHEMA_VERSION,
            "status": "complete",
            "evidence_tier": EVIDENCE_TIER,
            "claim_boundary": (
                "Exploratory model-based AIHWKit-1.1.0 IBM-OM ladder. The OM "
                "preset is a fitted hardware-derived model, not replay of raw "
                "measured traces. HWA and target recovery use teacher-logit BPTT; "
                "recovery additionally keeps digital Adam moments. Persistent "
                "full G only is evaluated, with no inference read noise or drift."
            ),
            "source_revision": store.manifest.get("source", {}),
            "teacher": {
                "path": str(teacher_path),
                "sha256": sha256_file(teacher_path),
                "metadata": teacher_metadata,
                "logical_weight_sha256": list(source_weight_hashes),
                "layer_absmax": list(source_absmax),
                "initialization": "relu_weight_divided_by_layer_absmax_fp32",
            },
            "protocol": to_plain_data(protocol),
            "source_93001": {
                "pair": dict(source_pair.report),
                "published_population": _population_report(source_pair.published),
                "repaired_mapping_twin": _population_report(source_pair.repaired),
                "mapping_oracle_only": "counterfactual_repaired",
                "physical_training_population": "published",
                "training_endpoint_seeds": list(
                    contract["source_training_endpoint_seeds"]
                ),
                "selection_endpoint_seeds": list(
                    contract["source_selection_endpoint_seeds"]
                ),
                "endpoint_banks_disjoint": True,
                "hwa": {
                    "arm": "mean2",
                    "epochs": int(contract["hwa_epochs"]),
                    "optimizer": "sgd",
                    "learning_rates": list(contract["hwa_learning_rates"]),
                    "objective_weights": {
                        "ideal": 0.25,
                        "persistent": [0.375, 0.375],
                    },
                    "initial_development": initial_development,
                    "selected_epoch": best_epoch,
                    "selected_development": best_development,
                    "epoch0_is_control_not_checkpoint_candidate": True,
                    "selected_hwa_underperformed_epoch0_control": (
                        hwa_underperformed_epoch0
                    ),
                    "history": history,
                },
                "selected_checkpoint": {
                    "path": str(selected_checkpoint),
                    "sha256": frozen_source_checkpoint_sha256,
                    "target_assignment_opened_before_freeze": False,
                },
                "test_after_selection_freeze": source_test,
            },
            "target_93002": {
                "opened_after_source_checkpoint_sha256": frozen_source_checkpoint_sha256,
                "pair": dict(target_pair.report),
                "published_population": _population_report(target_pair.published),
                "repaired_mapping_twin": _population_report(target_pair.repaired),
                "mapping": {
                    "epoch_0": dict(initial_target.report),
                    "selected_hwa": dict(selected_target.report),
                    "target_clipping": False,
                    "negative_conductance_count": 0,
                },
                "ideal_requested_no_write": target_no_write,
                "deterministic_fault_clamped_no_write": (
                    target_fault_clamped_no_write
                ),
                "program_verify": {
                    "initialization": "sampled_lower_RESET",
                    "controller": "one_pulse",
                    "endpoint_seeds": list(contract["target_endpoint_seeds"]),
                    "persistent_full_G_only": True,
                    "apparent_endpoint_applied_to_drn": False,
                    "epoch_0": {
                        "repeats": target_initial_deployments,
                        **initial_stats,
                    },
                    "selected_hwa": {
                        "repeats": target_selected_deployments,
                        **selected_stats,
                    },
                    "paired_hwa_gate": target_hwa_gate,
                },
                "p0": {
                    "endpoint_seed": int(contract["p0_endpoint_seed"]),
                    "path": str(p0_checkpoint),
                    "sha256": sha256_file(p0_checkpoint),
                    "exact_continuation": True,
                    "no_remap_after_p0": True,
                    "roundtrip": p0_roundtrip,
                },
                "open_loop_pulse_adam": recovery,
                "recovery_gate": recovery_gate,
            },
            "accuracy_ladder": {
                "relu_initialized_source_93001_persistent_mean_test_accuracy": float(
                    source_test["epoch_0"]["persistent_mean_accuracy"]
                ),
                "source_93001_selected_hwa_persistent_mean_test_accuracy": float(
                    source_test["selected_hwa"]["persistent_mean_accuracy"]
                ),
                "relu_initialized_target_93002_pv_mean_test_accuracy": float(
                    initial_stats["mean"]
                ),
                "hwa_selected_target_93002_pv_mean_test_accuracy": float(
                    selected_stats["mean"]
                ),
                "relu_initialized_target_93002_fault_clamped_no_write_test_accuracy": float(
                    target_fault_clamped_no_write["epoch_0"]["evaluation"][
                        "metrics"
                    ]["student_accuracy"]
                ),
                "hwa_selected_target_93002_fault_clamped_no_write_test_accuracy": float(
                    target_fault_clamped_no_write["selected_hwa"]["evaluation"][
                        "metrics"
                    ]["student_accuracy"]
                ),
                "target_93002_seed_93301_pre_adam_test_accuracy": float(
                    recovery["before_test"]["student_accuracy"]
                ),
                "target_93002_seed_93301_post_adam_test_accuracy": float(
                    recovery["after_test"]["student_accuracy"]
                ),
                "target_seed_93301_open_loop_adam_delta": float(
                    recovery["after_test"]["student_accuracy"]
                    - recovery["before_test"]["student_accuracy"]
                ),
            },
            "invariants": {
                "published_corrupt_mask_exposed_to_learner": False,
                "published_corrupt_physical_effects_present": True,
                "published_corrupt_cells_immobile": True,
                "deterministic_target_fault_only_rung_present": True,
                "full_conductance_formula": "G=2*x=a+1",
                "negative_conductance_count": 0,
                "target_opened_before_source_checkpoint_freeze": False,
                "source_target_distinct_array_gate": distinct_array_gate,
                "verify_reads_during_recovery": 0,
                "om_cycle_to_cycle_noise_retained": True,
                "om_apparent_write_noise_retained_during_pulses": True,
            },
            "limitations": [
                "exploratory_noncanonical_single_source_and_single_target_assignment",
                "hardware_derived_fitted_OM_model_not_raw_measured_trace_replay",
                "counterfactual_repaired_twins_are_mapping_oracles_only",
                "teacher_logit_BPTT_and_digital_optimizer_state",
                "column_serial_stochastic_coincidence_emulator_not_parallel_IBM_circuit",
                "one_target_endpoint_receives_recovery",
                "no_inference_read_noise_retention_or_temporal_drift",
            ],
        }
        summary_path = artifact_root / "scientific_summary.json"
        atomic_write_json(summary_path, summary)
        generated_artifacts.append(
            {"path": str(summary_path), "kind": "scientific_summary"}
        )
        terminal = {
            "evidence_tier": EVIDENCE_TIER,
            "source_assignment_seed": int(contract["source_assignment_seed"]),
            "target_assignment_seed": int(contract["target_assignment_seed"]),
            "selected_hwa_epoch": int(best_epoch),
            "source_epoch0_persistent_mean_test_accuracy": float(
                source_test["epoch_0"]["persistent_mean_accuracy"]
            ),
            "source_selected_hwa_persistent_mean_test_accuracy": float(
                source_test["selected_hwa"]["persistent_mean_accuracy"]
            ),
            "target_epoch0_pv_mean_test_accuracy": float(initial_stats["mean"]),
            "target_selected_hwa_pv_mean_test_accuracy": float(selected_stats["mean"]),
            "target_epoch0_fault_clamped_no_write_test_accuracy": float(
                target_fault_clamped_no_write["epoch_0"]["evaluation"]["metrics"][
                    "student_accuracy"
                ]
            ),
            "target_selected_hwa_fault_clamped_no_write_test_accuracy": float(
                target_fault_clamped_no_write["selected_hwa"]["evaluation"][
                    "metrics"
                ]["student_accuracy"]
            ),
            "target_p0_pre_adam_test_accuracy": float(
                recovery["before_test"]["student_accuracy"]
            ),
            "target_p0_post_adam_test_accuracy": float(
                recovery["after_test"]["student_accuracy"]
            ),
            "target_corrupt_cells": int(target_population.corrupt.sum().item()),
            "target_paired_hwa_mean_gain": float(
                target_hwa_gate["mean_selected_minus_epoch0_accuracy"]
            ),
            "target_paired_hwa_wins": int(target_hwa_gate["wins"]),
            "target_paired_hwa_gate_passed": bool(target_hwa_gate["passed"]),
            "target_recovery_gain": float(recovery_gate["post_minus_pre_accuracy"]),
            "target_recovery_gate_passed": bool(recovery_gate["passed"]),
            "verify_reads_during_recovery": 0,
            "negative_conductance_count": 0,
        }
        store.append_metric({"mode": "train_terminal", **terminal})
        store.complete(
            metrics=terminal,
            artifacts=_artifact_records(store, generated_artifacts),
        )
        print(f"run_dir={store.run_dir}", flush=True)
        del source_training_bank, source_selection_bank
        gc.collect()
        torch.cuda.empty_cache()
        return 0
    except BaseException as error:
        store.fail(error)
        raise


__all__ = [
    "_assert_corrupt_immobile",
    "_development_key",
    "_mean2_objective",
    "_train_source_epoch",
    "run_train",
]
