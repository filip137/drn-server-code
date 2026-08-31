"""Exploratory multi-source corrupt-OM HWA and sealed-target recovery runtime.

This successor deliberately reuses the positive-full-conductance and literal
published-corrupt primitives from the standard ladder.  Its interventions are
limited to (a) source-array coverage, (b) the output-layer HWA learning-rate
factor, and (c) a three-epoch output-KL target recovery whose exact data-order
state is serialized alongside the physical P0 continuation.  Output-KL names
the loss; both physical layers are updated.
"""

from __future__ import annotations

import gc
from itertools import combinations
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
from experiments.mnist_relu_drn.ibm_om_baseline_selection import quad_stack
from experiments.mnist_relu_drn.ibm_om_bounded_codebook_scheme_screen import (
    _tensor_sha256,
)
from experiments.mnist_relu_drn.ibm_om_corrupt_source_hwa_cross_array_pulse_adam import (
    build_corrupt_persistent_endpoint_codebook_ensemble,
    build_published_persistent_targets,
    build_repaired_mapping_templates,
    clamp_targets_to_published_corrupt_singletons,
)
from experiments.mnist_relu_drn.ibm_om_corrupt_source_hwa_cross_array_pulse_adam_runtime import (
    _apply_full_g,
    _assert_corrupt_immobile,
    _assert_distinct_array_pair,
    _assert_p0_roundtrip,
    _deployment_protocol_adapter,
    _endpoint_accuracies,
    _evaluate_endpoint_bank,
    _evaluate_full_g,
    _initial_logical_masters,
    _one_forward_gradient,
    _paired_target_hwa_gate,
    _population_report,
    _sample_and_save_pair,
    _save_endpoint_bank,
    _snapshot_masters,
)
from experiments.mnist_relu_drn.ibm_om_winsorized_cross_array_open_loop_adam import (
    CLAIM_LABEL,
    UPDATE_SURFACE,
    ColumnSerialOpenLoopAdam,
    open_loop_pulse_port,
)
from experiments.mnist_relu_drn.ibm_om_winsorized_cross_array_open_loop_adam_runtime import (
    _programming_summary,
    _restore_deployment_plant,
)
from experiments.mnist_relu_drn.ibm_om_winsorized_onchip_adam import (
    flatten_physical,
    split_flat_physical,
    sync_catalog_from_persistent,
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
    quantize_winsorized_logical_master,
)
from experiments.mnist_relu_drn.ibm_om_winsorized_qat_runtime import (
    NOMINAL_DELTA_X,
)
from experiments.mnist_relu_drn.runtime import _load_teacher
from experiments.mnist_shared import build_mnist_loaders, limited
from experiments.schema import to_plain_data
from training.checkpoint import atomic_torch_save
from training.ibm_reram_program_verify import (
    ControllerSettings,
    IbmReramRawActivePlant,
    run_program_verify,
)
from training.ibm_reram_raw_active_program_verify import (
    array_population_as_pulse_population,
    matched_trajectory_seeds,
)


if TYPE_CHECKING:
    from ebl.cli import TrainRequest


_ROOT = Path(__file__).resolve().parents[2]
SUMMARY_SCHEMA = "ebl.mnist_relu_drn.ibm_om_corrupt_multi_source_hwa_tuned_recovery"
SUMMARY_SCHEMA_VERSION = 1
EVIDENCE_TIER = "exploratory_noncanonical"
SIMPLICITY_ORDER = (
    "single_w2_x1",
    "single_w2_x30",
    "multi_w2_x1",
    "multi_w2_x30",
)
COMPARATOR_ASSIGNMENT_SEED = 93002
COMPARATOR_ENDPOINT_SEEDS = (93301, 93302, 93303, 93304, 93305)
COMPARATOR_HISTORICAL_MEAN = 0.42066
COMPARATOR_PUBLISHED_FINGERPRINT = (
    "59e365851c959a555778699d49385c41c49a63830a4d853596d49ea57cb7efce"
)
COMPARATOR_REPAIRED_FINGERPRINT = (
    "fa50c6910717d617049faff0cf504e51f8d3d912954b38d2bda0c417495527f6"
)


def _field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _mean2_objective(ideal_loss: float, persistent_losses: Sequence[float]) -> float:
    weights = gradient_mixture_weights(
        "mean2", persistent_losses, ideal_loss=ideal_loss
    )
    value = weights.ideal_weight * float(ideal_loss) + sum(
        weight * float(loss)
        for weight, loss in zip(
            weights.persistent_weights, persistent_losses, strict=True
        )
    )
    if not math.isfinite(value):
        raise FloatingPointError("Multi-source HWA objective became non-finite.")
    return value


def _arm_id(arm: Any) -> str:
    value = str(_field(arm, "arm_id", _field(arm, "name", "")))
    if value not in SIMPLICITY_ORDER:
        raise ValueError(f"Unexpected HWA arm id {value!r}.")
    return value


def _arm_source_seeds(arm: Any) -> tuple[int, ...]:
    raw = _field(
        arm,
        "training_assignment_seeds",
        _field(arm, "source_assignment_seeds", _field(arm, "assignment_seeds", ())),
    )
    result = tuple(map(int, raw))
    if result not in {(94001,), (94001, 94002)}:
        raise ValueError("HWA arm source schedule must be 94001 or alternating 94001/94002.")
    return result


def _arm_w2_factor(arm: Any) -> int:
    value = int(
        _field(
            arm,
            "w2_learning_rate_factor",
            _field(
                arm,
                "w2_learning_rate_multiplier",
                _field(arm, "w2_lr_factor", 0),
            ),
        )
    )
    if value not in {1, 30}:
        raise ValueError("Expected the frozen W2 learning-rate factor 1 or 30.")
    return value


def _generator_state_record(state: torch.Tensor) -> Mapping[str, Any]:
    value = state.detach().cpu().to(torch.uint8).contiguous().clone()
    return {
        "state": value,
        "sha256": _tensor_sha256(value),
        "bytes": int(value.numel()),
        "backend": "torch_cpu_generator",
    }


def _assert_generator_state_record(record: Mapping[str, Any]) -> torch.Tensor:
    state = record.get("state")
    if (
        not isinstance(state, torch.Tensor)
        or state.dtype != torch.uint8
        or state.ndim != 1
        or _tensor_sha256(state) != record.get("sha256")
        or int(state.numel()) != int(record.get("bytes", -1))
    ):
        raise RuntimeError("Serialized MNIST train-loader generator state is invalid.")
    return state.detach().cpu().clone()


def _train_hwa_epoch(
    *,
    stack: Any,
    teacher: Any,
    loader: Iterable,
    masters: Sequence[torch.nn.Parameter],
    templates_by_assignment: Mapping[int, Sequence[WinsorizedQatLayerTemplate]],
    banks_by_assignment: Mapping[int, Any],
    assignment_schedule: Sequence[int],
    optimizer: torch.optim.Optimizer,
    global_minibatch_ordinal: int,
    assignment_visit_ordinals: Mapping[int, int],
    maximum_batches: int | None,
) -> tuple[Mapping[str, Any], int, Mapping[int, int]]:
    """Train one epoch with global array alternation and local endpoint cycling."""

    schedule = tuple(map(int, assignment_schedule))
    if schedule not in {(94001,), (94001, 94002)}:
        raise ValueError("Unexpected HWA assignment schedule.")
    visits = {int(seed): int(assignment_visit_ordinals.get(int(seed), 0)) for seed in schedule}
    totals: dict[str, float | int] = {
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
    assignment_batches = {seed: 0 for seed in schedule}
    endpoint_exposures = {
        seed: {int(endpoint): 0 for endpoint in banks_by_assignment[seed].endpoint_seeds}
        for seed in schedule
    }
    start_global = int(global_minibatch_ordinal)
    start_visits = dict(visits)
    for inputs, labels in limited(loader, maximum_batches):
        schedule_index = int(global_minibatch_ordinal) % len(schedule)
        assignment_seed = schedule[schedule_index]
        bank = banks_by_assignment[assignment_seed]
        templates = templates_by_assignment[assignment_seed]
        visit_ordinal = visits[assignment_seed]
        sample_indices = _sample_indices("mean2", visit_ordinal)
        visits[assignment_seed] = visit_ordinal + 1
        assignment_batches[assignment_seed] += 1
        for sample_index in sample_indices:
            endpoint = int(bank.endpoint_seeds[sample_index])
            endpoint_exposures[assignment_seed][endpoint] += 1

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
        persistent_losses: list[float] = []
        persistent_gradients: list[tuple[torch.Tensor, ...]] = []
        persistent_metrics: list[Mapping[str, float | int]] = []
        for full_g in bank.full_conductance_views(
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
            raise RuntimeError("Mean-two HWA did not materialize exactly two endpoints.")
        mixed = mix_logical_gradients(
            "mean2",
            persistent_losses,
            persistent_gradients,
            ideal_loss=ideal_loss,
            ideal_gradients=ideal_gradients,
        )
        objective = _mean2_objective(ideal_loss, persistent_losses)
        optimizer.zero_grad(set_to_none=True)
        for layer, (master, gradient) in enumerate(zip(masters, mixed, strict=True)):
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
            int(item["correct"]) for item in persistent_metrics
        )
        global_minibatch_ordinal += 1

    examples = int(totals["examples"])
    batches = int(totals["batches"])
    if examples < 1 or batches < 1:
        raise RuntimeError("Expected at least one multi-source HWA minibatch.")
    return (
        {
            "examples": examples,
            "batches": batches,
            "global_minibatch_ordinal_start": start_global,
            "global_minibatch_ordinal_end": int(global_minibatch_ordinal),
            "assignment_schedule": list(schedule),
            "assignment_schedule_semantic": (
                "global_minibatch_ordinal_modulo_source_count_start_94001"
            ),
            "endpoint_schedule_semantic": (
                "cyclic_adjacent_pair_by_assignment_visit_ordinal"
            ),
            "assignment_visit_ordinal_start": {
                str(seed): value for seed, value in start_visits.items()
            },
            "assignment_visit_ordinal_end": {
                str(seed): value for seed, value in visits.items()
            },
            "assignment_batches": {
                str(seed): value for seed, value in assignment_batches.items()
            },
            "endpoint_exposures": {
                str(seed): {str(endpoint): count for endpoint, count in values.items()}
                for seed, values in endpoint_exposures.items()
            },
            "objective_weights": {"ideal": 0.25, "persistent": [0.375, 0.375]},
            "objective_kl_teacher_student": float(totals["objective_kl_sum"]) / examples,
            "ideal_kl_teacher_student": float(totals["ideal_kl_sum"]) / examples,
            "ideal_student_accuracy": int(totals["ideal_correct"]) / examples,
            "persistent_mean_kl_teacher_student": (
                float(totals["persistent_kl_sum"]) / (2 * examples)
            ),
            "persistent_mean_student_accuracy": (
                int(totals["persistent_correct"]) / (2 * examples)
            ),
            "logical_gradient_rms": [
                math.sqrt(total / count)
                for total, count in zip(gradient_squared, gradient_values, strict=True)
            ],
        },
        int(global_minibatch_ordinal),
        visits,
    )


def _hwa_selection_key(candidate: Mapping[str, Any]) -> tuple[float, float, float, int, int]:
    development = candidate["development"]
    arm_id = str(candidate["arm_id"])
    if arm_id not in SIMPLICITY_ORDER:
        raise ValueError("Unknown HWA arm in joint selection.")
    epoch = int(candidate["epoch"])
    if epoch < 1:
        raise ValueError("Epoch zero is not a trained HWA candidate.")
    return (
        float(development["persistent_mean_accuracy"]),
        float(development["persistent_minimum_accuracy"]),
        -float(development["persistent_mean_kl_teacher_student"]),
        -epoch,
        -SIMPLICITY_ORDER.index(arm_id),
    )


def _select_hwa_candidate(candidates: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    selected = tuple(candidates)
    if not selected:
        raise ValueError("Expected at least one trained HWA candidate.")
    identities = {(str(item["arm_id"]), int(item["epoch"])) for item in selected}
    if len(identities) != len(selected):
        raise ValueError("HWA candidate identities are not unique.")
    return max(selected, key=_hwa_selection_key)


def _corrected_bank_programmability(bank: Any, population: Any) -> Mapping[str, Any]:
    """Separate repaired-supported table entries from physically writable ones."""

    maximum = bank.maximum_index.detach().cpu().to(torch.int64)
    corrupt = population.corrupt.detach().cpu().to(torch.bool)
    if maximum.shape != corrupt.shape:
        raise ValueError("Endpoint-bank capacity and corrupt mask shapes differ.")
    supported = 0
    programmable = 0
    corrupt_supported = 0
    per_level = []
    for level in range(int(bank.num_levels)):
        level_supported = maximum >= level
        level_programmable = level_supported & ~corrupt
        level_corrupt = level_supported & corrupt
        row = {
            "level": level,
            "repaired_supported_cells": int(level_supported.sum().item()),
            "controller_programmable_healthy_cells": int(level_programmable.sum().item()),
            "immutable_corrupt_supported_cells": int(level_corrupt.sum().item()),
        }
        per_level.append(row)
        supported += row["repaired_supported_cells"]
        programmable += row["controller_programmable_healthy_cells"]
        corrupt_supported += row["immutable_corrupt_supported_cells"]
    return {
        "endpoint_repeats": len(tuple(bank.endpoint_seeds)),
        "per_endpoint_repaired_supported_entries": supported,
        "per_endpoint_controller_programmable_healthy_entries": programmable,
        "per_endpoint_immutable_corrupt_supported_entries": corrupt_supported,
        "all_endpoint_repaired_supported_entries": supported * len(tuple(bank.endpoint_seeds)),
        "all_endpoint_controller_programmable_healthy_entries": (
            programmable * len(tuple(bank.endpoint_seeds))
        ),
        "legacy_level_diagnostic_eligible_cells_semantic": (
            "repaired_supported_not_controller_programmable"
        ),
        "per_level": per_level,
    }


def _assert_pairwise_source_development_distinctness(
    pairs: Mapping[int, Any], assignment_seeds: Sequence[int]
) -> Mapping[str, Any]:
    """Fail closed on identity or published-fault reuse across source/dev arrays."""

    seeds = tuple(map(int, assignment_seeds))
    if len(seeds) != 3 or len(set(seeds)) != 3 or set(pairs) != set(seeds):
        raise ValueError("Expected exactly the frozen 94001/94002/94003 pair mapping.")
    rows: list[Mapping[str, Any]] = []
    for left_seed, right_seed in combinations(seeds, 2):
        left = pairs[left_seed]
        right = pairs[right_seed]
        base = _assert_distinct_array_pair(left, right)
        published_masks_differ = not torch.equal(
            left.published.corrupt.detach().cpu(),
            right.published.corrupt.detach().cpu(),
        )
        repaired_provenance_masks_differ = not torch.equal(
            left.repaired.published_corrupt.detach().cpu(),
            right.repaired.published_corrupt.detach().cpu(),
        )
        if not published_masks_differ or not repaired_provenance_masks_differ:
            raise RuntimeError(
                "Source/development assignments reused an exact published corrupt mask."
            )
        if bool(torch.any(left.repaired.corrupt)) or bool(
            torch.any(right.repaired.corrupt)
        ):
            raise RuntimeError("A repaired mapping twin retained active corrupt cells.")
        rows.append(
            {
                **dict(base),
                "left_assignment_seed": left_seed,
                "right_assignment_seed": right_seed,
                "published_corrupt_masks_differ": True,
                "repaired_published_corrupt_provenance_masks_differ": True,
                "repaired_active_corrupt_masks_are_both_empty": True,
            }
        )
    return {
        "assignment_seeds": list(seeds),
        "pair_count": len(rows),
        "all_published_fingerprints_pairwise_distinct": True,
        "all_repaired_fingerprints_pairwise_distinct": True,
        "all_published_corrupt_masks_pairwise_distinct": True,
        "pairs": rows,
    }


def _code_change_report(
    initial_masters: Sequence[torch.Tensor],
    current_masters: Sequence[torch.Tensor],
    templates: Sequence[WinsorizedQatLayerTemplate],
) -> Mapping[str, Any]:
    initial = quantize_winsorized_logical_master(initial_masters, templates)
    current = quantize_winsorized_logical_master(current_masters, templates)
    layers = []
    for layer, (
        before_index,
        after_index,
        before_weight,
        after_weight,
        template,
    ) in enumerate(
        zip(
            initial.requested_index,
            current.requested_index,
            initial.realized_normalized_weight,
            current.realized_normalized_weight,
            templates,
            strict=True,
        )
    ):
        changed_code = before_index != after_index
        logical_weights = int(before_weight.numel())
        changed_by_weight = quad_stack(
            changed_code, layout=template.layout
        ).any(dim=-1)
        if int(changed_by_weight.numel()) != logical_weights:
            raise RuntimeError("Physical code-change layout lost logical weight identity.")
        realized_delta = after_weight - before_weight
        layers.append(
            {
                "layer": layer,
                "logical_weights": logical_weights,
                "physical_requested_codes_changed": int(changed_code.sum().item()),
                "logical_weights_with_any_requested_code_change": int(
                    changed_by_weight.sum().item()
                ),
                "logical_weights_with_realized_change": int(
                    (realized_delta != 0.0).sum().item()
                ),
                "realized_normalized_weight_delta_rms": float(
                    realized_delta.to(torch.float64).square().mean().sqrt().item()
                ),
            }
        )
    return {"mapping_assignment_role": "development_94003", "layers": layers}


def _augment_deployment_fault_accounting(
    report: Mapping[str, Any], payload: Mapping[str, Any], population: Any
) -> Mapping[str, Any]:
    programming = payload["programming"]
    accepted = programming["accepted"].detach().cpu().to(torch.bool)
    exhausted = programming["budget_exhausted"].detach().cpu().to(torch.bool)
    corrupt = population.corrupt.detach().cpu().to(torch.bool)
    if accepted.shape != corrupt.shape or exhausted.shape != corrupt.shape:
        raise RuntimeError("Deployment masks do not match the target population.")
    return {
        **dict(report),
        "fault_accounting": {
            "healthy_programmable_cells": int((~corrupt).sum().item()),
            "immutable_corrupt_cells": int(corrupt.sum().item()),
            "apparent_accepted_healthy": int((accepted & ~corrupt).sum().item()),
            "apparent_accepted_corrupt_but_persistent_immobile": int(
                (accepted & corrupt).sum().item()
            ),
            "budget_exhausted_healthy": int((exhausted & ~corrupt).sum().item()),
            "budget_exhausted_corrupt": int((exhausted & corrupt).sum().item()),
            "programming_success_denominator": "healthy_programmable_cells_only",
        },
    }


def _output_kl_gradients(
    gradients: Sequence[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Validate the two physical gradients of the output-KL objective.

    "Output-only" names the loss surface: there is no hidden-state auxiliary
    loss.  Both physical bindings remain trainable, exactly as in the successful
    predecessor Adam recovery.
    """

    values = tuple(gradients)
    if len(values) != 2:
        raise ValueError("Expected hidden and output physical gradients.")
    if any(not bool(torch.all(torch.isfinite(value))) for value in values):
        raise FloatingPointError("Output-KL recovery gradient is non-finite.")
    return values  # type: ignore[return-value]


def _recovery_selection_key(report: Mapping[str, Any]) -> tuple[float, float, int]:
    validation = report["validation"]
    return (
        float(validation["student_accuracy"]),
        -float(validation["kl_teacher_student"]),
        -int(report["epoch"]),
    )


def _assert_p0_validation_parity(
    p0: Mapping[str, Any],
    observed: Mapping[str, Any],
    observed_prediction_sha256: str,
) -> Mapping[str, Any]:
    """Fail closed if restoring P0 changes its pre-update validation result."""

    expected = p0.get("validation")
    expected_prediction_sha256 = p0.get("validation_prediction_sha256")
    if not isinstance(expected, Mapping):
        raise RuntimeError("Exact P0 lacks its saved validation evaluation.")
    if (
        int(observed.get("student_correct", -1))
        != int(expected.get("student_correct", -2))
        or float(observed.get("kl_teacher_student", float("nan")))
        != float(expected.get("kl_teacher_student", float("inf")))
        or str(observed_prediction_sha256) != str(expected_prediction_sha256)
    ):
        raise RuntimeError(
            "Restored P0 validation result differs before the first recovery update."
        )
    return {
        "student_correct_exact": True,
        "kl_teacher_student_exact": True,
        "prediction_sha256_exact": True,
        "checked_before_first_update": True,
    }


def _fine_tune_output_kl(
    *,
    p0: Mapping[str, Any],
    population: Any,
    pulse_population: Any,
    stack: Any,
    teacher: Any,
    loaders: Any,
    protocol: Any,
    store: RunStore,
    generated_artifacts: list[Mapping[str, Any]],
) -> Mapping[str, Any]:
    """Run three persistent output-KL Adam epochs and select on validation."""

    execution = p0.get("recovery_execution_state")
    if not isinstance(execution, Mapping):
        raise RuntimeError("Exact P0 lacks recovery data-order state.")
    if int(execution.get("seed", -1)) != int(protocol.recovery.data_order_seed):
        raise RuntimeError("Exact P0 recovery data-order seed changed.")
    initial_generator_state = _assert_generator_state_record(execution)
    loaders.train_generator.set_state(initial_generator_state)
    plant = _restore_deployment_plant(
        p0,
        pulse_population=pulse_population,
        maximum_random_draws=int(protocol.deployment.maximum_random_draws),
        device=stack.device,
    )
    p0_persistent = plant.persistent.detach().cpu().clone()
    p0_draw_indices = plant.state_dict()["draw_indices"].detach().cpu().clone()
    sync_catalog_from_persistent(
        stack.bundle.catalog, plant, binding_shapes=population.binding_shapes
    )
    initial_validation, initial_validation_prediction = _evaluate_detailed(
        stack, teacher, loaders.validation, sample_limit=None
    )
    initial_validation_prediction_sha256 = _tensor_sha256(
        initial_validation_prediction.to(torch.int64)
    )
    p0_validation_parity = _assert_p0_validation_parity(
        p0,
        initial_validation,
        initial_validation_prediction_sha256,
    )
    recovery = protocol.recovery
    if int(recovery.epochs) != 3:
        raise ValueError("The tuned successor requires exactly three recovery epochs.")
    optimizer = ColumnSerialOpenLoopAdam(
        open_loop_pulse_port(plant),
        device=stack.device,
        binding_shapes=population.binding_shapes,
        learning_rate_raw_x=float(recovery.learning_rate_raw_x),
        nominal_delta_x=NOMINAL_DELTA_X,
        pulse_cap=int(recovery.pulse_cap),
        pulse_selection_seed=int(recovery.pulse_selection_seed),
        beta1=float(recovery.beta1),
        beta2=float(recovery.beta2),
        epsilon=float(recovery.epsilon),
    )
    corrupt = population.corrupt.to(device=stack.device, dtype=torch.bool)
    first_binding_cells = math.prod(population.binding_shapes[0])
    epoch_reports: list[Mapping[str, Any]] = []
    checkpoint_paths: dict[int, Path] = {}
    prior_issued = 0
    prior_effective = 0
    cumulative_effective = 0
    cumulative_effective_corrupt = 0
    cumulative_effective_healthy = 0
    cumulative_conceptual_phases = 0
    for epoch in range(1, int(recovery.epochs) + 1):
        examples = 0
        batches = 0
        train_kl = 0.0
        train_correct = 0
        issued_epoch = 0
        capped_epoch = 0
        probability_clipped_epoch = 0
        effective_epoch = 0
        effective_corrupt_epoch = 0
        conceptual_phases_epoch = 0
        for batch_index, (inputs, labels) in enumerate(
            limited(loaders.train, recovery.maximum_batches)
        ):
            sync_catalog_from_persistent(
                stack.bundle.catalog, plant, binding_shapes=population.binding_shapes
            )
            inputs = inputs.to(stack.device, dtype=torch.float32)
            labels = labels.to(stack.device, dtype=torch.long)
            with torch.no_grad():
                teacher_logits = teacher.logits(inputs)
            stack.network.set_input(inputs, reset=True)
            stack.minimizer.compute_equilibrium()
            stack.cost.set_teacher(teacher_logits, labels)
            with torch.no_grad():
                train_kl += float(stack.cost.eval().sum().item())
                train_correct += int(
                    stack.cost.student_logits().argmax(dim=1).eq(labels).sum().item()
                )
            gradients = _output_kl_gradients(
                tuple(stack.differentiator.compute_gradient())
            )
            before = plant.persistent.detach().clone()
            step = optimizer.step(gradients)
            moved = plant.persistent != before
            moved_count = int(moved.sum().item())
            moved_corrupt = int((moved & corrupt).sum().item())
            if moved_corrupt:
                raise RuntimeError("An immutable corrupt cell moved during recovery.")
            issued_epoch += int(step.applied_cell_pulses)
            capped_epoch += int(step.capped_cell_requests)
            probability_clipped_epoch += int(step.probability_clipped_cells)
            conceptual_phases_epoch += int(step.conceptual_column_phases)
            effective_epoch += moved_count
            effective_corrupt_epoch += moved_corrupt
            examples += int(labels.numel())
            batches = batch_index + 1
        if batches < 1:
            raise RuntimeError("Expected recovery to process at least one minibatch.")
        cumulative_effective += effective_epoch
        cumulative_effective_corrupt += effective_corrupt_epoch
        cumulative_effective_healthy += effective_epoch - effective_corrupt_epoch
        cumulative_conceptual_phases += conceptual_phases_epoch
        sync_catalog_from_persistent(
            stack.bundle.catalog, plant, binding_shapes=population.binding_shapes
        )
        validation, validation_prediction = _evaluate_detailed(
            stack, teacher, loaders.validation, sample_limit=None
        )
        optimizer_state = optimizer.state_dict()
        pulse_count = optimizer_state["pulse_count"]
        issued_cumulative = int(pulse_count.sum().item())
        if issued_cumulative - prior_issued != issued_epoch:
            raise RuntimeError("Recovery command accounting is not cumulative.")
        prior_issued = issued_cumulative
        if cumulative_effective - prior_effective != effective_epoch:
            raise RuntimeError("Recovery movement accounting is not cumulative.")
        prior_effective = cumulative_effective
        train_generator_record = _generator_state_record(
            loaders.train_generator.get_state()
        )
        checkpoint_payload = {
            **optimizer_state,
            "schema": "ebl.mnist_relu_drn.ibm_om_corrupt_output_kl_recovery",
            "schema_version": 1,
            "evidence_tier": EVIDENCE_TIER,
            "epoch": epoch,
            "plant_continuation_state": plant.state_dict(),
            "source_p0_sha256": sha256_file(Path(str(p0["artifact_path"]))),
            "recovery_train_generator_state": train_generator_record,
            "validation": dict(validation),
            "validation_prediction_sha256": _tensor_sha256(
                validation_prediction.to(torch.int64)
            ),
            "test": None,
            "output_kl_only": True,
            "updated_physical_layers": [0, 1],
            "frozen_physical_layers": [],
            "issued_commands_cumulative": issued_cumulative,
            "effective_state_change_events_cumulative": cumulative_effective,
            "effective_corrupt_state_change_events_cumulative": (
                cumulative_effective_corrupt
            ),
            "effective_healthy_state_change_events_cumulative": (
                cumulative_effective_healthy
            ),
            "conceptual_column_phases_cumulative": cumulative_conceptual_phases,
            "verify_reads_during_updates": 0,
            "authoritative_weight_shadow": None,
        }
        checkpoint = atomic_torch_save(
            checkpoint_payload,
            store.run_dir / f"checkpoints/output_kl_adam_epoch_{epoch}.pt",
        )
        generated_artifacts.append(
            {"path": str(checkpoint), "kind": "output_kl_adam_epoch_checkpoint"}
        )
        checkpoint_paths[epoch] = checkpoint
        report = {
            "epoch": epoch,
            "train": {
                "examples": examples,
                "batches": batches,
                "kl_teacher_student": train_kl / examples,
                "student_accuracy": train_correct / examples,
            },
            "validation": dict(validation),
            "validation_prediction_sha256": checkpoint_payload[
                "validation_prediction_sha256"
            ],
            "test": None,
            "issued_commands_epoch": issued_epoch,
            "issued_commands_cumulative": issued_cumulative,
            "effective_state_change_events_epoch": effective_epoch,
            "effective_state_change_events_cumulative": cumulative_effective,
            "effective_corrupt_state_change_events_epoch": effective_corrupt_epoch,
            "capped_requests_epoch": capped_epoch,
            "probability_clipped_cells_epoch": probability_clipped_epoch,
            "conceptual_column_phases_epoch": conceptual_phases_epoch,
            "conceptual_column_phases_cumulative": cumulative_conceptual_phases,
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": sha256_file(checkpoint),
            "train_generator_state_sha256": train_generator_record["sha256"],
        }
        epoch_reports.append(report)
        store.append_metric({"mode": "target_output_kl_adam", **report})
        print(
            f"output-KL Adam epoch={epoch}/{recovery.epochs} "
            f"validation={100.0*float(validation['student_accuracy']):.2f}% "
            f"issued={issued_epoch} moved={effective_epoch}",
            flush=True,
        )

    selected_report = max(epoch_reports, key=_recovery_selection_key)
    selected_epoch = int(selected_report["epoch"])
    selected_checkpoint = checkpoint_paths[selected_epoch]
    try:
        selected_state = torch.load(
            selected_checkpoint, map_location="cpu", weights_only=True
        )
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        raise RuntimeError("Could not reload selected recovery checkpoint.") from error
    selected_plant = _restore_deployment_plant(
        {"continuation_state": selected_state["plant_continuation_state"]},
        pulse_population=pulse_population,
        maximum_random_draws=int(protocol.deployment.maximum_random_draws),
        device=stack.device,
    )
    selected_persistent = selected_plant.persistent.detach().cpu()
    selected_pulse_count = selected_state["pulse_count"].detach().cpu()
    selected_draw_indices = selected_state["plant_continuation_state"][
        "draw_indices"
    ].detach().cpu()
    expected_draw_delta = 2 * selected_pulse_count
    if not torch.equal(selected_draw_indices - p0_draw_indices, expected_draw_delta):
        raise RuntimeError("Recovery plant RNG draws do not match issued pulse commands.")
    sync_catalog_from_persistent(
        stack.bundle.catalog, selected_plant, binding_shapes=population.binding_shapes
    )
    selected_test, selected_test_prediction = _evaluate_detailed(
        stack, teacher, loaders.test, sample_limit=None
    )
    selected_corrupt_immobility = _assert_corrupt_immobile(
        p0_persistent,
        selected_persistent,
        population.corrupt,
        context="selected_output_kl_open_loop_adam",
    )
    final_counts = selected_pulse_count
    corrupt_cpu = population.corrupt.detach().cpu().to(torch.bool)
    selected_issued = int(final_counts.sum().item())
    selected_issued_corrupt = int(final_counts[corrupt_cpu].sum().item())
    selected_issued_hidden = int(final_counts[:first_binding_cells].sum().item())
    selected_issued_output = int(final_counts[first_binding_cells:].sum().item())
    selected_effective = int(
        selected_state["effective_state_change_events_cumulative"]
    )
    selected_effective_corrupt = int(
        selected_state["effective_corrupt_state_change_events_cumulative"]
    )
    selected_effective_healthy = int(
        selected_state["effective_healthy_state_change_events_cumulative"]
    )
    del optimizer, plant, selected_plant
    gc.collect()
    torch.cuda.empty_cache()
    return {
        "claim_label": CLAIM_LABEL,
        "update_surface": UPDATE_SURFACE,
        "objective": "teacher_output_KL_only_no_hidden_auxiliary_loss",
        "updated_physical_layers": [0, 1],
        "frozen_physical_layers": [],
        "learning_rate_raw_x": float(recovery.learning_rate_raw_x),
        "epochs_run": int(recovery.epochs),
        "initial_validation": dict(initial_validation),
        "initial_validation_prediction_sha256": initial_validation_prediction_sha256,
        "p0_validation_parity_before_first_update": p0_validation_parity,
        "epoch_reports": epoch_reports,
        "selection_key": "validation_accuracy_then_KL_then_earlier_epoch",
        "selected_epoch": selected_epoch,
        "selected_checkpoint": str(selected_checkpoint),
        "selected_checkpoint_sha256": sha256_file(selected_checkpoint),
        "selected_test": dict(selected_test),
        "selected_test_prediction_sha256": _tensor_sha256(
            selected_test_prediction.to(torch.int64)
        ),
        "test_opened_only_after_epoch_selection_frozen": True,
        "pulse": {
            "issued_commands": selected_issued,
            "issued_layer0_commands": selected_issued_hidden,
            "issued_layer1_commands": selected_issued_output,
            "issued_commands_targeting_corrupt_cells": selected_issued_corrupt,
            "effective_state_change_events_selected": selected_effective,
            "effective_healthy_state_change_events_selected": selected_effective_healthy,
            "effective_corrupt_state_change_events_selected": selected_effective_corrupt,
            "effective_event_is_not_unique_moved_cell_count": True,
            "issued_command_is_not_synonymous_with_effective_motion": True,
            "maximum_per_cell_selected": int(final_counts.max().item()),
            "cells_at_cap_selected": int(
                (final_counts >= int(recovery.pulse_cap)).sum().item()
            ),
            "verify_reads_during_updates": 0,
        },
        "output_kl_objective_invariant": {
            "hidden_auxiliary_loss": False,
            "both_physical_layers_updated": True,
            "conceptual_column_phases_per_minibatch": [100, 20],
        },
        "corrupt_immobility": selected_corrupt_immobility,
        "rng_continuation": {
            "data_order_seed": int(execution["seed"]),
            "initial_train_generator_state_sha256": execution["sha256"],
            "selected_train_generator_state_sha256": selected_state[
                "recovery_train_generator_state"
            ]["sha256"],
            "plant_draw_delta_equals_two_per_issued_command": True,
        },
        "validation_used_for_selection": True,
        "test_used_for_selection": False,
    }


def _runtime_protocol_for_assignment(protocol: Any, assignment_seed: int) -> Any:
    base = _deployment_protocol_adapter(protocol)
    recovery = SimpleNamespace(
        **vars(base.recovery),
        data_order_seed=int(protocol.recovery.data_order_seed),
    )
    return SimpleNamespace(
        target=SimpleNamespace(
            assignment_seed=int(assignment_seed),
            fine_tune_endpoint_seed=int(base.target.fine_tune_endpoint_seed),
        ),
        deployment=base.deployment,
        recovery=recovery,
    )


def _deploy_once_split_safe(
    *,
    name: str,
    endpoint_seed: int,
    raw_x_targets: Sequence[torch.Tensor],
    target_kind: str,
    population: Any,
    pulse_population: Any,
    stack: Any,
    teacher: Any,
    evaluation_loader: Iterable,
    evaluation_label: str,
    protocol: Any,
    store: RunStore,
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    """Program once and evaluate only the explicitly named open split."""

    if evaluation_label not in {"validation", "test"}:
        raise ValueError("Deployment evaluation label must be validation or test.")
    flat_target = flatten_physical(raw_x_targets).to(
        device=stack.device, dtype=torch.float32
    )
    plant = IbmReramRawActivePlant(
        pulse_population,
        seeds=matched_trajectory_seeds(population, endpoint_seed=endpoint_seed),
        device=stack.device,
        maximum_random_draws=int(protocol.deployment.maximum_random_draws),
    )
    plant.initialize_at_sampled_lower()
    result = run_program_verify(
        plant.controller_port(),
        targets=flat_target,
        tolerance=float(protocol.deployment.verify_tolerance_raw_x),
        maximum_pulses=int(protocol.deployment.maximum_program_pulses),
        settings=ControllerSettings(kind="one_pulse"),
    )
    sync_catalog_from_persistent(
        stack.bundle.catalog, plant, binding_shapes=population.binding_shapes
    )
    metrics, prediction = _evaluate_detailed(
        stack, teacher, evaluation_loader, sample_limit=None
    )
    payload = {
        "schema": "ebl.mnist_relu_drn.ibm_om_winsorized_cross_array_deployment",
        "schema_version": 1,
        "evidence_tier": EVIDENCE_TIER,
        "assignment_seed": int(protocol.target.assignment_seed),
        "endpoint_seed": int(endpoint_seed),
        "target_kind": target_kind,
        "target_clipping": False,
        "maximum_program_pulses": int(protocol.deployment.maximum_program_pulses),
        "maximum_random_draws": int(protocol.deployment.maximum_random_draws),
        "binding_keys": list(population.binding_keys),
        "binding_shapes": [list(shape) for shape in population.binding_shapes],
        "target_raw_x": flat_target.detach().cpu().clone(),
        "programming": {
            "accepted": result.accepted.detach().cpu().clone(),
            "nonfinite": result.nonfinite.detach().cpu().clone(),
            "budget_exhausted": result.budget_exhausted.detach().cpu().clone(),
            "set_count": result.set_count.detach().cpu().clone(),
            "reset_count": result.reset_count.detach().cpu().clone(),
            "total_pulses": result.total_pulses.detach().cpu().clone(),
            "verify_count": result.verify_count.detach().cpu().clone(),
            "reversals": result.reversals.detach().cpu().clone(),
        },
        "continuation_state": plant.state_dict(),
        evaluation_label: dict(metrics),
        f"{evaluation_label}_prediction_sha256": _tensor_sha256(
            prediction.to(torch.int64)
        ),
        "unopened_split": "test" if evaluation_label == "validation" else None,
        "persistent_endpoint_applied_to_drn": True,
        "apparent_endpoint_applied_to_drn": False,
        "inference_read_noise": False,
    }
    path = atomic_torch_save(
        payload, store.run_dir / f"artifacts/deployments/{name}.pt"
    )
    report = {
        "name": name,
        "endpoint_seed": int(endpoint_seed),
        "target_kind": target_kind,
        "path": str(path),
        "sha256": sha256_file(path),
        "programming": _programming_summary(
            result,
            target_raw_x=flat_target,
            pulse_population=pulse_population,
        ),
        evaluation_label: dict(metrics),
        f"{evaluation_label}_prediction_sha256": _tensor_sha256(
            prediction.to(torch.int64)
        ),
        "test": None if evaluation_label == "validation" else dict(metrics),
        "test_prediction_sha256": (
            None
            if evaluation_label == "validation"
            else _tensor_sha256(prediction.to(torch.int64))
        ),
    }
    store.append_metric({"mode": "cross_array_deployment", **report})
    print(
        f"cross deploy {target_kind} seed={endpoint_seed}: "
        f"{evaluation_label}={100.0*float(metrics['student_accuracy']):.2f}% "
        f"exhausted={report['programming']['budget_exhausted']}",
        flush=True,
    )
    del plant
    gc.collect()
    torch.cuda.empty_cache()
    return report, payload


def _replay_deployment_test(
    report: Mapping[str, Any],
    payload: Mapping[str, Any],
    *,
    population: Any,
    stack: Any,
    teacher: Any,
    test_loader: Iterable,
) -> Mapping[str, Any]:
    """Evaluate one saved persistent endpoint after all selections are frozen."""

    persistent = payload["continuation_state"]["persistent"]
    if not isinstance(persistent, torch.Tensor):
        raise RuntimeError("Deployment payload lacks persistent tensor state.")
    full_g = split_flat_physical(
        persistent.detach().cpu().to(torch.float32) + 1.0,
        population.binding_shapes,
    )
    evaluation = _evaluate_full_g(
        stack=stack,
        teacher=teacher,
        loader=test_loader,
        full_g=full_g,
    )
    return {
        **dict(report),
        "test": dict(evaluation["metrics"]),
        "test_prediction_sha256": evaluation["prediction_sha256"],
        "test_replayed_from_saved_persistent_endpoint": True,
    }


def _deploy_pair(
    *,
    label: str,
    target: Any,
    endpoint_seeds: Sequence[int],
    population: Any,
    pulse_population: Any,
    stack: Any,
    teacher: Any,
    evaluation_loader: Iterable,
    evaluation_label: str,
    runtime_protocol: Any,
    store: RunStore,
    generated_artifacts: list[Mapping[str, Any]],
) -> tuple[list[Mapping[str, Any]], dict[int, Mapping[str, Any]]]:
    reports: list[Mapping[str, Any]] = []
    payloads: dict[int, Mapping[str, Any]] = {}
    for endpoint_seed in endpoint_seeds:
        report, payload = _deploy_once_split_safe(
            name=f"{label}_published_seed_{int(endpoint_seed)}",
            endpoint_seed=int(endpoint_seed),
            raw_x_targets=target.target_raw_x,
            target_kind=f"{label}_repaired_template_to_published_cells",
            population=population,
            pulse_population=pulse_population,
            stack=stack,
            teacher=teacher,
            evaluation_loader=evaluation_loader,
            evaluation_label=evaluation_label,
            protocol=runtime_protocol,
            store=store,
        )
        report = _augment_deployment_fault_accounting(report, payload, population)
        report = {
            **report,
            "corrupt_immobility": _assert_corrupt_immobile(
                population.min_bound,
                payload["continuation_state"]["persistent"],
                population.corrupt,
                context=f"{label}_seed_{int(endpoint_seed)}",
            ),
        }
        reports.append(report)
        payloads[int(endpoint_seed)] = payload
        generated_artifacts.append(
            {"path": str(report["path"]), "kind": f"{label}_deployment"}
        )
    return reports, payloads


def _protocol_hwa_arms(protocol: Any) -> tuple[Any, ...]:
    arms = tuple(_field(protocol, "hwa_arms", _field(protocol, "arms", ())))
    if not arms:
        training = _field(protocol, "training")
        arms = tuple(_field(training, "arms", ()))
    if tuple(_arm_id(arm) for arm in arms) != SIMPLICITY_ORDER:
        raise ValueError("Resolved config does not contain the four frozen HWA arms.")
    return arms


def _strict_protocol_values(protocol: Any) -> Mapping[str, Any]:
    training = _field(protocol, "training")
    mapping = _field(protocol, "mapping")
    target = _field(protocol, "target")
    deployment = _field(protocol, "deployment")
    recovery = _field(protocol, "recovery")
    source = _field(protocol, "source", _field(protocol, "sources"))
    train_assignments = tuple(
        map(
            int,
            _field(
                source,
                "train_assignment_seeds",
                _field(source, "training_assignment_seeds", (94001, 94002)),
            ),
        )
    )
    raw_train_endpoints = tuple(
        _field(source, "training_endpoint_seeds_by_assignment", ())
    )
    train_endpoints = {
        int(seed): tuple(map(int, values)) for seed, values in raw_train_endpoints
    }
    arms = _protocol_hwa_arms(protocol)
    w1_rates = {float(_field(arm, "w1_learning_rate")) for arm in arms}
    base_w2_rates = {float(_field(arm, "base_w2_learning_rate")) for arm in arms}
    if len(w1_rates) != 1 or len(base_w2_rates) != 1:
        raise ValueError("All HWA arms must share one frozen W1/base-W2 rate.")
    development_seed = int(
        _field(
            source,
            "development_assignment_seed",
            _field(protocol, "development_assignment_seed", 94003),
        )
    )
    development_endpoints = tuple(
        map(
            int,
            _field(
                source,
                "development_endpoint_seeds",
                _field(protocol, "development_endpoint_seeds", (94201, 94202, 94203, 94204)),
            ),
        )
    )
    values = {
        "train_assignments": train_assignments,
        "train_endpoints": train_endpoints,
        "development_assignment": development_seed,
        "development_endpoints": development_endpoints,
        "target_assignment": int(_field(target, "assignment_seed")),
        "target_endpoints": tuple(map(int, _field(target, "endpoint_seeds"))),
        "p0_endpoint": int(
            _field(target, "fine_tune_endpoint_seed", _field(target, "p0_endpoint_seed"))
        ),
        "hwa_epochs": int(_field(training, "epochs")),
        "w1_lr": next(iter(w1_rates)),
        "base_w2_lr": next(iter(base_w2_rates)),
        "spacing": float(_field(mapping, "level_spacing_raw_x")),
        "gain": float(_field(mapping, "fixed_logit_gain")),
        "maximum_program_pulses": int(_field(deployment, "maximum_program_pulses")),
        "maximum_random_draws": int(_field(deployment, "maximum_random_draws")),
        "verify_tolerance": float(_field(deployment, "verify_tolerance_raw_x")),
        "recovery_epochs": int(_field(recovery, "epochs")),
        "recovery_lr": float(_field(recovery, "learning_rate_raw_x")),
        "recovery_cap": int(_field(recovery, "pulse_cap")),
        "recovery_pulse_seed": int(_field(recovery, "pulse_selection_seed")),
        "recovery_data_order_seed": int(_field(recovery, "data_order_seed")),
    }
    expected = {
        "train_assignments": (94001, 94002),
        "development_assignment": 94003,
        "development_endpoints": (94201, 94202, 94203, 94204),
        "target_assignment": 94004,
        "target_endpoints": (94301, 94302, 94303, 94304, 94305),
        "p0_endpoint": 94301,
        "hwa_epochs": 6,
        "recovery_epochs": 3,
        "recovery_pulse_seed": 94401,
        "recovery_data_order_seed": 94402,
    }
    if any(values[name] != value for name, value in expected.items()):
        raise ValueError("Resolved protocol does not match the frozen multi-source ladder.")
    if (
        train_endpoints != {
            94001: (94101, 94102, 94103, 94104),
            94002: (94111, 94112, 94113, 94114),
        }
        or not math.isclose(values["w1_lr"], 0.004411914893617021, abs_tol=1e-18)
        or not math.isclose(values["base_w2_lr"], 1.1974808510638298e-5, abs_tol=1e-20)
        or not math.isclose(values["spacing"], NOMINAL_DELTA_X, abs_tol=1e-15)
        or not math.isclose(values["gain"], 14.12537544622754, abs_tol=1e-15)
        or not math.isclose(values["recovery_lr"], 3e-5, abs_tol=1e-20)
        or values["recovery_cap"] != 64
        or values["maximum_program_pulses"] != 128
        or values["maximum_random_draws"] != 385
        or str(_field(training, "optimizer")) != "sgd"
        or float(_field(training, "momentum")) != 0.0
        or float(_field(training, "weight_decay")) != 0.0
        or str(_field(training, "endpoint_objective")) != "mean2"
        or float(_field(training, "ideal_gradient_weight")) != 0.25
        or tuple(map(float, _field(training, "persistent_gradient_weights")))
        != (0.375, 0.375)
        or str(_field(training, "persistent_sample_schedule"))
        != "cyclic_adjacent_pair_by_assignment_visit_ordinal"
        or bool(_field(recovery, "output_kl_only")) is not True
        or tuple(map(int, _field(recovery, "updated_layer_indices"))) != (0, 1)
        or tuple(map(int, _field(recovery, "frozen_layer_indices"))) != ()
        or int(_field(recovery, "verify_reads_during_updates")) != 0
    ):
        raise ValueError("Resolved numerical protocol differs from the frozen design.")
    return values


def run_train(request: "TrainRequest") -> int:
    """Execute the four-arm source screen, sealed target, and tuned recovery."""

    from experiments.mnist_relu_drn.ibm_om_corrupt_multi_source_hwa_tuned_recovery_config import (
        CorruptMultiSourceHwaTunedRecoveryTrainSpec,
    )

    spec = request.spec
    if not isinstance(spec, CorruptMultiSourceHwaTunedRecoveryTrainSpec):
        raise TypeError("Expected the dedicated corrupt multi-source train spec.")
    if request.teacher_weights is None:
        raise ValueError("Expected explicit --teacher-weights for the frozen ReLU teacher.")
    for name in ("weights", "base_weights", "resume", "device_data", "device_model"):
        if getattr(request, name) is not None:
            raise ValueError(
                "The corrupt multi-source ladder accepts only --teacher-weights; "
                f"unexpected {name}."
            )
    protocol = spec.protocol
    contract = _strict_protocol_values(protocol)
    arms = _protocol_hwa_arms(protocol)
    teacher_path = request.teacher_weights.expanduser().resolve()
    if sha256_file(teacher_path) != str(protocol.expected_teacher_weights_sha256):
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
            raise RuntimeError("The corrupt multi-source ladder requires local CUDA.")
        torch.manual_seed(student.runtime.seed)
        torch.cuda.manual_seed_all(student.runtime.seed)
        device = torch.device("cuda")
        loaders = build_mnist_loaders(
            student.data,
            data_seed=student.runtime.data_seed,
            calibration_examples=student.mapping.calibration_examples,
            calibration_batch_size=student.mapping.calibration_batch_size,
        )
        hwa_initial_generator_state = loaders.train_generator.get_state().clone()
        recovery_generator = torch.Generator(device="cpu")
        recovery_generator.manual_seed(int(contract["recovery_data_order_seed"]))
        recovery_initial_generator_record = {
            **_generator_state_record(recovery_generator.get_state()),
            "seed": int(contract["recovery_data_order_seed"]),
            "semantic": "dedicated_recovery_shuffle_independent_of_HWA_arm_count",
        }
        teacher, teacher_metadata = _load_teacher(
            teacher_path, device=device, spec=student
        )
        stack = build_student_stack(student, enable_measured=False)
        stack.cost.gain = float(contract["gain"])
        base_parameters, source_absmax, source_weight_hashes = _initial_logical_masters(
            teacher, device=device
        )
        initial_master = _snapshot_masters(base_parameters)
        del base_parameters
        artifact_root = store.run_dir / "artifacts"
        generated_artifacts: list[Mapping[str, Any]] = []

        # Build the two training identities and the disjoint development identity
        # once.  Every arm shares these immutable endpoint banks.
        pairs: dict[int, Any] = {}
        templates_cpu: dict[int, tuple[WinsorizedQatLayerTemplate, ...]] = {}
        banks_cpu: dict[int, Any] = {}
        banks: dict[int, Any] = {}
        population_artifacts: dict[int, list[Mapping[str, Any]]] = {}
        all_source_seeds = (
            *contract["train_assignments"],
            int(contract["development_assignment"]),
        )
        endpoints_by_assignment = {
            **contract["train_endpoints"],
            int(contract["development_assignment"]): contract[
                "development_endpoints"
            ],
        }
        for assignment_seed in all_source_seeds:
            role = (
                f"train_{assignment_seed}"
                if assignment_seed in contract["train_assignments"]
                else f"development_{assignment_seed}"
            )
            pair, artifacts = _sample_and_save_pair(
                bindings=stack.bundle.catalog.trainable,
                assignment_seed=int(assignment_seed),
                role=role,
                sampler=sampler,
                artifact_root=artifact_root,
            )
            pairs[int(assignment_seed)] = pair
            population_artifacts[int(assignment_seed)] = artifacts
            generated_artifacts.extend(artifacts)
            template = build_repaired_mapping_templates(
                initial_master,
                pair,
                level_spacing_raw_x=float(contract["spacing"]),
            )
            templates_cpu[int(assignment_seed)] = template
            print(
                f"building shared endpoint bank assignment={assignment_seed} "
                f"seeds={list(endpoints_by_assignment[int(assignment_seed)])}",
                flush=True,
            )
            bank = build_corrupt_persistent_endpoint_codebook_ensemble(
                pair,
                template,
                endpoint_seeds=endpoints_by_assignment[int(assignment_seed)],
                tolerance_raw_x=float(contract["verify_tolerance"]),
                maximum_program_pulses=int(contract["maximum_program_pulses"]),
                device=device,
            )
            banks_cpu[int(assignment_seed)] = bank
            bank_path = _save_endpoint_bank(
                artifact_root / f"{role}/endpoint_bank.pt", bank
            )
            generated_artifacts.append(
                {"path": str(bank_path), "kind": f"{role}_shared_endpoint_bank"}
            )
            banks[int(assignment_seed)] = bank.to(device)

        source_development_distinctness = (
            _assert_pairwise_source_development_distinctness(
                pairs, all_source_seeds
            )
        )

        development_seed = int(contract["development_assignment"])
        development_templates = tuple(
            value.to(device) for value in templates_cpu[development_seed]
        )
        development_bank = banks[development_seed]
        development_initial = _evaluate_endpoint_bank(
            stack=stack,
            teacher=teacher,
            loader=loaders.validation,
            masters=initial_master,
            templates=development_templates,
            endpoint_bank=development_bank,
        )
        store.append_metric(
            {
                "mode": "development_epoch0_control",
                "assignment_seed": development_seed,
                "development": development_initial,
            }
        )

        hwa_histories: dict[str, list[Mapping[str, Any]]] = {}
        candidates: list[Mapping[str, Any]] = []
        candidate_masters: dict[tuple[str, int], tuple[torch.Tensor, ...]] = {}
        arm_generator_start_hashes: dict[str, str] = {}
        for arm in arms:
            arm_id = _arm_id(arm)
            source_schedule = _arm_source_seeds(arm)
            w2_factor = _arm_w2_factor(arm)
            w1_lr = float(_field(arm, "w1_learning_rate"))
            w2_lr = float(_field(arm, "w2_learning_rate"))
            if (
                not math.isclose(w1_lr, float(contract["w1_lr"]), abs_tol=1e-18)
                or not math.isclose(
                    w2_lr,
                    float(contract["base_w2_lr"]) * w2_factor,
                    abs_tol=1e-20,
                )
            ):
                raise ValueError(f"Arm {arm_id} learning rates drifted from protocol.")
            loaders.train_generator.set_state(hwa_initial_generator_state.clone())
            arm_generator_start_hashes[arm_id] = _tensor_sha256(
                loaders.train_generator.get_state()
            )
            masters = tuple(
                torch.nn.Parameter(value.to(device=device, dtype=torch.float32).clone())
                for value in initial_master
            )
            optimizer = torch.optim.SGD(
                (
                    {"params": [masters[0]], "lr": w1_lr},
                    {"params": [masters[1]], "lr": w2_lr},
                ),
                momentum=float(protocol.training.momentum),
                weight_decay=float(protocol.training.weight_decay),
            )
            selected_templates = {
                seed: tuple(value.to(device) for value in templates_cpu[seed])
                for seed in source_schedule
            }
            selected_banks = {seed: banks[seed] for seed in source_schedule}
            visits = {seed: 0 for seed in source_schedule}
            global_ordinal = 0
            history: list[Mapping[str, Any]] = []
            initial_for_device = tuple(value.to(device) for value in initial_master)
            for epoch in range(1, int(contract["hwa_epochs"]) + 1):
                train_report, global_ordinal, visits = _train_hwa_epoch(
                    stack=stack,
                    teacher=teacher,
                    loader=loaders.train,
                    masters=masters,
                    templates_by_assignment=selected_templates,
                    banks_by_assignment=selected_banks,
                    assignment_schedule=source_schedule,
                    optimizer=optimizer,
                    global_minibatch_ordinal=global_ordinal,
                    assignment_visit_ordinals=visits,
                    maximum_batches=student.settings.max_batches,
                )
                development = _evaluate_endpoint_bank(
                    stack=stack,
                    teacher=teacher,
                    loader=loaders.validation,
                    masters=masters,
                    templates=development_templates,
                    endpoint_bank=development_bank,
                )
                code_change = _code_change_report(
                    initial_for_device, masters, development_templates
                )
                record = {
                    "arm_id": arm_id,
                    "source_assignments": list(source_schedule),
                    "w2_learning_rate_multiplier": w2_factor,
                    "learning_rates": [w1_lr, w2_lr],
                    "epoch": epoch,
                    "train": train_report,
                    "development": development,
                    "development_code_change_from_epoch0": code_change,
                }
                history.append(record)
                store.append_metric({"mode": "multi_source_hwa", **record})
                candidate = {
                    "arm_id": arm_id,
                    "epoch": epoch,
                    "development": development,
                    "development_code_change_from_epoch0": code_change,
                }
                candidates.append(candidate)
                candidate_masters[(arm_id, epoch)] = _snapshot_masters(masters)
                print(
                    f"HWA arm={arm_id} epoch={epoch}/{contract['hwa_epochs']} "
                    f"dev_mean={100.0*float(development['persistent_mean_accuracy']):.2f}% "
                    f"dev_min={100.0*float(development['persistent_minimum_accuracy']):.2f}%",
                    flush=True,
                )
            if _tensor_sha256(hwa_initial_generator_state) != arm_generator_start_hashes[
                arm_id
            ]:
                raise RuntimeError("Matched HWA arm data-order start state changed.")
            hwa_histories[arm_id] = history
            del optimizer, masters, selected_templates, selected_banks
            gc.collect()

        winner = _select_hwa_candidate(candidates)
        winner_arm = str(winner["arm_id"])
        winner_epoch = int(winner["epoch"])
        winner_master = candidate_masters[(winner_arm, winner_epoch)]
        winner_checkpoint = atomic_torch_save(
            {
                "schema": "ebl.mnist_relu_drn.ibm_om_corrupt_multi_source_hwa_winner",
                "schema_version": 1,
                "evidence_tier": EVIDENCE_TIER,
                "arm_id": winner_arm,
                "epoch": winner_epoch,
                "logical_master": winner_master,
                "logical_master_sha256": [
                    _tensor_sha256(value) for value in winner_master
                ],
                "source_absmax": list(map(float, source_absmax)),
                "development": dict(winner["development"]),
                "development_code_change_from_epoch0": dict(
                    winner["development_code_change_from_epoch0"]
                ),
                "selection_key": list(_hwa_selection_key(winner)),
                "selection_assignment_seed": development_seed,
                "selection_endpoint_seeds": list(contract["development_endpoints"]),
                "epoch0_eligible": False,
                "target_94004_opened": False,
                "test_opened": False,
            },
            store.run_dir / "checkpoints/joint_hwa_winner.pt",
        )
        generated_artifacts.append(
            {"path": str(winner_checkpoint), "kind": "joint_hwa_winner"}
        )
        winner_checkpoint_sha256 = sha256_file(winner_checkpoint)

        target_seed = int(contract["target_assignment"])
        target_pair, target_artifacts = _sample_and_save_pair(
            bindings=stack.bundle.catalog.trainable,
            assignment_seed=target_seed,
            role=f"target_{target_seed}",
            sampler=sampler,
            artifact_root=artifact_root,
        )
        generated_artifacts.extend(target_artifacts)
        distinct_target_gates = {
            str(seed): _assert_distinct_array_pair(pairs[seed], target_pair)
            for seed in all_source_seeds
        }
        target_templates_cpu = build_repaired_mapping_templates(
            initial_master,
            target_pair,
            level_spacing_raw_x=float(contract["spacing"]),
        )
        initial_target = build_published_persistent_targets(
            initial_master, target_templates_cpu, target_pair
        )
        winner_target = build_published_persistent_targets(
            winner_master, target_templates_cpu, target_pair
        )
        initial_fault = clamp_targets_to_published_corrupt_singletons(
            initial_target.target_full_conductance, target_pair
        )
        winner_fault = clamp_targets_to_published_corrupt_singletons(
            winner_target.target_full_conductance, target_pair
        )
        target_population = target_pair.published
        target_pulse_population = array_population_as_pulse_population(
            target_population
        )
        runtime_protocol = _runtime_protocol_for_assignment(protocol, target_seed)
        initial_deployments_validation, initial_payloads = _deploy_pair(
            label="target_94004_epoch0",
            target=initial_target,
            endpoint_seeds=contract["target_endpoints"],
            population=target_population,
            pulse_population=target_pulse_population,
            stack=stack,
            teacher=teacher,
            evaluation_loader=loaders.validation,
            evaluation_label="validation",
            runtime_protocol=runtime_protocol,
            store=store,
            generated_artifacts=generated_artifacts,
        )
        winner_deployments_validation, winner_payloads = _deploy_pair(
            label="target_94004_winner",
            target=winner_target,
            endpoint_seeds=contract["target_endpoints"],
            population=target_population,
            pulse_population=target_pulse_population,
            stack=stack,
            teacher=teacher,
            evaluation_loader=loaders.validation,
            evaluation_label="validation",
            runtime_protocol=runtime_protocol,
            store=store,
            generated_artifacts=generated_artifacts,
        )
        p0_seed = int(contract["p0_endpoint"])
        p0_payload = dict(winner_payloads[p0_seed])
        p0_payload.update(
            {
                "p0_role": "selected_joint_hwa_target_94004_seed_94301",
                "source_hwa_checkpoint_sha256": winner_checkpoint_sha256,
                "no_remap_after_p0": True,
                "recovery_execution_state": recovery_initial_generator_record,
            }
        )
        p0_checkpoint = atomic_torch_save(
            p0_payload,
            store.run_dir / "checkpoints/target_94004_seed_94301_exact_p0.pt",
        )
        generated_artifacts.append(
            {"path": str(p0_checkpoint), "kind": "exact_target_p0_with_data_order"}
        )
        try:
            reloaded_p0 = torch.load(
                p0_checkpoint, map_location="cpu", weights_only=True
            )
        except (OSError, RuntimeError, TypeError, ValueError) as error:
            raise RuntimeError("Could not reload exact target P0.") from error
        p0_roundtrip = dict(_assert_p0_roundtrip(p0_payload, reloaded_p0))
        reloaded_execution = reloaded_p0.get("recovery_execution_state")
        if not isinstance(reloaded_execution, Mapping):
            raise RuntimeError("Reloaded P0 lost recovery execution state.")
        reloaded_generator_state = _assert_generator_state_record(reloaded_execution)
        if (
            int(reloaded_execution.get("seed", -1))
            != int(contract["recovery_data_order_seed"])
            or not torch.equal(
                reloaded_generator_state,
                recovery_initial_generator_record["state"],
            )
        ):
            raise RuntimeError("P0 recovery data-order roundtrip changed.")
        p0_roundtrip.update(
            {
                "recovery_data_order_seed_equal": True,
                "recovery_train_generator_state_equal": True,
                "recovery_train_generator_sha256": reloaded_execution["sha256"],
                "execution_exact_not_plant_only": True,
            }
        )
        p0_payload = dict(reloaded_p0)
        p0_payload["artifact_path"] = str(p0_checkpoint)
        recovery = _fine_tune_output_kl(
            p0=p0_payload,
            population=target_population,
            pulse_population=target_pulse_population,
            stack=stack,
            teacher=teacher,
            loaders=loaders,
            protocol=runtime_protocol,
            store=store,
            generated_artifacts=generated_artifacts,
        )

        # Both selections are now immutable.  Complete the terminal test phase:
        # source/development, target controls and saved P&V endpoints are all
        # evaluated without changing their physical states.  The selected
        # recovery checkpoint was the first post-freeze test evaluation above.
        development_test = {
            "epoch0": _evaluate_endpoint_bank(
                stack=stack,
                teacher=teacher,
                loader=loaders.test,
                masters=initial_master,
                templates=development_templates,
                endpoint_bank=development_bank,
            ),
            "winner": _evaluate_endpoint_bank(
                stack=stack,
                teacher=teacher,
                loader=loaders.test,
                masters=winner_master,
                templates=development_templates,
                endpoint_bank=development_bank,
            ),
        }
        target_no_write = {
            "epoch0": _evaluate_full_g(
                stack=stack,
                teacher=teacher,
                loader=loaders.test,
                full_g=initial_target.target_full_conductance,
            ),
            "winner": _evaluate_full_g(
                stack=stack,
                teacher=teacher,
                loader=loaders.test,
                full_g=winner_target.target_full_conductance,
            ),
        }
        target_fault_only = {
            "epoch0": {
                "fault_realization": dict(initial_fault.report),
                "evaluation": _evaluate_full_g(
                    stack=stack,
                    teacher=teacher,
                    loader=loaders.test,
                    full_g=initial_fault.realized_full_conductance,
                ),
            },
            "winner": {
                "fault_realization": dict(winner_fault.report),
                "evaluation": _evaluate_full_g(
                    stack=stack,
                    teacher=teacher,
                    loader=loaders.test,
                    full_g=winner_fault.realized_full_conductance,
                ),
            },
        }
        initial_deployments = [
            _replay_deployment_test(
                report,
                initial_payloads[int(report["endpoint_seed"])],
                population=target_population,
                stack=stack,
                teacher=teacher,
                test_loader=loaders.test,
            )
            for report in initial_deployments_validation
        ]
        winner_deployments = [
            _replay_deployment_test(
                report,
                winner_payloads[int(report["endpoint_seed"])],
                population=target_population,
                stack=stack,
                teacher=teacher,
                test_loader=loaders.test,
            )
            for report in winner_deployments_validation
        ]

        # Post-selection, non-selecting apples-to-apples comparator against the
        # old 93002 identity and exactly its five historical endpoint seeds.
        comparator_pair, comparator_artifacts = _sample_and_save_pair(
            bindings=stack.bundle.catalog.trainable,
            assignment_seed=COMPARATOR_ASSIGNMENT_SEED,
            role="post_selection_comparator_93002",
            sampler=sampler,
            artifact_root=artifact_root,
        )
        generated_artifacts.extend(comparator_artifacts)
        if (
            comparator_pair.published.fingerprint
            != COMPARATOR_PUBLISHED_FINGERPRINT
            or comparator_pair.repaired.fingerprint
            != COMPARATOR_REPAIRED_FINGERPRINT
        ):
            raise RuntimeError("Regenerated comparator 93002 identity changed.")
        comparator_templates = build_repaired_mapping_templates(
            initial_master,
            comparator_pair,
            level_spacing_raw_x=float(contract["spacing"]),
        )
        comparator_target = build_published_persistent_targets(
            winner_master, comparator_templates, comparator_pair
        )
        comparator_population = comparator_pair.published
        comparator_pulse_population = array_population_as_pulse_population(
            comparator_population
        )
        comparator_protocol = _runtime_protocol_for_assignment(
            protocol, COMPARATOR_ASSIGNMENT_SEED
        )
        comparator_deployments, _comparator_payloads = _deploy_pair(
            label="post_selection_comparator_93002_winner",
            target=comparator_target,
            endpoint_seeds=COMPARATOR_ENDPOINT_SEEDS,
            population=comparator_population,
            pulse_population=comparator_pulse_population,
            stack=stack,
            teacher=teacher,
            evaluation_loader=loaders.test,
            evaluation_label="test",
            runtime_protocol=comparator_protocol,
            store=store,
            generated_artifacts=generated_artifacts,
        )

        initial_target_stats = _endpoint_accuracies(initial_deployments)
        winner_target_stats = _endpoint_accuracies(winner_deployments)
        comparator_stats = _endpoint_accuracies(comparator_deployments)
        target_hwa_gate = _paired_target_hwa_gate(
            initial_deployments, winner_deployments
        )
        p0_before_test = next(
            item["test"]
            for item in winner_deployments
            if int(item["endpoint_seed"]) == p0_seed
        )
        recovery_gain = float(recovery["selected_test"]["student_accuracy"]) - float(
            p0_before_test["student_accuracy"]
        )
        recovery_gate = {
            "pre_recovery_accuracy": float(p0_before_test["student_accuracy"]),
            "post_recovery_accuracy": float(
                recovery["selected_test"]["student_accuracy"]
            ),
            "post_minus_pre_accuracy": recovery_gain,
            "minimum_gain": 0.10,
            "gain_passed": recovery_gain >= 0.10,
            "zero_verify_reads": recovery["pulse"]["verify_reads_during_updates"]
            == 0,
            "corrupt_cells_immobile": bool(
                recovery["corrupt_immobility"]["stuck_cells_immobile"]
            ),
        }
        recovery_gate["passed"] = all(
            (
                recovery_gate["gain_passed"],
                recovery_gate["zero_verify_reads"],
                recovery_gate["corrupt_cells_immobile"],
            )
        )

        bank_audits = {
            str(seed): _corrected_bank_programmability(
                banks_cpu[seed], pairs[seed].published
            )
            for seed in all_source_seeds
        }
        summary = {
            "schema": SUMMARY_SCHEMA,
            "schema_version": SUMMARY_SCHEMA_VERSION,
            "status": "complete",
            "evidence_tier": EVIDENCE_TIER,
            "claim_boundary": (
                "Exploratory model-based AIHWKit-1.1.0 IBM-OM ladder with literal "
                "published corrupt devices. HWA and recovery use teacher output-KL "
                "BPTT; recovery keeps digital Adam moments and issues open-loop "
                "stochastic pulse commands with zero verify reads. Persistent full "
                "G=a+1 only; no inference read noise, retention, or drift."
            ),
            "source_revision": store.manifest.get("source", {}),
            "teacher": {
                "path": str(teacher_path),
                "sha256": sha256_file(teacher_path),
                "metadata": teacher_metadata,
                "logical_weight_sha256": list(source_weight_hashes),
                "layer_absmax": list(source_absmax),
            },
            "protocol": to_plain_data(protocol),
            "data_order": {
                "hwa_arm_initial_generator_sha256": _tensor_sha256(
                    hwa_initial_generator_state
                ),
                "all_hwa_arm_start_states_equal": len(
                    set(arm_generator_start_hashes.values())
                )
                == 1,
                "hwa_arm_start_sha256": arm_generator_start_hashes,
                "recovery_dedicated_seed": int(
                    contract["recovery_data_order_seed"]
                ),
                "recovery_initial_generator_sha256": (
                    recovery_initial_generator_record["sha256"]
                ),
                "recovery_order_independent_of_HWA_arm_count": True,
            },
            "source_and_development": {
                "train_assignments": list(contract["train_assignments"]),
                "development_assignment": development_seed,
                "pairs": {
                    str(seed): dict(pairs[seed].report) for seed in all_source_seeds
                },
                "pairwise_distinctness": source_development_distinctness,
                "bank_programmability_corrected": bank_audits,
                "development_epoch0": development_initial,
                "arms": hwa_histories,
                "joint_selection": {
                    "candidate_count": len(candidates),
                    "epoch0_eligible": False,
                    "metric_order": list(protocol.selection.metric_order),
                    "simplicity_order": list(SIMPLICITY_ORDER),
                    "winner_arm": winner_arm,
                    "winner_epoch": winner_epoch,
                    "winner_development": dict(winner["development"]),
                    "winner_code_change_from_epoch0": dict(
                        winner["development_code_change_from_epoch0"]
                    ),
                    "checkpoint": str(winner_checkpoint),
                    "checkpoint_sha256": winner_checkpoint_sha256,
                    "target_94004_opened_before_freeze": False,
                    "test_opened_before_freeze": False,
                },
                "test_after_joint_selection_freeze": development_test,
            },
            "target_94004": {
                "opened_after_winner_checkpoint_sha256": winner_checkpoint_sha256,
                "pair": dict(target_pair.report),
                "published_population": _population_report(target_pair.published),
                "repaired_mapping_twin": _population_report(target_pair.repaired),
                "distinct_from_all_source_and_development": distinct_target_gates,
                "mapping": {
                    "epoch0": dict(initial_target.report),
                    "winner": dict(winner_target.report),
                    "target_clipping": False,
                    "full_conductance_formula": "G=2*x=a+1",
                },
                "ideal_requested_no_write": target_no_write,
                "deterministic_fault_only": target_fault_only,
                "program_verify": {
                    "endpoint_seeds": list(contract["target_endpoints"]),
                    "epoch0": {
                        "repeats": initial_deployments,
                        **initial_target_stats,
                    },
                    "winner": {
                        "repeats": winner_deployments,
                        **winner_target_stats,
                    },
                    "paired_hwa_gate": target_hwa_gate,
                },
                "p0": {
                    "endpoint_seed": p0_seed,
                    "path": str(p0_checkpoint),
                    "sha256": sha256_file(p0_checkpoint),
                    "exact_plant_and_data_order_continuation": True,
                    "roundtrip": p0_roundtrip,
                },
                "output_kl_open_loop_adam": recovery,
                "recovery_gate": recovery_gate,
            },
            "post_selection_comparator_93002": {
                "used_for_selection": False,
                "opened_after_winner_checkpoint_sha256": winner_checkpoint_sha256,
                "assignment_seed": COMPARATOR_ASSIGNMENT_SEED,
                "endpoint_seeds": list(COMPARATOR_ENDPOINT_SEEDS),
                "published_fingerprint": comparator_pair.published.fingerprint,
                "repaired_fingerprint": comparator_pair.repaired.fingerprint,
                "repeats": comparator_deployments,
                **comparator_stats,
                "historical_v1_selected_hwa_mean_accuracy": (
                    COMPARATOR_HISTORICAL_MEAN
                ),
                "new_winner_minus_historical_mean": (
                    float(comparator_stats["mean"]) - COMPARATOR_HISTORICAL_MEAN
                ),
            },
            "accuracy_ladder": {
                "development_94003_epoch0_persistent_mean_test_accuracy": float(
                    development_test["epoch0"]["persistent_mean_accuracy"]
                ),
                "development_94003_winner_persistent_mean_test_accuracy": float(
                    development_test["winner"]["persistent_mean_accuracy"]
                ),
                "target_94004_epoch0_fault_only_test_accuracy": float(
                    target_fault_only["epoch0"]["evaluation"]["metrics"][
                        "student_accuracy"
                    ]
                ),
                "target_94004_winner_fault_only_test_accuracy": float(
                    target_fault_only["winner"]["evaluation"]["metrics"][
                        "student_accuracy"
                    ]
                ),
                "target_94004_epoch0_pv_mean_test_accuracy": float(
                    initial_target_stats["mean"]
                ),
                "target_94004_winner_pv_mean_test_accuracy": float(
                    winner_target_stats["mean"]
                ),
                "target_94004_seed94301_pre_recovery_test_accuracy": float(
                    p0_before_test["student_accuracy"]
                ),
                "target_94004_selected_recovery_test_accuracy": float(
                    recovery["selected_test"]["student_accuracy"]
                ),
                "comparator_93002_winner_pv_mean_test_accuracy": float(
                    comparator_stats["mean"]
                ),
            },
            "invariants": {
                "full_conductance_formula": "G=2*x=a+1",
                "negative_conductance_count": 0,
                "joint_winner_frozen_before_target_94004_open": True,
                "test_sealed_until_joint_hwa_and_recovery_selection_freeze": True,
                "recovery_test_used_for_selection": False,
                "recovery_uses_dedicated_serialized_data_order": True,
                "verify_reads_during_recovery": 0,
                "issued_commands_reported_separately_from_state_change_events": True,
                "corrupt_supported_not_counted_as_programmable_success": True,
            },
            "limitations": [
                "exploratory_noncanonical_two_training_one_development_one_target_assignment",
                "hardware_derived_fitted_OM_model_not_raw_measured_trace_replay",
                "counterfactual_repaired_twins_are_mapping_oracles_only",
                "published_corrupt_persistent_forwards_use_nominal_repaired_envelope_STE",
                "teacher_output_KL_BPTT_and_digital_optimizer_state",
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
            "winner_arm": winner_arm,
            "winner_epoch": winner_epoch,
            "development_winner_persistent_mean_test_accuracy": float(
                development_test["winner"]["persistent_mean_accuracy"]
            ),
            "target_epoch0_pv_mean_test_accuracy": float(initial_target_stats["mean"]),
            "target_winner_pv_mean_test_accuracy": float(winner_target_stats["mean"]),
            "target_paired_hwa_mean_gain": float(
                target_hwa_gate["mean_selected_minus_epoch0_accuracy"]
            ),
            "target_paired_hwa_gate_passed": bool(target_hwa_gate["passed"]),
            "recovery_selected_epoch": int(recovery["selected_epoch"]),
            "target_pre_recovery_test_accuracy": float(
                p0_before_test["student_accuracy"]
            ),
            "target_post_recovery_test_accuracy": float(
                recovery["selected_test"]["student_accuracy"]
            ),
            "target_recovery_gain": recovery_gain,
            "target_recovery_gate_passed": bool(recovery_gate["passed"]),
            "comparator_93002_pv_mean_test_accuracy": float(comparator_stats["mean"]),
            "verify_reads_during_recovery": 0,
            "negative_conductance_count": 0,
        }
        store.append_metric({"mode": "train_terminal", **terminal})
        store.complete(
            metrics=terminal,
            artifacts=_artifact_records(store, generated_artifacts),
        )
        print(f"run_dir={store.run_dir}", flush=True)
        gc.collect()
        torch.cuda.empty_cache()
        return 0
    except BaseException as error:
        store.fail(error)
        raise


__all__ = [
    "_arm_source_seeds",
    "_arm_w2_factor",
    "_assert_p0_validation_parity",
    "_assert_pairwise_source_development_distinctness",
    "_corrected_bank_programmability",
    "_generator_state_record",
    "_hwa_selection_key",
    "_output_kl_gradients",
    "_recovery_selection_key",
    "_select_hwa_candidate",
    "_strict_protocol_values",
    "_train_hwa_epoch",
    "run_train",
]
