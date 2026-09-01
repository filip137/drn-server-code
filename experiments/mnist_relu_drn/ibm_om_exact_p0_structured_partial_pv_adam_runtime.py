"""CUDA runtime for structured partial P&V Adam recovery from exact P0."""

from __future__ import annotations

from hashlib import sha256
import gc
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence, TYPE_CHECKING

import torch

from experiments.artifacts import RunStore, atomic_write_json, sha256_file
from experiments.mnist_relu_drn.components import build_student_stack
from experiments.mnist_relu_drn.ibm_om_baseline_selection import quad_stack
from experiments.mnist_relu_drn.ibm_om_baseline_selection_runtime import (
    _artifact_records,
    _evaluate_detailed,
    _input,
)
from experiments.mnist_relu_drn.ibm_om_bounded_codebook_scheme_screen import (
    _tensor_sha256,
)
from experiments.mnist_relu_drn.ibm_om_corrupt_multi_source_hwa_tuned_recovery_runtime import (
    _generator_state_record,
    _output_kl_gradients,
)
from experiments.mnist_relu_drn.ibm_om_exact_p0_open_vs_closed_loop_adam_runtime import (
    EVIDENCE_TIER,
    _evaluate_payload_test,
    _load_device_model,
    _load_exact_p0,
    _run_arm,
    _strict_protocol as _strict_base_protocol,
)
from experiments.mnist_relu_drn.ibm_om_exact_p0_structured_partial_pv_adam import (
    ARM_IDS,
    LAYOUTS,
    StructuredPartialUpdateMask,
    build_structured_partial_update_masks,
)
from experiments.mnist_relu_drn.ibm_om_winsorized_onchip_adam import (
    sync_catalog_from_persistent,
)
from experiments.mnist_relu_drn.ibm_om_winsorized_cross_array_open_loop_adam_runtime import (
    _restore_deployment_plant,
)
from experiments.mnist_relu_drn.runtime import _load_teacher
from experiments.mnist_shared import build_mnist_loaders
from experiments.schema import to_plain_data
from training.checkpoint import atomic_torch_save
from training.ibm_reram_raw_active_program_verify import (
    array_population_as_pulse_population,
)


if TYPE_CHECKING:
    from ebl.cli import TrainRequest


_ROOT = Path(__file__).resolve().parents[2]
SUMMARY_SCHEMA = "ebl.mnist_relu_drn.ibm_om_exact_p0_structured_partial_pv_adam"
SUMMARY_SCHEMA_VERSION = 1
CLOSED_WRITER_ARM_ID = "incremental_one_pulse_closed_loop_target_tracking"

_FULL_VALIDATION = (
    {
        "correct": 4412,
        "kl": 0.3655388344794512,
        "prediction": "36b0f52276e1e2bee089e6ecad3e9f56205797e6e83f28ea32f9bf6e88d39ca0",
        "issued": 157234,
        "effective": 130800,
        "train_state": "56ad35bc1bb14e7274309735fc523734d65ce630057cb14ee2b535c92edb2c2f",
    },
    {
        "correct": 4577,
        "kl": 0.24129868309125305,
        "prediction": "9887d984353245e70280e819c6d6d0e45d8b17812c2222c71d200e865ba080cc",
        "issued": 240003,
        "effective": 197536,
        "train_state": "6fd2daf706ce9d027c809a8d47ed8c04e5f19467506022049f13231a85c2c037",
    },
    {
        "correct": 4635,
        "kl": 0.18168440887704493,
        "prediction": "2d08c0527f65532c28f0787098dec90af862c543c77b032141bb4a51b7a8c104",
        "issued": 303231,
        "effective": 247243,
        "train_state": "9f12c6bbeda63372fecc36702b8c5a5795a9510bc10f38327726aba24b0bd8a2",
    },
)
_FULL_TEST = {
    "correct": 9334,
    "kl": 0.14868692288342863,
    "prediction": "f49ac3f3bed9f5190ca2bc2eb2bc2ee8495faeeea12d4d47e27a0409f7728f1c",
}


def _float_summary(value: torch.Tensor) -> dict[str, float]:
    flat = value.detach().to(device="cpu", dtype=torch.float64).reshape(-1)
    if flat.numel() < 1 or not bool(torch.all(torch.isfinite(flat))):
        raise ValueError("Expected finite nonempty logical ranking scores.")
    return {
        "minimum": float(flat.min().item()),
        "mean": float(flat.mean().item()),
        "rms": float(flat.square().mean().sqrt().item()),
        "maximum": float(flat.max().item()),
    }


def _quad_score_batch(gradient_dL_dG: torch.Tensor, *, layout: str) -> torch.Tensor:
    """Return quad-mean absolute raw-x gradient for one calibration batch."""

    if gradient_dL_dG.ndim != 2 or not bool(
        torch.all(torch.isfinite(gradient_dL_dG))
    ):
        raise ValueError("Expected one finite rank-2 physical conductance gradient.")
    raw_x_quad_gradient = quad_stack(2.0 * gradient_dL_dG.detach(), layout=layout)
    return raw_x_quad_gradient.abs().mean(dim=-1).to(
        device="cpu", dtype=torch.float64
    )


def _strict_protocol(protocol: Any) -> None:
    _strict_base_protocol(protocol.base)
    partial = protocol.partial
    if (
        tuple(arm.arm_id for arm in partial.arms) != ARM_IDS
        or partial.controller != CLOSED_WRITER_ARM_ID
        or partial.epochs != 3
        or partial.matched_logical_count != 500
        or partial.mask_unit != "canonical_four_physical_cell_logical_quad"
        or partial.mask_application != ("Adam_target_updates", "pulse_eligibility")
        or partial.ranking.examples != 1024
        or partial.ranking.batch_size != 128
        or partial.ranking.minibatches != 8
        or partial.ranking.canonical_layouts != LAYOUTS
        or partial.ranking.uses_new_seed
        or partial.ranking.uses_device_bounds
        or partial.ranking.uses_corrupt_mask
        or partial.ranking.uses_validation
        or partial.ranking.uses_test
    ):
        raise ValueError("Resolved structured partial-recovery protocol drifted.")
    expected_counts = {
        "full": (39_700, 158_800),
        "w1_only": (39_200, 156_800),
        "w2_only": (500, 2_000),
        "w1_top500": (500, 2_000),
        "global_top500": (500, 2_000),
    }
    for arm in partial.arms:
        if (arm.expected_logical_quads, arm.expected_physical_cells) != expected_counts[
            arm.arm_id
        ]:
            raise ValueError("Structured partial arm count contract drifted.")


def _state_equal(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    return all(
        torch.equal(left[name].detach().cpu(), right[name].detach().cpu())
        for name in ("persistent", "apparent", "draw_indices")
    )


def _rank_partial_masks(
    *,
    p0: Mapping[str, Any],
    population: Any,
    pulse_population: Any,
    stack: Any,
    teacher: Any,
    loaders: Any,
    protocol: Any,
) -> tuple[Mapping[str, StructuredPartialUpdateMask], Mapping[str, Any]]:
    """Rank logical quads at exact P0 on the frozen train-only cohort."""

    before_train_state = _generator_state_record(loaders.train_generator.get_state())
    plant = _restore_deployment_plant(
        p0,
        pulse_population=pulse_population,
        maximum_random_draws=protocol.base.recovery.maximum_random_draws,
        device=stack.device,
    )
    before_plant = plant.state_dict()
    sync_catalog_from_persistent(
        stack.bundle.catalog, plant, binding_shapes=population.binding_shapes
    )
    score_sums: list[torch.Tensor] | None = None
    cohort_digest = sha256()
    input_digest = sha256()
    label_digest = sha256()
    examples = 0
    batches = 0
    for batch_index, (inputs_cpu, labels_cpu) in enumerate(loaders.calibration):
        cohort_digest.update(str(batch_index).encode("utf-8"))
        input_hash = _tensor_sha256(inputs_cpu)
        label_hash = _tensor_sha256(labels_cpu.to(torch.int64))
        cohort_digest.update(input_hash.encode("ascii"))
        cohort_digest.update(label_hash.encode("ascii"))
        input_digest.update(input_hash.encode("ascii"))
        label_digest.update(label_hash.encode("ascii"))
        inputs = inputs_cpu.to(stack.device, dtype=torch.float32)
        labels = labels_cpu.to(stack.device, dtype=torch.long)
        with torch.no_grad():
            teacher_logits = teacher.logits(inputs)
        stack.network.set_input(inputs, reset=True)
        stack.minimizer.compute_equilibrium()
        stack.cost.set_teacher(teacher_logits, labels)
        gradients = _output_kl_gradients(
            tuple(stack.differentiator.compute_gradient())
        )
        batch_scores = []
        for gradient, layout in zip(gradients, LAYOUTS, strict=True):
            batch_scores.append(_quad_score_batch(gradient, layout=layout))
        if score_sums is None:
            score_sums = [torch.zeros_like(value) for value in batch_scores]
        for cumulative, value in zip(score_sums, batch_scores, strict=True):
            cumulative.add_(value)
        examples += int(labels.numel())
        batches = batch_index + 1
    if score_sums is None or examples != 1024 or batches != 8:
        raise RuntimeError("The frozen train-only ranking cohort changed.")
    scores = tuple(value / batches for value in score_sums)
    after_train_state = _generator_state_record(loaders.train_generator.get_state())
    after_plant = plant.state_dict()
    train_state_unchanged = (
        before_train_state["sha256"] == after_train_state["sha256"]
        and before_train_state["bytes"] == after_train_state["bytes"]
        and torch.equal(before_train_state["state"], after_train_state["state"])
    )
    if not train_state_unchanged or not _state_equal(before_plant, after_plant):
        raise RuntimeError("Mask ranking changed the P0 plant or training data RNG.")
    masks = build_structured_partial_update_masks(
        scores,
        binding_shapes=population.binding_shapes,
        layouts=LAYOUTS,
        matched_logical_count=protocol.partial.matched_logical_count,
    )
    ranking = {
        "cohort": protocol.partial.ranking.cohort,
        "examples": examples,
        "batches": batches,
        "batch_size": protocol.partial.ranking.batch_size,
        "cohort_sha256": cohort_digest.hexdigest(),
        "input_batches_sha256": input_digest.hexdigest(),
        "label_batches_sha256": label_digest.hexdigest(),
        "objective": protocol.partial.ranking.objective,
        "aggregation": protocol.partial.ranking.aggregation,
        "tie_break": protocol.partial.ranking.tie_break,
        "global_flat_order": protocol.partial.ranking.global_flat_order,
        "score_sha256_by_layer": [_tensor_sha256(value) for value in scores],
        "score_summary_by_layer": [_float_summary(value) for value in scores],
        "train_generator_unchanged": True,
        "p0_plant_persistent_apparent_and_draw_indices_unchanged": True,
        "selector_inputs": {
            "training_calibration_inputs": True,
            "teacher_output_KL_gradients": True,
            "device_bounds": False,
            "published_corrupt_mask": False,
            "validation": False,
            "test": False,
        },
        "selector_overhead": {
            "dense_BPTT_gradient_evaluations": batches,
            "examples": examples,
            "logical_quads_scored": 39_700,
            "physical_cells_differentiated": population.size,
            "programming_pulses": 0,
            "verify_calls": 0,
            "classification": "off_chip_global_training_only_ranking_heuristic",
        },
    }
    return masks, {"scores": scores, "ranking": ranking}


def _save_mask_artifacts(
    *,
    store: RunStore,
    p0_sha256: str,
    masks: Mapping[str, StructuredPartialUpdateMask],
    ranking_payload: Mapping[str, Any],
    provenance: Mapping[str, Any],
) -> tuple[list[Mapping[str, Any]], Mapping[str, Any]]:
    overlap: dict[str, Mapping[str, Any]] = {}
    for left_index, left_id in enumerate(ARM_IDS):
        left = torch.cat(
            tuple(value.reshape(-1) for value in masks[left_id].logical_masks)
        )
        for right_id in ARM_IDS[left_index + 1 :]:
            right = torch.cat(
                tuple(value.reshape(-1) for value in masks[right_id].logical_masks)
            )
            intersection = int((left & right).sum().item())
            union = int((left | right).sum().item())
            overlap[f"{left_id}__{right_id}"] = {
                "logical_intersection": intersection,
                "logical_union": union,
                "logical_jaccard": intersection / union,
                "identical": bool(torch.equal(left, right)),
            }
    payload = {
        "schema": "ebl.mnist_relu_drn.ibm_om_structured_partial_update_masks",
        "schema_version": 1,
        "evidence_tier": EVIDENCE_TIER,
        "source_p0_sha256": p0_sha256,
        "provenance": dict(provenance),
        "ranking": ranking_payload["ranking"],
        "quad_scores": [value.detach().cpu() for value in ranking_payload["scores"]],
        "arms": {
            arm_id: {
                "logical_masks": [value.detach().cpu() for value in mask.logical_masks],
                "physical_masks": [value.detach().cpu() for value in mask.physical_masks],
                "report": mask.report(),
            }
            for arm_id, mask in masks.items()
        },
        "pairwise_logical_mask_overlap": overlap,
    }
    path = atomic_torch_save(payload, store.run_dir / "artifacts/partial_update_masks.pt")
    receipt = {
        "schema": payload["schema"] + ".receipt",
        "schema_version": 1,
        "path": str(path),
        "sha256": sha256_file(path),
        "source_p0_sha256": p0_sha256,
        "provenance": dict(provenance),
        "ranking": ranking_payload["ranking"],
        "arms": {arm_id: mask.report() for arm_id, mask in masks.items()},
        "pairwise_logical_mask_overlap": overlap,
    }
    receipt_path = store.run_dir / "artifacts/partial_update_masks.receipt.json"
    atomic_write_json(receipt_path, receipt)
    return (
        [
            {"path": str(path), "kind": "structured_partial_update_masks"},
            {"path": str(receipt_path), "kind": "structured_partial_update_masks_receipt"},
        ],
        receipt,
    )


def _selected_epoch_row(report: Mapping[str, Any]) -> Mapping[str, Any]:
    selected = int(report["selected_epoch"])
    return next(row for row in report["epoch_reports"] if int(row["epoch"]) == selected)


def _assert_full_historical_epoch_parity(report: Mapping[str, Any]) -> None:
    rows = report["epoch_reports"]
    if len(rows) != 3 or int(report["selected_epoch"]) != 3:
        raise RuntimeError("Full-mask historical epoch coverage changed.")
    for row, expected in zip(rows, _FULL_VALIDATION, strict=True):
        validation = row["validation"]
        if (
            int(validation["student_correct"]) != expected["correct"]
            or row["validation_prediction_sha256"] != expected["prediction"]
            or int(row["issued_commands_cumulative"]) != expected["issued"]
            or int(row["effective_state_change_events_cumulative"])
            != expected["effective"]
            or row["train_generator_state_sha256"] != expected["train_state"]
            or not math.isclose(
                float(validation["kl_teacher_student"]),
                float(expected["kl"]),
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        ):
            raise RuntimeError("Full-mask recovery no longer matches prior evidence.")


def run_train(request: "TrainRequest") -> int:
    from experiments.mnist_relu_drn.ibm_om_exact_p0_structured_partial_pv_adam_config import (
        ExactP0StructuredPartialPvAdamTrainSpec,
    )

    spec = request.spec
    if not isinstance(spec, ExactP0StructuredPartialPvAdamTrainSpec):
        raise TypeError("Expected the dedicated structured partial-recovery train spec.")
    if request.teacher_weights is None or request.weights is None or request.device_model is None:
        raise ValueError("Expected explicit --teacher-weights, --weights exact P0, and --device-model.")
    for name in ("base_weights", "resume", "device_data"):
        if getattr(request, name) is not None:
            raise ValueError(f"Structured partial recovery does not accept {name}.")
    protocol = spec.protocol
    _strict_protocol(protocol)
    base = protocol.base
    teacher_path = request.teacher_weights.expanduser().resolve()
    p0_path = request.weights.expanduser().resolve()
    device_model_path = request.device_model.expanduser().resolve()
    if not teacher_path.is_file() or sha256_file(teacher_path) != base.expected_teacher_weights_sha256:
        raise ValueError("Frozen teacher SHA-256 mismatch.")
    population, population_path, device_receipt = _load_device_model(
        device_model_path, base
    )
    p0 = _load_exact_p0(p0_path, protocol=base, population=population)
    pulse_population = array_population_as_pulse_population(population)
    store = RunStore.create(
        output_root=request.output_dir,
        experiment_id=spec.experiment_id,
        resolved_config=to_plain_data(spec),
        command=request.command,
        repo_root=_ROOT,
        input_artifacts=(
            _input("teacher_weights", teacher_path),
            _input("exact_target_p0", p0_path),
            _input("paired_target_device_model", device_model_path),
            _input("published_target_population", population_path),
        ),
        resume_capability="unsupported",
    )
    try:
        student = spec.student
        if student.runtime.device != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("Structured partial P&V recovery requires CUDA.")
        torch.manual_seed(student.runtime.seed)
        torch.cuda.manual_seed_all(student.runtime.seed)
        loaders = build_mnist_loaders(
            student.data,
            data_seed=student.runtime.data_seed,
            calibration_examples=student.mapping.calibration_examples,
            calibration_batch_size=student.mapping.calibration_batch_size,
        )
        teacher, teacher_metadata = _load_teacher(
            teacher_path, device=torch.device("cuda"), spec=student
        )
        stack = build_student_stack(student, enable_measured=False)
        stack.cost.gain = base.recovery.fixed_logit_gain
        masks, ranking_payload = _rank_partial_masks(
            p0=p0,
            population=population,
            pulse_population=pulse_population,
            stack=stack,
            teacher=teacher,
            loaders=loaders,
            protocol=protocol,
        )
        generated_artifacts, mask_receipt = _save_mask_artifacts(
            store=store,
            p0_sha256=base.p0.sha256,
            masks=masks,
            ranking_payload=ranking_payload,
            provenance={
                "teacher_weights_sha256": sha256_file(teacher_path),
                "exact_p0_sha256": sha256_file(p0_path),
                "source_hwa_checkpoint_sha256": (
                    base.p0.source_hwa_checkpoint_sha256
                ),
                "paired_device_model_receipt_sha256": sha256_file(
                    device_model_path
                ),
                "published_population_sha256": sha256_file(population_path),
                "published_population_fingerprint": population.fingerprint,
            },
        )
        arm_reports: dict[str, Mapping[str, Any]] = {}
        selected_payloads: dict[str, Mapping[str, Any]] = {}
        for arm_id in ARM_IDS:
            contract = next(value for value in protocol.partial.arms if value.arm_id == arm_id)
            mask = masks[arm_id]
            observed = mask.report()
            if (
                int(observed["logical_weights"]) != contract.expected_logical_quads
                or int(observed["physical_cells"]) != contract.expected_physical_cells
            ):
                raise RuntimeError("Frozen structured mask count differs from config.")
            report, selected_payload = _run_arm(
                CLOSED_WRITER_ARM_ID,
                artifact_arm_id=arm_id,
                trainable_cell_mask=mask.flat_physical_mask,
                p0=p0,
                population=population,
                pulse_population=pulse_population,
                stack=stack,
                teacher=teacher,
                loaders=loaders,
                protocol=base,
                store=store,
                generated_artifacts=generated_artifacts,
            )
            arm_reports[arm_id] = report
            selected_payloads[arm_id] = selected_payload
            if arm_id == "full":
                _assert_full_historical_epoch_parity(report)
        order_hashes = {
            arm_id: [row["train_generator_state_sha256"] for row in report["epoch_reports"]]
            for arm_id, report in arm_reports.items()
        }
        if any(value != order_hashes["full"] for value in order_hashes.values()):
            raise RuntimeError("Partial recovery arms consumed different minibatch orders.")

        # All five validation-only epoch selections are frozen before test access.
        p0_test = _evaluate_payload_test(
            p0,
            population=population,
            pulse_population=pulse_population,
            stack=stack,
            teacher=teacher,
            test_loader=loaders.test,
            maximum_random_draws=base.recovery.maximum_random_draws,
        )
        terminal_tests: dict[str, Mapping[str, Any]] = {}
        for arm_id in ARM_IDS:
            terminal_tests[arm_id] = _evaluate_payload_test(
                selected_payloads[arm_id],
                population=population,
                pulse_population=pulse_population,
                stack=stack,
                teacher=teacher,
                test_loader=loaders.test,
                maximum_random_draws=base.recovery.maximum_random_draws,
            )
            arm_reports[arm_id] = {
                **dict(arm_reports[arm_id]),
                "selected_test": terminal_tests[arm_id],
                "test_opened": True,
            }
        historical = base.historical_open_loop_parity
        full_test = terminal_tests["full"]
        if (
            int(p0_test["student_correct"]) != historical.p0_test_correct
            or p0_test["prediction_sha256"] != historical.p0_test_prediction_sha256
            or int(full_test["student_correct"]) != _FULL_TEST["correct"]
            or full_test["prediction_sha256"] != _FULL_TEST["prediction"]
            or not math.isclose(
                float(full_test["kl_teacher_student"]),
                float(_FULL_TEST["kl"]),
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        ):
            raise RuntimeError("P0/full-mask terminal historical parity failed.")

        full_selected = _selected_epoch_row(arm_reports["full"])
        full_val = float(full_selected["validation"]["student_accuracy"])
        full_pulses = int(full_selected["issued_commands_cumulative"])
        comparisons: dict[str, Mapping[str, Any]] = {}
        for arm_id in ARM_IDS:
            selected = _selected_epoch_row(arm_reports[arm_id])
            epoch_rows = list(arm_reports[arm_id]["epoch_reports"])
            selected_epoch = int(arm_reports[arm_id]["selected_epoch"])
            selected_rows = [
                row for row in epoch_rows if int(row["epoch"]) <= selected_epoch
            ]
            final_row = epoch_rows[-1]
            val = float(selected["validation"]["student_accuracy"])
            pulses = int(selected["issued_commands_cumulative"])
            test = float(terminal_tests[arm_id]["student_accuracy"])
            selected_api_calls = sum(
                int(row["api_full_port_verify_calls_epoch"])
                for row in selected_rows
            )
            full_horizon_api_calls = sum(
                int(row["api_full_port_verify_calls_epoch"]) for row in epoch_rows
            )
            comparisons[arm_id] = {
                "selected_epoch": selected_epoch,
                "validation_correct": int(
                    selected["validation"]["student_correct"]
                ),
                "validation_examples": int(selected["validation"]["examples"]),
                "validation_accuracy": val,
                "validation_KL": float(
                    selected["validation"]["kl_teacher_student"]
                ),
                "validation_minus_full": val - full_val,
                "test_correct": int(terminal_tests[arm_id]["student_correct"]),
                "test_examples": int(terminal_tests[arm_id]["examples"]),
                "test_accuracy": test,
                "test_KL": float(
                    terminal_tests[arm_id]["kl_teacher_student"]
                ),
                "test_teacher_agreement": float(
                    terminal_tests[arm_id]["teacher_agreement"]
                ),
                "test_minus_full": test - float(full_test["student_accuracy"]),
                "selected_checkpoint_cost": {
                    "issued_commands": pulses,
                    "effective_state_change_events": int(
                        selected["effective_state_change_events_cumulative"]
                    ),
                    "api_full_port_verify_calls": selected_api_calls,
                    "software_full_port_values_materialized": (
                        selected_api_calls * population.size
                    ),
                    "conceptual_pulsed_cell_observations": sum(
                        int(row["conceptual_post_pulse_cell_observations_epoch"])
                        for row in selected_rows
                    ),
                },
                "executed_three_epoch_selection_cost": {
                    "issued_commands": int(final_row["issued_commands_cumulative"]),
                    "effective_state_change_events": int(
                        final_row["effective_state_change_events_cumulative"]
                    ),
                    "api_full_port_verify_calls": full_horizon_api_calls,
                    "software_full_port_values_materialized": (
                        full_horizon_api_calls * population.size
                    ),
                    "conceptual_pulsed_cell_observations": sum(
                        int(row["conceptual_post_pulse_cell_observations_epoch"])
                        for row in epoch_rows
                    ),
                    "discarded_post_selection_issued_commands": (
                        int(final_row["issued_commands_cumulative"]) - pulses
                    ),
                },
                "selected_issued_commands_fraction_of_full": pulses / full_pulses,
                "recovered_full_gain_fraction": (
                    (test - float(p0_test["student_accuracy"]))
                    / (float(full_test["student_accuracy"]) - float(p0_test["student_accuracy"]))
                ),
                "within_one_validation_point_of_full": val >= full_val - 0.01,
                "at_most_one_quarter_full_pulses": pulses <= 0.25 * full_pulses,
            }
        primary_sparse = ("w2_only", "w1_top500", "global_top500")
        summary = {
            "schema": SUMMARY_SCHEMA,
            "schema_version": SUMMARY_SCHEMA_VERSION,
            "evidence_tier": EVIDENCE_TIER,
            "status": "completed",
            "question": (
                "How much exact-P0 recovery is retained when only a fixed, "
                "quad-coherent subset of physical conductances may update?"
            ),
            "teacher": {
                "path": str(teacher_path),
                "sha256": sha256_file(teacher_path),
                **teacher_metadata,
            },
            "exact_p0": {
                "path": str(p0_path),
                "sha256": sha256_file(p0_path),
                "assignment_seed": base.p0.assignment_seed,
                "endpoint_seed": base.p0.endpoint_seed,
                "test": p0_test,
                "initial_programming_cost": {
                    "issued_pulses": int(p0["programming"]["total_pulses"].sum().item()),
                    "verify_observations": int(
                        p0["programming"]["verify_count"].sum().item()
                    ),
                    "excluded_from_recovery_arm_cost_ratios": True,
                },
            },
            "device_model": {
                "receipt_path": str(device_model_path),
                "receipt_sha256": sha256_file(device_model_path),
                "population_path": str(population_path),
                "population_sha256": sha256_file(population_path),
                "population_fingerprint": population.fingerprint,
                "corrupt_cells": int(population.corrupt.sum().item()),
                "receipt_role": device_receipt["role"],
            },
            "mask_artifact": mask_receipt,
            "protocol": {
                "arms": list(ARM_IDS),
                "mask_unit": protocol.partial.mask_unit,
                "frozen_cell_policy": protocol.partial.frozen_cell_policy,
                "objective": base.recovery.objective,
                "optimizer": base.recovery.optimizer,
                "controller": protocol.partial.controller,
                "epochs": base.recovery.epochs,
                "learning_rate_raw_x": base.recovery.learning_rate_raw_x,
                "pulse_cap": base.recovery.pulse_cap,
                "persistent_forward": "G=a+1=2*x",
                "test_policy": protocol.partial.test_policy,
            },
            "arms": arm_reports,
            "comparison": {
                "by_arm": comparisons,
                "primary_eligible_address_matched_500_quad_arms": list(primary_sparse),
                "primary_success_arms": [
                    arm_id
                    for arm_id in primary_sparse
                    if comparisons[arm_id]["within_one_validation_point_of_full"]
                    and comparisons[arm_id]["at_most_one_quarter_full_pulses"]
                ],
                "selection_is_validation_only_within_each_arm": True,
                "success_cost_gate_is_selected_checkpoint_pulses_only": True,
                "verify_API_and_software_full_port_materialization_reported_separately": True,
            },
            "matched_gates": {
                "same_exact_p0": True,
                "same_three_epoch_minibatch_order_hashes": True,
                "all_masks_frozen_before_arm_validation_and_test": True,
                "all_four_cells_per_selected_logical_weight": True,
                "frozen_cells_exact_in_plant_rng_and_optimizer_state": True,
                "fault_mask_not_used_by_selector_or_controller": True,
                "persistent_G_nonnegative": True,
                "published_corrupt_cells_immobile": True,
                "test_sealed_until_all_epoch_selections_frozen": True,
                "full_mask_historical_parity_exact": True,
            },
            "limitations": [
                "exploratory_noncanonical_single_previously_inspected_target_endpoint",
                "model_based_aihwkit_1.1.0_IBM_OM_fitted_preset_not_raw_measured_traces",
                "static_P0_gradient_ranking_may_not_capture_later_update_turnover",
                "500_quad_arms_match_eligible_addresses_not_realized_total_cost",
                "digital_full_BPTT_and_Adam_are_not_sparse_or_autonomous_on_chip",
                "vectorized_verify_call_cost_need_not_scale_with_trainable_cell_count",
                "target_accumulation_and_[0,1]_projection_remain_controller_cointerventions",
                "three_epochs_are_a_fixed_horizon_not_a_convergence_claim",
            ],
        }
        summary_path = store.run_dir / "artifacts/scientific_summary.json"
        atomic_write_json(summary_path, summary)
        generated_artifacts.append({"path": str(summary_path), "kind": "scientific_summary"})
        terminal = {
            "evidence_tier": EVIDENCE_TIER,
            "p0_test_accuracy": float(p0_test["student_accuracy"]),
            "full_test_accuracy": float(full_test["student_accuracy"]),
            "primary_sparse_success_arms": summary["comparison"]["primary_success_arms"],
            "test_accuracy_by_arm": {
                arm_id: float(terminal_tests[arm_id]["student_accuracy"])
                for arm_id in ARM_IDS
            },
        }
        store.append_metric({"mode": "train_terminal", **terminal})
        store.complete(metrics=terminal, artifacts=_artifact_records(store, generated_artifacts))
        print(f"run_dir={store.run_dir}", flush=True)
        gc.collect()
        torch.cuda.empty_cache()
        return 0
    except BaseException as error:
        store.fail(error)
        raise


__all__ = [
    "_assert_full_historical_epoch_parity",
    "_quad_score_batch",
    "_rank_partial_masks",
    "_strict_protocol",
    "run_train",
]
