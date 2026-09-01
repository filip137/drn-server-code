"""CUDA runtime for matched exact-P0 open- and closed-loop Adam recovery."""

from __future__ import annotations

from dataclasses import asdict
import gc
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence, TYPE_CHECKING

import torch

from experiments.artifacts import RunStore, atomic_write_json, sha256_file
from experiments.mnist_relu_drn.components import build_student_stack
from experiments.mnist_relu_drn.ibm_om_baseline_selection_runtime import (
    _artifact_records,
    _evaluate_detailed,
    _input,
)
from experiments.mnist_relu_drn.ibm_om_bounded_codebook_scheme_screen import (
    _tensor_sha256,
)
from experiments.mnist_relu_drn.ibm_om_corrupt_multi_source_hwa_tuned_recovery_runtime import (
    _assert_generator_state_record,
    _assert_p0_validation_parity,
    _generator_state_record,
    _output_kl_gradients,
    _recovery_selection_key,
)
from experiments.mnist_relu_drn.ibm_om_corrupt_source_hwa_cross_array_pulse_adam_runtime import (
    _assert_corrupt_immobile,
)
from experiments.mnist_relu_drn.ibm_om_exact_p0_open_vs_closed_loop_adam import (
    IncrementalOnePulseProgramVerifyAdam,
)
from experiments.mnist_relu_drn.ibm_om_winsorized_cross_array_open_loop_adam import (
    CLAIM_LABEL as OPEN_LOOP_CLAIM_LABEL,
    ColumnSerialOpenLoopAdam,
    open_loop_pulse_port,
)
from experiments.mnist_relu_drn.ibm_om_winsorized_cross_array_open_loop_adam_runtime import (
    DEPLOYMENT_SCHEMA,
    DEPLOYMENT_SCHEMA_VERSION,
    _restore_deployment_plant,
)
from experiments.mnist_relu_drn.ibm_om_winsorized_onchip_adam import (
    sync_catalog_from_persistent,
)
from experiments.mnist_relu_drn.runtime import _load_teacher
from experiments.mnist_shared import build_mnist_loaders, limited
from experiments.schema import to_plain_data
from training.checkpoint import atomic_torch_save
from training.ibm_reram_hwa import load_om_array_population
from training.ibm_reram_raw_active_program_verify import (
    array_population_as_pulse_population,
    matched_trajectory_seeds,
)


if TYPE_CHECKING:
    from ebl.cli import TrainRequest


_ROOT = Path(__file__).resolve().parents[2]
SUMMARY_SCHEMA = "ebl.mnist_relu_drn.ibm_om_exact_p0_open_vs_closed_loop_adam"
SUMMARY_SCHEMA_VERSION = 1
EVIDENCE_TIER = "exploratory_noncanonical"
ARM_IDS = ("open_loop_adam", "incremental_one_pulse_closed_loop_target_tracking")


def _counts_by_layer(
    values: torch.Tensor, binding_shapes: Sequence[Sequence[int]]
) -> list[int]:
    result = []
    offset = 0
    cpu = values.detach().cpu()
    for shape in binding_shapes:
        size = math.prod(tuple(shape))
        result.append(int(cpu[offset : offset + size].sum().item()))
        offset += size
    return result


def _maximum_by_layer(
    values: torch.Tensor, binding_shapes: Sequence[Sequence[int]]
) -> list[int]:
    result = []
    offset = 0
    cpu = values.detach().cpu()
    for shape in binding_shapes:
        size = math.prod(tuple(shape))
        result.append(int(cpu[offset : offset + size].max().item()))
        offset += size
    return result


def _load_device_model(path: Path, protocol: Any) -> tuple[Any, Path, Mapping[str, Any]]:
    source = path.expanduser().resolve()
    if not source.is_file() or sha256_file(source) != protocol.device_model.receipt_sha256:
        raise ValueError("Exact target-94004 paired device-model receipt SHA-256 mismatch.")
    try:
        receipt = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("Expected a readable paired device-model receipt.") from error
    winsorized = receipt.get("winsorized")
    published = winsorized.get("published") if isinstance(winsorized, Mapping) else None
    pair = receipt.get("pair")
    if (
        receipt.get("schema")
        != "ebl.mnist_relu_drn.ibm_om_paired_winsorized_assignment_receipt"
        or receipt.get("schema_version") != 1
        or receipt.get("assignment_seed") != protocol.device_model.assignment_seed
        or receipt.get("role") != "target_94004"
        or not isinstance(pair, Mapping)
        or pair.get("assignment_seed") != protocol.device_model.assignment_seed
        or pair.get("published_winsorized_fingerprint")
        != protocol.device_model.published_population_fingerprint
        or pair.get("full_conductance_formula")
        != protocol.device_model.conductance_formula
        or not isinstance(published, Mapping)
        or published.get("sha256")
        != protocol.device_model.published_population_sha256
        or published.get("population_fingerprint")
        != protocol.device_model.published_population_fingerprint
    ):
        raise ValueError("Paired target device-model semantic contract mismatch.")
    population_path = Path(str(published.get("path", ""))).expanduser().resolve()
    if (
        not population_path.is_file()
        or sha256_file(population_path)
        != protocol.device_model.published_population_sha256
    ):
        raise ValueError("Published winsorized target population SHA-256 mismatch.")
    population = load_om_array_population(population_path)
    if (
        population.assignment_seed != protocol.device_model.assignment_seed
        or population.corruption_policy != protocol.device_model.corruption_policy
        or population.fingerprint
        != protocol.device_model.published_population_fingerprint
        or population.binding_keys
        != ("base.dense_weight.0", "base.dense_weight.1")
        or population.binding_shapes != ((1568, 100), (100, 20))
        or int(population.corrupt.sum().item()) != 21_345
        or bool(torch.any(population.min_bound < -1.0))
        or bool(torch.any(population.max_bound > 1.0))
    ):
        raise ValueError("Loaded target population differs from the frozen identity.")
    return population, population_path, receipt


def _load_exact_p0(path: Path, *, protocol: Any, population: Any) -> Mapping[str, Any]:
    source = path.expanduser().resolve()
    if not source.is_file() or sha256_file(source) != protocol.p0.sha256:
        raise ValueError("Exact target-94004/94301 P0 SHA-256 mismatch.")
    try:
        payload = torch.load(source, map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        raise ValueError("Expected a readable weights-only exact P0.") from error
    expected_fields = {
        "schema",
        "schema_version",
        "evidence_tier",
        "assignment_seed",
        "endpoint_seed",
        "target_kind",
        "target_clipping",
        "maximum_program_pulses",
        "maximum_random_draws",
        "binding_keys",
        "binding_shapes",
        "target_raw_x",
        "programming",
        "continuation_state",
        "validation",
        "validation_prediction_sha256",
        "unopened_split",
        "persistent_endpoint_applied_to_drn",
        "apparent_endpoint_applied_to_drn",
        "inference_read_noise",
        "p0_role",
        "source_hwa_checkpoint_sha256",
        "no_remap_after_p0",
        "recovery_execution_state",
    }
    if not isinstance(payload, Mapping) or set(payload) != expected_fields:
        raise ValueError("Exact P0 fields differ from the frozen schema.")
    state = payload.get("continuation_state")
    execution = payload.get("recovery_execution_state")
    validation = payload.get("validation")
    if (
        payload.get("schema") != DEPLOYMENT_SCHEMA
        or payload.get("schema_version") != DEPLOYMENT_SCHEMA_VERSION
        or payload.get("evidence_tier") != EVIDENCE_TIER
        or payload.get("assignment_seed") != protocol.p0.assignment_seed
        or payload.get("endpoint_seed") != protocol.p0.endpoint_seed
        or payload.get("target_kind")
        != "target_94004_winner_repaired_template_to_published_cells"
        or payload.get("target_clipping") is not False
        or payload.get("maximum_program_pulses") != 128
        or payload.get("maximum_random_draws") != protocol.p0.maximum_random_draws
        or payload.get("binding_keys") != list(population.binding_keys)
        or payload.get("binding_shapes")
        != [list(shape) for shape in population.binding_shapes]
        or payload.get("unopened_split") != "test"
        or payload.get("persistent_endpoint_applied_to_drn") is not True
        or payload.get("apparent_endpoint_applied_to_drn") is not False
        or payload.get("inference_read_noise") is not False
        or payload.get("p0_role") != "selected_joint_hwa_target_94004_seed_94301"
        or payload.get("source_hwa_checkpoint_sha256")
        != protocol.p0.source_hwa_checkpoint_sha256
        or payload.get("no_remap_after_p0") is not True
        or not isinstance(state, Mapping)
        or state.get("schema_version") != 2
        or state.get("state_coordinate") != protocol.p0.state_coordinate
        or state.get("maximum_random_draws") != protocol.p0.maximum_random_draws
        or not isinstance(execution, Mapping)
        or execution.get("seed") != protocol.p0.recovery_data_order_seed
        or not isinstance(validation, Mapping)
        or validation.get("student_correct")
        != protocol.historical_open_loop_parity.p0_validation_correct
        or payload.get("validation_prediction_sha256")
        != protocol.historical_open_loop_parity.p0_validation_prediction_sha256
    ):
        raise ValueError("Exact P0 semantic contract mismatch.")
    size = population.size
    required_float = ("persistent", "apparent")
    if any(
        not isinstance(state.get(name), torch.Tensor)
        or state[name].dtype != torch.float32
        or state[name].shape != (size,)
        or not bool(torch.all(torch.isfinite(state[name])))
        for name in required_float
    ):
        raise ValueError("Exact P0 physical state vectors are invalid.")
    draws = state.get("draw_indices")
    if (
        not isinstance(draws, torch.Tensor)
        or draws.dtype != torch.int64
        or draws.shape != (size,)
        or int(draws.max().item()) > 257
        or tuple(map(int, state.get("seeds", ())))
        != matched_trajectory_seeds(population, endpoint_seed=protocol.p0.endpoint_seed)
        or not isinstance(state.get("construction_seeds"), torch.Tensor)
        or not torch.equal(
            state["construction_seeds"],
            array_population_as_pulse_population(population).construction_seeds,
        )
    ):
        raise ValueError("Exact P0 identity or RNG continuation is invalid.")
    programming = payload.get("programming")
    total_pulses = programming.get("total_pulses") if isinstance(programming, Mapping) else None
    if (
        not isinstance(total_pulses, torch.Tensor)
        or total_pulses.dtype != torch.int64
        or total_pulses.shape != (size,)
        or not torch.equal(draws, 1 + 2 * total_pulses)
        or bool(torch.any(state["persistent"] + 1.0 < 0.0))
        or bool(
            torch.any(
                state["persistent"][population.corrupt]
                != population.min_bound[population.corrupt]
            )
        )
    ):
        raise ValueError("Exact P0 pulse accounting or physical invariants changed.")
    generator_state = _assert_generator_state_record(execution)
    if (
        execution.get("sha256")
        != protocol.historical_open_loop_parity.initial_train_generator_state_sha256
    ):
        raise ValueError("Exact P0 recovery loader-state hash changed.")
    result = dict(payload)
    result["artifact_path"] = str(source)
    result["recovery_generator_state"] = generator_state
    return result


def _strict_protocol(protocol: Any) -> None:
    recovery = protocol.recovery
    if (
        protocol.evidence_tier != EVIDENCE_TIER
        or protocol.p0.assignment_seed != 94004
        or protocol.p0.endpoint_seed != 94301
        or protocol.p0.recovery_data_order_seed != 94402
        or tuple(arm.arm_id for arm in recovery.arms) != ARM_IDS
        or recovery.objective != "teacher_output_kl_only"
        or recovery.optimizer != "digital_Adam_shared_equations"
        or not math.isclose(recovery.learning_rate_raw_x, 3e-5, abs_tol=0.0)
        or recovery.beta1 != 0.9
        or recovery.beta2 != 0.999
        or recovery.epsilon != 1e-8
        or recovery.epochs != 3
        or recovery.maximum_batches is not None
        or recovery.updated_layer_indices != (0, 1)
        or recovery.pulse_cap != 64
        or not math.isclose(recovery.nominal_delta_x, 0.04745, abs_tol=0.0)
        or recovery.maximum_random_draws != 385
        or recovery.data_order_seed != 94402
        or not math.isclose(recovery.fixed_logit_gain, 14.12537544622754, abs_tol=0.0)
        or recovery.arms[0].pulse_selection_seed != 94401
        or recovery.arms[0].maximum_pulses_per_cell_per_minibatch != 1
        or recovery.arms[1].pulse_selection_seed is not None
        or recovery.arms[1].maximum_pulses_per_cell_per_minibatch != 1
        or not math.isclose(
            float(recovery.arms[1].verify_tolerance_raw_x), 0.04745 / 2.0, abs_tol=0.0
        )
    ):
        raise ValueError("Resolved exact-P0 comparator protocol drifted.")


def _historical_open_loop_epoch_gate(
    report: Mapping[str, Any], *, protocol: Any
) -> Mapping[str, Any]:
    epoch = int(report["epoch"])
    index = epoch - 1
    historical = protocol.historical_open_loop_parity
    validation = report["validation"]
    expected_kl = historical.validation_kl_by_epoch[index]
    if (
        int(validation["student_correct"])
        != historical.validation_correct_by_epoch[index]
        or report["validation_prediction_sha256"]
        != historical.validation_prediction_sha256_by_epoch[index]
        or int(report["issued_commands_cumulative"])
        != historical.cumulative_issued_commands_by_epoch[index]
        or int(report["effective_state_change_events_cumulative"])
        != historical.cumulative_effective_state_change_events_by_epoch[index]
        or report["train_generator_state_sha256"]
        != historical.train_generator_state_sha256_by_epoch[index]
        or not math.isclose(
            float(validation["kl_teacher_student"]),
            float(expected_kl),
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    ):
        raise RuntimeError(f"Open-loop exact-P0 historical parity failed at epoch {epoch}.")
    return {
        "validation_correct_exact": True,
        "validation_prediction_sha256_exact": True,
        "validation_kl_absolute_tolerance": 1e-12,
        "cumulative_issued_commands_exact": True,
        "cumulative_effective_state_change_events_exact": True,
        "train_generator_state_sha256_exact": True,
    }


def _build_optimizer(
    arm_id: str,
    *,
    plant: Any,
    p0: Mapping[str, Any],
    population: Any,
    protocol: Any,
    device: torch.device,
) -> Any:
    recovery = protocol.recovery
    if arm_id == "open_loop_adam":
        return ColumnSerialOpenLoopAdam(
            open_loop_pulse_port(plant),
            device=device,
            binding_shapes=population.binding_shapes,
            learning_rate_raw_x=recovery.learning_rate_raw_x,
            nominal_delta_x=recovery.nominal_delta_x,
            pulse_cap=recovery.pulse_cap,
            pulse_selection_seed=recovery.arms[0].pulse_selection_seed,
            beta1=recovery.beta1,
            beta2=recovery.beta2,
            epsilon=recovery.epsilon,
        )
    if arm_id == "incremental_one_pulse_closed_loop_target_tracking":
        return IncrementalOnePulseProgramVerifyAdam(
            plant.controller_port(),
            initial_apparent_raw_a=p0["continuation_state"]["apparent"],
            device=device,
            binding_shapes=population.binding_shapes,
            learning_rate_raw_x=recovery.learning_rate_raw_x,
            nominal_delta_x=recovery.nominal_delta_x,
            verify_tolerance_raw_x=recovery.arms[1].verify_tolerance_raw_x,
            pulse_cap=recovery.pulse_cap,
            maximum_pulses_per_cell_per_minibatch=(
                recovery.arms[1].maximum_pulses_per_cell_per_minibatch
            ),
            beta1=recovery.beta1,
            beta2=recovery.beta2,
            epsilon=recovery.epsilon,
        )
    raise ValueError(f"Unknown recovery arm {arm_id!r}.")


def _restore_selected_validation(
    payload: Mapping[str, Any],
    *,
    population: Any,
    pulse_population: Any,
    stack: Any,
    teacher: Any,
    validation_loader: Iterable,
    maximum_random_draws: int,
) -> tuple[Mapping[str, Any], Any]:
    """Restore one selected plant and fail closed on saved validation parity."""

    expected = payload.get("validation")
    expected_prediction_sha256 = payload.get("validation_prediction_sha256")
    state = payload.get("plant_continuation_state")
    if not isinstance(expected, Mapping) or not isinstance(state, Mapping):
        raise RuntimeError("Selected checkpoint lacks validation or plant state.")
    plant = _restore_deployment_plant(
        {"continuation_state": state},
        pulse_population=pulse_population,
        maximum_random_draws=maximum_random_draws,
        device=stack.device,
    )
    if bool(torch.any(plant.persistent + 1.0 < 0.0)):
        raise RuntimeError("Selected checkpoint restored a negative conductance.")
    sync_catalog_from_persistent(
        stack.bundle.catalog, plant, binding_shapes=population.binding_shapes
    )
    observed, prediction = _evaluate_detailed(
        stack, teacher, validation_loader, sample_limit=None
    )
    observed_prediction_sha256 = _tensor_sha256(prediction.to(torch.int64))
    if (
        int(observed.get("student_correct", -1))
        != int(expected.get("student_correct", -2))
        or observed_prediction_sha256 != expected_prediction_sha256
        or not math.isclose(
            float(observed.get("kl_teacher_student", float("nan"))),
            float(expected.get("kl_teacher_student", float("inf"))),
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    ):
        raise RuntimeError(
            "Selected checkpoint restore changed its validation evaluation."
        )
    return (
        {
            "restored_before_any_test_access": True,
            "student_correct_exact": True,
            "prediction_sha256_exact": True,
            "kl_teacher_student_absolute_tolerance": 1e-12,
        },
        plant,
    )


def _run_arm(
    arm_id: str,
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
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    recovery = protocol.recovery
    loaders.train_generator.set_state(p0["recovery_generator_state"].clone())
    plant = _restore_deployment_plant(
        p0,
        pulse_population=pulse_population,
        maximum_random_draws=recovery.maximum_random_draws,
        device=stack.device,
    )
    p0_persistent = plant.persistent.detach().cpu().clone()
    p0_draw_indices = plant.state_dict()["draw_indices"].detach().cpu().clone()
    sync_catalog_from_persistent(
        stack.bundle.catalog, plant, binding_shapes=population.binding_shapes
    )
    initial_validation, initial_prediction = _evaluate_detailed(
        stack, teacher, loaders.validation, sample_limit=None
    )
    initial_prediction_sha256 = _tensor_sha256(initial_prediction.to(torch.int64))
    p0_parity = _assert_p0_validation_parity(
        p0, initial_validation, initial_prediction_sha256
    )
    historical = protocol.historical_open_loop_parity
    if (
        int(initial_validation["student_correct"]) != historical.p0_validation_correct
        or initial_prediction_sha256 != historical.p0_validation_prediction_sha256
        or not math.isclose(
            float(initial_validation["kl_teacher_student"]),
            historical.p0_validation_kl,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    ):
        raise RuntimeError(f"{arm_id} did not restore the frozen P0 validation state.")
    optimizer = _build_optimizer(
        arm_id,
        plant=plant,
        p0=p0,
        population=population,
        protocol=protocol,
        device=stack.device,
    )
    corrupt = population.corrupt.to(stack.device)
    reports: list[Mapping[str, Any]] = []
    checkpoints: dict[int, Path] = {}
    cumulative_effective = 0
    cumulative_corrupt_issued = 0
    cumulative_upward = 0
    cumulative_downward = 0
    cumulative_reversals = 0
    prior_issued = 0
    for epoch in range(1, recovery.epochs + 1):
        examples = 0
        batches = 0
        train_kl = 0.0
        train_correct = 0
        issued_epoch = 0
        effective_epoch = 0
        corrupt_issued_epoch = 0
        upward_epoch = 0
        downward_epoch = 0
        reversals_epoch = 0
        api_verify_calls_epoch = 0
        conceptual_observations_epoch = 0
        per_call_ceiling_epoch = 0
        cap_blocked_epoch = 0
        target_projection_cells_epoch = 0
        target_projection_signed_epoch = 0.0
        target_projection_absolute_epoch = 0.0
        target_projection_by_layer_epoch = [
            {
                "cell_events": 0,
                "signed_sum": 0.0,
                "absolute_sum": 0.0,
                "maximum_absolute": 0.0,
            }
            for _ in population.binding_shapes
        ]
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
            before_state = plant.persistent.detach().clone()
            before_count = optimizer.pulse_count.detach().clone()
            step = optimizer.step(gradients)
            pulse_delta = optimizer.pulse_count - before_count
            issued = int(pulse_delta.sum().item())
            expected_issued = (
                int(step.applied_cell_pulses)
                if arm_id == "open_loop_adam"
                else int(step.issued_cell_pulses)
            )
            if issued != expected_issued:
                raise RuntimeError("Recovery issued-command accounting diverged.")
            moved = plant.persistent != before_state
            if bool(torch.any(moved & corrupt)):
                raise RuntimeError("An immutable published-corrupt cell moved.")
            if bool(torch.any(plant.persistent + 1.0 < 0.0)):
                raise RuntimeError("Persistent G=a+1 became negative.")
            issued_epoch += issued
            corrupt_issued_epoch += int(pulse_delta[corrupt].sum().item())
            effective_epoch += int(moved.sum().item())
            upward_epoch += int(step.upward_pulses)
            downward_epoch += int(step.downward_pulses)
            if arm_id != "open_loop_adam":
                api_verify_calls_epoch += int(step.api_full_port_verify_calls)
                conceptual_observations_epoch += int(
                    step.conceptual_post_pulse_cell_observations
                )
                per_call_ceiling_epoch += int(
                    step.per_call_one_pulse_budget_reached_cells
                )
                cap_blocked_epoch += int(step.cap_blocked_cells)
                reversals_epoch += int(step.reversals)
                target_projection_cells_epoch += step.target_projection.cells
                target_projection_signed_epoch += step.target_projection.signed_sum
                target_projection_absolute_epoch += step.target_projection.absolute_sum
                for cumulative, layer in zip(
                    target_projection_by_layer_epoch,
                    step.target_projection_by_layer,
                    strict=True,
                ):
                    cumulative["cell_events"] += layer.cells
                    cumulative["signed_sum"] += layer.signed_sum
                    cumulative["absolute_sum"] += layer.absolute_sum
                    cumulative["maximum_absolute"] = max(
                        cumulative["maximum_absolute"], layer.maximum_absolute
                    )
            examples += int(labels.numel())
            batches = batch_index + 1
        if batches < 1:
            raise RuntimeError("Expected recovery to process at least one minibatch.")
        cumulative_effective += effective_epoch
        cumulative_corrupt_issued += corrupt_issued_epoch
        cumulative_upward += upward_epoch
        cumulative_downward += downward_epoch
        cumulative_reversals += reversals_epoch
        sync_catalog_from_persistent(
            stack.bundle.catalog, plant, binding_shapes=population.binding_shapes
        )
        validation, validation_prediction = _evaluate_detailed(
            stack, teacher, loaders.validation, sample_limit=None
        )
        validation_prediction_sha256 = _tensor_sha256(
            validation_prediction.to(torch.int64)
        )
        optimizer_state = optimizer.state_dict()
        issued_cumulative = int(optimizer.pulse_count.sum().item())
        if issued_cumulative - prior_issued != issued_epoch:
            raise RuntimeError("Cumulative issued commands do not match the epoch.")
        if cumulative_upward + cumulative_downward != issued_cumulative:
            raise RuntimeError("SET/RESET direction accounting does not sum to pulses.")
        if arm_id != "open_loop_adam" and (
            int(optimizer_state["total_upward_pulses"]) != cumulative_upward
            or int(optimizer_state["total_downward_pulses"]) != cumulative_downward
            or int(optimizer_state["total_reversals"]) != cumulative_reversals
            or int(
                optimizer_state[
                    "total_conceptual_post_pulse_cell_observations"
                ]
            )
            != issued_cumulative
        ):
            raise RuntimeError("Closed-loop controller accounting diverged.")
        prior_issued = issued_cumulative
        plant_state = plant.state_dict()
        draw_indices = plant_state["draw_indices"].detach().cpu()
        draw_gate = torch.equal(
            draw_indices,
            p0_draw_indices + 2 * optimizer.pulse_count.detach().cpu(),
        )
        if not draw_gate or int(draw_indices.max().item()) > recovery.maximum_random_draws:
            raise RuntimeError("Plant RNG draw continuation differs from two draws/pulse.")
        generator_record = _generator_state_record(
            loaders.train_generator.get_state()
        )
        checkpoint_payload = {
            "schema": "ebl.mnist_relu_drn.ibm_om_exact_p0_recovery_arm_state",
            "schema_version": 1,
            "evidence_tier": EVIDENCE_TIER,
            "arm_id": arm_id,
            "epoch": epoch,
            "source_p0_sha256": protocol.p0.sha256,
            "optimizer_state": optimizer_state,
            "plant_continuation_state": plant_state,
            "recovery_train_generator_state": generator_record,
            "validation": dict(validation),
            "validation_prediction_sha256": validation_prediction_sha256,
            "test": None,
            "output_kl_only": True,
            "updated_physical_layers": [0, 1],
            "issued_commands_cumulative": issued_cumulative,
            "effective_state_change_events_cumulative": cumulative_effective,
            "issued_commands_targeting_corrupt_cells_cumulative": (
                cumulative_corrupt_issued
            ),
            "upward_pulses_cumulative": cumulative_upward,
            "downward_pulses_cumulative": cumulative_downward,
            "closed_loop_direction_reversals_cumulative": (
                cumulative_reversals if arm_id != "open_loop_adam" else None
            ),
            "plant_draw_delta_equals_two_per_issued_command": True,
            "persistent_full_conductance_formula": "G=a+1=2*x",
        }
        checkpoint = atomic_torch_save(
            checkpoint_payload,
            store.run_dir / f"checkpoints/{arm_id}_epoch_{epoch}.pt",
        )
        checkpoints[epoch] = checkpoint
        generated_artifacts.append(
            {"path": str(checkpoint), "kind": f"{arm_id}_epoch_checkpoint"}
        )
        report = {
            "arm_id": arm_id,
            "epoch": epoch,
            "train": {
                "examples": examples,
                "batches": batches,
                "kl_teacher_student": train_kl / examples,
                "student_accuracy": train_correct / examples,
            },
            "validation": dict(validation),
            "validation_prediction_sha256": validation_prediction_sha256,
            "test": None,
            "issued_commands_epoch": issued_epoch,
            "issued_commands_cumulative": issued_cumulative,
            "issued_commands_by_layer_cumulative": _counts_by_layer(
                optimizer.pulse_count, population.binding_shapes
            ),
            "maximum_commands_per_cell_by_layer": _maximum_by_layer(
                optimizer.pulse_count, population.binding_shapes
            ),
            "issued_commands_targeting_corrupt_cells_epoch": corrupt_issued_epoch,
            "upward_pulses_epoch": upward_epoch,
            "downward_pulses_epoch": downward_epoch,
            "closed_loop_direction_reversals_epoch": (
                reversals_epoch if arm_id != "open_loop_adam" else None
            ),
            "effective_state_change_events_epoch": effective_epoch,
            "effective_state_change_events_cumulative": cumulative_effective,
            "api_full_port_verify_calls_epoch": api_verify_calls_epoch,
            "conceptual_post_pulse_cell_observations_epoch": (
                conceptual_observations_epoch
            ),
            "per_call_one_pulse_ceiling_reached_events_epoch": (
                per_call_ceiling_epoch
            ),
            "cap_blocked_target_debt_events_epoch": cap_blocked_epoch,
            "update_target_projection": {
                "cells_epoch": target_projection_cells_epoch,
                "lost_signed_displacement_epoch": target_projection_signed_epoch,
                "lost_absolute_displacement_epoch": target_projection_absolute_epoch,
                "by_layer_epoch": target_projection_by_layer_epoch,
            },
            "train_generator_state_sha256": generator_record["sha256"],
            "plant_draw_delta_equals_two_per_issued_command": True,
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": sha256_file(checkpoint),
        }
        if arm_id == "open_loop_adam":
            report["historical_open_loop_parity"] = _historical_open_loop_epoch_gate(
                report, protocol=protocol
            )
        reports.append(report)
        store.append_metric({"mode": "exact_p0_recovery", **report})
        print(
            f"recovery arm={arm_id} epoch={epoch}/{recovery.epochs} "
            f"val={100.0*float(validation['student_accuracy']):.2f}% "
            f"issued={issued_epoch}",
            flush=True,
        )
    selected = max(reports, key=_recovery_selection_key)
    selected_epoch = int(selected["epoch"])
    if arm_id == "open_loop_adam" and (
        selected_epoch != protocol.historical_open_loop_parity.selected_epoch
    ):
        raise RuntimeError("Open-loop selected epoch differs from exact prior evidence.")
    selected_path = checkpoints[selected_epoch]
    selected_payload = torch.load(selected_path, map_location="cpu", weights_only=True)
    if (
        selected_payload.get("arm_id") != arm_id
        or selected_payload.get("epoch") != selected_epoch
        or selected_payload.get("test") is not None
        or not isinstance(selected_payload.get("optimizer_state"), Mapping)
    ):
        raise RuntimeError("Selected recovery checkpoint roundtrip changed.")
    selected_validation_restore_parity, selected_plant = (
        _restore_selected_validation(
            selected_payload,
            population=population,
            pulse_population=pulse_population,
            stack=stack,
            teacher=teacher,
            validation_loader=loaders.validation,
            maximum_random_draws=recovery.maximum_random_draws,
        )
    )
    selected_optimizer_state = selected_payload["optimizer_state"]
    if arm_id == "incremental_one_pulse_closed_loop_target_tracking":
        replay_optimizer = _build_optimizer(
            arm_id,
            plant=selected_plant,
            p0=p0,
            population=population,
            protocol=protocol,
            device=stack.device,
        )
        replay_optimizer.load_state_dict(selected_optimizer_state)
        replay_state = replay_optimizer.state_dict()
        for name in (
            "first_moment",
            "second_moment",
            "pulse_count",
            "desired_raw_x",
            "cached_apparent_raw_x",
            "previous_pulse_direction",
            "ever_update_target_projected",
            "ever_cap_blocked",
        ):
            if not torch.equal(replay_state[name], selected_optimizer_state[name]):
                raise RuntimeError("Closed-loop optimizer checkpoint replay changed.")
        optimizer_state_contract_validated = True
        optimizer_state_replay_roundtrip = True
        del replay_optimizer
    else:
        if (
            selected_optimizer_state.get("schema")
            != "ebl.mnist_relu_drn.ibm_om_cross_array_open_loop_adam_state"
            or selected_optimizer_state.get("schema_version") != 1
            or selected_optimizer_state.get("binding_shapes")
            != [list(shape) for shape in population.binding_shapes]
            or selected_optimizer_state.get("pulse_cap") != recovery.pulse_cap
            or selected_optimizer_state.get("learning_rate_raw_x")
            != recovery.learning_rate_raw_x
            or selected_optimizer_state.get("nominal_delta_x")
            != recovery.nominal_delta_x
            or selected_optimizer_state.get("beta1") != recovery.beta1
            or selected_optimizer_state.get("beta2") != recovery.beta2
            or selected_optimizer_state.get("epsilon") != recovery.epsilon
            or selected_optimizer_state.get("step")
            != sum(
                int(row["train"]["batches"])
                for row in reports
                if int(row["epoch"]) <= selected_epoch
            )
            or not isinstance(
                selected_optimizer_state.get("pulse_selection_rng_state"),
                torch.Tensor,
            )
        ):
            raise RuntimeError("Open-loop optimizer checkpoint state is invalid.")
        optimizer_state_contract_validated = True
        optimizer_state_replay_roundtrip = None
    corrupt_immobility = _assert_corrupt_immobile(
        p0_persistent,
        selected_payload["plant_continuation_state"]["persistent"],
        population.corrupt,
        context=f"selected_{arm_id}",
    )
    initial_projection = None
    selected_controller_diagnostic = None
    if arm_id != "open_loop_adam":
        terminal_debt = (
            selected_optimizer_state["cached_apparent_raw_x"]
            - selected_optimizer_state["desired_raw_x"]
        ).abs() > float(protocol.recovery.arms[1].verify_tolerance_raw_x)
        initial_projection = {
            "all_cells": selected_optimizer_state["initial_target_projection"],
            "by_layer": selected_optimizer_state[
                "initial_target_projection_by_layer"
            ],
            "projection_domain": [0.0, 1.0],
            "per_device_support_projection": False,
        }
        selected_controller_diagnostic = {
            "update_target_projection_cell_events": int(
                selected_optimizer_state["total_target_projection_cell_events"]
            ),
            "unique_cells_with_update_target_projection": int(
                selected_optimizer_state["ever_update_target_projected"].sum().item()
            ),
            "unique_update_projected_cells_by_layer": _counts_by_layer(
                selected_optimizer_state["ever_update_target_projected"],
                population.binding_shapes,
            ),
            "cap_blocked_cell_events": int(
                selected_optimizer_state["total_cap_blocked_cell_events"]
            ),
            "unique_cells_ever_cap_blocked": int(
                selected_optimizer_state["ever_cap_blocked"].sum().item()
            ),
            "unique_cap_blocked_cells_by_layer": _counts_by_layer(
                selected_optimizer_state["ever_cap_blocked"], population.binding_shapes
            ),
            "pulse_cap_semantic": (
                "recovery_only_additional_pulses_after_exact_P0"
            ),
            "api_full_port_verify_calls": int(
                selected_optimizer_state["total_api_full_port_verify_calls"]
            ),
            "conceptual_post_pulse_cell_observations": int(
                selected_optimizer_state[
                    "total_conceptual_post_pulse_cell_observations"
                ]
            ),
            "upward_pulses": int(selected_optimizer_state["total_upward_pulses"]),
            "downward_pulses": int(
                selected_optimizer_state["total_downward_pulses"]
            ),
            "direction_reversals": int(selected_optimizer_state["total_reversals"]),
            "target_projection_cell_events_and_magnitudes_by_layer": (
                selected_optimizer_state["total_target_projection_by_layer"]
            ),
            "terminal_target_debt_cells": int(terminal_debt.sum().item()),
            "terminal_target_debt_cells_by_layer": _counts_by_layer(
                terminal_debt, population.binding_shapes
            ),
        }
    report = {
        "arm_id": arm_id,
        "claim_label": (
            OPEN_LOOP_CLAIM_LABEL
            if arm_id == "open_loop_adam"
            else "projected digital-target closed-loop Adam"
        ),
        "method": (
            "column_serial_open_loop_stochastic_coincidence_emulator"
            if arm_id == "open_loop_adam"
            else "incremental_one_pulse_closed_loop_target_tracking"
        ),
        "initial_validation": dict(initial_validation),
        "initial_validation_prediction_sha256": initial_prediction_sha256,
        "p0_validation_parity": p0_parity,
        "epoch_reports": reports,
        "selected_epoch": selected_epoch,
        "selected_validation": dict(selected["validation"]),
        "selected_checkpoint": str(selected_path),
        "selected_checkpoint_sha256": sha256_file(selected_path),
        "selected_test": None,
        "initial_desired_target_projection": initial_projection,
        "selected_controller_diagnostic": selected_controller_diagnostic,
        "corrupt_immobility": corrupt_immobility,
        "optimizer_state_contract_validated": optimizer_state_contract_validated,
        "optimizer_state_replay_roundtrip": optimizer_state_replay_roundtrip,
        "optimizer_state_replay_limitation": (
            "open_loop_optimizer_has_no_load_API_and_is_not_resumed_for_selection"
            if arm_id == "open_loop_adam"
            else None
        ),
        "selected_checkpoint_validation_restore_parity": (
            selected_validation_restore_parity
        ),
        "test_opened": False,
    }
    return report, selected_payload


def _evaluate_payload_test(
    payload: Mapping[str, Any],
    *,
    population: Any,
    pulse_population: Any,
    stack: Any,
    teacher: Any,
    test_loader: Iterable,
    maximum_random_draws: int,
) -> Mapping[str, Any]:
    source = (
        {"continuation_state": payload["plant_continuation_state"]}
        if "plant_continuation_state" in payload
        else payload
    )
    plant = _restore_deployment_plant(
        source,
        pulse_population=pulse_population,
        maximum_random_draws=maximum_random_draws,
        device=stack.device,
    )
    if bool(torch.any(plant.persistent + 1.0 < 0.0)):
        raise RuntimeError("Terminal persistent G=a+1 is negative.")
    sync_catalog_from_persistent(
        stack.bundle.catalog, plant, binding_shapes=population.binding_shapes
    )
    metrics, prediction = _evaluate_detailed(
        stack, teacher, test_loader, sample_limit=None
    )
    return {
        **dict(metrics),
        "prediction_sha256": _tensor_sha256(prediction.to(torch.int64)),
    }


def run_train(request: "TrainRequest") -> int:
    """Rerun matched open/closed recovery arms from the exact saved P0."""

    from experiments.mnist_relu_drn.ibm_om_exact_p0_open_vs_closed_loop_adam_config import (
        ExactP0OpenVsClosedLoopAdamTrainSpec,
    )

    spec = request.spec
    if not isinstance(spec, ExactP0OpenVsClosedLoopAdamTrainSpec):
        raise TypeError("Expected the dedicated exact-P0 comparator train spec.")
    if request.teacher_weights is None or request.weights is None or request.device_model is None:
        raise ValueError(
            "Expected explicit --teacher-weights, --weights exact P0, and --device-model."
        )
    for name in ("base_weights", "resume", "device_data"):
        if getattr(request, name) is not None:
            raise ValueError(f"Exact-P0 comparator does not accept {name}.")
    protocol = spec.protocol
    _strict_protocol(protocol)
    teacher_path = request.teacher_weights.expanduser().resolve()
    p0_path = request.weights.expanduser().resolve()
    device_model_path = request.device_model.expanduser().resolve()
    if (
        not teacher_path.is_file()
        or sha256_file(teacher_path) != protocol.expected_teacher_weights_sha256
    ):
        raise ValueError("Frozen teacher SHA-256 mismatch.")
    population, population_path, device_receipt = _load_device_model(
        device_model_path, protocol
    )
    p0 = _load_exact_p0(p0_path, protocol=protocol, population=population)
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
            raise RuntimeError("Exact-P0 recovery comparator requires CUDA.")
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
        stack.cost.gain = protocol.recovery.fixed_logit_gain
        generated_artifacts: list[Mapping[str, Any]] = []
        arm_reports: dict[str, Mapping[str, Any]] = {}
        selected_payloads: dict[str, Mapping[str, Any]] = {}
        for arm_id in ARM_IDS:
            report, selected_payload = _run_arm(
                arm_id,
                p0=p0,
                population=population,
                pulse_population=pulse_population,
                stack=stack,
                teacher=teacher,
                loaders=loaders,
                protocol=protocol,
                store=store,
                generated_artifacts=generated_artifacts,
            )
            arm_reports[arm_id] = report
            selected_payloads[arm_id] = selected_payload

        # Both validation-only selections are frozen before the first test item
        # is requested.  Terminal evaluation replays immutable saved states.
        open_hashes = [
            row["train_generator_state_sha256"]
            for row in arm_reports[ARM_IDS[0]]["epoch_reports"]
        ]
        closed_hashes = [
            row["train_generator_state_sha256"]
            for row in arm_reports[ARM_IDS[1]]["epoch_reports"]
        ]
        if open_hashes != closed_hashes:
            raise RuntimeError("Matched recovery arms consumed different minibatch orders.")
        p0_test = _evaluate_payload_test(
            p0,
            population=population,
            pulse_population=pulse_population,
            stack=stack,
            teacher=teacher,
            test_loader=loaders.test,
            maximum_random_draws=protocol.recovery.maximum_random_draws,
        )
        terminal_tests = {}
        for arm_id in ARM_IDS:
            terminal_tests[arm_id] = _evaluate_payload_test(
                selected_payloads[arm_id],
                population=population,
                pulse_population=pulse_population,
                stack=stack,
                teacher=teacher,
                test_loader=loaders.test,
                maximum_random_draws=protocol.recovery.maximum_random_draws,
            )
            arm_reports[arm_id] = {
                **dict(arm_reports[arm_id]),
                "selected_test": terminal_tests[arm_id],
                "test_opened": True,
            }
        historical = protocol.historical_open_loop_parity
        open_test = terminal_tests[ARM_IDS[0]]
        if (
            int(p0_test["student_correct"]) != historical.p0_test_correct
            or p0_test["prediction_sha256"]
            != historical.p0_test_prediction_sha256
            or int(open_test["student_correct"]) != historical.selected_test_correct
            or open_test["prediction_sha256"]
            != historical.selected_test_prediction_sha256
            or not math.isclose(
                float(open_test["kl_teacher_student"]),
                historical.selected_test_kl,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        ):
            raise RuntimeError("Terminal exact-P0/open-loop historical parity failed.")
        summary = {
            "schema": SUMMARY_SCHEMA,
            "schema_version": SUMMARY_SCHEMA_VERSION,
            "evidence_tier": EVIDENCE_TIER,
            "status": "completed",
            "question": (
                "Does cached apparent-feedback incremental one-pulse target tracking "
                "improve exact-P0 recovery over matched open-loop stochastic writes?"
            ),
            "teacher": {"path": str(teacher_path), "sha256": sha256_file(teacher_path), **teacher_metadata},
            "exact_p0": {
                "path": str(p0_path),
                "sha256": sha256_file(p0_path),
                "assignment_seed": protocol.p0.assignment_seed,
                "endpoint_seed": protocol.p0.endpoint_seed,
                "validation": dict(p0["validation"]),
                "test": p0_test,
                "plant_and_loader_rng_exact_continuation": True,
            },
            "device_model": {
                "receipt_path": str(device_model_path),
                "receipt_sha256": sha256_file(device_model_path),
                "population_path": str(population_path),
                "population_sha256": sha256_file(population_path),
                "population_fingerprint": population.fingerprint,
                "assignment_seed": population.assignment_seed,
                "corrupt_cells": int(population.corrupt.sum().item()),
                "receipt_role": device_receipt["role"],
            },
            "protocol": {
                "objective": protocol.recovery.objective,
                "optimizer": protocol.recovery.optimizer,
                "epochs": protocol.recovery.epochs,
                "learning_rate_raw_x": protocol.recovery.learning_rate_raw_x,
                "nominal_delta_x": protocol.recovery.nominal_delta_x,
                "pulse_cap": protocol.recovery.pulse_cap,
                "data_order_seed": protocol.recovery.data_order_seed,
                "fixed_logit_gain": protocol.recovery.fixed_logit_gain,
                "persistent_forward": "G=a+1=2*x",
                "closed_loop_target_projection": (
                    "global_nonnegative_conductance_domain_x_in_[0,1]; "
                    "not_per_device_support_clipping"
                ),
                "test_policy": protocol.recovery.test_policy,
            },
            "arms": arm_reports,
            "matched_gates": {
                "same_exact_p0_sha256": True,
                "same_initial_validation": True,
                "same_output_kl_and_digital_adam_equations": True,
                "same_three_epoch_minibatch_order_hashes": True,
                "selected_checkpoint_validation_restore_parity_both_arms": True,
                "both_physical_layers_updated": True,
                "persistent_G_nonnegative": True,
                "published_corrupt_cells_immobile": True,
                "test_sealed_until_both_epoch_selections_frozen": True,
                "test_used_for_selection": False,
                "open_loop_historical_parity_exact": True,
                "terminal_p0_and_open_loop_historical_test_parity_exact": True,
            },
            "comparison": {
                "p0_test_accuracy": float(p0_test["student_accuracy"]),
                "open_loop_selected_epoch": int(arm_reports[ARM_IDS[0]]["selected_epoch"]),
                "open_loop_test_accuracy": float(terminal_tests[ARM_IDS[0]]["student_accuracy"]),
                "closed_loop_selected_epoch": int(arm_reports[ARM_IDS[1]]["selected_epoch"]),
                "closed_loop_test_accuracy": float(terminal_tests[ARM_IDS[1]]["student_accuracy"]),
                "closed_minus_open_test_accuracy": float(
                    terminal_tests[ARM_IDS[1]]["student_accuracy"]
                    - terminal_tests[ARM_IDS[0]]["student_accuracy"]
                ),
                "selection_is_within_arm_validation_only": True,
            },
            "limitations": [
                "exploratory_noncanonical_single_target_endpoint",
                "model_based_aihwkit_1.1.0_IBM_OM_fitted_preset",
                "digital_Adam_and_BPTT_not_native_analog_optimizer",
                "incremental_one_pulse_target_tracking_not_full_convergence_program_verify",
                "target_accumulation_and_[0,1]_projection_are_cointerventions_with_feedback",
                "held_post_write_apparent_cache_no_fresh_read_noise_or_retention",
                "vectorized_full_port_API_verify_calls_reported_separately_from_pulsed_cell_observations",
            ],
        }
        summary_path = store.run_dir / "artifacts/scientific_summary.json"
        atomic_write_json(summary_path, summary)
        generated_artifacts.append({"path": str(summary_path), "kind": "scientific_summary"})
        terminal = {
            "evidence_tier": EVIDENCE_TIER,
            "p0_test_accuracy": float(p0_test["student_accuracy"]),
            "open_loop_selected_epoch": int(arm_reports[ARM_IDS[0]]["selected_epoch"]),
            "open_loop_test_accuracy": float(terminal_tests[ARM_IDS[0]]["student_accuracy"]),
            "closed_loop_selected_epoch": int(arm_reports[ARM_IDS[1]]["selected_epoch"]),
            "closed_loop_test_accuracy": float(terminal_tests[ARM_IDS[1]]["student_accuracy"]),
            "closed_minus_open_test_accuracy": summary["comparison"]["closed_minus_open_test_accuracy"],
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
    "_build_optimizer",
    "_historical_open_loop_epoch_gate",
    "_load_device_model",
    "_load_exact_p0",
    "_restore_selected_validation",
    "_run_arm",
    "_strict_protocol",
    "run_train",
]
