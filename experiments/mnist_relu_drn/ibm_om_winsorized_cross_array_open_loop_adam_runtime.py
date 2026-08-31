"""CUDA runtime for fresh-array transfer and open-loop pulse Adam."""

from __future__ import annotations

import gc
import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable, Mapping, Sequence, TYPE_CHECKING

import torch

from experiments.artifacts import RunStore, atomic_write_json, sha256_file
from experiments.mnist_relu_drn.components import apply_targets, build_student_stack
from experiments.mnist_relu_drn.ibm_om_baseline_selection import quad_stack
from experiments.mnist_relu_drn.ibm_om_baseline_selection_runtime import (
    _artifact_records,
    _evaluate_detailed,
    _input,
    _prepare_assignment,
    _sampler_python,
)
from experiments.mnist_relu_drn.ibm_om_baseline_spacing_pv_runtime import (
    _hardware_summary,
    _save_design_mapping,
)
from experiments.mnist_relu_drn.ibm_om_baseline_spacing_pv_truncated_nominal import (
    build_truncated_nominal_baseline_spacing_mapping,
)
from experiments.mnist_relu_drn.ibm_om_baseline_spacing_pv_truncated_nominal_runtime import (
    _native_assignment_from_joint_artifact,
    _save_winsorized_identity_artifacts,
)
from experiments.mnist_relu_drn.ibm_om_bounded_codebook_scheme_screen import (
    _tensor_sha256,
)
from experiments.mnist_relu_drn.ibm_om_winsorized_cross_array_open_loop_adam import (
    CLAIM_LABEL,
    EVIDENCE_TIER,
    UPDATE_SURFACE,
    ColumnSerialOpenLoopAdam,
    build_target_baseline_contrast_transfer,
    literal_full_g_transfer_diagnostic,
    load_selected_source_recovery,
    open_loop_pulse_port,
)
from experiments.mnist_relu_drn.ibm_om_winsorized_onchip_adam import (
    flatten_physical,
    split_flat_physical,
    sync_catalog_from_persistent,
)
from experiments.mnist_relu_drn.ibm_om_winsorized_cross_array_open_loop_adam_config import (
    FIXED_LOGIT_GAIN,
    NOMINAL_DELTA_X,
    WinsorizedCrossArrayOpenLoopAdamTrainSpec,
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
SUMMARY_SCHEMA = (
    "ebl.mnist_relu_drn.ibm_om_winsorized_cross_array_open_loop_adam_exploratory"
)
SUMMARY_SCHEMA_VERSION = 1
DEPLOYMENT_SCHEMA = (
    "ebl.mnist_relu_drn.ibm_om_winsorized_cross_array_deployment"
)
DEPLOYMENT_SCHEMA_VERSION = 1
LAYOUTS = ("halves", "paired")


def _compat_assignment_protocol(protocol: Any) -> Any:
    return SimpleNamespace(
        device=SimpleNamespace(
            corruption_policy="counterfactual_repaired",
            required_aihwkit_version="1.1.0",
        ),
        reset_commissioning=SimpleNamespace(
            samples_per_cell=protocol.target.reset_commissioning_samples,
        ),
        joint_assignment_repair=SimpleNamespace(
            maximum_candidates_per_quad=128,
        ),
    )


def _source_summary_path(protocol: Any) -> Path:
    return (
        _ROOT
        / "results"
        / protocol.source.result_id
        / protocol.source.run_relative_path
        / "artifacts/scientific_summary.json"
    )


def _load_and_validate_source_summary(protocol: Any) -> Mapping[str, Any]:
    path = _source_summary_path(protocol)
    if not path.is_file() or sha256_file(path) != protocol.source.summary_sha256:
        raise ValueError("Hash-pinned source recovery summary is missing or changed.")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("Expected a readable source recovery summary.") from error
    selected = payload.get("screen", {}).get("selected_arm")
    test = payload.get("selected_test", {}).get("metrics", {})
    if (
        selected != "stage2_direct_physical_rail_lr_3e-05"
        or test.get("student_correct") != protocol.source.test_correct
        or test.get("examples") != protocol.source.test_examples
    ):
        raise ValueError("Source recovery summary no longer names the frozen result.")
    return payload


def _programming_summary(
    result: Any,
    *,
    target_raw_x: torch.Tensor,
    pulse_population: Any,
) -> Mapping[str, Any]:
    lower = (pulse_population.min_bound.detach().cpu() + 1.0) / 2.0
    upper = (pulse_population.max_bound.detach().cpu() + 1.0) / 2.0
    target = target_raw_x.detach().cpu()
    persistent = result.apparent_endpoint.detach().cpu()
    del persistent
    exact_support = (target >= lower) & (target <= upper)
    verify_intersection = (
        target + 0.5 * NOMINAL_DELTA_X >= lower
    ) & (target - 0.5 * NOMINAL_DELTA_X <= upper)
    total = result.total_pulses.detach().cpu().to(torch.float64)
    return {
        "cells": int(target.numel()),
        "exact_target_in_support": int(exact_support.sum().item()),
        "exact_target_outside_support": int((~exact_support).sum().item()),
        "verify_window_intersects_support": int(verify_intersection.sum().item()),
        "apparent_accepted": int(result.accepted.sum().item()),
        "budget_exhausted": int(result.budget_exhausted.sum().item()),
        "set_pulses": int(result.set_count.sum().item()),
        "reset_pulses": int(result.reset_count.sum().item()),
        "total_pulses": int(result.total_pulses.sum().item()),
        "verify_reads": int(result.verify_count.sum().item()),
        "mean_total_pulses_per_cell": float(total.mean().item()),
        "maximum_total_pulses_per_cell": int(total.max().item()),
        "target_raw_x_minimum": float(target.min().item()),
        "target_raw_x_maximum": float(target.max().item()),
        "target_clipping": False,
        "apparent_endpoint_applied_to_drn": False,
    }


def _restore_deployment_plant(
    payload: Mapping[str, Any],
    *,
    pulse_population: Any,
    maximum_random_draws: int,
    device: torch.device,
) -> IbmReramRawActivePlant:
    state = payload["continuation_state"]
    if not isinstance(state, Mapping):
        raise ValueError("Deployment checkpoint lacks continuation state.")
    plant = IbmReramRawActivePlant(
        pulse_population,
        seeds=tuple(int(seed) for seed in state["seeds"]),
        device=device,
        maximum_random_draws=maximum_random_draws,
    )
    plant.load_state_dict(state)
    return plant


def _deploy_once(
    *,
    name: str,
    endpoint_seed: int,
    raw_x_targets: Sequence[torch.Tensor],
    target_kind: str,
    population: Any,
    pulse_population: Any,
    stack: Any,
    teacher: Any,
    test_loader: Iterable,
    protocol: Any,
    store: RunStore,
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    flat_target = flatten_physical(raw_x_targets).to(
        device=stack.device, dtype=torch.float32
    )
    plant = IbmReramRawActivePlant(
        pulse_population,
        seeds=matched_trajectory_seeds(population, endpoint_seed=endpoint_seed),
        device=stack.device,
        maximum_random_draws=protocol.deployment.maximum_random_draws,
    )
    plant.initialize_at_sampled_lower()
    result = run_program_verify(
        plant.controller_port(),
        targets=flat_target,
        tolerance=protocol.deployment.verify_tolerance_raw_x,
        maximum_pulses=protocol.deployment.maximum_program_pulses,
        settings=ControllerSettings(kind="one_pulse"),
    )
    sync_catalog_from_persistent(
        stack.bundle.catalog,
        plant,
        binding_shapes=population.binding_shapes,
    )
    metrics, prediction = _evaluate_detailed(
        stack, teacher, test_loader, sample_limit=None
    )
    payload = {
        "schema": DEPLOYMENT_SCHEMA,
        "schema_version": DEPLOYMENT_SCHEMA_VERSION,
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
        "test": dict(metrics),
        "test_prediction_sha256": _tensor_sha256(prediction.to(torch.int64)),
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
        "test": dict(metrics),
        "test_prediction_sha256": _tensor_sha256(prediction.to(torch.int64)),
    }
    store.append_metric({"mode": "cross_array_deployment", **report})
    print(
        f"cross deploy {target_kind} seed={endpoint_seed}: "
        f"test={100.0*float(metrics['student_accuracy']):.2f}% "
        f"exhausted={report['programming']['budget_exhausted']}",
        flush=True,
    )
    del plant
    gc.collect()
    torch.cuda.empty_cache()
    return report, payload


def _delta_summary(value: torch.Tensor) -> Mapping[str, float]:
    flat = value.detach().to(device="cpu", dtype=torch.float64).reshape(-1)
    return {
        "bias": float(flat.mean().item()),
        "mae": float(flat.abs().mean().item()),
        "rms": float(flat.square().mean().sqrt().item()),
        "maximum_absolute": float(flat.abs().max().item()),
    }


def _physical_drift(
    before_native: torch.Tensor,
    after_native: torch.Tensor,
    *,
    binding_shapes: Sequence[Sequence[int]],
    pulse_population: Any,
) -> Mapping[str, Any]:
    before_g = split_flat_physical(before_native.detach().cpu() + 1.0, binding_shapes)
    after_g = split_flat_physical(after_native.detach().cpu() + 1.0, binding_shapes)
    sign = torch.tensor((1.0, -1.0, -1.0, 1.0), dtype=torch.float32)
    layers = []
    load_all = []
    contrast_all = []
    for layer_index, (before, after, layout) in enumerate(
        zip(before_g, after_g, LAYOUTS)
    ):
        delta_quad = quad_stack(after - before, layout=layout)
        load = delta_quad.mean(dim=-1)
        contrast = (delta_quad * sign).sum(dim=-1) / 2.0
        load_all.append(load.reshape(-1))
        contrast_all.append(contrast.reshape(-1))
        layers.append(
            {
                "layer": layer_index,
                "layout": layout,
                "quad_mean_loading_delta_full_G": _delta_summary(load),
                "signed_contrast_delta_full_G": _delta_summary(contrast),
            }
        )
    lower = pulse_population.min_bound.detach().cpu()
    upper = pulse_population.max_bound.detach().cpu()
    return {
        "layers": layers,
        "all_quad_mean_loading_delta_full_G": _delta_summary(torch.cat(load_all)),
        "all_signed_contrast_delta_full_G": _delta_summary(torch.cat(contrast_all)),
        "changed_cells": int((before_native.detach().cpu() != after_native.detach().cpu()).sum().item()),
        "persistent_saturated_lower": int(
            (torch.abs(after_native.detach().cpu() - lower) <= 2e-6).sum().item()
        ),
        "persistent_saturated_upper": int(
            (torch.abs(after_native.detach().cpu() - upper) <= 2e-6).sum().item()
        ),
    }


def _fine_tune_open_loop(
    *,
    p0: Mapping[str, Any],
    population: Any,
    pulse_population: Any,
    stack: Any,
    teacher: Any,
    loaders: Any,
    protocol: Any,
    store: RunStore,
) -> Mapping[str, Any]:
    plant = _restore_deployment_plant(
        p0,
        pulse_population=pulse_population,
        maximum_random_draws=protocol.deployment.maximum_random_draws,
        device=stack.device,
    )
    before_native = plant.persistent.detach().cpu().clone()
    sync_catalog_from_persistent(
        stack.bundle.catalog, plant, binding_shapes=population.binding_shapes
    )
    before_validation, before_validation_prediction = _evaluate_detailed(
        stack, teacher, loaders.validation, sample_limit=None
    )
    before_test, before_test_prediction = _evaluate_detailed(
        stack, teacher, loaders.test, sample_limit=None
    )
    optimizer = ColumnSerialOpenLoopAdam(
        open_loop_pulse_port(plant),
        device=stack.device,
        binding_shapes=population.binding_shapes,
        learning_rate_raw_x=protocol.recovery.learning_rate_raw_x,
        nominal_delta_x=NOMINAL_DELTA_X,
        pulse_cap=protocol.recovery.pulse_cap,
        pulse_selection_seed=protocol.recovery.pulse_selection_seed,
        beta1=protocol.recovery.beta1,
        beta2=protocol.recovery.beta2,
        epsilon=protocol.recovery.epsilon,
    )
    examples = 0
    batches = 0
    train_kl = 0.0
    train_correct = 0
    conceptual_phases = 0
    row_assertions = 0
    requested = 0
    applied = 0
    capped = 0
    upward = 0
    downward = 0
    probability_clipped = 0
    for batch_index, (inputs, labels) in enumerate(
        limited(loaders.train, protocol.recovery.maximum_batches)
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
        gradients = tuple(stack.differentiator.compute_gradient())
        step = optimizer.step(gradients)
        conceptual_phases += step.conceptual_column_phases
        row_assertions += step.stochastic_row_line_assertions
        requested += step.requested_cell_coincidences
        applied += step.applied_cell_pulses
        capped += step.capped_cell_requests
        upward += step.upward_pulses
        downward += step.downward_pulses
        probability_clipped += step.probability_clipped_cells
        examples += int(labels.shape[0])
        batches = batch_index + 1
    if batches < 1:
        raise RuntimeError("Expected open-loop Adam to process at least one batch.")
    sync_catalog_from_persistent(
        stack.bundle.catalog, plant, binding_shapes=population.binding_shapes
    )
    after_validation, after_validation_prediction = _evaluate_detailed(
        stack, teacher, loaders.validation, sample_limit=None
    )
    after_test, after_test_prediction = _evaluate_detailed(
        stack, teacher, loaders.test, sample_limit=None
    )
    drift = _physical_drift(
        before_native,
        plant.persistent,
        binding_shapes=population.binding_shapes,
        pulse_population=pulse_population,
    )
    checkpoint_payload = {
        **optimizer.state_dict(),
        "plant_continuation_state": plant.state_dict(),
        "source_deployment_sha256": sha256_file(Path(str(p0["artifact_path"]))),
        "assignment_seed": int(protocol.target.assignment_seed),
        "endpoint_seed": int(protocol.target.fine_tune_endpoint_seed),
        "before_validation": dict(before_validation),
        "after_validation": dict(after_validation),
        "before_test": dict(before_test),
        "after_test": dict(after_test),
        "target_clipping": False,
        "verify_reads_during_updates": 0,
        "persistent_endpoint_applied_to_drn": True,
        "apparent_endpoint_applied_to_drn": False,
    }
    checkpoint = atomic_torch_save(
        checkpoint_payload,
        store.run_dir / "checkpoints/open_loop_adam_epoch_1.pt",
    )
    report = {
        "claim_label": CLAIM_LABEL,
        "update_surface": UPDATE_SURFACE,
        "learning_rate_raw_x": protocol.recovery.learning_rate_raw_x,
        "epochs": protocol.recovery.epochs,
        "train": {
            "examples": examples,
            "batches": batches,
            "kl_teacher_student": train_kl / examples,
            "student_accuracy": train_correct / examples,
        },
        "before_validation": dict(before_validation),
        "after_validation": dict(after_validation),
        "before_test": dict(before_test),
        "after_test": dict(after_test),
        "prediction_sha256": {
            "before_validation": _tensor_sha256(before_validation_prediction.to(torch.int64)),
            "after_validation": _tensor_sha256(after_validation_prediction.to(torch.int64)),
            "before_test": _tensor_sha256(before_test_prediction.to(torch.int64)),
            "after_test": _tensor_sha256(after_test_prediction.to(torch.int64)),
        },
        "pulse": {
            "conceptual_column_phases": conceptual_phases,
            "conceptual_column_phases_per_minibatch": list(
                protocol.recovery.conceptual_column_phases_per_minibatch
            ),
            "column_phase_vectorization": protocol.recovery.vectorization_equivalence,
            "stochastic_row_line_assertions": row_assertions,
            "requested_cell_coincidences": requested,
            "applied_cell_pulses": applied,
            "capped_cell_requests": capped,
            "upward": upward,
            "downward": downward,
            "probability_clipped_cells_across_steps": probability_clipped,
            "cells_at_cap": int((optimizer.pulse_count >= protocol.recovery.pulse_cap).sum().item()),
            "mean_per_cell": float(optimizer.pulse_count.to(torch.float64).mean().item()),
            "maximum_per_cell": int(optimizer.pulse_count.max().item()),
            "verify_reads_during_updates": 0,
        },
        "physical_drift": drift,
        "validation_used_for_selection": False,
        "hyperparameters_retuned_on_target": False,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
    }
    store.append_metric({"mode": "open_loop_adam", **report})
    print(
        "open-loop Adam epoch=1: "
        f"test={100.0*float(before_test['student_accuracy']):.2f}%->"
        f"{100.0*float(after_test['student_accuracy']):.2f}% "
        f"pulses={applied} verify_reads=0",
        flush=True,
    )
    del optimizer, plant
    gc.collect()
    torch.cuda.empty_cache()
    return report


def run_train(request: "TrainRequest") -> int:
    spec = request.spec
    if not isinstance(spec, WinsorizedCrossArrayOpenLoopAdamTrainSpec):
        raise TypeError("Expected the dedicated cross-array open-loop Adam spec.")
    if request.weights is None or request.teacher_weights is None:
        raise ValueError("Expected explicit --weights source recovery and --teacher-weights.")
    if (
        request.base_weights is not None
        or request.resume is not None
        or request.device_data is not None
        or request.device_model is not None
    ):
        raise ValueError("Cross-array recovery rejects base/resume/device inputs.")
    protocol = spec.protocol
    source_checkpoint_path = request.weights.expanduser().resolve()
    teacher_path = request.teacher_weights.expanduser().resolve()
    if sha256_file(teacher_path) != protocol.expected_teacher_weights_sha256:
        raise ValueError("Frozen teacher checkpoint SHA-256 mismatch.")
    source_summary = _load_and_validate_source_summary(protocol)
    source_summary_path = _source_summary_path(protocol)
    source_payload = load_selected_source_recovery(
        source_checkpoint_path,
        expected_sha256=protocol.source.checkpoint_sha256,
        expected_learning_rate_raw_x=protocol.source.selected_learning_rate_raw_x,
    )
    sampler = _sampler_python()
    store = RunStore.create(
        output_root=request.output_dir,
        experiment_id=spec.experiment_id,
        resolved_config=to_plain_data(spec),
        command=request.command,
        repo_root=_ROOT,
        input_artifacts=(
            _input("teacher_weights", teacher_path),
            _input("selected_87003_recovery", source_checkpoint_path),
            _input("selected_87003_recovery_summary", source_summary_path),
            _input("aihwkit_python", sampler),
        ),
        resume_capability="unsupported",
    )
    try:
        if spec.student.runtime.device != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("Fresh-array transfer and recovery require CUDA.")
        torch.manual_seed(spec.student.runtime.seed)
        torch.cuda.manual_seed_all(spec.student.runtime.seed)
        device = torch.device("cuda")
        loaders = build_mnist_loaders(
            spec.student.data,
            data_seed=spec.student.runtime.data_seed,
            calibration_examples=spec.student.mapping.calibration_examples,
            calibration_batch_size=spec.student.mapping.calibration_batch_size,
        )
        teacher, teacher_metadata = _load_teacher(
            teacher_path, device=device, spec=spec.student
        )
        logical_weights = tuple(
            value.detach().cpu().clone() for value in teacher.parameters()
        )
        if len(logical_weights) != 2:
            raise RuntimeError("Expected exactly two frozen teacher tensors.")
        stack = build_student_stack(spec.student, enable_measured=False)
        stack.cost.gain = FIXED_LOGIT_GAIN
        artifact_root = store.run_dir / "artifacts"
        target_hardware, target_artifacts = _prepare_assignment(
            assignment_seed=protocol.target.assignment_seed,
            assignment_role="fresh_target_87004",
            logical_weights=logical_weights,
            stack=stack,
            protocol=_compat_assignment_protocol(protocol),
            aihwkit_python=sampler,
            artifact_root=artifact_root,
        )
        target_native = _native_assignment_from_joint_artifact(target_hardware)
        target_artifacts.extend(
            _save_winsorized_identity_artifacts(
                artifact_root / "fresh_target_87004" / "winsorized_identity",
                target_native,
            )
        )
        anchor_mapping = build_truncated_nominal_baseline_spacing_mapping(
            logical_weights,
            baseline_position_fraction=0.0,
            spacing_delta_multiples=1,
            scale_fractions=(1.0, 1.0),
            nominal_dw_min=float(target_hardware["nominal_dw_min"]),
            cell_lower_raw_x=target_native["cell_lower_raw_x"],
            cell_upper_raw_x=target_native["cell_upper_raw_x"],
            reset_baseline_raw_x=target_native["reset_baseline_raw_x"],
            intrinsic_references_native=target_hardware["intrinsic_references_native"],
        )
        mapping_artifact = _save_design_mapping(
            artifact_root / "fresh_target_87004" / "baseline_anchor_mapping.npz",
            anchor_mapping,
            assignment_seed=protocol.target.assignment_seed,
            assignment_role="fresh_target_baseline_anchor_only",
            hardware_instance_id=target_hardware["hardware_instance_id"],
        )
        target_artifacts.append(
            {**mapping_artifact, "kind": "fresh_target_baseline_anchor_mapping"}
        )
        target_artifacts.append(
            {"path": mapping_artifact["context"], "kind": "fresh_target_baseline_anchor_context"}
        )
        transfer = build_target_baseline_contrast_transfer(
            source_payload,
            target_baseline_raw_x=anchor_mapping.baseline_raw_x,
            target_lower_raw_x=target_native["cell_lower_raw_x"],
            target_upper_raw_x=target_native["cell_upper_raw_x"],
        )
        literal = literal_full_g_transfer_diagnostic(
            source_payload,
            target_lower_raw_x=target_native["cell_lower_raw_x"],
            target_upper_raw_x=target_native["cell_upper_raw_x"],
        )
        literal_targets = literal["raw_x_targets"]
        literal_report = {key: value for key, value in literal.items() if key != "raw_x_targets"}
        # This deliberately evaluates the mathematical requested map even
        # where a target cell cannot reach it.  No clipping is performed and
        # this state is never used as a persistent deployment initializer.
        apply_targets(stack.bundle.catalog, transfer.target_full_conductance)
        requested_remap_metrics, requested_remap_prediction = _evaluate_detailed(
            stack, teacher, loaders.test, sample_limit=None
        )
        support_limited_raw_x = tuple(
            torch.minimum(
                torch.maximum(target, lower.to(target.dtype)),
                upper.to(target.dtype),
            )
            for target, lower, upper in zip(
                transfer.target_raw_x,
                target_native["cell_lower_raw_x"],
                target_native["cell_upper_raw_x"],
            )
        )
        support_limited_full_g = tuple(2.0 * value for value in support_limited_raw_x)
        apply_targets(stack.bundle.catalog, support_limited_full_g)
        support_limited_oracle_metrics, support_limited_oracle_prediction = (
            _evaluate_detailed(stack, teacher, loaders.test, sample_limit=None)
        )
        apply_targets(
            stack.bundle.catalog,
            tuple(2.0 * value for value in literal_targets),
        )
        literal_requested_metrics, literal_requested_prediction = _evaluate_detailed(
            stack, teacher, loaders.test, sample_limit=None
        )
        if (
            literal_requested_metrics["student_correct"]
            != protocol.source.test_correct
            or literal_requested_metrics["examples"]
            != protocol.source.test_examples
        ):
            raise RuntimeError(
                "Literal source-full-G no-write replay did not reproduce the "
                "frozen 9414/10000 source result; aborting before stochastic P&V."
            )
        population = target_native["population"]
        pulse_population = array_population_as_pulse_population(population)
        if tuple(population.binding_shapes) != tuple(
            tuple(shape) for shape in source_payload["binding_shapes"]
        ):
            raise RuntimeError("Source and target physical bindings differ.")
        print(
            "prepared fresh target 87004: "
            f"hardware={target_hardware['hardware_instance_id']} "
            f"unsupported_weights={transfer.report['unsupported_logical_weights']} "
            f"requested_remap={100.0*float(requested_remap_metrics['student_accuracy']):.2f}%",
            flush=True,
        )

        deployment_reports = []
        p0_payload = None
        for endpoint_seed in protocol.target.endpoint_seeds:
            report, payload = _deploy_once(
                name=f"contrast_transfer_seed_{endpoint_seed}",
                endpoint_seed=endpoint_seed,
                raw_x_targets=transfer.target_raw_x,
                target_kind="target_baseline_plus_source_signed_contrast",
                population=population,
                pulse_population=pulse_population,
                stack=stack,
                teacher=teacher,
                test_loader=loaders.test,
                protocol=protocol,
                store=store,
            )
            deployment_reports.append(report)
            if endpoint_seed == protocol.target.fine_tune_endpoint_seed:
                p0_payload = dict(payload)
                p0_payload["artifact_path"] = report["path"]
        if p0_payload is None:
            raise RuntimeError("Fine-tuning P0 seed was not deployed.")
        literal_deployment_report, _literal_payload = _deploy_once(
            name=f"literal_full_g_seed_{protocol.target.fine_tune_endpoint_seed}",
            endpoint_seed=protocol.target.fine_tune_endpoint_seed,
            raw_x_targets=literal_targets,
            target_kind="literal_source_full_G_diagnostic",
            population=population,
            pulse_population=pulse_population,
            stack=stack,
            teacher=teacher,
            test_loader=loaders.test,
            protocol=protocol,
            store=store,
        )
        recovery = _fine_tune_open_loop(
            p0=p0_payload,
            population=population,
            pulse_population=pulse_population,
            stack=stack,
            teacher=teacher,
            loaders=loaders,
            protocol=protocol,
            store=store,
        )
        accuracies = [float(value["test"]["student_accuracy"]) for value in deployment_reports]
        deployment_mean = sum(accuracies) / len(accuracies)
        deployment_variance = sum(
            (value - deployment_mean) ** 2 for value in accuracies
        ) / len(accuracies)
        source_accuracy = protocol.source.test_correct / protocol.source.test_examples
        summary = {
            "schema": SUMMARY_SCHEMA,
            "schema_version": SUMMARY_SCHEMA_VERSION,
            "evidence_tier": EVIDENCE_TIER,
            "claim_label": CLAIM_LABEL,
            "fixed_logit_gain": FIXED_LOGIT_GAIN,
            "source": {
                "assignment_seed": protocol.source.assignment_seed,
                "checkpoint_path": str(source_checkpoint_path),
                "checkpoint_sha256": protocol.source.checkpoint_sha256,
                "summary_path": str(source_summary_path),
                "summary_sha256": protocol.source.summary_sha256,
                "test_correct": protocol.source.test_correct,
                "test_examples": protocol.source.test_examples,
                "test_accuracy": source_accuracy,
                "selected_arm": source_summary["screen"]["selected_arm"],
                "transfer_object": protocol.source.transfer_object,
            },
            "target": {
                "assignment_seed": protocol.target.assignment_seed,
                "role": protocol.target.role,
                "hardware": _hardware_summary(target_hardware),
                "native_coordinate": target_native["report"],
                "baseline_anchor_mapping": anchor_mapping.report,
            },
            "contrast_transfer": dict(transfer.report),
            "requested_remap_no_write": {
                "role": (
                    "mathematical_unbounded_target_diagnostic_never_deployed_or_"
                    "used_for_initialization"
                ),
                "target_clipping": False,
                "metrics": dict(requested_remap_metrics),
                "prediction_sha256": _tensor_sha256(
                    requested_remap_prediction.to(torch.int64)
                ),
            },
            "support_limited_oracle": {
                "role": (
                    "analysis_only_cellwise_projection_diagnostic_never_deployed_"
                    "or_used_for_selection"
                ),
                "projection_is_physical_protocol": False,
                "projected_cells": transfer.report["unsupported_cells"],
                "metrics": dict(support_limited_oracle_metrics),
                "prediction_sha256": _tensor_sha256(
                    support_limited_oracle_prediction.to(torch.int64)
                ),
            },
            "literal_full_G_transfer_diagnostic": {
                **literal_report,
                "requested_no_write": {
                    "metrics": dict(literal_requested_metrics),
                    "prediction_sha256": _tensor_sha256(
                        literal_requested_prediction.to(torch.int64)
                    ),
                },
                "deployment": literal_deployment_report,
            },
            "deployments": {
                "endpoint_seeds": list(protocol.target.endpoint_seeds),
                "maximum_random_draws_identical": True,
                "maximum_random_draws": protocol.deployment.maximum_random_draws,
                "p0_endpoint_seed": protocol.target.fine_tune_endpoint_seed,
                "repeats": deployment_reports,
                "mean_test_accuracy": deployment_mean,
                "population_standard_deviation_test_accuracy": math.sqrt(
                    deployment_variance
                ),
                "minimum_test_accuracy": min(accuracies),
                "maximum_test_accuracy": max(accuracies),
                "retention_from_source_accuracy_mean": deployment_mean - source_accuracy,
            },
            "open_loop_recovery": recovery,
            "accuracy_ladder": {
                "source_87003_recovered_test_accuracy": source_accuracy,
                "target_87004_requested_remap_no_write_test_accuracy": (
                    requested_remap_metrics["student_accuracy"]
                ),
                "target_87004_five_seed_pv_mean_test_accuracy": deployment_mean,
                "target_87004_seed_89401_pre_adam_test_accuracy": recovery[
                    "before_test"
                ]["student_accuracy"],
                "target_87004_seed_89401_post_adam_test_accuracy": recovery[
                    "after_test"
                ]["student_accuracy"],
                "source_to_requested_remap_delta": (
                    requested_remap_metrics["student_accuracy"] - source_accuracy
                ),
                "requested_remap_to_five_seed_pv_mean_delta": (
                    deployment_mean - requested_remap_metrics["student_accuracy"]
                ),
                "seed_89401_open_loop_adam_delta": (
                    recovery["after_test"]["student_accuracy"]
                    - recovery["before_test"]["student_accuracy"]
                ),
            },
            "limitations": [
                "exploratory_noncanonical_single_fresh_model_assignment_87004",
                "model_based_nominal_bound_winsorization_not_measured_chip",
                "digital_teacher_BPTT_and_Adam_moments",
                "column_serial_coincidence_emulator_not_parallel_IBM_or_Tiki_Taka",
                "no_inference_read_noise_retention_or_drift",
                "target_requests_deliberately_unclipped_and_some_are_unreachable",
                "one_endpoint_only_receives_open_loop_recovery",
            ],
            "target_clipping": False,
            "verify_reads_during_fine_tuning": 0,
            "apparent_endpoint_applied_to_drn": False,
        }
        summary_path = artifact_root / "scientific_summary.json"
        atomic_write_json(summary_path, summary)
        target_artifacts.append(
            {"path": str(summary_path), "kind": "scientific_summary"}
        )
        for report in deployment_reports:
            target_artifacts.append(
                {"path": report["path"], "kind": "cross_array_deployment_state"}
            )
        target_artifacts.append(
            {
                "path": literal_deployment_report["path"],
                "kind": "literal_full_g_deployment_diagnostic",
            }
        )
        target_artifacts.append(
            {"path": recovery["checkpoint"], "kind": "open_loop_recovery_state"}
        )
        terminal = {
            "evidence_tier": EVIDENCE_TIER,
            "source_assignment_seed": protocol.source.assignment_seed,
            "target_assignment_seed": protocol.target.assignment_seed,
            "endpoint_count": len(deployment_reports),
            "source_test_accuracy": source_accuracy,
            "requested_remap_no_write_test_accuracy": requested_remap_metrics[
                "student_accuracy"
            ],
            "cross_array_mean_test_accuracy": deployment_mean,
            "cross_array_p0_test_accuracy": recovery["before_test"]["student_accuracy"],
            "open_loop_after_test_accuracy": recovery["after_test"]["student_accuracy"],
            "unsupported_logical_weights": transfer.report["unsupported_logical_weights"],
            "verify_reads_during_fine_tuning": 0,
            "target_clipping": False,
        }
        store.complete(
            metrics=terminal,
            artifacts=_artifact_records(store, target_artifacts),
        )
        print(f"run_dir={store.run_dir}", flush=True)
        return 0
    except BaseException as error:
        store.fail(error)
        raise


__all__ = ["run_train"]
