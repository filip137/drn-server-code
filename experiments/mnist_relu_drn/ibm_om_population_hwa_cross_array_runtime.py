"""Positive-G population HWA followed by independent A--D deployment.

The ordering is part of the numerical contract:

1. normalize the frozen ReLU weights into one clean signed master per layer;
2. map and physically program that raw initializer on Array A;
3. train fresh clean-master copies with a healthy population programming-error
   kernel that contains no Array-A identity, bounds, reference, or fault mask;
4. freeze and hash the fixed epoch-10 HWA checkpoints; and only then
5. sample Arrays B--D and independently program the raw and HWA states.

All network forwards use complete positive conductances in ``G=[0,2]``.  HWA
uses apparent accepted endpoints, while physical A--D evaluation uses the
persistent raw-active state.  Repaired and published-corrupt writes are kept
as a matched deployment intervention.
"""

from __future__ import annotations

import gc
import json
import math
from pathlib import Path
from statistics import fmean, pstdev
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
from experiments.mnist_relu_drn.ibm_om_corrupt_source_hwa_cross_array_pulse_adam import (
    build_published_persistent_targets,
    build_repaired_mapping_templates,
    clamp_targets_to_published_corrupt_singletons,
)
from experiments.mnist_relu_drn.ibm_om_corrupt_source_hwa_cross_array_pulse_adam_runtime import (
    _initial_logical_masters,
    _population_report,
    _sample_and_save_pair,
)
from experiments.mnist_relu_drn.ibm_om_global_positive_g import (
    lift_global_positive_g_gradients,
    map_global_positive_g,
)
from experiments.mnist_relu_drn.ibm_om_population_hwa_cross_array_config import (
    HWA_CONDITION_KEY,
    HWA_ESTIMATOR_KEY,
    PopulationHwaCrossArrayTrainSpec,
)
from experiments.mnist_relu_drn.runtime import _load_teacher
from experiments.mnist_relu_drn.ibm_om_winsorized_onchip_adam import (
    flatten_physical,
    split_flat_physical,
    sync_catalog_from_persistent,
)
from experiments.mnist_shared import build_mnist_loaders, limited
from experiments.schema import to_plain_data
from training.checkpoint import atomic_torch_save
from training.ibm_reram_population_hwa import (
    IbmReramPositiveGPopulationHwaConfig,
    IbmReramPositiveGProgrammingErrorSampler,
    load_raw_active_accepted_endpoint_model,
)
from training.ibm_reram_program_verify import (
    ControllerSettings,
    IbmReramRawActivePlant,
    PopulationStepEstimator,
    run_program_verify,
)
from training.ibm_reram_raw_active_program_verify import (
    RAW_ACTIVE_COORDINATE,
    array_population_as_pulse_population,
    matched_trajectory_seeds,
)


if TYPE_CHECKING:
    from ebl.cli import TrainRequest


_ROOT = Path(__file__).resolve().parents[2]
SUMMARY_SCHEMA = "ebl.mnist_relu_drn.ibm_om_population_hwa_cross_array"
SUMMARY_SCHEMA_VERSION = 1
CHECKPOINT_SCHEMA = "ebl.mnist_relu_drn.ibm_om_positive_g_population_hwa_master"
CHECKPOINT_SCHEMA_VERSION = 1
DEPLOYMENT_SCHEMA = "ebl.mnist_relu_drn.ibm_om_population_hwa_deployment"
DEPLOYMENT_SCHEMA_VERSION = 1


def _positive_full_g(values: Sequence[torch.Tensor]) -> tuple[torch.Tensor, ...]:
    selected = tuple(values)
    if len(selected) != 2:
        raise ValueError("Expected two positive-G DRN conductance tensors.")
    if any(
        not isinstance(value, torch.Tensor)
        or not value.is_floating_point()
        or not bool(torch.all(torch.isfinite(value)))
        or bool(torch.any(value < 0.0))
        or bool(torch.any(value > 2.0))
        for value in selected
    ):
        raise ValueError("Expected every DRN conductance to be finite in G=[0,2].")
    return selected


def _apply_full_g(stack: Any, values: Sequence[torch.Tensor]) -> None:
    selected = _positive_full_g(values)
    apply_full_conductance_targets(
        stack.bundle.catalog,
        selected,
        conductance_min=0.0,
        conductance_max=2.0,
    )


def _snapshot_masters(values: Sequence[torch.Tensor]) -> tuple[torch.Tensor, ...]:
    result = tuple(
        value.detach().cpu().to(torch.float32).contiguous().clone()
        for value in values
    )
    if len(result) != 2:
        raise ValueError("Expected exactly two logical masters.")
    return result


def _load_deployment_estimator(
    estimator_path: Path,
    *,
    expected_sha256: str,
    endpoint_metadata: Mapping[str, Any],
) -> tuple[PopulationStepEstimator, str, Path]:
    """Load the coordinate-matched calibration-only adaptive estimator."""

    resolved = estimator_path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(
            f"Expected a step-estimator artifact. Provided value: {str(resolved)!r}."
        )
    digest = sha256_file(resolved)
    if digest != expected_sha256:
        raise ValueError("Step-estimator artifact SHA-256 does not match endpoint metadata.")
    try:
        artifact = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("Expected a readable step-estimator JSON artifact.") from error
    metadata = artifact.get("metadata") if isinstance(artifact, Mapping) else None
    estimators = artifact.get("estimators") if isinstance(artifact, Mapping) else None
    encoded = estimators.get(HWA_ESTIMATOR_KEY) if isinstance(estimators, Mapping) else None
    expected_metadata = {
        "execution_profile": "hwa_production_cap128",
        "preset": "reram_array_om",
        "enable_published_corruption": False,
        "nominal_dw_min": endpoint_metadata.get("nominal_dw_min"),
        "coordinate": RAW_ACTIVE_COORDINATE,
        "state_coordinate": "raw_active",
        "trajectory_artifact_sha256": endpoint_metadata.get(
            "trajectory_artifact_sha256"
        ),
        "device_population_artifact_sha256": endpoint_metadata.get(
            "device_population_artifact_sha256"
        ),
    }
    metadata_mismatch = {
        key: {"expected": expected, "provided": metadata.get(key)}
        for key, expected in expected_metadata.items()
        if isinstance(metadata, Mapping) and metadata.get(key) != expected
    }
    if (
        not isinstance(artifact, Mapping)
        or artifact.get("schema") != "ebl.ibm_reram.step_estimators"
        or artifact.get("schema_version") != 1
        or artifact.get("calibration_partition_only") is not True
        or not isinstance(metadata, Mapping)
        or bool(metadata_mismatch)
        or not isinstance(encoded, Mapping)
    ):
        raise ValueError(
            "Expected a healthy raw-active calibration estimator matched to "
            "the endpoint artifact."
        )
    estimator = PopulationStepEstimator.from_mapping(encoded)
    expected_fallback = float(endpoint_metadata["nominal_dw_min"]) / 2.0
    if not math.isclose(
        estimator.fallback_step,
        expected_fallback,
        rel_tol=0.0,
        abs_tol=1e-15,
    ):
        raise ValueError(
            "Expected the step estimator fallback to equal half the raw-active "
            "OM nominal pulse step."
        )
    return estimator, digest, resolved


def _load_hwa_inputs(
    endpoint_path: Path,
    estimator_path: Path,
) -> tuple[Any, str, Path, PopulationStepEstimator, str, Path, Mapping[str, Any]]:
    model, endpoint_digest, endpoint_resolved = load_raw_active_accepted_endpoint_model(
        endpoint_path,
        condition_key=HWA_CONDITION_KEY,
    )
    try:
        endpoint_artifact = json.loads(endpoint_resolved.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:  # pragma: no cover - loader read it
        raise ValueError("Could not reread the validated endpoint artifact.") from error
    metadata = endpoint_artifact.get("metadata")
    expected_estimator_digest = (
        metadata.get("step_estimator_artifact_sha256")
        if isinstance(metadata, Mapping)
        else None
    )
    controller = metadata.get("controller") if isinstance(metadata, Mapping) else None
    if (
        not isinstance(metadata, Mapping)
        or not isinstance(expected_estimator_digest, str)
        or len(expected_estimator_digest) != 64
        or metadata.get("execution_profile") != "hwa_production_cap128"
        or metadata.get("preset") != "reram_array_om"
        or metadata.get("aihwkit_version") != "1.1.0"
        or metadata.get("state_coordinate") != "raw_active"
        or metadata.get("sampled_reference_consumed_by_plant") is not False
        or not isinstance(controller, Mapping)
        or controller.get("maximum_program_pulses") != 128
        or controller.get("adaptive")
        != {
            "eta": 0.75,
            "maximum_batch": 32,
            "epsilon": 1e-8,
            "force_one_within_steps": 2.0,
        }
    ):
        raise ValueError(
            "Expected the raw-active healthy cap-128 HWA characterization."
        )
    estimator, estimator_digest, estimator_resolved = _load_deployment_estimator(
        estimator_path,
        expected_sha256=expected_estimator_digest,
        endpoint_metadata=metadata,
    )
    return (
        model,
        endpoint_digest,
        endpoint_resolved,
        estimator,
        estimator_digest,
        estimator_resolved,
        metadata,
    )


def _one_forward_gradient(
    *,
    stack: Any,
    teacher: Any,
    inputs: torch.Tensor,
    labels: torch.Tensor,
    masters: Sequence[torch.Tensor],
    full_g: Sequence[torch.Tensor],
) -> tuple[float, tuple[torch.Tensor, ...], Mapping[str, int | float]]:
    _apply_full_g(stack, full_g)
    with torch.no_grad():
        teacher_logits = teacher.logits(inputs)
    stack.network.set_input(inputs, reset=True)
    stack.minimizer.compute_equilibrium()
    stack.cost.set_teacher(teacher_logits, labels)
    with torch.no_grad():
        per_example_kl = stack.cost.eval()
        prediction = stack.cost.student_logits().argmax(dim=1)
        teacher_prediction = teacher_logits.argmax(dim=1)
        loss = float(per_example_kl.to(torch.float64).mean().item())
        metrics: Mapping[str, int | float] = {
            "examples": int(labels.numel()),
            "correct": int(prediction.eq(labels).sum().item()),
            "teacher_agreement": int(prediction.eq(teacher_prediction).sum().item()),
        }
    physical_gradients = tuple(stack.differentiator.compute_gradient())
    logical_gradients = lift_global_positive_g_gradients(
        masters,
        physical_gradients,
    )
    return loss, logical_gradients, metrics


def _evaluate_global_clean(
    *,
    stack: Any,
    teacher: Any,
    loader: Iterable,
    masters: Sequence[torch.Tensor],
    quantization_spacing_raw_x: float | None,
    sample_limit: int | None,
) -> Mapping[str, Any]:
    selected = tuple(
        value.to(device=stack.device, dtype=torch.float32) for value in masters
    )
    view = map_global_positive_g(
        selected,
        quantization_spacing_x=quantization_spacing_raw_x,
    )
    _apply_full_g(stack, view.full_conductance)
    metrics, prediction = _evaluate_detailed(
        stack,
        teacher,
        loader,
        sample_limit=sample_limit,
    )
    return {
        "metrics": dict(metrics),
        "prediction_sha256": _tensor_sha256(prediction.to(torch.int64)),
        "mapping": dict(view.mapping_report),
    }


def _train_population_hwa_arm(
    *,
    arm: Any,
    initial_master: Sequence[torch.Tensor],
    stack: Any,
    teacher: Any,
    loaders: Any,
    model: Any,
    endpoint_digest: str,
    protocol: Any,
    checkpoint_dir: Path,
    store: RunStore,
) -> tuple[tuple[torch.Tensor, ...], Mapping[str, Any], Path]:
    masters = tuple(
        torch.nn.Parameter(value.to(device=stack.device, dtype=torch.float32))
        for value in initial_master
    )
    sampler = IbmReramPositiveGProgrammingErrorSampler(
        model,
        IbmReramPositiveGPopulationHwaConfig(
            seed=int(protocol.hwa["programming_error_seed"]),
            ramp_epochs=int(protocol.hwa["ramp_epochs"]),
            initial_strength=float(protocol.hwa["initial_strength"]),
            final_strength=float(protocol.hwa["final_strength"]),
        ),
        artifact_sha256=endpoint_digest,
    )
    optimizer = torch.optim.Adam(
        [
            {"params": [master], "lr": float(rate)}
            for master, rate in zip(
                masters,
                protocol.hwa["learning_rates"],
                strict=True,
            )
        ],
        betas=tuple(float(value) for value in protocol.hwa["betas"]),
        eps=float(protocol.hwa["epsilon"]),
    )
    spacing = arm.quantization_spacing_raw_x
    history = []
    global_minibatch_ordinal = 0
    for epoch in range(1, int(protocol.hwa["epochs"]) + 1):
        sampler.set_epoch(epoch)
        totals = {
            "examples": 0,
            "batches": 0,
            "kl_sum": 0.0,
            "correct": 0,
            "agreement": 0,
            "rejected": 0,
            "scalar_draws": 0,
            "residual_sum": 0.0,
            "residual_square_sum": 0.0,
            "residual_values": 0,
        }
        gradient_squared = [0.0, 0.0]
        gradient_values = [0, 0]
        for inputs, labels in limited(
            loaders.train,
            protocol.execution["maximum_training_batches_per_epoch"],
        ):
            inputs = inputs.to(stack.device, dtype=torch.float32)
            labels = labels.to(stack.device, dtype=torch.long)
            with torch.no_grad():
                clean_view = map_global_positive_g(
                    masters,
                    quantization_spacing_x=spacing,
                )
                clean_flat = flatten_physical(clean_view.full_conductance)
                draw = sampler.sample_full_conductance(clean_flat)
                noisy_full_g = split_flat_physical(
                    draw.applied_g,
                    tuple(value.shape for value in clean_view.full_conductance),
                )
            loss, gradients, metrics = _one_forward_gradient(
                stack=stack,
                teacher=teacher,
                inputs=inputs,
                labels=labels,
                masters=masters,
                full_g=noisy_full_g,
            )
            optimizer.zero_grad(set_to_none=True)
            for layer, (master, gradient) in enumerate(
                zip(masters, gradients, strict=True)
            ):
                master.grad = gradient
                gradient_squared[layer] += float(
                    gradient.detach().to(torch.float64).square().sum().item()
                )
                gradient_values[layer] += int(gradient.numel())
            optimizer.step()
            with torch.no_grad():
                for master in masters:
                    master.clamp_(-1.0, 1.0)
            count = int(metrics["examples"])
            residual64 = draw.residual_x.detach().to(torch.float64)
            totals["examples"] += count
            totals["batches"] += 1
            totals["kl_sum"] += loss * count
            totals["correct"] += int(metrics["correct"])
            totals["agreement"] += int(metrics["teacher_agreement"])
            totals["rejected"] += int(draw.rejected_scalar_draws)
            totals["scalar_draws"] += int(draw.total_scalar_draws)
            totals["residual_sum"] += float(residual64.sum().item())
            totals["residual_square_sum"] += float(residual64.square().sum().item())
            totals["residual_values"] += int(residual64.numel())
            global_minibatch_ordinal += 1
        examples = int(totals["examples"])
        values = int(totals["residual_values"])
        if examples < 1 or values < 1:
            raise RuntimeError("Population HWA processed no training examples.")
        residual_mean = float(totals["residual_sum"]) / values
        residual_variance = max(
            0.0,
            float(totals["residual_square_sum"]) / values - residual_mean**2,
        )
        clean_validation = _evaluate_global_clean(
            stack=stack,
            teacher=teacher,
            loader=loaders.validation,
            masters=masters,
            quantization_spacing_raw_x=spacing,
            sample_limit=protocol.execution["evaluation_sample_limit"],
        )
        record = {
            "mode": "population_hwa",
            "arm_id": arm.arm_id,
            "epoch": epoch,
            "strength": sampler.strength,
            "train": {
                "examples": examples,
                "batches": int(totals["batches"]),
                "kl_teacher_student": float(totals["kl_sum"]) / examples,
                "student_accuracy": int(totals["correct"]) / examples,
                "teacher_agreement": int(totals["agreement"]) / examples,
                "raw_x_residual_mean": residual_mean,
                "raw_x_residual_std": math.sqrt(residual_variance),
                "rejected_out_of_range_apparent_draws": int(totals["rejected"]),
                "total_scalar_draws_including_rejections": int(totals["scalar_draws"]),
                "logical_gradient_rms": [
                    math.sqrt(total / count)
                    for total, count in zip(
                        gradient_squared, gradient_values, strict=True
                    )
                ],
            },
            "clean_global_validation": clean_validation,
        }
        history.append(record)
        store.append_metric(record)
        print(
            f"population HWA {arm.arm_id} epoch={epoch}/"
            f"{protocol.hwa['epochs']} strength={sampler.strength:.1f} "
            f"clean_val={100.0*float(clean_validation['metrics']['student_accuracy']):.2f}%",
            flush=True,
        )
    final_master = _snapshot_masters(masters)
    final_clean_test = _evaluate_global_clean(
        stack=stack,
        teacher=teacher,
        loader=loaders.test,
        masters=final_master,
        quantization_spacing_raw_x=spacing,
        sample_limit=protocol.execution["evaluation_sample_limit"],
    )
    checkpoint_payload = {
        "schema": CHECKPOINT_SCHEMA,
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "evidence_tier": protocol.evidence_tier,
        "arm_id": arm.arm_id,
        "epoch": int(protocol.hwa["epochs"]),
        "fixed_final_no_selection": True,
        "logical_master": final_master,
        "logical_master_sha256": [
            _tensor_sha256(value) for value in final_master
        ],
        "optimizer_state": optimizer.state_dict(),
        "population_hwa_sampler_state": sampler.state_dict(),
        "endpoint_model_sha256": endpoint_digest,
        "quantization_spacing_raw_x": spacing,
        "history": history,
        "final_clean_global_test": final_clean_test,
    }
    checkpoint = atomic_torch_save(
        checkpoint_payload,
        checkpoint_dir / f"{arm.arm_id}.pt",
    )
    report = {
        "arm_id": arm.arm_id,
        "role": arm.role,
        "quantization_spacing_raw_x": spacing,
        "initial_master_sha256": [
            _tensor_sha256(value) for value in initial_master
        ],
        "fixed_final_master_sha256": checkpoint_payload["logical_master_sha256"],
        "fixed_final_epoch": int(protocol.hwa["epochs"]),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "history": history,
        "final_clean_global_test": final_clean_test,
        "sampler": sampler.report(),
    }
    return final_master, report, checkpoint


def _evaluate_full_g(
    *,
    stack: Any,
    teacher: Any,
    loader: Iterable,
    full_g: Sequence[torch.Tensor],
    sample_limit: int | None,
) -> Mapping[str, Any]:
    _apply_full_g(stack, full_g)
    metrics, prediction = _evaluate_detailed(
        stack,
        teacher,
        loader,
        sample_limit=sample_limit,
    )
    return {
        "metrics": dict(metrics),
        "prediction_sha256": _tensor_sha256(prediction.to(torch.int64)),
        "negative_conductance_count": 0,
    }


def _program_once(
    *,
    array_label: str,
    state_label: str,
    population_role: str,
    population: Any,
    raw_x_targets: Sequence[torch.Tensor],
    endpoint_seed: int,
    estimator: PopulationStepEstimator,
    protocol: Any,
    stack: Any,
    teacher: Any,
    test_loader: Iterable,
    store: RunStore,
) -> tuple[Mapping[str, Any], Path]:
    pulse_population = array_population_as_pulse_population(population)
    target = flatten_physical(raw_x_targets).to(
        device=stack.device,
        dtype=torch.float32,
    )
    deployment = protocol.deployment
    plant = IbmReramRawActivePlant(
        pulse_population,
        seeds=matched_trajectory_seeds(population, endpoint_seed=endpoint_seed),
        device=stack.device,
        maximum_random_draws=int(deployment["maximum_random_draws"]),
    )
    plant.initialize_at_sampled_lower()
    result = run_program_verify(
        plant.controller_port(),
        targets=target,
        tolerance=float(deployment["verify_tolerance_raw_x"]),
        maximum_pulses=int(deployment["maximum_program_pulses"]),
        settings=ControllerSettings(
            kind="adaptive",
            eta=0.75,
            maximum_batch=32,
            epsilon=1e-8,
            force_one_within_steps=2.0,
        ),
        estimator=estimator,
    )
    persistent_x = (plant.persistent + 1.0) / 2.0
    persistent_inside = (
        torch.abs(persistent_x - target)
        <= float(deployment["verify_tolerance_raw_x"])
    )
    lower = (plant.population.min_bound + 1.0) / 2.0
    upper = (plant.population.max_bound + 1.0) / 2.0
    exact_support = (target >= lower) & (target <= upper)
    sync_catalog_from_persistent(
        stack.bundle.catalog,
        plant,
        binding_shapes=population.binding_shapes,
    )
    metrics, prediction = _evaluate_detailed(
        stack,
        teacher,
        test_loader,
        sample_limit=protocol.execution["evaluation_sample_limit"],
    )
    name = (
        f"array_{array_label}/{state_label}/"
        f"{population_role}_seed_{endpoint_seed}"
    )
    payload = {
        "schema": DEPLOYMENT_SCHEMA,
        "schema_version": DEPLOYMENT_SCHEMA_VERSION,
        "evidence_tier": protocol.evidence_tier,
        "array_label": array_label,
        "assignment_seed": int(population.assignment_seed),
        "state_label": state_label,
        "population_role": population_role,
        "population_fingerprint": population.fingerprint,
        "endpoint_seed": int(endpoint_seed),
        "target_raw_x": target.detach().cpu().clone(),
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
        "persistent_inside_acceptance_window": persistent_inside.detach().cpu().clone(),
        "continuation_state": plant.state_dict(),
        "test": dict(metrics),
        "test_prediction_sha256": _tensor_sha256(prediction.to(torch.int64)),
        "persistent_endpoint_applied_to_drn": True,
        "apparent_endpoint_applied_to_drn": False,
        "target_clipping": False,
        "inference_read_noise": False,
    }
    path = atomic_torch_save(
        payload,
        store.run_dir / "artifacts" / "deployments" / f"{name}.pt",
    )
    total_pulses = result.total_pulses.detach().to(torch.float64)
    report = {
        "array_label": array_label,
        "assignment_seed": int(population.assignment_seed),
        "state_label": state_label,
        "population_role": population_role,
        "population_fingerprint": population.fingerprint,
        "endpoint_seed": int(endpoint_seed),
        "artifact_path": str(path),
        "artifact_sha256": sha256_file(path),
        "programming": {
            "cells": int(target.numel()),
            "exact_target_in_support": int(exact_support.sum().item()),
            "exact_target_outside_support": int((~exact_support).sum().item()),
            "apparent_accepted": int(result.accepted.sum().item()),
            "persistent_inside_acceptance_window": int(
                persistent_inside.sum().item()
            ),
            "budget_exhausted": int(result.budget_exhausted.sum().item()),
            "nonfinite": int(result.nonfinite.sum().item()),
            "total_pulses": int(result.total_pulses.sum().item()),
            "mean_pulses_per_cell": float(total_pulses.mean().item()),
            "maximum_pulses_per_cell": int(result.total_pulses.max().item()),
            "verify_reads": int(result.verify_count.sum().item()),
            "persistent_and_apparent_acceptance_are_separate": True,
        },
        "test": dict(metrics),
        "test_prediction_sha256": payload["test_prediction_sha256"],
    }
    store.append_metric({"mode": "array_deployment", **report})
    print(
        f"array {array_label} {state_label} {population_role} "
        f"seed={endpoint_seed} test={100.0*float(metrics['student_accuracy']):.2f}%",
        flush=True,
    )
    del plant
    gc.collect()
    torch.cuda.empty_cache()
    return report, path


def _repeat_summary(reports: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    selected = tuple(reports)
    if not selected:
        raise ValueError("Expected at least one deployment repeat.")
    accuracies = tuple(float(item["test"]["student_accuracy"]) for item in selected)
    divergences = tuple(float(item["test"]["kl_teacher_student"]) for item in selected)
    return {
        "repeats": list(selected),
        "mean_accuracy": fmean(accuracies),
        "minimum_accuracy": min(accuracies),
        "maximum_accuracy": max(accuracies),
        "population_std_accuracy": pstdev(accuracies),
        "mean_kl_teacher_student": fmean(divergences),
    }


def _evaluate_array_state(
    *,
    array: Any,
    pair: Any,
    templates: Any,
    state_label: str,
    masters: Sequence[torch.Tensor],
    estimator: PopulationStepEstimator,
    protocol: Any,
    stack: Any,
    teacher: Any,
    test_loader: Iterable,
    store: RunStore,
) -> tuple[Mapping[str, Any], list[Path]]:
    target = build_published_persistent_targets(masters, templates, pair)
    fault_only = clamp_targets_to_published_corrupt_singletons(
        target.target_full_conductance,
        pair,
    )
    ideal = _evaluate_full_g(
        stack=stack,
        teacher=teacher,
        loader=test_loader,
        full_g=target.target_full_conductance,
        sample_limit=protocol.execution["evaluation_sample_limit"],
    )
    fault = _evaluate_full_g(
        stack=stack,
        teacher=teacher,
        loader=test_loader,
        full_g=fault_only.realized_full_conductance,
        sample_limit=protocol.execution["evaluation_sample_limit"],
    )
    artifacts: list[Path] = []
    populations = {
        "counterfactual_repaired": pair.repaired,
        "published_corrupt": pair.published,
    }
    programmed = {}
    for population_role, population in populations.items():
        repeats = []
        for endpoint_seed in array.endpoint_seeds:
            report, path = _program_once(
                array_label=array.label,
                state_label=state_label,
                population_role=population_role,
                population=population,
                raw_x_targets=target.target_raw_x,
                endpoint_seed=int(endpoint_seed),
                estimator=estimator,
                protocol=protocol,
                stack=stack,
                teacher=teacher,
                test_loader=test_loader,
                store=store,
            )
            repeats.append(report)
            artifacts.append(path)
        programmed[population_role] = _repeat_summary(repeats)
    repaired_mean = float(programmed["counterfactual_repaired"]["mean_accuracy"])
    corrupt_mean = float(programmed["published_corrupt"]["mean_accuracy"])
    return {
        "state_label": state_label,
        "logical_master_sha256": [
            _tensor_sha256(value) for value in masters
        ],
        "mapping": dict(target.report),
        "ideal_requested_no_write": ideal,
        "deterministic_published_fault_only": {
            "fault_realization": dict(fault_only.report),
            "evaluation": fault,
        },
        "program_verify": programmed,
        "published_corruption_penalty_accuracy": repaired_mean - corrupt_mean,
    }, artifacts


def _array_comparisons(states: Mapping[str, Mapping[str, Any]]) -> Mapping[str, Any]:
    def accuracy(state: str, population: str) -> float:
        return float(states[state]["program_verify"][population]["mean_accuracy"])

    result = {}
    for population in ("counterfactual_repaired", "published_corrupt"):
        raw = accuracy("raw_relu", population)
        continuous = accuracy("population_hwa_continuous", population)
        quantized = accuracy("population_hwa_one_delta_qat", population)
        result[population] = {
            "raw_relu_mean_accuracy": raw,
            "continuous_hwa_mean_accuracy": continuous,
            "one_delta_qat_hwa_mean_accuracy": quantized,
            "continuous_hwa_minus_raw_relu": continuous - raw,
            "one_delta_qat_hwa_minus_raw_relu": quantized - raw,
            "one_delta_qat_minus_continuous_hwa": quantized - continuous,
        }
    result["corruption_penalty"] = {
        state: float(value["published_corruption_penalty_accuracy"])
        for state, value in states.items()
    }
    return result


def run_train(request: "TrainRequest") -> int:
    """Execute the raw-ReLU A write, population HWA, then B--D writes."""

    spec = request.spec
    if not isinstance(spec, PopulationHwaCrossArrayTrainSpec):
        raise TypeError("Expected a population-HWA cross-array train spec.")
    required = {
        "teacher_weights": request.teacher_weights,
        "device_model": request.device_model,
        "device_data": request.device_data,
    }
    missing = [name for name, value in required.items() if value is None]
    if missing:
        raise ValueError(
            "Population-HWA A--D execution requires explicit "
            f"{', '.join('--' + name.replace('_', '-') for name in missing)}."
        )
    for name in ("weights", "base_weights", "resume"):
        if getattr(request, name) is not None:
            raise ValueError(f"Population-HWA A--D execution does not accept --{name}.")
    assert request.teacher_weights is not None
    assert request.device_model is not None
    assert request.device_data is not None
    teacher_path = request.teacher_weights.expanduser().resolve()
    protocol = spec.protocol
    if sha256_file(teacher_path) != protocol.expected_teacher_weights_sha256:
        raise ValueError("Frozen ReLU teacher/source checkpoint SHA-256 mismatch.")
    (
        endpoint_model,
        endpoint_digest,
        endpoint_path,
        estimator,
        estimator_digest,
        estimator_path,
        endpoint_metadata,
    ) = _load_hwa_inputs(request.device_model, request.device_data)
    sampler_python = _sampler_python()
    store = RunStore.create(
        output_root=request.output_dir,
        experiment_id=spec.experiment_id,
        resolved_config=to_plain_data(spec),
        command=request.command,
        repo_root=_ROOT,
        input_artifacts=(
            _input("teacher_weights", teacher_path),
            _input("raw_active_accepted_endpoint_model", endpoint_path),
            _input("raw_active_step_estimator", estimator_path),
            _input("aihwkit_python", sampler_python),
        ),
        resume_capability="unsupported",
    )
    try:
        if spec.student.runtime.device != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("Population-HWA A--D execution requires local CUDA.")
        torch.manual_seed(spec.student.runtime.seed)
        torch.cuda.manual_seed_all(spec.student.runtime.seed)
        device = torch.device("cuda")
        base_loaders = build_mnist_loaders(
            spec.student.data,
            data_seed=spec.student.runtime.data_seed,
            calibration_examples=spec.student.mapping.calibration_examples,
            calibration_batch_size=spec.student.mapping.calibration_batch_size,
        )
        teacher, teacher_metadata = _load_teacher(
            teacher_path,
            device=device,
            spec=spec.student,
        )
        stack = build_student_stack(spec.student, enable_measured=False)
        stack.cost.gain = float(protocol.deployment["fixed_logit_gain"])
        masters, source_absmax, source_weight_hashes = _initial_logical_masters(
            teacher,
            device=device,
        )
        initial_master = _snapshot_masters(masters)
        global_initial = _evaluate_global_clean(
            stack=stack,
            teacher=teacher,
            loader=base_loaders.test,
            masters=initial_master,
            quantization_spacing_raw_x=None,
            sample_limit=protocol.execution["evaluation_sample_limit"],
        )
        artifact_root = store.run_dir / "artifacts"
        generated_artifacts: list[Mapping[str, Any]] = []

        # Array A is the pre-HWA baseline only.  Nothing from this assignment
        # enters either HWA arm or the raw-active population kernel.
        array_a = protocol.arrays[0]
        pair_a, pair_a_artifacts = _sample_and_save_pair(
            bindings=stack.bundle.catalog.trainable,
            assignment_seed=array_a.assignment_seed,
            role="array_A_pre_hwa",
            sampler=sampler_python,
            artifact_root=artifact_root,
        )
        generated_artifacts.extend(pair_a_artifacts)
        templates_a = build_repaired_mapping_templates(
            initial_master,
            pair_a,
            level_spacing_raw_x=float(protocol.deployment["level_spacing_raw_x"]),
        )
        raw_a_report, raw_a_artifacts = _evaluate_array_state(
            array=array_a,
            pair=pair_a,
            templates=templates_a,
            state_label="raw_relu",
            masters=initial_master,
            estimator=estimator,
            protocol=protocol,
            stack=stack,
            teacher=teacher,
            test_loader=base_loaders.test,
            store=store,
        )
        generated_artifacts.extend(
            {"path": str(path), "kind": "array_A_raw_relu_deployment"}
            for path in raw_a_artifacts
        )

        hwa_states: dict[str, tuple[torch.Tensor, ...]] = {}
        hwa_reports: dict[str, Mapping[str, Any]] = {}
        checkpoint_paths: list[Path] = []
        for arm in protocol.hwa_arms:
            # Rebuilding the loaders with the same data seed makes the two
            # HWA interventions consume the same shuffled training cohort.
            arm_loaders = build_mnist_loaders(
                spec.student.data,
                data_seed=spec.student.runtime.data_seed,
                calibration_examples=spec.student.mapping.calibration_examples,
                calibration_batch_size=spec.student.mapping.calibration_batch_size,
            )
            final_master, report, checkpoint = _train_population_hwa_arm(
                arm=arm,
                initial_master=initial_master,
                stack=stack,
                teacher=teacher,
                loaders=arm_loaders,
                model=endpoint_model,
                endpoint_digest=endpoint_digest,
                protocol=protocol,
                checkpoint_dir=store.run_dir / "checkpoints",
                store=store,
            )
            hwa_states[arm.arm_id] = final_master
            hwa_reports[arm.arm_id] = report
            checkpoint_paths.append(checkpoint)
            generated_artifacts.append(
                {"path": str(checkpoint), "kind": f"{arm.arm_id}_fixed_final_master"}
            )
        frozen_checkpoint_hashes = {
            path.stem: sha256_file(path) for path in checkpoint_paths
        }

        states: dict[str, tuple[torch.Tensor, ...]] = {
            "raw_relu": initial_master,
            **hwa_states,
        }
        arrays: dict[str, Any] = {
            "A": {
                "assignment": dict(pair_a.report),
                "published_population": _population_report(pair_a.published),
                "repaired_population": _population_report(pair_a.repaired),
                "opened_before_hwa": True,
                "consumed_by_hwa": False,
                "states": {"raw_relu": raw_a_report},
            }
        }
        # HWA-final Array-A writes are diagnostics only and occur after both
        # fixed-final checkpoints exist.
        for state_label, state_master in hwa_states.items():
            report, paths = _evaluate_array_state(
                array=array_a,
                pair=pair_a,
                templates=templates_a,
                state_label=state_label,
                masters=state_master,
                estimator=estimator,
                protocol=protocol,
                stack=stack,
                teacher=teacher,
                test_loader=base_loaders.test,
                store=store,
            )
            arrays["A"]["states"][state_label] = report
            generated_artifacts.extend(
                {"path": str(path), "kind": f"array_A_{state_label}_deployment"}
                for path in paths
            )
        arrays["A"]["comparisons"] = _array_comparisons(arrays["A"]["states"])

        # B--D are not sampled until every HWA checkpoint above is immutable.
        observed_fingerprints = {
            pair_a.published.fingerprint,
            pair_a.repaired.fingerprint,
        }
        for array in protocol.arrays[1:]:
            pair, pair_artifacts = _sample_and_save_pair(
                bindings=stack.bundle.catalog.trainable,
                assignment_seed=array.assignment_seed,
                role=f"array_{array.label}_post_hwa",
                sampler=sampler_python,
                artifact_root=artifact_root,
            )
            generated_artifacts.extend(pair_artifacts)
            if (
                pair.published.fingerprint in observed_fingerprints
                or pair.repaired.fingerprint in observed_fingerprints
            ):
                raise RuntimeError("A--D physical assignment fingerprints are not unique.")
            observed_fingerprints.update(
                (pair.published.fingerprint, pair.repaired.fingerprint)
            )
            templates = build_repaired_mapping_templates(
                initial_master,
                pair,
                level_spacing_raw_x=float(protocol.deployment["level_spacing_raw_x"]),
            )
            state_reports = {}
            for state_label, state_master in states.items():
                report, paths = _evaluate_array_state(
                    array=array,
                    pair=pair,
                    templates=templates,
                    state_label=state_label,
                    masters=state_master,
                    estimator=estimator,
                    protocol=protocol,
                    stack=stack,
                    teacher=teacher,
                    test_loader=base_loaders.test,
                    store=store,
                )
                state_reports[state_label] = report
                generated_artifacts.extend(
                    {
                        "path": str(path),
                        "kind": f"array_{array.label}_{state_label}_deployment",
                    }
                    for path in paths
                )
            arrays[array.label] = {
                "assignment": dict(pair.report),
                "published_population": _population_report(pair.published),
                "repaired_population": _population_report(pair.repaired),
                "opened_after_hwa_checkpoint_hashes": frozen_checkpoint_hashes,
                "states": state_reports,
                "comparisons": _array_comparisons(state_reports),
            }
            del pair
            gc.collect()
            torch.cuda.empty_cache()

        transfer_labels = ("B", "C", "D")
        transfer_summary = {}
        for population in ("counterfactual_repaired", "published_corrupt"):
            for state_label in states:
                key = f"{population}:{state_label}"
                values = [
                    float(
                        arrays[label]["states"][state_label]["program_verify"]
                        [population]["mean_accuracy"]
                    )
                    for label in transfer_labels
                ]
                transfer_summary[key] = {
                    "array_means": dict(zip(transfer_labels, values, strict=True)),
                    "mean_across_B_D": fmean(values),
                    "minimum_across_B_D": min(values),
                    "maximum_across_B_D": max(values),
                }
        corruption_penalties = {
            state_label: {
                label: float(
                    arrays[label]["states"][state_label]
                    ["published_corruption_penalty_accuracy"]
                )
                for label in transfer_labels
            }
            for state_label in states
        }
        for state_label, values in corruption_penalties.items():
            values["mean_B_D"] = fmean(
                float(values[label]) for label in transfer_labels
            )

        summary = {
            "schema": SUMMARY_SCHEMA,
            "schema_version": SUMMARY_SCHEMA_VERSION,
            "status": "complete",
            "evidence_tier": protocol.evidence_tier,
            "source_revision": store.manifest.get("source", {}),
            "sequence": [
                "raw_ReLU_clean_master",
                "independent_physical_write_on_A",
                "array_agnostic_positive_G_population_HWA",
                "freeze_epoch_10_clean_masters",
                "independent_physical_writes_on_B_C_D",
            ],
            "teacher": {
                "path": str(teacher_path),
                "sha256": sha256_file(teacher_path),
                "metadata": teacher_metadata,
                "logical_weight_sha256": list(source_weight_hashes),
                "layer_absmax": list(source_absmax),
            },
            "programming_error_model": {
                "path": str(endpoint_path),
                "sha256": endpoint_digest,
                "coordinate": RAW_ACTIVE_COORDINATE,
                "condition_key": HWA_CONDITION_KEY,
                "metadata": dict(endpoint_metadata),
                "fixed_array_identity": False,
                "accepted_healthy_apparent_endpoints_only": True,
            },
            "step_estimator": {
                "path": str(estimator_path),
                "sha256": estimator_digest,
                "key": HWA_ESTIMATOR_KEY,
                "calibration_partition_only": True,
            },
            "protocol": to_plain_data(protocol),
            "initial_global_clean_test": global_initial,
            "hwa": hwa_reports,
            "frozen_hwa_checkpoint_hashes_before_B_D_open": frozen_checkpoint_hashes,
            "arrays": arrays,
            "transfer_B_D": transfer_summary,
            "published_corruption_penalty_B_D": corruption_penalties,
            "invariants": {
                "all_circuit_weights_are_positive_G_0_2": True,
                "global_HWA_target_coordinate": "x=G/2",
                "array_A_state_or_bounds_consumed_by_HWA": False,
                "arrays_B_D_opened_before_HWA_freeze": False,
                "fresh_programming_error_per_physical_cell_per_minibatch": True,
                "same_HWA_draw_for_forward_and_backward": True,
                "HWA_corrupt_mask": False,
                "deployment_corruption_is_separate_intervention": True,
                "deployment_inference_uses_persistent_full_G": True,
                "deployment_apparent_verify_used_for_inference": False,
                "quantization_is_separate_HWA_arm": True,
            },
            "limitations": [
                "exploratory_noncanonical",
                "model_based_aihwkit_1.1.0_OM_preset_not_raw_measured_trace_replay",
                "single_frozen_ReLU_initializer_and_MNIST_split",
                "A_D_are_four_simulated_assignments_not_fabricated_arrays",
                "counterfactual_repair_uses_simulated_healthy_donors",
                "no_inference_read_noise_retention_drift_or_peripheral_nonideality",
            ],
        }
        summary_path = artifact_root / "scientific_summary.json"
        atomic_write_json(summary_path, summary)
        generated_artifacts.append(
            {"path": str(summary_path), "kind": "scientific_summary"}
        )
        terminal = {
            "evidence_tier": protocol.evidence_tier,
            "profile": protocol.profile,
            "array_A_assignment_seed": int(protocol.arrays[0].assignment_seed),
            "array_B_D_assignment_seeds": [
                int(value.assignment_seed) for value in protocol.arrays[1:]
            ],
            "hwa_fixed_final_epoch": int(protocol.hwa["epochs"]),
            "negative_conductance_count": 0,
            "array_A_consumed_by_hwa": False,
            "arrays_B_D_opened_before_hwa_freeze": False,
            "published_corruption_penalty_B_D": corruption_penalties,
        }
        store.append_metric({"mode": "train_terminal", **terminal})
        store.complete(
            metrics=terminal,
            artifacts=_artifact_records(store, generated_artifacts),
        )
        print(f"run_dir={store.run_dir}", flush=True)
        return 0
    except BaseException as error:
        store.fail(error)
        raise


__all__ = [
    "_array_comparisons",
    "_load_deployment_estimator",
    "_load_hwa_inputs",
    "_one_forward_gradient",
    "_repeat_summary",
    "_train_population_hwa_arm",
    "run_train",
]
