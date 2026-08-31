"""CUDA runtime for staged target-87003 persistent pulse-Adam recovery."""

from __future__ import annotations

import gc
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence, TYPE_CHECKING

import torch

from experiments.artifacts import RunStore, atomic_write_json, sha256_file
from experiments.mnist_relu_drn.components import build_student_stack
from experiments.mnist_relu_drn.ibm_om_baseline_selection_runtime import _input
from experiments.mnist_relu_drn.ibm_om_bounded_codebook_scheme_screen import (
    _evaluate_detailed,
    _tensor_sha256,
)
from experiments.mnist_relu_drn.ibm_om_winsorized_onchip_adam import (
    CLAIM_LABEL,
    EVIDENCE_TIER,
    STATE_AUTHORITY,
    PersistentPulseAdam,
    clone_p0_bundle,
    flatten_physical,
    load_p0_bundle,
    plant_from_p0,
    save_p0_bundle,
    split_flat_physical,
    sync_catalog_from_persistent,
)
from experiments.mnist_relu_drn.ibm_om_baseline_selection import quad_stack
from experiments.mnist_relu_drn.ibm_om_winsorized_qat import (
    WinsorizedQatLayerTemplate,
    load_winsorized_qat_templates,
    quantize_winsorized_logical_master,
)
from experiments.mnist_relu_drn.ibm_om_winsorized_qat_runtime import (
    _load_winsorized_population,
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
SUMMARY_SCHEMA = "ebl.mnist_relu_drn.ibm_om_winsorized_onchip_adam_exploratory"
SUMMARY_SCHEMA_VERSION = 1
NOMINAL_DELTA_X = 0.04745
MULTI_QAT_CHECKPOINT_SCHEMA = (
    "ebl.mnist_relu_drn.ibm_om_winsorized_multi_assignment_qat_logical_master"
)
MULTI_QAT_CHECKPOINT_SCHEMA_VERSION = 1


def _resolve_target_artifacts(protocol: Any) -> Mapping[str, Path]:
    root = _ROOT / "results" / protocol.artifacts.source_exploratory_result_id
    arm = (
        root
        / "runs"
        / (
            "alpha_000_spacing_"
            f"{protocol.spacing_delta_x_multiplier}delta-heldout-87003"
        )
    )
    candidates = []
    if arm.is_dir():
        for run_dir in sorted(item for item in arm.iterdir() if item.is_dir()):
            paths = {
                "mapping": run_dir / "artifacts/heldout/ideal_physical_mapping.npz",
                "population": (
                    run_dir
                    / "artifacts/heldout/winsorized_identity/winsorized_population.npz"
                ),
                "population_receipt": (
                    run_dir
                    / "artifacts/heldout/winsorized_identity/"
                    "winsorized_population.receipt.json"
                ),
                "commissioning": (
                    run_dir
                    / "artifacts/heldout/winsorized_identity/"
                    "winsorized_reset_commissioning.npz"
                ),
                "joint_assignment": run_dir / "artifacts/heldout/joint_assignment.npz",
                "scientific_summary": run_dir / "artifacts/scientific_summary.json",
            }
            if all(path.is_file() for path in paths.values()):
                candidates.append(paths)
    matched = [
        paths
        for paths in candidates
        if sha256_file(paths["mapping"])
        == protocol.artifacts.target_mapping_sha256
        and sha256_file(paths["population"])
        == protocol.artifacts.target_population_sha256
        and sha256_file(paths["commissioning"])
        == protocol.artifacts.target_commissioning_sha256
        and sha256_file(paths["joint_assignment"])
        == protocol.artifacts.target_joint_assignment_sha256
    ]
    if len(matched) != 1:
        raise RuntimeError(
            "Expected exactly one target-87003 predecessor artifact bundle "
            f"matching the declared hashes; matches={len(matched)}."
        )
    return matched[0]


def _load_qat_logical_master(
    path: Path,
    *,
    templates: Sequence[WinsorizedQatLayerTemplate],
    spacing_delta_x_multiplier: int,
    fixed_logit_gain: float,
    teacher_sha256: str,
    teacher: Any,
    source_contract: Any,
    device: torch.device,
) -> tuple[tuple[torch.Tensor, ...], tuple[str, str]]:
    source = path.expanduser().resolve()
    try:
        payload = torch.load(source, map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        raise ValueError("Expected a readable weights-only epoch-10 QAT checkpoint.") from error
    fields = {
        "schema",
        "schema_version",
        "epoch",
        "evidence_tier",
        "teacher_sha256",
        "spacing_delta_x_multiplier",
        "fixed_logit_gain",
        "source_absmax",
        "training_assignment_seeds",
        "training_population_fingerprints",
        "development_assignment_seed",
        "development_population_fingerprint",
        "assignment_cycle",
        "global_minibatch_ordinal",
        "logical_master",
        "logical_master_sha256",
        "optimizer_state",
        "development_validation",
    }
    if not isinstance(payload, Mapping) or set(payload) != fields:
        raise ValueError("Expected exact epoch-10 Winsorized QAT checkpoint fields.")
    if (
        payload["schema"] != MULTI_QAT_CHECKPOINT_SCHEMA
        or payload["schema_version"] != MULTI_QAT_CHECKPOINT_SCHEMA_VERSION
        or payload["epoch"] != 10
        or payload["evidence_tier"] != EVIDENCE_TIER
        or payload["teacher_sha256"] != teacher_sha256
        or payload["spacing_delta_x_multiplier"] != spacing_delta_x_multiplier
        or payload["fixed_logit_gain"] != fixed_logit_gain
    ):
        raise ValueError(
            "Multi-assignment QAT checkpoint does not match the selected "
            "on-chip protocol."
        )
    expected_fingerprints = {
        str(seed): fingerprint
        for seed, fingerprint in zip(
            source_contract.qat_training_assignment_seeds,
            source_contract.qat_training_population_fingerprints,
        )
    }
    if (
        payload["training_assignment_seeds"]
        != list(source_contract.qat_training_assignment_seeds)
        or payload["training_population_fingerprints"] != expected_fingerprints
        or payload["development_assignment_seed"]
        != source_contract.qat_development_assignment_seed
        or payload["development_population_fingerprint"]
        != source_contract.qat_development_population_fingerprint
        or payload["assignment_cycle"] != source_contract.qat_assignment_cycle
        or isinstance(payload["global_minibatch_ordinal"], bool)
        or not isinstance(payload["global_minibatch_ordinal"], int)
        or payload["global_minibatch_ordinal"]
        != source_contract.qat_expected_global_minibatch_ordinal
        or not isinstance(payload["optimizer_state"], Mapping)
        or not isinstance(payload["development_validation"], Mapping)
    ):
        raise ValueError(
            "Multi-assignment QAT checkpoint assignment/bundle contract mismatch."
        )
    teacher_absmax = tuple(
        float(value.detach().abs().max().item()) for value in teacher.parameters()
    )
    source_absmax = payload["source_absmax"]
    if (
        not isinstance(source_absmax, list)
        or len(source_absmax) != 2
        or tuple(float(value) for value in source_absmax) != teacher_absmax
    ):
        raise ValueError("QAT checkpoint source normalization mismatches the teacher.")
    values = payload["logical_master"]
    hashes = payload["logical_master_sha256"]
    if (
        not isinstance(values, (tuple, list))
        or not isinstance(hashes, (tuple, list))
        or len(values) != 2
        or len(hashes) != 2
    ):
        raise ValueError("Expected two hashed logical masters in the QAT checkpoint.")
    masters = []
    observed = []
    for value, template in zip(values, templates):
        if (
            not isinstance(value, torch.Tensor)
            or value.dtype != torch.float32
            or tuple(value.shape) != template.logical_shape
            or not bool(torch.all(torch.isfinite(value)))
            or bool(torch.any(value < -1.0))
            or bool(torch.any(value > 1.0))
        ):
            raise ValueError("Expected finite bounded float32 QAT logical masters.")
        clean = value.detach().cpu().contiguous().clone()
        observed.append(_tensor_sha256(clean))
        masters.append(clean.to(device))
    if tuple(observed) != tuple(hashes):
        raise ValueError("QAT logical-master hashes do not match checkpoint tensors.")
    return tuple(masters), (observed[0], observed[1])


def _programming_payload(result: Any) -> Mapping[str, Any]:
    return {
        "accepted": result.accepted.detach().cpu().clone(),
        "nonfinite": result.nonfinite.detach().cpu().clone(),
        "budget_exhausted": result.budget_exhausted.detach().cpu().clone(),
        "apparent_endpoint": result.apparent_endpoint.detach().cpu().clone(),
        "set_count": result.set_count.detach().cpu().clone(),
        "reset_count": result.reset_count.detach().cpu().clone(),
        "total_pulses": result.total_pulses.detach().cpu().clone(),
        "verify_count": result.verify_count.detach().cpu().clone(),
        "reversals": result.reversals.detach().cpu().clone(),
    }


def _create_p0(
    *,
    checkpoint_sha256: str,
    teacher_sha256: str,
    logical_master_sha256: Sequence[str],
    masters: Sequence[torch.Tensor],
    templates: Sequence[WinsorizedQatLayerTemplate],
    population: Any,
    stack: Any,
    teacher: Any,
    validation_loader: Iterable,
    protocol: Any,
    device: torch.device,
) -> dict[str, Any]:
    view = quantize_winsorized_logical_master(masters, templates)
    target_raw_x = flatten_physical(view.raw_x_targets).to(device=device)
    requested_index = flatten_physical(view.requested_index).to(torch.int64)
    pulse_population = array_population_as_pulse_population(population)
    lower = ((pulse_population.min_bound + 1.0) / 2.0).to(device)
    upper = ((pulse_population.max_bound + 1.0) / 2.0).to(device)
    if (
        target_raw_x.shape != lower.shape
        or bool(torch.any(target_raw_x < lower - 1e-6))
        or bool(torch.any(target_raw_x > upper + 1e-6))
    ):
        raise RuntimeError("Selected QAT deployment contains unsupported target cells.")
    seeds = matched_trajectory_seeds(
        population, endpoint_seed=protocol.p0.endpoint_seed
    )
    plant = IbmReramRawActivePlant(
        pulse_population,
        seeds=seeds,
        device=device,
        maximum_random_draws=protocol.p0.maximum_random_draws,
    )
    plant.initialize_at_sampled_lower()
    programming = run_program_verify(
        plant.controller_port(),
        targets=target_raw_x,
        tolerance=protocol.p0.verify_tolerance_raw_x,
        maximum_pulses=protocol.p0.maximum_program_pulses,
        settings=ControllerSettings(kind="one_pulse"),
    )
    sync_catalog_from_persistent(
        stack.bundle.catalog,
        plant,
        binding_shapes=population.binding_shapes,
    )
    validation, prediction = _evaluate_detailed(
        stack, teacher, validation_loader, sample_limit=None
    )
    continuation = plant.state_dict()
    payload = {
        "schema": "ebl.mnist_relu_drn.ibm_om_winsorized_onchip_adam_p0",
        "schema_version": 1,
        "evidence_tier": EVIDENCE_TIER,
        "claim_label": CLAIM_LABEL,
        "state_authority": STATE_AUTHORITY,
        "assignment_seed": int(protocol.p0.assignment_seed),
        "endpoint_seed": int(protocol.p0.endpoint_seed),
        "spacing_delta_x_multiplier": int(protocol.spacing_delta_x_multiplier),
        "qat_checkpoint_sha256": checkpoint_sha256,
        "teacher_sha256": teacher_sha256,
        "population_sha256": protocol.artifacts.target_population_sha256,
        "population_fingerprint": protocol.artifacts.target_population_fingerprint,
        "mapping_sha256": protocol.artifacts.target_mapping_sha256,
        "joint_assignment_sha256": protocol.artifacts.target_joint_assignment_sha256,
        "commissioning_sha256": protocol.artifacts.target_commissioning_sha256,
        "binding_keys": list(population.binding_keys),
        "binding_shapes": [list(shape) for shape in population.binding_shapes],
        "maximum_program_pulses": int(protocol.p0.maximum_program_pulses),
        "recovery_pulse_cap": int(protocol.p0.recovery_pulse_cap),
        "maximum_random_draws": int(protocol.p0.maximum_random_draws),
        "pulse_selection_seed": int(protocol.recovery.pulse_selection_seed),
        "target_raw_x": target_raw_x.detach().cpu().to(torch.float32).clone(),
        "requested_index": requested_index.detach().cpu().clone(),
        "logical_master_sha256": list(logical_master_sha256),
        "programming": _programming_payload(programming),
        "continuation_state": continuation,
        "validation": dict(validation),
        "validation_prediction_sha256": _tensor_sha256(
            prediction.to(torch.int64)
        ),
        "no_remap_after_p0": True,
        "apparent_endpoint_applied_to_drn": False,
    }
    del plant
    gc.collect()
    torch.cuda.empty_cache()
    return payload


def _validate_loaded_p0(payload: Mapping[str, Any], protocol: Any, teacher_sha: str) -> None:
    expected = {
        "assignment_seed": protocol.p0.assignment_seed,
        "endpoint_seed": protocol.p0.endpoint_seed,
        "spacing_delta_x_multiplier": protocol.spacing_delta_x_multiplier,
        "teacher_sha256": teacher_sha,
        "population_sha256": protocol.artifacts.target_population_sha256,
        "population_fingerprint": protocol.artifacts.target_population_fingerprint,
        "mapping_sha256": protocol.artifacts.target_mapping_sha256,
        "joint_assignment_sha256": protocol.artifacts.target_joint_assignment_sha256,
        "commissioning_sha256": protocol.artifacts.target_commissioning_sha256,
        "recovery_pulse_cap": protocol.p0.recovery_pulse_cap,
        "maximum_random_draws": protocol.p0.maximum_random_draws,
        "pulse_selection_seed": protocol.recovery.pulse_selection_seed,
    }
    mismatched = {
        name: (payload.get(name), value)
        for name, value in expected.items()
        if payload.get(name) != value
    }
    if mismatched:
        raise ValueError(f"P0 input does not match the selected protocol: {mismatched!r}.")


def _delta_summary(value: torch.Tensor) -> Mapping[str, float]:
    flat = value.detach().to(device="cpu", dtype=torch.float64).reshape(-1)
    return {
        "bias": float(flat.mean().item()),
        "mae": float(flat.abs().mean().item()),
        "rms": float(flat.square().mean().sqrt().item()),
        "maximum_absolute": float(flat.abs().max().item()),
    }


def _p0_programming_summary(p0: Mapping[str, Any]) -> Mapping[str, Any]:
    programming = p0["programming"]
    total = programming["total_pulses"].to(torch.float64)
    return {
        "cells": int(total.numel()),
        "apparent_accepted": int(programming["accepted"].sum().item()),
        "budget_exhausted": int(programming["budget_exhausted"].sum().item()),
        "set_pulses": int(programming["set_count"].sum().item()),
        "reset_pulses": int(programming["reset_count"].sum().item()),
        "mean_total_pulses_per_cell": float(total.mean().item()),
        "maximum_total_pulses_per_cell": int(total.max().item()),
        "apparent_endpoint_applied_to_drn": False,
    }


def _physical_drift_report(
    *,
    p0: Mapping[str, Any],
    plant: IbmReramRawActivePlant,
    pulse_population: Any,
    binding_shapes: Sequence[Sequence[int]],
) -> Mapping[str, Any]:
    """Separate full-G load drift from signed quad-contrast drift."""

    p0_native = p0["continuation_state"]["persistent"].detach().cpu()
    final_native = plant.persistent.detach().cpu()
    p0_full_g = split_flat_physical(p0_native + 1.0, binding_shapes)
    final_full_g = split_flat_physical(final_native + 1.0, binding_shapes)
    layers = []
    all_load = []
    all_contrast = []
    sign = torch.tensor((1.0, -1.0, -1.0, 1.0), dtype=torch.float32)
    for layer_index, (before, after, layout) in enumerate(
        zip(p0_full_g, final_full_g, ("halves", "paired"))
    ):
        before_quad = quad_stack(before, layout=layout)
        after_quad = quad_stack(after, layout=layout)
        load_delta = after_quad.mean(dim=-1) - before_quad.mean(dim=-1)
        contrast_delta = (
            ((after_quad - before_quad) * sign).sum(dim=-1) / 2.0
        )
        all_load.append(load_delta.reshape(-1))
        all_contrast.append(contrast_delta.reshape(-1))
        layers.append(
            {
                "layer": layer_index,
                "layout": layout,
                "quad_mean_loading_delta_full_G": _delta_summary(load_delta),
                "signed_contrast_delta_full_G": _delta_summary(contrast_delta),
            }
        )
    lower = pulse_population.min_bound.detach().cpu()
    upper = pulse_population.max_bound.detach().cpu()
    tolerance = 2e-6
    return {
        "layers": layers,
        "all_quad_mean_loading_delta_full_G": _delta_summary(torch.cat(all_load)),
        "all_signed_contrast_delta_full_G": _delta_summary(torch.cat(all_contrast)),
        "persistent_saturated_lower": int(
            (torch.abs(final_native - lower) <= tolerance).sum().item()
        ),
        "persistent_saturated_upper": int(
            (torch.abs(final_native - upper) <= tolerance).sum().item()
        ),
        "apparent_endpoint_applied_to_drn": False,
    }


def _train_arm(
    *,
    name: str,
    technique: str,
    learning_rate_raw_x: float,
    p0: Mapping[str, Any],
    pulse_population: Any,
    population: Any,
    stack: Any,
    teacher: Any,
    loaders: Any,
    train_generator_state: torch.Tensor,
    protocol: Any,
    store: RunStore,
) -> Mapping[str, Any]:
    plant = plant_from_p0(
        pulse_population,
        clone_p0_bundle(p0),
        device=stack.device,
    )
    sync_catalog_from_persistent(
        stack.bundle.catalog, plant, binding_shapes=population.binding_shapes
    )
    loaders.train_generator.set_state(train_generator_state.clone())
    optimizer = PersistentPulseAdam(
        plant.controller_port(),
        device=stack.device,
        binding_shapes=population.binding_shapes,
        technique=technique,
        learning_rate_raw_x=learning_rate_raw_x,
        nominal_delta_x=NOMINAL_DELTA_X,
        pulse_cap=protocol.p0.recovery_pulse_cap,
        pulse_selection_seed=protocol.recovery.pulse_selection_seed,
        beta1=protocol.recovery.beta1,
        beta2=protocol.recovery.beta2,
        epsilon=protocol.recovery.epsilon,
    )
    examples = 0
    batches = 0
    train_kl = 0.0
    train_correct = 0
    requested_pulses = 0
    applied_pulses = 0
    capped_pulses = 0
    upward_pulses = 0
    downward_pulses = 0
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
        physical_gradients = tuple(stack.differentiator.compute_gradient())
        step = optimizer.step(physical_gradients)
        requested_pulses += step.requested_cells
        applied_pulses += step.pulsed_cells
        capped_pulses += step.capped_cells
        upward_pulses += step.upward_pulses
        downward_pulses += step.downward_pulses
        examples += int(labels.shape[0])
        batches = batch_index + 1
    if batches < 1:
        raise RuntimeError("Expected a pulse-Adam arm to process training batches.")
    sync_catalog_from_persistent(
        stack.bundle.catalog, plant, binding_shapes=population.binding_shapes
    )
    validation, prediction = _evaluate_detailed(
        stack, teacher, loaders.validation, sample_limit=None
    )
    persistent_x = ((plant.persistent + 1.0) / 2.0).detach().cpu()
    p0_persistent_x = (
        (p0["continuation_state"]["persistent"].detach().cpu() + 1.0) / 2.0
    )
    changed = persistent_x != p0_persistent_x
    drift = _physical_drift_report(
        p0=p0,
        plant=plant,
        pulse_population=pulse_population,
        binding_shapes=population.binding_shapes,
    )
    checkpoint_payload = {
        **optimizer.state_dict(),
        "plant_continuation_state": plant.state_dict(),
        "evidence_tier": EVIDENCE_TIER,
        "arm": name,
        "p0_qat_checkpoint_sha256": p0["qat_checkpoint_sha256"],
        "p0_validation_prediction_sha256": p0["validation_prediction_sha256"],
        "validation": dict(validation),
        "validation_prediction_sha256": _tensor_sha256(prediction.to(torch.int64)),
        "no_remap_after_p0": True,
        "apparent_endpoint_applied_to_drn": False,
    }
    checkpoint = atomic_torch_save(
        checkpoint_payload,
        store.run_dir / f"checkpoints/{name}.pt",
    )
    result = {
        "arm": name,
        "technique": technique,
        "learning_rate_raw_x": float(learning_rate_raw_x),
        "train": {
            "examples": examples,
            "batches": batches,
            "kl_teacher_student": train_kl / examples,
            "student_accuracy": train_correct / examples,
        },
        "validation": dict(validation),
        "validation_prediction_sha256": _tensor_sha256(prediction.to(torch.int64)),
        "pulse": {
            "requested": requested_pulses,
            "applied": applied_pulses,
            "capped_requests": capped_pulses,
            "upward": upward_pulses,
            "downward": downward_pulses,
            "changed_cells_from_p0": int(changed.sum().item()),
            "cells_at_cap": int(
                (optimizer.pulse_count.detach().cpu() >= protocol.p0.recovery_pulse_cap)
                .sum()
                .item()
            ),
            "mean_per_cell": float(
                optimizer.pulse_count.to(torch.float64).mean().item()
            ),
            "maximum_per_cell": int(optimizer.pulse_count.max().item()),
        },
        "physical_drift": drift,
        "checkpoint": str(checkpoint),
    }
    store.append_metric({"mode": "train", **result, "checkpoint": str(checkpoint)})
    print(
        f"onchip Adam {name}: validation="
        f"{100.0*float(validation['student_accuracy']):.2f}% "
        f"pulses={applied_pulses}",
        flush=True,
    )
    del optimizer, plant
    gc.collect()
    torch.cuda.empty_cache()
    return result


def _frozen_arm(
    *,
    p0: Mapping[str, Any],
    pulse_population: Any,
    population: Any,
    stack: Any,
    teacher: Any,
    loaders: Any,
) -> Mapping[str, Any]:
    plant = plant_from_p0(pulse_population, clone_p0_bundle(p0), device=stack.device)
    sync_catalog_from_persistent(
        stack.bundle.catalog, plant, binding_shapes=population.binding_shapes
    )
    validation, prediction = _evaluate_detailed(
        stack, teacher, loaders.validation, sample_limit=None
    )
    result = {
        "arm": "frozen_p0",
        "technique": "frozen_no_pulse",
        "learning_rate_raw_x": 0.0,
        "validation": dict(validation),
        "validation_prediction_sha256": _tensor_sha256(prediction.to(torch.int64)),
        "pulse": {"requested": 0, "applied": 0, "changed_cells_from_p0": 0},
        "physical_drift": _physical_drift_report(
            p0=p0,
            plant=plant,
            pulse_population=pulse_population,
            binding_shapes=population.binding_shapes,
        ),
        "checkpoint": None,
    }
    del plant
    gc.collect()
    torch.cuda.empty_cache()
    return result


def _selection_key(result: Mapping[str, Any]) -> tuple[int, float, int, float]:
    validation = result["validation"]
    return (
        int(validation["student_correct"]),
        -float(validation["kl_teacher_student"]),
        -int(result["pulse"]["applied"]),
        -float(result["learning_rate_raw_x"]),
    )


def run_train(request: "TrainRequest") -> int:
    from experiments.mnist_relu_drn.ibm_om_winsorized_onchip_adam_config import (
        WinsorizedOnchipAdamTrainSpec,
    )

    spec = request.spec
    if not isinstance(spec, WinsorizedOnchipAdamTrainSpec):
        raise TypeError("Expected the dedicated Winsorized on-chip Adam train spec.")
    if request.teacher_weights is None:
        raise ValueError("Expected --teacher-weights for pulse-Adam recovery.")
    if request.resume is not None or request.device_data is not None or request.device_model is not None:
        raise ValueError("Pulse-Adam recovery rejects --resume and device inputs.")
    if (request.weights is None) == (request.base_weights is None):
        raise ValueError(
            "Expected exactly one explicit source: --weights epoch-10 QAT "
            "checkpoint to create P0, or --base-weights exact P0 bundle to replay it."
        )
    protocol = spec.protocol
    teacher_path = request.teacher_weights.expanduser().resolve()
    teacher_sha = sha256_file(teacher_path)
    if teacher_sha != protocol.expected_teacher_weights_sha256:
        raise ValueError("Frozen teacher checkpoint SHA-256 mismatch.")
    source_paths = _resolve_target_artifacts(protocol)
    input_path = (request.weights or request.base_weights).expanduser().resolve()
    input_role = "epoch_10_qat_checkpoint" if request.weights is not None else "exact_p0_bundle"
    store = RunStore.create(
        output_root=request.output_dir,
        experiment_id=spec.experiment_id,
        resolved_config=to_plain_data(spec),
        command=request.command,
        repo_root=_ROOT,
        input_artifacts=(
            _input("teacher_weights", teacher_path),
            _input(input_role, input_path),
            *tuple(_input(f"target_87003_{name}", path) for name, path in source_paths.items()),
        ),
        resume_capability="unsupported",
    )
    try:
        torch.manual_seed(spec.student.runtime.seed)
        torch.cuda.manual_seed_all(spec.student.runtime.seed)
        device = torch.device(spec.student.runtime.device)
        if device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("Winsorized persistent pulse-Adam requires CUDA.")
        loaders = build_mnist_loaders(
            spec.student.data,
            data_seed=spec.student.runtime.data_seed,
            calibration_examples=spec.student.mapping.calibration_examples,
            calibration_batch_size=spec.student.mapping.calibration_batch_size,
        )
        teacher, _teacher_metadata = _load_teacher(
            teacher_path, device=device, spec=spec.student
        )
        stack = build_student_stack(spec.student, enable_measured=False)
        stack.cost.gain = float(protocol.fixed_logit_gain)
        spacing_raw_x = protocol.spacing_delta_x_multiplier * NOMINAL_DELTA_X
        templates_cpu = load_winsorized_qat_templates(
            source_paths["mapping"], level_spacing_raw_x=spacing_raw_x
        )
        templates = tuple(template.to(device) for template in templates_cpu)
        population = _load_winsorized_population(
            source_paths["population"],
            source_paths["population_receipt"],
            expected_sha256=protocol.artifacts.target_population_sha256,
            expected_fingerprint=protocol.artifacts.target_population_fingerprint,
            expected_assignment_seed=protocol.p0.assignment_seed,
            source_joint_hardware_instance_id=(
                protocol.artifacts.target_hardware_instance_id
            ),
        )
        pulse_population = array_population_as_pulse_population(population)
        if request.weights is not None:
            checkpoint_sha = sha256_file(input_path)
            masters, logical_hashes = _load_qat_logical_master(
                input_path,
                templates=templates_cpu,
                spacing_delta_x_multiplier=protocol.spacing_delta_x_multiplier,
                fixed_logit_gain=protocol.fixed_logit_gain,
                teacher_sha256=teacher_sha,
                teacher=teacher,
                source_contract=protocol.artifacts,
                device=device,
            )
            p0 = _create_p0(
                checkpoint_sha256=checkpoint_sha,
                teacher_sha256=teacher_sha,
                logical_master_sha256=logical_hashes,
                masters=masters,
                templates=templates,
                population=population,
                stack=stack,
                teacher=teacher,
                validation_loader=loaders.validation,
                protocol=protocol,
                device=device,
            )
            p0_path = save_p0_bundle(p0, store.run_dir / "artifacts/p0.pt")
        else:
            p0 = load_p0_bundle(input_path)
            _validate_loaded_p0(p0, protocol, teacher_sha)
            p0_path = save_p0_bundle(p0, store.run_dir / "artifacts/p0.pt")
        _validate_loaded_p0(p0, protocol, teacher_sha)

        train_generator_state = loaders.train_generator.get_state().clone()
        frozen = _frozen_arm(
            p0=p0,
            pulse_population=pulse_population,
            population=population,
            stack=stack,
            teacher=teacher,
            loaders=loaders,
        )
        store.append_metric({"mode": "frozen_control", **frozen})
        center = protocol.recovery.learning_rates_raw_x[1]
        stage1 = []
        for technique in protocol.recovery.techniques:
            stage1.append(
                _train_arm(
                    name=f"stage1_{technique}_lr_{center:.0e}",
                    technique=technique,
                    learning_rate_raw_x=center,
                    p0=p0,
                    pulse_population=pulse_population,
                    population=population,
                    stack=stack,
                    teacher=teacher,
                    loaders=loaders,
                    train_generator_state=train_generator_state,
                    protocol=protocol,
                    store=store,
                )
            )
        selected_technique = max(stage1, key=_selection_key)["technique"]
        stage2 = []
        for rate in (
            protocol.recovery.learning_rates_raw_x[0],
            protocol.recovery.learning_rates_raw_x[2],
        ):
            stage2.append(
                _train_arm(
                    name=f"stage2_{selected_technique}_lr_{rate:.0e}",
                    technique=selected_technique,
                    learning_rate_raw_x=rate,
                    p0=p0,
                    pulse_population=pulse_population,
                    population=population,
                    stack=stack,
                    teacher=teacher,
                    loaders=loaders,
                    train_generator_state=train_generator_state,
                    protocol=protocol,
                    store=store,
                )
        )
        candidates = [item for item in (*stage1, *stage2) if item["technique"] == selected_technique]
        selected = max(candidates, key=_selection_key)
        # Test labels are opened only after both technique and LR are fixed.
        frozen_test_plant = plant_from_p0(
            pulse_population, clone_p0_bundle(p0), device=device
        )
        sync_catalog_from_persistent(
            stack.bundle.catalog,
            frozen_test_plant,
            binding_shapes=population.binding_shapes,
        )
        frozen_test_metrics, frozen_test_prediction = _evaluate_detailed(
            stack, teacher, loaders.test, sample_limit=None
        )
        del frozen_test_plant
        gc.collect()
        torch.cuda.empty_cache()
        recovery_payload = torch.load(
            Path(selected["checkpoint"]), map_location="cpu", weights_only=True
        )
        selected_plant = IbmReramRawActivePlant(
            pulse_population,
            seeds=tuple(
                int(seed)
                for seed in recovery_payload["plant_continuation_state"]["seeds"]
            ),
            device=device,
            maximum_random_draws=protocol.p0.maximum_random_draws,
        )
        selected_plant.load_state_dict(recovery_payload["plant_continuation_state"])
        sync_catalog_from_persistent(
            stack.bundle.catalog,
            selected_plant,
            binding_shapes=population.binding_shapes,
        )
        test_metrics, test_prediction = _evaluate_detailed(
            stack, teacher, loaders.test, sample_limit=None
        )
        summary = {
            "schema": SUMMARY_SCHEMA,
            "schema_version": SUMMARY_SCHEMA_VERSION,
            "evidence_tier": EVIDENCE_TIER,
            "claim_label": CLAIM_LABEL,
            "state_authority": STATE_AUTHORITY,
            "spacing_delta_x_multiplier": protocol.spacing_delta_x_multiplier,
            "target_assignment_seed": protocol.p0.assignment_seed,
            "p0_endpoint_seed": protocol.p0.endpoint_seed,
            "p0": {
                "path": str(p0_path),
                "sha256": sha256_file(p0_path),
                "qat_checkpoint_sha256": p0["qat_checkpoint_sha256"],
                "validation": p0["validation"],
                "programming": _p0_programming_summary(p0),
                "physical_state": frozen["physical_drift"],
                "shared_exact_seed_for_every_arm": True,
            },
            "screen": {
                "frozen": frozen,
                "stage1": stage1,
                "stage1_selected_technique": selected_technique,
                "stage2": stage2,
                "selected_arm": selected["arm"],
                "selection_split": protocol.recovery.selection_split,
                "test_used_for_selection": False,
            },
            "selected_test": {
                "metrics": test_metrics,
                "prediction_sha256": _tensor_sha256(test_prediction.to(torch.int64)),
            },
            "frozen_p0_test": {
                "metrics": frozen_test_metrics,
                "prediction_sha256": _tensor_sha256(
                    frozen_test_prediction.to(torch.int64)
                ),
            },
            "limitations": [
                "exploratory_noncanonical_single_target_assignment_87003",
                "digital_teacher_KL_and_BPTT_not_fully_on_chip",
                "digital_Adam_moments_and_pulse_selection_RNG",
                "model_based_winsorized_IBM_OM_not_measured_target_chip",
                "post_P0_state_left_original_uniform_codebook_manifold",
            ],
            "no_remap_after_p0": True,
            "apparent_endpoint_applied_to_drn": False,
        }
        summary_path = store.run_dir / "artifacts/scientific_summary.json"
        atomic_write_json(summary_path, summary)
        terminal = {
            "evidence_tier": EVIDENCE_TIER,
            "spacing_delta_x_multiplier": protocol.spacing_delta_x_multiplier,
            "screen_arms": 5,
            "selected_technique": selected_technique,
            "selected_learning_rate_raw_x": selected["learning_rate_raw_x"],
            "frozen_validation_accuracy": frozen["validation"]["student_accuracy"],
            "selected_validation_accuracy": selected["validation"]["student_accuracy"],
            "selected_test_accuracy": test_metrics["student_accuracy"],
            "frozen_test_accuracy": frozen_test_metrics["student_accuracy"],
        }
        artifacts = [
            store.artifact_record(p0_path, kind="exact_p0"),
            store.artifact_record(summary_path, kind="scientific_summary"),
        ]
        artifacts.extend(
            store.artifact_record(Path(item["checkpoint"]), kind="recovery_state")
            for item in (*stage1, *stage2)
        )
        store.complete(metrics=terminal, artifacts=artifacts)
        print(f"run_dir={store.run_dir}", flush=True)
        return 0
    except BaseException as error:
        store.fail(error)
        raise


__all__ = ["run_train"]
