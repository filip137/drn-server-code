"""Runtime for the matched IBM-OM standard-crossbar ReLU experiment."""

from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
import json
import math
import os
from pathlib import Path
from statistics import fmean, pstdev
from typing import Any, Iterable, Mapping, TYPE_CHECKING

import torch
import torch.nn.functional as F

from experiments.artifacts import RunStore, atomic_write_json, content_hash, sha256_file
from experiments.mnist_analog_relu.config import CrossbarTrainSpec
from experiments.mnist_relu.model import BiasFreeReluTeacher
from experiments.mnist_shared import build_mnist_loaders, limited
from experiments.schema import to_plain_data
from model.resistive.builders import ParameterBinding
from model.variable.parameter import DenseWeight
from training.checkpoint import atomic_torch_save, load_named_weights
from training.ibm_om_standard_crossbar import (
    CrossbarTileSpec,
    IbmOmEffectiveCrossbarPlant,
    PulseAdam,
    PulseSGD,
    apply_population_bound_policy,
    build_crossbar_layout,
    build_deterministic_effective_codebook,
    layer_cell_slices,
    local_star_crossbar_step,
    map_logical_weights,
    project_to_nearest_effective_code,
    project_to_nearest_effective_code_device,
    standard_crossbar_logits,
    standard_crossbar_forward_states,
    tensor_sha256,
    validate_population_layout,
)
from training.ibm_reram_hwa import (
    IbmReramArrayPopulation,
    sample_om_array_population_external,
    save_om_array_population,
)
from training.ibm_reram_program_verify import (
    ControllerSettings,
    OM_PRESET,
    PUBLISHED_CORRUPT_PROBABILITY,
    PUBLISHED_CORRUPT_RANGE,
    derive_seed,
    run_program_verify,
)
from training.star_targets import (
    CROSSBAR_STATE_REPRESENTATION,
    StarSourceBinding,
    StarTargetAccumulator,
    StarTargetBinding,
    build_calibration_binding,
    load_star_targets,
    save_star_targets,
)

if TYPE_CHECKING:
    from ebl.cli import TrainRequest


_ROOT = Path(__file__).resolve().parents[2]
_POST_DEPLOYMENT_FAULT_POLICIES = frozenset(
    {"star_local_pulse_sgd", "supervised_ce_pulse_adam"}
)


def _post_deployment_fault_settings(spec: CrossbarTrainSpec) -> Any | None:
    if spec.recovery.policy == "star_local_pulse_sgd":
        return spec.recovery.star
    if spec.recovery.policy == "supervised_ce_pulse_adam":
        return spec.recovery.supervised_bp
    return None


def _input(role: str, path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(
            f"Expected {role} to identify an existing file. Provided value: {str(resolved)!r}."
        )
    return {"role": role, "path": str(resolved), "sha256": sha256_file(resolved)}


def _resolve_repo_path(value: str) -> Path:
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        candidate = _ROOT / candidate
    return candidate.resolve()


def _load_drn_reference(spec: CrossbarTrainSpec) -> tuple[Path, dict[str, Any]]:
    path = _resolve_repo_path(spec.drn_reference.path)
    if sha256_file(path) != spec.drn_reference.sha256:
        raise ValueError("Expected the pinned DRN reference receipt SHA-256 to match.")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("Expected a readable strict DRN reference receipt.") from error
    required = {
        "architecture",
        "assignment_seed",
        "device_model",
        "endpoint_seeds",
        "evidence_tier",
        "metrics",
        "recovery_contract",
        "schema",
        "schema_version",
        "source",
    }
    if (
        not isinstance(value, dict)
        or set(value) != required
        or value["schema"] != "ebl.reference.ibm_om_drn_endpoint_recovery"
        or value["schema_version"] != 1
        or value["architecture"] != spec.drn_reference.architecture
        or int(value["assignment_seed"]) != spec.drn_reference.assignment_seed
        or tuple(value["endpoint_seeds"]) != spec.drn_reference.endpoint_seeds
    ):
        raise ValueError("Expected the DRN receipt to match the resolved comparison cohort.")
    metrics = value.get("metrics")
    if not isinstance(metrics, dict) or set(metrics) != {
        "open_loop_adam",
        "persistent_deployment",
    }:
        raise ValueError("Expected DRN deployment and recovery metrics in the receipt.")
    for name in ("open_loop_adam", "persistent_deployment"):
        row = metrics[name]
        if not isinstance(row, dict) or "per_endpoint_accuracy" not in row:
            raise ValueError("Expected per-endpoint DRN reference accuracy.")
        accuracy = tuple(float(item) for item in row["per_endpoint_accuracy"])
        if len(accuracy) != len(spec.device.endpoint_seeds) or any(
            not math.isfinite(item) or item < 0.0 or item > 1.0 for item in accuracy
        ):
            raise ValueError("Expected valid per-endpoint DRN reference accuracy.")
    recovery = value.get("recovery_contract")
    source = value.get("source")
    device_model = value.get("device_model")
    raw_x_learning_rate = (
        recovery.get("adam_learning_rate_raw_x")
        if isinstance(recovery, dict)
        else None
    )
    if (
        not isinstance(recovery, dict)
        or recovery.get("epochs") != 1
        or recovery.get("batch_size") != spec.data.batch_size
        or recovery.get("data_seed") != spec.runtime.data_seed
        or recovery.get("checkpoint_policy") != "fixed_final_epoch_no_selection"
        or recovery.get("pulse_cap_per_cell") != 64
        or isinstance(raw_x_learning_rate, bool)
        or not isinstance(raw_x_learning_rate, (int, float))
        or not math.isclose(
            float(raw_x_learning_rate),
            3e-5,
            rel_tol=0.0,
            abs_tol=1e-15,
        )
        or not isinstance(source, dict)
        or source.get("teacher_checkpoint_sha256")
        != spec.source.expected_teacher_weights_sha256
        or not math.isclose(
            float(source.get("teacher_accuracy", float("nan"))),
            0.9736,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        or not math.isclose(
            float(source.get("source_pretransfer_accuracy", float("nan"))),
            0.9414,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        or not math.isclose(
            float(source.get("requested_remap_no_write_accuracy", float("nan"))),
            0.9202,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        or not isinstance(device_model, dict)
        or device_model.get("preset") != spec.device.preset
        or device_model.get("evidence_class") != spec.device.evidence_class
    ):
        raise ValueError(
            "Expected the pinned DRN reference to match the teacher, device, "
            "data, checkpoint, and recovery contract."
        )
    return path, value


def _sampling_bindings(layout: Iterable[CrossbarTileSpec]) -> tuple[ParameterBinding, ...]:
    bindings = []
    for tile in layout:
        parameter = DenseWeight(
            (tile.shape[0],),
            (tile.shape[1],),
            gain=1.0,
            device="cpu",
            clamp=False,
        )
        bindings.append(
            ParameterBinding(
                key=tile.key,
                parameter=parameter,
                group="crossbar",
                role="dense_weight",
                trainable=False,
                checkpointed=False,
            )
        )
    return tuple(bindings)


def _sample_population(
    *,
    layout: tuple[CrossbarTileSpec, ...],
    assignment_seed: int,
    corruption_policy: str,
    sampler: Path,
    artifact_root: Path,
    role: str,
    bound_policy: str,
    required_aihwkit_version: str,
) -> tuple[
    IbmReramArrayPopulation,
    dict[str, Any],
    tuple[tuple[Path, str], ...],
]:
    sampled_path = artifact_root / f"{role}_sampled_population.npz"
    sampled_receipt_path = artifact_root / f"{role}_sampled_population.receipt.json"
    sampled, sampling_receipt = sample_om_array_population_external(
        _sampling_bindings(layout),
        assignment_seed=assignment_seed,
        corruption_policy=corruption_policy,
        aihwkit_python=sampler,
        population_path=sampled_path,
        receipt_path=sampled_receipt_path,
    )
    population, bound_report = apply_population_bound_policy(
        sampled,
        policy=bound_policy,
    )
    if population.aihwkit_version != required_aihwkit_version:
        raise RuntimeError(
            "Expected the sampled IBM OM population to match the pinned "
            "AIHWKit version."
        )
    validate_population_layout(population, layout)
    artifacts: tuple[tuple[Path, str], ...] = (
        (sampled_path, "ibm_om_sampled_population"),
        (sampled_receipt_path, "population_sampling_receipt"),
    )
    effective_artifact = None
    if population.fingerprint != sampled.fingerprint:
        effective_path = artifact_root / f"{role}_effective_population.npz"
        effective_receipt_path = artifact_root / f"{role}_effective_population.receipt.json"
        save_om_array_population(effective_path, population)
        effective_receipt = {
            "schema": "ebl.ibm_om_crossbar_effective_population_receipt",
            "schema_version": 1,
            "artifact": effective_path.name,
            "artifact_sha256": sha256_file(effective_path),
            "population_fingerprint": population.fingerprint,
            "source_population_fingerprint": sampled.fingerprint,
            "source_population_sha256": sha256_file(sampled_path),
            "sampling_receipt_sha256": sha256_file(sampled_receipt_path),
            "bound_treatment": bound_report,
        }
        atomic_write_json(effective_receipt_path, effective_receipt)
        artifacts += (
            (effective_path, "ibm_om_effective_population"),
            (effective_receipt_path, "effective_population_receipt"),
        )
        effective_artifact = effective_receipt
    provenance = {
        "sampling": sampling_receipt,
        "bound_treatment": bound_report,
        "effective_artifact": effective_artifact,
    }
    return population, provenance, artifacts


def _load_teacher(
    path: Path,
    *,
    expected_sha256: str,
    device: torch.device,
) -> tuple[BiasFreeReluTeacher, dict[str, Any]]:
    if sha256_file(path) != expected_sha256:
        raise ValueError("Expected the frozen ReLU teacher checkpoint SHA-256 to match.")
    teacher = BiasFreeReluTeacher(device=device)
    loaded = load_named_weights(path, teacher.catalog)
    if loaded.metadata.get("architecture") != "bias_free_relu_784_50_10":
        raise ValueError("Expected the frozen source architecture to be bias_free_relu_784_50_10.")
    return teacher, dict(loaded.metadata)


def _prediction_digest(predictions: list[torch.Tensor]) -> str:
    joined = torch.cat(predictions).to(dtype=torch.int64, device="cpu").contiguous()
    return sha256(joined.numpy().tobytes()).hexdigest()


def _evaluate(
    *,
    effective_state: torch.Tensor,
    digital_scales: tuple[float, float],
    layout: tuple[CrossbarTileSpec, ...],
    teacher: BiasFreeReluTeacher,
    loader: Iterable,
    device: torch.device,
    maximum_batches: int | None,
    sample_limit: int | None,
) -> dict[str, Any]:
    totals = {
        "student_correct": 0,
        "teacher_correct": 0,
        "agreement": 0,
        "kl": 0.0,
        "cross_entropy": 0.0,
        "student_score_squared": 0.0,
        "teacher_score_squared": 0.0,
    }
    examples = 0
    student_predictions: list[torch.Tensor] = []
    teacher_predictions: list[torch.Tensor] = []
    state = effective_state.detach().to(device=device, dtype=torch.float32)
    with torch.no_grad():
        for inputs, labels in limited(loader, maximum_batches):
            if sample_limit is not None:
                remaining = sample_limit - examples
                if remaining <= 0:
                    break
                inputs = inputs[:remaining]
                labels = labels[:remaining]
            inputs = inputs.to(device=device, dtype=torch.float32)
            labels = labels.to(device=device, dtype=torch.long)
            teacher_logits = teacher.logits(inputs)
            student_logits = standard_crossbar_logits(
                inputs,
                state,
                layout,
                digital_scales=digital_scales,
            )
            teacher_log_probability = F.log_softmax(teacher_logits, dim=1)
            teacher_probability = teacher_log_probability.exp()
            student_log_probability = F.log_softmax(student_logits, dim=1)
            student_prediction = student_logits.argmax(dim=1)
            teacher_prediction = teacher_logits.argmax(dim=1)
            totals["student_correct"] += int(student_prediction.eq(labels).sum().item())
            totals["teacher_correct"] += int(teacher_prediction.eq(labels).sum().item())
            totals["agreement"] += int(student_prediction.eq(teacher_prediction).sum().item())
            totals["kl"] += float(
                (teacher_probability * (teacher_log_probability - student_log_probability)).sum().item()
            )
            totals["cross_entropy"] += float(
                F.cross_entropy(student_logits, labels, reduction="sum").item()
            )
            totals["student_score_squared"] += float(student_logits.square().sum().item())
            totals["teacher_score_squared"] += float(teacher_logits.square().sum().item())
            examples += int(labels.shape[0])
            student_predictions.append(student_prediction.detach().cpu())
            teacher_predictions.append(teacher_prediction.detach().cpu())
    if examples == 0:
        raise ValueError("Expected evaluation to process at least one example.")
    score_count = examples * 10
    return {
        "examples": examples,
        "student_correct": totals["student_correct"],
        "student_accuracy": totals["student_correct"] / examples,
        "teacher_correct": totals["teacher_correct"],
        "teacher_accuracy": totals["teacher_correct"] / examples,
        "teacher_agreement": totals["agreement"] / examples,
        "prediction_flips_from_teacher": examples - totals["agreement"],
        "kl_teacher_student": totals["kl"] / examples,
        "cross_entropy": totals["cross_entropy"] / examples,
        "student_score_rms": math.sqrt(totals["student_score_squared"] / score_count),
        "teacher_score_rms": math.sqrt(totals["teacher_score_squared"] / score_count),
        "student_prediction_sha256": _prediction_digest(student_predictions),
        "teacher_prediction_sha256": _prediction_digest(teacher_predictions),
    }


def _evaluate_plant_states(
    *,
    plant: IbmOmEffectiveCrossbarPlant,
    digital_scales: tuple[float, float],
    layout: tuple[CrossbarTileSpec, ...],
    teacher: BiasFreeReluTeacher,
    loader: Iterable,
    device: torch.device,
    maximum_batches: int | None,
    sample_limit: int | None,
) -> dict[str, Any]:
    """Evaluate the AIHWKit-visible state and hidden update state separately."""

    common = {
        "digital_scales": digital_scales,
        "layout": layout,
        "teacher": teacher,
        "loader": loader,
        "device": device,
        "maximum_batches": maximum_batches,
        "sample_limit": sample_limit,
    }
    return {
        "network_forward_state": "apparent_q",
        "hidden_update_state": "persistent_q",
        "apparent_forward": _evaluate(
            effective_state=plant.apparent,
            **common,
        ),
        "persistent_diagnostic": _evaluate(
            effective_state=plant.persistent,
            **common,
        ),
    }


def _evaluate_local_star_state_errors(
    *,
    plant: IbmOmEffectiveCrossbarPlant,
    target_bundle: Any,
    digital_scales: tuple[float, float],
    layout: tuple[CrossbarTileSpec, ...],
    loader: Iterable,
    device: torch.device,
    maximum_batches: int | None,
    sample_limit: int | None,
) -> dict[str, Any]:
    """Measure label-addressed STAR errors without a teacher or backpropagation."""

    hidden_targets = target_bundle.means("hidden_post_relu").to(
        device=device,
        dtype=torch.float32,
    )
    output_targets = target_bundle.means("output_logits").to(
        device=device,
        dtype=torch.float32,
    )
    try:
        hidden_gain, output_gain = (
            float(value) for value in target_bundle.binding.component_gains
        )
    except (AttributeError, TypeError, ValueError) as error:
        raise ValueError("Expected two STAR component gains in the target binding.") from error
    if (
        hidden_targets.ndim != 2
        or output_targets.ndim != 2
        or hidden_targets.shape[0] != output_targets.shape[0]
        or not math.isfinite(hidden_gain)
        or not math.isfinite(output_gain)
        or hidden_gain < 0.0
        or output_gain < 0.0
    ):
        raise ValueError("Expected finite class-indexed STAR targets and gains.")

    totals = {
        "hidden_state_squared": 0.0,
        "output_state_squared": 0.0,
        "hidden_local_error_squared": 0.0,
        "output_local_error_squared": 0.0,
    }
    examples = 0
    state = plant.apparent.detach().to(device=device, dtype=torch.float32)
    with torch.no_grad():
        for inputs, labels in limited(loader, maximum_batches):
            if sample_limit is not None:
                remaining = sample_limit - examples
                if remaining <= 0:
                    break
                inputs = inputs[:remaining]
                labels = labels[:remaining]
            inputs = inputs.to(device=device, dtype=torch.float32)
            labels = labels.to(device=device, dtype=torch.int64)
            if (
                bool(torch.any(labels < 0))
                or bool(torch.any(labels >= hidden_targets.shape[0]))
            ):
                raise ValueError("Expected labels addressable by the STAR target table.")
            states = standard_crossbar_forward_states(
                inputs,
                state,
                layout,
                digital_scales=digital_scales,
            )
            hidden_delta = states.hidden_post_relu - hidden_targets[labels]
            output_delta = states.output_logits - output_targets[labels]
            hidden_local_error = (
                hidden_gain
                * hidden_delta
                * (states.hidden_preactivation > 0).to(hidden_delta.dtype)
            )
            output_local_error = output_gain * output_delta
            totals["hidden_state_squared"] += float(hidden_delta.square().sum().item())
            totals["output_state_squared"] += float(output_delta.square().sum().item())
            totals["hidden_local_error_squared"] += float(
                hidden_local_error.square().sum().item()
            )
            totals["output_local_error_squared"] += float(
                output_local_error.square().sum().item()
            )
            examples += int(labels.shape[0])
    if examples == 0:
        raise ValueError("Expected STAR state-error evaluation to process examples.")
    hidden_values = examples * hidden_targets.shape[1]
    output_values = examples * output_targets.shape[1]
    return {
        "examples": examples,
        "network_forward_state": "apparent_q",
        "teacher_access": False,
        "hidden_gain": hidden_gain,
        "output_gain": output_gain,
        "hidden_state_error_mean_squared_l2": (
            totals["hidden_state_squared"] / examples
        ),
        "output_state_error_mean_squared_l2": (
            totals["output_state_squared"] / examples
        ),
        "hidden_state_error_rms": math.sqrt(
            totals["hidden_state_squared"] / hidden_values
        ),
        "output_state_error_rms": math.sqrt(
            totals["output_state_squared"] / output_values
        ),
        "hidden_local_error_mean_squared_l2": (
            totals["hidden_local_error_squared"] / examples
        ),
        "output_local_error_mean_squared_l2": (
            totals["output_local_error_squared"] / examples
        ),
        "mean_local_objective": (
            0.5
            * (
                hidden_gain * totals["hidden_state_squared"]
                + output_gain * totals["output_state_squared"]
            )
            / examples
        ),
        "apparent_q_sha256": tensor_sha256(plant.apparent),
        "target_semantic_sha256": target_bundle.semantic_sha256,
    }


def _float_summary(value: torch.Tensor) -> dict[str, float]:
    flat = value.detach().cpu().to(torch.float64).reshape(-1)
    if flat.numel() == 0 or not bool(torch.all(torch.isfinite(flat))):
        raise ValueError("Expected a non-empty finite tensor summary input.")
    quantiles = torch.quantile(flat, torch.tensor((0.1, 0.5, 0.9, 0.99), dtype=torch.float64))
    return {
        "minimum": float(flat.min().item()),
        "p10": float(quantiles[0].item()),
        "median": float(quantiles[1].item()),
        "mean": float(flat.mean().item()),
        "p90": float(quantiles[2].item()),
        "p99": float(quantiles[3].item()),
        "maximum": float(flat.max().item()),
        "rms": float(flat.square().mean().sqrt().item()),
    }


def _ordered_labeled_cohort_report(
    *,
    inputs: list[torch.Tensor],
    labels: list[torch.Tensor],
    split: str,
) -> tuple[dict[str, Any], tuple[str, ...]]:
    """Hash an exact ordered update stream and its per-example identities."""

    if not inputs or len(inputs) != len(labels):
        raise ValueError("Expected non-empty matched cohort input and label batches.")
    joined_inputs = torch.cat(inputs).detach().cpu().to(torch.float32).contiguous()
    joined_labels = torch.cat(labels).detach().cpu().to(torch.int64).contiguous()
    if joined_inputs.shape[0] != joined_labels.numel():
        raise ValueError("Expected one cohort label per model input.")
    sample_ids = torch.arange(joined_labels.numel(), dtype=torch.int64)
    binding = build_calibration_binding(
        dataset_id="mnist",
        split=split,
        ordered_sample_ids=sample_ids,
        model_inputs=joined_inputs,
        labels=joined_labels,
    )
    identities = tuple(
        content_hash(
            {
                "model_input_sha256": tensor_sha256(joined_inputs[index]),
                "label": int(joined_labels[index].item()),
            }
        )
        for index in range(joined_labels.numel())
    )
    report = to_plain_data(binding)
    report["ordered_example_identity_sequence_sha256"] = content_hash(
        list(identities)
    )
    report["unique_example_identities"] = len(set(identities))
    return report, identities


def _population_defect_report(
    population: IbmReramArrayPopulation,
) -> dict[str, Any]:
    """Report the sampled defect intervention without inferring it from levels."""

    final = population.corrupt.detach().cpu()
    published = population.published_corrupt.detach().cpu()
    collapsed_zero_step = (
        (torch.abs(population.max_bound.detach().cpu() - population.min_bound.detach().cpu()) <= 1e-12)
        & (population.dwmin_up.detach().cpu() == 0.0)
        & (population.dwmin_down.detach().cpu() == 0.0)
    )
    if not torch.equal(final, collapsed_zero_step):
        raise RuntimeError(
            "Expected final corrupt cells to remain exactly the collapsed "
            "zero-step IBM OM identities."
        )
    if population.corruption_policy == "published" and not torch.equal(
        final, published
    ):
        raise RuntimeError(
            "Expected the published corruption policy to retain every sampled defect."
        )
    cells = population.size
    final_count = int(final.sum().item())
    published_count = int(published.sum().item())
    per_binding = []
    offset = 0
    for key, shape in zip(
        population.binding_keys,
        population.binding_shapes,
        strict=True,
    ):
        binding_cells = math.prod(shape)
        stop = offset + binding_cells
        binding_final = final[offset:stop]
        binding_published = published[offset:stop]
        binding_collapsed = collapsed_zero_step[offset:stop]
        per_binding.append(
            {
                "binding_key": key,
                "binding_shape": list(shape),
                "cell_offset_start": offset,
                "cell_offset_stop": stop,
                "cells": binding_cells,
                "final_corrupt_cells": int(binding_final.sum().item()),
                "published_corrupt_cells": int(binding_published.sum().item()),
                "repaired_published_corrupt_cells": int(
                    (binding_published & ~binding_final).sum().item()
                ),
                "unattributed_final_corrupt_cells": int(
                    (binding_final & ~binding_published).sum().item()
                ),
                "collapsed_zero_step_cells": int(binding_collapsed.sum().item()),
            }
        )
        offset = stop
    if offset != cells:  # pragma: no cover - population validation is upstream
        raise RuntimeError("Expected IBM OM binding offsets to cover every cell.")
    return {
        "corruption_policy": population.corruption_policy,
        "cells": cells,
        "final_corrupt_cells": final_count,
        "published_corrupt_cells": published_count,
        "repaired_published_corrupt_cells": int((published & ~final).sum().item()),
        "unattributed_final_corrupt_cells": int((final & ~published).sum().item()),
        "collapsed_zero_step_cells": int(collapsed_zero_step.sum().item()),
        "final_corrupt_fraction": final_count / cells,
        "published_corrupt_fraction": published_count / cells,
        "per_binding": per_binding,
    }


def _mapping_report(
    *,
    population: IbmReramArrayPopulation,
    requested: torch.Tensor,
    continuous: torch.Tensor,
    codebook: torch.Tensor,
    pulse_indices: torch.Tensor,
    level_counts: torch.Tensor,
) -> dict[str, Any]:
    logical_minimum = population.logical_min.detach().cpu()
    logical_maximum = population.logical_max.detach().cpu()
    below = requested < logical_minimum
    above = requested > logical_maximum
    requested_nonzero = requested != 0.0
    return {
        "requested_sha256": tensor_sha256(requested),
        "continuous_sha256": tensor_sha256(continuous),
        "codebook_sha256": tensor_sha256(codebook),
        "deterministic_projected_target_sha256": tensor_sha256(codebook),
        "requested": _float_summary(requested),
        "continuous": _float_summary(continuous),
        "codebook": _float_summary(codebook),
        "continuous_error": _float_summary(continuous - requested),
        "codebook_error": _float_summary(codebook - requested),
        "requested_below_support": int(below.sum().item()),
        "requested_above_support": int(above.sum().item()),
        "requested_outside_support": int((below | above).sum().item()),
        "requested_sign_counts": {
            "negative": int((requested < 0.0).sum().item()),
            "zero": int((requested == 0.0).sum().item()),
            "positive": int((requested > 0.0).sum().item()),
        },
        "continuous_sign_flips": int(((requested * continuous) < 0.0).sum().item()),
        "continuous_nonzero_to_zero": int(
            (requested_nonzero & (continuous == 0.0)).sum().item()
        ),
        "codebook_sign_flips": int(((requested * codebook) < 0.0).sum().item()),
        "codebook_nonzero_to_zero": int(
            (requested_nonzero & (codebook == 0.0)).sum().item()
        ),
        "pulse_indices": _float_summary(pulse_indices.to(torch.float32)),
        "effective_level_counts": _float_summary(level_counts.to(torch.float32)),
        "one_level_cells": int((level_counts == 1).sum().item()),
        "final_corrupt_one_level_cells": int(
            ((level_counts == 1) & population.corrupt.detach().cpu()).sum().item()
        ),
        "defects": _population_defect_report(population),
        "tie_rule": "lowest_pulse_index",
        "reference_projection_count": 0,
        "reference_reassignment_count": 0,
        "reference_failure_count": 0,
    }


def _program_endpoint(
    *,
    population: IbmReramArrayPopulation,
    requested: torch.Tensor,
    assignment_seed: int,
    endpoint_seed: int,
    maximum_pulses: int,
    tolerance_x: float,
    device: torch.device,
    stream_role: str,
    random_stream_fingerprint: str,
) -> tuple[IbmOmEffectiveCrossbarPlant, dict[str, Any]]:
    generator = torch.Generator(device=device.type)
    generator.manual_seed(
        derive_seed(
            endpoint_seed,
            assignment_seed,
            random_stream_fingerprint,
            stream_role,
        )
    )
    plant = IbmOmEffectiveCrossbarPlant(
        population,
        generator=generator,
        device=device,
    )
    result = run_program_verify(
        plant.controller_port(),
        targets=(requested.to(device=device, dtype=torch.float32) + 1.0) / 2.0,
        tolerance=tolerance_x,
        maximum_pulses=maximum_pulses,
        settings=ControllerSettings(kind="one_pulse"),
    )
    persistent = plant.persistent.detach().cpu()
    apparent = plant.apparent.detach().cpu()
    target = requested.detach().cpu().to(torch.float32)
    logical_minimum = population.logical_min.detach().cpu()
    logical_maximum = population.logical_max.detach().cpu()
    tolerance_q = 2.0 * tolerance_x
    exact_in_support = (target >= logical_minimum) & (target <= logical_maximum)
    verify_window_intersects_support = (
        (target + tolerance_q >= logical_minimum)
        & (target - tolerance_q <= logical_maximum)
    )
    persistent_within_tolerance = torch.abs(persistent - target) <= tolerance_q
    apparent_q = apparent
    final_corrupt = population.corrupt.detach().cpu()
    result_accepted = result.accepted.detach().cpu()
    result_budget_exhausted = result.budget_exhausted.detach().cpu()
    result_nonfinite = result.nonfinite.detach().cpu()
    result_verify_count = result.verify_count.detach().cpu()
    result_total_pulses = result.total_pulses.detach().cpu()
    final_active = persistent + population.reference.detach().cpu()
    final_saturated_lower = torch.isclose(
        final_active,
        population.min_bound.detach().cpu(),
        atol=1e-7,
        rtol=0.0,
    )
    final_saturated_upper = torch.isclose(
        final_active,
        population.max_bound.detach().cpu(),
        atol=1e-7,
        rtol=0.0,
    )
    report = {
        "endpoint_seed": endpoint_seed,
        "random_stream_population_fingerprint": random_stream_fingerprint,
        "persistent_sha256": tensor_sha256(persistent),
        "apparent_sha256": tensor_sha256(apparent),
        "cells": population.size,
        "defects": _population_defect_report(population),
        "exact_target_in_support": int(exact_in_support.sum().item()),
        "exact_target_outside_support": int((~exact_in_support).sum().item()),
        "verify_window_intersects_support": int(
            verify_window_intersects_support.sum().item()
        ),
        "apparent_accepted": int(result.accepted.sum().item()),
        "persistent_within_tolerance": int(persistent_within_tolerance.sum().item()),
        "apparent_accepted_persistent_outside_tolerance": int(
            (result_accepted & ~persistent_within_tolerance).sum().item()
        ),
        "budget_exhausted": int(result.budget_exhausted.sum().item()),
        "nonfinite": int(result.nonfinite.sum().item()),
        "verify_reads": int(result.verify_count.sum().item()),
        "total_programming_pulses": int(result.total_pulses.sum().item()),
        "mean_programming_pulses_per_cell": float(result.total_pulses.float().mean().item()),
        "maximum_programming_pulses_per_cell": int(result.total_pulses.max().item()),
        "direction_reversals": int(result.reversals.sum().item()),
        "target_q": _float_summary(target),
        "apparent_error_q": _float_summary(apparent_q - target),
        "persistent_error_q": _float_summary(persistent - target),
        "verify_tolerance_x": tolerance_x,
        "verify_tolerance_q": tolerance_q,
        "target_clipping": False,
        "network_forward_state": "apparent_q",
        "hidden_update_state": "persistent_q",
        "apparent_endpoint_applied_to_network": True,
        "plant_pulses": plant.pulse_statistics(),
        "final_corrupt_endpoint": {
            "cells": int(final_corrupt.sum().item()),
            "exact_target_in_support": int(
                (final_corrupt & exact_in_support).sum().item()
            ),
            "exact_target_outside_support": int(
                (final_corrupt & ~exact_in_support).sum().item()
            ),
            "verify_window_intersects_support": int(
                (final_corrupt & verify_window_intersects_support).sum().item()
            ),
            "verify_window_disjoint_support": int(
                (final_corrupt & ~verify_window_intersects_support).sum().item()
            ),
            "apparent_accepted": int(
                (final_corrupt & result_accepted).sum().item()
            ),
            "persistent_within_tolerance": int(
                (final_corrupt & persistent_within_tolerance).sum().item()
            ),
            "apparent_accepted_persistent_outside_tolerance": int(
                (
                    final_corrupt
                    & result_accepted
                    & ~persistent_within_tolerance
                ).sum().item()
            ),
            "budget_exhausted": int(
                (final_corrupt & result_budget_exhausted).sum().item()
            ),
            "nonfinite": int((final_corrupt & result_nonfinite).sum().item()),
            "verify_reads": int(result_verify_count[final_corrupt].sum().item()),
            "programming_pulses": int(
                result_total_pulses[final_corrupt].sum().item()
            ),
            "final_saturated_lower": int(
                (final_corrupt & final_saturated_lower).sum().item()
            ),
            "final_saturated_upper": int(
                (final_corrupt & final_saturated_upper).sum().item()
            ),
        },
    }
    return plant, report


def _matched_published_fault_overlay(
    *,
    healthy: IbmReramArrayPopulation,
    published: IbmReramArrayPopulation,
    preset_default_corrupt_devices_prob: float,
    enabled_corrupt_devices_prob: float,
    corrupt_devices_range: float,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Bind a repaired healthy identity to its published stuck-cell companion."""

    if (
        not math.isclose(
            preset_default_corrupt_devices_prob,
            0.0,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        or not math.isclose(
            enabled_corrupt_devices_prob,
            PUBLISHED_CORRUPT_PROBABILITY[OM_PRESET],
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        or not math.isclose(
            corrupt_devices_range,
            PUBLISHED_CORRUPT_RANGE[OM_PRESET],
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    ):
        raise ValueError("Expected the pinned AIHWKit OM corrupt-device settings.")
    if (
        healthy.assignment_seed != published.assignment_seed
        or healthy.binding_keys != published.binding_keys
        or healthy.binding_shapes != published.binding_shapes
        or healthy.binding_sampling_seeds != published.binding_sampling_seeds
        or healthy.size != published.size
        or healthy.nominal_dw_min != published.nominal_dw_min
        or healthy.dw_min_std != published.dw_min_std
        or healthy.write_noise_std != published.write_noise_std
        or healthy.aihwkit_version != published.aihwkit_version
        or healthy.corruption_policy != "counterfactual_repaired"
        or published.corruption_policy != "published"
    ):
        raise ValueError("Expected matched repaired and published IBM OM populations.")
    mask = published.corrupt.detach().cpu().to(torch.bool)
    if (
        not bool(torch.any(mask))
        or bool(torch.any(healthy.corrupt.detach().cpu()))
        or not torch.equal(healthy.published_corrupt.detach().cpu(), mask)
        or not torch.equal(published.published_corrupt.detach().cpu(), mask)
    ):
        raise ValueError("Expected the matched published defect mask to survive only in its companion.")
    healthy_mask = ~mask
    for name in (
        "min_bound",
        "max_bound",
        "dwmin_up",
        "dwmin_down",
        "reference",
    ):
        left = getattr(healthy, name).detach().cpu()[healthy_mask]
        right = getattr(published, name).detach().cpu()[healthy_mask]
        if not torch.equal(left, right):
            raise ValueError(
                f"Expected non-fault IBM OM field {name!r} to match its published companion."
            )
    stuck_persistent_q = published.logical_min.detach().cpu().to(torch.float32)
    stuck_active = published.min_bound.detach().cpu().to(torch.float32)
    corrupt_range = corrupt_devices_range
    if not torch.equal(
        stuck_persistent_q[mask],
        published.logical_max.detach().cpu()[mask],
    ) or bool(torch.any(stuck_active[mask].abs() > corrupt_range + 1e-7)):
        raise ValueError("Expected every published fault source cell to have collapsed support.")
    report = {
        "policy": "post_deployment_published_companion_replay",
        "healthy_population_fingerprint": healthy.fingerprint,
        "published_population_fingerprint": published.fingerprint,
        "assignment_seed": healthy.assignment_seed,
        "fault_mask_sha256": tensor_sha256(mask),
        "stuck_persistent_q_sha256": tensor_sha256(stuck_persistent_q[mask]),
        "stuck_state_source": (
            "AIHWKit_1.1.0_sampled_corrupt_active_bound_minus_intrinsic_reference"
        ),
        "published_corrupt_devices_probability": enabled_corrupt_devices_prob,
        "preset_default_corrupt_devices_probability": (
            preset_default_corrupt_devices_prob
        ),
        "corrupt_devices_range": corrupt_range,
        "persistent_pulse_increments": "zero_up_and_down",
        "apparent_write_noise_retained": True,
        "faulted_cells": int(mask.sum().item()),
        "cells": healthy.size,
        "fault_fraction": float(mask.float().mean().item()),
        "non_fault_identity_equal": True,
        "shared_nominal_device_parameters_equal": True,
    }
    return mask, stuck_persistent_q, report


def _recovery_seed(
    *,
    runtime_seed: int,
    assignment_seed: int,
    endpoint_seed: int,
) -> int:
    # Deliberately excludes recovery policy and layer scope so matched arms see
    # the same Bernoulli stream.
    return derive_seed(runtime_seed, assignment_seed, endpoint_seed, "crossbar_pulse_selection")


def _offchip_realized_state(
    *,
    master: torch.Tensor,
    policy: str,
    logical_minimum: torch.Tensor,
    logical_maximum: torch.Tensor,
    codebook_rows: torch.Tensor,
    validate_codebook: bool,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    if policy == "none":
        return master, None
    if policy == "continuous_hwa":
        return torch.maximum(torch.minimum(master, logical_maximum), logical_minimum), None
    if policy == "deterministic_qat":
        return project_to_nearest_effective_code_device(
            codebook_rows,
            master,
            rowwise=True,
            validate=validate_codebook,
        )
    raise ValueError(f"Unsupported off-chip policy: {policy!r}.")


def _map_transfer_source_state(
    *,
    source_state: str,
    offchip_policy: str,
    offchip_state: Mapping[str, Any],
    source_plant: IbmOmEffectiveCrossbarPlant,
    target_population: IbmReramArrayPopulation,
    target_codebook: Any,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Resolve and target-map the declared source of a fresh-array transfer."""

    if source_state == "offchip_fixed_final_master":
        source = offchip_state.get("fixed_final_master_q")
        if not isinstance(source, torch.Tensor):
            raise ValueError("Expected the off-chip checkpoint to contain fixed_final_master_q.")
        source = source.detach().cpu().to(torch.float32).clone()
        codebook_rows = (
            target_codebook.values.detach()
            .cpu()
            .to(torch.float32)
            .transpose(0, 1)
            .contiguous()
        )
        mapped, indices = _offchip_realized_state(
            master=source,
            policy=offchip_policy,
            logical_minimum=target_population.logical_min.detach().cpu(),
            logical_maximum=target_population.logical_max.detach().cpu(),
            codebook_rows=codebook_rows,
            validate_codebook=True,
        )
        mapping_rule = {
            "none": "identity_master_q",
            "continuous_hwa": "target_support_clamp",
            "deterministic_qat": "target_deterministic_codebook_projection",
        }[offchip_policy]
        source_role = "logical_offchip_fixed_final_master_q"
    elif source_state == "same_array_persistent":
        source = source_plant.persistent.detach().cpu().to(torch.float32).clone()
        mapped = source.clone()
        indices = None
        mapping_rule = "identity_source_final_hidden_persistent_q"
        source_role = "source_array_fixed_final_hidden_persistent_q"
    else:
        raise ValueError(f"Unsupported transfer source state: {source_state!r}.")
    mapped = mapped.detach().cpu().to(torch.float32).clone()
    return mapped, {
        "source_state": source_state,
        "source_role": source_role,
        "offchip_policy": offchip_policy,
        "target_mapping_rule": mapping_rule,
        "source_q_sha256": tensor_sha256(source),
        "target_mapped_q_sha256": tensor_sha256(mapped),
        "target_codebook_index_sha256": (
            None if indices is None else tensor_sha256(indices)
        ),
        "copied_source_offchip_realized_or_codebook_state": False,
        "copied_source_persistent_state": source_state == "same_array_persistent",
        "copied_source_apparent_state": False,
    }


def _offchip_adapt(
    *,
    source_requested: torch.Tensor,
    population: IbmReramArrayPopulation,
    codebook_values: torch.Tensor,
    spec: CrossbarTrainSpec,
    layout: tuple[CrossbarTileSpec, ...],
    digital_scales: tuple[float, float],
    teacher: BiasFreeReluTeacher,
    validation_loader: Iterable,
    train_loader: Iterable,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, Any], Mapping[str, Any]]:
    """Run fixed-final continuous HWA or deterministic-codebook QAT.

    Adam acts on two effective-``q`` shadow tensors.  Dividing each parameter
    group's rate by its fixed digital layer scale makes the Adam displacement
    correspond to the declared logical-weight displacement (up to Adam's
    epsilon term).  The forward pass uses a straight-through estimator; only
    the continuous support clamp or deterministic codebook state reaches the
    network.
    """

    slices = layer_cell_slices(layout)
    source = source_requested.detach().to(device=device, dtype=torch.float32)
    parameters = [
        torch.nn.Parameter(source[layer_slice].clone()) for layer_slice in slices
    ]
    master = torch.cat(parameters)
    minimum = population.logical_min.detach().to(device=device, dtype=torch.float32)
    maximum = population.logical_max.detach().to(device=device, dtype=torch.float32)
    codebook_rows = (
        codebook_values.detach()
        .to(device=device, dtype=torch.float32)
        .transpose(0, 1)
        .contiguous()
    )
    initial_realized, initial_indices = _offchip_realized_state(
        master=master,
        policy=spec.offchip.policy,
        logical_minimum=minimum,
        logical_maximum=maximum,
        codebook_rows=codebook_rows,
        validate_codebook=True,
    )
    initial_validation = _evaluate(
        effective_state=initial_realized,
        digital_scales=digital_scales,
        layout=layout,
        teacher=teacher,
        loader=validation_loader,
        device=device,
        maximum_batches=spec.evaluation.maximum_validation_batches,
        sample_limit=spec.evaluation.sample_limit,
    )
    initial_master = master.detach().cpu().clone()
    initial_realized_cpu = initial_realized.detach().cpu().clone()
    if spec.offchip.policy == "none":
        state = {
            "schema": "ebl.ibm_om_crossbar_offchip_state",
            "schema_version": 1,
            "policy": "none",
            "initial_master_q": initial_master,
            "fixed_final_master_q": initial_master.clone(),
            "fixed_final_realized_q": initial_realized_cpu,
            "optimizer_state_dict": None,
        }
        return initial_realized_cpu, {
            "policy": "none",
            "objective": spec.offchip.objective,
            "initial_master_sha256": tensor_sha256(initial_master),
            "initial_realized_sha256": tensor_sha256(initial_realized_cpu),
            "initial_codebook_index_sha256": None,
            "initial_validation": initial_validation,
            "epochs": [],
            "fixed_final_epoch": 0,
            "fixed_final_master_sha256": tensor_sha256(initial_master),
            "fixed_final_realized_sha256": tensor_sha256(initial_realized_cpu),
            "fixed_final_codebook_index_sha256": None,
            "fixed_final_validation": initial_validation,
            "optimizer_steps": 0,
            "logical_learning_rates": [0.0, 0.0],
            "effective_q_learning_rates": [0.0, 0.0],
            "checkpoint_policy": "fixed_source_no_selection",
            "stochastic_programming_during_training": False,
            "persistent_state_updates_during_training": False,
        }, state

    effective_rates = tuple(
        logical_rate / digital_scale
        for logical_rate, digital_scale in zip(
            spec.offchip.logical_learning_rates,
            digital_scales,
            strict=True,
        )
    )
    optimizer = torch.optim.Adam(
        [
            {"params": [parameter], "lr": rate}
            for parameter, rate in zip(parameters, effective_rates, strict=True)
        ],
        betas=(spec.offchip.beta_1, spec.offchip.beta_2),
        eps=spec.offchip.epsilon,
    )
    epoch_reports: list[dict[str, Any]] = []
    optimizer_steps = 0
    final_validation = initial_validation
    final_realized = initial_realized
    final_indices = initial_indices
    first_epoch_inputs: list[torch.Tensor] = []
    first_epoch_labels: list[torch.Tensor] = []
    for epoch in range(1, spec.offchip.epochs + 1):
        loss_sum = 0.0
        examples = 0
        batches = 0
        for batch_index, (inputs, labels) in enumerate(
            limited(train_loader, spec.offchip.maximum_batches)
        ):
            if epoch == 1:
                first_epoch_inputs.append(
                    inputs.detach().cpu().to(torch.float32).contiguous()
                )
                first_epoch_labels.append(
                    labels.detach().cpu().to(torch.int64).contiguous()
                )
            inputs = inputs.to(device=device, dtype=torch.float32)
            with torch.no_grad():
                teacher_logits = teacher.logits(inputs)
                teacher_log_probability = F.log_softmax(teacher_logits, dim=1)
                teacher_probability = teacher_log_probability.exp()
            optimizer.zero_grad(set_to_none=True)
            master = torch.cat(parameters)
            realized, _indices = _offchip_realized_state(
                master=master,
                policy=spec.offchip.policy,
                logical_minimum=minimum,
                logical_maximum=maximum,
                codebook_rows=codebook_rows,
                validate_codebook=False,
            )
            straight_through = master + (realized - master).detach()
            student_logits = standard_crossbar_logits(
                inputs,
                straight_through,
                layout,
                digital_scales=digital_scales,
            )
            student_log_probability = F.log_softmax(student_logits, dim=1)
            loss = (
                teacher_probability
                * (teacher_log_probability - student_log_probability)
            ).sum(dim=1).mean()
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                for parameter in parameters:
                    parameter.clamp_(
                        min=spec.offchip.master_q_bounds[0],
                        max=spec.offchip.master_q_bounds[1],
                    )
            examples += int(inputs.shape[0])
            loss_sum += float(loss.item()) * int(inputs.shape[0])
            batches = batch_index + 1
            optimizer_steps += 1
        if examples == 0:
            raise ValueError("Expected off-chip HWA/QAT to process training examples.")
        if any(not bool(torch.all(torch.isfinite(parameter))) for parameter in parameters):
            raise RuntimeError("Expected finite off-chip HWA/QAT shadow weights.")
        master = torch.cat(parameters)
        final_realized, final_indices = _offchip_realized_state(
            master=master,
            policy=spec.offchip.policy,
            logical_minimum=minimum,
            logical_maximum=maximum,
            codebook_rows=codebook_rows,
            validate_codebook=False,
        )
        final_validation = _evaluate(
            effective_state=final_realized,
            digital_scales=digital_scales,
            layout=layout,
            teacher=teacher,
            loader=validation_loader,
            device=device,
            maximum_batches=spec.evaluation.maximum_validation_batches,
            sample_limit=spec.evaluation.sample_limit,
        )
        epoch_reports.append(
            {
                "epoch": epoch,
                "batches": batches,
                "examples": examples,
                "train_kl_teacher_student": loss_sum / examples,
                "master_sha256": tensor_sha256(master),
                "realized_sha256": tensor_sha256(final_realized),
                "codebook_index_sha256": (
                    None if final_indices is None else tensor_sha256(final_indices)
                ),
                "validation": final_validation,
            }
        )

    final_master_cpu = torch.cat(parameters).detach().cpu().clone()
    final_realized_cpu = final_realized.detach().cpu().clone()
    first_epoch_training_cohort, _first_epoch_identities = (
        _ordered_labeled_cohort_report(
            inputs=first_epoch_inputs,
            labels=first_epoch_labels,
            split="predeployment_offchip_first_epoch_update_stream",
        )
    )
    state = {
        "schema": "ebl.ibm_om_crossbar_offchip_state",
        "schema_version": 1,
        "policy": spec.offchip.policy,
        "initial_master_q": initial_master,
        "fixed_final_master_q": final_master_cpu,
        "fixed_final_realized_q": final_realized_cpu,
        "optimizer_state_dict": optimizer.state_dict(),
    }
    return final_realized_cpu, {
        "policy": spec.offchip.policy,
        "objective": spec.offchip.objective,
        "initial_master_sha256": tensor_sha256(initial_master),
        "initial_realized_sha256": tensor_sha256(initial_realized_cpu),
        "initial_codebook_index_sha256": (
            None if initial_indices is None else tensor_sha256(initial_indices)
        ),
        "initial_validation": initial_validation,
        "first_epoch_training_cohort": first_epoch_training_cohort,
        "epochs": epoch_reports,
        "fixed_final_epoch": spec.offchip.epochs,
        "fixed_final_master_sha256": tensor_sha256(final_master_cpu),
        "fixed_final_realized_sha256": tensor_sha256(final_realized_cpu),
        "fixed_final_codebook_index_sha256": (
            None if final_indices is None else tensor_sha256(final_indices)
        ),
        "fixed_final_validation": final_validation,
        "optimizer_steps": optimizer_steps,
        "logical_learning_rates": list(spec.offchip.logical_learning_rates),
        "effective_q_learning_rates": list(effective_rates),
        "checkpoint_policy": spec.offchip.checkpoint_policy,
        "stochastic_programming_during_training": False,
        "persistent_state_updates_during_training": False,
    }, state


def _capture_crossbar_star_targets(
    *,
    plant: IbmOmEffectiveCrossbarPlant,
    layout: tuple[CrossbarTileSpec, ...],
    digital_scales: tuple[float, float],
    calibration_loader: Iterable,
    maximum_batches: int | None,
    device: torch.device,
    artifact_path: Path,
    model_checkpoint_sha256: str,
    healthy_deployment_artifact_sha256: str,
    hardware_instance_id: str,
    config_sha256: str,
    hidden_gain: float,
    output_gain: float,
    storage_dtype: str,
    expected_calibration_examples: int,
) -> tuple[Any, dict[str, Any], tuple[str, ...]]:
    """Record endpoint-local healthy states and save a strict STAR artifact."""

    if plant.fault_transition is not None:
        raise RuntimeError("Expected STAR targets to be captured before the fault transition.")
    before = plant.state_dict()
    accumulator = StarTargetAccumulator(CROSSBAR_STATE_REPRESENTATION)
    observed_inputs: list[torch.Tensor] = []
    observed_labels: list[torch.Tensor] = []
    observed_ids: list[torch.Tensor] = []
    offset = 0
    for inputs, labels in limited(calibration_loader, maximum_batches):
        cpu_inputs = inputs.detach().cpu().to(torch.float32).contiguous()
        cpu_labels = labels.detach().cpu().to(torch.int64).contiguous()
        sample_ids = torch.arange(
            offset,
            offset + cpu_labels.numel(),
            dtype=torch.int64,
        )
        states = standard_crossbar_forward_states(
            cpu_inputs.to(device=device),
            plant.apparent,
            layout,
            digital_scales=digital_scales,
        )
        accumulator.observe(
            cpu_labels,
            {
                "hidden_post_relu": states.hidden_post_relu.detach().cpu(),
                "output_logits": states.output_logits.detach().cpu(),
            },
            inputs=cpu_inputs,
            sample_ids=sample_ids,
        )
        observed_inputs.append(cpu_inputs)
        observed_labels.append(cpu_labels)
        observed_ids.append(sample_ids)
        offset += cpu_labels.numel()
    if not observed_inputs:
        raise ValueError("Expected STAR calibration to observe labeled examples.")
    if offset != expected_calibration_examples:
        raise ValueError(
            "Expected STAR calibration to consume exactly the configured cohort size. "
            f"Provided value: observed={offset}, expected={expected_calibration_examples}."
        )
    calibration = build_calibration_binding(
        dataset_id="mnist",
        split="training_calibration",
        ordered_sample_ids=torch.cat(observed_ids),
        model_inputs=torch.cat(observed_inputs),
        labels=torch.cat(observed_labels),
    )
    calibration_report, calibration_identities = _ordered_labeled_cohort_report(
        inputs=observed_inputs,
        labels=observed_labels,
        split="training_calibration",
    )
    if any(
        calibration_report[name] != getattr(calibration, name)
        for name in (
            "dataset_id",
            "split",
            "examples",
            "ordered_sample_ids_sha256",
            "model_inputs_sha256",
            "labels_sha256",
            "cohort_sha256",
        )
    ):
        raise RuntimeError("Expected one exact STAR calibration-cohort binding.")
    binding = StarTargetBinding(
        architecture_id="crossbar_784_50_10_digital_relu",
        state_representation_id=CROSSBAR_STATE_REPRESENTATION,
        source=StarSourceBinding(
            model_checkpoint_sha256=model_checkpoint_sha256,
            healthy_deployment_artifact_sha256=healthy_deployment_artifact_sha256,
            healthy_persistent_state_sha256=tensor_sha256(plant.persistent),
            healthy_forward_state_sha256=tensor_sha256(plant.apparent),
            hardware_instance_id=hardware_instance_id,
            config_sha256=config_sha256,
        ),
        calibration=calibration,
        component_gains=(hidden_gain, output_gain),
        storage_dtype=storage_dtype,
    )
    bundle = accumulator.finalize(binding)
    record = save_star_targets(artifact_path, bundle)
    loaded = load_star_targets(
        record.artifact_path,
        record.receipt_path,
        binding,
    )
    after = plant.state_dict()
    if set(before) != set(after):
        raise RuntimeError("Expected STAR target capture to preserve the plant schema.")
    for name in before:
        left = before[name]
        right = after[name]
        equal = (
            torch.equal(left, right)
            if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor)
            else left == right
        )
        if not equal:
            raise RuntimeError("Expected STAR target capture not to mutate the healthy plant.")
    return loaded, {
        "schema": "ebl.star.crossbar_target_capture",
        "schema_version": 1,
        "lifecycle_phase": "healthy_pre_fault_calibration",
        "state_representation": CROSSBAR_STATE_REPRESENTATION,
        "network_forward_state": "apparent_q",
        "examples": calibration.examples,
        "class_counts": loaded.class_counts.tolist(),
        "component_gains": [hidden_gain, output_gain],
        "storage_dtype": storage_dtype,
        "stored_values": loaded.stored_value_count,
        "stored_mean_bytes": loaded.stored_mean_nbytes,
        "cohort_sha256": calibration.cohort_sha256,
        "semantic_sha256": loaded.semantic_sha256,
        "artifact": artifact_path.name,
        "artifact_sha256": record.artifact_sha256,
        "receipt": record.receipt_path.name,
        "receipt_sha256": record.receipt_sha256,
        "healthy_plant_state_bit_exact_after_capture": True,
        "fault_mask_stored_in_target_artifact": False,
    }, calibration_identities


def _run_local_star_pulse_recovery(
    *,
    update_port: Any,
    spec: CrossbarTrainSpec,
    endpoint_seed: int,
    layout: tuple[CrossbarTileSpec, ...],
    digital_scales: tuple[float, float],
    target_bundle: Any,
    train_loader: Iterable,
    device: torch.device,
) -> tuple[list[dict[str, Any]], dict[str, Any], tuple[str, ...]]:
    """Run teacher-free, no-autograd local STAR outer-product updates."""

    star = spec.recovery.star
    if star is None:
        raise RuntimeError("Expected captured targets before STAR recovery.")
    selection_generator = torch.Generator(device=device.type)
    selection_generator.manual_seed(
        _recovery_seed(
            runtime_seed=spec.runtime.seed,
            assignment_seed=spec.device.assignment_seed,
            endpoint_seed=endpoint_seed,
        )
    )
    optimizer = PulseSGD(
        size=update_port.size,
        layout=layout,
        learning_rates=spec.recovery.learning_rates_q,
        layer_scope=spec.recovery.layer_scope,
        nominal_dw_min=update_port.nominal_dw_min,
        pulse_cap_per_cell=spec.recovery.pulse_cap_per_cell,
        pulse_rule=star.pulse_rule,
        generator=selection_generator,
        device=device,
    )
    hidden_targets = target_bundle.means("hidden_post_relu").to(device)
    output_targets = target_bundle.means("output_logits").to(device)
    epoch_reports: list[dict[str, Any]] = []
    repair_inputs: list[torch.Tensor] = []
    repair_labels: list[torch.Tensor] = []
    for epoch in range(1, spec.recovery.epochs + 1):
        examples = 0
        batches = 0
        hidden_loss_sum = 0.0
        output_loss_sum = 0.0
        commanded = 0
        for batch_index, (inputs, labels) in enumerate(
            limited(train_loader, spec.recovery.maximum_batches)
        ):
            repair_inputs.append(
                inputs.detach().cpu().to(torch.float32).contiguous()
            )
            repair_labels.append(
                labels.detach().cpu().to(torch.int64).contiguous()
            )
            inputs = inputs.to(device=device, dtype=torch.float32)
            labels = labels.to(device=device, dtype=torch.int64)
            # Sequential samples model online row/column pulse coincidence and
            # avoid an undeclared digital minibatch-gradient accumulator.
            for row in range(inputs.shape[0]):
                step = local_star_crossbar_step(
                    inputs[row : row + 1],
                    labels[row : row + 1],
                    update_port.apparent,
                    layout,
                    digital_scales=digital_scales,
                    hidden_targets=hidden_targets,
                    output_targets=output_targets,
                    hidden_gain=star.hidden_gain,
                    output_gain=star.output_gain,
                )
                pulse = optimizer.step(step.gradient_q, update_port)
                hidden_loss_sum += float(step.hidden_loss.item())
                output_loss_sum += float(step.output_loss.item())
                commanded += int(pulse["commanded_pulses"])
                examples += 1
            batches = batch_index + 1
        if examples == 0:
            raise ValueError("Expected local STAR recovery to process labeled examples.")
        state_hashes = update_port.state_hash_receipt()
        epoch_reports.append(
            {
                "epoch": epoch,
                "batches": batches,
                "examples": examples,
                "mean_hidden_local_state_loss": hidden_loss_sum / examples,
                "mean_output_local_state_loss": output_loss_sum / examples,
                "commanded_pulses": commanded,
                **state_hashes,
            }
        )
    repair_cohort, repair_identities = _ordered_labeled_cohort_report(
        inputs=repair_inputs,
        labels=repair_labels,
        split="post_fault_local_repair_update_stream",
    )
    optimizer_report = optimizer.report()
    optimizer_report["repair_cohort"] = repair_cohort
    return epoch_reports, optimizer_report, repair_identities


def _run_supervised_ce_pulse_recovery(
    *,
    update_port: Any,
    spec: CrossbarTrainSpec,
    endpoint_seed: int,
    layout: tuple[CrossbarTileSpec, ...],
    digital_scales: tuple[float, float],
    train_loader: Iterable,
    device: torch.device,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Run ordinary label-supervised BP with digital Adam and physical pulses."""

    settings = spec.recovery.supervised_bp
    if settings is None:
        raise RuntimeError("Expected supervised-BP recovery settings.")
    selection_generator = torch.Generator(device=device.type)
    selection_generator.manual_seed(
        _recovery_seed(
            runtime_seed=spec.runtime.seed,
            assignment_seed=spec.device.assignment_seed,
            endpoint_seed=endpoint_seed,
        )
    )
    optimizer = PulseAdam(
        size=update_port.size,
        layout=layout,
        learning_rates=spec.recovery.learning_rates_q,
        betas=(float(spec.recovery.beta_1), float(spec.recovery.beta_2)),
        epsilon=float(spec.recovery.epsilon),
        layer_scope=spec.recovery.layer_scope,
        nominal_dw_min=update_port.nominal_dw_min,
        pulse_cap_per_cell=spec.recovery.pulse_cap_per_cell,
        generator=selection_generator,
        device=device,
    )
    epoch_reports: list[dict[str, Any]] = []
    all_inputs: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []
    epoch_cohorts: list[dict[str, Any]] = []
    for epoch in range(1, spec.recovery.epochs + 1):
        epoch_inputs: list[torch.Tensor] = []
        epoch_labels: list[torch.Tensor] = []
        examples = 0
        batches = 0
        loss_sum = 0.0
        correct = 0
        commanded = 0
        for batch_index, (inputs, labels) in enumerate(
            limited(train_loader, spec.recovery.maximum_batches)
        ):
            cpu_inputs = inputs.detach().cpu().to(torch.float32).contiguous()
            cpu_labels = labels.detach().cpu().to(torch.int64).contiguous()
            epoch_inputs.append(cpu_inputs)
            epoch_labels.append(cpu_labels)
            all_inputs.append(cpu_inputs)
            all_labels.append(cpu_labels)
            inputs = cpu_inputs.to(device=device)
            labels = cpu_labels.to(device=device)
            apparent_q = update_port.apparent.to(device).requires_grad_(True)
            logits = standard_crossbar_logits(
                inputs,
                apparent_q,
                layout,
                digital_scales=digital_scales,
            )
            loss = F.cross_entropy(logits, labels)
            gradient = torch.autograd.grad(loss, apparent_q, only_inputs=True)[0]
            pulse = optimizer.step(gradient, update_port)
            batch_examples = int(inputs.shape[0])
            examples += batch_examples
            batches = batch_index + 1
            loss_sum += float(loss.item()) * batch_examples
            correct += int((logits.argmax(dim=1) == labels).sum().item())
            commanded += int(
                (
                    pulse["commanded_pulses"]
                    if "commanded_pulses" in pulse
                    else pulse["applied_pulses"]
                )
            )
        if examples != settings.repair_examples:
            raise ValueError(
                "Expected supervised BP to consume exactly its declared repair "
                f"cohort each epoch. Provided value: observed={examples}, "
                f"expected={settings.repair_examples}."
            )
        epoch_cohort, _ = _ordered_labeled_cohort_report(
            inputs=epoch_inputs,
            labels=epoch_labels,
            split=f"post_fault_supervised_bp_epoch_{epoch}_update_stream",
        )
        epoch_cohorts.append(epoch_cohort)
        epoch_reports.append(
            {
                "epoch": epoch,
                "batches": batches,
                "examples": examples,
                "train_cross_entropy": loss_sum / examples,
                "train_pre_update_accuracy": correct / examples,
                "commanded_pulses": commanded,
                "repair_cohort": epoch_cohort,
                **update_port.state_hash_receipt(),
            }
        )
    repair_cohort, _ = _ordered_labeled_cohort_report(
        inputs=all_inputs,
        labels=all_labels,
        split="post_fault_supervised_bp_full_update_stream",
    )
    optimizer_report = optimizer.report()
    optimizer_report["repair_cohort"] = repair_cohort
    optimizer_report["per_epoch_repair_cohorts"] = epoch_cohorts
    optimizer_report["nominal_dw_min"] = update_port.nominal_dw_min
    return epoch_reports, optimizer_report


def _posthoc_recovery_pulse_effects(
    *,
    faulted_state: Mapping[str, Any],
    final_state: Mapping[str, Any],
    fault_mask: torch.Tensor,
    layout: tuple[CrossbarTileSpec, ...],
) -> dict[str, Any]:
    """Audit command destinations and healthy persistent movement after recovery."""

    names = ("persistent", "upward_pulses", "downward_pulses")
    if any(
        not isinstance(faulted_state.get(name), torch.Tensor)
        or not isinstance(final_state.get(name), torch.Tensor)
        for name in names
    ):
        raise ValueError("Expected complete faulted and final post-fault plant states.")
    initial_pulses = (
        faulted_state["upward_pulses"] + faulted_state["downward_pulses"]
    ).detach().cpu()
    final_pulses = (
        final_state["upward_pulses"] + final_state["downward_pulses"]
    ).detach().cpu()
    commands = final_pulses - initial_pulses
    mask = fault_mask.detach().cpu().to(torch.bool)
    before = faulted_state["persistent"].detach().cpu().to(torch.float32)
    after = final_state["persistent"].detach().cpu().to(torch.float32)
    if (
        commands.shape != mask.shape
        or before.shape != mask.shape
        or after.shape != mask.shape
        or bool(torch.any(commands < 0))
        or not bool(torch.all(torch.isfinite(before)))
        or not bool(torch.all(torch.isfinite(after)))
        or not torch.equal(before[mask], after[mask])
    ):
        raise RuntimeError("Expected ordered pulse counts and immutable persistent faults.")
    delta = after - before
    changed_healthy = (delta != 0.0) & ~mask
    slices = layer_cell_slices(layout)
    per_layer = []
    for layer_index, layer_slice in enumerate(slices):
        layer_mask = mask[layer_slice]
        layer_commands = commands[layer_slice]
        layer_delta = delta[layer_slice]
        layer_changed = changed_healthy[layer_slice]
        per_layer.append(
            {
                "layer_index": layer_index,
                "commanded_pulses": int(layer_commands.sum().item()),
                "commands_to_immutable_cells": int(
                    layer_commands[layer_mask].sum().item()
                ),
                "commands_to_programmable_cells": int(
                    layer_commands[~layer_mask].sum().item()
                ),
                "persistent_changed_healthy_cells": int(
                    layer_changed.sum().item()
                ),
                "persistent_healthy_delta_l1": float(
                    layer_delta[~layer_mask].abs().sum().item()
                ),
                "persistent_healthy_delta_l2": float(
                    layer_delta[~layer_mask].square().sum().sqrt().item()
                ),
            }
        )
    return {
        "analysis_accesses_fault_mask_posthoc": True,
        "learner_accesses_fault_mask": False,
        "commanded_pulses": int(commands.sum().item()),
        "commands_to_immutable_cells": int(commands[mask].sum().item()),
        "commands_to_programmable_cells": int(commands[~mask].sum().item()),
        "persistent_changed_healthy_cells": int(changed_healthy.sum().item()),
        "persistent_healthy_delta_l1": float(delta[~mask].abs().sum().item()),
        "persistent_healthy_delta_l2": float(
            delta[~mask].square().sum().sqrt().item()
        ),
        "per_layer": per_layer,
    }


def _posthoc_local_star_pulse_effects(**kwargs: Any) -> dict[str, Any]:
    """Compatibility alias for the generalized post-fault pulse audit."""

    return _posthoc_recovery_pulse_effects(**kwargs)


def _recover(
    *,
    plant: IbmOmEffectiveCrossbarPlant,
    spec: CrossbarTrainSpec,
    endpoint_seed: int,
    layout: tuple[CrossbarTileSpec, ...],
    digital_scales: tuple[float, float],
    teacher: BiasFreeReluTeacher,
    validation_loader: Iterable,
    train_loader: Iterable,
    device: torch.device,
    calibration_loader: Iterable | None = None,
    post_fault_context: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], Mapping[str, Any]]:
    initial_state = plant.state_dict()
    initial_evaluation = _evaluate_plant_states(
        plant=plant,
        digital_scales=digital_scales,
        layout=layout,
        teacher=teacher,
        loader=validation_loader,
        device=device,
        maximum_batches=spec.evaluation.maximum_validation_batches,
        sample_limit=spec.evaluation.sample_limit,
    )
    final_state: Mapping[str, Any] = initial_state
    final_epoch = 0
    final_evaluation = initial_evaluation
    epoch_reports: list[dict[str, Any]] = []
    if spec.recovery.policy == "none":
        return {
            "policy": "none",
            "layer_scope": "all",
            "network_forward_state": "apparent_q",
            "hidden_update_state": "persistent_q",
            "gradient_handoff": "identity_ste_apparent_q_to_persistent_pulse_update",
            "initial_apparent_validation": initial_evaluation["apparent_forward"],
            "initial_persistent_validation": initial_evaluation[
                "persistent_diagnostic"
            ],
            "epochs": epoch_reports,
            "fixed_final_epoch": 0,
            "fixed_final_apparent_validation": initial_evaluation[
                "apparent_forward"
            ],
            "fixed_final_persistent_validation": initial_evaluation[
                "persistent_diagnostic"
            ],
            "optimizer": {
                "optimizer_steps": 0,
                "requested_nonzero_commands": 0,
                "applied_pulses": 0,
                "applied_pulses_by_layer": [0, 0],
                "probability_clipped": 0,
                "blocked_at_cap": 0,
                "pulse_cap_per_cell": 0,
                "cells_at_cap": 0,
                "changed_cells": 0,
                "maximum_pulses_per_cell": 0,
                "enabled_cells": 0,
            },
            "checkpoint_policy": "fixed_p0_no_selection",
        }, final_state

    if spec.recovery.policy == "star_local_pulse_sgd":
        star = spec.recovery.star
        required_context = {
            "artifact_path",
            "faulted_checkpoint_path",
            "model_checkpoint_sha256",
            "healthy_deployment_artifact_sha256",
            "hardware_instance_id",
            "config_sha256",
            "fault_mask",
            "stuck_persistent_q",
            "fault_report",
            "fault_source_population_fingerprint",
            "predeployment_training_cohort",
        }
        if (
            star is None
            or calibration_loader is None
            or not isinstance(post_fault_context, Mapping)
            or set(post_fault_context) != required_context
        ):
            raise ValueError("Expected a complete chronological STAR recovery context.")
        (
            target_bundle,
            target_report,
            calibration_example_identities,
        ) = _capture_crossbar_star_targets(
            plant=plant,
            layout=layout,
            digital_scales=digital_scales,
            calibration_loader=calibration_loader,
            maximum_batches=star.calibration_maximum_batches,
            device=device,
            artifact_path=Path(post_fault_context["artifact_path"]),
            model_checkpoint_sha256=str(post_fault_context["model_checkpoint_sha256"]),
            healthy_deployment_artifact_sha256=str(
                post_fault_context["healthy_deployment_artifact_sha256"]
            ),
            hardware_instance_id=str(post_fault_context["hardware_instance_id"]),
            config_sha256=str(post_fault_context["config_sha256"]),
            hidden_gain=star.hidden_gain,
            output_gain=star.output_gain,
            storage_dtype=star.target_quantization,
            expected_calibration_examples=star.calibration_examples,
        )
        state_error_evaluation = {
            "target_bundle": target_bundle,
            "digital_scales": digital_scales,
            "layout": layout,
            "loader": validation_loader,
            "device": device,
            "maximum_batches": spec.evaluation.maximum_validation_batches,
            "sample_limit": spec.evaluation.sample_limit,
        }
        healthy_state_errors = _evaluate_local_star_state_errors(
            plant=plant,
            **state_error_evaluation,
        )
        transition = plant.apply_stuck_at_fault_transition(
            mask=post_fault_context["fault_mask"],
            stuck_persistent_q=post_fault_context["stuck_persistent_q"],
            transition_id=(
                f"assignment-{spec.device.assignment_seed}-endpoint-{endpoint_seed}"
            ),
            source_population_fingerprint=str(
                post_fault_context["fault_source_population_fingerprint"]
            ),
        )
        faulted_state = plant.state_dict()
        faulted_checkpoint_path = Path(post_fault_context["faulted_checkpoint_path"])
        atomic_torch_save(faulted_state, faulted_checkpoint_path)
        damaged_evaluation = _evaluate_plant_states(
            plant=plant,
            digital_scales=digital_scales,
            layout=layout,
            teacher=teacher,
            loader=validation_loader,
            device=device,
            maximum_batches=spec.evaluation.maximum_validation_batches,
            sample_limit=spec.evaluation.sample_limit,
        )
        damaged_state_errors = _evaluate_local_star_state_errors(
            plant=plant,
            **state_error_evaluation,
        )
        update_port = plant.local_star_update_port()
        epoch_reports, optimizer_report, repair_example_identities = (
            _run_local_star_pulse_recovery(
                update_port=update_port,
                spec=spec,
                endpoint_seed=endpoint_seed,
                layout=layout,
                digital_scales=digital_scales,
                target_bundle=target_bundle,
                train_loader=train_loader,
                device=device,
            )
        )
        predeployment_cohort = post_fault_context["predeployment_training_cohort"]
        repair_cohort = optimizer_report["repair_cohort"]
        if not isinstance(predeployment_cohort, Mapping) or any(
            predeployment_cohort.get(name) != repair_cohort[name]
            for name in (
                "examples",
                "ordered_sample_ids_sha256",
                "model_inputs_sha256",
                "labels_sha256",
                "ordered_example_identity_sequence_sha256",
                "unique_example_identities",
            )
        ):
            raise RuntimeError(
                "Expected local repair to reuse the exact first predeployment "
                "training update stream."
            )
        calibration_overlap = len(
            set(calibration_example_identities).intersection(repair_example_identities)
        )
        if calibration_overlap != 0:
            raise RuntimeError(
                "Expected STAR target calibration and repair cohorts to be disjoint."
            )
        repair_cohort["same_as_predeployment_first_epoch_update_stream"] = True
        repair_cohort["calibration_example_identity_overlap"] = calibration_overlap
        repair_cohort["disjoint_from_target_calibration"] = True
        final_evaluation = _evaluate_plant_states(
            plant=plant,
            digital_scales=digital_scales,
            layout=layout,
            teacher=teacher,
            loader=validation_loader,
            device=device,
            maximum_batches=spec.evaluation.maximum_validation_batches,
            sample_limit=spec.evaluation.sample_limit,
        )
        final_state_errors = _evaluate_local_star_state_errors(
            plant=plant,
            **state_error_evaluation,
        )
        final_state = plant.state_dict()
        posthoc_pulse_effects = _posthoc_local_star_pulse_effects(
            faulted_state=faulted_state,
            final_state=final_state,
            fault_mask=post_fault_context["fault_mask"],
            layout=layout,
        )
        if (
            posthoc_pulse_effects["commanded_pulses"]
            != optimizer_report["commanded_pulses"]
        ):
            raise RuntimeError(
                "Expected the local optimizer and post-hoc plant command counts to match."
            )
        return {
            "policy": spec.recovery.policy,
            "objective": "local_star_state_matching",
            "layer_scope": spec.recovery.layer_scope,
            "network_forward_state": "apparent_q",
            "hidden_update_state": "persistent_q",
            "gradient_handoff": "identity_apparent_local_error_to_persistent_pulse_update",
            "local_error_contract": {
                "hidden": "rho_h*(h-mu_y_h)*relu_mask",
                "output": "rho_o*(z-mu_y_o)",
                "gradient_q_hidden": (
                    "alpha_1*x_transpose*[rho_h*(h-mu_y_h)*relu_mask]"
                ),
                "gradient_q_output": (
                    "alpha_2*h_transpose*[rho_o*(z-mu_y_o)]"
                ),
                "pulse_direction": "-sign(gradient_q)",
                "pulse_selection_probability": (
                    "min(1,eta_layer*abs(gradient_q)/dw_nominal)"
                ),
                "learning_rates_q": list(spec.recovery.learning_rates_q),
                "nominal_dw_min": optimizer_report["nominal_dw_min"],
                "label_use": "class_conditional_target_address_only",
                "transpose_mvm": False,
                "cross_entropy_or_softmax": False,
                "cross_layer_error_propagation": False,
                "teacher_access_during_updates": False,
                "autograd_during_updates": False,
                "adam_moments": False,
                "fault_mask_access_during_updates": False,
                "restricted_update_port": True,
                "update_port_exposes": [
                    "apparent_q",
                    "pulse_application",
                    "array_size",
                    "nominal_step",
                    "state_hash_receipt",
                ],
            },
            "healthy_pre_fault_apparent_validation": initial_evaluation[
                "apparent_forward"
            ],
            "healthy_pre_fault_persistent_validation": initial_evaluation[
                "persistent_diagnostic"
            ],
            "healthy_pre_fault_local_state_errors": healthy_state_errors,
            "target_capture": target_report,
            "fault_source": dict(post_fault_context["fault_report"]),
            "fault_transition": transition,
            "faulted_pre_recovery_checkpoint": faulted_checkpoint_path.name,
            "faulted_pre_recovery_checkpoint_sha256": sha256_file(
                faulted_checkpoint_path
            ),
            "faulted_pre_recovery_apparent_validation": damaged_evaluation[
                "apparent_forward"
            ],
            "faulted_pre_recovery_persistent_validation": damaged_evaluation[
                "persistent_diagnostic"
            ],
            "faulted_pre_recovery_local_state_errors": damaged_state_errors,
            "epochs": epoch_reports,
            "fixed_final_epoch": spec.recovery.epochs,
            "fixed_final_apparent_validation": final_evaluation["apparent_forward"],
            "fixed_final_persistent_validation": final_evaluation[
                "persistent_diagnostic"
            ],
            "fixed_final_local_state_errors": final_state_errors,
            "optimizer": optimizer_report,
            "posthoc_pulse_effects": posthoc_pulse_effects,
            "checkpoint_policy": "fixed_final_epoch_no_selection",
        }, final_state

    if spec.recovery.policy == "supervised_ce_pulse_adam":
        required_context = {
            "faulted_checkpoint_path",
            "fault_mask",
            "stuck_persistent_q",
            "fault_report",
            "fault_source_population_fingerprint",
            "predeployment_training_cohort",
        }
        if (
            spec.recovery.supervised_bp is None
            or not isinstance(post_fault_context, Mapping)
            or set(post_fault_context) != required_context
        ):
            raise ValueError(
                "Expected a complete chronological supervised-BP recovery context."
            )
        transition = plant.apply_stuck_at_fault_transition(
            mask=post_fault_context["fault_mask"],
            stuck_persistent_q=post_fault_context["stuck_persistent_q"],
            transition_id=(
                f"assignment-{spec.device.assignment_seed}-endpoint-{endpoint_seed}"
            ),
            source_population_fingerprint=str(
                post_fault_context["fault_source_population_fingerprint"]
            ),
        )
        faulted_state = plant.state_dict()
        faulted_checkpoint_path = Path(
            post_fault_context["faulted_checkpoint_path"]
        )
        atomic_torch_save(faulted_state, faulted_checkpoint_path)
        damaged_evaluation = _evaluate_plant_states(
            plant=plant,
            digital_scales=digital_scales,
            layout=layout,
            teacher=teacher,
            loader=validation_loader,
            device=device,
            maximum_batches=spec.evaluation.maximum_validation_batches,
            sample_limit=spec.evaluation.sample_limit,
        )
        update_port = plant.restricted_recovery_update_port()
        epoch_reports, optimizer_report = _run_supervised_ce_pulse_recovery(
            update_port=update_port,
            spec=spec,
            endpoint_seed=endpoint_seed,
            layout=layout,
            digital_scales=digital_scales,
            train_loader=train_loader,
            device=device,
        )
        repair_cohort = optimizer_report["repair_cohort"]
        predeployment_cohort = post_fault_context["predeployment_training_cohort"]
        if not isinstance(predeployment_cohort, Mapping):
            raise RuntimeError("Expected a bound predeployment training cohort.")
        comparison_fields = (
            "examples",
            "ordered_sample_ids_sha256",
            "model_inputs_sha256",
            "labels_sha256",
            "ordered_example_identity_sequence_sha256",
            "unique_example_identities",
        )
        same_predeployment_stream = all(
            predeployment_cohort.get(name) == repair_cohort[name]
            for name in comparison_fields
        )
        if (
            spec.recovery.supervised_bp.repair_examples == spec.data.num_points
            and spec.recovery.epochs == 1
            and not same_predeployment_stream
        ):
            raise RuntimeError(
                "Expected the matched supervised-BP arm to reuse the exact "
                "predeployment update stream."
            )
        repair_cohort["same_as_predeployment_first_epoch_update_stream"] = (
            same_predeployment_stream
        )
        repair_cohort["predeployment_first_epoch_update_stream"] = {
            name: predeployment_cohort.get(name) for name in comparison_fields
        }
        final_evaluation = _evaluate_plant_states(
            plant=plant,
            digital_scales=digital_scales,
            layout=layout,
            teacher=teacher,
            loader=validation_loader,
            device=device,
            maximum_batches=spec.evaluation.maximum_validation_batches,
            sample_limit=spec.evaluation.sample_limit,
        )
        final_state = plant.state_dict()
        posthoc_pulse_effects = _posthoc_recovery_pulse_effects(
            faulted_state=faulted_state,
            final_state=final_state,
            fault_mask=post_fault_context["fault_mask"],
            layout=layout,
        )
        if (
            posthoc_pulse_effects["commanded_pulses"]
            != optimizer_report["commanded_pulses"]
        ):
            raise RuntimeError(
                "Expected supervised BP and post-hoc plant command counts to match."
            )
        return {
            "policy": spec.recovery.policy,
            "objective": "cross_entropy",
            "layer_scope": spec.recovery.layer_scope,
            "network_forward_state": "apparent_q",
            "hidden_update_state": "persistent_q",
            "gradient_handoff": (
                "autograd_backprop_apparent_q_to_persistent_pulse_update"
            ),
            "ordinary_backprop_contract": {
                "loss": "ground_truth_label_cross_entropy",
                "output_error": "softmax_logits_minus_onehot_label",
                "hidden_error": "output_error_times_W2_transpose_times_relu_mask",
                "transpose_mvm": True,
                "cross_layer_backpropagation": True,
                "autograd_during_updates": True,
                "digital_adam_moments": True,
                "teacher_access_during_updates": False,
                "fault_mask_access_during_updates": False,
                "restricted_update_port": True,
                "update_port_exposes": [
                    "apparent_q",
                    "pulse_application",
                    "array_size",
                    "nominal_step",
                    "state_hash_receipt",
                ],
                "learning_rates_q": list(spec.recovery.learning_rates_q),
                "adam_betas": [spec.recovery.beta_1, spec.recovery.beta_2],
                "adam_epsilon": spec.recovery.epsilon,
                "nominal_dw_min": optimizer_report["nominal_dw_min"],
            },
            "healthy_pre_fault_apparent_validation": initial_evaluation[
                "apparent_forward"
            ],
            "healthy_pre_fault_persistent_validation": initial_evaluation[
                "persistent_diagnostic"
            ],
            "fault_source": dict(post_fault_context["fault_report"]),
            "fault_transition": transition,
            "faulted_pre_recovery_checkpoint": faulted_checkpoint_path.name,
            "faulted_pre_recovery_checkpoint_sha256": sha256_file(
                faulted_checkpoint_path
            ),
            "faulted_pre_recovery_apparent_validation": damaged_evaluation[
                "apparent_forward"
            ],
            "faulted_pre_recovery_persistent_validation": damaged_evaluation[
                "persistent_diagnostic"
            ],
            "epochs": epoch_reports,
            "fixed_final_epoch": spec.recovery.epochs,
            "fixed_final_apparent_validation": final_evaluation[
                "apparent_forward"
            ],
            "fixed_final_persistent_validation": final_evaluation[
                "persistent_diagnostic"
            ],
            "optimizer": optimizer_report,
            "posthoc_pulse_effects": posthoc_pulse_effects,
            "checkpoint_policy": "fixed_final_epoch_no_selection",
        }, final_state

    selection_generator = torch.Generator(device=device.type)
    selection_generator.manual_seed(
        _recovery_seed(
            runtime_seed=spec.runtime.seed,
            assignment_seed=spec.device.assignment_seed,
            endpoint_seed=endpoint_seed,
        )
    )
    optimizer = PulseAdam(
        size=plant.size,
        layout=layout,
        learning_rates=spec.recovery.learning_rates_q,
        betas=(float(spec.recovery.beta_1), float(spec.recovery.beta_2)),
        epsilon=float(spec.recovery.epsilon),
        layer_scope=spec.recovery.layer_scope,
        nominal_dw_min=population_nominal_step(plant.population),
        pulse_cap_per_cell=spec.recovery.pulse_cap_per_cell,
        generator=selection_generator,
        device=device,
    )
    for epoch in range(1, spec.recovery.epochs + 1):
        loss_sum = 0.0
        examples = 0
        batches = 0
        applied = 0
        for batch_index, (inputs, _labels) in enumerate(
            limited(train_loader, spec.recovery.maximum_batches)
        ):
            inputs = inputs.to(device=device, dtype=torch.float32)
            with torch.no_grad():
                teacher_logits = teacher.logits(inputs)
                teacher_log_probability = F.log_softmax(teacher_logits, dim=1)
                teacher_probability = teacher_log_probability.exp()
            state = plant.apparent.detach().clone().requires_grad_(True)
            student_logits = standard_crossbar_logits(
                inputs,
                state,
                layout,
                digital_scales=digital_scales,
            )
            student_log_probability = F.log_softmax(student_logits, dim=1)
            loss = (
                teacher_probability
                * (teacher_log_probability - student_log_probability)
            ).sum(dim=1).mean()
            gradient = torch.autograd.grad(loss, state, only_inputs=True)[0]
            step = optimizer.step(gradient, plant)
            applied += int(step["applied_pulses"])
            examples += int(inputs.shape[0])
            loss_sum += float(loss.item()) * int(inputs.shape[0])
            batches = batch_index + 1
        if examples == 0:
            raise ValueError("Expected pulse-Adam recovery to process training examples.")
        evaluation = _evaluate_plant_states(
            plant=plant,
            digital_scales=digital_scales,
            layout=layout,
            teacher=teacher,
            loader=validation_loader,
            device=device,
            maximum_batches=spec.evaluation.maximum_validation_batches,
            sample_limit=spec.evaluation.sample_limit,
        )
        report = {
            "epoch": epoch,
            "batches": batches,
            "examples": examples,
            "train_kl_teacher_student": loss_sum / examples,
            "applied_pulses": applied,
            "apparent_validation": evaluation["apparent_forward"],
            "persistent_validation": evaluation["persistent_diagnostic"],
            "apparent_sha256": tensor_sha256(plant.apparent),
            "persistent_sha256": tensor_sha256(plant.persistent),
        }
        epoch_reports.append(report)
        # The matched DRN endpoint study evaluates the fixed one-epoch state;
        # validation is diagnostic and never chooses between P0 and recovery.
        final_epoch = epoch
        final_evaluation = evaluation
        final_state = plant.state_dict()

    plant.load_state_dict(final_state)
    return {
        "policy": spec.recovery.policy,
        "layer_scope": spec.recovery.layer_scope,
        "network_forward_state": "apparent_q",
        "hidden_update_state": "persistent_q",
        "gradient_handoff": "identity_ste_apparent_q_to_persistent_pulse_update",
        "initial_apparent_validation": initial_evaluation["apparent_forward"],
        "initial_persistent_validation": initial_evaluation[
            "persistent_diagnostic"
        ],
        "epochs": epoch_reports,
        "fixed_final_epoch": final_epoch,
        "fixed_final_apparent_validation": final_evaluation["apparent_forward"],
        "fixed_final_persistent_validation": final_evaluation[
            "persistent_diagnostic"
        ],
        "optimizer": optimizer.report(),
        "checkpoint_policy": "fixed_final_epoch_no_selection",
    }, final_state


def population_nominal_step(population: IbmReramArrayPopulation) -> float:
    value = float(population.nominal_dw_min)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError("Expected a positive IBM OM nominal pulse scale.")
    return value


def _numeric_summary(values: Iterable[float]) -> dict[str, float]:
    resolved = [float(value) for value in values]
    if not resolved or any(not math.isfinite(value) for value in resolved):
        raise ValueError("Expected at least one finite value to summarize.")
    return {
        "mean": fmean(resolved),
        "population_sd": pstdev(resolved) if len(resolved) > 1 else 0.0,
        "minimum": min(resolved),
        "maximum": max(resolved),
    }


def _aggregate(rows: list[dict[str, Any]], key_path: tuple[str, ...]) -> dict[str, float]:
    values: list[float] = []
    for row in rows:
        value: Any = row
        for key in key_path:
            value = value[key]
        values.append(float(value))
    return _numeric_summary(values)


def _drn_comparison(
    *,
    rows: list[dict[str, Any]],
    reference: Mapping[str, Any],
    recovery_policy: str,
    bound_policy: str,
    offchip_policy: str = "none",
    layer_scope: str = "all",
    crossbar_deployment_source_accuracy: float | None = None,
) -> dict[str, Any]:
    metrics = reference["metrics"]
    drn_p0 = tuple(float(value) for value in metrics["persistent_deployment"]["per_endpoint_accuracy"])
    drn_recovered = tuple(float(value) for value in metrics["open_loop_adam"]["per_endpoint_accuracy"])
    drn_source_accuracy = float(reference["source"]["source_pretransfer_accuracy"])
    drn_requested_accuracy = float(
        reference["source"]["requested_remap_no_write_accuracy"]
    )
    def row_metric(
        row: Mapping[str, Any],
        explicit: str,
        legacy: str,
    ) -> Mapping[str, Any]:
        value = row.get(explicit, row.get(legacy))
        if not isinstance(value, Mapping):
            raise ValueError(f"Expected endpoint metric {explicit!r}.")
        return value

    p0_metrics = tuple(
        row_metric(row, "p0_apparent_test", "p0_test") for row in rows
    )
    final_metrics = tuple(
        row_metric(row, "fixed_final_apparent_test", "fixed_final_test")
        for row in rows
    )
    crossbar_p0 = tuple(float(value["student_accuracy"]) for value in p0_metrics)
    crossbar_final = tuple(
        float(value["student_accuracy"]) for value in final_metrics
    )
    teacher = tuple(float(value["teacher_accuracy"]) for value in p0_metrics)
    if crossbar_deployment_source_accuracy is None:
        crossbar_source = teacher
    else:
        value = float(crossbar_deployment_source_accuracy)
        if not math.isfinite(value) or value < 0.0 or value > 1.0:
            raise ValueError("Expected a valid crossbar deployment-source accuracy.")
        crossbar_source = tuple(value for _ in rows)

    def closure(final: float, initial: float, target: float) -> float | None:
        gap = target - initial
        return None if abs(gap) <= 1e-12 else (final - initial) / gap

    crossbar_closure = [
        closure(final, initial, target)
        for final, initial, target in zip(
            crossbar_final,
            crossbar_p0,
            crossbar_source,
            strict=True,
        )
    ]
    crossbar_teacher_closure = [
        closure(final, initial, target)
        for final, initial, target in zip(
            crossbar_final,
            crossbar_p0,
            teacher,
            strict=True,
        )
    ]
    drn_closure = [
        closure(final, initial, drn_source_accuracy)
        for final, initial in zip(drn_recovered, drn_p0, strict=True)
    ]
    crossbar_gain_pp = [
        100.0 * (final - initial)
        for final, initial in zip(crossbar_final, crossbar_p0, strict=True)
    ]
    drn_gain_pp = [
        100.0 * (final - initial)
        for final, initial in zip(drn_recovered, drn_p0, strict=True)
    ]
    crossbar_mean_p0 = fmean(crossbar_p0)
    crossbar_mean_final = fmean(crossbar_final)
    floor_crossing_candidate = (
        offchip_policy == "deterministic_qat"
        and recovery_policy == "pulse_adam"
        and layer_scope == "all"
        and bound_policy == "winsorize_raw_active_a_to_unit_interval"
        and crossbar_mean_p0 < 0.90
        and crossbar_mean_final >= 0.90
    )
    return {
        "endpoint_seeds": [int(row["endpoint_seed"]) for row in rows],
        "drn_p0_accuracy": list(drn_p0),
        "crossbar_p0_accuracy": list(crossbar_p0),
        "crossbar_minus_drn_p0_percentage_points": [
            100.0 * (crossbar - drn) for crossbar, drn in zip(crossbar_p0, drn_p0, strict=True)
        ],
        "drn_open_loop_adam_accuracy": list(drn_recovered),
        "crossbar_fixed_final_accuracy": list(crossbar_final),
        "crossbar_minus_drn_recovered_percentage_points": [
            100.0 * (crossbar - drn)
            for crossbar, drn in zip(crossbar_final, drn_recovered, strict=True)
        ],
        "crossbar_recovery_gain_percentage_points": crossbar_gain_pp,
        "drn_recovery_gain_percentage_points": drn_gain_pp,
        "crossbar_source_to_p0_loss_percentage_points": [
            100.0 * (source - initial)
            for source, initial in zip(crossbar_source, crossbar_p0, strict=True)
        ],
        "drn_source_to_p0_loss_percentage_points": [
            100.0 * (drn_source_accuracy - initial) for initial in drn_p0
        ],
        "drn_requested_remap_to_p0_loss_percentage_points": [
            100.0 * (drn_requested_accuracy - initial) for initial in drn_p0
        ],
        "crossbar_fraction_of_deployment_source_accuracy_gap_closed": crossbar_closure,
        "crossbar_fraction_of_teacher_accuracy_gap_closed": crossbar_teacher_closure,
        "drn_fraction_of_source_accuracy_gap_closed": drn_closure,
        "crossbar_source_accuracy": list(crossbar_source),
        "crossbar_source_kind": (
            "exact_teacher_mapping"
            if offchip_policy == "none"
            else f"fixed_final_{offchip_policy}_realized_target"
        ),
        "crossbar_teacher_accuracy": list(teacher),
        "drn_source_pretransfer_accuracy": drn_source_accuracy,
        "drn_requested_remap_no_write_accuracy": drn_requested_accuracy,
        "crossbar_mean_p0": crossbar_mean_p0,
        "drn_mean_p0": fmean(drn_p0),
        "crossbar_mean_fixed_final": crossbar_mean_final,
        "drn_mean_open_loop_adam": fmean(drn_recovered),
        "predeclared_importance_rule": {
            "negligible": (
                abs(fmean(crossbar_gain_pp)) < 1.0
                and max(abs(value) for value in crossbar_gain_pp) < 2.0
            ),
            "floor_crossing_candidate": floor_crossing_candidate,
            "needed_for_this_protocol": (
                None if floor_crossing_candidate else False
            ),
            "requires_independent_qat_frozen_target_and_p0_hash_audit": (
                floor_crossing_candidate
            ),
            "accuracy_floor": 0.90,
            "negligible_mean_gain_pp_threshold": 1.0,
            "negligible_every_endpoint_gain_pp_threshold": 2.0,
        },
        "recovery_policy": recovery_policy,
        "offchip_policy": offchip_policy,
        "layer_scope": layer_scope,
        "checkpoint_policy": "fixed_final_epoch_no_selection",
        "crossbar_network_forward_state": "apparent_q",
        "crossbar_hidden_update_state": "persistent_q",
        "bound_treatment_matched": (
            bound_policy == "winsorize_raw_active_a_to_unit_interval"
        ),
        "network_forward_state_matched": False,
        "device_model_treatment_matched": False,
        "claim_boundary": (
            "paired endpoint labels on topology-specific populations; not shared physical cells; "
            "the architectures share the teacher task and logical dimensions but not an exact "
            "trained weight tensor; the crossbar result uses AIHWKit apparent-q forwards while "
            "the pinned DRN receipt reports persistent physical conductances, so their absolute "
            "accuracies are architecture-specific outcomes rather than a forward-state-matched "
            "device-model contrast; the bounds are matched only under the Winsorized policy"
        ),
    }


def _validate_request(request: Any) -> None:
    if request.teacher_weights is None:
        raise ValueError("Expected --teacher-weights for the crossbar comparator.")
    for name in ("weights", "base_weights", "resume", "device_data", "device_model"):
        if getattr(request, name, None) is not None:
            raise ValueError(
                "Expected the crossbar comparator to accept only --teacher-weights. "
                f"Provided value: --{name.replace('_', '-')} was set."
            )


def run_train(request: "TrainRequest") -> int:
    spec = request.spec
    if not isinstance(spec, CrossbarTrainSpec):
        raise TypeError(
            "Expected mnist_ibm_om_crossbar_relu.v1 train to resolve CrossbarTrainSpec."
        )
    _validate_request(request)
    teacher_path = request.teacher_weights.expanduser().resolve()
    expected_teacher_path = _resolve_repo_path(
        spec.source.expected_teacher_weights_path
    )
    if teacher_path != expected_teacher_path:
        raise ValueError(
            "Expected --teacher-weights to match the pinned source path. "
            f"Provided value: {str(teacher_path)!r}."
        )
    reference_path, reference = _load_drn_reference(spec)
    sampler_value = os.environ.get("EBL_AIHWKIT_PYTHON")
    if not sampler_value:
        raise RuntimeError(
            "Expected EBL_AIHWKIT_PYTHON to name the pinned AIHWKit 1.1.0 interpreter."
        )
    sampler = Path(sampler_value).expanduser().resolve()
    if not sampler.is_file() or not os.access(sampler, os.X_OK):
        raise RuntimeError("Expected EBL_AIHWKIT_PYTHON to identify an executable file.")

    inputs = (
        _input("teacher_weights", teacher_path),
        _input("aihwkit_python", sampler),
        _input("drn_reference", reference_path),
    )
    store = RunStore.create(
        output_root=request.output_dir,
        experiment_id=spec.experiment_id,
        resolved_config=to_plain_data(spec),
        command=request.command,
        repo_root=_ROOT,
        input_artifacts=inputs,
        resume_capability="unsupported",
    )
    try:
        torch.manual_seed(spec.runtime.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(spec.runtime.seed)
        if spec.runtime.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("Expected CUDA to be available for runtime.device='cuda'.")
        device = torch.device(spec.runtime.device)
        teacher, teacher_metadata = _load_teacher(
            teacher_path,
            expected_sha256=spec.source.expected_teacher_weights_sha256,
            device=device,
        )
        layout = build_crossbar_layout(
            spec.model.dims,
            maximum_input_size=spec.model.maximum_input_size,
        )
        logical_weights = tuple(value.detach().cpu().clone() for value in teacher.parameters())
        source_requested, digital_scales = map_logical_weights(
            logical_weights,
            layout,
            weight_scaling_omega=spec.mapping.weight_scaling_omega,
        )
        artifact_root = store.run_dir / "artifacts"
        source_population, source_receipt, source_artifact_paths = _sample_population(
            layout=layout,
            assignment_seed=spec.device.assignment_seed,
            corruption_policy=spec.device.corruption_policy,
            sampler=sampler,
            artifact_root=artifact_root,
            role="source",
            bound_policy=spec.device.bound_policy,
            required_aihwkit_version=spec.runtime.required_aihwkit_version,
        )
        post_fault_population = None
        post_fault_overlay = None
        post_fault_artifact_paths: tuple[tuple[Path, str], ...] = ()
        fault_settings = _post_deployment_fault_settings(spec)
        if spec.recovery.policy in _POST_DEPLOYMENT_FAULT_POLICIES:
            if fault_settings is None:
                raise RuntimeError("Expected post-deployment fault settings.")
            (
                post_fault_population,
                post_fault_receipt,
                sampled_fault_artifacts,
            ) = _sample_population(
                layout=layout,
                assignment_seed=spec.device.assignment_seed,
                corruption_policy="published",
                sampler=sampler,
                artifact_root=artifact_root,
                role=(
                    "star_post_deployment_fault_source"
                    if spec.recovery.policy == "star_local_pulse_sgd"
                    else "post_deployment_fault_source"
                ),
                bound_policy=spec.device.bound_policy,
                required_aihwkit_version=spec.runtime.required_aihwkit_version,
            )
            fault_mask, stuck_persistent_q, fault_report = _matched_published_fault_overlay(
                healthy=source_population,
                published=post_fault_population,
                preset_default_corrupt_devices_prob=(
                    fault_settings.fault_source_preset_default_corrupt_devices_prob
                ),
                enabled_corrupt_devices_prob=(
                    fault_settings.fault_source_enabled_corrupt_devices_prob
                ),
                corrupt_devices_range=(
                    fault_settings.fault_source_corrupt_devices_range
                ),
            )
            post_fault_overlay = {
                "mask": fault_mask,
                "stuck_persistent_q": stuck_persistent_q,
                "report": fault_report,
                "receipt": post_fault_receipt,
            }
            post_fault_artifact_paths = tuple(sampled_fault_artifacts)
        codebook = build_deterministic_effective_codebook(
            source_population,
            maximum_pulses=spec.device.deterministic_codebook_pulses,
        )
        source_continuous = torch.maximum(
            torch.minimum(
                source_requested,
                source_population.logical_max.detach().cpu(),
            ),
            source_population.logical_min.detach().cpu(),
        )
        source_deterministic, source_pulse_indices = project_to_nearest_effective_code(
            codebook,
            source_requested,
        )
        source_mapping_report = _mapping_report(
            population=source_population,
            requested=source_requested,
            continuous=source_continuous,
            codebook=source_deterministic,
            pulse_indices=source_pulse_indices,
            level_counts=codebook.effective_level_counts,
        )

        diagnostic_loaders = build_mnist_loaders(spec.data, data_seed=spec.runtime.data_seed)
        source_requested_validation = _evaluate(
            effective_state=source_requested,
            digital_scales=digital_scales,
            layout=layout,
            teacher=teacher,
            loader=diagnostic_loaders.validation,
            device=device,
            maximum_batches=spec.evaluation.maximum_validation_batches,
            sample_limit=spec.evaluation.sample_limit,
        )
        source_deterministic_validation = _evaluate(
            effective_state=source_deterministic,
            digital_scales=digital_scales,
            layout=layout,
            teacher=teacher,
            loader=diagnostic_loaders.validation,
            device=device,
            maximum_batches=spec.evaluation.maximum_validation_batches,
            sample_limit=spec.evaluation.sample_limit,
        )
        source_continuous_validation = _evaluate(
            effective_state=source_continuous,
            digital_scales=digital_scales,
            layout=layout,
            teacher=teacher,
            loader=diagnostic_loaders.validation,
            device=device,
            maximum_batches=spec.evaluation.maximum_validation_batches,
            sample_limit=spec.evaluation.sample_limit,
        )
        source_requested_test = None
        source_deterministic_test = None
        source_continuous_test = None
        if spec.evaluation.evaluate_test:
            source_requested_test = _evaluate(
                effective_state=source_requested,
                digital_scales=digital_scales,
                layout=layout,
                teacher=teacher,
                loader=diagnostic_loaders.test,
                device=device,
                maximum_batches=None,
                sample_limit=spec.evaluation.sample_limit,
            )
            source_deterministic_test = _evaluate(
                effective_state=source_deterministic,
                digital_scales=digital_scales,
                layout=layout,
                teacher=teacher,
                loader=diagnostic_loaders.test,
                device=device,
                maximum_batches=None,
                sample_limit=spec.evaluation.sample_limit,
            )
            source_continuous_test = _evaluate(
                effective_state=source_continuous,
                digital_scales=digital_scales,
                layout=layout,
                teacher=teacher,
                loader=diagnostic_loaders.test,
                device=device,
                maximum_batches=None,
                sample_limit=spec.evaluation.sample_limit,
            )
        exact_mapping_checks = [source_requested_validation]
        if source_requested_test is not None:
            exact_mapping_checks.append(source_requested_test)
        if any(
            row["student_prediction_sha256"] != row["teacher_prediction_sha256"]
            for row in exact_mapping_checks
        ):
            raise RuntimeError(
                "Expected the requested standard-crossbar mapping to preserve "
                "the frozen teacher predictions before device constraints."
            )

        offchip_loaders = build_mnist_loaders(
            spec.data,
            data_seed=spec.runtime.data_seed,
        )
        requested, offchip_report, offchip_state = _offchip_adapt(
            source_requested=source_requested,
            population=source_population,
            codebook_values=codebook.values,
            spec=spec,
            layout=layout,
            digital_scales=digital_scales,
            teacher=teacher,
            validation_loader=offchip_loaders.validation,
            train_loader=offchip_loaders.train,
            device=device,
        )
        offchip_state_path = artifact_root / "offchip_state.pt"
        atomic_torch_save(offchip_state, offchip_state_path)

        continuous = torch.maximum(
            torch.minimum(requested, source_population.logical_max.detach().cpu()),
            source_population.logical_min.detach().cpu(),
        )
        deterministic, pulse_indices = project_to_nearest_effective_code(
            codebook,
            requested,
        )
        if spec.offchip.policy == "continuous_hwa" and not torch.equal(
            requested,
            continuous,
        ):
            raise RuntimeError("Expected continuous HWA to deploy its exact support-clamped state.")
        if spec.offchip.policy == "deterministic_qat" and not torch.equal(
            requested,
            deterministic,
        ):
            raise RuntimeError("Expected deterministic QAT to deploy its exact codebook state.")
        mapping_report = _mapping_report(
            population=source_population,
            requested=requested,
            continuous=continuous,
            codebook=deterministic,
            pulse_indices=pulse_indices,
            level_counts=codebook.effective_level_counts,
        )
        requested_validation = _evaluate(
            effective_state=requested,
            digital_scales=digital_scales,
            layout=layout,
            teacher=teacher,
            loader=diagnostic_loaders.validation,
            device=device,
            maximum_batches=spec.evaluation.maximum_validation_batches,
            sample_limit=spec.evaluation.sample_limit,
        )
        deterministic_validation = _evaluate(
            effective_state=deterministic,
            digital_scales=digital_scales,
            layout=layout,
            teacher=teacher,
            loader=diagnostic_loaders.validation,
            device=device,
            maximum_batches=spec.evaluation.maximum_validation_batches,
            sample_limit=spec.evaluation.sample_limit,
        )
        continuous_validation = _evaluate(
            effective_state=continuous,
            digital_scales=digital_scales,
            layout=layout,
            teacher=teacher,
            loader=diagnostic_loaders.validation,
            device=device,
            maximum_batches=spec.evaluation.maximum_validation_batches,
            sample_limit=spec.evaluation.sample_limit,
        )
        requested_test = None
        deterministic_test = None
        continuous_test = None
        if spec.evaluation.evaluate_test:
            requested_test = _evaluate(
                effective_state=requested,
                digital_scales=digital_scales,
                layout=layout,
                teacher=teacher,
                loader=diagnostic_loaders.test,
                device=device,
                maximum_batches=None,
                sample_limit=spec.evaluation.sample_limit,
            )
            deterministic_test = _evaluate(
                effective_state=deterministic,
                digital_scales=digital_scales,
                layout=layout,
                teacher=teacher,
                loader=diagnostic_loaders.test,
                device=device,
                maximum_batches=None,
                sample_limit=spec.evaluation.sample_limit,
            )
            continuous_test = _evaluate(
                effective_state=continuous,
                digital_scales=digital_scales,
                layout=layout,
                teacher=teacher,
                loader=diagnostic_loaders.test,
                device=device,
                maximum_batches=None,
                sample_limit=spec.evaluation.sample_limit,
            )

        target_contexts: list[dict[str, Any]] = []
        target_artifact_paths: list[tuple[Path, str]] = []
        if spec.transfer.enabled:
            for target_index, target in enumerate(spec.transfer.targets):
                role = f"transfer_target_{target_index}_assignment_{target.assignment_seed}"
                population, receipt, artifact_paths = _sample_population(
                    layout=layout,
                    assignment_seed=target.assignment_seed,
                    corruption_policy=spec.device.corruption_policy,
                    sampler=sampler,
                    artifact_root=artifact_root,
                    role=role,
                    bound_policy=spec.device.bound_policy,
                    required_aihwkit_version=spec.runtime.required_aihwkit_version,
                )
                codebook_value = build_deterministic_effective_codebook(
                    population,
                    maximum_pulses=spec.device.deterministic_codebook_pulses,
                )
                target_contexts.append(
                    {
                        "index": target_index,
                        "settings": target,
                        "population": population,
                        "receipt": receipt,
                        "codebook": codebook_value,
                    }
                )
                target_artifact_paths.extend(artifact_paths)

        endpoint_rows: list[dict[str, Any]] = []
        checkpoint_paths: list[Path] = []
        post_fault_runtime_artifact_paths: list[tuple[Path, str]] = []
        for endpoint_index, endpoint_seed in enumerate(spec.device.endpoint_seeds):
            loaders = build_mnist_loaders(
                spec.data,
                data_seed=spec.runtime.data_seed,
                calibration_examples=(
                    spec.recovery.star.calibration_examples
                    if spec.recovery.star is not None
                    else 1024
                ),
            )
            recovery_train_loader = loaders.train
            if spec.recovery.policy == "supervised_ce_pulse_adam":
                bp_settings = spec.recovery.supervised_bp
                if bp_settings is None:  # pragma: no cover - parser invariant
                    raise RuntimeError("Expected supervised-BP recovery settings.")
                full_training_examples = 60_000 - spec.data.validation_points
                repair_data = replace(
                    spec.data,
                    num_points=(
                        None
                        if bp_settings.repair_examples == full_training_examples
                        else bp_settings.repair_examples
                    ),
                )
                recovery_train_loader = build_mnist_loaders(
                    repair_data,
                    data_seed=spec.runtime.data_seed,
                ).train
            plant, program_report = _program_endpoint(
                population=source_population,
                requested=requested,
                assignment_seed=spec.device.assignment_seed,
                endpoint_seed=endpoint_seed,
                maximum_pulses=spec.device.maximum_programming_pulses,
                tolerance_x=spec.device.verify_tolerance_x,
                device=device,
                stream_role="source_program_verify",
                random_stream_fingerprint=source_receipt["bound_treatment"][
                    "source_population_fingerprint"
                ],
            )
            p0_state = plant.state_dict()
            post_fault_context = None
            if spec.recovery.policy in _POST_DEPLOYMENT_FAULT_POLICIES:
                if post_fault_overlay is None or post_fault_population is None:
                    raise RuntimeError(
                        "Expected the matched published post-deployment fault source."
                    )
                endpoint_artifact_root = artifact_root / "endpoints" / str(endpoint_seed)
                healthy_p0_path = endpoint_artifact_root / "healthy_p0.pt"
                atomic_torch_save(p0_state, healthy_p0_path)
                faulted_checkpoint_path = (
                    endpoint_artifact_root / "faulted_pre_recovery.pt"
                )
                post_fault_context = {
                    "faulted_checkpoint_path": faulted_checkpoint_path,
                    "fault_mask": post_fault_overlay["mask"],
                    "stuck_persistent_q": post_fault_overlay[
                        "stuck_persistent_q"
                    ],
                    "fault_report": post_fault_overlay["report"],
                    "fault_source_population_fingerprint": (
                        post_fault_population.fingerprint
                    ),
                    "predeployment_training_cohort": offchip_report[
                        "first_epoch_training_cohort"
                    ],
                }
                post_fault_runtime_artifact_paths.append(
                    (
                        healthy_p0_path,
                        (
                            "star_healthy_p0_state"
                            if spec.recovery.policy == "star_local_pulse_sgd"
                            else "post_fault_healthy_p0_state"
                        ),
                    )
                )
                if spec.recovery.policy == "star_local_pulse_sgd":
                    star_target_path = endpoint_artifact_root / "star_targets.npz"
                    hardware_instance_id = content_hash(
                        {
                            "assignment_seed": spec.device.assignment_seed,
                            "endpoint_seed": endpoint_seed,
                            "population_fingerprint": source_population.fingerprint,
                        }
                    )
                    post_fault_context.update(
                        {
                            "artifact_path": star_target_path,
                            "model_checkpoint_sha256": sha256_file(
                                offchip_state_path
                            ),
                            "healthy_deployment_artifact_sha256": sha256_file(
                                healthy_p0_path
                            ),
                            "hardware_instance_id": hardware_instance_id,
                            "config_sha256": str(
                                store.manifest["config"]["sha256"]
                            ),
                        }
                    )
            p0_evaluation = _evaluate_plant_states(
                plant=plant,
                digital_scales=digital_scales,
                layout=layout,
                teacher=teacher,
                loader=loaders.validation,
                device=device,
                maximum_batches=spec.evaluation.maximum_validation_batches,
                sample_limit=spec.evaluation.sample_limit,
            )
            recovery, final_state = _recover(
                plant=plant,
                spec=spec,
                endpoint_seed=endpoint_seed,
                layout=layout,
                digital_scales=digital_scales,
                teacher=teacher,
                validation_loader=loaders.validation,
                train_loader=recovery_train_loader,
                device=device,
                calibration_loader=loaders.calibration,
                post_fault_context=post_fault_context,
            )
            if spec.recovery.policy == "star_local_pulse_sgd":
                if post_fault_context is None:  # pragma: no cover - branch invariant
                    raise RuntimeError("Expected materialized STAR artifacts.")
                star_target_path = Path(post_fault_context["artifact_path"])
                post_fault_runtime_artifact_paths.extend(
                    [
                        (star_target_path, "star_state_targets"),
                        (
                            star_target_path.with_suffix(".receipt.json"),
                            "star_state_targets_receipt",
                        ),
                    ]
                )
            if spec.recovery.policy in _POST_DEPLOYMENT_FAULT_POLICIES:
                if post_fault_context is None:  # pragma: no cover - branch invariant
                    raise RuntimeError("Expected materialized post-fault artifacts.")
                post_fault_runtime_artifact_paths.append(
                    (
                        Path(post_fault_context["faulted_checkpoint_path"]),
                        (
                            "star_faulted_pre_recovery_state"
                            if spec.recovery.policy == "star_local_pulse_sgd"
                            else "post_fault_pre_recovery_state"
                        ),
                    )
                )
            p0_test_evaluation = None
            faulted_test_evaluation = None
            final_test_evaluation = None
            if spec.evaluation.evaluate_test:
                p0_plant_generator = torch.Generator(device=device.type)
                p0_plant_generator.manual_seed(1)
                p0_plant = IbmOmEffectiveCrossbarPlant(
                    source_population,
                    generator=p0_plant_generator,
                    device=device,
                )
                p0_plant.load_state_dict(p0_state)
                p0_test_evaluation = _evaluate_plant_states(
                    plant=p0_plant,
                    digital_scales=digital_scales,
                    layout=layout,
                    teacher=teacher,
                    loader=loaders.test,
                    device=device,
                    maximum_batches=None,
                    sample_limit=spec.evaluation.sample_limit,
                )
                if spec.recovery.policy in _POST_DEPLOYMENT_FAULT_POLICIES:
                    if post_fault_context is None:
                        raise RuntimeError("Expected a saved post-fault state.")
                    faulted_state = torch.load(
                        Path(post_fault_context["faulted_checkpoint_path"]),
                        map_location="cpu",
                        weights_only=True,
                    )
                    faulted_generator = torch.Generator(device=device.type)
                    faulted_generator.manual_seed(1)
                    faulted_plant = IbmOmEffectiveCrossbarPlant(
                        source_population,
                        generator=faulted_generator,
                        device=device,
                    )
                    faulted_plant.load_state_dict(
                        faulted_state,
                        expected_fault_source_population=post_fault_population,
                        expected_pre_fault_state=p0_state,
                    )
                    faulted_test_evaluation = _evaluate_plant_states(
                        plant=faulted_plant,
                        digital_scales=digital_scales,
                        layout=layout,
                        teacher=teacher,
                        loader=loaders.test,
                        device=device,
                        maximum_batches=None,
                        sample_limit=spec.evaluation.sample_limit,
                    )
                final_test_evaluation = _evaluate_plant_states(
                    plant=plant,
                    digital_scales=digital_scales,
                    layout=layout,
                    teacher=teacher,
                    loader=loaders.test,
                    device=device,
                    maximum_batches=None,
                    sample_limit=spec.evaluation.sample_limit,
                )

            transfer_reports: list[dict[str, Any]] = []
            if spec.transfer.enabled:
                for context in target_contexts:
                    target = context["settings"]
                    target_population = context["population"]
                    target_codebook = context["codebook"]
                    target_receipt = context["receipt"]
                    target_seed = target.endpoint_seeds[endpoint_index]
                    transfer_requested, transfer_source = _map_transfer_source_state(
                        source_state=spec.transfer.source_state,
                        offchip_policy=spec.offchip.policy,
                        offchip_state=offchip_state,
                        source_plant=plant,
                        target_population=target_population,
                        target_codebook=target_codebook,
                    )
                    transfer_continuous = torch.maximum(
                        torch.minimum(
                            transfer_requested,
                            target_population.logical_max.detach().cpu(),
                        ),
                        target_population.logical_min.detach().cpu(),
                    )
                    transfer_deterministic, transfer_indices = (
                        project_to_nearest_effective_code(
                            target_codebook,
                            transfer_requested,
                        )
                    )
                    no_write_validation = _evaluate(
                        effective_state=transfer_requested,
                        digital_scales=digital_scales,
                        layout=layout,
                        teacher=teacher,
                        loader=loaders.validation,
                        device=device,
                        maximum_batches=spec.evaluation.maximum_validation_batches,
                        sample_limit=spec.evaluation.sample_limit,
                    )
                    transfer_plant, transfer_program = _program_endpoint(
                        population=target_population,
                        requested=transfer_requested,
                        assignment_seed=target.assignment_seed,
                        endpoint_seed=target_seed,
                        maximum_pulses=spec.device.maximum_programming_pulses,
                        tolerance_x=spec.device.verify_tolerance_x,
                        device=device,
                        stream_role="fresh_transfer_program_verify",
                        random_stream_fingerprint=target_receipt["bound_treatment"][
                            "source_population_fingerprint"
                        ],
                    )
                    pv_validation = _evaluate_plant_states(
                        plant=transfer_plant,
                        digital_scales=digital_scales,
                        layout=layout,
                        teacher=teacher,
                        loader=loaders.validation,
                        device=device,
                        maximum_batches=spec.evaluation.maximum_validation_batches,
                        sample_limit=spec.evaluation.sample_limit,
                    )
                    no_write_test = None
                    pv_test = None
                    if spec.evaluation.evaluate_test:
                        no_write_test = _evaluate(
                            effective_state=transfer_requested,
                            digital_scales=digital_scales,
                            layout=layout,
                            teacher=teacher,
                            loader=loaders.test,
                            device=device,
                            maximum_batches=None,
                            sample_limit=spec.evaluation.sample_limit,
                        )
                        pv_test = _evaluate_plant_states(
                            plant=transfer_plant,
                            digital_scales=digital_scales,
                            layout=layout,
                            teacher=teacher,
                            loader=loaders.test,
                            device=device,
                            maximum_batches=None,
                            sample_limit=spec.evaluation.sample_limit,
                        )
                    transfer_reports.append(
                        {
                            "target_index": context["index"],
                            "target_assignment_seed": target.assignment_seed,
                            "source_endpoint_seed": endpoint_seed,
                            "target_endpoint_seed": target_seed,
                            "source": transfer_source,
                            "mapping": _mapping_report(
                                population=target_population,
                                requested=transfer_requested,
                                continuous=transfer_continuous,
                                codebook=transfer_deterministic,
                                pulse_indices=transfer_indices,
                                level_counts=target_codebook.effective_level_counts,
                            ),
                            "no_write_mapped_validation": no_write_validation,
                            "programming": transfer_program,
                            "apparent_pv_validation": pv_validation[
                                "apparent_forward"
                            ],
                            "persistent_pv_validation": pv_validation[
                                "persistent_diagnostic"
                            ],
                            "no_write_mapped_test": no_write_test,
                            "apparent_pv_test": (
                                None if pv_test is None else pv_test["apparent_forward"]
                            ),
                            "persistent_pv_test": (
                                None
                                if pv_test is None
                                else pv_test["persistent_diagnostic"]
                            ),
                        }
                    )

            checkpoint_path = artifact_root / "endpoints" / str(endpoint_seed) / "states.pt"
            atomic_torch_save(
                {
                    "schema": "ebl.ibm_om_crossbar_endpoint_states",
                    "schema_version": 1,
                    "assignment_seed": spec.device.assignment_seed,
                    "endpoint_seed": endpoint_seed,
                    "population_fingerprint": source_population.fingerprint,
                    "offchip_policy": spec.offchip.policy,
                    "source_requested_sha256": tensor_sha256(source_requested),
                    "requested_sha256": tensor_sha256(requested),
                    "offchip_checkpoint_sha256": sha256_file(offchip_state_path),
                    "p0": p0_state,
                    "fixed_final": final_state,
                    "fixed_final_epoch": recovery["fixed_final_epoch"],
                },
                checkpoint_path,
            )
            checkpoint_paths.append(checkpoint_path)
            row = {
                "endpoint_seed": endpoint_seed,
                "deployment_requested_sha256": tensor_sha256(requested),
                "programming": program_report,
                "p0_apparent_sha256": program_report["apparent_sha256"],
                "p0_persistent_sha256": program_report["persistent_sha256"],
                "p0_apparent_validation": p0_evaluation["apparent_forward"],
                "p0_persistent_validation": p0_evaluation[
                    "persistent_diagnostic"
                ],
                "p0_apparent_test": (
                    None
                    if p0_test_evaluation is None
                    else p0_test_evaluation["apparent_forward"]
                ),
                "p0_persistent_test": (
                    None
                    if p0_test_evaluation is None
                    else p0_test_evaluation["persistent_diagnostic"]
                ),
                "faulted_pre_recovery_apparent_test": (
                    None
                    if faulted_test_evaluation is None
                    else faulted_test_evaluation["apparent_forward"]
                ),
                "faulted_pre_recovery_persistent_test": (
                    None
                    if faulted_test_evaluation is None
                    else faulted_test_evaluation["persistent_diagnostic"]
                ),
                "recovery": recovery,
                "fixed_final_apparent_test": (
                    None
                    if final_test_evaluation is None
                    else final_test_evaluation["apparent_forward"]
                ),
                "fixed_final_persistent_test": (
                    None
                    if final_test_evaluation is None
                    else final_test_evaluation["persistent_diagnostic"]
                ),
                "fixed_final_apparent_sha256": tensor_sha256(plant.apparent),
                "fixed_final_persistent_sha256": tensor_sha256(plant.persistent),
                "fixed_final_plant_pulses": plant.pulse_statistics(),
                "fresh_transfers": transfer_reports,
                "checkpoint_sha256": sha256_file(checkpoint_path),
            }
            endpoint_rows.append(row)
            store.append_metric(
                {
                    "mode": "endpoint",
                    "endpoint_seed": endpoint_seed,
                    "network_forward_state": "apparent_q",
                    "p0_apparent_validation_accuracy": p0_evaluation[
                        "apparent_forward"
                    ]["student_accuracy"],
                    "p0_persistent_validation_accuracy": p0_evaluation[
                        "persistent_diagnostic"
                    ]["student_accuracy"],
                    "fixed_final_apparent_validation_accuracy": recovery[
                        "fixed_final_apparent_validation"
                    ]["student_accuracy"],
                    "fixed_final_persistent_validation_accuracy": recovery[
                        "fixed_final_persistent_validation"
                    ]["student_accuracy"],
                    "p0_apparent_test_accuracy": (
                        None
                        if p0_test_evaluation is None
                        else p0_test_evaluation["apparent_forward"]["student_accuracy"]
                    ),
                    "fixed_final_apparent_test_accuracy": (
                        None
                        if final_test_evaluation is None
                        else final_test_evaluation["apparent_forward"][
                            "student_accuracy"
                        ]
                    ),
                    "faulted_pre_recovery_apparent_test_accuracy": (
                        None
                        if faulted_test_evaluation is None
                        else faulted_test_evaluation["apparent_forward"][
                            "student_accuracy"
                        ]
                    ),
                }
            )

        aggregates: dict[str, Any] = {
            "network_forward_state": "apparent_q",
            "hidden_update_state": "persistent_q",
            "p0_apparent_validation_accuracy": _aggregate(
                endpoint_rows, ("p0_apparent_validation", "student_accuracy")
            ),
            "p0_persistent_validation_accuracy": _aggregate(
                endpoint_rows, ("p0_persistent_validation", "student_accuracy")
            ),
            "fixed_final_apparent_validation_accuracy": _aggregate(
                endpoint_rows,
                ("recovery", "fixed_final_apparent_validation", "student_accuracy"),
            ),
            "fixed_final_persistent_validation_accuracy": _aggregate(
                endpoint_rows,
                ("recovery", "fixed_final_persistent_validation", "student_accuracy"),
            ),
            "p0_programming_pulses": _aggregate(
                endpoint_rows, ("programming", "total_programming_pulses")
            ),
            "p0_verify_reads": _aggregate(
                endpoint_rows, ("programming", "verify_reads")
            ),
        }
        if spec.recovery.policy in _POST_DEPLOYMENT_FAULT_POLICIES:
            aggregates.update(
                {
                    "recovery_commanded_pulses": _aggregate(
                        endpoint_rows,
                        ("recovery", "optimizer", "commanded_pulses"),
                    ),
                    "recovery_commanded_cells": _aggregate(
                        endpoint_rows,
                        ("recovery", "optimizer", "commanded_cells"),
                    ),
                    "recovery_commands_to_immutable_cells": _aggregate(
                        endpoint_rows,
                        (
                            "recovery",
                            "posthoc_pulse_effects",
                            "commands_to_immutable_cells",
                        ),
                    ),
                    "recovery_persistent_changed_healthy_cells": _aggregate(
                        endpoint_rows,
                        (
                            "recovery",
                            "posthoc_pulse_effects",
                            "persistent_changed_healthy_cells",
                        ),
                    ),
                    "healthy_pre_fault_apparent_validation_accuracy": _aggregate(
                        endpoint_rows,
                        (
                            "recovery",
                            "healthy_pre_fault_apparent_validation",
                            "student_accuracy",
                        ),
                    ),
                    "faulted_pre_recovery_apparent_validation_accuracy": _aggregate(
                        endpoint_rows,
                        (
                            "recovery",
                            "faulted_pre_recovery_apparent_validation",
                            "student_accuracy",
                        ),
                    ),
                    "paired_post_fault_validation_recovery_gain_percentage_points": (
                        _numeric_summary(
                            100.0
                            * (
                                float(
                                    row["recovery"][
                                        "fixed_final_apparent_validation"
                                    ]["student_accuracy"]
                                )
                                - float(
                                    row["recovery"][
                                        "faulted_pre_recovery_apparent_validation"
                                    ]["student_accuracy"]
                                )
                            )
                            for row in endpoint_rows
                        )
                    ),
                }
            )
            if spec.recovery.policy == "star_local_pulse_sgd":
                aggregates.update(
                    {
                        "paired_star_validation_recovery_gain_percentage_points": (
                            aggregates[
                                "paired_post_fault_validation_recovery_gain_percentage_points"
                            ]
                        ),
                        "healthy_pre_fault_local_objective": _aggregate(
                            endpoint_rows,
                            (
                                "recovery",
                                "healthy_pre_fault_local_state_errors",
                                "mean_local_objective",
                            ),
                        ),
                        "faulted_pre_recovery_local_objective": _aggregate(
                            endpoint_rows,
                            (
                                "recovery",
                                "faulted_pre_recovery_local_state_errors",
                                "mean_local_objective",
                            ),
                        ),
                        "fixed_final_local_objective": _aggregate(
                            endpoint_rows,
                            (
                                "recovery",
                                "fixed_final_local_state_errors",
                                "mean_local_objective",
                            ),
                        ),
                    }
                )
        else:
            aggregates.update(
                {
                    "recovery_applied_pulses": _aggregate(
                        endpoint_rows,
                        ("recovery", "optimizer", "applied_pulses"),
                    ),
                    "recovery_changed_cells": _aggregate(
                        endpoint_rows,
                        ("recovery", "optimizer", "changed_cells"),
                    ),
                }
            )
        comparison = None
        if spec.evaluation.evaluate_test:
            aggregates["p0_apparent_test_accuracy"] = _aggregate(
                endpoint_rows, ("p0_apparent_test", "student_accuracy")
            )
            aggregates["p0_persistent_test_accuracy"] = _aggregate(
                endpoint_rows, ("p0_persistent_test", "student_accuracy")
            )
            aggregates["fixed_final_apparent_test_accuracy"] = _aggregate(
                endpoint_rows, ("fixed_final_apparent_test", "student_accuracy")
            )
            aggregates["fixed_final_persistent_test_accuracy"] = _aggregate(
                endpoint_rows, ("fixed_final_persistent_test", "student_accuracy")
            )
            recovery_baseline_key = (
                "faulted_pre_recovery_apparent_test"
                if spec.recovery.policy in _POST_DEPLOYMENT_FAULT_POLICIES
                else "p0_apparent_test"
            )
            aggregates["paired_recovery_gain_percentage_points"] = _numeric_summary(
                100.0
                * (
                    float(row["fixed_final_apparent_test"]["student_accuracy"])
                    - float(row[recovery_baseline_key]["student_accuracy"])
                )
                for row in endpoint_rows
            )
            if spec.recovery.policy in _POST_DEPLOYMENT_FAULT_POLICIES:
                aggregates["faulted_pre_recovery_apparent_test_accuracy"] = _aggregate(
                    endpoint_rows,
                    ("faulted_pre_recovery_apparent_test", "student_accuracy"),
                )
                aggregates["faulted_pre_recovery_persistent_test_accuracy"] = _aggregate(
                    endpoint_rows,
                    ("faulted_pre_recovery_persistent_test", "student_accuracy"),
                )
                damages = [
                    float(row["p0_apparent_test"]["student_accuracy"])
                    - float(
                        row["faulted_pre_recovery_apparent_test"][
                            "student_accuracy"
                        ]
                    )
                    for row in endpoint_rows
                ]
                gains = [
                    float(row["fixed_final_apparent_test"]["student_accuracy"])
                    - float(
                        row["faulted_pre_recovery_apparent_test"][
                            "student_accuracy"
                        ]
                    )
                    for row in endpoint_rows
                ]
                positive_damage_fractions = [
                    gain / damage
                    for gain, damage in zip(gains, damages, strict=True)
                    if damage > 0.0
                ]
                aggregates.update(
                    {
                        "paired_fault_damage_percentage_points": _numeric_summary(
                            100.0 * value for value in damages
                        ),
                        "paired_recovery_fraction": (
                            None
                            if not positive_damage_fractions
                            else _numeric_summary(positive_damage_fractions)
                        ),
                        "post_fault_recovery_endpoint_counts": {
                            "endpoints": len(endpoint_rows),
                            "positive_gain": sum(value > 0.0 for value in gains),
                            "zero_gain": sum(value == 0.0 for value in gains),
                            "negative_gain": sum(value < 0.0 for value in gains),
                            "loss_greater_than_two_percentage_points": sum(
                                value < -0.02 for value in gains
                            ),
                        },
                    }
                )
                if spec.recovery.policy == "star_local_pulse_sgd":
                    aggregates["star_recovery_endpoint_counts"] = aggregates[
                        "post_fault_recovery_endpoint_counts"
                    ]
            if (
                spec.evaluation.sample_limit is None
                and spec.recovery.policy not in _POST_DEPLOYMENT_FAULT_POLICIES
            ):
                comparison = _drn_comparison(
                    rows=endpoint_rows,
                    reference=reference,
                    recovery_policy=spec.recovery.policy,
                    bound_policy=spec.device.bound_policy,
                    offchip_policy=spec.offchip.policy,
                    layer_scope=spec.recovery.layer_scope,
                    crossbar_deployment_source_accuracy=float(
                        requested_test["student_accuracy"]
                    ),
                )

        transfer_aggregates = None
        if spec.transfer.enabled:
            all_transfers = [
                transfer
                for row in endpoint_rows
                for transfer in row["fresh_transfers"]
            ]

            def summarize_transfers(rows: list[dict[str, Any]]) -> dict[str, Any]:
                result: dict[str, Any] = {
                    "pairs": len(rows),
                    "no_write_mapped_validation_accuracy": _aggregate(
                        rows,
                        ("no_write_mapped_validation", "student_accuracy"),
                    ),
                    "apparent_pv_validation_accuracy": _aggregate(
                        rows, ("apparent_pv_validation", "student_accuracy")
                    ),
                    "persistent_pv_validation_accuracy": _aggregate(
                        rows, ("persistent_pv_validation", "student_accuracy")
                    ),
                    "programming_pulses": _aggregate(
                        rows, ("programming", "total_programming_pulses")
                    ),
                }
                if spec.evaluation.evaluate_test:
                    result.update(
                        {
                            "no_write_mapped_test_accuracy": _aggregate(
                                rows,
                                ("no_write_mapped_test", "student_accuracy"),
                            ),
                            "apparent_pv_test_accuracy": _aggregate(
                                rows, ("apparent_pv_test", "student_accuracy")
                            ),
                            "persistent_pv_test_accuracy": _aggregate(
                                rows, ("persistent_pv_test", "student_accuracy")
                            ),
                        }
                    )
                return result

            per_target = []
            for context in target_contexts:
                target_rows = [
                    value
                    for value in all_transfers
                    if value["target_index"] == context["index"]
                ]
                per_target.append(
                    {
                        "target_index": context["index"],
                        "assignment_seed": context["settings"].assignment_seed,
                        "endpoint_seeds": list(
                            context["settings"].endpoint_seeds
                        ),
                        "metrics": summarize_transfers(target_rows),
                    }
                )
            transfer_aggregates = {
                "source_state": spec.transfer.source_state,
                "network_forward_state": "apparent_q",
                "per_target": per_target,
                "pooled": summarize_transfers(all_transfers),
            }
            aggregates["fresh_transfer"] = transfer_aggregates

        summary = {
            "schema": "ebl.analysis.mnist_ibm_om_crossbar_relu",
            "schema_version": 1,
            "evidence_tier": "exploratory_noncanonical",
            "architecture": {
                "logical_dims": list(spec.model.dims),
                "logical_weights": sum(tile.cells for tile in layout),
                "programmable_active_states": source_population.size,
                "fixed_sampled_reference_values": source_population.size,
                "tile_layout": [
                    {"key": tile.key, "shape": list(tile.shape)} for tile in layout
                ],
                "forward": "analog_MVM(q=a-r)-digital_ReLU-analog_MVM(q=a-r)",
                "network_forward_state": "apparent_q",
                "hidden_device_update_state": "persistent_q",
                "recovery_gradient_handoff": (
                    "identity_apparent_local_error_to_persistent_pulse_update"
                    if spec.recovery.policy == "star_local_pulse_sgd"
                    else (
                        "autograd_backprop_apparent_q_to_persistent_pulse_update"
                        if spec.recovery.policy == "supervised_ce_pulse_adam"
                        else "identity_ste_apparent_q_to_persistent_pulse_update"
                    )
                ),
                "star_local_recovery": (
                    None
                    if spec.recovery.policy != "star_local_pulse_sgd"
                    else {
                        "hidden_error": "rho_h*(h-mu_y_h)*relu_mask",
                        "output_error": "rho_o*(z-mu_y_o)",
                        "transpose_mvm": False,
                        "cross_layer_backpropagation": False,
                        "label_role": "class_conditional_target_address",
                    }
                ),
                "ordinary_supervised_bp_recovery": (
                    None
                    if spec.recovery.policy != "supervised_ce_pulse_adam"
                    else {
                        "loss": "ground_truth_label_cross_entropy",
                        "transpose_mvm": True,
                        "cross_layer_backpropagation": True,
                        "autograd": True,
                        "digital_adam_moments": True,
                        "teacher_access_during_updates": False,
                    }
                ),
                "passive_kcl_denominator": "not_applicable_standard_crossbar_MVM",
            },
            "source": {
                "teacher_sha256": sha256_file(teacher_path),
                "teacher_metadata": teacher_metadata,
                "logical_weight_sha256": [tensor_sha256(value) for value in logical_weights],
                "requested_q_sha256": tensor_sha256(source_requested),
                "digital_scales": list(digital_scales),
            },
            "device": {
                "preset": spec.device.preset,
                "aihwkit_version": source_population.aihwkit_version,
                "evidence_class": spec.device.evidence_class,
                "corruption_policy": spec.device.corruption_policy,
                "bound_policy": spec.device.bound_policy,
                "assignment_seed": spec.device.assignment_seed,
                "population_fingerprint": source_population.fingerprint,
                "population_sampling_receipt": source_receipt,
                "defects": _population_defect_report(source_population),
                "deterministic_codebook": {
                    "maximum_set_pulses": codebook.maximum_pulses,
                    "values_sha256": tensor_sha256(codebook.values),
                    "effective_level_counts_sha256": tensor_sha256(
                        codebook.effective_level_counts
                    ),
                    "cycle_to_cycle_random_term": 0.0,
                    "apparent_write_noise": 0.0,
                    "tie_rule": spec.mapping.codebook_tie_rule,
                },
                "absolute_conductance_calibration": None,
                "power_claim": "not_available_without_absolute_conductance_and_periphery_model",
                "post_deployment_fault_source": (
                    None
                    if post_fault_overlay is None
                    else {
                        "report": post_fault_overlay["report"],
                        "population_fingerprint": (
                            post_fault_population.fingerprint
                            if post_fault_population is not None
                            else None
                        ),
                        "sampling_receipt": post_fault_overlay["receipt"],
                    }
                ),
                "post_deployment_star_fault_source": (
                    None
                    if spec.recovery.policy != "star_local_pulse_sgd"
                    or post_fault_overlay is None
                    else {
                        "report": post_fault_overlay["report"],
                        "population_fingerprint": (
                            post_fault_population.fingerprint
                            if post_fault_population is not None
                            else None
                        ),
                        "sampling_receipt": post_fault_overlay["receipt"],
                    }
                ),
            },
            "source_mapping": source_mapping_report,
            "source_deterministic_controls": {
                "requested_exact_mapping_validation": source_requested_validation,
                "continuous_envelope_validation": source_continuous_validation,
                "deterministic_codebook_validation": source_deterministic_validation,
                "requested_exact_mapping_test": source_requested_test,
                "continuous_envelope_test": source_continuous_test,
                "deterministic_codebook_test": source_deterministic_test,
                "optimizer_updates": 0,
                "stochastic_programming": False,
            },
            "offchip": {
                "settings": to_plain_data(spec.offchip),
                "report": offchip_report,
                "checkpoint_sha256": sha256_file(offchip_state_path),
            },
            "deployment_mapping": mapping_report,
            "deployment_deterministic_controls": {
                "requested_exact_mapping_validation": requested_validation,
                "continuous_envelope_validation": continuous_validation,
                "deterministic_codebook_validation": deterministic_validation,
                "requested_exact_mapping_test": requested_test,
                "continuous_envelope_test": continuous_test,
                "deterministic_codebook_test": deterministic_test,
                "optimizer_updates": 0,
                "stochastic_programming": False,
            },
            "recovery": to_plain_data(spec.recovery),
            "endpoints": endpoint_rows,
            "aggregates": aggregates,
            "drn_comparison": comparison,
            "drn_reference": reference,
            "transfer": {
                "settings": to_plain_data(spec.transfer),
                "source_state": spec.transfer.source_state,
                "target_populations": [
                    {
                        "target_index": context["index"],
                        "assignment_seed": context["settings"].assignment_seed,
                        "endpoint_seeds": list(context["settings"].endpoint_seeds),
                        "fingerprint": context["population"].fingerprint,
                        "sampling_receipt": context["receipt"],
                        "defects": _population_defect_report(
                            context["population"]
                        ),
                        "deterministic_codebook_values_sha256": tensor_sha256(
                            context["codebook"].values
                        ),
                    }
                    for context in target_contexts
                ],
                "aggregates": transfer_aggregates,
            },
            "limitations": [
                "model_based_AIHWKit_OM_preset_not_raw_measured_device_data",
                "standard_MVM_has_no_DRN_passive_sum_loading_or_equilibrium_voltage",
                "open_loop_single_pulse_emulator_not_native_parallel_outer_product_tile",
                "no_inference_read_noise_retention_line_resistance_or_absolute_power",
                "source_endpoint_seeds_share_one_assignment_and_are_not_independent_arrays",
                "target_endpoint_seeds_within_each_target_share_one_assignment",
                *(
                    [
                        "digital_teacher_KL_gradients_and_Adam_moments_remain_off_array",
                        "recovery_gradient_uses_identity_ste_from_apparent_to_hidden_persistent_state",
                    ]
                    if spec.recovery.policy
                    not in _POST_DEPLOYMENT_FAULT_POLICIES
                    else []
                ),
                *(
                    [
                        (
                            f"predeployment_{spec.offchip.policy}_uses_off_array_teacher_KL_autograd_and_Adam_but_the_local_post_fault_update_does_not"
                            if spec.offchip.policy != "none"
                            else "teacher_checkpoint_initializes_and_evaluates_the_source_but_is_absent_from_the_local_post_fault_update"
                        ),
                        "star_inspired_recovery_is_teacher_free_and_moment_free_but_software_emulates_local_subtraction_relu_mask_and_outer_product_pulses",
                        "star_fault_transition_replays_explicitly_enabled_AIHWKit_published_OM_corrupt_cells_after_healthy_programming",
                        "corrupt_persistent_q_is_stuck_but_apparent_q_retains_AIHWKit_write_noise_resampling",
                        "star_labels_address_sram_targets_and_are_not_generated_on_chip",
                    ]
                    if spec.recovery.policy == "star_local_pulse_sgd"
                    else []
                ),
                *(
                    [
                        f"predeployment_{spec.offchip.policy}_uses_off_array_teacher_KL_autograd_and_Adam",
                        "post_fault_retraining_uses_ground_truth_cross_entropy_full_network_autograd_backprop_and_digital_Adam",
                        "teacher_is_used_for_source_initialization_predeployment_HWA_and_diagnostics_but_not_post_fault_updates",
                        "ordinary_BP_recovery_is_a_privileged_algorithmic_control_not_fully_on_chip",
                        "post_deployment_fault_transition_replays_explicitly_enabled_AIHWKit_published_OM_corrupt_cells_after_healthy_programming",
                        "corrupt_persistent_q_is_stuck_but_apparent_q_retains_AIHWKit_write_noise_resampling",
                    ]
                    if spec.recovery.policy == "supervised_ce_pulse_adam"
                    else []
                ),
                *(
                    [
                        "same_array_persistent_transfer_uses_hidden_controller_inaccessible_state"
                    ]
                    if spec.transfer.source_state == "same_array_persistent"
                    else []
                ),
            ],
        }
        summary_path = artifact_root / "summary.json"
        atomic_write_json(summary_path, summary)
        artifacts = [
            store.artifact_record(path, kind=kind)
            for path, kind in (
                *source_artifact_paths,
                *post_fault_artifact_paths,
                *target_artifact_paths,
            )
        ]
        artifacts.append(store.artifact_record(summary_path, kind="scientific_summary"))
        artifacts.append(
            store.artifact_record(offchip_state_path, kind="crossbar_offchip_state")
        )
        artifacts.extend(
            store.artifact_record(path, kind="crossbar_endpoint_states")
            for path in checkpoint_paths
        )
        artifacts.extend(
            store.artifact_record(path, kind=kind)
            for path, kind in post_fault_runtime_artifact_paths
        )
        compact_metrics = {
            "evidence_tier": summary["evidence_tier"],
            "assignment_seed": spec.device.assignment_seed,
            "endpoint_seeds": list(spec.device.endpoint_seeds),
            "offchip_policy": spec.offchip.policy,
            "recovery_policy": spec.recovery.policy,
            "layer_scope": spec.recovery.layer_scope,
            "aggregates": aggregates,
            "drn_comparison": comparison,
        }
        store.complete(metrics=compact_metrics, artifacts=tuple(artifacts))
        return 0
    except BaseException as error:
        store.fail(error)
        raise


__all__ = ["run_train"]
