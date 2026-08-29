"""Direct CUDA runtime for exploratory Winsorized IBM OM STE-QAT."""

from __future__ import annotations

import gc
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence, TYPE_CHECKING

import numpy as np
import torch

from experiments.artifacts import RunStore, atomic_write_json, sha256_file
from experiments.mnist_relu_drn.components import build_student_stack
from experiments.mnist_relu_drn.ibm_om_baseline_selection_runtime import (
    _evaluate_detailed,
    _input,
)
from experiments.mnist_relu_drn.ibm_om_baseline_selection import (
    hardware_instance_fingerprint,
)
from experiments.mnist_relu_drn.ibm_om_baseline_spacing_pv import (
    apply_full_conductance_targets,
)
from experiments.mnist_relu_drn.ibm_om_baseline_spacing_pv_runtime import (
    _flatten,
    _maximum_supported_uniform_index,
    _split_population_vector,
)
from experiments.mnist_relu_drn.ibm_om_bounded_codebook_scheme_screen import (
    _tensor_sha256,
)
from experiments.mnist_relu_drn.ibm_om_winsorized_qat import (
    WinsorizedQatLayerTemplate,
    load_winsorized_qat_templates,
    logical_master_gradients,
    quantize_winsorized_logical_master,
)
from experiments.mnist_relu_drn.runtime import _load_teacher
from experiments.mnist_shared import build_mnist_loaders, limited
from experiments.schema import to_plain_data
from training.checkpoint import atomic_torch_save
from training.ibm_reram_hwa import IbmReramArrayPopulation
from training.ibm_reram_program_verify import ControllerSettings
from training.ibm_reram_raw_active_program_verify import (
    classify_persistent_uniform_codes,
    run_raw_active_program_verify,
)


if TYPE_CHECKING:
    from ebl.cli import TrainRequest


_ROOT = Path(__file__).resolve().parents[2]
SUMMARY_SCHEMA = "ebl.mnist_relu_drn.ibm_om_winsorized_qat_exploratory"
SUMMARY_SCHEMA_VERSION = 1
CHECKPOINT_SCHEMA = "ebl.mnist_relu_drn.ibm_om_winsorized_qat_logical_master"
CHECKPOINT_SCHEMA_VERSION = 1
NOMINAL_DELTA_X = 0.04745
MAXIMUM_PROGRAM_PULSES = 128
VERIFY_TOLERANCE_DELTA_X_RATIO = 0.5
_WINSORIZED_POPULATION_SCHEMA = "ebl.ibm_reram.om_array_population"
_WINSORIZED_POPULATION_FIELDS = {
    "schema",
    "schema_version",
    "assignment_seed",
    "corruption_policy",
    "binding_keys_json",
    "binding_shapes_json",
    "binding_sampling_seeds",
    "donor_sampling_seeds",
    "nominal_dw_min",
    "dw_min_std",
    "write_noise_std",
    "max_bound",
    "min_bound",
    "dwmin_up",
    "dwmin_down",
    "reference",
    "corrupt",
    "published_corrupt",
    "fingerprint",
    "aihwkit_version",
}


def _npz_scalar(payload: Mapping[str, np.ndarray], name: str) -> Any:
    value = payload[name]
    if value.shape != () or value.dtype.hasobject:
        raise ValueError(
            f"Expected scalar non-object Winsorized population field {name!r}."
        )
    return value.item()


def _load_winsorized_population(
    path: Path,
    receipt_path: Path,
    *,
    expected_sha256: str,
    expected_fingerprint: str,
    expected_assignment_seed: int,
    source_joint_hardware_instance_id: str,
) -> IbmReramArrayPopulation:
    """Strictly reload the study-local custom-fingerprint pulse population.

    The generic OM loader correctly rejects this artifact: the Winsorized
    intervention intentionally replaced the canonical sampling fingerprint by
    :func:`hardware_instance_fingerprint`.  This loader reproduces that exact
    intervention hash and does not relax the generic artifact contract.
    """

    source = path.expanduser().resolve()
    receipt_source = receipt_path.expanduser().resolve()
    if sha256_file(source) != expected_sha256:
        raise ValueError("Winsorized population artifact SHA-256 mismatch.")
    try:
        with np.load(source, allow_pickle=False) as raw:
            if set(raw.files) != _WINSORIZED_POPULATION_FIELDS:
                raise ValueError(
                    "Expected exact Winsorized OM population fields. "
                    f"Provided value: {sorted(raw.files)!r}."
                )
            payload = {name: raw[name].copy() for name in raw.files}
    except (OSError, ValueError) as error:
        raise ValueError(
            f"Expected a readable pickle-free Winsorized population at {source}."
        ) from error
    if (
        str(_npz_scalar(payload, "schema")) != _WINSORIZED_POPULATION_SCHEMA
        or int(_npz_scalar(payload, "schema_version")) != 1
        or payload["schema_version"].dtype != np.dtype(np.int64)
        or payload["assignment_seed"].dtype != np.dtype(np.int64)
    ):
        raise ValueError("Expected Winsorized OM population schema version 1.")
    assignment_seed = int(_npz_scalar(payload, "assignment_seed"))
    if assignment_seed != int(expected_assignment_seed):
        raise ValueError("Winsorized population assignment-seed mismatch.")
    corruption_policy = str(_npz_scalar(payload, "corruption_policy"))
    if corruption_policy != "counterfactual_repaired":
        raise ValueError("Expected the frozen counterfactual-repaired population.")
    try:
        keys_value = json.loads(str(_npz_scalar(payload, "binding_keys_json")))
        shapes_value = json.loads(str(_npz_scalar(payload, "binding_shapes_json")))
    except json.JSONDecodeError as error:
        raise ValueError("Expected canonical Winsorized binding JSON.") from error
    if (
        not isinstance(keys_value, list)
        or not keys_value
        or len(set(keys_value)) != len(keys_value)
        or not all(isinstance(key, str) and key for key in keys_value)
        or not isinstance(shapes_value, list)
        or len(shapes_value) != len(keys_value)
        or not all(
            isinstance(shape, list)
            and len(shape) == 2
            and all(
                isinstance(value, int)
                and not isinstance(value, bool)
                and value > 0
                for value in shape
            )
            for shape in shapes_value
        )
    ):
        raise ValueError("Expected valid named two-dimensional Winsorized bindings.")
    keys = tuple(keys_value)
    shapes = tuple(tuple(int(value) for value in shape) for shape in shapes_value)
    binding_seed_array = payload["binding_sampling_seeds"]
    donor_seed_array = payload["donor_sampling_seeds"]
    if (
        binding_seed_array.dtype != np.dtype(np.int64)
        or donor_seed_array.dtype != np.dtype(np.int64)
        or binding_seed_array.shape != (len(keys),)
        or donor_seed_array.shape != (len(keys),)
    ):
        raise ValueError("Expected one int64 sampling and donor seed per binding.")
    size = sum(math.prod(shape) for shape in shapes)
    tensors: dict[str, torch.Tensor] = {}
    for name in ("max_bound", "min_bound", "dwmin_up", "dwmin_down", "reference"):
        value = payload[name]
        if (
            value.dtype != np.dtype(np.float32)
            or value.shape != (size,)
            or not bool(np.all(np.isfinite(value)))
        ):
            raise ValueError(
                f"Expected finite float32 Winsorized population vector {name!r}."
            )
        tensors[name] = torch.from_numpy(value.copy())
    for name in ("corrupt", "published_corrupt"):
        value = payload[name]
        if value.dtype != np.dtype(np.bool_) or value.shape != (size,):
            raise ValueError(f"Expected boolean Winsorized population vector {name!r}.")
        tensors[name] = torch.from_numpy(value.copy())
    if (
        bool(torch.any(tensors["min_bound"] < -1.0))
        or bool(torch.any(tensors["max_bound"] > 1.0))
        or bool(torch.any(tensors["max_bound"] < tensors["min_bound"]))
    ):
        raise ValueError("Winsorized population bounds left ordered a in [-1,1].")
    scalars = {}
    for name in ("nominal_dw_min", "dw_min_std", "write_noise_std"):
        if payload[name].dtype != np.dtype(np.float64):
            raise ValueError(f"Expected float64 Winsorized scalar {name!r}.")
        value = float(_npz_scalar(payload, name))
        if not math.isfinite(value):
            raise ValueError(f"Expected finite Winsorized scalar {name!r}.")
        scalars[name] = value
    aihwkit_version = str(_npz_scalar(payload, "aihwkit_version"))
    if aihwkit_version != "1.1.0":
        raise ValueError("Expected AIHWKit 1.1.0 in the Winsorized population.")
    fingerprint = hardware_instance_fingerprint(
        {
            "source_joint_hardware_instance_id": (
                source_joint_hardware_instance_id
            ),
            "intervention": "nominal_bound_winsorization",
            "raw_a_minimum": -1.0,
            "raw_a_maximum": 1.0,
            "identity_resampled": False,
        },
        {
            name: tensors[name]
            for name in (
                "min_bound",
                "max_bound",
                "dwmin_up",
                "dwmin_down",
                "reference",
                "corrupt",
                "published_corrupt",
            )
        },
    )
    stored_fingerprint = str(_npz_scalar(payload, "fingerprint"))
    if fingerprint != stored_fingerprint or fingerprint != expected_fingerprint:
        raise ValueError("Winsorized custom population fingerprint mismatch.")
    try:
        receipt = json.loads(receipt_source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("Expected a readable Winsorized population receipt.") from error
    intervention = receipt.get("intervention")
    if (
        receipt.get("schema")
        != "ebl.mnist_relu_drn.ibm_om_winsorized_population_receipt"
        or receipt.get("schema_version") != 1
        or receipt.get("artifact") != source.name
        or receipt.get("artifact_sha256") != expected_sha256
        or receipt.get("assignment_seed") != assignment_seed
        or receipt.get("population_fingerprint") != fingerprint
        or receipt.get("source_population_fingerprint")
        != source_joint_hardware_instance_id
        or not isinstance(intervention, dict)
        or intervention.get("policy") != "nominal_bound_winsorization"
        or intervention.get("source_population_fingerprint")
        != source_joint_hardware_instance_id
        or intervention.get("winsorized_population_fingerprint") != fingerprint
        or intervention.get("raw_a_interval") != [-1.0, 1.0]
        or intervention.get("identity_resampled") is not False
    ):
        raise ValueError("Winsorized population receipt does not bind the artifact.")
    return IbmReramArrayPopulation(
        assignment_seed=assignment_seed,
        corruption_policy=corruption_policy,
        binding_keys=keys,
        binding_shapes=shapes,
        binding_sampling_seeds=tuple(
            int(value) for value in binding_seed_array.tolist()
        ),
        donor_sampling_seeds=tuple(int(value) for value in donor_seed_array.tolist()),
        nominal_dw_min=scalars["nominal_dw_min"],
        dw_min_std=scalars["dw_min_std"],
        write_noise_std=scalars["write_noise_std"],
        max_bound=tensors["max_bound"],
        min_bound=tensors["min_bound"],
        dwmin_up=tensors["dwmin_up"],
        dwmin_down=tensors["dwmin_down"],
        reference=tensors["reference"],
        corrupt=tensors["corrupt"],
        published_corrupt=tensors["published_corrupt"],
        fingerprint=fingerprint,
        aihwkit_version=aihwkit_version,
    )


def _completed_source_run(protocol: Any) -> Mapping[str, Path]:
    """Resolve the exact hash-pinned Winsorized predecessor bundle."""

    spacing = int(protocol.mapping.spacing_delta_x_multiplier)
    root = (
        _ROOT
        / "results"
        / str(protocol.artifacts.source_exploratory_result_id)
        / "runs"
        / f"alpha_000_spacing_{spacing}delta-heldout-87001"
    )
    candidates = []
    for status_path in sorted(root.glob("*/status.json")):
        try:
            status = json.loads(status_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if status.get("status") != "complete":
            continue
        run_dir = status_path.parent
        paths = {
            "run_dir": run_dir,
            "development_mapping": run_dir
            / "artifacts/development/selected_physical_mapping.npz",
            "evaluation_mapping": run_dir
            / "artifacts/heldout/ideal_physical_mapping.npz",
            "development_population": run_dir
            / "artifacts/development/winsorized_identity/winsorized_population.npz",
            "development_population_receipt": run_dir
            / "artifacts/development/winsorized_identity/winsorized_population.receipt.json",
            "evaluation_population": run_dir
            / "artifacts/heldout/winsorized_identity/winsorized_population.npz",
            "evaluation_population_receipt": run_dir
            / "artifacts/heldout/winsorized_identity/winsorized_population.receipt.json",
            "development_commissioning": run_dir
            / "artifacts/development/winsorized_identity/winsorized_reset_commissioning.npz",
            "evaluation_commissioning": run_dir
            / "artifacts/heldout/winsorized_identity/winsorized_reset_commissioning.npz",
            "development_joint": run_dir
            / "artifacts/development/joint_assignment.npz",
            "evaluation_joint": run_dir / "artifacts/heldout/joint_assignment.npz",
            "calibration": run_dir / "artifacts/development/calibration.json",
            "summary": run_dir / "artifacts/scientific_summary.json",
        }
        if all(path.is_file() for name, path in paths.items() if name != "run_dir"):
            candidates.append(paths)
    if not candidates:
        raise FileNotFoundError(
            "Expected one completed hash-pinned Winsorized predecessor run at "
            f"{root}."
        )

    def matches(paths: Mapping[str, Path]) -> bool:
        return (
            sha256_file(paths["development_mapping"])
            == protocol.evaluation.development_mapping_sha256
            and sha256_file(paths["evaluation_mapping"])
            == protocol.evaluation.evaluation_mapping_sha256
            and sha256_file(paths["development_population"])
            == protocol.artifacts.development_winsorized_population_sha256
            and sha256_file(paths["evaluation_population"])
            == protocol.artifacts.evaluation_winsorized_population_sha256
            and sha256_file(paths["development_commissioning"])
            == protocol.artifacts.development_winsorized_commissioning_sha256
            and sha256_file(paths["evaluation_commissioning"])
            == protocol.artifacts.evaluation_winsorized_commissioning_sha256
            and sha256_file(paths["development_joint"])
            == protocol.artifacts.development_joint_assignment_sha256
            and sha256_file(paths["evaluation_joint"])
            == protocol.artifacts.evaluation_joint_assignment_sha256
            and sha256_file(paths["calibration"])
            == protocol.artifacts.development_calibration_sha256
        )

    matched = [paths for paths in candidates if matches(paths)]
    if len(matched) != 1:
        raise RuntimeError(
            "Expected exactly one complete predecessor bundle matching every "
            f"declared artifact hash; matches={len(matched)}."
        )
    return matched[0]


def _artifact_inputs(paths: Mapping[str, Path]) -> tuple[Mapping[str, Any], ...]:
    roles = (
        "development_mapping",
        "evaluation_mapping",
        "development_population",
        "development_population_receipt",
        "evaluation_population",
        "evaluation_population_receipt",
        "development_commissioning",
        "evaluation_commissioning",
        "development_joint",
        "evaluation_joint",
        "calibration",
        "summary",
    )
    return tuple(_input(f"predecessor_{role}", paths[role]) for role in roles)


def _initial_logical_masters(
    teacher: Any,
    templates: Sequence[WinsorizedQatLayerTemplate],
    protocol: Any,
    *,
    device: torch.device,
) -> tuple[tuple[torch.nn.Parameter, ...], tuple[float, ...]]:
    source = tuple(value.detach().cpu().to(torch.float32) for value in teacher.parameters())
    observed_hashes = tuple(_tensor_sha256(value) for value in source)
    if observed_hashes != tuple(protocol.initialization.source_weight_sha256_by_layer):
        raise RuntimeError(
            "Frozen ReLU source tensors do not match the declared initialization."
        )
    maxima = tuple(float(value.abs().max().item()) for value in source)
    masters = []
    for value, maximum, template in zip(source, maxima, templates):
        normalized = (value.to(torch.float64) / maximum).to(torch.float32)
        if not torch.equal(normalized, template.initial_normalized_weight.cpu()):
            difference = float(
                (normalized - template.initial_normalized_weight.cpu()).abs().max().item()
            )
            if difference > 1e-7:
                raise RuntimeError(
                    "Predecessor logical initialization differs from the teacher; "
                    f"maximum_absolute_difference={difference}."
                )
            normalized = template.initial_normalized_weight.cpu().clone()
        masters.append(torch.nn.Parameter(normalized.to(device=device)))
    return tuple(masters), maxima


def _load_epoch_10_logical_master(
    path: Path,
    *,
    templates: Sequence[WinsorizedQatLayerTemplate],
    protocol: Any,
    teacher_sha256: str,
    source_absmax: Sequence[float],
    device: torch.device,
) -> tuple[tuple[torch.Tensor, ...], Mapping[str, Any]]:
    """Load one failed-run epoch-10 shadow for evaluation-only recovery."""

    source = path.expanduser().resolve()
    try:
        payload = torch.load(source, map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        raise ValueError("Expected a readable weights-only QAT checkpoint.") from error
    expected_fields = {
        "schema",
        "schema_version",
        "epoch",
        "evidence_tier",
        "teacher_sha256",
        "spacing_delta_x_multiplier",
        "fixed_logit_gain",
        "source_absmax",
        "logical_master",
        "logical_master_sha256",
        "optimizer_state",
        "validation",
    }
    if not isinstance(payload, dict) or set(payload) != expected_fields:
        raise ValueError("Expected exact epoch-10 QAT checkpoint fields.")
    if (
        payload.get("schema") != CHECKPOINT_SCHEMA
        or payload.get("schema_version") != CHECKPOINT_SCHEMA_VERSION
        or payload.get("epoch") != 10
        or payload.get("evidence_tier") != "exploratory_noncanonical"
        or payload.get("teacher_sha256") != teacher_sha256
        or payload.get("spacing_delta_x_multiplier")
        != int(protocol.mapping.spacing_delta_x_multiplier)
        or payload.get("fixed_logit_gain")
        != float(protocol.mapping.fixed_logit_gain)
    ):
        raise ValueError("Epoch-10 QAT checkpoint protocol mismatch.")
    maxima = payload.get("source_absmax")
    if (
        not isinstance(maxima, list)
        or len(maxima) != 2
        or any(type(value) is not float or not math.isfinite(value) for value in maxima)
        or tuple(maxima) != tuple(float(value) for value in source_absmax)
    ):
        raise ValueError("Epoch-10 QAT source normalization mismatch.")
    stored = payload.get("logical_master")
    stored_hashes = payload.get("logical_master_sha256")
    if (
        not isinstance(stored, (tuple, list))
        or len(stored) != 2
        or not isinstance(stored_hashes, (tuple, list))
        or len(stored_hashes) != 2
    ):
        raise ValueError("Expected two hashed epoch-10 logical masters.")
    masters = []
    observed_hashes = []
    for layer_index, (value, template) in enumerate(zip(stored, templates)):
        if (
            not isinstance(value, torch.Tensor)
            or value.dtype != torch.float32
            or tuple(value.shape) != template.logical_shape
            or not bool(torch.all(torch.isfinite(value)))
            or bool(torch.any(value < -1.0))
            or bool(torch.any(value > 1.0))
        ):
            raise ValueError(
                f"Expected finite bounded float32 epoch-10 master for layer {layer_index}."
            )
        clean = value.detach().cpu().contiguous().clone()
        observed_hashes.append(_tensor_sha256(clean))
        masters.append(clean.to(device=device))
    if tuple(observed_hashes) != tuple(stored_hashes):
        raise ValueError("Epoch-10 logical-master SHA-256 mismatch.")
    validation = payload.get("validation")
    if not isinstance(validation, dict):
        raise ValueError("Expected checkpoint development-validation metadata.")
    return tuple(masters), payload


def _apply_view(stack: Any, view: Any) -> None:
    apply_full_conductance_targets(
        stack.bundle.catalog,
        view.full_conductance_targets,
        conductance_min=0.0,
        conductance_max=2.0,
    )


def _evaluate_quantized(
    stack: Any,
    teacher: Any,
    loader: Iterable,
    masters: Sequence[torch.Tensor],
    templates: Sequence[WinsorizedQatLayerTemplate],
) -> tuple[Mapping[str, Any], torch.Tensor, Any]:
    view = quantize_winsorized_logical_master(masters, templates)
    _apply_view(stack, view)
    metrics, prediction = _evaluate_detailed(
        stack,
        teacher,
        loader,
        sample_limit=None,
    )
    return metrics, prediction, view


def _train_epoch(
    stack: Any,
    teacher: Any,
    loader: Iterable,
    masters: Sequence[torch.nn.Parameter],
    templates: Sequence[WinsorizedQatLayerTemplate],
    optimizer: torch.optim.Optimizer,
    *,
    maximum_batches: int | None,
) -> Mapping[str, Any]:
    examples = 0
    batches = 0
    kl_sum = 0.0
    correct = 0
    agreement = 0
    gradient_squared = [0.0, 0.0]
    gradient_values = [0, 0]
    for batch_index, (inputs, labels) in enumerate(limited(loader, maximum_batches)):
        inputs = inputs.to(stack.device, dtype=torch.float32)
        labels = labels.to(stack.device, dtype=torch.long)
        with torch.no_grad():
            teacher_logits = teacher.logits(inputs)
            view = quantize_winsorized_logical_master(
                masters, templates, include_report=False
            )
            _apply_view(stack, view)
        optimizer.zero_grad(set_to_none=True)
        stack.network.set_input(inputs, reset=True)
        stack.minimizer.compute_equilibrium()
        stack.cost.set_teacher(teacher_logits, labels)
        with torch.no_grad():
            batch_kl = stack.cost.eval()
            student_prediction = stack.cost.student_logits().argmax(dim=1)
            teacher_prediction = teacher_logits.argmax(dim=1)
            kl_sum += float(batch_kl.sum().item())
            correct += int(student_prediction.eq(labels).sum().item())
            agreement += int(student_prediction.eq(teacher_prediction).sum().item())
        physical_gradients = tuple(stack.differentiator.compute_gradient())
        logical_gradients = logical_master_gradients(
            masters, physical_gradients, templates
        )
        for layer_index, (master, gradient) in enumerate(
            zip(masters, logical_gradients)
        ):
            master.grad = gradient
            gradient_squared[layer_index] += float(
                gradient.detach().to(torch.float64).square().sum().item()
            )
            gradient_values[layer_index] += int(gradient.numel())
        optimizer.step()
        with torch.no_grad():
            for master in masters:
                master.clamp_(-1.0, 1.0)
        examples += int(labels.shape[0])
        batches = batch_index + 1
    if examples == 0:
        raise RuntimeError("Expected QAT to process at least one minibatch.")
    return {
        "examples": examples,
        "batches": batches,
        "kl_teacher_student": kl_sum / examples,
        "student_accuracy": correct / examples,
        "teacher_agreement": agreement / examples,
        "logical_gradient_rms": [
            math.sqrt(total / count)
            for total, count in zip(gradient_squared, gradient_values)
        ],
    }


def _checkpoint_payload(
    *,
    epoch: int,
    masters: Sequence[torch.Tensor],
    source_absmax: Sequence[float],
    optimizer: torch.optim.Optimizer,
    validation: Mapping[str, Any],
    protocol: Any,
    teacher_sha256: str,
) -> Mapping[str, Any]:
    return {
        "schema": CHECKPOINT_SCHEMA,
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "epoch": int(epoch),
        "evidence_tier": "exploratory_noncanonical",
        "teacher_sha256": teacher_sha256,
        "spacing_delta_x_multiplier": int(
            protocol.mapping.spacing_delta_x_multiplier
        ),
        "fixed_logit_gain": float(protocol.mapping.fixed_logit_gain),
        "source_absmax": list(map(float, source_absmax)),
        "logical_master": tuple(
            value.detach().cpu().to(torch.float32).clone() for value in masters
        ),
        "logical_master_sha256": tuple(_tensor_sha256(value) for value in masters),
        "optimizer_state": optimizer.state_dict(),
        "validation": dict(validation),
    }


def _programming_summary(outcome: Any, code: Any) -> Mapping[str, Any]:
    programming = outcome.programming
    residual = (
        outcome.persistent_endpoint_unit.to(torch.float64)
        - outcome.target_unit.to(torch.float64)
    )
    return {
        "endpoint_seed": int(outcome.endpoint_seed),
        "cells": int(outcome.target_unit.numel()),
        "apparent_accepted": int(programming.accepted.sum().item()),
        "persistent_inside_acceptance_window": int(
            outcome.persistent_inside_acceptance_window.sum().item()
        ),
        "requested_code_correct": int(code.requested_code_correct.sum().item()),
        "budget_exhausted": int(programming.budget_exhausted.sum().item()),
        "saturated_lower": int(outcome.saturated_lower.sum().item()),
        "saturated_upper": int(outcome.saturated_upper.sum().item()),
        "mean_total_pulses": float(
            programming.total_pulses.to(torch.float64).mean().item()
        ),
        "mean_verify_count": float(
            programming.verify_count.to(torch.float64).mean().item()
        ),
        "persistent_residual_raw_x": {
            "bias": float(residual.mean().item()),
            "mae": float(residual.abs().mean().item()),
            "rmse": float(residual.square().mean().sqrt().item()),
        },
        "apparent_endpoint_applied_to_drn": False,
        "persistent_projection_count": 0,
    }


def _evaluate_deployment(
    *,
    role: str,
    stack: Any,
    teacher: Any,
    loader: Iterable,
    masters: Sequence[torch.Tensor],
    templates: Sequence[WinsorizedQatLayerTemplate],
    population: Any,
    endpoint_seeds: Sequence[int],
    device: torch.device,
) -> Mapping[str, Any]:
    ideal_metrics, _ideal_prediction, view = _evaluate_quantized(
        stack, teacher, loader, masters, templates
    )
    target_raw = _flatten(view.raw_x_targets, dtype=torch.float32)
    baseline_raw = _flatten(
        tuple(template.baseline_raw_x for template in templates),
        dtype=torch.float32,
    )
    requested = _flatten(view.requested_index, dtype=torch.int64)
    upper = _flatten(
        tuple(template.cell_upper_raw_x for template in templates),
        dtype=torch.float32,
    )
    spacing = float(templates[0].level_spacing_raw_x)
    maximum = _maximum_supported_uniform_index(
        upper,
        baseline_raw,
        spacing_unit=spacing,
    )
    if not all(
        math.isclose(float(template.level_spacing_raw_x), spacing)
        for template in templates
    ):
        raise RuntimeError("Expected one common raw-x spacing across both layers.")
    if bool(torch.any(requested > maximum)):
        raise RuntimeError("QAT deployment requested an unsupported code index.")
    repeats = []
    for endpoint_seed in endpoint_seeds:
        outcome = run_raw_active_program_verify(
            population,
            targets_unit=target_raw,
            endpoint_seed=int(endpoint_seed),
            tolerance_unit=VERIFY_TOLERANCE_DELTA_X_RATIO * NOMINAL_DELTA_X,
            maximum_program_pulses=MAXIMUM_PROGRAM_PULSES,
            device=device,
            settings=ControllerSettings(kind="one_pulse"),
        )
        code = classify_persistent_uniform_codes(
            outcome.persistent_endpoint_unit,
            baseline_unit=baseline_raw,
            spacing_unit=spacing,
            requested_index=requested,
            maximum_index=maximum,
        )
        persistent_raw = outcome.persistent_endpoint_unit
        if bool(torch.any(persistent_raw < 0.0)) or bool(
            torch.any(persistent_raw > 1.0)
        ):
            raise RuntimeError(
                "Winsorized persistent endpoint left x in [0,1]; no projection "
                "is permitted."
            )
        persistent_g = _split_population_vector(2.0 * persistent_raw, population)
        apply_full_conductance_targets(
            stack.bundle.catalog,
            persistent_g,
            conductance_min=0.0,
            conductance_max=2.0,
        )
        metrics, _prediction = _evaluate_detailed(
            stack,
            teacher,
            loader,
            sample_limit=None,
        )
        repeats.append(
            {
                "metrics": metrics,
                "programming": _programming_summary(outcome, code),
            }
        )
        del outcome, code, persistent_g
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    mean_accuracy = sum(
        float(item["metrics"]["student_accuracy"]) for item in repeats
    ) / len(repeats)
    return {
        "role": role,
        "assignment_seed": int(population.assignment_seed),
        "population_fingerprint": population.fingerprint,
        "ideal_quantized": ideal_metrics,
        "quantizer": view.report,
        "persistent_repeats": repeats,
        "persistent_mean_accuracy": mean_accuracy,
        "endpoint_seeds": list(map(int, endpoint_seeds)),
    }


def _deployment_evaluations(
    *,
    stack: Any,
    teacher: Any,
    loader: Iterable,
    initial_master: Sequence[torch.Tensor],
    epoch_10_master: Sequence[torch.Tensor],
    development_templates: Sequence[WinsorizedQatLayerTemplate],
    evaluation_templates: Sequence[WinsorizedQatLayerTemplate],
    source_paths: Mapping[str, Path],
    protocol: Any,
    device: torch.device,
) -> Mapping[str, Any]:
    development_population = _load_winsorized_population(
        source_paths["development_population"],
        source_paths["development_population_receipt"],
        expected_sha256=(
            protocol.artifacts.development_winsorized_population_sha256
        ),
        expected_fingerprint=(
            protocol.artifacts.development_winsorized_population_fingerprint
        ),
        expected_assignment_seed=protocol.assignments.development_assignment_seed,
        source_joint_hardware_instance_id=(
            protocol.artifacts.development_hardware_instance_id
        ),
    )
    evaluation_population = _load_winsorized_population(
        source_paths["evaluation_population"],
        source_paths["evaluation_population_receipt"],
        expected_sha256=(
            protocol.artifacts.evaluation_winsorized_population_sha256
        ),
        expected_fingerprint=(
            protocol.artifacts.evaluation_winsorized_population_fingerprint
        ),
        expected_assignment_seed=protocol.assignments.evaluation_assignment_seed,
        source_joint_hardware_instance_id=(
            protocol.artifacts.evaluation_hardware_instance_id
        ),
    )

    def evaluate_pair(masters: Sequence[torch.Tensor]) -> Mapping[str, Any]:
        return {
            "same_hardware": _evaluate_deployment(
                role="same_hardware_development_86001",
                stack=stack,
                teacher=teacher,
                loader=loader,
                masters=masters,
                templates=development_templates,
                population=development_population,
                endpoint_seeds=protocol.assignments.development_endpoint_seeds,
                device=device,
            ),
            "cross_hardware": _evaluate_deployment(
                role="cross_hardware_seen_heldout_87001",
                stack=stack,
                teacher=teacher,
                loader=loader,
                masters=masters,
                templates=evaluation_templates,
                population=evaluation_population,
                endpoint_seeds=protocol.assignments.evaluation_endpoint_seeds,
                device=device,
            ),
        }

    return {
        "epoch_0": evaluate_pair(initial_master),
        "epoch_10": evaluate_pair(epoch_10_master),
    }


def _terminal_metrics(
    *,
    protocol: Any,
    initial_validation: Mapping[str, Any],
    epoch_10_validation: Mapping[str, Any],
    evaluations: Mapping[str, Any],
    best_epoch: int | None,
    evaluation_only_recovery: bool,
) -> Mapping[str, Any]:
    return {
        "evidence_tier": "exploratory_noncanonical",
        "evaluation_only_recovery": bool(evaluation_only_recovery),
        "spacing_delta_x_multiplier": int(
            protocol.mapping.spacing_delta_x_multiplier
        ),
        "epochs": int(protocol.training.epochs),
        "best_development_epoch_diagnostic": best_epoch,
        "epoch_0_validation_accuracy": float(initial_validation["student_accuracy"]),
        "epoch_10_validation_accuracy": float(
            epoch_10_validation["student_accuracy"]
        ),
        "epoch_0_same_hardware_ideal_accuracy": float(
            evaluations["epoch_0"]["same_hardware"]["ideal_quantized"][
                "student_accuracy"
            ]
        ),
        "epoch_10_same_hardware_ideal_accuracy": float(
            evaluations["epoch_10"]["same_hardware"]["ideal_quantized"][
                "student_accuracy"
            ]
        ),
        "epoch_0_same_hardware_pv_accuracy": float(
            evaluations["epoch_0"]["same_hardware"]["persistent_mean_accuracy"]
        ),
        "epoch_10_same_hardware_pv_accuracy": float(
            evaluations["epoch_10"]["same_hardware"]["persistent_mean_accuracy"]
        ),
        "epoch_0_cross_hardware_ideal_accuracy": float(
            evaluations["epoch_0"]["cross_hardware"]["ideal_quantized"][
                "student_accuracy"
            ]
        ),
        "epoch_10_cross_hardware_ideal_accuracy": float(
            evaluations["epoch_10"]["cross_hardware"]["ideal_quantized"][
                "student_accuracy"
            ]
        ),
        "epoch_0_cross_hardware_pv_accuracy": float(
            evaluations["epoch_0"]["cross_hardware"]["persistent_mean_accuracy"]
        ),
        "epoch_10_cross_hardware_pv_accuracy": float(
            evaluations["epoch_10"]["cross_hardware"]["persistent_mean_accuracy"]
        ),
    }


def _finish_evaluation_only_recovery(
    *,
    store: RunStore,
    resume_path: Path,
    resume_payload: Mapping[str, Any],
    stack: Any,
    teacher: Any,
    teacher_path: Path,
    teacher_metadata: Mapping[str, Any],
    loaders: Any,
    source_absmax: Sequence[float],
    initial_master: Sequence[torch.Tensor],
    epoch_10_master: Sequence[torch.Tensor],
    initial_hashes: Sequence[str],
    initial_validation: Mapping[str, Any],
    development_templates: Sequence[WinsorizedQatLayerTemplate],
    evaluation_templates: Sequence[WinsorizedQatLayerTemplate],
    source_paths: Mapping[str, Path],
    protocol: Any,
    device: torch.device,
) -> int:
    evaluations = _deployment_evaluations(
        stack=stack,
        teacher=teacher,
        loader=loaders.test,
        initial_master=initial_master,
        epoch_10_master=epoch_10_master,
        development_templates=development_templates,
        evaluation_templates=evaluation_templates,
        source_paths=source_paths,
        protocol=protocol,
        device=device,
    )
    epoch_10_validation = dict(resume_payload["validation"])
    summary = {
        "schema": SUMMARY_SCHEMA,
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "status": "complete",
        "evidence_tier": "exploratory_noncanonical",
        "claim_boundary": (
            "Evaluation-only recovery of a hash-validated epoch-10 logical "
            "master from a preserved failed exploratory run. Training was not "
            "repeated. Deterministic codebook QAT used one fixed model-based "
            "Winsorized development array and no P&V/read noise during "
            "training. Persistent one-pulse P&V is evaluated after training. "
            "Cross-hardware 87001 was inspected previously and is not an "
            "untouched target. No fabricated-device or on-chip-training claim."
        ),
        "source": {
            "teacher_path": str(teacher_path),
            "teacher_sha256": sha256_file(teacher_path),
            "teacher_metadata": teacher_metadata,
            "source_absmax": list(map(float, source_absmax)),
        },
        "protocol": to_plain_data(protocol),
        "predecessor_artifacts": {
            name: {"path": str(path), "sha256": sha256_file(path)}
            for name, path in source_paths.items()
            if name != "run_dir"
        },
        "recovery": {
            "mode": "evaluation_only_from_epoch_10_checkpoint",
            "checkpoint_path": str(resume_path),
            "checkpoint_sha256": sha256_file(resume_path),
            "checkpoint_epoch": int(resume_payload["epoch"]),
            "training_repeated": False,
        },
        "initialization": {
            "logical_master_sha256": [
                _tensor_sha256(value) for value in initial_master
            ],
            "quantized_conductance_sha256": list(initial_hashes),
            "epoch_0_validation": dict(initial_validation),
        },
        "training": {
            "epochs": int(protocol.training.epochs),
            "learning_rates": list(protocol.training.learning_rates),
            "training_program_verify": False,
            "training_read_noise": False,
            "fixed_headline_epoch": 10,
            "epoch_10_validation": epoch_10_validation,
            "history_available_in_source_failed_run": True,
            "history": [],
        },
        "evaluations": evaluations,
    }
    summary_path = store.run_dir / "artifacts/scientific_summary.json"
    atomic_write_json(summary_path, summary)
    terminal = _terminal_metrics(
        protocol=protocol,
        initial_validation=initial_validation,
        epoch_10_validation=epoch_10_validation,
        evaluations=evaluations,
        best_epoch=None,
        evaluation_only_recovery=True,
    )
    store.append_metric({"mode": "evaluation_recovery_terminal", **terminal})
    store.complete(
        metrics=terminal,
        artifacts=(
            store.artifact_record(summary_path, kind="scientific_summary"),
        ),
    )
    return 0


def run_train(request: "TrainRequest") -> int:
    from experiments.mnist_relu_drn.ibm_om_winsorized_qat_config import (
        WinsorizedQatTrainSpec,
    )

    spec = request.spec
    if not isinstance(spec, WinsorizedQatTrainSpec):
        raise TypeError("Expected the dedicated Winsorized QAT train spec.")
    if request.teacher_weights is None:
        raise ValueError("Expected --teacher-weights for Winsorized QAT.")
    for name in ("weights", "base_weights", "device_data", "device_model"):
        if getattr(request, name) is not None:
            raise ValueError(
                "Winsorized QAT accepts --teacher-weights and optional "
                "evaluation-only --resume; "
                f"unexpected {name}."
            )
    protocol = spec.protocol
    student = spec.student
    teacher_path = request.teacher_weights.expanduser().resolve()
    resume_path = (
        request.resume.expanduser().resolve()
        if request.resume is not None
        else None
    )
    if sha256_file(teacher_path) != protocol.source.expected_teacher_weights_sha256:
        raise ValueError("Frozen teacher/source checkpoint hash mismatch.")
    source_paths = _completed_source_run(protocol)
    store = RunStore.create(
        output_root=request.output_dir,
        experiment_id=spec.experiment_id,
        resolved_config=to_plain_data(spec),
        command=request.command,
        repo_root=_ROOT,
        input_artifacts=(
            _input("teacher_weights", teacher_path),
            *_artifact_inputs(source_paths),
            *(
                (_input("evaluation_recovery_checkpoint", resume_path),)
                if resume_path is not None
                else ()
            ),
        ),
        resume_capability="unsupported",
    )
    try:
        torch.manual_seed(student.runtime.seed)
        torch.cuda.manual_seed_all(student.runtime.seed)
        device = torch.device(student.runtime.device)
        if device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("Exploratory Winsorized QAT requires CUDA.")
        loaders = build_mnist_loaders(
            student.data,
            data_seed=student.runtime.data_seed,
            calibration_examples=student.mapping.calibration_examples,
            calibration_batch_size=student.mapping.calibration_batch_size,
        )
        teacher, teacher_metadata = _load_teacher(
            teacher_path,
            device=device,
            spec=student,
        )
        stack = build_student_stack(student, enable_measured=False)
        stack.cost.gain = float(protocol.mapping.fixed_logit_gain)
        spacing_raw_x = (
            int(protocol.mapping.spacing_delta_x_multiplier) * NOMINAL_DELTA_X
        )
        development_templates_cpu = load_winsorized_qat_templates(
            source_paths["development_mapping"],
            level_spacing_raw_x=spacing_raw_x,
        )
        evaluation_templates_cpu = load_winsorized_qat_templates(
            source_paths["evaluation_mapping"],
            level_spacing_raw_x=spacing_raw_x,
        )
        development_templates = tuple(
            template.to(device) for template in development_templates_cpu
        )
        evaluation_templates = tuple(
            template.to(device) for template in evaluation_templates_cpu
        )
        masters, source_absmax = _initial_logical_masters(
            teacher,
            development_templates_cpu,
            protocol,
            device=device,
        )
        initial_master = tuple(value.detach().clone() for value in masters)
        initial_view = quantize_winsorized_logical_master(
            masters, development_templates
        )
        initial_hashes = tuple(
            _tensor_sha256(value) for value in initial_view.full_conductance_targets
        )
        if initial_hashes != tuple(
            protocol.initialization.initial_quantized_conductance_sha256_by_layer
        ):
            raise RuntimeError(
                "Epoch-0 QAT forward does not reproduce the completed ideal map."
            )
        optimizer = torch.optim.SGD(
            [
                {"params": [master], "lr": float(rate)}
                for master, rate in zip(masters, protocol.training.learning_rates)
            ],
            momentum=0.0,
            weight_decay=0.0,
        )
        initial_validation, _prediction, _view = _evaluate_quantized(
            stack,
            teacher,
            loaders.validation,
            masters,
            development_templates,
        )
        if resume_path is not None:
            epoch_10_master, resume_payload = _load_epoch_10_logical_master(
                resume_path,
                templates=development_templates_cpu,
                protocol=protocol,
                teacher_sha256=sha256_file(teacher_path),
                source_absmax=source_absmax,
                device=device,
            )
            return _finish_evaluation_only_recovery(
                store=store,
                resume_path=resume_path,
                resume_payload=resume_payload,
                stack=stack,
                teacher=teacher,
                teacher_path=teacher_path,
                teacher_metadata=teacher_metadata,
                loaders=loaders,
                source_absmax=source_absmax,
                initial_master=initial_master,
                epoch_10_master=epoch_10_master,
                initial_hashes=initial_hashes,
                initial_validation=initial_validation,
                development_templates=development_templates,
                evaluation_templates=evaluation_templates,
                source_paths=source_paths,
                protocol=protocol,
                device=device,
            )
        store.append_metric(
            {
                "mode": "train",
                "epoch": 0,
                "spacing_delta_x_multiplier": int(
                    protocol.mapping.spacing_delta_x_multiplier
                ),
                "validation": initial_validation,
            }
        )
        best_epoch = 0
        best_validation = dict(initial_validation)
        best_payload = _checkpoint_payload(
            epoch=0,
            masters=masters,
            source_absmax=source_absmax,
            optimizer=optimizer,
            validation=initial_validation,
            protocol=protocol,
            teacher_sha256=sha256_file(teacher_path),
        )
        history = []
        for epoch in range(1, int(protocol.training.epochs) + 1):
            train_metrics = _train_epoch(
                stack,
                teacher,
                loaders.train,
                masters,
                development_templates,
                optimizer,
                maximum_batches=student.settings.max_batches,
            )
            validation, _prediction, view = _evaluate_quantized(
                stack,
                teacher,
                loaders.validation,
                masters,
                development_templates,
            )
            record = {
                "mode": "train",
                "epoch": epoch,
                "spacing_delta_x_multiplier": int(
                    protocol.mapping.spacing_delta_x_multiplier
                ),
                "train": train_metrics,
                "validation": validation,
                "quantizer": view.report,
            }
            history.append(record)
            store.append_metric(record)
            candidate_key = (
                int(validation["student_correct"]),
                -float(validation["kl_teacher_student"]),
                -epoch,
            )
            incumbent_key = (
                int(best_validation["student_correct"]),
                -float(best_validation["kl_teacher_student"]),
                -best_epoch,
            )
            if candidate_key > incumbent_key:
                best_epoch = epoch
                best_validation = dict(validation)
                best_payload = _checkpoint_payload(
                    epoch=epoch,
                    masters=masters,
                    source_absmax=source_absmax,
                    optimizer=optimizer,
                    validation=validation,
                    protocol=protocol,
                    teacher_sha256=sha256_file(teacher_path),
                )
            print(
                f"winsorized QAT h={protocol.mapping.spacing_delta_x_multiplier} "
                f"epoch={epoch}/10 train={100*train_metrics['student_accuracy']:.2f}% "
                f"validation={100*validation['student_accuracy']:.2f}%",
                flush=True,
            )

        final_validation = dict(history[-1]["validation"])
        final_payload = _checkpoint_payload(
            epoch=int(protocol.training.epochs),
            masters=masters,
            source_absmax=source_absmax,
            optimizer=optimizer,
            validation=final_validation,
            protocol=protocol,
            teacher_sha256=sha256_file(teacher_path),
        )
        final_checkpoint = atomic_torch_save(
            final_payload,
            store.run_dir / "checkpoints/epoch_10_logical_master.pt",
        )
        best_checkpoint = atomic_torch_save(
            best_payload,
            store.run_dir / "checkpoints/best_development_logical_master.pt",
        )

        evaluations = _deployment_evaluations(
            stack=stack,
            teacher=teacher,
            loader=loaders.test,
            initial_master=initial_master,
            epoch_10_master=masters,
            development_templates=development_templates,
            evaluation_templates=evaluation_templates,
            source_paths=source_paths,
            protocol=protocol,
            device=device,
        )
        summary = {
            "schema": SUMMARY_SCHEMA,
            "schema_version": SUMMARY_SCHEMA_VERSION,
            "status": "complete",
            "evidence_tier": "exploratory_noncanonical",
            "claim_boundary": (
                "Ten-epoch deterministic codebook-aware off-chip QAT on one "
                "fixed model-based AIHWKit OM Winsorized development array. "
                "Training contains quantization but no P&V/read noise; exact "
                "persistent one-pulse P&V is evaluated only after training. "
                "Cross-hardware 87001 was inspected previously and is not a "
                "new untouched target. No fabricated-device or on-chip-training claim."
            ),
            "source": {
                "teacher_path": str(teacher_path),
                "teacher_sha256": sha256_file(teacher_path),
                "teacher_metadata": teacher_metadata,
                "source_absmax": list(source_absmax),
            },
            "protocol": to_plain_data(protocol),
            "predecessor_artifacts": {
                name: {"path": str(path), "sha256": sha256_file(path)}
                for name, path in source_paths.items()
                if name != "run_dir"
            },
            "initialization": {
                "logical_master_sha256": [
                    _tensor_sha256(value) for value in initial_master
                ],
                "quantized_conductance_sha256": list(initial_hashes),
                "epoch_0_validation": initial_validation,
            },
            "training": {
                "epochs": int(protocol.training.epochs),
                "learning_rates": list(protocol.training.learning_rates),
                "training_program_verify": False,
                "training_read_noise": False,
                "best_development_epoch_diagnostic": best_epoch,
                "best_development_validation": best_validation,
                "fixed_headline_epoch": 10,
                "epoch_10_validation": final_validation,
                "history": history,
            },
            "evaluations": evaluations,
        }
        summary_path = store.run_dir / "artifacts/scientific_summary.json"
        atomic_write_json(summary_path, summary)
        terminal = {
            "evidence_tier": "exploratory_noncanonical",
            "spacing_delta_x_multiplier": int(
                protocol.mapping.spacing_delta_x_multiplier
            ),
            "epochs": int(protocol.training.epochs),
            "best_development_epoch_diagnostic": best_epoch,
            "epoch_0_validation_accuracy": float(
                initial_validation["student_accuracy"]
            ),
            "epoch_10_validation_accuracy": float(
                final_validation["student_accuracy"]
            ),
            "epoch_0_same_hardware_ideal_accuracy": float(
                evaluations["epoch_0"]["same_hardware"]["ideal_quantized"][
                    "student_accuracy"
                ]
            ),
            "epoch_10_same_hardware_ideal_accuracy": float(
                evaluations["epoch_10"]["same_hardware"]["ideal_quantized"][
                    "student_accuracy"
                ]
            ),
            "epoch_0_same_hardware_pv_accuracy": float(
                evaluations["epoch_0"]["same_hardware"][
                    "persistent_mean_accuracy"
                ]
            ),
            "epoch_10_same_hardware_pv_accuracy": float(
                evaluations["epoch_10"]["same_hardware"][
                    "persistent_mean_accuracy"
                ]
            ),
            "epoch_0_cross_hardware_ideal_accuracy": float(
                evaluations["epoch_0"]["cross_hardware"]["ideal_quantized"][
                    "student_accuracy"
                ]
            ),
            "epoch_10_cross_hardware_ideal_accuracy": float(
                evaluations["epoch_10"]["cross_hardware"]["ideal_quantized"][
                    "student_accuracy"
                ]
            ),
            "epoch_0_cross_hardware_pv_accuracy": float(
                evaluations["epoch_0"]["cross_hardware"][
                    "persistent_mean_accuracy"
                ]
            ),
            "epoch_10_cross_hardware_pv_accuracy": float(
                evaluations["epoch_10"]["cross_hardware"][
                    "persistent_mean_accuracy"
                ]
            ),
        }
        store.append_metric({"mode": "train_terminal", **terminal})
        store.complete(
            metrics=terminal,
            artifacts=(
                store.artifact_record(final_checkpoint, kind="epoch_10_logical_master"),
                store.artifact_record(best_checkpoint, kind="best_logical_master"),
                store.artifact_record(summary_path, kind="scientific_summary"),
            ),
        )
        return 0
    except BaseException as error:
        store.fail(error)
        raise


__all__ = ["run_train"]
