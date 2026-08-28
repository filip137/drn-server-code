"""Native ``ebl validate`` runtime for four-device OM baseline selection."""

from __future__ import annotations

from dataclasses import fields
from hashlib import sha256
from itertools import product
import gc
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
from types import SimpleNamespace
from typing import Any, Iterable, Mapping, Sequence, TYPE_CHECKING

import numpy as np
import torch

from experiments.artifacts import (
    RunStore,
    atomic_write_json,
    canonical_json_bytes,
    sha256_file,
)
from experiments.mnist_relu_drn.components import (
    build_student_stack,
    collect_calibration,
    fit_positive_logit_gain,
)
from experiments.mnist_relu_drn.ibm_om_baseline_selection import (
    BASELINE_POLICIES,
    INDEPENDENT_CELL_RESET_MEAN,
    JointRepairPlan,
    REFERENCE_ENFORCED_DESTINATION_COLUMNS,
    build_joint_repair_plan,
    build_physical_mapping,
    build_weight_error_decomposition,
    fit_weight_reconstruction_scales,
    hardware_instance_fingerprint,
    all_policy_feasibility_from_quads,
    apply_checked_physical_targets,
    apply_joint_repair_field,
    baseline_group_violation_mask,
    quad_stack,
    save_physical_mapping,
)
from experiments.mnist_relu_drn.ibm_om_bounded_codebook_scheme_screen import (
    _binding_slices,
    _evaluate_detailed,
    _float_summary,
    _population_for,
    _tensor_sha256,
)
from experiments.mnist_relu_drn.ibm_om_ideal_mapping_scheme_screen import (
    _ordered_labels,
)
from experiments.mnist_relu_drn.ibm_om_standard_level_scheme_screen import (
    _commission_reset_origin_for_population,
)
from experiments.mnist_relu_drn.runtime import _load_teacher
from experiments.mnist_shared import build_mnist_loaders
from experiments.schema import to_plain_data
from training.ibm_reram_hwa import (
    IbmReramArrayPopulation,
    load_om_array_population,
)
from training.ibm_reram_program_verify import OM_PRESET, derive_seed


if TYPE_CHECKING:
    from ebl.cli import ValidateRequest


_ROOT = Path(__file__).resolve().parents[2]
SUMMARY_SCHEMA = "ebl.mnist_relu_drn.ibm_om_baseline_selection_result"
SUMMARY_SCHEMA_VERSION = 1
JOINT_SCHEMA = "ebl.mnist_relu_drn.ibm_om_baseline_joint_assignment"
JOINT_SCHEMA_VERSION = 1
PREDICTION_SCHEMA = "ebl.mnist_relu_drn.ibm_om_baseline_predictions"
PREDICTION_SCHEMA_VERSION = 1
BASELINE_RESIDUAL_SCHEMA = (
    "ebl.mnist_relu_drn.ibm_om_baseline_rail_voltage_residual"
)
BASELINE_RESIDUAL_SCHEMA_VERSION = 1
LAYOUTS = ("halves", "paired")
_IDENTITY_FIELDS = (
    "max_bound",
    "min_bound",
    "dwmin_up",
    "dwmin_down",
    "reference",
    "corrupt",
    "published_corrupt",
)


def _input(role: str, path: Path) -> dict[str, Any]:
    source = path.expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Expected {role} to name an existing file: {source}.")
    return {"role": role, "path": str(source), "sha256": sha256_file(source)}


def _sampler_python() -> Path:
    raw = os.environ.get("EBL_AIHWKIT_PYTHON")
    if not raw:
        raise ValueError(
            "Expected EBL_AIHWKIT_PYTHON to name the pinned AIHWKit 1.1.0 "
            "interpreter."
        )
    path = Path(raw).expanduser().resolve()
    if not path.is_file() or not os.access(path, os.X_OK):
        raise ValueError(f"Expected an executable EBL_AIHWKIT_PYTHON: {path}.")
    probe = subprocess.run(
        (str(path), "-c", "import aihwkit; print(aihwkit.__version__)"),
        check=False,
        capture_output=True,
        text=True,
    )
    if probe.returncode != 0 or probe.stdout.strip() != "1.1.0":
        raise ValueError(
            "Expected EBL_AIHWKIT_PYTHON to import AIHWKit 1.1.0. "
            f"stdout={probe.stdout!r}, stderr={probe.stderr!r}."
        )
    return path


def _artifact_records(store: RunStore, items: Iterable[Mapping[str, Any]]):
    records = []
    for item in items:
        records.append(
            store.artifact_record(Path(str(item["path"])), kind=str(item["kind"]))
        )
        receipt = item.get("receipt")
        if receipt is not None:
            records.append(
                store.artifact_record(
                    Path(str(receipt)), kind=f"{item['kind']}_receipt"
                )
            )
    return tuple(records)


def _atomic_save_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    destination = path.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _sample_layout_external(
    *,
    binding_keys: Sequence[str],
    binding_shapes: Sequence[Sequence[int]],
    assignment_seed: int,
    corruption_policy: str,
    aihwkit_python: Path,
    population_path: Path,
    receipt_path: Path,
) -> tuple[IbmReramArrayPopulation, Mapping[str, Any]]:
    """Use the pinned sampler for a declared arbitrary two-dimensional layout."""

    request = {
        "preset": OM_PRESET,
        "assignment_seed": int(assignment_seed),
        "corruption_policy": corruption_policy,
        "binding_keys": [str(value) for value in binding_keys],
        "binding_shapes": [list(map(int, shape)) for shape in binding_shapes],
        "required_aihwkit_version": "1.1.0",
    }
    population_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    command = (
        str(aihwkit_python),
        "-m",
        "experiments.reram_program_verify.hwa_population_sampler",
        "--request-json",
        json.dumps(request, allow_nan=False, sort_keys=True, separators=(",", ":")),
        "--output",
        str(population_path),
        "--receipt",
        str(receipt_path),
    )
    environment = os.environ.copy()
    environment.pop("EBL_AIHWKIT_PYTHON", None)
    completed = subprocess.run(
        command,
        cwd=_ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout)[-4000:]
        raise RuntimeError(
            "Pinned external donor sampler failed: "
            f"exit={completed.returncode}, tail={detail!r}."
        )
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError("Expected a readable donor sampling receipt.") from error
    population = load_om_array_population(population_path)
    if (
        receipt.get("request") != request
        or receipt.get("population_sha256") != sha256_file(population_path)
        or receipt.get("population_fingerprint") != population.fingerprint
        or receipt.get("aihwkit_version") != "1.1.0"
        or receipt.get("sampler_source_sha256")
        != sha256_file(
            _ROOT / "experiments/reram_program_verify/hwa_population_sampler.py"
        )
        or receipt.get("population_implementation_sha256")
        != sha256_file(_ROOT / "training/ibm_reram_hwa.py")
        or population.assignment_seed != assignment_seed
        or population.binding_keys != tuple(binding_keys)
        or population.binding_shapes
        != tuple(tuple(int(value) for value in shape) for shape in binding_shapes)
    ):
        raise RuntimeError("Expected the donor population and receipt to match exactly.")
    return population, receipt


def _population_layers(
    population: IbmReramArrayPopulation, name: str
) -> tuple[torch.Tensor, torch.Tensor]:
    slices = _binding_slices(population)
    value = getattr(population, name).detach().cpu()
    layers = tuple(
        value[slices[key]].reshape(shape).clone()
        for key, shape in zip(population.binding_keys, population.binding_shapes)
    )
    if len(layers) != 2:
        raise ValueError("Expected two canonical four-device population bindings.")
    return layers[0], layers[1]


def _commissioning_layers(population, commissioning, name: str):
    slices = _binding_slices(population)
    value = getattr(commissioning, name).detach().cpu()
    layers = tuple(
        value[slices[key]].reshape(shape).clone()
        for key, shape in zip(population.binding_keys, population.binding_shapes)
    )
    if len(layers) != 2:
        raise ValueError("Expected two canonical commissioning bindings.")
    return layers


def _unit_bounds(native: torch.Tensor) -> torch.Tensor:
    return ((native.to(torch.float64) + 1.0) / 2.0).clamp(0.0, 1.0)


def _bounded_reset(
    mean_raw_a: torch.Tensor, lower_unit: torch.Tensor, upper_unit: torch.Tensor
) -> torch.Tensor:
    value = ((mean_raw_a.to(torch.float64) + 1.0) / 2.0).clamp(0.0, 1.0)
    return torch.minimum(torch.maximum(value, lower_unit), upper_unit)


def _base_feasibility(
    logical_weights: Sequence[torch.Tensor],
    lower_layers: Sequence[torch.Tensor],
    upper_layers: Sequence[torch.Tensor],
    reset_layers: Sequence[torch.Tensor],
    reference_layers: Sequence[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    result = []
    for logical, lower, upper, reset, reference, layout in zip(
        logical_weights,
        lower_layers,
        upper_layers,
        reset_layers,
        reference_layers,
        LAYOUTS,
    ):
        feasibility = all_policy_feasibility_from_quads(
            logical.detach().cpu().to(torch.float64),
            quad_stack(lower, layout=layout),
            quad_stack(upper, layout=layout),
            quad_stack(reset, layout=layout),
            quad_stack((reference.to(torch.float64) + 1.0) / 2.0, layout=layout),
        )["all_policies"]
        result.append(feasibility)
    return result[0], result[1]


def _donor_quads(
    population: IbmReramArrayPopulation,
    value: torch.Tensor,
    failure_counts: Sequence[int],
    maximum_attempts: int,
) -> torch.Tensor:
    slices = _binding_slices(population)
    blocks = []
    for key, count in zip(population.binding_keys, failure_counts):
        raw = value[slices[key]].reshape(-1, 4)
        required = int(count) * maximum_attempts
        if raw.shape[0] < max(required, 1):
            raise RuntimeError("Expected the donor binding to cover its private blocks.")
        if required:
            blocks.append(raw[:required].reshape(int(count), maximum_attempts, 4))
    if blocks:
        return torch.cat(blocks, dim=0)
    return torch.empty((0, maximum_attempts, 4), dtype=value.dtype)


def _save_joint_assignment(
    path: Path,
    *,
    assignment_seed: int,
    hardware_instance_id: str,
    base_population: IbmReramArrayPopulation,
    donor_population: IbmReramArrayPopulation | None,
    plan: JointRepairPlan,
    identity_layers: Mapping[str, Sequence[torch.Tensor]],
    reset_mean_layers: Sequence[torch.Tensor],
    reset_se_layers: Sequence[torch.Tensor],
    report: Mapping[str, Any],
) -> Mapping[str, Any]:
    arrays: dict[str, np.ndarray] = {
        "schema": np.asarray(JOINT_SCHEMA),
        "schema_version": np.asarray(JOINT_SCHEMA_VERSION, dtype=np.int64),
        "assignment_seed": np.asarray(assignment_seed, dtype=np.int64),
        "hardware_instance_id": np.asarray(hardware_instance_id),
        "base_population_fingerprint": np.asarray(base_population.fingerprint),
        "donor_population_fingerprint": np.asarray(
            donor_population.fingerprint if donor_population is not None else "none"
        ),
        "maximum_attempts": np.asarray(plan.maximum_attempts, dtype=np.int64),
        "failure_layer_index": plan.failure_layer_index.numpy(),
        "failure_flat_index": plan.failure_flat_index.numpy(),
        "selected_attempt": plan.selected_attempt.numpy(),
    }
    for layer_index, attempt in enumerate(plan.attempts_by_layer):
        arrays[f"layer_{layer_index}_selected_attempt_grid"] = attempt.numpy()
        arrays[f"layer_{layer_index}_reset_mean_raw_a"] = (
            reset_mean_layers[layer_index].contiguous().numpy()
        )
        arrays[f"layer_{layer_index}_reset_standard_error_raw_a"] = (
            reset_se_layers[layer_index].contiguous().numpy()
        )
        for name, values in identity_layers.items():
            arrays[f"layer_{layer_index}_{name}"] = (
                values[layer_index].contiguous().numpy()
            )
    _atomic_save_npz(path, arrays)
    receipt_path = path.with_suffix(".receipt.json")
    receipt = {
        "schema": JOINT_SCHEMA,
        "schema_version": JOINT_SCHEMA_VERSION,
        "assignment_seed": assignment_seed,
        "hardware_instance_id": hardware_instance_id,
        "artifact": path.name,
        "artifact_sha256": sha256_file(path),
        "report": dict(report),
        "tensor_hashes": {
            name: sha256(array.tobytes(order="C")).hexdigest()
            for name, array in arrays.items()
            if isinstance(array, np.ndarray) and array.ndim > 0
        },
    }
    atomic_write_json(receipt_path, receipt)
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "receipt": str(receipt_path),
        "receipt_sha256": sha256_file(receipt_path),
        "hardware_instance_id": hardware_instance_id,
    }


def _prepare_assignment(
    *,
    assignment_seed: int,
    assignment_role: str,
    logical_weights: Sequence[torch.Tensor],
    stack,
    protocol,
    aihwkit_python: Path,
    artifact_root: Path,
) -> tuple[Mapping[str, Any], list[Mapping[str, Any]]]:
    population_contract = SimpleNamespace(
        corruption_policy=protocol.device.corruption_policy,
        required_aihwkit_version=protocol.device.required_aihwkit_version,
    )
    base_population, population_report = _population_for(
        stack=stack,
        topology=4,
        assignment_seed=assignment_seed,
        contract=population_contract,
        aihwkit_python=aihwkit_python,
        output_dir=artifact_root / assignment_role / "base",
    )
    population_report = dict(population_report)
    try:
        declared_dw_min_std = population_report.pop(
            "declared_dw_min_std_but_disabled"
        )
        declared_write_noise_std = population_report.pop(
            "declared_write_noise_std_but_disabled"
        )
    except KeyError as error:
        raise RuntimeError(
            "Expected the reused population report to expose its declared OM "
            "commissioning-noise parameters."
        ) from error
    population_report["noise_partition"] = {
        "commissioning_cycle_to_cycle_noise": "enabled",
        "commissioning_apparent_write_noise": "enabled",
        "mapped_target_write_noise": "disabled",
        "declared_dw_min_std": declared_dw_min_std,
        "declared_write_noise_std": declared_write_noise_std,
    }
    base_commissioning, base_commissioning_report = (
        _commission_reset_origin_for_population(
            base_population,
            topology=4,
            read_samples=protocol.reset_commissioning.samples_per_cell,
            output_dir=artifact_root / assignment_role / "base",
        )
    )
    identity_base = {
        name: _population_layers(base_population, name) for name in _IDENTITY_FIELDS
    }
    mean_base = _commissioning_layers(
        base_population, base_commissioning, "reset_mean_raw_a"
    )
    se_base = _commissioning_layers(
        base_population, base_commissioning, "reset_standard_error_raw_a"
    )
    lower_base = tuple(_unit_bounds(value) for value in identity_base["min_bound"])
    upper_base = tuple(_unit_bounds(value) for value in identity_base["max_bound"])
    reset_base = tuple(
        _bounded_reset(mean, lower, upper)
        for mean, lower, upper in zip(mean_base, lower_base, upper_base)
    )
    feasible_base = _base_feasibility(
        logical_weights,
        lower_base,
        upper_base,
        reset_base,
        identity_base["reference"],
    )
    failure_counts = [int((~value).sum().item()) for value in feasible_base]
    maximum_attempts = protocol.joint_assignment_repair.maximum_candidates_per_quad
    donor_population = None
    donor_population_report = None
    donor_commissioning_report = None
    donor_identity: dict[str, torch.Tensor] = {}
    donor_mean = torch.empty((0, maximum_attempts, 4), dtype=torch.float32)
    donor_se = torch.empty((0, maximum_attempts, 4), dtype=torch.float32)
    if sum(failure_counts):
        donor_seed = derive_seed(
            assignment_seed,
            "mnist_ibm_om_baseline_selection_joint_donor_v1",
        )
        donor_shapes = tuple(
            (max(count * maximum_attempts, 1), 4) for count in failure_counts
        )
        donor_keys = ("joint_donor.layer0", "joint_donor.layer1")
        donor_path = artifact_root / assignment_role / "donor" / "population.npz"
        donor_receipt_path = donor_path.with_suffix(".receipt.json")
        donor_population, donor_receipt = _sample_layout_external(
            binding_keys=donor_keys,
            binding_shapes=donor_shapes,
            assignment_seed=donor_seed,
            corruption_policy=protocol.device.corruption_policy,
            aihwkit_python=aihwkit_python,
            population_path=donor_path,
            receipt_path=donor_receipt_path,
        )
        if (
            donor_population.nominal_dw_min != base_population.nominal_dw_min
            or donor_population.dw_min_std != base_population.dw_min_std
            or donor_population.write_noise_std != base_population.write_noise_std
        ):
            raise RuntimeError("Expected base and donor OM scalar parameters to match.")
        donor_population_report = {
            "path": str(donor_path),
            "sha256": sha256_file(donor_path),
            "receipt": str(donor_receipt_path),
            "receipt_sha256": sha256_file(donor_receipt_path),
            "assignment_seed": donor_seed,
            "population_fingerprint": donor_population.fingerprint,
            "cells": donor_population.size,
            "request": donor_receipt["request"],
            "noise_partition": {
                "commissioning_cycle_to_cycle_noise": "enabled",
                "commissioning_apparent_write_noise": "enabled",
                "mapped_target_write_noise": "disabled",
                "declared_dw_min_std": donor_population.dw_min_std,
                "declared_write_noise_std": donor_population.write_noise_std,
            },
        }
        donor_commissioning, donor_commissioning_report = (
            _commission_reset_origin_for_population(
                donor_population,
                topology=4,
                read_samples=protocol.reset_commissioning.samples_per_cell,
                output_dir=artifact_root / assignment_role / "donor",
            )
        )
        for name in _IDENTITY_FIELDS:
            donor_identity[name] = _donor_quads(
                donor_population,
                getattr(donor_population, name).detach().cpu(),
                failure_counts,
                maximum_attempts,
            )
        donor_mean = _donor_quads(
            donor_population,
            donor_commissioning.reset_mean_raw_a,
            failure_counts,
            maximum_attempts,
        )
        donor_se = _donor_quads(
            donor_population,
            donor_commissioning.reset_standard_error_raw_a,
            failure_counts,
            maximum_attempts,
        )
        donor_lower = _unit_bounds(donor_identity["min_bound"])
        donor_upper = _unit_bounds(donor_identity["max_bound"])
        donor_reset = _bounded_reset(donor_mean, donor_lower, donor_upper)
        failure_sign = torch.cat(
            tuple(
                logical.detach().cpu().to(torch.float64).reshape(-1)[
                    torch.where(~mask.reshape(-1))[0]
                ]
                for logical, mask in zip(logical_weights, feasible_base)
            )
        )
        donor_feasible = all_policy_feasibility_from_quads(
            failure_sign[:, None].expand(-1, maximum_attempts),
            donor_lower,
            donor_upper,
            donor_reset,
            (donor_identity["reference"].to(torch.float64) + 1.0) / 2.0,
        )["all_policies"]
    else:
        donor_feasible = torch.empty((0, maximum_attempts), dtype=torch.bool)

    plan = build_joint_repair_plan(
        feasible_base,
        donor_feasible,
        layouts=LAYOUTS,
        maximum_attempts=maximum_attempts,
    )
    identity_final = {
        name: apply_joint_repair_field(
            values,
            donor_identity.get(
                name,
                torch.empty((0, maximum_attempts, 4), dtype=values[0].dtype),
            ),
            plan,
        )
        for name, values in identity_base.items()
    }
    mean_final = apply_joint_repair_field(mean_base, donor_mean, plan)
    se_final = apply_joint_repair_field(se_base, donor_se, plan)
    lower_final = tuple(_unit_bounds(value) for value in identity_final["min_bound"])
    upper_final = tuple(_unit_bounds(value) for value in identity_final["max_bound"])
    reset_final = tuple(
        _bounded_reset(mean, lower, upper)
        for mean, lower, upper in zip(mean_final, lower_final, upper_final)
    )
    final_feasible = _base_feasibility(
        logical_weights,
        lower_final,
        upper_final,
        reset_final,
        identity_final["reference"],
    )
    if not all(bool(value.all()) for value in final_feasible):
        raise RuntimeError("Expected joint repair to make every policy feasible.")
    semantic_metadata = {
        "schema": JOINT_SCHEMA,
        "schema_version": JOINT_SCHEMA_VERSION,
        "assignment_seed": assignment_seed,
        "assignment_role": assignment_role,
        "base_population_fingerprint": base_population.fingerprint,
        "donor_population_fingerprint": (
            donor_population.fingerprint if donor_population is not None else None
        ),
        "binding_keys": list(base_population.binding_keys),
        "binding_shapes": [list(shape) for shape in base_population.binding_shapes],
        "layouts": list(LAYOUTS),
        "read_samples": protocol.reset_commissioning.samples_per_cell,
        "maximum_attempts": maximum_attempts,
        "selected_attempts": plan.selected_attempt.tolist(),
        "failure_layer_index": plan.failure_layer_index.tolist(),
        "failure_flat_index": plan.failure_flat_index.tolist(),
    }
    fingerprint_tensors: dict[str, torch.Tensor] = {}
    for layer_index in range(2):
        for name, values in identity_final.items():
            fingerprint_tensors[f"layer_{layer_index}_{name}"] = values[layer_index]
        fingerprint_tensors[f"layer_{layer_index}_reset_mean_raw_a"] = mean_final[
            layer_index
        ]
        fingerprint_tensors[f"layer_{layer_index}_reset_standard_error_raw_a"] = (
            se_final[layer_index]
        )
    hardware_id = hardware_instance_fingerprint(
        semantic_metadata,
        fingerprint_tensors,
    )
    repair_report = {
        **semantic_metadata,
        "hardware_instance_id": hardware_id,
        "failure_count": plan.failure_count,
        "failure_count_by_layer": failure_counts,
        "final_corrupt_count": sum(
            int(value.to(torch.int64).sum().item())
            for value in identity_final["corrupt"]
        ),
        "final_corrupt_count_by_layer": [
            int(value.to(torch.int64).sum().item())
            for value in identity_final["corrupt"]
        ],
        "published_corrupt_count": sum(
            int(value.to(torch.int64).sum().item())
            for value in identity_final["published_corrupt"]
        ),
        "published_corrupt_count_by_layer": [
            int(value.to(torch.int64).sum().item())
            for value in identity_final["published_corrupt"]
        ],
        "selected_attempt": {
            "minimum": int(plan.selected_attempt.min().item()) if plan.failure_count else None,
            "maximum": int(plan.selected_attempt.max().item()) if plan.failure_count else None,
            "mean": (
                float(plan.selected_attempt.to(torch.float64).mean().item())
                if plan.failure_count
                else None
            ),
        },
        "final_policy_coverage": {
            "feasible_quads": [int(value.sum().item()) for value in final_feasible],
            "total_quads": [int(value.numel()) for value in final_feasible],
        },
        "commissioned_reset_mean_raw_a": [
            _float_summary(value) for value in mean_final
        ],
        "bounded_reset_baseline_unit": [
            _float_summary(value) for value in reset_final
        ],
        "reference_outside_public_coordinate_count": [
            int(((value < -1.0) | (value > 1.0)).sum().item())
            for value in identity_final["reference"]
        ],
    }
    joint_artifact = _save_joint_assignment(
        artifact_root / assignment_role / "joint_assignment.npz",
        assignment_seed=assignment_seed,
        hardware_instance_id=hardware_id,
        base_population=base_population,
        donor_population=donor_population,
        plan=plan,
        identity_layers=identity_final,
        reset_mean_layers=mean_final,
        reset_se_layers=se_final,
        report=repair_report,
    )
    artifacts: list[Mapping[str, Any]] = [
        {**population_report, "kind": "ibm_om_base_population"},
        {**base_commissioning_report, "kind": "ibm_om_base_reset_commissioning"},
        {**joint_artifact, "kind": "ibm_om_joint_assignment"},
    ]
    if donor_population_report is not None and donor_commissioning_report is not None:
        artifacts.extend(
            (
                {**donor_population_report, "kind": "ibm_om_donor_population"},
                {
                    **donor_commissioning_report,
                    "kind": "ibm_om_donor_reset_commissioning",
                },
            )
        )
    prepared = {
        "assignment_seed": assignment_seed,
        "assignment_role": assignment_role,
        "hardware_instance_id": hardware_id,
        "population": population_report,
        "base_commissioning": base_commissioning_report,
        "donor_population": donor_population_report,
        "donor_commissioning": donor_commissioning_report,
        "joint_assignment": joint_artifact,
        "repair": repair_report,
        "nominal_dw_min": base_population.nominal_dw_min,
        "cell_lower_units": lower_final,
        "cell_upper_units": upper_final,
        "reset_baseline_units": reset_final,
        "intrinsic_references_native": identity_final["reference"],
    }
    return prepared, artifacts


def _fit_calibration(stack, teacher, loader, labels, spec) -> Mapping[str, Any]:
    raw_scores, teacher_logits = collect_calibration(stack, teacher, loader)
    fitted = fit_positive_logit_gain(
        raw_scores,
        teacher_logits,
        gain_min=spec.mapping.logit_gain_min,
        gain_max=spec.mapping.logit_gain_max,
        steps=spec.mapping.logit_gain_steps,
    )
    gain = float(fitted["gain"])
    student_prediction = (raw_scores * gain).argmax(dim=1).detach().cpu()
    teacher_prediction = teacher_logits.argmax(dim=1).detach().cpu()
    labels = labels[: student_prediction.numel()]
    return {
        **fitted,
        "student_correct": int(student_prediction.eq(labels).sum().item()),
        "teacher_correct": int(teacher_prediction.eq(labels).sum().item()),
        "teacher_agreement_count": int(
            student_prediction.eq(teacher_prediction).sum().item()
        ),
        "examples": int(labels.numel()),
        "student_accuracy": float(student_prediction.eq(labels).double().mean().item()),
        "teacher_accuracy": float(teacher_prediction.eq(labels).double().mean().item()),
        "teacher_agreement": float(
            student_prediction.eq(teacher_prediction).double().mean().item()
        ),
        "gain_source": "per_policy_continuous_development_fit",
        "test_labels_used": False,
    }


def _selected_candidate(candidates: Sequence[Mapping[str, Any]]) -> int:
    return min(
        range(len(candidates)),
        key=lambda index: (
            -int(candidates[index]["calibration"]["student_correct"]),
            float(candidates[index]["calibration"]["calibrated_kl"]),
            tuple(float(value) for value in candidates[index]["scale_fractions"]),
        ),
    )


def _teacher_predictions(teacher, loader, *, device, sample_limit: int | None):
    labels_out = []
    predictions = []
    examples = 0
    with torch.no_grad():
        for inputs, labels in loader:
            if sample_limit is not None:
                remaining = sample_limit - examples
                if remaining <= 0:
                    break
                inputs = inputs[:remaining]
                labels = labels[:remaining]
            inputs = inputs.to(device, dtype=torch.float32)
            labels_out.append(labels.detach().cpu().to(torch.int64))
            predictions.append(teacher.logits(inputs).argmax(dim=1).detach().cpu())
            examples += int(labels.shape[0])
    if not labels_out:
        raise RuntimeError("Expected prediction evaluation examples.")
    return torch.cat(labels_out), torch.cat(predictions)


def _dataset_provenance(loaders, calibration_labels: torch.Tensor) -> Mapping[str, Any]:
    """Hash the exact MNIST source tensors and frozen calibration indices."""

    test_wrapper = loaders.test.dataset
    test_dataset = getattr(test_wrapper, "dataset", None)
    raw_test = getattr(test_dataset, "dataset", test_dataset)
    data = getattr(raw_test, "data", None)
    targets = getattr(raw_test, "targets", None)
    calibration_subset = loaders.calibration.dataset
    indices = getattr(calibration_subset, "indices", None)
    calibration_wrapper = getattr(calibration_subset, "dataset", None)
    raw_train = getattr(calibration_wrapper, "dataset", calibration_wrapper)
    train_data = getattr(raw_train, "data", None)
    train_targets = getattr(raw_train, "targets", None)
    if not (
        isinstance(data, torch.Tensor)
        and isinstance(targets, torch.Tensor)
        and isinstance(train_data, torch.Tensor)
        and isinstance(train_targets, torch.Tensor)
        and indices is not None
        and len(targets) == len(test_wrapper)
    ):
        raise RuntimeError("Expected canonical tensor-backed MNIST test/calibration data.")
    index_tensor = torch.as_tensor(indices, dtype=torch.int64)
    selected_train_data = train_data[index_tensor]
    selected_train_targets = train_targets[index_tensor].to(torch.int64)
    if not torch.equal(
        selected_train_targets,
        calibration_labels.detach().cpu().to(torch.int64),
    ):
        raise RuntimeError("Expected calibration labels to match raw MNIST targets.")
    return {
        "dataset": "MNIST",
        "test_examples": int(len(test_wrapper)),
        "test_data_sha256": _tensor_sha256(data.detach().cpu()),
        "test_targets_sha256": _tensor_sha256(targets.detach().cpu().to(torch.int64)),
        "training_examples": int(train_targets.numel()),
        "training_data_sha256": _tensor_sha256(train_data.detach().cpu()),
        "training_targets_sha256": _tensor_sha256(
            train_targets.detach().cpu().to(torch.int64)
        ),
        "calibration_examples": int(index_tensor.numel()),
        "calibration_indices_sha256": _tensor_sha256(index_tensor),
        "calibration_raw_images_sha256": _tensor_sha256(selected_train_data),
        "calibration_raw_targets_sha256": _tensor_sha256(selected_train_targets),
        "calibration_labels_sha256": _tensor_sha256(calibration_labels),
        "test_split_labels_used_for_mapping_or_calibration": False,
    }


def _rail_difference(state: torch.Tensor, *, layout: str) -> torch.Tensor:
    if state.ndim != 2 or state.shape[1] % 2:
        raise ValueError("Expected an even-width rank-2 rail state.")
    if layout == "halves":
        half = state.shape[1] // 2
        return state[:, :half] - state[:, half:]
    if layout == "paired":
        return state[:, 0::2] - state[:, 1::2]
    raise ValueError(f"Unexpected rail layout: {layout!r}.")


def _baseline_rail_residual(
    stack,
    loader,
    *,
    sample_limit: int | None,
) -> tuple[Mapping[str, Any], tuple[torch.Tensor, ...]]:
    """Solve the all-``d=0`` circuit and report paired rail imbalance."""

    accumulators = [[], []]
    examples = 0
    with torch.no_grad():
        for inputs, _labels in loader:
            if sample_limit is not None:
                remaining = sample_limit - examples
                if remaining <= 0:
                    break
                inputs = inputs[:remaining]
            inputs = inputs.to(stack.device, dtype=torch.float32)
            stack.network.set_input(inputs, reset=True)
            stack.minimizer.compute_equilibrium()
            layers = tuple(stack.bundle.energy.layers())
            if len(layers) != 3:
                raise RuntimeError("Expected input, hidden, and output DRN layers.")
            accumulators[0].append(
                _rail_difference(layers[1].state.detach(), layout="halves").cpu()
            )
            accumulators[1].append(
                _rail_difference(layers[2].state.detach(), layout="paired").cpu()
            )
            examples += int(inputs.shape[0])
    if examples == 0:
        raise RuntimeError("Expected baseline-only residual examples.")
    reports = []
    hashes = []
    residuals = []
    for layer_index, pieces in enumerate(accumulators):
        residual = torch.cat(pieces).to(torch.float32)
        value = residual.to(torch.float64)
        absolute = value.abs().reshape(-1)
        reports.append(
            {
                "layer": layer_index,
                "layout": LAYOUTS[layer_index],
                "values": int(value.numel()),
                "signed_mean": float(value.mean().item()),
                "rms": float(value.square().mean().sqrt().item()),
                "p95_absolute": float(torch.quantile(absolute, 0.95).item()),
                "maximum_absolute": float(absolute.max().item()),
            }
        )
        hashes.append(_tensor_sha256(residual))
        residuals.append(residual)
    return (
        {
            "definition": (
                "paired_rail_voltage_difference_with_all_offsets_d_equal_zero"
            ),
            "examples": examples,
            "layers": reports,
            "residual_hashes": hashes,
        },
        tuple(residuals),
    )


def _save_baseline_rail_residual(
    path: Path,
    *,
    report: Mapping[str, Any],
    residuals: Sequence[torch.Tensor],
) -> Mapping[str, Any]:
    if len(residuals) != 2:
        raise ValueError("Expected hidden and output baseline rail residual tensors.")
    tensor_hashes = {
        f"layer_{index}_rail_voltage_residual": _tensor_sha256(value)
        for index, value in enumerate(residuals)
    }
    if list(tensor_hashes.values()) != list(report.get("residual_hashes", ())):
        raise ValueError("Expected residual tensor hashes to match the report.")
    arrays = {
        "schema": np.asarray(BASELINE_RESIDUAL_SCHEMA),
        "schema_version": np.asarray(
            BASELINE_RESIDUAL_SCHEMA_VERSION, dtype=np.int64
        ),
        "examples": np.asarray(int(report["examples"]), dtype=np.int64),
        **{
            f"layer_{index}_rail_voltage_residual": value.detach()
            .cpu()
            .to(torch.float32)
            .numpy()
            for index, value in enumerate(residuals)
        },
    }
    _atomic_save_npz(path, arrays)
    receipt_path = path.with_suffix(".receipt.json")
    receipt = {
        "schema": BASELINE_RESIDUAL_SCHEMA,
        "schema_version": BASELINE_RESIDUAL_SCHEMA_VERSION,
        "artifact": path.name,
        "artifact_sha256": sha256_file(path),
        "report": dict(report),
        "tensor_hashes": tensor_hashes,
    }
    atomic_write_json(receipt_path, receipt)
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "receipt": str(receipt_path),
        "receipt_sha256": sha256_file(receipt_path),
        "report": dict(report),
    }


def _apply_baseline(stack, mapping) -> None:
    span = mapping.conductance_max - mapping.conductance_min
    checked = []
    for binding, layer in zip(stack.bundle.catalog.trainable, mapping.layers):
        lower = mapping.conductance_min + span * layer.cell_lower_unit
        upper = mapping.conductance_min + span * layer.cell_upper_unit
        baseline = layer.baseline
        tolerance = max(1e-12, span * 1e-6)
        if (
            baseline.shape != binding.state.shape
            or not bool(torch.isfinite(baseline).all())
            or bool(torch.any(baseline < lower - tolerance))
            or bool(torch.any(baseline > upper + tolerance))
        ):
            raise RuntimeError("Expected a finite in-bound full baseline target.")
        checked.append((binding.state, baseline))
    with torch.no_grad():
        for state, value in checked:
            state.copy_(value.to(device=state.device, dtype=state.dtype))


def _save_predictions(
    path: Path,
    *,
    assignment_seed: int,
    policy: str,
    labels: torch.Tensor,
    teacher_prediction: torch.Tensor,
    continuous_prediction: torch.Tensor,
    quantized_prediction: torch.Tensor,
) -> Mapping[str, Any]:
    arrays = {
        "schema": np.asarray(PREDICTION_SCHEMA),
        "schema_version": np.asarray(PREDICTION_SCHEMA_VERSION, dtype=np.int64),
        "assignment_seed": np.asarray(assignment_seed, dtype=np.int64),
        "policy": np.asarray(policy),
        "labels": labels.detach().cpu().numpy().astype(np.int64, copy=False),
        "teacher_prediction": teacher_prediction.detach().cpu().numpy().astype(
            np.int64, copy=False
        ),
        "continuous_prediction": continuous_prediction.detach().cpu().numpy().astype(
            np.int64, copy=False
        ),
        "standard4delta_prediction": quantized_prediction.detach().cpu().numpy().astype(
            np.int64, copy=False
        ),
    }
    _atomic_save_npz(path, arrays)
    receipt_path = path.with_suffix(".receipt.json")
    receipt = {
        "schema": PREDICTION_SCHEMA,
        "schema_version": PREDICTION_SCHEMA_VERSION,
        "assignment_seed": assignment_seed,
        "policy": policy,
        "examples": int(labels.numel()),
        "artifact": path.name,
        "artifact_sha256": sha256_file(path),
        "correct": {
            "teacher": int(teacher_prediction.eq(labels).sum().item()),
            "continuous": int(continuous_prediction.eq(labels).sum().item()),
            "standard4delta": int(quantized_prediction.eq(labels).sum().item()),
        },
        "labels_sha256": _tensor_sha256(labels),
        "prediction_hashes": {
            "teacher": _tensor_sha256(teacher_prediction),
            "continuous": _tensor_sha256(continuous_prediction),
            "standard4delta": _tensor_sha256(quantized_prediction),
        },
    }
    atomic_write_json(receipt_path, receipt)
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "receipt": str(receipt_path),
        "receipt_sha256": sha256_file(receipt_path),
        **{key: value for key, value in receipt.items() if key not in {"artifact", "artifact_sha256"}},
    }


def _mapping_invariants(mapping) -> Mapping[str, Any]:
    shared_zero = mapping.policy != INDEPENDENT_CELL_RESET_MEAN
    layers = []
    valid = True
    reasons = []
    for layer in mapping.layers:
        decomposition_continuous = float(
            (layer.continuous_conductance - (layer.baseline + layer.continuous_offset))
            .abs()
            .max()
            .item()
        )
        decomposition_quantized = float(
            (layer.quantized_conductance - (layer.baseline + layer.quantized_offset))
            .abs()
            .max()
            .item()
        )
        zero_max = float(layer.baseline_contrast.abs().max().item())
        zero_tolerance = 1e-9 * layer.level_spacing_physical
        zero_ok = not shared_zero or zero_max <= zero_tolerance
        baseline_group_violation_count = int(
            baseline_group_violation_mask(
                layer.baseline,
                layout=layer.layout,
                policy=mapping.policy,
            )
            .sum()
            .item()
        )
        span = mapping.conductance_max - mapping.conductance_min
        lower = mapping.conductance_min + span * layer.cell_lower_unit
        upper = mapping.conductance_min + span * layer.cell_upper_unit
        target_tolerance = max(1e-12, span * 1e-6)
        bound_violations = 0
        for target in (layer.baseline, layer.continuous_conductance, layer.quantized_conductance):
            bound_violations += int(
                ((target < lower - target_tolerance) | (target > upper + target_tolerance))
                .sum()
                .item()
            )
        reference_unit = layer.intrinsic_reference_unit_unclipped
        reference_invalid = layer.reference_mask & (
            (reference_unit < 0.0)
            | (reference_unit > 1.0)
            | (reference_unit < layer.cell_lower_unit)
            | (reference_unit > layer.cell_upper_unit)
        )
        reference_invalid_count = int(reference_invalid.sum().item())
        layer_valid = (
            decomposition_continuous == 0.0
            and decomposition_quantized == 0.0
            and zero_ok
            and baseline_group_violation_count == 0
            and bound_violations == 0
            and reference_invalid_count == 0
        )
        if not layer_valid:
            valid = False
            reasons.append(f"layer_{layer.layer_index}_physical_invariant")
        zero_in_levels = layer.baseline_contrast.to(torch.float64) / float(
            layer.level_spacing_physical
        )
        zero_abs = zero_in_levels.abs().reshape(-1)
        baseline_rows = layer.baseline_row_loading.to(torch.float64)
        baseline_columns = layer.baseline_column_loading.to(torch.float64)
        row_imbalance = baseline_rows[..., 0] - baseline_rows[..., 1]
        column_imbalance = baseline_columns[..., 0] - baseline_columns[..., 1]
        layers.append(
            {
                "layer": layer.layer_index,
                "continuous_decomposition_max_abs_residual": decomposition_continuous,
                "standard4delta_decomposition_max_abs_residual": decomposition_quantized,
                "baseline_contrast_rms": float(
                    layer.baseline_contrast.to(torch.float64).square().mean().sqrt().item()
                ),
                "baseline_contrast_max_abs": zero_max,
                "zero_tolerance": zero_tolerance,
                "exact_zero_required": shared_zero,
                "exact_zero_passed": zero_ok,
                "declared_baseline_group_violation_count": (
                    baseline_group_violation_count
                ),
                "bound_violation_count": bound_violations,
                "selected_reference_public_or_bound_violation_count": (
                    reference_invalid_count
                ),
                "baseline_contrast_in_levels": {
                    "rms": float(zero_in_levels.square().mean().sqrt().item()),
                    "p95_absolute": float(torch.quantile(zero_abs, 0.95).item()),
                    "maximum_absolute": float(zero_abs.max().item()),
                    "fraction_at_least_half_level": float(
                        (zero_abs >= 0.5).to(torch.float64).mean().item()
                    ),
                },
                "baseline_loading": _float_summary(layer.baseline_loading),
                "baseline_source_row_loading_imbalance": _float_summary(
                    row_imbalance
                ),
                "baseline_destination_column_loading_imbalance": _float_summary(
                    column_imbalance
                ),
                "continuous_loading": _float_summary(layer.continuous_loading),
                "standard4delta_loading": _float_summary(layer.quantized_loading),
                "selected_headroom_unit": _float_summary(layer.selected_headroom_unit),
                "selected_level_capacity": _float_summary(
                    layer.selected_level_capacity.to(torch.float64)
                ),
            }
        )
    return {"valid": valid, "invalid_reasons": reasons, "layers": layers}


def run_validate(request: "ValidateRequest") -> int:
    from experiments.mnist_relu_drn.ibm_om_baseline_selection_config import (
        BaselineSelectionValidateSpec,
    )

    spec = request.spec
    if not isinstance(spec, BaselineSelectionValidateSpec):
        raise TypeError(
            "Expected mnist_ibm_om_baseline_selection.v1 to resolve its dedicated spec."
        )
    if request.teacher_weights is not None or request.device_model is not None:
        raise ValueError(
            "Expected only --weights; teacher/device-model inputs are excluded."
        )
    weights_path = request.weights.expanduser().resolve()
    protocol = spec.protocol
    if sha256_file(weights_path) != protocol.source.expected_weights_sha256:
        raise ValueError("Frozen ReLU source SHA-256 does not match the contract.")
    sampler = _sampler_python()
    store = RunStore.create(
        output_root=request.output_dir,
        experiment_id=spec.experiment_id,
        resolved_config=to_plain_data(spec),
        command=request.command,
        repo_root=_ROOT,
        input_artifacts=(
            _input("weights", weights_path),
            _input("aihwkit_python", sampler),
        ),
        resume_capability="unsupported",
    )
    try:
        student_spec = spec.student
        torch.manual_seed(student_spec.runtime.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(student_spec.runtime.seed)
        device = torch.device(student_spec.runtime.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("The production baseline contract requires CUDA.")
        loaders = build_mnist_loaders(
            student_spec.data,
            data_seed=student_spec.runtime.data_seed,
            calibration_examples=student_spec.mapping.calibration_examples,
            calibration_batch_size=student_spec.mapping.calibration_batch_size,
        )
        teacher, teacher_metadata = _load_teacher(
            weights_path,
            device=device,
            spec=student_spec,
        )
        logical_weights = tuple(
            value.detach().cpu().clone() for value in teacher.parameters()
        )
        if len(logical_weights) != 2:
            raise RuntimeError("Expected the frozen ReLU source to contain two weights.")
        source_report = {
            "role": protocol.source.weights_role,
            "path": str(weights_path),
            "sha256": sha256_file(weights_path),
            "metadata": teacher_metadata,
            "logical_weight_hashes": [_tensor_sha256(value) for value in logical_weights],
            "optimizer_updates": 0,
        }
        artifacts: list[Mapping[str, Any]] = []
        artifact_root = store.run_dir / "artifacts"

        torch.manual_seed(student_spec.runtime.seed)
        dev_stack = build_student_stack(student_spec, enable_measured=False)
        development_hardware, development_artifacts = _prepare_assignment(
            assignment_seed=protocol.assignments.development_seed,
            assignment_role="development",
            logical_weights=logical_weights,
            stack=dev_stack,
            protocol=protocol,
            aihwkit_python=sampler,
            artifact_root=artifact_root,
        )
        artifacts.extend(development_artifacts)
        calibration_labels = _ordered_labels(loaders.calibration)
        data_provenance = _dataset_provenance(loaders, calibration_labels)
        candidates = []
        for scale_pair in product(protocol.mapping.scale_fractions, repeat=2):
            mapping = build_physical_mapping(
                logical_weights,
                policy=protocol.baseline.selected_policy,
                scale_fractions=scale_pair,
                nominal_dw_min=development_hardware["nominal_dw_min"],
                conductance_min=student_spec.model.conductance_min,
                conductance_max=student_spec.model.conductance_max,
                cell_lower_units=development_hardware["cell_lower_units"],
                cell_upper_units=development_hardware["cell_upper_units"],
                reset_baseline_units=development_hardware["reset_baseline_units"],
                intrinsic_references_native=development_hardware[
                    "intrinsic_references_native"
                ],
            )
            apply_checked_physical_targets(
                dev_stack.bundle.catalog, mapping, endpoint="continuous"
            )
            calibration = _fit_calibration(
                dev_stack,
                teacher,
                loaders.calibration,
                calibration_labels,
                student_spec,
            )
            candidates.append(
                {
                    "scale_fractions": [float(value) for value in scale_pair],
                    "calibration": calibration,
                    "target_hashes": [
                        layer["hashes"]["continuous_conductance"]
                        for layer in mapping.report["layers"]
                    ],
                }
            )
            del mapping
        selected_index = _selected_candidate(candidates)
        selected = candidates[selected_index]
        selected_pair = tuple(float(value) for value in selected["scale_fractions"])
        selected_gain = float(selected["calibration"]["gain"])
        development_mapping = build_physical_mapping(
            logical_weights,
            policy=protocol.baseline.selected_policy,
            scale_fractions=selected_pair,
            nominal_dw_min=development_hardware["nominal_dw_min"],
            conductance_min=student_spec.model.conductance_min,
            conductance_max=student_spec.model.conductance_max,
            cell_lower_units=development_hardware["cell_lower_units"],
            cell_upper_units=development_hardware["cell_upper_units"],
            reset_baseline_units=development_hardware["reset_baseline_units"],
            intrinsic_references_native=development_hardware[
                "intrinsic_references_native"
            ],
        )
        development_invariants = _mapping_invariants(development_mapping)
        if not development_invariants["valid"]:
            raise RuntimeError("Development full-conductance invariants failed.")
        analysis_scales = fit_weight_reconstruction_scales(
            logical_weights, development_mapping
        )
        development_mapping_artifact = save_physical_mapping(
            artifact_root / "development" / "selected_physical_mapping.npz",
            development_mapping,
            assignment_seed=protocol.assignments.development_seed,
            assignment_role="development",
            hardware_instance_id=development_hardware["hardware_instance_id"],
        )
        artifacts.append(
            {**development_mapping_artifact, "kind": "ibm_om_physical_mapping"}
        )
        calibration_receipt = {
            "schema": "ebl.mnist_relu_drn.ibm_om_baseline_development_calibration",
            "schema_version": 1,
            "policy": protocol.baseline.selected_policy,
            "assignment_seed": protocol.assignments.development_seed,
            "hardware_instance_id": development_hardware["hardware_instance_id"],
            "selection_domain": protocol.mapping.selection_domain,
            "selection_metric": protocol.mapping.selection_metric,
            "candidate_count": len(candidates),
            "candidates": candidates,
            "selected_index": selected_index,
            "selected": selected,
            "analysis_scales": list(analysis_scales),
            "calibration_labels_sha256": data_provenance[
                "calibration_labels_sha256"
            ],
            "calibration_indices_sha256": data_provenance[
                "calibration_indices_sha256"
            ],
            "test_labels_used": False,
            "continuous_mapping_artifact_sha256": development_mapping_artifact[
                "sha256"
            ],
        }
        calibration_path = artifact_root / "development" / "calibration.json"
        atomic_write_json(calibration_path, calibration_receipt)
        artifacts.append({"path": str(calibration_path), "kind": "development_calibration"})
        print(
            f"calibrated {protocol.baseline.selected_policy}: "
            f"scales={selected_pair}, gain={selected_gain:.8g}, "
            f"accuracy={100.0 * selected['calibration']['student_accuracy']:.2f}%",
            flush=True,
        )

        del dev_stack
        gc.collect()
        torch.manual_seed(student_spec.runtime.seed)
        heldout_stack = build_student_stack(student_spec, enable_measured=False)
        heldout_seed = protocol.assignments.heldout_seed
        heldout_hardware, heldout_artifacts = _prepare_assignment(
            assignment_seed=heldout_seed,
            assignment_role="heldout",
            logical_weights=logical_weights,
            stack=heldout_stack,
            protocol=protocol,
            aihwkit_python=sampler,
            artifact_root=artifact_root,
        )
        artifacts.extend(heldout_artifacts)
        heldout_mapping = build_physical_mapping(
            logical_weights,
            policy=protocol.baseline.selected_policy,
            scale_fractions=selected_pair,
            nominal_dw_min=heldout_hardware["nominal_dw_min"],
            conductance_min=student_spec.model.conductance_min,
            conductance_max=student_spec.model.conductance_max,
            cell_lower_units=heldout_hardware["cell_lower_units"],
            cell_upper_units=heldout_hardware["cell_upper_units"],
            reset_baseline_units=heldout_hardware["reset_baseline_units"],
            intrinsic_references_native=heldout_hardware[
                "intrinsic_references_native"
            ],
        )
        heldout_invariants = _mapping_invariants(heldout_mapping)
        if not heldout_invariants["valid"]:
            raise RuntimeError("Held-out full-conductance invariants failed.")
        heldout_mapping_artifact = save_physical_mapping(
            artifact_root / "heldout" / "physical_mapping.npz",
            heldout_mapping,
            assignment_seed=heldout_seed,
            assignment_role="heldout",
            hardware_instance_id=heldout_hardware["hardware_instance_id"],
        )
        artifacts.append({**heldout_mapping_artifact, "kind": "ibm_om_physical_mapping"})
        heldout_stack.cost.gain = selected_gain
        apply_checked_physical_targets(
            heldout_stack.bundle.catalog, heldout_mapping, endpoint="continuous"
        )
        continuous_metrics, continuous_prediction = _evaluate_detailed(
            heldout_stack,
            teacher,
            loaders.test,
            sample_limit=student_spec.settings.sample_limit,
        )
        apply_checked_physical_targets(
            heldout_stack.bundle.catalog, heldout_mapping, endpoint="standard4delta"
        )
        standard_metrics, standard_prediction = _evaluate_detailed(
            heldout_stack,
            teacher,
            loaders.test,
            sample_limit=student_spec.settings.sample_limit,
        )
        _apply_baseline(heldout_stack, heldout_mapping)
        baseline_metrics, _baseline_prediction = _evaluate_detailed(
            heldout_stack,
            teacher,
            loaders.test,
            sample_limit=student_spec.settings.sample_limit,
        )
        baseline_rail_residual, baseline_residual_tensors = _baseline_rail_residual(
            heldout_stack,
            loaders.test,
            sample_limit=student_spec.settings.sample_limit,
        )
        baseline_residual_artifact = _save_baseline_rail_residual(
            artifact_root / "heldout" / "baseline_rail_residual.npz",
            report=baseline_rail_residual,
            residuals=baseline_residual_tensors,
        )
        artifacts.append(
            {
                **baseline_residual_artifact,
                "kind": "baseline_rail_residual",
            }
        )
        labels, teacher_prediction = _teacher_predictions(
            teacher,
            loaders.test,
            device=device,
            sample_limit=student_spec.settings.sample_limit,
        )
        if not (
            labels.numel()
            == teacher_prediction.numel()
            == continuous_prediction.numel()
            == standard_prediction.numel()
        ):
            raise RuntimeError("Expected all prediction vectors to cover identical examples.")
        expected_examples = (
            10_000 if protocol.execution.profile == "production" else 32
        )
        observed_example_counts = {
            "continuous": int(continuous_metrics["examples"]),
            "standard4delta": int(standard_metrics["examples"]),
            "baseline_only": int(baseline_metrics["examples"]),
            "baseline_rail_residual": int(baseline_rail_residual["examples"]),
            "predictions": int(labels.numel()),
        }
        expected_examples_passed = all(
            value == expected_examples for value in observed_example_counts.values()
        )
        if not expected_examples_passed:
            raise RuntimeError(
                "Expected exact evaluation coverage: "
                f"expected={expected_examples}, observed={observed_example_counts!r}."
            )
        if (
            int(continuous_prediction.eq(labels).sum().item())
            != continuous_metrics["student_correct"]
            or int(standard_prediction.eq(labels).sum().item())
            != standard_metrics["student_correct"]
            or int(teacher_prediction.eq(labels).sum().item())
            != continuous_metrics["teacher_correct"]
        ):
            raise RuntimeError("Prediction artifact exact counts do not match evaluation.")
        prediction_artifact = _save_predictions(
            artifact_root / "heldout" / "predictions.npz",
            assignment_seed=heldout_seed,
            policy=protocol.baseline.selected_policy,
            labels=labels,
            teacher_prediction=teacher_prediction,
            continuous_prediction=continuous_prediction,
            quantized_prediction=standard_prediction,
        )
        artifacts.append({**prediction_artifact, "kind": "predictions"})
        weight_errors = build_weight_error_decomposition(
            logical_weights,
            heldout_mapping,
            analysis_scales,
        )
        weight_error_report = [dict(value.report) for value in weight_errors]
        weight_error_arrays: dict[str, np.ndarray] = {
            "schema": np.asarray(
                "ebl.mnist_relu_drn.ibm_om_baseline_weight_errors"
            ),
            "schema_version": np.asarray(1, dtype=np.int64),
            "assignment_seed": np.asarray(heldout_seed, dtype=np.int64),
            "policy": np.asarray(protocol.baseline.selected_policy),
        }
        for value in weight_errors:
            for field in fields(value):
                tensor = getattr(value, field.name)
                if isinstance(tensor, torch.Tensor):
                    weight_error_arrays[
                        f"layer_{value.layer_index}_{field.name}"
                    ] = tensor.detach().cpu().numpy()
        weight_error_path = artifact_root / "heldout" / "weight_errors.npz"
        _atomic_save_npz(weight_error_path, weight_error_arrays)
        weight_error_receipt_path = weight_error_path.with_suffix(".receipt.json")
        atomic_write_json(
            weight_error_receipt_path,
            {
                "schema": "ebl.mnist_relu_drn.ibm_om_baseline_weight_errors",
                "schema_version": 1,
                "assignment_seed": heldout_seed,
                "policy": protocol.baseline.selected_policy,
                "artifact": weight_error_path.name,
                "artifact_sha256": sha256_file(weight_error_path),
                "analysis_scales": list(analysis_scales),
                "layers": weight_error_report,
            },
        )
        artifacts.append(
            {
                "path": str(weight_error_path),
                "receipt": str(weight_error_receipt_path),
                "kind": "weight_error_decomposition",
            }
        )

        development_repair_coverage = all(
            feasible == total
            for feasible, total in zip(
                development_hardware["repair"]["final_policy_coverage"][
                    "feasible_quads"
                ],
                development_hardware["repair"]["final_policy_coverage"][
                    "total_quads"
                ],
            )
        )
        heldout_repair_coverage = all(
            feasible == total
            for feasible, total in zip(
                heldout_hardware["repair"]["final_policy_coverage"][
                    "feasible_quads"
                ],
                heldout_hardware["repair"]["final_policy_coverage"][
                    "total_quads"
                ],
            )
        )
        validity_gates = {
            "development_joint_policy_coverage_complete": development_repair_coverage,
            "heldout_joint_policy_coverage_complete": heldout_repair_coverage,
            "development_repaired_population_has_no_corrupt_cells": (
                development_hardware["repair"]["final_corrupt_count"] == 0
            ),
            "heldout_repaired_population_has_no_corrupt_cells": (
                heldout_hardware["repair"]["final_corrupt_count"] == 0
            ),
            "development_full_conductance_invariants": bool(
                development_invariants["valid"]
            ),
            "heldout_full_conductance_invariants": bool(
                heldout_invariants["valid"]
            ),
            "development_candidate_count_is_16": len(candidates) == 16,
            "standard4delta_uses_continuous_selected_calibration": (
                protocol.mapping.standard4delta_refit is False
            ),
            "expected_evaluation_examples": expected_examples_passed,
            "optimizer_updates_are_zero": protocol.exclusions.optimizer_updates == 0,
            "target_write_noise_is_zero": protocol.exclusions.target_write_noise == 0.0,
            "inference_read_noise_is_zero": protocol.exclusions.inference_read_noise
            == 0.0,
        }
        coverage_valid = all(validity_gates.values())
        if not coverage_valid:
            failed_gates = sorted(
                name for name, passed in validity_gates.items() if not passed
            )
            raise RuntimeError(
                "Refusing to complete an invalid ideal-mapped initialization "
                f"bundle; failed gates={failed_gates!r}."
            )
        scientific_summary = {
            "schema": SUMMARY_SCHEMA,
            "schema_version": SUMMARY_SCHEMA_VERSION,
            "status": "complete",
            "claim_boundary": (
                "Model-based AIHWKit 1.1.0 OM preset, four-device ideal bounded "
                "initialization only. Commissioning RESET/read noise is frozen; "
                "optimizer updates, HWA, P&V, target-write noise, inference-read "
                "noise, retention, drift, and fabricated-device claims are absent."
            ),
            "metric_definition_ids": {
                "primary": protocol.metrics.primary,
                "diagnostic": protocol.metrics.diagnostic,
            },
            "policy": protocol.baseline.selected_policy,
            "source": source_report,
            "data": data_provenance,
            "contract": to_plain_data(protocol),
            "development": {
                "hardware": {
                    key: value
                    for key, value in development_hardware.items()
                    if key
                    not in {
                        "cell_lower_units",
                        "cell_upper_units",
                        "reset_baseline_units",
                        "intrinsic_references_native",
                    }
                },
                "calibration": calibration_receipt,
                "mapping": development_mapping.report,
                "invariants": development_invariants,
            },
            "heldout": {
                "assignment_seed": heldout_seed,
                "hardware": {
                    key: value
                    for key, value in heldout_hardware.items()
                    if key
                    not in {
                        "cell_lower_units",
                        "cell_upper_units",
                        "reset_baseline_units",
                        "intrinsic_references_native",
                    }
                },
                "scale_fractions": list(selected_pair),
                "fixed_logit_gain": selected_gain,
                "mapping": heldout_mapping.report,
                "invariants": heldout_invariants,
                "ideal_bounded_continuous_init": continuous_metrics,
                "ideal_bounded_standard4delta_init": standard_metrics,
                "baseline_only_full_circuit": baseline_metrics,
                "baseline_rail_voltage_residual": baseline_rail_residual,
                "weight_errors": weight_error_report,
                "predictions": prediction_artifact,
            },
            "validity": {
                "coverage_valid": coverage_valid,
                "gates": validity_gates,
                "expected_examples": expected_examples,
                "observed_examples": observed_example_counts,
                "optimizer_updates": 0,
                "program_verify_enabled": False,
                "hardware_aware_training_enabled": False,
                "target_write_noise": 0.0,
                "inference_read_noise": 0.0,
                "retention_or_drift": False,
                "noise_partition": {
                    "commissioning_reset_pulse_and_apparent_read_noise": "enabled_and_frozen",
                    "mapped_target_write_noise": "disabled",
                    "inference_read_noise": "disabled",
                },
            },
        }
        summary_path = artifact_root / "scientific_summary.json"
        atomic_write_json(summary_path, scientific_summary)
        artifacts.append({"path": str(summary_path), "kind": "scientific_summary"})
        terminal_metrics = {
            "metric_definition_ids": scientific_summary["metric_definition_ids"],
            "policy": protocol.baseline.selected_policy,
            "heldout_assignment_seed": heldout_seed,
            "development_hardware_instance_id": development_hardware[
                "hardware_instance_id"
            ],
            "heldout_hardware_instance_id": heldout_hardware["hardware_instance_id"],
            "selected_scale_fractions": list(selected_pair),
            "fixed_logit_gain": selected_gain,
            "ideal_bounded_continuous_init": {
                key: continuous_metrics[key]
                for key in (
                    "student_correct",
                    "examples",
                    "student_accuracy",
                    "teacher_correct",
                    "teacher_accuracy",
                    "teacher_agreement_count",
                    "teacher_agreement",
                    "prediction_sha256",
                )
            },
            "ideal_bounded_standard4delta_init": {
                key: standard_metrics[key]
                for key in (
                    "student_correct",
                    "examples",
                    "student_accuracy",
                    "teacher_correct",
                    "teacher_accuracy",
                    "teacher_agreement_count",
                    "teacher_agreement",
                    "prediction_sha256",
                )
            },
            "coverage_valid": coverage_valid,
            "validity_gates": validity_gates,
            "expected_examples": expected_examples,
            "optimizer_updates": 0,
        }
        print(
            f"heldout {heldout_seed} {protocol.baseline.selected_policy}: "
            f"continuous={100.0 * continuous_metrics['student_accuracy']:.2f}% "
            f"({continuous_metrics['student_correct']}/{continuous_metrics['examples']}), "
            f"4delta={100.0 * standard_metrics['student_accuracy']:.2f}%",
            flush=True,
        )
        store.append_metric({"mode": "validate", **terminal_metrics})
        store.complete(
            metrics=terminal_metrics,
            artifacts=_artifact_records(store, artifacts),
        )
        return 0
    except BaseException as error:
        store.fail(error)
        raise


__all__ = [
    "BASELINE_RESIDUAL_SCHEMA",
    "BASELINE_RESIDUAL_SCHEMA_VERSION",
    "SUMMARY_SCHEMA",
    "SUMMARY_SCHEMA_VERSION",
    "run_validate",
]
