"""Native ``ebl validate`` runtime for four-device reference balancing."""

from __future__ import annotations

from dataclasses import replace
from itertools import product
import gc
import json
import math
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable, Mapping, Sequence, TYPE_CHECKING

import torch

from experiments.artifacts import RunStore, atomic_write_json, sha256_file
from experiments.mnist_relu_drn.components import (
    apply_targets,
    build_student_stack,
)
from experiments.mnist_relu_drn.ibm_om_bounded_codebook_scheme_screen import (
    _evaluate_detailed,
    _float_summary,
    _population_for,
    _tensor_sha256,
)
from experiments.mnist_relu_drn.ibm_om_four_reference_balance import (
    BALANCED_POLICY,
    POLICIES,
    RANDOM_POLICY,
    build_continuous_reference_targets,
    build_reference_binding_plan,
    save_continuous_mapping,
    save_reference_binding_plan,
)
from experiments.mnist_relu_drn.ibm_om_ideal_mapping_scheme_screen import (
    _normalized,
    _ordered_labels,
    _quad_contrast,
)
from experiments.mnist_relu_drn.ibm_om_standard_level_scheme_screen import (
    _fit_scheme_calibration,
)
from experiments.mnist_relu_drn.runtime import _load_teacher
from experiments.mnist_shared import build_mnist_loaders
from experiments.schema import to_plain_data


if TYPE_CHECKING:
    from ebl.cli import ValidateRequest


_ROOT = Path(__file__).resolve().parents[2]
SUMMARY_SCHEMA = "ebl.mnist_relu_drn.ibm_om_four_reference_balance_result"
SUMMARY_SCHEMA_VERSION = 1


def _input(role: str, path: Path) -> dict[str, Any]:
    source = path.expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Expected {role} to name an existing file: {source}.")
    return {"role": role, "path": str(source), "sha256": sha256_file(source)}


def _sampler_python() -> Path:
    raw = os.environ.get("EBL_AIHWKIT_PYTHON")
    if not raw:
        raise ValueError(
            "Expected EBL_AIHWKIT_PYTHON to name the pinned AIHWKit 1.1.0 interpreter."
        )
    path = Path(raw).expanduser().resolve()
    if not path.is_file() or not os.access(path, os.X_OK):
        raise ValueError(f"Expected an executable EBL_AIHWKIT_PYTHON: {path}.")
    return path


def _source_logical_weights(
    weights_path: Path,
    *,
    student_spec,
    teacher,
) -> tuple[tuple[torch.Tensor, torch.Tensor], Mapping[str, Any]]:
    try:
        payload = torch.load(weights_path, map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, TypeError) as error:
        raise ValueError("Expected a readable safe named logical checkpoint.") from error
    if not isinstance(payload, Mapping) or set(payload) != {
        "schema",
        "schema_version",
        "catalog",
        "weights",
        "metadata",
    }:
        raise ValueError("Expected the exact named-weight checkpoint fields.")
    if payload["schema"] != "drn.named-weights" or payload["schema_version"] != 1:
        raise ValueError("Expected named-weight checkpoint schema version 1.")
    expected = (
        ("base.dense_weight.0", (1568, 100), "torch.float32"),
        ("base.dense_weight.1", (100, 20), "torch.float32"),
    )
    catalog = payload["catalog"]
    weights = payload["weights"]
    if (
        not isinstance(catalog, list)
        or len(catalog) != len(expected)
        or not isinstance(weights, Mapping)
    ):
        raise ValueError("Expected named checkpoint catalog and weight mapping.")
    if tuple(weights) != tuple(item[0] for item in expected):
        raise ValueError("Expected canonical source checkpoint weight order.")
    matrices = []
    for index, (key, shape, dtype_name) in enumerate(expected):
        descriptor = catalog[index] if index < len(catalog) else None
        if not isinstance(descriptor, Mapping) or (
            descriptor.get("key") != key
            or tuple(descriptor.get("shape", ())) != shape
            or descriptor.get("dtype") != dtype_name
        ):
            raise ValueError("Expected canonical source checkpoint catalog descriptors.")
        value = weights[key]
        if (
            not isinstance(value, torch.Tensor)
            or tuple(value.shape) != shape
            or value.dtype != torch.float32
            or not bool(torch.isfinite(value).all())
        ):
            raise ValueError("Expected finite float32 source checkpoint tensors.")
        matrices.append(value.detach().cpu())
    logical = tuple(
        _quad_contrast(matrix, layout=layout) / 2.0
        for matrix, layout in zip(matrices, ("halves", "paired"))
    )
    teacher_weights = tuple(value.detach().cpu() for value in teacher.parameters())
    if len(teacher_weights) != 2:
        raise ValueError("Expected the bias-free ReLU teacher to expose two weights.")
    deviations = []
    for source, expected in zip(logical, teacher_weights):
        if source.shape != expected.shape:
            raise ValueError("Expected source checkpoint and teacher logical shapes to match.")
        deviations.append(
            float((_normalized(source) - _normalized(expected)).abs().max().item())
        )
    if max(deviations) > 2e-6:
        raise ValueError(
            "Expected the explicit pre-BPTT checkpoint to encode the named teacher weights. "
            f"Maximum normalized deviation: {max(deviations):.12g}."
        )
    report = {
        "checkpoint_sha256": sha256_file(weights_path),
        "checkpoint_metadata": payload["metadata"],
        "logical_weight_hashes": [_tensor_sha256(value) for value in logical],
        "normalized_teacher_max_abs_deviation": deviations,
        "extraction": "half_four_rail_contrast_then_layerwise_absmax_normalization",
    }
    return (logical[0], logical[1]), report


def _selected_candidate(candidates: Sequence[Mapping[str, Any]]) -> int:
    if not candidates:
        raise RuntimeError("Expected at least one development calibration candidate.")
    return min(
        range(len(candidates)),
        key=lambda index: (
            -float(candidates[index]["calibration"]["student_accuracy"]),
            float(candidates[index]["calibration"]["calibrated_kl"]),
            tuple(float(value) for value in candidates[index]["scale_fractions"]),
        ),
    )


def _accuracy_summary(values: Sequence[float]) -> Mapping[str, float]:
    if not values or not all(math.isfinite(value) for value in values):
        raise ValueError("Expected at least one finite accuracy.")
    tensor = torch.tensor(tuple(values), dtype=torch.float64)
    return {
        "mean": float(tensor.mean().item()),
        "minimum": float(tensor.min().item()),
        "maximum": float(tensor.max().item()),
        "sample_standard_deviation": (
            float(tensor.std(unbiased=True).item()) if tensor.numel() > 1 else 0.0
        ),
    }


def _artifact_records(store: RunStore, paths: Iterable[Mapping[str, Any]]):
    records = []
    for item in paths:
        records.append(store.artifact_record(Path(item["path"]), kind=str(item["kind"])))
        if item.get("receipt") is not None:
            records.append(
                store.artifact_record(Path(item["receipt"]), kind=f"{item['kind']}_receipt")
            )
    return tuple(records)


def _compact_mapping(mapping: Mapping[str, Any]) -> Mapping[str, Any]:
    return {
        "policy": mapping["policy"],
        "assignment_seed": mapping["assignment_seed"],
        "scale_fractions": mapping["scale_fractions"],
        "target_hashes": mapping["target_hashes"],
        "layers": [
            {
                "layer": layer["layer"],
                "baseline_contrast": layer["baseline_contrast"],
                "realized_contrast": layer["realized_contrast"],
                "quad_loading": layer["quad_loading"],
                "common_positive_headroom": layer["common_positive_headroom"],
                "zero_positive_headroom_count": layer["zero_positive_headroom_count"],
                "reference_outside_active_bounds_count": layer[
                    "reference_outside_active_bounds_count"
                ],
                "continuous_logical_sign_flip_count": layer[
                    "continuous_logical_sign_flip_count"
                ],
                "continuous_logical_sign_flip_fraction_nonzero": layer[
                    "continuous_logical_sign_flip_fraction_nonzero"
                ],
                "full_g_decomposition_max_abs_residual": layer[
                    "full_g_decomposition_max_abs_residual"
                ],
                "active_bound_violation_count": layer[
                    "active_bound_violation_count"
                ],
                "float32_inward_baseline_adjustment_count": layer[
                    "float32_inward_baseline_adjustment_count"
                ],
                "float32_inward_conductance_adjustment_count": layer[
                    "float32_inward_conductance_adjustment_count"
                ],
                "float32_inward_conductance_max_abs_adjustment": layer[
                    "float32_inward_conductance_max_abs_adjustment"
                ],
                "hashes": layer["hashes"],
            }
            for layer in mapping["layers"]
        ],
    }


def _evaluate_assignment(
    *,
    assignment_seed: int,
    population,
    plans,
    logical_weights,
    development,
    student_spec,
    teacher,
    test_loader,
    sample_limit,
    artifact_directory: Path,
) -> tuple[list[Mapping[str, Any]], list[Mapping[str, Any]], Mapping[str, Any]]:
    torch.manual_seed(student_spec.runtime.seed)
    stack = build_student_stack(student_spec, enable_measured=False)
    arm_reports: list[Mapping[str, Any]] = []
    artifacts: list[Mapping[str, Any]] = []
    predictions: dict[tuple[str, str], torch.Tensor] = {}
    for binding_policy in POLICIES:
        for calibration_policy in POLICIES:
            selected = development[calibration_policy]["selected"]
            pair = tuple(float(value) for value in selected["scale_fractions"])
            gain = float(selected["calibration"]["gain"])
            targets, mapping, components = build_continuous_reference_targets(
                logical_weights,
                population,
                plans[binding_policy],
                scale_fractions=(pair[0], pair[1]),
                conductance_min=student_spec.model.conductance_min,
                conductance_max=student_spec.model.conductance_max,
            )
            apply_targets(stack.bundle.catalog, targets)
            stack.cost.gain = gain
            metrics, prediction = _evaluate_detailed(
                stack,
                teacher,
                test_loader,
                sample_limit=sample_limit,
            )
            mapping_artifact = save_continuous_mapping(
                artifact_directory,
                assignment_seed=assignment_seed,
                binding_policy=binding_policy,
                calibration_policy=calibration_policy,
                mapping_report=mapping,
                components=components,
            )
            artifacts.append({**mapping_artifact, "kind": "ibm_om_full_g_mapping"})
            predictions[(binding_policy, calibration_policy)] = prediction
            arm_reports.append(
                {
                    "assignment_seed": assignment_seed,
                    "binding_policy": binding_policy,
                    "calibration_policy": calibration_policy,
                    "scale_fractions": list(pair),
                    "fixed_logit_gain": gain,
                    "calibration_source": "frozen_development_assignment_86001",
                    "mapping": _compact_mapping(mapping),
                    "mapping_artifact": mapping_artifact,
                    "ideal_bounded_continuous_test": metrics,
                }
            )
    paired = {}
    for calibration_policy in POLICIES:
        random_prediction = predictions[(RANDOM_POLICY, calibration_policy)]
        balanced_prediction = predictions[(BALANCED_POLICY, calibration_policy)]
        random_arm = next(
            item
            for item in arm_reports
            if item["binding_policy"] == RANDOM_POLICY
            and item["calibration_policy"] == calibration_policy
        )
        balanced_arm = next(
            item
            for item in arm_reports
            if item["binding_policy"] == BALANCED_POLICY
            and item["calibration_policy"] == calibration_policy
        )
        random_accuracy = float(
            random_arm["ideal_bounded_continuous_test"]["student_accuracy"]
        )
        balanced_accuracy = float(
            balanced_arm["ideal_bounded_continuous_test"]["student_accuracy"]
        )
        paired[calibration_policy] = {
            "balanced_minus_random_accuracy": balanced_accuracy - random_accuracy,
            "prediction_flip_count": int(
                (random_prediction != balanced_prediction).sum().item()
            ),
            "prediction_flip_fraction": float(
                (random_prediction != balanced_prediction).to(torch.float64).mean().item()
            ),
        }
    return arm_reports, artifacts, paired


def run_validate(request: "ValidateRequest") -> int:
    # Imported lazily so lightweight ebl discovery does not import torch-heavy
    # config/runtime modules unnecessarily.
    from experiments.mnist_relu_drn.ibm_om_four_reference_balance_config import (
        BalanceValidateSpec,
    )

    spec = request.spec
    if not isinstance(spec, BalanceValidateSpec):
        raise TypeError(
            "Expected ibm_om_four_reference_balance.v1 validate to resolve its "
            f"dedicated spec. Provided value: {type(spec).__name__}."
        )
    if request.teacher_weights is None:
        raise ValueError("Expected --teacher-weights for reference-balance validation.")
    if request.device_model is not None:
        raise ValueError("Expected no --device-model for the ideal reference-balance study.")
    sampler = _sampler_python()
    contract = spec.protocol
    weights_path = request.weights.expanduser().resolve()
    teacher_path = request.teacher_weights.expanduser().resolve()
    if sha256_file(weights_path) != contract.source.expected_weights_sha256:
        raise ValueError("Source checkpoint SHA-256 does not match the frozen contract.")
    if sha256_file(teacher_path) != contract.source.expected_teacher_sha256:
        raise ValueError("Teacher checkpoint SHA-256 does not match the frozen contract.")
    store = RunStore.create(
        output_root=request.output_dir,
        experiment_id=spec.experiment_id,
        resolved_config=to_plain_data(spec),
        command=request.command,
        repo_root=_ROOT,
        input_artifacts=(
            _input("weights", weights_path),
            _input("teacher_weights", teacher_path),
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
        loaders = build_mnist_loaders(
            student_spec.data,
            data_seed=student_spec.runtime.data_seed,
            calibration_examples=student_spec.mapping.calibration_examples,
            calibration_batch_size=student_spec.mapping.calibration_batch_size,
        )
        teacher, teacher_metadata = _load_teacher(
            teacher_path, device=device, spec=student_spec
        )
        logical_weights, source_report = _source_logical_weights(
            weights_path,
            student_spec=student_spec,
            teacher=teacher,
        )
        calibration_labels = _ordered_labels(loaders.calibration)
        fraction_pairs = tuple(
            product(contract.continuous_mapping.scale_fractions, repeat=2)
        )
        artifacts: list[Mapping[str, Any]] = []
        population_contract = SimpleNamespace(
            corruption_policy=contract.device.corruption_policy,
            required_aihwkit_version=contract.device.required_aihwkit_version,
        )

        torch.manual_seed(student_spec.runtime.seed)
        dev_stack = build_student_stack(student_spec, enable_measured=False)
        development_population, development_population_report = _population_for(
            stack=dev_stack,
            topology=4,
            assignment_seed=contract.assignments.development_seed,
            contract=population_contract,
            aihwkit_python=sampler,
            output_dir=store.run_dir / "artifacts",
        )
        artifacts.extend(
            (
                {
                    "path": development_population_report["path"],
                    "receipt": development_population_report["receipt"],
                    "kind": "ibm_om_population",
                },
            )
        )
        development_plans = {}
        development: dict[str, Any] = {}
        for policy in POLICIES:
            plan = build_reference_binding_plan(development_population, policy=policy)
            development_plans[policy] = plan
            plan_artifact = save_reference_binding_plan(
                store.run_dir / "artifacts" / "binding_plans",
                development_population,
                plan,
            )
            artifacts.append({**plan_artifact, "kind": "ibm_om_reference_binding_plan"})
            candidates = []
            retained = []
            for pair in fraction_pairs:
                targets, mapping, components = build_continuous_reference_targets(
                    logical_weights,
                    development_population,
                    plan,
                    scale_fractions=(float(pair[0]), float(pair[1])),
                    conductance_min=student_spec.model.conductance_min,
                    conductance_max=student_spec.model.conductance_max,
                )
                apply_targets(dev_stack.bundle.catalog, targets)
                calibration = _fit_scheme_calibration(
                    dev_stack,
                    teacher,
                    loaders.calibration,
                    calibration_labels,
                    gain_min=student_spec.mapping.logit_gain_min,
                    gain_max=student_spec.mapping.logit_gain_max,
                    gain_steps=student_spec.mapping.logit_gain_steps,
                )
                calibration["gain_source"] = (
                    "per_binding_continuous_development_fit"
                )
                candidates.append(
                    {
                        "scale_fractions": [float(pair[0]), float(pair[1])],
                        "calibration": calibration,
                        "target_hashes": mapping["target_hashes"],
                        "baseline_contrast_rms": [
                            layer["baseline_contrast"]["rms"]
                            for layer in mapping["layers"]
                        ],
                    }
                )
                retained.append((targets, mapping, components))
            selected_index = _selected_candidate(candidates)
            selected = candidates[selected_index]
            selected_targets, selected_mapping, selected_components = retained[
                selected_index
            ]
            mapping_artifact = save_continuous_mapping(
                store.run_dir / "artifacts" / "mappings",
                assignment_seed=contract.assignments.development_seed,
                binding_policy=policy,
                calibration_policy=policy,
                mapping_report=selected_mapping,
                components=selected_components,
            )
            artifacts.append({**mapping_artifact, "kind": "ibm_om_full_g_mapping"})
            development[policy] = {
                "binding_policy": policy,
                "selection_domain": "development_assignment_86001_calibration_subset",
                "selection_metric": "mapped_accuracy_then_calibrated_kl_then_scale_pair",
                "selected_index": selected_index,
                "selected": selected,
                "selected_mapping": _compact_mapping(selected_mapping),
                "selected_mapping_artifact": mapping_artifact,
                "binding_plan": plan.report,
                "binding_plan_artifact": plan_artifact,
                "candidates": candidates,
            }
            print(
                f"calibrated {policy}: scales={selected['scale_fractions']}, "
                f"accuracy={100.0 * selected['calibration']['student_accuracy']:.2f}%",
                flush=True,
            )

        heldout = []
        for assignment_seed in contract.assignments.heldout_seeds:
            torch.manual_seed(student_spec.runtime.seed)
            stack = build_student_stack(student_spec, enable_measured=False)
            population, population_report = _population_for(
                stack=stack,
                topology=4,
                assignment_seed=assignment_seed,
                contract=population_contract,
                aihwkit_python=sampler,
                output_dir=store.run_dir / "artifacts",
            )
            artifacts.append(
                {
                    "path": population_report["path"],
                    "receipt": population_report["receipt"],
                    "kind": "ibm_om_population",
                }
            )
            plans = {}
            plan_reports = {}
            for policy in POLICIES:
                plan = build_reference_binding_plan(population, policy=policy)
                plans[policy] = plan
                plan_artifact = save_reference_binding_plan(
                    store.run_dir / "artifacts" / "binding_plans", population, plan
                )
                artifacts.append({**plan_artifact, "kind": "ibm_om_reference_binding_plan"})
                plan_reports[policy] = {
                    "report": plan.report,
                    "artifact": plan_artifact,
                }
            arm_reports, assignment_artifacts, paired = _evaluate_assignment(
                assignment_seed=assignment_seed,
                population=population,
                plans=plans,
                logical_weights=logical_weights,
                development=development,
                student_spec=student_spec,
                teacher=teacher,
                test_loader=loaders.test,
                sample_limit=student_spec.settings.sample_limit,
                artifact_directory=store.run_dir / "artifacts" / "mappings",
            )
            artifacts.extend(assignment_artifacts)
            heldout.append(
                {
                    "assignment_seed": assignment_seed,
                    "population": population_report,
                    "binding_plans": plan_reports,
                    "arms": arm_reports,
                    "paired_binding_effect": paired,
                }
            )
            optimized = {
                arm["binding_policy"]: arm
                for arm in arm_reports
                if arm["binding_policy"] == arm["calibration_policy"]
            }
            print(
                f"heldout {assignment_seed}: random="
                f"{100.0 * optimized[RANDOM_POLICY]['ideal_bounded_continuous_test']['student_accuracy']:.2f}%, "
                f"balanced="
                f"{100.0 * optimized[BALANCED_POLICY]['ideal_bounded_continuous_test']['student_accuracy']:.2f}%",
                flush=True,
            )
            del population, stack
            gc.collect()

        aggregate: dict[str, Any] = {"scheme_optimized": {}, "fixed_calibration": {}}
        for policy in POLICIES:
            values = [
                float(
                    next(
                        arm
                        for arm in block["arms"]
                        if arm["binding_policy"] == policy
                        and arm["calibration_policy"] == policy
                    )["ideal_bounded_continuous_test"]["student_accuracy"]
                )
                for block in heldout
            ]
            aggregate["scheme_optimized"][policy] = _accuracy_summary(values)
        for calibration_policy in POLICIES:
            deltas = [
                float(
                    block["paired_binding_effect"][calibration_policy][
                        "balanced_minus_random_accuracy"
                    ]
                )
                for block in heldout
            ]
            aggregate["fixed_calibration"][calibration_policy] = {
                "balanced_minus_random_accuracy": _accuracy_summary(deltas),
                "per_assignment": deltas,
            }
        balanced_mean = float(
            aggregate["scheme_optimized"][BALANCED_POLICY]["mean"]
        )
        aggregate["balanced_accuracy_gate"] = {
            "threshold": 0.90,
            "observed_mean": balanced_mean,
            "passed": balanced_mean >= 0.90,
        }
        scientific_summary = {
            "schema": SUMMARY_SCHEMA,
            "schema_version": SUMMARY_SCHEMA_VERSION,
            "status": "complete",
            "claim_boundary": (
                "Model-based AIHWKit 1.1.0 OM preset; ideal continuous bounded "
                "initialization only. No quantization, pulse programming, HWA, "
                "optimizer update, inference read noise, or fabricated-device claim."
            ),
            "metric_definition_id": contract.metric_definition,
            "source": source_report,
            "teacher": {
                "sha256": sha256_file(teacher_path),
                "metadata": teacher_metadata,
            },
            "contract": to_plain_data(contract),
            "development_population": development_population_report,
            "development": development,
            "heldout": heldout,
            "aggregate": aggregate,
            "exclusions": {
                "optimizer_updates": 0,
                "quantization_enabled": False,
                "program_verify_enabled": False,
                "hwa_enabled": False,
                "deployment_write_noise": 0.0,
                "inference_read_noise": 0.0,
                "retention_or_drift": False,
            },
        }
        summary_path = store.run_dir / "artifacts" / "scientific_summary.json"
        atomic_write_json(summary_path, scientific_summary)
        artifacts.append({"path": str(summary_path), "kind": "scientific_summary"})
        compact_heldout = [
            {
                "assignment_seed": block["assignment_seed"],
                "scheme_optimized": {
                    policy: next(
                        arm["ideal_bounded_continuous_test"]
                        for arm in block["arms"]
                        if arm["binding_policy"] == policy
                        and arm["calibration_policy"] == policy
                    )
                    for policy in POLICIES
                },
                "paired_binding_effect": block["paired_binding_effect"],
            }
            for block in heldout
        ]
        store.append_metric(
            {
                "mode": "validate",
                "metric_definition_id": contract.metric_definition,
                "aggregate": aggregate,
            }
        )
        store.complete(
            metrics={
                "metric_definition_id": contract.metric_definition,
                "evidence_class": contract.device.evidence_class,
                "initialization": "pre_bptt_continuous_bounded_fixed_r",
                "heldout": compact_heldout,
                "aggregate": aggregate,
                "coverage_valid": True,
                "optimizer_updates": 0,
            },
            artifacts=_artifact_records(store, artifacts),
        )
        return 0
    except BaseException as error:
        store.fail(error)
        raise


__all__ = ["run_validate"]
