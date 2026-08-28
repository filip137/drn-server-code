"""Native ``ebl validate`` runtime for local IBM OM baseline compensation.

The runtime keeps the sampled identity at every physical address.  It compares
the clipped intrinsic symmetry baseline with the nearest in-bounds baseline
whose four-cell zero contrast is exactly zero.  Both arms use a common
headroom envelope, so a fixed scale pair produces the exact same ``d`` tensor;
only ``B`` changes.  The complete ``G = B + d`` tensors are applied to the
resistive solver.
"""

from __future__ import annotations

from itertools import product
import gc
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, TYPE_CHECKING

import torch

from experiments.artifacts import RunStore, atomic_write_json, sha256_file
from experiments.mnist_relu_drn.components import apply_targets, build_student_stack
from experiments.mnist_relu_drn.ibm_om_bounded_codebook_scheme_screen import (
    _evaluate_detailed,
    _population_for,
)
from experiments.mnist_relu_drn.ibm_om_four_reference_balance_runtime import (
    _accuracy_summary,
    _artifact_records,
    _input,
    _sampler_python,
    _selected_candidate,
    _source_logical_weights,
)
from experiments.mnist_relu_drn.ibm_om_local_reference_compensation import (
    CONTROL_POLICY,
    POLICIES,
    TREATMENT_POLICY,
    build_matched_continuous_targets,
    build_local_baseline_plan,
    save_continuous_mapping,
    save_local_baseline_plan,
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
SUMMARY_SCHEMA = "ebl.mnist_relu_drn.ibm_om_local_reference_compensation_result"
SUMMARY_SCHEMA_VERSION = 1


def _offset_hashes(mapping: Mapping[str, Any]) -> tuple[str, ...]:
    """Return the canonical per-layer ``d`` hashes from a mapping report."""

    try:
        hashes = tuple(str(layer["hashes"]["offset"]) for layer in mapping["layers"])
    except (KeyError, TypeError) as error:
        raise ValueError("Expected every mapping layer to expose an offset hash.") from error
    if len(hashes) != 2 or any(len(value) != 64 for value in hashes):
        raise ValueError("Expected exactly two SHA-256 offset hashes.")
    return hashes


def _assert_matched_offsets(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
    *,
    context: str,
) -> tuple[str, ...]:
    """Fail closed unless two policy arms use bit-identical offset tensors."""

    left_hashes = _offset_hashes(left)
    right_hashes = _offset_hashes(right)
    if left_hashes != right_hashes:
        raise RuntimeError(f"Matched-offset invariant failed for {context}.")
    return left_hashes


def _compact_mapping(mapping: Mapping[str, Any]) -> Mapping[str, Any]:
    """Keep the scientific invariants needed by result and study analyzers."""

    baseline_policy = mapping["baseline_policy"]
    if baseline_policy not in POLICIES:
        raise ValueError("Expected a canonical baseline policy in the mapping report.")
    compact_layers = []
    for layer in mapping["layers"]:
        retained = {
            name: layer[name]
            for name in (
                "layer",
                "baseline_contrast",
                "offset_contrast",
                "realized_contrast",
                "quad_loading",
                "baseline_loading",
                "common_positive_headroom",
                "zero_positive_headroom_count",
                "continuous_logical_sign_flip_count",
                "continuous_logical_sign_flip_fraction_nonzero",
                "logical_sign_flip_count",
                "logical_sign_flip_fraction_nonzero",
                "full_g_decomposition_max_abs_residual",
                "active_bound_violation_count",
                "bound_violation_count",
                "hashes",
            )
            if name in layer
        }
        # Baseline-plan diagnostics are deliberately copied when present.  The
        # core mapper owns their exact naming and the artifact retains the full
        # report; this compaction remains forwards-compatible with added fields.
        for name in (
            "exact_zero_residual",
            "baseline_correction",
            "bound_active_quad_count",
            "analytic_checkerboard_quad_count",
            "row_loading_change",
            "column_loading_change",
            "total_loading_change",
        ):
            if name in layer:
                retained[name] = layer[name]
        for canonical, source in (
            ("continuous_logical_sign_flip_count", "logical_sign_flip_count"),
            (
                "continuous_logical_sign_flip_fraction_nonzero",
                "logical_sign_flip_fraction_nonzero",
            ),
            ("active_bound_violation_count", "bound_violation_count"),
        ):
            if canonical not in retained and source in layer:
                retained[canonical] = layer[source]
        compact_layers.append(retained)
    return {
        "baseline_policy": baseline_policy,
        "assignment_seed": mapping["assignment_seed"],
        "scale_fractions": mapping["scale_fractions"],
        "target_hashes": mapping["target_hashes"],
        "offset_hashes": list(_offset_hashes(mapping)),
        "identity_binding": mapping["identity_binding"],
        "shared_matching": mapping["shared_matching"],
        "layers": compact_layers,
    }


def _paired_effect(
    control_metrics: Mapping[str, Any],
    treatment_metrics: Mapping[str, Any],
    control_prediction: torch.Tensor,
    treatment_prediction: torch.Tensor,
) -> Mapping[str, Any]:
    """Build an exact-count paired effect for one shared calibration."""

    control_examples = int(control_metrics["examples"])
    treatment_examples = int(treatment_metrics["examples"])
    if control_examples <= 0 or treatment_examples != control_examples:
        raise ValueError("Expected paired arms to evaluate the same nonempty cohort.")
    control_correct = int(control_metrics["student_correct"])
    treatment_correct = int(treatment_metrics["student_correct"])
    if control_prediction.shape != treatment_prediction.shape:
        raise ValueError("Expected paired prediction tensors to have identical shape.")
    if int(control_prediction.numel()) != control_examples:
        raise ValueError("Expected one paired prediction per evaluated example.")
    correct_delta = treatment_correct - control_correct
    accuracy_delta = correct_delta / control_examples
    reported_delta = float(treatment_metrics["student_accuracy"]) - float(
        control_metrics["student_accuracy"]
    )
    if not math.isclose(accuracy_delta, reported_delta, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("Exact correct-count delta disagrees with reported accuracies.")
    flips = control_prediction != treatment_prediction
    return {
        "treatment_minus_control_correct": correct_delta,
        "examples": control_examples,
        "treatment_minus_control_accuracy": accuracy_delta,
        "prediction_flip_count": int(flips.sum().item()),
        "prediction_flip_fraction": float(flips.to(torch.float64).mean().item()),
    }


def _material_artifact_contract(
    *, development_assignments: int, heldout_assignments: int
) -> Mapping[str, int]:
    """Declare the exact scientific artifact cardinalities for one native run."""

    if development_assignments != 1 or heldout_assignments <= 0:
        raise ValueError("Expected one development assignment and held-out assignments.")
    assignment_count = development_assignments + heldout_assignments
    return {
        "population_artifacts": assignment_count,
        "baseline_plan_artifacts": assignment_count * len(POLICIES),
        "mapping_artifacts": len(POLICIES)
        + heldout_assignments * len(POLICIES) * len(POLICIES),
        "identity_reassignment_artifacts": 0,
        "scientific_summary_artifacts": 1,
    }


def _build_plans(
    population,
    *,
    artifact_directory: Path,
) -> tuple[Mapping[str, Any], list[Mapping[str, Any]], Mapping[str, Any]]:
    plans: dict[str, Any] = {}
    artifacts: list[Mapping[str, Any]] = []
    reports: dict[str, Any] = {}
    for policy in POLICIES:
        plan = build_local_baseline_plan(population, policy=policy)
        artifact = save_local_baseline_plan(artifact_directory, population, plan)
        plans[policy] = plan
        artifacts.append({**artifact, "kind": "ibm_om_local_baseline_plan"})
        reports[policy] = {"report": plan.report, "artifact": artifact}
    return plans, artifacts, reports


def _map(
    *,
    logical_weights,
    population,
    plans,
    baseline_policy: str,
    scale_fractions: tuple[float, float],
    student_spec,
):
    return build_matched_continuous_targets(
        logical_weights,
        population,
        plans,
        policy=baseline_policy,
        scale_fractions=scale_fractions,
        conductance_min=student_spec.model.conductance_min,
        conductance_max=student_spec.model.conductance_max,
    )


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
    mappings: dict[tuple[str, str], Mapping[str, Any]] = {}
    for baseline_policy in POLICIES:
        for calibration_policy in POLICIES:
            selected = development[calibration_policy]["selected"]
            pair = tuple(float(value) for value in selected["scale_fractions"])
            gain = float(selected["calibration"]["gain"])
            targets, mapping, components = _map(
                logical_weights=logical_weights,
                population=population,
                plans=plans,
                baseline_policy=baseline_policy,
                scale_fractions=(pair[0], pair[1]),
                student_spec=student_spec,
            )
            apply_targets(stack.bundle.catalog, targets)
            stack.cost.gain = gain
            metrics, prediction = _evaluate_detailed(
                stack, teacher, test_loader, sample_limit=sample_limit
            )
            mapping_artifact = save_continuous_mapping(
                artifact_directory,
                assignment_seed=assignment_seed,
                baseline_policy=baseline_policy,
                calibration_policy=calibration_policy,
                mapping_report=mapping,
                components=components,
            )
            artifacts.append({**mapping_artifact, "kind": "ibm_om_full_g_mapping"})
            predictions[(baseline_policy, calibration_policy)] = prediction
            mappings[(baseline_policy, calibration_policy)] = mapping
            arm_reports.append(
                {
                    "assignment_seed": assignment_seed,
                    "baseline_policy": baseline_policy,
                    "calibration_policy": calibration_policy,
                    "scale_fractions": list(pair),
                    "fixed_logit_gain": gain,
                    "calibration_source": "frozen_development_assignment_86001",
                    "mapping": _compact_mapping(mapping),
                    "mapping_artifact": mapping_artifact,
                    "ideal_bounded_continuous_test": metrics,
                }
            )
    paired: dict[str, Any] = {}
    for calibration_policy in POLICIES:
        control_mapping = mappings[(CONTROL_POLICY, calibration_policy)]
        treatment_mapping = mappings[(TREATMENT_POLICY, calibration_policy)]
        offset_hashes = _assert_matched_offsets(
            control_mapping,
            treatment_mapping,
            context=(
                f"assignment {assignment_seed}, calibration {calibration_policy}"
            ),
        )
        control_arm = next(
            item
            for item in arm_reports
            if item["baseline_policy"] == CONTROL_POLICY
            and item["calibration_policy"] == calibration_policy
        )
        treatment_arm = next(
            item
            for item in arm_reports
            if item["baseline_policy"] == TREATMENT_POLICY
            and item["calibration_policy"] == calibration_policy
        )
        paired[calibration_policy] = {
            **_paired_effect(
                control_arm["ideal_bounded_continuous_test"],
                treatment_arm["ideal_bounded_continuous_test"],
                predictions[(CONTROL_POLICY, calibration_policy)],
                predictions[(TREATMENT_POLICY, calibration_policy)],
            ),
            "identical_offset_hashes": list(offset_hashes),
        }
    return arm_reports, artifacts, paired


def run_validate(request: "ValidateRequest") -> int:
    """Execute the predeclared ideal-only local-baseline comparison."""

    from experiments.mnist_relu_drn.ibm_om_local_reference_compensation_config import (
        LocalCompensationValidateSpec,
    )

    spec = request.spec
    if not isinstance(spec, LocalCompensationValidateSpec):
        raise TypeError(
            "Expected ibm_om_local_reference_compensation.v1 validate to resolve "
            f"its dedicated spec. Provided value: {type(spec).__name__}."
        )
    if request.teacher_weights is None:
        raise ValueError("Expected --teacher-weights for local compensation validation.")
    if request.device_model is not None:
        raise ValueError("Expected no --device-model for the ideal compensation study.")
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
            weights_path, student_spec=student_spec, teacher=teacher
        )
        from experiments.mnist_relu_drn.ibm_om_ideal_mapping_scheme_screen import (
            _ordered_labels,
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
        artifacts.append(
            {
                "path": development_population_report["path"],
                "receipt": development_population_report["receipt"],
                "kind": "ibm_om_population",
            }
        )
        development_plans, plan_artifacts, development_plan_reports = _build_plans(
            development_population,
            artifact_directory=store.run_dir / "artifacts" / "baseline_plans",
        )
        artifacts.extend(plan_artifacts)

        development: dict[str, Any] = {}
        candidate_offset_hashes: dict[str, list[tuple[str, ...]]] = {}
        for policy in POLICIES:
            candidates = []
            for pair in fraction_pairs:
                targets, mapping, _components = _map(
                    logical_weights=logical_weights,
                    population=development_population,
                    plans=development_plans,
                    baseline_policy=policy,
                    scale_fractions=(float(pair[0]), float(pair[1])),
                    student_spec=student_spec,
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
                calibration["gain_source"] = "per_baseline_continuous_development_fit"
                candidates.append(
                    {
                        "scale_fractions": [float(pair[0]), float(pair[1])],
                        "calibration": calibration,
                        "target_hashes": mapping["target_hashes"],
                        "offset_hashes": list(_offset_hashes(mapping)),
                        "baseline_contrast_rms": [
                            layer["baseline_contrast"]["rms"]
                            for layer in mapping["layers"]
                        ],
                    }
                )
            selected_index = _selected_candidate(candidates)
            selected = candidates[selected_index]
            selected_pair = tuple(float(value) for value in selected["scale_fractions"])
            _selected_targets, selected_mapping, selected_components = _map(
                logical_weights=logical_weights,
                population=development_population,
                plans=development_plans,
                baseline_policy=policy,
                scale_fractions=(selected_pair[0], selected_pair[1]),
                student_spec=student_spec,
            )
            if list(_offset_hashes(selected_mapping)) != selected["offset_hashes"]:
                raise RuntimeError("Selected development mapping did not replay exactly.")
            mapping_artifact = save_continuous_mapping(
                store.run_dir / "artifacts" / "mappings",
                assignment_seed=contract.assignments.development_seed,
                baseline_policy=policy,
                calibration_policy=policy,
                mapping_report=selected_mapping,
                components=selected_components,
            )
            artifacts.append({**mapping_artifact, "kind": "ibm_om_full_g_mapping"})
            development[policy] = {
                "baseline_policy": policy,
                "selection_domain": "development_assignment_86001_calibration_subset",
                "selection_metric": (
                    "mapped_accuracy_then_calibrated_kl_then_scale_pair"
                ),
                "selected_index": selected_index,
                "selected": selected,
                "selected_mapping": _compact_mapping(selected_mapping),
                "selected_mapping_artifact": mapping_artifact,
                "baseline_plan": development_plan_reports[policy],
                "candidates": candidates,
            }
            candidate_offset_hashes[policy] = [
                tuple(str(value) for value in candidate["offset_hashes"])
                for candidate in candidates
            ]
            print(
                f"calibrated {policy}: scales={selected['scale_fractions']}, "
                f"accuracy={100.0 * selected['calibration']['student_accuracy']:.2f}%",
                flush=True,
            )
        for pair_index, pair in enumerate(fraction_pairs):
            if (
                candidate_offset_hashes[CONTROL_POLICY][pair_index]
                != candidate_offset_hashes[TREATMENT_POLICY][pair_index]
            ):
                raise RuntimeError(
                    "Matched-offset invariant failed for development scale pair "
                    f"{tuple(pair)}."
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
            plans, assignment_plan_artifacts, plan_reports = _build_plans(
                population,
                artifact_directory=store.run_dir / "artifacts" / "baseline_plans",
            )
            artifacts.extend(assignment_plan_artifacts)
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
                    "baseline_plans": plan_reports,
                    "arms": arm_reports,
                    "paired_baseline_effect": paired,
                }
            )
            optimized = {
                arm["baseline_policy"]: arm
                for arm in arm_reports
                if arm["baseline_policy"] == arm["calibration_policy"]
            }
            control_accuracy = optimized[CONTROL_POLICY][
                "ideal_bounded_continuous_test"
            ]["student_accuracy"]
            treatment_accuracy = optimized[TREATMENT_POLICY][
                "ideal_bounded_continuous_test"
            ]["student_accuracy"]
            print(
                f"heldout {assignment_seed}: control="
                f"{100.0 * control_accuracy:.2f}%, "
                f"treatment={100.0 * treatment_accuracy:.2f}%",
                flush=True,
            )
            del population, stack
            gc.collect()

        aggregate: dict[str, Any] = {
            "scheme_optimized": {},
            "fixed_calibration": {},
        }
        for policy in POLICIES:
            values = [
                float(
                    next(
                        arm
                        for arm in block["arms"]
                        if arm["baseline_policy"] == policy
                        and arm["calibration_policy"] == policy
                    )["ideal_bounded_continuous_test"]["student_accuracy"]
                )
                for block in heldout
            ]
            aggregate["scheme_optimized"][policy] = _accuracy_summary(values)
        for calibration_policy in POLICIES:
            effects = [
                block["paired_baseline_effect"][calibration_policy]
                for block in heldout
            ]
            deltas = [float(item["treatment_minus_control_accuracy"]) for item in effects]
            aggregate["fixed_calibration"][calibration_policy] = {
                "treatment_minus_control_accuracy": _accuracy_summary(deltas),
                "per_assignment": effects,
            }
        primary_effects = [
            block["paired_baseline_effect"][CONTROL_POLICY] for block in heldout
        ]
        aggregate["primary_fixed_control_calibration"] = {
            "calibration_policy": CONTROL_POLICY,
            "treatment_minus_control_accuracy": _accuracy_summary(
                [float(item["treatment_minus_control_accuracy"]) for item in primary_effects]
            ),
            "total_treatment_minus_control_correct": sum(
                int(item["treatment_minus_control_correct"]) for item in primary_effects
            ),
            "total_examples": sum(int(item["examples"]) for item in primary_effects),
        }
        primary_mean = float(
            aggregate["primary_fixed_control_calibration"][
                "treatment_minus_control_accuracy"
            ]["mean"]
        )
        aggregate["primary_effect_gate"] = {
            "threshold": 0.05,
            "observed_mean": primary_mean,
            "passed": primary_mean >= 0.05,
        }
        optimized_control = aggregate["scheme_optimized"][CONTROL_POLICY]
        optimized_treatment = aggregate["scheme_optimized"][TREATMENT_POLICY]
        scheme_deltas = []
        scheme_correct_deltas = []
        scheme_examples = []
        for block in heldout:
            control = next(
                arm
                for arm in block["arms"]
                if arm["baseline_policy"] == CONTROL_POLICY
                and arm["calibration_policy"] == CONTROL_POLICY
            )["ideal_bounded_continuous_test"]
            treatment = next(
                arm
                for arm in block["arms"]
                if arm["baseline_policy"] == TREATMENT_POLICY
                and arm["calibration_policy"] == TREATMENT_POLICY
            )["ideal_bounded_continuous_test"]
            examples = int(control["examples"])
            if int(treatment["examples"]) != examples:
                raise RuntimeError("Scheme-optimized arms used different test cohorts.")
            correct_delta = int(treatment["student_correct"]) - int(
                control["student_correct"]
            )
            scheme_deltas.append(correct_delta / examples)
            scheme_correct_deltas.append(correct_delta)
            scheme_examples.append(examples)
        aggregate["secondary_scheme_optimized"] = {
            "control_accuracy": optimized_control,
            "treatment_accuracy": optimized_treatment,
            "treatment_minus_control_accuracy": _accuracy_summary(scheme_deltas),
            "total_treatment_minus_control_correct": sum(scheme_correct_deltas),
            "total_examples": sum(scheme_examples),
        }
        treatment_mean = float(optimized_treatment["mean"])
        aggregate["treatment_accuracy_gate"] = {
            "threshold": 0.90,
            "observed_mean": treatment_mean,
            "passed": treatment_mean >= 0.90,
        }
        artifact_contract = _material_artifact_contract(
            development_assignments=1,
            heldout_assignments=len(contract.assignments.heldout_seeds),
        )
        observed_contract = {
            "population_artifacts": sum(
                item["kind"] == "ibm_om_population" for item in artifacts
            ),
            "baseline_plan_artifacts": sum(
                item["kind"] == "ibm_om_local_baseline_plan" for item in artifacts
            ),
            "mapping_artifacts": sum(
                item["kind"] == "ibm_om_full_g_mapping" for item in artifacts
            ),
            "identity_reassignment_artifacts": 0,
            "scientific_summary_artifacts": 1,
        }
        if observed_contract != artifact_contract:
            raise RuntimeError(
                "Scientific artifact cardinality mismatch: "
                f"expected {artifact_contract!r}, observed {observed_contract!r}."
            )

        scientific_summary = {
            "schema": SUMMARY_SCHEMA,
            "schema_version": SUMMARY_SCHEMA_VERSION,
            "status": "complete",
            "claim_boundary": (
                "Model-based AIHWKit 1.1.0 OM preset; ideal continuous bounded "
                "initialization with fixed sampled identities only. No quantization, "
                "pulse programming, HWA, optimizer update, inference read noise, or "
                "fabricated-device claim."
            ),
            "metric_definition_id": contract.metric_definition,
            "source": source_report,
            "teacher": {
                "sha256": sha256_file(teacher_path),
                "metadata": teacher_metadata,
            },
            "contract": to_plain_data(contract),
            "artifact_contract": artifact_contract,
            "development_population": development_population_report,
            "development_baseline_plans": development_plan_reports,
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
                "identity_reassignment": False,
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
                        if arm["baseline_policy"] == policy
                        and arm["calibration_policy"] == policy
                    )
                    for policy in POLICIES
                },
                "paired_baseline_effect": block["paired_baseline_effect"],
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
                "initialization": "pre_bptt_continuous_bounded_local_baseline",
                "heldout": compact_heldout,
                "aggregate": aggregate,
                "artifact_contract": artifact_contract,
                "coverage_valid": True,
                "optimizer_updates": 0,
            },
            artifacts=_artifact_records(store, artifacts),
        )
        return 0
    except BaseException as error:
        store.fail(error)
        raise


__all__ = [
    "SUMMARY_SCHEMA",
    "SUMMARY_SCHEMA_VERSION",
    "run_validate",
]
