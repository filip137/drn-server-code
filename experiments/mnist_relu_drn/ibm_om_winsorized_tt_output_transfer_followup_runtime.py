"""CUDA runtime for the isolated IBM-OM TT output-transfer follow-up."""

from __future__ import annotations

import gc
import json
from pathlib import Path
from typing import Any, Mapping, TYPE_CHECKING

import torch

from experiments.artifacts import RunStore, atomic_write_json, sha256_file
from experiments.mnist_relu_drn.components import build_student_stack
from experiments.mnist_relu_drn.ibm_om_baseline_selection_runtime import (
    _artifact_records,
    _input,
)
from experiments.mnist_relu_drn.ibm_om_winsorized_endpoint_optimizer_recovery import (
    intrinsic_symmetry_raw_a,
)
from experiments.mnist_relu_drn.ibm_om_winsorized_endpoint_optimizer_recovery_config import (
    FIXED_LOGIT_GAIN,
)
from experiments.mnist_relu_drn.ibm_om_winsorized_endpoint_optimizer_recovery_runtime import (
    SUMMARY_SCHEMA as SOURCE_SUMMARY_SCHEMA,
    SUMMARY_SCHEMA_VERSION as SOURCE_SUMMARY_SCHEMA_VERSION,
    _evaluate_endpoint_p0,
    _expected_summary_path,
    _load_predecessor_bundle,
    _load_predecessor_summary,
    _predecessor_run_dir,
    _run_arm,
    _selection_key,
)
from experiments.mnist_relu_drn.ibm_om_winsorized_tt_output_transfer_followup_config import (
    TT_VARIANTS,
    TtOutputTransferFollowupTrainSpec,
)
from experiments.mnist_relu_drn.runtime import _load_teacher
from experiments.mnist_shared import build_mnist_loaders
from experiments.schema import to_plain_data


if TYPE_CHECKING:
    from ebl.cli import TrainRequest


_ROOT = Path(__file__).resolve().parents[2]
SUMMARY_SCHEMA = (
    "ebl.mnist_relu_drn.ibm_om_winsorized_tt_output_transfer_followup_exploratory"
)
SUMMARY_SCHEMA_VERSION = 1


def _source_run_dir(protocol: Any) -> Path:
    source = protocol.source_optimizer_recovery
    return (_ROOT / "results" / source.result_id / source.run_relative_path).resolve()


def _expected_source_summary_path(protocol: Any) -> Path:
    return (
        _source_run_dir(protocol)
        / protocol.source_optimizer_recovery.summary_relative_path
    ).resolve()


def _frozen_lambdas_by_variant(protocol: Any) -> dict[str, tuple[float, float]]:
    result = {
        str(name): tuple(float(value) for value in lambdas)
        for name, lambdas in (
            protocol.source_optimizer_recovery.selected_effective_lambdas
        )
    }
    if set(result) != set(TT_VARIANTS) or any(
        len(value) != 2 or any(item <= 0.0 for item in value)
        for value in result.values()
    ):
        raise ValueError("Frozen source TT lambdas are incomplete or invalid.")
    return result


def _load_source_optimizer_summary(
    path: Path, protocol: Any
) -> Mapping[str, Any]:
    expected = _expected_source_summary_path(protocol)
    if path.resolve() != expected:
        raise ValueError(
            "Expected --weights to name the exact endpoint-optimizer summary."
        )
    if not path.is_file() or sha256_file(path) != (
        protocol.source_optimizer_recovery.summary_sha256
    ):
        raise ValueError("Endpoint-optimizer summary SHA-256 mismatch.")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("Expected a readable endpoint-optimizer summary.") from error
    development = payload.get("development")
    evaluation = payload.get("evaluation")
    frozen = payload.get("frozen_settings_before_evaluation_test")
    if (
        payload.get("schema") != SOURCE_SUMMARY_SCHEMA
        or payload.get("schema_version") != SOURCE_SUMMARY_SCHEMA_VERSION
        or payload.get("evidence_tier") != "exploratory_noncanonical"
        or payload.get("target_assignment_seed") != protocol.target_assignment_seed
        or payload.get("full_conductance_forward")
        != protocol.full_conductance_forward
        or payload.get("verify_reads_during_training") != 0
        or not isinstance(development, Mapping)
        or development.get("endpoint_seed") != protocol.development.endpoint_seed
        or development.get("test_opened") is not False
        or not isinstance(evaluation, Mapping)
        or evaluation.get("endpoint_seeds")
        != list(protocol.evaluation.endpoint_seeds)
        or evaluation.get("test_used_for_selection") is not False
        or not isinstance(frozen, Mapping)
    ):
        raise ValueError("Endpoint-optimizer source summary contract mismatch.")
    expected_lambdas = _frozen_lambdas_by_variant(protocol)
    for variant in TT_VARIANTS:
        setting = frozen.get(variant)
        if (
            not isinstance(setting, Mapping)
            or setting.get("effective_transfer_lambdas")
            != list(expected_lambdas[variant])
            or setting.get("scale_transfer_lr") is not False
            or float(setting.get("fast_learning_rate_raw_x", -1.0))
            != protocol.base_recovery.optimizers.tt_fast_learning_rate_raw_x
        ):
            raise ValueError(f"Frozen source setting changed for {variant}.")
    return payload


def _candidate_effective_lambdas(
    protocol: Any, *, variant: str, w2_multiplier: float
) -> tuple[float, float]:
    if variant not in TT_VARIANTS:
        raise ValueError(f"Unexpected TT variant: {variant!r}.")
    multiplier = float(w2_multiplier)
    if multiplier not in protocol.development.w2_multiplier_grid:
        raise ValueError("W2 multiplier is outside the frozen development grid.")
    selected_w1, selected_w2 = _frozen_lambdas_by_variant(protocol)[variant]
    return selected_w1, selected_w2 * multiplier


def _with_output_layer_diagnostic(
    report: Mapping[str, Any],
    *,
    variant: str,
    w2_multiplier: float,
    source_lambdas: tuple[float, float],
) -> Mapping[str, Any]:
    enriched = dict(report)
    slow = report["slow_c_pulse"]
    h = report["h"]
    effective = tuple(
        float(value) for value in report["hyperparameters"]["effective_transfer_lambdas"]
    )
    if effective[0] != source_lambdas[0]:
        raise RuntimeError("Follow-up changed the frozen W1 transfer lambda.")
    if effective[1] != source_lambdas[1] * float(w2_multiplier):
        raise RuntimeError("Follow-up W2 transfer lambda differs from its grid point.")
    output = {
        "w1_effective_lambda_frozen": effective[0],
        "source_selected_w2_effective_lambda": source_lambdas[1],
        "w2_multiplier": float(w2_multiplier),
        "w2_effective_lambda": effective[1],
        "w2_slow_c_pulses": int(slow["by_layer"][1]),
        "w2_slow_c_cells_at_cap": int(slow["cells_at_cap_by_layer"][1]),
        "w2_slow_c_maximum_pulses_per_cell": int(
            slow["maximum_per_cell_by_layer"][1]
        ),
        "w2_h_threshold_crossings": (
            None if variant == "tt_v1" else int(h["threshold_crossings_by_layer"][1])
        ),
        "w2_h_cap_blocked_debt_retained_across_steps": (
            None
            if variant == "tt_v1"
            else int(h["cap_blocked_debt_retained_by_layer_across_steps"][1])
        ),
        "w2_final_h_pulse_unit_state": (
            None if variant == "tt_v1" else h["final_pulse_unit_state_by_layer"][1]
        ),
    }
    enriched["output_layer_transfer"] = output
    return enriched


def run_train(request: "TrainRequest") -> int:
    spec = request.spec
    if not isinstance(spec, TtOutputTransferFollowupTrainSpec):
        raise TypeError("Expected the dedicated TT output-transfer follow-up spec.")
    if request.weights is None or request.teacher_weights is None:
        raise ValueError(
            "Expected --weights endpoint-optimizer summary and --teacher-weights."
        )
    if (
        request.base_weights is not None
        or request.resume is not None
        or request.device_data is not None
        or request.device_model is not None
    ):
        raise ValueError("TT output-transfer follow-up rejects base/resume/device inputs.")
    protocol = spec.protocol
    base = protocol.base_recovery
    source_summary_path = request.weights.expanduser().resolve()
    teacher_path = request.teacher_weights.expanduser().resolve()
    _load_source_optimizer_summary(source_summary_path, protocol)
    # Revalidate the original deployment summary and every exact P0 bundle.
    original_summary_path = _expected_summary_path(base)
    _load_predecessor_summary(original_summary_path, base)
    if (
        teacher_path != (_ROOT / base.expected_teacher_weights_path).resolve()
        or sha256_file(teacher_path) != base.expected_teacher_weights_sha256
    ):
        raise ValueError("Frozen teacher path/SHA-256 mismatch.")
    population, pulse_population, endpoints = _load_predecessor_bundle(base)
    input_artifacts = [
        _input("teacher_weights", teacher_path),
        _input("source_endpoint_optimizer_summary", source_summary_path),
        _input("original_cross_array_summary", original_summary_path),
        _input(
            "target_87004_winsorized_population",
            _predecessor_run_dir(base) / base.predecessor.population_relative_path,
        ),
        _input(
            "target_87004_winsorized_population_receipt",
            _predecessor_run_dir(base)
            / base.predecessor.population_receipt_relative_path,
        ),
    ]
    for endpoint_seed, (path, _payload) in endpoints.items():
        input_artifacts.append(_input(f"target_87004_endpoint_{endpoint_seed}", path))
    store = RunStore.create(
        output_root=request.output_dir,
        experiment_id=spec.experiment_id,
        resolved_config=to_plain_data(spec),
        command=request.command,
        repo_root=_ROOT,
        input_artifacts=tuple(input_artifacts),
        resume_capability="unsupported",
    )
    try:
        if spec.student.runtime.device != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("TT output-transfer follow-up requires CUDA.")
        torch.manual_seed(spec.student.runtime.seed)
        torch.cuda.manual_seed_all(spec.student.runtime.seed)
        device = torch.device("cuda")
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
        stack.cost.gain = FIXED_LOGIT_GAIN
        if tuple(population.binding_shapes) != ((1568, 100), (100, 20)):
            raise RuntimeError("Saved endpoint bindings differ from the frozen DRN.")
        symmetry = intrinsic_symmetry_raw_a(pulse_population)
        train_generator_state = loaders.train_generator.get_state().clone()
        source_lambdas = _frozen_lambdas_by_variant(protocol)

        development_seed = protocol.development.endpoint_seed
        development_contract = next(
            item
            for item in base.predecessor.endpoints
            if item.endpoint_seed == development_seed
        )
        development_path, development_endpoint = endpoints[development_seed]
        development_p0 = _evaluate_endpoint_p0(
            endpoint=development_endpoint,
            endpoint_seed=development_seed,
            contract=development_contract,
            pulse_population=pulse_population,
            population=population,
            stack=stack,
            teacher=teacher,
            loaders=loaders,
            protocol=base,
            open_test=False,
        )
        development_reports: dict[str, list[Mapping[str, Any]]] = {}
        selected: dict[str, Mapping[str, Any]] = {}
        artifacts: list[Mapping[str, Any]] = []
        for variant in TT_VARIANTS:
            reports = []
            for multiplier in protocol.development.w2_multiplier_grid:
                lambdas = _candidate_effective_lambdas(
                    protocol, variant=variant, w2_multiplier=multiplier
                )
                raw_report = _run_arm(
                    phase="development_w2_screen",
                    optimizer_name=variant,
                    candidate_scale=float(multiplier),
                    maximum_batches=protocol.development.maximum_batches,
                    endpoint_path=development_path,
                    endpoint=development_endpoint,
                    endpoint_seed=development_seed,
                    population=population,
                    pulse_population=pulse_population,
                    symmetry=symmetry,
                    stack=stack,
                    teacher=teacher,
                    loaders=loaders,
                    train_generator_state=train_generator_state,
                    protocol=base,
                    store=store,
                    before_validation=development_p0["validation"],
                    before_test=None,
                    tt_effective_transfer_lambdas=lambdas,
                )
                report = _with_output_layer_diagnostic(
                    raw_report,
                    variant=variant,
                    w2_multiplier=float(multiplier),
                    source_lambdas=source_lambdas[variant],
                )
                reports.append(report)
                artifacts.append(
                    {"path": report["checkpoint"], "kind": "development_TT_state"}
                )
            development_reports[variant] = reports
            selected[variant] = max(reports, key=_selection_key)
        frozen_settings = {
            variant: {
                "w1_effective_lambda_frozen": source_lambdas[variant][0],
                "source_selected_w2_effective_lambda": source_lambdas[variant][1],
                "w2_multiplier": float(
                    selected[variant]["output_layer_transfer"]["w2_multiplier"]
                ),
                "effective_transfer_lambdas": list(
                    selected[variant]["hyperparameters"][
                        "effective_transfer_lambdas"
                    ]
                ),
                "selected_validation": dict(
                    selected[variant]["after_validation"]
                ),
            }
            for variant in TT_VARIANTS
        }
        # Evaluation test is still sealed until both variant-specific W2
        # multipliers have been frozen from endpoint-89401 validation only.
        evaluation_reports = []
        for endpoint_seed in protocol.evaluation.endpoint_seeds:
            contract = next(
                item
                for item in base.predecessor.endpoints
                if item.endpoint_seed == endpoint_seed
            )
            endpoint_path, endpoint = endpoints[endpoint_seed]
            p0 = _evaluate_endpoint_p0(
                endpoint=endpoint,
                endpoint_seed=endpoint_seed,
                contract=contract,
                pulse_population=pulse_population,
                population=population,
                stack=stack,
                teacher=teacher,
                loaders=loaders,
                protocol=base,
                open_test=True,
            )
            arms = []
            for variant in TT_VARIANTS:
                multiplier = float(frozen_settings[variant]["w2_multiplier"])
                lambdas = tuple(
                    float(value)
                    for value in frozen_settings[variant][
                        "effective_transfer_lambdas"
                    ]
                )
                raw_report = _run_arm(
                    phase="evaluation_w2_followup",
                    optimizer_name=variant,
                    candidate_scale=multiplier,
                    maximum_batches=None,
                    endpoint_path=endpoint_path,
                    endpoint=endpoint,
                    endpoint_seed=endpoint_seed,
                    population=population,
                    pulse_population=pulse_population,
                    symmetry=symmetry,
                    stack=stack,
                    teacher=teacher,
                    loaders=loaders,
                    train_generator_state=train_generator_state,
                    protocol=base,
                    store=store,
                    before_validation=p0["validation"],
                    before_test=p0["test"],
                    tt_effective_transfer_lambdas=lambdas,
                )
                report = _with_output_layer_diagnostic(
                    raw_report,
                    variant=variant,
                    w2_multiplier=multiplier,
                    source_lambdas=source_lambdas[variant],
                )
                arms.append(report)
                artifacts.append(
                    {"path": report["checkpoint"], "kind": "evaluation_TT_state"}
                )
            evaluation_reports.append(
                {
                    "endpoint_seed": endpoint_seed,
                    "semantics": "P&V_endpoint_realization_not_independent_array",
                    "p0": p0,
                    "arms": arms,
                }
            )
        aggregate = {}
        for variant in TT_VARIANTS:
            matching = [
                arm
                for endpoint_report in evaluation_reports
                for arm in endpoint_report["arms"]
                if arm["optimizer"] == variant
            ]
            before = [float(item["before_test"]["student_accuracy"]) for item in matching]
            after = [float(item["after_test"]["student_accuracy"]) for item in matching]
            aggregate[variant] = {
                "endpoints": len(matching),
                "mean_before_test_accuracy": sum(before) / len(before),
                "mean_after_test_accuracy": sum(after) / len(after),
                "mean_test_accuracy_delta": sum(
                    after_value - before_value
                    for before_value, after_value in zip(before, after)
                )
                / len(after),
                "minimum_after_test_accuracy": min(after),
                "maximum_after_test_accuracy": max(after),
                "total_w2_slow_c_pulses": sum(
                    int(item["output_layer_transfer"]["w2_slow_c_pulses"])
                    for item in matching
                ),
                "total_w2_h_threshold_crossings": (
                    None
                    if variant == "tt_v1"
                    else sum(
                        int(
                            item["output_layer_transfer"][
                                "w2_h_threshold_crossings"
                            ]
                        )
                        for item in matching
                    )
                ),
                "total_w2_slow_c_cells_at_cap": sum(
                    int(
                        item["output_layer_transfer"][
                            "w2_slow_c_cells_at_cap"
                        ]
                    )
                    for item in matching
                ),
            }
        summary = {
            "schema": SUMMARY_SCHEMA,
            "schema_version": SUMMARY_SCHEMA_VERSION,
            "evidence_tier": protocol.evidence_tier,
            "target_assignment_seed": protocol.target_assignment_seed,
            "endpoint_semantics": protocol.evaluation.endpoint_semantics,
            "independent_population_count": 1,
            "source_optimizer_recovery": {
                "path": str(source_summary_path),
                "sha256": sha256_file(source_summary_path),
                "role": "frozen_TT_W1_and_source_W2_settings_only",
                "test_metrics_used_for_followup_selection": False,
                "selected_effective_lambdas": {
                    name: list(value) for name, value in source_lambdas.items()
                },
            },
            "development": {
                "endpoint_seed": development_seed,
                "role": protocol.development.role,
                "maximum_batches_per_candidate": (
                    protocol.development.maximum_batches
                ),
                "w2_multiplier_grid": list(
                    protocol.development.w2_multiplier_grid
                ),
                "candidate_reports": development_reports,
                "selected_by_variant": frozen_settings,
                "test_opened": False,
            },
            "frozen_settings_before_evaluation_test": frozen_settings,
            "evaluation": {
                "endpoint_seeds": list(protocol.evaluation.endpoint_seeds),
                "endpoint_realizations": evaluation_reports,
                "aggregate": aggregate,
                "test_used_for_selection": False,
                "validation_used_after_freeze_for_selection": False,
                "sgd_or_adam_rerun": False,
            },
            "optimizer_semantics": {
                "tt_v1": "qualified_OM_plant_TT_v1_minibatch_equation_emulator",
                "tt_v2": "qualified_unchopped_OM_plant_TTv2_minibatch_equation_emulator",
            },
            "full_conductance_forward": protocol.full_conductance_forward,
            "verify_reads_during_training": 0,
            "limitations": [
                "exploratory_noncanonical_one_target_87004_device_population",
                "89402_to_89405_are_PV_endpoint_realizations_not_independent_arrays",
                "model_based_winsorized_IBM_OM_not_a_fabricated_chip",
                "only_W2_transfer_lambda_changed_from_the_completed_TT_comparison",
                "TT_arms_are_minibatch_equation_emulators_not_native_parallel_tiles",
                "fast_A_uses_hidden_model_oracle_intrinsic_symmetry_initialization",
                "fast_A_transfer_uses_held_apparent_write_sample_not_fresh_MVM_read_noise",
                "A_and_C_each_have_a_64_pulse_per_cell_training_cap",
                "one_epoch_recovery_only",
            ],
        }
        summary_path_out = store.run_dir / "artifacts/scientific_summary.json"
        atomic_write_json(summary_path_out, summary)
        artifacts.append({"path": str(summary_path_out), "kind": "scientific_summary"})
        terminal = {
            "evidence_tier": protocol.evidence_tier,
            "target_assignment_seed": protocol.target_assignment_seed,
            "independent_population_count": 1,
            "evaluation_endpoint_count": len(protocol.evaluation.endpoint_seeds),
            "evaluation_arm_count": sum(
                len(item["arms"]) for item in evaluation_reports
            ),
            "mean_after_test_accuracy_by_optimizer": {
                name: value["mean_after_test_accuracy"]
                for name, value in aggregate.items()
            },
            "total_w2_slow_c_pulses_by_optimizer": {
                name: value["total_w2_slow_c_pulses"]
                for name, value in aggregate.items()
            },
            "verify_reads_during_training": 0,
            "test_used_for_selection": False,
            "sgd_or_adam_rerun": False,
        }
        store.complete(metrics=terminal, artifacts=_artifact_records(store, artifacts))
        print(f"run_dir={store.run_dir}", flush=True)
        return 0
    except BaseException as error:
        store.fail(error)
        raise
    finally:
        gc.collect()


__all__ = [
    "_candidate_effective_lambdas",
    "_load_source_optimizer_summary",
    "_with_output_layer_diagnostic",
    "run_train",
]
