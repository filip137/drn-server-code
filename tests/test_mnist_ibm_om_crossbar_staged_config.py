from __future__ import annotations

import pytest

from experiments.mnist_analog_relu.staged_config import (
    ADAM_LEARNING_RATE_GRID,
    ADAM_PULSE_CAP_GRID,
    EXPERIMENT_ID,
    ApplyCorruptionStageSettings,
    DeployStageSettings,
    LiteralAdamHyperparameters,
    OffchipHwaStageSettings,
    OnChipAdamStageSettings,
    SelectionReceiptAdamHyperparameters,
    StagedCrossbarTrainSpec,
    parse_staged_crossbar_config,
    resolve_staged_crossbar_spec,
)
from experiments.schema import ConfigError, RunMode


def _common_payload() -> dict:
    return {
        "schema_version": 2,
        "experiment_id": EXPERIMENT_ID,
        "runtime": {
            "seed": 42,
            "data_seed": 42,
            "device": "cuda",
            "dtype": "float32",
            "required_aihwkit_version": "1.1.0",
        },
        "data": {
            "batch_size": 16,
            "validation_points": 5000,
            "num_points": None,
            "shuffle": True,
        },
        "model": {
            "dims": [784, 256, 10],
            "bias": False,
            "digital_non_linearity": "relu",
            "maximum_input_size": 512,
            "maximum_output_size": 512,
        },
        "source": {
            "teacher_experiment_id": "mnist_relu.v2",
            "teacher_architecture": "bias_free_relu_784_256_10",
            "minimum_validation_accuracy": 0.97,
            "accuracy_comparison": ">",
        },
        "device": {
            "preset": "ReRamArrayOMPresetDevice",
            "evidence_class": "model_based_aihwkit_preset",
            "device_pair_topology": "one_active_om_plus_fixed_intrinsic_reference",
            "effective_state": "q_equals_a_minus_r",
            "assignment_seed": 87008,
            "endpoint_seed": 89802,
            "healthy_programming_corruption_policy": "counterfactual_repaired",
            "fault_source_corruption_policy": "published",
            "bound_policy": "native_sampled_bounds",
            "reference_policy": (
                "fixed_sampled_intrinsic_reference_q_equals_a_minus_r"
            ),
            "deterministic_codebook_pulses": 128,
            "maximum_programming_pulses": 128,
            "verify_tolerance_x": 0.023725,
            "controller": "one_pulse_apparent_verify_persistent_handoff",
            "fault_transition": "post_deployment_published_companion_replay",
            "fault_source_preset_default_corrupt_devices_prob": 0.0,
            "fault_source_enabled_corrupt_devices_prob": 0.1348,
            "fault_source_corrupt_devices_range": 0.01,
        },
        "mapping": {
            "trained_source_weight_scaling_omega": [1, 1],
            "trained_source_scaling_policy": (
                "one_shared_absmax_scale_per_logical_layer"
            ),
            "scratch_weight_scaling_omega": [0, 0],
            "scratch_scaling_policy": (
                "aihwkit_default_no_weight_scaling_direct_q"
            ),
            "scratch_initialization": (
                "aihwkit_analog_linear_default_kaiming_uniform"
            ),
            "scratch_initialization_seed": 42,
            "out_of_bounds_policy": (
                "retain_request_and_report_endpoint_saturation"
            ),
            "codebook_tie_rule": "lowest_pulse_index",
        },
        "evaluation": {"profile": "production_full_validation_and_test"},
        "stage": {"kind": "deploy", "source_kind": "teacher"},
    }


def _forward_noise() -> dict:
    return {
        "model": "aihwkit_1_1_0_softbounds_reference_additive_write_noise",
        "base_state": "support_clamped_digital_master_q_surrogate",
        "equation": (
            "q_apparent_equals_q_digital_master_surrogate_plus_write_noise_std_times_"
            "nominal_dw_min_times_standard_normal"
        ),
        "resampling": "independent_full_array_draw_per_minibatch",
        "samples_per_minibatch": 1,
        "relative_scale": 1,
        "seed": 88042,
        "gradient_estimator": "identity_straight_through",
    }


def _offchip_hwa_stage() -> dict:
    return {
        "kind": "offchip_hwa",
        "policy": "stochastic_apparent_hwa",
        "training_protocol": "full_mnist_ten_epoch_fixed_final",
        "epochs": 10,
        "optimizer": "adam",
        "learning_rate": 1e-4,
        "betas": [0.9, 0.999],
        "epsilon": 1e-8,
        "objective": "teacher_kl",
        "master_q_bounds": [-1, 1],
        "maximum_batches": 3438,
        "checkpoint_policy": "fixed_final_epoch_no_selection",
        "deployment_target": "fixed_final_master_fault_blind_pv",
        "evaluation_forward_policy": (
            "one_sampled_held_apparent_q_per_evaluation"
        ),
        "forward_noise": _forward_noise(),
    }


def _literal_adam_stage(
    learning_rate: float = 1e-3,
    pulse_cap_per_cell: int | None = 128,
    start_state: str = "hwa_healthy_p0",
) -> dict:
    return {
        "kind": "on_chip_adam",
        "start_state": start_state,
        "epochs": 10,
        "optimizer": "pulse_adam",
        "objective": "supervised_cross_entropy",
        "repair_examples": 55000,
        "maximum_batches": 3438,
        "betas": [0.9, 0.999],
        "epsilon": 1e-8,
        "layer_scope": "all",
        "forward_state": "held_apparent_q",
        "write_state": "persistent_q",
        "gradient_estimator": (
            "identity_ste_apparent_q_to_persistent_pulse_update"
        ),
        "checkpoint_policy": "epoch_boundary_exact_resume_fixed_final",
        "hyperparameters": {
            "source": "literal",
            "learning_rate": learning_rate,
            "pulse_cap_per_cell": pulse_cap_per_cell,
        },
    }


def _receipt_adam_stage() -> dict:
    stage = _literal_adam_stage()
    stage["hyperparameters"] = {
        "source": "strict_selection_receipt",
        "allowed_learning_rates": [3e-4, 1e-3, 3e-3],
        "allowed_pulse_caps_per_cell": [128, None],
        "required_start_states": [
            "hwa_healthy_p0",
            "hwa_published_fault",
            "scratch_healthy_p0",
            "scratch_published_fault",
        ],
        "score": (
            "mean_final_apparent_state_validation_cross_entropy_"
            "across_four_matched_starts"
        ),
        "tie_break_policy": "prefer_pulse_cap_128_then_lower_learning_rate",
    }
    return stage


def _parse_with_stage(stage: dict, *, evaluation: str | None = None):
    payload = _common_payload()
    payload["stage"] = stage
    if evaluation is None and stage.get("kind") == "offchip_hwa":
        payload["evaluation"]["profile"] = "tuning_validation_only"
    if evaluation is not None:
        payload["evaluation"]["profile"] = evaluation
    return parse_staged_crossbar_config(payload)


@pytest.mark.parametrize(
    ("stage", "expected_type"),
    [
        (_offchip_hwa_stage(), OffchipHwaStageSettings),
        ({"kind": "deploy", "source_kind": "teacher"}, DeployStageSettings),
        (
            {
                "kind": "apply_corruption",
                "source_kind": "teacher",
                "input_state_kind": "healthy_p0",
                "fault_source": "published_companion",
            },
            ApplyCorruptionStageSettings,
        ),
        (_literal_adam_stage(), OnChipAdamStageSettings),
    ],
)
def test_each_scientific_boundary_is_one_discriminated_stage(
    stage: dict,
    expected_type: type,
) -> None:
    document = _parse_with_stage(stage)
    spec = resolve_staged_crossbar_spec(document, RunMode.TRAIN)

    assert isinstance(spec, StagedCrossbarTrainSpec)
    assert isinstance(spec.stage, expected_type)
    assert spec.model.dims == (784, 256, 10)
    assert spec.device.assignment_seed == 87008
    assert spec.device.endpoint_seed == 89802


@pytest.mark.parametrize("source_kind", ["teacher", "hwa_master", "scratch"])
def test_deploy_accepts_only_the_three_declared_logical_sources(
    source_kind: str,
) -> None:
    document = _parse_with_stage(
        {"kind": "deploy", "source_kind": source_kind}
    )

    assert isinstance(document.stage, DeployStageSettings)
    assert document.stage.source_kind == source_kind
    assert document.mapping.trained_source_weight_scaling_omega == (1.0, 1.0)
    assert document.mapping.scratch_weight_scaling_omega == (0.0, 0.0)
    assert document.mapping.scratch_initialization_seed == 42


def test_hwa_contract_is_ten_epoch_teacher_kl_on_stochastic_apparent_state() -> None:
    document = _parse_with_stage(_offchip_hwa_stage())
    stage = document.stage

    assert isinstance(stage, OffchipHwaStageSettings)
    assert stage.epochs == 10
    assert stage.learning_rate == 1e-4
    assert (stage.beta_1, stage.beta_2) == (0.9, 0.999)
    assert stage.objective == "teacher_kl"
    assert stage.maximum_batches == 3438
    assert stage.evaluation_forward_policy == (
        "one_sampled_held_apparent_q_per_evaluation"
    )
    assert stage.forward_noise.base_state == (
        "support_clamped_digital_master_q_surrogate"
    )
    assert stage.forward_noise.resampling == "independent_full_array_draw_per_minibatch"
    assert stage.forward_noise.gradient_estimator == "identity_straight_through"


def test_hwa_rejects_test_evaluation_during_protocol_development() -> None:
    with pytest.raises(ConfigError, match="tuning_validation_only"):
        _parse_with_stage(
            _offchip_hwa_stage(),
            evaluation="production_full_validation_and_test",
        )


@pytest.mark.parametrize("learning_rate", ADAM_LEARNING_RATE_GRID)
@pytest.mark.parametrize("pulse_cap", ADAM_PULSE_CAP_GRID)
def test_literal_adam_grid_contains_exactly_six_candidates(
    learning_rate: float,
    pulse_cap: int | None,
) -> None:
    document = _parse_with_stage(_literal_adam_stage(learning_rate, pulse_cap))
    stage = document.stage

    assert isinstance(stage, OnChipAdamStageSettings)
    assert isinstance(stage.hyperparameters, LiteralAdamHyperparameters)
    assert stage.hyperparameters.learning_rate == learning_rate
    assert stage.hyperparameters.pulse_cap_per_cell == pulse_cap
    assert stage.epochs == 10
    assert stage.objective == "supervised_cross_entropy"
    assert stage.forward_state == "held_apparent_q"
    assert stage.write_state == "persistent_q"


def test_selection_receipt_freezes_grid_four_starts_score_and_tie_rule() -> None:
    document = _parse_with_stage(_receipt_adam_stage())
    stage = document.stage

    assert isinstance(stage, OnChipAdamStageSettings)
    selection = stage.hyperparameters
    assert isinstance(selection, SelectionReceiptAdamHyperparameters)
    assert selection.allowed_learning_rates == ADAM_LEARNING_RATE_GRID
    assert selection.allowed_pulse_caps_per_cell == ADAM_PULSE_CAP_GRID
    assert selection.required_start_states == (
        "hwa_healthy_p0",
        "hwa_published_fault",
        "scratch_healthy_p0",
        "scratch_published_fault",
    )
    assert selection.tie_break_policy == (
        "prefer_pulse_cap_128_then_lower_learning_rate"
    )


@pytest.mark.parametrize(
    ("profile", "evaluate_test", "test_points"),
    [
        ("tuning_validation_only", False, None),
        ("production_full_validation_and_test", True, 10000),
    ],
)
def test_evaluation_profiles_never_subsample_validation_or_test(
    profile: str,
    evaluate_test: bool,
    test_points: int | None,
) -> None:
    document = _parse_with_stage(
        {"kind": "deploy", "source_kind": "teacher"},
        evaluation=profile,
    )

    assert document.evaluation.evaluate_validation is True
    assert document.evaluation.validation_points == 5000
    assert document.evaluation.evaluate_test is evaluate_test
    assert document.evaluation.test_points == test_points
    assert document.evaluation.selection_metric == "fixed_final_epoch_no_selection"
    assert document.evaluation.primary_state == "held_apparent_q"
    assert document.evaluation.secondary_state == (
        "hidden_persistent_q_diagnostic"
    )


def test_device_contract_programs_repaired_p0_then_allows_published_fault() -> None:
    device = _parse_with_stage(
        {"kind": "deploy", "source_kind": "teacher"}
    ).device

    assert device.device_pair_topology == "one_active_om_plus_fixed_intrinsic_reference"
    assert device.effective_state == "q_equals_a_minus_r"
    assert device.reference_policy == (
        "fixed_sampled_intrinsic_reference_q_equals_a_minus_r"
    )
    assert device.healthy_programming_corruption_policy == "counterfactual_repaired"
    assert device.fault_source_corruption_policy == "published"
    assert device.maximum_programming_pulses == 128
    assert device.verify_tolerance_x == 0.023725
    assert device.fault_source_enabled_corrupt_devices_prob == 0.1348
    assert device.fault_source_corrupt_devices_range == 0.01


@pytest.mark.parametrize(
    ("section", "field", "invalid"),
    [
        ("runtime", "device", "cpu"),
        ("runtime", "seed", 43),
        ("data", "batch_size", 32),
        ("data", "validation_points", 1000),
        ("data", "num_points", 1000),
        ("data", "shuffle", False),
        ("model", "dims", [784, 50, 10]),
        ("model", "bias", True),
        ("source", "minimum_validation_accuracy", 0.9701),
        ("source", "accuracy_comparison", ">="),
        ("device", "reference_policy", "reference_disabled"),
        ("device", "healthy_programming_corruption_policy", "published"),
        ("device", "fault_source_corruption_policy", "counterfactual_repaired"),
        ("device", "maximum_programming_pulses", 127),
        ("device", "verify_tolerance_x", 0.02),
        ("mapping", "trained_source_weight_scaling_omega", [0, 0]),
        ("mapping", "scratch_weight_scaling_omega", [1, 1]),
        ("evaluation", "profile", "validation_and_test_sampled"),
    ],
)
def test_common_large_run_contract_fails_closed(
    section: str,
    field: str,
    invalid: object,
) -> None:
    payload = _common_payload()
    payload[section][field] = invalid

    with pytest.raises(ConfigError):
        parse_staged_crossbar_config(payload)


@pytest.mark.parametrize(
    ("field", "invalid"),
    [
        ("policy", "continuous_hwa"),
        ("epochs", 9),
        ("optimizer", "sgd"),
        ("learning_rate", 3e-4),
        ("objective", "cross_entropy"),
        ("maximum_batches", 3437),
        ("checkpoint_policy", "best_validation"),
        ("evaluation_forward_policy", "support_clamped_master"),
    ],
)
def test_hwa_stage_rejects_protocol_drift(field: str, invalid: object) -> None:
    stage = _offchip_hwa_stage()
    stage[field] = invalid

    with pytest.raises(ConfigError):
        _parse_with_stage(stage)


@pytest.mark.parametrize(
    ("field", "invalid"),
    [
        ("epochs", 5),
        ("optimizer", "adam"),
        ("objective", "teacher_kl"),
        ("repair_examples", 1000),
        ("maximum_batches", 63),
        ("forward_state", "persistent_q"),
        ("write_state", "apparent_q"),
        ("checkpoint_policy", "unsupported"),
    ],
)
def test_on_chip_adam_rejects_protocol_drift(field: str, invalid: object) -> None:
    stage = _literal_adam_stage()
    stage[field] = invalid

    with pytest.raises(ConfigError):
        _parse_with_stage(stage)


@pytest.mark.parametrize(
    ("learning_rate", "pulse_cap"),
    [(1e-4, 128), (1e-3, 64), (1e-2, None)],
)
def test_literal_adam_rejects_values_outside_frozen_grid(
    learning_rate: float,
    pulse_cap: int | None,
) -> None:
    with pytest.raises(ConfigError):
        _parse_with_stage(_literal_adam_stage(learning_rate, pulse_cap))


@pytest.mark.parametrize(
    ("field", "invalid"),
    [
        ("allowed_learning_rates", [1e-3, 3e-4, 3e-3]),
        ("allowed_pulse_caps_per_cell", [None, 128]),
        (
            "required_start_states",
            [
                "hwa_healthy_p0",
                "hwa_published_fault",
                "scratch_healthy_p0",
            ],
        ),
        ("score", "best_single_start_accuracy"),
        ("tie_break_policy", "lower_learning_rate_only"),
    ],
)
def test_selection_receipt_contract_rejects_incomplete_or_reordered_coverage(
    field: str,
    invalid: object,
) -> None:
    stage = _receipt_adam_stage()
    stage["hyperparameters"][field] = invalid

    with pytest.raises(ConfigError):
        _parse_with_stage(stage)


def test_stage_discriminator_rejects_cross_stage_keys() -> None:
    stage = {"kind": "deploy", "source_kind": "teacher", "epochs": 10}

    with pytest.raises(ConfigError):
        _parse_with_stage(stage)


def test_one_assignment_and_one_endpoint_are_scalar_config_fields() -> None:
    payload = _common_payload()
    payload["device"]["endpoint_seeds"] = [89802, 89803]

    with pytest.raises(ConfigError):
        parse_staged_crossbar_config(payload)


def test_config_contains_no_artifact_or_output_path_escape_hatches() -> None:
    for forbidden in ("teacher_weights_path", "device_state_path", "output"):
        payload = _common_payload()
        payload[forbidden] = "/tmp/not-allowed.pt"
        with pytest.raises(ConfigError):
            parse_staged_crossbar_config(payload)


@pytest.mark.parametrize(
    ("field", "invalid"),
    [("schema_version", 1), ("schema_version", 2.0), ("experiment_id", "v2")],
)
def test_schema_identity_is_exact(field: str, invalid: object) -> None:
    payload = _common_payload()
    payload[field] = invalid

    with pytest.raises(ConfigError):
        parse_staged_crossbar_config(payload)


def test_only_train_mode_resolves() -> None:
    document = parse_staged_crossbar_config(_common_payload())

    with pytest.raises(ConfigError):
        resolve_staged_crossbar_spec(document, RunMode.VALIDATE)
