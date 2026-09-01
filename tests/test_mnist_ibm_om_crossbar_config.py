from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import sys
from types import ModuleType

import pytest

from ebl.cli import TrainRequest, main
from experiments.definitions import (
    EXPERIMENT_REGISTRY,
    parse_experiment_config,
    resolve_experiment_config,
)
from experiments.mnist_analog_relu.config import (
    CrossbarTrainSpec,
    parse_crossbar_config,
    resolve_crossbar_spec,
)
from experiments.mnist_analog_relu.runtime import _drn_comparison
from experiments.schema import ConfigError, RunMode
from experiments.study_workflow import load_study_plan, prepare_study


ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIRECTORY = (
    ROOT / "examples" / "mnist_analog_relu" / "ibm_om_onchip_importance"
)
STUDY_PATH = (
    ROOT
    / "studies"
    / "mnist-ibm-om-crossbar-relu-onchip-importance-20260830-v1.json"
)
PUBLISHED_DEFECT_STUDY_PATH = (
    ROOT
    / "studies"
    / "mnist-ibm-om-crossbar-fresh-array-published-defects-smoke-20260831-v1.json"
)
SUPERVISED_RETRAINING_STUDY_PATH = (
    ROOT
    / "studies"
    / "mnist-ibm-om-crossbar-supervised-retraining-recovery-20260831-v1.json"
)
STOCHASTIC_TIKI_TAKA_STUDY_PATH = (
    ROOT
    / "studies"
    / "mnist-ibm-om-crossbar-stochastic-tiki-taka-recovery-20260901-v1.json"
)
RUNTIME_MODULE = "experiments.mnist_analog_relu.runtime"


def _payload(name: str = "matched_winsorized_onchip_all.json") -> dict:
    return json.loads((CONFIG_DIRECTORY / name).read_text(encoding="utf-8"))


@pytest.mark.parametrize(
    "path",
    sorted(CONFIG_DIRECTORY.glob("*.json")),
    ids=lambda path: path.stem,
)
def test_all_declared_crossbar_configs_parse_and_resolve(path: Path) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    definition, document = parse_experiment_config(payload)
    spec = definition.resolve(document, RunMode.TRAIN)

    assert definition is EXPERIMENT_REGISTRY["mnist_ibm_om_crossbar_relu.v1"]
    assert isinstance(spec, CrossbarTrainSpec)
    assert spec.model.dims == (784, 50, 10)
    assert spec.device.endpoint_seeds == (89402, 89403, 89404, 89405)
    assert spec.drn_reference.endpoint_seeds == spec.device.endpoint_seeds


@pytest.mark.parametrize(
    ("repaired_name", "published_name"),
    [
        (
            "smoke_matched_winsorized_frozen.json",
            "smoke_published_defects_matched_winsorized_frozen.json",
        ),
        (
            "smoke_matched_winsorized_continuous_hwa_frozen.json",
            "smoke_published_defects_matched_winsorized_continuous_hwa_frozen.json",
        ),
        (
            "smoke_matched_winsorized_qat_frozen.json",
            "smoke_published_defects_matched_winsorized_qat_frozen.json",
        ),
    ],
)
def test_published_defect_smokes_change_only_corruption_policy(
    repaired_name: str,
    published_name: str,
) -> None:
    repaired = _payload(repaired_name)
    published = _payload(published_name)

    assert repaired["device"].pop("corruption_policy") == "counterfactual_repaired"
    assert published["device"].pop("corruption_policy") == "published"
    assert published == repaired


def test_published_defect_transfer_study_declares_three_matched_arms() -> None:
    plan = load_study_plan(PUBLISHED_DEFECT_STUDY_PATH)

    assert len(plan["arms"]) == 3
    assert [arm["arm_id"] for arm in plan["arms"]] == [
        "direct-frozen-master-transfer-published-defects-smoke",
        "continuous-hwa-frozen-master-transfer-published-defects-smoke",
        "qat-frozen-master-transfer-published-defects-smoke",
    ]
    for arm in plan["arms"]:
        assert len(arm["configs"]) == 1
        _, spec = resolve_experiment_config(
            arm["configs"][0]["resolved_path"],
            RunMode.TRAIN,
        )
        assert isinstance(spec, CrossbarTrainSpec)
        assert spec.device.corruption_policy == "published"
        assert tuple(target.assignment_seed for target in spec.transfer.targets) == (
            87005,
            87006,
            87007,
        )


def test_coordinate_matched_recovery_contract_is_frozen() -> None:
    spec = resolve_crossbar_spec(
        parse_crossbar_config(_payload()),
        RunMode.TRAIN,
    )

    assert spec.recovery.learning_rates_q == (6e-5, 6e-5)
    assert spec.recovery.pulse_cap_per_cell == 64
    assert spec.recovery.epochs == 1
    assert spec.offchip.policy == "deterministic_qat"
    assert spec.offchip.epochs == 5
    assert spec.offchip.logical_learning_rates == (1e-4, 1e-4)
    assert spec.offchip.checkpoint_policy == "fixed_final_epoch_no_selection"
    assert spec.evaluation.selection_metric == "fixed_final_epoch_no_selection"
    assert spec.device.bound_policy == "winsorize_raw_active_a_to_unit_interval"


def test_direct_frozen_control_has_no_optimizer_updates() -> None:
    spec = resolve_crossbar_spec(
        parse_crossbar_config(_payload("matched_winsorized_frozen.json")),
        RunMode.TRAIN,
    )

    assert spec.offchip.policy == "none"
    assert spec.offchip.epochs == 0
    assert spec.offchip.logical_learning_rates == (0.0, 0.0)
    assert spec.recovery.policy == "none"
    assert spec.recovery.epochs == 0


def test_star_recovery_contract_is_local_moment_free_and_chronological() -> None:
    spec = resolve_crossbar_spec(
        parse_crossbar_config(_payload("smoke_star_local_recovery.json")),
        RunMode.TRAIN,
    )

    assert spec.recovery.policy == "star_local_pulse_sgd"
    assert spec.recovery.objective == "local_star_state_matching"
    assert spec.recovery.beta_1 is None
    assert spec.recovery.beta_2 is None
    assert spec.recovery.epsilon is None
    assert spec.recovery.star is not None
    assert spec.recovery.star.pulse_rule == "stochastic_pulse_sgd"
    assert spec.recovery.star.calibration_examples == 1024
    assert spec.recovery.star.fault_mask_access == "forbidden"
    assert spec.recovery.star.update_batching == "sequential_per_example"
    assert spec.recovery.star.fault_source_preset_default_corrupt_devices_prob == 0.0
    assert spec.recovery.star.fault_source_enabled_corrupt_devices_prob == 0.1348
    assert spec.recovery.star.fault_source_corrupt_devices_range == 0.01
    assert spec.device.corruption_policy == "counterfactual_repaired"
    assert spec.transfer.enabled is False


@pytest.mark.parametrize(
    ("name", "repair_examples", "maximum_batches"),
    [
        ("smoke_supervised_ce_retraining_matched_16.json", 16, 1),
        ("smoke_supervised_ce_retraining_full_epoch.json", 55_000, 3_438),
    ],
)
def test_supervised_retraining_contract_is_label_bp_and_chronological(
    name: str,
    repair_examples: int,
    maximum_batches: int,
) -> None:
    spec = resolve_crossbar_spec(parse_crossbar_config(_payload(name)), RunMode.TRAIN)

    assert spec.recovery.policy == "supervised_ce_pulse_adam"
    assert spec.recovery.objective == "cross_entropy"
    assert spec.recovery.layer_scope == "all"
    assert spec.recovery.learning_rates_q == (6e-5, 6e-5)
    assert spec.recovery.supervised_bp is not None
    assert spec.recovery.supervised_bp.repair_examples == repair_examples
    assert spec.recovery.supervised_bp.label_source == "ground_truth"
    assert spec.recovery.supervised_bp.gradient_engine == (
        "autograd_full_network_backprop"
    )
    assert spec.recovery.supervised_bp.fault_mask_access == "forbidden"
    assert spec.recovery.maximum_batches == maximum_batches
    assert spec.data.num_points == 16
    assert spec.offchip.maximum_batches == 1
    assert spec.device.corruption_policy == "counterfactual_repaired"
    assert spec.transfer.enabled is False


def test_supervised_retraining_study_declares_matched_and_full_budget_arms() -> None:
    plan = load_study_plan(SUPERVISED_RETRAINING_STUDY_PATH)

    assert [arm["arm_id"] for arm in plan["arms"]] == [
        "supervised-ce-pulse-adam-matched-16",
        "supervised-ce-pulse-adam-full-55000",
    ]


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda value: value["recovery"].__setitem__(
                "objective", "teacher_kl"
            ),
            "cross_entropy",
        ),
        (
            lambda value: value["recovery"]["supervised_bp"].__setitem__(
                "fault_mask_access", "available"
            ),
            "forbidden",
        ),
        (
            lambda value: value["recovery"]["supervised_bp"].__setitem__(
                "fault_source_enabled_corrupt_devices_prob", 0.0
            ),
            "0.1348",
        ),
        (
            lambda value: value["recovery"].__setitem__(
                "maximum_batches", 2
            ),
            "complete declared supervised repair cohort",
        ),
        (
            lambda value: value["device"].__setitem__(
                "corruption_policy", "published"
            ),
            "counterfactually repaired healthy array",
        ),
    ],
)
def test_supervised_retraining_rejects_scientific_drift(
    mutation,
    message: str,
) -> None:
    payload = _payload("smoke_supervised_ce_retraining_matched_16.json")
    mutation(payload)

    with pytest.raises(ConfigError, match=message):
        parse_crossbar_config(payload)


def test_supervised_shadow_program_verify_contract_is_strict_and_chronological() -> None:
    payload = _payload("smoke_supervised_ce_shadow_program_verify_full_epoch.json")
    spec = resolve_crossbar_spec(parse_crossbar_config(payload), RunMode.TRAIN)

    assert spec.recovery.policy == "supervised_ce_shadow_program_verify"
    assert spec.recovery.objective == "cross_entropy"
    assert spec.recovery.layer_scope == "all"
    assert spec.recovery.epochs == 1
    assert spec.recovery.maximum_batches == 3_438
    assert "learning_rates_q" not in payload["recovery"]
    assert spec.recovery.learning_rates_q == (0.0, 0.0)
    assert spec.recovery.pulse_cap_per_cell == 128
    assert spec.recovery.supervised_bp is None
    settings = spec.recovery.supervised_shadow_pv
    assert settings is not None
    assert settings.repair_examples == 55_000
    assert settings.label_source == "ground_truth"
    assert settings.update_batching == "minibatch"
    assert settings.gradient_engine == "autograd_full_network_backprop"
    assert settings.optimizer_state == "digital_fp32_shadow_and_adam"
    assert settings.optimizer_coordinate == (
        "logical_weight_learning_rate_converted_to_q"
    )
    assert settings.logical_learning_rates == (1e-3, 1e-3)
    assert settings.shadow_initial_state == "post_fault_apparent_q"
    assert settings.shadow_bounds == "healthy_source_population_q_bounds"
    assert settings.write_schedule == "fixed_final_epoch_program_verify"
    assert settings.writer == spec.device.controller
    assert settings.maximum_programming_pulses == (
        spec.device.maximum_programming_pulses
    )
    assert settings.verify_tolerance_x == spec.device.verify_tolerance_x
    assert settings.fault_transition == (
        "post_deployment_published_companion_replay"
    )
    assert settings.fault_source_corruption_policy == "published"
    assert settings.fault_source_preset_default_corrupt_devices_prob == 0.0
    assert settings.fault_source_enabled_corrupt_devices_prob == 0.1348
    assert settings.fault_source_corrupt_devices_range == 0.01
    assert settings.fault_mask_access == "forbidden"
    assert spec.device.corruption_policy == "counterfactual_repaired"
    assert spec.transfer.enabled is False


@pytest.mark.parametrize(
    ("name", "policy", "fast_corruption_policy", "expected_learning_rates_q"),
    [
        (
            "supervised_ce_stochastic_pulse_sgd_full_epoch.json",
            "supervised_ce_stochastic_pulse_sgd",
            None,
            (0.0001, 0.0001),
        ),
        (
            "supervised_ce_tiki_taka_v1_repaired_fast_full_epoch.json",
            "supervised_ce_tiki_taka_v1",
            "counterfactual_repaired",
            (0.001, 0.001),
        ),
        (
            "supervised_ce_tiki_taka_v1_published_fast_full_epoch.json",
            "supervised_ce_tiki_taka_v1",
            "published",
            (0.001, 0.001),
        ),
    ],
)
def test_stochastic_pulse_recovery_contract_is_physical_and_chronological(
    name: str,
    policy: str,
    fast_corruption_policy: str | None,
    expected_learning_rates_q: tuple[float, float],
) -> None:
    payload = _payload(name)
    spec = resolve_crossbar_spec(parse_crossbar_config(payload), RunMode.TRAIN)

    assert spec.recovery.policy == policy
    assert spec.recovery.objective == "cross_entropy"
    assert spec.recovery.layer_scope == "all"
    assert spec.recovery.epochs == 1
    assert spec.recovery.learning_rates_q == expected_learning_rates_q
    assert spec.recovery.maximum_batches == 3_438
    assert spec.recovery.pulse_cap_per_cell is None
    assert spec.recovery.beta_1 is None
    assert spec.recovery.beta_2 is None
    assert spec.recovery.epsilon is None
    assert "betas" not in payload["recovery"]
    assert "epsilon" not in payload["recovery"]
    assert "pulse_cap_per_cell" not in payload["recovery"]
    assert spec.data.num_points == 16
    assert spec.offchip.policy == "continuous_hwa"
    assert spec.offchip.maximum_batches == 1
    assert spec.device.corruption_policy == "counterfactual_repaired"
    assert spec.transfer.enabled is False

    settings = spec.recovery.supervised_stochastic_bp
    assert settings is not None
    assert settings.repair_examples == 55_000
    assert settings.label_source == "ground_truth"
    assert settings.update_batching == "minibatch"
    assert settings.gradient_engine == "manual_cross_entropy_backprop"
    assert settings.optimizer_state == "none"
    assert settings.weight_state == (
        "physical_persistent_and_apparent_device_state_no_shadow"
    )
    assert settings.pulse_type == "stochastic_compressed"
    assert settings.desired_bl == 31
    assert settings.fixed_bl is True
    assert settings.update_bl_management is True
    assert settings.update_management is True
    assert settings.um_grad_scale == 1.0
    assert settings.bit_line_seed == 108402
    assert settings.cumulative_pulse_cap is None
    assert settings.final_program_verify is False
    assert settings.fault_transition == (
        "post_deployment_published_companion_replay"
    )
    assert settings.fault_source_corruption_policy == "published"
    assert settings.fault_source_preset_default_corrupt_devices_prob == 0.0
    assert settings.fault_source_enabled_corrupt_devices_prob == 0.1348
    assert settings.fault_source_corrupt_devices_range == 0.01
    assert settings.fault_mask_access == "forbidden"

    tiki_taka = spec.recovery.tiki_taka
    if fast_corruption_policy is None:
        assert tiki_taka is None
        assert "tiki_taka" not in payload["recovery"]
    else:
        assert tiki_taka is not None
        assert tiki_taka.algorithm == "tiki_taka_transfer_compound_v1"
        assert tiki_taka.gamma == 0.0
        assert tiki_taka.fast_lr == 1.0
        assert tiki_taka.transfer_every == 1
        assert tiki_taka.units_in_mbatch is True
        assert tiki_taka.n_reads_per_transfer == 1
        assert tiki_taka.transfer_selection == (
            "sequential_physical_tile_columns"
        )
        assert tiki_taka.transfer_lr == 1.0
        assert tiki_taka.scale_transfer_lr is True
        assert tiki_taka.transfer_columns is True
        assert tiki_taka.with_reset_prob == 0.0
        assert tiki_taka.random_selection is False
        assert tiki_taka.fast_preset == "ReRamArrayOMPresetDevice"
        assert tiki_taka.fast_evidence_class == "model_based_aihwkit_preset"
        assert tiki_taka.fast_assignment_seed == 88004
        assert tiki_taka.fast_endpoint_seeds == (98402, 98403, 98404, 98405)
        assert tiki_taka.fast_initialization == (
            "strict_q_zero_program_verify_mask_blind"
        )
        assert tiki_taka.fast_corruption_policy == fast_corruption_policy
        assert tiki_taka.fast_preset_default_corrupt_devices_prob == 0.0
        assert tiki_taka.fast_enabled_corrupt_devices_prob == 0.1348
        assert tiki_taka.fast_corrupt_devices_range == 0.01


def test_tiki_taka_fast_defect_control_changes_only_fast_corruption_policy() -> None:
    repaired = _payload(
        "supervised_ce_tiki_taka_v1_repaired_fast_full_epoch.json"
    )
    published = _payload(
        "supervised_ce_tiki_taka_v1_published_fast_full_epoch.json"
    )

    assert (
        repaired["recovery"]["tiki_taka"].pop("fast_corruption_policy")
        == "counterfactual_repaired"
    )
    assert (
        published["recovery"]["tiki_taka"].pop("fast_corruption_policy")
        == "published"
    )
    assert published == repaired


def test_stochastic_tiki_taka_study_freezes_three_arm_heldout_contract() -> None:
    plan = load_study_plan(STOCHASTIC_TIKI_TAKA_STUDY_PATH)

    assert [arm["arm_id"] for arm in plan["arms"]] == [
        "direct-bl31-stochastic-pulse-sgd",
        "tiki-taka-v1-repaired-fast-om",
        "tiki-taka-v1-published-fast-om",
    ]
    specs = [
        resolve_experiment_config(arm["configs"][0]["resolved_path"], RunMode.TRAIN)[
            1
        ]
        for arm in plan["arms"]
    ]
    assert all(isinstance(spec, CrossbarTrainSpec) for spec in specs)
    assert [spec.recovery.policy for spec in specs] == [
        "supervised_ce_stochastic_pulse_sgd",
        "supervised_ce_tiki_taka_v1",
        "supervised_ce_tiki_taka_v1",
    ]
    assert [spec.recovery.learning_rates_q for spec in specs] == [
        (0.0001, 0.0001),
        (0.001, 0.001),
        (0.001, 0.001),
    ]
    assert specs[1].recovery.tiki_taka is not None
    assert specs[2].recovery.tiki_taka is not None
    assert specs[1].recovery.tiki_taka.fast_corruption_policy == (
        "counterfactual_repaired"
    )
    assert specs[2].recovery.tiki_taka.fast_corruption_policy == "published"
    assert all(
        spec.device.endpoint_seeds == (89402, 89403, 89404, 89405)
        for spec in specs
    )
    assert any(
        "Endpoint 89402 is development-only; endpoints 89403-89405"
        in criterion
        for criterion in plan["completion_criteria"]
    )


def test_stochastic_pulse_recovery_allows_positive_declared_q_rates() -> None:
    payload = _payload("supervised_ce_stochastic_pulse_sgd_full_epoch.json")
    payload["recovery"]["learning_rates_q"] = [0.002, 0.003]

    spec = parse_crossbar_config(payload)

    assert spec.recovery.learning_rates_q == (0.002, 0.003)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda value: value["recovery"].__setitem__(
                "learning_rates_q", [0.0, 0.01]
            ),
            "two positive numbers",
        ),
        (
            lambda value: value["recovery"].__setitem__("epochs", 2),
            "exactly one epoch",
        ),
        (
            lambda value: value["data"].__setitem__("num_points", 32),
            "exact prior P0 source contract",
        ),
        (
            lambda value: value["offchip"].__setitem__("maximum_batches", 2),
            "exact prior P0 source contract",
        ),
        (
            lambda value: value["recovery"]["supervised_stochastic_bp"].__setitem__(
                "gradient_engine", "autograd_full_network_backprop"
            ),
            "manual_cross_entropy_backprop",
        ),
        (
            lambda value: value["recovery"]["supervised_stochastic_bp"].__setitem__(
                "optimizer_state", "digital_adam"
            ),
            "to equal 'none'",
        ),
        (
            lambda value: value["recovery"]["supervised_stochastic_bp"].__setitem__(
                "weight_state", "persistent_device_q_only"
            ),
            "physical_persistent_and_apparent_device_state_no_shadow",
        ),
        (
            lambda value: value["recovery"]["supervised_stochastic_bp"].__setitem__(
                "desired_bl", 30
            ),
            "maximum bit-line length 31",
        ),
        (
            lambda value: value["recovery"]["supervised_stochastic_bp"].__setitem__(
                "update_bl_management", False
            ),
            "to be true",
        ),
        (
            lambda value: value["recovery"]["supervised_stochastic_bp"].__setitem__(
                "cumulative_pulse_cap", 64
            ),
            "no cumulative per-cell pulse cap",
        ),
        (
            lambda value: value["recovery"]["supervised_stochastic_bp"].__setitem__(
                "final_program_verify", True
            ),
            "to be false",
        ),
        (
            lambda value: value["recovery"]["supervised_stochastic_bp"].__setitem__(
                "fault_mask_access", "available"
            ),
            "forbidden",
        ),
        (
            lambda value: value["recovery"].__setitem__(
                "betas", [0.9, 0.999]
            ),
            "contain only keys",
        ),
        (
            lambda value: value["device"].__setitem__(
                "corruption_policy", "published"
            ),
            "counterfactually repaired healthy array",
        ),
    ],
)
def test_stochastic_pulse_recovery_rejects_scientific_drift(
    mutation,
    message: str,
) -> None:
    payload = _payload("supervised_ce_stochastic_pulse_sgd_full_epoch.json")
    mutation(payload)

    with pytest.raises(ConfigError, match=message):
        parse_crossbar_config(payload)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda value: value["recovery"]["tiki_taka"].__setitem__(
                "gamma", 0.1
            ),
            "to equal 0.0",
        ),
        (
            lambda value: value["recovery"]["tiki_taka"].__setitem__(
                "fast_lr", 0.5
            ),
            "to equal 1.0",
        ),
        (
            lambda value: value["recovery"]["tiki_taka"].__setitem__(
                "transfer_every", 2
            ),
            "to equal 1",
        ),
        (
            lambda value: value["recovery"]["tiki_taka"].__setitem__(
                "transfer_selection", "sequential_global_columns"
            ),
            "sequential_physical_tile_columns",
        ),
        (
            lambda value: value["recovery"]["tiki_taka"].__setitem__(
                "n_reads_per_transfer", 2
            ),
            "to equal 1",
        ),
        (
            lambda value: value["recovery"]["tiki_taka"].__setitem__(
                "scale_transfer_lr", False
            ),
            "to be true",
        ),
        (
            lambda value: value["recovery"]["tiki_taka"].__setitem__(
                "with_reset_prob", 1.0
            ),
            "to equal 0.0",
        ),
        (
            lambda value: value["recovery"]["tiki_taka"].__setitem__(
                "random_selection", True
            ),
            "to be false",
        ),
        (
            lambda value: value["recovery"]["tiki_taka"].__setitem__(
                "fast_assignment_seed", 88005
            ),
            "to equal 88004",
        ),
        (
            lambda value: value["recovery"]["tiki_taka"].__setitem__(
                "fast_endpoint_seeds", [98402, 98403]
            ),
            "frozen independent fast-array seeds",
        ),
        (
            lambda value: value["recovery"]["tiki_taka"].__setitem__(
                "fast_initialization", "exact_zero"
            ),
            "strict_q_zero_program_verify_mask_blind",
        ),
        (
            lambda value: value["recovery"]["tiki_taka"].__setitem__(
                "fast_corruption_policy", "none"
            ),
            "counterfactual_repaired.*published",
        ),
    ],
)
def test_tiki_taka_v1_rejects_scientific_drift(mutation, message: str) -> None:
    payload = _payload(
        "supervised_ce_tiki_taka_v1_repaired_fast_full_epoch.json"
    )
    mutation(payload)

    with pytest.raises(ConfigError, match=message):
        parse_crossbar_config(payload)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda value: value["recovery"].__setitem__(
                "objective", "teacher_kl"
            ),
            "cross_entropy",
        ),
        (
            lambda value: value["recovery"].__setitem__(
                "learning_rates_q", [1e-3, 1e-3]
            ),
            "contain only keys",
        ),
        (
            lambda value: value["recovery"]["supervised_shadow_pv"].__setitem__(
                "logical_learning_rates", [6e-5, 6e-5]
            ),
            "predeclared logical rates",
        ),
        (
            lambda value: value["recovery"]["supervised_shadow_pv"].__setitem__(
                "optimizer_coordinate", "q_coordinate"
            ),
            "logical_weight_learning_rate_converted_to_q",
        ),
        (
            lambda value: value["recovery"]["supervised_shadow_pv"].__setitem__(
                "optimizer_state", "device_local"
            ),
            "digital_fp32_shadow_and_adam",
        ),
        (
            lambda value: value["recovery"]["supervised_shadow_pv"].__setitem__(
                "fault_mask_access", "available"
            ),
            "forbidden",
        ),
        (
            lambda value: value["recovery"]["supervised_shadow_pv"].__setitem__(
                "fault_source_preset_default_corrupt_devices_prob", 0.1348
            ),
            "to equal 0.0",
        ),
        (
            lambda value: value["recovery"]["supervised_shadow_pv"].__setitem__(
                "fault_source_enabled_corrupt_devices_prob", 0.0
            ),
            "to equal 0.1348",
        ),
        (
            lambda value: value["recovery"]["supervised_shadow_pv"].__setitem__(
                "fault_source_corrupt_devices_range", 0.1
            ),
            "to equal 0.01",
        ),
        (
            lambda value: value["recovery"]["supervised_shadow_pv"].__setitem__(
                "maximum_programming_pulses", 127
            ),
            "128-pulse",
        ),
        (
            lambda value: value["recovery"]["supervised_shadow_pv"].__setitem__(
                "verify_tolerance_x", 0.02
            ),
            "0.023725",
        ),
        (
            lambda value: value["recovery"]["supervised_shadow_pv"].__setitem__(
                "write_schedule", "after_every_minibatch"
            ),
            "fixed_final_epoch_program_verify",
        ),
        (
            lambda value: value["recovery"].update(
                {"maximum_batches": 1}
            )
            or value["recovery"]["supervised_shadow_pv"].__setitem__(
                "repair_examples", 16
            ),
            "complete 55,000-example",
        ),
        (
            lambda value: value["device"].__setitem__(
                "corruption_policy", "published"
            ),
            "counterfactually repaired healthy array",
        ),
        (
            lambda value: value["transfer"].update(
                {
                    "enabled": True,
                    "source_state": "offchip_fixed_final_master",
                    "targets": [
                        {
                            "assignment_seed": 87005,
                            "endpoint_seeds": [89502, 89503, 89504, 89505],
                        }
                    ],
                }
            ),
            "fresh-array transfer disabled",
        ),
    ],
)
def test_supervised_shadow_program_verify_rejects_scientific_drift(
    mutation,
    message: str,
) -> None:
    payload = _payload("smoke_supervised_ce_shadow_program_verify_full_epoch.json")
    mutation(payload)

    with pytest.raises(ConfigError, match=message):
        parse_crossbar_config(payload)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda value: value["device"].__setitem__("corruption_policy", "published"),
            "counterfactually repaired healthy array",
        ),
        (
            lambda value: value["transfer"].update(
                {
                    "enabled": True,
                    "source_state": "offchip_fixed_final_master",
                    "targets": [
                        {
                            "assignment_seed": 87005,
                            "endpoint_seeds": [89502, 89503, 89504, 89505],
                        }
                    ],
                }
            ),
            "fresh-array transfer disabled",
        ),
        (
            lambda value: value["offchip"].update(
                {
                    "policy": "none",
                    "epochs": 0,
                    "logical_learning_rates": [0.0, 0.0],
                }
            ),
            "after continuous HWA",
        ),
        (
            lambda value: value["recovery"].__setitem__("epochs", 2),
            "exactly one epoch",
        ),
        (
            lambda value: value["recovery"].__setitem__("maximum_batches", None),
            "reuse its first-epoch batch budget",
        ),
        (
            lambda value: value["recovery"]["star"].__setitem__(
                "fault_mask_access", "available"
            ),
            "forbidden",
        ),
        (
            lambda value: value["recovery"]["star"].__setitem__(
                "fault_source_preset_default_corrupt_devices_prob", 0.1348
            ),
            "to equal 0.0",
        ),
        (
            lambda value: value["recovery"]["star"].__setitem__(
                "fault_source_enabled_corrupt_devices_prob", 0.0
            ),
            "to equal 0.1348",
        ),
        (
            lambda value: value["recovery"]["star"].__setitem__(
                "fault_source_corrupt_devices_range", 0.1
            ),
            "to equal 0.01",
        ),
        (
            lambda value: value["recovery"].__setitem__("betas", [0.9, 0.999]),
            "contain only keys",
        ),
    ],
)
def test_star_recovery_rejects_nonlocal_or_nonchronological_drift(
    mutation,
    message: str,
) -> None:
    payload = _payload("smoke_star_local_recovery.json")
    mutation(payload)

    with pytest.raises(ConfigError, match=message):
        parse_crossbar_config(payload)


def test_qat_recovery_arms_share_one_offchip_starting_state_contract() -> None:
    names = (
        "matched_winsorized_qat_frozen.json",
        "matched_winsorized_onchip_all.json",
        "matched_winsorized_onchip_input_only.json",
        "matched_winsorized_onchip_output_only.json",
    )
    specs = [
        resolve_crossbar_spec(parse_crossbar_config(_payload(name)), RunMode.TRAIN)
        for name in names
    ]

    assert all(spec.offchip == specs[0].offchip for spec in specs[1:])
    assert all(spec.data == specs[0].data for spec in specs[1:])
    assert all(spec.source == specs[0].source for spec in specs[1:])
    assert all(spec.device == specs[0].device for spec in specs[1:])
    assert all(spec.mapping == specs[0].mapping for spec in specs[1:])


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda value: value["recovery"].__setitem__("pulse_cap_per_cell", 63),
            "64-pulse",
        ),
        (
            lambda value: value["evaluation"].__setitem__(
                "selection_metric", "validation_accuracy_then_kl"
            ),
            "fixed_final_epoch_no_selection",
        ),
        (
            lambda value: value["offchip"].__setitem__("epochs", 4),
            "five epochs",
        ),
        (
            lambda value: value["drn_reference"].__setitem__(
                "endpoint_seeds", [89402]
            ),
            "same assignment and endpoint cohort",
        ),
    ],
)
def test_matched_contract_rejects_scientific_drift(mutation, message: str) -> None:
    payload = deepcopy(_payload())
    mutation(payload)

    with pytest.raises(ConfigError, match=message):
        parse_crossbar_config(payload)


def test_crossbar_experiment_rejects_non_train_modes() -> None:
    document = parse_crossbar_config(_payload())

    with pytest.raises(ConfigError, match="equal 'train'"):
        resolve_crossbar_spec(document, RunMode.VALIDATE)


def test_default_cli_lazily_dispatches_crossbar_train(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = ModuleType(RUNTIME_MODULE)
    seen: list[TrainRequest] = []

    def run_train(request: TrainRequest) -> int:
        seen.append(request)
        return 37

    runtime.run_train = run_train  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, RUNTIME_MODULE, runtime)
    config_path = CONFIG_DIRECTORY / "matched_winsorized_frozen.json"
    teacher = ROOT / "data" / "mnist_relu_teacher_fixed_init_20260816.pt"
    result = main(
        [
            "train",
            "--config",
            str(config_path),
            "--output-dir",
            str(tmp_path / "run"),
            "--teacher-weights",
            str(teacher),
        ]
    )

    assert result == 37
    assert len(seen) == 1
    assert seen[0].definition is EXPERIMENT_REGISTRY["mnist_ibm_om_crossbar_relu.v1"]
    assert isinstance(seen[0].spec, CrossbarTrainSpec)
    assert seen[0].teacher_weights == teacher


def test_study_plan_prepares_and_resolves_exact_production_matrix(
    tmp_path: Path,
) -> None:
    plan = load_study_plan(STUDY_PATH)
    declared = [config for arm in plan["arms"] for config in arm["configs"]]

    assert len(plan["arms"]) == 8
    assert len(declared) == 8
    assert [arm["arm_id"] for arm in plan["arms"]] == [
        "matched-direct-frozen",
        "matched-continuous-hwa-frozen",
        "matched-qat-frozen",
        "matched-qat-onchip-all",
        "matched-qat-onchip-input-only",
        "matched-qat-onchip-output-only",
        "native-qat-frozen",
        "native-qat-onchip-all",
    ]
    assert all("smoke" not in Path(item["resolved_path"]).name for item in declared)
    for item in declared:
        definition, spec = resolve_experiment_config(
            item["resolved_path"],
            RunMode.TRAIN,
        )
        assert definition is EXPERIMENT_REGISTRY["mnist_ibm_om_crossbar_relu.v1"]
        assert isinstance(spec, CrossbarTrainSpec)

    study_root = prepare_study(STUDY_PATH, tmp_path / "results")
    prepared = json.loads((study_root / "study.json").read_text(encoding="utf-8"))
    assert len(prepared["arms"]) == 8
    assert all(
        (study_root / "runs" / arm["arm_id"]).is_dir()
        for arm in prepared["arms"]
    )


def test_drn_comparison_reports_paired_gain_and_teacher_gap_closure() -> None:
    reference = json.loads(
        (ROOT / "studies" / "references" / "drn_ibm_om_endpoint_recovery_20260829.json").read_text(
            encoding="utf-8"
        )
    )
    rows = [
        {
            "endpoint_seed": seed,
            "p0_test": {"student_accuracy": initial, "teacher_accuracy": 0.9736},
            "fixed_final_test": {"student_accuracy": final},
        }
        for seed, initial, final in zip(
            (89402, 89403, 89404, 89405),
            (0.80, 0.81, 0.82, 0.83),
            (0.90, 0.91, 0.92, 0.93),
            strict=True,
        )
    ]

    comparison = _drn_comparison(
        rows=rows,
        reference=reference,
        recovery_policy="pulse_adam",
        bound_policy="winsorize_raw_active_a_to_unit_interval",
        offchip_policy="deterministic_qat",
        layer_scope="all",
        crossbar_deployment_source_accuracy=0.95,
    )

    assert comparison["bound_treatment_matched"] is True
    assert comparison["network_forward_state_matched"] is False
    assert comparison["device_model_treatment_matched"] is False
    assert comparison["crossbar_recovery_gain_percentage_points"] == pytest.approx(
        [10.0, 10.0, 10.0, 10.0]
    )
    assert comparison["crossbar_fraction_of_deployment_source_accuracy_gap_closed"][0] == pytest.approx(
        0.1 / (0.95 - 0.8)
    )
    assert comparison["crossbar_fraction_of_teacher_accuracy_gap_closed"][0] == pytest.approx(
        0.1 / (0.9736 - 0.8)
    )
    assert comparison["drn_mean_p0"] == pytest.approx(0.6316)
    assert comparison["drn_mean_open_loop_adam"] == pytest.approx(0.93065)
    assert comparison["drn_source_pretransfer_accuracy"] == pytest.approx(0.9414)
    assert comparison["drn_requested_remap_no_write_accuracy"] == pytest.approx(0.9202)
    assert comparison["drn_source_to_p0_loss_percentage_points"][0] == pytest.approx(
        100.0 * (0.9414 - 0.6249)
    )
    assert comparison["predeclared_importance_rule"] == {
        "negligible": False,
        "floor_crossing_candidate": True,
        "needed_for_this_protocol": None,
        "requires_independent_qat_frozen_target_and_p0_hash_audit": True,
        "accuracy_floor": 0.9,
        "negligible_mean_gain_pp_threshold": 1.0,
        "negligible_every_endpoint_gain_pp_threshold": 2.0,
    }
