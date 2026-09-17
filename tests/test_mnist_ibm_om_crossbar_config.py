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
from experiments.mnist_analog_relu.runtime import _cpu_thread_contract, _drn_comparison
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
STOCHASTIC_HWA_DIAGNOSTIC_STUDY_PATH = (
    ROOT
    / "studies"
    / "mnist-ibm-om-crossbar-stochastic-apparent-hwa-diagnostic-20260901-v1.json"
)
LONG_HWA_EXACT_PV_STUDY_PATH = (
    ROOT
    / "studies"
    / "mnist-ibm-om-crossbar-long-hwa-exact-pv-20260901-v1.json"
)
LONG_HWA_CROSS_DEFECT_STUDY_PATH = (
    ROOT
    / "studies"
    / "mnist-ibm-om-crossbar-long-hwa-cross-defect-transfer-20260901-v1.json"
)
POPULATION_HWA_STUDY_PATH = (
    ROOT
    / "studies"
    / "mnist-ibm-om-crossbar-population-programming-error-hwa-20260902-v1.json"
)
POPULATION_HWA_CPU1_STUDY_PATH = (
    ROOT
    / "studies"
    / "mnist-ibm-om-crossbar-population-programming-error-hwa-20260902-v2.json"
)
RUNTIME_MODULE = "experiments.mnist_analog_relu.runtime"


def _payload(name: str = "matched_winsorized_onchip_all.json") -> dict:
    return json.loads((CONFIG_DIRECTORY / name).read_text(encoding="utf-8"))


def test_cpu_thread_contract_is_explicit_and_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _payload("hwa_long_repaired_stochastic_hwa.json")
    unpinned = resolve_crossbar_spec(parse_crossbar_config(payload), RunMode.TRAIN)
    assert unpinned.runtime.required_cpu_threads is None

    payload["runtime"]["required_cpu_threads"] = 1
    pinned = resolve_crossbar_spec(parse_crossbar_config(payload), RunMode.TRAIN)
    assert pinned.runtime.required_cpu_threads == 1
    for name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        monkeypatch.setenv(name, "1")
    monkeypatch.setattr(
        "experiments.mnist_analog_relu.runtime.torch.get_num_threads",
        lambda: 1,
    )
    monkeypatch.setattr(
        "experiments.mnist_analog_relu.runtime.torch.get_num_interop_threads",
        lambda: 7,
    )

    report = _cpu_thread_contract(pinned)
    assert report == {
        "required_cpu_threads": 1,
        "environment": {
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
        },
        "torch_intraop_threads": 1,
        "torch_interop_threads": 7,
    }

    monkeypatch.delenv("MKL_NUM_THREADS")
    with pytest.raises(RuntimeError, match="CPU thread contract"):
        _cpu_thread_contract(pinned)


def test_cpu_thread_contract_rejects_cuda() -> None:
    payload = _payload("native_frozen.json")
    payload["runtime"]["required_cpu_threads"] = 1

    with pytest.raises(ConfigError, match="omitted unless runtime.device='cpu'"):
        parse_crossbar_config(payload)


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


@pytest.mark.parametrize(
    ("name", "policy", "epochs", "rates", "maximum_batches"),
    [
        (
            "hwa_noise_diagnostic_repaired_exact_no_hwa.json",
            "none",
            0,
            (0.0, 0.0),
            None,
        ),
        (
            "hwa_noise_diagnostic_repaired_support_only.json",
            "support_clamped_no_update",
            0,
            (0.0, 0.0),
            None,
        ),
        (
            "hwa_noise_diagnostic_repaired_deterministic_hwa.json",
            "continuous_hwa",
            5,
            (1e-4, 1e-4),
            256,
        ),
        (
            "hwa_noise_diagnostic_repaired_stochastic_hwa.json",
            "stochastic_apparent_hwa",
            5,
            (1e-4, 1e-4),
            256,
        ),
    ],
)
def test_hwa_noise_diagnostic_contract(
    name: str,
    policy: str,
    epochs: int,
    rates: tuple[float, float],
    maximum_batches: int | None,
) -> None:
    spec = resolve_crossbar_spec(parse_crossbar_config(_payload(name)), RunMode.TRAIN)

    assert spec.data.num_points == 4096
    assert spec.offchip.policy == policy
    assert spec.offchip.epochs == epochs
    assert spec.offchip.logical_learning_rates == rates
    assert spec.offchip.maximum_batches == maximum_batches
    assert spec.recovery.policy == "none"
    assert spec.transfer.enabled is True
    if policy == "stochastic_apparent_hwa":
        noise = spec.offchip.forward_noise
        assert noise is not None
        assert noise.model == (
            "aihwkit_1_1_0_softbounds_reference_additive_write_noise"
        )
        assert noise.base_state == "support_clamped_persistent_q"
        assert noise.resampling == "independent_full_array_draw_per_minibatch"
        assert noise.samples_per_minibatch == 1
        assert noise.relative_scale == 1.0
        assert noise.seed == 88042
        assert noise.gradient_estimator == "identity_straight_through"
    else:
        assert spec.offchip.forward_noise is None


@pytest.mark.parametrize(
    "role",
    ["exact_no_hwa", "support_only", "deterministic_hwa", "stochastic_hwa"],
)
def test_hwa_noise_diagnostic_corruption_pair_changes_only_policy(role: str) -> None:
    repaired = _payload(f"hwa_noise_diagnostic_repaired_{role}.json")
    published = _payload(f"hwa_noise_diagnostic_published_{role}.json")

    assert repaired["device"].pop("corruption_policy") == "counterfactual_repaired"
    assert published["device"].pop("corruption_policy") == "published"
    assert repaired == published


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda value: value["offchip"].pop("forward_noise"),
            "present for stochastic_apparent_hwa",
        ),
        (
            lambda value: value["offchip"]["forward_noise"].__setitem__(
                "relative_scale", 0.5
            ),
            "to equal 1.0",
        ),
        (
            lambda value: value["offchip"]["forward_noise"].__setitem__(
                "resampling", "once_per_epoch"
            ),
            "independent_full_array_draw_per_minibatch",
        ),
    ],
)
def test_stochastic_hwa_rejects_noise_contract_drift(mutation, message: str) -> None:
    payload = _payload("hwa_noise_diagnostic_repaired_stochastic_hwa.json")
    mutation(payload)

    with pytest.raises(ConfigError, match=message):
        parse_crossbar_config(payload)


def test_stochastic_hwa_diagnostic_study_declares_complete_factorial() -> None:
    plan = load_study_plan(STOCHASTIC_HWA_DIAGNOSTIC_STUDY_PATH)

    assert len(plan["arms"]) == 8
    assert [arm["arm_id"] for arm in plan["arms"]] == [
        "repaired-exact-no-hwa",
        "repaired-support-clamped-no-update",
        "repaired-deterministic-support-hwa",
        "repaired-stochastic-apparent-hwa",
        "published-exact-no-hwa",
        "published-support-clamped-no-update",
        "published-deterministic-support-hwa",
        "published-stochastic-apparent-hwa",
    ]
    for arm in plan["arms"]:
        _, spec = resolve_experiment_config(
            arm["configs"][0]["resolved_path"], RunMode.TRAIN
        )
        assert isinstance(spec, CrossbarTrainSpec)
        assert spec.data.num_points == 4096
        assert spec.transfer.enabled is True


@pytest.mark.parametrize(
    ("name", "policy", "epochs", "maximum_batches"),
    [
        ("hwa_long_repaired_exact_no_hwa.json", "none", 0, None),
        (
            "hwa_long_repaired_deterministic_hwa.json",
            "continuous_hwa",
            10,
            3438,
        ),
        (
            "hwa_long_repaired_stochastic_hwa.json",
            "stochastic_apparent_hwa",
            10,
            3438,
        ),
    ],
)
def test_long_hwa_exact_pv_contract(
    name: str,
    policy: str,
    epochs: int,
    maximum_batches: int | None,
) -> None:
    spec = resolve_crossbar_spec(parse_crossbar_config(_payload(name)), RunMode.TRAIN)

    assert spec.data.num_points is None
    assert spec.offchip.policy == policy
    assert spec.offchip.epochs == epochs
    assert spec.offchip.maximum_batches == maximum_batches
    assert spec.offchip.deployment_target == "fixed_final_master_fault_blind_pv"
    assert spec.offchip.epoch_evaluation == "validation_and_test"
    assert spec.evaluation.sample_limit == 1000
    assert spec.evaluation.maximum_validation_batches == 63
    assert spec.transfer.enabled is True
    if epochs:
        assert spec.offchip.training_protocol == "full_mnist_ten_epoch_fixed_final"
        assert spec.offchip.logical_learning_rates == (1e-4, 1e-4)
    else:
        assert spec.offchip.training_protocol == "no_update"


@pytest.mark.parametrize(
    "role",
    ["exact_no_hwa", "deterministic_hwa", "stochastic_hwa"],
)
def test_long_hwa_corruption_pair_changes_only_policy(role: str) -> None:
    repaired = _payload(f"hwa_long_repaired_{role}.json")
    published = _payload(f"hwa_long_published_{role}.json")

    assert repaired["device"].pop("corruption_policy") == "counterfactual_repaired"
    assert published["device"].pop("corruption_policy") == "published"
    assert repaired == published


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda value: value["offchip"].__setitem__("epochs", 9),
            "ten full-MNIST",
        ),
        (
            lambda value: value["offchip"].__setitem__("maximum_batches", 3437),
            "maximum_batches=3438",
        ),
        (
            lambda value: value["data"].__setitem__("num_points", 4096),
            "all 55,000 training examples",
        ),
        (
            lambda value: value["offchip"].__setitem__(
                "epoch_evaluation", "validation_only"
            ),
            "validation_and_test",
        ),
    ],
)
def test_long_hwa_rejects_budget_drift(mutation, message: str) -> None:
    payload = _payload("hwa_long_repaired_stochastic_hwa.json")
    mutation(payload)

    with pytest.raises(ConfigError, match=message):
        parse_crossbar_config(payload)


def test_long_hwa_exact_pv_study_declares_six_matched_arms() -> None:
    plan = load_study_plan(LONG_HWA_EXACT_PV_STUDY_PATH)

    assert [arm["arm_id"] for arm in plan["arms"]] == [
        "repaired-exact-no-hwa",
        "repaired-long-deterministic-hwa",
        "repaired-long-stochastic-hwa",
        "published-exact-no-hwa",
        "published-long-deterministic-hwa",
        "published-long-stochastic-hwa",
    ]
    for arm in plan["arms"]:
        _, spec = resolve_experiment_config(
            arm["configs"][0]["resolved_path"], RunMode.TRAIN
        )
        assert isinstance(spec, CrossbarTrainSpec)
        assert spec.offchip.deployment_target == "fixed_final_master_fault_blind_pv"
        assert spec.transfer.enabled is True


@pytest.mark.parametrize(
    ("name", "source_policy", "target_policy"),
    [
        (
            "hwa_long_repaired_stochastic_hwa_to_published_targets.json",
            "counterfactual_repaired",
            "published",
        ),
        (
            "hwa_long_published_stochastic_hwa_to_repaired_targets.json",
            "published",
            "counterfactual_repaired",
        ),
    ],
)
def test_long_hwa_cross_defect_configs_separate_source_and_target_policy(
    name: str, source_policy: str, target_policy: str
) -> None:
    spec = resolve_crossbar_spec(parse_crossbar_config(_payload(name)), RunMode.TRAIN)

    assert spec.device.corruption_policy == source_policy
    assert spec.offchip.policy == "stochastic_apparent_hwa"
    assert spec.offchip.training_protocol == "full_mnist_ten_epoch_fixed_final"
    assert spec.offchip.deployment_target == "fixed_final_master_fault_blind_pv"
    assert all(
        target.corruption_policy == target_policy
        for target in spec.transfer.targets
    )


def test_long_hwa_cross_defect_configs_change_only_crossed_policies() -> None:
    repaired = _payload(
        "hwa_long_repaired_stochastic_hwa_to_published_targets.json"
    )
    published = _payload(
        "hwa_long_published_stochastic_hwa_to_repaired_targets.json"
    )

    assert repaired["device"].pop("corruption_policy") == "counterfactual_repaired"
    assert published["device"].pop("corruption_policy") == "published"
    repaired_target_policies = {
        target.pop("corruption_policy")
        for target in repaired["transfer"]["targets"]
    }
    published_target_policies = {
        target.pop("corruption_policy")
        for target in published["transfer"]["targets"]
    }
    assert repaired_target_policies == {"published"}
    assert published_target_policies == {"counterfactual_repaired"}
    assert repaired == published


@pytest.mark.parametrize(
    ("cross_name", "diagonal_name", "expected_target_policy"),
    [
        (
            "hwa_long_repaired_stochastic_hwa_to_published_targets.json",
            "hwa_long_repaired_stochastic_hwa.json",
            "published",
        ),
        (
            "hwa_long_published_stochastic_hwa_to_repaired_targets.json",
            "hwa_long_published_stochastic_hwa.json",
            "counterfactual_repaired",
        ),
    ],
)
def test_long_hwa_cross_config_preserves_diagonal_training_contract(
    cross_name: str, diagonal_name: str, expected_target_policy: str
) -> None:
    cross = _payload(cross_name)
    diagonal = _payload(diagonal_name)

    assert {
        target.pop("corruption_policy")
        for target in cross["transfer"]["targets"]
    } == {expected_target_policy}
    assert cross == diagonal


def test_long_hwa_cross_defect_study_declares_two_off_diagonal_arms() -> None:
    plan = load_study_plan(LONG_HWA_CROSS_DEFECT_STUDY_PATH)

    assert [arm["arm_id"] for arm in plan["arms"]] == [
        "repaired-a-stochastic-hwa-to-published-bd",
        "published-a-stochastic-hwa-to-repaired-bd",
    ]
    for arm in plan["arms"]:
        _, spec = resolve_experiment_config(
            arm["configs"][0]["resolved_path"], RunMode.TRAIN
        )
        assert isinstance(spec, CrossbarTrainSpec)
        assert spec.offchip.policy == "stochastic_apparent_hwa"
        assert all(
            target.corruption_policy != "inherit_source"
            for target in spec.transfer.targets
        )


@pytest.mark.parametrize(
    ("name", "target_policy"),
    [
        ("hwa_long_population_programming_error_hwa.json", "inherit_source"),
        (
            "hwa_long_population_programming_error_hwa_to_published_targets.json",
            "published",
        ),
    ],
)
def test_population_programming_error_hwa_contract(
    name: str,
    target_policy: str,
) -> None:
    spec = resolve_crossbar_spec(parse_crossbar_config(_payload(name)), RunMode.TRAIN)

    assert spec.device.corruption_policy == "counterfactual_repaired"
    assert spec.offchip.policy == "population_programming_error_hwa"
    assert spec.offchip.training_protocol == "full_mnist_ten_epoch_fixed_final"
    assert spec.offchip.epochs == 10
    assert spec.offchip.maximum_batches == 3438
    assert spec.offchip.deployment_target == "fixed_final_master_fault_blind_pv"
    assert spec.offchip.forward_noise is None
    programming = spec.offchip.programming_error
    assert programming is not None
    assert programming.model == (
        "ibm_reram_om_pv128_healthy_accepted_endpoint_residual_v1"
    )
    assert programming.artifact_sha256 == (
        "3030e04d6205dc90d0894ac453d2c1c522dffdc004f69b9c6ab7eaf7ef4b8ba3"
    )
    assert programming.condition_key == (
        "adaptive__lower_to_target__tau_step_0.5"
    )
    assert programming.controller == "adaptive"
    assert programming.start_protocol == "lower_to_target"
    assert programming.maximum_programming_pulses == 128
    assert programming.verify_tolerance_x == pytest.approx(0.023725)
    assert programming.initial_strength == 0.0
    assert programming.final_strength == 1.0
    assert programming.ramp_epochs == 10
    assert programming.seed == 88042
    assert all(
        target.corruption_policy == target_policy
        for target in spec.transfer.targets
    )


def test_population_hwa_configs_change_only_target_corruption_policy() -> None:
    repaired = _payload("hwa_long_population_programming_error_hwa.json")
    published = _payload(
        "hwa_long_population_programming_error_hwa_to_published_targets.json"
    )

    assert {
        target.get("corruption_policy", "inherit_source")
        for target in repaired["transfer"]["targets"]
    } == {"inherit_source"}
    assert {
        target.pop("corruption_policy")
        for target in published["transfer"]["targets"]
    } == {"published"}
    assert published == repaired


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda value: value["offchip"].pop("programming_error"),
            "present for population_programming_error_hwa",
        ),
        (
            lambda value: value["offchip"]["programming_error"].__setitem__(
                "training_population", "fixed_array_a"
            ),
            "healthy_noncorrupt_accepted_endpoints_no_fixed_array",
        ),
        (
            lambda value: value["offchip"]["programming_error"].__setitem__(
                "ramp_epochs", 5
            ),
            "ten-epoch linear programming-error strength ramp",
        ),
        (
            lambda value: value["offchip"].__setitem__(
                "deployment_target", "policy_realized_state"
            ),
            "fixed_final_master_fault_blind_pv",
        ),
    ],
)
def test_population_hwa_rejects_contract_drift(mutation, message: str) -> None:
    payload = _payload("hwa_long_population_programming_error_hwa.json")
    mutation(payload)

    with pytest.raises(ConfigError, match=message):
        parse_crossbar_config(payload)


def test_population_hwa_study_declares_fixed_and_population_cross() -> None:
    plan = load_study_plan(POPULATION_HWA_STUDY_PATH)

    assert [arm["arm_id"] for arm in plan["arms"]] == [
        "fixed-repaired-a-stochastic-hwa-to-repaired-bd",
        "population-programming-error-hwa-to-repaired-bd",
        "fixed-repaired-a-stochastic-hwa-to-published-bd",
        "population-programming-error-hwa-to-published-bd",
    ]
    population_specs = []
    for arm in plan["arms"]:
        _, spec = resolve_experiment_config(
            arm["configs"][0]["resolved_path"], RunMode.TRAIN
        )
        assert isinstance(spec, CrossbarTrainSpec)
        if spec.offchip.policy == "population_programming_error_hwa":
            population_specs.append(spec)
    assert len(population_specs) == 2
    assert all(
        spec.offchip.programming_error is not None
        for spec in population_specs
    )


@pytest.mark.parametrize(
    ("base_name", "cpu1_name"),
    [
        (
            "hwa_long_repaired_stochastic_hwa.json",
            "hwa_long_repaired_stochastic_hwa_cpu1.json",
        ),
        (
            "hwa_long_repaired_stochastic_hwa_to_published_targets.json",
            "hwa_long_repaired_stochastic_hwa_to_published_targets_cpu1.json",
        ),
        (
            "hwa_long_population_programming_error_hwa.json",
            "hwa_long_population_programming_error_hwa_cpu1.json",
        ),
        (
            "hwa_long_population_programming_error_hwa_to_published_targets.json",
            "hwa_long_population_programming_error_hwa_to_published_targets_cpu1.json",
        ),
    ],
)
def test_cpu1_configs_add_only_the_thread_contract(
    base_name: str,
    cpu1_name: str,
) -> None:
    base = _payload(base_name)
    cpu1 = _payload(cpu1_name)

    assert cpu1["runtime"].pop("required_cpu_threads") == 1
    assert cpu1 == base


def test_population_hwa_v2_study_requires_cpu1_for_every_arm() -> None:
    plan = load_study_plan(POPULATION_HWA_CPU1_STUDY_PATH)

    assert len(plan["arms"]) == 4
    for arm in plan["arms"]:
        config = arm["configs"][0]
        assert str(config["resolved_path"]).endswith("_cpu1.json")
        _, spec = resolve_experiment_config(config["resolved_path"], RunMode.TRAIN)
        assert isinstance(spec, CrossbarTrainSpec)
        assert spec.runtime.required_cpu_threads == 1
    assert any(
        "OMP_NUM_THREADS=1" in criterion
        for criterion in plan["completion_criteria"]
    )


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
