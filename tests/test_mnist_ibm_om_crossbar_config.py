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
