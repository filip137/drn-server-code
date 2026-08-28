from __future__ import annotations

from copy import deepcopy
from dataclasses import FrozenInstanceError
import json
from pathlib import Path

import pytest

from experiments.mnist_relu_drn.config import StudentValidateSpec
from experiments.mnist_relu_drn.ibm_om_baseline_spacing_pv_config import (
    BASELINE_POSITION_FRACTIONS,
    CONTINUOUS_DIAGNOSTIC_METRIC_DEFINITION,
    ENDPOINT_SEEDS_BY_ASSIGNMENT,
    EXPERIMENT_ID,
    HELDOUT_ASSIGNMENT_SEEDS,
    IDEAL_QUANTIZED_METRIC_DEFINITION,
    P_AND_V_REPEAT_COUNT,
    PV_PERSISTENT_METRIC_DEFINITION,
    SPACING_DELTA_X_MULTIPLIERS,
    STUDY_ID,
    BaselineSpacingPvValidateSpec,
    parse_baseline_spacing_pv_config,
    resolve_baseline_spacing_pv_spec,
)
from experiments.schema import ConfigError, RunMode
from experiments.study_workflow import load_study_plan


ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = (
    ROOT / "examples" / "mnist_relu_drn" / "ibm_om_baseline_spacing_pv"
)
STUDY_PATH = ROOT / "studies" / f"{STUDY_ID}.json"


def _alpha_stem(alpha: float) -> str:
    return {0.0: "000", 0.25: "025", 0.5: "050"}[alpha]


def _production_paths() -> tuple[Path, ...]:
    return tuple(
        CONFIG_ROOT
        / (
            f"alpha_{_alpha_stem(alpha)}_spacing_{spacing}delta-"
            f"heldout-{seed}.json"
        )
        for alpha in BASELINE_POSITION_FRACTIONS
        for spacing in SPACING_DELTA_X_MULTIPLIERS
        for seed in HELDOUT_ASSIGNMENT_SEEDS
    )


def _payload(path: Path | None = None) -> dict:
    source = path or _production_paths()[0]
    return json.loads(source.read_text(encoding="utf-8"))


def test_all_twenty_seven_production_configs_resolve_exact_matrix() -> None:
    observed = set()
    for path in _production_paths():
        assert path.is_file()
        document = parse_baseline_spacing_pv_config(_payload(path))
        spec = resolve_baseline_spacing_pv_spec(document, RunMode.VALIDATE)

        assert isinstance(spec, BaselineSpacingPvValidateSpec)
        assert isinstance(spec.student, StudentValidateSpec)
        assert spec.experiment_id == EXPERIMENT_ID
        assert spec.student.runtime.device == "cuda"
        assert spec.student.settings.sample_limit is None
        assert spec.student.settings.weight_modifier.type == "none"
        assert spec.protocol.execution.profile == "production"
        assert spec.protocol.assignments.endpoint_seeds == (
            ENDPOINT_SEEDS_BY_ASSIGNMENT[
                spec.protocol.assignments.heldout_seed
            ]
        )
        observed.add(
            (
                spec.protocol.baseline_position_fraction,
                spec.protocol.spacing_delta_x_multiplier,
                spec.protocol.assignments.heldout_seed,
            )
        )

    assert observed == {
        (alpha, spacing, seed)
        for alpha in BASELINE_POSITION_FRACTIONS
        for spacing in SPACING_DELTA_X_MULTIPLIERS
        for seed in HELDOUT_ASSIGNMENT_SEEDS
    }


def test_resolved_contract_freezes_calibration_pv_and_historical_parity() -> None:
    spec = resolve_baseline_spacing_pv_spec(
        parse_baseline_spacing_pv_config(_payload()),
        RunMode.VALIDATE,
    )
    protocol = spec.protocol

    assert protocol.metrics.ideal_quantized == IDEAL_QUANTIZED_METRIC_DEFINITION
    assert protocol.metrics.pv_persistent == PV_PERSISTENT_METRIC_DEFINITION
    assert (
        protocol.metrics.continuous_diagnostic
        == CONTINUOUS_DIAGNOSTIC_METRIC_DEFINITION
    )
    assert protocol.calibration.selection_domain.startswith("continuous_bounded")
    assert protocol.calibration.sharing == (
        "one_per_alpha_frozen_across_spacing_heldout_and_endpoint_repeats"
    )
    assert protocol.program_verify.controller == "one_pulse"
    assert protocol.program_verify.tolerance_delta_x_ratio == 0.5
    assert protocol.program_verify.maximum_program_pulses == 128
    assert protocol.program_verify.verify_coordinate == "raw_x=(a+1)/2"
    assert protocol.program_verify.persistent_coordinate == "raw_x=(a+1)/2"
    assert protocol.program_verify.inference_read_noise is False
    assert protocol.program_verify.endpoint_for_accuracy == "persistent"
    assert protocol.program_verify.public_conductance_handoff == (
        "hard_clip_raw_x_to_public_0_1_at_circuit_handoff_with_saved_masks"
    )
    assert protocol.program_verify.repeat_count == P_AND_V_REPEAT_COUNT == 5
    assert protocol.group_lower_definition.startswith("L_j=max_bounded")
    assert protocol.group_upper_definition.startswith("U_j=min_sampled")
    assert protocol.level_capacity_policy == (
        "whole_in_bound_levels_no_short_terminal_interval"
    )
    parity = protocol.reference_study.alpha0_spacing4_parity
    assert parity.development_hardware_instance_id == (
        "feddd5a62ae3ace700dc46f155f54236c58fa769c0d4d6603613ed857b7a0150"
    )
    assert parity.heldout_hardware_instance_ids == (
        "58fb1079e7bd60f53ba93ba61a3dca887352846b9f5a5768e9cd7bb72aed8492",
        "215d46a5ec563abdf2c9450625dea0d6620280dec8a18a413e6199adc72efb50",
        "3be072fb1dcad07d54176b8e0fd30e8f376306853f1955729f0a00254c5475d9",
    )
    assert parity.ideal_correct == (9098, 8580, 8889)
    assert parity.selected_scale_fractions == (1.0, 1.0)
    assert parity.fixed_logit_gain == 14.12537544622754
    assert protocol.physical_invariants.destination_column_groups == (
        ("G++", "G-+"),
        ("G+-", "G--"),
    )
    assert protocol.physical_invariants.loading == "full_conductance_sum"


def test_smoke_config_is_cuda_and_preserves_all_five_pv_repeats() -> None:
    smoke = CONFIG_ROOT / "smoke-alpha_000_spacing_4delta-heldout-87001.json"
    spec = resolve_baseline_spacing_pv_spec(
        parse_baseline_spacing_pv_config(_payload(smoke)),
        RunMode.VALIDATE,
    )

    assert spec.protocol.execution.profile == "smoke"
    assert spec.student.runtime.device == "cuda"
    assert spec.student.settings.sample_limit == 32
    assert spec.protocol.baseline_position_fraction == 0.0
    assert spec.protocol.spacing_delta_x_multiplier == 4
    assert spec.protocol.assignments.endpoint_seeds == (
        89101,
        89102,
        89103,
        89104,
        89105,
    )


def test_spec_is_immutable_and_validate_only() -> None:
    document = parse_baseline_spacing_pv_config(_payload())
    spec = resolve_baseline_spacing_pv_spec(document, RunMode.VALIDATE)
    with pytest.raises(FrozenInstanceError):
        spec.protocol.baseline_position_fraction = 0.5  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        spec.protocol.assignments.heldout_seed = 1  # type: ignore[misc]
    with pytest.raises(ConfigError, match="validate-only"):
        resolve_baseline_spacing_pv_spec(document, RunMode.TRAIN)


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        (
            lambda p: p["baseline_spacing_pv"].__setitem__(
                "baseline_position_fraction", 0
            ),
            "baseline_position_fraction",
        ),
        (
            lambda p: p["baseline_spacing_pv"].__setitem__(
                "baseline_position_fraction", 0.75
            ),
            "baseline_position_fraction",
        ),
        (
            lambda p: p["baseline_spacing_pv"].__setitem__(
                "spacing_delta_x_multiplier", 3
            ),
            "spacing_delta_x_multiplier",
        ),
        (
            lambda p: p["baseline_spacing_pv"]["assignments"].__setitem__(
                "endpoint_seeds", [89101]
            ),
            "endpoint_seeds",
        ),
        (
            lambda p: p["baseline_spacing_pv"]["program_verify"].__setitem__(
                "controller", "adaptive"
            ),
            "controller",
        ),
        (
            lambda p: p["baseline_spacing_pv"]["program_verify"].__setitem__(
                "tolerance_delta_x_ratio", 2.0
            ),
            "tolerance_delta_x_ratio",
        ),
        (
            lambda p: p["baseline_spacing_pv"]["program_verify"].__setitem__(
                "inference_read_noise", True
            ),
            "inference_read_noise",
        ),
        (
            lambda p: p["baseline_spacing_pv"]["program_verify"].__setitem__(
                "verify_coordinate", "apparent_clipped_x"
            ),
            "verify_coordinate",
        ),
        (
            lambda p: p["baseline_spacing_pv"]["calibration"].__setitem__(
                "sharing", "refit_per_spacing"
            ),
            "sharing",
        ),
        (
            lambda p: p["baseline_spacing_pv"]["reference_study"][
                "alpha0_spacing4_parity"
            ].__setitem__("ideal_correct", [9098, 8580, 8890]),
            "ideal_correct",
        ),
    ),
)
def test_protocol_rejects_semantic_drift(mutation, message: str) -> None:
    payload = _payload()
    mutation(payload)
    with pytest.raises(ConfigError, match=message):
        parse_baseline_spacing_pv_config(payload)


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        (
            lambda p: p["runtime"].__setitem__("device", "cpu"),
            "runtime.device",
        ),
        (
            lambda p: p["model"].__setitem__("encoding", "differential"),
            "model.encoding",
        ),
        (
            lambda p: p["modes"]["validate"].__setitem__(
                "weight_modifier", {"type": "add_normal", "parameters": {}}
            ),
            "weight_modifier",
        ),
    ),
)
def test_parent_student_surface_cannot_change(mutation, message: str) -> None:
    payload = _payload()
    mutation(payload)
    with pytest.raises(ConfigError, match=message):
        parse_baseline_spacing_pv_config(payload)


def test_unknown_protocol_keys_and_extra_modes_fail_closed() -> None:
    payload = _payload()
    payload["baseline_spacing_pv"]["unexpected"] = True
    with pytest.raises(ConfigError, match="contain only keys"):
        parse_baseline_spacing_pv_config(payload)

    payload = _payload()
    payload["modes"]["train"] = deepcopy(payload["modes"]["validate"])
    with pytest.raises(ConfigError, match="exactly the validate mode"):
        parse_baseline_spacing_pv_config(payload)


def test_study_declares_nine_arms_and_exact_production_configs() -> None:
    plan = load_study_plan(STUDY_PATH)
    assert plan["study_id"] == STUDY_ID
    assert len(plan["arms"]) == 9

    declared_paths = []
    for arm in plan["arms"]:
        assert arm["experiment_id"] == EXPERIMENT_ID
        assert arm["mode"] == "validate"
        assert len(arm["configs"]) == 3
        assert len({config["sha256"] for config in arm["configs"]}) == 3
        declared_paths.extend(config["resolved_path"] for config in arm["configs"])

    assert len(declared_paths) == 27
    assert len(set(declared_paths)) == 27
    assert all("smoke-" not in Path(path).name for path in declared_paths)
    assert {Path(path) for path in declared_paths} == set(_production_paths())
