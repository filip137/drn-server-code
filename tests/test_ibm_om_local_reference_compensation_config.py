from __future__ import annotations

from copy import deepcopy
from dataclasses import FrozenInstanceError
import json
from pathlib import Path

import pytest

from experiments.mnist_relu_drn.config import StudentValidateSpec
from experiments.mnist_relu_drn.ibm_om_local_reference_compensation_config import (
    BASELINE_POLICIES,
    DEVELOPMENT_ASSIGNMENT_SEED,
    EXPECTED_TEACHER_SHA256,
    EXPECTED_WEIGHTS_SHA256,
    EXPERIMENT_ID,
    HELDOUT_ASSIGNMENT_SEEDS,
    METRIC_DEFINITION,
    LocalCompensationValidateSpec,
    parse_local_reference_compensation_config,
    resolve_local_reference_compensation_spec,
)
from experiments.schema import ConfigError, RunMode


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT
    / "examples/mnist_relu_drn/ibm_om_local_reference_compensation/continuous.json"
)


def _payload() -> dict:
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def test_local_compensation_contract_resolves_student_validation() -> None:
    document = parse_local_reference_compensation_config(_payload())
    spec = resolve_local_reference_compensation_spec(document, RunMode.VALIDATE)

    assert isinstance(spec, LocalCompensationValidateSpec)
    assert isinstance(spec.student, StudentValidateSpec)
    assert spec.experiment_id == EXPERIMENT_ID
    assert spec.student.experiment_id == "mnist_relu_drn_kd.v1"
    assert spec.student.model.encoding == "single"
    assert spec.student.settings.split == "test"
    assert spec.student.settings.sample_limit is None
    assert spec.student.settings.weight_modifier.type == "none"
    assert spec.protocol.metric_definition == METRIC_DEFINITION
    assert spec.protocol.source.expected_weights_sha256 == EXPECTED_WEIGHTS_SHA256
    assert spec.protocol.source.expected_teacher_sha256 == EXPECTED_TEACHER_SHA256
    assert spec.protocol.assignments.development_seed == DEVELOPMENT_ASSIGNMENT_SEED
    assert spec.protocol.assignments.heldout_seeds == HELDOUT_ASSIGNMENT_SEEDS
    assert spec.protocol.baselines.policies == BASELINE_POLICIES
    assert spec.protocol.continuous_mapping.scale_fractions == (
        0.125,
        0.25,
        0.5,
        1.0,
    )


def test_contract_freezes_local_identity_and_exact_zero_intervention() -> None:
    spec = resolve_local_reference_compensation_spec(
        parse_local_reference_compensation_config(_payload()),
        RunMode.VALIDATE,
    )

    assert spec.protocol.identity_binding.policy == "sampled_order"
    assert spec.protocol.identity_binding.reassignment == "none"
    exact = spec.protocol.baselines.local_min_l2_exact_zero
    assert exact.scope == "each_existing_four_cell_quad"
    assert exact.sign_order == ("++", "+-", "-+", "--")
    assert exact.constraints[-1] == "B_++-B_+--B_-++B_--=0"
    assert exact.intrinsic_reference_mutation is False
    assert spec.protocol.continuous_mapping.identical_offsets_across_policies
    assert spec.protocol.continuous_mapping.physical_conductance == "G=B+d"
    assert spec.protocol.continuous_mapping.loading == "full_conductance_sum"


def test_contract_freezes_primary_and_secondary_calibration_comparisons() -> None:
    spec = resolve_local_reference_compensation_spec(
        parse_local_reference_compensation_config(_payload()),
        RunMode.VALIDATE,
    )
    calibration = spec.protocol.calibration

    assert calibration.primary_control == (
        "local_nearest_symmetry@cal_local_nearest_symmetry"
    )
    assert calibration.primary_treatment == (
        "local_min_l2_exact_zero@cal_local_nearest_symmetry"
    )
    assert calibration.primary_comparison == "primary_treatment-minus-primary_control"
    assert calibration.secondary_matrix == (
        "all_two_by_two_policy_by_calibration_policy"
    )


def test_local_compensation_spec_is_immutable() -> None:
    spec = resolve_local_reference_compensation_spec(
        parse_local_reference_compensation_config(_payload()),
        RunMode.VALIDATE,
    )
    with pytest.raises(FrozenInstanceError):
        spec.experiment_id = "changed"  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        spec.protocol.assignments.development_seed = 1  # type: ignore[misc]


def test_explicit_frozen_scale_fractions_must_match_student_grid() -> None:
    payload = _payload()
    mapping = payload["local_reference_compensation"]["continuous_mapping"]
    mapping["scale_fraction_source"] = "explicit_frozen"
    mapping["scale_fractions"] = payload["mapping"]["scale_fractions"]

    document = parse_local_reference_compensation_config(payload)
    assert document.protocol.continuous_mapping.scale_fraction_source == (
        "explicit_frozen"
    )

    payload["local_reference_compensation"]["continuous_mapping"][
        "scale_fractions"
    ] = [0.25, 1.0]
    with pytest.raises(ConfigError, match="to equal"):
        parse_local_reference_compensation_config(payload)


def test_contract_is_validate_only() -> None:
    document = parse_local_reference_compensation_config(_payload())
    with pytest.raises(ConfigError, match="validate-only"):
        resolve_local_reference_compensation_spec(document, RunMode.TRAIN)

    payload = _payload()
    payload["modes"]["train"] = deepcopy(payload["modes"]["validate"])
    with pytest.raises(ConfigError, match="exactly the validate mode"):
        parse_local_reference_compensation_config(payload)


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        (
            lambda p: p["local_reference_compensation"].__setitem__(
                "metric_definition", "wrong"
            ),
            "metric_definition",
        ),
        (
            lambda p: p["local_reference_compensation"]["device"].__setitem__(
                "required_aihwkit_version", "1.0.0"
            ),
            "required_aihwkit_version",
        ),
        (
            lambda p: p["local_reference_compensation"]["source"].__setitem__(
                "expected_weights_sha256", "0" * 64
            ),
            "expected_weights_sha256",
        ),
        (
            lambda p: p["local_reference_compensation"]["assignments"].__setitem__(
                "development_seed", 86001.0
            ),
            "development_seed",
        ),
        (
            lambda p: p["local_reference_compensation"]["assignments"].__setitem__(
                "heldout_seeds", [87001, 87003, 87002]
            ),
            "heldout_seeds",
        ),
        (
            lambda p: p["local_reference_compensation"][
                "identity_binding"
            ].__setitem__("reassignment", "within_layer"),
            "reassignment",
        ),
        (
            lambda p: p["local_reference_compensation"]["baselines"].__setitem__(
                "policies", list(reversed(BASELINE_POLICIES))
            ),
            "policies",
        ),
        (
            lambda p: p["local_reference_compensation"]["baselines"][
                "local_min_l2_exact_zero"
            ].__setitem__("intrinsic_reference_mutation", True),
            "intrinsic_reference_mutation",
        ),
        (
            lambda p: p["local_reference_compensation"]["baselines"][
                "local_min_l2_exact_zero"
            ].__setitem__("solver", "unbounded_closed_form"),
            "solver",
        ),
        (
            lambda p: p["local_reference_compensation"][
                "continuous_mapping"
            ].__setitem__("identical_offsets_across_policies", False),
            "identical_offsets_across_policies",
        ),
        (
            lambda p: p["local_reference_compensation"][
                "continuous_mapping"
            ].__setitem__("loading", "offset_only"),
            "loading",
        ),
        (
            lambda p: p["local_reference_compensation"]["calibration"].__setitem__(
                "primary_treatment",
                "local_min_l2_exact_zero@cal_local_min_l2_exact_zero",
            ),
            "primary_treatment",
        ),
        (
            lambda p: p["local_reference_compensation"]["exclusions"].__setitem__(
                "program_and_verify", True
            ),
            "program_and_verify",
        ),
        (
            lambda p: p["local_reference_compensation"]["exclusions"].__setitem__(
                "write_noise", False
            ),
            "write_noise",
        ),
    ),
)
def test_contract_rejects_semantic_drift(mutation, message: str) -> None:
    payload = _payload()
    mutation(payload)
    with pytest.raises(ConfigError, match=message):
        parse_local_reference_compensation_config(payload)


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        (
            lambda p: p["model"].__setitem__("encoding", "differential"),
            "four devices",
        ),
        (
            lambda p: p["model"].__setitem__("include_biases", True),
            "include_biases",
        ),
        (
            lambda p: p["modes"]["validate"].__setitem__("sample_limit", 10),
            "sample_limit",
        ),
        (
            lambda p: p["modes"]["validate"].__setitem__(
                "weight_modifier", {"type": "add_normal", "parameters": {}}
            ),
            "weight_modifier",
        ),
        (
            lambda p: p["mapping"].__setitem__(
                "scale_fraction_pairs", [[1.0, 1.0]]
            ),
            "scale_fraction_pairs",
        ),
    ),
)
def test_parent_student_surface_cannot_enable_excluded_paths(
    mutation, message: str
) -> None:
    payload = _payload()
    mutation(payload)
    with pytest.raises(ConfigError, match=message):
        parse_local_reference_compensation_config(payload)


def test_unknown_protocol_keys_fail_closed() -> None:
    payload = _payload()
    payload["local_reference_compensation"]["baselines"]["unexpected"] = True
    with pytest.raises(ConfigError, match="contain only keys"):
        parse_local_reference_compensation_config(payload)
