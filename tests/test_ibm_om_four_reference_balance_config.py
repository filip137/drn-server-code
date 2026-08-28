from __future__ import annotations

from copy import deepcopy
from dataclasses import FrozenInstanceError
import json
from pathlib import Path

import pytest

from experiments.mnist_relu_drn.config import StudentValidateSpec
from experiments.mnist_relu_drn.ibm_om_four_reference_balance_config import (
    BINDING_POLICIES,
    DEVELOPMENT_ASSIGNMENT_SEED,
    EXPECTED_TEACHER_SHA256,
    EXPECTED_WEIGHTS_SHA256,
    EXPERIMENT_ID,
    HELDOUT_ASSIGNMENT_SEEDS,
    METRIC_DEFINITION,
    BalanceValidateSpec,
    parse_balance_config,
    resolve_balance_spec,
)
from experiments.schema import ConfigError, RunMode


ROOT = Path(__file__).resolve().parents[1]
BASE_CONFIG = (
    ROOT
    / "examples/mnist_relu_drn/ibm_om_differential_pair_hwa_pilot/clean.json"
)


def _reference_balance() -> dict:
    return {
        "metric_definition": METRIC_DEFINITION,
        "device": {
            "evidence_class": "model_based_aihwkit_preset",
            "preset": "reram_array_om",
            "required_aihwkit_version": "1.1.0",
            "corruption_policy": "counterfactual_repaired",
        },
        "source": {
            "expected_weights_sha256": EXPECTED_WEIGHTS_SHA256,
            "expected_teacher_sha256": EXPECTED_TEACHER_SHA256,
        },
        "assignments": {
            "development_seed": DEVELOPMENT_ASSIGNMENT_SEED,
            "heldout_seeds": list(HELDOUT_ASSIGNMENT_SEEDS),
        },
        "binding": {
            "policies": list(BINDING_POLICIES),
            "reference_balanced_binding_v1": {
                "scope": "global_within_layer",
                "sort_key": "mapped_reference_then_original_flat_index",
                "quartet_grouping": "sorted_consecutive_quartets",
                "partition_candidates": (
                    "three_unique_two_vs_two_sign_partitions"
                ),
                "partition_objective": (
                    "minimum_absolute_signed_reference_sum_mismatch"
                ),
                "tie_break": "lexicographic_original_flat_indices",
                "quad_placement": "deterministic_quad_shuffle",
                "quad_shuffle_seed": (
                    "derive_seed(assignment_seed,layer_key,"
                    "reference_balanced_binding_v1)"
                ),
            },
        },
        "continuous_mapping": {
            "coordinate": "clip((a+1)/2,0,1)",
            "active_offsets": "positive_only",
            "active_start": "clip_reference_to_active_bounds",
            "quad_headroom": "minimum_positive_headroom_across_four_cells",
            "logical_sign": "dual_rail_role_placement",
            "physical_conductance": "G=baseline+offset",
            "transfer": "full_signed_conductance_contrast",
            "loading": "full_conductance_sum",
            "quantization": "none",
            "four_delta_spacing": None,
            "pulse_cap": None,
            "scale_fraction_source": "inherit_student_mapping",
            "scale_fractions": None,
        },
        "exclusions": {
            "optimizer_updates": 0,
            "program_and_verify": False,
            "hardware_aware_training": False,
            "write_noise": 0.0,
            "read_noise": 0.0,
            "retention_drift": 0.0,
        },
    }


def _payload() -> dict:
    payload = json.loads(BASE_CONFIG.read_text(encoding="utf-8"))
    payload["experiment_id"] = EXPERIMENT_ID
    payload["model"]["encoding"] = "single"
    payload["modes"] = {
        "validate": {
            "split": "test",
            "sample_limit": None,
            "noise_repeats": 1,
            "weight_modifier": {"type": "none", "parameters": {}},
        }
    }
    payload["reference_balance"] = _reference_balance()
    return payload


def test_balance_contract_resolves_nested_student_validation() -> None:
    document = parse_balance_config(_payload())
    spec = resolve_balance_spec(document, RunMode.VALIDATE)

    assert isinstance(spec, BalanceValidateSpec)
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
    assert spec.protocol.assignments.development_seed == 86001
    assert spec.protocol.assignments.heldout_seeds == (87001, 87002, 87003)
    assert spec.protocol.binding.policies == BINDING_POLICIES
    assert spec.protocol.continuous_mapping.scale_fraction_source == (
        "inherit_student_mapping"
    )
    assert spec.protocol.continuous_mapping.scale_fractions == (
        0.125,
        0.25,
        0.5,
        1.0,
    )


def test_balance_spec_is_immutable() -> None:
    spec = resolve_balance_spec(parse_balance_config(_payload()), RunMode.VALIDATE)
    with pytest.raises(FrozenInstanceError):
        spec.experiment_id = "changed"  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        spec.protocol.assignments.development_seed = 1  # type: ignore[misc]


def test_explicit_frozen_scale_fractions_must_match_student_grid() -> None:
    payload = _payload()
    mapping = payload["reference_balance"]["continuous_mapping"]
    mapping["scale_fraction_source"] = "explicit_frozen"
    mapping["scale_fractions"] = payload["mapping"]["scale_fractions"]

    document = parse_balance_config(payload)
    assert document.protocol.continuous_mapping.scale_fraction_source == (
        "explicit_frozen"
    )
    assert document.protocol.continuous_mapping.scale_fractions == (
        0.125,
        0.25,
        0.5,
        1.0,
    )

    payload["reference_balance"]["continuous_mapping"]["scale_fractions"] = [
        0.25,
        1.0,
    ]
    with pytest.raises(ConfigError, match="to equal"):
        parse_balance_config(payload)


def test_balance_contract_is_validate_only() -> None:
    document = parse_balance_config(_payload())
    with pytest.raises(ConfigError, match="validate-only"):
        resolve_balance_spec(document, RunMode.TRAIN)

    payload = _payload()
    payload["modes"]["train"] = deepcopy(
        json.loads(BASE_CONFIG.read_text(encoding="utf-8"))["modes"]["train"]
    )
    with pytest.raises(ConfigError, match="exactly the validate mode"):
        parse_balance_config(payload)


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        (
            lambda p: p["reference_balance"].__setitem__(
                "metric_definition", "wrong"
            ),
            "metric_definition",
        ),
        (
            lambda p: p["reference_balance"]["device"].__setitem__(
                "evidence_class", "measured"
            ),
            "evidence_class",
        ),
        (
            lambda p: p["reference_balance"]["device"].__setitem__(
                "required_aihwkit_version", "1.0.0"
            ),
            "required_aihwkit_version",
        ),
        (
            lambda p: p["reference_balance"]["source"].__setitem__(
                "expected_weights_sha256", "0" * 64
            ),
            "expected_weights_sha256",
        ),
        (
            lambda p: p["reference_balance"]["assignments"].__setitem__(
                "development_seed", 86002
            ),
            "development_seed",
        ),
        (
            lambda p: p["reference_balance"]["assignments"].__setitem__(
                "development_seed", 86001.0
            ),
            "development_seed",
        ),
        (
            lambda p: p["reference_balance"]["assignments"].__setitem__(
                "heldout_seeds", [87001, 87003, 87002]
            ),
            "heldout_seeds",
        ),
        (
            lambda p: p["reference_balance"]["binding"].__setitem__(
                "policies", list(reversed(BINDING_POLICIES))
            ),
            "policies",
        ),
        (
            lambda p: p["reference_balance"]["binding"][
                "reference_balanced_binding_v1"
            ].__setitem__("quartet_grouping", "local_quartets"),
            "quartet_grouping",
        ),
        (
            lambda p: p["reference_balance"]["binding"][
                "reference_balanced_binding_v1"
            ].__setitem__("tie_break", "random"),
            "tie_break",
        ),
        (
            lambda p: p["reference_balance"]["continuous_mapping"].__setitem__(
                "active_offsets", "signed"
            ),
            "active_offsets",
        ),
        (
            lambda p: p["reference_balance"]["continuous_mapping"].__setitem__(
                "four_delta_spacing", 4
            ),
            "four_delta_spacing",
        ),
        (
            lambda p: p["reference_balance"]["continuous_mapping"].__setitem__(
                "pulse_cap", 128
            ),
            "pulse_cap",
        ),
        (
            lambda p: p["reference_balance"]["exclusions"].__setitem__(
                "optimizer_updates", 1
            ),
            "optimizer_updates",
        ),
        (
            lambda p: p["reference_balance"]["exclusions"].__setitem__(
                "read_noise", 0.01
            ),
            "read_noise",
        ),
        (
            lambda p: p["reference_balance"]["exclusions"].__setitem__(
                "write_noise", False
            ),
            "write_noise",
        ),
    ),
)
def test_balance_contract_rejects_semantic_drift(mutation, message: str) -> None:
    payload = _payload()
    mutation(payload)
    with pytest.raises(ConfigError, match=message):
        parse_balance_config(payload)


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
        parse_balance_config(payload)


def test_unknown_protocol_keys_fail_closed() -> None:
    payload = _payload()
    payload["reference_balance"]["binding"]["unexpected"] = True
    with pytest.raises(ConfigError, match="contain only keys"):
        parse_balance_config(payload)
