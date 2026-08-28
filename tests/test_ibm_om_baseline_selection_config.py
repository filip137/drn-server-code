from __future__ import annotations

from copy import deepcopy
from dataclasses import FrozenInstanceError
from hashlib import sha256
import json
from pathlib import Path

import pytest

from experiments.mnist_relu_drn.config import StudentValidateSpec
from experiments.mnist_relu_drn.ibm_om_baseline_selection_config import (
    BASELINE_POLICIES,
    CONTINUOUS_METRIC_DEFINITION,
    DEVELOPMENT_ASSIGNMENT_SEED,
    EXPECTED_WEIGHTS_SHA256,
    EXPERIMENT_ID,
    HELDOUT_ASSIGNMENT_SEEDS,
    MAXIMUM_DONOR_CANDIDATES_PER_QUAD,
    RESET_READ_SAMPLES,
    SCALE_FRACTIONS,
    SPACING_DELTA_MULTIPLES,
    STANDARD4DELTA_METRIC_DEFINITION,
    BaselineSelectionValidateSpec,
    parse_baseline_selection_config,
    resolve_baseline_selection_spec,
)
from experiments.schema import ConfigError, RunMode
from experiments.study_workflow import load_study_plan


ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = (
    ROOT / "examples" / "mnist_relu_drn" / "ibm_om_baseline_selection"
)
STUDY_PATH = (
    ROOT
    / "studies"
    / "mnist-ibm-om-four-device-baseline-selection-20260828-v1.json"
)


def _production_paths() -> tuple[Path, ...]:
    return tuple(
        CONFIG_ROOT / f"{policy}-heldout-{seed}.json"
        for policy in BASELINE_POLICIES
        for seed in HELDOUT_ASSIGNMENT_SEEDS
    )


def _payload(path: Path | None = None) -> dict:
    source = path or _production_paths()[0]
    return json.loads(source.read_text(encoding="utf-8"))


def test_all_twelve_production_configs_resolve_one_policy_assignment() -> None:
    observed = set()
    hashes = set()
    for path in _production_paths():
        assert path.is_file()
        raw = path.read_bytes()
        assert sha256(raw).hexdigest() not in hashes
        hashes.add(sha256(raw).hexdigest())
        document = parse_baseline_selection_config(
            json.loads(raw.decode("utf-8"))
        )
        spec = resolve_baseline_selection_spec(document, RunMode.VALIDATE)

        assert isinstance(spec, BaselineSelectionValidateSpec)
        assert isinstance(spec.student, StudentValidateSpec)
        assert spec.experiment_id == EXPERIMENT_ID
        assert spec.student.experiment_id == "mnist_relu_drn_kd.v1"
        assert spec.student.runtime.device == "cuda"
        assert spec.student.settings.split == "test"
        assert spec.student.settings.sample_limit is None
        assert spec.student.settings.weight_modifier.type == "none"
        assert spec.protocol.execution.profile == "production"
        assert spec.protocol.assignments.development_seed == 86001
        assert spec.protocol.assignments.allowed_heldout_seeds == (
            87001,
            87002,
            87003,
        )
        observed.add(
            (
                spec.protocol.baseline.selected_policy,
                spec.protocol.assignments.heldout_seed,
            )
        )

    assert len(hashes) == 12
    assert observed == {
        (policy, seed)
        for policy in BASELINE_POLICIES
        for seed in HELDOUT_ASSIGNMENT_SEEDS
    }


def test_resolved_contract_freezes_baseline_and_mapping_decisions() -> None:
    spec = resolve_baseline_selection_spec(
        parse_baseline_selection_config(_payload()),
        RunMode.VALIDATE,
    )

    assert spec.protocol.metrics.primary == CONTINUOUS_METRIC_DEFINITION
    assert spec.protocol.metrics.diagnostic == STANDARD4DELTA_METRIC_DEFINITION
    assert spec.protocol.source.expected_weights_sha256 == EXPECTED_WEIGHTS_SHA256
    assert spec.protocol.source.weights_role == "frozen_relu_source_and_teacher"
    assert spec.protocol.assignments.development_seed == (
        DEVELOPMENT_ASSIGNMENT_SEED
    )
    assert spec.protocol.baseline.policies == BASELINE_POLICIES
    assert spec.protocol.baseline.destination_column_groups == (
        ("G++", "G-+"),
        ("G+-", "G--"),
    )
    assert spec.protocol.reset_commissioning.samples_per_cell == (
        RESET_READ_SAMPLES
    )
    assert spec.protocol.joint_assignment_repair.maximum_candidates_per_quad == (
        MAXIMUM_DONOR_CANDIDATES_PER_QUAD
    )
    assert spec.protocol.mapping.scale_fractions == SCALE_FRACTIONS
    assert spec.protocol.mapping.spacing_delta_multiples == (
        SPACING_DELTA_MULTIPLES
    )
    assert spec.protocol.mapping.standard4delta_refit is False
    assert spec.protocol.mapping.active_headroom == (
        "minimum_across_two_sign_selected_active_cells"
    )
    assert spec.protocol.mapping.physical_conductance == "G=B+d"
    assert spec.protocol.mapping.loading == "full_conductance_sum"
    assert spec.protocol.exclusions.optimizer_updates == 0
    assert spec.protocol.exclusions.hardware_aware_training is False


def test_smoke_config_is_explicitly_nonproduction_and_small() -> None:
    smoke = (
        CONFIG_ROOT
        / "smoke-reference_enforced_destination_columns-heldout-87001.json"
    )
    spec = resolve_baseline_selection_spec(
        parse_baseline_selection_config(_payload(smoke)),
        RunMode.VALIDATE,
    )

    assert spec.protocol.execution.profile == "smoke"
    assert spec.protocol.baseline.selected_policy == (
        "reference_enforced_destination_columns"
    )
    assert spec.student.runtime.device == "cuda"
    assert spec.student.settings.sample_limit == 32


def test_baseline_selection_spec_is_immutable_and_validate_only() -> None:
    document = parse_baseline_selection_config(_payload())
    spec = resolve_baseline_selection_spec(document, RunMode.VALIDATE)
    with pytest.raises(FrozenInstanceError):
        spec.experiment_id = "changed"  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        spec.protocol.assignments.heldout_seed = 1  # type: ignore[misc]
    with pytest.raises(ConfigError, match="validate-only"):
        resolve_baseline_selection_spec(document, RunMode.TRAIN)


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        (
            lambda p: p["baseline_selection"]["metrics"].__setitem__(
                "primary", "wrong"
            ),
            "metrics.primary",
        ),
        (
            lambda p: p["baseline_selection"]["source"].__setitem__(
                "expected_weights_sha256", "0" * 64
            ),
            "expected_weights_sha256",
        ),
        (
            lambda p: p["baseline_selection"]["assignments"].__setitem__(
                "development_seed", 86002
            ),
            "development_seed",
        ),
        (
            lambda p: p["baseline_selection"]["assignments"].__setitem__(
                "heldout_seed", 87001.0
            ),
            "heldout_seed",
        ),
        (
            lambda p: p["baseline_selection"]["assignments"].__setitem__(
                "allowed_heldout_seeds", [87003, 87002, 87001]
            ),
            "allowed_heldout_seeds",
        ),
        (
            lambda p: p["baseline_selection"]["baseline"].__setitem__(
                "selected_policy", "unknown"
            ),
            "selected_policy",
        ),
        (
            lambda p: p["baseline_selection"]["baseline"].__setitem__(
                "destination_column_groups",
                [["G++", "G+-"], ["G-+", "G--"]],
            ),
            "destination_column_groups",
        ),
        (
            lambda p: p["baseline_selection"]["reset_commissioning"].__setitem__(
                "samples_per_cell", 4
            ),
            "samples_per_cell",
        ),
        (
            lambda p: p["baseline_selection"]["joint_assignment_repair"].__setitem__(
                "maximum_candidates_per_quad", 127
            ),
            "maximum_candidates_per_quad",
        ),
        (
            lambda p: p["baseline_selection"]["mapping"].__setitem__(
                "standard4delta_refit", True
            ),
            "standard4delta_refit",
        ),
        (
            lambda p: p["baseline_selection"]["mapping"].__setitem__(
                "active_headroom", "minimum_across_four_cells"
            ),
            "active_headroom",
        ),
        (
            lambda p: p["baseline_selection"]["exclusions"].__setitem__(
                "optimizer_updates", 1
            ),
            "optimizer_updates",
        ),
    ),
)
def test_protocol_rejects_semantic_drift(mutation, message: str) -> None:
    payload = _payload()
    mutation(payload)
    with pytest.raises(ConfigError, match=message):
        parse_baseline_selection_config(payload)


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        (
            lambda p: p["model"].__setitem__("encoding", "differential"),
            "model.encoding",
        ),
        (
            lambda p: p["model"].__setitem__("include_biases", True),
            "include_biases",
        ),
        (
            lambda p: p["solver"].__setitem__("inference_iterations", 5),
            "inference_iterations",
        ),
        (
            lambda p: p["mapping"].__setitem__(
                "scale_fraction_pairs", [[1.0, 1.0]]
            ),
            "scale_fraction_pairs",
        ),
        (
            lambda p: p["modes"]["validate"].__setitem__(
                "sample_limit", 100
            ),
            "sample_limit",
        ),
        (
            lambda p: p["modes"]["validate"].__setitem__(
                "weight_modifier", {"type": "add_normal", "parameters": {}}
            ),
            "weight_modifier",
        ),
    ),
)
def test_parent_student_surface_cannot_change_the_matched_control(
    mutation, message: str
) -> None:
    payload = _payload()
    mutation(payload)
    with pytest.raises(ConfigError, match=message):
        parse_baseline_selection_config(payload)


def test_unknown_protocol_keys_and_extra_modes_fail_closed() -> None:
    payload = _payload()
    payload["baseline_selection"]["unexpected"] = True
    with pytest.raises(ConfigError, match="contain only keys"):
        parse_baseline_selection_config(payload)

    payload = _payload()
    payload["modes"]["train"] = deepcopy(payload["modes"]["validate"])
    with pytest.raises(ConfigError, match="exactly the validate mode"):
        parse_baseline_selection_config(payload)


def test_study_declares_four_policy_arms_and_three_unique_configs_each() -> None:
    plan = load_study_plan(STUDY_PATH)
    assert plan["study_id"] == (
        "mnist-ibm-om-four-device-baseline-selection-20260828-v1"
    )
    assert len(plan["arms"]) == 4
    expected_arm_ids = {
        "independent-cell-reset-mean",
        "shared-quad-reset-max",
        "shared-destination-columns-reset-max",
        "reference-enforced-destination-columns",
    }
    assert {arm["arm_id"] for arm in plan["arms"]} == expected_arm_ids

    declared_paths = []
    for arm in plan["arms"]:
        assert arm["experiment_id"] == EXPERIMENT_ID
        assert arm["mode"] == "validate"
        assert len(arm["configs"]) == 3
        hashes = {config["sha256"] for config in arm["configs"]}
        assert len(hashes) == 3
        declared_paths.extend(config["resolved_path"] for config in arm["configs"])
    assert len(declared_paths) == 12
    assert len(set(declared_paths)) == 12
    assert all("smoke-" not in Path(path).name for path in declared_paths)
    assert {Path(path) for path in declared_paths} == set(_production_paths())
