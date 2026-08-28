from __future__ import annotations

from copy import deepcopy
from dataclasses import FrozenInstanceError
import json
from pathlib import Path

import pytest

from experiments.definitions import EXPERIMENT_REGISTRY, parse_experiment_config
from experiments.mnist_relu_drn.config import StudentValidateSpec
from experiments.mnist_relu_drn.ibm_om_baseline_spacing_pv_no_clip_config import (
    AFFINE_SLOPE_G_PER_X,
    AFFINE_SUPPORT_ORIGIN_X,
    BASELINE_POSITION_FRACTIONS,
    ENDPOINT_SEEDS_BY_ASSIGNMENT,
    EXACT_SUPPORT_CEILING_X,
    EXACT_SUPPORT_FLOOR_X,
    EXPERIMENT_ID,
    HELDOUT_ASSIGNMENT_SEEDS,
    MAPPED_CEILING_G,
    MAPPED_MINIMUM_G,
    SPACING_DELTA_X_MULTIPLIERS,
    STRICT_POSITIVE_MARGIN_X,
    BaselineSpacingPvNoClipValidateSpec,
    parse_baseline_spacing_pv_no_clip_config,
    resolve_baseline_spacing_pv_no_clip_spec,
)
from experiments.schema import ConfigError, RunMode


ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = (
    ROOT
    / "examples"
    / "mnist_relu_drn"
    / "ibm_om_baseline_spacing_pv_no_clip"
)


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
    return json.loads((path or _production_paths()[0]).read_text(encoding="utf-8"))


def test_exact_twenty_seven_cuda_production_configs_resolve() -> None:
    assert set(CONFIG_ROOT.glob("*.json")) == set(_production_paths())
    observed = set()
    for path in _production_paths():
        document = parse_baseline_spacing_pv_no_clip_config(_payload(path))
        spec = resolve_baseline_spacing_pv_no_clip_spec(document, RunMode.VALIDATE)
        assert isinstance(spec, BaselineSpacingPvNoClipValidateSpec)
        assert isinstance(spec.student, StudentValidateSpec)
        assert spec.student.runtime.device == "cuda"
        assert spec.student.settings.sample_limit is None
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


def test_affine_translation_is_exact_and_has_no_projection() -> None:
    spec = resolve_baseline_spacing_pv_no_clip_spec(
        parse_baseline_spacing_pv_no_clip_config(_payload()),
        RunMode.VALIDATE,
    )
    affine = spec.protocol.affine_embedding
    assert affine.exact_support_floor_x == EXACT_SUPPORT_FLOOR_X
    assert affine.strict_positive_margin_x == STRICT_POSITIVE_MARGIN_X
    assert affine.support_origin_x == AFFINE_SUPPORT_ORIGIN_X
    assert affine.support_origin_x == pytest.approx(
        EXACT_SUPPORT_FLOOR_X - STRICT_POSITIVE_MARGIN_X,
        abs=0.0,
    )
    assert affine.slope_g_per_x == AFFINE_SLOPE_G_PER_X
    assert affine.exact_support_ceiling_x == EXACT_SUPPORT_CEILING_X
    assert affine.mapped_minimum_g == pytest.approx(
        AFFINE_SLOPE_G_PER_X
        * (EXACT_SUPPORT_FLOOR_X - AFFINE_SUPPORT_ORIGIN_X)
    )
    assert affine.mapped_minimum_g == MAPPED_MINIMUM_G
    assert affine.mapped_ceiling_g == pytest.approx(
        AFFINE_SLOPE_G_PER_X
        * (EXACT_SUPPORT_CEILING_X - AFFINE_SUPPORT_ORIGIN_X)
    )
    assert affine.mapped_ceiling_g == MAPPED_CEILING_G
    assert spec.student.model.conductance_min == 0.0
    assert spec.student.model.conductance_max == MAPPED_CEILING_G
    assert spec.protocol.program_verify.endpoint_for_accuracy == "persistent"
    assert spec.protocol.program_verify.circuit_handoff == (
        "affine_map_persistent_raw_x_without_projection"
    )
    assert spec.protocol.program_verify.apparent_endpoint_role == (
        "controller_and_diagnostic_only_never_applied_to_drn"
    )
    assert spec.protocol.physical_invariants.out_of_bounds.endswith(
        "no_projection"
    )


def test_reset_bounding_uses_only_literal_raw_support() -> None:
    spec = resolve_baseline_spacing_pv_no_clip_spec(
        parse_baseline_spacing_pv_no_clip_config(_payload()),
        RunMode.VALIDATE,
    )
    assert spec.protocol.commissioning.bound_policy == (
        "bound_mean_to_exact_sampled_raw_x_min_max_without_public_0_1_intersection"
    )
    assert spec.protocol.group_lower_definition.endswith(
        "in_raw_x_destination_column"
    )
    assert spec.protocol.group_upper_definition.startswith(
        "U_j=min_exact_sampled_raw_x_upper_bound"
    )
    assert spec.protocol.physical_invariants.support == (
        "literal_sampled_native_min_bound_to_max_bound_no_public_intersection"
    )


def test_registry_is_additive_and_validate_only() -> None:
    definition = EXPERIMENT_REGISTRY[EXPERIMENT_ID]
    assert definition.supported_modes == (RunMode.VALIDATE,)
    selected, document = parse_experiment_config(_payload())
    assert selected is definition
    assert document.experiment_id == EXPERIMENT_ID


def test_spec_is_immutable_and_validate_only() -> None:
    document = parse_baseline_spacing_pv_no_clip_config(_payload())
    spec = resolve_baseline_spacing_pv_no_clip_spec(document, RunMode.VALIDATE)
    with pytest.raises(FrozenInstanceError):
        spec.protocol.affine_embedding.support_origin_x = -1.0  # type: ignore[misc]
    with pytest.raises(ConfigError, match="validate-only"):
        resolve_baseline_spacing_pv_no_clip_spec(document, RunMode.TRAIN)


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        (
            lambda p: p["baseline_spacing_pv_no_clip"]["affine_embedding"].__setitem__(
                "support_origin_x", -1.0
            ),
            "support_origin_x",
        ),
        (
            lambda p: p["baseline_spacing_pv_no_clip"]["affine_embedding"].__setitem__(
                "slope_g_per_x", 0.0001
            ),
            "slope_g_per_x",
        ),
        (
            lambda p: p["baseline_spacing_pv_no_clip"]["program_verify"].__setitem__(
                "circuit_handoff", "clip_to_0_1"
            ),
            "circuit_handoff",
        ),
        (
            lambda p: p["baseline_spacing_pv_no_clip"]["program_verify"].__setitem__(
                "endpoint_for_accuracy", "apparent"
            ),
            "endpoint_for_accuracy",
        ),
        (
            lambda p: p["baseline_spacing_pv_no_clip"]["commissioning"].__setitem__(
                "bound_policy", "public_0_1"
            ),
            "bound_policy",
        ),
        (
            lambda p: p["baseline_spacing_pv_no_clip"].__setitem__(
                "baseline_position_fraction", 0.75
            ),
            "baseline_position_fraction",
        ),
        (
            lambda p: p["baseline_spacing_pv_no_clip"]["assignments"].__setitem__(
                "endpoint_seeds", [89101]
            ),
            "endpoint_seeds",
        ),
    ),
)
def test_protocol_rejects_semantic_drift(mutation, message: str) -> None:
    payload = _payload()
    mutation(payload)
    with pytest.raises(ConfigError, match=message):
        parse_baseline_spacing_pv_no_clip_config(payload)


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        (lambda p: p["runtime"].__setitem__("device", "cpu"), "runtime.device"),
        (
            lambda p: p["model"].__setitem__("conductance_max", 0.00011),
            "conductance_max",
        ),
        (
            lambda p: p["model"].__setitem__("encoding", "differential"),
            "model.encoding",
        ),
        (
            lambda p: p["modes"]["validate"].__setitem__("sample_limit", 32),
            "sample_limit",
        ),
    ),
)
def test_parent_student_surface_cannot_change(mutation, message: str) -> None:
    payload = _payload()
    mutation(payload)
    with pytest.raises(ConfigError, match=message):
        parse_baseline_spacing_pv_no_clip_config(payload)


def test_unknown_protocol_keys_and_extra_modes_fail_closed() -> None:
    payload = _payload()
    payload["baseline_spacing_pv_no_clip"]["unexpected"] = True
    with pytest.raises(ConfigError, match="contain only keys"):
        parse_baseline_spacing_pv_no_clip_config(payload)

    payload = _payload()
    payload["modes"]["train"] = deepcopy(payload["modes"]["validate"])
    with pytest.raises(ConfigError, match="exactly the validate mode"):
        parse_baseline_spacing_pv_no_clip_config(payload)
