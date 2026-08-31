import json

import pytest
import torch

from experiments.artifacts import sha256_file
from experiments.mnist_relu_drn.ibm_om_baseline_selection import (
    quad_stack,
    scatter_quads,
)
from experiments.mnist_relu_drn.ibm_om_positive_conductance import (
    CALIBRATION_SCHEMA,
    DESTINATION_PAIR_BASELINE,
    EMPIRICAL_QUANTILES,
    PROGRESS_COORDINATE,
    QUAD_BASELINE,
    SET_DECREASES_CONDUCTANCE,
    SET_INCREASES_CONDUCTANCE,
    PositiveConductanceEmbedding,
    build_common_conductance_baseline,
    commission_positive_conductance_endpoints,
    load_positive_conductance_calibration,
)


SOURCE_SHA256 = "a" * 64
FIGURE_CROP_SHA256 = "c" * 64


def _paired_payload() -> dict:
    return {
        "schema": CALIBRATION_SCHEMA,
        "schema_version": 1,
        "calibration_id": "unit-test-positive-g-v1",
        "provenance": {
            "citation": "Synthetic unit-test fixture",
            "year": 2022,
            "figure": "Figure 6",
            "evidence_class": "published_figure_digitization",
            "extraction_method": "unit_test_values_not_device_evidence",
            "source_sha256": SOURCE_SHA256,
            "figure_crop_sha256": FIGURE_CROP_SHA256,
        },
        "conductance": {
            "unit": "uS",
            "direction": SET_INCREASES_CONDUCTANCE,
            "progress_coordinate": PROGRESS_COORDINATE,
            "representation": "paired_endpoints",
            "sampling_policy": "without_replacement",
            "reset": [10.0, 12.0, 14.0, 16.0],
            "set": [80.0, 90.0, 100.0, 110.0],
        },
    }


def _quantile_payload(*, pairing_policy: str = "independent") -> dict:
    payload = _paired_payload()
    payload["calibration_id"] = "unit-test-positive-g-quantiles-v1"
    payload["conductance"] = {
        "unit": "uS",
        "direction": SET_INCREASES_CONDUCTANCE,
        "progress_coordinate": PROGRESS_COORDINATE,
        "representation": EMPIRICAL_QUANTILES,
        "pairing_policy": pairing_policy,
        "reset": {
            "probabilities": [0.0, 0.5, 1.0],
            "values": [8.0, 10.0, 12.0],
        },
        "set": {
            "probabilities": [0.0, 0.5, 1.0],
            "values": [60.0, 80.0, 100.0],
        },
    }
    return payload


def _write_payload(tmp_path, payload: dict, *, name: str = "calibration.json"):
    path = tmp_path / name
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_loads_versioned_positive_physical_endpoints_and_records_hashes(
    tmp_path,
) -> None:
    path = _write_payload(tmp_path, _paired_payload())

    calibration = load_positive_conductance_calibration(path)

    assert calibration.artifact_sha256 == sha256_file(path)
    assert calibration.source_sha256 == SOURCE_SHA256
    assert calibration.report()["figure_crop_sha256"] == FIGURE_CROP_SHA256
    assert calibration.input_unit == "uS"
    assert torch.equal(
        calibration.reset_values_s,
        torch.tensor([10.0, 12.0, 14.0, 16.0], dtype=torch.float64) * 1e-6,
    )
    population = calibration.sample_population(devices=4, assignment_seed=19)
    replay = calibration.sample_population(devices=4, assignment_seed=19)
    assert torch.equal(population.reset_conductance_s, replay.reset_conductance_s)
    assert torch.equal(population.set_conductance_s, replay.set_conductance_s)
    assert population.population_sha256 == replay.population_sha256
    assert population.report["calibration_sha256"] == sha256_file(path)
    assert population.report["source_sha256"] == SOURCE_SHA256
    assert bool(torch.all(population.reset_conductance_s > 0.0))
    assert bool(
        torch.all(population.set_conductance_s > population.reset_conductance_s)
    )


def test_strict_loader_rejects_nonpositive_ambiguous_or_unknown_data(tmp_path) -> None:
    nonpositive = _paired_payload()
    nonpositive["conductance"]["reset"][0] = 0.0
    with pytest.raises(ValueError, match="strictly positive"):
        load_positive_conductance_calibration(
            _write_payload(tmp_path, nonpositive, name="nonpositive.json")
        )

    wrong_direction = _paired_payload()
    wrong_direction["conductance"]["direction"] = SET_DECREASES_CONDUCTANCE
    with pytest.raises(ValueError, match="violate the declared"):
        load_positive_conductance_calibration(
            _write_payload(tmp_path, wrong_direction, name="direction.json")
        )

    unknown = _paired_payload()
    unknown["conductance"]["implicit_normalization"] = "[-1,1]"
    with pytest.raises(ValueError, match="extra=.*implicit_normalization"):
        load_positive_conductance_calibration(
            _write_payload(tmp_path, unknown, name="unknown.json")
        )


def test_strict_loader_rejects_duplicate_json_keys(tmp_path) -> None:
    path = tmp_path / "duplicate.json"
    path.write_text(
        '{"schema":"x","schema":"y","schema_version":1}',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Duplicate JSON key"):
        load_positive_conductance_calibration(path)


def test_empirical_marginals_require_and_record_an_explicit_pairing_assumption(
    tmp_path,
) -> None:
    payload = _quantile_payload(pairing_policy="independent")
    path = _write_payload(tmp_path, payload)
    calibration = load_positive_conductance_calibration(path)

    first = calibration.sample_population(devices=32, assignment_seed=41)
    second = calibration.sample_population(devices=32, assignment_seed=41)

    assert calibration.pairing_policy == "independent"
    assert first.report["pairing_policy"] == "independent"
    assert torch.equal(first.reset_conductance_s, second.reset_conductance_s)
    assert torch.equal(first.set_conductance_s, second.set_conductance_s)
    assert bool(torch.all(first.reset_conductance_s >= 8e-6))
    assert bool(torch.all(first.reset_conductance_s <= 12e-6))
    assert bool(torch.all(first.set_conductance_s >= 60e-6))
    assert bool(torch.all(first.set_conductance_s <= 100e-6))

    missing = _quantile_payload()
    del missing["conductance"]["pairing_policy"]
    with pytest.raises(ValueError, match="missing=.*pairing_policy"):
        load_positive_conductance_calibration(
            _write_payload(tmp_path, missing, name="missing-pairing.json")
        )


def test_independent_marginals_must_guarantee_direction_for_every_pair(tmp_path) -> None:
    payload = _quantile_payload(pairing_policy="independent")
    payload["conductance"]["set"]["values"] = [11.0, 20.0, 30.0]
    with pytest.raises(ValueError, match="cannot guarantee"):
        load_positive_conductance_calibration(_write_payload(tmp_path, payload))


def test_progress_maps_to_full_positive_G_and_never_clips(tmp_path) -> None:
    calibration = load_positive_conductance_calibration(
        _write_payload(tmp_path, _paired_payload())
    )
    population = calibration.sample_population(devices=4, assignment_seed=3)
    embedding = PositiveConductanceEmbedding.from_population(population)

    assert torch.equal(
        embedding.to_full_conductance(torch.zeros(4, dtype=torch.float64)),
        population.reset_conductance_s,
    )
    assert torch.equal(
        embedding.to_full_conductance(torch.ones(4, dtype=torch.float64)),
        population.set_conductance_s,
    )
    midpoint = embedding.to_full_conductance(
        torch.full((4,), 0.5, dtype=torch.float64)
    )
    assert torch.allclose(
        midpoint,
        (population.reset_conductance_s + population.set_conductance_s) / 2.0,
        rtol=0.0,
        atol=1e-20,
    )
    assert embedding.report()["reference_subtraction"] is False
    assert embedding.report()["post_handoff_clipping"] is False
    with pytest.raises(ValueError, match="clipping is forbidden"):
        embedding.to_full_conductance(torch.tensor([-1e-12, 0.5, 0.5, 1.0]))
    with pytest.raises(ValueError, match="clipping is forbidden"):
        embedding.to_full_conductance(torch.tensor([0.0, 0.5, 0.5, 1.000001]))


def test_commissions_both_physical_endpoints_per_cell_with_provenance(
    tmp_path,
) -> None:
    calibration = load_positive_conductance_calibration(
        _write_payload(tmp_path, _paired_payload())
    )
    population = calibration.sample_population(devices=4, assignment_seed=11)
    reset_reads = torch.stack(
        (
            population.reset_conductance_s - 0.2e-6,
            population.reset_conductance_s,
            population.reset_conductance_s + 0.2e-6,
        )
    )
    set_reads = torch.stack(
        (
            population.set_conductance_s - 1e-6,
            population.set_conductance_s,
            population.set_conductance_s + 1e-6,
        )
    )

    commissioning = commission_positive_conductance_endpoints(
        population,
        reset_readings_s=reset_reads,
        set_readings_s=set_reads,
        commissioning_seed=701,
    )

    assert torch.allclose(
        commissioning.reset_mean_s,
        population.reset_conductance_s,
        rtol=0.0,
        atol=1e-18,
    )
    assert torch.allclose(
        commissioning.set_mean_s,
        population.set_conductance_s,
        rtol=0.0,
        atol=1e-18,
    )
    assert commissioning.report["projection"] == "none"
    assert commissioning.report["post_handoff_clipping"] is False
    assert commissioning.report["calibration_sha256"] == calibration.artifact_sha256
    assert commissioning.report["source_sha256"] == SOURCE_SHA256
    commissioned_embedding = PositiveConductanceEmbedding.from_commissioning(
        commissioning
    )
    assert torch.equal(
        commissioned_embedding.to_full_conductance(
            torch.zeros(4, dtype=torch.float64)
        ),
        commissioning.reset_mean_s,
    )


def _physical_quad(values_uS: list[float], *, layout: str) -> torch.Tensor:
    return scatter_quads(
        torch.tensor(values_uS, dtype=torch.float64).reshape(1, 1, 4) * 1e-6,
        shape=(2, 2),
        layout=layout,
    )


@pytest.mark.parametrize("layout", ["halves", "paired"])
def test_constructs_exact_pair_and_quad_reset_side_baselines_without_clipping(
    layout: str,
) -> None:
    reset = _physical_quad([10.0, 20.0, 15.0, 25.0], layout=layout)
    set_state = _physical_quad([80.0, 70.0, 75.0, 65.0], layout=layout)

    pair = build_common_conductance_baseline(
        reset,
        set_state,
        direction=SET_INCREASES_CONDUCTANCE,
        policy=DESTINATION_PAIR_BASELINE,
        layout=layout,
        calibration_sha256="b" * 64,
        source_sha256=SOURCE_SHA256,
    )
    quad = build_common_conductance_baseline(
        reset,
        set_state,
        direction=SET_INCREASES_CONDUCTANCE,
        policy=QUAD_BASELINE,
        layout=layout,
        calibration_sha256="b" * 64,
        source_sha256=SOURCE_SHA256,
    )

    assert torch.equal(
        quad_stack(pair.baseline_s, layout=layout),
        torch.tensor([15.0, 25.0, 15.0, 25.0], dtype=torch.float64).reshape(
            1, 1, 4
        )
        * 1e-6,
    )
    assert torch.allclose(
        quad_stack(quad.baseline_s, layout=layout),
        torch.full((1, 1, 4), 25e-6, dtype=torch.float64),
        rtol=0.0,
        atol=1e-20,
    )
    assert torch.equal(
        pair.group_set_frontier_s,
        torch.tensor([75.0, 65.0], dtype=torch.float64).reshape(1, 1, 2)
        * 1e-6,
    )
    assert pair.report["target_projection"] == "none"
    assert pair.report["post_handoff_clipping"] is False
    assert bool(torch.all(pair.baseline_s > 0.0))


def test_set_decreasing_direction_uses_the_correct_reset_side_frontier() -> None:
    reset = _physical_quad([100.0, 90.0, 80.0, 70.0], layout="halves")
    set_state = _physical_quad([20.0, 30.0, 25.0, 35.0], layout="halves")

    pair = build_common_conductance_baseline(
        reset,
        set_state,
        direction=SET_DECREASES_CONDUCTANCE,
        policy=DESTINATION_PAIR_BASELINE,
        layout="halves",
        baseline_position_fraction=0.25,
        calibration_sha256="b" * 64,
        source_sha256=SOURCE_SHA256,
    )

    # Pair frontiers are RESET=min([100,80])=80, SET=max([20,25])=25;
    # and RESET=min([90,70])=70, SET=max([30,35])=35.
    expected_groups = torch.tensor([66.25, 61.25], dtype=torch.float64).reshape(
        1, 1, 2
    ) * 1e-6
    assert torch.equal(
        pair.group_reset_frontier_s,
        torch.tensor([80.0, 70.0], dtype=torch.float64).reshape(1, 1, 2)
        * 1e-6,
    )
    assert torch.equal(
        pair.group_set_frontier_s,
        torch.tensor([25.0, 35.0], dtype=torch.float64).reshape(1, 1, 2)
        * 1e-6,
    )
    assert torch.equal(
        quad_stack(pair.baseline_s, layout="halves")[..., :2], expected_groups
    )


def test_empty_common_window_fails_instead_of_projecting_or_clipping() -> None:
    reset = _physical_quad([10.0, 10.0, 70.0, 10.0], layout="halves")
    set_state = _physical_quad([60.0, 80.0, 90.0, 70.0], layout="halves")
    # All individual cells are valid, but destination pair (0,2) has
    # max(RESET)=70 uS > min(SET)=60 uS.
    with pytest.raises(ValueError, match="no nonzero exact common"):
        build_common_conductance_baseline(
            reset,
            set_state,
            direction=SET_INCREASES_CONDUCTANCE,
            policy=DESTINATION_PAIR_BASELINE,
            layout="halves",
            calibration_sha256="b" * 64,
            source_sha256=SOURCE_SHA256,
        )
