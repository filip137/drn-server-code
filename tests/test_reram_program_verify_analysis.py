from __future__ import annotations

import json
from pathlib import Path
import sqlite3
import sys
from types import ModuleType

import pytest
import torch

from experiments.reram_program_verify.analysis import (
    build_empirical_kernel,
    build_wan_comparison,
    fit_bounded_uniform_models,
    fit_gaussian_surrogates,
)
from experiments.reram_program_verify.integrity import validate_trajectory_database
from experiments.reram_program_verify.storage import TrajectoryStore


def _begin_row(
    *,
    device_id: int,
    partition: str,
    target: float = 0.5,
    target_index: int = 0,
    corrupt: bool = False,
    conditioned: float = 0.5,
    sampled_lower: float = 0.0,
    sampled_upper: float = 1.0,
) -> dict[str, object]:
    return {
        "preset": "reram_array_om",
        "corrupt_population": int(corrupt),
        "controller": "one_pulse",
        "start_protocol": "lower_to_target",
        "tolerance_ratio": 0.5,
        "tolerance": 0.01,
        "target_index": target_index,
        "target": target,
        "device_id": device_id,
        "repeat_id": 0,
        "partition_name": partition,
        "construction_seed": 100 + device_id,
        "repeat_seed": 200 + device_id,
        "conditioning_seed": 300 + device_id,
        "pulse_seed": 400 + device_id,
        "controller_seed": None,
        "corrupt": int(corrupt),
        "conditioning_success": 1,
        "conditioning_pulses": 3,
        "conditioned_apparent": conditioned,
        "conditioned_persistent": conditioned,
        "sampled_lower_persistent": sampled_lower,
        "sampled_upper_persistent": sampled_upper,
    }


def _finish_row(
    trajectory_id: int,
    *,
    target: float = 0.5,
    endpoint: float = 0.5,
    set_count: int = 0,
    verify_count: int = 1,
    saturated: bool = False,
) -> dict[str, object]:
    return {
        "trajectory_id": trajectory_id,
        "accepted": int(abs(endpoint - target) <= 0.01),
        "initialization_failed": 0,
        "nonfinite": 0,
        "budget_exhausted": 0,
        "saturated": int(saturated),
        "endpoint_apparent": endpoint,
        "endpoint_persistent": endpoint,
        "residual_apparent": endpoint - target,
        "residual_persistent": endpoint - target,
        "set_count": set_count,
        "reset_count": 0,
        "total_pulses": set_count,
        "verify_count": verify_count,
        "reversals": 0,
    }


def _write_valid_ledger(path: Path) -> None:
    with TrajectoryStore(path) as store:
        ids = store.begin_trajectories(
            [
                _begin_row(device_id=0, partition="fit", conditioned=0.5),
                _begin_row(device_id=1, partition="validation", conditioned=0.4),
            ]
        )
        store.insert_events(
            [
                (ids[0], 0, 0.5, 0.5, 0.5, 0, 0, 0, 0),
                (ids[1], 0, 0.4, 0.4, 0.4, 0, 0, 0, 0),
                (ids[1], 1, 0.4, 0.5, 0.5, 1, 1, 1, 1),
            ]
        )
        store.finish_trajectories(
            [
                _finish_row(ids[0]),
                _finish_row(ids[1], endpoint=0.5, set_count=1, verify_count=2),
            ]
        )


def test_integrity_report_proves_persisted_event_and_trajectory_accounting(
    tmp_path: Path,
) -> None:
    database = tmp_path / "trajectories.sqlite3"
    report = tmp_path / "integrity.json"
    _write_valid_ledger(database)

    artifact = validate_trajectory_database(
        database,
        output_path=report,
        expected_trajectory_count=2,
        expected_condition_count=1,
        trajectories_per_condition=2,
        maximum_program_pulses=4,
        controllers=("one_pulse",),
        start_protocols=("lower_to_target",),
        tolerance_ratios=(0.5,),
        target_points=1,
        repeats_per_device=1,
        expected_partition_device_counts={"fit": 1, "validation": 1},
    )

    assert artifact["passed"] is True
    assert artifact["summary"] == {
        "trajectory_count": 2,
        "verify_event_count": 3,
        "maximum_program_pulses": 4,
        "conditioning_scope": "per_target",
    }
    assert all(record["passed"] for record in artifact["checks"])
    assert json.loads(report.read_text(encoding="utf-8")) == artifact


def test_integrity_report_fails_closed_on_corrupt_pulse_accounting(
    tmp_path: Path,
) -> None:
    database = tmp_path / "trajectories.sqlite3"
    report = tmp_path / "integrity.json"
    _write_valid_ledger(database)
    connection = sqlite3.connect(database)
    connection.execute("UPDATE trajectories SET set_count=7 WHERE trajectory_id=2")
    connection.commit()
    connection.close()

    with pytest.raises(RuntimeError, match="trajectory_pulse_accounting"):
        validate_trajectory_database(
            database,
            output_path=report,
            expected_trajectory_count=2,
            expected_condition_count=1,
            trajectories_per_condition=2,
            maximum_program_pulses=4,
            controllers=("one_pulse",),
            start_protocols=("lower_to_target",),
            tolerance_ratios=(0.5,),
            target_points=1,
            repeats_per_device=1,
            expected_partition_device_counts={"fit": 1, "validation": 1},
        )
    artifact = json.loads(report.read_text(encoding="utf-8"))
    assert artifact["passed"] is False


def test_integrity_report_accepts_controller_specific_condition_counts(
    tmp_path: Path,
) -> None:
    database = tmp_path / "focused.sqlite3"
    report = tmp_path / "focused-integrity.json"
    with TrajectoryStore(database) as store:
        rows = [
            _begin_row(device_id=0, partition="calibration"),
            {
                **_begin_row(device_id=0, partition="calibration"),
                "controller": "adaptive",
            },
            {
                **_begin_row(device_id=1, partition="fit"),
                "controller": "adaptive",
            },
        ]
        ids = store.begin_trajectories(rows)
        store.insert_events(
            [(trajectory_id, 0, 0.5, 0.5, 0.5, 0, 0, 0, 0) for trajectory_id in ids]
        )
        store.finish_trajectories([_finish_row(trajectory_id) for trajectory_id in ids])

    artifact = validate_trajectory_database(
        database,
        output_path=report,
        expected_trajectory_count=3,
        expected_condition_count=2,
        trajectories_per_condition=2,
        maximum_program_pulses=128,
        controllers=("one_pulse", "adaptive"),
        start_protocols=("lower_to_target",),
        tolerance_ratios=(0.5,),
        target_points=1,
        repeats_per_device=1,
        expected_partition_device_counts={"calibration": 1, "fit": 1},
        trajectories_per_controller={"one_pulse": 1, "adaptive": 2},
    )

    checks = {item["name"]: item for item in artifact["checks"]}
    assert artifact["passed"] is True
    assert checks["condition_coverage"]["observed"] == {
        "groups": 2,
        "minimum_trajectories": 1,
        "maximum_trajectories": 2,
    }
    assert checks["controller_specific_condition_counts"]["observed"] == []


def test_integrity_report_fails_closed_when_healthy_sampled_bounds_collapse(
    tmp_path: Path,
) -> None:
    database = tmp_path / "changed-bounds.sqlite3"
    report = tmp_path / "changed-bounds-integrity.json"
    _write_valid_ledger(database)
    connection = sqlite3.connect(database)
    connection.execute(
        "UPDATE trajectories SET sampled_lower_persistent=1.0 "
        "WHERE trajectory_id=1"
    )
    connection.commit()
    connection.close()

    with pytest.raises(RuntimeError, match="sampled_device_bound_invariants"):
        validate_trajectory_database(
            database,
            output_path=report,
            expected_trajectory_count=2,
            expected_condition_count=1,
            trajectories_per_condition=2,
            maximum_program_pulses=4,
            controllers=("one_pulse",),
            start_protocols=("lower_to_target",),
            tolerance_ratios=(0.5,),
            target_points=1,
            repeats_per_device=1,
            expected_partition_device_counts={"fit": 1, "validation": 1},
        )


def test_integrity_report_proves_blocked_conditioning_reuse_across_targets(
    tmp_path: Path,
) -> None:
    database = tmp_path / "blocked.sqlite3"
    report = tmp_path / "blocked-integrity.json"
    with TrajectoryStore(database) as store:
        rows = [
            _begin_row(
                device_id=0,
                partition="validation",
                target=0.4,
                target_index=index,
                conditioned=0.4,
            )
            for index in range(2)
        ]
        ids = store.begin_trajectories(rows)
        store.insert_events(
            [
                (trajectory_id, 0, 0.4, 0.4, 0.4, 0, 0, 0, 0)
                for trajectory_id in ids
            ]
        )
        store.finish_trajectories(
            [
                _finish_row(
                    trajectory_id,
                    target=0.4,
                    endpoint=0.4,
                    verify_count=1,
                )
                for trajectory_id in ids
            ]
        )

    artifact = validate_trajectory_database(
        database,
        output_path=report,
        expected_trajectory_count=2,
        expected_condition_count=2,
        trajectories_per_condition=1,
        maximum_program_pulses=4,
        controllers=("one_pulse",),
        start_protocols=("lower_to_target",),
        tolerance_ratios=(0.5,),
        target_points=2,
        repeats_per_device=1,
        expected_partition_device_counts={"validation": 1},
        conditioning_scope="per_start",
    )

    check = next(
        record
        for record in artifact["checks"]
        if record["name"] == "blocked_conditioning_reused_across_targets"
    )
    assert check["passed"] is True


def test_integrity_report_preserves_a_nonfinite_terminal_verify_as_sql_null(
    tmp_path: Path,
) -> None:
    database = tmp_path / "nonfinite.sqlite3"
    report = tmp_path / "integrity.json"
    with TrajectoryStore(database) as store:
        trajectory_id = store.begin_trajectories(
            [_begin_row(device_id=0, partition="validation", conditioned=0.4)]
        )[0]
        store.insert_events(
            [
                (trajectory_id, 0, 0.4, 0.4, 0.4, 0, 0, 0, 0),
                (trajectory_id, 1, 0.4, None, 0.41, 1, 1, 1, 1),
            ]
        )
        store.finish_trajectories(
            [
                {
                    "trajectory_id": trajectory_id,
                    "accepted": 0,
                    "initialization_failed": 0,
                    "nonfinite": 1,
                    "budget_exhausted": 0,
                    "saturated": 0,
                    "endpoint_apparent": None,
                    "endpoint_persistent": 0.41,
                    "residual_apparent": None,
                    "residual_persistent": -0.09,
                    "set_count": 1,
                    "reset_count": 0,
                    "total_pulses": 1,
                    "verify_count": 2,
                    "reversals": 0,
                }
            ]
        )

    artifact = validate_trajectory_database(
        database,
        output_path=report,
        expected_trajectory_count=1,
        expected_condition_count=1,
        trajectories_per_condition=1,
        maximum_program_pulses=4,
        controllers=("one_pulse",),
        start_protocols=("lower_to_target",),
        tolerance_ratios=(0.5,),
        target_points=1,
        repeats_per_device=1,
        expected_partition_device_counts={"validation": 1},
    )

    assert artifact["passed"] is True


def test_empirical_kernel_keeps_identity_partitions_and_corrupt_outcomes_separate(
    tmp_path: Path,
) -> None:
    database = tmp_path / "trajectories.sqlite3"
    output = tmp_path / "kernel.json"
    partitions = ("calibration", "fit", "validation", "validation")
    with TrajectoryStore(database) as store:
        ids = store.begin_trajectories(
            [
                _begin_row(
                    device_id=index,
                    partition=partition,
                    corrupt=index == 3,
                )
                for index, partition in enumerate(partitions)
            ]
        )
        store.finish_trajectories(
            [
                _finish_row(
                    trajectory_id,
                    endpoint=0.5 + 0.001 * index,
                    saturated=index == 3,
                )
                for index, trajectory_id in enumerate(ids)
            ]
        )

    artifact = build_empirical_kernel(
        database,
        output_path=output,
        metadata={"evidence_class": "operational_smoke"},
    )
    target = next(iter(artifact["conditions"].values()))["bins"][0]

    assert artifact["claim_class"] == "operational_smoke"
    assert {
        name: summary["trajectory_count"]
        for name, summary in target["partitions"].items()
    } == {"calibration": 1, "fit": 1, "validation": 2}
    assert target["partitions"]["validation"]["apparent_residual"]["count"] == 1
    assert target["population_breakdown"]["corrupt"]["trajectory_count"] == 1
    assert target["population_breakdown"]["corrupt"]["behavior_classes"] == {
        "accepted_at_stuck_state": 1
    }
    assert target["apparent_residual"]["count"] == 3


def test_gaussian_fit_and_validation_use_disjoint_identity_partitions(
    tmp_path: Path,
) -> None:
    database = tmp_path / "trajectories.sqlite3"
    output = tmp_path / "fit.json"
    targets = [index / 4 for index in range(5)]
    with TrajectoryStore(database) as store:
        begin_rows = []
        descriptors = []
        for target_index, target in enumerate(targets):
            for device_id, partition, residual in (
                (0, "calibration", -0.009),
                (1, "fit", 0.005),
                (2, "validation", 0.002),
            ):
                begin_rows.append(
                    _begin_row(
                        device_id=device_id,
                        partition=partition,
                        target=target,
                        target_index=target_index,
                    )
                )
                descriptors.append((target, residual))
        ids = store.begin_trajectories(begin_rows)
        store.finish_trajectories(
            [
                _finish_row(
                    trajectory_id,
                    target=target,
                    endpoint=target + residual,
                )
                for trajectory_id, (target, residual) in zip(ids, descriptors)
            ]
        )

    artifact = fit_gaussian_surrogates(
        database,
        output_path=output,
        polynomial_order=4,
        standard_deviation_floor=1e-8,
        analysis_seed=17,
        metadata={"evidence_class": "test"},
    )
    condition = next(iter(artifact["conditions"].values()))

    assert condition["fit_status"] == "fit"
    assert condition["mu_coefficients"][0] == pytest.approx(0.005, abs=1e-7)
    assert condition["validation"]["accepted_noncorrupt_count"] == 5
    assert [
        record["held_out_residual"]["count"]
        for record in condition["validation"]["per_target"]
    ] == [1, 1, 1, 1, 1]
    assert condition["validation"]["residual"]["bias"] == pytest.approx(0.002)


def test_bounded_uniform_model_fits_successes_and_separates_outcomes(
    tmp_path: Path,
) -> None:
    database = tmp_path / "trajectories.sqlite3"
    output = tmp_path / "bounded-uniform.json"
    fit_residuals = [-0.01 + (index + 0.5) * 0.02 / 80 for index in range(80)]
    validation_residuals = [
        -0.01 + (index + 0.5) * 0.02 / 100 for index in range(100)
    ]
    descriptors: list[tuple[float, bool]] = []
    with TrajectoryStore(database) as store:
        begin_rows = []
        for device_id, residual in enumerate(fit_residuals):
            begin_rows.append(_begin_row(device_id=device_id, partition="fit"))
            descriptors.append((0.5 + residual, False))
        for offset, residual in enumerate(validation_residuals, start=100):
            begin_rows.append(_begin_row(device_id=offset, partition="validation"))
            descriptors.append((0.5 + residual, False))
        begin_rows.append(_begin_row(device_id=300, partition="fit"))
        descriptors.append((0.8, False))
        begin_rows.append(
            _begin_row(device_id=301, partition="fit", corrupt=True)
        )
        descriptors.append((0.503, True))
        # The structural reachability model pairs the independently
        # conditioned persistent RESET and SET states for each identity.
        lower_rows = list(begin_rows)
        lower_descriptors = list(descriptors)
        for row, descriptor in zip(lower_rows, lower_descriptors):
            begin_rows.append(
                {
                    **row,
                    "start_protocol": "upper_to_target",
                    "conditioned_apparent": 0.9,
                    "conditioned_persistent": 0.9,
                }
            )
            descriptors.append(descriptor)
        ids = store.begin_trajectories(begin_rows)
        store.finish_trajectories(
            [
                _finish_row(trajectory_id, endpoint=endpoint)
                for trajectory_id, (endpoint, _) in zip(ids, descriptors)
            ]
        )

    artifact = fit_bounded_uniform_models(
        database,
        output_path=output,
        metadata={"evidence_class": "test"},
    )
    condition = next(iter(artifact["conditions"].values()))
    target = condition["validation"]["per_target"][0]
    histogram = target["accepted_noncorrupt_residual"]
    outcomes = target["outcome_model"]

    assert artifact["target_support"]["outside_support"].startswith("reject")
    assert histogram["support"] == [-0.01, 0.01]
    assert histogram["fit_count"] == 80
    assert sum(histogram["bin_probabilities"]) == pytest.approx(1.0)
    assert histogram["bin_probabilities"] == pytest.approx([0.125] * 8)
    assert outcomes["failed_noncorrupt_terminal"]["count"] == 1
    assert outcomes["failed_noncorrupt_terminal"]["apparent_endpoint"][
        "values"
    ] == pytest.approx([0.8] * 11)
    assert outcomes["corrupt_terminal"]["count"] == 1
    assert outcomes["noncorrupt_reachability"]["available"] is True
    assert outcomes["noncorrupt_reachability"]["classes"][
        "target_inside_bounds"
    ]["probability"] == 1.0
    assert condition["reachability_fit_status"] == "fit"
    assert condition["validation"]["piecewise_uniform"]["adequate"] is True
    assert condition["validation"]["acceptance_uniform_baseline"][
        "adequate"
    ] is True
    assert condition["adequate"] is True
    assert json.loads(output.read_text(encoding="utf-8")) == artifact


def test_bounded_model_separates_lower_inside_and_upper_reachability(
    tmp_path: Path,
) -> None:
    database = tmp_path / "reachability.sqlite3"
    output = tmp_path / "reachability.json"
    # target=0.5 is respectively below, inside, and above these paired
    # persistent boundary intervals.
    intervals = ((0.6, 0.9, 0.6), (0.1, 0.9, 0.5), (0.1, 0.4, 0.4))
    with TrajectoryStore(database) as store:
        begin_rows = []
        endpoints = []
        for partition, offset in (("fit", 0), ("validation", 10)):
            for index, (lower, upper, endpoint) in enumerate(intervals):
                device_id = offset + index
                begin_rows.extend(
                    [
                        _begin_row(
                            device_id=device_id,
                            partition=partition,
                            conditioned=lower,
                        ),
                        {
                            **_begin_row(
                                device_id=device_id,
                                partition=partition,
                                conditioned=upper,
                            ),
                            "start_protocol": "upper_to_target",
                        },
                    ]
                )
                endpoints.extend((endpoint, endpoint))
        ids = store.begin_trajectories(begin_rows)
        store.finish_trajectories(
            [
                _finish_row(trajectory_id, endpoint=endpoint)
                for trajectory_id, endpoint in zip(ids, endpoints)
            ]
        )

    artifact = fit_bounded_uniform_models(
        database,
        output_path=output,
        metadata={"evidence_class": "test"},
    )
    condition = artifact["conditions"][
        "one_pulse__lower_to_target__tau_step_0.5"
    ]
    reachability = condition["validation"]["per_target"][0]["outcome_model"][
        "noncorrupt_reachability"
    ]
    classes = reachability["classes"]

    assert reachability["classified_noncorrupt_count"] == 3
    assert reachability["missing_paired_boundary_count"] == 0
    for reachability_class in (
        "target_below_lower_bound",
        "target_inside_bounds",
        "target_above_upper_bound",
    ):
        assert classes[reachability_class]["probability"] == pytest.approx(1 / 3)
    assert classes["target_below_lower_bound"]["success_probability"] == 0.0
    assert classes["target_inside_bounds"]["success_probability"] == 1.0
    assert classes["target_above_upper_bound"]["success_probability"] == 0.0
    assert classes["target_below_lower_bound"][
        "acceptance_window_reachable_probability"
    ] == 0.0
    assert classes["target_inside_bounds"][
        "acceptance_window_reachable_probability"
    ] == 1.0
    assert classes["target_above_upper_bound"][
        "acceptance_window_reachable_probability"
    ] == 0.0


def test_bounded_model_uses_sampled_bounds_for_lower_only_hwa_profile(
    tmp_path: Path,
) -> None:
    database = tmp_path / "sampled-reachability.sqlite3"
    output = tmp_path / "sampled-reachability.json"
    intervals = ((0.6, 0.9, 0.6), (0.1, 0.9, 0.5), (0.1, 0.4, 0.4))
    with TrajectoryStore(database) as store:
        begin_rows = []
        endpoints = []
        for partition, offset in (("fit", 0), ("validation", 10)):
            for index, (lower, upper, endpoint) in enumerate(intervals):
                begin_rows.append(
                    _begin_row(
                        device_id=offset + index,
                        partition=partition,
                        conditioned=lower,
                        sampled_lower=lower,
                        sampled_upper=upper,
                    )
                )
                endpoints.append(endpoint)
        ids = store.begin_trajectories(begin_rows)
        store.finish_trajectories(
            [
                _finish_row(trajectory_id, endpoint=endpoint)
                for trajectory_id, endpoint in zip(ids, endpoints)
            ]
        )

    artifact = fit_bounded_uniform_models(
        database,
        output_path=output,
        metadata={"evidence_class": "test"},
    )
    condition = artifact["conditions"][
        "one_pulse__lower_to_target__tau_step_0.5"
    ]
    reachability = condition["validation"]["per_target"][0]["outcome_model"][
        "noncorrupt_reachability"
    ]

    assert condition["reachability_fit_status"] == "fit"
    assert reachability["classified_noncorrupt_count"] == 3
    assert reachability["missing_paired_boundary_count"] == 0
    assert {
        name: row["probability"]
        for name, row in reachability["classes"].items()
    } == pytest.approx(
        {
            "target_below_lower_bound": 1 / 3,
            "target_inside_bounds": 1 / 3,
            "target_above_upper_bound": 1 / 3,
        }
    )


def test_wan_comparison_records_full_statistics_cost_and_conditioning_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    aihwkit = ModuleType("aihwkit")
    aihwkit.__version__ = "test"  # type: ignore[attr-defined]
    inference = ModuleType("aihwkit.inference")
    noise = ModuleType("aihwkit.inference.noise")
    reram = ModuleType("aihwkit.inference.noise.reram")

    class FakeWan:
        coeff_dic = {1.0: [0.0, 0.0, 0.0, 0.0, 0.0]}
        coeff_g_max_reference = 40.0

        def __init__(self, *, g_max: float, noise_scale: float) -> None:
            self.g_max = g_max
            self.noise_scale = noise_scale

        def apply_programming_noise_to_conductance(
            self, conductance: torch.Tensor
        ) -> torch.Tensor:
            return conductance.clone()

    reram.ReRamWan2022NoiseModel = FakeWan  # type: ignore[attr-defined]
    for name, module in (
        ("aihwkit", aihwkit),
        ("aihwkit.inference", inference),
        ("aihwkit.inference.noise", noise),
        ("aihwkit.inference.noise.reram", reram),
    ):
        monkeypatch.setitem(sys.modules, name, module)

    empty_cost = {
        name: {"count": 2, "mean": 1.0, "median": 1.0, "p95": 1.0, "maximum": 1}
        for name in (
            "set_pulses",
            "reset_pulses",
            "total_pulses",
            "verify_reads",
            "reversals",
        )
    }
    outcome = {
        "trajectory_count": 2,
        "success_count": 2,
        "success_probability": 1.0,
        "failure_probability": 0.0,
        "failure_classes": {},
        "behavior_classes": {"accepted": 2},
        "saturated_fraction": 0.0,
        "cost": empty_cost,
    }
    per_target = []
    for index, target in enumerate((0.0, 1.0)):
        per_target.append(
            {
                "target_index": index,
                "target": target,
                "held_out_residual": {
                    "count": 2,
                    "bias": 0.0,
                    "standard_deviation": 0.0,
                    "mae": 0.0,
                    "rmse": 0.0,
                    "quantiles": {"q01": 0.0, "q05": 0.0, "q50": 0.0, "q95": 0.0, "q99": 0.0},
                    "skewness": None,
                    "excess_kurtosis": None,
                },
                "absolute_error_exceedance": {},
                "endpoint_outside_0_1": {
                    "below_zero_fraction": 0.0,
                    "above_one_fraction": 0.0,
                },
                "noncorrupt_success_count": 2,
                "trajectory_count": 2,
                "success_count": 2,
                "failure_probability": 0.0,
                "failure_classes": {},
                "saturated_fraction": 0.0,
                "cost": empty_cost,
                "population_breakdown": {
                    "all": outcome,
                    "noncorrupt": outcome,
                    "corrupt": {**outcome, "trajectory_count": 0, "success_count": 0},
                },
            }
        )
    fit = {"conditions": {"condition": {"validation": {"per_target": per_target}}}}

    artifact = build_wan_comparison(
        targets=(0.0, 1.0),
        samples_per_target=2,
        g_max_us=40.0,
        noise_scale=1.0,
        wan_seed=5,
        fit_artifact=fit,
        output_path=tmp_path / "wan.json",
    )

    comparison = artifact["ibm_condition_comparisons"]["condition"][0]
    assert comparison["ibm_population_outcomes"]["cost"]["set_pulses"]["mean"] == 1.0
    assert comparison["ibm_success_conditioned_noncorrupt"]["sample_count"] == 2
    assert comparison["wan_raw_one_second"]["error"]["skewness"] is None
    assert artifact["wan_cost_and_convergence"]["program_and_verify_cost_available"] is False
    assert artifact["sampling"]["base_seed"] == 5
    assert artifact["normalization"].startswith("G_target=40*x")
