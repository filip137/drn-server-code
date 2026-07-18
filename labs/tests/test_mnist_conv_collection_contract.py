from __future__ import annotations

import json

import pytest

from experiments.mnist_conv.backend import ExecutionContext
from experiments.mnist_conv.cli import main
from experiments.mnist_conv.collection import (
    CollectionBusyError,
    _acquire_lock,
    _release_lock,
    collect_sweep,
)
from experiments.mnist_conv.layout import ResultLayout
from experiments.mnist_conv.manifest import publish_manifest
from experiments.mnist_conv.runner import execute_run
from experiments.mnist_conv.specs import RunSpec, SweepSpec
from labs.tests.test_mnist_conv_canonical_v1 import (
    FakeBackend,
    PROVENANCE,
    run_value,
    sweep_value,
)


def test_collection_groups_each_declared_axis_before_computing_seed_coverage(tmp_path):
    value = sweep_value(run_value(epochs=1), seeds=(0, 1))
    lr_pointer = "/run/training/learning_rate/0"
    value["axes"].append({"path": lr_pointer, "values": [0.01, 0.02]})
    value["varying_fields"].append(lr_pointer)
    value["collection"]["group_by"] = [lr_pointer]
    sweep = SweepSpec.from_dict(value)
    layout = ResultLayout(tmp_path / "results")
    manifest, manifest_path = publish_manifest(sweep, layout, PROVENANCE)

    backend = FakeBackend()
    for entry in manifest["entries"]:
        if entry["run_spec"]["seed"] == 0:
            execute_run(
                RunSpec.from_dict(entry["run_spec"]),
                ExecutionContext(tmp_path / "data", device="cpu"),
                layout,
                PROVENANCE,
                backend=backend,
            )

    collection = collect_sweep(manifest_path, layout, allow_incomplete=True)
    assert collection["complete"] is False
    assert len(collection["cases"]) == 2
    assert {
        json.loads(group["group_values"])[lr_pointer]
        for group in collection["cases"]
    } == {0.01, 0.02}
    for group in collection["cases"]:
        assert json.loads(group["complete_seeds"]) == [0]
        assert json.loads(group["missing_seeds"]) == [1]
        assert group["complete"] is False
    assert (manifest_path.parent / "summary.partial.csv").is_file()
    assert not (manifest_path.parent / "summary.csv").exists()


def test_group_seed_is_complete_only_when_every_omitted_axis_job_is_complete(tmp_path):
    value = sweep_value(run_value(epochs=1), seeds=(0, 1))
    lr_pointer = "/run/training/learning_rate/0"
    value["axes"].append({"path": lr_pointer, "values": [0.01, 0.02]})
    value["varying_fields"].append(lr_pointer)
    # Deliberately omit LR from group_by: each grouped seed now owns two jobs.
    value["collection"]["group_by"] = []
    sweep = SweepSpec.from_dict(value)
    layout = ResultLayout(tmp_path / "results")
    manifest, manifest_path = publish_manifest(sweep, layout, PROVENANCE)

    backend = FakeBackend()
    for entry in manifest["entries"]:
        spec = RunSpec.from_dict(entry["run_spec"])
        lr = spec.data["run"]["training"]["learning_rate"][0]
        if spec.data["seed"] == 1 or lr == 0.01:
            execute_run(
                spec,
                ExecutionContext(tmp_path / "data", device="cpu"),
                layout,
                PROVENANCE,
                backend=backend,
            )

    collection = collect_sweep(manifest_path, layout, allow_incomplete=True)
    assert collection["complete"] is False
    assert collection["coverage"] == {
        "complete": 3,
        "missing": 1,
        "running": 0,
        "failed": 0,
        "pruned": 0,
        "stale": 0,
        "invalid": 0,
    }
    assert len(collection["cases"]) == 1
    group = collection["cases"][0]
    assert json.loads(group["complete_seeds"]) == [1]
    assert json.loads(group["missing_seeds"]) == [0]
    assert group["complete"] is False
    assert group["run_count"] == 4
    assert group["complete_count"] == 3
    assert collection["coverage_checks"]["complete_required_cases_and_seeds"] is False


def test_default_collection_exits_nonzero_but_monitoring_mode_returns_partial(tmp_path, capsys):
    layout = ResultLayout(tmp_path / "results")
    sweep = SweepSpec.from_dict(sweep_value(run_value(epochs=1), seeds=(0, 1)))
    _, manifest_path = publish_manifest(sweep, layout, PROVENANCE)

    with pytest.raises(SystemExit) as failure:
        main(["collect", "--sweep", str(manifest_path.parent)])
    assert failure.value.code == 2
    assert (manifest_path.parent / "summary.partial.csv").is_file()
    assert not (manifest_path.parent / "summary.csv").exists()
    capsys.readouterr()

    assert main([
        "collect", "--sweep", str(manifest_path.parent), "--allow-incomplete",
    ]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["complete"] is False
    assert result["coverage"]["missing"] == 2


def test_collection_lock_recovers_after_crash_but_rejects_a_live_second_writer(tmp_path):
    layout = ResultLayout(tmp_path / "results")
    sweep = SweepSpec.from_dict(sweep_value(run_value(epochs=1), seeds=(0,)))
    _, manifest_path = publish_manifest(sweep, layout, PROVENANCE)
    lock = manifest_path.parent / ".collect.lock"
    lock.write_text('{"pid": 1, "created_at": "2000-01-01T00:00:00Z"}\n')

    # A leftover file has no live kernel lock and is safely reclaimed.
    collection = collect_sweep(manifest_path, layout, allow_incomplete=True)
    assert collection["complete"] is False
    assert not lock.exists()

    held = _acquire_lock(lock)
    try:
        with pytest.raises(CollectionBusyError, match="at most one sweep collector"):
            collect_sweep(manifest_path, layout, allow_incomplete=True)
        assert lock.is_file()
    finally:
        _release_lock(lock, held)


def test_amplification_grid_cannot_be_combined_across_architectures(tmp_path):
    base = run_value(epochs=1)
    cases = [
        {
            "id": "conv1-v1-c1",
            "set": {
                "/run/model/voltage_amp": 1.0,
                "/run/model/current_amp": 1.0,
                "/run/calibration/calibration_id": "conv1-v1-c1",
            },
        },
        {
            "id": "conv2-v4-c1",
            "set": {
                "/run/architecture/profile": "conv2",
                "/run/architecture/channels": [64, 128],
                "/run/architecture/kernel_sizes": [3, 3],
                "/run/architecture/strides": [2, 2],
                "/run/architecture/paddings": [1, 1],
                "/run/model/weight_gains": [1.0, 1.0, 1.0],
                "/run/model/voltage_amp": 4.0,
                "/run/model/current_amp": 1.0,
                "/run/training/learning_rate": [0.01] * 5,
                "/run/calibration/calibration_id": "conv2-v4-c1",
                "/run/calibration/layer_measurements": [
                    {"layer_index": 1, "measured_saturation": 0.3},
                    {"layer_index": 2, "measured_saturation": 0.31},
                ],
            },
        },
        {
            "id": "conv1-v4-c0p25",
            "set": {
                "/run/model/voltage_amp": 4.0,
                "/run/model/current_amp": 0.25,
                "/run/calibration/calibration_id": "conv1-v4-c0p25",
            },
        },
    ]
    varying = sorted({pointer for case in cases for pointer in case["set"]})
    sweep = SweepSpec.from_dict({
        "schema_version": "mnist-conv-sweep/v1",
        "name": "mixed-architecture-grid",
        "base_run": base,
        "axes": [],
        "cases": cases,
        "varying_fields": varying,
        "collection": {
            "expected_seeds": [0],
            "required_cases": [case["id"] for case in cases],
            "group_by": [
                "/run/architecture/profile",
                "/run/model/non_linearity",
                "/run/model/voltage_amp",
                "/run/model/current_amp",
            ],
        },
    })
    layout = ResultLayout(tmp_path / "results")
    manifest, manifest_path = publish_manifest(sweep, layout, PROVENANCE)
    backend = FakeBackend()
    for entry in manifest["entries"]:
        execute_run(
            RunSpec.from_dict(entry["run_spec"]),
            ExecutionContext(tmp_path / "data", device="cpu"),
            layout,
            PROVENANCE,
            backend=backend,
        )

    collection = collect_sweep(manifest_path, layout)
    assert collection["complete"] is True
    assert collection["coverage_checks"]["approved_amplification_grid"] is False
    assert all(
        "amplification_grid_incomplete" in json.loads(row["eligibility_reasons"])
        for row in collection["rows"]
    )
    assert all(
        "comparison_architecture_mismatch" in json.loads(row["eligibility_reasons"])
        for row in collection["rows"]
    )


@pytest.mark.parametrize(
    ("pointer", "replacement", "reason"),
    [
        (
            "/run/dataset/normalization/scale",
            0.2,
            "comparison_preprocessing_mismatch",
        ),
        ("/run/dataset/batch_size", 32, "comparison_batch_size_mismatch"),
        ("/run/training/epochs", 2, "comparison_epoch_budget_mismatch"),
        (
            "/run/solver/inference_iterations",
            2,
            "comparison_solver_tk_mismatch",
        ),
        (
            "/run/solver/minimizer/overrelaxation_factor",
            1.2,
            "comparison_minimizer_mismatch",
        ),
        ("/run/training/lr_decay", 0.75, "comparison_lr_policy_mismatch"),
        (
            "/run/calibration/target_initial_saturation",
            0.25,
            "comparison_calibration_protocol_mismatch",
        ),
        (
            "/run/model/weight_gains/0",
            2.0,
            "comparison_model_initialization_mismatch",
        ),
    ],
)
def test_mixed_amplification_case_protocols_are_never_paper_eligible(
    tmp_path,
    pointer,
    replacement,
    reason,
):
    value = sweep_value(run_value(epochs=1), seeds=(0, 1), three_cases=True)
    value["cases"][1]["set"][pointer] = replacement
    value["varying_fields"].append(pointer)
    sweep = SweepSpec.from_dict(value)
    layout = ResultLayout(tmp_path / "results")
    _, manifest_path = publish_manifest(sweep, layout, PROVENANCE)

    collection = collect_sweep(manifest_path, layout, allow_incomplete=True)
    assert collection["paper_eligible"] is False
    assert collection["coverage_checks"]["fixed_comparison_contract"] is False
    assert all(
        reason in json.loads(row["eligibility_reasons"])
        for row in collection["rows"]
    )


def test_case_linked_learning_rates_and_calibration_measurements_may_differ(tmp_path):
    value = sweep_value(run_value(epochs=1), seeds=(0, 1), three_cases=True)
    allowed_overrides = [
        ("/run/model/input_gain", [1.0, 2.0, 3.0]),
        ("/run/training/learning_rate", [[0.01] * 3, [0.02] * 3, [0.03] * 3]),
        ("/run/calibration/measured_initial_saturation", [0.30, 0.31, 0.32]),
        (
            "/run/calibration/layer_measurements",
            [
                [{"layer_index": 1, "measured_saturation": 0.30}],
                [{"layer_index": 1, "measured_saturation": 0.31}],
                [{"layer_index": 1, "measured_saturation": 0.32}],
            ],
        ),
    ]
    for pointer, replacements in allowed_overrides:
        value["varying_fields"].append(pointer)
        for case, replacement in zip(value["cases"], replacements):
            case["set"][pointer] = replacement
    sweep = SweepSpec.from_dict(value)
    layout = ResultLayout(tmp_path / "results")
    _, manifest_path = publish_manifest(sweep, layout, PROVENANCE)

    collection = collect_sweep(manifest_path, layout, allow_incomplete=True)
    assert collection["coverage_checks"]["fixed_comparison_contract"] is True
    disallowed_reasons = {
        "comparison_lr_policy_mismatch",
        "comparison_calibration_protocol_mismatch",
        "case_input_gain_mismatch",
        "case_learning_rate_mismatch",
        "case_calibration_binding_mismatch",
    }
    assert all(
        disallowed_reasons.isdisjoint(json.loads(row["eligibility_reasons"]))
        for row in collection["rows"]
    )


def test_numeric_learning_rate_axis_is_not_a_selected_case_value(tmp_path):
    value = sweep_value(run_value(epochs=1), seeds=(0, 1), three_cases=True)
    lr_pointer = "/run/training/learning_rate/0"
    value["axes"].append({"path": lr_pointer, "values": [0.01, 0.02]})
    value["varying_fields"].append(lr_pointer)
    sweep = SweepSpec.from_dict(value)
    layout = ResultLayout(tmp_path / "results")
    _, manifest_path = publish_manifest(sweep, layout, PROVENANCE)

    collection = collect_sweep(manifest_path, layout, allow_incomplete=True)
    assert collection["coverage_checks"]["fixed_comparison_contract"] is False
    assert all(
        "case_learning_rate_mismatch" in json.loads(row["eligibility_reasons"])
        for row in collection["rows"]
    )
