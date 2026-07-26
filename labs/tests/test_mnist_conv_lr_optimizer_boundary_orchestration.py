from __future__ import annotations

import copy
import hashlib
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments.mnist_conv.io import (
    atomic_write_bytes,
    atomic_write_json,
    read_json,
)
from experiments.mnist_conv.lr_optimizer_boundary import (
    CONCURRENCY_LEVELS,
    FROZEN_INITIALIZATION_CHECKPOINT_SHA256,
    FROZEN_INITIALIZATION_TENSOR_SHA256,
    FROZEN_SPLIT_SHA256,
    FROZEN_TRAIN_BATCH_ORDER_SHA256,
    REUSE_SOURCE_DIRECTORIES,
    RHO_DENSE_BY_SCHEME,
    ROW_IDS,
    TOTAL_STEPS,
    arm_id,
    bind_stage_entries,
    bias_capped_confirmation_entries,
    capped_bias_learning_rates,
    completion_action,
    entry_id,
    execution_entry_provenance,
    artifact_record,
    load_main_manifest,
    main_grid_entries,
    publish_once,
    reuse_cells_from_root,
    select_all_arms,
    select_arm,
    select_concurrency,
    terminal_completion,
    upper_sentinel_entries,
    verify_reuse_cells_against_root,
)
from experiments.mnist_conv.identity import sha256_json
from experiments.mnist_conv.lr_optimizer_boundary_plot import (
    _selected_replays,
    render_boundary_report,
)
from experiments.mnist_conv.lr_optimizer_boundary_spec import (
    OptimizerBoundaryStudySpec,
    optimizer_boundary_study_template,
)
import experiments.run_mnist_conv_lr_optimizer_boundary as boundary_runner
import experiments.submit_mnist_conv_lr_optimizer_boundary_jeanzay as boundary_submit


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_protocol(scheme: str) -> tuple[str, str, str]:
    if scheme == "baseline":
        return (
            "mnist-conv-run/v5",
            "conv-hardsigmoid-lr-conv2-two-rho-median-constant-sgd-bs16-v6",
            "lrstudy_c735d2beead9fcbadda89a65ffe86257da53f15e6cbfab7b4f04b14b08ff6c2f",
        )
    return (
        "mnist-conv-run/v6",
        "conv-hardsigmoid-lr-conv2-amplified-scheme-two-rho-median-constant-sgd-bs16",
        "lrsweep_07eed332608b684ff77bcf994071b90230fd877345b2e5132d9144cc4fa40435",
    )


def _build_reuse_root(root: Path) -> Path:
    for scheme, dense_values in RHO_DENSE_BY_SCHEME.items():
        for rho_dense in dense_values:
            source_name = REUSE_SOURCE_DIRECTORIES[(scheme, rho_dense)]
            source = root / source_name
            source.mkdir(parents=True)
            schema, protocol, collection = _source_protocol(scheme)
            run_spec_name = "run_spec.v5.json" if scheme == "baseline" else "run_spec.v6.json"
            rates = {
                "Bias_0": 0.01,
                "Bias_1": 0.01,
                "ConvWeight_0": 0.01,
                "ConvWeight_1": 0.01,
                "DenseWeight_0": rho_dense,
            }
            run_spec = {
                "schema_version": schema,
                "protocol_id": protocol,
                "run": {
                    "dataset": {
                        "validation": {"official_test_enabled": False}
                    },
                    "training": {
                        "optimizer": {
                            "name": "SGD",
                            "momentum": 0.0,
                            "weight_decay": 0.0,
                        },
                        "learning_rates_by_parameter": rates,
                    },
                    "lr_provenance": {
                        "row_id": ROW_IDS[scheme],
                        "rho_conv": 0.01,
                        "rho_dense": rho_dense,
                        "median_unit_by_weight": {
                            "ConvWeight_0": 1.0,
                            "ConvWeight_1": 1.0,
                            "DenseWeight_0": 1.0,
                        },
                        "split_sha256": FROZEN_SPLIT_SHA256,
                        "batch_order_sha256": FROZEN_TRAIN_BATCH_ORDER_SHA256,
                        "initialization_checkpoint_sha256": (
                            FROZEN_INITIALIZATION_CHECKPOINT_SHA256
                        ),
                        "initialization_tensor_sha256": (
                            FROZEN_INITIALIZATION_TENSOR_SHA256
                        ),
                        "probe_sha256": "a" * 64,
                    },
                },
            }
            atomic_write_json(source / run_spec_name, run_spec, canonical=True)
            atomic_write_json(
                source / "summary.json",
                {
                    "schema_version": "candidate",
                    "status": "complete",
                    "training_completed": True,
                    "completed_steps": TOTAL_STEPS,
                    "admissible": True,
                    "rho_conv": 0.01,
                    "rho_dense": rho_dense,
                    "row": {
                        "row_id": ROW_IDS[scheme],
                        "scheme": scheme,
                    },
                    "final_validation_loss": 0.1,
                    "final_validation_accuracy": 0.95,
                    "median_projection_efficiency": 0.99,
                    "learning_rates_by_parameter": rates,
                },
                canonical=True,
            )
            for name in (
                "best_validation.pt",
                "final.pt",
                "minibatches.json",
                "parameter_diagnostics.csv",
                "step_log.csv",
                "validation.json",
            ):
                atomic_write_bytes(source / name, f"{source_name}:{name}".encode())
            output_names = sorted(
                (
                    "best_validation.pt",
                    "final.pt",
                    "minibatches.json",
                    "parameter_diagnostics.csv",
                    run_spec_name,
                    "step_log.csv",
                    "summary.json",
                    "validation.json",
                )
            )
            outputs = [
                {
                    "path": f"old/stage/{source_name}/{name}",
                    "sha256": _sha(source / name),
                    "bytes": (source / name).stat().st_size,
                }
                for name in output_names
            ]
            completion = {
                "schema_version": "old-completion",
                "state": "complete",
                "study_id": collection if collection.startswith("lrstudy_") else None,
                "sweep_id": collection if collection.startswith("lrsweep_") else None,
                "outputs": outputs,
            }
            atomic_write_json(source / "complete.json", completion, canonical=True)
    return root


def _result(
    scheme: str,
    optimizer: str,
    rho_conv: float,
    rho_dense: float,
    *,
    loss: float,
    accuracy: float,
    projection: float,
    stage: str = "main_grid",
) -> dict:
    return {
        "entry_id": entry_id(
            scheme=scheme,
            optimizer=optimizer,
            stage=stage,
            rho_conv=rho_conv,
            rho_dense=rho_dense,
        ),
        "scheme": scheme,
        "optimizer": optimizer,
        "stage": stage,
        "rho_conv": rho_conv,
        "rho_dense": rho_dense,
        "status": "complete",
        "completed_steps": TOTAL_STEPS,
        "admissible": True,
        "final_validation_loss": loss,
        "final_validation_accuracy": accuracy,
        "median_projection_efficiency": projection,
        "official_test_read": False,
    }


def _probe_bundle(spec: OptimizerBoundaryStudySpec) -> dict:
    probes = {}
    for scheme in ("baseline", "ours", "legacy"):
        for optimizer in ("sgd", "adam"):
            key = f"{scheme}--{optimizer}"
            weight_hash = "a" * 64 if optimizer == "sgd" else "b" * 64
            probes[key] = {
                "schema_version": "effective-probe-test",
                "scheme": scheme,
                "optimizer": optimizer,
                "optimizer_name": "SGD" if optimizer == "sgd" else "Adam",
                "optimizer_parameters": spec.data["optimizer_arms"][optimizer],
                "probe_sha256": weight_hash,
                "weight_normalization_probe_sha256": weight_hash,
                "bias_q90_probe_sha256": "c" * 64,
                "normalization_unit_by_weight": {
                    "ConvWeight_0": 1.0,
                    "ConvWeight_1": 1.0,
                    "DenseWeight_0": 1.0,
                },
                "bias_q90_unit_by_parameter": {
                    "Bias_0": 1.0,
                    "Bias_1": 2.0,
                },
                "official_test_read": False,
            }
    return {
        "schema_version": "mnist-conv-lr-optimizer-boundary-probes/v1",
        "study_id": spec.study_id,
        "code_fingerprint": "d" * 64,
        "parent_sha256": "e" * 64,
        "checkpoint_sha256": "f" * 64,
        "minibatch_sha256": "1" * 64,
        "probes_by_arm": probes,
        "official_test_read": False,
    }


def _manifest_fixture(tmp_path: Path) -> tuple[Path, Path, Path, dict, dict]:
    reuse_root = _build_reuse_root(tmp_path / "reuse")
    study_path = tmp_path / "study.json"
    spec, _ = boundary_runner.create_study_config(
        reuse_root=reuse_root, output_path=study_path
    )
    probes = _probe_bundle(spec)
    probes_path = tmp_path / "probes.json"
    publish_once(probes_path, probes)
    manifest_path = tmp_path / "manifest.json"
    manifest = boundary_runner.create_main_manifest(
        study_config=study_path,
        reuse_root=reuse_root,
        probes_path=probes_path,
        output_path=manifest_path,
    )
    return reuse_root, study_path, probes_path, manifest, {
        "path": manifest_path,
        "spec": spec,
    }


def _benchmark_fixture(
    tmp_path: Path,
    *,
    study_path: Path,
    study_dir: Path,
    manifest_path: Path,
    manifest: dict,
    spec: OptimizerBoundaryStudySpec,
) -> Path:
    entry = next(
        item
        for item in manifest["entries"]
        if item["scheme"] == "ours"
        and item["optimizer"] == "adam"
        and item["rho_conv"] == 0.01
        and item["rho_dense"] == 0.003
    )
    levels = [
        {
            "concurrency": width,
            "failed_children": 0,
            "completed_children": width,
            "peak_gpu_memory_mib": 1000.0 * width,
            "mean_gpu_utilization_percent": 80.0,
            "aggregate_steps_per_second": float(width),
            "measured_steps_total": width * 256,
            "measurement_wall_seconds": 256.0,
            "wall_seconds": 257.0,
            "child_failures": [],
            "children": [{"barrier_synchronized": True}],
        }
        for width in CONCURRENCY_LEVELS
    ]
    selection = select_concurrency(levels)
    scientific = boundary_runner._benchmark_scientific_payload(
        spec=spec,
        study_config_path=study_path,
        study_dir=study_dir,
        manifest_path=manifest_path,
        manifest=manifest,
        entry=entry,
        levels=levels,
        selection=selection,
        device_identity={
            "name": "Tesla V100-SXM2-32GB",
            "total_memory_mib": 32768.0,
            "uuid": "GPU-test",
        },
    )
    report = {
        **scientific,
        "benchmark_id": "lroptbenchmark_" + sha256_json(scientific),
    }
    path = tmp_path / "benchmark.json"
    publish_once(path, report)
    return path


def test_exact_main_grid_counts_and_reuse_coordinates() -> None:
    entries = main_grid_entries()

    assert len(entries) == 36
    assert sum(entry["mode"] == "reuse_hash_verified_sgd" for entry in entries) == 6
    assert sum(
        entry["optimizer"] == "sgd"
        and entry["mode"] == "new_five_epoch_training"
        for entry in entries
    ) == 12
    assert sum(
        entry["optimizer"] == "adam"
        and entry["mode"] == "new_five_epoch_training"
        for entry in entries
    ) == 18
    assert all(entry["expected_steps"] == 17_190 for entry in entries)
    assert all(entry["official_test_read"] is False for entry in entries)


def test_reuse_audit_builds_valid_study_and_fails_closed(tmp_path: Path) -> None:
    root = _build_reuse_root(tmp_path / "reuse")
    cells = reuse_cells_from_root(root)
    spec = OptimizerBoundaryStudySpec.from_dict(
        optimizer_boundary_study_template(cells)
    )

    assert len(cells) == 6
    assert spec.data["reuse"]["mismatch_policy"] == "abort_without_rerun"
    assert spec.data["optimizer_arms"]["adam"]["name"] == "Adam"
    verify_reuse_cells_against_root(spec.data["reuse"]["cells"], root)

    target = root / REUSE_SOURCE_DIRECTORIES[("legacy", 0.01)] / "summary.json"
    target.write_bytes(target.read_bytes() + b"\n")
    with pytest.raises(RuntimeError, match="aborting without rerun"):
        verify_reuse_cells_against_root(spec.data["reuse"]["cells"], root)


def test_selection_uses_plateau_tie_breakers_and_triggers_sentinel() -> None:
    records = [
        _result(
            "ours",
            "sgd",
            0.01,
            0.003,
            loss=1.0,
            accuracy=0.94,
            projection=0.99,
        ),
        _result(
            "ours",
            "sgd",
            0.015,
            0.003,
            loss=1.01,
            accuracy=0.96,
            projection=0.98,
        ),
        _result(
            "ours",
            "sgd",
            0.03,
            0.003,
            loss=1.019,
            accuracy=0.96,
            projection=0.995,
        ),
    ]

    selection = select_arm(records, require_provenance=False)

    assert selection["selected"]["rho_conv"] == 0.03
    assert selection["trigger_upper_sentinel"] is True
    assert selection["status"] == "pending_upper_sentinel"


def test_sentinel_stops_at_point_one_and_adam_reports_both_edges() -> None:
    records = [
        _result(
            "baseline",
            "adam",
            0.01,
            0.01,
            loss=1.0,
            accuracy=0.95,
            projection=0.99,
        ),
        _result(
            "baseline",
            "adam",
            0.03,
            0.01,
            loss=1.01,
            accuracy=0.95,
            projection=0.99,
        ),
        _result(
            "baseline",
            "adam",
            0.1,
            0.01,
            loss=1.015,
            accuracy=0.96,
            projection=0.99,
            stage="upper_sentinel",
        ),
    ]

    selection = select_arm(records, require_provenance=False)

    assert selection["trigger_upper_sentinel"] is False
    assert selection["boundary_statuses"] == [
        "unbracketed_high",
        "unbracketed_low",
    ]
    assert selection["status"] == "selected"


def test_terminal_failed_sentinels_do_not_retrigger_forever() -> None:
    records = [
        _result(
            "legacy",
            "sgd",
            rho_conv,
            0.003,
            loss=1.0 if rho_conv == 0.03 else 2.0,
            accuracy=0.95,
            projection=0.99,
        )
        for rho_conv in (0.01, 0.015, 0.03)
    ]
    for rho_dense in (0.003, 0.01):
        failed = _result(
            "legacy",
            "sgd",
            0.1,
            rho_dense,
            loss=9.0,
            accuracy=0.0,
            projection=0.0,
            stage="upper_sentinel",
        )
        failed.update(
            {
                "status": "safety_failure",
                "completed_steps": 10,
                "admissible": False,
                "safety_failure": {"kind": "non_finite_optimizer_state"},
            }
        )
        records.append(failed)

    selection = select_arm(records, require_provenance=False)

    assert selection["trigger_upper_sentinel"] is False
    assert selection["status"] == "selected"
    assert selection["boundary_statuses"] == ["bracketed"]


def test_all_arms_emit_only_triggered_two_slice_sentinels() -> None:
    selections = {}
    for scheme in ("baseline", "ours", "legacy"):
        for optimizer in ("sgd", "adam"):
            selections[arm_id(scheme, optimizer)] = {
                "trigger_upper_sentinel": scheme == "legacy"
                and optimizer == "adam"
            }

    entries = upper_sentinel_entries(selections)

    assert len(entries) == 2
    assert {entry["rho_dense"] for entry in entries} == {0.003, 0.01}
    assert {entry["rho_conv"] for entry in entries} == {0.1}


def test_confirmations_reject_any_pending_sentinel_selection() -> None:
    selections = {}
    parents = {}
    q90 = {}
    for scheme in ("baseline", "ours", "legacy"):
        rho_dense = RHO_DENSE_BY_SCHEME[scheme][0]
        for optimizer in ("sgd", "adam"):
            key = arm_id(scheme, optimizer)
            parent_id = entry_id(
                scheme=scheme,
                optimizer=optimizer,
                stage="main_grid",
                rho_conv=0.015,
                rho_dense=rho_dense,
            )
            result_sha = ("1" if optimizer == "sgd" else "2") * 64
            completion_sha = ("3" if optimizer == "sgd" else "4") * 64
            selections[key] = {
                "status": "selected",
                "trigger_upper_sentinel": False,
                "selected": {
                    "entry_id": parent_id,
                    "rho_conv": 0.015,
                    "rho_dense": rho_dense,
                    "result_sha256": result_sha,
                    "completion_sha256": completion_sha,
                },
            }
            parents[parent_id] = {
                "entry_id": parent_id,
                "scheme": scheme,
                "optimizer": optimizer,
                "stage": "main_grid",
                "bias_policy": "attached",
                "result_sha256": result_sha,
                "completion_sha256": completion_sha,
                "raw_learning_rates_by_parameter": {
                    "ConvWeight_0": 0.1,
                    "Bias_0": 0.1,
                    "ConvWeight_1": 0.2,
                    "Bias_1": 0.2,
                    "DenseWeight_0": 0.01,
                },
            }
            q90[key] = {"Bias_0": 1.0, "Bias_1": 1.0}
    selections["legacy--adam"] = {
        **selections["legacy--adam"],
        "status": "pending_upper_sentinel",
        "trigger_upper_sentinel": True,
    }

    with pytest.raises(ValueError, match="fully resolved post-sentinel"):
        bias_capped_confirmation_entries(selections, parents, q90)


def test_bias_cap_preserves_weights_and_never_increases_bias() -> None:
    attached = {
        "ConvWeight_0": 0.2,
        "Bias_0": 0.2,
        "ConvWeight_1": 0.03,
        "Bias_1": 0.03,
        "DenseWeight_0": 0.004,
    }

    capped, details = capped_bias_learning_rates(
        attached,
        q90_nominal_bias_units={"Bias_0": 0.02, "Bias_1": 1.0},
        bias_to_weight={
            "Bias_0": "ConvWeight_0",
            "Bias_1": "ConvWeight_1",
        },
    )

    assert capped["ConvWeight_0"] == attached["ConvWeight_0"]
    assert capped["ConvWeight_1"] == attached["ConvWeight_1"]
    assert capped["DenseWeight_0"] == attached["DenseWeight_0"]
    assert capped["Bias_0"] == pytest.approx(0.05)
    assert capped["Bias_1"] == pytest.approx(0.001)
    assert all(
        item["capped_learning_rate"] <= item["attached_learning_rate"]
        for item in details["biases"].values()
    )


def test_incomplete_adam_is_never_model_only_resumed(tmp_path: Path) -> None:
    entry_dir = tmp_path / "entry"
    entry_dir.mkdir()
    atomic_write_bytes(entry_dir / "final.pt", b"incomplete model only")

    assert (
        completion_action(
            entry_dir,
            optimizer="adam",
            expected_entry_id="main_grid--baseline--adam--c0p01--d0p01",
        )
        == "restart_from_initialization"
    )


def test_adam_restart_result_is_attempt_scoped_and_preserves_stale_base(
    tmp_path: Path,
) -> None:
    entry_dir = tmp_path / "entry"
    entry_dir.mkdir()
    publish_once(entry_dir / "result.json", {"stale": True})
    result = {"attempt_relative_path": "attempts/attempt-000001"}

    relative = boundary_runner._attempt_result_relative_path(
        result, resume_action="restart_from_initialization"
    )
    publish_once(entry_dir / relative, {"fresh": True})

    assert read_json(entry_dir / "result.json") == {"stale": True}
    assert read_json(entry_dir / relative) == {"fresh": True}


def test_concurrency_uses_largest_safe_width_with_throughput_gate() -> None:
    levels = [
        {
            "concurrency": width,
            "failed_children": 1 if width == 16 else 0,
            "peak_gpu_memory_mib": 1000.0 * width,
            "aggregate_steps_per_second": {
                1: 1.0,
                2: 2.0,
                4: 4.0,
                8: 8.0,
                12: 7.4,
                16: 8.5,
            }[width],
        }
        for width in CONCURRENCY_LEVELS
    ]

    selected = select_concurrency(levels)

    assert selected["selected_concurrency"] == 12


def test_concurrency_reserves_exactly_ten_percent_of_capacity() -> None:
    threshold = 32_768.0 * 0.90

    def levels(last_memory: float) -> list[dict]:
        return [
            {
                "concurrency": width,
                "failed_children": 0,
                "peak_gpu_memory_mib": (
                    last_memory if width == 16 else 1000.0 * width
                ),
                "aggregate_steps_per_second": float(width),
            }
            for width in CONCURRENCY_LEVELS
        ]

    assert (
        select_concurrency(levels(threshold))["selected_concurrency"] == 16
    )
    assert (
        select_concurrency(levels(threshold + 1.0e-6))[
            "selected_concurrency"
        ]
        == 12
    )


def test_benchmark_child_command_has_exact_barrier_arguments(
    tmp_path: Path,
) -> None:
    command = boundary_runner._benchmark_child_command(
        python="/python",
        study_config=tmp_path / "study.json",
        study_dir=tmp_path / "study",
        manifest_path=tmp_path / "manifest.json",
        reuse_root=tmp_path / "reuse",
        entry_id_value="entry",
        data_root=tmp_path / "data",
        output_path=tmp_path / "child.json",
        device="cuda",
        ready_path=tmp_path / "child.ready",
        start_path=tmp_path / "start.json",
    )

    study = str((tmp_path / "study.json").resolve())
    assert command.count("--study-config") == 1
    assert command[command.index("--study-config") + 1] == study
    assert command.count(study) == 1
    assert command[command.index("--ready-path") + 1].endswith("child.ready")
    assert command[command.index("--start-path") + 1].endswith("start.json")
    assert command[command.index("--start-timeout-seconds") + 1] == "600"


def test_cli_dispatches_barrier_and_benchmark_arguments_exactly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = {}

    def fake_child(**kwargs):
        captured["child"] = kwargs
        return {"status": "complete"}

    def fake_pack(**kwargs):
        captured["pack"] = kwargs
        return {"status": "complete"}

    monkeypatch.setattr(boundary_runner, "run_benchmark_child", fake_child)
    monkeypatch.setattr(boundary_runner, "run_pack", fake_pack)
    assert boundary_runner.main(
        [
            "benchmark-child",
            "--study-config",
            "study.json",
            "--study-dir",
            "study",
            "--manifest",
            "manifest.json",
            "--reuse-root",
            "reuse",
            "--entry-id",
            "entry",
            "--data-root",
            "data",
            "--output",
            "child.json",
            "--ready-path",
            "child.ready",
            "--start-path",
            "start.json",
            "--start-timeout-seconds",
            "123",
        ]
    ) == 0
    assert captured["child"]["ready_path"] == "child.ready"
    assert captured["child"]["start_path"] == "start.json"
    assert captured["child"]["start_timeout_seconds"] == 123.0

    assert boundary_runner.main(
        [
            "run-pack",
            "--study-config",
            "study.json",
            "--study-dir",
            "study",
            "--manifest",
            "manifest.json",
            "--benchmark",
            "benchmark.json",
            "--reuse-root",
            "reuse",
            "--data-root",
            "data",
            "--scheme",
            "ours",
            "--concurrency",
            "8",
            "--log-dir",
            "logs",
        ]
    ) == 0
    assert captured["pack"]["benchmark_path"] == "benchmark.json"
    assert captured["pack"]["concurrency"] == 8


def test_main_manifest_is_path_free_and_revalidates_after_relocation(
    tmp_path: Path,
) -> None:
    reuse_root, _study, _probes, manifest, metadata = _manifest_fixture(
        tmp_path
    )
    manifest_path = metadata["path"]

    assert str(reuse_root) not in manifest_path.read_text(encoding="utf-8")
    assert all(
        "source_root" not in entry["reuse_binding"]
        for entry in manifest["entries"]
        if entry["reuse_binding"] is not None
    )
    moved = tmp_path / "moved-reuse"
    shutil.copytree(reuse_root, moved)
    relocated = load_main_manifest(manifest_path, reuse_root=moved)
    assert relocated == manifest
    for entry in manifest["entries"]:
        if entry["optimizer"] == "sgd":
            assert entry["probe_sha256"] == "a" * 64
            assert entry["bias_q90_probe_sha256"] == "c" * 64
            assert entry["normalization_unit_by_weight"] == {
                "ConvWeight_0": 1.0,
                "ConvWeight_1": 1.0,
                "DenseWeight_0": 1.0,
            }


def test_foreign_probe_bundle_and_separate_bias_bundle_fail_closed(
    tmp_path: Path,
) -> None:
    _reuse, study_path, probes_path, _manifest, metadata = _manifest_fixture(
        tmp_path
    )
    foreign = read_json(probes_path)
    foreign["study_id"] = "lrstudy_" + "9" * 64
    foreign_path = tmp_path / "foreign-probes.json"
    publish_once(foreign_path, foreign)
    with pytest.raises(ValueError, match="match the study config"):
        boundary_runner.create_main_manifest(
            study_config=study_path,
            reuse_root=tmp_path / "reuse",
            probes_path=foreign_path,
            output_path=tmp_path / "foreign-manifest.json",
        )

    results_path = tmp_path / "empty-results.json"
    publish_once(
        results_path,
        {
            "schema_version": "results-test",
            "records": [],
            "official_test_read": False,
        },
    )
    selection_path = tmp_path / "selection-inputs.json"
    publish_once(
        selection_path,
        {
            "schema_version": "selection-test",
            "results_sha256": _sha(results_path),
            "main_manifest_sha256": _sha(metadata["path"]),
            "probe_bundle_sha256": _sha(probes_path),
            "selections_by_arm": {},
            "official_test_read": False,
        },
    )
    with pytest.raises(RuntimeError, match="same canonical probe bundle"):
        boundary_runner.create_confirmations(
            selections_path=selection_path,
            results_path=results_path,
            bias_probe_path=foreign_path,
            main_manifest_path=metadata["path"],
            probes_path=probes_path,
            output_path=tmp_path / "confirmations.json",
        )


def test_completion_refuses_same_coordinate_with_changed_provenance(
    tmp_path: Path,
) -> None:
    _reuse, _study, _probes, manifest, _metadata = _manifest_fixture(tmp_path)
    entry = next(
        item
        for item in manifest["entries"]
        if item["mode"] == "new_five_epoch_training"
    )
    entry_dir = tmp_path / "entry"
    entry_dir.mkdir()
    result = {
        "schema_version": "test-result",
        "entry_id": entry["entry_id"],
        "scheme": entry["scheme"],
        "optimizer": entry["optimizer"],
        "stage": entry["stage"],
        "rho_conv": entry["rho_conv"],
        "rho_dense": entry["rho_dense"],
        "status": "complete",
        "completed_steps": TOTAL_STEPS,
        "admissible": True,
        "final_validation_loss": 0.1,
        "final_validation_accuracy": 0.95,
        "median_projection_efficiency": 0.99,
        **execution_entry_provenance(entry),
        "official_test_read": False,
    }
    publish_once(entry_dir / "result.json", result)
    completion = terminal_completion(
        entry_id_value=entry["entry_id"],
        result=result,
        outputs=[artifact_record(entry_dir, "result.json")],
        entry=entry,
    )
    publish_once(entry_dir / "complete.json", completion)

    assert completion_action(
        entry_dir,
        optimizer=entry["optimizer"],
        expected_entry_id=entry["entry_id"],
        expected_entry=entry,
    ) == "skip_immutable_complete"
    changed = {**entry, "probe_sha256": "9" * 64}
    with pytest.raises(RuntimeError, match="stale coordinate"):
        completion_action(
            entry_dir,
            optimizer=entry["optimizer"],
            expected_entry_id=entry["entry_id"],
            expected_entry=changed,
        )


def test_reduced_manifest_results_selection_confirmation_e2e(
    tmp_path: Path,
) -> None:
    _reuse, study_path, probes_path, manifest, metadata = _manifest_fixture(
        tmp_path
    )
    manifest_path = metadata["path"]
    study_dir = tmp_path / "study-output"
    for entry in manifest["entries"]:
        entry_dir = (
            study_dir
            / "stages"
            / entry["stage"]
            / "entries"
            / entry["entry_id"]
        )
        entry_dir.mkdir(parents=True)
        rho_loss = {
            0.01: 1.2,
            0.015: 0.5,
            0.03: 1.3,
        }[entry["rho_conv"]]
        result = {
            "schema_version": "synthetic-e2e-result",
            "entry_id": entry["entry_id"],
            "scheme": entry["scheme"],
            "optimizer": entry["optimizer"],
            "stage": entry["stage"],
            "rho_conv": entry["rho_conv"],
            "rho_dense": entry["rho_dense"],
            "bias_policy": entry["bias_policy"],
            "status": "complete",
            "completed_steps": TOTAL_STEPS,
            "admissible": True,
            "final_validation_loss": rho_loss + entry["rho_dense"],
            "final_validation_accuracy": 0.95,
            "median_projection_efficiency": 0.99,
            **execution_entry_provenance(entry),
            "official_test_read": False,
        }
        publish_once(entry_dir / "result.json", result)
        completion = terminal_completion(
            entry_id_value=entry["entry_id"],
            result=result,
            outputs=[artifact_record(entry_dir, "result.json")],
            entry=entry,
        )
        publish_once(entry_dir / "complete.json", completion)

    results_path = tmp_path / "results.json"
    aggregated = boundary_runner.aggregate_results(
        study_dir=study_dir,
        plan_paths=[manifest_path],
        output_path=results_path,
    )
    assert aggregated["record_count"] == 36
    selection_path = tmp_path / "selection.json"
    sentinel_path = tmp_path / "sentinels.json"
    selection = boundary_runner.select_and_plan_sentinels(
        results_path=results_path,
        main_manifest_path=manifest_path,
        probes_path=probes_path,
        selection_path=selection_path,
        sentinel_path=sentinel_path,
    )
    assert selection["status"] == "base_selection_complete"
    assert read_json(sentinel_path)["entry_count"] == 0

    confirmation_path = tmp_path / "confirmations.json"
    confirmation = boundary_runner.create_confirmations(
        selections_path=selection_path,
        results_path=results_path,
        bias_probe_path=probes_path,
        main_manifest_path=manifest_path,
        probes_path=probes_path,
        output_path=confirmation_path,
    )
    assert confirmation["entry_count"] == 6
    selection_sha = _sha(selection_path)
    records_by_id = {
        record["entry_id"]: record for record in aggregated["records"]
    }
    for entry in confirmation["entries"]:
        parent = records_by_id[entry["parent_entry_id"]]
        assert (
            entry["parent_entry_completion_sha256"]
            == parent["completion_sha256"]
        )
        assert entry["parent_decision_sha256"] == selection_sha
        assert entry["bias_policy"] == "capped"
        entry_dir = (
            study_dir
            / "stages"
            / entry["stage"]
            / "entries"
            / entry["entry_id"]
        )
        entry_dir.mkdir(parents=True)
        result = {
            "schema_version": "synthetic-confirmation-result",
            "entry_id": entry["entry_id"],
            "scheme": entry["scheme"],
            "optimizer": entry["optimizer"],
            "stage": entry["stage"],
            "rho_conv": entry["rho_conv"],
            "rho_dense": entry["rho_dense"],
            "bias_policy": "capped",
            "status": "complete",
            "completed_steps": TOTAL_STEPS,
            "admissible": True,
            "final_validation_loss": 0.4,
            "final_validation_accuracy": 0.96,
            "median_projection_efficiency": 0.99,
            **execution_entry_provenance(entry),
            "official_test_read": False,
        }
        publish_once(entry_dir / "result.json", result)
        publish_once(
            entry_dir / "complete.json",
            terminal_completion(
                entry_id_value=entry["entry_id"],
                result=result,
                outputs=[artifact_record(entry_dir, "result.json")],
                entry=entry,
            ),
        )

    final_results_path = tmp_path / "final-results.json"
    final_results = boundary_runner.aggregate_results(
        study_dir=study_dir,
        plan_paths=[manifest_path, confirmation_path],
        output_path=final_results_path,
    )
    assert final_results["record_count"] == 42
    tampered_results = copy.deepcopy(final_results)
    tampered_confirmation = next(
        record
        for record in tampered_results["records"]
        if record["stage"] == "bias_capped_confirmation"
    )
    tampered_confirmation["parent_result_sha256"] = "0" * 64
    tampered_results_path = tmp_path / "tampered-final-results.json"
    publish_once(tampered_results_path, tampered_results)
    with pytest.raises(RuntimeError, match="coordinates and parent hashes"):
        boundary_runner.finalize_boundary_study(
            results_path=tampered_results_path,
            selections_path=selection_path,
            output_dir=tmp_path / "tampered-final",
        )

    pending_selection = read_json(selection_path)
    pending_key = "legacy--adam"
    pending_selection["selections_by_arm"][pending_key]["status"] = (
        "pending_upper_sentinel"
    )
    pending_selection["selections_by_arm"][pending_key][
        "trigger_upper_sentinel"
    ] = True
    pending_selection_path = tmp_path / "pending-selection.json"
    publish_once(pending_selection_path, pending_selection)
    with pytest.raises(RuntimeError, match="fully resolved"):
        boundary_runner.finalize_boundary_study(
            results_path=final_results_path,
            selections_path=pending_selection_path,
            output_dir=tmp_path / "pending-final",
        )

    final_dir = tmp_path / "final"
    final = boundary_runner.finalize_boundary_study(
        results_path=final_results_path,
        selections_path=selection_path,
        output_dir=final_dir,
    )
    assert final["selection_input_results_sha256"] == _sha(results_path)
    assert final["final_results_sha256"] == _sha(final_results_path)
    plot_path = final_dir / "plots" / "epochwise_effective_rho.png"
    original_plot = plot_path.read_bytes()
    assert boundary_runner.finalize_boundary_study(
        results_path=final_results_path,
        selections_path=selection_path,
        output_dir=final_dir,
    ) == final
    assert plot_path.read_bytes() == original_plot
    plot_path.write_bytes(original_plot + b"tampered")
    with pytest.raises(RuntimeError, match="hash/size"):
        boundary_runner.finalize_boundary_study(
            results_path=final_results_path,
            selections_path=selection_path,
            output_dir=final_dir,
        )
    assert plot_path.read_bytes() == original_plot + b"tampered"


def test_selected_replays_retain_attached_and_capped_policies() -> None:
    attached = {
        **_result(
            "ours",
            "adam",
            0.015,
            0.003,
            loss=0.1,
            accuracy=0.95,
            projection=0.99,
        ),
        "diagnostic_best": True,
        "bias_policy": "attached",
    }
    capped = {
        **attached,
        "entry_id": entry_id(
            scheme="ours",
            optimizer="adam",
            stage="bias_capped_confirmation",
            rho_conv=0.015,
            rho_dense=0.003,
        ),
        "stage": "bias_capped_confirmation",
        "diagnostic_best": False,
        "bias_policy": "capped",
    }

    selected = _selected_replays([attached, capped])

    assert selected[("ours", "adam")]["attached"] is attached
    assert selected[("ours", "adam")]["capped"] is capped


def test_jeanzay_submitter_builds_three_benchmark_selected_main_packs(
    tmp_path: Path,
) -> None:
    reuse, study_path, _probes, manifest, metadata = _manifest_fixture(
        tmp_path
    )
    study_dir = tmp_path / "study-output"
    study_dir.mkdir()
    benchmark = _benchmark_fixture(
        tmp_path,
        study_path=study_path,
        study_dir=study_dir,
        manifest_path=metadata["path"],
        manifest=manifest,
        spec=metadata["spec"],
    )
    data_root = tmp_path / "data"
    data_root.mkdir()
    args = SimpleNamespace(
        repo_root=str(Path(__file__).resolve().parents[2]),
        python=sys.executable,
        study_config=str(study_path),
        study_dir=str(study_dir),
        manifest=str(metadata["path"]),
        reuse_root=str(reuse),
        data_root=str(data_root),
        benchmark=str(benchmark),
        output_root=str(study_dir / "pack-records"),
        log_root=str(study_dir / "logs"),
        module="pytorch-gpu/py3/2.5.0",
    )

    commands, submit_metadata = (
        boundary_submit.prepare_main_pack_submissions(args)
    )

    assert len(commands) == 3
    assert submit_metadata["schemes"] == ["baseline", "ours", "legacy"]
    assert submit_metadata["selected_concurrency"] == 16
    joined = "\n".join(" ".join(command) for command in commands)
    assert "--account=fmu@v100" in joined
    assert "--constraint=v100-32g" in joined
    assert "--cpus-per-task=16" in joined
    assert str(benchmark) in joined
    assert "MNIST_CONV_BOUNDARY_CONCURRENCY=16" in joined


def test_conditional_submitter_skips_zero_work_and_only_submits_triggered_scheme(
    tmp_path: Path,
) -> None:
    reuse, study_path, probes_path, manifest, metadata = _manifest_fixture(
        tmp_path
    )
    study_dir = tmp_path / "study-output"
    study_dir.mkdir()
    benchmark = _benchmark_fixture(
        tmp_path,
        study_path=study_path,
        study_dir=study_dir,
        manifest_path=metadata["path"],
        manifest=manifest,
        spec=metadata["spec"],
    )
    probes = read_json(probes_path)
    decisions = {
        arm_id(scheme, optimizer): {
            "trigger_upper_sentinel": scheme == "legacy"
            and optimizer == "adam"
        }
        for scheme in ("baseline", "ours", "legacy")
        for optimizer in ("sgd", "adam")
    }
    decision_sha = "2" * 64
    entries = [
        {
            **entry,
            "parent_sha256": decision_sha,
            "parent_entry_completion_sha256": None,
            "parent_decision_sha256": decision_sha,
        }
        for entry in upper_sentinel_entries(decisions)
    ]
    partial = bind_stage_entries(
        entries,
        main_manifest=manifest,
        probe_bundle=probes,
        stage="upper_sentinel",
    )
    partial_path = tmp_path / "partial-sentinel.json"
    publish_once(partial_path, partial)
    empty = bind_stage_entries(
        [],
        main_manifest=manifest,
        probe_bundle=probes,
        stage="upper_sentinel",
    )
    empty_path = tmp_path / "empty-sentinel.json"
    publish_once(empty_path, empty)
    data_root = tmp_path / "data"
    data_root.mkdir()

    def args_for(path: Path) -> SimpleNamespace:
        return SimpleNamespace(
            repo_root=str(Path(__file__).resolve().parents[2]),
            python=sys.executable,
            study_config=str(study_path),
            study_dir=str(study_dir),
            manifest=str(path),
            reuse_root=str(reuse),
            data_root=str(data_root),
            benchmark=str(benchmark),
            stage="upper_sentinel",
            output_root=str(study_dir / "pack-records"),
            log_root=str(study_dir / "logs"),
            module="pytorch-gpu/py3/2.5.0",
        )

    partial_commands, partial_metadata = (
        boundary_submit.prepare_conditional_pack_submissions(
            args_for(partial_path)
        )
    )
    empty_commands, empty_metadata = (
        boundary_submit.prepare_conditional_pack_submissions(
            args_for(empty_path)
        )
    )

    assert len(partial_commands) == 1
    assert partial_metadata["schemes"] == ["legacy"]
    assert "MNIST_CONV_BOUNDARY_SCHEME=legacy" in " ".join(
        partial_commands[0]
    )
    assert empty_commands == []
    assert empty_metadata["schemes"] == []


def test_submission_record_preflight_prevents_duplicate_sbatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        boundary_submit,
        "verify_login_r3_allocation",
        lambda runner: {"verified": True},
    )
    calls = []

    def fake_runner(command, **_kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(
            command, 0, stdout=f"{100 + len(calls)}\n", stderr=""
        )

    record = tmp_path / "submission.json"
    first = boundary_submit.submit_prepared(
        commands=[["sbatch", "one"], ["sbatch", "two"]],
        metadata={
            "operation": "main_packs",
            "log_root": str(tmp_path / "logs"),
            "pack_outputs": {
                "baseline": str(tmp_path / "out" / "baseline.json"),
                "ours": str(tmp_path / "out" / "ours.json"),
                "legacy": str(tmp_path / "out" / "legacy.json"),
            },
        },
        record_path=record,
        submit=True,
        runner=fake_runner,
    )
    assert [job["job_id"] for job in first["jobs"]] == ["101", "102"]
    with pytest.raises(RuntimeError, match="potentially duplicate"):
        boundary_submit.submit_prepared(
            commands=[["sbatch", "one"], ["sbatch", "two"]],
            metadata={
                "operation": "main_packs",
                "log_root": str(tmp_path / "logs"),
                "pack_outputs": {},
            },
            record_path=record,
            submit=True,
            runner=fake_runner,
        )
    assert len(calls) == 2


def test_report_plots_all_requested_diagnostics(tmp_path: Path) -> None:
    records = []
    for scheme in ("baseline", "ours", "legacy"):
        for optimizer in ("sgd", "adam"):
            records.append(
                {
                    **_result(
                        scheme,
                        optimizer,
                        0.01,
                        RHO_DENSE_BY_SCHEME[scheme][0],
                        loss=0.1,
                        accuracy=0.95,
                        projection=0.99,
                    ),
                    "diagnostic_best": True,
                    "bias_policy": "attached",
                    "epoch_diagnostics": [
                        {
                            "epoch": epoch,
                            "achieved_weight_rho_median_by_parameter": {
                                "ConvWeight_0": 0.01 / (epoch + 1),
                                "ConvWeight_1": 0.008 / (epoch + 1),
                                "DenseWeight_0": 0.006 / (epoch + 1),
                            },
                            "bias_weight_step_ratio_median_by_parameter": {
                                "Bias_0": 0.1 + epoch * 0.01,
                                "Bias_1": 0.2 + epoch * 0.01,
                            },
                        }
                        for epoch in range(6)
                    ],
                }
            )

    outputs = render_boundary_report(records, tmp_path / "plots")

    assert {path.name for path in outputs} == {
        "final_validation_loss_heatmap.png",
        "final_validation_accuracy_heatmap.png",
        "epochwise_effective_rho.png",
        "epochwise_bias_weight_ratio.png",
    }
    assert all(path.stat().st_size > 0 for path in outputs)
