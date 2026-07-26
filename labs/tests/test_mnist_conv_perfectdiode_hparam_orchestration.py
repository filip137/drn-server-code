from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pytest

import experiments.mnist_conv.perfectdiode_hparam as orchestration
from experiments.mnist_conv.identity import sha256_file
from experiments.mnist_conv.io import (
    atomic_write_bytes,
    atomic_write_csv,
    atomic_write_json,
    read_json,
)
from experiments.mnist_conv.perfectdiode_hparam_spec import (
    PerfectDiodeHparamStudySpec,
    core_grid,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG = (
    REPO_ROOT
    / "configs"
    / "conv"
    / "perfectdiode_conv12_sgd_adam_hparam_ordinary_mnist_v1.json"
)
PROVENANCE = {
    "git_revision": "a" * 40,
    "dirty_source_digest": None,
    "effective_code_fingerprint": "b" * 64,
}
SOURCE_COMMIT = "a" * 40
SOURCE_ARCHIVE = "c" * 64
ENVIRONMENT = {
    "python_version": "3.12.test",
    "python_implementation": "CPython",
    "platform": "test",
    "executable_name": "python",
    "torch": "test",
    "cuda_runtime": "test",
}


def _spec() -> PerfectDiodeHparamStudySpec:
    return PerfectDiodeHparamStudySpec.from_path(CONFIG)


def _write_declared_outputs(
    shard: Path,
    stage: str,
    entry_id: str,
    result: dict[str, Any],
    payload: dict[str, Any],
) -> None:
    entry_dir = shard / "stages" / stage / "entries" / entry_id
    outputs = (
        orchestration._PREFLIGHT_OUTPUTS
        if stage == "preflight_canary"
        else orchestration._RUNTIME_OUTPUTS[stage]
    )
    for name in outputs:
        path = entry_dir / name
        if name == "result.json":
            if stage in {
                "rho_core_candidates",
                "rho_extension_candidates",
                "long_confirm",
            }:
                result["checkpoints"] = {
                    "best_validation": {
                        "path": "best_validation.pt",
                        "sha256": sha256_file(entry_dir / "best_validation.pt"),
                    },
                    "final": {
                        "path": "final.pt",
                        "sha256": sha256_file(entry_dir / "final.pt"),
                    },
                }
            atomic_write_json(path, result, canonical=True)
        elif name == "run_spec.json":
            atomic_write_json(
                path,
                {
                    "study_id": _spec().study_id,
                    "config_sha256": _spec().config_sha256,
                    "official_test_read": False,
                    "provenance": {
                        "initialization_checkpoint_sha256": result[
                            "initialization_checkpoint_sha256"
                        ],
                        "initialization_tensor_sha256": result[
                            "initialization_tensor_sha256"
                        ],
                        "train_indices_sha256": result["train_indices_sha256"],
                        "validation_indices_sha256": result[
                            "validation_indices_sha256"
                        ],
                        "probe_entry_id": result["probe_entry_id"],
                        "source_commit": payload["source_commit"],
                        "source_archive_sha256": payload[
                            "source_archive_sha256"
                        ],
                        "environment_sha256": payload["environment_sha256"],
                        "source_shard_environment_sha256": payload.get(
                            "source_shard_environment_sha256",
                            payload["environment_sha256"],
                        ),
                        "execution_environment_sha256": payload.get(
                            "execution_environment_sha256",
                            payload["environment_sha256"],
                        ),
                    },
                },
                canonical=True,
            )
        elif name == "validation.json":
            atomic_write_json(
                path,
                {
                    "official_test_read": False,
                    "validation_indices_sha256": result[
                        "validation_indices_sha256"
                    ],
                    "records": [
                        {
                            "epoch": result.get("epochs_completed", 1),
                            "loss": result["final_validation_loss"],
                            "accuracy": result["final_validation_accuracy"],
                        }
                    ],
                },
                canonical=True,
            )
        elif name.endswith(".json"):
            atomic_write_json(
                path,
                {"official_test_read": False, "artifact": name},
                canonical=True,
            )
        elif name.endswith(".csv"):
            atomic_write_csv(path, ["value"], [{"value": 1}])
        else:
            atomic_write_bytes(path, b"checkpoint")


def _fake_runtime_executor(**kwargs: Any) -> dict[str, Any]:
    stage = kwargs["stage"]
    payload = kwargs["payload"]
    shard = Path(kwargs["shard_dir"])
    entry_id = payload["entry_id"]
    common = {
        "status": "complete",
        "entry_id": entry_id,
        "official_test_read": False,
    }
    if stage == "assets":
        result = {
            **common,
            "initialization_checkpoint_sha256": "1" * 64,
            "initialization_tensor_sha256": "2" * 64,
        }
    elif stage == "fixed_tk_gradient_security":
        result = {**common, "passed": True}
    elif stage == "optimizer_probe":
        result = {
            **common,
            "probe_stable": True,
            "used_batches": 32,
        }
    elif stage == "rho_canary_core":
        center_conv, center_dense = 3e-3, 1e-2
        cells = []
        pairs = core_grid(center_conv, center_dense)
        for rho_conv, rho_dense in pairs:
            cell_id = orchestration.rho_cell_id(
                _spec().study_id,
                payload["surface_id"],
                rho_conv,
                rho_dense,
            )
            # One rejected corner verifies that selection still receives the
            # complete declared 3x3 grid.
            safety_clean = (rho_conv, rho_dense) != pairs[-1]
            cells.append(
                {
                    "cell_id": cell_id,
                    "rho_conv": rho_conv,
                    "rho_dense": rho_dense,
                    "status": "complete" if safety_clean else "failed",
                    "safety_clean": safety_clean,
                    "safety_failure": (
                        None if safety_clean else "projection_efficiency"
                    ),
                }
            )
        result = {
            **common,
            "surface_id": payload["surface_id"],
            "safe_center": {
                "rho_conv": center_conv,
                "rho_dense": center_dense,
                "cell_id": orchestration.rho_cell_id(
                    _spec().study_id,
                    payload["surface_id"],
                    center_conv,
                    center_dense,
                ),
            },
            "center_reused_as_core": True,
            "cells": cells,
        }
    elif stage == "preflight_canary":
        result = {
            **common,
            "canonical_stage": False,
            "preflight_passed": True,
            "cell": {
                "completed_steps": 640,
                "expected_steps": 640,
                "safety_clean": True,
            },
            "benchmark": {
                "elapsed_seconds": 1.0,
                "successful_steps_per_second": 640.0,
                "cuda_peak_memory_allocated_bytes": 1,
                "cuda_peak_memory_reserved_bytes": 1,
            },
        }
    elif stage in {
        "rho_core_candidates",
        "rho_extension_candidates",
    }:
        rho_conv = float(payload["rho_conv"])
        rho_dense = float(payload["rho_dense"])
        centered = rho_conv == 3e-3 and rho_dense == 1e-2
        expansion_surface = (
            payload["surface_id"] == "conv2_legacy_v4_c0p25--adam"
        )
        expansion_best = (
            expansion_surface
            and rho_conv == pytest.approx(9e-3)
            and rho_dense == pytest.approx(1e-2)
        )
        result = {
            **common,
            "admissible": True,
            "completed_steps": 10_314,
            "epochs_completed": 3,
            "final_validation_accuracy": (
                0.94 if expansion_best else 0.93 if centered else 0.91
            ),
            "final_validation_loss": (
                0.05 if expansion_best else 0.10 if centered else 0.20
            ),
            "best_validation_loss": (
                0.05 if expansion_best else 0.10 if centered else 0.20
            ),
            "best_validation_epoch": 3,
            "inclusive_90_percent_accuracy_gate_passed": True,
            "safety": {"median_projection_efficiency": 0.80},
            "median_projection_efficiency": 0.80,
            "achieved_updates_by_parameter": {
                "ConvWeight_0": {
                    "step_count": 10_314,
                    "achieved_relative_update_median": 1e-3,
                }
            },
            "raw_learning_rates_by_parameter": {
                "ConvWeight_0": 1e-3,
                "Bias_0": 1e-3,
                "DenseWeight_0": 2e-3,
            },
            "study_id": _spec().study_id,
            "config_sha256": _spec().config_sha256,
            "cell_id": payload["cell_id"],
            "rho_conv": rho_conv,
            "rho_dense": rho_dense,
            "initialization_checkpoint_sha256": "1" * 64,
            "initialization_tensor_sha256": "2" * 64,
            "train_indices_sha256": "3" * 64,
            "validation_indices_sha256": "4" * 64,
            "probe_entry_id": payload["probe_entry_id"],
        }
    elif stage == "rho_canary_extension":
        cell = copy.deepcopy(payload["cell"])
        result = {
            **common,
            "cells": [
                {
                    **cell,
                    "status": "complete",
                    "safety_clean": True,
                }
            ],
        }
    elif stage == "long_confirm":
        result = {
            **common,
            "admissible": True,
            "completed_steps": payload["expected_steps"],
            "epochs_completed": payload["epochs"],
            "final_validation_accuracy": 0.95,
            "final_validation_loss": 0.08,
            "best_validation_loss": 0.07,
            "best_validation_epoch": payload["epochs"] - 1,
            "safety": {"median_projection_efficiency": 0.82},
            "median_projection_efficiency": 0.82,
            "achieved_updates_by_parameter": {
                "ConvWeight_0": {
                    "step_count": payload["expected_steps"],
                    "achieved_relative_update_median": 8e-4,
                }
            },
            "raw_learning_rates_by_parameter": {
                "ConvWeight_0": 1e-3,
                "Bias_0": 1e-3,
                "DenseWeight_0": 2e-3,
            },
            "study_id": _spec().study_id,
            "config_sha256": _spec().config_sha256,
            "cell_id": payload["cell_id"],
            "rho_conv": payload["rho_conv"],
            "rho_dense": payload["rho_dense"],
            "initialization_checkpoint_sha256": "1" * 64,
            "initialization_tensor_sha256": "2" * 64,
            "train_indices_sha256": "3" * 64,
            "validation_indices_sha256": "4" * 64,
            "probe_entry_id": payload["probe_entry_id"],
        }
    else:
        raise AssertionError(stage)
    _write_declared_outputs(shard, stage, entry_id, result, payload)
    return result


def _run_screen(tmp_path: Path, host: str) -> dict[str, Any]:
    return orchestration.run_host_screen(
        spec=_spec(),
        results_root=tmp_path / "results",
        host=host,
        data_root=tmp_path / "data",
        device="cuda",
        provenance=PROVENANCE,
        source_commit=SOURCE_COMMIT,
        source_archive_sha256=SOURCE_ARCHIVE,
        environment=ENVIRONMENT,
        runtime_executor=_fake_runtime_executor,
    )


def test_plan_has_exact_public_order_routes_and_zero_jobs() -> None:
    spec = _spec()
    akib = orchestration.plan_host_screen(spec, "akib")
    trex = orchestration.plan_host_screen(spec, "trex")
    assert akib["study_id"] == spec.study_id
    assert akib["stage_sequence"] == list(orchestration.PUBLIC_STAGE_SEQUENCE)
    assert akib["architecture"] == "conv1"
    assert trex["architecture"] == "conv2"
    assert len(akib["surface_ids"]) == len(trex["surface_ids"]) == 6
    assert all("--sgd" in value or "--adam" in value for value in akib["surface_ids"])
    assert akib["launched_jobs"] == trex["launched_jobs"] == 0
    assert akib["one_process_per_gpu"] is True


def test_shard_uses_semantic_config_hash_and_exact_clean_git_source(
    tmp_path: Path,
) -> None:
    spec = _spec()
    _root, shard = orchestration.create_host_shard(
        spec,
        tmp_path / "valid",
        "akib",
        provenance=PROVENANCE,
        source_commit=SOURCE_COMMIT,
        source_archive_sha256=SOURCE_ARCHIVE,
        environment=ENVIRONMENT,
    )
    descriptor = read_json(shard / "shard.resolved.json")
    assert descriptor["config_sha256"] == spec.config_sha256
    assert descriptor["resolved_config_file_sha256"] == sha256_file(
        shard / "study.resolved.json"
    )
    assert (
        descriptor["resolved_config_file_sha256"]
        != descriptor["config_sha256"]
    )

    invalid_sources = (
        {**PROVENANCE, "git_revision": "d" * 40},
        {**PROVENANCE, "dirty_source_digest": "e" * 64},
        {
            "git_revision": None,
            "dirty_source_digest": PROVENANCE["effective_code_fingerprint"],
            "effective_code_fingerprint": PROVENANCE[
                "effective_code_fingerprint"
            ],
        },
    )
    for index, invalid in enumerate(invalid_sources):
        with pytest.raises(RuntimeError, match="clean Git checkout"):
            orchestration.create_host_shard(
                spec,
                tmp_path / f"invalid-{index}",
                "akib",
                provenance=invalid,
                source_commit=SOURCE_COMMIT,
                source_archive_sha256=SOURCE_ARCHIVE,
                environment=ENVIRONMENT,
            )


def test_stage_worker_rejects_wrong_clean_commit_with_same_fingerprint(
    tmp_path: Path,
) -> None:
    spec = _spec()
    _root, shard = orchestration.create_host_shard(
        spec,
        tmp_path / "results",
        "akib",
        provenance=PROVENANCE,
        source_commit=SOURCE_COMMIT,
        source_archive_sha256=SOURCE_ARCHIVE,
        environment=ENVIRONMENT,
    )
    _manifest, manifest_path = orchestration.publish_screen_stage_manifest(
        shard, "audit", provenance=PROVENANCE
    )
    wrong_commit = {**PROVENANCE, "git_revision": "d" * 40}
    with pytest.raises(RuntimeError, match="clean Git checkout"):
        orchestration.execute_screen_manifest_entry(
            shard_dir=shard,
            manifest_path=manifest_path,
            entry_index=0,
            data_root=tmp_path / "data",
            device="cpu",
            provenance=wrong_commit,
        )


def test_host_pipeline_is_resumable_and_selection_keeps_failed_canary_cells(
    tmp_path: Path,
) -> None:
    first = _run_screen(tmp_path, "akib")
    assert first["status"] == "complete_through_select_final"
    second = _run_screen(tmp_path, "akib")
    assert all(
        stage["resumed_count"] == stage["entry_count"]
        for stage in second["stages"]
    )

    shard = Path(first["shard_dir"])
    preflight = read_json(
        shard / "preflight" / "adam-center-ours" / "receipt.json"
    )
    assert preflight["execution_passed"] is True
    assert preflight["expected_steps"] == 640
    assert not (
        shard / "stages" / "preflight_canary" / "manifest.json"
    ).exists()
    selection = read_json(
        shard
        / "stages"
        / "select_core"
        / "entries"
        / "conv1_baseline_v1_c1--sgd"
        / "result.json"
    )
    assert selection["status"] == "selected"
    assert len(selection["evaluated_candidates"]) == 9
    assert sum(
        candidate["admissible"] is False
        for candidate in selection["evaluated_candidates"]
    ) == 1
    assert selection["selected"]["rho_conv"] == pytest.approx(3e-3)
    assert selection["selected"]["rho_dense"] == pytest.approx(1e-2)

    extension_manifest = read_json(
        shard / "stages" / "rho_canary_extension" / "manifest.json"
    )
    assert len(extension_manifest["entries"]) == 6
    assert all(
        entry["payload"]["no_op"] is True
        for entry in extension_manifest["entries"]
    )


def test_merge_verifies_source_and_aggregate_runs_long_sequentially(
    tmp_path: Path,
) -> None:
    akib = _run_screen(tmp_path, "akib")
    trex = _run_screen(tmp_path, "trex")
    aggregate, receipt = orchestration.merge_screen_shards(
        tmp_path / "aggregate",
        akib_shard=akib["shard_dir"],
        trex_shard=trex["shard_dir"],
    )
    assert receipt["staged_source"]["commit"] == SOURCE_COMMIT
    assert receipt["staged_source"]["archive_sha256"] == SOURCE_ARCHIVE
    assert receipt["staged_source"]["effective_code_fingerprint"] == "b" * 64

    complete = orchestration.run_aggregate_confirmations(
        aggregate_dir=aggregate,
        data_root=tmp_path / "data",
        device="cuda",
        provenance=PROVENANCE,
        runtime_executor=_fake_runtime_executor,
    )
    assert complete["execution_host"] == "trex"
    assert complete["stages"][0]["entry_count"] == 12
    assert complete["stages"][0]["workers"] == 1
    long_manifest = read_json(
        aggregate / "stages" / "long_confirm" / "manifest.json"
    )
    trex_descriptor = read_json(
        aggregate / "shards" / "trex" / "shard.resolved.json"
    )
    for entry in long_manifest["entries"]:
        payload = entry["payload"]
        source_descriptor = read_json(
            aggregate / payload["source_shard"] / "shard.resolved.json"
        )
        assert payload["source_commit"] == source_descriptor[
            "staged_source"
        ]["commit"]
        assert payload["source_archive_sha256"] == source_descriptor[
            "staged_source"
        ]["archive_sha256"]
        assert payload[
            "source_shard_environment_sha256"
        ] == source_descriptor["environment_sha256"]
        assert payload[
            "execution_environment_sha256"
        ] == trex_descriptor["environment_sha256"]
        run_spec = read_json(
            aggregate
            / "stages"
            / "long_confirm"
            / "entries"
            / entry["entry_id"]
            / "run_spec.json"
        )
        provenance = run_spec["provenance"]
        assert provenance["source_commit"] == SOURCE_COMMIT
        assert provenance["source_archive_sha256"] == SOURCE_ARCHIVE
        assert provenance[
            "source_shard_environment_sha256"
        ] == source_descriptor["environment_sha256"]
        assert provenance[
            "execution_environment_sha256"
        ] == trex_descriptor["environment_sha256"]
    final = read_json(
        aggregate
        / "stages"
        / "finalize"
        / "entries"
        / "finalization"
        / "result.json"
    )
    assert final["selected_surface_count"] == 12
    assert len(final["rows"]) == 12
    assert final["official_test_read"] is False
    baseline = next(
        row
        for row in final["rows"]
        if row["surface_id"] == "conv1_baseline_v1_c1--sgd"
    )
    assert baseline["selected_rho_targets"] == {
        "rho_conv": pytest.approx(3e-3),
        "rho_dense": pytest.approx(1e-2),
        "cell_id": baseline["selection"]["cell_id"],
    }
    assert set(baseline["raw_learning_rates_by_parameter"]) == {
        "ConvWeight_0",
        "Bias_0",
        "DenseWeight_0",
    }
    assert baseline["validation_metrics"]["final_accuracy"] == pytest.approx(
        0.93
    )
    assert baseline["validation_metrics"]["records"]
    assert baseline["safety_diagnostics"]["median_projection_efficiency"] == (
        pytest.approx(0.80)
    )
    assert baseline["median_projection_efficiency"] == pytest.approx(0.80)
    assert baseline["achieved_updates_by_parameter"]["ConvWeight_0"][
        "step_count"
    ] == 10_314
    assert set(baseline["checkpoints"]) == {"best_validation", "final"}
    for field in (
        "config_sha256",
        "source_commit",
        "source_archive_sha256",
        "source_shard_environment_sha256",
        "train_indices_sha256",
        "validation_indices_sha256",
        "initialization_checkpoint_sha256",
        "initialization_tensor_sha256",
        "probe_result_sha256",
        "candidate_result_sha256",
        "candidate_run_spec_sha256",
        "candidate_validation_sha256",
        "selection_stage_manifest_sha256",
        "candidate_stage_manifest_sha256",
        "merge_receipt_sha256",
    ):
        assert baseline["provenance"][field]
    assert baseline["long_confirmation"]["zero_work"] is False
    assert baseline["long_confirmation"]["result"][
        "final_validation_accuracy"
    ] == pytest.approx(0.95)
    assert baseline["long_confirmation"]["run_provenance"][
        "execution_environment_sha256"
    ] == trex_descriptor["environment_sha256"]

    expanded = next(
        row
        for row in final["rows"]
        if row["surface_id"] == "conv2_legacy_v4_c0p25--adam"
    )
    assert expanded["boundary_status"] == "resolved_after_single_expansion"
    assert expanded["boundary_diagnostics"]["core_selection_status"] == (
        "needs_expansion"
    )
    assert expanded["boundary_diagnostics"]["core_expansion_axes"] == [
        "rho_conv"
    ]


def test_merge_rejects_archive_mismatch_and_copied_artifact_tamper(
    tmp_path: Path,
) -> None:
    akib = _run_screen(tmp_path, "akib")
    trex = _run_screen(tmp_path, "trex")
    trex_descriptor_path = Path(trex["shard_dir"]) / "shard.resolved.json"
    trex_descriptor = read_json(trex_descriptor_path)
    trex_descriptor["staged_source"]["archive_sha256"] = "d" * 64
    atomic_write_json(trex_descriptor_path, trex_descriptor, canonical=True)
    # The immutable upstream hash may reject the tamper before the explicit
    # cross-host source comparison; either fail-closed path is correct.
    with pytest.raises((RuntimeError, ValueError)):
        orchestration.merge_screen_shards(
            tmp_path / "bad-aggregate",
            akib_shard=akib["shard_dir"],
            trex_shard=trex["shard_dir"],
        )


def test_aggregate_rejects_non_exact_or_dirty_source(tmp_path: Path) -> None:
    akib = _run_screen(tmp_path, "akib")
    trex = _run_screen(tmp_path, "trex")
    aggregate, _receipt = orchestration.merge_screen_shards(
        tmp_path / "aggregate",
        akib_shard=akib["shard_dir"],
        trex_shard=trex["shard_dir"],
    )
    invalid_sources = (
        {**PROVENANCE, "effective_code_fingerprint": "e" * 64},
        {**PROVENANCE, "git_revision": "d" * 40},
        {**PROVENANCE, "dirty_source_digest": "f" * 64},
    )
    for invalid in invalid_sources:
        with pytest.raises(RuntimeError, match="clean Git checkout"):
            orchestration.publish_aggregate_stage_manifest(
                aggregate, "long_confirm", provenance=invalid
            )


def test_default_executor_adapter_puts_entry_id_inside_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, Any] = {}

    def dispatcher(**kwargs: Any) -> dict[str, Any]:
        observed.update(kwargs)
        return {"status": "complete"}

    monkeypatch.setattr(
        "experiments.mnist_conv.perfectdiode_hparam_runtime."
        "execute_perfectdiode_stage_entry",
        dispatcher,
    )
    result = orchestration._default_runtime_executor(
        study=_spec().data,
        shard_dir=Path("/tmp/example"),
        stage="assets",
        payload={"entry_id": "conv1-assets"},
        data_root=Path("/tmp/data"),
        device="cuda",
        download=False,
    )
    assert result["status"] == "complete"
    assert observed["payload"]["entry_id"] == "conv1-assets"
    assert "entry_id" not in {
        key for key in observed if key != "payload"
    }


def test_rho_cell_identity_matches_runtime() -> None:
    from experiments.mnist_conv.perfectdiode_hparam_runtime import (
        rho_cell_id as runtime_cell_id,
    )

    study_id = _spec().study_id
    expected = orchestration.rho_cell_id(
        study_id, "conv1_baseline_v1_c1--adam", 3e-3, 1e-2
    )
    assert expected == runtime_cell_id(
        study_id=study_id,
        surface_id="conv1_baseline_v1_c1--adam",
        rho_conv=3e-3,
        rho_dense=1e-2,
    )
    assert expected.startswith("pdcell_")
    assert len(expected) == 71
