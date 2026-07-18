from __future__ import annotations

import copy
import json
import os
import shutil
from pathlib import Path

import numpy as np
import pytest
import torch

from experiments.mnist_conv.backend import ExecutionContext
from experiments.mnist_conv.identity import run_fingerprint, sha256_file
from experiments.mnist_conv.layout import ResultLayout
from experiments.mnist_conv.runner import (
    ExecutionPolicy,
    InvalidBundleError,
    RetryExhaustedError,
    execute_run,
    validate_bundle,
)
from experiments.mnist_conv.specs import RunSpec


REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = REPO_ROOT / "configs/conv/run_v1.diagnostic.example.json"
PROVENANCE = {
    "git_revision": "1" * 40,
    "dirty_source_digest": "2" * 64,
    "effective_code_fingerprint": "3" * 64,
}


def _run_value(*, replicate_id: str | None = None) -> dict:
    value = json.loads(EXAMPLE.read_text())
    value["replicate_id"] = replicate_id
    value["run"]["training"]["epochs"] = 1
    return value


def _checkpoint_payload(value: float = 1.0) -> dict:
    return {
        "format": "drn.function.parameters",
        "version": 1,
        "schema": [{
            "name": "weight 0",
            "type": "model.variable.parameter.Weight",
            "shape": [2, 2],
            "dtype": "torch.float32",
        }],
        "states": [torch.full((2, 2), value, dtype=torch.float32)],
    }


def _write_pair(output_dir: Path, role: str, *, checkpoint_value: float = 1.0, npz_value: float = 1.0) -> None:
    checkpoint_name = "best_model.pt" if role == "best" else "final_model.pt"
    npz_name = "weights_best.npz" if role == "best" else "weights_final.npz"
    torch.save(_checkpoint_payload(checkpoint_value), output_dir / checkpoint_name)
    np.savez(
        output_dir / npz_name,
        weight_0=np.full((2, 2), npz_value, dtype=np.float32),
        param_names=np.asarray(["weight_0"]),
        param_types=np.asarray(["Weight"]),
        param_shapes_json=np.asarray(json.dumps([[2, 2]])),
        metadata_json=np.asarray("{}"),
    )


def _write_complete_backend_output(
    output_dir: Path,
    *,
    npz_value: float = 1.0,
    history_value: float = 0.5,
) -> dict:
    _write_pair(output_dir, "best", npz_value=npz_value)
    _write_pair(output_dir, "final", npz_value=npz_value)
    for filename, value in (
        ("loss_train.npy", history_value),
        ("loss_test.npy", 0.4),
        ("accuracy_train.npy", 0.8),
        ("accuracy_test.npy", 0.9),
    ):
        np.save(output_dir / filename, np.asarray([value], dtype=np.float64))
    (output_dir / "events.out.tfevents.test").write_bytes(b"event")
    (output_dir / "metrics.json").write_text(json.dumps({
        "best_epoch": 1,
        "best_test_accuracy": 0.9,
        "final_test_accuracy": 0.9,
        "final_train_loss": history_value,
        "final_test_loss": 0.4,
    }))
    return {"learning_rate": [[0.01, 0.01, 0.01]]}


class CompleteBackend:
    def __init__(self, *, npz_value: float = 1.0, history_value: float = 0.5):
        self.npz_value = npz_value
        self.history_value = history_value

    def __call__(self, *, output_dir, **kwargs):
        return _write_complete_backend_output(
            output_dir,
            npz_value=self.npz_value,
            history_value=self.history_value,
        )


def _execute(tmp_path: Path, spec: RunSpec, backend, *, policy: ExecutionPolicy | None = None):
    layout = ResultLayout(tmp_path / "results")
    result = execute_run(
        spec,
        ExecutionContext(tmp_path / "data", device="cpu"),
        layout,
        PROVENANCE,
        backend=backend,
        policy=policy,
    )
    return layout, result


def _latest_status(layout: ResultLayout, run_id: str) -> dict:
    paths = sorted(layout.attempt_root(run_id).glob("*/status.json"))
    assert paths
    return json.loads(paths[-1].read_text())


def test_npz_values_must_exactly_match_checkpoint_tensors(tmp_path):
    spec = RunSpec.from_dict(_run_value())
    layout = ResultLayout(tmp_path / "results")
    run_id = run_fingerprint(spec, PROVENANCE)
    with pytest.raises(InvalidBundleError, match="values to exactly match"):
        execute_run(
            spec,
            ExecutionContext(tmp_path / "data"),
            layout,
            PROVENANCE,
            backend=CompleteBackend(npz_value=2.0),
        )
    assert layout.find_run_dir(run_id) is None
    assert not layout.claim_path(run_id).exists()
    assert _latest_status(layout, run_id)["state"] == "failed"


def test_npz_rejects_undeclared_arrays_and_history_rejects_lr_drift(tmp_path):
    layout = ResultLayout(tmp_path / "results")

    extra_spec = RunSpec.from_dict(_run_value(replicate_id="npz-extra"))

    def extra_backend(*, output_dir, **kwargs):
        result = _write_complete_backend_output(output_dir)
        source = np.load(output_dir / "weights_best.npz", allow_pickle=False)
        arrays = {name: source[name] for name in source.files}
        source.close()
        np.savez(output_dir / "weights_best.npz", **arrays, undeclared=np.ones(1))
        return result

    with pytest.raises(InvalidBundleError, match="exactly the declared parameters"):
        execute_run(
            extra_spec,
            ExecutionContext(tmp_path / "data"),
            layout,
            PROVENANCE,
            backend=extra_backend,
        )

    lr_spec = RunSpec.from_dict(_run_value(replicate_id="lr-drift"))

    def lr_backend(*, output_dir, **kwargs):
        result = _write_complete_backend_output(output_dir)
        result["learning_rate"] = [[0.02, 0.01, 0.01]]
        return result

    with pytest.raises(InvalidBundleError, match="learning rates"):
        execute_run(
            lr_spec,
            ExecutionContext(tmp_path / "data"),
            layout,
            PROVENANCE,
            backend=lr_backend,
        )


def test_manifest_records_are_complete_and_artifact_corruption_is_rejected(tmp_path):
    spec = RunSpec.from_dict(_run_value())
    _, result = _execute(tmp_path, spec, CompleteBackend())
    manifest_path = result.run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["artifacts"] = manifest["artifacts"][:-1]
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(InvalidBundleError, match="artifact records to be complete"):
        validate_bundle(result.run_dir, result.run_id)


def test_completed_bundle_is_relocatable_without_rewriting_serialized_paths(tmp_path):
    spec = RunSpec.from_dict(_run_value(replicate_id="relocation"))
    _, result = _execute(tmp_path, spec, CompleteBackend())
    relocated = tmp_path / "relocated" / "runs" / result.run_dir.name
    relocated.parent.mkdir(parents=True)
    shutil.copytree(result.run_dir, relocated)

    assert validate_bundle(relocated, result.run_id)["state"] == "complete"
    metrics = json.loads((relocated / "metrics.json").read_text())
    assert all(
        not Path(value).is_absolute()
        for key, value in metrics.items()
        if key.endswith("_path") and isinstance(value, str)
    )


def test_backend_dangling_paths_are_replaced_with_published_artifacts(tmp_path):
    spec = RunSpec.from_dict(_run_value(replicate_id="canonical-metric-paths"))

    def backend(*, output_dir, **kwargs):
        result = _write_complete_backend_output(output_dir)
        (output_dir / "config.json").write_text(json.dumps({"temporary": True}))
        metrics_path = output_dir / "metrics.json"
        metrics = json.loads(metrics_path.read_text())
        metrics.update({
            "run_dir": str(output_dir.resolve()),
            "config_path": str((output_dir / "config.json").resolve()),
            "best_checkpoint_path": str((output_dir / "best_model.pt").resolve()),
            "checkpoint_path": str((output_dir / "final_model.pt").resolve()),
            "weights_best_path": str((output_dir / "weights_best.npz").resolve()),
            "weights_final_path": str((output_dir / "weights_final.npz").resolve()),
            "history_paths": {
                "loss_train": str((output_dir / "loss_train.npy").resolve()),
                "loss_test": str((output_dir / "loss_test.npy").resolve()),
                "accuracy_train": str((output_dir / "accuracy_train.npy").resolve()),
                "accuracy_test": str((output_dir / "accuracy_test.npy").resolve()),
            },
        })
        metrics_path.write_text(json.dumps(metrics))
        return result

    _, result = _execute(tmp_path, spec, backend)
    metrics = json.loads((result.run_dir / "metrics.json").read_text())
    expected_paths = {
        "config_path": "config.resolved.json",
        "best_checkpoint_path": "checkpoints/best.pt",
        "checkpoint_path": "checkpoints/final.pt",
        "weights_best_path": "weights/best.npz",
        "weights_final_path": "weights/final.npz",
        "history_path": "history.csv",
    }
    assert metrics["run_dir"] == "."
    assert "history_paths" not in metrics
    assert {key: metrics[key] for key in expected_paths} == expected_paths
    assert all((result.run_dir / relative).is_file() for relative in expected_paths.values())
    assert not (result.run_dir / "config.json").exists()
    assert not list(result.run_dir.glob("*.npy"))


def test_npz_metadata_rejects_absolute_paths(tmp_path):
    spec = RunSpec.from_dict(_run_value(replicate_id="npz-absolute-metadata"))
    layout = ResultLayout(tmp_path / "results")
    run_id = run_fingerprint(spec, PROVENANCE)

    def backend(*, output_dir, **kwargs):
        result = _write_complete_backend_output(output_dir)
        source_path = output_dir / "weights_best.npz"
        with np.load(source_path, allow_pickle=False) as source:
            arrays = {name: source[name] for name in source.files}
        arrays["metadata_json"] = np.asarray(json.dumps({"source_path": "/host/private/model.pt"}))
        np.savez(source_path, **arrays)
        return result

    with pytest.raises(InvalidBundleError, match="NPZ metadata_json.*absolute path"):
        execute_run(
            spec,
            ExecutionContext(tmp_path / "data"),
            layout,
            PROVENANCE,
            backend=backend,
        )
    assert layout.find_run_dir(run_id) is None
    assert _latest_status(layout, run_id)["state"] == "failed"
    assert not layout.claim_path(run_id).exists()


def test_recorded_artifact_corruption_and_symlinks_are_rejected(tmp_path):
    first = RunSpec.from_dict(_run_value(replicate_id="corrupt"))
    _, result = _execute(tmp_path, first, CompleteBackend())
    (result.run_dir / "metrics.json").write_bytes(b"corrupt")
    with pytest.raises(InvalidBundleError, match="artifact records to be complete"):
        validate_bundle(result.run_dir, result.run_id)

    second = RunSpec.from_dict(_run_value(replicate_id="symlink"))
    _, result = _execute(tmp_path, second, CompleteBackend())
    os.symlink(result.run_dir / "metrics.json", result.run_dir / "logs" / "linked.json")
    with pytest.raises(InvalidBundleError, match="not symlinks"):
        validate_bundle(result.run_dir, result.run_id)


def test_setup_failure_always_cleans_claim_and_records_failure(tmp_path, monkeypatch):
    spec = RunSpec.from_dict(_run_value())
    layout = ResultLayout(tmp_path / "results")
    run_id = run_fingerprint(spec, PROVENANCE)

    def fail_setup(*args, **kwargs):
        raise RuntimeError("setup exploded")

    monkeypatch.setattr("experiments.mnist_conv.runner.build_engine_config", fail_setup)
    with pytest.raises(RuntimeError, match="setup exploded"):
        execute_run(
            spec,
            ExecutionContext(tmp_path / "data"),
            layout,
            PROVENANCE,
            backend=pytest.fail,
        )
    assert not layout.claim_path(run_id).exists()
    assert _latest_status(layout, run_id)["state"] == "failed"


def test_stale_claim_recovery_and_retry_exhaustion(tmp_path):
    spec = RunSpec.from_dict(_run_value(replicate_id="stale"))
    layout = ResultLayout(tmp_path / "results")
    run_id = run_fingerprint(spec, PROVENANCE)
    old_attempt = "20000101T000000.000000Z-1-stale"
    old_dir = layout.attempt_dir(run_id, old_attempt)
    old_dir.mkdir(parents=True)
    interrupted_staging = old_dir / ".staging"
    interrupted_staging.mkdir()
    (interrupted_staging / "partial-checkpoint.pt").write_bytes(b"interrupted")
    stale = {
        "schema_version": "mnist-conv-status/v1",
        "run_id": run_id,
        "attempt_id": old_attempt,
        "state": "running",
        "updated_at": "2000-01-01T00:00:00Z",
        "host": "old-host",
        "pid": 1,
    }
    (old_dir / "status.json").write_text(json.dumps(stale))
    layout.claim_path(run_id).write_text(json.dumps(stale))
    result = execute_run(
        spec,
        ExecutionContext(tmp_path / "data"),
        layout,
        PROVENANCE,
        backend=CompleteBackend(),
        policy=ExecutionPolicy(max_attempts=3, reclaim_stale_after_seconds=1),
    )
    assert result.status == "complete"
    assert json.loads((old_dir / "status.json").read_text())["state"] == "stale"
    assert (interrupted_staging / "partial-checkpoint.pt").read_bytes() == b"interrupted"
    assert layout.find_run_dir(run_id) == result.run_dir
    assert list((layout.attempt_root(run_id) / "recovered_claims").glob("*.json"))

    invalid_spec = RunSpec.from_dict(_run_value(replicate_id="invalid-stale-claim"))
    invalid_id = run_fingerprint(invalid_spec, PROVENANCE)
    invalid_claim = layout.claim_path(invalid_id)
    invalid_claim.parent.mkdir(parents=True)
    invalid_claim.write_text("{truncated")
    os.utime(invalid_claim, (1, 1))
    invalid_result = execute_run(
        invalid_spec,
        ExecutionContext(tmp_path / "data"),
        layout,
        PROVENANCE,
        backend=CompleteBackend(),
        policy=ExecutionPolicy(max_attempts=2, reclaim_stale_after_seconds=1),
    )
    assert invalid_result.status == "complete"
    assert list((layout.attempt_root(invalid_id) / "recovered_claims").glob("invalid-*.json"))

    failed_spec = RunSpec.from_dict(_run_value(replicate_id="retry"))
    failed_id = run_fingerprint(failed_spec, PROVENANCE)

    def fail_backend(**kwargs):
        raise RuntimeError("backend failed")

    with pytest.raises(RuntimeError, match="backend failed"):
        execute_run(
            failed_spec,
            ExecutionContext(tmp_path / "data"),
            layout,
            PROVENANCE,
            backend=fail_backend,
            policy=ExecutionPolicy(max_attempts=1),
        )
    with pytest.raises(RetryExhaustedError):
        execute_run(
            failed_spec,
            ExecutionContext(tmp_path / "data"),
            layout,
            PROVENANCE,
            backend=pytest.fail,
            policy=ExecutionPolicy(max_attempts=1, retry_failed=True),
        )
    assert not layout.claim_path(failed_id).exists()


@pytest.mark.parametrize("failure", ["checksum", "format", "schema"])
def test_initialization_checkpoint_checksum_format_and_schema_are_validated(tmp_path, failure):
    layout = ResultLayout(tmp_path / "results")
    checkpoint_path = layout.root / "imports" / "initial.pt"
    checkpoint_path.parent.mkdir(parents=True)
    payload = _checkpoint_payload()
    if failure == "schema":
        payload["schema"][0]["shape"] = [3, 3]
    torch.save(payload, checkpoint_path)
    value = _run_value(replicate_id=f"init-{failure}")
    value["run"]["initialization"]["checkpoint"] = {
        "path": "imports/initial.pt",
        "sha256": ("0" * 64 if failure == "checksum" else sha256_file(checkpoint_path)),
        "format": ("legacy_tensor_list" if failure == "format" else "drn.function.parameters/v1"),
        "source_run_id": None,
        "role": "initialization",
    }
    spec = RunSpec.from_dict(value)
    with pytest.raises(InvalidBundleError):
        execute_run(
            spec,
            ExecutionContext(tmp_path / "data"),
            layout,
            PROVENANCE,
            backend=pytest.fail,
        )
    assert not layout.attempt_root(run_fingerprint(spec, PROVENANCE)).exists()


def test_initialization_source_run_and_role_must_match_path(tmp_path):
    source = RunSpec.from_dict(_run_value(replicate_id="source"))
    layout, source_result = _execute(tmp_path, source, CompleteBackend())
    source_checkpoint = source_result.run_dir / "checkpoints/best.pt"
    checkpoint_record = {
        "path": source_checkpoint.relative_to(layout.root).as_posix(),
        "sha256": sha256_file(source_checkpoint),
        "format": "drn.function.parameters/v1",
        "source_run_id": source_result.run_id,
        "role": "best",
    }

    valid_value = _run_value(replicate_id="target-valid")
    valid_value["run"]["initialization"]["checkpoint"] = checkpoint_record
    valid_target = RunSpec.from_dict(valid_value)

    def backend(*, context, output_dir, **kwargs):
        assert context.initialization_checkpoint_path == source_checkpoint
        return _write_complete_backend_output(output_dir)

    valid_result = execute_run(
        valid_target,
        ExecutionContext(tmp_path / "data"),
        layout,
        PROVENANCE,
        backend=backend,
    )
    assert valid_result.status == "complete"

    target_value = _run_value(replicate_id="target-invalid-role")
    target_value["run"]["initialization"]["checkpoint"] = {
        **checkpoint_record,
        "role": "final",
    }
    target = RunSpec.from_dict(target_value)
    with pytest.raises(InvalidBundleError, match="source-run role"):
        execute_run(
            target,
            ExecutionContext(tmp_path / "data"),
            layout,
            PROVENANCE,
            backend=pytest.fail,
        )


def test_nonfinite_backend_output_fails_without_publication(tmp_path):
    spec = RunSpec.from_dict(_run_value())
    layout = ResultLayout(tmp_path / "results")
    run_id = run_fingerprint(spec, PROVENANCE)

    def backend(*, output_dir, **kwargs):
        result = _write_complete_backend_output(output_dir)
        np.save(output_dir / "loss_train.npy", np.asarray([float("nan")]))
        return result

    with pytest.raises(InvalidBundleError, match="finite history"):
        execute_run(
            spec,
            ExecutionContext(tmp_path / "data"),
            layout,
            PROVENANCE,
            backend=backend,
        )
    assert layout.find_run_dir(run_id) is None
    assert _latest_status(layout, run_id)["state"] == "failed"
    assert not layout.claim_path(run_id).exists()


def test_absolute_public_json_path_outside_bundle_is_rejected(tmp_path):
    spec = RunSpec.from_dict(_run_value(replicate_id="absolute-path"))
    layout = ResultLayout(tmp_path / "results")
    run_id = run_fingerprint(spec, PROVENANCE)

    def backend(*, output_dir, **kwargs):
        result = _write_complete_backend_output(output_dir)
        metrics_path = output_dir / "metrics.json"
        metrics = json.loads(metrics_path.read_text())
        metrics["external_path"] = "/outside/result-store/file.pt"
        metrics_path.write_text(json.dumps(metrics))
        return result

    with pytest.raises(InvalidBundleError, match="absolute path"):
        execute_run(
            spec,
            ExecutionContext(tmp_path / "data"),
            layout,
            PROVENANCE,
            backend=backend,
        )
    assert layout.find_run_dir(run_id) is None
    assert _latest_status(layout, run_id)["state"] == "failed"
    assert not layout.claim_path(run_id).exists()


@pytest.mark.parametrize("mode", ["missing", "mismatched"])
def test_pruning_requires_valid_matching_best_artifacts(tmp_path, mode):
    spec = RunSpec.from_dict(_run_value(replicate_id=f"pruned-{mode}"))
    layout = ResultLayout(tmp_path / "results")
    run_id = run_fingerprint(spec, PROVENANCE)

    def backend(*, output_dir, **kwargs):
        if mode == "mismatched":
            _write_pair(output_dir, "best", checkpoint_value=1.0, npz_value=2.0)
        return {"status": "pruned", "epoch": 1}

    with pytest.raises(InvalidBundleError):
        execute_run(
            spec,
            ExecutionContext(tmp_path / "data"),
            layout,
            PROVENANCE,
            backend=backend,
        )
    assert layout.find_run_dir(run_id) is None
    assert _latest_status(layout, run_id)["state"] == "failed"
    assert not layout.claim_path(run_id).exists()


def test_pruning_with_valid_best_artifacts_is_terminal_but_unpublished(tmp_path):
    spec = RunSpec.from_dict(_run_value(replicate_id="pruned-valid"))

    def backend(*, output_dir, **kwargs):
        _write_pair(output_dir, "best")
        return {"status": "pruned", "epoch": 1}

    layout, result = _execute(tmp_path, spec, backend)
    assert result.status == "pruned"
    assert layout.find_run_dir(result.run_id) is None
    assert _latest_status(layout, result.run_id)["state"] == "pruned"
    assert not layout.claim_path(result.run_id).exists()
