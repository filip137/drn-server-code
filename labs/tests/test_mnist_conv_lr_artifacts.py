from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from experiments.mnist_conv.io import atomic_write_json
from experiments.mnist_conv.lr_artifacts import (
    LRArtifactConflictError,
    LRArtifactError,
    artifact_record,
    build_stage_manifest,
    entry_is_complete,
    load_stage_manifest,
    publish_entry_completion,
    publish_stage_completion,
    publish_stage_manifest,
    stage_entry,
    stage_is_complete,
    validate_entry_completion,
    validate_stage_completion,
)


STUDY_ID = "lrstudy_" + "1" * 64
PROVENANCE = {
    "git_revision": "2" * 40,
    "dirty_source_digest": None,
    "effective_code_fingerprint": "3" * 64,
}


def _write(path: Path, data: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def _plan(tmp_path: Path, *, count: int = 2):
    study_dir = tmp_path / "study"
    study_dir.mkdir()
    config = study_dir / "study.resolved.json"
    atomic_write_json(config, {"schema_version": "mnist-conv-lr-study/v1"}, canonical=True)
    entries = []
    for index in range(count):
        entries.append(
            stage_entry(
                index,
                f"row-{index}",
                completion_path=f"probe/row-{index}/complete.json",
                outputs=[
                    f"probe/row-{index}/diagnostics.jsonl",
                    f"probe/row-{index}/summary.json",
                ],
                payload={"row_id": f"row-{index}"},
            )
        )
    manifest = build_stage_manifest(
        study_dir=study_dir,
        study_id=STUDY_ID,
        study_config_path=config,
        code_provenance=PROVENANCE,
        stage_name="probe",
        entries=entries,
    )
    manifest_path = study_dir / "probe" / "manifest.json"
    publish_stage_manifest(manifest_path, manifest, study_dir=study_dir)
    return study_dir, manifest_path, manifest


def _write_outputs(study_dir: Path, manifest: dict, entry_index: int) -> None:
    for output in manifest["entries"][entry_index]["outputs"]:
        _write(study_dir / output, f"entry {entry_index}: {output}\n".encode())


def test_manifest_is_canonical_immutable_and_revalidates_config(tmp_path: Path) -> None:
    study_dir, manifest_path, manifest = _plan(tmp_path)

    assert json.loads(manifest_path.read_text()) == manifest
    assert manifest_path.read_text() == json.dumps(
        manifest, sort_keys=True, separators=(",", ":")
    ) + "\n"
    assert "timestamp" not in manifest_path.read_text()
    assert load_stage_manifest(
        manifest_path,
        study_dir=study_dir,
        expected_study_id=STUDY_ID,
        expected_stage_name="probe",
    ) == manifest
    assert publish_stage_manifest(manifest_path, manifest, study_dir=study_dir) == manifest_path

    changed = copy.deepcopy(manifest)
    changed["entries"][0]["payload"]["row_id"] = "different"
    with pytest.raises(LRArtifactConflictError, match="Expected stage_manifest"):
        publish_stage_manifest(manifest_path, changed, study_dir=study_dir)


def test_manifest_rejects_upstream_or_config_hash_drift(tmp_path: Path) -> None:
    study_dir = tmp_path / "study"
    study_dir.mkdir()
    config = _write(study_dir / "study.resolved.json", b"{}\n")
    upstream = _write(study_dir / "probe" / "complete.json", b"probe\n")
    entry = stage_entry(
        0,
        "range-row",
        completion_path="range/range-row/complete.json",
        outputs=["range/range-row/summary.json"],
    )
    manifest = build_stage_manifest(
        study_dir=study_dir,
        study_id=STUDY_ID,
        study_config_path=config,
        code_provenance=PROVENANCE,
        stage_name="range",
        entries=[entry],
        upstream_paths=[upstream],
    )
    manifest_path = study_dir / "range" / "manifest.json"
    publish_stage_manifest(manifest_path, manifest, study_dir=study_dir)

    upstream.write_bytes(b"changed\n")
    with pytest.raises(LRArtifactError, match="Expected artifacts to be the current artifact hashes"):
        load_stage_manifest(manifest_path, study_dir=study_dir)

    upstream.write_bytes(b"probe\n")
    config.write_bytes(b'{"changed":true}\n')
    with pytest.raises(LRArtifactError, match="Expected study config SHA-256"):
        load_stage_manifest(manifest_path, study_dir=study_dir)


def test_entry_marker_is_last_resumable_and_detects_corruption(tmp_path: Path) -> None:
    study_dir, manifest_path, manifest = _plan(tmp_path, count=1)
    marker = study_dir / manifest["entries"][0]["completion_path"]

    assert entry_is_complete(
        study_dir=study_dir, manifest_path=manifest_path, entry_id="row-0"
    ) is False
    with pytest.raises(LRArtifactError, match="Expected artifact to be an existing regular file"):
        publish_entry_completion(
            study_dir=study_dir, manifest_path=manifest_path, entry_id="row-0"
        )
    assert not marker.exists()

    _write_outputs(study_dir, manifest, 0)
    assert publish_entry_completion(
        study_dir=study_dir, manifest_path=manifest_path, entry_id="row-0"
    ) == marker
    completion = validate_entry_completion(
        study_dir=study_dir, manifest_path=manifest_path, entry_id="row-0"
    )
    assert [record["path"] for record in completion["outputs"]] == manifest["entries"][0]["outputs"]
    assert all(set(record) == {"path", "sha256", "bytes"} for record in completion["outputs"])
    assert entry_is_complete(
        study_dir=study_dir, manifest_path=manifest_path, entry_id="row-0"
    ) is True

    (study_dir / manifest["entries"][0]["outputs"][0]).write_text("corrupt\n")
    with pytest.raises(LRArtifactError, match="Expected entry_completion"):
        entry_is_complete(
            study_dir=study_dir, manifest_path=manifest_path, entry_id="row-0"
        )


def test_stage_completion_requires_all_entries_and_revalidates_them(tmp_path: Path) -> None:
    study_dir, manifest_path, manifest = _plan(tmp_path)
    for index in range(2):
        _write_outputs(study_dir, manifest, index)
    publish_entry_completion(
        study_dir=study_dir, manifest_path=manifest_path, entry_id="row-0"
    )
    aggregate = manifest_path.parent / "complete.json"
    with pytest.raises(LRArtifactError, match="Expected entry_completion"):
        publish_stage_completion(study_dir=study_dir, manifest_path=manifest_path)
    assert not aggregate.exists()

    publish_entry_completion(
        study_dir=study_dir, manifest_path=manifest_path, entry_id="row-1"
    )
    assert publish_stage_completion(
        study_dir=study_dir, manifest_path=manifest_path
    ) == aggregate
    completion = validate_stage_completion(
        study_dir=study_dir, manifest_path=manifest_path
    )
    assert [entry["entry_id"] for entry in completion["entries"]] == ["row-0", "row-1"]
    assert stage_is_complete(study_dir=study_dir, manifest_path=manifest_path) is True

    (study_dir / manifest["entries"][1]["outputs"][1]).write_text("tampered\n")
    with pytest.raises(LRArtifactError, match="Expected entry_completion"):
        stage_is_complete(study_dir=study_dir, manifest_path=manifest_path)


def test_rejects_nonportable_duplicate_and_symlink_artifacts(tmp_path: Path) -> None:
    study_dir, _, _ = _plan(tmp_path, count=1)
    with pytest.raises(LRArtifactError, match="Expected entry.outputs to be sorted unique"):
        stage_entry(
            0,
            "bad",
            completion_path="bad/complete.json",
            outputs=["bad/z.json", "bad/a.json"],
        )
    outside = _write(tmp_path / "outside", b"outside")
    link = study_dir / "link"
    link.symlink_to(outside)
    with pytest.raises(LRArtifactError, match="Expected artifact to be a path contained"):
        artifact_record(study_dir, link)
