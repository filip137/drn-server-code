from __future__ import annotations

import json
import shutil
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from experiments.mnist_conv import lr_stages
from experiments.mnist_conv.identity import sha256_file
from experiments.mnist_conv.lr_study_spec import LRStudySpec


REPO_ROOT = Path(__file__).resolve().parents[2]
V3_STUDY_CONFIG = (
    REPO_ROOT / "configs/conv/hardsigmoid_lr_relative_rho_sgd_bs16_v3.json"
)
REAL_STUDIES = REPO_ROOT / "simulation_results/conv_lr_protocol_20260719/lr_studies"


def _iter_strings(value: Any) -> Iterator[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for key, item in value.items():
            yield from _iter_strings(key)
            yield from _iter_strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from _iter_strings(item)


def _copy_minimal_completed_parent(source: Path, destination: Path) -> None:
    """Copy the files needed to revalidate the real probe/range/select chain."""

    required = {"study.resolved.json"}
    for stage in ("probe", "range", "select"):
        manifest_relative = f"stages/{stage}/manifest.json"
        manifest = json.loads((source / manifest_relative).read_text(encoding="utf-8"))
        required.update(
            {
                manifest_relative,
                f"stages/{stage}/complete.json",
                *(record["path"] for record in manifest["upstream_artifacts"]),
            }
        )
        for entry in manifest["entries"]:
            required.add(entry["completion_path"])
            required.update(entry["outputs"])

    for relative in sorted(required):
        source_path = source / relative
        assert source_path.is_file(), relative
        destination_path = destination / relative
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, destination_path)


def _copy_v2_evidence(source: Path, destination: Path, spec: LRStudySpec) -> None:
    evidence = spec.data["rescue"]["evidence_study"]
    required = {
        "study.resolved.json",
        "stages/select/entries/selection/selection.json",
        *(
            f"stages/range/entries/{row_id}/summary.json"
            for row_id in evidence["rows"]
        ),
    }
    for relative in sorted(required):
        source_path = source / relative
        assert source_path.is_file(), relative
        destination_path = destination / relative
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, destination_path)


@pytest.fixture
def real_v3_import_tree(
    tmp_path: Path,
) -> tuple[LRStudySpec, Path, Path, Path]:
    spec = LRStudySpec.from_path(V3_STUDY_CONFIG)
    rescue = spec.data["rescue"]
    parent = rescue["parent_study"]
    evidence = rescue["evidence_study"]
    parent_name = f"{parent['name']}--{parent['study_id']}"
    evidence_name = f"{evidence['name']}--{evidence['study_id']}"
    real_parent = REAL_STUDIES / parent_name
    real_evidence = REAL_STUDIES / evidence_name
    if not real_parent.is_dir() or not real_evidence.is_dir():
        pytest.skip(
            "completed v1 parent and v2 evidence LR studies are unavailable: "
            f"{real_parent}, {real_evidence}"
        )

    temporary_parent = tmp_path / parent_name
    temporary_evidence = tmp_path / evidence_name
    _copy_minimal_completed_parent(real_parent, temporary_parent)
    _copy_v2_evidence(real_evidence, temporary_evidence, spec)
    v3_root = tmp_path / f"relative-rho-v3--{spec.study_id}"
    return spec, v3_root, temporary_parent, temporary_evidence


def _forbidden_builder(*args: Any, **kwargs: Any) -> None:
    del args, kwargs
    raise AssertionError("v3 asset import must not construct a dataset or model")


def _fresh_probe_builder(*args: Any, **kwargs: Any) -> None:
    del args, kwargs
    raise AssertionError("v3 probe must be recomputed from the imported checkpoint")


def test_v3_import_copies_only_split_and_initialization_and_is_idempotent(
    real_v3_import_tree: tuple[LRStudySpec, Path, Path, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec, v3_root, _, _ = real_v3_import_tree
    monkeypatch.setattr(lr_stages, "build_loader_bundle", _forbidden_builder)
    monkeypatch.setattr(lr_stages, "build_model_runtime", _forbidden_builder)

    result = lr_stages.prepare_study_assets(
        spec.data,
        v3_root,
        data_root=tmp_path / "dataset-must-not-be-read",
        download=True,
        device="device-must-not-be-used",
    )
    receipt_path = v3_root / "rescue/import_receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    rescue = spec.data["rescue"]
    frozen = rescue["frozen_provenance"]

    assert result["import_receipt"] == receipt
    assert receipt["schema_version"] == "mnist-conv-lr-parent-import/v2"
    assert receipt["mode"] == "reused_parent_assets_recomputed_probe"
    assert receipt["probe_recomputed_under_relative_rho"] is True
    assert receipt["official_test_read"] is False
    assert result["split_provenance"]["official_test_read"] is False
    assert all(not Path(value).is_absolute() for value in _iter_strings(receipt))
    assert all(str(tmp_path) not in value for value in _iter_strings(receipt))

    expected_copies = {
        "split/indices.json": frozen["split_indices_sha256"],
        "split/provenance.json": frozen["split_provenance_sha256"],
    }
    for architecture, values in frozen["architectures"].items():
        expected_copies[f"initialization/{architecture}.pt"] = values[
            "checkpoint_sha256"
        ]
        expected_copies[f"initialization/{architecture}.json"] = values[
            "metadata_sha256"
        ]
    copied = {item["destination_path"]: item for item in receipt["copied_artifacts"]}
    assert set(copied) == set(expected_copies)
    assert not any(path.startswith("stages/probe/") for path in copied)
    assert not any("baseline" in path for path in copied)
    assert not (v3_root / "stages/probe").exists()
    for relative, expected_digest in expected_copies.items():
        destination = v3_root / relative
        assert copied[relative] == {
            "source_path": relative,
            "destination_path": relative,
            "sha256": expected_digest,
            "bytes": destination.stat().st_size,
        }
        assert sha256_file(destination) == expected_digest

    assert receipt["validated_parent_stages"] == {
        stage: {
            "manifest_sha256": frozen[f"parent_{stage}_manifest_sha256"],
            "completion_sha256": frozen[f"parent_{stage}_completion_sha256"],
        }
        for stage in ("probe", "range", "select")
    }
    verified_parent = {
        item["row_id"]: item for item in receipt["verified_parent_range_summaries"]
    }
    assert set(verified_parent) == set(rescue["eligibility"]["included_row_ids"])
    for row_id, values in frozen["rows"].items():
        assert verified_parent[row_id]["sha256"] == values[
            "parent_range_summary_sha256"
        ]

    evidence = rescue["evidence_study"]
    verified_evidence = receipt["verified_evidence_study"]
    assert verified_evidence["study_id"] == evidence["study_id"]
    assert verified_evidence["directory_name"] == (
        f"{evidence['name']}--{evidence['study_id']}"
    )
    evidence_rows = {item["row_id"]: item for item in verified_evidence["rows"]}
    assert set(evidence_rows) == set(evidence["rows"])
    for row_id, values in evidence["rows"].items():
        assert evidence_rows[row_id]["sha256"] == values["range_summary_sha256"]

    tracked_files = [receipt_path, *(v3_root / path for path in copied)]
    before = {
        path: (sha256_file(path), path.stat().st_mtime_ns) for path in tracked_files
    }
    repeated = lr_stages.prepare_study_assets(
        spec.data,
        v3_root,
        data_root=tmp_path / "dataset-must-not-be-read",
        download=False,
        device="device-must-not-be-used",
    )
    after = {
        path: (sha256_file(path), path.stat().st_mtime_ns) for path in tracked_files
    }
    assert repeated["import_receipt"] == receipt
    assert after == before

    monkeypatch.setattr(lr_stages, "build_model_runtime", _fresh_probe_builder)
    with pytest.raises(AssertionError, match="probe must be recomputed"):
        lr_stages.execute_probe_entry(
            spec.data,
            v3_root,
            spec.rows[0]["row_id"],
            data_root=tmp_path / "fresh-probe-dataset",
            download=False,
            device="fresh-probe-device",
        )


def test_v3_import_rejects_a_same_size_corruption_of_a_parent_checkpoint(
    real_v3_import_tree: tuple[LRStudySpec, Path, Path, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec, v3_root, parent_root, _ = real_v3_import_tree
    monkeypatch.setattr(lr_stages, "build_loader_bundle", _forbidden_builder)
    monkeypatch.setattr(lr_stages, "build_model_runtime", _forbidden_builder)

    corrupted = parent_root / "initialization/conv2.pt"
    value = bytearray(corrupted.read_bytes())
    value[-1] ^= 1
    corrupted.write_bytes(value)

    with pytest.raises(ValueError, match="current artifact hashes and sizes"):
        lr_stages.prepare_study_assets(
            spec.data,
            v3_root,
            data_root=tmp_path / "dataset-must-not-be-read",
            download=False,
            device="device-must-not-be-used",
        )
    assert not (v3_root / "rescue/import_receipt.json").exists()
