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
RESCUE_STUDY_CONFIG = (
    REPO_ROOT / "configs/conv/hardsigmoid_lr_rescue_warmin_sgd_bs16_v2.json"
)
REAL_PARENT_STUDIES = (
    REPO_ROOT / "simulation_results/conv_lr_protocol_20260719/lr_studies"
)


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
    """Copy only files needed to validate the real probe/range/select chain."""

    required = {"study.resolved.json"}
    for stage in ("probe", "range", "select"):
        manifest_relative = f"stages/{stage}/manifest.json"
        manifest = json.loads((source / manifest_relative).read_text())
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


@pytest.fixture
def real_parent_and_rescue_root(tmp_path: Path) -> tuple[LRStudySpec, Path]:
    spec = LRStudySpec.from_path(RESCUE_STUDY_CONFIG)
    parent = spec.data["rescue"]["parent_study"]
    parent_name = f"{parent['name']}--{parent['study_id']}"
    real_parent = REAL_PARENT_STUDIES / parent_name
    if not real_parent.is_dir():
        pytest.skip(f"completed parent LR study is unavailable: {real_parent}")

    temporary_parent = tmp_path / parent_name
    _copy_minimal_completed_parent(real_parent, temporary_parent)
    return spec, tmp_path / f"rescue--{spec.study_id}"


def _forbidden_builder(*args: Any, **kwargs: Any) -> None:
    del args, kwargs
    raise AssertionError("rescue import must not build a dataset or model")


def test_rescue_import_reuses_real_parent_probe_without_building(
    real_parent_and_rescue_root: tuple[LRStudySpec, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec, rescue_root = real_parent_and_rescue_root
    monkeypatch.setattr(lr_stages, "build_loader_bundle", _forbidden_builder)
    monkeypatch.setattr(lr_stages, "build_model_runtime", _forbidden_builder)

    result = lr_stages.prepare_study_assets(
        spec.data,
        rescue_root,
        data_root=tmp_path / "dataset-must-not-be-read",
        download=True,
        device="device-must-not-be-used",
    )
    receipt_path = rescue_root / "rescue/import_receipt.json"
    receipt = json.loads(receipt_path.read_text())

    assert result["import_receipt"] == receipt
    assert receipt["schema_version"] == "mnist-conv-lr-parent-import/v1"
    assert receipt["mode"] == "reused_parent_artifact"
    assert receipt["official_test_read"] is False
    assert all(not Path(value).is_absolute() for value in _iter_strings(receipt))
    assert all(str(tmp_path) not in value for value in _iter_strings(receipt))

    frozen_rows = spec.data["rescue"]["frozen_provenance"]["rows"]
    copied = {item["destination_path"]: item for item in receipt["copied_artifacts"]}
    expected_probe_paths = {
        f"stages/probe/entries/{row_id}/{filename}": digest
        for row_id, row in frozen_rows.items()
        for filename, digest in row["probe_outputs"].items()
    }
    for relative, expected_digest in expected_probe_paths.items():
        assert copied[relative] == {
            "source_path": relative,
            "destination_path": relative,
            "sha256": expected_digest,
            "bytes": (rescue_root / relative).stat().st_size,
        }
        assert sha256_file(rescue_root / relative) == expected_digest

    expected_row_ids = tuple(row["row_id"] for row in spec.rows)
    assert tuple(
        item["row_id"] for item in receipt["verified_parent_range_summaries"]
    ) == tuple(sorted(expected_row_ids))
    assert sorted(
        path.name for path in (rescue_root / "stages/probe/entries").iterdir()
    ) == sorted(expected_row_ids)
    assert not (rescue_root / "initialization/conv2.pt").exists()
    assert not any("conv2." in path for path in copied)
    assert not any("baseline" in path for path in copied)

    for row_id in expected_row_ids:
        assert lr_stages.execute_probe_entry(
            spec.data,
            rescue_root,
            row_id,
            data_root=tmp_path / "dataset-must-not-be-read",
            download=True,
            device="device-must-not-be-used",
        ) == {
            "status": "complete",
            "mode": "reused_parent_artifact",
            "row_id": row_id,
            "rho_unit": frozen_rows[row_id]["rho_unit"],
        }

    tracked_files = [
        receipt_path,
        *(rescue_root / relative for relative in copied),
    ]
    before = {
        path: (sha256_file(path), path.stat().st_mtime_ns) for path in tracked_files
    }
    repeated = lr_stages.prepare_study_assets(
        spec.data,
        rescue_root,
        data_root=tmp_path / "dataset-must-not-be-read",
        download=True,
        device="device-must-not-be-used",
    )
    after = {
        path: (sha256_file(path), path.stat().st_mtime_ns) for path in tracked_files
    }
    assert repeated["import_receipt"] == receipt
    assert after == before


def test_rescue_import_and_probe_reuse_reject_corrupted_import(
    real_parent_and_rescue_root: tuple[LRStudySpec, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec, rescue_root = real_parent_and_rescue_root
    monkeypatch.setattr(lr_stages, "build_loader_bundle", _forbidden_builder)
    monkeypatch.setattr(lr_stages, "build_model_runtime", _forbidden_builder)
    lr_stages.prepare_study_assets(
        spec.data,
        rescue_root,
        data_root=tmp_path / "dataset-must-not-be-read",
        download=False,
        device="device-must-not-be-used",
    )

    row_id = spec.rows[0]["row_id"]
    corrupted = rescue_root / f"stages/probe/entries/{row_id}/step_log.csv"
    corrupted.write_bytes(corrupted.read_bytes() + b"corrupt\n")

    with pytest.raises(RuntimeError, match="imported probe output.*SHA-256"):
        lr_stages.execute_probe_entry(
            spec.data,
            rescue_root,
            row_id,
            data_root=tmp_path / "dataset-must-not-be-read",
            download=False,
            device="device-must-not-be-used",
        )
    with pytest.raises(RuntimeError, match="imported artifact.*SHA-256"):
        lr_stages.prepare_study_assets(
            spec.data,
            rescue_root,
            data_root=tmp_path / "dataset-must-not-be-read",
            download=False,
            device="device-must-not-be-used",
        )
