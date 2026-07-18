from __future__ import annotations

import copy
import json
import shutil
from pathlib import Path

import pytest

from experiments.mnist_conv.identity import run_fingerprint
from experiments.mnist_conv.layout import ResultLayout
from experiments.mnist_conv.manifest import (
    ManifestConflictError,
    load_manifest,
    manifest_entry,
    publish_manifest,
    validate_manifest,
)
from experiments.mnist_conv.specs import RunSpec, SpecValidationError, SweepSpec, pointer_get


REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_EXAMPLE = REPO_ROOT / "configs/conv/run_v1.diagnostic.example.json"
PROVENANCE = {
    "git_revision": "1" * 40,
    "dirty_source_digest": "2" * 64,
    "effective_code_fingerprint": "3" * 64,
}


def _run_value() -> dict:
    value = json.loads(RUN_EXAMPLE.read_text())
    value["run"]["training"].setdefault(
        "optimizer",
        {"name": "SGD", "momentum": 0.0, "weight_decay": 0.0},
    )
    return value


def _sweep(
    *,
    name: str = "integrity-sweep",
    run_label: str | None = None,
) -> SweepSpec:
    base_run = _run_value()
    if run_label is not None:
        base_run["label"] = run_label
    return SweepSpec.from_dict(
        {
            "schema_version": "mnist-conv-sweep/v1",
            "name": name,
            "base_run": base_run,
            "axes": [{"path": "/seed", "values": [0, 1]}],
            "cases": [{"id": "baseline", "set": {}}],
            "varying_fields": ["/seed"],
            "collection": {
                "expected_seeds": [0, 1],
                "required_cases": ["baseline"],
                "group_by": [
                    "/run/architecture/profile",
                    "/run/model/non_linearity",
                ],
            },
        }
    )


def _published(tmp_path: Path) -> tuple[dict, Path, ResultLayout]:
    layout = ResultLayout(tmp_path / "results")
    manifest, path = publish_manifest(_sweep(), layout, PROVENANCE)
    return manifest, path, layout


@pytest.mark.parametrize(
    "tamper",
    [
        "top_level_extra",
        "collection_shape",
        "invalid_seed_contract",
        "duplicate_varying_field",
        "boolean_job_index",
        "entry_extra",
        "axis_disagrees_with_run",
        "logical_key",
        "run_spec",
        "run_relpath",
        "sweep_id",
    ],
)
def test_manifest_rejects_tampered_contract_entries_and_sweep_id(tmp_path, tamper):
    manifest, _, _ = _published(tmp_path)
    value = copy.deepcopy(manifest)
    if tamper == "top_level_extra":
        value["unexpected"] = True
    elif tamper == "collection_shape":
        value["collection"]["unexpected"] = []
    elif tamper == "invalid_seed_contract":
        value["collection"]["expected_seeds"] = [0, True]
    elif tamper == "duplicate_varying_field":
        value["varying_fields"].append("/seed")
    elif tamper == "boolean_job_index":
        value["entries"][0]["job_index"] = False
    elif tamper == "entry_extra":
        value["entries"][0]["unexpected"] = True
    elif tamper == "axis_disagrees_with_run":
        value["entries"][0]["axes"]["/seed"] = 99
    elif tamper == "logical_key":
        value["entries"][0]["logical_key"] = "not-the-canonical-logical-key"
    elif tamper == "run_spec":
        value["entries"][0]["run_spec"]["seed"] = 99
    elif tamper == "run_relpath":
        value["entries"][0]["run_relpath"] = "../../../outside"
    elif tamper == "sweep_id":
        value["sweep_id"] = "sweep_" + "f" * 64
    else:  # pragma: no cover - the parametrization is exhaustive
        raise AssertionError(tamper)

    with pytest.raises(SpecValidationError):
        validate_manifest(value)


def test_load_crosschecks_resolved_sweep_and_jobs(tmp_path):
    manifest, manifest_path, _ = _published(tmp_path)
    resolved_path = manifest_path.parent / "sweep.resolved.json"
    jobs_path = manifest_path.parent / "jobs.jsonl"
    resolved_bytes = resolved_path.read_bytes()
    jobs_bytes = jobs_path.read_bytes()

    resolved = json.loads(resolved_bytes)
    resolved["name"] = "tampered-display-name"
    resolved_path.write_text(json.dumps(resolved))
    with pytest.raises(SpecValidationError):
        load_manifest(manifest_path)

    resolved_path.write_bytes(resolved_bytes)
    jobs_path.write_bytes(jobs_bytes + b" \n")
    with pytest.raises(SpecValidationError):
        load_manifest(manifest_path)

    jobs_path.write_bytes(jobs_bytes)
    assert load_manifest(manifest_path) == manifest


def test_publish_reuses_sweep_id_across_display_names_and_manifest_is_relocatable(tmp_path):
    manifest, first_path, layout = _published(tmp_path)
    renamed_manifest, renamed_path = publish_manifest(
        _sweep(
            name="same-plan-new-display-name",
            run_label="same-run-new-display-label",
        ),
        layout,
        PROVENANCE,
    )
    assert renamed_path == first_path
    assert renamed_manifest == manifest
    assert len(list(layout.sweeps_root.glob(f"*--{manifest['sweep_id']}"))) == 1

    copied_manifest = tmp_path / "worker-copy" / "internal-manifest.json"
    copied_manifest.parent.mkdir(parents=True)
    shutil.copy2(first_path, copied_manifest)
    assert load_manifest(copied_manifest) == manifest

    relocated_dir = (
        tmp_path
        / "relocated-results"
        / "sweeps"
        / first_path.parent.name
    )
    relocated_dir.parent.mkdir(parents=True)
    shutil.copytree(first_path.parent, relocated_dir)
    assert load_manifest(relocated_dir / "manifest.json") == manifest


def test_publish_rejects_conflicting_existing_sweep_files(tmp_path):
    _, manifest_path, layout = _published(tmp_path)
    jobs_path = manifest_path.parent / "jobs.jsonl"
    jobs_path.write_bytes(jobs_path.read_bytes() + b"tampered\n")

    with pytest.raises(ManifestConflictError):
        publish_manifest(_sweep(name="renamed-but-same-id"), layout, PROVENANCE)


def test_every_internal_job_is_self_identifying_and_detached(tmp_path):
    manifest, _, _ = _published(tmp_path)
    for index in range(len(manifest["entries"])):
        entry = manifest_entry(manifest, index)
        spec = RunSpec.from_dict(entry["run_spec"])
        assert entry["run_id"] == run_fingerprint(spec, PROVENANCE)
        for pointer, value in entry["axes"].items():
            assert pointer_get(spec.to_dict(), pointer) == value

        entry["run_spec"]["seed"] = 99
        assert manifest["entries"][index]["run_spec"]["seed"] != 99
