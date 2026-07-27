from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from experiments.mnist_conv.identity import sha256_file
from experiments.mnist_conv import perfectdiode_hparam_v2_orchestration as orchestration


STUDY_ID = "lrstudy_" + "1" * 64
CONFIG_SHA256 = "2" * 64
MANIFEST_ID = "pdlrmanifest_" + "3" * 64
SELECTED = "selected_seed0_ordinary_mnist_three_epoch_screen"


class _FakeSpec:
    study_id = STUDY_ID
    config_sha256 = CONFIG_SHA256
    data = {
        "selection": {
            "selected_status": SELECTED,
            "unresolved_statuses": [
                "unresolved_no_pass",
                "unresolved_boundary",
                "unresolved_probe",
                "unresolved_post_training_tk",
            ],
        }
    }

    @classmethod
    def from_path(cls, _path: Path) -> "_FakeSpec":
        return cls()


class _FakeOperatingPointSpec(_FakeSpec):
    data = {
        **_FakeSpec.data,
        "upstream_tk": {
            "study_id": "tkstudy_" + "a" * 64,
            "config_sha256": "b" * 64,
        },
        "operating_point": {
            "mode": "user_fixed_after_residual_gradient_review",
            "inference_iterations": 8,
            "training_iterations": 8,
            "require_fresh_manifest_bound_preflight_gate": True,
        },
    }
    rows = tuple(
        {
            "row_id": f"{scheme}_row",
            "upstream_tk": {"selected_t": 4, "selected_k": 4},
        }
        for scheme in ("baseline", "ours", "legacy")
    )


@pytest.fixture(autouse=True)
def _fake_scientific_authority(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        orchestration,
        "PerfectDiodeConv3HparamStudySpec",
        _FakeSpec,
    )
    monkeypatch.setattr(
        orchestration,
        "validate_surface_manifest",
        lambda manifest, *, spec: dict(manifest),
    )


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


def _record(path: Path, base: Path) -> dict[str, Any]:
    return {
        "path": path.relative_to(base).as_posix(),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }


def _records(root: Path, *, exclude: tuple[Path, ...] = ()) -> list[dict[str, Any]]:
    excluded = {path.resolve() for path in exclude}
    files = sorted(
        (
            path
            for path in root.rglob("*")
            if path.is_file() and path.resolve() not in excluded
        ),
        key=lambda path: path.relative_to(root).as_posix(),
    )
    return [_record(path, root) for path in files]


def _surfaces(
    *,
    operating_point_contract: bool = False,
) -> list[dict[str, Any]]:
    values: list[dict[str, Any]] = []
    routes = (
        ("baseline", "main", "fixed_high_3x3"),
        ("ours", "akibscomputer", "fixed_high_3x3"),
        ("legacy", "main", "adaptive_safe_center_factor_three_3x3"),
    )
    for scheme, host, mode in routes:
        for optimizer in ("sgd", "adam"):
            surface_id = f"{scheme}_row--{optimizer}"
            surface: dict[str, Any] = {
                "surface_index": len(values),
                "surface_id": surface_id,
                "row_id": f"{scheme}_row",
                "scheme": scheme,
                "optimizer": optimizer,
                "host": host,
                "lr_eligible": True,
                "zero_work": False,
                "inference_iterations": (
                    8 if operating_point_contract else 4
                ),
                "training_iterations": (
                    8 if operating_point_contract else 4
                ),
                "rho_policy": {"core_mode": mode},
                "output_path": f"surfaces/{surface_id}",
                "completion_path": f"surfaces/{surface_id}/completion.json",
                "finalization_path": (
                    f"surfaces/{surface_id}/stages/finalize_lr/result.json"
                ),
            }
            if operating_point_contract:
                surface.update(
                    {
                        "shared_asset_hashes": {
                            "initialization_checkpoint_sha256": "4" * 64,
                            "initialization_tensor_sha256": "5" * 64,
                            "t_cohort_indices_sha256": "6" * 64,
                            "k_cohort_indices_sha256": "7" * 64,
                        },
                        "execution_source": {
                            "archive_sha256": "8" * 64,
                            "effective_source_fingerprint": "9" * 64,
                        },
                        "execution_environment_sha256": "c" * 64,
                        "upstream_tk": {
                            "selected_t": 4,
                            "selected_k": 4,
                        },
                    }
                )
            values.append(surface)
    return values


def _authority(
    root: Path,
    *,
    operating_point_contract: bool = False,
) -> list[dict[str, Any]]:
    surfaces = _surfaces(
        operating_point_contract=operating_point_contract,
    )
    _write(root / orchestration.STUDY_FILENAME, {"fake": "study"})
    _write(
        root / orchestration.MANIFEST_FILENAME,
        {
            "manifest_id": MANIFEST_ID,
            "study_id": STUDY_ID,
            "config_sha256": CONFIG_SHA256,
            "surface_count": 6,
            "public_stage_sequence": list(orchestration.PUBLIC_STAGE_SEQUENCE),
            "surfaces": surfaces,
        },
    )
    return surfaces


def _cell(
    root: Path,
    surface: dict[str, Any],
    stage: str,
    cell_id: str = "rho_conv_0p009--rho_dense_0p03",
) -> None:
    destination = root / surface["output_path"] / "stages" / stage / "cells" / cell_id
    result_path = destination / "result.json"
    completion_path = destination / "completion.json"
    _write(
        result_path,
        {
            "cell_id": cell_id,
            "surface_id": surface["surface_id"],
            "stage": stage,
            "official_test_read": False,
        },
    )
    _write(
        completion_path,
        {
            "schema_version": orchestration.CELL_COMPLETION_SCHEMA_VERSION,
            "state": "complete",
            "manifest_id": MANIFEST_ID,
            "surface_id": surface["surface_id"],
            "stage": stage,
            "cell_id": cell_id,
            "official_test_read": False,
            "outputs": [_record(result_path, destination)],
        },
    )


def _complete_surface(
    root: Path,
    surface: dict[str, Any],
    *,
    final_status: str = SELECTED,
) -> None:
    surface_root = root / surface["output_path"]
    for stage in orchestration.PUBLIC_STAGE_SEQUENCE:
        stage_root = surface_root / "stages" / stage
        if stage == "rho_canary_core":
            _cell(root, surface, stage)
        result: dict[str, Any] = {
            "schema_version": orchestration.EXECUTION_STAGE_SCHEMA_VERSION,
            "study_id": STUDY_ID,
            "config_sha256": CONFIG_SHA256,
            "manifest_id": MANIFEST_ID,
            "surface_id": surface["surface_id"],
            "stage": stage,
            "status": final_status if stage == "finalize_lr" else "complete",
            "zero_work": False,
            "reason": (
                "legacy_center_canary_failed_all_attempts"
                if stage == "finalize_lr"
                and final_status == "unresolved_no_safe_center"
                else None
            ),
            "official_test_read": False,
        }
        if stage == "finalize_lr" and final_status == SELECTED:
            result.update(
                {
                    "selected_t": surface["inference_iterations"],
                    "selected_k": surface["training_iterations"],
                    "selected_cell_id": "rho_conv_0p009--rho_dense_0p03",
                    "selected_rho": {"rho_conv": 0.009, "rho_dense": 0.03},
                    "final_validation_loss": 0.2,
                    "final_validation_accuracy": 0.91,
                    "median_projection_efficiency": 0.8,
                    "expansion_used": False,
                }
            )
        result_path = stage_root / "result.json"
        completion_path = stage_root / "completion.json"
        _write(result_path, result)
        _write(
            completion_path,
            {
                "schema_version": orchestration.EXECUTION_COMPLETION_SCHEMA_VERSION,
                "state": "complete",
                "study_id": STUDY_ID,
                "config_sha256": CONFIG_SHA256,
                "manifest_id": MANIFEST_ID,
                "surface_id": surface["surface_id"],
                "stage": stage,
                "official_test_read": False,
                "outputs": _records(stage_root, exclude=(completion_path,)),
            },
        )
    final_root = surface_root / "stages" / "finalize_lr"
    _write(
        surface_root / "completion.json",
        {
            "schema_version": orchestration.SURFACE_COMPLETION_SCHEMA_VERSION,
            "state": "complete",
            "study_id": STUDY_ID,
            "config_sha256": CONFIG_SHA256,
            "manifest_id": MANIFEST_ID,
            "surface_id": surface["surface_id"],
            "host": surface["host"],
            "status": final_status,
            "zero_work": False,
            "official_test_read": False,
            "outputs": [
                _record(final_root / "completion.json", surface_root),
                _record(final_root / "result.json", surface_root),
            ],
        },
    )


def _complete_host(
    root: Path,
    host: str,
    *,
    legacy_status: str = SELECTED,
    operating_point_contract: bool = False,
) -> list[dict[str, Any]]:
    surfaces = _authority(
        root,
        operating_point_contract=operating_point_contract,
    )
    selected = [surface for surface in surfaces if surface["host"] == host]
    for surface in selected:
        status = legacy_status if surface["scheme"] == "legacy" else SELECTED
        _complete_surface(root, surface, final_status=status)
    return selected


def _fixed_tk_gate(
    root: Path,
    surface: dict[str, Any],
) -> None:
    row_id = surface["row_id"]
    host = surface["host"]
    destination = (
        root / "preflight" / "fixed_tk_gate" / host / row_id
    )
    result_path = destination / "result.json"
    completion_path = destination / "completion.json"
    common = {
        "study_id": STUDY_ID,
        "config_sha256": CONFIG_SHA256,
        "manifest_id": MANIFEST_ID,
        "surface_id": surface["surface_id"],
        "row_id": row_id,
        "stage": "fixed_tk_gate",
        "gate_id": f"{row_id}--t8-k8",
        "representative_surface_id": surface["surface_id"],
        "selected_t": 8,
        "selected_k": 8,
        "official_test_read": False,
    }
    _write(
        result_path,
        {
            "schema_version": (
                orchestration.FIXED_TK_GATE_RESULT_SCHEMA_VERSION
            ),
            **common,
            "scheme": surface["scheme"],
            "optimizer": "adam",
            "host": host,
            "surface_manifest_sha256": sha256_file(
                root / orchestration.MANIFEST_FILENAME
            ),
            "diagnostic_selected_t": 4,
            "diagnostic_selected_k": 4,
            "restart_from_shared_initialization": True,
            "shared_asset_hashes": surface["shared_asset_hashes"],
            "execution_source": surface["execution_source"],
            "status": "passed",
            "passed": True,
            "operating_point_audit": {
                "schema_version": (
                    orchestration.TK_OPERATING_POINT_AUDIT_SCHEMA_VERSION
                ),
                "study_id": "tkstudy_" + "a" * 64,
                "config_sha256": "b" * 64,
                "entry_id": f"{row_id}--lr-fixed-tk-gate",
                "row_id": row_id,
                "scheme": surface["scheme"],
                "execution_backend": "tmux",
                "execution_host": host,
                "execution_environment_sha256": surface.get(
                    "execution_environment_sha256"
                ),
                "official_test_read": False,
                "selected_t": 8,
                "selected_k": 8,
                "k_audit_reference": 64,
                "fresh_replay_after_selection": True,
                "t_extension_used": False,
                "t_measurement": {
                    "cohort_source_indices_sha256": (
                        surface["shared_asset_hashes"][
                            "t_cohort_indices_sha256"
                        ]
                    ),
                    "initialization_checkpoint_sha256": (
                        surface["shared_asset_hashes"][
                            "initialization_checkpoint_sha256"
                        ]
                    ),
                    "initialization_tensor_sha256": (
                        surface["shared_asset_hashes"][
                            "initialization_tensor_sha256"
                        ]
                    ),
                    "official_test_read": False,
                },
                "t64_sentinel_measurement": {
                    "cohort_source_indices_sha256": (
                        surface["shared_asset_hashes"][
                            "t_cohort_indices_sha256"
                        ]
                    ),
                    "initialization_checkpoint_sha256": (
                        surface["shared_asset_hashes"][
                            "initialization_checkpoint_sha256"
                        ]
                    ),
                    "initialization_tensor_sha256": (
                        surface["shared_asset_hashes"][
                            "initialization_tensor_sha256"
                        ]
                    ),
                    "official_test_read": False,
                },
                "k_measurement": {
                    "cohort_source_indices_sha256": (
                        surface["shared_asset_hashes"][
                            "k_cohort_indices_sha256"
                        ]
                    ),
                    "initialization_checkpoint_sha256": (
                        surface["shared_asset_hashes"][
                            "initialization_checkpoint_sha256"
                        ]
                    ),
                    "initialization_tensor_sha256": (
                        surface["shared_asset_hashes"][
                            "initialization_tensor_sha256"
                        ]
                    ),
                    "official_test_read": False,
                },
                "t256_extension_sentinel_measurement": None,
                "t256_extension_sentinel_passed": None,
                "status": "passed",
                "passed": True,
            },
        },
    )
    _write(
        completion_path,
        {
            "schema_version": (
                orchestration.FIXED_TK_GATE_COMPLETION_SCHEMA_VERSION
            ),
            "state": "complete",
            **common,
            "status": "passed",
            "passed": True,
            "outputs": [_record(result_path, destination)],
        },
    )


def _snapshot(root: Path) -> list[tuple[str, str]]:
    return [
        (path.relative_to(root).as_posix(), sha256_file(path))
        for path in sorted(root.rglob("*"))
        if path.is_file()
    ]


def test_status_is_read_only_during_partially_completed_cell_stage(
    tmp_path: Path,
) -> None:
    surfaces = _authority(tmp_path)
    surface = next(value for value in surfaces if value["host"] == "main")
    _cell(tmp_path, surface, "rho_canary_core")
    before = _snapshot(tmp_path)

    status = orchestration.host_status(tmp_path, "main")

    assert status["state"] == "pending"
    assert status["launched_jobs"] == 0
    assert _snapshot(tmp_path) == before


def test_finalize_host_validates_recursive_hashes_and_legacy_terminal(
    tmp_path: Path,
) -> None:
    surfaces = _complete_host(
        tmp_path,
        "main",
        legacy_status="unresolved_no_safe_center",
    )
    before = _snapshot(tmp_path)
    status = orchestration.host_status(tmp_path, "main")
    assert status["state"] == "ready_to_finalize"
    assert _snapshot(tmp_path) == before

    terminal = orchestration.finalize_host_shard(tmp_path, "main")
    assert terminal["surface_count"] == 4
    assert terminal["launched_jobs"] == 0

    nested = (
        tmp_path
        / surfaces[0]["output_path"]
        / "stages"
        / "rho_canary_core"
        / "cells"
        / "rho_conv_0p009--rho_dense_0p03"
        / "result.json"
    )
    value = json.loads(nested.read_text(encoding="utf-8"))
    value["tampered"] = True
    _write(nested, value)
    with pytest.raises(orchestration.PerfectDiodeConv3OrchestrationError):
        orchestration.validate_host_terminal(tmp_path, "main")


def test_operating_point_contract_requires_fixed_tk_gate_receipts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        orchestration,
        "PerfectDiodeConv3HparamStudySpec",
        _FakeOperatingPointSpec,
    )
    _complete_host(
        tmp_path,
        "main",
        operating_point_contract=True,
    )

    status = orchestration.host_status(tmp_path, "main")

    assert status["state"] == "pending"
    assert status["complete"] is False
    assert status["fixed_tk_gates"] == [
        {"row_id": "baseline_row", "state": "missing"},
        {"row_id": "legacy_row", "state": "missing"},
    ]
    with pytest.raises(orchestration.PerfectDiodeConv3OrchestrationError):
        orchestration.finalize_host_shard(tmp_path, "main")
    assert not (tmp_path / "host_terminals" / "main.json").exists()


def test_operating_point_gate_receipts_are_host_terminal_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        orchestration,
        "PerfectDiodeConv3HparamStudySpec",
        _FakeOperatingPointSpec,
    )
    monkeypatch.setattr(
        orchestration,
        "_fixed_tk_audit_passed",
        lambda audit, *, selected_t, selected_k: True,
    )
    surfaces = _complete_host(
        tmp_path,
        "main",
        operating_point_contract=True,
    )
    gate_surfaces = [
        surface for surface in surfaces if surface["optimizer"] == "adam"
    ]
    for surface in gate_surfaces:
        _fixed_tk_gate(tmp_path, surface)

    status = orchestration.host_status(tmp_path, "main")
    assert status["state"] == "ready_to_finalize"
    assert status["fixed_tk_gates"] == [
        {"row_id": "baseline_row", "state": "passed"},
        {"row_id": "legacy_row", "state": "passed"},
    ]

    terminal = orchestration.finalize_host_shard(tmp_path, "main")
    artifact_paths = {
        artifact["path"] for artifact in terminal["artifacts"]
    }
    expected_gate_paths = {
        (
            "preflight/fixed_tk_gate/"
            f"main/{surface['row_id']}/{filename}"
        )
        for surface in gate_surfaces
        for filename in ("completion.json", "result.json")
    }
    assert expected_gate_paths <= artifact_paths
    assert orchestration.validate_host_terminal(tmp_path, "main") == terminal


def test_atomic_import_and_six_surface_study_finalization(tmp_path: Path) -> None:
    collector = tmp_path / "collector"
    _complete_host(collector, "main")
    orchestration.finalize_host_shard(collector, "main")

    incoming = (
        collector / ".incoming" / "bundle-001" / "shards" / "akibscomputer"
    )
    _complete_host(incoming, "akibscomputer")
    orchestration.finalize_host_shard(incoming, "akibscomputer")

    receipt = orchestration.import_host_shard(
        collector,
        incoming,
        "akibscomputer",
    )
    assert receipt["publication_mode"] == "same_filesystem_atomic_rename"
    assert not incoming.exists()
    assert (collector / "imported_shards" / "akibscomputer").is_dir()

    result = orchestration.finalize_study(collector)
    assert result["surface_count"] == 6
    assert result["selected_surface_count"] == 6
    assert result["status"] == "complete"
    assert len({item["surface_id"] for item in result["surfaces"]}) == 6
    assert orchestration.validate_study(collector) == result


def test_import_rejects_authority_drift(tmp_path: Path) -> None:
    collector = tmp_path / "collector"
    _authority(collector)
    incoming = collector / ".incoming" / "bundle-002" / "shards" / "akibscomputer"
    _complete_host(incoming, "akibscomputer")
    orchestration.finalize_host_shard(incoming, "akibscomputer")
    _write(incoming / orchestration.STUDY_FILENAME, {"fake": "different"})

    with pytest.raises(orchestration.PerfectDiodeConv3OrchestrationError):
        orchestration.import_host_shard(collector, incoming, "akibscomputer")


def test_public_cli_imports_repo_when_invoked_by_absolute_path(
    tmp_path: Path,
) -> None:
    script = (
        Path(__file__).resolve().parents[2]
        / "experiments"
        / "run_mnist_conv_perfectdiode_hparam_v2_orchestration.py"
    )
    completed = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "never launches jobs" in completed.stdout
