from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from experiments.analyze_bounded_wmax_sweep import (
    EXPECTED_SOURCE_ARCHIVE_SHA256,
    analyze,
    main,
)
from experiments.reporting import MANIFEST_SCHEMA, RESULT_SCHEMA, STATUS_SCHEMA


STUDY_ID = "synthetic-bounded-wmax-study"
EVIDENCE_CLASS = "ordinary_mnist_bounded_wmax_sensitivity"
SOURCE_COMMIT = "0" * 40


def _json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_json_bytes(value))


def _study_manifest(tmp_path: Path) -> tuple[Path, list[dict[str, object]]]:
    rows: list[dict[str, object]] = []
    for weight_max in (1e-4, 3e-4):
        tag = "1em4" if weight_max == 1e-4 else "3em4"
        for scheme in ("baseline", "ours", "legacy"):
            arm_id = f"conv1_wmax_{tag}_{scheme}_sgd_seed0"
            rows.append(
                {
                    "architecture": "conv1",
                    "arm_id": arm_id,
                    "config": f"conv1/{len(rows):02d}_{arm_id}.json",
                    "config_sha256": hashlib.sha256(arm_id.encode()).hexdigest(),
                    "epochs": 10,
                    "global_array_index": len(rows),
                    "optimizer": "sgd",
                    "scheme": scheme,
                    "target": "akib",
                    "weight_max": weight_max,
                    "weight_min": 1e-5,
                    "weight_range_ratio": weight_max / 1e-5,
                }
            )
    ordered_digest = hashlib.sha256(
        "".join(f"{row['config_sha256']}\n" for row in rows).encode("ascii")
    ).hexdigest()
    manifest = {
        "evidence_class": EVIDENCE_CLASS,
        "ordered_config_set_sha256": ordered_digest,
        "paper_facing": False,
        "parent": {"source_commit": SOURCE_COMMIT},
        "run_count": len(rows),
        "runs": rows,
        "schema_version": "perfectdiode-bounded-wmax-sweep-config-set/v1",
        "study_id": STUDY_ID,
        "varied_field": "model_base.weight_max",
        "weight_max_values": [1e-4, 3e-4],
    }
    path = tmp_path / "study_manifest.json"
    _write_json(path, manifest)
    return path, rows


def _canonical_run(
    root: Path,
    row: dict[str, object],
    *,
    final_accuracy: float,
    best_accuracy: float,
    suffix: str = "",
    config_sha256: str | None = None,
    source_archive_sha256: str = EXPECTED_SOURCE_ARCHIVE_SHA256,
) -> Path:
    run_id = f"run_{row['global_array_index']}{suffix}"
    run_dir = root / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "metrics.jsonl").write_text("", encoding="utf-8")
    manifest = {
        "schema_version": MANIFEST_SCHEMA,
        "created_at": "2026-08-09T00:00:00+00:00",
        "study_id": STUDY_ID,
        "run_id": run_id,
        "arm_id": row["arm_id"],
        "evidence_class": EVIDENCE_CLASS,
        "smoke": False,
        "configuration": {
            "sha256": config_sha256 or row["config_sha256"],
            "epochs": row["epochs"],
            "resolved": {
                "arm_id": row["arm_id"],
                "study_id": STUDY_ID,
                "lab": {"epochs": row["epochs"]},
                "model_base": {
                    "weight_min": row["weight_min"],
                    "weight_max": row["weight_max"],
                },
                "optimizer": {"name": "SGD"},
            },
        },
        "dataset": {"official_test_read": False},
        "git": {
            "commit": SOURCE_COMMIT,
            "source_archive_sha256": source_archive_sha256,
        },
        "runtime": {"target": row["target"]},
    }
    _write_json(run_dir / "manifest.json", manifest)
    result = {
        "schema_version": RESULT_SCHEMA,
        "study_id": STUDY_ID,
        "run_id": run_id,
        "arm_id": row["arm_id"],
        "evidence_class": EVIDENCE_CLASS,
        "smoke": False,
        "completion": {"criteria_met": True},
        "manifest_sha256": _sha256_bytes((run_dir / "manifest.json").read_bytes()),
        "metrics_sha256": _sha256_bytes((run_dir / "metrics.jsonl").read_bytes()),
        "terminal_metrics": {
            "best_epoch": 3,
            "validation": {
                "best_accuracy": best_accuracy,
                "final_accuracy": final_accuracy,
            },
        },
        "artifacts": [],
    }
    _write_json(run_dir / "result.json", result)
    status = {
        "schema_version": STATUS_SCHEMA,
        "study_id": STUDY_ID,
        "run_id": run_id,
        "arm_id": row["arm_id"],
        "state": "complete",
    }
    _write_json(run_dir / "status.json", status)
    return run_dir


def _metric(row: dict[str, object]) -> tuple[float, float]:
    scheme_offset = {"baseline": 0.0, "ours": 0.10, "legacy": 0.04}[
        str(row["scheme"])
    ]
    bound_offset = 0.02 if row["weight_max"] == 3e-4 else 0.0
    final = 0.60 + scheme_offset + bound_offset
    return final, final + 0.05


def test_complete_analysis_writes_metrics_deltas_gaps_and_plots(
    tmp_path: Path,
) -> None:
    manifest_path, rows = _study_manifest(tmp_path)
    result_root = tmp_path / "results"
    for row in rows:
        final, best = _metric(row)
        _canonical_run(
            result_root, row, final_accuracy=final, best_accuracy=best
        )

    analysis_dir = tmp_path / "analysis"
    report = analyze(
        study_manifest_path=manifest_path,
        result_roots=[result_root],
        analysis_dir=analysis_dir,
        require_complete=True,
    )

    assert report["coverage"]["complete"] is True
    assert report["coverage"]["usable_arm_count"] == 6
    ours_wide = next(
        row
        for row in report["accuracy_by_surface"]
        if row["scheme"] == "ours" and row["weight_max"] == 3e-4
    )
    assert ours_wide["delta_final_vs_1e-4"] == pytest.approx(0.02)
    ours_baseline = next(
        row
        for row in report["scheme_gaps"]
        if row["gap_definition"] == "ours_minus_baseline"
        and row["weight_max"] == 3e-4
    )
    assert ours_baseline["final_validation_accuracy_gap"] == pytest.approx(0.10)
    assert (analysis_dir / "accuracy_by_surface.csv").is_file()
    assert (analysis_dir / "anchor_deltas.csv").is_file()
    assert (analysis_dir / "scheme_gaps.csv").is_file()
    assert (analysis_dir / "analysis_summary.json").is_file()
    assert (analysis_dir / "accuracy_vs_wmax_conv1.png").stat().st_size > 0
    assert (analysis_dir / "delta_vs_anchor_conv1.png").stat().st_size > 0
    assert (analysis_dir / "scheme_gaps_vs_wmax_conv1.png").stat().st_size > 0


def test_partial_analysis_is_written_but_require_complete_exits_nonzero(
    tmp_path: Path,
) -> None:
    manifest_path, rows = _study_manifest(tmp_path)
    result_root = tmp_path / "partial_results"
    final, best = _metric(rows[0])
    _canonical_run(result_root, rows[0], final_accuracy=final, best_accuracy=best)
    analysis_dir = tmp_path / "partial_analysis"

    report = analyze(
        study_manifest_path=manifest_path,
        result_roots=[result_root],
        analysis_dir=analysis_dir,
    )
    assert report["coverage"]["complete"] is False
    assert report["coverage"]["usable_arm_count"] == 1
    assert len(report["coverage"]["missing_arm_ids"]) == 5

    return_code = main(
        [
            "--study-manifest",
            str(manifest_path),
            "--result-root",
            str(result_root),
            "--analysis-dir",
            str(analysis_dir),
            "--require-complete",
        ]
    )
    assert return_code == 2
    written = json.loads(
        (analysis_dir / "analysis_summary.json").read_text(encoding="utf-8")
    )
    assert written["coverage"]["complete"] is False


def test_duplicate_valid_arm_is_ambiguous_and_excluded(tmp_path: Path) -> None:
    manifest_path, rows = _study_manifest(tmp_path)
    result_root = tmp_path / "duplicate_results"
    final, best = _metric(rows[0])
    _canonical_run(result_root, rows[0], final_accuracy=final, best_accuracy=best)
    _canonical_run(
        result_root,
        rows[0],
        final_accuracy=final,
        best_accuracy=best,
        suffix="_retry",
    )

    report = analyze(
        study_manifest_path=manifest_path,
        result_roots=[result_root],
        analysis_dir=tmp_path / "duplicate_analysis",
    )

    assert report["coverage"]["duplicate_arm_ids"] == [rows[0]["arm_id"]]
    assert report["coverage"]["usable_arm_count"] == 0
    first = report["accuracy_by_surface"][0]
    assert first["available"] is False
    assert first["final_validation_accuracy"] is None


def test_invalid_config_digest_is_reported_as_invalid_evidence(tmp_path: Path) -> None:
    manifest_path, rows = _study_manifest(tmp_path)
    result_root = tmp_path / "invalid_results"
    final, best = _metric(rows[0])
    _canonical_run(
        result_root,
        rows[0],
        final_accuracy=final,
        best_accuracy=best,
        config_sha256="f" * 64,
    )

    report = analyze(
        study_manifest_path=manifest_path,
        result_roots=[result_root],
        analysis_dir=tmp_path / "invalid_analysis",
    )

    assert report["coverage"]["usable_arm_count"] == 0
    assert len(report["coverage"]["invalid_runs"]) == 1
    assert "scientific config digest" in " ".join(
        report["coverage"]["invalid_runs"][0]["errors"]
    )


def test_wrong_source_archive_is_reported_as_invalid_evidence(
    tmp_path: Path,
) -> None:
    manifest_path, rows = _study_manifest(tmp_path)
    result_root = tmp_path / "wrong_source_results"
    final, best = _metric(rows[0])
    _canonical_run(
        result_root,
        rows[0],
        final_accuracy=final,
        best_accuracy=best,
        source_archive_sha256="f" * 64,
    )

    report = analyze(
        study_manifest_path=manifest_path,
        result_roots=[result_root],
        analysis_dir=tmp_path / "wrong_source_analysis",
    )

    assert report["coverage"]["usable_arm_count"] == 0
    assert len(report["coverage"]["invalid_runs"]) == 1
    assert "source archive" in " ".join(
        report["coverage"]["invalid_runs"][0]["errors"]
    )
