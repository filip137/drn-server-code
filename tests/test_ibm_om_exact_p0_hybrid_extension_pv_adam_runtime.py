from __future__ import annotations

import copy
from dataclasses import replace
import json
from pathlib import Path

import pytest

from experiments.artifacts import sha256_file
from experiments.definitions import resolve_experiment_config
from experiments.mnist_relu_drn import (
    ibm_om_exact_p0_hybrid_extension_pv_adam_runtime as runtime,
)
from experiments.mnist_relu_drn.ibm_om_exact_p0_hybrid_extension_pv_adam import (
    ANCHOR_ARM_ID,
)
from experiments.schema import RunMode


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT
    / "examples/mnist_relu_drn/ibm_om_exact_p0_hybrid_extension_pv_adam/validation_sweep.json"
)


def _spec():
    _definition, spec = resolve_experiment_config(CONFIG, RunMode.TRAIN)
    return spec


def _stage1_fixture() -> dict[str, object]:
    return {
        "schema": runtime.STAGE1_SUMMARY_SCHEMA,
        "schema_version": 1,
        "status": "completed",
        "evidence_tier": runtime.EVIDENCE_TIER,
        "exact_p0": {
            "sha256": _spec().protocol.base.p0.sha256,
            "test": {"student_accuracy": 0.4445},
        },
        "protocol": {
            "arms": ["full", "w1_only", "w2_only", "w1_top500", "global_top500"]
        },
        "matched_gates": {
            "test_sealed_until_all_epoch_selections_frozen": True,
            "full_mask_historical_parity_exact": True,
        },
        "comparison": {
            "by_arm": {
                "full": {
                    "validation_accuracy": 0.927,
                    "test_accuracy": 0.9334,
                    "selected_checkpoint_cost": {"issued_commands": 303_231},
                }
            }
        },
    }


def _anchor_rows() -> list[dict[str, object]]:
    return [
        {
            "epoch": epoch,
            "checkpoint": f"/historical/top500_epoch_{epoch}.pt",
            "checkpoint_sha256": str(epoch) * 64,
            "issued_commands_cumulative": epoch * 10_000,
            "validation_prediction_sha256": str(epoch + 3) * 64,
        }
        for epoch in (1, 2, 3)
    ]


def _stage2_fixture(
    *, stage1_sha256: str, stage1_full: object
) -> dict[str, object]:
    return {
        "schema": runtime.STAGE2_SUMMARY_SCHEMA,
        "schema_version": 1,
        "status": "completed",
        "evidence_tier": runtime.EVIDENCE_TIER,
        "exact_p0": {"sha256": _spec().protocol.base.p0.sha256},
        "protocol": {"arms": list(runtime.STAGE2_ARM_IDS)},
        "matched_gates": {
            "stage1_ranking_hashes_exact": True,
            "all_five_validation_epoch_selections_frozen_before_test": True,
            "random500_test_never_opened": True,
        },
        "comparison": {
            "selected_smallest_qualifying_union": ANCHOR_ARM_ID,
            "test_opened_arms": ["w2_only", ANCHOR_ARM_ID],
        },
        "stage1_reference": {
            "sha256": stage1_sha256,
            "full_arm": stage1_full,
        },
        "mask_artifact": {
            "arms": {
                ANCHOR_ARM_ID: {"flat_physical_mask_sha256": "a" * 64}
            }
        },
        "arms": {
            ANCHOR_ARM_ID: {
                "selected_epoch": 3,
                "epoch_reports": _anchor_rows(),
                "selected_test": {
                    "student_correct": 8939,
                    "student_accuracy": 0.8939,
                    "prediction_sha256": "b" * 64,
                    "kl_teacher_student": 0.3032,
                },
                "test_opened": True,
            }
        },
    }


def _fixture_protocol(
    tmp_path: Path,
) -> tuple[object, dict[str, object], dict[str, object]]:
    spec = _spec()
    stage1 = _stage1_fixture()
    stage1_path = tmp_path / "stage1.json"
    stage1_path.write_text(json.dumps(stage1), encoding="utf-8")
    stage1_hash = sha256_file(stage1_path)
    stage2 = _stage2_fixture(
        stage1_sha256=stage1_hash,
        stage1_full=stage1["comparison"]["by_arm"]["full"],
    )
    stage2_path = tmp_path / "stage2.json"
    stage2_path.write_text(json.dumps(stage2), encoding="utf-8")
    anchor = stage2["arms"][ANCHOR_ARM_ID]
    epoch_hashes = tuple(
        runtime._epoch_trajectory_sha256(row) for row in anchor["epoch_reports"]
    )
    test_hash = runtime._json_sha256(anchor["selected_test"])
    partial = spec.protocol.partial
    stage1_reference = replace(
        partial.stage1_reference,
        summary_path="stage1.json",
        summary_sha256=stage1_hash,
    )
    stage2_reference = replace(
        partial.stage2_reference,
        summary=replace(
            partial.stage2_reference.summary,
            summary_path="stage2.json",
            summary_sha256=sha256_file(stage2_path),
        ),
        expected_epoch_trajectory_sha256=epoch_hashes,
        expected_selected_test_sha256=test_hash,
        expected_flat_physical_mask_sha256="a" * 64,
    )
    protocol = replace(
        spec.protocol,
        partial=replace(
            partial,
            stage1_reference=stage1_reference,
            expected_stage1_full_reference_sha256=runtime._json_sha256(
                stage1["comparison"]["by_arm"]["full"]
            ),
            stage2_reference=stage2_reference,
        ),
    )
    return protocol, stage1, stage2


def test_runtime_loads_both_hashed_references_and_accepts_strict_protocol(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = _spec()
    runtime._strict_protocol(spec.protocol)
    protocol, _stage1, _stage2 = _fixture_protocol(tmp_path)
    monkeypatch.setattr(runtime, "_ROOT", tmp_path)
    _stage1_path, loaded_stage1 = runtime._load_stage1_reference(protocol)
    stage2_path, loaded_stage2 = runtime._load_stage2_reference(
        protocol, stage1=loaded_stage1
    )
    assert stage2_path.is_file()
    assert loaded_stage2["status"] == "completed"
    stage2_path.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        runtime._load_stage2_reference(protocol, stage1=loaded_stage1)


def test_top500_parity_hashes_path_independent_epoch_rows_and_full_test(
    tmp_path: Path,
) -> None:
    protocol, _stage1, stage2 = _fixture_protocol(tmp_path)
    historical = stage2["arms"][ANCHOR_ARM_ID]
    report = {
        "selected_epoch": historical["selected_epoch"],
        "epoch_reports": copy.deepcopy(historical["epoch_reports"]),
    }
    for row in report["epoch_reports"]:
        row["checkpoint"] = "/different/output/root/checkpoint.pt"
        row["checkpoint_sha256"] = "different-wrapper-byte-hash"
    gate = runtime._assert_anchor_historical_epoch_parity(
        report, stage2=stage2, protocol=protocol
    )
    assert gate["selected_epoch_exact"]
    report["epoch_reports"][1]["issued_commands_cumulative"] += 1
    with pytest.raises(RuntimeError, match="trajectory hash"):
        runtime._assert_anchor_historical_epoch_parity(
            report, stage2=stage2, protocol=protocol
        )
    test = copy.deepcopy(historical["selected_test"])
    assert runtime._assert_anchor_historical_test_parity(
        test, stage2=stage2, protocol=protocol
    )["exact"]
    test["student_correct"] -= 1
    with pytest.raises(RuntimeError, match="terminal test hash"):
        runtime._assert_anchor_historical_test_parity(
            test, stage2=stage2, protocol=protocol
        )


def _report(accuracy: float, kl_value: float, pulses: int) -> dict[str, object]:
    examples = 5_000
    return {
        "selected_epoch": 1,
        "epoch_reports": [
            {
                "epoch": 1,
                "validation": {
                    "student_accuracy": accuracy,
                    "student_correct": round(accuracy * examples),
                    "examples": examples,
                    "kl_teacher_student": kl_value,
                },
                "issued_commands_cumulative": pulses,
            }
        ],
    }


def test_selector_uses_top500_baseline_and_ignores_random_validation_control() -> None:
    spec = _spec()
    stage1 = _stage1_fixture()
    reports = {
        ANCHOR_ARM_ID: _report(0.8800, 0.35, 30_000),
        "w2_plus_w1_top1000": _report(0.8890, 0.30, 40_000),
        "w2_plus_w1_top2000": _report(0.8950, 0.29, 60_000),
        "w2_plus_w1_top4000": _report(0.9100, 0.25, 70_000),
        "w2_plus_w1_random2000": _report(0.9990, 0.01, 1),
    }
    winner, gates = runtime._select_smallest_qualifying_larger_union(
        reports, stage1=stage1, protocol=spec.protocol
    )
    assert winner == "w2_plus_w1_top2000"
    assert not gates["w2_plus_w1_top1000"]["validation_accuracy_gate"]
    assert gates["w2_plus_w1_top2000"]["qualifies"]
    assert "w2_plus_w1_random2000" not in gates


def test_selector_requires_lower_kl_and_stage1_full_pulse_gate() -> None:
    spec = _spec()
    stage1 = _stage1_fixture()
    reports = {
        ANCHOR_ARM_ID: _report(0.8800, 0.35, 30_000),
        # Accuracy passes, but KL is worse than the anchor.
        "w2_plus_w1_top1000": _report(0.8950, 0.36, 40_000),
        # Accuracy and KL pass, but 80k exceeds 25% of 303,231 pulses.
        "w2_plus_w1_top2000": _report(0.9000, 0.29, 80_000),
        "w2_plus_w1_top4000": _report(0.9100, 0.25, 70_000),
        "w2_plus_w1_random2000": _report(0.9990, 0.01, 1),
    }
    winner, gates = runtime._select_smallest_qualifying_larger_union(
        reports, stage1=stage1, protocol=spec.protocol
    )
    assert winner == "w2_plus_w1_top4000"
    assert not gates["w2_plus_w1_top1000"]["lower_validation_KL_gate"]
    assert not gates["w2_plus_w1_top2000"]["pulse_gate"]
    assert gates["w2_plus_w1_top4000"]["qualifies"]
