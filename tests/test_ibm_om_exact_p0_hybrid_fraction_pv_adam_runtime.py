from __future__ import annotations

import copy
from dataclasses import replace
import json
from pathlib import Path

import pytest

from experiments.definitions import resolve_experiment_config
from experiments.artifacts import sha256_file
from experiments.mnist_relu_drn import (
    ibm_om_exact_p0_hybrid_fraction_pv_adam_runtime as runtime,
)
from experiments.schema import RunMode


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "examples/mnist_relu_drn/ibm_om_exact_p0_hybrid_fraction_pv_adam/fraction_sweep.json"


def _spec():
    _definition, spec = resolve_experiment_config(CONFIG, RunMode.TRAIN)
    return spec


def _stage1_fixture() -> dict[str, object]:
    rows = [
        {
            "epoch": epoch,
            "checkpoint": f"/historical/w2_epoch_{epoch}.pt",
            "checkpoint_sha256": str(epoch) * 64,
            "issued_commands_cumulative": epoch * 100,
            "validation_prediction_sha256": str(epoch + 3) * 64,
        }
        for epoch in (1, 2, 3)
    ]
    return {
        "schema": runtime.STAGE1_SUMMARY_SCHEMA,
        "schema_version": 1,
        "status": "completed",
        "evidence_tier": runtime.EVIDENCE_TIER,
        "exact_p0": {"sha256": _spec().protocol.base.p0.sha256},
        "protocol": {
            "arms": ["full", "w1_only", "w2_only", "w1_top500", "global_top500"]
        },
        "matched_gates": {
            "test_sealed_until_all_epoch_selections_frozen": True,
            "full_mask_historical_parity_exact": True,
        },
        "arms": {
            "w2_only": {
                "selected_epoch": 3,
                "epoch_reports": rows,
                "selected_test": {
                    "student_correct": 8762,
                    "prediction_sha256": "e" * 64,
                    "kl_teacher_student": 0.3469,
                },
            }
        },
        "comparison": {
            "by_arm": {
                "full": {
                    "selected_checkpoint_cost": {"issued_commands": 303_231}
                }
            }
        },
    }


def test_runtime_loads_hashed_stage1_reference_and_accepts_strict_protocol(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = _spec()
    runtime._strict_protocol(spec.protocol)
    payload = _stage1_fixture()
    path = tmp_path / "stage1.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    reference = replace(
        spec.protocol.partial.stage1_reference,
        summary_path="stage1.json",
        summary_sha256=sha256_file(path),
    )
    protocol = replace(
        spec.protocol,
        partial=replace(spec.protocol.partial, stage1_reference=reference),
    )
    monkeypatch.setattr(runtime, "_ROOT", tmp_path)
    path, stage1 = runtime._load_stage1_reference(protocol)
    assert path.is_file()
    assert stage1["status"] == "completed"
    path.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        runtime._load_stage1_reference(protocol)


def test_w2_historical_gate_hashes_the_complete_path_independent_epoch_rows() -> None:
    stage1 = _stage1_fixture()
    historical = stage1["arms"]["w2_only"]
    report = {
        "selected_epoch": historical["selected_epoch"],
        "epoch_reports": copy.deepcopy(historical["epoch_reports"]),
    }
    for row in report["epoch_reports"]:
        row["checkpoint"] = "/different/output/root/checkpoint.pt"
        row["checkpoint_sha256"] = "not-byte-stable-because-atomic-temp-name"
    gate = runtime._assert_w2_historical_epoch_parity(report, stage1=stage1)
    assert gate["stage1_epoch_trajectory_sha256"] == gate[
        "observed_epoch_trajectory_sha256"
    ]
    report["epoch_reports"][1]["issued_commands_cumulative"] += 1
    with pytest.raises(RuntimeError, match="trajectory hash"):
        runtime._assert_w2_historical_epoch_parity(report, stage1=stage1)


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


def test_selector_chooses_smallest_union_passing_accuracy_kl_and_pulse_gates() -> None:
    spec = _spec()
    stage1 = _stage1_fixture()
    reports = {
        "w2_only": _report(0.8656, 0.3977, 17_844),
        "w2_plus_w1_top125": _report(0.8740, 0.35, 30_000),
        "w2_plus_w1_top250": _report(0.8760, 0.39, 40_000),
        "w2_plus_w1_top500": _report(0.8900, 0.30, 50_000),
        "w2_plus_w1_random500": _report(0.99, 0.01, 1),
    }
    winner, gates = runtime._select_smallest_qualifying_union(
        reports, stage1=stage1, protocol=spec.protocol
    )
    assert winner == "w2_plus_w1_top250"
    assert not gates["w2_plus_w1_top125"]["validation_accuracy_gate"]
    assert gates["w2_plus_w1_top250"]["qualifies"]
    assert "w2_plus_w1_random500" not in gates


def test_w2_terminal_test_parity_uses_full_payload_hash() -> None:
    stage1 = _stage1_fixture()
    test = copy.deepcopy(stage1["arms"]["w2_only"]["selected_test"])
    gate = runtime._assert_w2_historical_test_parity(test, stage1=stage1)
    assert gate["exact"]
    test["student_correct"] -= 1
    with pytest.raises(RuntimeError, match="terminal test hash"):
        runtime._assert_w2_historical_test_parity(test, stage1=stage1)
