from __future__ import annotations

import json

import pytest

from scripts.verify_canary import (
    parse_sacct,
    require_semantic_json,
    task_rows,
    validate_task_rows,
)


def _row(job_id: str, state: str = "COMPLETED", exit_code: str = "0:0") -> str:
    return (
        f"{job_id}|{state}|{exit_code}|fmu@v100|gpu_p13|qos_gpu-t3|"
        "v100-16g|billing=1,cpu=10,gres/gpu=1,node=1"
    )


def test_singleton_array_parent_representation_is_one_task() -> None:
    rows = parse_sacct("\n".join([_row("123"), _row("123.batch"), _row("123.extern")]))

    assert [row["JobIDRaw"] for row in task_rows("123", rows)] == ["123"]


def test_explicit_array_task_representation_is_one_task() -> None:
    rows = parse_sacct("\n".join([_row("123_0"), _row("123_0.batch")]))

    assert [row["JobIDRaw"] for row in task_rows("123", rows)] == ["123_0"]


def test_complete_multi_task_array_keeps_every_task() -> None:
    rows = parse_sacct("\n".join([_row("123_0"), _row("123_1"), _row("123_2")]))

    assert [row["JobIDRaw"] for row in task_rows("123", rows)] == [
        "123_0",
        "123_1",
        "123_2",
    ]


def _validate(job_id, rows, expected_tasks):
    validate_task_rows(
        job_id,
        rows,
        expected_tasks=expected_tasks,
        expected_account="fmu@v100",
        expected_partition="gpu_p13",
        expected_qos="qos_gpu-t3",
        expected_constraint="v100-16g",
    )


def test_scheduler_contract_accepts_complete_array() -> None:
    rows = parse_sacct("\n".join([_row("123_0"), _row("123_1"), _row("123_2")]))

    _validate("123", task_rows("123", rows), expected_tasks=3)


@pytest.mark.parametrize(
    "text, expected_tasks",
    [
        ("", 1),
        ("\n".join([_row("123_0"), _row("123_0")]), 1),
        (_row("123_7"), 1),
        ("\n".join([_row("123_0"), _row("123_2")]), 2),
    ],
)
def test_scheduler_contract_rejects_missing_duplicate_or_unexpected_rows(
    text,
    expected_tasks,
) -> None:
    rows = parse_sacct(text)

    with pytest.raises(RuntimeError):
        _validate("123", task_rows("123", rows), expected_tasks=expected_tasks)


def test_scheduler_contract_rejects_failed_and_mixed_rows() -> None:
    failed = parse_sacct(_row("123_0", state="FAILED", exit_code="1:0"))
    mixed = parse_sacct("\n".join([_row("123_0"), _row("123_1", state="FAILED")]))

    with pytest.raises(RuntimeError, match="mismatch"):
        _validate("123", task_rows("123", failed), expected_tasks=1)
    with pytest.raises(RuntimeError, match="mismatch"):
        _validate("123", task_rows("123", mixed), expected_tasks=2)


def test_malformed_sacct_row_fails() -> None:
    with pytest.raises(ValueError, match="Malformed"):
        parse_sacct("123|COMPLETED")


def test_semantic_summary_requires_one_complete_row(tmp_path) -> None:
    path = tmp_path / "summary.json"
    path.write_text(json.dumps([{"status": "complete", "rho_conv": 0.009}]))

    result = require_semantic_json(path)

    assert result["row"]["status"] == "complete"


@pytest.mark.parametrize("payload", [[], [{"status": "running"}], {}])
def test_semantic_summary_rejects_incomplete_payload(tmp_path, payload) -> None:
    path = tmp_path / "summary.json"
    path.write_text(json.dumps(payload))

    with pytest.raises(RuntimeError):
        require_semantic_json(path)
