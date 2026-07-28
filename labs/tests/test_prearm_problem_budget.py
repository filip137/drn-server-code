from datetime import datetime, timezone

from experiments.prearm_problem_budget import (
    record_failed_repair,
    start_budget,
)


NOW = datetime(2026, 7, 28, 14, 0, tzinfo=timezone.utc)


def test_second_independent_failure_asks_user() -> None:
    state = start_budget("flow-a", now=NOW)
    state = record_failed_repair(
        state,
        stage="target_probe",
        error="dataset missing",
        attempted_repair="corrected data root",
        elapsed_seconds=120,
        now=NOW,
    )
    assert state["status"] == "active"
    state = record_failed_repair(
        state,
        stage="functional_smoke",
        error="CUDA operation unsupported",
        attempted_repair="used supported adapter",
        elapsed_seconds=240,
        now=NOW,
    )
    assert state["status"] == "ask_user"
    assert state["stop_reason"] == "repair_attempt_budget_exhausted"


def test_repeated_unchanged_error_stops_immediately() -> None:
    state = start_budget("flow-b", now=NOW, maximum_failed_attempts=4)
    for attempt in ("first change", "second change"):
        state = record_failed_repair(
            state,
            stage="parser",
            error="same schema error",
            attempted_repair=attempt,
            elapsed_seconds=30,
            now=NOW,
        )
    assert state["status"] == "ask_user"
    assert state["stop_reason"] == "repeated_unchanged_error"


def test_time_limit_asks_user_before_second_attempt() -> None:
    state = start_budget("flow-c", now=NOW)
    state = record_failed_repair(
        state,
        stage="staging",
        error="remote path unavailable",
        attempted_repair="checked target profile",
        elapsed_seconds=600,
        valid_artifacts=["local-study.json"],
        now=NOW,
    )
    assert state["status"] == "ask_user"
    assert state["stop_reason"] == "time_budget_exhausted"
    assert state["failed_attempts"][0]["valid_artifacts"] == [
        "local-study.json"
    ]
