"""Small state machine that prevents pre-arm repair loops."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from typing import Any, Mapping


SCHEMA = "experiment-prearm-problem-budget/v1"
DEFAULT_SECONDS = 10 * 60
DEFAULT_FAILED_ATTEMPTS = 2


def _timestamp(value: datetime | None = None) -> str:
    current = value or datetime.now(timezone.utc)
    if current.tzinfo is None or current.utcoffset() is None:
        raise ValueError(
            "Expected a timezone-aware timestamp. "
            f"Provided value: {current!r}."
        )
    return current.isoformat()


def _fingerprint(stage: str, error: str) -> str:
    payload = json.dumps(
        {"stage": stage, "error": error},
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def start_budget(
    workflow_id: str,
    *,
    now: datetime | None = None,
    maximum_seconds: int = DEFAULT_SECONDS,
    maximum_failed_attempts: int = DEFAULT_FAILED_ATTEMPTS,
) -> dict[str, Any]:
    if not isinstance(workflow_id, str) or not workflow_id:
        raise ValueError(
            "Expected workflow_id to be non-empty text. "
            f"Provided value: {workflow_id!r}."
        )
    if maximum_seconds <= 0 or maximum_failed_attempts <= 0:
        raise ValueError(
            "Expected positive pre-arm budget limits. "
            f"Provided value: seconds={maximum_seconds!r}, "
            f"failed_attempts={maximum_failed_attempts!r}."
        )
    started = _timestamp(now)
    return {
        "schema_version": SCHEMA,
        "workflow_id": workflow_id,
        "status": "active",
        "started_at": started,
        "maximum_seconds": maximum_seconds,
        "maximum_failed_attempts": maximum_failed_attempts,
        "failed_attempts": [],
        "stop_reason": None,
        "next_action": "diagnose_one_bounded_repair",
    }


def record_failed_repair(
    state: Mapping[str, Any],
    *,
    stage: str,
    error: str,
    attempted_repair: str,
    elapsed_seconds: float,
    valid_artifacts: list[str] | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    value = dict(state)
    if value.get("schema_version") != SCHEMA:
        raise ValueError(
            f"Expected schema_version {SCHEMA!r}. "
            f"Provided value: {value.get('schema_version')!r}."
        )
    if value.get("status") != "active":
        raise ValueError(
            "Expected an active pre-arm budget. "
            f"Provided value: {value.get('status')!r}."
        )
    if elapsed_seconds < 0:
        raise ValueError(
            "Expected elapsed_seconds to be non-negative. "
            f"Provided value: {elapsed_seconds!r}."
        )
    attempts = [dict(item) for item in value.get("failed_attempts", [])]
    fingerprint = _fingerprint(stage, error)
    attempts.append(
        {
            "attempt_number": len(attempts) + 1,
            "stage": stage,
            "error": error,
            "error_fingerprint": fingerprint,
            "attempted_repair": attempted_repair,
            "elapsed_seconds": float(elapsed_seconds),
            "valid_artifacts": list(valid_artifacts or []),
            "recorded_at": _timestamp(now),
        }
    )
    repeated = sum(
        item["error_fingerprint"] == fingerprint for item in attempts
    ) >= 2
    time_exhausted = elapsed_seconds >= int(value["maximum_seconds"])
    attempts_exhausted = len(attempts) >= int(
        value["maximum_failed_attempts"]
    )
    if repeated:
        stop_reason = "repeated_unchanged_error"
    elif time_exhausted:
        stop_reason = "time_budget_exhausted"
    elif attempts_exhausted:
        stop_reason = "repair_attempt_budget_exhausted"
    else:
        stop_reason = None
    value.update(
        {
            "status": "ask_user" if stop_reason else "active",
            "failed_attempts": attempts,
            "stop_reason": stop_reason,
            "next_action": (
                "summarize_blocker_and_ask_user_how_to_proceed"
                if stop_reason
                else "diagnose_one_bounded_repair"
            ),
        }
    )
    return value


__all__ = [
    "DEFAULT_FAILED_ATTEMPTS",
    "DEFAULT_SECONDS",
    "SCHEMA",
    "record_failed_repair",
    "start_budget",
]
