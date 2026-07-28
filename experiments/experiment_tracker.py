#!/usr/bin/env python3
"""Executor-owned updates for the concise current-experiments tracker.

The user never authors this Markdown.  A prepared simplified flow supplies
the scientific purpose and target/result locations; the executor supplies the
observed lifecycle state and concise detail.  The repository's canonical
tracker validator checks the complete document before publication.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
from pathlib import Path
import re
import sys
from typing import Any, Mapping, Sequence
import uuid

from experiments.mnist_conv.io import read_json


FLOW_SCHEMA = "experiment-prepared-flow/v1"
REPO_ROOT = Path(__file__).resolve().parents[1]
VALIDATOR_PATH = (
    REPO_ROOT
    / "skills"
    / "run-experiment-pipeline"
    / "scripts"
    / "validate_current_experiments.py"
)
DEFAULT_TRACKER = REPO_ROOT / "docs" / "current_experiments.md"
MARKER_RE = re.compile(
    r"(?m)^<!-- experiment-id: (?P<experiment_id>[a-z0-9][a-z0-9._-]*) -->"
    r"[ \t]*$"
)
ENTRY_BLOCK_RE = re.compile(
    r"(?m)^<!-- experiment-id: "
    r"(?P<experiment_id>[a-z0-9][a-z0-9._-]*) -->[ \t]*\n"
    r"#{2,6}[ \t]+\S.*[ \t]*\n"
    r"(?:[ \t]*\n)*"
    r"-[ \t]+\*\*Testing:\*\*[ \t]+\S.*[ \t]*\n"
    r"-[ \t]+\*\*Where:\*\*[ \t]+\S.*[ \t]*\n"
    r"-[ \t]+\*\*Status:\*\*[ \t]+\S.*(?:\n|$)"
)
ALLOWED_STATES = {
    "planned",
    "preflighting",
    "launch-ready",
    "queued",
    "running",
    "remote-closed",
    "transferring",
    "validating",
    "review-pending",
    "blocked",
}


class ExperimentTrackerError(ValueError):
    """A prepared flow cannot be projected into valid tracker state."""


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ExperimentTrackerError(
            f"Expected {label} to be non-empty text. "
            f"Provided value: {value!r}."
        )
    return " ".join(value.strip().split())


def _validator() -> Any:
    spec = importlib.util.spec_from_file_location(
        "_generated_current_experiments_validator",
        VALIDATOR_PATH,
    )
    if spec is None or spec.loader is None:
        raise ExperimentTrackerError(
            f"Expected an importable tracker validator. "
            f"Provided value: {VALIDATOR_PATH}."
        )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def tracker_entry_from_prepared(
    prepared: Mapping[str, Any],
    *,
    status: str,
    where: str | None = None,
    detail: str,
    request_link: str | None = None,
) -> dict[str, str]:
    """Build one minimal tracker entry entirely from generated state."""

    if prepared.get("schema_version") != FLOW_SCHEMA:
        raise ExperimentTrackerError(
            f"Expected schema_version {FLOW_SCHEMA!r}. "
            f"Provided value: {prepared.get('schema_version')!r}."
        )
    if prepared.get("status") not in {
        "ready_for_functional_smokes",
        "ready_for_production",
        "preflight_failed",
    }:
        raise ExperimentTrackerError(
            "Expected an approved prepared flow before tracker publication. "
            f"Provided value: {prepared.get('status')!r}."
        )
    if status not in ALLOWED_STATES:
        raise ExperimentTrackerError(
            f"Expected status to be one of {sorted(ALLOWED_STATES)!r}. "
            f"Provided value: {status!r}."
        )
    resolved = prepared.get("resolved_study")
    if not isinstance(resolved, Mapping):
        raise ExperimentTrackerError(
            "Expected resolved_study to be a mapping. "
            f"Provided value: {resolved!r}."
        )
    study = resolved.get("study")
    if not isinstance(study, Mapping):
        raise ExperimentTrackerError(
            "Expected resolved_study.study to be a mapping. "
            f"Provided value: {study!r}."
        )
    binding = resolved.get("bindings_and_provenance")
    binding = binding if isinstance(binding, Mapping) else {}
    catalog = binding.get("catalog_resolution")
    catalog = catalog if isinstance(catalog, Mapping) else {}
    metadata = catalog.get("request_metadata")
    metadata = metadata if isinstance(metadata, Mapping) else {}

    experiment_id = _text(resolved.get("study_id"), "resolved study ID")
    title = _text(
        metadata.get("experiment", experiment_id),
        "tracker title",
    )
    testing = _text(study.get("purpose"), "tracker purpose")
    if request_link is not None:
        testing += f" [{request_link}]"

    if where is None:
        plan = prepared.get("production_plan") or prepared.get(
            "validation_plan"
        )
        attempts = (
            plan.get("attempts")
            if isinstance(plan, Mapping)
            else None
        )
        if not isinstance(attempts, list) or not attempts:
            raise ExperimentTrackerError(
                "Expected a generated execution plan with target attempts. "
                f"Provided value: {attempts!r}."
            )
        locations = []
        for attempt in attempts:
            if not isinstance(attempt, Mapping):
                raise ExperimentTrackerError(
                    "Expected each execution attempt to be a mapping. "
                    f"Provided value: {attempt!r}."
                )
            locations.append(
                f"{_text(attempt.get('target_id'), 'target ID')}: "
                f"{_text(attempt.get('result_root'), 'result root')}"
            )
        where = "; ".join(locations)

    return {
        "experiment_id": experiment_id,
        "title": title,
        "testing": testing,
        "where": _text(where, "tracker location"),
        "status": status,
        "detail": _text(detail, "tracker status detail"),
    }


def render_tracker_entry(entry: Mapping[str, str]) -> str:
    return (
        f"<!-- experiment-id: {entry['experiment_id']} -->\n"
        f"### {entry['title']}\n\n"
        f"- **Testing:** {entry['testing']}\n"
        f"- **Where:** {entry['where']}\n"
        f"- **Status:** `{entry['status']}` — {entry['detail']}\n"
    )


def _replace_or_append(text: str, entry: Mapping[str, str]) -> str:
    matches = list(MARKER_RE.finditer(text))
    selected = [
        (index, match)
        for index, match in enumerate(matches)
        if match.group("experiment_id") == entry["experiment_id"]
    ]
    if len(selected) > 1:
        raise ExperimentTrackerError(
            "Expected at most one existing tracker entry for "
            f"{entry['experiment_id']!r}. Provided value: {len(selected)}."
        )
    rendered = render_tracker_entry(entry).rstrip()
    if selected:
        blocks = [
            match
            for match in ENTRY_BLOCK_RE.finditer(text)
            if match.group("experiment_id") == entry["experiment_id"]
        ]
        if len(blocks) != 1:
            raise ExperimentTrackerError(
                "Expected the existing tracker entry to match the minimal "
                f"three-field contract. Provided value: {len(blocks)} "
                "matching blocks."
            )
        block = blocks[0]
        suffix = text[block.end() :]
        newline = "" if suffix.startswith("\n") else "\n"
        return text[: block.start()] + rendered + newline + suffix
    return text.rstrip() + "\n\n" + rendered + "\n"


def upsert_tracker(path: str | Path, entry: Mapping[str, str]) -> str:
    """Validate then atomically publish one generated tracker transition."""

    destination = Path(path).expanduser().resolve()
    try:
        original = destination.read_text(encoding="utf-8")
    except OSError as exc:
        raise ExperimentTrackerError(
            f"Expected a readable tracker at {destination}. "
            f"Provided value: {exc!r}."
        ) from exc
    candidate = _replace_or_append(original, entry)
    validator = _validator()
    try:
        validator.validate_tracker(
            candidate,
            required_experiment_id=entry["experiment_id"],
            require_launch_ready=entry["status"] == "launch-ready",
        )
    except validator.TrackerError as exc:
        raise ExperimentTrackerError(str(exc)) from exc
    if candidate == original:
        return "unchanged"
    temporary = destination.with_name(
        f".{destination.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    )
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            handle.write(candidate)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return "updated"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate and upsert an executor-owned tracker entry."
    )
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--tracker", type=Path, default=DEFAULT_TRACKER)
    parser.add_argument("--status", choices=sorted(ALLOWED_STATES), required=True)
    parser.add_argument("--where")
    parser.add_argument("--detail", required=True)
    parser.add_argument("--request-link")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    prepared = read_json(args.prepared)
    entry = tracker_entry_from_prepared(
        prepared,
        status=args.status,
        where=args.where,
        detail=args.detail,
        request_link=args.request_link,
    )
    outcome = upsert_tracker(args.tracker, entry)
    print(f"{outcome}: {entry['experiment_id']} -> {entry['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ALLOWED_STATES",
    "ExperimentTrackerError",
    "render_tracker_entry",
    "tracker_entry_from_prepared",
    "upsert_tracker",
]
