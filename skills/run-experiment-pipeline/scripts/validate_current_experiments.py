#!/usr/bin/env python3
"""Validate the minimal human-readable current-experiments tracker."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import sys
from typing import Any
import uuid


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_TRACKER = REPO_ROOT / "docs" / "current_experiments.md"
ID_PATTERN = r"[a-z0-9][a-z0-9._-]*"
ID_RE = re.compile(rf"^{ID_PATTERN}$")
MARKER_PREFIX_RE = re.compile(r"(?m)^.*<!--\s*experiment-id:")
MARKER_RE = re.compile(
    rf"(?m)^<!-- experiment-id: (?P<experiment_id>{ID_PATTERN}) -->[ \t]*$"
)
HEADING_RE = re.compile(r"^(?P<marks>#{2,6})[ \t]+(?P<title>\S.*?)[ \t]*$")
FIELD_RE = re.compile(
    r"^[ \t]*-[ \t]+\*\*(?P<label>[^*:\n]+):\*\*[ \t]*(?P<value>.*?)[ \t]*$"
)
REQUIRED_FIELDS = ("Testing", "Where", "Status")
LIVE_STATUSES = frozenset(
    {
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
)
STATUS_RE = re.compile(
    r"^`(?P<status>[a-z][a-z-]*)`"
    r"(?:[ \t]+\N{EM DASH}[ \t]+(?P<detail>\S.*))?$"
)
PLACEHOLDER_RE = re.compile(
    r"(?:\bTODO\b|\bTBD\b|REPLACE_ME|<[^>\n]+>|\bunknown\b|"
    r"\bnot assigned(?: yet)?\b|\bto be (?:decided|determined|assigned)\b)",
    re.IGNORECASE,
)
GATE_RECEIPT_SCHEMA_VERSION = "current-experiments-launch-gate/v1"
GATE_STAGE_STATUS = {
    "canary": "preflighting",
    "production": "launch-ready",
}


class TrackerError(ValueError):
    """Raised when the live experiment tracker violates its contract."""


@dataclass(frozen=True)
class TrackerEntry:
    """One minimal live experiment entry."""

    experiment_id: str
    title: str
    testing: str
    where: str
    status: str


def require(condition: bool, expected: str, provided: Any) -> None:
    """Raise an expected-first validation error when ``condition`` is false."""

    if not condition:
        raise TrackerError(f"Expected {expected}; got {provided!r}")


def load_tracker(path: Path) -> str:
    """Read a tracker as UTF-8 text."""

    try:
        return path.read_text(encoding="utf-8")
    except OSError as exc:
        raise TrackerError(f"Expected a readable tracker at {path}; got {exc}") from exc


def load_tracker_snapshot(path: Path) -> tuple[str, str]:
    """Read and hash the exact tracker bytes used for validation once."""

    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise TrackerError(
            f"Expected a readable tracker at {path}; got {exc}"
        ) from exc
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise TrackerError(
            f"Expected UTF-8 tracker text at {path}; got {exc}"
        ) from exc
    return text, hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_gate_receipt(
    path: Path,
    *,
    tracker_path: Path,
    tracker_sha256: str,
    experiment: TrackerEntry,
    gate_stage: str,
    gate_state: Path,
    gate_output_root: Path,
) -> dict[str, Any]:
    expected_status = GATE_STAGE_STATUS[gate_stage]
    require(
        experiment.status == expected_status,
        f"experiment_id {experiment.experiment_id!r} with Status "
        f"{expected_status!r} for the {gate_stage!r} gate",
        experiment.status,
    )
    destination = Path(os.path.abspath(path.expanduser()))
    require(
        destination.parent.is_dir(),
        "gate-receipt parent to be an existing directory",
        destination.parent,
    )
    require(
        not destination.exists() and not destination.is_symlink(),
        "gate-receipt path not to exist",
        destination,
    )
    validator_path = Path(__file__).resolve()
    value = {
        "schema_version": GATE_RECEIPT_SCHEMA_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment_id": experiment.experiment_id,
        "gate_stage": gate_stage,
        "status": experiment.status,
        "state_path": str(Path(os.path.abspath(gate_state.expanduser()))),
        "output_root": str(
            Path(os.path.abspath(gate_output_root.expanduser()))
        ),
        "tracker_path": str(tracker_path),
        "tracker_sha256": tracker_sha256,
        "validator_path": str(validator_path),
        "validator_sha256": sha256_file(validator_path),
    }
    payload = (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    )
    temporary = destination.with_name(
        f".{destination.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    )
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        # Publish a fully written file with no-replace semantics. A competing
        # writer or an old attempt receipt wins rather than being overwritten.
        os.link(temporary, destination)
        directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        directory_fd = os.open(destination.parent, directory_flags)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except OSError as exc:
        raise TrackerError(
            "Expected gate-receipt publication to create one fully written "
            f"new file at {destination}; got {exc}"
        ) from exc
    finally:
        if temporary.exists():
            temporary.unlink()
    return value


def _marker_matches(text: str) -> list[re.Match[str]]:
    marker_like = MARKER_PREFIX_RE.findall(text)
    markers = list(MARKER_RE.finditer(text))
    require(
        len(marker_like) == len(markers),
        "every experiment marker to use '<!-- experiment-id: ID -->' "
        "with a filesystem-safe lowercase ID",
        len(marker_like) - len(markers),
    )
    return markers


def _entry_from_block(experiment_id: str, block: str) -> TrackerEntry:
    lines = block.splitlines()
    first_content_index = next(
        (index for index, line in enumerate(lines) if line.strip()),
        None,
    )
    require(
        first_content_index is not None,
        f"a Markdown heading after experiment marker {experiment_id!r}",
        "end of entry",
    )
    first_content = lines[first_content_index]  # type: ignore[index]
    heading_match = HEADING_RE.fullmatch(first_content)
    require(
        heading_match is not None,
        f"a level 2-6 Markdown heading immediately after experiment marker "
        f"{experiment_id!r}",
        first_content.strip(),
    )
    title = heading_match.group("title").strip()  # type: ignore[union-attr]
    require(bool(title), f"a non-empty heading for {experiment_id!r}", title)

    field_values: dict[str, list[str]] = {}
    unknown_fields: list[str] = []
    for line in lines[first_content_index + 1 :]:  # type: ignore[operator]
        match = FIELD_RE.fullmatch(line)
        if match is None:
            continue
        label = match.group("label").strip()
        value = match.group("value").strip()
        if label not in REQUIRED_FIELDS:
            unknown_fields.append(label)
            continue
        field_values.setdefault(label, []).append(value)

    require(
        not unknown_fields,
        f"only fields {list(REQUIRED_FIELDS)!r} in experiment entry "
        f"{experiment_id!r}",
        unknown_fields,
    )
    for label in REQUIRED_FIELDS:
        occurrences = field_values.get(label, [])
        require(
            len(occurrences) == 1,
            f"exactly one '{label}' field in experiment entry {experiment_id!r}",
            len(occurrences),
        )
        require(
            bool(occurrences[0]),
            f"a non-empty '{label}' field in experiment entry {experiment_id!r}",
            occurrences[0],
        )

    raw_status = field_values["Status"][0]
    status_match = STATUS_RE.fullmatch(raw_status)
    require(
        status_match is not None,
        "Status beginning with one backticked lowercase token, optionally "
        "followed by ' — concise progress and next action'",
        raw_status,
    )
    status = status_match.group("status")  # type: ignore[union-attr]
    require(
        status in LIVE_STATUSES,
        f"a live Status in {sorted(LIVE_STATUSES)!r}",
        status,
    )
    if status == "launch-ready":
        for label in ("Testing", "Where"):
            value = field_values[label][0]
            require(
                PLACEHOLDER_RE.search(value) is None,
                f"a non-placeholder '{label}' field for launch-ready "
                f"experiment {experiment_id!r}",
                value,
            )
    return TrackerEntry(
        experiment_id=experiment_id,
        title=title,
        testing=field_values["Testing"][0],
        where=field_values["Where"][0],
        status=status,
    )


def validate_tracker(
    text: str,
    *,
    required_experiment_id: str | None = None,
    require_launch_ready: bool = False,
) -> dict[str, TrackerEntry]:
    """Validate every marked entry and optional pre-launch requirements."""

    if required_experiment_id is not None:
        require(
            ID_RE.fullmatch(required_experiment_id) is not None,
            "a filesystem-safe lowercase --require-experiment-id",
            required_experiment_id,
        )
    require(
        not require_launch_ready or required_experiment_id is not None,
        "--require-experiment-id when --require-launch-ready is used",
        required_experiment_id,
    )

    markers = _marker_matches(text)
    entries: dict[str, TrackerEntry] = {}
    for index, marker in enumerate(markers):
        experiment_id = marker.group("experiment_id")
        require(
            experiment_id not in entries,
            "unique experiment-id markers",
            experiment_id,
        )
        block_end = markers[index + 1].start() if index + 1 < len(markers) else len(text)
        entries[experiment_id] = _entry_from_block(
            experiment_id,
            text[marker.end() : block_end],
        )

    if required_experiment_id is not None:
        require(
            required_experiment_id in entries,
            f"a tracker entry for experiment_id {required_experiment_id!r}",
            sorted(entries),
        )
        if require_launch_ready:
            observed = entries[required_experiment_id].status
            require(
                observed == "launch-ready",
                f"experiment_id {required_experiment_id!r} with Status "
                "'launch-ready'",
                observed,
            )
    return entries


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "tracker",
        nargs="?",
        type=Path,
        default=DEFAULT_TRACKER,
        help="tracker Markdown path (default: docs/current_experiments.md)",
    )
    parser.add_argument("--require-experiment-id")
    parser.add_argument(
        "--require-launch-ready",
        action="store_true",
        help="require the selected experiment entry to have Status `launch-ready`",
    )
    parser.add_argument(
        "--write-gate-receipt",
        type=Path,
        help=(
            "write one immutable, hash-bound canary/production gate receipt"
        ),
    )
    parser.add_argument(
        "--gate-stage",
        choices=sorted(GATE_STAGE_STATUS),
        help="stage authorized by --write-gate-receipt",
    )
    parser.add_argument(
        "--gate-state",
        type=Path,
        help=(
            "exact supervisor state path authorized by "
            "--write-gate-receipt"
        ),
    )
    parser.add_argument(
        "--gate-output-root",
        type=Path,
        help=(
            "exact experiment output root authorized by "
            "--write-gate-receipt"
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    tracker_path = args.tracker.expanduser().resolve()
    try:
        gate_arguments = {
            "write_gate_receipt": args.write_gate_receipt,
            "gate_stage": args.gate_stage,
            "gate_state": args.gate_state,
            "gate_output_root": args.gate_output_root,
        }
        require(
            all(value is None for value in gate_arguments.values())
            or all(value is not None for value in gate_arguments.values()),
            "--write-gate-receipt, --gate-stage, --gate-state, and "
            "--gate-output-root to be supplied together",
            gate_arguments,
        )
        require(
            args.write_gate_receipt is None
            or args.require_experiment_id is not None,
            "--require-experiment-id when writing a gate receipt",
            args.require_experiment_id,
        )
        tracker_text, tracker_sha256 = load_tracker_snapshot(tracker_path)
        entries = validate_tracker(
            tracker_text,
            required_experiment_id=args.require_experiment_id,
            require_launch_ready=args.require_launch_ready,
        )
        receipt = None
        if args.write_gate_receipt is not None:
            receipt = write_gate_receipt(
                args.write_gate_receipt,
                tracker_path=tracker_path,
                tracker_sha256=tracker_sha256,
                experiment=entries[args.require_experiment_id],
                gate_stage=args.gate_stage,
                gate_state=args.gate_state,
                gate_output_root=args.gate_output_root,
            )
        print(f"OK {tracker_path} ({len(entries)} live experiment entries)")
        if receipt is not None:
            print(
                "GATE_RECEIPT "
                + json.dumps(
                    {
                        **receipt,
                        "receipt_path": str(
                            Path(
                                os.path.abspath(
                                    args.write_gate_receipt.expanduser()
                                )
                            )
                        ),
                        "receipt_sha256": sha256_file(
                            Path(
                                os.path.abspath(
                                    args.write_gate_receipt.expanduser()
                                )
                            )
                        ),
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                )
            )
        return 0
    except TrackerError as exc:
        print(f"Current-experiments validation failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
