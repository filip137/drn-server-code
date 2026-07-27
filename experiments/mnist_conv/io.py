"""Atomic, deterministic filesystem helpers."""

from __future__ import annotations

import csv
import json
import os
import uuid
from pathlib import Path
from typing import Any, Iterable, Mapping

from .identity import canonical_json_bytes


def _reject_json_constant(value: str) -> None:
    raise ValueError(
        f"Expected strict JSON without NaN or Infinity. Provided value: {value!r}."
    )


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(
                f"Expected every JSON object key to be unique. Provided duplicate key: {key!r}."
            )
        value[key] = item
    return value


def atomic_write_bytes(path: str | Path, data: bytes) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()
    return target


def atomic_create_bytes(path: str | Path, data: bytes) -> Path:
    """Publish fully written bytes exactly once without replacing evidence."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        raise FileExistsError(
            f"Expected immutable output path not to exist. Provided value: "
            f"{target}."
        )
    temporary = target.with_name(
        f".{target.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    )
    try:
        with temporary.open("xb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        # A hard link is atomic and fails if another writer won the exact
        # destination; unlike os.replace, it never destroys prior evidence.
        os.link(temporary, target)
        directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        directory_fd = os.open(target.parent, directory_flags)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary.exists():
            temporary.unlink()
    return target


def atomic_write_json(path: str | Path, value: Any, *, canonical: bool = False) -> Path:
    if canonical:
        data = canonical_json_bytes(value) + b"\n"
    else:
        data = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    return atomic_write_bytes(path, data)


def atomic_create_json(
    path: str | Path,
    value: Any,
    *,
    canonical: bool = False,
) -> Path:
    if canonical:
        data = canonical_json_bytes(value) + b"\n"
    else:
        data = (
            json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
        ).encode("utf-8")
    return atomic_create_bytes(path, data)


def atomic_write_csv(
    path: str | Path,
    fieldnames: list[str],
    rows: Iterable[Mapping[str, Any]],
) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()
    return target


def read_json(path: str | Path) -> Any:
    return json.loads(
        Path(path).read_text(),
        parse_constant=_reject_json_constant,
        object_pairs_hook=_unique_json_object,
    )


def relative_posix(path: str | Path, start: str | Path) -> str:
    return Path(os.path.relpath(Path(path), start=Path(start))).as_posix()
