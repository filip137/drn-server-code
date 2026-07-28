"""Shared validation primitives for experiment control-plane contracts."""

from __future__ import annotations

from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, NoReturn, Sequence


ID_RE = re.compile(r"^[a-z0-9][a-z0-9._-]*$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
SOURCE_ID_RE = re.compile(r"^[0-9a-f]{40}$")
PLACEHOLDER_RE = re.compile(
    r"(?:REPLACE_ME|replace-me|TODO|TBD|<[^>]+>|__[A-Z0-9_]+__)"
)


class ContractError(ValueError):
    """A structured, pre-arm contract failure."""

    def __init__(
        self,
        *,
        stage: str,
        code: str,
        path: str,
        expected: str,
        provided: Any,
        next_action: str | None = None,
    ) -> None:
        self.stage = stage
        self.code = code
        self.path = path
        self.expected = expected
        self.provided = provided
        self.recovery_scope = "pre_arm_repairable"
        self.next_action = next_action
        super().__init__(
            f"Expected {expected} at {path}; provided value: {provided!r}"
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "experiment-contract-error/v1",
            "status": "failed",
            "stage": self.stage,
            "error_code": self.code,
            "path": self.path,
            "expected": self.expected,
            "provided": self.provided,
            "recovery_scope": self.recovery_scope,
            "launched_job_count": 0,
            "next_action": self.next_action,
        }


def fail(
    *,
    stage: str,
    code: str,
    path: str,
    expected: str,
    provided: Any,
    next_action: str | None = None,
) -> NoReturn:
    raise ContractError(
        stage=stage,
        code=code,
        path=path,
        expected=expected,
        provided=provided,
        next_action=next_action,
    )


def require(
    condition: bool,
    *,
    stage: str,
    code: str,
    path: str,
    expected: str,
    provided: Any,
    next_action: str | None = None,
) -> None:
    if not condition:
        fail(
            stage=stage,
            code=code,
            path=path,
            expected=expected,
            provided=provided,
            next_action=next_action,
        )


def require_mapping(
    value: Any,
    *,
    stage: str,
    path: str,
) -> dict[str, Any]:
    require(
        isinstance(value, dict),
        stage=stage,
        code="wrong_type",
        path=path,
        expected="an object",
        provided=type(value).__name__,
    )
    return value


def require_exact_keys(
    value: Any,
    *,
    keys: Iterable[str],
    stage: str,
    path: str,
) -> dict[str, Any]:
    mapping = require_mapping(value, stage=stage, path=path)
    required = set(keys)
    missing = sorted(required - set(mapping))
    unknown = sorted(set(mapping) - required)
    require(
        not missing and not unknown,
        stage=stage,
        code="object_keys_mismatch",
        path=path,
        expected=(
            "exact documented keys with separate missing and unknown sets"
        ),
        provided={"missing": missing, "unknown": unknown},
    )
    return mapping


def require_text(
    value: Any,
    *,
    stage: str,
    path: str,
) -> str:
    require(
        isinstance(value, str) and bool(value.strip()),
        stage=stage,
        code="empty_text",
        path=path,
        expected="non-empty text",
        provided=value,
    )
    return value


def require_id(
    value: Any,
    *,
    stage: str,
    path: str,
) -> str:
    text = require_text(value, stage=stage, path=path)
    require(
        ID_RE.fullmatch(text) is not None and text not in {".", ".."},
        stage=stage,
        code="unsafe_id",
        path=path,
        expected="a lowercase filesystem-safe identifier",
        provided=text,
    )
    return text


def require_sha256(
    value: Any,
    *,
    stage: str,
    path: str,
) -> str:
    require(
        isinstance(value, str) and SHA256_RE.fullmatch(value) is not None,
        stage=stage,
        code="invalid_sha256",
        path=path,
        expected="a lowercase 64-character SHA-256",
        provided=value,
    )
    return value


def require_source_id(
    value: Any,
    *,
    stage: str,
    path: str,
) -> str:
    require(
        isinstance(value, str)
        and SOURCE_ID_RE.fullmatch(value) is not None,
        stage=stage,
        code="invalid_source_id",
        path=path,
        expected="a lowercase 40-character Git commit",
        provided=value,
    )
    return value


def require_bool(
    value: Any,
    *,
    stage: str,
    path: str,
) -> bool:
    require(
        isinstance(value, bool),
        stage=stage,
        code="wrong_type",
        path=path,
        expected="a boolean",
        provided=value,
    )
    return value


def require_positive_number(
    value: Any,
    *,
    stage: str,
    path: str,
) -> float:
    require(
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
        and float(value) > 0,
        stage=stage,
        code="invalid_number",
        path=path,
        expected="a positive finite number",
        provided=value,
    )
    return float(value)


def require_nonnegative_int(
    value: Any,
    *,
    stage: str,
    path: str,
) -> int:
    require(
        isinstance(value, int) and not isinstance(value, bool) and value >= 0,
        stage=stage,
        code="invalid_integer",
        path=path,
        expected="a non-negative integer",
        provided=value,
    )
    return value


def require_positive_int(
    value: Any,
    *,
    stage: str,
    path: str,
) -> int:
    require(
        isinstance(value, int) and not isinstance(value, bool) and value > 0,
        stage=stage,
        code="invalid_integer",
        path=path,
        expected="a positive integer",
        provided=value,
    )
    return value


def require_string_list(
    value: Any,
    *,
    stage: str,
    path: str,
    allow_empty: bool = False,
    unique: bool = True,
) -> list[str]:
    require(
        isinstance(value, list)
        and (allow_empty or bool(value))
        and all(isinstance(item, str) and bool(item.strip()) for item in value),
        stage=stage,
        code="invalid_string_list",
        path=path,
        expected=(
            "a list of non-empty strings"
            if allow_empty
            else "a non-empty list of non-empty strings"
        ),
        provided=value,
    )
    if unique:
        require(
            len(value) == len(set(value)),
            stage=stage,
            code="duplicate_value",
            path=path,
            expected="unique string values",
            provided=value,
        )
    return value


def require_argv(
    value: Any,
    *,
    stage: str,
    path: str,
) -> list[str]:
    result = require_string_list(
        value,
        stage=stage,
        path=path,
        unique=False,
    )
    require(
        all("\0" not in item for item in result),
        stage=stage,
        code="unsafe_argv",
        path=path,
        expected="argv tokens without NUL bytes",
        provided=result,
    )
    return result


def safe_relative_path(
    value: Any,
    *,
    stage: str,
    path: str,
) -> Path:
    text = require_text(value, stage=stage, path=path)
    candidate = Path(text)
    require(
        not candidate.is_absolute()
        and ".." not in candidate.parts
        and "\0" not in text,
        stage=stage,
        code="unsafe_path",
        path=path,
        expected="a repository-relative path without '..' or NUL",
        provided=text,
    )
    return candidate


def require_absolute_path(
    value: Any,
    *,
    stage: str,
    path: str,
) -> Path:
    text = require_text(value, stage=stage, path=path)
    candidate = Path(text)
    require(
        candidate.is_absolute() and ".." not in candidate.parts and "\0" not in text,
        stage=stage,
        code="unsafe_path",
        path=path,
        expected="an absolute path without '..' or NUL",
        provided=text,
    )
    return candidate


def canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        fail(
            stage="contract_serialization",
            code="noncanonical_json",
            path="$",
            expected="a finite JSON value",
            provided=str(exc),
        )


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        fail(
            stage="binding_verification",
            code="unreadable_file",
            path=str(path),
            expected="a readable regular file",
            provided=str(exc),
        )
    return digest.hexdigest()


def _reject_duplicate_pairs(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    duplicates: list[str] = []
    for key, value in pairs:
        if key in result:
            duplicates.append(key)
        result[key] = value
    if duplicates:
        fail(
            stage="contract_loading",
            code="duplicate_json_key",
            path="$",
            expected="unique JSON object keys",
            provided=sorted(set(duplicates)),
        )
    return result


def _reject_nonstandard_constant(value: str) -> NoReturn:
    fail(
        stage="contract_loading",
        code="nonstandard_json_number",
        path="$",
        expected="finite numbers in standard JSON syntax",
        provided=value,
    )


def loads_json(payload: str, *, source: str = "<memory>") -> dict[str, Any]:
    """Load one strict JSON object, rejecting duplicate object keys."""

    try:
        value = json.loads(
            payload,
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=_reject_nonstandard_constant,
        )
    except json.JSONDecodeError as exc:
        fail(
            stage="contract_loading",
            code="invalid_json",
            path=source,
            expected="valid JSON",
            provided=str(exc),
        )
    return require_mapping(value, stage="contract_loading", path="$")


def load_json(path: Path) -> dict[str, Any]:
    try:
        payload = path.read_text(encoding="utf-8")
    except OSError as exc:
        fail(
            stage="contract_loading",
            code="unreadable_contract",
            path=str(path),
            expected="a readable UTF-8 JSON file",
            provided=str(exc),
        )
    return loads_json(payload, source=str(path))


def write_json_first(path: Path, value: Mapping[str, Any]) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        value,
        indent=2,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
    ) + "\n"
    try:
        descriptor = os.open(
            path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
        )
    except FileExistsError:
        fail(
            stage="contract_publication",
            code="output_exists",
            path=str(path),
            expected="a new output path",
            provided="already exists",
        )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        try:
            path.unlink()
        except OSError:
            pass
        raise


def resolve_pointer(value: Any, pointer: Any, *, path: str) -> Any:
    text = require_text(pointer, stage="binding_verification", path=path)
    require(
        text.startswith("/"),
        stage="binding_verification",
        code="invalid_json_pointer",
        path=path,
        expected="a JSON pointer beginning with '/'",
        provided=text,
    )
    current = value
    for raw_part in text[1:].split("/"):
        part = raw_part.replace("~1", "/").replace("~0", "~")
        if isinstance(current, dict):
            require(
                part in current,
                stage="binding_verification",
                code="pointer_component_missing",
                path=path,
                expected=f"JSON pointer component {part!r}",
                provided=sorted(current),
            )
            current = current[part]
        elif isinstance(current, list):
            require(
                part.isdigit() and 0 <= int(part) < len(current),
                stage="binding_verification",
                code="pointer_index_invalid",
                path=path,
                expected=f"a valid list index for component {part!r}",
                provided=part,
            )
            current = current[int(part)]
        else:
            fail(
                stage="binding_verification",
                code="pointer_noncontainer",
                path=path,
                expected="the pointer to traverse an object or list",
                provided=type(current).__name__,
            )
    return current


def require_timezone_timestamp(
    value: Any,
    *,
    stage: str,
    path: str,
) -> str:
    text = require_text(value, stage=stage, path=path)
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        fail(
            stage=stage,
            code="invalid_timestamp",
            path=path,
            expected="an ISO-8601 timestamp",
            provided=text,
        )
    require(
        parsed.tzinfo is not None,
        stage=stage,
        code="naive_timestamp",
        path=path,
        expected="a timezone-aware ISO-8601 timestamp",
        provided=text,
    )
    return text


def reject_placeholders(value: Any, *, path: str = "$") -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            reject_placeholders(item, path=f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            reject_placeholders(item, path=f"{path}[{index}]")
    elif isinstance(value, str):
        require(
            PLACEHOLDER_RE.search(value) is None,
            stage="approval_validation",
            code="placeholder_in_approved_contract",
            path=path,
            expected="no placeholder in an approved or authorized contract",
            provided=value,
        )
