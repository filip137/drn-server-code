#!/usr/bin/env python3
"""Fast, duplicate-safe dispatch of approved long simulations to main or Trex."""

from __future__ import annotations

import argparse
import ctypes
from datetime import datetime, timezone
import fcntl
import getpass
import hashlib
import json
import math
import os
from pathlib import Path
import pwd
import re
import shlex
import signal
import subprocess
import sys
import time
import uuid
from typing import Any, Callable, Mapping, Sequence


REQUEST_SCHEMA = "server-code-local-dispatch-request/v2"
PROFILE_SCHEMA = "server-code-local-dispatch-profile/v2"
DISPATCH_SCHEMA = "server-code-local-dispatch-receipt/v1"
FAILURE_SCHEMA = "server-code-local-dispatch-failure/v1"
COMPLETION_SCHEMA = "server-code-local-dispatch-completion/v1"
RUNNING_SCHEMA = "server-code-local-dispatch-running/v1"
QUIESCENCE_SCHEMA = "server-code-local-dispatch-quiescence/v1"
TRACKER_GATE_SCHEMA = "current-experiments-launch-gate/v1"
PREFLIGHT_SCHEMA = "server-code-long-run-preflight/v2"
SUPPORTED_CHECK_SCHEMAS = {
    (
        "smoke",
        "mnist-conv-perfectdiode-preflight-receipt/v1",
    ): "perfectdiode-hparam-smoke/v1",
    (
        "tk_reference",
        "mnist-conv-perfectdiode-tk-selection/v1",
    ): "perfectdiode-tk-selection/v1",
    (
        "scheduled_run_preflight",
        "scheduled-run-preflight-receipt/v1",
    ): "scheduled-run-preflight/v1",
}
ATTEMPT_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{2,79}$")
EXPERIMENT_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{2,127}$")
SUBJECT_RE = re.compile(r"^[a-z0-9][a-z0-9._-]*$")
SOURCE_RE = re.compile(r"^[0-9a-f]{40}$")
ENV_RE = re.compile(r"^[A-Z_][A-Z0-9_]*$")
SAFE_REQUEST_ENV = {
    "CUDA_VISIBLE_DEVICES",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "KMP_DISABLE_SHM",
    "KMP_SHM_DISABLE",
    "PYTHONUNBUFFERED",
}
PINNED_BASE_ENV = {
    "HOME",
    "USER",
    "LOGNAME",
    "PATH",
    "LANG",
    "LC_ALL",
    "TZ",
    "TMPDIR",
    "PYTHONNOUSERSITE",
}
SSH_TARGET_RE = re.compile(
    r"^[A-Za-z0-9_][A-Za-z0-9_.-]*@[A-Za-z0-9][A-Za-z0-9.-]*$"
)
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
PLAN_JSON_BLOCK_RE = re.compile(r"```json[ \t]*\n(.*?)\n```", re.DOTALL)
MAX_REQUEST_BYTES = 256 * 1024
LOCK_TIMEOUT_SECONDS = 5.0
MAX_TRACKER_GATE_AGE_SECONDS = 10 * 60
MAX_SCHEDULED_PREFLIGHT_AGE_SECONDS = 10 * 60
MAX_SCHEDULED_PREFLIGHT_TIMEOUT_SECONDS = 15 * 60
ENV_EXECUTABLE = "/usr/bin/env"
PERFECTDIODE_SMOKE_BINDINGS = {
    "main": {"host": "akib", "architecture": "conv1"},
    "trex": {"host": "trex", "architecture": "conv2"},
}
Runner = Callable[..., subprocess.CompletedProcess[str]]


class DispatchError(RuntimeError):
    """Structured fail-stop error for one dispatch attempt."""

    def __init__(
        self,
        stage: str,
        message: str,
        *,
        launched_job_count: int | None = 0,
        launch_id: str | None = None,
        attempt_id: str | None = None,
        experiment_id: str | None = None,
        target: str | None = None,
        completed_stages: Sequence[str] = (),
        last_valid_artifacts: Mapping[str, Any] | None = None,
        smallest_next_action: str = (
            "Report this terminal attempt and wait for explicit authorization "
            "before diagnosing or creating a new attempt."
        ),
        terminal: bool = True,
    ) -> None:
        super().__init__(message)
        self.stage = stage
        self.launched_job_count = launched_job_count
        self.launch_id = launch_id
        self.attempt_id = attempt_id
        self.experiment_id = experiment_id
        self.target = target
        self.completed_stages = list(completed_stages)
        self.last_valid_artifacts = dict(last_valid_artifacts or {})
        self.smallest_next_action = smallest_next_action
        self.terminal = terminal

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": FAILURE_SCHEMA,
            "status": "failed",
            "terminal": self.terminal,
            "stage": self.stage,
            "error": str(self),
            "launched_job_count": self.launched_job_count,
            "launch_id": self.launch_id,
            "retry_authorized": not self.terminal,
            "attempt_id": self.attempt_id,
            "experiment_id": self.experiment_id,
            "target": self.target,
            "completed_stages": self.completed_stages,
            "last_valid_artifacts": self.last_valid_artifacts,
            "smallest_next_action": self.smallest_next_action,
            "failed_at_utc": datetime.now(timezone.utc).isoformat(),
        }

    def bind_request(
        self,
        request: Mapping[str, Any],
        *,
        completed_stages: Sequence[str] = (),
        last_valid_artifacts: Mapping[str, Any] | None = None,
    ) -> "DispatchError":
        self.attempt_id = str(request["attempt_id"])
        self.experiment_id = str(request["experiment_id"])
        self.target = str(request["target"])
        if completed_stages and not self.completed_stages:
            self.completed_stages = list(completed_stages)
        if last_valid_artifacts:
            self.last_valid_artifacts.update(last_valid_artifacts)
        return self

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "DispatchError":
        return cls(
            str(value.get("stage", "remote_dispatch")),
            str(value.get("error", "Remote dispatch failed.")),
            launched_job_count=value.get("launched_job_count"),
            launch_id=value.get("launch_id"),
            attempt_id=value.get("attempt_id"),
            experiment_id=value.get("experiment_id"),
            target=value.get("target"),
            completed_stages=value.get("completed_stages", ()),
            last_valid_artifacts=value.get("last_valid_artifacts", {}),
            smallest_next_action=str(
                value.get(
                    "smallest_next_action",
                    "Report this terminal attempt and wait for explicit "
                    "authorization before diagnosing or creating a new attempt.",
                )
            ),
            terminal=value.get("terminal") is not False,
        )


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _strict_json_bytes(payload: bytes, *, label: str) -> Any:
    if len(payload) > MAX_REQUEST_BYTES:
        raise ValueError(
            f"Expected {label} to be at most {MAX_REQUEST_BYTES} bytes. "
            f"Provided value: {len(payload)} bytes."
        )

    def reject_constant(value: str) -> None:
        raise ValueError(
            f"Expected {label} to use strict JSON. Provided value: {value!r}."
        )

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                raise ValueError(
                    f"Expected {label} keys to be unique. "
                    f"Provided duplicate key: {key!r}."
                )
            value[key] = item
        return value

    return json.loads(
        payload.decode("utf-8"),
        parse_constant=reject_constant,
        object_pairs_hook=unique_object,
    )


def _atomic_create(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise FileExistsError(
            f"Expected immutable receipt path not to exist. Provided value: {path}."
        )
    temporary = path.with_name(
        f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    )
    try:
        with temporary.open("xb") as handle:
            handle.write(_canonical_bytes(value) + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.chmod(0o444)
        os.link(temporary, path)
        directory_fd = os.open(
            path.parent,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
        )
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(
            f"Expected {label} to be an existing non-symlink file. "
            f"Provided value: {path}."
        )
    value = _strict_json_bytes(path.read_bytes(), label=label)
    if not isinstance(value, dict):
        raise ValueError(
            f"Expected {label} to be a JSON object. Provided value: {value!r}."
        )
    return value


def _expand_template(value: str, *, user: str) -> str:
    if set(re.findall(r"{([^{}]+)}", value)) - {"user"}:
        raise ValueError(
            "Expected profile path templates to use only '{user}'. "
            f"Provided value: {value!r}."
        )
    return value.format(user=user)


def _login_user() -> str:
    try:
        value = pwd.getpwuid(os.getuid()).pw_name
    except (KeyError, OSError):
        value = getpass.getuser()
    if re.fullmatch(r"[A-Za-z0-9_-]+", value or "") is None:
        raise ValueError(
            f"Expected the operating-system login to be safe. Provided value: {value!r}."
        )
    return value


def _lexical_absolute(value: str | Path) -> Path:
    return Path(os.path.abspath(os.path.expanduser(str(value))))


def _require_profile_path(value: Any, *, label: str) -> str:
    if not isinstance(value, str):
        raise ValueError(
            f"Expected profile {label} to be an absolute path template. "
            f"Provided value: {value!r}."
        )
    expanded = _expand_template(value, user=_login_user())
    path = _lexical_absolute(expanded)
    if not Path(expanded).is_absolute() or "\0" in expanded:
        raise ValueError(
            f"Expected profile {label} to be an absolute safe path. "
            f"Provided value: {value!r}."
        )
    return str(path)


def load_profile(path: str | Path) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    raw = _read_json(source, label="dispatch profile")
    expected_keys = {
        "schema_version",
        "target",
        "transport",
        "python_path_template",
        "helper_path_template",
        "profile_path_template",
        "repo_root_template",
        "state_root_template",
        "allowed_cwd_roots",
        "allowed_executable_roots",
        "base_env",
        "tmux",
        "gpu",
        "ssh",
        "command_timeout_seconds",
    }
    if set(raw) != expected_keys or raw.get("schema_version") != PROFILE_SCHEMA:
        raise ValueError(
            f"Expected dispatch profile keys to be exactly "
            f"{sorted(expected_keys)!r} with schema {PROFILE_SCHEMA!r}. "
            f"Provided value: {raw!r}."
        )
    target = raw["target"]
    transport = raw["transport"]
    if target not in {"main", "trex"} or transport not in {"local", "ssh"}:
        raise ValueError(
            "Expected target main/trex and transport local/ssh. "
            f"Provided value: target={target!r}, transport={transport!r}."
        )
    if (target, transport) not in {("main", "local"), ("trex", "ssh")}:
        raise ValueError(
            "Expected main to use local transport and trex to use ssh. "
            f"Provided value: {(target, transport)!r}."
        )
    user = _login_user()
    expanded = dict(raw)
    for key in (
        "python_path_template",
        "helper_path_template",
        "profile_path_template",
        "repo_root_template",
        "state_root_template",
    ):
        expanded[key.removesuffix("_template")] = _require_profile_path(
            raw[key],
            label=key,
        )
    for key in ("allowed_cwd_roots", "allowed_executable_roots"):
        values = raw[key]
        if not isinstance(values, list) or not values:
            raise ValueError(
                f"Expected profile {key} to be a non-empty list. "
                f"Provided value: {values!r}."
            )
        expanded[key] = [
            _require_profile_path(value, label=f"{key} entry") for value in values
        ]
    base_env = raw["base_env"]
    if not isinstance(base_env, dict) or set(base_env) != PINNED_BASE_ENV:
        raise ValueError(
            f"Expected profile base_env keys to be exactly "
            f"{sorted(PINNED_BASE_ENV)!r}. Provided value: {base_env!r}."
        )
    expanded_base_env: dict[str, str] = {}
    for key, item in base_env.items():
        if not isinstance(item, str) or not item or "\0" in item:
            raise ValueError(
                f"Expected profile base_env.{key} to be a non-empty safe "
                f"string. Provided value: {item!r}."
            )
        expanded_base_env[key] = _expand_template(item, user=user)
    expected_home = f"/home/{user}"
    if (
        expanded_base_env["USER"] != user
        or expanded_base_env["LOGNAME"] != user
        or expanded_base_env["HOME"] != expected_home
        or expanded_base_env["PYTHONNOUSERSITE"] != "1"
    ):
        raise ValueError(
            f"Expected pinned login identity USER=LOGNAME={user!r}, "
            f"HOME={expected_home!r}, and PYTHONNOUSERSITE='1'. "
            f"Provided value: {expanded_base_env!r}."
        )
    expanded["base_env"] = expanded_base_env
    timeout = raw["command_timeout_seconds"]
    if (
        isinstance(timeout, bool)
        or not isinstance(timeout, (int, float))
        or not 1 <= float(timeout) <= 300
    ):
        raise ValueError(
            "Expected profile command_timeout_seconds in [1, 300]. "
            f"Provided value: {timeout!r}."
        )
    tmux = raw["tmux"]
    if not isinstance(tmux, dict) or set(tmux) != {
        "mode",
        "session",
        "session_prefix",
    }:
        raise ValueError(
            "Expected exact tmux mode/session/session_prefix profile. "
            f"Provided value: {tmux!r}."
        )
    expected_mode = "existing_session" if target == "main" else "attempt_session"
    if tmux["mode"] != expected_mode:
        raise ValueError(
            f"Expected {target} tmux mode {expected_mode!r}. "
            f"Provided value: {tmux['mode']!r}."
        )
    for key in ("session", "session_prefix"):
        if (
            not isinstance(tmux[key], str)
            or re.fullmatch(r"[A-Za-z0-9_-]*", tmux[key]) is None
        ):
            raise ValueError(
                f"Expected safe tmux {key}. Provided value: {tmux[key]!r}."
            )
    if target == "main" and not tmux["session"]:
        raise ValueError(
            f"Expected main tmux session to be non-empty. Provided value: {tmux!r}."
        )
    if target == "trex" and not tmux["session_prefix"]:
        raise ValueError(
            f"Expected Trex tmux session_prefix to be non-empty. "
            f"Provided value: {tmux!r}."
        )
    gpu = raw["gpu"]
    if (
        not isinstance(gpu, dict)
        or set(gpu) != {"require_no_compute_processes", "device_index"}
        or gpu["require_no_compute_processes"] is not True
        or isinstance(gpu["device_index"], bool)
        or not isinstance(gpu["device_index"], int)
        or gpu["device_index"] < 0
    ):
        raise ValueError(
            "Expected GPU policy with require_no_compute_processes=true and "
            "a non-negative device_index. "
            f"Provided value: {gpu!r}."
        )
    ssh = raw["ssh"]
    if transport == "ssh":
        if not isinstance(ssh, dict) or set(ssh) != {
            "target",
            "connect_timeout_seconds",
        }:
            raise ValueError(
                f"Expected exact SSH profile for Trex. Provided value: {ssh!r}."
            )
        expanded_target = _expand_template(str(ssh["target"]), user=user)
        if SSH_TARGET_RE.fullmatch(expanded_target) is None:
            raise ValueError(
                f"Expected a safe user@host SSH target. "
                f"Provided value: {expanded_target!r}."
            )
        connect_timeout = ssh["connect_timeout_seconds"]
        if (
            isinstance(connect_timeout, bool)
            or not isinstance(connect_timeout, int)
            or not 1 <= connect_timeout <= 60
        ):
            raise ValueError(
                "Expected SSH connect_timeout_seconds in [1, 60]. "
                f"Provided value: {connect_timeout!r}."
            )
        expanded["ssh"] = {
            "target": expanded_target,
            "connect_timeout_seconds": connect_timeout,
        }
    elif ssh is not None:
        raise ValueError(
            f"Expected local profile ssh to be null. Provided value: {ssh!r}."
        )
    expanded["_source_path"] = str(source)
    expanded["_sha256"] = _sha256_bytes(source.read_bytes())
    return expanded


def _under_roots(path: Path, roots: Sequence[str], *, resolve: bool) -> bool:
    candidate = path.resolve() if resolve else _lexical_absolute(path)
    for raw_root in roots:
        root_path = Path(raw_root)
        root = root_path.resolve() if resolve else _lexical_absolute(root_path)
        try:
            candidate.relative_to(root)
        except ValueError:
            continue
        return True
    return False


def _require_safe_relative(value: Any, *, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(
            f"Expected {label} to be a non-empty repository-relative path. "
            f"Provided value: {value!r}."
        )
    path = Path(value)
    if path.is_absolute() or ".." in path.parts or "\0" in value:
        raise ValueError(
            f"Expected {label} to be a safe repository-relative path. "
            f"Provided value: {value!r}."
        )
    return path


def _load_plan_bindings(
    plan_path: Path,
    *,
    cwd: Path,
    experiment_id: str,
) -> dict[str, Any]:
    if plan_path.is_symlink() or not plan_path.is_file():
        raise ValueError(
            "Expected --approved-plan to be an existing non-symlink file. "
            f"Provided value: {plan_path}."
        )
    payload = plan_path.read_bytes()
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(
            f"Expected approved plan to be UTF-8. Provided value: {plan_path}."
        ) from exc
    blocks = PLAN_JSON_BLOCK_RE.findall(text)
    if len(blocks) != 1:
        raise ValueError(
            "Expected approved plan to contain exactly one fenced JSON contract. "
            f"Provided value: {len(blocks)} blocks in {plan_path}."
        )
    plan = _strict_json_bytes(
        blocks[0].encode("utf-8"),
        label="approved plan JSON contract",
    )
    if not isinstance(plan, dict):
        raise ValueError(
            f"Expected approved plan contract to be an object. "
            f"Provided value: {type(plan).__name__}."
        )
    if plan.get("schema_version") != "experiment-run-plan/v1":
        raise ValueError(
            "Expected approved plan schema 'experiment-run-plan/v1'. "
            f"Provided value: {plan.get('schema_version')!r}."
        )
    if plan.get("experiment_id") != experiment_id:
        raise ValueError(
            f"Expected approved plan experiment_id {experiment_id!r}. "
            f"Provided value: {plan.get('experiment_id')!r}."
        )
    approval = plan.get("approval")
    if not isinstance(approval, dict) or approval.get("status") != "approved":
        raise ValueError(
            "Expected approved plan approval.status='approved'. "
            f"Provided value: {approval!r}."
        )
    execution = plan.get("execution")
    if not isinstance(execution, dict):
        raise ValueError(
            f"Expected approved plan execution object. Provided value: {execution!r}."
        )
    launcher = execution.get("launcher")
    if (
        not isinstance(launcher, list)
        or not launcher
        or not all(isinstance(item, str) and item for item in launcher)
    ):
        raise ValueError(
            f"Expected approved plan execution.launcher string list. "
            f"Provided value: {launcher!r}."
        )
    preflight = execution.get("preflight")
    if not isinstance(preflight, dict):
        raise ValueError(
            f"Expected approved plan execution.preflight object. "
            f"Provided value: {preflight!r}."
        )
    if preflight.get("smoke_required") is not True:
        raise ValueError(
            "Expected approved plan to require a payload smoke. "
            f"Provided value: {preflight.get('smoke_required')!r}."
        )
    for key in (
        "tk_reference_required",
        "scheduled_run_preflight_required",
    ):
        if not isinstance(preflight.get(key), bool):
            raise ValueError(
                f"Expected approved plan execution.preflight.{key} to be "
                f"boolean. Provided value: {preflight.get(key)!r}."
            )
    receipt_schemas = preflight.get("receipt_schemas")
    receipt_bindings = preflight.get("receipt_bindings")
    check_roles = {
        "smoke",
        "tk_reference",
        "scheduled_run_preflight",
    }
    if not isinstance(receipt_schemas, dict) or set(receipt_schemas) != check_roles:
        raise ValueError(
            f"Expected approved plan execution.preflight.receipt_schemas "
            f"keys {sorted(check_roles)!r}. Provided value: "
            f"{receipt_schemas!r}. Amend and reapprove legacy plans rather "
            "than inferring preflight semantics."
        )
    if not isinstance(receipt_bindings, dict) or set(receipt_bindings) != check_roles:
        raise ValueError(
            f"Expected approved plan execution.preflight.receipt_bindings "
            f"keys {sorted(check_roles)!r}. Provided value: "
            f"{receipt_bindings!r}. Amend and reapprove legacy plans rather "
            "than inferring receipt paths or subjects."
        )
    required_by_role = {
        "smoke": True,
        "tk_reference": preflight["tk_reference_required"],
        "scheduled_run_preflight": preflight[
            "scheduled_run_preflight_required"
        ],
    }
    normalized_schemas: dict[str, str | None] = {}
    normalized_bindings: dict[str, dict[str, Any] | None] = {}
    relative_binding_paths: list[Path] = []
    for role, required in required_by_role.items():
        schema = receipt_schemas.get(role)
        binding = receipt_bindings.get(role)
        if required:
            if (
                not isinstance(schema, str)
                or not schema
                or (role, schema) not in SUPPORTED_CHECK_SCHEMAS
            ):
                supported = sorted(
                    candidate
                    for candidate_role, candidate in SUPPORTED_CHECK_SCHEMAS
                    if candidate_role == role
                )
                raise ValueError(
                    f"Expected required preflight role {role!r} to name one "
                    f"supported schema {supported!r}. Provided value: {schema!r}."
                )
            normalized_schemas[role] = schema
            if not isinstance(binding, dict) or set(binding) != {
                "receipt_path",
                "subject_id",
                "producer_source_id",
            }:
                raise ValueError(
                    f"Expected required preflight role {role!r} to have exact "
                    "receipt_path/subject_id/producer_source_id binding. "
                    f"Provided value: {binding!r}."
                )
            binding_relative = _require_safe_relative(
                binding.get("receipt_path"),
                label=(
                    "approved plan execution.preflight.receipt_bindings."
                    f"{role}.receipt_path"
                ),
            )
            if (
                len(binding_relative.parts) < 2
                or binding_relative.parts[0] != "results"
            ):
                raise ValueError(
                    f"Expected required preflight role {role!r} receipt_path "
                    "below results/. "
                    f"Provided value: {binding_relative}."
                )
            subject_id = binding.get("subject_id")
            producer_source_id = binding.get("producer_source_id")
            if role == "scheduled_run_preflight":
                if subject_id is not None:
                    raise ValueError(
                        "Expected scheduled_run_preflight subject_id to be "
                        f"null. Provided value: {subject_id!r}."
                    )
                if producer_source_id is not None:
                    raise ValueError(
                        "Expected scheduled_run_preflight producer_source_id "
                        f"to be null. Provided value: {producer_source_id!r}."
                    )
            elif (
                not isinstance(subject_id, str)
                or SUBJECT_RE.fullmatch(subject_id) is None
            ):
                raise ValueError(
                    f"Expected required preflight role {role!r} subject_id "
                    "to be a nonempty safe identifier. "
                    f"Provided value: {subject_id!r}."
                )
            elif (
                not isinstance(producer_source_id, str)
                or SOURCE_RE.fullmatch(producer_source_id) is None
            ):
                raise ValueError(
                    f"Expected required preflight role {role!r} "
                    "producer_source_id to be an exact lowercase "
                    "40-character Git commit. "
                    f"Provided value: {producer_source_id!r}."
                )
            normalized_bindings[role] = {
                "receipt_relative_path": str(binding_relative),
                "subject_id": subject_id,
                "producer_source_id": producer_source_id,
            }
            relative_binding_paths.append(binding_relative)
        else:
            if schema is not None:
                raise ValueError(
                    f"Expected nonrequired preflight role {role!r} schema "
                    f"to be null. Provided value: {schema!r}."
                )
            if binding is not None:
                raise ValueError(
                    f"Expected nonrequired preflight role {role!r} binding "
                    f"to be null. Provided value: {binding!r}."
                )
            normalized_schemas[role] = None
            normalized_bindings[role] = None
    preflight_relative = _require_safe_relative(
        preflight.get("receipt_path"),
        label="approved plan execution.preflight.receipt_path",
    )
    storage = plan.get("storage")
    if not isinstance(storage, dict):
        raise ValueError(
            f"Expected approved plan storage object. Provided value: {storage!r}."
        )
    output_relative = _require_safe_relative(
        storage.get("local_bundle_path"),
        label="approved plan storage.local_bundle_path",
    )
    if len(output_relative.parts) < 2 or output_relative.parts[0] != "results":
        raise ValueError(
            "Expected approved plan storage.local_bundle_path below results/. "
            f"Provided value: {output_relative}."
        )
    try:
        preflight_relative.relative_to(output_relative)
    except ValueError as exc:
        raise ValueError(
            "Expected approved plan outer preflight receipt below "
            f"storage.local_bundle_path {output_relative}. "
            f"Provided value: {preflight_relative}."
        ) from exc
    if preflight_relative in relative_binding_paths:
        raise ValueError(
            "Expected approved plan outer preflight receipt path to differ "
            f"from every inner receipt path. Provided value: {preflight_relative}."
        )
    if len(relative_binding_paths) != len(set(relative_binding_paths)):
        raise ValueError(
            "Expected distinct approved inner preflight receipt paths. "
            f"Provided value: {[str(path) for path in relative_binding_paths]!r}."
        )
    for binding in normalized_bindings.values():
        if binding is None:
            continue
        relative = Path(str(binding["receipt_relative_path"]))
        binding["receipt_path"] = str(_lexical_absolute(cwd / relative))
    return {
        "approved_plan_path": str(plan_path),
        "approved_plan_sha256": _sha256_bytes(payload),
        "preflight_receipt_path": str(_lexical_absolute(cwd / preflight_relative)),
        "output_root": str(_lexical_absolute(cwd / output_relative)),
        "launcher": list(launcher),
        "required_checks": {
            "smoke": True,
            "tk_reference": preflight["tk_reference_required"],
            "scheduled_run_preflight": preflight[
                "scheduled_run_preflight_required"
            ],
        },
        "receipt_schemas": normalized_schemas,
        "receipt_bindings": normalized_bindings,
    }


def _command_matches_launcher(
    argv: Sequence[str],
    expected_launcher: Sequence[str],
    *,
    profile: Mapping[str, Any],
) -> bool:
    if len(argv) != len(expected_launcher):
        return False
    actual = list(argv)
    expected = list(expected_launcher)
    approved_executable = expected[0]
    if approved_executable == "python":
        executable_matches = actual[0] == str(profile["python_path"])
    elif Path(approved_executable).is_absolute():
        executable_matches = actual[0] == approved_executable
    else:
        executable_matches = actual[0] == approved_executable
    if not executable_matches:
        return False
    return actual[1:] == expected[1:]


def _runtime_environment(
    profile: Mapping[str, Any],
    request_env: Mapping[str, str],
) -> dict[str, str]:
    environment = dict(profile["base_env"])
    environment.update(request_env)
    return dict(sorted(environment.items()))


def _environment_binding(
    profile: Mapping[str, Any],
    request_env: Mapping[str, str],
) -> dict[str, Any]:
    effective = _runtime_environment(profile, request_env)
    return {
        "overrides": dict(sorted(request_env.items())),
        "effective": effective,
        "sha256": _sha256_bytes(_canonical_bytes(effective)),
    }


def _isolated_command(
    environment: Mapping[str, str],
    command: Sequence[str],
) -> list[str]:
    return [
        ENV_EXECUTABLE,
        "-i",
        *(f"{key}={value}" for key, value in sorted(environment.items())),
        *command,
    ]


def _require_canonical_below(
    path: Path,
    *,
    root: Path,
    label: str,
    require_file: bool,
) -> Path:
    lexical = _lexical_absolute(path)
    _ensure_no_symlink_components(lexical)
    if require_file and not lexical.is_file():
        raise DispatchError(
            "launch_authorization",
            f"Expected {label} to be an existing file. Provided value: {lexical}.",
        )
    resolved = lexical.resolve(strict=require_file)
    try:
        resolved.relative_to(root.resolve(strict=True))
    except ValueError as exc:
        raise DispatchError(
            "launch_authorization",
            f"Expected {label} below canonical repo root {root}. "
            f"Provided value: {resolved}.",
        ) from exc
    if resolved != lexical:
        raise DispatchError(
            "launch_authorization",
            f"Expected {label} without symlink traversal. "
            f"Provided value: {lexical} -> {resolved}.",
        )
    return lexical


def _validate_artifact_records(
    records: Any,
    *,
    root: Path,
    label: str,
) -> list[str]:
    if not isinstance(records, list) or not records:
        raise ValueError(
            f"Expected {label} to be a non-empty artifact list. "
            f"Provided value: {records!r}."
        )
    validated: list[str] = []
    for index, record in enumerate(records):
        if not isinstance(record, dict) or set(record) != {
            "path",
            "sha256",
            "bytes",
        }:
            raise ValueError(
                f"Expected {label}[{index}] to contain path/sha256/bytes. "
                f"Provided value: {record!r}."
            )
        relative = _require_safe_relative(
            record["path"],
            label=f"{label}[{index}].path",
        )
        path = _require_canonical_below(
            root / relative,
            root=root,
            label=f"{label}[{index}]",
            require_file=True,
        )
        payload = path.read_bytes()
        if (
            not isinstance(record["sha256"], str)
            or _sha256_bytes(payload) != record["sha256"]
            or isinstance(record["bytes"], bool)
            or record["bytes"] != len(payload)
        ):
            raise ValueError(
                f"Expected {label}[{index}] hash and size to match. "
                f"Provided value: {record!r}."
            )
        validated.append(str(path))
    return validated


def _validate_perfectdiode_hparam_smoke(
    receipt_path: Path,
    value: Mapping[str, Any],
    *,
    cwd: Path,
    experiment_id: str,
    subject_id: str | None,
    producer_source_id: str | None,
    source_id: str,
    target: str,
    output_root: Path,
) -> dict[str, Any]:
    del experiment_id
    if not isinstance(subject_id, str):
        raise ValueError(
            "Expected a bound perfect-diode smoke subject_id. "
            f"Provided value: {subject_id!r}."
        )
    if (
        not isinstance(producer_source_id, str)
        or producer_source_id != source_id
    ):
        raise ValueError(
            "Expected the perfect-diode smoke producer_source_id to equal "
            f"the downstream launch source_id {source_id!r}. "
            f"Provided value: {producer_source_id!r}."
        )
    target_binding = PERFECTDIODE_SMOKE_BINDINGS.get(target)
    if target_binding is None:
        raise ValueError(
            "Expected a dispatch target with a frozen perfect-diode smoke "
            f"binding {sorted(PERFECTDIODE_SMOKE_BINDINGS)!r}. "
            f"Provided value: {target!r}."
        )
    expected_host = target_binding["host"]
    expected_architecture = target_binding["architecture"]
    expected_keys = {
        "schema_version",
        "study_id",
        "host",
        "architecture",
        "surface_id",
        "entry_id",
        "execution_passed",
        "scientific_center_safety_clean",
        "completed_steps",
        "expected_steps",
        "benchmark",
        "request_sha256",
        "outputs",
        "official_test_read",
    }
    if set(value) != expected_keys:
        raise ValueError(
            f"Expected perfect-diode smoke keys {sorted(expected_keys)!r}. "
            f"Provided value: {sorted(value)!r}."
        )
    if (
        value.get("schema_version")
        != "mnist-conv-perfectdiode-preflight-receipt/v1"
        or value.get("study_id") != subject_id
        or value.get("host") != expected_host
        or value.get("architecture") != expected_architecture
        or value.get("execution_passed") is not True
        or value.get("scientific_center_safety_clean") is not True
        or value.get("official_test_read") is not False
        or isinstance(value.get("expected_steps"), bool)
        or not isinstance(value.get("expected_steps"), int)
        or value["expected_steps"] <= 0
        or value.get("completed_steps") != value["expected_steps"]
    ):
        raise ValueError(
            "Expected a complete, scientifically clean, non-test "
            "perfect-diode smoke bound to the approved subject and target's "
            f"producer host/architecture {target_binding!r}. "
            f"Provided value: {dict(value)!r}."
        )
    try:
        receipt_path.relative_to(output_root)
    except ValueError as exc:
        raise ValueError(
            f"Expected smoke receipt below output root {output_root}. "
            f"Provided value: {receipt_path}."
        ) from exc
    request_path = receipt_path.parent / "request.json"
    request_path = _require_canonical_below(
        request_path,
        root=cwd,
        label="perfect-diode smoke request",
        require_file=True,
    )
    request_payload = request_path.read_bytes()
    if _sha256_bytes(request_payload) != value.get("request_sha256"):
        raise ValueError(
            "Expected perfect-diode smoke request hash to match its receipt. "
            f"Provided value: {value.get('request_sha256')!r}."
        )
    request = _strict_json_bytes(
        request_payload,
        label="perfect-diode smoke request",
    )
    provenance = request.get("code_provenance") if isinstance(request, dict) else None
    if (
        not isinstance(request, dict)
        or request.get("schema_version")
        != "mnist-conv-perfectdiode-preflight-request/v1"
        or request.get("study_id") != subject_id
        or request.get("host") != expected_host
        or request.get("architecture") != expected_architecture
        or request.get("architecture") != value.get("architecture")
        or not isinstance(provenance, dict)
        or provenance.get("git_revision") != producer_source_id
        or provenance.get("dirty_source_digest") is not None
    ):
        raise ValueError(
            "Expected perfect-diode smoke request to bind the clean source, "
            f"subject, architecture, and target. Provided value: {request!r}."
        )
    output_artifacts = _validate_artifact_records(
        value["outputs"],
        root=output_root,
        label="perfect-diode smoke outputs",
    )
    return {
        "study_id": value["study_id"],
        "producer_source_id": producer_source_id,
        "host": value["host"],
        "architecture": value["architecture"],
        "surface_id": value["surface_id"],
        "entry_id": value["entry_id"],
        "completed_steps": value["completed_steps"],
        "request_path": str(request_path),
        "request_sha256": value["request_sha256"],
        "output_artifacts": output_artifacts,
        "official_test_read": False,
    }


def _validate_perfectdiode_tk_selection(
    receipt_path: Path,
    value: Mapping[str, Any],
    *,
    cwd: Path,
    experiment_id: str,
    subject_id: str | None,
    producer_source_id: str | None,
    source_id: str,
    target: str,
    output_root: Path,
) -> dict[str, Any]:
    del experiment_id, source_id, target
    if not isinstance(subject_id, str):
        raise ValueError(
            "Expected a bound perfect-diode T/K subject_id. "
            f"Provided value: {subject_id!r}."
        )
    if not isinstance(producer_source_id, str):
        raise ValueError(
            "Expected a bound perfect-diode T/K producer_source_id. "
            f"Provided value: {producer_source_id!r}."
        )
    try:
        receipt_path.relative_to(output_root)
    except ValueError as exc:
        raise ValueError(
            f"Expected T/K selection below output root {output_root}. "
            f"Provided value: {receipt_path}."
        ) from exc
    manifest_path = _require_canonical_below(
        receipt_path.parent / "manifest.json",
        root=cwd,
        label="perfect-diode T/K manifest",
        require_file=True,
    )
    cwd_text = str(cwd)
    if cwd_text not in sys.path:
        sys.path.insert(0, cwd_text)
    from experiments.run_mnist_conv_perfectdiode_tk import (
        _load_manifest,
        _validate_entry_completion,
    )

    loaded_path, loaded_manifest, _spec = _load_manifest(
        manifest_path,
        enforce_code_identity=False,
    )
    manifest_sha256 = _sha256_bytes(loaded_path.read_bytes())
    expected_selection_path = loaded_path.parent / str(
        loaded_manifest["selection_path"]
    )
    if expected_selection_path.resolve() != receipt_path:
        raise ValueError(
            "Expected T/K receipt to be the manifest-declared selection. "
            f"Provided value: expected={expected_selection_path}, "
            f"observed={receipt_path}."
        )
    entries = loaded_manifest.get("entries")
    if not isinstance(entries, list) or len(entries) != 3:
        raise ValueError(
            "Expected the frozen T/K manifest to contain exactly three rows. "
            f"Provided value: {entries!r}."
        )
    rows: list[dict[str, Any]] = []
    row_hashes: dict[str, dict[str, str]] = {}
    for entry in entries:
        _completion, result = _validate_entry_completion(
            loaded_path.parent,
            loaded_manifest,
            entry,
            manifest_sha256=manifest_sha256,
        )
        result_path = loaded_path.parent / str(entry["outputs"][-1])
        completion_path = loaded_path.parent / str(entry["completion_path"])
        result_sha256 = _sha256_bytes(result_path.read_bytes())
        completion_sha256 = _sha256_bytes(completion_path.read_bytes())
        rows.append(
            {
                "row_id": result["row_id"],
                "architecture": "conv3",
                "scheme": result["scheme"],
                "execution_backend": result["execution_backend"],
                "execution_host": result["execution_host"],
                "execution_environment_path": str(entry["outputs"][0]),
                "execution_environment_sha256": result[
                    "execution_environment_sha256"
                ],
                "run_name": result["run_name"],
                "voltage_amp": result["voltage_amp"],
                "current_amp": result["current_amp"],
                "input_gain": result["input_gain"],
                "status": result["status"],
                "selected_t": result["selected_t"],
                "selected_k": result["selected_k"],
                "t_reference": result["t_reference"],
                "k_reference": result["k_reference"],
                "t_extension_used": result["t_extension_used"],
                "k128_sentinel_used": result["k128_sentinel_used"],
                "result_path": str(entry["outputs"][-1]),
                "result_sha256": result_sha256,
                "completion_path": str(entry["completion_path"]),
                "completion_sha256": completion_sha256,
                "operating_point_audit_path": str(entry["outputs"][-2]),
                "operating_point_audit_sha256": result[
                    "operating_point_audit_sha256"
                ],
                "operating_point_audit_passed": result[
                    "operating_point_audit_passed"
                ],
                "t_cohort_indices_sha256": result[
                    "t_cohort_indices_sha256"
                ],
                "k_cohort_indices_sha256": result[
                    "k_cohort_indices_sha256"
                ],
            }
        )
        row_hashes[result["row_id"]] = {
            "result_sha256": result_sha256,
            "completion_sha256": completion_sha256,
            "operating_point_audit_sha256": result[
                "operating_point_audit_sha256"
            ],
            "execution_environment_sha256": result[
                "execution_environment_sha256"
            ],
        }
    rederived = {
        "schema_version": "mnist-conv-perfectdiode-tk-selection/v1",
        "study_id": loaded_manifest["study_id"],
        "config_sha256": loaded_manifest["config_sha256"],
        "manifest_sha256": manifest_sha256,
        "official_test_read": False,
        "status": (
            "selected"
            if all(row["status"] == "selected" for row in rows)
            else "unresolved"
        ),
        "rows": rows,
        "artifact_sha256s": {
            "asset_bundle": loaded_manifest["asset_binding"],
            "rows": row_hashes,
        },
    }
    manifest = _read_json(
        manifest_path,
        label="perfect-diode T/K manifest",
    )
    staged = manifest.get("staged_source")
    if (
        value.get("schema_version")
        != "mnist-conv-perfectdiode-tk-selection/v1"
        or value.get("study_id") != subject_id
        or value.get("status") != "selected"
        or value.get("official_test_read") is not False
        or rederived != dict(value)
        or not rows
        or any(row.get("status") != "selected" for row in rows)
        or not isinstance(staged, dict)
        or staged.get("commit") != producer_source_id
    ):
        raise ValueError(
            "Expected a fully revalidated selected T/K reference bound to the "
            f"approved producer source and subject. Provided value: {dict(value)!r}."
        )
    return {
        "study_id": value["study_id"],
        "producer_source_id": producer_source_id,
        "manifest_path": str(manifest_path),
        "manifest_sha256": manifest_sha256,
        "row_count": len(rows),
        "selected_rows": [row["row_id"] for row in rows],
        "official_test_read": False,
    }


def _validate_scheduled_run_preflight(
    receipt_path: Path,
    value: Mapping[str, Any],
    *,
    cwd: Path,
    experiment_id: str,
    subject_id: str | None,
    producer_source_id: str | None,
    source_id: str,
    target: str,
    output_root: Path,
) -> dict[str, Any]:
    if subject_id is not None:
        raise ValueError(
            "Expected scheduled-run preflight subject_id to be null. "
            f"Provided value: {subject_id!r}."
        )
    if producer_source_id is not None:
        raise ValueError(
            "Expected scheduled-run preflight producer_source_id to be null. "
            f"Provided value: {producer_source_id!r}."
        )
    expected_keys = {
        "schema_version",
        "status",
        "timestamp_utc",
        "runner",
        "runner_sha256",
        "cwd",
        "preflight_command",
        "timeout_seconds",
        "exit_code",
        "stdout",
        "stderr",
    }
    try:
        receipt_path.relative_to(output_root)
    except ValueError as exc:
        raise ValueError(
            f"Expected scheduled-run preflight receipt below output root "
            f"{output_root}. Provided value: {receipt_path}."
        ) from exc
    runner_value = value.get("runner")
    if not isinstance(runner_value, str) or not Path(runner_value).is_absolute():
        raise ValueError(
            "Expected scheduled-run runner to be an absolute path string. "
            f"Provided value: {runner_value!r}."
        )
    runner_path = Path(runner_value)
    command = value.get("preflight_command")
    timeout_seconds = value.get("timeout_seconds")
    stdout = value.get("stdout")
    stderr = value.get("stderr")
    if (
        set(value) != expected_keys
        or value.get("schema_version") != "scheduled-run-preflight-receipt/v1"
        or value.get("status") != "passed"
        or value.get("exit_code") != 0
        or _lexical_absolute(str(value.get("cwd", ""))) != cwd
        or not isinstance(command, list)
        or not all(isinstance(item, str) and item for item in command)
        or isinstance(timeout_seconds, bool)
        or not isinstance(timeout_seconds, (int, float))
        or not math.isfinite(float(timeout_seconds))
        or not 0 < float(timeout_seconds) <= MAX_SCHEDULED_PREFLIGHT_TIMEOUT_SECONDS
        or not isinstance(stdout, str)
        or not isinstance(stderr, str)
    ):
        raise ValueError(
            "Expected a successful canonical scheduled-run preflight bound "
            f"to this cwd and runner. Provided value: {dict(value)!r}."
        )
    runner_path = _require_canonical_below(
        runner_path,
        root=cwd,
        label="scheduled-run preflight runner",
        require_file=True,
    )
    runner_text = str(runner_path)
    if runner_path.suffix == ".sh":
        expected_command = [
            "/bin/bash",
            "-euo",
            "pipefail",
            runner_text,
            "--preflight",
        ]
        command_matches = command == expected_command
    elif runner_path.suffix == ".py":
        command_matches = (
            len(command) == 3
            and Path(command[0]).is_absolute()
            and Path(command[0]).is_file()
            and os.access(command[0], os.X_OK)
            and Path(command[0]).resolve() == Path(sys.executable).resolve()
            and command[1:] == [runner_text, "--preflight"]
        )
    else:
        command_matches = (
            os.access(runner_path, os.X_OK)
            and command == [runner_text, "--preflight"]
        )
    if not command_matches:
        raise ValueError(
            "Expected scheduled-run preflight_command to exactly match the "
            f"official producer command for {runner_path.suffix or 'executable'!r}. "
            f"Provided value: {command!r}."
        )
    if _sha256_bytes(runner_path.read_bytes()) != value.get("runner_sha256"):
        raise ValueError(
            "Expected scheduled-run runner hash to match the live file. "
            f"Provided value: {value.get('runner_sha256')!r}."
        )
    timestamp = value.get("timestamp_utc")
    if not isinstance(timestamp, str):
        raise ValueError(
            "Expected scheduled-run timestamp_utc to be timezone-aware "
            f"ISO-8601. Provided value: {timestamp!r}."
        )
    try:
        generated_at = datetime.fromisoformat(timestamp)
    except ValueError as exc:
        raise ValueError(
            "Expected scheduled-run timestamp_utc to be timezone-aware "
            f"ISO-8601. Provided value: {timestamp!r}."
        ) from exc
    if generated_at.tzinfo is None:
        raise ValueError(
            "Expected scheduled-run timestamp_utc to be timezone-aware "
            f"ISO-8601. Provided value: {timestamp!r}."
        )
    age = (
        datetime.now(timezone.utc) - generated_at.astimezone(timezone.utc)
    ).total_seconds()
    if not -60 <= age <= MAX_SCHEDULED_PREFLIGHT_AGE_SECONDS:
        raise ValueError(
            "Expected a fresh scheduled-run preflight no older than "
            f"{MAX_SCHEDULED_PREFLIGHT_AGE_SECONDS} seconds. "
            f"Provided age: {age:.3f} seconds."
        )
    return {
        "experiment_id": experiment_id,
        "source_id": source_id,
        "target": target,
        "output_root": str(output_root),
        "timestamp_utc": timestamp,
        "runner": str(runner_path),
        "runner_sha256": value["runner_sha256"],
        "preflight_command": command,
        "timeout_seconds": timeout_seconds,
        "exit_code": 0,
        "stdout_sha256": _sha256_bytes(stdout.encode("utf-8")),
        "stderr_sha256": _sha256_bytes(stderr.encode("utf-8")),
    }


def _validate_preflight_check(
    *,
    role: str,
    expected_schema: str,
    receipt_path_value: Any,
    receipt_sha256_value: Any,
    profile: Mapping[str, Any],
    cwd: Path,
    experiment_id: str,
    plan_bindings: Mapping[str, Any],
    source_id: str,
) -> tuple[Path, dict[str, Any]]:
    receipt_path, receipt, _payload = _validate_bound_json_file(
        receipt_path_value,
        receipt_sha256_value,
        label=f"{role} gate receipt",
    )
    _require_canonical_below(
        receipt_path,
        root=cwd,
        label=f"{role} gate receipt",
        require_file=True,
    )
    binding = plan_bindings["receipt_bindings"].get(role)
    if not isinstance(binding, dict):
        raise DispatchError(
            "launch_authorization",
            f"Expected required role {role!r} to have an approved receipt "
            f"binding. Provided value: {binding!r}.",
        )
    expected_receipt_path = Path(str(binding["receipt_path"]))
    if receipt_path != expected_receipt_path:
        raise DispatchError(
            "launch_authorization",
            f"Expected {role!r} receipt path to equal the approved-plan "
            f"binding {expected_receipt_path}. Provided value: {receipt_path}.",
        )
    if receipt.get("schema_version") != expected_schema:
        raise DispatchError(
            "launch_authorization",
            f"Expected {role!r} receipt schema {expected_schema!r}. "
            f"Provided value: {receipt.get('schema_version')!r}.",
        )
    validator_id = SUPPORTED_CHECK_SCHEMAS.get((role, expected_schema))
    validators = {
        "perfectdiode-hparam-smoke/v1": _validate_perfectdiode_hparam_smoke,
        "perfectdiode-tk-selection/v1": _validate_perfectdiode_tk_selection,
        "scheduled-run-preflight/v1": _validate_scheduled_run_preflight,
    }
    validator = validators.get(str(validator_id))
    if validator is None:
        raise DispatchError(
            "launch_authorization",
            f"Expected a checked-in validator for {(role, expected_schema)!r}. "
            f"Provided value: {validator_id!r}.",
        )
    try:
        claims = validator(
            receipt_path,
            receipt,
            cwd=cwd,
            experiment_id=experiment_id,
            subject_id=binding["subject_id"],
            producer_source_id=binding["producer_source_id"],
            source_id=source_id,
            target=str(profile["target"]),
            output_root=Path(str(plan_bindings["output_root"])),
        )
    except DispatchError:
        raise
    except Exception as exc:
        raise DispatchError(
            "launch_authorization",
            f"Expected {role!r} schema validator {validator_id!r} to pass. "
            f"Provided error: {type(exc).__name__}: {exc}.",
        ) from exc
    return receipt_path, {
        "validator_id": validator_id,
        "claims": claims,
    }


def _validate_preflight_envelope(
    value: Mapping[str, Any],
    *,
    profile: Mapping[str, Any],
    cwd: Path,
    experiment_id: str,
    plan_bindings: Mapping[str, Any],
    source_id: str,
    argv: Sequence[str],
    env: Mapping[str, str],
) -> dict[str, Any]:
    expected_keys = {
        "schema_version",
        "status",
        "created_at_utc",
        "experiment_id",
        "approved_plan_sha256",
        "profile_sha256",
        "source_id",
        "target",
        "device_index",
        "output_root",
        "argv",
        "env",
        "checks",
    }
    if set(value) != expected_keys:
        raise DispatchError(
            "launch_authorization",
            f"Expected canonical preflight envelope keys "
            f"{sorted(expected_keys)!r}. Provided value: {sorted(value)!r}.",
        )
    expected_values = {
        "schema_version": PREFLIGHT_SCHEMA,
        "status": "passed",
        "experiment_id": experiment_id,
        "approved_plan_sha256": plan_bindings["approved_plan_sha256"],
        "profile_sha256": profile["_sha256"],
        "source_id": source_id,
        "target": profile["target"],
        "device_index": profile["gpu"]["device_index"],
        "output_root": plan_bindings["output_root"],
        "argv": list(argv),
        "env": _environment_binding(profile, env),
    }
    for key, expected in expected_values.items():
        if value.get(key) != expected:
            raise DispatchError(
                "launch_authorization",
                f"Expected preflight envelope {key}={expected!r}. "
                f"Provided value: {value.get(key)!r}.",
            )
    created = value.get("created_at_utc")
    if not isinstance(created, str):
        raise DispatchError(
            "launch_authorization",
            f"Expected preflight envelope created_at_utc. "
            f"Provided value: {created!r}.",
        )
    try:
        created_at = datetime.fromisoformat(created)
    except ValueError as exc:
        raise DispatchError(
            "launch_authorization",
            f"Expected timezone-aware ISO-8601 preflight created_at_utc. "
            f"Provided value: {created!r}.",
        ) from exc
    if created_at.tzinfo is None:
        raise DispatchError(
            "launch_authorization",
            f"Expected timezone-aware preflight created_at_utc. "
            f"Provided value: {created!r}.",
        )
    checks = value.get("checks")
    required_checks = plan_bindings["required_checks"]
    if not isinstance(checks, dict) or set(checks) != set(required_checks):
        raise DispatchError(
            "launch_authorization",
            f"Expected preflight checks {sorted(required_checks)!r}. "
            f"Provided value: {checks!r}.",
        )
    validated: dict[str, Any] = {}
    for name, required in required_checks.items():
        check = checks[name]
        expected_check_keys = {
            "required",
            "status",
            "schema_version",
            "validator_id",
            "receipt_path",
            "receipt_sha256",
            "claims",
        }
        if not isinstance(check, dict) or set(check) != expected_check_keys:
            raise DispatchError(
                "launch_authorization",
                f"Expected exact preflight check contract for {name!r}. "
                f"Provided value: {check!r}.",
            )
        if check.get("required") is not required:
            raise DispatchError(
                "launch_authorization",
                f"Expected preflight check {name!r} required={required!r}. "
                f"Provided value: {check.get('required')!r}.",
            )
        if not required:
            if check != {
                "required": False,
                "status": "not_required",
                "schema_version": None,
                "validator_id": None,
                "receipt_path": None,
                "receipt_sha256": None,
                "claims": None,
            }:
                raise DispatchError(
                    "launch_authorization",
                    f"Expected canonical not-required check for {name!r}. "
                    f"Provided value: {check!r}.",
                )
            validated[name] = {"required": False, "status": "not_required"}
            continue
        if check.get("status") != "passed":
            raise DispatchError(
                "launch_authorization",
                f"Expected required preflight check {name!r} to pass. "
                f"Provided value: {check.get('status')!r}.",
            )
        expected_schema = plan_bindings["receipt_schemas"][name]
        if check.get("schema_version") != expected_schema:
            raise DispatchError(
                "launch_authorization",
                f"Expected preflight check {name!r} schema "
                f"{expected_schema!r}. Provided value: "
                f"{check.get('schema_version')!r}.",
            )
        receipt_path, validated_check = _validate_preflight_check(
            role=name,
            expected_schema=str(expected_schema),
            receipt_path_value=check.get("receipt_path"),
            receipt_sha256_value=check.get("receipt_sha256"),
            profile=profile,
            cwd=cwd,
            experiment_id=experiment_id,
            plan_bindings=plan_bindings,
            source_id=source_id,
        )
        if (
            check.get("validator_id") != validated_check["validator_id"]
            or check.get("claims") != validated_check["claims"]
        ):
            raise DispatchError(
                "launch_authorization",
                f"Expected preflight check {name!r} validator claims to "
                f"revalidate exactly. Provided value: {check!r}.",
            )
        validated[name] = {
            "required": True,
            "status": "passed",
            "schema_version": expected_schema,
            "validator_id": validated_check["validator_id"],
            "receipt_path": str(receipt_path),
            "receipt_sha256": check["receipt_sha256"],
            "claims": validated_check["claims"],
        }
    return validated


def _validate_bound_file_bytes(
    path_value: Any,
    sha_value: Any,
    *,
    label: str,
) -> tuple[Path, bytes]:
    if not isinstance(path_value, str):
        raise DispatchError(
            "launch_authorization",
            f"Expected {label} path to be an absolute string. "
            f"Provided value: {path_value!r}.",
        )
    path = _lexical_absolute(path_value)
    _ensure_no_symlink_components(path)
    if not Path(path_value).is_absolute() or path.is_symlink() or not path.is_file():
        raise DispatchError(
            "launch_authorization",
            f"Expected {label} to be an existing non-symlink file. "
            f"Provided value: {path}.",
        )
    payload = path.read_bytes()
    observed = _sha256_bytes(payload)
    if not isinstance(sha_value, str) or observed != sha_value:
        raise DispatchError(
            "launch_authorization",
            f"Expected {label} SHA-256 {sha_value!r}. "
            f"Provided value: {observed!r}.",
        )
    return path, payload


def _validate_bound_json_file(
    path_value: Any,
    sha_value: Any,
    *,
    label: str,
) -> tuple[Path, dict[str, Any], bytes]:
    path, payload = _validate_bound_file_bytes(
        path_value,
        sha_value,
        label=label,
    )
    value = _strict_json_bytes(payload, label=label)
    if not isinstance(value, dict):
        raise DispatchError(
            "launch_authorization",
            f"Expected {label} to contain a JSON object. "
            f"Provided value: {type(value).__name__}.",
        )
    return path, value, payload


def _normalize_request_env(
    raw: Mapping[str, Any],
    *,
    profile: Mapping[str, Any],
) -> dict[str, str]:
    env = dict(raw)
    if not all(
        isinstance(key, str)
        and ENV_RE.fullmatch(key) is not None
        and isinstance(item, str)
        and "\0" not in item
        for key, item in env.items()
    ):
        raise ValueError(
            f"Expected env to be a safe string mapping. Provided value: {raw!r}."
        )
    unknown_env = set(env) - SAFE_REQUEST_ENV
    if unknown_env:
        raise ValueError(
            f"Expected env names from {sorted(SAFE_REQUEST_ENV)!r}. "
            f"Provided value: {sorted(unknown_env)!r}."
        )
    for key, item in env.items():
        if key in {
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
        }:
            if not item.isdigit() or not 1 <= int(item) <= 256:
                raise ValueError(
                    f"Expected {key} to be an integer string from 1 through "
                    f"256. Provided value: {item!r}."
                )
        elif key != "CUDA_VISIBLE_DEVICES" and item != "1":
            raise ValueError(
                f"Expected safe runtime flag {key}=1. "
                f"Provided value: {item!r}."
            )
    expected_device = str(profile["gpu"]["device_index"])
    if (
        "CUDA_VISIBLE_DEVICES" in env
        and env["CUDA_VISIBLE_DEVICES"] != expected_device
    ):
        raise ValueError(
            f"Expected CUDA_VISIBLE_DEVICES={expected_device!r} for this profile. "
            f"Provided value: {env['CUDA_VISIBLE_DEVICES']!r}."
        )
    env.setdefault("CUDA_VISIBLE_DEVICES", expected_device)
    return dict(sorted(env.items()))


def validate_request(
    raw: Mapping[str, Any],
    *,
    profile: Mapping[str, Any],
) -> dict[str, Any]:
    expected_keys = {
        "schema_version",
        "attempt_id",
        "experiment_id",
        "target",
        "cwd",
        "argv",
        "env",
        "source_id",
        "expected_duration_seconds",
        "hard_deadline_seconds",
        "profile_sha256",
        "approved_plan_path",
        "approved_plan_sha256",
        "preflight_receipt_path",
        "preflight_receipt_sha256",
        "tracker_gate_receipt_path",
        "tracker_gate_receipt_sha256",
        "output_root",
    }
    value = dict(raw)
    if set(value) != expected_keys or value.get("schema_version") != REQUEST_SCHEMA:
        raise ValueError(
            f"Expected launch-request keys to be exactly "
            f"{sorted(expected_keys)!r} with schema {REQUEST_SCHEMA!r}. "
            f"Provided value: {value!r}."
        )
    if (
        not isinstance(value["attempt_id"], str)
        or ATTEMPT_RE.fullmatch(value["attempt_id"]) is None
    ):
        raise ValueError(
            "Expected attempt_id to match [a-z0-9][a-z0-9._-]{2,79}. "
            f"Provided value: {value['attempt_id']!r}."
        )
    if (
        not isinstance(value["experiment_id"], str)
        or EXPERIMENT_RE.fullmatch(value["experiment_id"]) is None
    ):
        raise ValueError(
            "Expected experiment_id to be a safe lowercase identifier. "
            f"Provided value: {value['experiment_id']!r}."
        )
    if value["target"] != profile["target"]:
        raise ValueError(
            f"Expected request target {profile['target']!r}. "
            f"Provided value: {value['target']!r}."
        )
    if value["profile_sha256"] != profile["_sha256"]:
        raise ValueError(
            "Expected request profile SHA-256 to match the loaded profile. "
            f"Provided value: {value['profile_sha256']!r}."
        )
    if not isinstance(value["cwd"], str):
        raise ValueError(
            f"Expected cwd to be an absolute string. "
            f"Provided value: {value['cwd']!r}."
        )
    if not Path(value["cwd"]).is_absolute():
        raise ValueError(
            f"Expected cwd to be absolute. Provided value: {value['cwd']!r}."
        )
    cwd = _lexical_absolute(value["cwd"])
    if cwd != _lexical_absolute(str(profile["repo_root"])):
        raise ValueError(
            f"Expected cwd to equal profile repo root {profile['repo_root']!r}. "
            f"Provided value: {value['cwd']!r}."
        )
    if not _under_roots(
        cwd, profile["allowed_cwd_roots"], resolve=False
    ):
        raise ValueError(
            "Expected cwd to be absolute and under an approved root. "
            f"Provided value: {value['cwd']!r}."
        )
    argv = value["argv"]
    if (
        not isinstance(argv, list)
        or not argv
        or not all(
            isinstance(item, str) and item and "\0" not in item
            for item in argv
        )
    ):
        raise ValueError(
            f"Expected argv to be a non-empty string list. Provided value: {argv!r}."
        )
    executable = Path(argv[0]).expanduser()
    if not executable.is_absolute() or not _under_roots(
        _lexical_absolute(executable),
        profile["allowed_executable_roots"],
        resolve=False,
    ):
        raise ValueError(
            "Expected argv[0] to be an absolute executable under an approved "
            f"root. Provided value: {argv[0]!r}."
        )
    env = value["env"]
    if not isinstance(env, dict):
        raise ValueError(
            f"Expected env to be a safe string mapping. Provided value: {env!r}."
        )
    normalized_env = _normalize_request_env(env, profile=profile)
    if (
        not isinstance(value["source_id"], str)
        or SOURCE_RE.fullmatch(value["source_id"]) is None
    ):
        raise ValueError(
            "Expected source_id to be an exact lowercase 40-character Git "
            f"commit. Provided value: {value['source_id']!r}."
        )
    expected = value["expected_duration_seconds"]
    deadline = value["hard_deadline_seconds"]
    if (
        isinstance(expected, bool)
        or not isinstance(expected, (int, float))
        or not math.isfinite(float(expected))
        or float(expected) < 600
    ):
        raise ValueError(
            "Expected long-run duration to be at least 600 seconds. "
            f"Provided value: {expected!r}."
        )
    if (
        isinstance(deadline, bool)
        or not isinstance(deadline, (int, float))
        or not math.isfinite(float(deadline))
        or float(deadline) < float(expected)
        or float(deadline) > 7 * 24 * 60 * 60
    ):
        raise ValueError(
            "Expected hard deadline to be >= duration and <= 604800 seconds. "
            f"Provided value: duration={expected!r}, deadline={deadline!r}."
        )
    for path_key in (
        "approved_plan_path",
        "preflight_receipt_path",
        "tracker_gate_receipt_path",
        "output_root",
    ):
        raw_path = value[path_key]
        if not isinstance(raw_path, str) or not Path(raw_path).is_absolute():
            raise ValueError(
                f"Expected {path_key} to be an absolute string. "
                f"Provided value: {raw_path!r}."
            )
        normalized = _lexical_absolute(raw_path)
        if not _under_roots(
            normalized,
            profile["allowed_cwd_roots"],
            resolve=False,
        ):
            raise ValueError(
                f"Expected {path_key} under an approved cwd root. "
                f"Provided value: {raw_path!r}."
            )
        value[path_key] = str(normalized)
    for sha_key in (
        "approved_plan_sha256",
        "preflight_receipt_sha256",
        "tracker_gate_receipt_sha256",
    ):
        sha = value[sha_key]
        if not isinstance(sha, str) or SHA256_RE.fullmatch(sha) is None:
            raise ValueError(
                f"Expected {sha_key} to be a lowercase SHA-256. "
                f"Provided value: {sha!r}."
            )
    value["cwd"] = str(cwd)
    value["argv"] = list(argv)
    value["env"] = normalized_env
    value["expected_duration_seconds"] = float(expected)
    value["hard_deadline_seconds"] = float(deadline)
    serialized = _canonical_bytes(value)
    if len(serialized) > MAX_REQUEST_BYTES:
        raise ValueError(
            f"Expected canonical launch request to be at most "
            f"{MAX_REQUEST_BYTES} bytes. Provided value: {len(serialized)} bytes."
        )
    return value


def build_request(
    *,
    profile: Mapping[str, Any],
    attempt_id: str,
    experiment_id: str,
    cwd: str,
    argv: Sequence[str],
    env: Mapping[str, str],
    source_id: str,
    expected_duration_seconds: float,
    hard_deadline_seconds: float,
    approved_plan_path: str,
    tracker_gate_receipt_path: str,
) -> dict[str, Any]:
    cwd_path = _lexical_absolute(cwd)
    plan_path = _lexical_absolute(approved_plan_path)
    bindings = _load_plan_bindings(
        plan_path,
        cwd=cwd_path,
        experiment_id=experiment_id,
    )
    if not _command_matches_launcher(
        argv,
        bindings["launcher"],
        profile=profile,
    ):
        raise ValueError(
            "Expected command argv to exactly match the approved plan launcher. "
            f"Provided value: argv={list(argv)!r}, "
            f"launcher={bindings['launcher']!r}."
        )
    preflight_path = Path(bindings["preflight_receipt_path"])
    if preflight_path.is_symlink() or not preflight_path.is_file():
        raise ValueError(
            "Expected the approved plan preflight receipt to be an existing "
            f"non-symlink file. Provided value: {preflight_path}."
        )
    preflight_payload = preflight_path.read_bytes()
    preflight_value = _strict_json_bytes(
        preflight_payload,
        label="approved plan preflight receipt",
    )
    if not isinstance(preflight_value, dict):
        raise ValueError(
            "Expected the approved plan preflight receipt to be an object. "
            f"Provided value: {preflight_value!r}."
        )
    normalized_env = dict(env)
    normalized_env.setdefault(
        "CUDA_VISIBLE_DEVICES",
        str(profile["gpu"]["device_index"]),
    )
    try:
        _validate_preflight_envelope(
            preflight_value,
            profile=profile,
            cwd=cwd_path,
            experiment_id=experiment_id,
            plan_bindings=bindings,
            source_id=source_id,
            argv=argv,
            env=dict(sorted(normalized_env.items())),
        )
    except DispatchError as exc:
        raise ValueError(str(exc)) from exc
    tracker_path = _lexical_absolute(tracker_gate_receipt_path)
    if tracker_path.is_symlink() or not tracker_path.is_file():
        raise ValueError(
            "Expected tracker gate receipt to be an existing non-symlink file. "
            f"Provided value: {tracker_path}."
        )
    tracker_payload = tracker_path.read_bytes()
    return validate_request(
        {
            "schema_version": REQUEST_SCHEMA,
            "attempt_id": attempt_id,
            "experiment_id": experiment_id,
            "target": profile["target"],
            "cwd": cwd,
            "argv": list(argv),
            "env": dict(env),
            "source_id": source_id,
            "expected_duration_seconds": expected_duration_seconds,
            "hard_deadline_seconds": hard_deadline_seconds,
            "profile_sha256": profile["_sha256"],
            "approved_plan_path": bindings["approved_plan_path"],
            "approved_plan_sha256": bindings["approved_plan_sha256"],
            "preflight_receipt_path": str(preflight_path),
            "preflight_receipt_sha256": _sha256_bytes(preflight_payload),
            "tracker_gate_receipt_path": str(tracker_path),
            "tracker_gate_receipt_sha256": _sha256_bytes(tracker_payload),
            "output_root": bindings["output_root"],
        },
        profile=profile,
    )


def _validate_preflight_inputs(
    *,
    profile: Mapping[str, Any],
    experiment_id: str,
    cwd: str,
    argv: Sequence[str],
    env: Mapping[str, str],
    source_id: str,
    approved_plan_path: str,
) -> tuple[Path, dict[str, Any], list[str], dict[str, str]]:
    if (
        not isinstance(experiment_id, str)
        or EXPERIMENT_RE.fullmatch(experiment_id) is None
    ):
        raise ValueError(
            "Expected experiment_id to be a safe lowercase identifier. "
            f"Provided value: {experiment_id!r}."
        )
    if SOURCE_RE.fullmatch(source_id) is None:
        raise ValueError(
            "Expected source_id to be an exact lowercase 40-character Git "
            f"commit. Provided value: {source_id!r}."
        )
    cwd_path = _lexical_absolute(cwd)
    if cwd_path != _lexical_absolute(str(profile["repo_root"])):
        raise ValueError(
            f"Expected cwd to equal profile repo root {profile['repo_root']!r}. "
            f"Provided value: {cwd!r}."
        )
    plan_path = _lexical_absolute(approved_plan_path)
    try:
        _require_canonical_below(
            plan_path,
            root=cwd_path,
            label="approved experiment plan",
            require_file=True,
        )
    except DispatchError as exc:
        raise ValueError(str(exc)) from exc
    bindings = _load_plan_bindings(
        plan_path,
        cwd=cwd_path,
        experiment_id=experiment_id,
    )
    normalized_argv = list(argv)
    if (
        not normalized_argv
        or not all(
            isinstance(item, str) and item and "\0" not in item
            for item in normalized_argv
        )
        or not _command_matches_launcher(
            normalized_argv,
            bindings["launcher"],
            profile=profile,
        )
    ):
        raise ValueError(
            "Expected exact production argv from the approved plan launcher. "
            f"Provided value: argv={normalized_argv!r}, "
            f"launcher={bindings['launcher']!r}."
        )
    normalized_env = _normalize_request_env(env, profile=profile)
    return cwd_path, bindings, normalized_argv, normalized_env


def build_preflight(
    *,
    profile: Mapping[str, Any],
    experiment_id: str,
    cwd: str,
    argv: Sequence[str],
    env: Mapping[str, str],
    source_id: str,
    approved_plan_path: str,
    smoke_receipt_path: str,
    tk_reference_receipt_path: str | None = None,
    scheduled_run_preflight_receipt_path: str | None = None,
) -> dict[str, Any]:
    """Build the canonical pre-arm envelope from schema-validated receipts."""

    cwd_path, bindings, normalized_argv, normalized_env = (
        _validate_preflight_inputs(
            profile=profile,
            experiment_id=experiment_id,
            cwd=cwd,
            argv=argv,
            env=env,
            source_id=source_id,
            approved_plan_path=approved_plan_path,
        )
    )
    provided_paths = {
        "smoke": smoke_receipt_path,
        "tk_reference": tk_reference_receipt_path,
        "scheduled_run_preflight": scheduled_run_preflight_receipt_path,
    }
    checks: dict[str, Any] = {}
    for role, required in bindings["required_checks"].items():
        path_value = provided_paths[role]
        if not required:
            if path_value is not None:
                raise ValueError(
                    f"Expected no receipt for nonrequired role {role!r}. "
                    f"Provided value: {path_value!r}."
                )
            checks[role] = {
                "required": False,
                "status": "not_required",
                "schema_version": None,
                "validator_id": None,
                "receipt_path": None,
                "receipt_sha256": None,
                "claims": None,
            }
            continue
        if not isinstance(path_value, str) or not path_value:
            raise ValueError(
                f"Expected --{role.replace('_', '-')}-receipt for required "
                f"role {role!r}. Provided value: {path_value!r}."
            )
        receipt_path = _lexical_absolute(path_value)
        if not receipt_path.is_file() or receipt_path.is_symlink():
            raise ValueError(
                f"Expected {role!r} receipt to be an existing non-symlink "
                f"file. Provided value: {receipt_path}."
            )
        receipt_sha256 = _sha256_bytes(receipt_path.read_bytes())
        expected_schema = bindings["receipt_schemas"][role]
        try:
            validated_path, validated = _validate_preflight_check(
                role=role,
                expected_schema=str(expected_schema),
                receipt_path_value=str(receipt_path),
                receipt_sha256_value=receipt_sha256,
                profile=profile,
                cwd=cwd_path,
                experiment_id=experiment_id,
                plan_bindings=bindings,
                source_id=source_id,
            )
        except DispatchError as exc:
            raise ValueError(str(exc)) from exc
        checks[role] = {
            "required": True,
            "status": "passed",
            "schema_version": expected_schema,
            "validator_id": validated["validator_id"],
            "receipt_path": str(validated_path),
            "receipt_sha256": receipt_sha256,
            "claims": validated["claims"],
        }
    envelope_path = Path(str(bindings["preflight_receipt_path"]))
    if envelope_path.exists() or envelope_path.is_symlink():
        if envelope_path.is_symlink() or not envelope_path.is_file():
            raise ValueError(
                "Expected existing preflight envelope to be a real file. "
                f"Provided value: {envelope_path}."
            )
        existing = _read_json(
            envelope_path,
            label="existing preflight envelope",
        )
        try:
            _validate_preflight_envelope(
                existing,
                profile=profile,
                cwd=cwd_path,
                experiment_id=experiment_id,
                plan_bindings=bindings,
                source_id=source_id,
                argv=normalized_argv,
                env=normalized_env,
            )
        except DispatchError as exc:
            raise ValueError(str(exc)) from exc
        expected_paths = {
            role: (
                str(_lexical_absolute(path))
                if path is not None
                else None
            )
            for role, path in provided_paths.items()
        }
        observed_paths = {
            role: existing["checks"][role]["receipt_path"]
            for role in provided_paths
        }
        if observed_paths != expected_paths:
            raise ValueError(
                "Expected existing preflight envelope to bind the supplied "
                f"receipts. Provided value: expected={expected_paths!r}, "
                f"observed={observed_paths!r}."
            )
        return {
            "status": "already_built",
            "side_effects": 0,
            "preflight_receipt_path": str(envelope_path),
            "preflight_receipt_sha256": _sha256_bytes(
                envelope_path.read_bytes()
            ),
            "envelope": existing,
        }
    envelope = {
        "schema_version": PREFLIGHT_SCHEMA,
        "status": "passed",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment_id": experiment_id,
        "approved_plan_sha256": bindings["approved_plan_sha256"],
        "profile_sha256": profile["_sha256"],
        "source_id": source_id,
        "target": profile["target"],
        "device_index": profile["gpu"]["device_index"],
        "output_root": bindings["output_root"],
        "argv": normalized_argv,
        "env": _environment_binding(profile, normalized_env),
        "checks": checks,
    }
    try:
        _validate_preflight_envelope(
            envelope,
            profile=profile,
            cwd=cwd_path,
            experiment_id=experiment_id,
            plan_bindings=bindings,
            source_id=source_id,
            argv=normalized_argv,
            env=normalized_env,
        )
    except DispatchError as exc:
        raise ValueError(str(exc)) from exc
    _atomic_create(envelope_path, envelope)
    return {
        "status": "built",
        "side_effects": 1,
        "preflight_receipt_path": str(envelope_path),
        "preflight_receipt_sha256": _sha256_bytes(
            envelope_path.read_bytes()
        ),
        "envelope": envelope,
    }


def verify_preflight(
    *,
    profile: Mapping[str, Any],
    experiment_id: str,
    cwd: str,
    argv: Sequence[str],
    env: Mapping[str, str],
    source_id: str,
    approved_plan_path: str,
) -> dict[str, Any]:
    """Read-only verification of the plan-declared preflight envelope."""

    cwd_path, bindings, normalized_argv, normalized_env = (
        _validate_preflight_inputs(
            profile=profile,
            experiment_id=experiment_id,
            cwd=cwd,
            argv=argv,
            env=env,
            source_id=source_id,
            approved_plan_path=approved_plan_path,
        )
    )
    envelope_path = Path(str(bindings["preflight_receipt_path"]))
    envelope = _read_json(
        envelope_path,
        label="preflight envelope",
    )
    try:
        checks = _validate_preflight_envelope(
            envelope,
            profile=profile,
            cwd=cwd_path,
            experiment_id=experiment_id,
            plan_bindings=bindings,
            source_id=source_id,
            argv=normalized_argv,
            env=normalized_env,
        )
    except DispatchError as exc:
        raise ValueError(str(exc)) from exc
    return {
        "status": "verified",
        "side_effects": 0,
        "preflight_receipt_path": str(envelope_path),
        "preflight_receipt_sha256": _sha256_bytes(
            envelope_path.read_bytes()
        ),
        "checks": checks,
    }


def _run(
    runner: Runner,
    command: Sequence[str],
    *,
    timeout: float,
    input_text: str | None = None,
) -> subprocess.CompletedProcess[str]:
    return runner(
        list(command),
        check=False,
        text=True,
        capture_output=True,
        timeout=timeout,
        input=input_text,
    )


def _require_success(
    completed: subprocess.CompletedProcess[str],
    *,
    stage: str,
) -> None:
    if completed.returncode != 0:
        raise DispatchError(
            stage,
            f"Expected command to exit zero. Provided command={completed.args!r}, "
            f"returncode={completed.returncode}, stdout={completed.stdout!r}, "
            f"stderr={completed.stderr!r}.",
        )


def _validate_target_authorization(
    request: Mapping[str, Any],
    *,
    profile: Mapping[str, Any],
    attempt_root: Path,
    runner: Runner,
    timeout: float,
) -> dict[str, Any]:
    cwd = Path(str(request["cwd"]))
    repo_root = Path(str(profile["repo_root"]))
    if (
        cwd != repo_root
        or cwd.is_symlink()
        or not cwd.is_dir()
        or cwd.resolve() != cwd
    ):
        raise DispatchError(
            "launch_authorization",
            f"Expected cwd to equal canonical profile repo root {repo_root}. "
            f"Provided value: {cwd}.",
        )
    executable = Path(str(request["argv"][0]))
    if (
        not executable.is_file()
        or not os.access(executable, os.X_OK)
        or not _under_roots(
            executable,
            profile["allowed_executable_roots"],
            resolve=True,
        )
    ):
        raise DispatchError(
            "launch_authorization",
            "Expected argv[0] to be an executable whose resolved target is "
            f"under an approved root. Provided value: {executable}.",
        )
    profile_python = Path(str(profile["python_path"]))
    env_executable = Path(ENV_EXECUTABLE)
    helper_path = Path(str(profile["helper_path"]))
    profile_path = Path(str(profile["profile_path"]))
    if not profile_python.is_file() or not os.access(profile_python, os.X_OK):
        raise DispatchError(
            "launch_authorization",
            f"Expected profile Python to be executable. "
            f"Provided value: {profile_python}.",
        )
    if not env_executable.is_file() or not os.access(env_executable, os.X_OK):
        raise DispatchError(
            "launch_authorization",
            f"Expected isolated-environment executable at {env_executable}. "
            f"Provided value: {env_executable}.",
        )
    for label, path in (
        ("dispatch helper", helper_path),
        ("dispatch profile", profile_path),
    ):
        _require_canonical_below(
            path,
            root=cwd,
            label=label,
            require_file=True,
        )
    if _sha256_bytes(profile_path.read_bytes()) != profile["_sha256"]:
        raise DispatchError(
            "launch_authorization",
            f"Expected target dispatch profile hash {profile['_sha256']!r}. "
            f"Provided value: {_sha256_bytes(profile_path.read_bytes())!r}.",
        )

    plan_path, _plan_payload = _validate_bound_file_bytes(
        request["approved_plan_path"],
        request["approved_plan_sha256"],
        label="approved experiment plan",
    )
    _require_canonical_below(
        plan_path,
        root=cwd,
        label="approved experiment plan",
        require_file=True,
    )
    plan_bindings = _load_plan_bindings(
        plan_path,
        cwd=cwd,
        experiment_id=str(request["experiment_id"]),
    )
    expected_binding = {
        key: request[key]
        for key in (
            "approved_plan_path",
            "approved_plan_sha256",
            "preflight_receipt_path",
            "output_root",
        )
    }
    observed_binding = {
        key: plan_bindings[key]
        for key in (
            "approved_plan_path",
            "approved_plan_sha256",
            "preflight_receipt_path",
            "output_root",
        )
    }
    if observed_binding != expected_binding:
        raise DispatchError(
            "launch_authorization",
            "Expected approved plan bindings to remain unchanged. "
            f"Provided value: expected={expected_binding!r}, "
            f"observed={observed_binding!r}.",
        )
    if not _command_matches_launcher(
        request["argv"],
        plan_bindings["launcher"],
        profile=profile,
    ):
        raise DispatchError(
            "launch_authorization",
            "Expected payload command to exactly match the approved plan launcher. "
            f"Provided value: argv={request['argv']!r}, "
            f"launcher={plan_bindings['launcher']!r}.",
        )
    validator = (
        cwd
        / "skills"
        / "run-experiment-pipeline"
        / "scripts"
        / "validate_experiment_plan.py"
    )
    _require_canonical_below(
        validator,
        root=cwd,
        label="canonical experiment-plan validator",
        require_file=True,
    )
    plan_validation = _run(
        runner,
        [
            str(profile["python_path"]),
            str(validator),
            str(plan_path),
            "--require-approved",
            "--verify-files",
        ],
        timeout=timeout,
    )
    _require_success(plan_validation, stage="approved_plan_validation")

    _preflight_path, preflight, _preflight_payload = _validate_bound_json_file(
        request["preflight_receipt_path"],
        request["preflight_receipt_sha256"],
        label="approved preflight receipt",
    )
    _require_canonical_below(
        _preflight_path,
        root=cwd,
        label="approved preflight receipt",
        require_file=True,
    )
    validated_checks = _validate_preflight_envelope(
        preflight,
        profile=profile,
        cwd=cwd,
        experiment_id=str(request["experiment_id"]),
        plan_bindings=plan_bindings,
        source_id=str(request["source_id"]),
        argv=request["argv"],
        env=request["env"],
    )

    gate_path, gate, _gate_payload = _validate_bound_json_file(
        request["tracker_gate_receipt_path"],
        request["tracker_gate_receipt_sha256"],
        label="tracker production-gate receipt",
    )
    _require_canonical_below(
        gate_path,
        root=cwd,
        label="tracker production-gate receipt",
        require_file=True,
    )
    _require_canonical_below(
        Path(str(request["output_root"])),
        root=cwd,
        label="experiment output root",
        require_file=False,
    )
    expected_gate_values = {
        "schema_version": TRACKER_GATE_SCHEMA,
        "experiment_id": request["experiment_id"],
        "gate_stage": "production",
        "status": "launch-ready",
        "state_path": str(attempt_root / "dispatch.json"),
        "output_root": request["output_root"],
    }
    for key, expected in expected_gate_values.items():
        if gate.get(key) != expected:
            raise DispatchError(
                "launch_authorization",
                f"Expected tracker gate {key}={expected!r}. "
                f"Provided value: {gate.get(key)!r} in {gate_path}.",
            )
    generated = gate.get("generated_at_utc")
    if not isinstance(generated, str):
        raise DispatchError(
            "launch_authorization",
            f"Expected tracker gate generated_at_utc. Provided value: {generated!r}.",
        )
    try:
        generated_at = datetime.fromisoformat(generated)
    except ValueError as exc:
        raise DispatchError(
            "launch_authorization",
            f"Expected tracker gate generated_at_utc to be ISO-8601. "
            f"Provided value: {generated!r}.",
        ) from exc
    if generated_at.tzinfo is None:
        raise DispatchError(
            "launch_authorization",
            f"Expected timezone-aware tracker gate timestamp. "
            f"Provided value: {generated!r}.",
        )
    gate_age = (
        datetime.now(timezone.utc) - generated_at.astimezone(timezone.utc)
    ).total_seconds()
    if not -60 <= gate_age <= MAX_TRACKER_GATE_AGE_SECONDS:
        raise DispatchError(
            "launch_authorization",
            "Expected tracker production gate no older than "
            f"{MAX_TRACKER_GATE_AGE_SECONDS} seconds. "
            f"Provided age: {gate_age:.3f} seconds.",
        )
    tracker_path = Path(str(gate.get("tracker_path", "")))
    if not tracker_path.is_absolute():
        raise DispatchError(
            "launch_authorization",
            "Expected tracker gate to bind an absolute canonical tracker path. "
            f"Provided value: {tracker_path}.",
        )
    if profile["target"] == "main":
        if tracker_path.is_symlink() or not tracker_path.is_file():
            raise DispatchError(
                "launch_authorization",
                "Expected the local tracker gate to bind a live non-symlink "
                f"tracker. Provided value: {tracker_path}.",
            )
        tracker_payload = tracker_path.read_bytes()
        if _sha256_bytes(tracker_payload) != gate.get("tracker_sha256"):
            raise DispatchError(
                "launch_authorization",
                "Expected the live local tracker bytes to match the production "
                f"gate. Provided value: {tracker_path}.",
            )
    tracker_validator = (
        cwd
        / "skills"
        / "run-experiment-pipeline"
        / "scripts"
        / "validate_current_experiments.py"
    )
    _require_canonical_below(
        tracker_validator,
        root=cwd,
        label="canonical tracker validator",
        require_file=True,
    )
    if _sha256_bytes(tracker_validator.read_bytes()) != gate.get(
        "validator_sha256"
    ):
        raise DispatchError(
            "launch_authorization",
            "Expected tracker gate validator hash to match the target checkout. "
            f"Provided value: {gate.get('validator_sha256')!r}.",
        )

    git_head = _run(
        runner,
        ["git", "-C", str(cwd), "rev-parse", "HEAD"],
        timeout=timeout,
    )
    _require_success(git_head, stage="source_identity")
    observed_head = git_head.stdout.strip()
    if observed_head != request["source_id"]:
        raise DispatchError(
            "source_identity",
            f"Expected target Git HEAD {request['source_id']!r}. "
            f"Provided value: {observed_head!r}.",
        )
    git_top = _run(
        runner,
        ["git", "-C", str(cwd), "rev-parse", "--show-toplevel"],
        timeout=timeout,
    )
    _require_success(git_top, stage="source_identity")
    if _lexical_absolute(git_top.stdout.strip()) != cwd:
        raise DispatchError(
            "source_identity",
            f"Expected target Git top-level {cwd}. "
            f"Provided value: {git_top.stdout.strip()!r}.",
        )
    tracked_clean = _run(
        runner,
        ["git", "-C", str(cwd), "diff-index", "--quiet", "HEAD", "--"],
        timeout=timeout,
    )
    _require_success(tracked_clean, stage="source_identity")
    untracked_clean = _run(
        runner,
        [
            "git",
            "-C",
            str(cwd),
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
        ],
        timeout=timeout,
    )
    _require_success(untracked_clean, stage="source_identity")
    if untracked_clean.stdout.strip():
        raise DispatchError(
            "source_identity",
            "Expected target worktree to contain no tracked changes or "
            f"untracked files. Provided value: {untracked_clean.stdout!r}.",
        )
    tracked_authority_paths = [
        str(path.relative_to(cwd))
        for path in (
            plan_path,
            helper_path,
            profile_path,
            validator,
            tracker_validator,
        )
    ]
    tracked_authority = _run(
        runner,
        [
            "git",
            "-C",
            str(cwd),
            "ls-files",
            "--error-unmatch",
            "--",
            *tracked_authority_paths,
        ],
        timeout=timeout,
    )
    _require_success(tracked_authority, stage="source_identity")
    return {
        "approved_plan_path": str(plan_path),
        "approved_plan_sha256": request["approved_plan_sha256"],
        "preflight_receipt_path": request["preflight_receipt_path"],
        "preflight_receipt_sha256": request["preflight_receipt_sha256"],
        "preflight_checks": validated_checks,
        "tracker_gate_receipt_path": str(gate_path),
        "tracker_gate_receipt_sha256": request["tracker_gate_receipt_sha256"],
        "tracker_sha256": gate["tracker_sha256"],
        "source_id": request["source_id"],
    }


def _parse_tmux_identity(stdout: str, *, stage: str) -> dict[str, Any]:
    lines = [line.strip() for line in stdout.splitlines() if line.strip()]
    if len(lines) != 1:
        raise DispatchError(
            stage,
            f"Expected one tmux identity line. Provided value: {stdout!r}.",
            launched_job_count=1,
        )
    fields = lines[0].split("|")
    if len(fields) != 3:
        raise DispatchError(
            stage,
            f"Expected window|pane|pid tmux identity. Provided value: {lines[0]!r}.",
            launched_job_count=1,
        )
    window_id, pane_id, pane_pid = fields
    if (
        not window_id.startswith("@")
        or not pane_id.startswith("%")
        or not pane_pid.isdigit()
    ):
        raise DispatchError(
            stage,
            f"Expected valid tmux identifiers. Provided value: {fields!r}.",
            launched_job_count=1,
        )
    return {
        "window_id": window_id,
        "pane_id": pane_id,
        "pane_pid": pane_pid,
    }


def _write_failure_once(path: Path, error: DispatchError) -> None:
    try:
        _atomic_create(path, error.as_dict())
    except FileExistsError:
        pass


def _acquire_lock(lock: Any, *, timeout: float) -> None:
    deadline = time.monotonic() + min(timeout, LOCK_TIMEOUT_SECONDS)
    while True:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            return
        except BlockingIOError:
            if time.monotonic() >= deadline:
                raise DispatchError(
                    "dispatch_lock_timeout",
                    "Expected the target dispatch lock to become available "
                    f"within {min(timeout, LOCK_TIMEOUT_SECONDS):g} seconds.",
                    launched_job_count=0,
                    smallest_next_action=(
                        "Report this terminal attempt, inspect the current "
                        "lock owner read-only, and use a new attempt ID only "
                        "after explicit authorization."
                    ),
                )
            time.sleep(0.05)


def _ensure_no_symlink_components(path: Path) -> None:
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current = current / part
        if current.is_symlink():
            raise DispatchError(
                "state_root",
                f"Expected dispatch state path without symlink components. "
                f"Provided value: {current}.",
            )


def _tmux_attempt_name(prefix: str, attempt_id: str, *, label: str) -> str:
    normalized = attempt_id.replace(".", "-")
    suffix = hashlib.sha256(attempt_id.encode("utf-8")).hexdigest()[:10]
    available = max(1, 80 - len(prefix) - len(suffix) - 1)
    return f"{prefix}{normalized[:available]}-{suffix}"


def _validate_tmux_readback(
    stdout: str,
    *,
    identity: Mapping[str, Any],
) -> str:
    lines = [line.strip() for line in stdout.splitlines() if line.strip()]
    if len(lines) != 1:
        raise DispatchError(
            "post_dispatch_readback",
            f"Expected one tmux readback line. Provided value: {stdout!r}.",
            launched_job_count=1,
            launch_id=str(identity["pane_id"]),
        )
    fields = lines[0].split("|")
    expected = [
        str(identity["session"]),
        str(identity["window_id"]),
        str(identity["pane_id"]),
        str(identity["pane_pid"]),
        "0",
    ]
    if fields != expected:
        raise DispatchError(
            "post_dispatch_readback",
            "Expected live tmux session|window|pane|pid|0 matching the launch. "
            f"Provided value: {fields!r}; expected: {expected!r}.",
            launched_job_count=1,
            launch_id=str(identity["pane_id"]),
        )
    return lines[0]


def _validate_existing_dispatch(
    path: Path,
    *,
    request: Mapping[str, Any],
    request_path: Path,
    profile: Mapping[str, Any],
) -> dict[str, Any]:
    value = _read_json(path, label="existing dispatch receipt")
    expected = {
        "schema_version": DISPATCH_SCHEMA,
        "status": "dispatched",
        "attempt_id": request["attempt_id"],
        "experiment_id": request["experiment_id"],
        "target": request["target"],
        "request_path": str(request_path),
        "request_sha256": _sha256_bytes(request_path.read_bytes()),
        "profile_sha256": profile["_sha256"],
        "source_id": request["source_id"],
    }
    for key, expected_value in expected.items():
        if value.get(key) != expected_value:
            raise DispatchError(
                "dispatch_receipt_mismatch",
                f"Expected existing dispatch {key}={expected_value!r}. "
                f"Provided value: {value.get(key)!r}.",
                launched_job_count=None,
            )
    if (
        value.get("launched_job_count") != 1
        or not isinstance(value.get("authorization"), dict)
        or not isinstance(value.get("tmux"), dict)
    ):
        raise DispatchError(
            "dispatch_receipt_mismatch",
            f"Expected a launched, authorization-bound tmux dispatch receipt. "
            f"Provided value: {value!r}.",
            launched_job_count=None,
        )
    return value


def _validate_attempt_receipt(
    value: Mapping[str, Any],
    *,
    schema: str,
    status: str,
    request: Mapping[str, Any],
    request_path: Path,
) -> dict[str, Any]:
    expected = {
        "schema_version": schema,
        "status": status,
        "attempt_id": request["attempt_id"],
        "experiment_id": request["experiment_id"],
        "target": request["target"],
    }
    for key, expected_value in expected.items():
        if value.get(key) != expected_value:
            raise DispatchError(
                "attempt_receipt_mismatch",
                f"Expected {status} receipt {key}={expected_value!r}. "
                f"Provided value: {value.get(key)!r}.",
                launched_job_count=None,
            )
    expected_request_sha = _sha256_bytes(request_path.read_bytes())
    if schema == FAILURE_SCHEMA:
        artifacts = value.get("last_valid_artifacts")
        if (
            value.get("terminal") is not True
            or value.get("retry_authorized") is not False
            or not isinstance(artifacts, dict)
            or artifacts.get("request_path") != str(request_path)
            or artifacts.get("request_sha256") != expected_request_sha
        ):
            raise DispatchError(
                "attempt_receipt_mismatch",
                "Expected terminal failure receipt to bind the immutable "
                f"request. Provided value: {value!r}.",
                launched_job_count=None,
            )
    elif value.get("request_sha256") != expected_request_sha:
        raise DispatchError(
            "attempt_receipt_mismatch",
            f"Expected {status} receipt request_sha256 "
            f"{expected_request_sha!r}. "
            f"Provided value: {value.get('request_sha256')!r}.",
            launched_job_count=None,
        )
    return dict(value)


def _active_attempts(state_root: Path, *, exclude: str) -> list[str]:
    active: list[str] = []
    for child in sorted(state_root.iterdir(), key=lambda item: item.name):
        if (
            child.name.startswith(".")
            or child.name == exclude
            or child.is_symlink()
            or not child.is_dir()
        ):
            continue
        request_path = child / "request.json"
        has_request = request_path.exists()
        has_dispatch = (child / "dispatch.json").exists()
        request: dict[str, Any] = {}
        request_payload = b""
        if request_path.is_file() and not request_path.is_symlink():
            try:
                request_payload = request_path.read_bytes()
                request = _read_json(
                    request_path,
                    label=f"attempt {child.name} request receipt",
                )
            except (OSError, ValueError, json.JSONDecodeError):
                request = {}
                request_payload = b""
        request_sha256 = _sha256_bytes(request_payload)
        request_is_bound = (
            request.get("schema_version") == REQUEST_SCHEMA
            and request.get("attempt_id") == child.name
            and isinstance(request.get("experiment_id"), str)
            and request.get("target") in {"main", "trex"}
        )
        completion_path = child / "completion.json"
        failure_path = child / "failure.json"
        quiescence_path = child / "quiescence.json"
        safely_terminal = False
        if (
            request_is_bound
            and completion_path.is_file()
            and not completion_path.is_symlink()
        ):
            try:
                completion = _read_json(
                    completion_path,
                    label=f"attempt {child.name} completion receipt",
                )
            except (OSError, ValueError, json.JSONDecodeError):
                completion = {}
            safely_terminal = (
                completion.get("schema_version") == COMPLETION_SCHEMA
                and completion.get("status") == "complete"
                and completion.get("attempt_id") == child.name
                and completion.get("experiment_id")
                == request["experiment_id"]
                and completion.get("target") == request["target"]
                and completion.get("request_sha256") == request_sha256
            )
        if (
            not safely_terminal
            and request_is_bound
            and failure_path.is_file()
            and not failure_path.is_symlink()
        ):
            try:
                failure = _read_json(
                    failure_path,
                    label=f"attempt {child.name} failure receipt",
                )
            except (OSError, ValueError, json.JSONDecodeError):
                failure = {}
            safely_terminal = (
                failure.get("schema_version") == FAILURE_SCHEMA
                and failure.get("status") == "failed"
                and failure.get("terminal") is True
                and failure.get("launched_job_count") == 0
                and failure.get("attempt_id") == child.name
                and failure.get("experiment_id") == request["experiment_id"]
                and failure.get("target") == request["target"]
                and isinstance(failure.get("last_valid_artifacts"), dict)
                and failure["last_valid_artifacts"].get("request_path")
                == str(request_path)
                and failure["last_valid_artifacts"].get("request_sha256")
                == request_sha256
            )
        if (
            not safely_terminal
            and request_is_bound
            and quiescence_path.is_file()
            and not quiescence_path.is_symlink()
        ):
            try:
                quiescence = _read_json(
                    quiescence_path,
                    label=f"attempt {child.name} quiescence receipt",
                )
            except (OSError, ValueError, json.JSONDecodeError):
                quiescence = {}
            safely_terminal = (
                quiescence.get("schema_version") == QUIESCENCE_SCHEMA
                and quiescence.get("status") == "quiescent"
                and quiescence.get("attempt_id") == child.name
                and quiescence.get("experiment_id") == request["experiment_id"]
                and quiescence.get("target") == request["target"]
                and quiescence.get("request_sha256") == request_sha256
                and quiescence.get("live_descendant_count") == 0
            )
        if (has_request or has_dispatch) and not safely_terminal:
            active.append(child.name)
    return active


def dispatch_on_target(
    request: Mapping[str, Any],
    *,
    profile: Mapping[str, Any],
    runner: Runner = subprocess.run,
) -> dict[str, Any]:
    """Validate, lock, probe, dispatch once, read back once, and return."""

    value = validate_request(request, profile=profile)
    state_root = _lexical_absolute(str(profile["state_root"]))
    _require_canonical_below(
        state_root,
        root=Path(str(profile["repo_root"])),
        label="dispatch state root",
        require_file=False,
    )
    _ensure_no_symlink_components(state_root)
    state_root.mkdir(parents=True, exist_ok=True)
    if state_root.is_symlink() or not state_root.is_dir():
        raise DispatchError(
            "state_root",
            f"Expected a real dispatch state directory. Provided value: {state_root}.",
        ).bind_request(value)
    lock_path = state_root / ".dispatch.lock"
    if lock_path.is_symlink():
        raise DispatchError(
            "state_root",
            f"Expected dispatch lock not to be a symlink. "
            f"Provided value: {lock_path}.",
        ).bind_request(value)
    timeout = float(profile["command_timeout_seconds"])
    with lock_path.open("a+", encoding="utf-8") as lock:
        try:
            _acquire_lock(lock, timeout=timeout)
        except DispatchError as exc:
            error = exc.bind_request(value)
            attempt_root = state_root / value["attempt_id"]
            request_path = attempt_root / "request.json"
            failure_path = attempt_root / "failure.json"
            try:
                attempt_root.mkdir(parents=False)
                _atomic_create(request_path, value)
            except FileExistsError:
                pass
            else:
                error.bind_request(
                    value,
                    last_valid_artifacts={
                        "request_path": str(request_path),
                        "request_sha256": _sha256_bytes(
                            request_path.read_bytes()
                        ),
                    },
                )
                _write_failure_once(failure_path, error)
            raise error from exc
        attempt_root = state_root / value["attempt_id"]
        request_path = attempt_root / "request.json"
        dispatch_path = attempt_root / "dispatch.json"
        failure_path = attempt_root / "failure.json"
        completion_path = attempt_root / "completion.json"
        request_bytes = _canonical_bytes(value) + b"\n"
        if request_path.exists() or request_path.is_symlink():
            if request_path.is_symlink() or request_path.read_bytes() != request_bytes:
                raise DispatchError(
                    "attempt_collision",
                    "Expected an existing attempt ID to retain an identical "
                    f"immutable request. Provided value: {value['attempt_id']!r}.",
                ).bind_request(value)
            if failure_path.exists() or failure_path.is_symlink():
                persisted = _read_json(
                    failure_path,
                    label="terminal failure receipt",
                )
                _validate_attempt_receipt(
                    persisted,
                    schema=FAILURE_SCHEMA,
                    status="failed",
                    request=value,
                    request_path=request_path,
                )
                raise DispatchError.from_dict(persisted)
            if completion_path.exists() or completion_path.is_symlink():
                if completion_path.is_symlink() or not completion_path.is_file():
                    raise DispatchError(
                        "completion_receipt_mismatch",
                        f"Expected a real completion receipt. "
                        f"Provided value: {completion_path}.",
                        launched_job_count=1,
                    ).bind_request(value)
                completion = _read_json(
                    completion_path,
                    label="existing completion receipt",
                )
                _validate_attempt_receipt(
                    completion,
                    schema=COMPLETION_SCHEMA,
                    status="complete",
                    request=value,
                    request_path=request_path,
                )
                return completion
            if dispatch_path.exists() or dispatch_path.is_symlink():
                if dispatch_path.is_symlink() or not dispatch_path.is_file():
                    raise DispatchError(
                        "dispatch_receipt_mismatch",
                        f"Expected a real dispatch receipt. "
                        f"Provided value: {dispatch_path}.",
                        launched_job_count=None,
                    ).bind_request(value)
                existing = _validate_existing_dispatch(
                    dispatch_path,
                    request=value,
                    request_path=request_path,
                    profile=profile,
                )
                return {**existing, "status": "already_dispatched"}
            error = DispatchError(
                "dispatch_outcome_unknown",
                "Expected an existing request to have a dispatch or terminal "
                "receipt. The prior process may have stopped after launching; "
                "this attempt will not be resumed.",
                launched_job_count=None,
            ).bind_request(
                value,
                last_valid_artifacts={
                    "request_path": str(request_path),
                    "request_sha256": _sha256_bytes(
                        request_path.read_bytes()
                    ),
                },
            )
            _write_failure_once(failure_path, error)
            raise error
        if attempt_root.exists() or attempt_root.is_symlink():
            raise DispatchError(
                "attempt_state_collision",
                f"Expected a new attempt state path. Provided value: {attempt_root}.",
                launched_job_count=None,
            ).bind_request(value)
        attempt_root.mkdir(parents=False)
        _atomic_create(request_path, value)

        completed_stages = ["request_persisted"]
        artifacts: dict[str, Any] = {
            "request_path": str(request_path),
            "request_sha256": _sha256_bytes(request_path.read_bytes()),
        }
        current_stage = "launch_authorization"
        launched_count_for_exception: int | None = 0
        try:
            authorization = _validate_target_authorization(
                value,
                profile=profile,
                attempt_root=attempt_root,
                runner=runner,
                timeout=timeout,
            )
            artifacts.update(authorization)
            completed_stages.append("launch_authorization")

            active = _active_attempts(
                state_root,
                exclude=str(value["attempt_id"]),
            )
            if active:
                raise DispatchError(
                    "target_reserved",
                    "Expected no unresolved long-run dispatch on this target. "
                    f"Provided active attempts: {active!r}.",
                )
            completed_stages.append("target_reservation")

            current_stage = "gpu_probe"
            device_index = int(profile["gpu"]["device_index"])
            gpu_probe = _run(
                runner,
                [
                    "nvidia-smi",
                    f"--id={device_index}",
                    "--query-compute-apps=pid",
                    "--format=csv,noheader,nounits",
                ],
                timeout=timeout,
            )
            _require_success(gpu_probe, stage="gpu_probe")
            compute_pids = [
                line.strip()
                for line in gpu_probe.stdout.splitlines()
                if line.strip()
            ]
            if any(not pid.isdigit() for pid in compute_pids):
                raise DispatchError(
                    "gpu_probe",
                    f"Expected numeric compute PIDs from nvidia-smi. "
                    f"Provided value: {compute_pids!r}.",
                )
            if compute_pids:
                raise DispatchError(
                    "gpu_busy",
                    f"Expected target GPU {device_index} to have no compute "
                    f"processes. Provided PIDs: {compute_pids!r}.",
                )
            completed_stages.append("gpu_probe")

            tmux = profile["tmux"]
            if tmux["mode"] == "existing_session":
                session = str(tmux["session"])
                current_stage = "tmux_session_probe"
                session_probe = _run(
                    runner,
                    ["tmux", "has-session", "-t", session],
                    timeout=timeout,
                )
                _require_success(session_probe, stage="tmux_session_probe")
                completed_stages.append("tmux_session_probe")
            else:
                session = _tmux_attempt_name(
                    str(tmux["session_prefix"]),
                    str(value["attempt_id"]),
                    label="session",
                )

            window_name = _tmux_attempt_name(
                "run-",
                str(value["attempt_id"]),
                label="window",
            )
            execute_command = shlex.join(
                _isolated_command(
                    _runtime_environment(profile, value["env"]),
                    [
                        str(profile["python_path"]),
                        str(profile["helper_path"]),
                        "execute",
                        "--profile",
                        str(profile["profile_path"]),
                        "--request",
                        str(request_path),
                        "--request-sha256",
                        _sha256_bytes(request_path.read_bytes()),
                    ],
                )
            )
            tmux_format = "#{window_id}|#{pane_id}|#{pane_pid}"
            if tmux["mode"] == "existing_session":
                launch_command = [
                    "tmux",
                    "new-window",
                    "-d",
                    "-P",
                    "-F",
                    tmux_format,
                    "-t",
                    session,
                    "-n",
                    window_name,
                    execute_command,
                ]
            else:
                launch_command = [
                    "tmux",
                    "new-session",
                    "-d",
                    "-P",
                    "-F",
                    tmux_format,
                    "-s",
                    session,
                    "-n",
                    window_name,
                    execute_command,
                ]
            current_stage = "tmux_dispatch"
            launched_count_for_exception = None
            launch = _run(runner, launch_command, timeout=timeout)
            if launch.returncode != 0:
                raise DispatchError(
                    "tmux_dispatch",
                    f"Expected one detached tmux launch. Provided returncode="
                    f"{launch.returncode}, stdout={launch.stdout!r}, "
                    f"stderr={launch.stderr!r}.",
                    launched_job_count=None,
                )
            launched_count_for_exception = 1
            identity = _parse_tmux_identity(
                launch.stdout,
                stage="tmux_dispatch",
            )
            identity["session"] = session
            completed_stages.append("tmux_dispatch")
            artifacts["tmux"] = identity

            current_stage = "post_dispatch_readback"
            readback = _run(
                runner,
                [
                    "tmux",
                    "display-message",
                    "-p",
                    "-t",
                    identity["pane_id"],
                    "#{session_name}|#{window_id}|#{pane_id}|#{pane_pid}|#{pane_dead}",
                ],
                timeout=timeout,
            )
            if readback.returncode != 0:
                raise DispatchError(
                    "post_dispatch_readback",
                    f"Expected immediate tmux readback for {identity!r}. "
                    f"Provided returncode={readback.returncode}, "
                    f"stderr={readback.stderr!r}.",
                    launched_job_count=1,
                    launch_id=identity["pane_id"],
                )
            readback_value = _validate_tmux_readback(
                readback.stdout,
                identity=identity,
            )
            completed_stages.append("post_dispatch_readback")
            if failure_path.exists() or failure_path.is_symlink():
                if failure_path.is_symlink() or not failure_path.is_file():
                    raise DispatchError(
                        "payload_failed_during_dispatch",
                        "Expected a real payload failure receipt. "
                        f"Provided value: {failure_path}.",
                        launched_job_count=1,
                        launch_id=identity["pane_id"],
                    )
                persisted_failure = _read_json(
                    failure_path,
                    label="payload failure receipt",
                )
                _validate_attempt_receipt(
                    persisted_failure,
                    schema=FAILURE_SCHEMA,
                    status="failed",
                    request=value,
                    request_path=request_path,
                )
                raise DispatchError.from_dict(persisted_failure)
            receipt = {
                "schema_version": DISPATCH_SCHEMA,
                "status": "dispatched",
                "attempt_id": value["attempt_id"],
                "experiment_id": value["experiment_id"],
                "target": value["target"],
                "request_path": str(request_path),
                "request_sha256": _sha256_bytes(request_path.read_bytes()),
                "profile_sha256": profile["_sha256"],
                "source_id": value["source_id"],
                "command_sha256": _sha256_bytes(
                    _canonical_bytes(
                        {
                            "cwd": value["cwd"],
                            "argv": value["argv"],
                            "env": value["env"],
                        }
                    )
                ),
                "authorization": authorization,
                "launched_job_count": 1,
                "tmux": identity,
                "readback": readback_value,
                "log_path": str(attempt_root / "run.log"),
                "dispatched_at_utc": datetime.now(timezone.utc).isoformat(),
            }
            _atomic_create(dispatch_path, receipt)
            return receipt
        except DispatchError as exc:
            error = exc.bind_request(
                value,
                completed_stages=completed_stages,
                last_valid_artifacts=artifacts,
            )
            _write_failure_once(failure_path, error)
            raise error
        except (OSError, subprocess.TimeoutExpired) as exc:
            error = DispatchError(
                current_stage,
                f"Expected bounded {current_stage} to complete. "
                f"Provided error: {exc!r}.",
                launched_job_count=launched_count_for_exception,
                completed_stages=completed_stages,
                last_valid_artifacts=artifacts,
            ).bind_request(value)
            _write_failure_once(failure_path, error)
            raise error from exc
        except Exception as exc:
            error = DispatchError(
                current_stage,
                f"Expected armed dispatch stage {current_stage!r} to complete "
                f"without an unexpected exception. Provided error: "
                f"{type(exc).__name__}: {exc}.",
                launched_job_count=launched_count_for_exception,
                completed_stages=completed_stages,
                last_valid_artifacts=artifacts,
            ).bind_request(value)
            _write_failure_once(failure_path, error)
            raise error from exc


def start_request(
    request: Mapping[str, Any],
    *,
    profile: Mapping[str, Any],
    runner: Runner = subprocess.run,
) -> dict[str, Any]:
    """Start locally or perform exactly one remote Trex transaction."""

    value = validate_request(request, profile=profile)
    if profile["transport"] == "local":
        return dispatch_on_target(value, profile=profile, runner=runner)
    ssh = profile["ssh"]
    remote_command = shlex.join(
        _isolated_command(
            profile["base_env"],
            [
                str(profile["python_path"]),
                str(profile["helper_path"]),
                "remote-dispatch",
                "--profile",
                str(profile["profile_path"]),
            ],
        )
    )
    command = [
        "ssh",
        "-o",
        "BatchMode=yes",
        "-o",
        f"ConnectTimeout={int(ssh['connect_timeout_seconds'])}",
        str(ssh["target"]),
        remote_command,
    ]
    try:
        completed = _run(
            runner,
            command,
            timeout=float(profile["command_timeout_seconds"]),
            input_text=(_canonical_bytes(value) + b"\n").decode("utf-8"),
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise DispatchError(
            "ssh_dispatch_ambiguous",
            f"Expected one bounded BatchMode Trex dispatch transaction. "
            f"Provided error: {exc!r}.",
            launched_job_count=None,
        ) from exc
    if completed.returncode != 0:
        try:
            failure = _strict_json_bytes(
                completed.stdout.encode("utf-8"),
                label="Trex dispatch failure",
            )
        except (UnicodeError, ValueError, json.JSONDecodeError):
            failure = None
        if (
            isinstance(failure, dict)
            and failure.get("schema_version") == FAILURE_SCHEMA
            and failure.get("attempt_id") == value["attempt_id"]
            and failure.get("experiment_id") == value["experiment_id"]
            and failure.get("target") == "trex"
            and failure.get("terminal") is True
        ):
            raise DispatchError.from_dict(failure)
        raise DispatchError(
            "ssh_dispatch_ambiguous",
            f"Expected Trex dispatch to exit zero. Provided returncode="
            f"{completed.returncode}, stdout={completed.stdout!r}, "
            f"stderr={completed.stderr!r}.",
            launched_job_count=None,
        )
    try:
        result = _strict_json_bytes(
            completed.stdout.encode("utf-8"),
            label="Trex dispatch response",
        )
    except (UnicodeError, ValueError, json.JSONDecodeError) as exc:
        raise DispatchError(
            "ssh_dispatch_response",
            f"Expected one structured Trex dispatch receipt. "
            f"Provided stdout={completed.stdout!r}.",
            launched_job_count=None,
        ) from exc
    if (
        not isinstance(result, dict)
        or result.get("schema_version")
        not in {DISPATCH_SCHEMA, COMPLETION_SCHEMA}
        or result.get("attempt_id") != value["attempt_id"]
        or result.get("target") != "trex"
    ):
        raise DispatchError(
            "ssh_dispatch_response",
            f"Expected a matching Trex dispatch receipt. Provided value: {result!r}.",
            launched_job_count=None,
        )
    return result


def status_request(
    *,
    profile: Mapping[str, Any],
    attempt_id: str,
    runner: Runner = subprocess.run,
) -> dict[str, Any]:
    if profile["transport"] == "local":
        return status_on_target(profile=profile, attempt_id=attempt_id)
    if ATTEMPT_RE.fullmatch(attempt_id) is None:
        raise ValueError(
            f"Expected a safe attempt ID. Provided value: {attempt_id!r}."
        )
    ssh = profile["ssh"]
    remote_command = shlex.join(
        _isolated_command(
            profile["base_env"],
            [
                str(profile["python_path"]),
                str(profile["helper_path"]),
                "remote-status",
                "--profile",
                str(profile["profile_path"]),
                "--attempt-id",
                attempt_id,
            ],
        )
    )
    command = [
        "ssh",
        "-o",
        "BatchMode=yes",
        "-o",
        f"ConnectTimeout={int(ssh['connect_timeout_seconds'])}",
        str(ssh["target"]),
        remote_command,
    ]
    try:
        completed = _run(
            runner,
            command,
            timeout=float(profile["command_timeout_seconds"]),
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise DispatchError(
            "ssh_status",
            f"Expected one bounded read-only Trex status transaction. "
            f"Provided error: {exc!r}.",
            launched_job_count=None,
            attempt_id=attempt_id,
            target="trex",
            terminal=False,
            smallest_next_action=(
                "This read-only query did not alter the simulation attempt; "
                "diagnose the transport normally or retry once with evidence."
            ),
        ) from exc
    if completed.returncode != 0:
        raise DispatchError(
            "ssh_status",
            f"Expected Trex status to exit zero. Provided returncode="
            f"{completed.returncode}, stdout={completed.stdout!r}, "
            f"stderr={completed.stderr!r}.",
            launched_job_count=None,
            attempt_id=attempt_id,
            target="trex",
            terminal=False,
            smallest_next_action=(
                "This read-only query did not alter the simulation attempt; "
                "diagnose the transport normally or retry once with evidence."
            ),
        )
    result = _strict_json_bytes(
        completed.stdout.encode("utf-8"),
        label="Trex status response",
    )
    if (
        not isinstance(result, dict)
        or result.get("attempt_id") != attempt_id
        or result.get("target") != "trex"
        or result.get("status")
        not in {
            "not_found",
            "initializing",
            "running",
            "stale_running",
            "dispatched",
            "complete",
            "failed",
        }
    ):
        raise DispatchError(
            "ssh_status_response",
            f"Expected a matching structured Trex status. "
            f"Provided value: {result!r}.",
            launched_job_count=None,
            attempt_id=attempt_id,
            target="trex",
            terminal=False,
            smallest_next_action=(
                "This read-only query did not alter the simulation attempt; "
                "inspect the response contract normally."
            ),
        )
    return result


def status_on_target(
    *,
    profile: Mapping[str, Any],
    attempt_id: str,
) -> dict[str, Any]:
    if ATTEMPT_RE.fullmatch(attempt_id) is None:
        raise ValueError(
            f"Expected a safe attempt ID. Provided value: {attempt_id!r}."
        )
    state_root = _lexical_absolute(str(profile["state_root"]))
    _ensure_no_symlink_components(state_root)
    root = state_root / attempt_id
    if not root.exists():
        return {
            "status": "not_found",
            "attempt_id": attempt_id,
            "target": profile["target"],
        }
    if root.is_symlink() or not root.is_dir():
        raise ValueError(
            f"Expected attempt state to be a real directory. Provided value: {root}."
        )
    request_path = root / "request.json"
    if request_path.is_symlink() or not request_path.is_file():
        raise ValueError(
            f"Expected existing attempt to contain a real request receipt. "
            f"Provided value: {request_path}."
        )
    request = _read_json(request_path, label="attempt request")
    if (
        request.get("schema_version") != REQUEST_SCHEMA
        or request.get("attempt_id") != attempt_id
        or request.get("target") != profile["target"]
    ):
        raise ValueError(
            f"Expected attempt request bound to {attempt_id!r}. "
            f"Provided value: {request!r}."
        )
    for name, schema, status in (
        ("failure.json", FAILURE_SCHEMA, "failed"),
        ("completion.json", COMPLETION_SCHEMA, "complete"),
        ("running.json", RUNNING_SCHEMA, "running"),
        ("dispatch.json", DISPATCH_SCHEMA, "dispatched"),
    ):
        path = root / name
        if path.exists() or path.is_symlink():
            if path.is_symlink() or not path.is_file():
                raise ValueError(
                    f"Expected attempt {name} to be a real file. "
                    f"Provided value: {path}."
                )
            result = _read_json(path, label=f"attempt {name}")
            try:
                if schema == DISPATCH_SCHEMA:
                    _validate_existing_dispatch(
                        path,
                        request=request,
                        request_path=request_path,
                        profile=profile,
                    )
                else:
                    _validate_attempt_receipt(
                        result,
                        schema=schema,
                        status=status,
                        request=request,
                        request_path=request_path,
                    )
            except DispatchError as exc:
                raise ValueError(str(exc)) from exc
            if schema == RUNNING_SCHEMA:
                worker_pid = result.get("worker_pid")
                worker_starttime = result.get("worker_starttime")
                live = (
                    _proc_snapshot().get(worker_pid)
                    if isinstance(worker_pid, int)
                    and not isinstance(worker_pid, bool)
                    else None
                )
                if (
                    live is None
                    or live[1] == "Z"
                    or live[2] != worker_starttime
                ):
                    return {
                        **result,
                        "status": "stale_running",
                        "terminal": False,
                        "lane_reserved": True,
                        "worker_identity_live": False,
                        "smallest_next_action": (
                            "Inspect the immutable attempt and target "
                            "read-only; do not launch another attempt until "
                            "quiescence is proven."
                        ),
                    }
            return result
    return {
        "status": "initializing",
        "attempt_id": attempt_id,
        "target": profile["target"],
    }


def _proc_snapshot() -> dict[int, tuple[int, str, str]]:
    snapshot: dict[int, tuple[int, str, str]] = {}
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            raw = (entry / "stat").read_text(encoding="utf-8")
            remainder = raw[raw.rfind(")") + 2 :].split()
            snapshot[int(entry.name)] = (
                int(remainder[1]),
                remainder[0],
                remainder[19],
            )
        except (OSError, ValueError, IndexError):
            continue
    return snapshot


def _descendant_identities(root_pid: int) -> dict[int, str]:
    snapshot = _proc_snapshot()
    identities: dict[int, str] = {}
    frontier = [root_pid]
    while frontier:
        parent = frontier.pop()
        children = [
            pid
            for pid, (ppid, _state, _starttime) in snapshot.items()
            if ppid == parent and pid not in identities
        ]
        for child in children:
            identities[child] = snapshot[child][2]
            frontier.append(child)
    return identities


def _signal_descendants(identities: Mapping[int, str], sig: signal.Signals) -> None:
    snapshot = _proc_snapshot()
    for pid, starttime in identities.items():
        current = snapshot.get(pid)
        if current is None or current[1] == "Z" or current[2] != starttime:
            continue
        try:
            os.kill(pid, sig)
        except ProcessLookupError:
            pass


def _set_child_subreaper(enabled: bool) -> bool:
    if sys.platform != "linux":
        raise RuntimeError(
            "Expected Linux PR_SET_CHILD_SUBREAPER support for long-run cleanup."
        )
    libc = ctypes.CDLL(None, use_errno=True)
    current = ctypes.c_int()
    if libc.prctl(37, ctypes.byref(current), 0, 0, 0) != 0:
        errno_value = ctypes.get_errno()
        raise OSError(
            errno_value,
            "PR_GET_CHILD_SUBREAPER failed",
        )
    if libc.prctl(36, int(enabled), 0, 0, 0) != 0:
        errno_value = ctypes.get_errno()
        raise OSError(
            errno_value,
            "PR_SET_CHILD_SUBREAPER failed",
        )
    return bool(current.value)


def _live_identities(identities: Mapping[int, str]) -> dict[int, str]:
    snapshot = _proc_snapshot()
    return {
        pid: starttime
        for pid, starttime in identities.items()
        if pid in snapshot
        and snapshot[pid][1] != "Z"
        and snapshot[pid][2] == starttime
    }


def _reap_children(identities: Mapping[int, str]) -> None:
    for pid in identities:
        try:
            os.waitpid(pid, os.WNOHANG)
        except (ChildProcessError, ProcessLookupError):
            pass


def _terminate_payload(
    process: subprocess.Popen[bytes],
    *,
    identities: Mapping[int, str],
    baseline_descendants: Mapping[int, str],
) -> int:
    descendants = {
        **identities,
        **_descendant_identities(process.pid),
    }

    def discover() -> None:
        for pid, starttime in _descendant_identities(process.pid).items():
            descendants[pid] = starttime
        for pid, starttime in _descendant_identities(os.getpid()).items():
            if baseline_descendants.get(pid) != starttime:
                descendants[pid] = starttime

    discover()
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    term_signaled: dict[int, str] = {}
    term_deadline = time.monotonic() + 2
    returncode = process.poll()
    while returncode is None and time.monotonic() < term_deadline:
        discover()
        pending_term = {
            pid: starttime
            for pid, starttime in descendants.items()
            if term_signaled.get(pid) != starttime
        }
        _signal_descendants(pending_term, signal.SIGTERM)
        term_signaled.update(pending_term)
        try:
            returncode = process.wait(timeout=0.05)
        except subprocess.TimeoutExpired:
            returncode = None
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    discover()
    _signal_descendants(descendants, signal.SIGKILL)
    if returncode is None:
        returncode = process.wait(timeout=10)
    deadline = time.monotonic() + 5
    quiet_since: float | None = None
    while time.monotonic() < deadline:
        discover()
        live = _live_identities(descendants)
        if live:
            _signal_descendants(live, signal.SIGKILL)
            quiet_since = None
        elif quiet_since is None:
            quiet_since = time.monotonic()
        elif time.monotonic() - quiet_since >= 0.25:
            _reap_children(descendants)
            return returncode
        time.sleep(0.05)
    remaining = sorted(_live_identities(descendants))
    raise RuntimeError(
        f"Expected payload descendants to terminate. Provided PIDs: {remaining!r}."
    )


def _wait_payload(
    process: subprocess.Popen[bytes],
    *,
    hard_deadline_seconds: float,
    baseline_descendants: Mapping[int, str],
    stop_signal: Callable[[], int | None] | None = None,
) -> tuple[int, bool, list[int], int | None]:
    deadline = time.monotonic() + hard_deadline_seconds
    tracked: dict[int, str] = {}
    while True:
        observed = _descendant_identities(os.getpid())
        for pid, starttime in observed.items():
            if baseline_descendants.get(pid) != starttime:
                tracked[pid] = starttime
        interrupted_signal = stop_signal() if stop_signal is not None else None
        if interrupted_signal is not None:
            returncode = _terminate_payload(
                process,
                identities=tracked,
                baseline_descendants=baseline_descendants,
            )
            return returncode, False, [], interrupted_signal
        returncode = process.poll()
        if returncode is not None:
            settle_deadline = time.monotonic() + 0.25
            while time.monotonic() < settle_deadline:
                observed = _descendant_identities(os.getpid())
                for pid, starttime in observed.items():
                    if baseline_descendants.get(pid) != starttime:
                        tracked[pid] = starttime
                time.sleep(0.025)
            remaining = sorted(
                pid
                for pid in _live_identities(tracked)
                if pid != process.pid
            )
            if remaining:
                _terminate_payload(
                    process,
                    identities=tracked,
                    baseline_descendants=baseline_descendants,
                )
            _reap_children(tracked)
            return returncode, False, remaining, None
        if time.monotonic() >= deadline:
            returncode = _terminate_payload(
                process,
                identities=tracked,
                baseline_descendants=baseline_descendants,
            )
            return returncode, True, [], None
        time.sleep(min(0.1, max(deadline - time.monotonic(), 0.01)))


def _execute_payload(
    request_path: Path,
    *,
    request_sha256: str,
    profile: Mapping[str, Any],
) -> int:
    request_path = _lexical_absolute(request_path)
    if request_path.is_symlink() or not request_path.is_file():
        raise ValueError(
            f"Expected execution request to be a real file. "
            f"Provided value: {request_path}."
        )
    request_payload = request_path.read_bytes()
    observed_request_sha = _sha256_bytes(request_payload)
    if observed_request_sha != request_sha256:
        raise ValueError(
            f"Expected execution request SHA-256 {request_sha256!r}. "
            f"Provided value: {observed_request_sha!r}."
        )
    raw_request = _strict_json_bytes(
        request_payload,
        label="execution request",
    )
    if not isinstance(raw_request, dict):
        raise ValueError(
            f"Expected execution request to be an object. "
            f"Provided value: {raw_request!r}."
        )
    request = validate_request(raw_request, profile=profile)
    state_root = _lexical_absolute(str(profile["state_root"]))
    expected_path = state_root / request["attempt_id"] / "request.json"
    if request_path != expected_path:
        raise ValueError(
            f"Expected execution request path {expected_path}. "
            f"Provided value: {request_path}."
        )
    attempt_root = request_path.parent
    log_path = attempt_root / "run.log"
    running_path = attempt_root / "running.json"
    quiescence_path = attempt_root / "quiescence.json"
    completion_path = attempt_root / "completion.json"
    failure_path = attempt_root / "failure.json"
    artifacts = {
        "request_path": str(request_path),
        "request_sha256": observed_request_sha,
        "log_path": str(log_path),
    }
    previous_subreaper: bool | None = None
    previous_handlers: dict[signal.Signals, Any] = {}
    received_signal: list[int] = []
    payload_started = False
    payload_quiescent = False
    lingering_descendants: list[int] = []
    interrupted_signal: int | None = None

    def record_signal(signum: int, _frame: Any) -> None:
        if not received_signal:
            received_signal.append(signum)

    try:
        _atomic_create(
            running_path,
            {
                "schema_version": RUNNING_SCHEMA,
                "status": "running",
                "attempt_id": request["attempt_id"],
                "experiment_id": request["experiment_id"],
                "target": request["target"],
                "request_sha256": observed_request_sha,
                "started_at_utc": datetime.now(timezone.utc).isoformat(),
                "worker_pid": os.getpid(),
                "worker_starttime": _proc_snapshot()[os.getpid()][2],
            },
        )
        environment = _runtime_environment(profile, request["env"])
        flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND
        flags |= getattr(os, "O_NOFOLLOW", 0)
        log_fd = os.open(log_path, flags, 0o600)
        with os.fdopen(log_fd, "ab", buffering=0) as log:
            handled_signals = [
                signal.SIGTERM,
                signal.SIGINT,
                signal.SIGHUP,
            ]
            for handled_signal in handled_signals:
                previous_handlers[handled_signal] = signal.getsignal(
                    handled_signal
                )
                signal.signal(handled_signal, record_signal)
            previous_subreaper = _set_child_subreaper(True)
            baseline_descendants = _descendant_identities(os.getpid())
            if received_signal:
                raise InterruptedError(
                    f"worker interrupted by signal {received_signal[0]} "
                    "before payload launch"
                )
            process = subprocess.Popen(
                request["argv"],
                cwd=request["cwd"],
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            payload_started = True
            (
                returncode,
                timed_out,
                lingering_descendants,
                interrupted_signal,
            ) = _wait_payload(
                process,
                hard_deadline_seconds=float(
                    request["hard_deadline_seconds"]
                ),
                baseline_descendants=baseline_descendants,
                stop_signal=(
                    lambda: received_signal[0] if received_signal else None
                ),
            )
            payload_quiescent = True
    except Exception as exc:
        if not payload_started:
            payload_quiescent = True
        if payload_quiescent:
            try:
                _atomic_create(
                    quiescence_path,
                    {
                        "schema_version": QUIESCENCE_SCHEMA,
                        "status": "quiescent",
                        "attempt_id": request["attempt_id"],
                        "experiment_id": request["experiment_id"],
                        "target": request["target"],
                        "request_sha256": observed_request_sha,
                        "live_descendant_count": 0,
                        "worker_pid": os.getpid(),
                        "confirmed_at_utc": datetime.now(
                            timezone.utc
                        ).isoformat(),
                    },
                )
            except OSError:
                pass
        error = DispatchError(
            "payload_execution",
            f"Expected the long simulation worker to start and terminate "
            f"cleanly. Provided error: {exc!r}.",
            launched_job_count=1,
            completed_stages=["request_verified"],
            last_valid_artifacts=artifacts,
        ).bind_request(request)
        _write_failure_once(failure_path, error)
        return 1
    finally:
        for handled_signal, previous_handler in previous_handlers.items():
            try:
                signal.signal(handled_signal, previous_handler)
            except (OSError, ValueError):
                pass
        if previous_subreaper is not None:
            try:
                _set_child_subreaper(previous_subreaper)
            except OSError:
                pass
    try:
        _atomic_create(
            quiescence_path,
            {
                "schema_version": QUIESCENCE_SCHEMA,
                "status": "quiescent",
                "attempt_id": request["attempt_id"],
                "experiment_id": request["experiment_id"],
                "target": request["target"],
                "request_sha256": observed_request_sha,
                "live_descendant_count": 0,
                "worker_pid": os.getpid(),
                "returncode": returncode,
                "timed_out": timed_out,
                "interrupted_signal": interrupted_signal,
                "confirmed_at_utc": datetime.now(timezone.utc).isoformat(),
            },
        )
    except OSError as exc:
        error = DispatchError(
            "quiescence_publication",
            "Expected the worker to publish immutable proof that no payload "
            f"descendants remain. Provided error: {exc!r}.",
            launched_job_count=1,
            completed_stages=["request_verified", "payload_execution"],
            last_valid_artifacts=artifacts,
        ).bind_request(request)
        _write_failure_once(failure_path, error)
        return 1
    terminal = {
        "attempt_id": request["attempt_id"],
        "experiment_id": request["experiment_id"],
        "target": request["target"],
        "request_sha256": observed_request_sha,
        "returncode": returncode,
        "timed_out": timed_out,
        "interrupted_signal": interrupted_signal,
        "lingering_descendants": lingering_descendants,
        "log_path": str(log_path),
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    if (
        returncode == 0
        and not timed_out
        and interrupted_signal is None
        and not lingering_descendants
    ):
        try:
            _atomic_create(
                completion_path,
                {
                    "schema_version": COMPLETION_SCHEMA,
                    "status": "complete",
                    **terminal,
                },
            )
        except OSError as exc:
            error = DispatchError(
                "completion_publication",
                f"Expected immutable completion publication. "
                f"Provided error: {exc!r}.",
                launched_job_count=1,
                completed_stages=["request_verified", "payload_execution"],
                last_valid_artifacts=artifacts,
            ).bind_request(request)
            _write_failure_once(failure_path, error)
            return 1
        return 0
    error = DispatchError(
        "payload_execution",
        f"Expected long simulation to exit zero before its hard deadline. "
        f"Provided returncode={returncode}, timed_out={timed_out}, "
        f"interrupted_signal={interrupted_signal}, "
        f"lingering_descendants={lingering_descendants!r}.",
        launched_job_count=1,
        completed_stages=["request_verified"],
        last_valid_artifacts=artifacts,
    ).bind_request(request)
    _write_failure_once(failure_path, error)
    if interrupted_signal is not None:
        return 128 + interrupted_signal
    return returncode if returncode != 0 else 124


def _default_profile_path(target: str) -> Path:
    return (
        Path(__file__).resolve().parents[1]
        / "configs"
        / "dispatch"
        / f"{target}.json"
    )


def _parse_env(values: Sequence[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for raw in values:
        if "=" not in raw:
            raise ValueError(
                f"Expected --env NAME=VALUE. Provided value: {raw!r}."
            )
        key, value = raw.split("=", 1)
        if ENV_RE.fullmatch(key) is None:
            raise ValueError(
                f"Expected a safe environment name. Provided value: {key!r}."
            )
        result[key] = value
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("build-preflight", "verify-preflight"):
        child = subparsers.add_parser(name)
        child.add_argument("--target", choices=("main", "trex"), required=True)
        child.add_argument("--profile")
        child.add_argument("--experiment-id", required=True)
        child.add_argument("--cwd", required=True)
        child.add_argument("--source-id", required=True)
        child.add_argument("--approved-plan", required=True)
        child.add_argument("--env", action="append", default=[])
        if name == "build-preflight":
            child.add_argument("--smoke-receipt", required=True)
            child.add_argument("--tk-reference-receipt")
            child.add_argument("--scheduled-run-preflight-receipt")
        child.add_argument("argv", nargs=argparse.REMAINDER)
    for name in ("plan", "start"):
        child = subparsers.add_parser(name)
        child.add_argument("--target", choices=("main", "trex"), required=True)
        child.add_argument("--profile")
        child.add_argument("--attempt-id", required=True)
        child.add_argument("--experiment-id", required=True)
        child.add_argument("--cwd", required=True)
        child.add_argument("--source-id", required=True)
        child.add_argument("--approved-plan", required=True)
        child.add_argument("--tracker-gate-receipt", required=True)
        child.add_argument(
            "--expected-duration-seconds",
            type=float,
            required=True,
        )
        child.add_argument(
            "--hard-deadline-seconds",
            type=float,
            required=True,
        )
        child.add_argument("--env", action="append", default=[])
        child.add_argument("argv", nargs=argparse.REMAINDER)
    status = subparsers.add_parser("status")
    status.add_argument("--target", choices=("main", "trex"), required=True)
    status.add_argument("--profile")
    status.add_argument("--attempt-id", required=True)
    remote = subparsers.add_parser("remote-dispatch", help=argparse.SUPPRESS)
    remote.add_argument("--profile", required=True)
    remote_status = subparsers.add_parser(
        "remote-status",
        help=argparse.SUPPRESS,
    )
    remote_status.add_argument("--profile", required=True)
    remote_status.add_argument("--attempt-id", required=True)
    execute = subparsers.add_parser("execute", help=argparse.SUPPRESS)
    execute.add_argument("--profile", required=True)
    execute.add_argument("--request", required=True)
    execute.add_argument("--request-sha256", required=True)
    return parser


def _profile_for_args(args: argparse.Namespace) -> dict[str, Any]:
    target = getattr(args, "target", None)
    path = (
        Path(args.profile)
        if args.profile
        else _default_profile_path(str(target))
    )
    profile = load_profile(path)
    if target is not None and profile["target"] != target:
        raise ValueError(
            f"Expected profile target {target!r}. "
            f"Provided value: {profile['target']!r}."
        )
    return profile


def _request_from_args(
    args: argparse.Namespace,
    *,
    profile: Mapping[str, Any],
) -> dict[str, Any]:
    argv = list(args.argv)
    if argv and argv[0] == "--":
        argv = argv[1:]
    return build_request(
        profile=profile,
        attempt_id=args.attempt_id,
        experiment_id=args.experiment_id,
        cwd=args.cwd,
        argv=argv,
        env=_parse_env(args.env),
        source_id=args.source_id,
        expected_duration_seconds=args.expected_duration_seconds,
        hard_deadline_seconds=args.hard_deadline_seconds,
        approved_plan_path=args.approved_plan,
        tracker_gate_receipt_path=args.tracker_gate_receipt,
    )


def main(
    argv: Sequence[str] | None = None,
    *,
    runner: Runner = subprocess.run,
) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "remote-dispatch":
            profile = load_profile(args.profile)
            request = _strict_json_bytes(
                sys.stdin.buffer.read(),
                label="remote dispatch request",
            )
            if not isinstance(request, dict):
                raise ValueError(
                    f"Expected remote request to be an object. "
                    f"Provided value: {request!r}."
                )
            result = dispatch_on_target(
                request,
                profile=profile,
                runner=runner,
            )
        elif args.command == "remote-status":
            profile = load_profile(args.profile)
            result = status_on_target(
                profile=profile,
                attempt_id=args.attempt_id,
            )
        elif args.command == "execute":
            profile = load_profile(args.profile)
            return _execute_payload(
                Path(args.request).expanduser().resolve(),
                request_sha256=args.request_sha256,
                profile=profile,
            )
        else:
            profile = _profile_for_args(args)
            if args.command == "status":
                result = status_request(
                    profile=profile,
                    attempt_id=args.attempt_id,
                    runner=runner,
                )
            elif args.command in {"build-preflight", "verify-preflight"}:
                command_argv = list(args.argv)
                if command_argv and command_argv[0] == "--":
                    command_argv = command_argv[1:]
                common = {
                    "profile": profile,
                    "experiment_id": args.experiment_id,
                    "cwd": args.cwd,
                    "argv": command_argv,
                    "env": _parse_env(args.env),
                    "source_id": args.source_id,
                    "approved_plan_path": args.approved_plan,
                }
                if args.command == "build-preflight":
                    result = build_preflight(
                        **common,
                        smoke_receipt_path=args.smoke_receipt,
                        tk_reference_receipt_path=(
                            args.tk_reference_receipt
                        ),
                        scheduled_run_preflight_receipt_path=(
                            args.scheduled_run_preflight_receipt
                        ),
                    )
                else:
                    result = verify_preflight(**common)
            else:
                request = _request_from_args(args, profile=profile)
                if args.command == "plan":
                    result = {
                        "status": "planned",
                        "side_effects": 0,
                        "request": request,
                        "request_sha256": _sha256_bytes(
                            _canonical_bytes(request) + b"\n"
                        ),
                    }
                else:
                    print(
                        "LONG-RUN ATTEMPT ARMED "
                        f"attempt_id={request['attempt_id']} "
                        f"experiment_id={request['experiment_id']} "
                        f"target={request['target']} "
                        f"expected_seconds="
                        f"{request['expected_duration_seconds']:g} "
                        f"deadline_seconds="
                        f"{request['hard_deadline_seconds']:g} "
                        f"request_path="
                        f"{profile['state_root']}/{request['attempt_id']}/request.json "
                        f"dispatch_path="
                        f"{profile['state_root']}/{request['attempt_id']}/dispatch.json "
                        f"failure_path="
                        f"{profile['state_root']}/{request['attempt_id']}/failure.json",
                        file=sys.stderr,
                        flush=True,
                    )
                    result = start_request(
                        request,
                        profile=profile,
                        runner=runner,
                    )
        print(
            json.dumps(result, sort_keys=True, allow_nan=False),
            flush=True,
        )
        return 0
    except DispatchError as exc:
        destination = (
            sys.stdout
            if args.command == "remote-dispatch"
            else sys.stderr
        )
        print(
            json.dumps(exc.as_dict(), sort_keys=True, allow_nan=False),
            file=destination,
            flush=True,
        )
        return 2
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(str(exc), file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
