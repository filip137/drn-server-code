#!/usr/bin/env python3
"""Capability-based planning for resolved experiments.

This module deliberately has no launch side effects.  It turns a resolved
scientific study and an execution proposal into deterministic, target-specific
argv plans, and it evaluates already-observed host facts.  An
experiment-specific adapter remains responsible for the numerical one-batch
validation and production job.

Scientific job ownership is strict.  Runtime implementation details are not:
Python, PyTorch, CUDA, and GPU descriptions are recorded as provenance and are
accepted whenever the declared functional capabilities pass.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path, PurePosixPath
import re
import shlex
import sys
from typing import Any, Iterable, Mapping, Sequence


RESOLVED_STUDY_SCHEMA = "resolved-experiment-study/v1"
EXECUTION_PROPOSAL_SCHEMA = "experiment-execution-proposal/v1"
TARGET_PROFILE_SCHEMA = "functional-target-profile/v1"
HOST_FACTS_SCHEMA = "functional-host-facts/v1"
EXECUTION_PLAN_SCHEMA = "experiment-execution-plan/v1"
PREFLIGHT_RESULT_SCHEMA = "functional-preflight-result/v1"
JOB_RESULT_SCHEMA = "experiment-job-result/v1"
ATTEMPT_RESULT_SCHEMA = "experiment-attempt-result/v1"

RUN_CLASSES = {"validation", "production"}
TARGET_KINDS = {"local_tmux", "ssh_tmux", "ssh_process"}
TRANSPORT_KINDS = {"local", "ssh"}
SMOKE_STATES = {"not_run", "passed", "failed"}
TK_STATES = {"not_required", "reused", "passed", "failed", "not_run"}
ID_RE = re.compile(r"^[a-z0-9][a-z0-9._-]*$")
PLACEHOLDER_RE = re.compile(r"\{([a-z_][a-z0-9_]*|job\.[a-zA-Z0-9_.-]+)\}")


class ExecutorError(ValueError):
    """Structured, pre-arm error with expected-first wording."""

    def __init__(
        self,
        *,
        code: str,
        path: str,
        expected: str,
        provided: Any,
    ) -> None:
        self.code = code
        self.path = path
        self.expected = expected
        self.provided = provided
        super().__init__(
            f"Expected {expected} at {path}; provided value: {provided!r}"
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "experiment-executor-error/v1",
            "status": "failed",
            "stage": "execution_planning",
            "error_code": self.code,
            "path": self.path,
            "expected": self.expected,
            "provided": self.provided,
            "launched_job_count": 0,
        }


def _fail(
    *,
    code: str,
    path: str,
    expected: str,
    provided: Any,
) -> None:
    raise ExecutorError(
        code=code,
        path=path,
        expected=expected,
        provided=provided,
    )


def _require(
    condition: bool,
    *,
    code: str,
    path: str,
    expected: str,
    provided: Any,
) -> None:
    if not condition:
        _fail(
            code=code,
            path=path,
            expected=expected,
            provided=provided,
        )


def _mapping(value: Any, *, path: str) -> dict[str, Any]:
    _require(
        isinstance(value, dict),
        code="wrong_type",
        path=path,
        expected="an object",
        provided=type(value).__name__,
    )
    return value


def _text(value: Any, *, path: str) -> str:
    _require(
        isinstance(value, str) and bool(value.strip()),
        code="empty_text",
        path=path,
        expected="non-empty text",
        provided=value,
    )
    return value


def _identifier(value: Any, *, path: str) -> str:
    text = _text(value, path=path)
    _require(
        ID_RE.fullmatch(text) is not None and text not in {".", ".."},
        code="unsafe_identifier",
        path=path,
        expected="a lowercase filesystem-safe identifier",
        provided=text,
    )
    return text


def _positive_int(value: Any, *, path: str) -> int:
    _require(
        isinstance(value, int) and not isinstance(value, bool) and value > 0,
        code="invalid_integer",
        path=path,
        expected="a positive integer",
        provided=value,
    )
    return value


def _nonnegative_int(value: Any, *, path: str) -> int:
    _require(
        isinstance(value, int) and not isinstance(value, bool) and value >= 0,
        code="invalid_integer",
        path=path,
        expected="a non-negative integer",
        provided=value,
    )
    return value


def _bool(value: Any, *, path: str) -> bool:
    _require(
        isinstance(value, bool),
        code="wrong_type",
        path=path,
        expected="a boolean",
        provided=value,
    )
    return value


def _string_list(value: Any, *, path: str, nonempty: bool = True) -> list[str]:
    _require(
        isinstance(value, list)
        and (bool(value) or not nonempty)
        and all(isinstance(item, str) and bool(item) for item in value),
        code="invalid_string_list",
        path=path,
        expected=(
            "a non-empty list of non-empty strings"
            if nonempty
            else "a list of non-empty strings"
        ),
        provided=value,
    )
    return list(value)


def _absolute_posix_path(value: Any, *, path: str) -> str:
    text = _text(value, path=path)
    candidate = PurePosixPath(text)
    _require(
        candidate.is_absolute()
        and ".." not in candidate.parts
        and "\x00" not in text,
        code="unsafe_path",
        path=path,
        expected="an absolute normalized POSIX path without '..'",
        provided=text,
    )
    return str(candidate)


def _load_json(path: str | Path, *, label: str) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    _require(
        resolved.is_file() and not resolved.is_symlink(),
        code="file_missing",
        path=label,
        expected="an existing non-symlink JSON file",
        provided=str(resolved),
    )

    def reject_constant(value: str) -> None:
        _fail(
            code="nonfinite_json",
            path=label,
            expected="strict JSON without NaN or Infinity",
            provided=value,
        )

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, item in pairs:
            if key in result:
                _fail(
                    code="duplicate_json_key",
                    path=label,
                    expected="unique JSON object keys",
                    provided=key,
                )
            result[key] = item
        return result

    try:
        value = json.loads(
            resolved.read_text(encoding="utf-8"),
            parse_constant=reject_constant,
            object_pairs_hook=unique_object,
        )
    except ExecutorError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        _fail(
            code="invalid_json",
            path=label,
            expected="valid UTF-8 JSON",
            provided=str(error),
        )
    return _mapping(value, path=label)


def load_target_profile(path: str | Path) -> dict[str, Any]:
    """Load and validate one functional target profile."""

    return validate_target_profile(
        _load_json(path, label="target_profile")
    )


def load_document(path: str | Path, *, label: str = "document") -> dict[str, Any]:
    """Load strict JSON for the public CLI and integrations."""

    return _load_json(path, label=label)


def _schema(value: Mapping[str, Any], expected: str, *, path: str) -> None:
    _require(
        value.get("schema_version") == expected,
        code="schema_mismatch",
        path=f"{path}.schema_version",
        expected=expected,
        provided=value.get("schema_version"),
    )


def validate_target_profile(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate a target profile without pinning runtime versions."""

    profile = deepcopy(_mapping(value, path="target_profile"))
    _schema(profile, TARGET_PROFILE_SCHEMA, path="target_profile")
    target_id = _identifier(
        profile.get("target_id"),
        path="target_profile.target_id",
    )
    target_kind = profile.get("target_kind")
    _require(
        target_kind in TARGET_KINDS,
        code="unsupported_target_kind",
        path="target_profile.target_kind",
        expected=f"one of {sorted(TARGET_KINDS)}",
        provided=target_kind,
    )
    _text(profile.get("host"), path="target_profile.host")

    transport = _mapping(
        profile.get("transport"),
        path="target_profile.transport",
    )
    kind = transport.get("kind")
    _require(
        kind in TRANSPORT_KINDS,
        code="unsupported_transport",
        path="target_profile.transport.kind",
        expected=f"one of {sorted(TRANSPORT_KINDS)}",
        provided=kind,
    )
    expected_kind = "local" if target_kind == "local_tmux" else "ssh"
    _require(
        kind == expected_kind,
        code="transport_target_mismatch",
        path="target_profile.transport.kind",
        expected=expected_kind,
        provided=kind,
    )
    ssh_target = transport.get("ssh_target")
    ssh_config = transport.get("ssh_config")
    if kind == "local":
        _require(
            ssh_target is None,
            code="unexpected_ssh_target",
            path="target_profile.transport.ssh_target",
            expected="null for a local transport",
            provided=ssh_target,
        )
        _require(
            ssh_config is None,
            code="unexpected_ssh_config",
            path="target_profile.transport.ssh_config",
            expected="null for a local transport",
            provided=ssh_config,
        )
    else:
        _text(
            ssh_target,
            path="target_profile.transport.ssh_target",
        )
        if ssh_config is not None:
            _absolute_posix_path(
                ssh_config,
                path="target_profile.transport.ssh_config",
            )

    _absolute_posix_path(
        profile.get("workspace_root"),
        path="target_profile.workspace_root",
    )
    _absolute_posix_path(
        profile.get("result_root"),
        path="target_profile.result_root",
    )
    _absolute_posix_path(
        profile.get("data_root"),
        path="target_profile.data_root",
    )
    _text(profile.get("device"), path="target_profile.device")
    _string_list(
        profile.get("python_candidates"),
        path="target_profile.python_candidates",
    )
    imports = _string_list(
        profile.get("required_imports"),
        path="target_profile.required_imports",
        nonempty=False,
    )
    _require(
        len(imports) == len(set(imports)),
        code="duplicate_required_import",
        path="target_profile.required_imports",
        expected="unique import names",
        provided=imports,
    )
    requires_cuda = _bool(
        profile.get("requires_cuda"),
        path="target_profile.requires_cuda",
    )
    minimum_gpus = profile.get("minimum_gpu_count")
    if requires_cuda:
        _positive_int(
            minimum_gpus,
            path="target_profile.minimum_gpu_count",
        )
    else:
        _require(
            minimum_gpus == 0,
            code="unexpected_gpu_requirement",
            path="target_profile.minimum_gpu_count",
            expected="zero when CUDA is not required",
            provided=minimum_gpus,
        )
    _positive_int(
        profile.get("default_concurrency"),
        path="target_profile.default_concurrency",
    )
    prefix = _text(
        profile.get("tmux_session_prefix"),
        path="target_profile.tmux_session_prefix",
    )
    _require(
        re.fullmatch(r"[a-z0-9][a-z0-9-]*", prefix) is not None,
        code="unsafe_tmux_prefix",
        path="target_profile.tmux_session_prefix",
        expected="lowercase letters, digits, and hyphens",
        provided=prefix,
    )
    return profile


def validate_resolved_study(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the small executor-facing projection of a resolved study."""

    study = deepcopy(_mapping(value, path="resolved_study"))
    _schema(study, RESOLVED_STUDY_SCHEMA, path="resolved_study")
    _identifier(study.get("study_id"), path="resolved_study.study_id")
    jobs = study.get("jobs")
    _require(
        isinstance(jobs, list) and bool(jobs),
        code="missing_jobs",
        path="resolved_study.jobs",
        expected="a non-empty ordered job list",
        provided=jobs,
    )
    job_ids: list[str] = []
    for index, raw_job in enumerate(jobs):
        job = _mapping(raw_job, path=f"resolved_study.jobs[{index}]")
        job_id = _identifier(
            job.get("job_id"),
            path=f"resolved_study.jobs[{index}].job_id",
        )
        job_ids.append(job_id)
    duplicates = sorted(
        job_id for job_id in set(job_ids) if job_ids.count(job_id) > 1
    )
    _require(
        not duplicates,
        code="duplicate_job_id",
        path="resolved_study.jobs",
        expected="globally unique job IDs",
        provided=duplicates,
    )
    return study


def validate_execution_proposal(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate operational choices without treating them as science."""

    proposal = deepcopy(_mapping(value, path="execution_proposal"))
    _schema(proposal, EXECUTION_PROPOSAL_SCHEMA, path="execution_proposal")
    _identifier(
        proposal.get("proposal_id"),
        path="execution_proposal.proposal_id",
    )
    run_class = proposal.get("run_class")
    _require(
        run_class in RUN_CLASSES,
        code="invalid_run_class",
        path="execution_proposal.run_class",
        expected=f"one of {sorted(RUN_CLASSES)}",
        provided=run_class,
    )
    _bool(
        proposal.get("launch_after_checks"),
        path="execution_proposal.launch_after_checks",
    )

    runner = _mapping(
        proposal.get("runner"),
        path="execution_proposal.runner",
    )
    _identifier(
        runner.get("adapter_id"),
        path="execution_proposal.runner.adapter_id",
    )
    argv_template = _string_list(
        runner.get("argv_template"),
        path="execution_proposal.runner.argv_template",
    )
    placeholders = {
        match.group(1)
        for token in argv_template
        for match in PLACEHOLDER_RE.finditer(token)
    }
    unknown_placeholders = sorted(
        placeholder
        for placeholder in placeholders
        if placeholder
        not in {
            "data_root",
            "device",
            "python",
            "job_id",
            "run_class",
            "output_dir",
            "result_root",
            "study_id",
            "proposal_id",
            "target_id",
            "workspace_root",
        }
        and not placeholder.startswith("job.")
    )
    _require(
        not unknown_placeholders,
        code="unknown_argv_placeholder",
        path="execution_proposal.runner.argv_template",
        expected="documented executor placeholders",
        provided=unknown_placeholders,
    )
    required_placeholders = {"python", "job_id", "run_class", "output_dir"}
    _require(
        required_placeholders <= placeholders,
        code="missing_argv_placeholder",
        path="execution_proposal.runner.argv_template",
        expected=f"placeholders {sorted(required_placeholders)}",
        provided=sorted(placeholders),
    )

    preflight = _mapping(
        proposal.get("preflight"),
        path="execution_proposal.preflight",
    )
    _bool(
        preflight.get("functional_smoke_required"),
        path="execution_proposal.preflight.functional_smoke_required",
    )
    _bool(
        preflight.get("fresh_tk_gate_required"),
        path="execution_proposal.preflight.fresh_tk_gate_required",
    )

    targets = proposal.get("targets")
    _require(
        isinstance(targets, list) and bool(targets),
        code="missing_targets",
        path="execution_proposal.targets",
        expected="a non-empty target assignment list",
        provided=targets,
    )
    target_ids: list[str] = []
    for index, raw_target in enumerate(targets):
        path = f"execution_proposal.targets[{index}]"
        target = _mapping(raw_target, path=path)
        target_id = _identifier(
            target.get("target_id"),
            path=f"{path}.target_id",
        )
        target_ids.append(target_id)
        job_ids = _string_list(
            target.get("job_ids"),
            path=f"{path}.job_ids",
        )
        for job_index, job_id in enumerate(job_ids):
            _identifier(job_id, path=f"{path}.job_ids[{job_index}]")
        _require(
            len(job_ids) == len(set(job_ids)),
            code="duplicate_job_in_target",
            path=f"{path}.job_ids",
            expected="unique job IDs within a target",
            provided=job_ids,
        )
        concurrency = target.get("concurrency")
        _require(
            concurrency == "auto"
            or (
                isinstance(concurrency, int)
                and not isinstance(concurrency, bool)
                and concurrency > 0
            ),
            code="invalid_concurrency",
            path=f"{path}.concurrency",
            expected="'auto' or a positive integer",
            provided=concurrency,
        )
        validation_job_id = target.get("validation_job_id")
        if validation_job_id is not None:
            _identifier(
                validation_job_id,
                path=f"{path}.validation_job_id",
            )
            _require(
                validation_job_id in job_ids,
                code="validation_job_not_owned",
                path=f"{path}.validation_job_id",
                expected="one of the target's owned job IDs",
                provided=validation_job_id,
            )
    _require(
        len(target_ids) == len(set(target_ids)),
        code="duplicate_target",
        path="execution_proposal.targets",
        expected="unique target IDs",
        provided=target_ids,
    )
    return proposal


def _normalise_profiles(
    profiles: Mapping[str, Mapping[str, Any]]
    | Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    if isinstance(profiles, Mapping):
        items: Iterable[tuple[str | None, Mapping[str, Any]]] = (
            (key, value) for key, value in profiles.items()
        )
    else:
        items = ((None, value) for value in profiles)
    result: dict[str, dict[str, Any]] = {}
    for supplied_key, raw_profile in items:
        profile = validate_target_profile(raw_profile)
        target_id = profile["target_id"]
        if supplied_key is not None:
            _require(
                supplied_key == target_id,
                code="profile_key_mismatch",
                path=f"target_profiles.{supplied_key}",
                expected=target_id,
                provided=supplied_key,
            )
        _require(
            target_id not in result,
            code="duplicate_target_profile",
            path="target_profiles",
            expected="one profile per target",
            provided=target_id,
        )
        result[target_id] = profile
    return result


def _normalise_facts(
    facts: Mapping[str, Mapping[str, Any]]
    | Sequence[Mapping[str, Any]]
    | None,
) -> dict[str, dict[str, Any]]:
    if facts is None:
        return {}
    if isinstance(facts, Mapping):
        items: Iterable[tuple[str | None, Mapping[str, Any]]] = (
            (key, value) for key, value in facts.items()
        )
    else:
        items = ((None, value) for value in facts)
    result: dict[str, dict[str, Any]] = {}
    for supplied_key, raw_fact in items:
        fact = deepcopy(_mapping(raw_fact, path="host_facts"))
        _schema(fact, HOST_FACTS_SCHEMA, path="host_facts")
        target_id = _identifier(
            fact.get("target_id"),
            path="host_facts.target_id",
        )
        if supplied_key is not None:
            _require(
                supplied_key == target_id,
                code="host_facts_key_mismatch",
                path=f"host_facts.{supplied_key}",
                expected=target_id,
                provided=supplied_key,
            )
        _require(
            target_id not in result,
            code="duplicate_host_facts",
            path="host_facts",
            expected="one observation per target",
            provided=target_id,
        )
        result[target_id] = fact
    return result


def _verify_job_ownership(
    study: Mapping[str, Any],
    proposal: Mapping[str, Any],
) -> None:
    expected = [job["job_id"] for job in study["jobs"]]
    owners: dict[str, list[str]] = {}
    for target in proposal["targets"]:
        for job_id in target["job_ids"]:
            owners.setdefault(job_id, []).append(target["target_id"])
    duplicate = {
        job_id: targets
        for job_id, targets in owners.items()
        if len(targets) > 1
    }
    expected_set = set(expected)
    missing = [job_id for job_id in expected if job_id not in owners]
    unknown = [
        job_id for job_id in owners if job_id not in expected_set
    ]
    _require(
        not duplicate and not missing and not unknown,
        code="job_ownership_mismatch",
        path="execution_proposal.targets",
        expected="every resolved study job owned by exactly one target",
        provided={
            "missing": missing,
            "unknown": unknown,
            "duplicate": duplicate,
        },
    )


def _lookup_job_value(job: Mapping[str, Any], path: str) -> Any:
    value: Any = job
    for part in path.split("."):
        if not isinstance(value, Mapping) or part not in value:
            _fail(
                code="missing_job_placeholder_value",
                path=f"resolved_study.jobs[{job.get('job_id')}].{path}",
                expected="a value referenced by the runner argv template",
                provided=None,
            )
        value = value[part]
    _require(
        isinstance(value, (str, int, float))
        and not isinstance(value, bool),
        code="nonscalar_job_placeholder_value",
        path=f"resolved_study.jobs[{job.get('job_id')}].{path}",
        expected="a string or numeric scalar",
        provided=value,
    )
    return value


def _expand_argv(
    template: Sequence[str],
    *,
    common: Mapping[str, str],
    job: Mapping[str, Any],
) -> list[str]:
    def replace(match: re.Match[str]) -> str:
        key = match.group(1)
        if key.startswith("job."):
            return str(_lookup_job_value(job, key[4:]))
        if key not in common:
            _fail(
                code="unknown_argv_placeholder",
                path="execution_proposal.runner.argv_template",
                expected="a documented executor placeholder",
                provided=key,
            )
        return common[key]

    result = [PLACEHOLDER_RE.sub(replace, token) for token in template]
    unresolved = [
        token for token in result if "{" in token or "}" in token
    ]
    _require(
        not unresolved,
        code="unresolved_argv_placeholder",
        path="execution_proposal.runner.argv_template",
        expected="fully resolved argv tokens",
        provided=unresolved,
    )
    return result


def _safe_tmux_component(value: str, *, maximum: int = 50) -> str:
    component = re.sub(r"[^a-zA-Z0-9_-]", "-", value)
    component = re.sub(r"-+", "-", component).strip("-")
    return (component or "job")[:maximum]


def _remote_wrap(ssh_target: str, argv: Sequence[str]) -> list[str]:
    return [
        "ssh",
        "-o",
        "BatchMode=yes",
        ssh_target,
        shlex.join(argv),
    ]


def _transport_argv(
    profile: Mapping[str, Any],
    argv: Sequence[str],
) -> list[str]:
    if profile["transport"]["kind"] == "local":
        return list(argv)
    return _remote_wrap(profile["transport"]["ssh_target"], argv)


def _selected_python(
    profile: Mapping[str, Any],
    facts: Mapping[str, Any] | None,
) -> str:
    if facts is not None:
        python = facts.get("python")
        if isinstance(python, Mapping):
            executable = python.get("executable")
            if isinstance(executable, str) and executable:
                return executable
    return profile["python_candidates"][0]


def _effective_concurrency(
    *,
    requested: int | str,
    profile: Mapping[str, Any],
    facts: Mapping[str, Any] | None,
    job_count: int,
    run_class: str,
) -> tuple[int, list[str]]:
    if run_class == "validation":
        return 1, []
    safe = None
    if facts is not None:
        raw_safe = facts.get("max_safe_concurrency")
        if raw_safe is not None:
            safe = _positive_int(
                raw_safe,
                path=f"host_facts.{profile['target_id']}.max_safe_concurrency",
            )
    if requested == "auto":
        value = safe or profile["default_concurrency"]
    else:
        value = requested
    warnings: list[str] = []
    if safe is not None and value > safe:
        warnings.append(
            f"reduced concurrency from {value} to observed safe limit {safe}"
        )
        value = safe
    return min(value, job_count), warnings


def _batches(job_ids: Sequence[str], concurrency: int) -> list[list[str]]:
    return [
        list(job_ids[index : index + concurrency])
        for index in range(0, len(job_ids), concurrency)
    ]


def build_execution_plan(
    resolved_study: Mapping[str, Any],
    execution_proposal: Mapping[str, Any],
    *,
    target_profiles: Mapping[str, Mapping[str, Any]]
    | Sequence[Mapping[str, Any]],
    host_facts: Mapping[str, Mapping[str, Any]]
    | Sequence[Mapping[str, Any]]
    | None = None,
) -> dict[str, Any]:
    """Build a deterministic, non-executing plan.

    ``validation`` plans one representative job per target.  The adapter is
    expected to interpret that class as one real numerical batch and optimizer
    step.  ``production`` plans every owned job.
    """

    study = validate_resolved_study(resolved_study)
    proposal = validate_execution_proposal(execution_proposal)
    profiles = _normalise_profiles(target_profiles)
    facts_by_target = _normalise_facts(host_facts)
    _verify_job_ownership(study, proposal)

    requested_target_ids = [
        target["target_id"] for target in proposal["targets"]
    ]
    missing_profiles = [
        target_id
        for target_id in requested_target_ids
        if target_id not in profiles
    ]
    unknown_facts = [
        target_id
        for target_id in facts_by_target
        if target_id not in requested_target_ids
    ]
    _require(
        not missing_profiles,
        code="target_profile_missing",
        path="target_profiles",
        expected="one profile for every proposed target",
        provided=missing_profiles,
    )
    _require(
        not unknown_facts,
        code="unexpected_host_facts",
        path="host_facts",
        expected="facts only for proposed targets",
        provided=unknown_facts,
    )

    jobs_by_id = {
        job["job_id"]: job for job in study["jobs"]
    }
    attempts: list[dict[str, Any]] = []
    total_planned_jobs = 0
    for target in proposal["targets"]:
        target_id = target["target_id"]
        profile = profiles[target_id]
        facts = facts_by_target.get(target_id)
        owned_job_ids = list(target["job_ids"])
        if proposal["run_class"] == "validation":
            validation_job = target.get("validation_job_id")
            planned_job_ids = [validation_job or owned_job_ids[0]]
        else:
            planned_job_ids = owned_job_ids
        concurrency, warnings = _effective_concurrency(
            requested=target["concurrency"],
            profile=profile,
            facts=facts,
            job_count=len(planned_job_ids),
            run_class=proposal["run_class"],
        )
        total_planned_jobs += len(planned_job_ids)

        attempt_id = (
            f"{proposal['proposal_id']}-{target_id}-"
            f"{proposal['run_class']}"
        )
        session = (
            profile["tmux_session_prefix"]
            + _safe_tmux_component(attempt_id, maximum=48)
        )[:80]
        init_tmux = [
            "tmux",
            "new-session",
            "-d",
            "-s",
            session,
            "-n",
            "control",
            "-c",
            profile["workspace_root"],
        ]
        initialize_argv = _transport_argv(profile, init_tmux)
        python = _selected_python(profile, facts)
        result_root = (
            PurePosixPath(profile["result_root"])
            / study["study_id"]
            / proposal["proposal_id"]
            / target_id
        )
        planned_jobs: list[dict[str, Any]] = []
        for index, job_id in enumerate(planned_job_ids):
            job = jobs_by_id[job_id]
            output_dir = result_root / "jobs" / job_id
            common = {
                "data_root": profile["data_root"],
                "device": profile["device"],
                "python": python,
                "job_id": job_id,
                "run_class": proposal["run_class"],
                "output_dir": str(output_dir),
                "result_root": profile["result_root"],
                "study_id": study["study_id"],
                "proposal_id": proposal["proposal_id"],
                "target_id": target_id,
                "workspace_root": profile["workspace_root"],
            }
            worker_argv = _expand_argv(
                proposal["runner"]["argv_template"],
                common=common,
                job=job,
            )
            window_name = (
                f"{index:02d}-"
                f"{_safe_tmux_component(job_id, maximum=28)}"
            )
            tmux_argv = [
                "tmux",
                "new-window",
                "-d",
                "-t",
                session,
                "-n",
                window_name,
                "-c",
                profile["workspace_root"],
                *worker_argv,
            ]
            planned_jobs.append(
                {
                    "job_id": job_id,
                    "output_dir": str(output_dir),
                    "worker_argv": worker_argv,
                    "dispatch_argv": _transport_argv(profile, tmux_argv),
                }
            )

        provenance: dict[str, Any] | None = None
        if facts is not None:
            provenance = {
                "observed_at": facts.get("observed_at"),
                "python": deepcopy(facts.get("python")),
                "imports": deepcopy(facts.get("imports")),
                "cuda": deepcopy(facts.get("cuda")),
            }
        attempts.append(
            {
                "attempt_id": attempt_id,
                "target_id": target_id,
                "target_kind": profile["target_kind"],
                "host": profile["host"],
                "transport": deepcopy(profile["transport"]),
                "owned_job_ids": owned_job_ids,
                "planned_job_ids": planned_job_ids,
                "requested_concurrency": target["concurrency"],
                "effective_concurrency": concurrency,
                "launch_batches": _batches(planned_job_ids, concurrency),
                "workspace_root": profile["workspace_root"],
                "result_root": str(result_root),
                "tmux_session": session,
                "initialize_argv": initialize_argv,
                "jobs": planned_jobs,
                "capability_provenance": provenance,
                "warnings": warnings,
            }
        )

    return {
        "schema_version": EXECUTION_PLAN_SCHEMA,
        "status": "planned",
        "side_effects_performed": False,
        "study_id": study["study_id"],
        "proposal_id": proposal["proposal_id"],
        "run_class": proposal["run_class"],
        "resolved_job_count": len(study["jobs"]),
        "planned_job_count": total_planned_jobs,
        "target_count": len(attempts),
        "launch_after_checks": proposal["launch_after_checks"],
        "runner_adapter_id": proposal["runner"]["adapter_id"],
        "preflight_policy": deepcopy(proposal["preflight"]),
        "attempts": attempts,
        "result_schemas": {
            "preflight": PREFLIGHT_RESULT_SCHEMA,
            "job": JOB_RESULT_SCHEMA,
            "attempt": ATTEMPT_RESULT_SCHEMA,
        },
    }


plan_attempts = build_execution_plan


def _check(
    checks: list[dict[str, Any]],
    *,
    role: str,
    passed: bool,
    observed: Any,
    required: Any,
    blocking: bool = True,
) -> None:
    checks.append(
        {
            "role": role,
            "status": "passed" if passed else ("failed" if blocking else "planned"),
            "blocking": blocking and not passed,
            "required": required,
            "observed": observed,
        }
    )


def _evaluate_target_capabilities(
    *,
    attempt: Mapping[str, Any],
    profile: Mapping[str, Any],
    facts: Mapping[str, Any],
    run_class: str,
    preflight_policy: Mapping[str, Any],
) -> dict[str, Any]:
    path = f"host_facts.{attempt['target_id']}"
    _schema(facts, HOST_FACTS_SCHEMA, path=path)
    _require(
        facts.get("target_id") == attempt["target_id"],
        code="host_facts_target_mismatch",
        path=f"{path}.target_id",
        expected=attempt["target_id"],
        provided=facts.get("target_id"),
    )
    checks: list[dict[str, Any]] = []
    reachable = _bool(
        facts.get("reachable"),
        path=f"{path}.reachable",
    )
    _check(
        checks,
        role="reachable",
        passed=reachable,
        observed=reachable,
        required=True,
    )

    python = _mapping(facts.get("python"), path=f"{path}.python")
    python_available = _bool(
        python.get("available"),
        path=f"{path}.python.available",
    )
    if python_available:
        _text(python.get("executable"), path=f"{path}.python.executable")
        _text(python.get("version"), path=f"{path}.python.version")
    _check(
        checks,
        role="python",
        passed=python_available,
        observed=deepcopy(python),
        required="a working Python; version is provenance only",
    )

    observed_imports = _mapping(
        facts.get("imports"),
        path=f"{path}.imports",
    )
    missing_imports: list[str] = []
    for import_name in profile["required_imports"]:
        observation = observed_imports.get(import_name)
        if not isinstance(observation, Mapping):
            missing_imports.append(import_name)
            continue
        available = _bool(
            observation.get("available"),
            path=f"{path}.imports.{import_name}.available",
        )
        if not available:
            missing_imports.append(import_name)
    _check(
        checks,
        role="imports",
        passed=not missing_imports,
        observed=deepcopy(observed_imports),
        required=profile["required_imports"],
    )

    cuda = _mapping(facts.get("cuda"), path=f"{path}.cuda")
    cuda_available = _bool(
        cuda.get("available"),
        path=f"{path}.cuda.available",
    )
    device_count = cuda.get("device_count")
    device_count = _nonnegative_int(
        device_count,
        path=f"{path}.cuda.device_count",
    )
    cuda_passed = (
        not profile["requires_cuda"]
        or (
            cuda_available
            and device_count >= profile["minimum_gpu_count"]
        )
    )
    _check(
        checks,
        role="cuda",
        passed=cuda_passed,
        observed=deepcopy(cuda),
        required={
            "required": profile["requires_cuda"],
            "minimum_gpu_count": profile["minimum_gpu_count"],
        },
    )

    dataset = _mapping(
        facts.get("dataset"),
        path=f"{path}.dataset",
    )
    dataset_readable = _bool(
        dataset.get("readable"),
        path=f"{path}.dataset.readable",
    )
    _check(
        checks,
        role="dataset",
        passed=dataset_readable,
        observed=deepcopy(dataset),
        required="the resolved study dataset is readable",
    )
    output = _mapping(
        facts.get("output"),
        path=f"{path}.output",
    )
    output_writable = _bool(
        output.get("writable"),
        path=f"{path}.output.writable",
    )
    _check(
        checks,
        role="output",
        passed=output_writable,
        observed=deepcopy(output),
        required="the attempt output root is writable",
    )

    smoke = _mapping(facts.get("smoke"), path=f"{path}.smoke")
    smoke_status = smoke.get("status")
    _require(
        smoke_status in SMOKE_STATES,
        code="invalid_smoke_status",
        path=f"{path}.smoke.status",
        expected=f"one of {sorted(SMOKE_STATES)}",
        provided=smoke_status,
    )
    completed_steps = _nonnegative_int(
        smoke.get("completed_optimizer_steps"),
        path=f"{path}.smoke.completed_optimizer_steps",
    )
    smoke_finite = _bool(
        smoke.get("finite"),
        path=f"{path}.smoke.finite",
    )
    smoke_artifact = _bool(
        smoke.get("artifact_written"),
        path=f"{path}.smoke.artifact_written",
    )
    smoke_required_now = (
        run_class == "production"
        and preflight_policy["functional_smoke_required"]
    )
    if smoke_required_now:
        smoke_passed = (
            smoke_status == "passed"
            and completed_steps >= 1
            and smoke_finite
            and smoke_artifact
        )
        _check(
            checks,
            role="functional_smoke",
            passed=smoke_passed,
            observed=deepcopy(smoke),
            required=(
                "one optimizer step with finite outputs and a disposable artifact"
            ),
        )
    else:
        _check(
            checks,
            role="functional_smoke",
            passed=False,
            observed=deepcopy(smoke),
            required="executed by this validation plan",
            blocking=False,
        )

    tk = _mapping(
        facts.get("tk_reference"),
        path=f"{path}.tk_reference",
    )
    tk_status = tk.get("status")
    _require(
        tk_status in TK_STATES,
        code="invalid_tk_status",
        path=f"{path}.tk_reference.status",
        expected=f"one of {sorted(TK_STATES)}",
        provided=tk_status,
    )
    fresh_tk_required_now = (
        run_class == "production"
        and preflight_policy["fresh_tk_gate_required"]
    )
    if fresh_tk_required_now:
        _check(
            checks,
            role="tk_reference",
            passed=tk_status == "passed",
            observed=deepcopy(tk),
            required="a passing fresh T/K reference gate",
        )
    else:
        _check(
            checks,
            role="tk_reference",
            passed=True,
            observed=deepcopy(tk),
            required=(
                "no fresh gate for this unchanged continuation"
                if not preflight_policy["fresh_tk_gate_required"]
                else "executed by this validation plan"
            ),
        )

    blocking_reasons = [
        check["role"] for check in checks if check["blocking"]
    ]
    return {
        "target_id": attempt["target_id"],
        "attempt_id": attempt["attempt_id"],
        "status": "passed" if not blocking_reasons else "failed",
        "checks": checks,
        "blocking_reasons": blocking_reasons,
        "effective_concurrency": attempt["effective_concurrency"],
        "provenance": {
            "observed_at": facts.get("observed_at"),
            "host": profile["host"],
            "python": deepcopy(python),
            "imports": deepcopy(observed_imports),
            "cuda": deepcopy(cuda),
        },
    }


def functional_preflight(
    plan: Mapping[str, Any],
    *,
    host_facts: Mapping[str, Mapping[str, Any]]
    | Sequence[Mapping[str, Any]],
    target_profiles: Mapping[str, Mapping[str, Any]]
    | Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Evaluate functional facts and return one compact preflight receipt.

    Capability failures are represented in the receipt instead of raised.
    Malformed inputs still raise :class:`ExecutorError`.
    """

    plan_value = deepcopy(_mapping(plan, path="execution_plan"))
    _schema(plan_value, EXECUTION_PLAN_SCHEMA, path="execution_plan")
    _require(
        plan_value.get("run_class") in RUN_CLASSES,
        code="invalid_run_class",
        path="execution_plan.run_class",
        expected=f"one of {sorted(RUN_CLASSES)}",
        provided=plan_value.get("run_class"),
    )
    profiles = _normalise_profiles(target_profiles)
    facts_by_target = _normalise_facts(host_facts)
    target_ids = [
        attempt["target_id"] for attempt in plan_value.get("attempts", [])
    ]
    missing = [
        target_id for target_id in target_ids if target_id not in facts_by_target
    ]
    unknown = [
        target_id for target_id in facts_by_target if target_id not in target_ids
    ]
    missing_profiles = [
        target_id for target_id in target_ids if target_id not in profiles
    ]
    _require(
        not missing and not unknown,
        code="host_facts_coverage_mismatch",
        path="host_facts",
        expected="exactly one host-facts observation per planned target",
        provided={"missing": missing, "unknown": unknown},
    )
    _require(
        not missing_profiles,
        code="target_profile_missing",
        path="target_profiles",
        expected="one profile per planned target",
        provided=missing_profiles,
    )

    targets = [
        _evaluate_target_capabilities(
            attempt=attempt,
            profile=profiles[attempt["target_id"]],
            facts=facts_by_target[attempt["target_id"]],
            run_class=plan_value["run_class"],
            preflight_policy=plan_value["preflight_policy"],
        )
        for attempt in plan_value["attempts"]
    ]
    blocking = [
        {
            "target_id": target["target_id"],
            "reasons": target["blocking_reasons"],
        }
        for target in targets
        if target["blocking_reasons"]
    ]
    return {
        "schema_version": PREFLIGHT_RESULT_SCHEMA,
        "status": "passed" if not blocking else "failed",
        "study_id": plan_value["study_id"],
        "proposal_id": plan_value["proposal_id"],
        "run_class": plan_value["run_class"],
        "target_count": len(targets),
        "targets": targets,
        "blocking_reasons": blocking,
        "versions_are_provenance_only": True,
        "side_effects_performed": False,
    }


def receipt_schema_definitions() -> dict[str, dict[str, Any]]:
    """Return small JSON-schema definitions generated by the executor."""

    return {
        "preflight": {
            "$id": PREFLIGHT_RESULT_SCHEMA,
            "type": "object",
            "required": [
                "schema_version",
                "status",
                "study_id",
                "proposal_id",
                "run_class",
                "targets",
                "blocking_reasons",
            ],
            "properties": {
                "schema_version": {"const": PREFLIGHT_RESULT_SCHEMA},
                "status": {"enum": ["passed", "failed"]},
                "study_id": {"type": "string"},
                "proposal_id": {"type": "string"},
                "run_class": {"enum": sorted(RUN_CLASSES)},
                "targets": {"type": "array"},
                "blocking_reasons": {"type": "array"},
            },
        },
        "job": {
            "$id": JOB_RESULT_SCHEMA,
            "type": "object",
            "required": [
                "schema_version",
                "status",
                "study_id",
                "proposal_id",
                "attempt_id",
                "target_id",
                "job_id",
                "run_class",
                "exit_code",
                "artifacts",
                "provenance",
            ],
            "properties": {
                "schema_version": {"const": JOB_RESULT_SCHEMA},
                "status": {"enum": ["complete", "failed"]},
                "exit_code": {"type": "integer"},
                "artifacts": {"type": "array"},
                "provenance": {"type": "object"},
            },
        },
        "attempt": {
            "$id": ATTEMPT_RESULT_SCHEMA,
            "type": "object",
            "required": [
                "schema_version",
                "status",
                "study_id",
                "proposal_id",
                "attempt_id",
                "target_id",
                "run_class",
                "job_results",
            ],
            "properties": {
                "schema_version": {"const": ATTEMPT_RESULT_SCHEMA},
                "status": {"enum": ["complete", "failed"]},
                "job_results": {"type": "array"},
            },
        },
    }


def make_job_result(
    *,
    plan: Mapping[str, Any],
    attempt_id: str,
    target_id: str,
    job_id: str,
    status: str,
    exit_code: int,
    artifacts: Sequence[str],
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the compact result object expected from a numerical adapter."""

    _schema(plan, EXECUTION_PLAN_SCHEMA, path="execution_plan")
    _require(
        status in {"complete", "failed"},
        code="invalid_job_status",
        path="job_result.status",
        expected="'complete' or 'failed'",
        provided=status,
    )
    _require(
        isinstance(exit_code, int) and not isinstance(exit_code, bool),
        code="invalid_exit_code",
        path="job_result.exit_code",
        expected="an integer process exit code",
        provided=exit_code,
    )
    _require(
        (status == "complete" and exit_code == 0)
        or (status == "failed" and exit_code != 0),
        code="job_status_exit_mismatch",
        path="job_result",
        expected="complete with exit code 0, or failed with nonzero exit",
        provided={"status": status, "exit_code": exit_code},
    )
    return {
        "schema_version": JOB_RESULT_SCHEMA,
        "status": status,
        "study_id": plan["study_id"],
        "proposal_id": plan["proposal_id"],
        "attempt_id": attempt_id,
        "target_id": target_id,
        "job_id": job_id,
        "run_class": plan["run_class"],
        "exit_code": exit_code,
        "artifacts": list(artifacts),
        "provenance": deepcopy(dict(provenance)),
    }


def make_attempt_result(
    *,
    plan: Mapping[str, Any],
    attempt_id: str,
    target_id: str,
    job_results: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Summarize adapter-produced job results without extra receipt plumbing."""

    _schema(plan, EXECUTION_PLAN_SCHEMA, path="execution_plan")
    expected_attempt = next(
        (
            attempt
            for attempt in plan["attempts"]
            if attempt["attempt_id"] == attempt_id
            and attempt["target_id"] == target_id
        ),
        None,
    )
    _require(
        expected_attempt is not None,
        code="attempt_not_planned",
        path="attempt_result",
        expected="an attempt and target present in the execution plan",
        provided={"attempt_id": attempt_id, "target_id": target_id},
    )
    by_job: dict[str, Mapping[str, Any]] = {}
    for result in job_results:
        _require(
            result.get("schema_version") == JOB_RESULT_SCHEMA,
            code="job_result_schema_mismatch",
            path="attempt_result.job_results",
            expected=JOB_RESULT_SCHEMA,
            provided=result.get("schema_version"),
        )
        expected_bindings = {
            "study_id": plan["study_id"],
            "proposal_id": plan["proposal_id"],
            "attempt_id": attempt_id,
            "target_id": target_id,
            "run_class": plan["run_class"],
        }
        mismatched_bindings = {
            key: {
                "expected": expected,
                "provided": result.get(key),
            }
            for key, expected in expected_bindings.items()
            if result.get(key) != expected
        }
        _require(
            not mismatched_bindings,
            code="job_result_binding_mismatch",
            path="attempt_result.job_results",
            expected="job results bound to this exact planned attempt",
            provided=mismatched_bindings,
        )
        by_job[result["job_id"]] = result
    expected_job_ids = expected_attempt["planned_job_ids"]
    missing = [job_id for job_id in expected_job_ids if job_id not in by_job]
    unknown = [job_id for job_id in by_job if job_id not in expected_job_ids]
    duplicate_count = len(job_results) != len(by_job)
    _require(
        not missing and not unknown and not duplicate_count,
        code="job_result_coverage_mismatch",
        path="attempt_result.job_results",
        expected="exactly one result for every planned job",
        provided={
            "missing": missing,
            "unknown": unknown,
            "duplicates": duplicate_count,
        },
    )
    ordered = [deepcopy(by_job[job_id]) for job_id in expected_job_ids]
    status = (
        "complete"
        if all(result["status"] == "complete" for result in ordered)
        else "failed"
    )
    return {
        "schema_version": ATTEMPT_RESULT_SCHEMA,
        "status": status,
        "study_id": plan["study_id"],
        "proposal_id": plan["proposal_id"],
        "attempt_id": attempt_id,
        "target_id": target_id,
        "run_class": plan["run_class"],
        "job_results": ordered,
    }


def _profiles_from_paths(paths: Sequence[str]) -> dict[str, dict[str, Any]]:
    profiles = [load_target_profile(path) for path in paths]
    return {profile["target_id"]: profile for profile in profiles}


def _facts_from_path(path: str | Path) -> dict[str, dict[str, Any]]:
    document = load_document(path, label="host_facts")
    values = document.get("targets")
    _require(
        isinstance(values, list),
        code="host_facts_targets_missing",
        path="host_facts.targets",
        expected="a list of host-facts objects",
        provided=values,
    )
    return _normalise_facts(values)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m experiments.experiment_executor",
        description="Build or inspect capability-based experiment plans.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("plan", "doctor"):
        subparser = subparsers.add_parser(command)
        subparser.add_argument("--study", required=True)
        subparser.add_argument("--proposal", required=True)
        subparser.add_argument(
            "--profile",
            action="append",
            required=True,
            help="repeat once for each target profile",
        )
        if command == "doctor":
            subparser.add_argument("--facts", required=True)
        else:
            subparser.add_argument("--facts")
    subparsers.add_parser("schemas")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "schemas":
            result: Mapping[str, Any] = receipt_schema_definitions()
        else:
            study = load_document(args.study, label="resolved_study")
            proposal = load_document(
                args.proposal,
                label="execution_proposal",
            )
            profiles = _profiles_from_paths(args.profile)
            facts = _facts_from_path(args.facts) if args.facts else None
            plan = build_execution_plan(
                study,
                proposal,
                target_profiles=profiles,
                host_facts=facts,
            )
            if args.command == "doctor":
                if facts is None:
                    raise AssertionError("doctor requires --facts")
                result = functional_preflight(
                    plan,
                    host_facts=facts,
                    target_profiles=profiles,
                )
            else:
                result = plan
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except ExecutorError as error:
        print(json.dumps(error.as_dict(), sort_keys=True), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ATTEMPT_RESULT_SCHEMA",
    "EXECUTION_PLAN_SCHEMA",
    "EXECUTION_PROPOSAL_SCHEMA",
    "ExecutorError",
    "HOST_FACTS_SCHEMA",
    "JOB_RESULT_SCHEMA",
    "PREFLIGHT_RESULT_SCHEMA",
    "RESOLVED_STUDY_SCHEMA",
    "TARGET_PROFILE_SCHEMA",
    "build_execution_plan",
    "functional_preflight",
    "load_document",
    "load_target_profile",
    "make_attempt_result",
    "make_job_result",
    "plan_attempts",
    "receipt_schema_definitions",
    "validate_execution_proposal",
    "validate_resolved_study",
    "validate_target_profile",
]
