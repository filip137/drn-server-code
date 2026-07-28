"""Execution-attempt contract bound to, but separate from, a study."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from .common import (
    canonical_sha256,
    file_sha256,
    load_json,
    reject_placeholders,
    require,
    require_absolute_path,
    require_argv,
    require_bool,
    require_exact_keys,
    require_id,
    require_positive_int,
    require_positive_number,
    require_sha256,
    require_source_id,
    require_string_list,
    require_text,
    require_timezone_timestamp,
    resolve_pointer,
    safe_relative_path,
)
from .study import (
    load_study,
    study_bindings_sha256,
    study_spec_sha256,
    validate_study,
)


ATTEMPT_SCHEMA = "experiment-execution-attempt/v1"
ATTEMPT_AUTHORIZATION_STATES = {"draft", "authorized"}
TARGET_KINDS = {"local_tmux", "ssh_tmux", "slurm"}
RUN_CLASSES = {"validation", "production"}


def load_attempt(path: str | Path) -> dict[str, Any]:
    return load_json(Path(path).expanduser().resolve())


def attempt_spec_sha256(value: Mapping[str, Any]) -> str:
    attempt = value.get("attempt") if "attempt" in value else value
    return canonical_sha256(attempt)


def _study_ref(
    value: Any,
    *,
    repo_root: Path,
    require_study_approved: bool,
    verify_files: bool,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    stage = "attempt_validation"
    result = require_exact_keys(
        value,
        keys={
            "path",
            "study_id",
            "study_spec_sha256",
            "study_bindings_sha256",
        },
        stage=stage,
        path="$.attempt.study_ref",
    )
    relative = safe_relative_path(
        result["path"],
        stage=stage,
        path="$.attempt.study_ref.path",
    )
    study_path = repo_root / relative
    require(
        study_path.is_file() and not study_path.is_symlink(),
        stage=stage,
        code="study_contract_missing",
        path="$.attempt.study_ref.path",
        expected="an existing non-symlink study contract",
        provided=str(relative),
    )
    study_document = load_study(study_path)
    study_summary = validate_study(
        study_document,
        repo_root=repo_root,
        verify_bindings=verify_files,
        require_approved=require_study_approved,
    )
    study_id = require_id(
        result["study_id"],
        stage=stage,
        path="$.attempt.study_ref.study_id",
    )
    require(
        study_id == study_summary["study_id"],
        stage=stage,
        code="study_id_mismatch",
        path="$.attempt.study_ref.study_id",
        expected=study_summary["study_id"],
        provided=study_id,
    )
    spec_hash = require_sha256(
        result["study_spec_sha256"],
        stage=stage,
        path="$.attempt.study_ref.study_spec_sha256",
    )
    require(
        spec_hash == study_spec_sha256(study_document),
        stage=stage,
        code="study_spec_hash_mismatch",
        path="$.attempt.study_ref.study_spec_sha256",
        expected=study_spec_sha256(study_document),
        provided=spec_hash,
    )
    binding_hash = require_sha256(
        result["study_bindings_sha256"],
        stage=stage,
        path="$.attempt.study_ref.study_bindings_sha256",
    )
    require(
        binding_hash == study_bindings_sha256(study_document),
        stage=stage,
        code="study_bindings_hash_mismatch",
        path="$.attempt.study_ref.study_bindings_sha256",
        expected=study_bindings_sha256(study_document),
        provided=binding_hash,
    )
    return result, study_document, study_summary


def _source(value: Any) -> dict[str, Any]:
    stage = "attempt_validation"
    result = require_exact_keys(
        value,
        keys={
            "git_commit",
            "control_plane_release_id",
            "control_plane_release_path",
            "control_plane_release_sha256",
        },
        stage=stage,
        path="$.attempt.source",
    )
    require_source_id(
        result["git_commit"],
        stage=stage,
        path="$.attempt.source.git_commit",
    )
    require_id(
        result["control_plane_release_id"],
        stage=stage,
        path="$.attempt.source.control_plane_release_id",
    )
    safe_relative_path(
        result["control_plane_release_path"],
        stage=stage,
        path="$.attempt.source.control_plane_release_path",
    )
    require_sha256(
        result["control_plane_release_sha256"],
        stage=stage,
        path="$.attempt.source.control_plane_release_sha256",
    )
    return result


def _target(value: Any) -> dict[str, Any]:
    stage = "attempt_validation"
    result = require_exact_keys(
        value,
        keys={
            "kind",
            "host",
            "device",
            "profile_path",
            "profile_sha256",
        },
        stage=stage,
        path="$.attempt.target",
    )
    require(
        result["kind"] in TARGET_KINDS,
        stage=stage,
        code="invalid_target_kind",
        path="$.attempt.target.kind",
        expected=f"one of {sorted(TARGET_KINDS)}",
        provided=result["kind"],
    )
    require_text(
        result["host"],
        stage=stage,
        path="$.attempt.target.host",
    )
    require_text(
        result["device"],
        stage=stage,
        path="$.attempt.target.device",
    )
    safe_relative_path(
        result["profile_path"],
        stage=stage,
        path="$.attempt.target.profile_path",
    )
    require_sha256(
        result["profile_sha256"],
        stage=stage,
        path="$.attempt.target.profile_sha256",
    )
    return result


def _execution(
    value: Any,
    *,
    study_document: Mapping[str, Any],
) -> dict[str, Any]:
    stage = "attempt_validation"
    result = require_exact_keys(
        value,
        keys={
            "launcher",
            "collector",
            "validator",
            "routed_manifest",
            "environment_contract",
        },
        stage=stage,
        path="$.attempt.execution",
    )
    for key in ("launcher", "collector", "validator"):
        require_argv(
            result[key],
            stage=stage,
            path=f"$.attempt.execution.{key}",
        )
    routed = require_exact_keys(
        result["routed_manifest"],
        keys={
            "path",
            "sha256",
            "source_study_manifest_sha256",
            "job_count_pointer",
            "job_count",
            "selection",
        },
        stage=stage,
        path="$.attempt.execution.routed_manifest",
    )
    safe_relative_path(
        routed["path"],
        stage=stage,
        path="$.attempt.execution.routed_manifest.path",
    )
    require_sha256(
        routed["sha256"],
        stage=stage,
        path="$.attempt.execution.routed_manifest.sha256",
    )
    source_manifest_hash = require_sha256(
        routed["source_study_manifest_sha256"],
        stage=stage,
        path=(
            "$.attempt.execution.routed_manifest."
            "source_study_manifest_sha256"
        ),
    )
    expected_source_hash = study_document["study"]["design"]["artifacts"][
        "study_manifest_sha256"
    ]
    require(
        source_manifest_hash == expected_source_hash,
        stage=stage,
        code="source_manifest_hash_mismatch",
        path=(
            "$.attempt.execution.routed_manifest."
            "source_study_manifest_sha256"
        ),
        expected=expected_source_hash,
        provided=source_manifest_hash,
    )
    pointer = require_text(
        routed["job_count_pointer"],
        stage=stage,
        path="$.attempt.execution.routed_manifest.job_count_pointer",
    )
    require(
        pointer.startswith("/"),
        stage=stage,
        code="invalid_json_pointer",
        path="$.attempt.execution.routed_manifest.job_count_pointer",
        expected="a JSON pointer beginning with '/'",
        provided=pointer,
    )
    job_count = require_positive_int(
        routed["job_count"],
        stage=stage,
        path="$.attempt.execution.routed_manifest.job_count",
    )
    selection = require_exact_keys(
        routed["selection"],
        keys={"mode", "job_id_key", "selected_study_job_ids"},
        stage=stage,
        path="$.attempt.execution.routed_manifest.selection",
    )
    require(
        selection["mode"] in {"all", "subset"},
        stage=stage,
        code="invalid_route_selection_mode",
        path="$.attempt.execution.routed_manifest.selection.mode",
        expected="all or subset",
        provided=selection["mode"],
    )
    require_id(
        selection["job_id_key"],
        stage=stage,
        path="$.attempt.execution.routed_manifest.selection.job_id_key",
    )
    selected_ids = require_string_list(
        selection["selected_study_job_ids"],
        stage=stage,
        path=(
            "$.attempt.execution.routed_manifest.selection."
            "selected_study_job_ids"
        ),
    )
    for index, selected_id in enumerate(selected_ids):
        require_id(
            selected_id,
            stage=stage,
            path=(
                "$.attempt.execution.routed_manifest.selection."
                f"selected_study_job_ids[{index}]"
            ),
        )
    require(
        job_count == len(selected_ids),
        stage=stage,
        code="route_selection_count_mismatch",
        path="$.attempt.execution.routed_manifest.job_count",
        expected="the number of selected study job IDs",
        provided={
            "job_count": job_count,
            "selected_id_count": len(selected_ids),
        },
    )
    expected_count = study_document["study"]["design"][
        "expected_initial_job_count"
    ]
    if selection["mode"] == "all":
        require(
            job_count == expected_count,
            stage=stage,
            code="routed_job_count_mismatch",
            path="$.attempt.execution.routed_manifest.job_count",
            expected=expected_count,
            provided=job_count,
        )
    else:
        require(
            job_count < expected_count,
            stage=stage,
            code="nonproper_route_subset",
            path="$.attempt.execution.routed_manifest.job_count",
            expected=(
                "fewer jobs than the study total when selection mode is subset"
            ),
            provided={
                "routed": job_count,
                "study_total": expected_count,
            },
        )
    environment = require_exact_keys(
        result["environment_contract"],
        keys={"path", "sha256"},
        stage=stage,
        path="$.attempt.execution.environment_contract",
    )
    safe_relative_path(
        environment["path"],
        stage=stage,
        path="$.attempt.execution.environment_contract.path",
    )
    require_sha256(
        environment["sha256"],
        stage=stage,
        path="$.attempt.execution.environment_contract.sha256",
    )
    return result


def _storage(
    value: Any,
    *,
    attempt_id: str,
    target_kind: str,
) -> dict[str, Any]:
    stage = "attempt_validation"
    result = require_exact_keys(
        value,
        keys={"local_bundle_path", "remote_staging"},
        stage=stage,
        path="$.attempt.storage",
    )
    local_bundle = safe_relative_path(
        result["local_bundle_path"],
        stage=stage,
        path="$.attempt.storage.local_bundle_path",
    )
    require(
        len(local_bundle.parts) > 1
        and local_bundle.parts[0] == "results"
        and attempt_id in local_bundle.parts,
        stage=stage,
        code="nonisolated_attempt_bundle",
        path="$.attempt.storage.local_bundle_path",
        expected=(
            "a path below results/ with the exact attempt_id as one component"
        ),
        provided=str(local_bundle),
    )
    remote = result["remote_staging"]
    require(
        isinstance(remote, list),
        stage=stage,
        code="wrong_type",
        path="$.attempt.storage.remote_staging",
        expected="a remote staging list",
        provided=remote,
    )
    if target_kind == "local_tmux":
        require(
            not remote,
            stage=stage,
            code="unexpected_remote_staging",
            path="$.attempt.storage.remote_staging",
            expected="an empty list for local_tmux",
            provided=remote,
        )
    else:
        require(
            bool(remote),
            stage=stage,
            code="missing_remote_staging",
            path="$.attempt.storage.remote_staging",
            expected="at least one remote staging entry",
            provided=remote,
        )
    for index, raw_entry in enumerate(remote):
        entry_path = f"$.attempt.storage.remote_staging[{index}]"
        entry = require_exact_keys(
            raw_entry,
            keys={"host", "path", "intended_local_destination"},
            stage=stage,
            path=entry_path,
        )
        require_text(
            entry["host"],
            stage=stage,
            path=f"{entry_path}.host",
        )
        require_absolute_path(
            entry["path"],
            stage=stage,
            path=f"{entry_path}.path",
        )
        destination = safe_relative_path(
            entry["intended_local_destination"],
            stage=stage,
            path=f"{entry_path}.intended_local_destination",
        )
        require(
            destination == local_bundle,
            stage=stage,
            code="remote_destination_mismatch",
            path=f"{entry_path}.intended_local_destination",
            expected=str(local_bundle),
            provided=str(destination),
        )
    return result


def _preflight(
    value: Any,
    *,
    study_document: Mapping[str, Any],
    local_bundle: Path,
    target_kind: str,
) -> dict[str, Any]:
    stage = "attempt_validation"
    result = require_exact_keys(
        value,
        keys={"outer_receipt_path", "requirements"},
        stage=stage,
        path="$.attempt.preflight",
    )
    outer = safe_relative_path(
        result["outer_receipt_path"],
        stage=stage,
        path="$.attempt.preflight.outer_receipt_path",
    )
    try:
        outer.relative_to(local_bundle)
    except ValueError:
        require(
            False,
            stage=stage,
            code="preflight_path_outside_bundle",
            path="$.attempt.preflight.outer_receipt_path",
            expected=f"a path below {local_bundle}",
            provided=str(outer),
        )
    requirements = result["requirements"]
    require(
        isinstance(requirements, list) and bool(requirements),
        stage=stage,
        code="missing_preflight_requirements",
        path="$.attempt.preflight.requirements",
        expected="a non-empty preflight requirement list",
        provided=requirements,
    )
    by_role: dict[str, dict[str, Any]] = {}
    receipt_paths: list[Path] = []
    for index, raw_requirement in enumerate(requirements):
        requirement_path = f"$.attempt.preflight.requirements[{index}]"
        requirement = require_exact_keys(
            raw_requirement,
            keys={
                "role",
                "required",
                "schema_version",
                "receipt_path",
                "subject_id",
                "producer_source_id",
            },
            stage=stage,
            path=requirement_path,
        )
        role = require_id(
            requirement["role"],
            stage=stage,
            path=f"{requirement_path}.role",
        )
        require(
            role not in by_role,
            stage=stage,
            code="duplicate_preflight_role",
            path=f"{requirement_path}.role",
            expected="a unique preflight role",
            provided=role,
        )
        by_role[role] = requirement
        required = require_bool(
            requirement["required"],
            stage=stage,
            path=f"{requirement_path}.required",
        )
        if required:
            require_text(
                requirement["schema_version"],
                stage=stage,
                path=f"{requirement_path}.schema_version",
            )
            receipt = safe_relative_path(
                requirement["receipt_path"],
                stage=stage,
                path=f"{requirement_path}.receipt_path",
            )
            try:
                receipt.relative_to(local_bundle)
            except ValueError:
                require(
                    False,
                    stage=stage,
                    code="preflight_path_outside_bundle",
                    path=f"{requirement_path}.receipt_path",
                    expected=f"a path below {local_bundle}",
                    provided=str(receipt),
                )
            receipt_paths.append(receipt)
            subject = requirement["subject_id"]
            if subject is not None:
                require_id(
                    subject,
                    stage=stage,
                    path=f"{requirement_path}.subject_id",
                )
            producer = requirement["producer_source_id"]
            if producer is not None:
                require_source_id(
                    producer,
                    stage=stage,
                    path=f"{requirement_path}.producer_source_id",
                )
        else:
            for key in (
                "schema_version",
                "receipt_path",
                "subject_id",
                "producer_source_id",
            ):
                require(
                    requirement[key] is None,
                    stage=stage,
                    code="nonrequired_preflight_value",
                    path=f"{requirement_path}.{key}",
                    expected="null when the role is not required",
                    provided=requirement[key],
                )
    require(
        "smoke" in by_role and by_role["smoke"]["required"] is True,
        stage=stage,
        code="smoke_not_required",
        path="$.attempt.preflight.requirements",
        expected="a required smoke role",
        provided=sorted(by_role),
    )
    scientific_gates = set(
        study_document["study"]["acceptance"]["required_scientific_gates"]
    )
    missing_gates = sorted(
        gate
        for gate in scientific_gates
        if gate not in by_role or by_role[gate]["required"] is not True
    )
    require(
        not missing_gates,
        stage=stage,
        code="scientific_gate_missing",
        path="$.attempt.preflight.requirements",
        expected="every study-required scientific gate to be required",
        provided={"missing": missing_gates},
    )
    if target_kind in {"ssh_tmux", "slurm"}:
        require(
            "scheduled_run_preflight" in by_role
            and by_role["scheduled_run_preflight"]["required"] is True,
            stage=stage,
            code="scheduled_preflight_missing",
            path="$.attempt.preflight.requirements",
            expected=(
                "a required scheduled_run_preflight role for remote execution"
            ),
            provided=sorted(by_role),
        )
    require(
        outer not in receipt_paths
        and len(receipt_paths) == len(set(receipt_paths)),
        stage=stage,
        code="preflight_receipt_path_collision",
        path="$.attempt.preflight",
        expected="distinct inner receipt paths and a distinct outer receipt",
        provided=[str(outer), *[str(path) for path in receipt_paths]],
    )
    return result


def _deadlines(value: Any, *, target_kind: str) -> dict[str, Any]:
    stage = "attempt_validation"
    result = require_exact_keys(
        value,
        keys={
            "run_class",
            "expected_duration_seconds",
            "hard_deadline_seconds",
            "scheduler_start_timeout_seconds",
        },
        stage=stage,
        path="$.attempt.deadlines",
    )
    run_class = result["run_class"]
    require(
        run_class in RUN_CLASSES,
        stage=stage,
        code="invalid_run_class",
        path="$.attempt.deadlines.run_class",
        expected=f"one of {sorted(RUN_CLASSES)}",
        provided=run_class,
    )
    expected = require_positive_number(
        result["expected_duration_seconds"],
        stage=stage,
        path="$.attempt.deadlines.expected_duration_seconds",
    )
    hard = require_positive_number(
        result["hard_deadline_seconds"],
        stage=stage,
        path="$.attempt.deadlines.hard_deadline_seconds",
    )
    require(
        hard >= expected and hard <= 7 * 24 * 60 * 60,
        stage=stage,
        code="invalid_hard_deadline",
        path="$.attempt.deadlines.hard_deadline_seconds",
        expected="a deadline >= expected duration and <= 604800 seconds",
        provided={"expected": expected, "hard_deadline": hard},
    )
    if run_class == "validation":
        require(
            expected < 600 and hard <= 600,
            stage=stage,
            code="validation_duration_too_long",
            path="$.attempt.deadlines",
            expected="validation duration and deadline below 600 seconds",
            provided={"expected": expected, "hard_deadline": hard},
        )
    else:
        require(
            expected >= 600,
            stage=stage,
            code="production_duration_too_short",
            path="$.attempt.deadlines.expected_duration_seconds",
            expected="at least 600 seconds for production",
            provided=expected,
        )
    scheduler_timeout = result["scheduler_start_timeout_seconds"]
    if target_kind == "slurm":
        timeout = require_positive_number(
            scheduler_timeout,
            stage=stage,
            path=(
                "$.attempt.deadlines.scheduler_start_timeout_seconds"
            ),
        )
        require(
            timeout <= hard,
            stage=stage,
            code="scheduler_timeout_exceeds_deadline",
            path="$.attempt.deadlines.scheduler_start_timeout_seconds",
            expected="a scheduler timeout no greater than the hard deadline",
            provided=timeout,
        )
    else:
        require(
            scheduler_timeout is None,
            stage=stage,
            code="unexpected_scheduler_timeout",
            path="$.attempt.deadlines.scheduler_start_timeout_seconds",
            expected=f"null for target kind {target_kind}",
            provided=scheduler_timeout,
        )
    return result


def _authorization(
    value: Any,
    *,
    expected_attempt_sha256: str,
) -> dict[str, Any]:
    stage = "attempt_authorization_validation"
    result = require_exact_keys(
        value,
        keys={
            "status",
            "authorized_by",
            "authorized_at",
            "authorized_attempt_sha256",
        },
        stage=stage,
        path="$.authorization",
    )
    status = result["status"]
    require(
        status in ATTEMPT_AUTHORIZATION_STATES,
        stage=stage,
        code="invalid_authorization_state",
        path="$.authorization.status",
        expected=f"one of {sorted(ATTEMPT_AUTHORIZATION_STATES)}",
        provided=status,
    )
    if status == "authorized":
        require_text(
            result["authorized_by"],
            stage=stage,
            path="$.authorization.authorized_by",
        )
        require_timezone_timestamp(
            result["authorized_at"],
            stage=stage,
            path="$.authorization.authorized_at",
        )
        bound_hash = require_sha256(
            result["authorized_attempt_sha256"],
            stage=stage,
            path="$.authorization.authorized_attempt_sha256",
        )
        require(
            bound_hash == expected_attempt_sha256,
            stage=stage,
            code="authorization_hash_mismatch",
            path="$.authorization.authorized_attempt_sha256",
            expected=(
                "the canonical SHA-256 of the execution-attempt object"
            ),
            provided=bound_hash,
        )
    else:
        for key in (
            "authorized_by",
            "authorized_at",
            "authorized_attempt_sha256",
        ):
            require(
                result[key] is None,
                stage=stage,
                code="unauthorized_field_not_null",
                path=f"$.authorization.{key}",
                expected="null while authorization status is draft",
                provided=result[key],
            )
    return result


def _provenance(value: Any) -> dict[str, Any]:
    result = require_exact_keys(
        value,
        keys={"origin", "notes"},
        stage="attempt_validation",
        path="$.provenance",
    )
    require_id(
        result["origin"],
        stage="attempt_validation",
        path="$.provenance.origin",
    )
    require_string_list(
        result["notes"],
        stage="attempt_validation",
        path="$.provenance.notes",
        allow_empty=True,
    )
    return result


def _verify_attempt_files(
    attempt: Mapping[str, Any],
    *,
    repo_root: Path,
    study_document: Mapping[str, Any],
) -> None:
    stage = "attempt_binding_verification"
    target = attempt["target"]
    source = attempt["source"]
    execution = attempt["execution"]
    for label, relative_value, expected_hash in (
        (
            "control-plane release",
            source["control_plane_release_path"],
            source["control_plane_release_sha256"],
        ),
        (
            "target profile",
            target["profile_path"],
            target["profile_sha256"],
        ),
        (
            "routed manifest",
            execution["routed_manifest"]["path"],
            execution["routed_manifest"]["sha256"],
        ),
        (
            "environment contract",
            execution["environment_contract"]["path"],
            execution["environment_contract"]["sha256"],
        ),
    ):
        relative = Path(relative_value)
        absolute = repo_root / relative
        require(
            absolute.is_file() and not absolute.is_symlink(),
            stage=stage,
            code="attempt_file_missing",
            path=str(relative),
            expected=f"an existing non-symlink {label}",
            provided=str(relative),
        )
        observed = file_sha256(absolute)
        require(
            observed == expected_hash,
            stage=stage,
            code="attempt_file_hash_mismatch",
            path=str(relative),
            expected=expected_hash,
            provided=observed,
        )
    routed = execution["routed_manifest"]
    manifest = load_json(repo_root / Path(routed["path"]))
    count_value = resolve_pointer(
        manifest,
        routed["job_count_pointer"],
        path="$.attempt.execution.routed_manifest.job_count_pointer",
    )
    observed_count = (
        len(count_value)
        if isinstance(count_value, list)
        else require_positive_int(
            count_value,
            stage=stage,
            path="routed manifest job-count pointer result",
        )
    )
    require(
        observed_count == routed["job_count"],
        stage=stage,
        code="routed_manifest_job_count_mismatch",
        path="$.attempt.execution.routed_manifest.path",
        expected=routed["job_count"],
        provided=observed_count,
    )
    selection = routed["selection"]
    job_id_key = selection["job_id_key"]
    selected_ids = selection["selected_study_job_ids"]
    routed_jobs = resolve_pointer(
        manifest,
        routed["job_count_pointer"],
        path="$.attempt.execution.routed_manifest.job_count_pointer",
    )
    require(
        isinstance(routed_jobs, list),
        stage=stage,
        code="routed_jobs_not_list",
        path="$.attempt.execution.routed_manifest.job_count_pointer",
        expected="a list so routed job identities can be verified",
        provided=type(routed_jobs).__name__,
    )
    routed_ids: list[str] = []
    for index, job in enumerate(routed_jobs):
        require(
            isinstance(job, dict) and job_id_key in job,
            stage=stage,
            code="routed_job_id_missing",
            path=(
                "$.attempt.execution.routed_manifest."
                f"jobs[{index}].{job_id_key}"
            ),
            expected=f"a routed job object with {job_id_key!r}",
            provided=job,
        )
        routed_ids.append(
            require_id(
                job[job_id_key],
                stage=stage,
                path=(
                    "$.attempt.execution.routed_manifest."
                    f"jobs[{index}].{job_id_key}"
                ),
            )
        )
    require(
        routed_ids == selected_ids,
        stage=stage,
        code="routed_job_order_mismatch",
        path="$.attempt.execution.routed_manifest.selection",
        expected="routed job IDs in the exact selected study order",
        provided={"selected": selected_ids, "routed": routed_ids},
    )

    study_manifest_path = (
        repo_root / study_document["bindings"]["study_manifest_path"]
    )
    study_manifest = load_json(study_manifest_path)
    study_jobs = resolve_pointer(
        study_manifest,
        study_document["study"]["design"]["artifacts"][
            "manifest_job_count_pointer"
        ],
        path="$.study.design.artifacts.manifest_job_count_pointer",
    )
    require(
        isinstance(study_jobs, list),
        stage=stage,
        code="study_jobs_not_list",
        path="$.study.design.artifacts.manifest_job_count_pointer",
        expected="a list so study job identities can be verified",
        provided=type(study_jobs).__name__,
    )
    study_ids: list[str] = []
    for index, job in enumerate(study_jobs):
        require(
            isinstance(job, dict) and job_id_key in job,
            stage=stage,
            code="study_job_id_missing",
            path=f"study manifest jobs[{index}].{job_id_key}",
            expected=f"a study job object with {job_id_key!r}",
            provided=job,
        )
        study_ids.append(
            require_id(
                job[job_id_key],
                stage=stage,
                path=f"study manifest jobs[{index}].{job_id_key}",
            )
        )
    require(
        len(study_ids) == len(set(study_ids)),
        stage=stage,
        code="duplicate_study_job_id",
        path="$.bindings.study_manifest_path",
        expected="unique study job IDs",
        provided=study_ids,
    )
    unknown = [job_id for job_id in selected_ids if job_id not in study_ids]
    require(
        not unknown,
        stage=stage,
        code="route_selects_unknown_study_job",
        path="$.attempt.execution.routed_manifest.selection",
        expected="only job IDs declared by the study manifest",
        provided={"unknown": unknown},
    )
    if selection["mode"] == "all":
        require(
            selected_ids == study_ids,
            stage=stage,
            code="all_route_order_mismatch",
            path="$.attempt.execution.routed_manifest.selection",
            expected="every study job ID in study-manifest order",
            provided={"study": study_ids, "selected": selected_ids},
        )


def validate_attempt(
    document: Mapping[str, Any],
    *,
    repo_root: str | Path,
    verify_files: bool = False,
    require_authorized: bool = False,
) -> dict[str, Any]:
    root = Path(repo_root).expanduser().resolve()
    top = require_exact_keys(
        document,
        keys={"schema_version", "attempt", "authorization", "provenance"},
        stage="attempt_validation",
        path="$",
    )
    require(
        top["schema_version"] == ATTEMPT_SCHEMA,
        stage="attempt_validation",
        code="unsupported_schema",
        path="$.schema_version",
        expected=ATTEMPT_SCHEMA,
        provided=top["schema_version"],
    )
    attempt = require_exact_keys(
        top["attempt"],
        keys={
            "attempt_id",
            "study_ref",
            "supersedes_attempt_id",
            "source",
            "target",
            "execution",
            "preflight",
            "storage",
            "deadlines",
        },
        stage="attempt_validation",
        path="$.attempt",
    )
    attempt_id = require_id(
        attempt["attempt_id"],
        stage="attempt_validation",
        path="$.attempt.attempt_id",
    )
    supersedes = attempt["supersedes_attempt_id"]
    if supersedes is not None:
        require_id(
            supersedes,
            stage="attempt_validation",
            path="$.attempt.supersedes_attempt_id",
        )
        require(
            supersedes != attempt_id,
            stage="attempt_validation",
            code="self_superseding_attempt",
            path="$.attempt.supersedes_attempt_id",
            expected="a different prior attempt ID",
            provided=supersedes,
        )
    _, study_document, study_summary = _study_ref(
        attempt["study_ref"],
        repo_root=root,
        require_study_approved=require_authorized,
        verify_files=verify_files or require_authorized,
    )
    _source(attempt["source"])
    target = _target(attempt["target"])
    _execution(attempt["execution"], study_document=study_document)
    storage = _storage(
        attempt["storage"],
        attempt_id=attempt_id,
        target_kind=target["kind"],
    )
    _preflight(
        attempt["preflight"],
        study_document=study_document,
        local_bundle=Path(storage["local_bundle_path"]),
        target_kind=target["kind"],
    )
    deadlines = _deadlines(
        attempt["deadlines"],
        target_kind=target["kind"],
    )
    spec_hash = attempt_spec_sha256(top)
    authorization = _authorization(
        top["authorization"],
        expected_attempt_sha256=spec_hash,
    )
    _provenance(top["provenance"])
    if require_authorized:
        require(
            authorization["status"] == "authorized",
            stage="attempt_authorization_validation",
            code="attempt_not_authorized",
            path="$.authorization.status",
            expected="authorized",
            provided=authorization["status"],
        )
    if authorization["status"] == "authorized":
        reject_placeholders(top)
    if verify_files or require_authorized:
        _verify_attempt_files(
            attempt,
            repo_root=root,
            study_document=study_document,
        )
    return {
        "schema_version": ATTEMPT_SCHEMA,
        "status": "passed",
        "attempt_id": attempt_id,
        "study_id": study_summary["study_id"],
        "study_spec_sha256": study_summary["study_spec_sha256"],
        "study_bindings_sha256": study_summary["study_bindings_sha256"],
        "authorization_status": authorization["status"],
        "attempt_spec_sha256": spec_hash,
        "target_kind": target["kind"],
        "run_class": deadlines["run_class"],
    }
