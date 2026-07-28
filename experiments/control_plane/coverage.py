"""Cross-attempt coverage checks for a sharded scientific study."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from .attempt import load_attempt, validate_attempt
from .common import (
    load_json,
    require,
    require_id,
    resolve_pointer,
)
from .study import (
    load_study,
    study_bindings_sha256,
    study_spec_sha256,
    validate_study,
)


def validate_attempt_set(
    *,
    study_path: str | Path,
    attempt_paths: Sequence[str | Path],
    repo_root: str | Path,
    verify_files: bool = False,
    require_authorized: bool = False,
) -> dict[str, Any]:
    """Require exact, non-overlapping coverage of every study-manifest job."""

    root = Path(repo_root).expanduser().resolve()
    resolved_study_path = Path(study_path).expanduser().resolve()
    try:
        relative_study_path = resolved_study_path.relative_to(root)
    except ValueError:
        require(
            False,
            stage="attempt_coverage_validation",
            code="study_path_outside_repo",
            path="study_path",
            expected=f"a study below repository root {root}",
            provided=str(resolved_study_path),
        )
        raise AssertionError("unreachable")
    study = load_study(resolved_study_path)
    study_summary = validate_study(
        study,
        repo_root=root,
        verify_bindings=verify_files or require_authorized,
        require_approved=require_authorized,
    )
    require(
        bool(attempt_paths),
        stage="attempt_coverage_validation",
        code="missing_attempts",
        path="attempt_paths",
        expected="at least one execution-attempt contract",
        provided=[],
    )

    attempts: list[dict[str, Any]] = []
    attempt_ids: list[str] = []
    selection_key: str | None = None
    routes: list[dict[str, Any]] = []
    observed_owners: dict[str, str] = {}
    duplicates: dict[str, list[str]] = {}
    for raw_path in attempt_paths:
        path = Path(raw_path).expanduser().resolve()
        document = load_attempt(path)
        summary = validate_attempt(
            document,
            repo_root=root,
            verify_files=verify_files or require_authorized,
            require_authorized=require_authorized,
        )
        attempt = document["attempt"]
        attempt_id = summary["attempt_id"]
        require(
            attempt_id not in attempt_ids,
            stage="attempt_coverage_validation",
            code="duplicate_attempt_id",
            path="attempt_paths",
            expected="unique execution attempt IDs",
            provided=attempt_id,
        )
        attempt_ids.append(attempt_id)
        study_ref = attempt["study_ref"]
        for key, expected in (
            ("path", str(relative_study_path)),
            ("study_id", study_summary["study_id"]),
            ("study_spec_sha256", study_spec_sha256(study)),
            ("study_bindings_sha256", study_bindings_sha256(study)),
        ):
            require(
                study_ref[key] == expected,
                stage="attempt_coverage_validation",
                code="attempt_study_ref_mismatch",
                path=f"{path}:$.attempt.study_ref.{key}",
                expected=expected,
                provided=study_ref[key],
            )
        selection = attempt["execution"]["routed_manifest"]["selection"]
        key = selection["job_id_key"]
        if selection_key is None:
            selection_key = key
        require(
            key == selection_key,
            stage="attempt_coverage_validation",
            code="attempt_job_id_key_mismatch",
            path=f"{path}:$.attempt.execution.routed_manifest.selection",
            expected=selection_key,
            provided=key,
        )
        selected = list(selection["selected_study_job_ids"])
        for job_id in selected:
            if job_id in observed_owners:
                duplicates.setdefault(
                    job_id,
                    [observed_owners[job_id]],
                ).append(attempt_id)
            else:
                observed_owners[job_id] = attempt_id
        routes.append(
            {
                "attempt_id": attempt_id,
                "target_host": attempt["target"]["host"],
                "selected_study_job_ids": selected,
            }
        )
        attempts.append(document)

    require(
        not duplicates,
        stage="attempt_coverage_validation",
        code="duplicate_study_job_route",
        path="attempt_paths",
        expected="each study job assigned to exactly one attempt",
        provided=duplicates,
    )
    if selection_key is None:
        raise AssertionError("attempt list was required non-empty")
    study_manifest = load_json(
        root / study["bindings"]["study_manifest_path"]
    )
    study_jobs = resolve_pointer(
        study_manifest,
        study["study"]["design"]["artifacts"][
            "manifest_job_count_pointer"
        ],
        path="$.study.design.artifacts.manifest_job_count_pointer",
    )
    require(
        isinstance(study_jobs, list),
        stage="attempt_coverage_validation",
        code="study_jobs_not_list",
        path="$.study.design.artifacts.manifest_job_count_pointer",
        expected="a list for cross-attempt coverage",
        provided=type(study_jobs).__name__,
    )
    study_ids: list[str] = []
    for index, job in enumerate(study_jobs):
        require(
            isinstance(job, dict) and selection_key in job,
            stage="attempt_coverage_validation",
            code="study_job_id_missing",
            path=f"study manifest jobs[{index}].{selection_key}",
            expected=f"a job object with {selection_key!r}",
            provided=job,
        )
        study_ids.append(
            require_id(
                job[selection_key],
                stage="attempt_coverage_validation",
                path=f"study manifest jobs[{index}].{selection_key}",
            )
        )
    require(
        len(study_ids) == len(set(study_ids)),
        stage="attempt_coverage_validation",
        code="duplicate_study_job_id",
        path="$.bindings.study_manifest_path",
        expected="unique study job IDs",
        provided=study_ids,
    )
    missing = [job_id for job_id in study_ids if job_id not in observed_owners]
    unknown = [
        job_id for job_id in observed_owners if job_id not in set(study_ids)
    ]
    require(
        not missing and not unknown,
        stage="attempt_coverage_validation",
        code="attempt_set_coverage_mismatch",
        path="attempt_paths",
        expected="every study job exactly once and no unknown jobs",
        provided={"missing": missing, "unknown": unknown},
    )
    return {
        "schema_version": "experiment-attempt-set-validation/v1",
        "status": "passed",
        "study_id": study_summary["study_id"],
        "study_job_count": len(study_ids),
        "attempt_count": len(attempts),
        "authorization_required": require_authorized,
        "routes": routes,
    }


__all__ = ["validate_attempt_set"]
