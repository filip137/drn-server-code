#!/usr/bin/env python3
"""One conversational request-to-execution-plan facade.

The facade has no launch side effects.  Before review it returns the complete
scientific summary and asks for the two user decisions.  After both decisions
it deterministically produces three validation attempts and the corresponding
production plan through the common executor.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

from experiments import experiment_executor as executor
from experiments import functional_dispatch
from experiments.experiment_catalog import load_catalog
from experiments.experiment_request import (
    load_request,
    parse_request_text,
    validate_request,
)
from experiments.experiment_study import (
    build_resolved_study,
    validate_resolved_study,
)
from experiments.mnist_conv.identity import sha256_file, sha256_json
from experiments.mnist_conv.io import atomic_create_json, read_json


FLOW_SCHEMA = "experiment-prepared-flow/v1"
EXECUTOR_STUDY_SCHEMA = executor.RESOLVED_STUDY_SCHEMA
DEFAULT_CATALOG = Path(__file__).with_name("experiment_catalog.json")
PROFILE_DIR = Path(__file__).resolve().parents[1] / "configs" / "dispatch"
DEFAULT_VALIDATION_EXPECTED_SECONDS = 5 * 60
DEFAULT_VALIDATION_DEADLINE_SECONDS = 6 * 60
DEFAULT_PRODUCTION_EXPECTED_SECONDS = 6 * 60 * 60
DEFAULT_PRODUCTION_DEADLINE_SECONDS = 24 * 60 * 60


class ExperimentFlowError(ValueError):
    """The intake could not be resolved without inventing a user choice."""


def _executor_study(study: Mapping[str, Any]) -> dict[str, Any]:
    scientific = study.get("study")
    if not isinstance(scientific, Mapping):
        raise ExperimentFlowError(
            "Expected resolved study.study to be a mapping. "
            f"Provided value: {scientific!r}."
        )
    jobs = scientific.get("jobs")
    if not isinstance(jobs, list) or not jobs:
        raise ExperimentFlowError(
            "Expected resolved study.study.jobs to be non-empty. "
            f"Provided value: {jobs!r}."
        )
    return {
        "schema_version": EXECUTOR_STUDY_SCHEMA,
        "study_id": study["study_id"],
        "jobs": copy.deepcopy(jobs),
    }


def load_profiles(
    target_ids: Sequence[str] = ("local", "trex", "akib"),
    *,
    profile_dir: str | Path = PROFILE_DIR,
) -> dict[str, dict[str, Any]]:
    root = Path(profile_dir).expanduser().resolve()
    return {
        target_id: executor.load_target_profile(
            root / f"functional_{target_id}.json"
        )
        for target_id in target_ids
    }


def apply_workspace_overrides(
    profiles: Mapping[str, Mapping[str, Any]],
    overrides: Sequence[str],
) -> dict[str, dict[str, Any]]:
    """Apply executor-owned isolated deployment roots from CLI values."""

    resolved = {
        target_id: copy.deepcopy(dict(profile))
        for target_id, profile in profiles.items()
    }
    seen: set[str] = set()
    for provided in overrides:
        target_id, separator, raw_path = provided.partition("=")
        candidate = Path(raw_path)
        if (
            not separator
            or target_id not in resolved
            or target_id in seen
            or not candidate.is_absolute()
            or ".." in candidate.parts
        ):
            raise ExperimentFlowError(
                "Expected each workspace override once as "
                "TARGET=/absolute/path for an available target. "
                f"Provided value: {provided!r}."
            )
        resolved[target_id]["workspace_root"] = str(candidate)
        resolved[target_id] = executor.validate_target_profile(
            resolved[target_id]
        )
        seen.add(target_id)
    return resolved


def load_smoke_host_facts(
    profiles: Mapping[str, Mapping[str, Any]],
    bindings: Sequence[str],
) -> dict[str, dict[str, Any]]:
    """Load target=receipt bindings and derive capability facts."""

    facts: dict[str, dict[str, Any]] = {}
    for provided in bindings:
        target_id, separator, raw_path = provided.partition("=")
        path = Path(raw_path).expanduser()
        if (
            not separator
            or target_id not in profiles
            or target_id in facts
            or not path.is_absolute()
            or not path.is_file()
            or path.is_symlink()
        ):
            raise ExperimentFlowError(
                "Expected each smoke receipt once as "
                "TARGET=/absolute/path to an existing regular JSON file. "
                f"Provided value: {provided!r}."
            )
        facts[target_id] = functional_dispatch.host_facts_from_smoke(
            read_json(path),
            profiles[target_id],
        )
    return facts


def _requested_targets(
    request: Mapping[str, Any],
    available: Sequence[str],
) -> list[str]:
    answers = request.get("user_answers")
    answers = answers if isinstance(answers, Mapping) else {}
    text = str(answers.get("preferred_compute") or "").casefold()
    aliases = {
        "local": ("local", "main"),
        "trex": ("trex",),
        "akib": ("akib", "akibscomputer", "integnano-akib"),
    }
    selected = [
        target_id
        for target_id in available
        if any(
            re.search(rf"\b{re.escape(alias)}\b", text)
            for alias in aliases.get(target_id, (target_id,))
        )
    ]
    if not selected and re.search(r"\bauto\b", text):
        selected = list(available)
    if not selected:
        raise ExperimentFlowError(
            "Expected preferred_compute to name at least one available target "
            f"{list(available)!r}. Provided value: {text!r}."
        )
    return selected


def _assign_jobs(
    jobs: Sequence[Mapping[str, Any]],
    target_ids: Sequence[str],
) -> list[dict[str, Any]]:
    groups: list[list[str]] = []
    group_index: dict[str, int] = {}
    for job in jobs:
        job_id = str(job["job_id"])
        scheme = job.get("scheme")
        key = str(scheme) if scheme is not None else job_id
        if key not in group_index:
            group_index[key] = len(groups)
            groups.append([])
        groups[group_index[key]].append(job_id)
    ownership = {target_id: [] for target_id in target_ids}
    for index, group in enumerate(groups):
        ownership[target_ids[index % len(target_ids)]].extend(group)
    return [
        {
            "target_id": target_id,
            "job_ids": ownership[target_id],
            "concurrency": "auto",
            "validation_job_id": ownership[target_id][0],
        }
        for target_id in target_ids
        if ownership[target_id]
    ]


def discover_parent_bundle(
    *,
    repo_root: Path,
    parent_experiment_id: str,
    parent_config_path: str,
    selected_job_ids: Sequence[str],
) -> Path:
    """Choose the newest complete compatible bundle as an operational detail."""

    source_config = (repo_root / parent_config_path).resolve()
    config_sha256 = sha256_file(source_config)
    staging = (
        repo_root
        / "results"
        / ".launch_staging"
        / parent_experiment_id
    )
    candidates: list[tuple[int, str, Path]] = []
    for manifest_path in staging.glob("*/manifest.json"):
        try:
            manifest = read_json(manifest_path)
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            continue
        if manifest.get("config_file_sha256") != config_sha256:
            continue
        entries = manifest.get("entries")
        if not isinstance(entries, list):
            continue
        by_id = {
            item.get("entry_id"): item
            for item in entries
            if isinstance(item, Mapping)
        }
        if any(job_id not in by_id for job_id in selected_job_ids):
            continue
        bundle = manifest_path.parent
        required = [bundle / "study.resolved.json"]
        for job_id in selected_job_ids:
            entry = by_id[job_id]
            asset = entry.get("asset_entry_id")
            probe = entry.get("probe_result")
            if isinstance(asset, str):
                required.extend(
                    [
                        bundle
                        / "stages"
                        / "assets"
                        / "entries"
                        / asset
                        / "initialization.pt",
                        bundle
                        / "stages"
                        / "assets"
                        / "entries"
                        / asset
                        / "split_indices.json",
                    ]
                )
            if isinstance(probe, str):
                required.append(bundle / probe)
        if not all(path.is_file() and not path.is_symlink() for path in required):
            continue
        candidates.append(
            (
                manifest_path.stat().st_mtime_ns,
                manifest_path.parent.name,
                bundle,
            )
        )
    if not candidates:
        raise ExperimentFlowError(
            "Expected at least one complete staged parent bundle matching the "
            f"current scientific config and selected jobs. Provided value: "
            f"staging={staging}, config_sha256={config_sha256!r}, "
            f"jobs={list(selected_job_ids)!r}."
        )
    candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return candidates[0][2].relative_to(repo_root)


def _execution_proposal(
    *,
    request: Mapping[str, Any],
    study: Mapping[str, Any],
    bundle_relative: Path,
    target_ids: Sequence[str],
    run_class: str,
    attempt_label: str | None = None,
) -> dict[str, Any]:
    answers = request.get("user_answers")
    answers = answers if isinstance(answers, Mapping) else {}
    jobs = study["study"]["jobs"]
    assignment = _assign_jobs(jobs, target_ids)
    proposal_seed = {
        "study_id": study["study_id"],
        "targets": assignment,
        "bundle": bundle_relative.as_posix(),
        "run_class": run_class,
    }
    if attempt_label is not None:
        proposal_seed["attempt_label"] = attempt_label
    proposal_id = (
        "proposal_"
        + sha256_json(proposal_seed)[:20]
        + "-"
        + run_class
    )
    study_runtime_path = (
        "{workspace_root}/"
        + (bundle_relative / "study.resolved.json").as_posix()
    )
    bundle_path = "{workspace_root}/" + bundle_relative.as_posix()
    return {
        "schema_version": executor.EXECUTION_PROPOSAL_SCHEMA,
        "proposal_id": proposal_id,
        "run_class": run_class,
        "launch_after_checks": (
            answers.get("executor_launch_after_checks") == "yes"
        ),
        "runner": {
            "adapter_id": "mnist-conv-perfectdiode-common-job-v1",
            "argv_template": [
                "{python}",
                "-m",
                "experiments.run_mnist_conv_perfectdiode_job",
                "--run-class",
                "{run_class}",
                "--study",
                study_runtime_path,
                "--bundle",
                bundle_path,
                "--entry-id",
                "{job_id}",
                "--attempt-id",
                "{proposal_id}-{target_id}-{run_class}",
                "--target",
                "{target_id}",
                "--data-root",
                "{data_root}",
                "--device",
                "{device}",
                "--output-dir",
                "{output_dir}",
            ],
        },
        "preflight": {
            "functional_smoke_required": True,
            "fresh_tk_gate_required": bool(
                study["study"]["tk_gate"]["fresh_required"]
            ),
        },
        "targets": assignment,
    }


def _review_state(request: Mapping[str, Any]) -> dict[str, Any]:
    review = request.get("user_review")
    review = review if isinstance(review, Mapping) else {}
    scientific = review.get("scientific_summary_approved")
    execution = review.get("execution_proposal_authorized")
    return {
        "scientific_summary_approved": scientific == "yes",
        "execution_proposal_authorized": execution == "yes",
        "scientific_value": scientific,
        "execution_value": execution,
    }


def prepare_request(
    request: Mapping[str, Any],
    *,
    catalog: Mapping[str, Any],
    repo_root: Path,
    target_profiles: Mapping[str, Mapping[str, Any]] | None = None,
    bundle_relative: Path | None = None,
    host_facts: Mapping[str, Mapping[str, Any]] | None = None,
    validation_expected_seconds: int = DEFAULT_VALIDATION_EXPECTED_SECONDS,
    validation_deadline_seconds: int = DEFAULT_VALIDATION_DEADLINE_SECONDS,
    production_expected_seconds: int = DEFAULT_PRODUCTION_EXPECTED_SECONDS,
    production_deadline_seconds: int = DEFAULT_PRODUCTION_DEADLINE_SECONDS,
    attempt_label: str | None = None,
) -> dict[str, Any]:
    """Prepare the whole flow, stopping cleanly at the user-review boundary."""

    intake = validate_request(request)
    if intake["status"] != "valid":
        return {
            "schema_version": FLOW_SCHEMA,
            "status": "needs_user_input",
            "side_effects_performed": False,
            "launched_job_count": 0,
            "intake_validation": intake,
            "next_step": "ask_for_all_missing_or_invalid_intake_fields_once",
        }
    if attempt_label is not None and executor.ID_RE.fullmatch(
        attempt_label
    ) is None:
        raise ExperimentFlowError(
            "Expected attempt_label to be a lowercase filesystem-safe "
            f"identifier. Provided value: {attempt_label!r}."
        )
    study = build_resolved_study(request, catalog, repo_root=repo_root)
    study_validation = validate_resolved_study(study)
    review = _review_state(request)
    base = {
        "schema_version": FLOW_SCHEMA,
        "side_effects_performed": False,
        "launched_job_count": 0,
        "intake_validation": intake,
        "study_validation": study_validation,
        "review": review,
        "resolved_study": study,
    }
    if not (
        review["scientific_summary_approved"]
        and review["execution_proposal_authorized"]
    ):
        return {
            **base,
            "status": "awaiting_user_review",
            "validation_plan": None,
            "production_plan": None,
            "next_step": (
                "ask_user_to_approve_or_correct_the_scientific_summary_and_"
                "execution_proposal"
            ),
        }

    profiles = (
        {
            key: copy.deepcopy(dict(value))
            for key, value in target_profiles.items()
        }
        if target_profiles is not None
        else load_profiles()
    )
    target_ids = _requested_targets(request, list(profiles))
    selected_job_ids = [
        str(job["job_id"]) for job in study["study"]["jobs"]
    ]
    if bundle_relative is None:
        bundle_relative = discover_parent_bundle(
            repo_root=repo_root,
            parent_experiment_id=study["study"][
                "parent_experiment_id"
            ],
            parent_config_path=study["bindings_and_provenance"][
                "parent_config_path"
            ],
            selected_job_ids=selected_job_ids,
        )
    runtime_study = _executor_study(study)
    validation_proposal = _execution_proposal(
        request=request,
        study=study,
        bundle_relative=bundle_relative,
        target_ids=target_ids,
        run_class="validation",
        attempt_label=attempt_label,
    )
    production_proposal = _execution_proposal(
        request=request,
        study=study,
        bundle_relative=bundle_relative,
        target_ids=target_ids,
        run_class="production",
        attempt_label=attempt_label,
    )
    validation_plan = executor.build_execution_plan(
        runtime_study,
        validation_proposal,
        target_profiles=profiles,
        host_facts=host_facts,
    )
    production_plan = executor.build_execution_plan(
        runtime_study,
        production_proposal,
        target_profiles=profiles,
        host_facts=host_facts,
    )
    validation_dispatch_requests = functional_dispatch.build_dispatch_requests(
        validation_plan,
        expected_duration_seconds=validation_expected_seconds,
        hard_deadline_seconds=validation_deadline_seconds,
    )
    preflight = (
        None
        if host_facts is None
        else executor.functional_preflight(
            production_plan,
            host_facts=host_facts,
            target_profiles=profiles,
        )
    )
    production_dispatch_requests = (
        None
        if preflight is None or preflight["status"] != "passed"
        else functional_dispatch.build_dispatch_requests(
            production_plan,
            expected_duration_seconds=production_expected_seconds,
            hard_deadline_seconds=production_deadline_seconds,
            production_preflight=preflight,
        )
    )
    return {
        **base,
        "status": (
            "ready_for_functional_smokes"
            if preflight is None
            else "ready_for_production"
            if preflight["status"] == "passed"
            else "preflight_failed"
        ),
        "bundle_relative": bundle_relative.as_posix(),
        "executor_study": runtime_study,
        "validation_proposal": validation_proposal,
        "production_proposal": production_proposal,
        "validation_plan": validation_plan,
        "production_plan": production_plan,
        "validation_dispatch_requests": validation_dispatch_requests,
        "production_dispatch_requests": production_dispatch_requests,
        "functional_preflight": preflight,
        "next_step": (
            "run_three_bounded_one_batch_functional_smokes"
            if preflight is None
            else "arm_and_launch_production"
            if preflight["status"] == "passed"
            else "report_preflight_blockers_and_ask_user_after_budget"
        ),
    }


def prepare_text(
    text: str,
    *,
    repo_root: Path,
    catalog_path: Path = DEFAULT_CATALOG,
    **kwargs: Any,
) -> dict[str, Any]:
    return prepare_request(
        parse_request_text(text),
        catalog=load_catalog(catalog_path),
        repo_root=repo_root,
        **kwargs,
    )


def publish_prepared_flow(
    prepared: Mapping[str, Any],
    output_dir: str | Path,
) -> dict[str, str]:
    """Publish authorized generated documents idempotently."""

    if prepared.get("status") not in {
        "ready_for_functional_smokes",
        "ready_for_production",
    }:
        raise ExperimentFlowError(
            "Expected an authorized prepared flow before publishing. "
            f"Provided value: status={prepared.get('status')!r}."
        )
    root = Path(output_dir).expanduser().resolve()
    documents = {
        "prepared_flow.json": prepared,
        "resolved_study.json": prepared["resolved_study"],
        "executor_study.json": prepared["executor_study"],
        "validation_proposal.json": prepared["validation_proposal"],
        "production_proposal.json": prepared["production_proposal"],
        "validation_plan.json": prepared["validation_plan"],
        "production_plan.json": prepared["production_plan"],
    }
    for request in prepared["validation_dispatch_requests"]:
        documents[
            f"validation_dispatch_{request['target_id']}.json"
        ] = request
    for request in prepared.get("production_dispatch_requests") or []:
        documents[
            f"production_dispatch_{request['target_id']}.json"
        ] = request
    published: dict[str, str] = {}
    for name, value in documents.items():
        path = root / name
        if path.exists():
            if read_json(path) != value:
                raise FileExistsError(
                    "Expected an existing generated document to be byte-"
                    f"equivalent in meaning. Provided value: {path}."
                )
        else:
            atomic_create_json(path, value, canonical=True)
        published[name] = str(path)
    return published


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Resolve a filled experiment request into a scientific summary "
            "and, only after review, dry execution plans."
        )
    )
    parser.add_argument("request", type=Path)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--profile-dir", type=Path, default=PROFILE_DIR)
    parser.add_argument(
        "--workspace-override",
        action="append",
        default=[],
        metavar="TARGET=/ABSOLUTE/PATH",
        help="override one target workspace with an isolated staged checkout",
    )
    parser.add_argument(
        "--smoke-receipt",
        action="append",
        default=[],
        metavar="TARGET=/ABSOLUTE/PATH",
        help="derive production capability facts from one passed target smoke",
    )
    parser.add_argument(
        "--attempt-label",
        help="operational nonce for a fresh immutable attempt after repair",
    )
    parser.add_argument("--publish-dir", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    root = args.repo_root.expanduser().resolve()
    request = load_request(args.request)
    profiles = apply_workspace_overrides(
        load_profiles(profile_dir=args.profile_dir),
        args.workspace_override,
    )
    host_facts = (
        load_smoke_host_facts(profiles, args.smoke_receipt)
        if args.smoke_receipt
        else None
    )
    result = prepare_request(
        request,
        catalog=load_catalog(args.catalog),
        repo_root=root,
        target_profiles=profiles,
        host_facts=host_facts,
        attempt_label=args.attempt_label,
    )
    if args.publish_dir is not None:
        result["published"] = publish_prepared_flow(
            result, args.publish_dir
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"].startswith("ready_") else 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ExperimentFlowError",
    "apply_workspace_overrides",
    "discover_parent_bundle",
    "load_smoke_host_facts",
    "load_profiles",
    "prepare_request",
    "prepare_text",
    "publish_prepared_flow",
]
