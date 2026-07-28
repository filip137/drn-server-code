#!/usr/bin/env python3
"""One-command Jean Zay validation fast path.

An approved continuation request is the sole user-authored authority. This
module resolves its existing parent case, creates one content-addressed
package, runs one import-only remote preflight, asks Slurm to validate the
stable runner, submits one singleton validation job, monitors it, and uses the
repository transfer workflow to repatriate the completed result.

The launcher never creates a catalog entry, never treats the tracker as launch
authority, never runs broad tests, and never retries a failed operation.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import tarfile
import time
from typing import Any, Callable, Mapping, Sequence

from experiments.experiment_catalog import load_catalog
from experiments.experiment_request import load_request, validate_request
from experiments.experiment_study import (
    build_resolved_study,
    validate_resolved_study,
)
from experiments.mnist_conv.identity import sha256_file
from experiments.mnist_conv.io import atomic_create_json, read_json
from experiments.mnist_conv.perfectdiode_functional_smoke import (
    validate_functional_preflight,
)


PROFILE_SCHEMA = "jean-zay-validation-profile/v1"
PACKAGE_SCHEMA = "jean-zay-validation-package/v1"
FAILURE_SCHEMA = "jean-zay-validation-failure/v1"
IMPORT_PREFLIGHT_SCHEMA = "jean-zay-import-preflight/v1"
DEFAULT_PROFILE = (
    Path(__file__).resolve().parents[1]
    / "configs"
    / "dispatch"
    / "functional_jeanzay.json"
)
DEFAULT_CATALOG = Path(__file__).with_name("experiment_catalog.json")
STABLE_RUNNER_RELATIVE = Path("experiments/run_jeanzay_validation.slurm")
ID_RE = re.compile(r"^[a-z0-9][a-z0-9._-]*$")
TERMINAL_STATES = {
    "BOOT_FAIL",
    "CANCELLED",
    "COMPLETED",
    "DEADLINE",
    "FAILED",
    "NODE_FAIL",
    "OUT_OF_MEMORY",
    "PREEMPTED",
    "REVOKED",
    "TIMEOUT",
}
FOCUSED_TESTS = (
    (
        "labs/tests/test_jeanzay_validation_fastpath.py::"
        "test_stable_runner_is_syntax_valid_and_remote_preflight_is_import_only"
    ),
    (
        "labs/tests/test_jeanzay_validation_fastpath.py::"
        "test_cached_profile_builds_one_exact_singleton_sbatch_command"
    ),
    (
        "labs/tests/test_jeanzay_validation_fastpath.py::"
        "test_approved_request_prepares_one_package_without_its_own_catalog_entry"
    ),
)

RunCommand = Callable[..., subprocess.CompletedProcess[str]]


class JeanZayValidationError(RuntimeError):
    """One bounded fast-path stage failed without retry."""

    def __init__(
        self,
        stage: str,
        message: str,
        *,
        command: Sequence[str] | None = None,
        job_id: str | None = None,
    ) -> None:
        super().__init__(message)
        self.stage = stage
        self.command = list(command) if command is not None else None
        self.job_id = job_id


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            f"Expected {label} to be non-empty text. Provided value: {value!r}."
        )
    return value.strip()


def _identifier(value: Any, label: str) -> str:
    text = _text(value, label)
    if ID_RE.fullmatch(text) is None or text in {".", ".."}:
        raise ValueError(
            f"Expected {label} to be a lowercase filesystem-safe identifier. "
            f"Provided value: {value!r}."
        )
    return text


def _absolute(value: Any, label: str) -> str:
    path = Path(_text(value, label))
    if not path.is_absolute() or ".." in path.parts:
        raise ValueError(
            f"Expected {label} to be an absolute normalized path. "
            f"Provided value: {value!r}."
        )
    return str(path)


def _positive_int(value: Any, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(
            f"Expected {label} to be a positive integer. "
            f"Provided value: {value!r}."
        )
    return value


def validate_profile(value: Mapping[str, Any]) -> dict[str, Any]:
    profile = json.loads(json.dumps(value))
    if profile.get("schema_version") != PROFILE_SCHEMA:
        raise ValueError(
            f"Expected schema_version {PROFILE_SCHEMA!r}. "
            f"Provided value: {profile.get('schema_version')!r}."
        )
    if profile.get("target_id") != "jean-zay":
        raise ValueError(
            "Expected target_id='jean-zay'. "
            f"Provided value: {profile.get('target_id')!r}."
        )
    transport = profile.get("transport")
    scheduler = profile.get("scheduler")
    environment = profile.get("environment")
    paths = profile.get("paths")
    deadlines = profile.get("deadlines")
    failure_budget = profile.get("failure_budget")
    for label, item in (
        ("transport", transport),
        ("scheduler", scheduler),
        ("environment", environment),
        ("paths", paths),
        ("deadlines", deadlines),
        ("failure_budget", failure_budget),
    ):
        if not isinstance(item, Mapping):
            raise ValueError(
                f"Expected profile.{label} to be an object. "
                f"Provided value: {item!r}."
            )
    _text(transport.get("ssh_target"), "transport.ssh_target")
    _absolute(transport.get("ssh_config"), "transport.ssh_config")
    for key in ("account", "partition", "qos", "constraint", "hint"):
        _text(scheduler.get(key), f"scheduler.{key}")
    for key in ("nodes", "tasks", "gpus", "cpus_per_task"):
        _positive_int(scheduler.get(key), f"scheduler.{key}")
    if scheduler.get("array") != "0-0":
        raise ValueError(
            "Expected scheduler.array='0-0'. "
            f"Provided value: {scheduler.get('array')!r}."
        )
    _text(scheduler.get("time_limit"), "scheduler.time_limit")
    for key in ("project", "module"):
        _text(environment.get(key), f"environment.{key}")
    _absolute(environment.get("python"), "environment.python")
    required_imports = environment.get("required_imports")
    if not (
        isinstance(required_imports, list)
        and required_imports
        and all(isinstance(item, str) and item for item in required_imports)
    ):
        raise ValueError(
            "Expected environment.required_imports to be a non-empty list. "
            f"Provided value: {required_imports!r}."
        )
    for key in ("remote_stage_root", "remote_result_root", "data_root"):
        _absolute(paths.get(key), f"paths.{key}")
    for key in (
        "local_stage_root",
        "local_collection_root",
        "transfer_receipt_root",
    ):
        local = Path(_text(paths.get(key), f"paths.{key}"))
        if local.is_absolute() or ".." in local.parts:
            raise ValueError(
                f"Expected paths.{key} to be repository-relative. "
                f"Provided value: {paths.get(key)!r}."
            )
    for key in ("preparation_seconds", "scheduler_start_seconds", "poll_seconds"):
        _positive_int(deadlines.get(key), f"deadlines.{key}")
    if deadlines["poll_seconds"] > 60:
        raise ValueError(
            "Expected deadlines.poll_seconds <= 60. "
            f"Provided value: {deadlines['poll_seconds']!r}."
        )
    if failure_budget.get("prearm_repair_attempts") != 1:
        raise ValueError(
            "Expected failure_budget.prearm_repair_attempts=1. "
            f"Provided value: {failure_budget.get('prearm_repair_attempts')!r}."
        )
    if failure_budget.get("broad_tests_on_passing_path") is not False:
        raise ValueError(
            "Expected failure_budget.broad_tests_on_passing_path=false. "
            "Provided value: "
            f"{failure_budget.get('broad_tests_on_passing_path')!r}."
        )
    return profile


def load_profile(path: str | Path = DEFAULT_PROFILE) -> dict[str, Any]:
    return validate_profile(read_json(Path(path).expanduser().resolve()))


def _hash_tree(root: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.is_symlink():
            continue
        records.append(
            {
                "path": path.relative_to(root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return records


def _copy_python_tree(source: Path, destination: Path) -> None:
    for path in sorted(source.rglob("*.py")):
        if "__pycache__" in path.parts or path.is_symlink():
            continue
        target = destination / path.relative_to(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)


def _copy_runtime_source(repo_root: Path, destination: Path) -> None:
    for directory in ("experiments", "model", "training"):
        _copy_python_tree(repo_root / directory, destination / directory)
    source_runner = repo_root / STABLE_RUNNER_RELATIVE
    packaged_runner = destination / STABLE_RUNNER_RELATIVE
    packaged_runner.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_runner, packaged_runner)
    labs = destination / "labs"
    labs.mkdir(parents=True, exist_ok=True)
    for name in (
        "__init__.py",
        "custom_classes.py",
        "custom_minimizer.py",
        "datasets.py",
        "mnist_train.py",
    ):
        source = repo_root / "labs" / name
        if source.is_file():
            shutil.copy2(source, labs / name)
    if not packaged_runner.is_file():
        raise FileNotFoundError(
            "Expected the stable Jean Zay runner in packaged source. "
            f"Provided value: {packaged_runner}."
        )
    packaged_runner.chmod(0o755)


def discover_parent_bundle(
    *,
    repo_root: Path,
    parent_experiment_id: str,
    parent_config_path: str,
    selected_job_ids: Sequence[str],
) -> Path:
    """Return the newest complete bundle matching the parent and exact jobs."""

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
        raise ValueError(
            "Expected at least one complete staged parent bundle matching the "
            f"current scientific config and selected jobs. Provided value: "
            f"staging={staging}, config_sha256={config_sha256!r}, "
            f"jobs={list(selected_job_ids)!r}."
        )
    candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return candidates[0][2]


def _layout(
    *,
    repo_root: Path,
    profile: Mapping[str, Any],
    experiment_id: str,
    attempt_id: str,
) -> dict[str, str]:
    paths = profile["paths"]
    remote_stage = (
        Path(paths["remote_stage_root"]) / experiment_id / attempt_id
    )
    remote_result = (
        Path(paths["remote_result_root"]) / experiment_id / attempt_id
    )
    local_attempt = (
        repo_root
        / paths["local_stage_root"]
        / experiment_id
        / attempt_id
    )
    local_collection = (
        repo_root
        / paths["local_collection_root"]
        / experiment_id
        / attempt_id
        / "shards"
        / "jean-zay"
    )
    transfer_root = (
        repo_root
        / paths["transfer_receipt_root"]
        / experiment_id
        / attempt_id
    )
    return {
        "remote_stage": str(remote_stage),
        "remote_result": str(remote_result),
        "remote_runner": str(remote_stage / "source" / STABLE_RUNNER_RELATIVE),
        "local_attempt": str(local_attempt),
        "local_package": str(local_attempt / "package"),
        "local_archive": str(local_attempt / "package.tar.gz"),
        "local_receipts": str(local_attempt / "receipts"),
        "local_collection": str(local_collection),
        "transfer_root": str(transfer_root),
    }


def _request_title(request: Mapping[str, Any], experiment_id: str) -> str:
    answers = request.get("user_answers")
    if isinstance(answers, Mapping):
        value = answers.get("short_name")
        if isinstance(value, str) and value.strip():
            return " ".join(value.split())
    return experiment_id


def prepare_package(
    *,
    request_path: str | Path,
    experiment_id: str,
    attempt_id: str,
    profile_path: str | Path = DEFAULT_PROFILE,
    catalog_path: str | Path = DEFAULT_CATALOG,
    repo_root: str | Path = Path("."),
    bundle_override: str | Path | None = None,
    focused_test_receipt: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Create one immutable source+bundle+launch-spec package."""

    root = Path(repo_root).expanduser().resolve()
    request_file = Path(request_path).expanduser().resolve()
    experiment = _identifier(experiment_id, "experiment_id")
    attempt = _identifier(attempt_id, "attempt_id")
    profile_file = Path(profile_path).expanduser().resolve()
    profile = load_profile(profile_file)
    request = load_request(request_file)
    validation = validate_request(request, require_review=True)
    if validation.get("status") != "valid":
        raise ValueError(
            "Expected an approved valid experiment request. "
            f"Provided value: {validation!r}."
        )
    catalog = load_catalog(Path(catalog_path).expanduser().resolve())
    study = build_resolved_study(request, catalog, repo_root=root)
    study_validation = validate_resolved_study(study)
    if study_validation.get("status") != "valid":
        raise ValueError(
            "Expected a valid resolved continuation study. "
            f"Provided value: {study_validation!r}."
        )
    jobs = study["study"]["jobs"]
    if len(jobs) != 1:
        raise ValueError(
            "Expected the Jean Zay validation fast path to select exactly one "
            f"job. Provided value: {[item.get('job_id') for item in jobs]!r}."
        )
    job = jobs[0]
    if job.get("official_test_read") is not False:
        raise ValueError(
            "Expected the selected validation job to forbid official-test "
            f"access. Provided value: {job.get('official_test_read')!r}."
        )
    if study["study"]["tk_gate"].get("fresh_required") is True:
        raise ValueError(
            "Expected a fixed continuation that can reuse its accepted T/K "
            "gate. Provided value: fresh_required=true."
        )
    entry_id = _identifier(job.get("job_id"), "selected job_id")
    selected_ids = [entry_id]
    bundle = (
        Path(bundle_override).expanduser().resolve()
        if bundle_override is not None
        else discover_parent_bundle(
            repo_root=root,
            parent_experiment_id=study["study"]["parent_experiment_id"],
            parent_config_path=study["bindings_and_provenance"][
                "parent_config_path"
            ],
            selected_job_ids=selected_ids,
        )
    )
    manifest = read_json(bundle / "manifest.json")
    matches = [
        item
        for item in manifest.get("entries", [])
        if isinstance(item, Mapping) and item.get("entry_id") == entry_id
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected bundle manifest to contain {entry_id!r} exactly once. "
            f"Provided value: {len(matches)} matches."
        )

    layout = _layout(
        repo_root=root,
        profile=profile,
        experiment_id=experiment,
        attempt_id=attempt,
    )
    local_attempt = Path(layout["local_attempt"])
    if local_attempt.exists():
        raise FileExistsError(
            "Expected a fresh local attempt directory. "
            f"Provided value: {local_attempt}."
        )
    package = Path(layout["local_package"])
    package.mkdir(parents=True)
    source = package / "source"
    _copy_runtime_source(root, source)
    shutil.copytree(bundle, package / "bundle", symlinks=False)
    shutil.copy2(request_file, package / "approved-request.md")
    atomic_create_json(package / "resolved-study.json", study, canonical=True)

    remote_stage = Path(layout["remote_stage"])
    remote_result = Path(layout["remote_result"])
    launch_spec = {
        "schema_version": "jean-zay-validation-launch/v1",
        "experiment_id": experiment,
        "attempt_id": attempt,
        "entry_id": entry_id,
        "run_class": "validation",
        "target": "jean-zay",
        "study_path": str(remote_stage / "bundle" / "study.resolved.json"),
        "bundle_dir": str(remote_stage / "bundle"),
        "data_root": profile["paths"]["data_root"],
        "device": "cuda:0",
        "output_dir": str(remote_result / "job"),
        "official_test_read": False,
        "maximum_jobs": 1,
    }
    atomic_create_json(package / "launch-spec.json", launch_spec, canonical=True)
    package_manifest = {
        "schema_version": PACKAGE_SCHEMA,
        "experiment_id": experiment,
        "attempt_id": attempt,
        "entry_id": entry_id,
        "request_sha256": sha256_file(request_file),
        "profile_sha256": sha256_file(profile_file),
        "source_bundle": str(bundle),
        "files": _hash_tree(package),
    }
    atomic_create_json(
        package / "package-manifest.json",
        package_manifest,
        canonical=True,
    )

    archive_path = Path(layout["local_archive"])
    with tarfile.open(archive_path, "w:gz") as archive:
        for child in sorted(package.iterdir()):
            archive.add(child, arcname=child.name, recursive=True)
    receipt_dir = Path(layout["local_receipts"])
    receipt_dir.mkdir()
    package_receipt = {
        "schema_version": PACKAGE_SCHEMA,
        "status": "prepared",
        "prepared_at": _now(),
        "experiment_id": experiment,
        "attempt_id": attempt,
        "entry_id": entry_id,
        "request_path": str(request_file),
        "request_sha256": sha256_file(request_file),
        "profile_path": str(profile_file),
        "profile_sha256": sha256_file(profile_file),
        "archive_path": str(archive_path),
        "archive_sha256": sha256_file(archive_path),
        "layout": layout,
        "catalog_role": "parent_discovery_only",
        "tracker_role": "best_effort_observation_only",
        "focused_tests": (
            {"status": "not_run_direct_api"}
            if focused_test_receipt is None
            else dict(focused_test_receipt)
        ),
        "broad_tests_run": [],
    }
    atomic_create_json(
        receipt_dir / "package-receipt.json",
        package_receipt,
        canonical=True,
    )
    return {
        "request": request,
        "study": study,
        "profile": profile,
        "profile_path": str(profile_file),
        "package_receipt": package_receipt,
        "layout": layout,
        "title": _request_title(request, experiment),
    }


def runner_arguments(
    profile: Mapping[str, Any],
    layout: Mapping[str, str],
    *,
    preflight: bool = False,
) -> list[str]:
    values = [
        layout["remote_runner"],
        "--package-root",
        layout["remote_stage"],
        "--project",
        profile["environment"]["project"],
        "--module",
        profile["environment"]["module"],
        "--python",
        profile["environment"]["python"],
    ]
    if preflight:
        values.insert(1, "--preflight")
    return values


def sbatch_command(
    profile: Mapping[str, Any],
    layout: Mapping[str, str],
    *,
    attempt_id: str,
    test_only: bool = False,
) -> list[str]:
    scheduler = profile["scheduler"]
    result = layout["remote_result"]
    job_name = f"jzval-{attempt_id}"[:100]
    command = ["sbatch"]
    if test_only:
        command.append("--test-only")
    else:
        command.append("--parsable")
    command.extend(
        [
            f"--job-name={job_name}",
            f"--nodes={scheduler['nodes']}",
            f"--ntasks={scheduler['tasks']}",
            f"--gres=gpu:{scheduler['gpus']}",
            f"--cpus-per-task={scheduler['cpus_per_task']}",
            f"--hint={scheduler['hint']}",
            f"--time={scheduler['time_limit']}",
            f"--account={scheduler['account']}",
            f"--partition={scheduler['partition']}",
            f"--qos={scheduler['qos']}",
            f"--constraint={scheduler['constraint']}",
            f"--array={scheduler['array']}",
            f"--output={result}/slurm/%x-%A_%a.out",
            f"--error={result}/slurm/%x-%A_%a.err",
            *runner_arguments(profile, layout),
        ]
    )
    return command


def _ssh_prefix(profile: Mapping[str, Any]) -> list[str]:
    return [
        "ssh",
        "-F",
        profile["transport"]["ssh_config"],
        profile["transport"]["ssh_target"],
    ]


def _remote_command(
    profile: Mapping[str, Any],
    argv: Sequence[str],
) -> list[str]:
    return [*_ssh_prefix(profile), shlex.join(list(argv))]


def _scp_prefix(profile: Mapping[str, Any]) -> list[str]:
    return ["scp", "-F", profile["transport"]["ssh_config"]]


def _run_checked(
    command: Sequence[str],
    *,
    stage: str,
    timeout: float,
    runner: RunCommand = subprocess.run,
) -> subprocess.CompletedProcess[str]:
    try:
        completed = runner(
            list(command),
            text=True,
            capture_output=True,
            check=False,
            timeout=timeout,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise JeanZayValidationError(
            stage,
            f"Expected the command to complete within {timeout} seconds. "
            f"Provided value: {exc}.",
            command=command,
        ) from exc
    if completed.returncode != 0:
        raise JeanZayValidationError(
            stage,
            "Expected command exit code 0. "
            f"Provided value: exit={completed.returncode}, "
            f"stdout={completed.stdout!r}, stderr={completed.stderr!r}.",
            command=command,
        )
    return completed


def run_focused_tests(
    repo_root: str | Path,
    *,
    runner: RunCommand = subprocess.run,
) -> dict[str, Any]:
    """Run the complete three-test local launch gate and nothing broader."""

    root = Path(repo_root).expanduser().resolve()
    command = [sys.executable, "-m", "pytest", "-q", *FOCUSED_TESTS]
    environment = os.environ.copy()
    environment.update(
        {
            "KMP_DISABLE_SHM": "1",
            "KMP_SHM_DISABLE": "1",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
        }
    )
    started = time.monotonic()
    try:
        completed = runner(
            command,
            cwd=root,
            env=environment,
            text=True,
            capture_output=True,
            check=False,
            timeout=120,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise JeanZayValidationError(
            "focused_local_tests",
            f"Expected exactly three focused tests to finish. Provided value: {exc}.",
            command=command,
        ) from exc
    if completed.returncode != 0:
        raise JeanZayValidationError(
            "focused_local_tests",
            "Expected exactly three focused tests to pass. "
            f"Provided value: exit={completed.returncode}, "
            f"stdout={completed.stdout!r}, stderr={completed.stderr!r}.",
            command=command,
        )
    return {
        "status": "passed",
        "count": 3,
        "tests": list(FOCUSED_TESTS),
        "elapsed_seconds": time.monotonic() - started,
        "stdout": completed.stdout,
        "broad_tests_run": [],
    }


def _tracker_observation(
    prepared: Mapping[str, Any],
    *,
    status: str,
    detail: str,
) -> str | None:
    """Best-effort visibility; tracker failure never blocks the launch."""

    try:
        from experiments.experiment_tracker import upsert_tracker

        study = prepared["study"]["study"]
        request = prepared["request"]
        answers = request.get("user_answers")
        answers = answers if isinstance(answers, Mapping) else {}
        entry = {
            "experiment_id": prepared["package_receipt"]["experiment_id"],
            "title": prepared["title"],
            "testing": " ".join(str(study["purpose"]).split()),
            "where": (
                "Jean Zay; "
                + prepared["layout"]["remote_result"]
                + "; attempt "
                + prepared["package_receipt"]["attempt_id"]
            ),
            "status": status,
            "detail": detail,
        }
        return upsert_tracker(
            Path(__file__).resolve().parents[1]
            / "docs"
            / "current_experiments.md",
            entry,
        )
    except Exception as exc:  # tracker is explicitly observational
        return f"warning: tracker observation failed: {exc}"


def _stage_and_preflight(
    prepared: Mapping[str, Any],
    *,
    started: float,
    runner: RunCommand = subprocess.run,
) -> dict[str, Any]:
    profile = prepared["profile"]
    layout = prepared["layout"]
    archive = prepared["package_receipt"]["archive_path"]
    archive_sha = prepared["package_receipt"]["archive_sha256"]
    remote_stage = layout["remote_stage"]
    remote_result = layout["remote_result"]
    timeout = profile["deadlines"]["preparation_seconds"]

    _run_checked(
        _remote_command(profile, ["test", "!", "-e", remote_stage]),
        stage="remote_stage_freshness",
        timeout=30,
        runner=runner,
    )
    _run_checked(
        _remote_command(profile, ["test", "!", "-e", remote_result]),
        stage="remote_result_freshness",
        timeout=30,
        runner=runner,
    )
    _run_checked(
        _remote_command(
            profile,
            [
                "mkdir",
                "-p",
                remote_stage,
                f"{remote_result}/slurm",
                f"{remote_result}/receipts",
            ],
        ),
        stage="remote_directories",
        timeout=30,
        runner=runner,
    )
    remote_archive = f"{remote_stage}/package.tar.gz"
    _run_checked(
        [
            *_scp_prefix(profile),
            archive,
            f"{profile['transport']['ssh_target']}:{remote_archive}",
        ],
        stage="package_upload",
        timeout=timeout,
        runner=runner,
    )
    observed_hash = _run_checked(
        _remote_command(profile, ["sha256sum", remote_archive]),
        stage="package_hash",
        timeout=30,
        runner=runner,
    ).stdout.split()[0]
    if observed_hash != archive_sha:
        raise JeanZayValidationError(
            "package_hash",
            f"Expected remote package SHA-256 {archive_sha}. "
            f"Provided value: {observed_hash}.",
        )
    _run_checked(
        _remote_command(
            profile,
            ["tar", "-xzf", remote_archive, "-C", remote_stage],
        ),
        stage="package_extract",
        timeout=timeout,
        runner=runner,
    )
    _run_checked(
        _remote_command(profile, ["chmod", "755", layout["remote_runner"]]),
        stage="runner_permissions",
        timeout=30,
        runner=runner,
    )
    import_result = _run_checked(
        _remote_command(
            profile,
            runner_arguments(profile, layout, preflight=True),
        ),
        stage="remote_import_preflight",
        timeout=60,
        runner=runner,
    )
    lines = [
        line
        for line in import_result.stdout.splitlines()
        if line.strip().startswith("{")
    ]
    if len(lines) != 1:
        raise JeanZayValidationError(
            "remote_import_preflight",
            "Expected one JSON import-preflight marker. "
            f"Provided value: {import_result.stdout!r}.",
        )
    try:
        import_receipt = json.loads(lines[0])
    except json.JSONDecodeError as exc:
        raise JeanZayValidationError(
            "remote_import_preflight",
            f"Expected valid JSON import-preflight output. Provided value: {exc}.",
        ) from exc
    if (
        import_receipt.get("schema_version") != IMPORT_PREFLIGHT_SCHEMA
        or import_receipt.get("status") != "passed"
        or import_receipt.get("side_effects_performed") is not False
    ):
        raise JeanZayValidationError(
            "remote_import_preflight",
            "Expected a passing side-effect-free import receipt. "
            f"Provided value: {import_receipt!r}.",
        )
    preflight_path = (
        Path(layout["local_receipts"]) / "import-preflight.json"
    )
    atomic_create_json(preflight_path, import_receipt, canonical=True)
    _run_checked(
        [
            *_scp_prefix(profile),
            str(preflight_path),
            (
                f"{profile['transport']['ssh_target']}:"
                f"{remote_result}/receipts/import-preflight.json"
            ),
        ],
        stage="preflight_receipt_publish",
        timeout=30,
        runner=runner,
    )
    test_only = sbatch_command(
        profile,
        layout,
        attempt_id=prepared["package_receipt"]["attempt_id"],
        test_only=True,
    )
    test_result = _run_checked(
        _remote_command(profile, test_only),
        stage="sbatch_test_only",
        timeout=60,
        runner=runner,
    )
    elapsed = time.monotonic() - started
    if elapsed > profile["deadlines"]["preparation_seconds"]:
        raise JeanZayValidationError(
            "preparation_deadline",
            "Expected preparation to finish before submission within "
            f"{profile['deadlines']['preparation_seconds']} seconds. "
            f"Provided value: {elapsed:.3f} seconds.",
        )
    return {
        "import_preflight": import_receipt,
        "sbatch_test_only_stdout": test_result.stdout,
        "elapsed_seconds": elapsed,
    }


def _scheduler_state(
    profile: Mapping[str, Any],
    job_id: str,
    *,
    runner: RunCommand = subprocess.run,
) -> dict[str, str]:
    queued = _run_checked(
        _remote_command(
            profile,
            [
                "squeue",
                f"--jobs={job_id}",
                "--array",
                "--noheader",
                "--format=%T|%R",
            ],
        ),
        stage="scheduler_poll",
        timeout=30,
        runner=runner,
    ).stdout.strip()
    if queued:
        state, _, reason = queued.partition("|")
        return {"state": state.splitlines()[0], "reason": reason}
    accounting = _run_checked(
        _remote_command(
            profile,
            [
                "sacct",
                "-j",
                job_id,
                "--noheader",
                "--parsable2",
                "--format=JobIDRaw,State,ExitCode",
            ],
        ),
        stage="accounting_poll",
        timeout=30,
        runner=runner,
    ).stdout
    for line in accounting.splitlines():
        raw, separator, remainder = line.partition("|")
        if separator and raw == job_id:
            state, _, exit_code = remainder.partition("|")
            return {"state": state.split()[0], "reason": exit_code}
    return {"state": "UNKNOWN", "reason": "no scheduler row"}


def _monitor(
    prepared: Mapping[str, Any],
    job_id: str,
    *,
    runner: RunCommand = subprocess.run,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, str]:
    profile = prepared["profile"]
    poll = profile["deadlines"]["poll_seconds"]
    start_deadline = (
        time.monotonic() + profile["deadlines"]["scheduler_start_seconds"]
    )
    running = False
    while True:
        observed = _scheduler_state(profile, job_id, runner=runner)
        state = observed["state"]
        print(
            json.dumps(
                {
                    "job_id": job_id,
                    "state": state,
                    "reason": observed["reason"],
                    "observed_at": _now(),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        if state == "RUNNING":
            if not running:
                _tracker_observation(
                    prepared,
                    status="running",
                    detail=f"Jean Zay job {job_id} is running.",
                )
            running = True
        elif state in TERMINAL_STATES:
            if state != "COMPLETED":
                raise JeanZayValidationError(
                    "monitor",
                    f"Expected job {job_id} to complete. "
                    f"Provided value: state={state}, reason={observed['reason']}.",
                    job_id=job_id,
                )
            return observed
        elif not running and time.monotonic() >= start_deadline:
            raise JeanZayValidationError(
                "scheduler_start_deadline",
                f"Expected job {job_id} to start within "
                f"{profile['deadlines']['scheduler_start_seconds']} seconds. "
                f"Provided value: state={state}, reason={observed['reason']}.",
                job_id=job_id,
            )
        sleep(poll)


def _collect(
    prepared: Mapping[str, Any],
    job_id: str,
    *,
    runner: RunCommand = subprocess.run,
) -> dict[str, str]:
    root = Path(__file__).resolve().parents[1]
    profile = prepared["profile"]
    layout = prepared["layout"]
    collection = Path(layout["local_collection"])
    transfer = Path(layout["transfer_root"])
    transfer.mkdir(parents=True, exist_ok=True)
    sync = (
        root
        / "skills"
        / "sync-remote-results"
        / "scripts"
        / "sync_remote_results.py"
    )
    common = [
        str(sync),
        "--host",
        profile["transport"]["ssh_target"],
        "--remote-path",
        layout["remote_result"],
        "--local-stage",
        str(collection),
        "--completion-marker",
        "job/functional_preflight.json",
    ]
    _tracker_observation(
        prepared,
        status="transferring",
        detail=f"Jean Zay job {job_id} is terminal; pulling metadata first.",
    )
    metadata_receipt = transfer / "jean-zay-metadata.json"
    _run_checked(
        [
            str(Path("/home/filip/miniconda3/envs/py312/bin/python")),
            *common,
            "--receipt",
            str(metadata_receipt),
            "--mode",
            "metadata",
        ],
        stage="metadata_transfer",
        timeout=300,
        runner=runner,
    )
    local_marker = collection / "job" / "functional_preflight.json"
    validate_functional_preflight(read_json(local_marker))
    _tracker_observation(
        prepared,
        status="validating",
        detail=f"Metadata for Jean Zay job {job_id} is local and valid.",
    )
    full_receipt = transfer / "jean-zay-full.json"
    _run_checked(
        [
            str(Path("/home/filip/miniconda3/envs/py312/bin/python")),
            *common,
            "--receipt",
            str(full_receipt),
            "--mode",
            "full",
        ],
        stage="full_transfer",
        timeout=900,
        runner=runner,
    )
    validate_functional_preflight(read_json(local_marker))
    _tracker_observation(
        prepared,
        status="review-pending",
        detail=(
            f"Jean Zay validation job {job_id} completed; metadata and full "
            "attempt snapshot are locally validated."
        ),
    )
    return {
        "local_collection": str(collection),
        "metadata_receipt": str(metadata_receipt),
        "full_receipt": str(full_receipt),
    }


def launch(
    prepared: Mapping[str, Any],
    *,
    wait: bool = True,
    collect: bool = True,
    started_at_monotonic: float | None = None,
    runner: RunCommand = subprocess.run,
) -> dict[str, Any]:
    """Stage, preflight, submit exactly once, monitor, and optionally collect."""

    started = (
        time.monotonic()
        if started_at_monotonic is None
        else started_at_monotonic
    )
    attempt_id = prepared["package_receipt"]["attempt_id"]
    profile = prepared["profile"]
    layout = prepared["layout"]
    tracker = _tracker_observation(
        prepared,
        status="preflighting",
        detail=(
            "Approved validation package is being staged; tracker state is "
            "informational and is not launch authority."
        ),
    )
    gates = _stage_and_preflight(prepared, started=started, runner=runner)
    command = sbatch_command(
        profile,
        layout,
        attempt_id=attempt_id,
        test_only=False,
    )
    print(
        "LONG-RUN ATTEMPT ARMED — "
        f"experiment {prepared['package_receipt']['experiment_id']}; "
        f"attempt {attempt_id}; Jean Zay singleton validation; "
        f"scheduler-start deadline "
        f"{profile['deadlines']['scheduler_start_seconds']} seconds; "
        f"immutable package {prepared['package_receipt']['archive_sha256']}.",
        flush=True,
    )
    submitted = _run_checked(
        _remote_command(profile, command),
        stage="sbatch_submit",
        timeout=60,
        runner=runner,
    )
    job_id = submitted.stdout.strip().split(";", 1)[0]
    if not job_id.isdigit():
        raise JeanZayValidationError(
            "sbatch_submit",
            f"Expected a numeric Slurm job ID. Provided value: {submitted.stdout!r}.",
            command=command,
        )
    readback = _run_checked(
        _remote_command(profile, ["scontrol", "show", "job", job_id]),
        stage="immediate_readback",
        timeout=30,
        runner=runner,
    )
    expected_tokens = (
        f"Account={profile['scheduler']['account']}",
        f"QOS={profile['scheduler']['qos']}",
        f"Partition={profile['scheduler']['partition']}",
        f"Features={profile['scheduler']['constraint']}",
        f"Command={layout['remote_runner']}",
    )
    missing = [token for token in expected_tokens if token not in readback.stdout]
    if missing:
        raise JeanZayValidationError(
            "immediate_readback",
            f"Expected scheduler readback tokens {list(expected_tokens)!r}. "
            f"Provided value: missing={missing!r}.",
            job_id=job_id,
        )
    submission = {
        "schema_version": "jean-zay-validation-submission/v1",
        "status": "queued",
        "submitted_at": _now(),
        "job_id": job_id,
        "command": command,
        "readback": readback.stdout,
        "gates": gates,
        "tracker_observation": tracker,
        "retry_authorized": False,
    }
    submission_path = Path(layout["local_receipts"]) / "submission.json"
    atomic_create_json(submission_path, submission, canonical=True)
    _run_checked(
        [
            *_scp_prefix(profile),
            str(submission_path),
            (
                f"{profile['transport']['ssh_target']}:"
                f"{layout['remote_result']}/receipts/submission.json"
            ),
        ],
        stage="submission_receipt_publish",
        timeout=30,
        runner=runner,
    )
    _tracker_observation(
        prepared,
        status="queued",
        detail=f"Jean Zay singleton validation job {job_id} is queued.",
    )
    if not wait:
        return submission
    terminal = _monitor(prepared, job_id, runner=runner)
    _tracker_observation(
        prepared,
        status="remote-closed",
        detail=f"Jean Zay job {job_id} completed; remote writer is closed.",
    )
    completion_check = _run_checked(
        _remote_command(
            profile,
            [
                "test",
                "-f",
                f"{layout['remote_result']}/job/functional_preflight.json",
            ],
        ),
        stage="completion_marker",
        timeout=30,
        runner=runner,
    )
    result = {
        **submission,
        "status": "completed",
        "terminal": terminal,
        "completion_marker_check": completion_check.returncode,
    }
    if collect:
        result["collection"] = _collect(
            prepared,
            job_id,
            runner=runner,
        )
    return result


def _failure_record(
    error: Exception,
    *,
    prepared: Mapping[str, Any] | None,
) -> dict[str, Any]:
    stage = (
        error.stage
        if isinstance(error, JeanZayValidationError)
        else "prepare"
    )
    job_id = (
        error.job_id if isinstance(error, JeanZayValidationError) else None
    )
    command = (
        error.command if isinstance(error, JeanZayValidationError) else None
    )
    return {
        "schema_version": FAILURE_SCHEMA,
        "status": "failed",
        "failed_at": _now(),
        "stage": stage,
        "error_type": type(error).__name__,
        "error": str(error),
        "command": command,
        "job_id": job_id,
        "retry_attempted": False,
        "broad_tests_run": [],
        "next_step": "report_exact_blocker_and_ask_user",
        "attempt_id": (
            None
            if prepared is None
            else prepared["package_receipt"]["attempt_id"]
        ),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare or launch one approved Jean Zay validation."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare", "launch"):
        child = commands.add_parser(name)
        child.add_argument("--request", required=True, type=Path)
        child.add_argument("--experiment-id", required=True)
        child.add_argument("--attempt-id", required=True)
        child.add_argument("--profile", type=Path, default=DEFAULT_PROFILE)
        child.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
        child.add_argument("--repo-root", type=Path, default=Path("."))
        child.add_argument("--bundle", type=Path)
    launch_parser = commands.choices["launch"]
    launch_parser.add_argument("--no-wait", action="store_true")
    launch_parser.add_argument("--no-collect", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    prepared: dict[str, Any] | None = None
    command_started = time.monotonic()
    try:
        focused_tests = run_focused_tests(args.repo_root)
        prepared = prepare_package(
            request_path=args.request,
            experiment_id=args.experiment_id,
            attempt_id=args.attempt_id,
            profile_path=args.profile,
            catalog_path=args.catalog,
            repo_root=args.repo_root,
            bundle_override=args.bundle,
            focused_test_receipt=focused_tests,
        )
        if args.command == "prepare":
            result = {
                "schema_version": PACKAGE_SCHEMA,
                "status": "prepared",
                "package_receipt": prepared["package_receipt"],
            }
        else:
            result = launch(
                prepared,
                wait=not args.no_wait,
                collect=not args.no_collect,
                started_at_monotonic=command_started,
            )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except Exception as exc:
        failure = _failure_record(exc, prepared=prepared)
        if prepared is not None:
            failure_path = (
                Path(prepared["layout"]["local_receipts"]) / "failure.json"
            )
            if not failure_path.exists():
                atomic_create_json(failure_path, failure, canonical=True)
            if failure["job_id"] is None:
                _tracker_observation(
                    prepared,
                    status="blocked",
                    detail=(
                        f"Pre-arm fast path stopped at {failure['stage']}: "
                        f"{failure['error']}"
                    ),
                )
        print(json.dumps(failure, indent=2, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "JeanZayValidationError",
    "launch",
    "load_profile",
    "prepare_package",
    "runner_arguments",
    "sbatch_command",
    "validate_profile",
]
