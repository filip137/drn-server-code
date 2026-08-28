"""Persistent formal launcher for the IBM OM baseline-selection study.

The launcher is deliberately only an orchestration surface.  Every numerical
task enters through the public ``python -m ebl validate`` command, in the
predeclared study arm, with the frozen ReLU checkpoint supplied through
``--weights``.  Failed native attempts are retained and an exact completed
config is the only reason a later launcher attempt may skip work.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import signal
import subprocess
import time
from typing import Any, Mapping, Sequence

from experiments.artifacts import atomic_write_json, sha256_file
from experiments.study_workflow import load_study_plan, load_study_record


_ROOT = Path(__file__).resolve().parents[2]
STUDY_ID = "mnist-ibm-om-four-device-baseline-selection-20260828-v1"
EXPERIMENT_ID = "mnist_ibm_om_baseline_selection.v1"
PLAN = _ROOT / "studies" / f"{STUDY_ID}.json"
CONFIG_ROOT = (
    _ROOT / "examples" / "mnist_relu_drn" / "ibm_om_baseline_selection"
)
TEACHER = _ROOT / "data" / "mnist_relu_teacher_fixed_init_20260816.pt"
EXPECTED_TEACHER_SHA256 = (
    "9a961a77628e365b54fdf59304ec7fddf46f1f4834c4c03cecea9c204f837f52"
)
TASK_PYTHON = Path("/home/filip/miniconda3/envs/py312/bin/python")
AIHWKIT_PYTHON = Path("/home/filip/miniconda3/envs/aihwkit/bin/python")
EXPECTED_AIHWKIT_VERSION = "1.1.0"
HELDOUT_SEEDS = (87001, 87002, 87003)
HEARTBEAT_SECONDS = 15.0
MONITOR_INTERVAL_SECONDS = 1800

# Ordered to make the sequential launch and every receipt deterministic.
ARM_POLICIES: tuple[tuple[str, str], ...] = (
    ("independent-cell-reset-mean", "independent_cell_reset_mean"),
    ("shared-quad-reset-max", "shared_quad_reset_max"),
    (
        "shared-destination-columns-reset-max",
        "shared_destination_columns_reset_max",
    ),
    (
        "reference-enforced-destination-columns",
        "reference_enforced_destination_columns",
    ),
)


@dataclass(frozen=True)
class Task:
    arm_id: str
    policy: str
    heldout_seed: int
    config: Path

    @property
    def label(self) -> str:
        return f"{self.arm_id}.heldout-{self.heldout_seed}"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _attempt_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--study-dir",
        type=Path,
        help=f"prepared results/{STUDY_ID} directory",
    )
    return parser


def _read_json(path: Path) -> Mapping[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, Mapping) else None


def _require_executable(path: Path, *, label: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_file() or not os.access(resolved, os.X_OK):
        raise RuntimeError(f"Expected {label} to be executable: {resolved}.")
    return resolved


def _require_clean_source_commit(repo_root: Path = _ROOT) -> str:
    """Return HEAD after failing closed on tracked or untracked source work."""

    try:
        commit = subprocess.check_output(
            ("git", "-C", str(repo_root), "rev-parse", "HEAD"),
            stderr=subprocess.PIPE,
        ).decode("ascii").strip()
        status = subprocess.check_output(
            (
                "git",
                "-C",
                str(repo_root),
                "status",
                "--porcelain=v1",
                "--untracked-files=all",
                "-z",
            ),
            stderr=subprocess.PIPE,
        )
    except (OSError, subprocess.CalledProcessError, UnicodeError) as error:
        raise RuntimeError("Expected a clean resolvable Git launch commit.") from error
    if not commit or status:
        raise RuntimeError(
            "Expected the formal baseline-selection launcher to use a clean "
            "worktree. Commit every source, config, and study change first."
        )
    return commit


def tasks() -> tuple[Task, ...]:
    return tuple(
        Task(
            arm_id=arm_id,
            policy=policy,
            heldout_seed=seed,
            config=CONFIG_ROOT / f"{policy}-heldout-{seed}.json",
        )
        for arm_id, policy in ARM_POLICIES
        for seed in HELDOUT_SEEDS
    )


def _nested(value: Mapping[str, Any], *keys: str) -> Any:
    current: Any = value
    for key in keys:
        if not isinstance(current, Mapping) or key not in current:
            return None
        current = current[key]
    return current


def _validate_task_config(task: Task) -> str:
    config = _read_json(task.config)
    if config is None:
        raise RuntimeError(f"Expected strict JSON config: {task.config}.")
    checks = {
        "schema_version": (config.get("schema_version"), 1),
        "experiment_id": (config.get("experiment_id"), EXPERIMENT_ID),
        "execution profile": (
            _nested(config, "baseline_selection", "execution", "profile"),
            "production",
        ),
        "runtime device": (config.get("runtime", {}).get("device"), "cuda"),
        "selected baseline policy": (
            _nested(
                config,
                "baseline_selection",
                "baseline",
                "selected_policy",
            ),
            task.policy,
        ),
        "held-out assignment seed": (
            _nested(
                config,
                "baseline_selection",
                "assignments",
                "heldout_seed",
            ),
            task.heldout_seed,
        ),
        "source checkpoint hash": (
            _nested(
                config,
                "baseline_selection",
                "source",
                "expected_weights_sha256",
            ),
            EXPECTED_TEACHER_SHA256,
        ),
        "optimizer updates": (
            _nested(
                config,
                "baseline_selection",
                "exclusions",
                "optimizer_updates",
            ),
            0,
        ),
    }
    mismatches = [
        f"{label}: observed={observed!r}, expected={expected!r}"
        for label, (observed, expected) in checks.items()
        if observed != expected
    ]
    if mismatches:
        raise RuntimeError(
            f"Config does not match formal task {task.label!r}: "
            + "; ".join(mismatches)
        )
    return sha256_file(task.config)


def _validate_prepared_study(study_dir: Path) -> dict[str, str]:
    """Validate tracked-plan identity and exact 4 x 3 production coverage."""

    plan = load_study_plan(PLAN)
    study = load_study_record(study_dir)
    if plan["study_id"] != STUDY_ID or study["study_id"] != STUDY_ID:
        raise RuntimeError(f"Expected the prepared {STUDY_ID!r} study.")
    if study["source_plan"]["sha256"] != sha256_file(PLAN):
        raise RuntimeError("Prepared study does not match the tracked study plan hash.")

    expected_tasks = tasks()
    expected_by_arm = {
        arm_id: [task for task in expected_tasks if task.arm_id == arm_id]
        for arm_id, _policy in ARM_POLICIES
    }
    plan_arms = {arm["arm_id"]: arm for arm in plan["arms"]}
    study_arms = {arm["arm_id"]: arm for arm in study["arms"]}
    expected_arm_ids = set(expected_by_arm)
    if set(plan_arms) != expected_arm_ids or set(study_arms) != expected_arm_ids:
        raise RuntimeError(
            "Expected tracked and prepared studies to declare exactly the four "
            "reviewed baseline-policy arms."
        )

    config_hashes: dict[str, str] = {}
    for arm_id, arm_tasks in expected_by_arm.items():
        plan_arm = plan_arms[arm_id]
        study_arm = study_arms[arm_id]
        for label, arm in (("tracked", plan_arm), ("prepared", study_arm)):
            if arm["experiment_id"] != EXPERIMENT_ID or arm["mode"] != "validate":
                raise RuntimeError(
                    f"Expected {label} arm {arm_id!r} to be validate-only "
                    f"{EXPERIMENT_ID!r}."
                )
            if len(arm["configs"]) != len(HELDOUT_SEEDS):
                raise RuntimeError(
                    f"Expected {label} arm {arm_id!r} to declare exactly three configs."
                )
        expected_hashes = {_validate_task_config(task) for task in arm_tasks}
        tracked_hashes = {item["sha256"] for item in plan_arm["configs"]}
        prepared_hashes = {item["sha256"] for item in study_arm["configs"]}
        if tracked_hashes != expected_hashes or prepared_hashes != expected_hashes:
            raise RuntimeError(
                f"Config coverage mismatch for arm {arm_id!r}; expected the "
                "three local production config hashes exactly."
            )
        for task in arm_tasks:
            config_hashes[task.label] = sha256_file(task.config)

    discovered_production = {
        path.resolve()
        for path in CONFIG_ROOT.glob("*.json")
        if not path.name.startswith("smoke-")
    }
    expected_paths = {task.config.resolve() for task in expected_tasks}
    if discovered_production != expected_paths:
        raise RuntimeError(
            "Expected the production config directory to contain exactly the "
            "twelve declared configs; smoke-* files are explicitly excluded."
        )
    return config_hashes


def _validate_prerequisites() -> dict[str, Any]:
    task_python = _require_executable(TASK_PYTHON, label="task Python")
    aihwkit_python = _require_executable(
        AIHWKIT_PYTHON,
        label="AIHWKit Python",
    )
    if not TEACHER.is_file():
        raise RuntimeError(f"Missing frozen teacher/source checkpoint: {TEACHER}.")
    teacher_sha = sha256_file(TEACHER)
    if teacher_sha != EXPECTED_TEACHER_SHA256:
        raise RuntimeError(
            "Frozen teacher/source checkpoint hash mismatch: "
            f"expected {EXPECTED_TEACHER_SHA256}, observed {teacher_sha}."
        )
    probe = subprocess.run(
        (
            str(aihwkit_python),
            "-c",
            "import aihwkit; print(aihwkit.__version__)",
        ),
        check=False,
        capture_output=True,
        text=True,
    )
    if probe.returncode != 0 or probe.stdout.strip() != EXPECTED_AIHWKIT_VERSION:
        raise RuntimeError(
            f"Expected pinned AIHWKit {EXPECTED_AIHWKIT_VERSION}; "
            f"stdout={probe.stdout!r}, stderr={probe.stderr!r}."
        )
    cuda_probe = subprocess.run(
        (
            str(task_python),
            "-c",
            (
                "import json, torch; "
                "print(json.dumps({'torch_version': torch.__version__, "
                "'cuda_available': torch.cuda.is_available(), "
                "'cuda_device_count': torch.cuda.device_count(), "
                "'cuda_device_name': (torch.cuda.get_device_name(0) "
                "if torch.cuda.is_available() else None)}))"
            ),
        ),
        check=False,
        capture_output=True,
        text=True,
    )
    try:
        cuda = json.loads(cuda_probe.stdout)
    except json.JSONDecodeError as error:
        raise RuntimeError(
            "Expected the task interpreter to return a readable CUDA probe."
        ) from error
    if (
        cuda_probe.returncode != 0
        or not isinstance(cuda, Mapping)
        or cuda.get("cuda_available") is not True
        or not isinstance(cuda.get("cuda_device_count"), int)
        or cuda["cuda_device_count"] < 1
        or not isinstance(cuda.get("cuda_device_name"), str)
    ):
        raise RuntimeError(
            "Expected at least one CUDA GPU for the formal study; "
            f"probe={cuda!r}, stderr={cuda_probe.stderr!r}."
        )
    return {
        "task_python": str(task_python),
        "teacher": str(TEACHER.resolve()),
        "teacher_sha256": teacher_sha,
        "aihwkit_python": str(aihwkit_python),
        "aihwkit_version": EXPECTED_AIHWKIT_VERSION,
        "cuda": dict(cuda),
    }


def _task_environment(base: Mapping[str, str]) -> dict[str, str]:
    environment = dict(base)
    environment["EBL_AIHWKIT_PYTHON"] = str(AIHWKIT_PYTHON)
    environment["EBL_DEFER_CURRENT_SIMULATIONS"] = "1"
    environment["PYTHONUNBUFFERED"] = "1"
    return environment


def _command(task: Task, *, study_dir: Path) -> list[str]:
    return [
        str(TASK_PYTHON),
        "-m",
        "ebl",
        "validate",
        "--config",
        str(task.config.resolve()),
        "--output-dir",
        str((study_dir / "runs" / task.arm_id).resolve()),
        "--weights",
        str(TEACHER.resolve()),
    ]


def _native_runs(arm_root: Path) -> list[Path]:
    if not arm_root.is_dir():
        return []
    return sorted(
        child
        for child in arm_root.iterdir()
        if child.is_dir()
        and (child / "manifest.json").is_file()
        and (child / "status.json").is_file()
    )


def _artifact_records_are_intact(
    run_dir: Path,
    result: Mapping[str, Any],
) -> bool:
    artifacts = result.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        return False
    root = run_dir.resolve()
    for record in artifacts:
        if not isinstance(record, Mapping):
            return False
        relative = record.get("path")
        if not isinstance(relative, str) or not relative:
            return False
        try:
            path = (run_dir / relative).resolve(strict=True)
            path.relative_to(root)
        except (OSError, ValueError):
            return False
        if (
            not path.is_file()
            or record.get("size_bytes") != path.stat().st_size
            or record.get("sha256") != sha256_file(path)
        ):
            return False
    return True


def _matching_run(
    arm_root: Path,
    *,
    config_sha256: str,
    source_commit: str | None = None,
    require_complete: bool,
) -> Path | None:
    matches: list[Path] = []
    for run_dir in _native_runs(arm_root):
        manifest = _read_json(run_dir / "manifest.json")
        status = _read_json(run_dir / "status.json")
        study = manifest.get("study") if manifest is not None else None
        source = manifest.get("source") if manifest is not None else None
        if (
            manifest is None
            or status is None
            or manifest.get("experiment_id") != EXPERIMENT_ID
            or not isinstance(study, Mapping)
            or study.get("study_id") != STUDY_ID
            or study.get("arm_id") != arm_root.name
            or study.get("source_config_sha256") != config_sha256
            or (
                source_commit is not None
                and (
                    not isinstance(source, Mapping)
                    or source.get("commit") != source_commit
                    or source.get("dirty") is not False
                )
            )
        ):
            continue
        if not require_complete:
            matches.append(run_dir)
            continue
        result = _read_json(run_dir / "result.json")
        if (
            status.get("status") == "complete"
            and result is not None
            and result.get("status") == "complete"
            and result.get("run_id") == run_dir.name
            and _artifact_records_are_intact(run_dir, result)
        ):
            matches.append(run_dir)
    if require_complete and len(matches) > 1:
        raise RuntimeError(
            "Expected at most one exact completed run for config hash "
            f"{config_sha256}; found {[str(path) for path in matches]!r}."
        )
    return matches[-1] if matches else None


def _running_run(arm_root: Path, *, config_sha256: str) -> Path | None:
    matches = []
    for run_dir in _native_runs(arm_root):
        manifest = _read_json(run_dir / "manifest.json")
        status = _read_json(run_dir / "status.json")
        study = manifest.get("study") if manifest is not None else None
        if (
            isinstance(study, Mapping)
            and study.get("study_id") == STUDY_ID
            and study.get("arm_id") == arm_root.name
            and study.get("source_config_sha256") == config_sha256
            and status is not None
            and status.get("status") == "running"
        ):
            matches.append(run_dir)
    if len(matches) > 1:
        raise RuntimeError(
            "Expected at most one running native attempt for config hash "
            f"{config_sha256}; found {[str(path) for path in matches]!r}."
        )
    return matches[0] if matches else None


def _native_snapshot(
    task: Task,
    *,
    study_dir: Path,
    config_sha256: str,
    process: subprocess.Popen[str],
) -> dict[str, Any]:
    run_dir = _matching_run(
        study_dir / "runs" / task.arm_id,
        config_sha256=config_sha256,
        require_complete=False,
    )
    status = _read_json(run_dir / "status.json") if run_dir is not None else None
    artifact_root = run_dir / "artifacts" if run_dir is not None else None
    artifact_files = (
        [path for path in artifact_root.rglob("*") if path.is_file()]
        if artifact_root is not None and artifact_root.is_dir()
        else []
    )
    metrics = run_dir / "metrics.jsonl" if run_dir is not None else None
    return {
        "arm_id": task.arm_id,
        "policy": task.policy,
        "heldout_seed": task.heldout_seed,
        "pid": process.pid,
        "launcher_returncode": process.poll(),
        "native_run_dir": str(run_dir) if run_dir is not None else None,
        "native_status": status.get("status") if status is not None else None,
        "native_started_at": (
            status.get("started_at") if status is not None else None
        ),
        "artifact_file_count": len(artifact_files),
        "artifact_size_bytes": sum(path.stat().st_size for path in artifact_files),
        "metrics_size_bytes": (
            metrics.stat().st_size if metrics is not None and metrics.is_file() else 0
        ),
        "terminal_result_present": (
            run_dir is not None and (run_dir / "result.json").is_file()
        ),
    }


def _terminate_process(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass


def _persist_task_result(attempt_dir: Path, task: Task, result: Mapping[str, Any]) -> None:
    atomic_write_json(attempt_dir / f"{task.label}.result.json", dict(result))


def _run_task(
    task: Task,
    *,
    study_dir: Path,
    attempt_dir: Path,
    environment: Mapping[str, str],
    source_commit: str,
    stop_requested: list[bool],
) -> dict[str, Any]:
    config_sha = sha256_file(task.config)
    exact = _matching_run(
        study_dir / "runs" / task.arm_id,
        config_sha256=config_sha,
        source_commit=source_commit,
        require_complete=True,
    )
    if exact is not None:
        result = {
            "arm_id": task.arm_id,
            "policy": task.policy,
            "heldout_seed": task.heldout_seed,
            "config": str(task.config.resolve()),
            "config_sha256": config_sha,
            "status": "skipped_exact_completed",
            "run_dir": str(exact),
            "finished_at": _utc_now(),
        }
        _persist_task_result(attempt_dir, task, result)
        return result

    other_source = _matching_run(
        study_dir / "runs" / task.arm_id,
        config_sha256=config_sha,
        require_complete=True,
    )
    if other_source is not None:
        raise RuntimeError(
            "Refusing to mix clean source commits in one formal study: exact "
            f"config {task.label!r} already completed at {other_source}, but "
            "its launch commit differs from the current commit."
        )
    running = _running_run(
        study_dir / "runs" / task.arm_id,
        config_sha256=config_sha,
    )
    if running is not None:
        raise RuntimeError(
            "Refusing to launch a duplicate native task while a matching run "
            f"is still marked running: {running}. Reconcile that attempt first."
        )

    command = _command(task, study_dir=study_dir)
    log_path = attempt_dir / f"{task.label}.log"
    started = time.monotonic()
    launched_at = _utc_now()
    with log_path.open("w", encoding="utf-8", buffering=1) as log:
        process = subprocess.Popen(
            command,
            cwd=_ROOT,
            env=dict(environment),
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        atomic_write_json(
            attempt_dir / f"{task.label}.launch.json",
            {
                "arm_id": task.arm_id,
                "policy": task.policy,
                "heldout_seed": task.heldout_seed,
                "config": str(task.config.resolve()),
                "config_sha256": config_sha,
                "command": command,
                "pid": process.pid,
                "started_at": launched_at,
                "log": str(log_path),
            },
        )
        while True:
            snapshot = _native_snapshot(
                task,
                study_dir=study_dir,
                config_sha256=config_sha,
                process=process,
            )
            atomic_write_json(
                attempt_dir / "heartbeat.json",
                {
                    "schema": "ebl.mnist_ibm_om_baseline_selection.heartbeat",
                    "schema_version": 1,
                    "study_id": STUDY_ID,
                    "launcher_pid": os.getpid(),
                    "updated_at": _utc_now(),
                    "heartbeat_seconds": HEARTBEAT_SECONDS,
                    "active_task": task.label,
                    "elapsed_task_seconds": time.monotonic() - started,
                    "stop_requested": stop_requested[0],
                    "native": snapshot,
                },
            )
            if process.poll() is not None:
                break
            if stop_requested[0]:
                _terminate_process(process)
            try:
                process.wait(timeout=HEARTBEAT_SECONDS)
            except subprocess.TimeoutExpired:
                pass
        returncode = process.wait()

    completed = _matching_run(
        study_dir / "runs" / task.arm_id,
        config_sha256=config_sha,
        source_commit=source_commit,
        require_complete=True,
    )
    latest = _matching_run(
        study_dir / "runs" / task.arm_id,
        config_sha256=config_sha,
        require_complete=False,
    )
    success = returncode == 0 and completed is not None and not stop_requested[0]
    result = {
        "arm_id": task.arm_id,
        "policy": task.policy,
        "heldout_seed": task.heldout_seed,
        "config": str(task.config.resolve()),
        "config_sha256": config_sha,
        "status": "completed" if success else "failed",
        "returncode": returncode,
        "run_dir": str(completed or latest) if (completed or latest) else None,
        "started_at": launched_at,
        "finished_at": _utc_now(),
        "elapsed_seconds": time.monotonic() - started,
        "log": str(log_path),
        "stop_requested": stop_requested[0],
    }
    _persist_task_result(attempt_dir, task, result)
    if not success:
        raise RuntimeError(f"Native task failed: {task.label}; see {log_path}.")
    return result


def _summarize(
    *,
    study_dir: Path,
    attempt_dir: Path,
    environment: Mapping[str, str],
) -> dict[str, Any]:
    command = [
        str(TASK_PYTHON),
        "-m",
        "ebl",
        "study",
        "summarize",
        "--study-dir",
        str(study_dir),
        "--verify-artifacts",
        "--json",
    ]
    completed = subprocess.run(
        command,
        cwd=_ROOT,
        env=dict(environment),
        check=False,
        capture_output=True,
        text=True,
    )
    log_path = attempt_dir / "study-summarize.log"
    log_path.write_text(completed.stdout + completed.stderr, encoding="utf-8")
    summary = _read_json(study_dir / "analysis" / "summary.json")
    result = {
        "command": command,
        "returncode": completed.returncode,
        "log": str(log_path),
        "summary_path": str(study_dir / "analysis" / "summary.json"),
        "validation_mode": (
            summary.get("validation_mode") if summary is not None else None
        ),
        "state": summary.get("state") if summary is not None else None,
        "ready_for_review": (
            summary.get("ready_for_review") if summary is not None else None
        ),
    }
    atomic_write_json(attempt_dir / "study-summarize.result.json", result)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    study_dir = (
        args.study_dir or (_ROOT / "results" / STUDY_ID)
    ).expanduser().resolve()
    source_commit = _require_clean_source_commit()
    prerequisites = _validate_prerequisites()
    config_hashes = _validate_prepared_study(study_dir)
    environment = _task_environment(os.environ)
    attempt_dir = study_dir / "launch" / _attempt_id()
    attempt_dir.mkdir(parents=True, exist_ok=False)
    planned_tasks = tasks()
    contract = {
        "schema": "ebl.mnist_ibm_om_baseline_selection.launch_contract",
        "schema_version": 1,
        "study_id": STUDY_ID,
        "formal_evidence": True,
        "source_commit": source_commit,
        "source_clean": True,
        "launcher_pid": os.getpid(),
        "target": {
            "type": "local_sequential",
            "hostname": platform.node(),
            "working_directory": str(_ROOT),
        },
        "coverage": {
            "expected_arms": [arm_id for arm_id, _policy in ARM_POLICIES],
            "expected_heldout_seeds": list(HELDOUT_SEEDS),
            "expected_tasks": len(planned_tasks),
        },
        "prerequisites": prerequisites,
        "heartbeat": {
            "path": str(attempt_dir / "heartbeat.json"),
            "cadence_seconds": HEARTBEAT_SECONDS,
            "external_monitor_interval_seconds": MONITOR_INTERVAL_SECONDS,
            "stale_after_seconds": 2700,
        },
        "runtime_expectation": {
            "basis": "rounded up from the reviewed full-10000-example RTX-3090 preflight",
            "seconds_per_task": 30,
            "seconds_total_sequential": 360,
        },
        "safe_retry": (
            "rerun this launcher unchanged; retain every attempt and skip only "
            "an artifact-intact completed config from this exact clean commit"
        ),
        "tasks": [
            {
                "index": index,
                "arm_id": task.arm_id,
                "policy": task.policy,
                "heldout_seed": task.heldout_seed,
                "config": str(task.config.resolve()),
                "config_sha256": config_hashes[task.label],
                "command": _command(task, study_dir=study_dir),
                "output_root": str(study_dir / "runs" / task.arm_id),
                "log": str(attempt_dir / f"{task.label}.log"),
                "launch_receipt": str(
                    attempt_dir / f"{task.label}.launch.json"
                ),
                "terminal_result": str(
                    attempt_dir / f"{task.label}.result.json"
                ),
            }
            for index, task in enumerate(planned_tasks)
        ],
        "created_at": _utc_now(),
    }
    atomic_write_json(attempt_dir / "contract.json", contract)

    stop_requested = [False]

    def request_stop(signum, frame) -> None:  # type: ignore[no-untyped-def]
        stop_requested[0] = True

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    started = time.monotonic()
    task_results: list[dict[str, Any]] = []
    failure: BaseException | None = None
    try:
        for task in planned_tasks:
            if stop_requested[0]:
                raise InterruptedError("Launcher stop requested before next task.")
            task_results.append(
                _run_task(
                    task,
                    study_dir=study_dir,
                    attempt_dir=attempt_dir,
                    environment=environment,
                    source_commit=source_commit,
                    stop_requested=stop_requested,
                )
            )
    except BaseException as error:
        failure = error

    try:
        summary = _summarize(
            study_dir=study_dir,
            attempt_dir=attempt_dir,
            environment=environment,
        )
    except BaseException as error:
        summary = {
            "returncode": None,
            "validation_mode": None,
            "state": None,
            "ready_for_review": None,
            "error": {"type": type(error).__name__, "message": str(error)},
        }
        if failure is None:
            failure = error

    if failure is None and (
        summary.get("returncode") != 0
        or summary.get("validation_mode") != "full_artifact_hashes"
        or summary.get("ready_for_review") is not True
    ):
        failure = RuntimeError(
            "Artifact-verified study summary did not reach ready_for_review; "
            f"observed {summary!r}."
        )

    launcher_result = {
        "schema": "ebl.mnist_ibm_om_baseline_selection.launcher_result",
        "schema_version": 1,
        "study_id": STUDY_ID,
        "status": "complete" if failure is None else "failed",
        "source_commit": source_commit,
        "finished_at": _utc_now(),
        "elapsed_seconds": time.monotonic() - started,
        "tasks": task_results,
        "study_summary": summary,
        "error": (
            None
            if failure is None
            else {"type": type(failure).__name__, "message": str(failure)}
        ),
    }
    atomic_write_json(attempt_dir / "launcher_result.json", launcher_result)
    atomic_write_json(
        attempt_dir / "heartbeat.json",
        {
            "schema": "ebl.mnist_ibm_om_baseline_selection.heartbeat",
            "schema_version": 1,
            "study_id": STUDY_ID,
            "launcher_pid": os.getpid(),
            "updated_at": _utc_now(),
            "terminal": True,
            "launcher_status": launcher_result["status"],
            "completed_or_skipped_tasks": len(task_results),
            "expected_tasks": len(planned_tasks),
        },
    )
    if failure is not None:
        raise failure
    return 0


if __name__ == "__main__":  # pragma: no cover - operational entry point
    raise SystemExit(main())
