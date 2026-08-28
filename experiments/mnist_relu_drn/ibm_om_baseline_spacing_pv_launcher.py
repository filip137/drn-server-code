"""Persistent CUDA launcher for the IBM OM baseline/spacing P&V study.

The launcher is orchestration only.  A CUDA smoke canary and every production
task enter through the public ``python -m ebl validate`` command.  Production
tasks run sequentially because each task owns five network-scale pulse plants
on the one local GPU.  Failed native attempts are retained; a task is skipped
only when the same config already completed from the exact clean source
commit and every registered artifact is intact.
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
from experiments.mnist_relu_drn.ibm_om_baseline_spacing_pv_config import (
    BASELINE_POSITION_FRACTIONS,
    ENDPOINT_SEEDS_BY_ASSIGNMENT,
    EXPECTED_WEIGHTS_SHA256,
    EXPERIMENT_ID,
    HELDOUT_ASSIGNMENT_SEEDS,
    P_AND_V_REPEAT_COUNT,
    SPACING_DELTA_X_MULTIPLIERS,
    STUDY_ID,
    parse_baseline_spacing_pv_config,
)
from experiments.study_workflow import load_study_plan, load_study_record


_ROOT = Path(__file__).resolve().parents[2]
PLAN = _ROOT / "studies" / f"{STUDY_ID}.json"
CONFIG_ROOT = (
    _ROOT / "examples" / "mnist_relu_drn" / "ibm_om_baseline_spacing_pv"
)
TEACHER = _ROOT / "data" / "mnist_relu_teacher_fixed_init_20260816.pt"
TASK_PYTHON = Path("/home/filip/miniconda3/envs/py312/bin/python")
AIHWKIT_PYTHON = Path("/home/filip/miniconda3/envs/aihwkit/bin/python")
EXPECTED_AIHWKIT_VERSION = "1.1.0"
SMOKE_CONFIG = CONFIG_ROOT / "smoke-alpha_000_spacing_4delta-heldout-87001.json"
HEARTBEAT_SECONDS = 15.0
MONITOR_INTERVAL_SECONDS = 1800
ANALYZER_MODULE = (
    "experiments.mnist_relu_drn.analyze_ibm_om_baseline_spacing_pv"
)
ANALYSIS_RESULT = "baseline_spacing_pv_analysis.json"


def _alpha_token(alpha: float) -> str:
    return {0.0: "000", 0.25: "025", 0.5: "050"}[float(alpha)]


@dataclass(frozen=True)
class Policy:
    arm_id: str
    config_stem: str
    alpha: float
    spacing: int


@dataclass(frozen=True)
class Task:
    policy: Policy
    heldout_seed: int
    config: Path

    @property
    def arm_id(self) -> str:
        return self.policy.arm_id

    @property
    def label(self) -> str:
        return f"{self.arm_id}.heldout-{self.heldout_seed}"


POLICIES: tuple[Policy, ...] = tuple(
    Policy(
        arm_id=(
            f"alpha-{_alpha_token(alpha)}-spacing-{spacing}delta"
        ),
        config_stem=(
            f"alpha_{_alpha_token(alpha)}_spacing_{spacing}delta"
        ),
        alpha=float(alpha),
        spacing=int(spacing),
    )
    for alpha in BASELINE_POSITION_FRACTIONS
    for spacing in SPACING_DELTA_X_MULTIPLIERS
)


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
            "Expected the formal baseline/spacing P&V launcher to use a clean "
            "worktree. Commit every source, config, and study change first."
        )
    return commit


def tasks() -> tuple[Task, ...]:
    return tuple(
        Task(
            policy=policy,
            heldout_seed=seed,
            config=(
                CONFIG_ROOT
                / f"{policy.config_stem}-heldout-{seed}.json"
            ),
        )
        for policy in POLICIES
        for seed in HELDOUT_ASSIGNMENT_SEEDS
    )


def _load_strict_config(path: Path) -> tuple[Mapping[str, Any], Any]:
    payload = _read_json(path)
    if payload is None:
        raise RuntimeError(f"Expected strict JSON config: {path}.")
    try:
        document = parse_baseline_spacing_pv_config(payload)
    except Exception as error:
        raise RuntimeError(f"Strict config failed validation: {path}.") from error
    return payload, document


def _validate_task_config(task: Task) -> str:
    payload, document = _load_strict_config(task.config)
    protocol = document.protocol
    mismatches = {
        "experiment_id": (document.experiment_id, EXPERIMENT_ID),
        "execution profile": (protocol.execution.profile, "production"),
        "runtime device": (payload.get("runtime", {}).get("device"), "cuda"),
        "baseline alpha": (
            protocol.baseline_position_fraction,
            task.policy.alpha,
        ),
        "spacing multiplier": (
            protocol.spacing_delta_x_multiplier,
            task.policy.spacing,
        ),
        "held-out assignment": (
            protocol.assignments.heldout_seed,
            task.heldout_seed,
        ),
        "endpoint seeds": (
            tuple(protocol.assignments.endpoint_seeds),
            ENDPOINT_SEEDS_BY_ASSIGNMENT[task.heldout_seed],
        ),
        "P&V repeats": (
            protocol.program_verify.repeat_count,
            P_AND_V_REPEAT_COUNT,
        ),
        "P&V controller": (protocol.program_verify.controller, "one_pulse"),
        "P&V endpoint": (
            protocol.program_verify.endpoint_for_accuracy,
            "persistent",
        ),
        "inference read noise": (
            protocol.program_verify.inference_read_noise,
            False,
        ),
        "source hash": (
            protocol.source.expected_weights_sha256,
            EXPECTED_WEIGHTS_SHA256,
        ),
    }
    failed = [
        f"{name}: observed={observed!r}, expected={expected!r}"
        for name, (observed, expected) in mismatches.items()
        if observed != expected
    ]
    if failed:
        raise RuntimeError(
            f"Config does not match formal task {task.label!r}: "
            + "; ".join(failed)
        )
    return sha256_file(task.config)


def _validate_smoke_config() -> str:
    payload, document = _load_strict_config(SMOKE_CONFIG)
    protocol = document.protocol
    expected = {
        "profile": (protocol.execution.profile, "smoke"),
        "device": (payload.get("runtime", {}).get("device"), "cuda"),
        "sample limit": (
            payload.get("modes", {}).get("validate", {}).get("sample_limit"),
            32,
        ),
        "alpha": (protocol.baseline_position_fraction, 0.0),
        "spacing": (protocol.spacing_delta_x_multiplier, 4),
        "held-out seed": (protocol.assignments.heldout_seed, 87001),
        "endpoint seeds": (
            tuple(protocol.assignments.endpoint_seeds),
            ENDPOINT_SEEDS_BY_ASSIGNMENT[87001],
        ),
        "repeat count": (
            protocol.program_verify.repeat_count,
            P_AND_V_REPEAT_COUNT,
        ),
    }
    failed = [
        f"{name}: observed={observed!r}, expected={wanted!r}"
        for name, (observed, wanted) in expected.items()
        if observed != wanted
    ]
    if failed:
        raise RuntimeError("CUDA smoke config mismatch: " + "; ".join(failed))
    return sha256_file(SMOKE_CONFIG)


def _validate_prepared_study(study_dir: Path) -> dict[str, str]:
    plan = load_study_plan(PLAN)
    study = load_study_record(study_dir)
    if plan["study_id"] != STUDY_ID or study["study_id"] != STUDY_ID:
        raise RuntimeError(f"Expected the prepared {STUDY_ID!r} study.")
    if study["source_plan"]["sha256"] != sha256_file(PLAN):
        raise RuntimeError("Prepared study does not match the tracked plan hash.")

    expected_tasks = tasks()
    expected_by_arm = {
        policy.arm_id: [task for task in expected_tasks if task.arm_id == policy.arm_id]
        for policy in POLICIES
    }
    plan_arms = {arm["arm_id"]: arm for arm in plan["arms"]}
    study_arms = {arm["arm_id"]: arm for arm in study["arms"]}
    if set(plan_arms) != set(expected_by_arm) or set(study_arms) != set(
        expected_by_arm
    ):
        raise RuntimeError(
            "Expected tracked and prepared studies to declare exactly nine "
            "alpha-by-spacing arms."
        )

    config_hashes: dict[str, str] = {}
    for arm_id, arm_tasks in expected_by_arm.items():
        expected_hashes = {_validate_task_config(task) for task in arm_tasks}
        for label, arm in (
            ("tracked", plan_arms[arm_id]),
            ("prepared", study_arms[arm_id]),
        ):
            if arm["experiment_id"] != EXPERIMENT_ID or arm["mode"] != "validate":
                raise RuntimeError(
                    f"Expected {label} arm {arm_id!r} to be validate-only "
                    f"{EXPERIMENT_ID!r}."
                )
            if len(arm["configs"]) != len(HELDOUT_ASSIGNMENT_SEEDS):
                raise RuntimeError(
                    f"Expected {label} arm {arm_id!r} to declare three configs."
                )
            provided = {item["sha256"] for item in arm["configs"]}
            if provided != expected_hashes:
                raise RuntimeError(
                    f"Config hash coverage mismatch for {label} arm {arm_id!r}."
                )
        for task in arm_tasks:
            config_hashes[task.label] = sha256_file(task.config)

    smoke_hash = _validate_smoke_config()
    discovered_production = {
        path.resolve()
        for path in CONFIG_ROOT.glob("*.json")
        if not path.name.startswith("smoke-")
    }
    expected_paths = {task.config.resolve() for task in expected_tasks}
    if discovered_production != expected_paths:
        raise RuntimeError(
            "Expected the config directory to contain exactly the 27 declared "
            "production configs plus explicitly excluded smoke-* configs."
        )
    config_hashes["smoke-canary"] = smoke_hash
    return config_hashes


def _validate_prerequisites() -> dict[str, Any]:
    task_python = _require_executable(TASK_PYTHON, label="task Python")
    aihwkit_python = _require_executable(AIHWKIT_PYTHON, label="AIHWKit Python")
    if not TEACHER.is_file():
        raise RuntimeError(f"Missing frozen source checkpoint: {TEACHER}.")
    teacher_sha = sha256_file(TEACHER)
    if teacher_sha != EXPECTED_WEIGHTS_SHA256:
        raise RuntimeError(
            "Frozen source checkpoint hash mismatch: "
            f"expected {EXPECTED_WEIGHTS_SHA256}, observed {teacher_sha}."
        )
    version_probe = subprocess.run(
        (str(aihwkit_python), "-c", "import aihwkit; print(aihwkit.__version__)"),
        check=False,
        capture_output=True,
        text=True,
    )
    if (
        version_probe.returncode != 0
        or version_probe.stdout.strip() != EXPECTED_AIHWKIT_VERSION
    ):
        raise RuntimeError(
            f"Expected pinned AIHWKit {EXPECTED_AIHWKIT_VERSION}; "
            f"stdout={version_probe.stdout!r}, stderr={version_probe.stderr!r}."
        )
    cuda_probe = subprocess.run(
        (
            str(task_python),
            "-c",
            (
                "import json,torch; "
                "print(json.dumps({'torch_version':torch.__version__,"
                "'cuda_available':torch.cuda.is_available(),"
                "'cuda_device_count':torch.cuda.device_count(),"
                "'cuda_device_name':torch.cuda.get_device_name(0) if "
                "torch.cuda.is_available() else None,"
                "'cuda_free_total_bytes':torch.cuda.mem_get_info(0) if "
                "torch.cuda.is_available() else None}))"
            ),
        ),
        check=False,
        capture_output=True,
        text=True,
    )
    try:
        cuda = json.loads(cuda_probe.stdout)
    except json.JSONDecodeError as error:
        raise RuntimeError("Expected a readable CUDA probe.") from error
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


def _command(config: Path, *, output_root: Path) -> list[str]:
    return [
        str(TASK_PYTHON),
        "-m",
        "ebl",
        "validate",
        "--config",
        str(config.resolve()),
        "--output-dir",
        str(output_root.resolve()),
        "--weights",
        str(TEACHER.resolve()),
    ]


def _task_command(task: Task, *, study_dir: Path) -> list[str]:
    return _command(task.config, output_root=study_dir / "runs" / task.arm_id)


def _native_runs(root: Path) -> list[Path]:
    if not root.is_dir():
        return []
    return sorted(
        child
        for child in root.iterdir()
        if child.is_dir()
        and (child / "manifest.json").is_file()
        and (child / "status.json").is_file()
    )


def _artifact_records_are_intact(run_dir: Path, result: Mapping[str, Any]) -> bool:
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


def _process_snapshot(
    *,
    output_root: Path,
    process: subprocess.Popen[str],
) -> dict[str, Any]:
    native = _native_runs(output_root)
    run_dir = native[-1] if native else None
    status = _read_json(run_dir / "status.json") if run_dir is not None else None
    artifact_root = run_dir / "artifacts" if run_dir is not None else None
    artifact_files = (
        [path for path in artifact_root.rglob("*") if path.is_file()]
        if artifact_root is not None and artifact_root.is_dir()
        else []
    )
    return {
        "pid": process.pid,
        "launcher_returncode": process.poll(),
        "native_run_dir": str(run_dir) if run_dir is not None else None,
        "native_status": status.get("status") if status is not None else None,
        "artifact_file_count": len(artifact_files),
        "artifact_size_bytes": sum(path.stat().st_size for path in artifact_files),
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


def _run_process(
    *,
    label: str,
    command: Sequence[str],
    output_root: Path,
    attempt_dir: Path,
    environment: Mapping[str, str],
    stop_requested: list[bool],
) -> tuple[int, float, Path]:
    log_path = attempt_dir / f"{label}.log"
    started = time.monotonic()
    launched_at = _utc_now()
    with log_path.open("w", encoding="utf-8", buffering=1) as log:
        process = subprocess.Popen(
            list(command),
            cwd=_ROOT,
            env=dict(environment),
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        atomic_write_json(
            attempt_dir / f"{label}.launch.json",
            {
                "label": label,
                "command": list(command),
                "pid": process.pid,
                "started_at": launched_at,
                "log": str(log_path),
                "output_root": str(output_root),
            },
        )
        while True:
            atomic_write_json(
                attempt_dir / "heartbeat.json",
                {
                    "schema": "ebl.mnist_ibm_om_baseline_spacing_pv.heartbeat",
                    "schema_version": 1,
                    "study_id": STUDY_ID,
                    "launcher_pid": os.getpid(),
                    "updated_at": _utc_now(),
                    "heartbeat_seconds": HEARTBEAT_SECONDS,
                    "active_task": label,
                    "elapsed_task_seconds": time.monotonic() - started,
                    "stop_requested": stop_requested[0],
                    "native": _process_snapshot(
                        output_root=output_root,
                        process=process,
                    ),
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
    return returncode, time.monotonic() - started, log_path


def _run_smoke(
    *,
    attempt_dir: Path,
    environment: Mapping[str, str],
    source_commit: str,
    stop_requested: list[bool],
) -> dict[str, Any]:
    output_root = attempt_dir / "smoke-native"
    command = _command(SMOKE_CONFIG, output_root=output_root)
    returncode, elapsed, log_path = _run_process(
        label="cuda-smoke-canary",
        command=command,
        output_root=output_root,
        attempt_dir=attempt_dir,
        environment=environment,
        stop_requested=stop_requested,
    )
    native = _native_runs(output_root)
    if len(native) != 1:
        raise RuntimeError(
            f"Expected exactly one CUDA smoke RunStore bundle; found {native!r}."
        )
    run_dir = native[0]
    manifest = _read_json(run_dir / "manifest.json")
    status = _read_json(run_dir / "status.json")
    result = _read_json(run_dir / "result.json")
    source = manifest.get("source") if manifest is not None else None
    summary = _read_json(run_dir / "artifacts" / "scientific_summary.json")
    valid = (
        returncode == 0
        and manifest is not None
        and manifest.get("experiment_id") == EXPERIMENT_ID
        and manifest.get("config", {}).get("sha256") is not None
        and isinstance(source, Mapping)
        and source.get("commit") == source_commit
        and source.get("dirty") is False
        and status is not None
        and status.get("status") == "complete"
        and result is not None
        and result.get("status") == "complete"
        and result.get("run_id") == run_dir.name
        and _artifact_records_are_intact(run_dir, result)
        and summary is not None
        and summary.get("schema")
        == "ebl.mnist_relu_drn.ibm_om_baseline_spacing_pv_result"
        and summary.get("status") == "complete"
        and summary.get("contract", {}).get("execution", {}).get("profile")
        == "smoke"
        and summary.get("validity", {}).get("expected_examples") == 32
        and len(summary.get("heldout", {}).get("pv_endpoint_artifacts", ()))
        == P_AND_V_REPEAT_COUNT
    )
    record = {
        "status": "complete" if valid else "failed",
        "returncode": returncode,
        "elapsed_seconds": elapsed,
        "run_dir": str(run_dir),
        "log": str(log_path),
        "config": str(SMOKE_CONFIG.resolve()),
        "config_sha256": sha256_file(SMOKE_CONFIG),
        "source_commit": source_commit,
        "five_repeat_semantics": True,
    }
    atomic_write_json(attempt_dir / "cuda-smoke-canary.result.json", record)
    if not valid:
        raise RuntimeError(f"CUDA smoke canary failed; see {log_path}.")
    return record


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
    arm_root = study_dir / "runs" / task.arm_id
    exact = _matching_run(
        arm_root,
        config_sha256=config_sha,
        source_commit=source_commit,
        require_complete=True,
    )
    if exact is not None:
        result = {
            "arm_id": task.arm_id,
            "alpha": task.policy.alpha,
            "spacing_delta_x_multiplier": task.policy.spacing,
            "heldout_seed": task.heldout_seed,
            "config": str(task.config.resolve()),
            "config_sha256": config_sha,
            "status": "skipped_exact_completed",
            "run_dir": str(exact),
            "finished_at": _utc_now(),
        }
        atomic_write_json(attempt_dir / f"{task.label}.result.json", result)
        return result
    other_source = _matching_run(
        arm_root,
        config_sha256=config_sha,
        require_complete=True,
    )
    if other_source is not None:
        raise RuntimeError(
            "Refusing to mix clean source commits in one formal study: exact "
            f"config {task.label!r} completed at {other_source}."
        )
    running = _running_run(arm_root, config_sha256=config_sha)
    if running is not None:
        raise RuntimeError(
            f"Refusing duplicate task {task.label!r}; matching run is active: {running}."
        )

    command = _task_command(task, study_dir=study_dir)
    returncode, elapsed, log_path = _run_process(
        label=task.label,
        command=command,
        output_root=arm_root,
        attempt_dir=attempt_dir,
        environment=environment,
        stop_requested=stop_requested,
    )
    completed = _matching_run(
        arm_root,
        config_sha256=config_sha,
        source_commit=source_commit,
        require_complete=True,
    )
    latest = _matching_run(
        arm_root,
        config_sha256=config_sha,
        require_complete=False,
    )
    success = returncode == 0 and completed is not None and not stop_requested[0]
    result = {
        "arm_id": task.arm_id,
        "alpha": task.policy.alpha,
        "spacing_delta_x_multiplier": task.policy.spacing,
        "heldout_seed": task.heldout_seed,
        "endpoint_seeds": list(ENDPOINT_SEEDS_BY_ASSIGNMENT[task.heldout_seed]),
        "config": str(task.config.resolve()),
        "config_sha256": config_sha,
        "status": "completed" if success else "failed",
        "returncode": returncode,
        "run_dir": str(completed or latest) if (completed or latest) else None,
        "finished_at": _utc_now(),
        "elapsed_seconds": elapsed,
        "log": str(log_path),
        "stop_requested": stop_requested[0],
    }
    atomic_write_json(attempt_dir / f"{task.label}.result.json", result)
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


def _run_custom_analysis(
    *,
    study_dir: Path,
    attempt_dir: Path,
    environment: Mapping[str, str],
) -> dict[str, Any]:
    output_dir = study_dir / "analysis"
    command = [
        str(TASK_PYTHON),
        "-m",
        ANALYZER_MODULE,
        "--study-dir",
        str(study_dir),
        "--output-dir",
        str(output_dir),
    ]
    completed = subprocess.run(
        command,
        cwd=_ROOT,
        env=dict(environment),
        check=False,
        capture_output=True,
        text=True,
    )
    log_path = attempt_dir / "custom-analysis.log"
    log_path.write_text(completed.stdout + completed.stderr, encoding="utf-8")
    report_path = output_dir / ANALYSIS_RESULT
    report = _read_json(report_path)
    valid = (
        completed.returncode == 0
        and report is not None
        and report.get("schema")
        == "ebl.mnist_relu_drn.ibm_om_baseline_spacing_pv_analysis"
        and report.get("schema_version") == 1
        and report.get("study_id") == STUDY_ID
        and report.get("audit", {}).get("complete_run_count") == 27
        and report.get("audit", {}).get("registered_artifacts_verified") is True
        and report.get("audit", {}).get("matched_hardware_and_rng_streams") is True
    )
    result = {
        "command": command,
        "returncode": completed.returncode,
        "log": str(log_path),
        "report": str(report_path),
        "report_sha256": sha256_file(report_path) if report_path.is_file() else None,
        "valid": valid,
    }
    atomic_write_json(attempt_dir / "custom-analysis.result.json", result)
    if not valid:
        raise RuntimeError(
            f"Artifact-level baseline/spacing P&V analyzer failed; see {log_path}."
        )
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
    planned = tasks()
    atomic_write_json(
        attempt_dir / "contract.json",
        {
            "schema": "ebl.mnist_ibm_om_baseline_spacing_pv.launch_contract",
            "schema_version": 1,
            "study_id": STUDY_ID,
            "formal_evidence": True,
            "source_commit": source_commit,
            "source_clean": True,
            "launcher_pid": os.getpid(),
            "target": {
                "type": "local_sequential_cuda",
                "hostname": platform.node(),
                "working_directory": str(_ROOT),
            },
            "coverage": {
                "expected_arms": [policy.arm_id for policy in POLICIES],
                "expected_heldout_seeds": list(HELDOUT_ASSIGNMENT_SEEDS),
                "expected_tasks": len(planned),
                "pv_repeats_per_task": P_AND_V_REPEAT_COUNT,
                "expected_pv_endpoints": len(planned) * P_AND_V_REPEAT_COUNT,
            },
            "prerequisites": prerequisites,
            "smoke_canary": {
                "config": str(SMOKE_CONFIG.resolve()),
                "config_sha256": config_hashes["smoke-canary"],
                "output_root": str(attempt_dir / "smoke-native"),
                "must_complete_before_production": True,
            },
            "heartbeat": {
                "path": str(attempt_dir / "heartbeat.json"),
                "cadence_seconds": HEARTBEAT_SECONDS,
                "external_monitor_interval_seconds": MONITOR_INTERVAL_SECONDS,
                "stale_after_seconds": 2700,
            },
            "runtime_expectation": {
                "basis": (
                    "RTX-3090 baseline tasks plus measured five-pass raw-active "
                    "pulse benchmark; five P&V passes are sequential"
                ),
                "seconds_per_task_range": [40, 70],
                "seconds_total_sequential_range": [1080, 1890],
                "pulse_stream_peak_mib_per_repeat": 187.36,
            },
            "safe_retry": (
                "rerun this launcher unchanged; retain every attempt and skip "
                "only artifact-intact completed configs from this clean commit"
            ),
            "tasks": [
                {
                    "index": index,
                    "arm_id": task.arm_id,
                    "alpha": task.policy.alpha,
                    "spacing_delta_x_multiplier": task.policy.spacing,
                    "heldout_seed": task.heldout_seed,
                    "endpoint_seeds": list(
                        ENDPOINT_SEEDS_BY_ASSIGNMENT[task.heldout_seed]
                    ),
                    "config": str(task.config.resolve()),
                    "config_sha256": config_hashes[task.label],
                    "command": _task_command(task, study_dir=study_dir),
                    "output_root": str(study_dir / "runs" / task.arm_id),
                    "log": str(attempt_dir / f"{task.label}.log"),
                    "launch_receipt": str(
                        attempt_dir / f"{task.label}.launch.json"
                    ),
                    "terminal_result": str(
                        attempt_dir / f"{task.label}.result.json"
                    ),
                }
                for index, task in enumerate(planned)
            ],
            "created_at": _utc_now(),
        },
    )

    stop_requested = [False]

    def request_stop(signum, frame) -> None:  # type: ignore[no-untyped-def]
        stop_requested[0] = True

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    started = time.monotonic()
    task_results: list[dict[str, Any]] = []
    smoke_result: dict[str, Any] | None = None
    custom_analysis: dict[str, Any] | None = None
    failure: BaseException | None = None
    try:
        smoke_result = _run_smoke(
            attempt_dir=attempt_dir,
            environment=environment,
            source_commit=source_commit,
            stop_requested=stop_requested,
        )
        for task in planned:
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
        custom_analysis = _run_custom_analysis(
            study_dir=study_dir,
            attempt_dir=attempt_dir,
            environment=environment,
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
        "schema": "ebl.mnist_ibm_om_baseline_spacing_pv.launcher_result",
        "schema_version": 1,
        "study_id": STUDY_ID,
        "status": "complete" if failure is None else "failed",
        "source_commit": source_commit,
        "smoke_canary": smoke_result,
        "finished_at": _utc_now(),
        "elapsed_seconds": time.monotonic() - started,
        "tasks": task_results,
        "custom_analysis": custom_analysis,
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
            "schema": "ebl.mnist_ibm_om_baseline_spacing_pv.heartbeat",
            "schema_version": 1,
            "study_id": STUDY_ID,
            "launcher_pid": os.getpid(),
            "updated_at": _utc_now(),
            "terminal": True,
            "launcher_status": launcher_result["status"],
            "completed_or_skipped_tasks": len(task_results),
            "expected_tasks": len(planned),
        },
    )
    if failure is not None:
        raise failure
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
