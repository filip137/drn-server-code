"""Persistent fail-closed launcher for the cell-aware exact-bounds study.

Every numerical task enters through ``python -m ebl``.  This module only
checks immutable prerequisites, enforces the assignment-85001 embargo,
persists launch/heartbeat receipts, and resumes by skipping exact completed
config bundles while preserving failed attempts.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence

from experiments.artifacts import atomic_write_json, sha256_file


_ROOT = Path(__file__).resolve().parents[2]
STUDY_ID = "mnist-ibm-om-cell-aware-exact-bounds-transfer-20260824-v1"
PLAN = _ROOT / "studies" / f"{STUDY_ID}.json"
CONFIG_ROOT = (
    _ROOT / "examples" / "mnist_relu_drn" / "ibm_om_cell_aware_exact_bounds"
)
FULL_SPAN = _ROOT / "data" / "ibm_om_cell_aware_full_span_v1.pt"
FULL_SPAN_RECEIPT = (
    _ROOT / "data" / "ibm_om_cell_aware_full_span_v1.receipt.json"
)
TEACHER = _ROOT / "data" / "mnist_relu_teacher_fixed_init_20260816.pt"
DEVICE_MODEL = _ROOT / "data" / "ibm_reram_om_pv128_hwa_v1.json"
SCHEDULE_RECEIPT = CONFIG_ROOT / "reset_relative_schedule_receipt.json"
SCHEDULE_RECEIPT_SHA256 = (
    "003f3d15af6fe3c0097f1668d2922dc4b6b1fb290098f7a70efb67d76bb15aa5"
)
DESIGN_RECEIPT = CONFIG_ROOT / "cell_aware_design_receipt.json"
DESIGN_RECEIPT_SHA256 = (
    "c573ba73737ae1b8e3b0c0dd32535b91665346e5e18700cf0679168c66fd8f6c"
)
EXPECTED_INPUT_HASHES = {
    FULL_SPAN: "a99995a3e5b321a56bb7e00e29840c3b76f3fee95b8d8c80fdf0fa16e93b3563",
    FULL_SPAN_RECEIPT: "d58b0d709312061dc2425d143744f45fdf9de9d5e583a619e7a91325c53059b8",
    TEACHER: "9a961a77628e365b54fdf59304ec7fddf46f1f4834c4c03cecea9c204f837f52",
    DEVICE_MODEL: "3030e04d6205dc90d0894ac453d2c1c522dffdc004f69b9c6ab7eaf7ef4b8ba3",
}
DEVELOPMENT_ARMS = {
    "train-zero-update-quantized": "zero_update_quantized.json",
    "train-clean-bptt-quantized-deploy": "clean_bptt_quantized_deploy.json",
    "train-continuous-cell-aware-hwa": "continuous_hwa.json",
    "train-quantized-cell-aware-qat": "quantized_qat.json",
}
HELDOUT_ARM_MODES = {
    "deploy-zero-update-heldout": "quantized",
    "deploy-clean-bptt-heldout": "quantized",
    "deploy-continuous-hwa-heldout": "continuous",
    "deploy-quantized-qat-heldout": "quantized",
}
HELDOUT_SOURCE_ARMS = {
    "deploy-zero-update-heldout": "train-zero-update-quantized",
    "deploy-clean-bptt-heldout": "train-clean-bptt-quantized-deploy",
    "deploy-continuous-hwa-heldout": "train-continuous-cell-aware-hwa",
    "deploy-quantized-qat-heldout": "train-quantized-cell-aware-qat",
}
ENDPOINT_SEEDS = tuple(range(85101, 85106))
HEARTBEAT_SECONDS = 15.0
_CUDA_VISIBLE_DEVICES = re.compile(
    r"(?:0|[1-9][0-9]*)(?:,(?:0|[1-9][0-9]*))*"
)


@dataclass(frozen=True)
class Task:
    arm_id: str
    mode: str
    config: Path
    weights: Path


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _attempt_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--phase",
        choices=("development", "heldout", "all"),
        default="all",
    )
    parser.add_argument("--study-dir", type=Path)
    parser.add_argument("--aihwkit-python", type=Path, required=True)
    parser.add_argument(
        "--cuda-visible-devices",
        default="inherit",
        help=(
            "CUDA visibility passed to native tasks, or 'inherit' to retain "
            "the launcher's existing device namespace."
        ),
    )
    return parser


def _read_json(path: Path) -> Mapping[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, Mapping) else None


def _task_environment(
    base: Mapping[str, str],
    *,
    aihwkit_python: str,
    cuda_visible_devices: str,
) -> dict[str, str]:
    environment = dict(base)
    environment["EBL_AIHWKIT_PYTHON"] = aihwkit_python
    if cuda_visible_devices != "inherit":
        if _CUDA_VISIBLE_DEVICES.fullmatch(cuda_visible_devices) is None:
            raise RuntimeError(
                "Expected --cuda-visible-devices to be 'inherit' or a "
                "comma-separated list of non-negative integer device indices."
            )
        environment["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
    environment["EBL_DEFER_CURRENT_SIMULATIONS"] = "1"
    environment["PYTHONUNBUFFERED"] = "1"
    return environment


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
            "Expected the formal cell-aware launcher to use a clean worktree. "
            "Commit every source/config/study change before launch."
        )
    return commit


def _validate_prerequisites(aihwkit_python: Path) -> dict[str, Any]:
    resolved_python = aihwkit_python.expanduser().resolve()
    if not resolved_python.is_file() or not os.access(resolved_python, os.X_OK):
        raise RuntimeError("Expected --aihwkit-python to be executable.")
    observed = {}
    for path, expected in EXPECTED_INPUT_HASHES.items():
        if not path.is_file():
            raise RuntimeError(f"Missing frozen input: {path}.")
        actual = sha256_file(path)
        if actual != expected:
            raise RuntimeError(
                f"Frozen input hash mismatch for {path}: expected {expected}, "
                f"observed {actual}."
            )
        observed[str(path)] = actual
    schedule = _read_json(SCHEDULE_RECEIPT)
    if (
        schedule is None
        or sha256_file(SCHEDULE_RECEIPT) != SCHEDULE_RECEIPT_SHA256
        or schedule.get("source_study_id")
        != "mnist-ibm-om-shared-reset-relative-quantized-hwa-20260824-v1"
        or schedule.get("cell_aware_learning_rate_selection_performed") is not False
        or schedule.get("heldout_assignment_consulted") is not False
    ):
        raise RuntimeError("Expected the strict RESET-schedule reuse receipt.")
    design = _read_json(DESIGN_RECEIPT)
    if (
        design is None
        or sha256_file(DESIGN_RECEIPT) != DESIGN_RECEIPT_SHA256
        or design.get("schema")
        != "ebl.mnist_relu_drn.ibm_om_cell_aware_exact_bounds_design"
        or design.get("scope") != "development_assignment_84001_only"
        or design.get("heldout_assignment_consulted") is not False
        or design.get("assignment_seed") != 84001
        or design.get("population_fingerprint")
        != "4887ad89abdd16193448c54a0cbe97be915cb4b9bc04cf1151c5e2a96ee5a3ac"
        or design.get("gain_fit", {}).get("continuous", {}).get("gain")
        != 3.5481338923357533
        or design.get("gain_fit", {})
        .get("quantized_9_level", {})
        .get("gain")
        != 2.818382931264453
    ):
        raise RuntimeError("Expected the frozen cell-aware design receipt.")
    probe = subprocess.run(
        (
            str(resolved_python),
            "-c",
            "import aihwkit; print(aihwkit.__version__)",
        ),
        check=False,
        capture_output=True,
        text=True,
    )
    if probe.returncode != 0 or probe.stdout.strip() != "1.1.0":
        raise RuntimeError(
            "Expected --aihwkit-python to import AIHWKit 1.1.0. "
            f"stdout={probe.stdout!r}, stderr={probe.stderr!r}."
        )
    return {
        "frozen_inputs": observed,
        "schedule_receipt": {
            "path": str(SCHEDULE_RECEIPT),
            "sha256": SCHEDULE_RECEIPT_SHA256,
        },
        "design_receipt": {
            "path": str(DESIGN_RECEIPT),
            "sha256": DESIGN_RECEIPT_SHA256,
        },
        "aihwkit_python": str(resolved_python),
        "aihwkit_version": "1.1.0",
    }


def _validate_prepared_study(study_dir: Path) -> None:
    study = _read_json(study_dir / "study.json")
    if study is None or study.get("study_id") != STUDY_ID:
        raise RuntimeError(
            f"Expected --study-dir to be the prepared {STUDY_ID!r} study."
        )
    declared = {
        str(arm.get("arm_id"))
        for arm in study.get("arms", [])
        if isinstance(arm, Mapping)
    }
    expected = set(DEVELOPMENT_ARMS) | set(HELDOUT_ARM_MODES)
    if declared != expected:
        raise RuntimeError(
            "Expected the prepared study to declare exactly the four "
            "development and four held-out pipeline arms."
        )


def _native_runs(arm_root: Path) -> list[Path]:
    if not arm_root.is_dir():
        return []
    return sorted(
        child
        for child in arm_root.iterdir()
        if child.is_dir() and (child / "status.json").is_file()
    )


def _completed_run(arm_root: Path, config: Path) -> Path | None:
    expected_config_sha = sha256_file(config)
    matches = []
    for run_dir in _native_runs(arm_root):
        status = _read_json(run_dir / "status.json")
        manifest = _read_json(run_dir / "manifest.json")
        study = manifest.get("study") if manifest is not None else None
        if (
            status is not None
            and status.get("status") == "completed"
            and isinstance(study, Mapping)
            and study.get("source_config_sha256") == expected_config_sha
        ):
            matches.append(run_dir)
    if len(matches) > 1:
        raise RuntimeError(
            f"Expected at most one completed run for {config}; found {matches!r}."
        )
    return matches[0] if matches else None


def development_tasks() -> tuple[Task, ...]:
    return tuple(
        Task(
            arm_id=arm_id,
            mode="train",
            config=CONFIG_ROOT / config_name,
            weights=FULL_SPAN,
        )
        for arm_id, config_name in DEVELOPMENT_ARMS.items()
    )


def _development_freeze(study_dir: Path, *, source_commit: str) -> dict[str, Any]:
    checkpoints = {}
    for task in development_tasks():
        run_dir = _completed_run(study_dir / "runs" / task.arm_id, task.config)
        if run_dir is None:
            raise RuntimeError(
                "Held-out assignment embargo remains active because the "
                f"development arm {task.arm_id!r} is incomplete."
            )
        weights = run_dir / "checkpoints" / "weights.pt"
        result = _read_json(run_dir / "result.json")
        if not weights.is_file() or result is None:
            raise RuntimeError(
                f"Expected completed development artifacts for {task.arm_id!r}."
            )
        selected = result.get("metrics", {}).get("selected")
        if not isinstance(selected, Mapping):
            raise RuntimeError(
                f"Expected selected metrics for {task.arm_id!r}."
            )
        checkpoints[task.arm_id] = {
            "run_dir": str(run_dir),
            "config": str(task.config),
            "config_sha256": sha256_file(task.config),
            "weights": str(weights),
            "weights_sha256": sha256_file(weights),
            "selected_epoch": selected.get("epoch"),
            "selected_student_accuracy": selected.get("student_accuracy"),
        }
    return {
        "schema": "ebl.mnist_relu_drn.ibm_om_cell_aware_development_freeze",
        "schema_version": 1,
        "study_id": STUDY_ID,
        "source_commit": source_commit,
        "assignment_85001_embargo_released": True,
        "release_condition": "all_four_development_checkpoints_frozen",
        "created_at": _utc_now(),
        "checkpoints": checkpoints,
    }


def _load_development_freeze(study_dir: Path, *, source_commit: str) -> Mapping[str, Any]:
    path = study_dir / "launch" / "development_freeze.json"
    freeze = _read_json(path)
    if (
        freeze is None
        or freeze.get("study_id") != STUDY_ID
        or freeze.get("source_commit") != source_commit
        or freeze.get("assignment_85001_embargo_released") is not True
    ):
        raise RuntimeError(
            "Expected a matching immutable development freeze before any "
            "assignment-85001 task."
        )
    current = _development_freeze(study_dir, source_commit=source_commit)
    frozen_checkpoints = freeze.get("checkpoints")
    if not isinstance(frozen_checkpoints, Mapping):
        raise RuntimeError("Expected development freeze checkpoint records.")
    for arm_id, record in current["checkpoints"].items():
        frozen = frozen_checkpoints.get(arm_id)
        if not isinstance(frozen, Mapping) or any(
            frozen.get(field) != record[field]
            for field in (
                "run_dir",
                "config_sha256",
                "weights",
                "weights_sha256",
                "selected_epoch",
            )
        ):
            raise RuntimeError(
                "Development checkpoint changed after assignment-85001 "
                f"embargo release: {arm_id!r}."
            )
    return freeze


def heldout_tasks(freeze: Mapping[str, Any]) -> tuple[Task, ...]:
    checkpoints = freeze.get("checkpoints")
    if not isinstance(checkpoints, Mapping):
        raise RuntimeError("Expected frozen checkpoint records.")
    tasks = []
    for arm_id, mode in HELDOUT_ARM_MODES.items():
        source_arm = HELDOUT_SOURCE_ARMS[arm_id]
        source = checkpoints.get(source_arm)
        if not isinstance(source, Mapping):
            raise RuntimeError(f"Missing frozen source arm {source_arm!r}.")
        weights = Path(str(source.get("weights"))).resolve()
        if not weights.is_file() or sha256_file(weights) != source.get(
            "weights_sha256"
        ):
            raise RuntimeError(f"Frozen source weights changed for {source_arm!r}.")
        for endpoint_seed in ENDPOINT_SEEDS:
            tasks.append(
                Task(
                    arm_id=arm_id,
                    mode="validate",
                    config=(
                        CONFIG_ROOT
                        / f"heldout_{mode}_seed_{endpoint_seed}.json"
                    ),
                    weights=weights,
                )
            )
    return tuple(tasks)


def _command(task: Task, *, python: Path, study_dir: Path) -> list[str]:
    command = [
        str(python),
        "-m",
        "ebl",
        task.mode,
        "--config",
        str(task.config),
        "--output-dir",
        str(study_dir / "runs" / task.arm_id),
        "--weights",
        str(task.weights),
        "--teacher-weights",
        str(TEACHER),
        "--device-model",
        str(DEVICE_MODEL),
    ]
    return command


def _terminate_process(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return


def _run_task(
    task: Task,
    *,
    python: Path,
    study_dir: Path,
    attempt_dir: Path,
    environment: Mapping[str, str],
    stop_requested: list[bool],
) -> dict[str, Any]:
    completed = _completed_run(study_dir / "runs" / task.arm_id, task.config)
    if completed is not None:
        return {
            "arm_id": task.arm_id,
            "config": str(task.config),
            "config_sha256": sha256_file(task.config),
            "status": "skipped_completed",
            "run_dir": str(completed),
        }
    command = _command(task, python=python, study_dir=study_dir)
    label = f"{task.arm_id}.{task.config.stem}"
    log_path = attempt_dir / f"{label}.log"
    started = time.monotonic()
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
            attempt_dir / f"{label}.launch.json",
            {
                "task": task.__dict__ | {"config": str(task.config), "weights": str(task.weights)},
                "command": command,
                "pid": process.pid,
                "started_at": _utc_now(),
                "log": str(log_path),
            },
        )
        while process.poll() is None:
            runs = _native_runs(study_dir / "runs" / task.arm_id)
            latest = runs[-1] if runs else None
            native_status = (
                _read_json(latest / "status.json") if latest is not None else None
            )
            atomic_write_json(
                attempt_dir / "heartbeat.json",
                {
                    "updated_at": _utc_now(),
                    "active_task": label,
                    "pid": process.pid,
                    "elapsed_seconds": time.monotonic() - started,
                    "stop_requested": stop_requested[0],
                    "native_run_dir": str(latest) if latest is not None else None,
                    "native_status": (
                        native_status.get("status")
                        if native_status is not None
                        else None
                    ),
                    "metrics_size_bytes": (
                        (latest / "metrics.jsonl").stat().st_size
                        if latest is not None
                        and (latest / "metrics.jsonl").is_file()
                        else 0
                    ),
                },
            )
            if stop_requested[0]:
                _terminate_process(process)
            try:
                process.wait(timeout=HEARTBEAT_SECONDS)
            except subprocess.TimeoutExpired:
                pass
        returncode = process.wait()
    completed = _completed_run(study_dir / "runs" / task.arm_id, task.config)
    result = {
        "arm_id": task.arm_id,
        "config": str(task.config),
        "config_sha256": sha256_file(task.config),
        "status": "completed" if returncode == 0 and completed else "failed",
        "returncode": returncode,
        "run_dir": str(completed) if completed is not None else None,
        "elapsed_seconds": time.monotonic() - started,
        "log": str(log_path),
    }
    atomic_write_json(attempt_dir / f"{label}.result.json", result)
    if result["status"] != "completed":
        raise RuntimeError(f"Native task failed: {label}; see {log_path}.")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    study_dir = (
        args.study_dir or (_ROOT / "results" / STUDY_ID)
    ).expanduser().resolve()
    source_commit = _require_clean_source_commit()
    prerequisites = _validate_prerequisites(args.aihwkit_python)
    _validate_prepared_study(study_dir)
    attempt_dir = study_dir / "launch" / _attempt_id()
    attempt_dir.mkdir(parents=True, exist_ok=False)
    environment = _task_environment(
        os.environ,
        aihwkit_python=prerequisites["aihwkit_python"],
        cuda_visible_devices=args.cuda_visible_devices,
    )
    stop_requested = [False]

    def request_stop(signum, frame) -> None:  # type: ignore[no-untyped-def]
        stop_requested[0] = True

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    contract = {
        "schema": "ebl.mnist_relu_drn.ibm_om_cell_aware_launcher",
        "schema_version": 1,
        "study_id": STUDY_ID,
        "formal_evidence": True,
        "phase": args.phase,
        "source_commit": source_commit,
        "cuda_visibility": {
            "policy": args.cuda_visible_devices,
            "effective_environment": environment.get("CUDA_VISIBLE_DEVICES"),
        },
        "heartbeat_seconds": HEARTBEAT_SECONDS,
        "monitoring_interval_seconds": 1800,
        "prerequisites": prerequisites,
        "assignment_85001_embargo": (
            "released_only_after_development_freeze"
        ),
        "created_at": _utc_now(),
    }
    atomic_write_json(attempt_dir / "contract.json", contract)
    task_results = []
    started = time.monotonic()
    try:
        if args.phase in {"development", "all"}:
            for task in development_tasks():
                task_results.append(
                    _run_task(
                        task,
                        python=Path(sys.executable).resolve(),
                        study_dir=study_dir,
                        attempt_dir=attempt_dir,
                        environment=environment,
                        stop_requested=stop_requested,
                    )
                )
            freeze = _development_freeze(
                study_dir,
                source_commit=source_commit,
            )
            freeze_path = study_dir / "launch" / "development_freeze.json"
            existing = _read_json(freeze_path)
            if existing is None:
                atomic_write_json(freeze_path, freeze)
            else:
                _load_development_freeze(
                    study_dir,
                    source_commit=source_commit,
                )
        if args.phase in {"heldout", "all"}:
            freeze = _load_development_freeze(
                study_dir,
                source_commit=source_commit,
            )
            for task in heldout_tasks(freeze):
                task_results.append(
                    _run_task(
                        task,
                        python=Path(sys.executable).resolve(),
                        study_dir=study_dir,
                        attempt_dir=attempt_dir,
                        environment=environment,
                        stop_requested=stop_requested,
                    )
                )
        summarize = subprocess.run(
            (
                str(Path(sys.executable).resolve()),
                "-m",
                "ebl",
                "study",
                "summarize",
                "--study-dir",
                str(study_dir),
            ),
            cwd=_ROOT,
            env=environment,
            check=False,
            capture_output=True,
            text=True,
        )
        summary_log = attempt_dir / "study-summarize.log"
        summary_log.write_text(
            summarize.stdout + summarize.stderr,
            encoding="utf-8",
        )
        if summarize.returncode != 0:
            raise RuntimeError(f"Study summarize failed; see {summary_log}.")
        atomic_write_json(
            attempt_dir / "launcher_result.json",
            {
                "status": "complete",
                "phase": args.phase,
                "source_commit": source_commit,
                "finished_at": _utc_now(),
                "elapsed_seconds": time.monotonic() - started,
                "tasks": task_results,
                "study_summarize_returncode": summarize.returncode,
            },
        )
        return 0
    except BaseException as error:
        atomic_write_json(
            attempt_dir / "launcher_result.json",
            {
                "status": "failed",
                "phase": args.phase,
                "source_commit": source_commit,
                "finished_at": _utc_now(),
                "elapsed_seconds": time.monotonic() - started,
                "tasks": task_results,
                "error": {
                    "type": type(error).__name__,
                    "message": str(error),
                },
            },
        )
        raise


if __name__ == "__main__":  # pragma: no cover - operational entry point
    raise SystemExit(main())
