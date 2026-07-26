#!/usr/bin/env python3
"""Guard and resume the Conv3 v7 LR pipeline from a Jean Zay login node.

This supervisor is deliberately a login-node process.  It performs only light
manifest/finalization work locally and submits GPU stages through Slurm.  It
never submits a later stage until the prior stage's immutable completion has
revalidated, and it refuses to run from inside a Slurm allocation.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.mnist_conv.identity import (
    code_provenance,
    normalize_code_provenance,
    sha256_file,
)
from experiments.mnist_conv.io import atomic_write_json, read_json
from experiments.mnist_conv.lr_artifacts import (
    load_stage_manifest,
    validate_stage_completion,
)
from experiments.mnist_conv.lr_study import (
    create_study,
    execute_local_stage,
    finalize_stage,
    publish_lr_stage_manifest,
    submit_slurm_stage,
)
from experiments.mnist_conv.lr_study_spec import LRStudySpec
from experiments.mnist_conv.lr_v6_jeanzay import verify_login_r3_allocation
from experiments.submit_mnist_conv_slurm import (
    _submitted_job_id,
    load_profile,
)


STATE_SCHEMA_VERSION = "mnist-conv-lr-v7-jeanzay-supervisor/v1"
V7_STUDY_ID = (
    "lrstudy_6e56b399cf3e522a2c3f39f66a51611147b6171dd3b9e223f20791bc6275caf0"
)
PIPELINE = (
    "audit",
    "probe",
    "preflight",
    "core_candidates",
    "select_core",
    "extension_candidates",
    "finalize",
)
LOCAL_CPU_STAGES = {"select_core", "finalize"}
SLURM_ACTIVE_STATES = {
    "CONFIGURING",
    "COMPLETING",
    "PENDING",
    "REQUEUED",
    "REQUEUE_FED",
    "REQUEUE_HOLD",
    "RESIZING",
    "RUNNING",
    "SIGNALING",
    "STAGE_OUT",
    "SUSPENDED",
}
SLURM_FAILED_STATES = {
    "BOOT_FAIL",
    "CANCELLED",
    "DEADLINE",
    "FAILED",
    "NODE_FAIL",
    "OUT_OF_MEMORY",
    "PREEMPTED",
    "REVOKED",
    "SPECIAL_EXIT",
    "STOPPED",
    "TIMEOUT",
}
THREAD_EXPORTS = {
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}


def _error(expected: str, provided: Any) -> ValueError:
    return ValueError(f"Expected {expected}. Provided value: {provided!r}.")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def require_login_node_context(
    environment: Mapping[str, str] | None = None,
) -> None:
    """Reject execution from a Slurm allocation or job step."""

    values = os.environ if environment is None else environment
    observed = {
        key: values.get(key)
        for key in (
            "SLURM_JOB_ID",
            "SLURM_JOB_NODELIST",
            "SLURM_STEP_ID",
            "SLURM_STEP_NODELIST",
        )
        if values.get(key)
    }
    if observed:
        raise RuntimeError(
            "Expected the v7 supervisor to run on a Jean Zay login node, "
            "outside every Slurm allocation and job step. "
            f"Provided Slurm environment: {observed!r}."
        )


def _normalized_slurm_state(value: str, *, job_id: str) -> str:
    state = value.strip().split()[0].rstrip("+") if value.strip() else ""
    if (
        state != "COMPLETED"
        and state not in SLURM_ACTIVE_STATES
        and state not in SLURM_FAILED_STATES
    ):
        raise RuntimeError(
            "Expected a recognized Slurm allocation state. "
            f"Provided value: job_id={job_id!r}, state={state!r}."
        )
    return state


def query_array_worker_state(
    job_id: str,
    expected_task_count: int,
    *,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> dict[str, Any]:
    """Validate one Slurm array parent and all expected allocation tasks."""

    if not isinstance(job_id, str) or not job_id.isdigit():
        raise _error("a numeric Slurm array worker job id", job_id)
    if type(expected_task_count) is not int or expected_task_count < 1:
        raise _error(
            "expected_task_count to be an integer >= 1",
            expected_task_count,
        )
    command = [
        "sacct",
        "-n",
        "-X",
        "-j",
        job_id,
        "--format=JobIDRaw,JobID,State",
        "--parsable2",
    ]
    completed = runner(command, check=True, text=True, capture_output=True)
    parent_states: list[str] = []
    task_states: dict[int, str] = {}
    compressed_rows: list[str] = []
    compressed_states: list[str] = []
    task_raw_ids: dict[int, str] = {}
    task_pattern = re.compile(rf"{re.escape(job_id)}_([0-9]+)\Z")
    for raw_line in completed.stdout.splitlines():
        fields = raw_line.split("|")
        if len(fields) < 3:
            continue
        raw_id = fields[0].strip()
        display_id = fields[1].strip()
        raw_state = fields[2].strip()
        if display_id == job_id:
            parent_states.append(
                _normalized_slurm_state(raw_state, job_id=display_id)
            )
            continue
        task_match = task_pattern.fullmatch(display_id)
        if task_match is not None:
            task_index = int(task_match.group(1))
            if task_index in task_states:
                raise RuntimeError(
                    "Expected one sacct allocation row per v7 worker task. "
                    f"Provided duplicate task: {display_id!r}."
                )
            task_states[task_index] = _normalized_slurm_state(
                raw_state, job_id=display_id
            )
            task_raw_ids[task_index] = raw_id
            continue
        if display_id.startswith(f"{job_id}_["):
            # Slurm may retain one compressed pending-array row before tasks
            # receive individual accounting records.
            compressed_rows.append(display_id)
            compressed_states.append(
                _normalized_slurm_state(raw_state, job_id=display_id)
            )
    if len(parent_states) > 1:
        raise RuntimeError(
            "Expected sacct to return at most one allocation row for the v7 "
            f"worker array parent {job_id!r}. Provided states: {parent_states!r}."
        )
    parent_state = (
        parent_states[0] if parent_states else "ACCOUNTING_PENDING"
    )
    if not parent_states and compressed_states:
        unique_compressed_states = set(compressed_states)
        if len(unique_compressed_states) != 1:
            raise RuntimeError(
                "Expected one consistent state across compressed v7 array "
                f"rows. Provided states: {compressed_states!r}."
            )
        parent_state = compressed_states[0]
    elif not parent_states:
        active_task_states = [
            state
            for state in task_states.values()
            if state in SLURM_ACTIVE_STATES
        ]
        if "RUNNING" in active_task_states:
            parent_state = "RUNNING"
        elif active_task_states:
            parent_state = active_task_states[0]
    expected_indices = set(range(expected_task_count))
    observed_indices = set(task_states)
    unexpected = sorted(observed_indices - expected_indices)
    missing = sorted(expected_indices - observed_indices)
    if unexpected:
        raise RuntimeError(
            "Expected only the manifest-indexed v7 worker array tasks. "
            f"Provided unexpected indices: {unexpected!r}."
        )
    failed_tasks = {
        index: state
        for index, state in sorted(task_states.items())
        if state in SLURM_FAILED_STATES
    }
    if failed_tasks:
        raise RuntimeError(
            "Expected every observed v7 worker array task to avoid terminal "
            f"failure. Provided failed tasks: {failed_tasks!r}."
        )
    if parent_state in SLURM_FAILED_STATES:
        raise RuntimeError(
            "Expected the v7 worker array parent to avoid terminal failure. "
            f"Provided job_id={job_id!r}, state={parent_state!r}."
        )
    non_complete = {
        index: state
        for index, state in sorted(task_states.items())
        if state != "COMPLETED"
    }
    completion_basis: str | None = None
    if parent_state == "COMPLETED":
        if missing:
            raise RuntimeError(
                "Expected a completed v7 worker array parent to retain every "
                f"manifest task row. Provided missing indices: {missing!r}."
            )
        if non_complete:
            raise RuntimeError(
                "Expected every task under a completed v7 worker array parent "
                f"to be COMPLETED. Provided mixed task states: {non_complete!r}."
            )
        status = "complete"
        completion_basis = "explicit_parent_and_all_indexed_tasks"
    elif (
        parent_state == "ACCOUNTING_PENDING"
        and not missing
        and not non_complete
    ):
        # Jean Zay may omit a separate array-parent allocation row after all
        # children finish. The indexed ``JobID`` field remains authoritative;
        # ``JobIDRaw`` is only a per-allocation numeric identifier.
        parent_state = "COMPLETED"
        status = "complete"
        completion_basis = "all_indexed_tasks_without_parent_row"
    else:
        # Accounting records can lag briefly after sbatch. Missing tasks are
        # acceptable only while the parent itself is not terminal.
        status = "waiting"
    return {
        "status": status,
        "parent_state": parent_state,
        "expected_task_count": expected_task_count,
        "observed_task_count": len(task_states),
        "completed_task_count": sum(
            state == "COMPLETED" for state in task_states.values()
        ),
        "missing_task_indices": missing,
        "task_states": {
            str(index): state for index, state in sorted(task_states.items())
        },
        "compressed_rows": compressed_rows,
        "compressed_states": compressed_states,
        "task_raw_ids": {
            str(index): raw_id
            for index, raw_id in sorted(task_raw_ids.items())
        },
        "singleton_task_recorded_as_parent": (
            expected_task_count == 1
            and task_raw_ids.get(0) == job_id
        ),
        "completion_basis": completion_basis,
    }


def _convert_legacy_cpu_finalizer_record(
    record: dict[str, Any],
) -> None:
    """Preserve, then disable, any pre-supervisor-fix CPU finalizer state."""

    mode = "login_node_local_after_array_validation"
    if record.get("finalizer_mode") == mode:
        if record.get("finalizer_submission_status") != "disabled":
            raise RuntimeError(
                "Expected converted CPU-finalizer state to remain disabled. "
                f"Provided value: {record.get('finalizer_submission_status')!r}."
            )
        return
    legacy_fields = (
        "finalizer_job_id",
        "finalizer_submission_status",
        "finalizer_command",
        "finalizer_submission_started_at_utc",
        "finalizer_submitted_at_utc",
        "finalizer_submission_error",
    )
    observed = {
        field: record[field] for field in legacy_fields if field in record
    }
    previous = record.get("legacy_cpu_finalizer")
    if previous is not None and not isinstance(previous, dict):
        raise _error("legacy_cpu_finalizer to be a JSON object", previous)
    if observed:
        if previous is not None and previous != observed:
            raise RuntimeError(
                "Expected the preserved legacy CPU finalizer record to remain "
                f"unchanged. Provided old={previous!r}, current={observed!r}."
            )
        record["legacy_cpu_finalizer"] = observed
        for field in legacy_fields:
            record.pop(field, None)
    record["finalizer_mode"] = mode
    record["finalizer_submission_status"] = "disabled"


def require_full_preflight_pass(summary_path: str | Path) -> dict[str, Any]:
    """Recheck every frozen pass condition before core publication."""

    source = Path(summary_path).expanduser().resolve()
    value = read_json(source)
    if not isinstance(value, dict):
        raise _error("the v7 preflight summary to be a JSON object", value)
    required_true = (
        "device_is_v100",
        "device_capacity_passed",
        "memory_passed",
        "runtime_passed",
    )
    failures: list[str] = []
    if value.get("schema_version") != "mnist-conv-lr-v7-preflight/v1":
        failures.append("schema_version")
    if value.get("status") != "passed":
        failures.append("status")
    if value.get("failure_reason") is not None:
        failures.append("failure_reason")
    for field in required_true:
        if value.get(field) is not True:
            failures.append(field)
    if value.get("safety_gate_failure") is not None:
        failures.append("safety_gate_failure")
    measured = value.get("measured_steps")
    completed = value.get("completed_measured_steps")
    if (
        type(measured) is not int
        or type(completed) is not int
        or measured <= 0
        or completed != measured
    ):
        failures.append("completed_measured_steps")
    if not isinstance(value.get("validation_metrics"), dict):
        failures.append("validation_metrics")
    headroom = value.get("memory_headroom_fraction")
    required_headroom = value.get("required_memory_headroom_fraction")
    if (
        isinstance(headroom, bool)
        or not isinstance(headroom, (int, float))
        or isinstance(required_headroom, bool)
        or not isinstance(required_headroom, (int, float))
        or float(headroom) < float(required_headroom)
    ):
        failures.append("memory_headroom_fraction")
    projected = value.get("projected_candidate_hours")
    maximum = value.get("maximum_projected_candidate_hours")
    if (
        isinstance(projected, bool)
        or not isinstance(projected, (int, float))
        or isinstance(maximum, bool)
        or not isinstance(maximum, (int, float))
        or float(projected) > float(maximum)
    ):
        failures.append("projected_candidate_hours")
    if failures:
        raise RuntimeError(
            "Expected the complete Conv3 v7 V100 preflight contract to pass "
            f"before core submission. Provided failing fields: {sorted(set(failures))!r}; "
            f"summary={source}."
        )
    return value


def extension_is_explicit_no_op(manifest: Mapping[str, Any]) -> bool:
    """Recognize only the canonical single-entry zero-work extension."""

    entries = manifest.get("entries")
    if not isinstance(entries, list) or not entries:
        raise _error("extension manifest entries to be a non-empty list", entries)
    no_op_entries = [
        entry
        for entry in entries
        if isinstance(entry, dict)
        and isinstance(entry.get("payload"), dict)
        and entry["payload"].get("no_op") is True
    ]
    if no_op_entries and (
        len(entries) != 1
        or len(no_op_entries) != 1
        or no_op_entries[0].get("entry_id") != "no-expansion-required"
        or no_op_entries[0]["payload"].get("reason")
        != "no_scheme_requested_expansion"
    ):
        raise RuntimeError(
            "Expected a v7 extension manifest to contain either only training "
            "entries or the exact single no-expansion entry. "
            f"Provided entries: {entries!r}."
        )
    return bool(no_op_entries)


def _augment_worker_contract(command: Sequence[str]) -> list[str]:
    """Freeze physical-core and one-thread exports for supervisor submissions."""

    values = list(command)
    if "--hint=nomultithread" not in values:
        export_index = next(
            (
                index
                for index, item in enumerate(values)
                if item.startswith("--export=")
            ),
            None,
        )
        if export_index is None:
            raise RuntimeError(
                "Expected the generated v7 worker command to contain --export."
            )
        values.insert(export_index, "--hint=nomultithread")
    export_indices = [
        index for index, item in enumerate(values) if item.startswith("--export=")
    ]
    if len(export_indices) != 1:
        raise RuntimeError(
            "Expected exactly one Slurm export argument in the v7 worker command. "
            f"Provided value: {values!r}."
        )
    export_index = export_indices[0]
    export_value = values[export_index]
    existing_keys = {
        item.split("=", maxsplit=1)[0]
        for item in export_value.removeprefix("--export=").split(",")
        if "=" in item
    }
    additions = [
        f"{key}={value}"
        for key, value in THREAD_EXPORTS.items()
        if key not in existing_keys
    ]
    if additions:
        values[export_index] = export_value + "," + ",".join(additions)
    return values


class V7JeanZaySupervisor:
    """Persistent state machine for the guarded v7 public stages."""

    def __init__(
        self,
        *,
        config_path: str | Path,
        profile_path: str | Path,
        results_root: str | Path,
        data_root: str | Path,
        runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
        sleeper: Callable[[float], None] = time.sleep,
        allocation_verifier: Callable[..., dict[str, Any]] = (
            verify_login_r3_allocation
        ),
    ):
        self.config_path = Path(config_path).expanduser().resolve()
        self.profile_path = Path(profile_path).expanduser().resolve()
        self.results_root = Path(results_root).expanduser().resolve()
        self.data_root = Path(data_root).expanduser().resolve()
        self.runner = runner
        self.sleeper = sleeper
        self.allocation_verifier = allocation_verifier
        self.spec: LRStudySpec | None = None
        self.study_dir: Path | None = None
        self.state_path: Path | None = None
        self.state: dict[str, Any] | None = None
        self.provenance: dict[str, Any] | None = None

    def initialize(self) -> None:
        require_login_node_context()
        for path, label in (
            (self.config_path, "v7 study config"),
            (self.profile_path, "v7 executor profile"),
        ):
            if not path.is_file():
                raise _error(f"{label} to be an existing file", path)
        if not self.data_root.is_dir():
            raise _error(
                "ordinary-MNIST data root to be an existing directory",
                self.data_root,
            )
        spec = LRStudySpec.from_path(self.config_path)
        if (
            spec.data.get("schema_version") != "mnist-conv-lr-study/v7"
            or spec.study_id != V7_STUDY_ID
        ):
            raise _error(
                f"the frozen Conv3 v7 study {V7_STUDY_ID!r}",
                {
                    "schema_version": spec.data.get("schema_version"),
                    "study_id": spec.study_id,
                },
            )
        profile = load_profile(self.profile_path)
        if (
            profile["name"] != "jeanzay-v100-conv3-lr-v7"
            or profile["concurrency"] != 27
        ):
            raise _error(
                "the frozen Conv3 v7 Jean Zay profile with concurrency 27",
                {
                    "name": profile["name"],
                    "concurrency": profile["concurrency"],
                },
            )
        provenance = normalize_code_provenance(code_provenance())
        allocation = self.allocation_verifier(runner=self.runner)
        if (
            allocation.get("verified") is not True
            or allocation.get("allocation_id") != "AD010913993R3"
            or allocation.get("slurm_account") != "fmu@v100"
        ):
            raise RuntimeError(
                "Expected a verified Jean Zay R3 fmu@v100 login allocation. "
                f"Provided value: {allocation!r}."
            )
        study_dir, _contract = create_study(spec, self.results_root)
        state_path = study_dir / "supervisor" / "jeanzay-v7.json"
        expected_binding = {
            "study_id": spec.study_id,
            "study_dir": str(study_dir),
            "config_path": str(self.config_path),
            "config_sha256": sha256_file(self.config_path),
            "profile_path": str(self.profile_path),
            "profile_sha256": sha256_file(self.profile_path),
            "data_root": str(self.data_root),
            "code_provenance": provenance,
        }
        if state_path.is_file():
            state = read_json(state_path)
            if not isinstance(state, dict):
                raise _error("supervisor state to be a JSON object", state)
            if state.get("schema_version") != STATE_SCHEMA_VERSION:
                raise _error(
                    f"supervisor schema_version {STATE_SCHEMA_VERSION!r}",
                    state.get("schema_version"),
                )
            observed_binding = {
                key: state.get(key) for key in expected_binding
            }
            if observed_binding != expected_binding:
                raise RuntimeError(
                    "Expected a resumed v7 supervisor to retain identical source, "
                    "config, profile, data, and study bindings. "
                    f"Provided value: expected={expected_binding!r}, "
                    f"observed={observed_binding!r}."
                )
            if not isinstance(state.get("stages"), dict):
                raise _error("supervisor stages to be a JSON object", state.get("stages"))
            state["live_allocation_audit"] = allocation
        else:
            state = {
                "schema_version": STATE_SCHEMA_VERSION,
                **expected_binding,
                "execution_host": "jean_zay_login_node",
                "created_at_utc": _utc_now(),
                "updated_at_utc": _utc_now(),
                "status": "active",
                "live_allocation_audit": allocation,
                "stages": {},
            }
        self.spec = spec
        self.study_dir = study_dir
        self.state_path = state_path
        self.state = state
        self.provenance = provenance
        self._save()

    def _require_initialized(
        self,
    ) -> tuple[LRStudySpec, Path, dict[str, Any], dict[str, Any]]:
        if (
            self.spec is None
            or self.study_dir is None
            or self.state is None
            or self.provenance is None
            or self.state_path is None
        ):
            raise RuntimeError("Expected the v7 supervisor to be initialized.")
        return self.spec, self.study_dir, self.state, self.provenance

    def _save(self) -> None:
        _spec, _study, state, _provenance = self._require_initialized()
        state["updated_at_utc"] = _utc_now()
        atomic_write_json(self.state_path, state, canonical=True)

    def _completion_path(self, stage: str) -> Path:
        _spec, study, _state, _provenance = self._require_initialized()
        return study / "stages" / stage / "complete.json"

    def _manifest_path(self, stage: str) -> Path:
        _spec, study, _state, _provenance = self._require_initialized()
        return study / "stages" / stage / "manifest.json"

    def _validate_and_mark_complete(
        self, stage: str, manifest_path: Path
    ) -> None:
        _spec, study, state, _provenance = self._require_initialized()
        validate_stage_completion(
            study_dir=study,
            manifest_path=manifest_path,
        )
        record = state["stages"].setdefault(stage, {})
        record.update(
            {
                "status": "complete",
                "completion_path": str(self._completion_path(stage)),
                "completed_at_utc": _utc_now(),
            }
        )
        self._save()

    def _ensure_manifest(
        self, stage: str
    ) -> tuple[dict[str, Any], Path, dict[str, Any]]:
        spec, study, state, provenance = self._require_initialized()
        manifest_path = self._manifest_path(stage)
        record = state["stages"].get(stage)
        if manifest_path.is_file() and record is None:
            if self._completion_path(stage).is_file():
                manifest = load_stage_manifest(
                    manifest_path,
                    study_dir=study,
                    expected_study_id=spec.study_id,
                    expected_stage_name=stage,
                )
                state["stages"][stage] = {
                    "status": "adopted_complete",
                    "manifest_path": str(manifest_path),
                    "manifest_sha256": sha256_file(manifest_path),
                    "entry_count": len(manifest["entries"]),
                }
                self._validate_and_mark_complete(stage, manifest_path)
                return manifest, manifest_path, state["stages"][stage]
            raise RuntimeError(
                "Expected an incomplete stage manifest to have a supervisor "
                "submission record before resumption; refusing a potentially "
                f"duplicate submission. Provided stage: {stage!r}."
            )
        if record is None:
            record = {
                "status": "planning",
                "planning_started_at_utc": _utc_now(),
            }
            state["stages"][stage] = record
            self._save()
        manifest, published_path = publish_lr_stage_manifest(
            study, spec, stage, provenance
        )
        digest = sha256_file(published_path)
        if record.get("manifest_sha256") not in {None, digest}:
            raise RuntimeError(
                "Expected the resumed stage manifest digest to remain unchanged. "
                f"Provided stage={stage!r}, recorded={record.get('manifest_sha256')!r}, "
                f"observed={digest!r}."
            )
        record.update(
            {
                "manifest_path": str(published_path),
                "manifest_sha256": digest,
                "entry_count": len(manifest["entries"]),
            }
        )
        self._save()
        return manifest, published_path, record

    def _run_local_stage(
        self,
        stage: str,
        manifest_path: Path,
        record: dict[str, Any],
    ) -> None:
        _spec, study, _state, provenance = self._require_initialized()
        record.update(
            {
                "executor": "login_node_local_cpu",
                "status": "local_running",
                "local_started_at_utc": _utc_now(),
            }
        )
        self._save()
        execute_local_stage(
            study_dir=study,
            manifest_path=manifest_path,
            data_root=self.data_root,
            download=False,
            device="cpu",
            workers=1,
            provenance=provenance,
        )
        self._validate_and_mark_complete(stage, manifest_path)

    def _submit_one(
        self,
        *,
        record: dict[str, Any],
        kind: str,
        command: Sequence[str],
    ) -> str:
        id_key = f"{kind}_job_id"
        status_key = f"{kind}_submission_status"
        if record.get(id_key) is not None:
            value = record[id_key]
            if not isinstance(value, str) or not value.isdigit():
                raise _error(f"recorded {id_key} to be numeric", value)
            return value
        previous = record.get(status_key)
        if previous in {"started", "ambiguous"}:
            raise RuntimeError(
                "Expected no ambiguous prior sbatch call before resumption; "
                "refusing a duplicate submission. "
                f"Provided kind={kind!r}, status={previous!r}."
            )
        record.update(
            {
                status_key: "started",
                f"{kind}_command": list(command),
                f"{kind}_submission_started_at_utc": _utc_now(),
            }
        )
        self._save()
        try:
            completed = self.runner(
                list(command),
                check=True,
                text=True,
                capture_output=True,
            )
        except subprocess.CalledProcessError as exc:
            record[status_key] = "rejected"
            record[f"{kind}_submission_error"] = str(exc)
            self._save()
            raise
        except BaseException:
            # The call may have reached Slurm even when the local process did
            # not receive its reply.  Fail closed rather than submit twice.
            record[status_key] = "ambiguous"
            self._save()
            raise
        job_id = _submitted_job_id(completed)
        record.update(
            {
                id_key: job_id,
                status_key: "submitted",
                f"{kind}_submitted_at_utc": _utc_now(),
            }
        )
        self._save()
        return job_id

    def _advance_slurm_stage(
        self,
        stage: str,
        manifest_path: Path,
        record: dict[str, Any],
    ) -> dict[str, Any]:
        _spec, study, _state, _provenance = self._require_initialized()
        # ``submit_slurm_stage(..., dry_run=True)`` intentionally does not
        # create log directories.  The supervisor submits the returned worker
        # command itself, so it must materialize the compute-visible path
        # before either a new or already-pending array starts.
        (study / "slurm" / stage).mkdir(parents=True, exist_ok=True)
        commands = submit_slurm_stage(
            study_dir=study,
            manifest_path=manifest_path,
            data_root=self.data_root,
            device="cuda",
            profile_path=self.profile_path,
            dry_run=True,
        )
        worker_command = _augment_worker_contract(commands["worker"])
        if stage == "core_candidates" and "--array=0-26%27" not in worker_command:
            raise RuntimeError(
                "Expected the frozen v7 core submission to be array 0-26%27. "
                f"Provided value: {worker_command!r}."
            )
        record["executor"] = "slurm_v100"
        worker_id = self._submit_one(
            record=record,
            kind="worker",
            command=worker_command,
        )
        _convert_legacy_cpu_finalizer_record(record)
        record["status"] = "waiting_for_worker_array"
        self._save()
        try:
            array = query_array_worker_state(
                worker_id,
                len(
                    load_stage_manifest(
                        manifest_path, study_dir=study
                    )["entries"]
                ),
                runner=self.runner,
            )
        except RuntimeError as exc:
            record["status"] = "failed_worker_array_validation"
            record["worker_array_validation_error"] = str(exc)
            self._save()
            raise
        record["last_worker_array_state"] = array
        record["last_polled_at_utc"] = _utc_now()
        self._save()
        if array["status"] == "waiting":
            return {
                "status": "waiting",
                "stage": stage,
                "worker_job_id": worker_id,
                "worker_parent_state": array["parent_state"],
                "completed_task_count": array["completed_task_count"],
                "expected_task_count": array["expected_task_count"],
                "missing_task_indices": array["missing_task_indices"],
            }
        if array["status"] != "complete":
            raise RuntimeError(
                "Expected a complete or waiting worker-array state. "
                f"Provided value: {array!r}."
            )
        finalize_stage(study, manifest_path)
        self._validate_and_mark_complete(stage, manifest_path)
        return {
            "status": "stage_complete",
            "stage": stage,
            "worker_job_id": worker_id,
            "worker_parent_state": array["parent_state"],
            "completed_task_count": array["completed_task_count"],
            "expected_task_count": array["expected_task_count"],
        }

    def advance_once(self) -> dict[str, Any]:
        spec, study, state, _provenance = self._require_initialized()
        for stage in PIPELINE:
            manifest_path = self._manifest_path(stage)
            completion_path = self._completion_path(stage)
            record = state["stages"].get(stage)
            if completion_path.is_file():
                if not manifest_path.is_file():
                    raise RuntimeError(
                        "Expected a completed stage to retain its manifest. "
                        f"Provided stage: {stage!r}."
                    )
                self._validate_and_mark_complete(stage, manifest_path)
                continue
            if record is not None and record.get("status") == "complete":
                raise RuntimeError(
                    "Expected a supervisor-complete stage to retain its immutable "
                    f"completion marker. Provided stage: {stage!r}."
                )
            if stage == "core_candidates":
                require_full_preflight_pass(
                    study
                    / "stages"
                    / "preflight"
                    / "entries"
                    / "ours-worst-cost"
                    / "summary.json"
                )
            manifest, manifest_path, record = self._ensure_manifest(stage)
            if stage in LOCAL_CPU_STAGES:
                self._run_local_stage(stage, manifest_path, record)
                continue
            if stage == "extension_candidates" and extension_is_explicit_no_op(
                manifest
            ):
                self._run_local_stage(stage, manifest_path, record)
                continue
            result = self._advance_slurm_stage(
                stage,
                manifest_path,
                record,
            )
            if result["status"] in {"waiting", "stage_complete"}:
                return result
        state["status"] = "complete"
        state["completed_at_utc"] = _utc_now()
        self._save()
        return {
            "status": "complete",
            "study_id": spec.study_id,
            "study_dir": str(study),
            "state_path": str(self.state_path),
        }

    def run(self, *, poll_seconds: float, once: bool = False) -> dict[str, Any]:
        if (
            isinstance(poll_seconds, bool)
            or not isinstance(poll_seconds, (int, float))
            or poll_seconds <= 0
        ):
            raise _error("poll_seconds to be a positive number", poll_seconds)
        while True:
            result = self.advance_once()
            print(json.dumps(result, sort_keys=True, allow_nan=False), flush=True)
            if once or result["status"] == "complete":
                return result
            if result["status"] == "stage_complete":
                continue
            self.sleeper(float(poll_seconds))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    repo_root = Path(__file__).resolve().parents[1]
    parser.add_argument(
        "--config",
        default=str(
            repo_root
            / "configs"
            / "conv"
            / "hardsigmoid_lr_conv3_scheme_two_rho_constant_sgd_bs16_v7.json"
        ),
    )
    parser.add_argument(
        "--profile",
        default=str(
            repo_root
            / "configs"
            / "executors"
            / "jeanzay-v100-conv3-lr-v7.json"
        ),
    )
    parser.add_argument("--results-root", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    parser.add_argument(
        "--once",
        action="store_true",
        help="Advance or poll once, then exit without waiting.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    require_login_node_context()
    results_root = Path(args.results_root).expanduser().resolve()
    results_root.mkdir(parents=True, exist_ok=True)
    lock_path = results_root / f".{V7_STUDY_ID}.supervisor.lock"
    with lock_path.open("a+", encoding="utf-8") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(
                "Expected only one active Conv3 v7 Jean Zay supervisor. "
                f"Provided locked path: {lock_path}."
            ) from exc
        supervisor = V7JeanZaySupervisor(
            config_path=args.config,
            profile_path=args.profile,
            results_root=results_root,
            data_root=args.data_root,
        )
        supervisor.initialize()
        supervisor.run(
            poll_seconds=args.poll_seconds,
            once=args.once,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
