"""Subprocess-only runner for experiments in multiple Git worktrees."""

from __future__ import annotations

from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
from typing import Any, Mapping

from campaigns.schema import CampaignSpec, InputRef, StageSpec, TargetSpec
from campaigns.schema import topological_stages
from experiments.artifacts import (
    atomic_write_json,
    content_hash,
    sha256_file,
)


_SOURCE_SUFFIXES = {".json", ".md", ".py", ".toml", ".yaml", ".yml"}


def _git_identity(target: TargetSpec) -> dict[str, Any]:
    commit = subprocess.check_output(
        ("git", "-C", str(target.worktree), "rev-parse", "HEAD"),
    ).decode("ascii").strip()
    status = subprocess.check_output(
        (
            "git",
            "-C",
            str(target.worktree),
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
            "-z",
        ),
    )
    diff = subprocess.check_output(
        ("git", "-C", str(target.worktree), "diff", "--binary", "HEAD"),
    )
    untracked_source_hashes = {}
    for record in status.split(b"\0"):
        if not record.startswith(b"?? "):
            continue
        relative = record[3:].decode(
            "utf-8",
            errors="surrogateescape",
        )
        candidate = target.worktree / relative
        if (
            candidate.is_file()
            and candidate.suffix.lower() in _SOURCE_SUFFIXES
        ):
            untracked_source_hashes[relative] = sha256_file(candidate)
    return {
        "commit": commit,
        "dirty": bool(status),
        "dirty_hash": (
            content_hash(
                {
                    "status": status.decode(
                        "utf-8",
                        errors="surrogateescape",
                    ),
                    "diff_sha256": sha256(diff).hexdigest(),
                    "untracked_source_sha256": untracked_source_hashes,
                }
            )
            if status
            else None
        ),
    }


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def _preflight(
    target: TargetSpec,
    *,
    allow_dirty: bool,
) -> Mapping[str, Any]:
    if not target.worktree.is_dir():
        raise ValueError(
            "Expected campaign target.worktree to be an existing directory. "
            f"Provided value: {str(target.worktree)!r}."
        )
    if not target.python.is_file() or not os.access(target.python, os.X_OK):
        raise ValueError(
            "Expected campaign target.python to be an existing executable. "
            f"Provided value: {str(target.python)!r}."
        )
    source = _git_identity(target)
    if source["dirty"] and not allow_dirty:
        raise RuntimeError(
            "Expected campaign target to have a clean worktree unless "
            f"--allow-dirty is used. Provided value: {target.target_id!r}."
        )
    completed = subprocess.run(
        (
            str(target.python),
            "-m",
            "ebl",
            "describe",
            "--experiment",
            "small_drn.v1",
            "--json",
        ),
        cwd=target.worktree,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "Expected campaign target describe preflight to succeed. "
            f"Provided value: target={target.target_id!r}, "
            f"returncode={completed.returncode}, output="
            f"{completed.stderr.strip() or completed.stdout.strip()!r}."
        )
    try:
        description = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            "Expected campaign target describe to return valid JSON. "
            f"Provided value: target={target.target_id!r}, "
            f"stdout={completed.stdout!r}."
        ) from exc
    if description.get("protocol_version") != 1:
        raise RuntimeError(
            "Expected target protocol_version to be 1. "
            f"Provided value: {description.get('protocol_version')!r}."
        )
    return {"source": source, "description": description}


def _string_list(value: Any, *, name: str) -> tuple[str, ...]:
    if not isinstance(value, list) or any(
        not isinstance(item, str) for item in value
    ):
        raise RuntimeError(
            f"Expected target describe {name} to be a list of strings. "
            f"Provided value: {value!r}."
        )
    return tuple(value)


def _validate_stage_capability(
    stage: StageSpec,
    *,
    target: TargetSpec,
    description: Mapping[str, Any],
) -> None:
    capabilities = description.get("capabilities")
    if not isinstance(capabilities, Mapping):
        raise RuntimeError(
            "Expected target describe capabilities to be an object. "
            f"Provided value: target={target.target_id!r}, "
            f"capabilities={capabilities!r}."
        )
    supported = _string_list(
        capabilities.get("supported_commands"),
        name="capabilities.supported_commands",
    )
    if stage.command not in supported:
        raise RuntimeError(
            "Expected campaign stage.command to be advertised by its target. "
            f"Provided value: target={target.target_id!r}, "
            f"stage={stage.stage_id!r}, command={stage.command!r}, "
            f"supported_commands={list(supported)!r}."
        )

    commands = description.get("commands")
    if not isinstance(commands, Mapping):
        raise RuntimeError(
            "Expected target describe commands to be an object. "
            f"Provided value: target={target.target_id!r}, "
            f"commands={commands!r}."
        )
    command_capability = commands.get(stage.command)
    if (
        not isinstance(command_capability, Mapping)
        or command_capability.get("available") is not True
    ):
        raise RuntimeError(
            "Expected campaign stage.command to be available on its target. "
            f"Provided value: target={target.target_id!r}, "
            f"stage={stage.stage_id!r}, command={stage.command!r}, "
            f"capability={command_capability!r}."
        )

    required_options = _string_list(
        command_capability.get("required_options", []),
        name=f"commands.{stage.command}.required_options",
    )
    exclusive_options = _string_list(
        command_capability.get("exclusive_input_options", []),
        name=f"commands.{stage.command}.exclusive_input_options",
    )
    provided_options = {
        "--" + name.replace("_", "-")
        for name in stage.inputs
    }
    operational_options = {"--config", "--output-dir"}
    missing_options = sorted(
        set(required_options) - operational_options - provided_options
    )
    if missing_options:
        raise RuntimeError(
            "Expected campaign stage.inputs to satisfy target command "
            f"requirements. Provided value: target={target.target_id!r}, "
            f"stage={stage.stage_id!r}, missing={missing_options!r}, "
            f"inputs={sorted(stage.inputs)!r}."
        )
    supplied_exclusive = sorted(
        provided_options & set(exclusive_options)
    )
    if len(supplied_exclusive) > 1:
        raise RuntimeError(
            "Expected campaign stage.inputs to select at most one mutually "
            "exclusive target input. "
            f"Provided value: target={target.target_id!r}, "
            f"stage={stage.stage_id!r}, "
            f"options={supplied_exclusive!r}."
        )
    known_input_options = (
        set(required_options)
        | set(exclusive_options)
    ) - operational_options
    unsupported_inputs = sorted(provided_options - known_input_options)
    if unsupported_inputs:
        raise RuntimeError(
            "Expected campaign stage.inputs to be accepted by its target "
            f"command. Provided value: target={target.target_id!r}, "
            f"stage={stage.stage_id!r}, "
            f"unsupported={unsupported_inputs!r}."
        )

    resume_capabilities = capabilities.get("resume")
    if not isinstance(resume_capabilities, Mapping):
        raise RuntimeError(
            "Expected target describe capabilities.resume to be an object. "
            f"Provided value: target={target.target_id!r}, "
            f"resume={resume_capabilities!r}."
        )
    capability_by_input = {
        "weights": "weights",
        "base_weights": "base_weights",
        "resume": "full_training_state",
    }
    unavailable = sorted(
        name
        for name in stage.inputs
        if resume_capabilities.get(capability_by_input[name]) is not True
    )
    if unavailable:
        raise RuntimeError(
            "Expected campaign stage input artifact modes to be advertised "
            f"by its target. Provided value: target={target.target_id!r}, "
            f"stage={stage.stage_id!r}, unavailable={unavailable!r}."
        )


def _validate_campaign_capabilities(
    spec: CampaignSpec,
    *,
    targets: Mapping[str, TargetSpec],
    target_info: Mapping[str, Mapping[str, Any]],
) -> None:
    for stage in spec.stages:
        _validate_stage_capability(
            stage,
            target=targets[stage.target_id],
            description=target_info[stage.target_id]["description"],
        )


def _resolved_input(
    input_ref: InputRef,
    *,
    stage_records: Mapping[str, Mapping[str, Any]],
) -> Path:
    if input_ref.path is not None:
        path = input_ref.path
    else:
        source = stage_records[input_ref.stage]
        result_path = Path(source["run_result"])
        result = _read_json(result_path)
        matches = [
            artifact
            for artifact in result.get("artifacts", ())
            if artifact.get("kind") == input_ref.artifact_kind
        ]
        if len(matches) != 1:
            raise RuntimeError(
                "Expected a stage input artifact kind to resolve exactly once. "
                f"Provided value: stage={input_ref.stage!r}, "
                f"kind={input_ref.artifact_kind!r}, count={len(matches)}."
            )
        path = result_path.parent / matches[0]["path"]
    if not path.is_file():
        raise ValueError(
            "Expected campaign stage input to be an existing file. "
            f"Provided value: {str(path)!r}."
        )
    return path.resolve()


def _stage_fingerprint(
    stage: StageSpec,
    *,
    target_identity: Mapping[str, Any],
    resolved_inputs: Mapping[str, Path | str],
) -> str:
    value = {
        "protocol_version": 1,
        "stage": {
            "id": stage.stage_id,
            "case_id": stage.case_id,
            "target": stage.target_id,
            "command": stage.command,
            "config_sha256": sha256_file(stage.config),
            "inputs": {
                key: (
                    {"path_sha256": sha256_file(value)}
                    if isinstance(value, Path)
                    else {"deferred": value}
                )
                for key, value in sorted(resolved_inputs.items())
            },
        },
        "target": target_identity,
    }
    return content_hash(value)


def _next_attempt(stage_root: Path) -> Path:
    existing = sorted(
        path
        for path in stage_root.glob("attempt-*")
        if path.is_dir() and path.name[8:].isdigit()
    )
    attempt_number = (
        max(int(path.name[8:]) for path in existing) + 1 if existing else 1
    )
    attempt = stage_root / f"attempt-{attempt_number:03d}"
    attempt.mkdir(parents=True, exist_ok=False)
    return attempt


def _reusable_record(
    stage_root: Path,
    *,
    fingerprint: str,
) -> Mapping[str, Any] | None:
    for candidate in sorted(
        stage_root.glob("attempt-*/stage_result.json"),
        reverse=True,
    ):
        record = _read_json(candidate)
        if (
            record.get("status") != "complete"
            or record.get("fingerprint") != fingerprint
        ):
            continue
        result_path = Path(record["run_result"])
        if (
            result_path.is_file()
            and sha256_file(result_path) == record.get("run_result_sha256")
        ):
            return record
    return None


def _find_run_result(output_dir: Path) -> Path:
    matches = sorted(output_dir.glob("*/result.json"))
    if len(matches) != 1:
        raise RuntimeError(
            "Expected one result.json below the stage output directory. "
            f"Provided value: directory={str(output_dir)!r}, "
            f"matches={[str(path) for path in matches]!r}."
        )
    result = _read_json(matches[0])
    if result.get("status") != "complete":
        raise RuntimeError(
            "Expected stage result status to be 'complete'. "
            f"Provided value: {result.get('status')!r}."
        )
    return matches[0].resolve()


def run_campaign(
    spec: CampaignSpec,
    *,
    output_root: Path,
    resume: bool = False,
    dry_run: bool = False,
    allow_dirty: bool = False,
    fail_fast: bool = False,
) -> dict[str, Mapping[str, Any]]:
    """Execute a campaign in dependency order.

    Independent stages continue after failures.  A dependent stage is recorded
    as ``skipped_dependency`` without launching its target process.
    """

    campaign_root = output_root.expanduser().resolve() / spec.campaign_id
    campaign_root.mkdir(parents=True, exist_ok=True)
    (campaign_root / "stages").mkdir(exist_ok=True)
    (campaign_root / "aggregate").mkdir(exist_ok=True)
    atomic_write_json(
        campaign_root / "campaign.resolved.json",
        {
            "schema_version": 1,
            "campaign_id": spec.campaign_id,
            "targets": [
                {
                    "id": item.target_id,
                    "worktree": str(item.worktree),
                    "python": str(item.python),
                }
                for item in spec.targets
            ],
            "stages": [
                {
                    "id": item.stage_id,
                    "case_id": item.case_id,
                    "target": item.target_id,
                    "command": item.command,
                    "config": str(item.config),
                    "depends_on": list(item.depends_on),
                    "inputs": {
                        key: {
                            "path": (
                                str(ref.path)
                                if ref.path is not None
                                else None
                            ),
                            "stage": ref.stage,
                            "artifact_kind": ref.artifact_kind,
                        }
                        for key, ref in item.inputs.items()
                    },
                }
                for item in spec.stages
            ],
        },
    )

    targets = {item.target_id: item for item in spec.targets}
    target_info = {
        target_id: _preflight(target, allow_dirty=allow_dirty)
        for target_id, target in targets.items()
    }
    _validate_campaign_capabilities(
        spec,
        targets=targets,
        target_info=target_info,
    )
    records: dict[str, Mapping[str, Any]] = {}
    stop_launching = False

    for stage in topological_stages(spec):
        stage_root = (
            campaign_root
            / "stages"
            / stage.case_id
            / stage.target_id
            / stage.stage_id
        )
        stage_root.mkdir(parents=True, exist_ok=True)
        failed_dependencies = [
            dependency
            for dependency in stage.depends_on
            if records[dependency]["status"]
            not in ({"complete", "dry_run"} if dry_run else {"complete"})
        ]
        if failed_dependencies or stop_launching:
            record = {
                "schema_version": 1,
                "stage_id": stage.stage_id,
                "case_id": stage.case_id,
                "target_id": stage.target_id,
                "status": (
                    "skipped_dependency"
                    if failed_dependencies
                    else "skipped_fail_fast"
                ),
                "failed_dependencies": failed_dependencies,
            }
            atomic_write_json(stage_root / "stage_result.json", record)
            records[stage.stage_id] = record
            continue

        if dry_run:
            resolved_inputs: Mapping[str, Path | str] = {
                key: (
                    ref.path
                    if ref.path is not None
                    else f"@stage:{ref.stage}:{ref.artifact_kind}"
                )
                for key, ref in stage.inputs.items()
            }
            missing_paths = [
                str(value)
                for value in resolved_inputs.values()
                if isinstance(value, Path) and not value.is_file()
            ]
            if missing_paths:
                raise ValueError(
                    "Expected literal campaign stage inputs to exist during "
                    f"dry-run. Provided value: missing {missing_paths!r}."
                )
        else:
            resolved_inputs = {
                key: _resolved_input(ref, stage_records=records)
                for key, ref in stage.inputs.items()
            }
        if not stage.config.is_file():
            raise ValueError(
                "Expected campaign stage.config to be an existing file. "
                f"Provided value: {str(stage.config)!r}."
            )
        fingerprint = _stage_fingerprint(
            stage,
            target_identity=target_info[stage.target_id]["source"],
            resolved_inputs=resolved_inputs,
        )
        if resume:
            reusable = _reusable_record(
                stage_root,
                fingerprint=fingerprint,
            )
            if reusable is not None:
                records[stage.stage_id] = reusable
                continue

        attempt = _next_attempt(stage_root)
        output_dir = attempt / "runs"
        command = [
            str(targets[stage.target_id].python),
            "-m",
            "ebl",
            stage.command,
            "--config",
            str(stage.config),
            "--output-dir",
            str(output_dir),
        ]
        for name, path in sorted(resolved_inputs.items()):
            command.extend((f"--{name.replace('_', '-')}", str(path)))
        plan = {
            "schema_version": 1,
            "stage_id": stage.stage_id,
            "case_id": stage.case_id,
            "target_id": stage.target_id,
            "target_source": dict(
                target_info[stage.target_id]["source"]
            ),
            "fingerprint": fingerprint,
            "command": command,
            "cwd": str(targets[stage.target_id].worktree),
        }
        atomic_write_json(attempt / "stage.plan.json", plan)
        if dry_run:
            record = {**plan, "status": "dry_run"}
            atomic_write_json(attempt / "stage_result.json", record)
            records[stage.stage_id] = record
            continue

        completed = subprocess.run(
            command,
            cwd=targets[stage.target_id].worktree,
            check=False,
            capture_output=True,
            text=True,
        )
        (attempt / "stdout.log").write_text(
            completed.stdout,
            encoding="utf-8",
        )
        (attempt / "stderr.log").write_text(
            completed.stderr,
            encoding="utf-8",
        )
        if completed.returncode == 0:
            try:
                run_result = _find_run_result(output_dir)
            except Exception as exc:
                status = "failed"
                error = {
                    "type": type(exc).__name__,
                    "message": str(exc),
                }
            else:
                status = "complete"
                error = None
        else:
            status = "failed"
            error = {
                "type": "SubprocessError",
                "message": (
                    completed.stderr.strip()
                    or completed.stdout.strip()
                    or f"exit code {completed.returncode}"
                ),
            }
            run_result = None
        record = {
            **plan,
            "status": status,
            "returncode": completed.returncode,
            "run_result": (
                str(run_result) if run_result is not None else None
            ),
            "run_result_sha256": (
                sha256_file(run_result) if run_result is not None else None
            ),
            "error": error,
        }
        atomic_write_json(attempt / "stage_result.json", record)
        records[stage.stage_id] = record
        if status == "failed" and fail_fast:
            stop_launching = True

    atomic_write_json(
        campaign_root / "aggregate" / "results.json",
        {
            "schema_version": 1,
            "campaign_id": spec.campaign_id,
            "stages": list(records.values()),
        },
    )
    return records
