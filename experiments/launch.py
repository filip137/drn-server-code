#!/usr/bin/env python3
"""Launch an existing experiment command locally, in tmux, or through Slurm."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TARGETS = REPO_ROOT / "configs" / "experiment_targets.json"
TARGET_KINDS = {"local", "tmux", "ssh-tmux", "slurm"}


def load_targets(path: Path) -> dict[str, dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object of targets, got {data!r}.")
    for name, target in data.items():
        if not isinstance(target, dict) or target.get("kind") not in TARGET_KINDS:
            raise ValueError(
                f"Expected target {name!r} to have kind in {sorted(TARGET_KINDS)}, "
                f"got {target!r}."
            )
        if not isinstance(target.get("workdir"), str):
            raise ValueError(
                f"Expected target {name!r} to have a string workdir, got {target!r}."
            )
    return data


def _detached_shell(argv: Sequence[str], workdir: str, log: str) -> str:
    log_path = Path(log)
    parent = str(log_path.parent)
    command = shlex.join(list(argv))
    return (
        f"cd {shlex.quote(workdir)} && "
        f"mkdir -p {shlex.quote(parent)} && "
        f"{command} > {shlex.quote(log)} 2>&1; "
        f"rc=$?; printf '%s\\n' \"$rc\" > {shlex.quote(log + '.exitcode')}; "
        'exit "$rc"'
    )


def _local_path(workdir: str, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else Path(workdir) / path


def build_launch_command(
    target: dict[str, Any],
    argv: Sequence[str],
    *,
    name: str,
    log: str | None,
    slurm_args: Sequence[str] = (),
) -> list[str]:
    kind = target["kind"]
    workdir = target["workdir"]

    if slurm_args and kind != "slurm":
        raise ValueError(
            f"Expected --slurm-arg only for a Slurm target. Provided kind: {kind!r}."
        )
    if kind == "local":
        return list(argv)
    if not log and kind in {"tmux", "ssh-tmux"}:
        raise ValueError(f"Expected --log for detached target kind {kind!r}, got {log!r}.")

    if kind == "tmux":
        return [
            "tmux",
            "new-window",
            "-dP",
            "-F",
            "#{window_id}",
            "-t",
            target["session"],
            "-n",
            name,
            "bash",
            "-lc",
            _detached_shell(argv, workdir, str(log)),
        ]

    if kind == "ssh-tmux":
        remote = [
            "tmux",
            "new-session",
            "-d",
            "-s",
            name,
            "bash",
            "-lc",
            _detached_shell(argv, workdir, str(log)),
        ]
        return ["ssh", target["host"], shlex.join(remote)]

    sbatch = [
        "sbatch",
        "--parsable",
        f"--job-name={name}",
        f"--chdir={workdir}",
        *target.get("sbatch", []),
        *slurm_args,
    ]
    if log:
        sbatch.append(f"--output={log}")
    sbatch.append(f"--wrap={shlex.join(list(argv))}")
    remote = shlex.join(sbatch)
    if log:
        remote = f"mkdir -p {shlex.quote(str(Path(log).parent))} && {remote}"
    return ["ssh", target["host"], remote]


def launch(
    target_name: str,
    target: dict[str, Any],
    argv: Sequence[str],
    *,
    name: str,
    log: str | None = None,
    slurm_args: Sequence[str] = (),
    dry_run: bool = False,
) -> dict[str, Any]:
    invocation = build_launch_command(
        target,
        argv,
        name=name,
        log=log,
        slurm_args=slurm_args,
    )
    result: dict[str, Any] = {
        "target": target_name,
        "kind": target["kind"],
        "name": name,
        "command": list(argv),
        "log": log,
        "slurm_args": list(slurm_args),
        "invocation": invocation,
    }
    if dry_run:
        result["state"] = "planned"
        return result

    if target["kind"] == "local":
        if log:
            log_path = _local_path(target["workdir"], log)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with log_path.open("w", encoding="utf-8") as handle:
                completed = subprocess.run(
                    invocation,
                    cwd=target["workdir"],
                    check=False,
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                )
            Path(str(log_path) + ".exitcode").write_text(
                f"{completed.returncode}\n", encoding="utf-8"
            )
        else:
            completed = subprocess.run(
                invocation,
                cwd=target["workdir"],
                check=False,
            )
        result.update(state="finished", returncode=completed.returncode)
        return result

    completed = subprocess.run(
        invocation,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode:
        raise RuntimeError(
            f"Launch failed with exit {completed.returncode}: "
            f"{completed.stderr.strip() or completed.stdout.strip()}"
        )
    output = completed.stdout.strip()
    if target["kind"] == "slurm":
        lines = [line.strip() for line in output.splitlines() if line.strip()]
        handle = lines[-1].split(";", 1)[0] if lines else ""
        if not handle.isdigit():
            raise RuntimeError(f"Expected numeric sbatch job ID, got {output!r}.")
    elif target["kind"] == "ssh-tmux":
        handle = name
    else:
        handle = output
    result.update(state="submitted", handle=handle)
    return result


def build_status_command(target: dict[str, Any], handle: str) -> list[str]:
    kind = target["kind"]
    if kind == "tmux":
        return ["tmux", "list-panes", "-t", handle, "-F", "#{pane_dead} #{pane_dead_status}"]
    if kind == "ssh-tmux":
        remote = ["tmux", "list-panes", "-t", handle, "-F", "#{pane_dead} #{pane_dead_status}"]
        return ["ssh", target["host"], shlex.join(remote)]
    if kind == "slurm":
        remote = [
            "sacct",
            "-j",
            handle,
            "--noheader",
            "--parsable2",
            "--format=State,ExitCode",
        ]
        return ["ssh", target["host"], shlex.join(remote)]
    raise ValueError(f"Expected a detached target for status, got kind {kind!r}.")


def _read_exitcode(target: dict[str, Any], log: str | None) -> int | None:
    if not log or target["kind"] not in {"tmux", "ssh-tmux"}:
        return None
    exit_path = log + ".exitcode"
    if target["kind"] == "tmux":
        path = _local_path(target["workdir"], exit_path)
        text = path.read_text(encoding="utf-8").strip() if path.exists() else ""
    else:
        quoted = shlex.quote(exit_path)
        remote = f"if test -f {quoted}; then cat {quoted}; fi"
        completed = subprocess.run(
            ["ssh", target["host"], remote],
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode:
            raise RuntimeError(
                f"Status transport failed with exit {completed.returncode}: "
                f"{completed.stderr.strip() or completed.stdout.strip()}"
            )
        text = completed.stdout.strip()
    if not text:
        return None
    try:
        return int(text)
    except ValueError as error:
        raise RuntimeError(f"Expected integer exit code in {exit_path!r}, got {text!r}.") from error


def status(
    target_name: str,
    target: dict[str, Any],
    handle: str,
    *,
    log: str | None = None,
) -> dict[str, Any]:
    exitcode = _read_exitcode(target, log)
    if exitcode is not None:
        return {
            "target": target_name,
            "kind": target["kind"],
            "handle": handle,
            "log": log,
            "state": "completed" if exitcode == 0 else "failed",
            "detail": f"exitcode={exitcode}",
        }

    completed = subprocess.run(
        build_status_command(target, handle),
        check=False,
        capture_output=True,
        text=True,
    )
    output = completed.stdout.strip()
    if completed.returncode:
        detail = completed.stderr.strip() or output
        missing = ("can't find session", "can't find window", "no server running")
        if target["kind"] in {"tmux", "ssh-tmux"} and any(
            marker in detail.lower() for marker in missing
        ):
            state = "not-running"
        else:
            raise RuntimeError(
                f"Status failed with exit {completed.returncode}: {detail}"
            )
    elif target["kind"] == "slurm":
        state = output.split("|", 1)[0].strip().lower() if output else "unknown"
    else:
        pane_dead = output.split(maxsplit=1)[0] if output else ""
        state = "finished" if pane_dead == "1" else "running"
    return {
        "target": target_name,
        "kind": target["kind"],
        "handle": handle,
        "log": log,
        "state": state,
        "detail": output,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--targets", type=Path, default=DEFAULT_TARGETS)
    subparsers = parser.add_subparsers(dest="action", required=True)

    run = subparsers.add_parser("run", help="launch an opaque command")
    run.add_argument("target")
    run.add_argument("--name", required=True)
    run.add_argument("--log")
    run.add_argument(
        "--slurm-arg",
        action="append",
        default=[],
        help="extra sbatch option; use --slurm-arg=--array=0-5",
    )
    run.add_argument("--dry-run", action="store_true")
    show = subparsers.add_parser("status", help="query a detached handle")
    show.add_argument("target")
    show.add_argument("handle")
    show.add_argument("--log", help="detached log path used to read the saved exit code")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args, remainder = _parser().parse_known_args(argv)
    targets = load_targets(args.targets)
    if args.target not in targets:
        raise ValueError(
            f"Expected one of targets {sorted(targets)}, got {args.target!r}."
        )
    target = targets[args.target]

    if args.action == "status":
        if remainder:
            raise ValueError(f"Expected no arguments after HANDLE, got {remainder!r}.")
        payload = status(args.target, target, args.handle, log=args.log)
    else:
        command = list(remainder)
        if command[:1] == ["--"]:
            command = command[1:]
        if not command:
            raise ValueError(f"Expected a command after '--', got {remainder!r}.")
        payload = launch(
            args.target,
            target,
            command,
            name=args.name,
            log=args.log,
            slurm_args=args.slurm_arg,
            dry_run=args.dry_run,
        )
    print(json.dumps(payload, indent=2))
    return int(payload.get("returncode", 0))


if __name__ == "__main__":
    sys.exit(main())
