"""Execute one immutable v6 pack as sequential waves on one allocated GPU."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any, Callable

from .lr_v6_packing import validate_v6_pack_manifest_files


def _error(expected: str, provided: Any) -> ValueError:
    return ValueError(f"Expected {expected}. Provided value: {provided!r}.")


def entry_command(
    *,
    python: str,
    stage: str,
    study_dir: Path,
    manifest_path: Path,
    entry_index: int,
    data_root: Path,
    device: str,
) -> list[str]:
    return [
        python,
        "-m",
        "experiments.mnist_conv",
        "lr-study",
        "--stage",
        stage,
        "--study",
        str(study_dir),
        "--manifest",
        str(manifest_path),
        "--entry-index",
        str(entry_index),
        "--data-root",
        str(data_root),
        "--device",
        device,
    ]


def execute_packed_waves(
    *,
    study_dir: str | Path,
    manifest_path: str | Path,
    pack_manifest_path: str | Path,
    benchmark_path: str | Path | None,
    stage: str,
    data_root: str | Path,
    device: str,
    python: str,
    log_dir: str | Path,
    process_factory: Callable[..., Any] = subprocess.Popen,
) -> dict[str, Any]:
    """Launch every wave concurrently, waiting before starting the next wave."""

    study = Path(study_dir).expanduser().resolve()
    manifest = Path(manifest_path).expanduser().resolve()
    pack_path = Path(pack_manifest_path).expanduser().resolve()
    benchmark = (
        None if benchmark_path is None else Path(benchmark_path).expanduser().resolve()
    )
    logs = Path(log_dir).expanduser().resolve()
    logs.mkdir(parents=True, exist_ok=True)
    pack = validate_v6_pack_manifest_files(
        manifest, pack_path, benchmark_path=benchmark
    )
    if pack["stage_name"] != stage:
        raise _error("requested stage to match the pack manifest", stage)
    if device not in {"cuda", "cuda:0"}:
        raise _error("device to identify the one allocated CUDA GPU", device)

    executions: list[dict[str, Any]] = []
    active: list[Any] = []
    streams: list[Any] = []
    try:
        for wave in pack["waves"]:
            active = []
            streams = []
            commands: list[list[str]] = []
            for entry_index in wave["entry_indices"]:
                command = entry_command(
                    python=python,
                    stage=stage,
                    study_dir=study,
                    manifest_path=manifest,
                    entry_index=entry_index,
                    data_root=Path(data_root).expanduser().resolve(),
                    device=device,
                )
                stream = (
                    logs / f"wave_{wave['wave_index']}_entry_{entry_index}.log"
                ).open("w")
                process = process_factory(
                    command,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
                commands.append(command)
                streams.append(stream)
                active.append(process)
            failures = []
            for entry_index, command, process in zip(
                wave["entry_indices"], commands, active, strict=True
            ):
                returncode = process.wait()
                executions.append(
                    {
                        "wave_index": wave["wave_index"],
                        "entry_index": entry_index,
                        "returncode": returncode,
                        "command": command,
                    }
                )
                if returncode != 0:
                    failures.append(
                        {"entry_index": entry_index, "returncode": returncode}
                    )
            for stream in streams:
                stream.close()
            streams = []
            active = []
            if failures:
                raise RuntimeError(
                    "Expected every entry in a v6 wave to complete before starting "
                    f"the next wave. Provided wave={wave['wave_index']}, "
                    f"failures={failures!r}."
                )
    finally:
        for process in active:
            if process.poll() is None:
                process.terminate()
        for process in active:
            if process.poll() is None:
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
        for stream in streams:
            stream.close()
    return {
        "status": "complete",
        "study_id": pack["study_id"],
        "stage": stage,
        "wave_count": len(pack["waves"]),
        "entry_count": len(executions),
        "executions": executions,
    }


__all__ = ["entry_command", "execute_packed_waves"]
