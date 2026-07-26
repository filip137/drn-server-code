"""Run one disjoint scheme slice of the immutable Conv1 rho sweep.

This is an execution-only wrapper. It never changes the scientific manifest
and never finalizes the sweep; finalization happens only after both scheme
workers have completed successfully.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

from experiments.run_mnist_conv_lr_conv1_scheme_rho_sweep import load_manifest


SCHEMES = ("ours", "legacy")


def _error(expected: str, provided: Any) -> ValueError:
    return ValueError(f"Expected {expected}. Provided value: {provided!r}.")


def scheme_entries(
    manifest: Mapping[str, Any], scheme: str
) -> list[Mapping[str, Any]]:
    if scheme not in SCHEMES:
        raise _error(f"scheme to be one of {SCHEMES!r}", scheme)
    entries = [
        entry
        for entry in manifest["entries"]
        if entry["scheme"] == scheme
        and entry["mode"] == "new_five_epoch_training"
    ]
    if len(entries) != 9:
        raise _error(
            f"exactly nine new five-epoch entries for scheme {scheme!r}",
            [entry.get("entry_index") for entry in entries],
        )
    return entries


def run_scheme_pack(
    *,
    manifest_path: str | Path,
    source_study: str | Path,
    data_root: str | Path,
    device: str,
    python: str,
    log_dir: str | Path,
    scheme: str,
    concurrency: int,
) -> dict[str, Any]:
    manifest, path = load_manifest(manifest_path)
    if type(concurrency) is not int or concurrency < 1:
        raise _error("a positive integer concurrency", concurrency)
    entries = scheme_entries(manifest, scheme)
    logs = Path(log_dir).expanduser().resolve()
    logs.mkdir(parents=True, exist_ok=True)
    executions: list[dict[str, Any]] = []

    for wave_index, start in enumerate(range(0, len(entries), concurrency)):
        wave = entries[start : start + concurrency]
        active: list[
            tuple[Mapping[str, Any], list[str], subprocess.Popen[str], Any]
        ] = []
        try:
            for entry in wave:
                command = [
                    python,
                    "-m",
                    "experiments.run_mnist_conv_lr_conv1_scheme_rho_sweep",
                    "run-entry",
                    "--manifest",
                    str(path),
                    "--source-study",
                    str(Path(source_study).expanduser().resolve()),
                    "--entry-index",
                    str(entry["entry_index"]),
                    "--data-root",
                    str(Path(data_root).expanduser().resolve()),
                    "--device",
                    device,
                ]
                stream = (
                    logs
                    / f"wave_{wave_index:02d}_entry_{entry['entry_index']:02d}_{scheme}.log"
                ).open("w")
                process = subprocess.Popen(
                    command,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
                active.append((entry, command, process, stream))

            for entry, command, process, stream in active:
                returncode = process.wait()
                stream.close()
                executions.append(
                    {
                        "wave_index": wave_index,
                        "entry_index": entry["entry_index"],
                        "scheme": scheme,
                        "returncode": returncode,
                        "command": command,
                    }
                )
            failures = [
                execution
                for execution in executions
                if execution["wave_index"] == wave_index
                and execution["returncode"] != 0
            ]
            if failures:
                raise RuntimeError(
                    "Expected every Conv1 scheme-slice child in the wave to "
                    f"complete. Provided failures: {failures!r}."
                )
        finally:
            for _entry, _command, process, stream in active:
                if process.poll() is None:
                    process.terminate()
                if not stream.closed:
                    stream.close()

    return {
        "status": "complete",
        "scheme": scheme,
        "entry_indices": [entry["entry_index"] for entry in entries],
        "executions": executions,
        "finalized": False,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--source-study", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--python", required=True)
    parser.add_argument("--log-dir", required=True)
    parser.add_argument("--scheme", choices=SCHEMES, required=True)
    parser.add_argument("--concurrency", type=int, default=9)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = run_scheme_pack(
            manifest_path=args.manifest,
            source_study=args.source_study,
            data_root=args.data_root,
            device=args.device,
            python=args.python,
            log_dir=args.log_dir,
            scheme=args.scheme,
            concurrency=args.concurrency,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        print(str(exc), file=sys.stderr, flush=True)
        return 2
    print(json.dumps(result, sort_keys=True, separators=(",", ":")), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
