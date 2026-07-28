#!/usr/bin/env python3
"""Execute one immutable Jean Zay validation launch specification."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from experiments.mnist_conv.io import read_json
from experiments.run_mnist_conv_perfectdiode_job import run_job


LAUNCH_SPEC_SCHEMA = "jean-zay-validation-launch/v1"


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            f"Expected {label} to be non-empty text. Provided value: {value!r}."
        )
    return value.strip()


def _absolute(value: Any, label: str) -> Path:
    path = Path(_text(value, label))
    if not path.is_absolute() or ".." in path.parts:
        raise ValueError(
            f"Expected {label} to be an absolute normalized path. "
            f"Provided value: {value!r}."
        )
    return path


def validate_launch_spec(value: Mapping[str, Any]) -> dict[str, Any]:
    spec = dict(value)
    if spec.get("schema_version") != LAUNCH_SPEC_SCHEMA:
        raise ValueError(
            f"Expected schema_version {LAUNCH_SPEC_SCHEMA!r}. "
            f"Provided value: {spec.get('schema_version')!r}."
        )
    if spec.get("run_class") != "validation":
        raise ValueError(
            "Expected run_class='validation'. "
            f"Provided value: {spec.get('run_class')!r}."
        )
    if spec.get("official_test_read") is not False:
        raise ValueError(
            "Expected official_test_read=false. "
            f"Provided value: {spec.get('official_test_read')!r}."
        )
    if spec.get("maximum_jobs") != 1:
        raise ValueError(
            "Expected maximum_jobs=1. "
            f"Provided value: {spec.get('maximum_jobs')!r}."
        )
    for key in ("experiment_id", "attempt_id", "entry_id", "target"):
        _text(spec.get(key), key)
    for key in (
        "study_path",
        "bundle_dir",
        "data_root",
        "output_dir",
    ):
        _absolute(spec.get(key), key)
    if _text(spec.get("device"), "device") != "cuda:0":
        raise ValueError(
            "Expected device='cuda:0'. "
            f"Provided value: {spec.get('device')!r}."
        )
    return spec


def execute_spec(path: str | Path) -> dict[str, Any]:
    spec_path = Path(path).expanduser().resolve()
    spec = validate_launch_spec(read_json(spec_path))
    output_dir = _absolute(spec["output_dir"], "output_dir")
    if output_dir.exists():
        raise FileExistsError(
            "Expected a fresh attempt-specific validation output directory. "
            f"Provided value: {output_dir}."
        )
    return run_job(
        run_class="validation",
        study_path=spec["study_path"],
        bundle_dir=spec["bundle_dir"],
        entry_id=spec["entry_id"],
        attempt_id=spec["attempt_id"],
        target=spec["target"],
        data_root=spec["data_root"],
        device=spec["device"],
        output_dir=output_dir,
        download=False,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Execute one immutable Jean Zay one-batch validation."
    )
    parser.add_argument("--launch-spec", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = execute_spec(args.launch_spec)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
