#!/usr/bin/env python3
"""Host-neutral validation and production adapter for one Conv job.

The same public command serves the bounded one-batch functional preflight and
the approved production horizon.  Environment details are capability
provenance, not an exact-version admission gate.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from experiments.mnist_conv.identity import sha256_file
from experiments.mnist_conv.io import atomic_create_json, read_json
from experiments.mnist_conv.perfectdiode_functional_smoke import (
    capture_environment,
    run_perfectdiode_one_batch_smoke,
)
from experiments.mnist_conv.perfectdiode_successor_confirmation import (
    execute_successor_stage_entry,
)


JOB_COMPLETION_SCHEMA = "experiment-job-completion/v1"
RUN_CLASSES = {"validation", "production"}


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            f"Expected {label} to be non-empty text. Provided value: {value!r}."
        )
    return value.strip()


def _entry(manifest: Any, entry_id: str) -> dict[str, Any]:
    if not isinstance(manifest, Mapping) or not isinstance(
        manifest.get("entries"), list
    ):
        raise ValueError(
            "Expected bundle manifest.entries to be a list. "
            f"Provided value: {manifest!r}."
        )
    matches = [
        item
        for item in manifest["entries"]
        if isinstance(item, Mapping) and item.get("entry_id") == entry_id
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected entry_id {entry_id!r} to identify exactly one manifest "
            f"job. Provided value: {len(matches)} matches."
        )
    return copy.deepcopy(dict(matches[0]))


def _runtime_payload(entry: Mapping[str, Any]) -> dict[str, Any]:
    payload = copy.deepcopy(dict(entry))
    probe_result = payload.get("probe_result")
    if probe_result is not None:
        payload["probe_result_path"] = probe_result
    return payload


def validate_job_completion(
    value: Mapping[str, Any],
    *,
    expected_attempt_id: str | None = None,
    expected_job_id: str | None = None,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(
            "Expected a job completion receipt mapping. "
            f"Provided value: {type(value).__name__}."
        )
    if value.get("schema_version") != JOB_COMPLETION_SCHEMA:
        raise ValueError(
            f"Expected schema_version {JOB_COMPLETION_SCHEMA!r}. "
            f"Provided value: {value.get('schema_version')!r}."
        )
    if value.get("status") != "complete":
        raise ValueError(
            "Expected job completion status='complete'. "
            f"Provided value: {value.get('status')!r}."
        )
    if value.get("official_test_read") is not False:
        raise ValueError(
            "Expected official_test_read=false. "
            f"Provided value: {value.get('official_test_read')!r}."
        )
    if expected_attempt_id is not None and (
        value.get("attempt_id") != expected_attempt_id
    ):
        raise ValueError(
            f"Expected attempt_id {expected_attempt_id!r}. "
            f"Provided value: {value.get('attempt_id')!r}."
        )
    if expected_job_id is not None and value.get("job_id") != expected_job_id:
        raise ValueError(
            f"Expected job_id {expected_job_id!r}. "
            f"Provided value: {value.get('job_id')!r}."
        )
    if value.get("outcome") not in {"completed", "negative_evidence"}:
        raise ValueError(
            "Expected outcome to be 'completed' or 'negative_evidence'. "
            f"Provided value: {value.get('outcome')!r}."
        )
    result = value.get("result")
    if not isinstance(result, Mapping):
        raise ValueError(
            "Expected result to be a mapping. "
            f"Provided value: {result!r}."
        )
    if result.get("official_test_read") is not False:
        raise ValueError(
            "Expected result.official_test_read=false. "
            f"Provided value: {result.get('official_test_read')!r}."
        )
    artifacts = value.get("artifacts")
    if (
        not isinstance(artifacts, list)
        or not artifacts
        or any(
            not isinstance(item, Mapping)
            or not isinstance(item.get("path"), str)
            or not isinstance(item.get("sha256"), str)
            or len(item["sha256"]) != 64
            for item in artifacts
        )
    ):
        raise ValueError(
            "Expected artifacts to be a non-empty list of path/SHA-256 "
            f"records. Provided value: {artifacts!r}."
        )
    return dict(value)


def _production_artifacts(output: Path) -> list[dict[str, Any]]:
    required = (
        "result.json",
        "run_spec.json",
        "step_log.csv",
        "validation.json",
        "best_validation.pt",
        "final.pt",
    )
    records = []
    for name in required:
        path = output / name
        if not path.is_file() or path.is_symlink():
            raise ValueError(
                "Expected production output to contain a regular artifact "
                f"{name!r}. Provided value: {path}."
            )
        records.append(
            {
                "path": name,
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
        )
    return records


def run_production_job(
    study: Mapping[str, Any],
    bundle_dir: str | Path,
    entry: Mapping[str, Any],
    *,
    attempt_id: str,
    target: str,
    data_root: str | Path,
    device: str,
    output_dir: str | Path,
    download: bool = False,
) -> dict[str, Any]:
    """Run one fixed job and publish one compact completion receipt."""

    attempt = _text(attempt_id, "attempt_id")
    target_name = _text(target, "target")
    job_id = _text(entry.get("entry_id"), "entry.entry_id")
    output = Path(output_dir).expanduser().resolve()
    completion_path = output / "job_completion.json"
    if completion_path.exists() or completion_path.is_symlink():
        existing = read_json(completion_path)
        return validate_job_completion(
            existing,
            expected_attempt_id=attempt,
            expected_job_id=job_id,
        )
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(
            "Expected a fresh job output directory or a valid existing "
            f"completion receipt. Provided value: {output}."
        )

    payload = _runtime_payload(entry)
    result = execute_successor_stage_entry(
        study,
        bundle_dir,
        "successor_confirm",
        payload,
        data_root=data_root,
        device=device,
        download=download,
        output_dir=output,
    )
    status = result.get("status")
    if status not in {"complete", "safety_failure"}:
        raise ValueError(
            "Expected production result status to be 'complete' or the "
            "predeclared scientific outcome 'safety_failure'. "
            f"Provided value: {status!r}."
        )
    if result.get("official_test_read") is not False:
        raise ValueError(
            "Expected production result official_test_read=false. "
            f"Provided value: {result.get('official_test_read')!r}."
        )
    artifacts = _production_artifacts(output)
    receipt = {
        "schema_version": JOB_COMPLETION_SCHEMA,
        "status": "complete",
        "outcome": (
            "completed" if status == "complete" else "negative_evidence"
        ),
        "attempt_id": attempt,
        "target": target_name,
        "job_id": job_id,
        "official_test_read": False,
        "scientific_binding": {
            "entry_id": job_id,
            "row_id": entry.get("row_id"),
            "architecture": entry.get("architecture"),
            "scheme": entry.get("scheme"),
            "optimizer": entry.get("optimizer"),
            "rho_conv": entry.get("rho_conv"),
            "rho_dense": entry.get("rho_dense"),
            "raw_learning_rates_by_parameter": copy.deepcopy(
                entry.get("raw_learning_rates_by_parameter")
            ),
            "epochs": entry.get("epochs"),
            "expected_steps": entry.get("expected_steps"),
        },
        "environment": capture_environment(device),
        "adapter_provenance": {
            "adapter_path": Path(__file__).name,
            "adapter_sha256": sha256_file(Path(__file__).resolve()),
        },
        "result": copy.deepcopy(dict(result)),
        "artifacts": artifacts,
    }
    validate_job_completion(
        receipt,
        expected_attempt_id=attempt,
        expected_job_id=job_id,
    )
    atomic_create_json(completion_path, receipt, canonical=True)
    return receipt


def run_job(
    *,
    run_class: str,
    study_path: str | Path,
    bundle_dir: str | Path,
    entry_id: str,
    attempt_id: str,
    target: str,
    data_root: str | Path,
    device: str,
    output_dir: str | Path,
    download: bool = False,
) -> dict[str, Any]:
    if run_class not in RUN_CLASSES:
        raise ValueError(
            f"Expected run_class to be one of {sorted(RUN_CLASSES)!r}. "
            f"Provided value: {run_class!r}."
        )
    bundle = Path(bundle_dir).expanduser().resolve()
    study = read_json(study_path)
    entry = _entry(read_json(bundle / "manifest.json"), entry_id)
    if run_class == "validation":
        return run_perfectdiode_one_batch_smoke(
            study,
            bundle,
            _runtime_payload(entry),
            attempt_id=attempt_id,
            target=target,
            data_root=data_root,
            device=device,
            output_dir=output_dir,
            download=download,
        )
    return run_production_job(
        study,
        bundle,
        entry,
        attempt_id=attempt_id,
        target=target,
        data_root=data_root,
        device=device,
        output_dir=output_dir,
        download=download,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run one perfect-diode Conv job through the common validation or "
            "production adapter."
        )
    )
    parser.add_argument(
        "--run-class", required=True, choices=sorted(RUN_CLASSES)
    )
    parser.add_argument("--study", required=True)
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--entry-id", required=True)
    parser.add_argument("--attempt-id", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--download", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = run_job(
        run_class=args.run_class,
        study_path=args.study,
        bundle_dir=args.bundle,
        entry_id=args.entry_id,
        attempt_id=args.attempt_id,
        target=args.target,
        data_root=args.data_root,
        device=args.device,
        output_dir=args.output_dir,
        download=args.download,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
