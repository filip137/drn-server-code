"""Run the two-row exploratory transfer of the v6 diagnostic-best rho pair.

This is deliberately not a v6 ``confirmations`` stage.  The exact v6 selector
reported an unbracketed baseline optimum, so the frozen v6 contract prohibits
canonical confirmations.  This module binds that unbracketed result and the
scheme-specific v6 probes into a separate content-addressed diagnostic.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

from experiments.mnist_conv.identity import (
    canonical_json_bytes,
    code_provenance,
    normalize_code_provenance,
    sha256_file,
    sha256_json,
)
from experiments.mnist_conv.io import atomic_write_bytes, atomic_write_json, read_json
from experiments.mnist_conv.lr_artifacts import validate_stage_completion
from experiments.mnist_conv.lr_stages import (
    execute_candidate_entry,
    layerwise_learning_rate_report,
    two_rho_learning_rates,
)
from experiments.mnist_conv.lr_study import _require_current_source, load_study


SCHEMA_VERSION = "mnist-conv-lr-exploratory-transfer/v1"
COMPLETION_SCHEMA_VERSION = "mnist-conv-lr-exploratory-transfer-completion/v1"
SUMMARY_SCHEMA_VERSION = "mnist-conv-lr-exploratory-transfer-summary/v1"
SOURCE_STUDY_ID = (
    "lrstudy_c735d2beead9fcbadda89a65ffe86257da53f15e6cbfab7b4f04b14b08ff6c2f"
)
RHO_CONV = 1.0e-2
RHO_DENSE = 3.0e-2
STAGE_NAME = "transfer_candidates"
ROLE = "exploratory-transfer"
OUTPUT_FILENAMES = (
    "best_validation.pt",
    "final.pt",
    "minibatches.json",
    "parameter_diagnostics.csv",
    "run_spec.v5.json",
    "step_log.csv",
    "summary.json",
    "validation.json",
)


def _error(expected: str, provided: Any) -> ValueError:
    return ValueError(f"Expected {expected}. Provided value: {provided!r}.")


def _same_float(left: Any, right: float) -> bool:
    try:
        return math.isclose(float(left), right, rel_tol=0.0, abs_tol=0.0)
    except (TypeError, ValueError):
        return False


def _publish_once(path: Path, value: Mapping[str, Any]) -> None:
    encoded = canonical_json_bytes(dict(value))
    if path.exists():
        if path.read_bytes() != encoded:
            raise RuntimeError(
                f"Expected immutable artifact {path} to retain identical bytes. "
                "Provided value: conflicting content."
            )
        return
    atomic_write_bytes(path, encoded)


def _source_artifact(root: Path, relative: str) -> dict[str, Any]:
    path = root / relative
    if not path.is_file():
        raise FileNotFoundError(
            f"Expected source artifact to be a regular file. Provided value: {path}."
        )
    return {
        "path": relative,
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }


def _validate_source_artifacts(root: Path, artifacts: Sequence[Mapping[str, Any]]) -> None:
    for artifact in artifacts:
        path = root / str(artifact["path"])
        observed = sha256_file(path) if path.is_file() else None
        if observed != artifact["sha256"]:
            raise RuntimeError(
                "Expected exploratory-transfer source artifact to retain its frozen "
                f"SHA-256. Provided value: path={path}, expected={artifact['sha256']!r}, "
                f"observed={observed!r}."
            )


def _entry_dir(transfer_root: Path, row_id: str) -> Path:
    return transfer_root / "stages" / STAGE_NAME / "entries" / f"{row_id}--{ROLE}"


def _completion_path(transfer_root: Path, row_id: str) -> Path:
    return _entry_dir(transfer_root, row_id) / "complete.json"


def _output_records(transfer_root: Path, row_id: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for filename in OUTPUT_FILENAMES:
        path = _entry_dir(transfer_root, row_id) / filename
        if not path.is_file():
            raise RuntimeError(
                "Expected exploratory transfer entry to publish every declared output. "
                f"Provided missing value: {path}."
            )
        records.append(
            {
                "path": path.relative_to(transfer_root).as_posix(),
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
        )
    return records


def _validate_completion(
    transfer_root: Path, manifest: Mapping[str, Any], entry: Mapping[str, Any]
) -> dict[str, Any]:
    path = _completion_path(transfer_root, str(entry["row_id"]))
    completion = read_json(path)
    expected = {
        "schema_version": COMPLETION_SCHEMA_VERSION,
        "state": "complete",
        "transfer_id": manifest["transfer_id"],
        "entry_index": entry["entry_index"],
        "row_id": entry["row_id"],
        "scheme": entry["scheme"],
        "outputs": _output_records(transfer_root, str(entry["row_id"])),
    }
    if completion != expected:
        raise RuntimeError(
            "Expected exploratory transfer completion to match all immutable outputs. "
            f"Provided value: {completion!r}."
        )
    return completion


def create_manifest(
    *,
    source_study: str | Path,
    results_root: str | Path,
    provenance: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], Path]:
    source_root, spec = load_study(source_study)
    if spec.study_id != SOURCE_STUDY_ID or spec.data.get("schema_version") != "mnist-conv-lr-study/v6":
        raise _error(
            f"the completed v6 source study {SOURCE_STUDY_ID!r}",
            {"study_id": spec.study_id, "schema_version": spec.data.get("schema_version")},
        )

    for stage in ("probe", "baseline_candidates", "select_baseline"):
        validate_stage_completion(
            study_dir=source_root,
            manifest_path=source_root / "stages" / stage / "manifest.json",
        )

    selection_path = source_root / "stages/select_baseline/entries/selection/selection.json"
    selection = read_json(selection_path)
    diagnostic = selection.get("diagnostic_best_entry_id")
    candidates = [
        candidate
        for candidate in selection.get("candidates", [])
        if candidate.get("entry_id") == diagnostic
    ]
    if (
        selection.get("status") != "unbracketed"
        or selection.get("reason") != "passing_plateau_confined_to_outer_boundary"
        or len(candidates) != 1
        or not _same_float(candidates[0].get("rho_conv"), RHO_CONV)
        or not _same_float(candidates[0].get("rho_dense"), RHO_DENSE)
    ):
        raise RuntimeError(
            "Expected the exact v6 selector to identify the unbracketed diagnostic-best "
            f"pair ({RHO_CONV}, {RHO_DENSE}). Provided value: {selection!r}."
        )

    rows_by_id = {row["row_id"]: row for row in spec.rows}
    row_ids = tuple(spec.data["rho_grid"]["confirmation_row_ids"])
    if row_ids != ("conv2_ours_v4_c1", "conv2_legacy_v4_c0p25"):
        raise _error("the frozen ours/legacy Conv2 row order", row_ids)

    entries: list[dict[str, Any]] = []
    artifacts = [
        _source_artifact(source_root, "study.resolved.json"),
        _source_artifact(source_root, "stages/probe/complete.json"),
        _source_artifact(source_root, "stages/baseline_candidates/complete.json"),
        _source_artifact(source_root, "stages/select_baseline/complete.json"),
        _source_artifact(
            source_root, "stages/select_baseline/entries/selection/selection.json"
        ),
        _source_artifact(source_root, "split/indices.json"),
        _source_artifact(source_root, "initialization/conv2.pt"),
    ]
    for index, row_id in enumerate(row_ids):
        row = rows_by_id[row_id]
        relative_probe = f"stages/probe/entries/{row_id}/summary.json"
        artifacts.append(_source_artifact(source_root, relative_probe))
        probe = read_json(source_root / relative_probe)
        rates = two_rho_learning_rates(
            probe["median_units_by_weight"],
            rho_conv=RHO_CONV,
            rho_dense=RHO_DENSE,
            bias_weight_lr_groups=probe["bias_weight_lr_groups"],
        )
        report = layerwise_learning_rate_report(
            rates,
            thresholds=spec.data["artifacts"]["large_raw_lr_reporting"]["thresholds"],
        )
        entries.append(
            {
                "entry_index": index,
                "row_id": row_id,
                "scheme": row["scheme"],
                "payload": {
                    "row_id": row_id,
                    "architecture": "conv2",
                    "scheme": row["scheme"],
                    "candidate_role": ROLE,
                    "candidate_stage": "confirmation",
                    "rho_conv": RHO_CONV,
                    "rho_dense": RHO_DENSE,
                    "median_units_by_weight": probe["median_units_by_weight"],
                    "learning_rates_by_parameter": rates,
                    "learning_rates_by_weight": report["weight_learning_rates"],
                    "peak_learning_rate": report["maximum_weight_learning_rate"],
                },
            }
        )

    normalized_provenance = normalize_code_provenance(
        dict(provenance or code_provenance())
    )
    scientific = {
        "schema_version": SCHEMA_VERSION,
        "study_role": "ordinary_mnist_exploratory_transfer_diagnostic",
        "source_study_id": spec.study_id,
        "source_artifacts": sorted(artifacts, key=lambda item: item["path"]),
        "source_selection_status": "unbracketed",
        "transfer_reason": "user_requested_transfer_of_v6_diagnostic_best_pair",
        "rho_conv": RHO_CONV,
        "rho_dense": RHO_DENSE,
        "entries": entries,
        "candidate_training": spec.data["candidate_training"],
        "safety_gates": spec.data["range_test"]["gates"],
        "minimum_final_validation_accuracy": 0.9,
        "official_test_read": False,
        "v6_pair_frozen": False,
        "final_paper_training_authorized": False,
        "code_provenance": normalized_provenance,
    }
    transfer_id = "lrtransfer_" + sha256_json(scientific)
    manifest = {**scientific, "transfer_id": transfer_id}
    destination = (
        Path(results_root).expanduser().resolve()
        / "lr_transfers"
        / f"conv2-amplified-rho-transfer--{transfer_id}"
    )
    destination.mkdir(parents=True, exist_ok=True)
    manifest_path = destination / "manifest.json"
    _publish_once(manifest_path, manifest)
    return manifest, manifest_path


def load_manifest(path: str | Path) -> tuple[dict[str, Any], Path]:
    manifest_path = Path(path).expanduser().resolve()
    manifest = read_json(manifest_path)
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise _error(f"manifest schema {SCHEMA_VERSION!r}", manifest.get("schema_version"))
    scientific = dict(manifest)
    transfer_id = scientific.pop("transfer_id", None)
    expected_id = "lrtransfer_" + sha256_json(scientific)
    if transfer_id != expected_id or manifest_path.parent.name != f"conv2-amplified-rho-transfer--{expected_id}":
        raise _error(
            "a content-addressed exploratory transfer manifest",
            {"transfer_id": transfer_id, "expected_id": expected_id, "path": str(manifest_path)},
        )
    if len(manifest.get("entries", [])) != 2:
        raise _error("exactly two exploratory entries", manifest.get("entries"))
    return manifest, manifest_path


def run_entry(
    *,
    manifest_path: str | Path,
    source_study: str | Path,
    entry_index: int,
    data_root: str | Path,
    device: str,
) -> dict[str, Any]:
    manifest, path = load_manifest(manifest_path)
    transfer_root = path.parent
    source_root, spec = load_study(source_study)
    if spec.study_id != manifest["source_study_id"]:
        raise _error("source study ID to match the transfer manifest", spec.study_id)
    _validate_source_artifacts(source_root, manifest["source_artifacts"])
    _require_current_source(manifest["code_provenance"], code_provenance())
    if type(entry_index) is not int or not 0 <= entry_index < len(manifest["entries"]):
        raise _error("entry_index in [0, 1]", entry_index)
    entry = manifest["entries"][entry_index]
    completion_path = _completion_path(transfer_root, entry["row_id"])
    if completion_path.is_file():
        _validate_completion(transfer_root, manifest, entry)
        return {"status": "resumed_complete", "entry_index": entry_index, "row_id": entry["row_id"]}

    execute_candidate_entry(
        spec.data,
        source_root,
        entry["row_id"],
        ROLE,
        candidate_payload=entry["payload"],
        output_stage=STAGE_NAME,
        output_root=transfer_root,
        data_root=data_root,
        download=False,
        device=device,
    )
    completion = {
        "schema_version": COMPLETION_SCHEMA_VERSION,
        "state": "complete",
        "transfer_id": manifest["transfer_id"],
        "entry_index": entry_index,
        "row_id": entry["row_id"],
        "scheme": entry["scheme"],
        "outputs": _output_records(transfer_root, entry["row_id"]),
    }
    _publish_once(completion_path, completion)
    return {"status": "complete", "entry_index": entry_index, "row_id": entry["row_id"]}


def finalize_transfer(manifest_path: str | Path) -> dict[str, Any]:
    manifest, path = load_manifest(manifest_path)
    transfer_root = path.parent
    records: list[dict[str, Any]] = []
    for entry in manifest["entries"]:
        _validate_completion(transfer_root, manifest, entry)
        summary = read_json(_entry_dir(transfer_root, entry["row_id"]) / "summary.json")
        records.append(
            {
                "row_id": entry["row_id"],
                "scheme": entry["scheme"],
                "rho_conv": summary["rho_conv"],
                "rho_dense": summary["rho_dense"],
                "status": summary["status"],
                "admissible": summary["admissible"],
                "inadmissible_reason": summary["inadmissible_reason"],
                "final_validation_accuracy": summary["final_validation_accuracy"],
                "final_validation_loss": summary["final_validation_loss"],
                "median_projection_efficiency": summary["median_projection_efficiency"],
                "learning_rates_by_weight": summary["learning_rates_by_weight"],
                "achieved_relative_updates_by_weight": summary[
                    "observed_rho_relative_q90_complete_run_by_parameter"
                ],
                "maximum_bound_occupancy_by_parameter": summary[
                    "maximum_bound_occupancy_by_parameter"
                ],
            }
        )
    result = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "status": "complete",
        "transfer_id": manifest["transfer_id"],
        "study_role": manifest["study_role"],
        "rho_conv": RHO_CONV,
        "rho_dense": RHO_DENSE,
        "all_rows_passed": all(record["admissible"] for record in records),
        "records": records,
        "source_v6_status": "unbracketed",
        "v6_pair_frozen": False,
        "official_test_read": False,
        "final_paper_training_authorized": False,
    }
    _publish_once(transfer_root / "summary.json", result)
    fields = [
        "row_id",
        "scheme",
        "rho_conv",
        "rho_dense",
        "status",
        "admissible",
        "inadmissible_reason",
        "final_validation_accuracy",
        "final_validation_loss",
        "median_projection_efficiency",
        "learning_rates_by_weight",
        "achieved_relative_updates_by_weight",
        "maximum_bound_occupancy_by_parameter",
    ]
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for record in records:
        writer.writerow(
            {
                **record,
                "learning_rates_by_weight": json.dumps(record["learning_rates_by_weight"], sort_keys=True),
                "achieved_relative_updates_by_weight": json.dumps(record["achieved_relative_updates_by_weight"], sort_keys=True),
                "maximum_bound_occupancy_by_parameter": json.dumps(record["maximum_bound_occupancy_by_parameter"], sort_keys=True),
            }
        )
    csv_path = transfer_root / "summary.csv"
    encoded = stream.getvalue().encode()
    if csv_path.exists() and csv_path.read_bytes() != encoded:
        raise RuntimeError(f"Expected immutable summary CSV bytes. Provided value: {csv_path}.")
    if not csv_path.exists():
        atomic_write_bytes(csv_path, encoded)
    return result


def run_pack(
    *,
    manifest_path: str | Path,
    source_study: str | Path,
    data_root: str | Path,
    device: str,
    python: str,
    log_dir: str | Path,
) -> dict[str, Any]:
    manifest, path = load_manifest(manifest_path)
    logs = Path(log_dir).expanduser().resolve()
    logs.mkdir(parents=True, exist_ok=True)
    active: list[subprocess.Popen[str]] = []
    streams: list[Any] = []
    commands: list[list[str]] = []
    try:
        for entry in manifest["entries"]:
            command = [
                python,
                "-m",
                "experiments.run_mnist_conv_lr_v6_exploratory_transfer",
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
            stream = (logs / f"entry_{entry['entry_index']}_{entry['scheme']}.log").open("w")
            process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT, text=True)
            commands.append(command)
            streams.append(stream)
            active.append(process)
        executions = []
        for entry, command, process in zip(manifest["entries"], commands, active, strict=True):
            returncode = process.wait()
            executions.append(
                {
                    "entry_index": entry["entry_index"],
                    "scheme": entry["scheme"],
                    "returncode": returncode,
                    "command": command,
                }
            )
        for stream in streams:
            stream.close()
        streams = []
        active = []
        failures = [entry for entry in executions if entry["returncode"] != 0]
        if failures:
            raise RuntimeError(
                "Expected both exploratory transfer children to complete. "
                f"Provided failures: {failures!r}."
            )
        summary = finalize_transfer(path)
        return {"status": "complete", "executions": executions, "summary": summary}
    finally:
        for process in active:
            if process.poll() is None:
                process.terminate()
        for process in active:
            if process.poll() is None:
                process.wait(timeout=30)
        for stream in streams:
            stream.close()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    create = commands.add_parser("create")
    create.add_argument("--source-study", required=True)
    create.add_argument("--results-root", required=True)
    run = commands.add_parser("run-entry")
    run.add_argument("--manifest", required=True)
    run.add_argument("--source-study", required=True)
    run.add_argument("--entry-index", required=True, type=int)
    run.add_argument("--data-root", required=True)
    run.add_argument("--device", default="cuda")
    pack = commands.add_parser("run-pack")
    pack.add_argument("--manifest", required=True)
    pack.add_argument("--source-study", required=True)
    pack.add_argument("--data-root", required=True)
    pack.add_argument("--device", default="cuda")
    pack.add_argument("--python", required=True)
    pack.add_argument("--log-dir", required=True)
    finalize = commands.add_parser("finalize")
    finalize.add_argument("--manifest", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "create":
            manifest, path = create_manifest(
                source_study=args.source_study,
                results_root=args.results_root,
            )
            result: Any = {
                "status": "created",
                "transfer_id": manifest["transfer_id"],
                "manifest": str(path),
                "entries": manifest["entries"],
            }
        elif args.command == "run-entry":
            result = run_entry(
                manifest_path=args.manifest,
                source_study=args.source_study,
                entry_index=args.entry_index,
                data_root=args.data_root,
                device=args.device,
            )
        elif args.command == "run-pack":
            result = run_pack(
                manifest_path=args.manifest,
                source_study=args.source_study,
                data_root=args.data_root,
                device=args.device,
                python=args.python,
                log_dir=args.log_dir,
            )
        else:
            result = finalize_transfer(args.manifest)
    except (OSError, RuntimeError, ValueError) as exc:
        print(str(exc), file=sys.stderr, flush=True)
        return 2
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
