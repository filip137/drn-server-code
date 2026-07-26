"""Run a content-addressed Conv2 amplified-scheme two-rho diagnostic.

The sweep binds exact completed v5 candidates and the failed v6 exploratory
transfer, then trains only the remaining grid cells. Ours and legacy are
selected independently; the unamplified Conv2 baseline is not rerun.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
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
from experiments.mnist_conv.io import atomic_write_bytes, read_json
from experiments.mnist_conv.lr_artifacts import validate_stage_completion
from experiments.mnist_conv.lr_protocol import select_v6_conv2_baseline
from experiments.mnist_conv.lr_stages import (
    execute_candidate_entry,
    layerwise_learning_rate_report,
    two_rho_learning_rates,
)
from experiments.mnist_conv.lr_study import _require_current_source, load_study
from experiments.run_mnist_conv_lr_v6_exploratory_transfer import (
    load_manifest as load_transfer_manifest,
)


SCHEMA_VERSION = "mnist-conv-lr-conv2-scheme-rho-sweep/v1"
ENTRY_COMPLETION_SCHEMA = "mnist-conv-lr-conv2-scheme-rho-entry-completion/v1"
SUMMARY_SCHEMA = "mnist-conv-lr-conv2-scheme-rho-summary/v1"
FINAL_COMPLETION_SCHEMA = "mnist-conv-lr-conv2-scheme-rho-completion/v1"
V6_STUDY_ID = "lrstudy_c735d2beead9fcbadda89a65ffe86257da53f15e6cbfab7b4f04b14b08ff6c2f"
V5_STUDY_ID = "lrstudy_5da3a7452c00d42325bfe930d80f037791125d3ce7b5ddd79be13ff8b96b49e2"
TRANSFER_ID = "lrtransfer_b75712190eb0ae6ea818fd6b54f335ee22171d34aeed2c9b82de9bbd454d7d14"
SCHEME_ROWS = (
    ("ours", "conv2_ours_v4_c1"),
    ("legacy", "conv2_legacy_v4_c0p25"),
)
RHO_CONV_GRID = (5.0e-4, 1.5e-3, 3.0e-3, 1.0e-2)
RHO_DENSE_GRID = (3.0e-3, 1.0e-2, 3.0e-2)
STAGE_NAME = "candidates"
ROLE_PREFIX = "scheme-rho"
OUTPUT_FILENAMES = (
    "best_validation.pt",
    "final.pt",
    "minibatches.json",
    "parameter_diagnostics.csv",
    "run_spec.v6.json",
    "step_log.csv",
    "summary.json",
    "validation.json",
)
V5_REUSE_BY_PAIR = {
    (5.0e-4, 1.0e-2): "historical_profile--center",
    (1.5e-3, 3.0e-2): "historical_profile--upper",
    (3.0e-3, 3.0e-3): "strict_equal--upper",
}
TRANSFER_REUSE_PAIR = (1.0e-2, 3.0e-2)


def _error(expected: str, provided: Any) -> ValueError:
    return ValueError(f"Expected {expected}. Provided value: {provided!r}.")


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


def _artifact(root: Path, relative: str) -> dict[str, Any]:
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


def _validate_artifacts(
    root: Path, artifacts: Sequence[Mapping[str, Any]], label: str
) -> None:
    for artifact in artifacts:
        path = root / str(artifact["path"])
        observed = sha256_file(path) if path.is_file() else None
        if observed != artifact["sha256"]:
            raise RuntimeError(
                f"Expected {label} artifact to retain its frozen SHA-256. "
                f"Provided value: path={path}, expected={artifact['sha256']!r}, "
                f"observed={observed!r}."
            )


def _same_pair(left: tuple[float, float], right: tuple[float, float]) -> bool:
    return all(
        math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-15)
        for a, b in zip(left, right, strict=True)
    )


def _v5_pair(summary: Mapping[str, Any]) -> tuple[float, float]:
    alpha = float(summary["alpha_arch"])
    multipliers = summary["target_multipliers_by_weight"]
    return (
        alpha * float(multipliers["ConvWeight_0"]),
        alpha * float(multipliers["DenseWeight_0"]),
    )


def _role(conv_index: int, dense_index: int) -> str:
    return f"{ROLE_PREFIX}--c{conv_index:02d}-d{dense_index:02d}"


def _entry_dir(root: Path, entry: Mapping[str, Any]) -> Path:
    return root / "stages" / STAGE_NAME / "entries" / f"{entry['row_id']}--{entry['role']}"


def _completion_path(root: Path, entry: Mapping[str, Any]) -> Path:
    return _entry_dir(root, entry) / "complete.json"


def _output_records(root: Path, entry: Mapping[str, Any]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for filename in OUTPUT_FILENAMES:
        path = _entry_dir(root, entry) / filename
        if not path.is_file():
            raise RuntimeError(
                "Expected every declared Conv2 rho candidate output. "
                f"Provided missing value: {path}."
            )
        records.append(
            {
                "path": path.relative_to(root).as_posix(),
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
        )
    return records


def _validate_completion(
    root: Path, manifest: Mapping[str, Any], entry: Mapping[str, Any]
) -> dict[str, Any]:
    completion = read_json(_completion_path(root, entry))
    expected = {
        "schema_version": ENTRY_COMPLETION_SCHEMA,
        "state": "complete",
        "sweep_id": manifest["sweep_id"],
        "entry_index": entry["entry_index"],
        "row_id": entry["row_id"],
        "scheme": entry["scheme"],
        "rho_conv": entry["rho_conv"],
        "rho_dense": entry["rho_dense"],
        "outputs": _output_records(root, entry),
    }
    if completion != expected:
        raise RuntimeError(
            "Expected Conv2 rho candidate completion to match all immutable "
            f"outputs. Provided value: {completion!r}."
        )
    return completion


def _source_summary_path(
    *,
    mode: str,
    row_id: str,
    source_entry_id: str,
) -> str:
    if mode == "reuse_v5_candidate":
        return f"stages/candidates/entries/{source_entry_id}/summary.json"
    if mode == "reuse_transfer_candidate":
        return (
            "stages/transfer_candidates/entries/"
            f"{row_id}--exploratory-transfer/summary.json"
        )
    raise _error("a reusable source mode", mode)


def create_manifest(
    *,
    source_study: str | Path,
    v5_source_study: str | Path,
    transfer_root: str | Path,
    results_root: str | Path,
    provenance: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], Path]:
    v6_root, v6_spec = load_study(source_study)
    v5_root, v5_spec = load_study(v5_source_study)
    transfer_dir = Path(transfer_root).expanduser().resolve()
    transfer, transfer_manifest_path = load_transfer_manifest(
        transfer_dir / "manifest.json"
    )
    if (
        v6_spec.study_id != V6_STUDY_ID
        or v6_spec.data.get("schema_version") != "mnist-conv-lr-study/v6"
    ):
        raise _error(
            f"the completed v6 source study {V6_STUDY_ID!r}",
            {
                "study_id": v6_spec.study_id,
                "schema_version": v6_spec.data.get("schema_version"),
            },
        )
    if (
        v5_spec.study_id != V5_STUDY_ID
        or v5_spec.data.get("schema_version") != "mnist-conv-lr-study/v5"
    ):
        raise _error(
            f"the completed v5 source study {V5_STUDY_ID!r}",
            {
                "study_id": v5_spec.study_id,
                "schema_version": v5_spec.data.get("schema_version"),
            },
        )
    if transfer.get("transfer_id") != TRANSFER_ID:
        raise _error(f"the completed transfer {TRANSFER_ID!r}", transfer.get("transfer_id"))

    for stage in ("probe", "baseline_candidates", "select_baseline"):
        validate_stage_completion(
            study_dir=v6_root,
            manifest_path=v6_root / "stages" / stage / "manifest.json",
        )
    for stage in ("audit", "probe", "candidates", "select"):
        validate_stage_completion(
            study_dir=v5_root,
            manifest_path=v5_root / "stages" / stage / "manifest.json",
        )

    transfer_summary = read_json(transfer_dir / "summary.json")
    if (
        transfer_summary.get("all_rows_passed") is not False
        or not _same_pair(
            (float(transfer_summary["rho_conv"]), float(transfer_summary["rho_dense"])),
            TRANSFER_REUSE_PAIR,
        )
    ):
        raise RuntimeError(
            "Expected the exploratory transfer to bind the failed upper rho "
            f"sentinel. Provided value: {transfer_summary!r}."
        )

    v6_artifacts = [
        _artifact(v6_root, "study.resolved.json"),
        _artifact(v6_root, "stages/probe/complete.json"),
        _artifact(v6_root, "stages/baseline_candidates/complete.json"),
        _artifact(v6_root, "stages/select_baseline/complete.json"),
        _artifact(v6_root, "stages/select_baseline/entries/selection/selection.json"),
        _artifact(v6_root, "split/indices.json"),
        _artifact(v6_root, "initialization/conv2.pt"),
        _artifact(v6_root, "initialization/conv2.json"),
    ]
    v5_artifacts = [
        _artifact(v5_root, "study.resolved.json"),
        _artifact(v5_root, "stages/candidates/complete.json"),
    ]
    transfer_artifacts = [
        _artifact(transfer_dir, transfer_manifest_path.name),
        _artifact(transfer_dir, "summary.json"),
    ]

    rows_by_id = {row["row_id"]: row for row in v6_spec.rows}
    entries: list[dict[str, Any]] = []
    for scheme, row_id in SCHEME_ROWS:
        row = rows_by_id[row_id]
        probe_relative = f"stages/probe/entries/{row_id}/summary.json"
        v6_artifacts.append(_artifact(v6_root, probe_relative))
        probe = read_json(v6_root / probe_relative)
        medians = {
            name: float(value)
            for name, value in probe["median_units_by_weight"].items()
        }
        groups = probe["bias_weight_lr_groups"]
        for conv_index, rho_conv in enumerate(RHO_CONV_GRID):
            for dense_index, rho_dense in enumerate(RHO_DENSE_GRID):
                pair = (rho_conv, rho_dense)
                role = _role(conv_index, dense_index)
                rates = two_rho_learning_rates(
                    medians,
                    rho_conv=rho_conv,
                    rho_dense=rho_dense,
                    bias_weight_lr_groups=groups,
                )
                report = layerwise_learning_rate_report(
                    rates,
                    thresholds=v6_spec.data["artifacts"]["large_raw_lr_reporting"][
                        "thresholds"
                    ],
                )
                v5_role = V5_REUSE_BY_PAIR.get(pair)
                if v5_role is not None:
                    mode = "reuse_v5_candidate"
                    source_entry_id = f"{row_id}--{v5_role}"
                    source_relative = _source_summary_path(
                        mode=mode,
                        row_id=row_id,
                        source_entry_id=source_entry_id,
                    )
                    source_summary = read_json(v5_root / source_relative)
                    if not _same_pair(_v5_pair(source_summary), pair):
                        raise RuntimeError(
                            "Expected reused v5 target pair to match the Conv2 "
                            f"grid. Provided value: {source_entry_id}."
                        )
                    v5_artifacts.append(_artifact(v5_root, source_relative))
                elif _same_pair(pair, TRANSFER_REUSE_PAIR):
                    mode = "reuse_transfer_candidate"
                    source_entry_id = f"{row_id}--exploratory-transfer"
                    source_relative = _source_summary_path(
                        mode=mode,
                        row_id=row_id,
                        source_entry_id=source_entry_id,
                    )
                    source_summary = read_json(transfer_dir / source_relative)
                    if not _same_pair(
                        (
                            float(source_summary["rho_conv"]),
                            float(source_summary["rho_dense"]),
                        ),
                        pair,
                    ):
                        raise RuntimeError(
                            "Expected reused transfer target pair to match the "
                            f"Conv2 grid. Provided value: {source_entry_id}."
                        )
                    transfer_artifacts.append(_artifact(transfer_dir, source_relative))
                    transfer_artifacts.append(
                        _artifact(
                            transfer_dir,
                            "stages/transfer_candidates/entries/"
                            f"{source_entry_id}/complete.json",
                        )
                    )
                else:
                    mode = "new_five_epoch_training"
                    source_entry_id = None
                    source_summary = None

                if source_summary is not None:
                    for name, rate in report["weight_learning_rates"].items():
                        observed = float(source_summary["learning_rates_by_weight"][name])
                        if not math.isclose(
                            observed, float(rate), rel_tol=1e-12, abs_tol=1e-15
                        ):
                            raise RuntimeError(
                                "Expected reused raw LR to match direct rho / m. "
                                f"Provided value: entry={source_entry_id}, "
                                f"parameter={name}, expected={rate}, observed={observed}."
                            )

                entries.append(
                    {
                        "entry_index": len(entries),
                        "row_id": row_id,
                        "scheme": scheme,
                        "role": role,
                        "rho_conv": rho_conv,
                        "rho_dense": rho_dense,
                        "mode": mode,
                        "source_entry_id": source_entry_id,
                        "payload": {
                            "candidate_coordinate": "conv2_scheme_direct_two_rho",
                            "candidate_stage": "scheme_grid",
                            "candidate_role": role,
                            "row_id": row_id,
                            "architecture": "conv2",
                            "scheme": scheme,
                            "rho_conv": rho_conv,
                            "rho_dense": rho_dense,
                            "median_units_by_weight": medians,
                            "learning_rates_by_parameter": rates,
                            "learning_rates_by_weight": report["weight_learning_rates"],
                            "peak_learning_rate": report["maximum_weight_learning_rate"],
                        },
                    }
                )

    mode_counts = {
        mode: sum(entry["mode"] == mode for entry in entries)
        for mode in (
            "new_five_epoch_training",
            "reuse_v5_candidate",
            "reuse_transfer_candidate",
        )
    }
    if len(entries) != 24 or mode_counts != {
        "new_five_epoch_training": 16,
        "reuse_v5_candidate": 6,
        "reuse_transfer_candidate": 2,
    }:
        raise RuntimeError(
            "Expected a 24-cell Conv2 scheme grid with 16 new, six v5, and "
            f"two transfer entries. Provided value: total={len(entries)}, "
            f"counts={mode_counts!r}."
        )

    normalized_provenance = normalize_code_provenance(
        dict(provenance or code_provenance())
    )
    scientific = {
        "schema_version": SCHEMA_VERSION,
        "study_role": "ordinary_mnist_conv2_scheme_specific_rho_diagnostic",
        "source_v6_study_id": v6_spec.study_id,
        "source_v5_study_id": v5_spec.study_id,
        "source_transfer_id": transfer["transfer_id"],
        "source_v6_artifacts": sorted(v6_artifacts, key=lambda item: item["path"]),
        "source_v5_artifacts": sorted(v5_artifacts, key=lambda item: item["path"]),
        "source_transfer_artifacts": sorted(
            transfer_artifacts, key=lambda item: item["path"]
        ),
        "row_order": [row_id for _scheme, row_id in SCHEME_ROWS],
        "rho_conv_grid": list(RHO_CONV_GRID),
        "rho_dense_grid": list(RHO_DENSE_GRID),
        "entries": entries,
        "mode_counts": mode_counts,
        "candidate_training": v6_spec.data["candidate_training"],
        "safety_gates": v6_spec.data["range_test"]["gates"],
        "minimum_final_validation_accuracy": 0.9,
        "plateau_relative_tolerance": 0.02,
        "batch_size": 16,
        "epochs": 5,
        "steps_per_epoch": 3438,
        "total_steps": 17190,
        "official_test_read": False,
        "medium_affine_handoff_replaced": False,
        "final_paper_training_authorized": False,
        "code_provenance": normalized_provenance,
    }
    sweep_id = "lrsweep_" + sha256_json(scientific)
    manifest = {**scientific, "sweep_id": sweep_id}
    destination = (
        Path(results_root).expanduser().resolve()
        / "lr_sweeps"
        / f"conv2-amplified-scheme-rho--{sweep_id}"
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
    sweep_id = scientific.pop("sweep_id", None)
    expected_id = "lrsweep_" + sha256_json(scientific)
    if (
        sweep_id != expected_id
        or manifest_path.parent.name != f"conv2-amplified-scheme-rho--{expected_id}"
    ):
        raise _error(
            "a content-addressed Conv2 scheme-rho manifest",
            {"sweep_id": sweep_id, "expected_id": expected_id, "path": str(manifest_path)},
        )
    if len(manifest.get("entries", [])) != 24:
        raise _error("exactly 24 scheme-rho entries", manifest.get("entries"))
    return manifest, manifest_path


def _validate_sources(
    manifest: Mapping[str, Any],
    *,
    source_study: str | Path,
    v5_source_study: str | Path,
    transfer_root: str | Path,
) -> tuple[Path, Mapping[str, Any], Path, Path]:
    v6_root, v6_spec = load_study(source_study)
    v5_root, v5_spec = load_study(v5_source_study)
    transfer_dir = Path(transfer_root).expanduser().resolve()
    if v6_spec.study_id != manifest["source_v6_study_id"]:
        raise _error("v6 source study ID to match the sweep manifest", v6_spec.study_id)
    if v5_spec.study_id != manifest["source_v5_study_id"]:
        raise _error("v5 source study ID to match the sweep manifest", v5_spec.study_id)
    transfer, _path = load_transfer_manifest(transfer_dir / "manifest.json")
    if transfer["transfer_id"] != manifest["source_transfer_id"]:
        raise _error("transfer ID to match the sweep manifest", transfer["transfer_id"])
    _validate_artifacts(v6_root, manifest["source_v6_artifacts"], "v6 source")
    _validate_artifacts(v5_root, manifest["source_v5_artifacts"], "v5 source")
    _validate_artifacts(
        transfer_dir, manifest["source_transfer_artifacts"], "transfer source"
    )
    return v6_root, v6_spec.data, v5_root, transfer_dir


def run_entry(
    *,
    manifest_path: str | Path,
    source_study: str | Path,
    v5_source_study: str | Path,
    transfer_root: str | Path,
    entry_index: int,
    data_root: str | Path,
    device: str,
) -> dict[str, Any]:
    manifest, path = load_manifest(manifest_path)
    sweep_root = path.parent
    v6_root, study, _v5_root, _transfer_dir = _validate_sources(
        manifest,
        source_study=source_study,
        v5_source_study=v5_source_study,
        transfer_root=transfer_root,
    )
    _require_current_source(manifest["code_provenance"], code_provenance())
    if type(entry_index) is not int or not 0 <= entry_index < len(manifest["entries"]):
        raise _error("entry_index in [0, 23]", entry_index)
    entry = manifest["entries"][entry_index]
    if entry["mode"] != "new_five_epoch_training":
        raise _error("a new_five_epoch_training entry", entry)
    completion_path = _completion_path(sweep_root, entry)
    if completion_path.is_file():
        _validate_completion(sweep_root, manifest, entry)
        return {"status": "resumed_complete", "entry_index": entry_index}

    execute_candidate_entry(
        study,
        v6_root,
        entry["row_id"],
        entry["role"],
        candidate_payload=entry["payload"],
        output_stage=STAGE_NAME,
        output_root=sweep_root,
        data_root=data_root,
        download=False,
        device=device,
    )
    completion = {
        "schema_version": ENTRY_COMPLETION_SCHEMA,
        "state": "complete",
        "sweep_id": manifest["sweep_id"],
        "entry_index": entry["entry_index"],
        "row_id": entry["row_id"],
        "scheme": entry["scheme"],
        "rho_conv": entry["rho_conv"],
        "rho_dense": entry["rho_dense"],
        "outputs": _output_records(sweep_root, entry),
    }
    _publish_once(completion_path, completion)
    return {"status": "complete", "entry_index": entry_index}


def scheme_entries(
    manifest: Mapping[str, Any], scheme: str
) -> list[Mapping[str, Any]]:
    if scheme not in {name for name, _row_id in SCHEME_ROWS}:
        raise _error("scheme to be ours or legacy", scheme)
    entries = [
        entry
        for entry in manifest["entries"]
        if entry["scheme"] == scheme
        and entry["mode"] == "new_five_epoch_training"
    ]
    if len(entries) != 8:
        raise _error(
            f"exactly eight new five-epoch entries for scheme {scheme!r}",
            [entry.get("entry_index") for entry in entries],
        )
    return entries


def _summary_for_entry(
    *,
    sweep_root: Path,
    v5_root: Path,
    transfer_dir: Path,
    entry: Mapping[str, Any],
) -> Mapping[str, Any]:
    if entry["mode"] == "new_five_epoch_training":
        return read_json(_entry_dir(sweep_root, entry) / "summary.json")
    relative = _source_summary_path(
        mode=str(entry["mode"]),
        row_id=str(entry["row_id"]),
        source_entry_id=str(entry["source_entry_id"]),
    )
    root = v5_root if entry["mode"] == "reuse_v5_candidate" else transfer_dir
    return read_json(root / relative)


def _record(
    *,
    sweep_root: Path,
    v5_root: Path,
    transfer_dir: Path,
    entry: Mapping[str, Any],
) -> dict[str, Any]:
    summary = _summary_for_entry(
        sweep_root=sweep_root,
        v5_root=v5_root,
        transfer_dir=transfer_dir,
        entry=entry,
    )
    accuracy = summary.get("final_validation_accuracy")
    admissible = bool(summary["admissible"])
    passed = admissible and accuracy is not None and float(accuracy) >= 0.9
    return {
        "entry_id": f"{entry['row_id']}--{entry['role']}",
        "row_id": entry["row_id"],
        "scheme": entry["scheme"],
        "rho_conv": float(entry["rho_conv"]),
        "rho_dense": float(entry["rho_dense"]),
        "evidence_mode": entry["mode"],
        "source_entry_id": entry["source_entry_id"],
        "status": summary["status"],
        "admissible": admissible,
        "passes": passed,
        "inadmissible_reason": (
            summary.get("inadmissible_reason")
            if not admissible
            else "final_validation_accuracy_below_0.9"
            if accuracy is not None and float(accuracy) < 0.9
            else None
        ),
        "final_validation_accuracy": accuracy,
        "final_validation_loss": summary.get("final_validation_loss"),
        "median_projection_efficiency": summary.get(
            "median_projection_efficiency"
        ),
        "learning_rates_by_weight": summary.get("learning_rates_by_weight", {}),
        "achieved_relative_updates_by_weight": summary.get(
            "observed_rho_relative_q90_complete_run_by_parameter", {}
        ),
        "early_relative_updates_by_weight": summary.get(
            "observed_peak_rho_relative_by_parameter", {}
        ),
        "maximum_bound_occupancy_by_parameter": summary.get(
            "maximum_bound_occupancy_by_parameter", {}
        ),
        "median_projection_efficiency_by_parameter": summary.get(
            "median_projection_efficiency_by_parameter", {}
        ),
    }


def _plot_heatmaps(
    records: Sequence[Mapping[str, Any]], root: Path, *, field: str, filename: str
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    parameters: tuple[str | None, ...]
    if field in {
        "final_validation_accuracy",
        "final_validation_loss",
        "median_projection_efficiency",
    }:
        parameters = (None,)
    else:
        parameters = ("ConvWeight_0", "ConvWeight_1", "DenseWeight_0")
    fig, axes = plt.subplots(
        len(SCHEME_ROWS),
        len(parameters),
        figsize=(5.2 * len(parameters), 4.2 * len(SCHEME_ROWS)),
        squeeze=False,
    )
    for row_index, (scheme, _row_id) in enumerate(SCHEME_ROWS):
        subset = [record for record in records if record["scheme"] == scheme]
        by_pair = {
            (record["rho_conv"], record["rho_dense"]): record for record in subset
        }
        for column_index, parameter in enumerate(parameters):
            matrix = []
            for rho_conv in RHO_CONV_GRID:
                values = []
                for rho_dense in RHO_DENSE_GRID:
                    value = by_pair[(rho_conv, rho_dense)].get(field)
                    if parameter is not None:
                        value = (value or {}).get(parameter)
                    values.append(math.nan if value is None else float(value))
                matrix.append(values)
            axis = axes[row_index][column_index]
            image = axis.imshow(matrix, origin="lower", aspect="auto")
            axis.set_xticks(range(len(RHO_DENSE_GRID)))
            axis.set_xticklabels([f"{value:g}" for value in RHO_DENSE_GRID])
            axis.set_yticks(range(len(RHO_CONV_GRID)))
            axis.set_yticklabels([f"{value:g}" for value in RHO_CONV_GRID])
            axis.set_xlabel("rho_dense")
            axis.set_ylabel("rho_conv")
            axis.set_title(f"{scheme}: {parameter or field}")
            fig.colorbar(image, ax=axis)
    fig.suptitle("Ordinary-MNIST Conv2 amplified-scheme rho diagnostic")
    fig.tight_layout()
    fig.savefig(root / filename, dpi=180)
    plt.close(fig)


def finalize_sweep(
    manifest_path: str | Path,
    *,
    source_study: str | Path,
    v5_source_study: str | Path,
    transfer_root: str | Path,
) -> dict[str, Any]:
    manifest, path = load_manifest(manifest_path)
    sweep_root = path.parent
    _v6_root, _study, v5_root, transfer_dir = _validate_sources(
        manifest,
        source_study=source_study,
        v5_source_study=v5_source_study,
        transfer_root=transfer_root,
    )
    records: list[dict[str, Any]] = []
    for entry in manifest["entries"]:
        if entry["mode"] == "new_five_epoch_training":
            _validate_completion(sweep_root, manifest, entry)
        records.append(
            _record(
                sweep_root=sweep_root,
                v5_root=v5_root,
                transfer_dir=transfer_dir,
                entry=entry,
            )
        )

    selections: dict[str, Any] = {}
    for scheme, _row_id in SCHEME_ROWS:
        subset = [record for record in records if record["scheme"] == scheme]
        selection = select_v6_conv2_baseline(
            [
                {
                    "candidate_id": record["entry_id"],
                    "rho_conv": record["rho_conv"],
                    "rho_dense": record["rho_dense"],
                    "admissible": record["admissible"],
                    "inadmissible_reason": record["inadmissible_reason"],
                    "final_validation_loss": record["final_validation_loss"],
                    "final_validation_accuracy": record["final_validation_accuracy"],
                    "median_projection_efficiency": record[
                        "median_projection_efficiency"
                    ],
                }
                for record in subset
            ],
            rho_conv_values=RHO_CONV_GRID,
            rho_dense_values=RHO_DENSE_GRID,
            minimum_accuracy=0.9,
            plateau_relative_tolerance=0.02,
        )
        selections[scheme] = selection
        plateau_ids = {item["candidate_id"] for item in selection["plateau"]}
        best_id = (
            None
            if selection["diagnostic_best"] is None
            else selection["diagnostic_best"]["candidate_id"]
        )
        for record in subset:
            record["in_loss_plateau"] = record["entry_id"] in plateau_ids
            record["diagnostic_best"] = record["entry_id"] == best_id

    result = {
        "schema_version": SUMMARY_SCHEMA,
        "status": "complete",
        "sweep_id": manifest["sweep_id"],
        "study_role": manifest["study_role"],
        "rho_conv_grid": list(RHO_CONV_GRID),
        "rho_dense_grid": list(RHO_DENSE_GRID),
        "minimum_final_validation_accuracy": 0.9,
        "selections_by_scheme": selections,
        "records": records,
        "official_test_read": False,
        "medium_affine_handoff_replaced": False,
        "final_paper_training_authorized": False,
    }
    _publish_once(sweep_root / "summary.json", result)

    fields = [
        "entry_id", "row_id", "scheme", "rho_conv", "rho_dense",
        "evidence_mode", "source_entry_id", "status", "admissible", "passes",
        "inadmissible_reason", "final_validation_accuracy", "final_validation_loss",
        "median_projection_efficiency", "in_loss_plateau", "diagnostic_best",
        "learning_rates_by_weight", "early_relative_updates_by_weight",
        "achieved_relative_updates_by_weight", "maximum_bound_occupancy_by_parameter",
        "median_projection_efficiency_by_parameter",
    ]
    mapping_fields = {
        "learning_rates_by_weight",
        "early_relative_updates_by_weight",
        "achieved_relative_updates_by_weight",
        "maximum_bound_occupancy_by_parameter",
        "median_projection_efficiency_by_parameter",
    }
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for record in records:
        row = dict(record)
        for field in mapping_fields:
            row[field] = json.dumps(row[field], sort_keys=True, separators=(",", ":"))
        writer.writerow(row)
    csv_path = sweep_root / "summary.csv"
    encoded = stream.getvalue().encode()
    if csv_path.exists() and csv_path.read_bytes() != encoded:
        raise RuntimeError(f"Expected immutable summary CSV bytes. Provided value: {csv_path}.")
    if not csv_path.exists():
        atomic_write_bytes(csv_path, encoded)

    plots = (
        ("final_validation_accuracy", "epoch5_accuracy.png"),
        ("final_validation_loss", "epoch5_loss.png"),
        ("learning_rates_by_weight", "raw_learning_rates.png"),
        ("early_relative_updates_by_weight", "early_relative_updates.png"),
        ("achieved_relative_updates_by_weight", "complete_relative_updates.png"),
        ("maximum_bound_occupancy_by_parameter", "maximum_occupancy.png"),
        ("median_projection_efficiency", "projection_efficiency.png"),
    )
    for field, filename in plots:
        if not (sweep_root / filename).exists():
            _plot_heatmaps(records, sweep_root, field=field, filename=filename)
    final_outputs = []
    for filename in ("summary.json", "summary.csv") + tuple(
        filename for _field, filename in plots
    ):
        output = sweep_root / filename
        final_outputs.append(
            {"path": filename, "sha256": sha256_file(output), "bytes": output.stat().st_size}
        )
    _publish_once(
        sweep_root / "complete.json",
        {
            "schema_version": FINAL_COMPLETION_SCHEMA,
            "state": "complete",
            "sweep_id": manifest["sweep_id"],
            "outputs": final_outputs,
        },
    )
    return result


def run_pack(
    *,
    manifest_path: str | Path,
    source_study: str | Path,
    v5_source_study: str | Path,
    transfer_root: str | Path,
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
        active: list[tuple[Mapping[str, Any], list[str], subprocess.Popen[str], Any]] = []
        try:
            for entry in wave:
                command = [
                    python,
                    "-m",
                    "experiments.run_mnist_conv_lr_conv2_scheme_rho_sweep",
                    "run-entry",
                    "--manifest", str(path),
                    "--source-study", str(Path(source_study).expanduser().resolve()),
                    "--v5-source-study", str(Path(v5_source_study).expanduser().resolve()),
                    "--transfer-root", str(Path(transfer_root).expanduser().resolve()),
                    "--entry-index", str(entry["entry_index"]),
                    "--data-root", str(Path(data_root).expanduser().resolve()),
                    "--device", device,
                ]
                stream = (
                    logs / f"wave_{wave_index:02d}_entry_{entry['entry_index']:02d}_{scheme}.log"
                ).open("w")
                process = subprocess.Popen(
                    command, stdout=stream, stderr=subprocess.STDOUT, text=True
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
                    "Expected every Conv2 scheme child in the wave to complete. "
                    f"Provided failures: {failures!r}."
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


def _add_sources(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--source-study", required=True)
    parser.add_argument("--v5-source-study", required=True)
    parser.add_argument("--transfer-root", required=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    create = commands.add_parser("create")
    _add_sources(create)
    create.add_argument("--results-root", required=True)
    run = commands.add_parser("run-entry")
    _add_sources(run)
    run.add_argument("--manifest", required=True)
    run.add_argument("--entry-index", required=True, type=int)
    run.add_argument("--data-root", required=True)
    run.add_argument("--device", default="cuda")
    pack = commands.add_parser("run-pack")
    _add_sources(pack)
    pack.add_argument("--manifest", required=True)
    pack.add_argument("--data-root", required=True)
    pack.add_argument("--device", default="cuda")
    pack.add_argument("--python", required=True)
    pack.add_argument("--log-dir", required=True)
    pack.add_argument("--scheme", choices=("ours", "legacy"), required=True)
    pack.add_argument("--concurrency", type=int, default=8)
    finalize = commands.add_parser("finalize")
    _add_sources(finalize)
    finalize.add_argument("--manifest", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "create":
            manifest, path = create_manifest(
                source_study=args.source_study,
                v5_source_study=args.v5_source_study,
                transfer_root=args.transfer_root,
                results_root=args.results_root,
            )
            result: Any = {
                "status": "created",
                "sweep_id": manifest["sweep_id"],
                "manifest": str(path),
                "mode_counts": manifest["mode_counts"],
            }
        elif args.command == "run-entry":
            result = run_entry(
                manifest_path=args.manifest,
                source_study=args.source_study,
                v5_source_study=args.v5_source_study,
                transfer_root=args.transfer_root,
                entry_index=args.entry_index,
                data_root=args.data_root,
                device=args.device,
            )
        elif args.command == "run-pack":
            result = run_pack(
                manifest_path=args.manifest,
                source_study=args.source_study,
                v5_source_study=args.v5_source_study,
                transfer_root=args.transfer_root,
                data_root=args.data_root,
                device=args.device,
                python=args.python,
                log_dir=args.log_dir,
                scheme=args.scheme,
                concurrency=args.concurrency,
            )
        else:
            result = finalize_sweep(
                args.manifest,
                source_study=args.source_study,
                v5_source_study=args.v5_source_study,
                transfer_root=args.transfer_root,
            )
    except (OSError, RuntimeError, ValueError) as exc:
        print(str(exc), file=sys.stderr, flush=True)
        return 2
    print(json.dumps(result, sort_keys=True, separators=(",", ":")), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
