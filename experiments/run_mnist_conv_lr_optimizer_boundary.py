"""CLI orchestration for the immutable Conv2 SGD/Adam boundary diagnostic."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

from experiments.mnist_conv.identity import code_provenance, sha256_file, sha256_json
from experiments.mnist_conv.io import read_json
from experiments.mnist_conv.lr_optimizer_boundary import (
    COMPLETION_SCHEMA_VERSION,
    REUSE_SOURCE_DIRECTORIES,
    SCHEMES,
    arm_id,
    artifact_record,
    assert_result_matches_entry,
    bias_capped_confirmation_entries,
    bind_stage_entries,
    build_main_manifest,
    completion_action,
    execution_entry_provenance,
    load_main_manifest,
    load_execution_plan,
    main_grid_entries,
    normalize_probe_bundle,
    pack_entries,
    publish_once,
    reuse_cells_from_root,
    raw_main_learning_rates,
    select_all_arms,
    select_concurrency,
    terminal_completion,
    upper_sentinel_entries,
    validate_result,
    verify_reuse_cells_against_root,
)
from experiments.mnist_conv.lr_optimizer_boundary_plot import (
    render_boundary_report,
)
from experiments.mnist_conv.lr_optimizer_boundary_spec import (
    OptimizerBoundaryStudySpec,
    optimizer_boundary_study_template,
)


def _error(expected: str, provided: Any) -> ValueError:
    return ValueError(f"Expected {expected}. Provided value: {provided!r}.")


_BENCHMARK_SCHEMA_VERSION = (
    "mnist-conv-lr-optimizer-boundary-benchmark/v1"
)
_BENCHMARK_WIDTHS = (1, 2, 4, 8, 12, 16)
_BENCHMARK_WARMUP_STEPS = 32
_BENCHMARK_MEASURED_STEPS = 256
_BENCHMARK_READY_TIMEOUT_SECONDS = 600.0
_RESOURCE_CONTRACT = {
    "account": "fmu@v100",
    "partition": "gpu_p13",
    "qos": "qos_gpu-t3",
    "constraint": "v100-32g",
    "gpus": 1,
    "cpus": 16,
    "walltime_hours": 20,
}


def _attempt_result_relative_path(
    result: Mapping[str, Any], *, resume_action: str
) -> str:
    attempt = result.get("attempt_relative_path")
    if attempt is None:
        if resume_action == "restart_from_initialization":
            raise _error(
                "a fresh Adam attempt path after restart",
                attempt,
            )
        return "result.json"
    if (
        resume_action != "restart_from_initialization"
        or not isinstance(attempt, str)
        or len(Path(attempt).parts) != 2
        or Path(attempt).parts[0] != "attempts"
        or not Path(attempt).parts[1].startswith("attempt-")
        or len(Path(attempt).parts[1]) != len("attempt-000001")
        or not Path(attempt).parts[1][len("attempt-") :].isdigit()
    ):
        raise _error(
            "restart result attempt_relative_path matching "
            "'attempts/attempt-NNNNNN'",
            {"resume_action": resume_action, "attempt_relative_path": attempt},
        )
    return f"{attempt}/result.json"


def create_study_config(
    *, reuse_root: str | Path, output_path: str | Path
) -> tuple[OptimizerBoundaryStudySpec, Path]:
    cells = reuse_cells_from_root(reuse_root)
    spec = OptimizerBoundaryStudySpec.from_dict(
        optimizer_boundary_study_template(cells)
    )
    output = Path(output_path).expanduser().resolve()
    publish_once(output, spec.to_dict())
    return spec, output


def verify_study_config(
    *, study_config: str | Path, reuse_root: str | Path
) -> OptimizerBoundaryStudySpec:
    spec = OptimizerBoundaryStudySpec.from_path(study_config)
    verify_reuse_cells_against_root(spec.data["reuse"]["cells"], reuse_root)
    return spec


def _expected_initialization_sha256(reuse_root: str | Path) -> str:
    root = Path(reuse_root).expanduser().resolve()
    source = root / REUSE_SOURCE_DIRECTORIES[("baseline", 0.01)]
    run_specs = sorted(source.glob("run_spec.v*.json"))
    if len(run_specs) != 1:
        raise _error(
            "exactly one baseline source run spec",
            [path.name for path in run_specs],
        )
    value = (
        read_json(run_specs[0])
        .get("run", {})
        .get("lr_provenance", {})
        .get("initialization_checkpoint_sha256")
    )
    if not isinstance(value, str) or len(value) != 64:
        raise _error("a source initialization SHA-256", value)
    return value


def _historical_sgd_normalization_by_scheme(
    spec: OptimizerBoundaryStudySpec, reuse_root: str | Path
) -> dict[str, dict[str, Any]]:
    """Read the authoritative SGD units/hashes from the six frozen sources."""

    root = Path(reuse_root).expanduser().resolve()
    scheme_by_row = {row["row_id"]: row["scheme"] for row in spec.rows}
    records: dict[str, list[dict[str, Any]]] = {
        scheme: [] for scheme in SCHEMES
    }
    for cell in spec.data["reuse"]["cells"]:
        scheme = scheme_by_row[cell["row_id"]]
        source = root / cell["source_entry_id"]
        run_specs = sorted(source.glob("run_spec.v*.json"))
        if len(run_specs) != 1:
            raise _error(
                "one frozen source run spec",
                [path.name for path in run_specs],
            )
        provenance = read_json(run_specs[0])["run"]["lr_provenance"]
        units = provenance.get("median_unit_by_weight")
        probe_sha = provenance.get("probe_sha256")
        if (
            not isinstance(units, Mapping)
            or set(units)
            != {"ConvWeight_0", "ConvWeight_1", "DenseWeight_0"}
            or not isinstance(probe_sha, str)
            or len(probe_sha) != 64
        ):
            raise _error(
                "source median units and a source probe SHA-256",
                provenance,
            )
        records[scheme].append(
            {
                "source_entry_id": cell["source_entry_id"],
                "probe_sha256": probe_sha,
                "normalization_unit_by_weight": {
                    name: float(units[name]) for name in sorted(units)
                },
            }
        )
    result: dict[str, dict[str, Any]] = {}
    for scheme, items in records.items():
        if len(items) != 2:
            raise _error("two source dense slices per SGD scheme", items)
        signatures = {
            (
                item["probe_sha256"],
                tuple(item["normalization_unit_by_weight"].items()),
            )
            for item in items
        }
        if len(signatures) != 1:
            raise RuntimeError(
                "Expected both hash-verified SGD dense slices to share the "
                "same historical normalization units and probe hash. "
                f"Provided value: scheme={scheme!r}, records={items!r}."
            )
        result[scheme] = {
            "probe_sha256": items[0]["probe_sha256"],
            "normalization_unit_by_weight": items[0][
                "normalization_unit_by_weight"
            ],
            "source_entry_ids": sorted(
                item["source_entry_id"] for item in items
            ),
        }
    return result


def prepare_optimizer_probes(
    *,
    study_config: str | Path,
    study_dir: str | Path,
    reuse_root: str | Path,
    data_root: str | Path,
    initialization_source: str | Path,
    output_path: str | Path,
    device: str,
) -> dict[str, Any]:
    """Prepare shared assets and publish all three SGD plus three Adam probes."""

    spec = verify_study_config(
        study_config=study_config, reuse_root=reuse_root
    )
    from experiments.mnist_conv.lr_optimizer_boundary_runtime import (
        compose_effective_optimizer_probe,
        measure_optimizer_probe,
        prepare_boundary_assets,
    )

    assets = prepare_boundary_assets(
        spec.data,
        study_dir,
        data_root=data_root,
        download=False,
        device=device,
        initialization_source=initialization_source,
        expected_checkpoint_sha256=_expected_initialization_sha256(reuse_root),
    )
    historical_sgd = _historical_sgd_normalization_by_scheme(spec, reuse_root)
    probes_by_arm = {}
    for row in spec.rows:
        for optimizer in ("sgd", "adam"):
            measured = measure_optimizer_probe(
                spec.data,
                assets,
                row_id=row["row_id"],
                optimizer_name=optimizer,
                device=device,
                publish=True,
            )
            if optimizer == "sgd":
                source = historical_sgd[row["scheme"]]
                effective = compose_effective_optimizer_probe(
                    measured,
                    source_weight_units_by_parameter=source[
                        "normalization_unit_by_weight"
                    ],
                    source_weight_probe_sha256=source["probe_sha256"],
                )
            else:
                effective = compose_effective_optimizer_probe(measured)
            key = arm_id(row["scheme"], optimizer)
            relative_measured_path = (
                Path("probes") / key / "probe.json"
            ).as_posix()
            effective_record = {
                "schema_version": (
                    "mnist-conv-lr-optimizer-boundary-effective-probe/v1"
                ),
                "row_id": row["row_id"],
                "scheme": row["scheme"],
                "optimizer": optimizer,
                "optimizer_name": effective["optimizer_name"],
                "optimizer_parameters": effective["optimizer_parameters"],
                "normalization_unit_by_weight": effective[
                    "normalization_unit_by_weight"
                ],
                "shadow_weight_unit_by_parameter": effective[
                    "shadow_weight_unit_by_parameter"
                ],
                "q90_bias_units_by_parameter": effective[
                    "bias_q90_unit_by_parameter"
                ],
                "weight_normalization_source": effective[
                    "weight_normalization_source"
                ],
                "probe_sha256": effective[
                    "weight_normalization_probe_sha256"
                ],
                "weight_normalization_probe_sha256": effective[
                    "weight_normalization_probe_sha256"
                ],
                "bias_q90_probe_sha256": effective[
                    "bias_q90_probe_sha256"
                ],
                "fresh_shadow_probe_relative_path": relative_measured_path,
                "fresh_shadow_probe_sha256": measured["artifact_sha256"],
                "normalization_batch_count": effective[
                    "normalization_batch_count"
                ],
                "normalization_batch_order_sha256": effective[
                    "normalization_batch_order_sha256"
                ],
                "checkpoint_sha256": effective[
                    "initialization_checkpoint_sha256"
                ],
                "effective_probe_composed": True,
                "official_test_read": False,
            }
            effective_path = (
                Path(study_dir).expanduser().resolve()
                / "probes"
                / key
                / "effective.json"
            )
            publish_once(effective_path, effective_record)
            probes_by_arm[key] = {
                **effective_record,
                "effective_probe_relative_path": (
                    Path("probes") / key / "effective.json"
                ).as_posix(),
                "effective_probe_artifact_sha256": sha256_file(effective_path),
                "weight_units_by_parameter": effective_record[
                    "normalization_unit_by_weight"
                ],
            }
    provenance = code_provenance()
    bundle = {
        "schema_version": "mnist-conv-lr-optimizer-boundary-probes/v1",
        "study_id": spec.study_id,
        "code_provenance": provenance,
        "code_fingerprint": provenance["effective_code_fingerprint"],
        "parent_sha256": sha256_json(spec.data["reuse"]),
        "checkpoint_sha256": assets.initialization["checkpoint_sha256"],
        "checkpoint_tensor_sha256": assets.initialization[
            "parameter_tensor_sha256"
        ],
        "minibatch_sha256": sha256_file(assets.minibatches_path),
        "probes_by_arm": probes_by_arm,
        "official_test_read": False,
    }
    normalize_probe_bundle(bundle)
    publish_once(output_path, bundle)
    return bundle


def _bindings_from_spec(
    spec: OptimizerBoundaryStudySpec,
) -> list[dict[str, Any]]:
    bindings: list[dict[str, Any]] = []
    scheme_by_row = {row["row_id"]: row["scheme"] for row in spec.rows}
    for cell in spec.data["reuse"]["cells"]:
        scheme = scheme_by_row[cell["row_id"]]
        source_entry_id = cell["source_entry_id"]
        artifacts = [
            {
                "path": f"{source_entry_id}/complete.json",
                "sha256": cell["completion_sha256"],
                "bytes": cell["completion_bytes"],
            },
            *[
                {
                    "path": f"{source_entry_id}/{output['name']}",
                    "sha256": output["sha256"],
                    "bytes": output["bytes"],
                }
                for output in cell["outputs"]
            ],
        ]
        artifacts.sort(key=lambda record: record["path"])
        bindings.append(
            {
                "entry": {
                    "entry_id": next(
                        entry["entry_id"]
                        for entry in main_grid_entries()
                        if entry["scheme"] == scheme
                        and entry["optimizer"] == "sgd"
                        and entry["rho_conv"] == cell["rho_conv"]
                        and entry["rho_dense"] == cell["rho_dense"]
                    ),
                    "scheme": scheme,
                    "optimizer": "sgd",
                    "rho_conv": cell["rho_conv"],
                    "rho_dense": cell["rho_dense"],
                },
                "source_id": cell["source_collection_id"],
                "source_entry_id": source_entry_id,
                "artifacts": artifacts,
            }
        )
    return bindings


def create_main_manifest(
    *,
    study_config: str | Path,
    reuse_root: str | Path,
    probes_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    """Bind probe outputs and all 36 per-entry LR vectors."""

    spec = verify_study_config(
        study_config=study_config, reuse_root=reuse_root
    )
    probes = read_json(probes_path)
    normalized_probes = normalize_probe_bundle(probes)
    if normalized_probes["study_id"] != spec.study_id:
        raise _error(
            "probe bundle study id to match the study config exactly",
            {
                "probes": normalized_probes["study_id"],
                "study": spec.study_id,
            },
        )
    probe_sha256_by_arm = {
        key: value["probe_sha256"]
        for key, value in normalized_probes["probes_by_arm"].items()
    }
    bias_q90_probe_sha256_by_arm = {
        key: value["bias_q90_probe_sha256"]
        for key, value in normalized_probes["probes_by_arm"].items()
    }
    normalization_unit_by_arm = {
        key: value["weight_units_by_parameter"]
        for key, value in normalized_probes["probes_by_arm"].items()
    }
    bias_q90_unit_by_arm = {
        key: value["q90_bias_units_by_parameter"]
        for key, value in normalized_probes["probes_by_arm"].items()
    }
    manifest = build_main_manifest(
        study_id=spec.study_id,
        study_config_sha256=sha256_file(study_config),
        code_fingerprint=normalized_probes["code_fingerprint"],
        parent_sha256=normalized_probes["parent_sha256"],
        checkpoint_sha256=normalized_probes["checkpoint_sha256"],
        minibatch_sha256=normalized_probes["minibatch_sha256"],
        probe_sha256_by_arm=probe_sha256_by_arm,
        bias_q90_probe_sha256_by_arm=bias_q90_probe_sha256_by_arm,
        normalization_unit_by_arm=normalization_unit_by_arm,
        bias_q90_unit_by_arm=bias_q90_unit_by_arm,
        raw_learning_rates_by_entry=raw_main_learning_rates(probes),
        optimizer_parameters_by_name=spec.data["optimizer_arms"],
        reuse_bindings=_bindings_from_spec(spec),
        reuse_root=reuse_root,
    )
    publish_once(output_path, manifest)
    return manifest


def _entry_directory(
    study_dir: Path, entry: Mapping[str, Any]
) -> Path:
    return (
        study_dir
        / "stages"
        / str(entry["stage"])
        / "entries"
        / str(entry["entry_id"])
    )


def _find_entry(
    manifest: Mapping[str, Any], entry_id_value: str
) -> dict[str, Any]:
    matches = [
        dict(entry)
        for entry in manifest["entries"]
        if entry["entry_id"] == entry_id_value
    ]
    if len(matches) != 1:
        raise _error("one manifest entry id", entry_id_value)
    return matches[0]


def _reuse_result(
    *,
    entry: Mapping[str, Any],
    reuse_root: Path,
) -> dict[str, Any]:
    binding = entry["reuse_binding"]
    source_entry_id = binding["source_entry_id"]
    summary_path = reuse_root / source_entry_id / "summary.json"
    summary = read_json(summary_path)
    return {
        "schema_version": "mnist-conv-lr-optimizer-boundary-result/v1",
        "entry_id": entry["entry_id"],
        "scheme": entry["scheme"],
        "optimizer": entry["optimizer"],
        "stage": entry["stage"],
        "rho_conv": entry["rho_conv"],
        "rho_dense": entry["rho_dense"],
        "bias_policy": "attached",
        "status": "complete",
        "completed_steps": summary["completed_steps"],
        "admissible": bool(summary["admissible"]),
        "final_validation_loss": summary["final_validation_loss"],
        "final_validation_accuracy": summary["final_validation_accuracy"],
        "median_projection_efficiency": summary[
            "median_projection_efficiency"
        ],
        "optimizer_parameters": entry["optimizer_parameters"],
        "probe_sha256": entry["probe_sha256"],
        "bias_q90_probe_sha256": entry["bias_q90_probe_sha256"],
        "normalization_unit_by_weight": entry[
            "normalization_unit_by_weight"
        ],
        "bias_q90_unit_by_parameter": entry[
            "bias_q90_unit_by_parameter"
        ],
        "parent_sha256": entry["parent_sha256"],
        "parent_entry_completion_sha256": entry[
            "parent_entry_completion_sha256"
        ],
        "raw_learning_rates_by_parameter": entry[
            "raw_learning_rates_by_parameter"
        ],
        "minibatch_sha256": entry["minibatch_sha256"],
        "checkpoint_sha256": entry["checkpoint_sha256"],
        "code_fingerprint": entry["code_fingerprint"],
        "evidence_mode": "reuse_hash_verified_sgd",
        "source_binding_sha256": binding["binding_sha256"],
        "source_summary_sha256": sha256_file(summary_path),
        "official_test_read": False,
    }


def _output_records(entry_dir: Path) -> list[dict[str, Any]]:
    records = []
    for path in sorted(entry_dir.rglob("*")):
        if (
            not path.is_file()
            or path.is_symlink()
            or path.name == "complete.json"
            or ".tmp" in path.name
        ):
            continue
        records.append(
            artifact_record(entry_dir, path.relative_to(entry_dir).as_posix())
        )
    if not records:
        raise RuntimeError(
            f"Expected entry execution to publish at least one output. Provided value: {entry_dir}."
        )
    return records


def run_entry(
    *,
    study_config: str | Path,
    study_dir: str | Path,
    manifest_path: str | Path,
    reuse_root: str | Path,
    entry_id_value: str,
    data_root: str | Path,
    device: str,
) -> dict[str, Any]:
    """Execute one entry or materialize one hash-verified reused result."""

    spec = verify_study_config(
        study_config=study_config, reuse_root=reuse_root
    )
    manifest = load_execution_plan(manifest_path, reuse_root=reuse_root)
    if manifest["study_id"] != spec.study_id:
        raise _error("manifest and study config to have the same study id", {
            "manifest": manifest["study_id"],
            "study": spec.study_id,
        })
    entry = _find_entry(manifest, entry_id_value)
    current_fingerprint = code_provenance()["effective_code_fingerprint"]
    if current_fingerprint != entry["code_fingerprint"]:
        raise _error(
            "current code fingerprint to match the immutable plan entry",
            {
                "current": current_fingerprint,
                "entry": entry["code_fingerprint"],
            },
        )
    root = Path(study_dir).expanduser().resolve()
    entry_dir = _entry_directory(root, entry)
    action = completion_action(
        entry_dir,
        optimizer=entry["optimizer"],
        expected_entry_id=entry["entry_id"],
        expected_entry=entry,
    )
    if action == "skip_immutable_complete":
        return {
            "status": "resumed_complete",
            "entry_id": entry["entry_id"],
        }
    entry_dir.mkdir(parents=True, exist_ok=True)
    if entry["mode"] == "reuse_hash_verified_sgd":
        result = _reuse_result(
            entry=entry,
            reuse_root=Path(reuse_root).expanduser().resolve(),
        )
    else:
        # Imported lazily so pure manifest/selection work never constructs a
        # model and old v1--v6 readers remain independent of this new runtime.
        from experiments.mnist_conv.lr_optimizer_boundary_runtime import (
            execute_optimizer_boundary_entry,
        )

        runtime_entry = {**entry, "resume_action": action}
        result = execute_optimizer_boundary_entry(
            spec.data,
            root,
            runtime_entry,
            data_root=data_root,
            download=False,
            device=device,
        )
    provenance = execution_entry_provenance(entry)
    result = dict(result)
    for field, expected in provenance.items():
        if field in result and result[field] != expected:
            raise RuntimeError(
                "Expected runtime result provenance to match its plan entry. "
                f"Provided value: field={field!r}, "
                f"runtime={result[field]!r}, entry={expected!r}."
            )
        result[field] = expected
    normalized = validate_result(result)
    result_relative_path = _attempt_result_relative_path(
        normalized, resume_action=action
    )
    result_path = entry_dir / result_relative_path
    publish_once(result_path, normalized)
    records = _output_records(entry_dir)
    completion = terminal_completion(
        entry_id_value=entry["entry_id"],
        result=normalized,
        outputs=records,
        entry=entry,
        result_relative_path=result_relative_path,
    )
    if completion.get("schema_version") != COMPLETION_SCHEMA_VERSION:
        raise RuntimeError("Internal completion-schema mismatch.")
    publish_once(entry_dir / "complete.json", completion)
    return {
        "status": completion["state"],
        "entry_id": entry["entry_id"],
        "completed_steps": completion["completed_steps"],
    }


def _load_results(path: str | Path) -> list[dict[str, Any]]:
    value = read_json(path)
    records = value.get("records") if isinstance(value, Mapping) else value
    if not isinstance(records, list):
        raise _error("a list or an object containing records", value)
    return [dict(record) for record in records]


def materialize_reuse_entries(
    *,
    study_config: str | Path,
    study_dir: str | Path,
    manifest_path: str | Path,
    reuse_root: str | Path,
    data_root: str | Path,
    device: str,
) -> dict[str, Any]:
    manifest = load_main_manifest(manifest_path, reuse_root=reuse_root)
    reuse_entries = [
        entry
        for entry in manifest["entries"]
        if entry["mode"] == "reuse_hash_verified_sgd"
    ]
    results = [
        run_entry(
            study_config=study_config,
            study_dir=study_dir,
            manifest_path=manifest_path,
            reuse_root=reuse_root,
            entry_id_value=entry["entry_id"],
            data_root=data_root,
            device=device,
        )
        for entry in reuse_entries
    ]
    return {
        "status": "complete",
        "entry_count": len(results),
        "results": results,
    }


def aggregate_results(
    *,
    study_dir: str | Path,
    plan_paths: Sequence[str | Path],
    output_path: str | Path,
) -> dict[str, Any]:
    root = Path(study_dir).expanduser().resolve()
    expected: dict[str, Mapping[str, Any]] = {}
    for plan_path in plan_paths:
        plan = load_execution_plan(plan_path)
        for entry in plan["entries"]:
            if entry["entry_id"] in expected:
                raise _error("unique entry ids across plans", entry["entry_id"])
            expected[entry["entry_id"]] = entry
    records = []
    for entry_id_value, entry in sorted(expected.items()):
        entry_dir = _entry_directory(root, entry)
        action = completion_action(
            entry_dir,
            optimizer=entry["optimizer"],
            expected_entry_id=entry_id_value,
            expected_entry=entry,
        )
        if action != "skip_immutable_complete":
            raise RuntimeError(
                "Expected every aggregated entry to have an immutable terminal "
                f"completion. Provided value: {entry_id_value} -> {action}."
            )
        completion = read_json(entry_dir / "complete.json")
        result_relative_path = completion.get(
            "result_relative_path", "result.json"
        )
        if (
            not isinstance(result_relative_path, str)
            or Path(result_relative_path).is_absolute()
            or ".." in Path(result_relative_path).parts
        ):
            raise _error(
                "a portable completion result_relative_path",
                result_relative_path,
            )
        result_path = entry_dir / result_relative_path
        result = assert_result_matches_entry(read_json(result_path), entry)
        records.append(
            {
                **result,
                "result_path": result_path.relative_to(root).as_posix(),
                "result_sha256": sha256_file(result_path),
                "completion_path": (
                    entry_dir / "complete.json"
                ).relative_to(root).as_posix(),
                "completion_sha256": sha256_file(entry_dir / "complete.json"),
            }
        )
    payload = {
        "schema_version": "mnist-conv-lr-optimizer-boundary-results/v1",
        "record_count": len(records),
        "records": records,
        "official_test_read": False,
    }
    publish_once(output_path, payload)
    return payload


def _verified_existing_final(
    root: Path,
    *,
    selection_input_results_sha256: str,
    final_results_sha256: str,
    selections_sha256: str,
) -> dict[str, Any] | None:
    """Return a verified immutable final summary, or require a clean directory."""

    if not root.exists():
        return None
    if root.is_symlink() or not root.is_dir():
        raise RuntimeError(
            f"Expected final output root to be a regular directory: {root}."
        )
    existing = sorted(path for path in root.rglob("*") if path.is_file() or path.is_symlink())
    if not existing:
        return None
    completion_path = root / "complete.json"
    if completion_path.is_symlink() or not completion_path.is_file():
        raise RuntimeError(
            "Expected a non-empty final directory to contain a verifiable "
            "immutable complete.json before any plot rendering."
        )
    completion = read_json(completion_path)
    expected_inputs = {
        "selection_input_results_sha256": selection_input_results_sha256,
        "final_results_sha256": final_results_sha256,
        "selections_sha256": selections_sha256,
    }
    if (
        completion.get("schema_version")
        != "mnist-conv-lr-optimizer-boundary-final-completion/v1"
        or completion.get("state") != "complete"
        or completion.get("official_test_read") is not False
        or any(completion.get(key) != value for key, value in expected_inputs.items())
    ):
        raise RuntimeError(
            "Expected existing final completion provenance to match the exact "
            "current results and selections before immutable skip."
        )
    outputs = completion.get("outputs")
    if not isinstance(outputs, list) or not outputs:
        raise RuntimeError("Expected final completion to bind non-empty outputs.")
    expected_paths: set[str] = set()
    for record in outputs:
        if not isinstance(record, Mapping) or set(record) != {
            "path",
            "sha256",
            "bytes",
        }:
            raise RuntimeError(
                f"Expected path/hash/size final output record: {record!r}."
            )
        relative = Path(str(record["path"]))
        if relative.is_absolute() or ".." in relative.parts:
            raise RuntimeError(
                f"Expected portable final output path: {record['path']!r}."
            )
        target = root / relative
        if (
            target.is_symlink()
            or not target.is_file()
            or sha256_file(target) != record["sha256"]
            or target.stat().st_size != record["bytes"]
        ):
            raise RuntimeError(
                f"Expected immutable final output hash/size to verify: {target}."
            )
        expected_paths.add(relative.as_posix())
    actual_paths = {
        path.relative_to(root).as_posix()
        for path in existing
        if path != completion_path
    }
    if actual_paths != expected_paths:
        raise RuntimeError(
            "Expected existing final directory to contain exactly the "
            "completion-bound outputs."
        )
    summary_path = root / "summary.json"
    if "summary.json" not in expected_paths:
        raise RuntimeError("Expected final completion to bind summary.json.")
    summary = read_json(summary_path)
    if any(summary.get(key) != value for key, value in expected_inputs.items()):
        raise RuntimeError(
            "Expected immutable final summary input hashes to match completion."
        )
    return summary


def finalize_boundary_study(
    *,
    results_path: str | Path,
    selections_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    selection_payload = read_json(selections_path)
    final_results_sha256 = sha256_file(results_path)
    selections_sha256 = sha256_file(selections_path)
    selection_results_sha256 = selection_payload.get("results_sha256")
    if (
        not isinstance(selection_results_sha256, str)
        or len(selection_results_sha256) != 64
    ):
        raise _error(
            "selection artifact to bind its attached-candidate results hash",
            selection_results_sha256,
        )
    root = Path(output_dir).expanduser().resolve()
    existing_summary = _verified_existing_final(
        root,
        selection_input_results_sha256=selection_results_sha256,
        final_results_sha256=final_results_sha256,
        selections_sha256=selections_sha256,
    )
    if existing_summary is not None:
        return existing_summary
    records = _load_results(results_path)
    selections = selection_payload["selections_by_arm"]
    expected_arms = {
        arm_id(scheme, optimizer)
        for scheme in ("baseline", "ours", "legacy")
        for optimizer in ("sgd", "adam")
    }
    if not isinstance(selections, Mapping) or set(selections) != expected_arms:
        raise _error(
            f"resolved selections for all six arms {sorted(expected_arms)!r}",
            sorted(selections) if isinstance(selections, Mapping) else selections,
        )
    for key, selection in selections.items():
        if (
            not isinstance(selection, Mapping)
            or selection.get("status") != "selected"
            or selection.get("trigger_upper_sentinel") is not False
            or selection.get("selected") is None
        ):
            raise RuntimeError(
                "Expected finalization to receive only fully resolved "
                f"post-sentinel selections. Provided arm: {key!r}."
            )
    selected_ids = {
        selection["selected"]["entry_id"]
        for selection in selections.values()
        if selection.get("selected") is not None
    }
    records_by_id = {record["entry_id"]: record for record in records}
    confirmations_by_arm: dict[str, list[dict[str, Any]]] = {
        key: [] for key in expected_arms
    }
    for record in records:
        if record.get("stage") == "bias_capped_confirmation":
            key = arm_id(record["scheme"], record["optimizer"])
            confirmations_by_arm[key].append(record)
    for key, selection in selections.items():
        selected = selection.get("selected")
        final_record = records_by_id.get(selected["entry_id"])
        if (
            final_record is None
            or final_record.get("bias_policy", "attached") != "attached"
            or final_record.get("result_sha256")
            != selected.get("result_sha256")
            or final_record.get("completion_sha256")
            != selected.get("completion_sha256")
        ):
            raise RuntimeError(
                "Expected every selected attached candidate and its result/"
                "completion hashes to remain present unchanged in the final "
                f"aggregate. Provided arm: {key!r}."
            )
        confirmations = confirmations_by_arm[key]
        if len(confirmations) != 1:
            raise RuntimeError(
                "Expected exactly one capped confirmation for every resolved "
                f"arm. Provided arm={key!r}, count={len(confirmations)}."
            )
        confirmation = confirmations[0]
        if (
            confirmation.get("bias_policy") != "capped"
            or confirmation.get("rho_conv") != selected.get("rho_conv")
            or confirmation.get("rho_dense") != selected.get("rho_dense")
            or confirmation.get("parent_entry_id") != selected["entry_id"]
            or confirmation.get("parent_result_sha256")
            != selected.get("result_sha256")
            or confirmation.get("parent_entry_completion_sha256")
            != selected.get("completion_sha256")
            or confirmation.get("parent_decision_sha256")
            != selections_sha256
        ):
            raise RuntimeError(
                "Expected capped confirmation coordinates and parent hashes "
                f"to bind the resolved attached selection for arm {key!r}."
            )
        selected_rates = final_record["raw_learning_rates_by_parameter"]
        confirmation_rates = confirmation["raw_learning_rates_by_parameter"]
        for weight in ("ConvWeight_0", "ConvWeight_1", "DenseWeight_0"):
            if confirmation_rates.get(weight) != selected_rates.get(weight):
                raise RuntimeError(
                    "Expected capped confirmation weight LRs to remain "
                    f"unchanged for arm {key!r}, parameter {weight!r}."
                )
        for bias in ("Bias_0", "Bias_1"):
            if confirmation_rates.get(bias, math.inf) > selected_rates.get(
                bias, -math.inf
            ):
                raise RuntimeError(
                    "Expected capped confirmation bias LRs never to exceed "
                    f"their attached-selection values for arm {key!r}."
                )
    plot_records = [
        {
            **record,
            "diagnostic_best": record["entry_id"] in selected_ids,
        }
        for record in records
    ]
    plot_paths = render_boundary_report(plot_records, root / "plots")
    summary = {
        "schema_version": "mnist-conv-lr-optimizer-boundary-final/v1",
        "status": "complete",
        "study_role": "ordinary_mnist_local_high_rho_optimizer_diagnostic",
        "adam_scope": "local_high_rho_surface_only",
        "adam_globally_tuned": False,
        "medium_affine_handoff_replaced": False,
        "final_paper_optimizer_changed": False,
        "selections_by_arm": selections,
        "selection_input_results_sha256": selection_results_sha256,
        "final_results_sha256": final_results_sha256,
        "selections_sha256": selections_sha256,
        "record_count": len(records),
        "new_run_count": sum(
            record.get("evidence_mode") != "reuse_hash_verified_sgd"
            for record in records
        ),
        "official_test_read": False,
    }
    summary_path = root / "summary.json"
    records_path = root / "records.json"
    publish_once(summary_path, summary)
    publish_once(
        records_path,
        {
            "schema_version": "mnist-conv-lr-optimizer-boundary-final-records/v1",
            "records": plot_records,
            "official_test_read": False,
        },
    )
    outputs = [
        {
            "path": path.relative_to(root).as_posix(),
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        }
        for path in sorted([summary_path, records_path, *plot_paths])
    ]
    publish_once(
        root / "complete.json",
        {
            "schema_version": "mnist-conv-lr-optimizer-boundary-final-completion/v1",
            "state": "complete",
            "selection_input_results_sha256": selection_results_sha256,
            "final_results_sha256": final_results_sha256,
            "selections_sha256": selections_sha256,
            "outputs": outputs,
            "official_test_read": False,
        },
    )
    return summary


def select_and_plan_sentinels(
    *,
    results_path: str | Path,
    main_manifest_path: str | Path,
    probes_path: str | Path,
    selection_path: str | Path,
    sentinel_path: str | Path,
) -> dict[str, Any]:
    records = _load_results(results_path)
    selections = select_all_arms(records)
    sentinel_entries = upper_sentinel_entries(selections)
    payload = {
        "schema_version": "mnist-conv-lr-optimizer-boundary-selections/v1",
        "status": (
            "upper_sentinels_required"
            if sentinel_entries
            else "base_selection_complete"
        ),
        "results_sha256": sha256_file(results_path),
        "main_manifest_sha256": sha256_file(main_manifest_path),
        "probe_bundle_sha256": sha256_file(probes_path),
        "selections_by_arm": selections,
        "official_test_read": False,
    }
    publish_once(selection_path, payload)
    selection_sha256 = sha256_file(selection_path)
    sentinel_entries = [
        {
            **entry,
            "parent_sha256": selection_sha256,
            "parent_entry_completion_sha256": None,
            "parent_decision_sha256": selection_sha256,
        }
        for entry in sentinel_entries
    ]
    main_manifest = load_main_manifest(main_manifest_path)
    sentinel_plan = bind_stage_entries(
        sentinel_entries,
        main_manifest=main_manifest,
        probe_bundle=read_json(probes_path),
        stage="upper_sentinel",
        stage_provenance={
            "selection_sha256": selection_sha256,
            "results_sha256": sha256_file(results_path),
            "main_manifest_sha256": sha256_file(main_manifest_path),
            "probe_bundle_sha256": sha256_file(probes_path),
        },
    )
    publish_once(sentinel_path, sentinel_plan)
    return payload


def create_confirmations(
    *,
    selections_path: str | Path,
    results_path: str | Path,
    bias_probe_path: str | Path,
    main_manifest_path: str | Path,
    probes_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    selection_payload = read_json(selections_path)
    expected_inputs = {
        "results_sha256": sha256_file(results_path),
        "main_manifest_sha256": sha256_file(main_manifest_path),
        "probe_bundle_sha256": sha256_file(probes_path),
    }
    observed_inputs = {
        key: selection_payload.get(key) for key in expected_inputs
    }
    if observed_inputs != expected_inputs:
        raise RuntimeError(
            "Expected the selection artifact to bind the exact results, "
            "main manifest, and probe bundle supplied for confirmations. "
            f"Provided value: expected={expected_inputs!r}, "
            f"observed={observed_inputs!r}."
        )
    if sha256_file(bias_probe_path) != expected_inputs["probe_bundle_sha256"]:
        raise RuntimeError(
            "Expected bias-cap units to come from the same canonical probe "
            "bundle bound by selection and execution provenance."
        )
    records = _load_results(results_path)
    result_by_id = {}
    for record in records:
        result = dict(record)
        if "result_sha256" not in result:
            raise _error(
                "every aggregated result to carry result_sha256",
                result,
            )
        result_by_id[result["entry_id"]] = result
    normalized_bias_probes = normalize_probe_bundle(
        read_json(bias_probe_path)
    )
    bias_units_by_arm = {
        key: probe["q90_bias_units_by_parameter"]
        for key, probe in normalized_bias_probes["probes_by_arm"].items()
    }
    entries = bias_capped_confirmation_entries(
        selection_payload["selections_by_arm"],
        result_by_id,
        bias_units_by_arm,
    )
    selection_sha256 = sha256_file(selections_path)
    entries = [
        {**entry, "parent_decision_sha256": selection_sha256}
        for entry in entries
    ]
    payload = bind_stage_entries(
        entries,
        main_manifest=load_main_manifest(main_manifest_path),
        probe_bundle=read_json(probes_path),
        stage="bias_capped_confirmation",
        stage_provenance={
            "selection_sha256": selection_sha256,
            "results_sha256": sha256_file(results_path),
            "main_manifest_sha256": sha256_file(main_manifest_path),
            "probe_bundle_sha256": sha256_file(probes_path),
            "bias_probe_bundle_sha256": sha256_file(bias_probe_path),
        },
    )
    publish_once(output_path, payload)
    return payload


def run_benchmark_child(
    *,
    study_config: str | Path,
    study_dir: str | Path,
    manifest_path: str | Path,
    reuse_root: str | Path,
    entry_id_value: str,
    data_root: str | Path,
    output_path: str | Path,
    device: str,
    ready_path: str | Path | None = None,
    start_path: str | Path | None = None,
    start_timeout_seconds: float = _BENCHMARK_READY_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    spec = verify_study_config(
        study_config=study_config, reuse_root=reuse_root
    )
    plan = load_execution_plan(manifest_path, reuse_root=reuse_root)
    entry = _find_entry(plan, entry_id_value)
    from experiments.mnist_conv.lr_optimizer_boundary_runtime import (
        execute_boundary_shadow_benchmark,
    )

    result = execute_boundary_shadow_benchmark(
        spec.data,
        study_dir,
        entry,
        data_root=data_root,
        device=device,
        warmup_steps=_BENCHMARK_WARMUP_STEPS,
        measured_steps=_BENCHMARK_MEASURED_STEPS,
        ready_path=ready_path,
        start_path=start_path,
        start_timeout_seconds=start_timeout_seconds,
    )
    publish_once(output_path, result)
    return result


def _gpu_sample() -> tuple[float, float]:
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    rows = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    if len(rows) != 1:
        raise RuntimeError(
            "Expected exactly one visible benchmark GPU. "
            f"Provided value: {rows!r}."
        )
    fields = [field.strip() for field in rows[0].split(",")]
    if len(fields) != 2:
        raise RuntimeError(
            f"Expected GPU memory/utilization fields. Provided value: {rows[0]!r}."
        )
    return float(fields[0]), float(fields[1])


def _gpu_identity() -> dict[str, Any]:
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,uuid",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    rows = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    if len(rows) != 1:
        raise RuntimeError(
            "Expected exactly one visible benchmark GPU. "
            f"Provided value: {rows!r}."
        )
    fields = [field.strip() for field in rows[0].split(",")]
    if len(fields) != 3:
        raise RuntimeError(
            "Expected GPU name, total memory, and UUID. "
            f"Provided value: {rows[0]!r}."
        )
    return {
        "name": fields[0],
        "total_memory_mib": float(fields[1]),
        "uuid": fields[2],
    }


def _benchmark_child_command(
    *,
    python: str,
    study_config: str | Path,
    study_dir: str | Path,
    manifest_path: str | Path,
    reuse_root: str | Path,
    entry_id_value: str,
    data_root: str | Path,
    output_path: str | Path,
    device: str,
    ready_path: str | Path,
    start_path: str | Path,
) -> list[str]:
    """Build the exact synchronized child command used by every width."""

    return [
        python,
        "-m",
        "experiments.run_mnist_conv_lr_optimizer_boundary",
        "benchmark-child",
        "--study-config",
        str(Path(study_config).expanduser().resolve()),
        "--study-dir",
        str(Path(study_dir).expanduser().resolve()),
        "--manifest",
        str(Path(manifest_path).expanduser().resolve()),
        "--reuse-root",
        str(Path(reuse_root).expanduser().resolve()),
        "--entry-id",
        entry_id_value,
        "--data-root",
        str(Path(data_root).expanduser().resolve()),
        "--output",
        str(Path(output_path).expanduser().resolve()),
        "--device",
        device,
        "--ready-path",
        str(Path(ready_path).expanduser().resolve()),
        "--start-path",
        str(Path(start_path).expanduser().resolve()),
        "--start-timeout-seconds",
        f"{_BENCHMARK_READY_TIMEOUT_SECONDS:g}",
    ]


def _benchmark_scientific_payload(
    *,
    spec: OptimizerBoundaryStudySpec,
    study_config_path: Path,
    study_dir: Path,
    manifest_path: Path,
    manifest: Mapping[str, Any],
    entry: Mapping[str, Any],
    levels: Sequence[Mapping[str, Any]],
    selection: Mapping[str, Any],
    device_identity: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": _BENCHMARK_SCHEMA_VERSION,
        "status": "complete",
        "study_id": spec.study_id,
        "study_config_sha256": sha256_file(study_config_path),
        "manifest_id": manifest["manifest_id"],
        "manifest_sha256": sha256_file(manifest_path),
        "entry_id": entry["entry_id"],
        "optimizer": "adam",
        "probe_sha256": entry["probe_sha256"],
        "checkpoint_sha256": entry["checkpoint_sha256"],
        "minibatch_sha256": entry["minibatch_sha256"],
        "code_fingerprint": entry["code_fingerprint"],
        "resource_contract": dict(_RESOURCE_CONTRACT),
        "device": dict(device_identity),
        "warmup_steps_per_child": _BENCHMARK_WARMUP_STEPS,
        "measured_steps_per_child": _BENCHMARK_MEASURED_STEPS,
        "barrier": {
            "kind": "per_width_ready_files_and_common_start_file",
            "ready_timeout_seconds": _BENCHMARK_READY_TIMEOUT_SECONDS,
        },
        "levels": [dict(level) for level in levels],
        "selection": dict(selection),
        "official_test_read": False,
    }


def load_benchmark_report(
    *,
    benchmark_path: str | Path,
    study_config: str | Path,
    execution_plan: Mapping[str, Any],
    execution_plan_path: str | Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate a V100 benchmark and return it plus its pack binding."""

    path = Path(benchmark_path).expanduser().resolve()
    report = read_json(path)
    if report.get("schema_version") != _BENCHMARK_SCHEMA_VERSION:
        raise _error(
            f"benchmark schema {_BENCHMARK_SCHEMA_VERSION!r}",
            report.get("schema_version"),
        )
    scientific = dict(report)
    observed_id = scientific.pop("benchmark_id", None)
    expected_id = "lroptbenchmark_" + sha256_json(scientific)
    if observed_id != expected_id:
        raise _error(expected_id, observed_id)
    if (
        report.get("status") != "complete"
        or report.get("official_test_read") is not False
    ):
        raise _error("a complete benchmark with official_test_read=false", report)
    study_sha = sha256_file(Path(study_config).expanduser().resolve())
    if report.get("study_config_sha256") != study_sha:
        raise _error(
            "benchmark study-config hash to match the execution",
            {
                "benchmark": report.get("study_config_sha256"),
                "execution": study_sha,
            },
        )
    if report.get("study_id") != execution_plan.get("study_id"):
        raise _error(
            "benchmark study id to match the execution plan",
            {
                "benchmark": report.get("study_id"),
                "plan": execution_plan.get("study_id"),
            },
        )
    expected_manifest_id = execution_plan.get(
        "manifest_id", execution_plan.get("main_manifest_id")
    )
    if report.get("manifest_id") != expected_manifest_id:
        raise _error(
            "benchmark main-manifest id to match the execution plan",
            {
                "benchmark": report.get("manifest_id"),
                "plan": expected_manifest_id,
            },
        )
    if execution_plan.get("manifest_id") is not None:
        # For the main grid the exact manifest bytes are directly available.
        # Conditional plans bind the same main manifest by content-addressed id.
        expected_manifest_sha = sha256_file(
            Path(execution_plan_path).expanduser().resolve()
        )
        if report.get("manifest_sha256") != expected_manifest_sha:
            raise _error(
                "benchmark main-manifest hash to match the execution",
                {
                    "benchmark": report.get("manifest_sha256"),
                    "execution": expected_manifest_sha,
                },
            )
    entry_fingerprints = {
        entry.get("code_fingerprint") for entry in execution_plan["entries"]
    }
    if entry_fingerprints and entry_fingerprints != {
        report.get("code_fingerprint")
    }:
        raise _error(
            "benchmark code fingerprint to match every execution entry",
            {
                "benchmark": report.get("code_fingerprint"),
                "entries": sorted(str(item) for item in entry_fingerprints),
            },
        )
    if report.get("resource_contract") != _RESOURCE_CONTRACT:
        raise _error("the frozen fmu@v100 resource contract", report.get("resource_contract"))
    device = report.get("device")
    if (
        not isinstance(device, Mapping)
        or "V100" not in str(device.get("name", ""))
        or float(device.get("total_memory_mib", 0.0)) != 32768.0
    ):
        raise _error("one benchmarked V100 with 32768 MiB", device)
    levels = report.get("levels")
    if (
        not isinstance(levels, list)
        or [level.get("concurrency") for level in levels]
        != list(_BENCHMARK_WIDTHS)
        or any(
            any(
                child.get("barrier_synchronized") is not True
                for child in level.get("children", [])
            )
            for level in levels
        )
    ):
        raise _error(
            "all six ordered widths with synchronized child reports", levels
        )
    selection = report.get("selection")
    selected = (
        selection.get("selected_concurrency")
        if isinstance(selection, Mapping)
        else None
    )
    if selected not in _BENCHMARK_WIDTHS:
        raise _error("a benchmark-selected supported concurrency", selected)
    binding = {
        "benchmark_id": report["benchmark_id"],
        "benchmark_sha256": sha256_file(path),
        "manifest_id": report["manifest_id"],
        "manifest_sha256": report["manifest_sha256"],
        "study_id": report["study_id"],
        "study_config_sha256": report["study_config_sha256"],
        "selected_concurrency": selected,
    }
    return report, binding


def run_concurrency_benchmark(
    *,
    study_config: str | Path,
    study_dir: str | Path,
    manifest_path: str | Path,
    reuse_root: str | Path,
    data_root: str | Path,
    output_path: str | Path,
    scratch_dir: str | Path,
    python: str,
    device: str,
) -> dict[str, Any]:
    """Measure the exact six widths on one V100 with an Adam entry."""

    spec = verify_study_config(
        study_config=study_config, reuse_root=reuse_root
    )
    manifest = load_main_manifest(manifest_path, reuse_root=reuse_root)
    if manifest["study_id"] != spec.study_id:
        raise _error(
            "benchmark manifest and study config to have the same study id",
            {"manifest": manifest["study_id"], "study": spec.study_id},
        )
    current_fingerprint = code_provenance()["effective_code_fingerprint"]
    if manifest["code_fingerprint"] != current_fingerprint:
        raise _error(
            "benchmark code fingerprint to match the probe-bound manifest",
            {
                "manifest": manifest["code_fingerprint"],
                "current": current_fingerprint,
            },
        )
    matches = [
        entry
        for entry in manifest["entries"]
        if entry["scheme"] == "ours"
        and entry["optimizer"] == "adam"
        and entry["mode"] == "new_five_epoch_training"
        and entry["rho_conv"] == 0.01
        and entry["rho_dense"] == 0.003
    ]
    if len(matches) != 1:
        raise _error("one ours/Adam lower-corner benchmark entry", matches)
    entry = matches[0]
    root = Path(scratch_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=False)
    device_identity = _gpu_identity()
    levels = []
    for concurrency in _BENCHMARK_WIDTHS:
        level = root / f"concurrency-{concurrency:02d}"
        level.mkdir()
        start_path = level / "start.json"
        active = []
        launched = time.perf_counter()
        readiness_started = time.perf_counter()
        measurement_wall_seconds = 0.0
        readiness_failure: str | None = None
        peak_memory = 0.0
        utilization_samples: list[float] = []
        try:
            for child_index in range(concurrency):
                output = level / f"child-{child_index:02d}.json"
                ready = level / f"child-{child_index:02d}.ready"
                log = (level / f"child-{child_index:02d}.log").open(
                    "w", encoding="utf-8"
                )
                command = _benchmark_child_command(
                    python=python,
                    study_config=study_config,
                    study_dir=study_dir,
                    manifest_path=manifest_path,
                    reuse_root=reuse_root,
                    entry_id_value=entry["entry_id"],
                    data_root=data_root,
                    output_path=output,
                    device=device,
                    ready_path=ready,
                    start_path=start_path,
                )
                process = subprocess.Popen(
                    command,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
                active.append((process, log, output, ready, command))
            while True:
                if all(ready.is_file() for _process, _log, _output, ready, _command in active):
                    break
                early = [
                    process.returncode
                    for process, _log, _output, ready, _command in active
                    if not ready.is_file() and process.poll() is not None
                ]
                if early:
                    readiness_failure = (
                        "child exited before publishing its ready file: "
                        f"{early!r}"
                    )
                    break
                if (
                    time.perf_counter() - readiness_started
                    > _BENCHMARK_READY_TIMEOUT_SECONDS
                ):
                    readiness_failure = (
                        "children did not all reach the start barrier within "
                        f"{_BENCHMARK_READY_TIMEOUT_SECONDS:g} seconds"
                    )
                    break
                memory, utilization = _gpu_sample()
                peak_memory = max(peak_memory, memory)
                utilization_samples.append(utilization)
                time.sleep(0.25)
            if readiness_failure is None:
                publish_once(
                    start_path,
                    {
                        "schema_version": (
                            "mnist-conv-lr-optimizer-boundary-start-barrier/v1"
                        ),
                        "concurrency": concurrency,
                        "entry_id": entry["entry_id"],
                    },
                )
                measurement_started = time.perf_counter()
            else:
                # Release children already waiting so cleanup never leaves a
                # process blocked on the barrier timeout.
                publish_once(
                    start_path,
                    {
                        "schema_version": (
                            "mnist-conv-lr-optimizer-boundary-start-barrier/v1"
                        ),
                        "concurrency": concurrency,
                        "entry_id": entry["entry_id"],
                        "aborted_before_measurement": True,
                    },
                )
                measurement_started = None
            while any(
                process.poll() is None
                for process, _log, _output, _ready, _command in active
            ):
                memory, utilization = _gpu_sample()
                peak_memory = max(peak_memory, memory)
                utilization_samples.append(utilization)
                time.sleep(0.25)
            if measurement_started is not None:
                measurement_wall_seconds = (
                    time.perf_counter() - measurement_started
                )
            failures = []
            children = []
            for child_index, (process, log, output, ready, command) in enumerate(active):
                returncode = process.wait()
                log.close()
                if returncode != 0 or not output.is_file():
                    failures.append(
                        {
                            "child_index": child_index,
                            "returncode": returncode,
                            "ready": ready.is_file(),
                            "command": command,
                        }
                    )
                else:
                    child = read_json(output)
                    if (
                        child.get("barrier_synchronized") is not True
                        or child.get("measured_successful_steps")
                        != _BENCHMARK_MEASURED_STEPS
                        or child.get("entry_id") != entry["entry_id"]
                        or child.get("probe_sha256") != entry["probe_sha256"]
                        or child.get("checkpoint_sha256")
                        != entry["checkpoint_sha256"]
                    ):
                        failures.append(
                            {
                                "child_index": child_index,
                                "returncode": returncode,
                                "ready": ready.is_file(),
                                "reason": "invalid synchronized child report",
                                "command": command,
                            }
                        )
                    else:
                        children.append(child)
            wall_seconds = time.perf_counter() - launched
            if readiness_failure is not None:
                failures.append(
                    {
                        "child_index": None,
                        "returncode": None,
                        "reason": readiness_failure,
                    }
                )
            measured_steps_total = sum(
                int(child["measured_successful_steps"]) for child in children
            )
            aggregate_throughput = (
                measured_steps_total / measurement_wall_seconds
                if measurement_wall_seconds > 0.0
                else 0.0
            )
            levels.append(
                {
                    "concurrency": concurrency,
                    "failed_children": max(
                        len(failures), concurrency - len(children)
                    ),
                    "completed_children": len(children),
                    "all_children_ready": readiness_failure is None,
                    "ready_wait_seconds": (
                        (measurement_started or time.perf_counter())
                        - readiness_started
                    ),
                    "peak_gpu_memory_mib": peak_memory,
                    "mean_gpu_utilization_percent": (
                        sum(utilization_samples) / len(utilization_samples)
                        if utilization_samples
                        else 0.0
                    ),
                    "aggregate_steps_per_second": aggregate_throughput,
                    "measured_steps_total": measured_steps_total,
                    "measurement_wall_seconds": measurement_wall_seconds,
                    "wall_seconds": wall_seconds,
                    "child_failures": failures,
                    "children": children,
                }
            )
        finally:
            for process, log, _output, _ready, _command in active:
                if process.poll() is None:
                    process.terminate()
                    process.wait()
                if not log.closed:
                    log.close()
    selection = select_concurrency(levels)
    scientific = _benchmark_scientific_payload(
        spec=spec,
        study_config_path=Path(study_config).expanduser().resolve(),
        study_dir=Path(study_dir).expanduser().resolve(),
        manifest_path=Path(manifest_path).expanduser().resolve(),
        manifest=manifest,
        entry=entry,
        levels=levels,
        selection=selection,
        device_identity=device_identity,
    )
    report = {
        **scientific,
        "benchmark_id": "lroptbenchmark_" + sha256_json(scientific),
    }
    publish_once(output_path, report)
    return report


def _run_entry_command(
    *,
    python: str,
    study_config: str | Path,
    study_dir: str | Path,
    manifest_path: str | Path,
    reuse_root: str | Path,
    entry_id_value: str,
    data_root: str | Path,
    device: str,
) -> list[str]:
    return [
        python,
        "-m",
        "experiments.run_mnist_conv_lr_optimizer_boundary",
        "run-entry",
        "--study-config",
        str(Path(study_config).expanduser().resolve()),
        "--study-dir",
        str(Path(study_dir).expanduser().resolve()),
        "--manifest",
        str(Path(manifest_path).expanduser().resolve()),
        "--reuse-root",
        str(Path(reuse_root).expanduser().resolve()),
        "--entry-id",
        entry_id_value,
        "--data-root",
        str(Path(data_root).expanduser().resolve()),
        "--device",
        device,
    ]


def run_pack(
    *,
    study_config: str | Path,
    study_dir: str | Path,
    manifest_path: str | Path,
    benchmark_path: str | Path,
    reuse_root: str | Path,
    data_root: str | Path,
    scheme: str,
    stage: str,
    concurrency: int,
    python: str,
    device: str,
    log_dir: str | Path,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    manifest = load_execution_plan(manifest_path, reuse_root=reuse_root)
    current_fingerprint = code_provenance()["effective_code_fingerprint"]
    manifest_fingerprints = {
        entry["code_fingerprint"] for entry in manifest["entries"]
    }
    if manifest_fingerprints != {current_fingerprint}:
        raise _error(
            "current code fingerprint to match every pack entry",
            {
                "current": current_fingerprint,
                "entries": sorted(manifest_fingerprints),
            },
        )
    _report, benchmark_binding = load_benchmark_report(
        benchmark_path=benchmark_path,
        study_config=study_config,
        execution_plan=manifest,
        execution_plan_path=manifest_path,
    )
    if concurrency != benchmark_binding["selected_concurrency"]:
        raise _error(
            "pack concurrency to equal the benchmark-selected width",
            {
                "pack": concurrency,
                "benchmark": benchmark_binding["selected_concurrency"],
            },
        )
    pack = pack_entries(
        manifest["entries"],
        scheme=scheme,
        concurrency=concurrency,
        stage=stage,
        benchmark_binding=benchmark_binding,
    )
    logs = Path(log_dir).expanduser().resolve()
    logs.mkdir(parents=True, exist_ok=True)
    executions = []
    for wave in pack["waves"]:
        active = []
        try:
            for entry_id_value in wave["entry_ids"]:
                command = _run_entry_command(
                    python=python,
                    study_config=study_config,
                    study_dir=study_dir,
                    manifest_path=manifest_path,
                    reuse_root=reuse_root,
                    entry_id_value=entry_id_value,
                    data_root=data_root,
                    device=device,
                )
                stream = (logs / f"{wave['wave_index']:02d}-{entry_id_value}.log").open(
                    "w", encoding="utf-8"
                )
                process = subprocess.Popen(
                    command,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
                active.append((entry_id_value, command, process, stream))
            failures = []
            for entry_id_value, command, process, stream in active:
                returncode = process.wait()
                stream.close()
                execution = {
                    "wave_index": wave["wave_index"],
                    "entry_id": entry_id_value,
                    "returncode": returncode,
                    "command": command,
                }
                executions.append(execution)
                if returncode != 0:
                    failures.append(execution)
            if failures:
                raise RuntimeError(
                    "Expected every optimizer-boundary child to complete before "
                    f"the next wave. Provided failures: {failures!r}."
                )
        finally:
            for _entry_id, _command, process, stream in active:
                if process.poll() is None:
                    process.terminate()
                    process.wait()
                if not stream.closed:
                    stream.close()
    result = {
        "schema_version": "mnist-conv-lr-optimizer-boundary-pack-execution/v1",
        "status": "complete",
        "pack_id": pack["pack_id"],
        "benchmark_id": benchmark_binding["benchmark_id"],
        "benchmark_sha256": benchmark_binding["benchmark_sha256"],
        "selected_concurrency": benchmark_binding["selected_concurrency"],
        "scheme": scheme,
        "stage": stage,
        "entry_count": len(executions),
        "executions": executions,
        "official_test_read": False,
    }
    if output_path is not None:
        publish_once(output_path, result)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    create = subparsers.add_parser("create-study")
    create.add_argument("--reuse-root", required=True)
    create.add_argument("--output", required=True)

    verify = subparsers.add_parser("verify-study")
    verify.add_argument("--study-config", required=True)
    verify.add_argument("--reuse-root", required=True)

    probes = subparsers.add_parser("prepare-probes")
    probes.add_argument("--study-config", required=True)
    probes.add_argument("--study-dir", required=True)
    probes.add_argument("--reuse-root", required=True)
    probes.add_argument("--data-root", required=True)
    probes.add_argument("--initialization-source", required=True)
    probes.add_argument("--output", required=True)
    probes.add_argument("--device", default="cuda")

    manifest = subparsers.add_parser("create-main-manifest")
    manifest.add_argument("--study-config", required=True)
    manifest.add_argument("--reuse-root", required=True)
    manifest.add_argument("--probes", required=True)
    manifest.add_argument("--output", required=True)

    entry = subparsers.add_parser("run-entry")
    entry.add_argument("--study-config", required=True)
    entry.add_argument("--study-dir", required=True)
    entry.add_argument("--manifest", required=True)
    entry.add_argument("--reuse-root", required=True)
    entry.add_argument("--entry-id", required=True)
    entry.add_argument("--data-root", required=True)
    entry.add_argument("--device", default="cuda")

    benchmark_child = subparsers.add_parser("benchmark-child")
    benchmark_child.add_argument("--study-config", required=True)
    benchmark_child.add_argument("--study-dir", required=True)
    benchmark_child.add_argument("--manifest", required=True)
    benchmark_child.add_argument("--reuse-root", required=True)
    benchmark_child.add_argument("--entry-id", required=True)
    benchmark_child.add_argument("--data-root", required=True)
    benchmark_child.add_argument("--output", required=True)
    benchmark_child.add_argument("--device", default="cuda")
    benchmark_child.add_argument("--ready-path")
    benchmark_child.add_argument("--start-path")
    benchmark_child.add_argument(
        "--start-timeout-seconds",
        type=float,
        default=_BENCHMARK_READY_TIMEOUT_SECONDS,
    )

    benchmark = subparsers.add_parser("benchmark")
    benchmark.add_argument("--study-config", required=True)
    benchmark.add_argument("--study-dir", required=True)
    benchmark.add_argument("--manifest", required=True)
    benchmark.add_argument("--reuse-root", required=True)
    benchmark.add_argument("--data-root", required=True)
    benchmark.add_argument("--output", required=True)
    benchmark.add_argument("--scratch-dir", required=True)
    benchmark.add_argument("--python", default=sys.executable)
    benchmark.add_argument("--device", default="cuda")

    pack = subparsers.add_parser("run-pack")
    pack.add_argument("--study-config", required=True)
    pack.add_argument("--study-dir", required=True)
    pack.add_argument("--manifest", required=True)
    pack.add_argument("--benchmark", required=True)
    pack.add_argument("--reuse-root", required=True)
    pack.add_argument("--data-root", required=True)
    pack.add_argument("--scheme", choices=SCHEMES, required=True)
    pack.add_argument(
        "--stage",
        choices=("main_grid", "upper_sentinel", "bias_capped_confirmation"),
        default="main_grid",
    )
    pack.add_argument("--concurrency", type=int, required=True)
    pack.add_argument("--python", default=sys.executable)
    pack.add_argument("--device", default="cuda")
    pack.add_argument("--log-dir", required=True)
    pack.add_argument("--output")

    select = subparsers.add_parser("select")
    select.add_argument("--results", required=True)
    select.add_argument("--main-manifest", required=True)
    select.add_argument("--probes", required=True)
    select.add_argument("--output", required=True)
    select.add_argument("--sentinel-output", required=True)

    confirmations = subparsers.add_parser("create-confirmations")
    confirmations.add_argument("--selections", required=True)
    confirmations.add_argument("--results", required=True)
    confirmations.add_argument("--bias-probes", required=True)
    confirmations.add_argument("--main-manifest", required=True)
    confirmations.add_argument("--probes", required=True)
    confirmations.add_argument("--output", required=True)

    reuse = subparsers.add_parser("materialize-reuse")
    reuse.add_argument("--study-config", required=True)
    reuse.add_argument("--study-dir", required=True)
    reuse.add_argument("--manifest", required=True)
    reuse.add_argument("--reuse-root", required=True)
    reuse.add_argument("--data-root", required=True)
    reuse.add_argument("--device", default="cpu")

    aggregate = subparsers.add_parser("aggregate")
    aggregate.add_argument("--study-dir", required=True)
    aggregate.add_argument("--plan", action="append", required=True)
    aggregate.add_argument("--output", required=True)

    finalize = subparsers.add_parser("finalize")
    finalize.add_argument("--results", required=True)
    finalize.add_argument("--selections", required=True)
    finalize.add_argument("--output-dir", required=True)

    plot = subparsers.add_parser("plot")
    plot.add_argument("--results", required=True)
    plot.add_argument("--output-dir", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    if arguments.command == "create-study":
        spec, path = create_study_config(
            reuse_root=arguments.reuse_root, output_path=arguments.output
        )
        result = {"study_id": spec.study_id, "path": str(path)}
    elif arguments.command == "verify-study":
        spec = verify_study_config(
            study_config=arguments.study_config,
            reuse_root=arguments.reuse_root,
        )
        result = {"status": "verified", "study_id": spec.study_id}
    elif arguments.command == "prepare-probes":
        bundle = prepare_optimizer_probes(
            study_config=arguments.study_config,
            study_dir=arguments.study_dir,
            reuse_root=arguments.reuse_root,
            data_root=arguments.data_root,
            initialization_source=arguments.initialization_source,
            output_path=arguments.output,
            device=arguments.device,
        )
        result = {
            "status": "complete",
            "study_id": bundle["study_id"],
            "probe_count": len(bundle["probes_by_arm"]),
            "adam_probe_count": sum(
                key.endswith("--adam") for key in bundle["probes_by_arm"]
            ),
        }
    elif arguments.command == "create-main-manifest":
        manifest = create_main_manifest(
            study_config=arguments.study_config,
            reuse_root=arguments.reuse_root,
            probes_path=arguments.probes,
            output_path=arguments.output,
        )
        result = {
            "status": "complete",
            "manifest_id": manifest["manifest_id"],
        }
    elif arguments.command == "run-entry":
        result = run_entry(
            study_config=arguments.study_config,
            study_dir=arguments.study_dir,
            manifest_path=arguments.manifest,
            reuse_root=arguments.reuse_root,
            entry_id_value=arguments.entry_id,
            data_root=arguments.data_root,
            device=arguments.device,
        )
    elif arguments.command == "benchmark-child":
        result = run_benchmark_child(
            study_config=arguments.study_config,
            study_dir=arguments.study_dir,
            manifest_path=arguments.manifest,
            reuse_root=arguments.reuse_root,
            entry_id_value=arguments.entry_id,
            data_root=arguments.data_root,
            output_path=arguments.output,
            device=arguments.device,
            ready_path=arguments.ready_path,
            start_path=arguments.start_path,
            start_timeout_seconds=arguments.start_timeout_seconds,
        )
    elif arguments.command == "benchmark":
        result = run_concurrency_benchmark(
            study_config=arguments.study_config,
            study_dir=arguments.study_dir,
            manifest_path=arguments.manifest,
            reuse_root=arguments.reuse_root,
            data_root=arguments.data_root,
            output_path=arguments.output,
            scratch_dir=arguments.scratch_dir,
            python=arguments.python,
            device=arguments.device,
        )
    elif arguments.command == "run-pack":
        result = run_pack(
            study_config=arguments.study_config,
            study_dir=arguments.study_dir,
            manifest_path=arguments.manifest,
            benchmark_path=arguments.benchmark,
            reuse_root=arguments.reuse_root,
            data_root=arguments.data_root,
            scheme=arguments.scheme,
            stage=arguments.stage,
            concurrency=arguments.concurrency,
            python=arguments.python,
            device=arguments.device,
            log_dir=arguments.log_dir,
            output_path=arguments.output,
        )
    elif arguments.command == "select":
        result = select_and_plan_sentinels(
            results_path=arguments.results,
            main_manifest_path=arguments.main_manifest,
            probes_path=arguments.probes,
            selection_path=arguments.output,
            sentinel_path=arguments.sentinel_output,
        )
    elif arguments.command == "create-confirmations":
        result = create_confirmations(
            selections_path=arguments.selections,
            results_path=arguments.results,
            bias_probe_path=arguments.bias_probes,
            main_manifest_path=arguments.main_manifest,
            probes_path=arguments.probes,
            output_path=arguments.output,
        )
    elif arguments.command == "materialize-reuse":
        result = materialize_reuse_entries(
            study_config=arguments.study_config,
            study_dir=arguments.study_dir,
            manifest_path=arguments.manifest,
            reuse_root=arguments.reuse_root,
            data_root=arguments.data_root,
            device=arguments.device,
        )
    elif arguments.command == "aggregate":
        result = aggregate_results(
            study_dir=arguments.study_dir,
            plan_paths=arguments.plan,
            output_path=arguments.output,
        )
    elif arguments.command == "finalize":
        result = finalize_boundary_study(
            results_path=arguments.results,
            selections_path=arguments.selections,
            output_dir=arguments.output_dir,
        )
    elif arguments.command == "plot":
        outputs = render_boundary_report(
            _load_results(arguments.results), arguments.output_dir
        )
        result = {
            "status": "complete",
            "outputs": [str(path) for path in outputs],
        }
    else:
        raise AssertionError(arguments.command)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
