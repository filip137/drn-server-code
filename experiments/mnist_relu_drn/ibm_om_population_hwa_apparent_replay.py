"""Read-only apparent-state replay of positive-G IBM OM DRN deployments.

The source experiment programmed and saved both the hidden persistent state and
the held apparent post-write state for every physical cell, but evaluated the
DRN with the persistent state.  This diagnostic keeps every source artifact,
assignment, endpoint seed, target, and programming trajectory fixed.  It
changes only the state presented to the forward pass.

Literal apparent raw-``x`` values can leave ``[0, 1]`` because AIHWKit write
noise is added after the bounded persistent update.  A passive DRN cannot
consume negative conductances, so the network diagnostic explicitly projects
the held apparent state into ``x=[0, 1]`` before applying ``G=2*x``.  Literal
out-of-range values and the projection displacement remain reported.  No
device is programmed, resampled, or updated by this module.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
from pathlib import Path
from statistics import fmean, pstdev
import sys
from typing import Any, Iterable, Mapping, Sequence

import torch

from experiments.artifacts import RunStore, atomic_write_json, sha256_file
from experiments.mnist_relu_drn.components import build_student_stack
from experiments.mnist_relu_drn.ibm_om_bounded_codebook_scheme_screen import (
    _tensor_sha256,
)
from experiments.mnist_relu_drn.ibm_om_population_hwa_cross_array_config import (
    EXPERIMENT_ID as SOURCE_EXPERIMENT_ID,
    PopulationHwaCrossArrayTrainSpec,
    parse_population_hwa_cross_array_config,
    resolve_population_hwa_cross_array_spec,
)
from experiments.mnist_relu_drn.ibm_om_population_hwa_cross_array_runtime import (
    DEPLOYMENT_SCHEMA,
    DEPLOYMENT_SCHEMA_VERSION,
    _apply_full_g,
    _evaluate_detailed,
)
from experiments.mnist_relu_drn.ibm_om_winsorized_onchip_adam import (
    split_flat_physical,
)
from experiments.mnist_relu_drn.runtime import _load_teacher
from experiments.mnist_shared import build_mnist_loaders
from experiments.schema import RunMode, to_plain_data


_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_ID = "mnist_ibm_om_population_hwa_apparent_replay.v1"
SUMMARY_SCHEMA = "ebl.mnist_relu_drn.ibm_om_population_hwa_apparent_replay"
SUMMARY_SCHEMA_VERSION = 1
EVIDENCE_TIER = "exploratory_noncanonical"
STATE_LABELS = (
    "raw_relu",
    "population_hwa_continuous",
    "population_hwa_one_delta_qat",
)
POPULATION_ROLES = ("counterfactual_repaired", "published_corrupt")
FORWARD_POLICY = "held_apparent_raw_x_projected_to_0_1_then_G_equals_2x"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--teacher-weights", type=Path, required=True)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--sample-limit",
        type=int,
        default=None,
        help="Optional reduced test-example count for an exploratory smoke.",
    )
    parser.add_argument(
        "--maximum-endpoints",
        type=int,
        default=None,
        help="Optional prefix length for an exploratory smoke.",
    )
    return parser


def _require_file(path: Path, *, label: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(
            f"Expected {label} to name an existing file: {str(resolved)!r}."
        )
    return resolved


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    source = _require_file(path, label=label)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Expected a readable {label} JSON object.") from error
    if not isinstance(payload, dict):
        raise ValueError(f"Expected {label} to contain one JSON object.")
    return payload


def _strict_torch_load(path: Path) -> Mapping[str, Any]:
    source = _require_file(path, label="deployment artifact")
    try:
        payload = torch.load(source, map_location="cpu", weights_only=True)
    except TypeError:  # pragma: no cover - older supported PyTorch
        payload = torch.load(source, map_location="cpu")
    if not isinstance(payload, Mapping):
        raise ValueError("Expected deployment artifact to contain one mapping.")
    return payload


def _input(role: str, path: Path) -> dict[str, Any]:
    source = _require_file(path, label=role)
    return {"role": role, "path": str(source), "sha256": sha256_file(source)}


def _finite_float32_vector(value: Any, *, name: str, size: int) -> torch.Tensor:
    if (
        not isinstance(value, torch.Tensor)
        or value.dtype != torch.float32
        or tuple(value.shape) != (size,)
        or not bool(torch.all(torch.isfinite(value)))
    ):
        raise ValueError(f"Expected {name} to be a finite float32 vector of size {size}.")
    return value.detach().cpu().contiguous()


def _bool_vector(value: Any, *, name: str, size: int) -> torch.Tensor:
    if (
        not isinstance(value, torch.Tensor)
        or value.dtype != torch.bool
        or tuple(value.shape) != (size,)
    ):
        raise ValueError(f"Expected {name} to be a bool vector of size {size}.")
    return value.detach().cpu().contiguous()


def _tensor_summary(value: torch.Tensor) -> dict[str, float | int]:
    flat = value.detach().to(device="cpu", dtype=torch.float64).reshape(-1)
    if flat.numel() == 0 or not bool(torch.all(torch.isfinite(flat))):
        raise ValueError("Expected a non-empty finite tensor for a summary.")
    return {
        "count": int(flat.numel()),
        "minimum": float(flat.min().item()),
        "maximum": float(flat.max().item()),
        "mean": float(flat.mean().item()),
        "population_std": float(flat.std(unbiased=False).item()),
        "rms": float(flat.square().mean().sqrt().item()),
    }


def prepare_apparent_endpoint(
    payload: Mapping[str, Any],
    *,
    binding_shapes: Sequence[Sequence[int]],
) -> tuple[torch.Tensor, torch.Tensor, Mapping[str, Any]]:
    """Validate one saved endpoint and return persistent/apparent positive ``G``.

    The returned apparent state is the deterministic passivity projection
    ``G=2*clamp((a_apparent+1)/2, 0, 1)``.  The saved payload is never mutated.
    """

    if (
        payload.get("schema") != DEPLOYMENT_SCHEMA
        or payload.get("schema_version") != DEPLOYMENT_SCHEMA_VERSION
        or payload.get("evidence_tier") != EVIDENCE_TIER
        or payload.get("persistent_endpoint_applied_to_drn") is not True
        or payload.get("apparent_endpoint_applied_to_drn") is not False
        or payload.get("inference_read_noise") is not False
    ):
        raise ValueError("Saved deployment semantic contract mismatch.")
    size = sum(math.prod(tuple(int(item) for item in shape)) for shape in binding_shapes)
    target = _finite_float32_vector(payload.get("target_raw_x"), name="target_raw_x", size=size)
    continuation = payload.get("continuation_state")
    programming = payload.get("programming")
    if not isinstance(continuation, Mapping) or not isinstance(programming, Mapping):
        raise ValueError("Expected saved continuation and programming mappings.")
    if continuation.get("state_coordinate") != "native_raw_active_a":
        raise ValueError("Expected native raw-active continuation coordinates.")
    persistent_a = _finite_float32_vector(
        continuation.get("persistent"), name="persistent raw a", size=size
    )
    apparent_a = _finite_float32_vector(
        continuation.get("apparent"), name="apparent raw a", size=size
    )
    accepted = _bool_vector(programming.get("accepted"), name="accepted", size=size)

    persistent_x = (persistent_a + 1.0) / 2.0
    if bool(torch.any(persistent_x < 0.0)) or bool(torch.any(persistent_x > 1.0)):
        raise ValueError("Saved persistent raw x left the declared [0,1] range.")
    apparent_x = (apparent_a + 1.0) / 2.0
    projected_x = apparent_x.clamp(0.0, 1.0)
    below = apparent_x < 0.0
    above = apparent_x > 1.0
    out_of_range = below | above
    projection = projected_x - apparent_x
    apparent_minus_persistent = apparent_x - persistent_x
    target_residual = apparent_x - target

    layer_reports: dict[str, Any] = {}
    offset = 0
    for index, shape_value in enumerate(binding_shapes):
        shape = tuple(int(item) for item in shape_value)
        count = math.prod(shape)
        section = slice(offset, offset + count)
        layer_oob = out_of_range[section]
        layer_reports[str(index)] = {
            "shape": list(shape),
            "cells": count,
            "below_zero": int(below[section].sum().item()),
            "above_one": int(above[section].sum().item()),
            "out_of_range": int(layer_oob.sum().item()),
            "out_of_range_fraction": float(layer_oob.to(torch.float64).mean().item()),
            "apparent_minus_persistent_raw_x": _tensor_summary(
                apparent_minus_persistent[section]
            ),
            "projection_delta_raw_x": _tensor_summary(projection[section]),
        }
        offset += count
    if offset != size:
        raise RuntimeError("Binding-shape accounting did not cover the endpoint.")

    report = {
        "policy": FORWARD_POLICY,
        "cells": size,
        "literal_apparent_raw_x": _tensor_summary(apparent_x),
        "persistent_raw_x": _tensor_summary(persistent_x),
        "apparent_minus_persistent_raw_x": _tensor_summary(
            apparent_minus_persistent
        ),
        "literal_apparent_minus_target_raw_x": _tensor_summary(target_residual),
        "below_zero": int(below.sum().item()),
        "above_one": int(above.sum().item()),
        "out_of_range": int(out_of_range.sum().item()),
        "out_of_range_fraction": float(out_of_range.to(torch.float64).mean().item()),
        "accepted_cells": int(accepted.sum().item()),
        "accepted_out_of_range": int((accepted & out_of_range).sum().item()),
        "projection_delta_raw_x": _tensor_summary(projection),
        "literal_apparent_raw_x_sha256": _tensor_sha256(apparent_x),
        "projected_apparent_raw_x_sha256": _tensor_sha256(projected_x),
        "layers": layer_reports,
        "literal_apparent_used_as_unbounded_network_conductance": False,
        "projection_is_diagnostic_not_additional_programming": True,
    }
    return 2.0 * persistent_x, 2.0 * projected_x, report


def _expected_endpoint_ids(spec: PopulationHwaCrossArrayTrainSpec) -> set[tuple[Any, ...]]:
    return {
        (array.label, state, population, int(endpoint_seed))
        for array in spec.protocol.arrays
        for state in STATE_LABELS
        for population in POPULATION_ROLES
        for endpoint_seed in array.endpoint_seeds
    }


def _endpoint_id(payload: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        payload.get("array_label"),
        payload.get("state_label"),
        payload.get("population_role"),
        payload.get("endpoint_seed"),
    )


def _repeat_summary(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not records:
        raise ValueError("Expected at least one endpoint record.")
    apparent = tuple(
        float(item["apparent_projected_test"]["student_accuracy"]) for item in records
    )
    persistent = tuple(
        float(item["persistent_comparison_test"]["student_accuracy"])
        for item in records
    )
    return {
        "repeats": len(records),
        "apparent_projected_mean_accuracy": fmean(apparent),
        "apparent_projected_minimum_accuracy": min(apparent),
        "apparent_projected_maximum_accuracy": max(apparent),
        "apparent_projected_population_std_accuracy": pstdev(apparent),
        "persistent_comparison_mean_accuracy": fmean(persistent),
        "apparent_projected_minus_persistent_mean_accuracy": (
            fmean(apparent) - fmean(persistent)
        ),
        "cells": sum(int(item["state_diagnostics"]["cells"]) for item in records),
        "literal_apparent_out_of_range": sum(
            int(item["state_diagnostics"]["out_of_range"]) for item in records
        ),
        "accepted_literal_apparent_out_of_range": sum(
            int(item["state_diagnostics"]["accepted_out_of_range"])
            for item in records
        ),
    }


def summarize_endpoints(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate complete endpoint reports without mixing intervention axes."""

    selected = tuple(records)
    if not selected:
        raise ValueError("Expected endpoint reports to summarize.")
    arrays: dict[str, Any] = {}
    for array_label in sorted({str(item["array_label"]) for item in selected}):
        array_records = tuple(
            item for item in selected if item["array_label"] == array_label
        )
        states: dict[str, Any] = {}
        for state_label in STATE_LABELS:
            populations: dict[str, Any] = {}
            for population in POPULATION_ROLES:
                group = tuple(
                    item
                    for item in array_records
                    if item["state_label"] == state_label
                    and item["population_role"] == population
                )
                if group:
                    populations[population] = _repeat_summary(group)
            if populations:
                state_summary: dict[str, Any] = {"populations": populations}
                if set(populations) == set(POPULATION_ROLES):
                    state_summary["published_corruption_penalty_accuracy"] = (
                        populations["counterfactual_repaired"]
                        ["apparent_projected_mean_accuracy"]
                        - populations["published_corrupt"]
                        ["apparent_projected_mean_accuracy"]
                    )
                states[state_label] = state_summary
        arrays[array_label] = {"states": states}

    transfer_records = tuple(
        item for item in selected if item["array_label"] in {"B", "C", "D"}
    )
    transfer: dict[str, Any] = {}
    for state_label in STATE_LABELS:
        state_summary: dict[str, Any] = {}
        for population in POPULATION_ROLES:
            group = tuple(
                item
                for item in transfer_records
                if item["state_label"] == state_label
                and item["population_role"] == population
            )
            if group:
                state_summary[population] = _repeat_summary(group)
        if state_summary:
            if set(state_summary) == set(POPULATION_ROLES):
                state_summary["published_corruption_penalty_accuracy"] = (
                    state_summary["counterfactual_repaired"]
                    ["apparent_projected_mean_accuracy"]
                    - state_summary["published_corrupt"]
                    ["apparent_projected_mean_accuracy"]
                )
            transfer[state_label] = state_summary

    for array_payload in arrays.values():
        states = array_payload["states"]
        for population in POPULATION_ROLES:
            if all(
                state in states and population in states[state]["populations"]
                for state in STATE_LABELS
            ):
                raw = states["raw_relu"]["populations"][population][
                    "apparent_projected_mean_accuracy"
                ]
                continuous = states["population_hwa_continuous"]["populations"][
                    population
                ]["apparent_projected_mean_accuracy"]
                quantized = states["population_hwa_one_delta_qat"]["populations"][
                    population
                ]["apparent_projected_mean_accuracy"]
                array_payload.setdefault("comparisons", {})[population] = {
                    "continuous_hwa_minus_raw_relu": continuous - raw,
                    "one_delta_qat_hwa_minus_raw_relu": quantized - raw,
                    "one_delta_qat_minus_continuous_hwa": quantized - continuous,
                }

    return {"arrays": arrays, "transfer_B_D": transfer}


def _source_bundle(
    source_run: Path,
    *,
    config_path: Path,
) -> tuple[
    Path,
    Mapping[str, Any],
    PopulationHwaCrossArrayTrainSpec,
    tuple[Mapping[str, Any], ...],
]:
    run_dir = source_run.expanduser().resolve()
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Expected --source-run directory: {str(run_dir)!r}.")
    manifest = _read_json(run_dir / "manifest.json", label="source manifest")
    status = _read_json(run_dir / "status.json", label="source status")
    result = _read_json(run_dir / "result.json", label="source result")
    resolved = _read_json(
        run_dir / "config.resolved.json", label="source resolved config"
    )
    if (
        manifest.get("experiment_id") != SOURCE_EXPERIMENT_ID
        or result.get("experiment_id") != SOURCE_EXPERIMENT_ID
        or status.get("status") != "complete"
        or result.get("status") != "complete"
    ):
        raise ValueError("Expected one completed population-HWA source run.")
    document = parse_population_hwa_cross_array_config(
        _read_json(config_path, label="source config")
    )
    spec = resolve_population_hwa_cross_array_spec(document, RunMode.TRAIN)
    if resolved != to_plain_data(spec):
        raise ValueError("Source resolved config differs from the supplied config.")
    artifacts = result.get("artifacts")
    if not isinstance(artifacts, list):
        raise ValueError("Expected source result artifact records.")
    deployments = tuple(
        item
        for item in artifacts
        if isinstance(item, Mapping)
        and isinstance(item.get("kind"), str)
        and str(item["kind"]).endswith("_deployment")
        and isinstance(item.get("path"), str)
        and str(item["path"]).endswith(".pt")
    )
    expected_count = len(_expected_endpoint_ids(spec))
    if len(deployments) != expected_count:
        raise ValueError(
            "Expected complete source deployment coverage: "
            f"expected={expected_count}, observed={len(deployments)}."
        )
    if len({str(item["path"]) for item in deployments}) != len(deployments):
        raise ValueError("Source result contains duplicate deployment paths.")
    return run_dir, manifest, spec, deployments


def _artifact_path(run_dir: Path, record: Mapping[str, Any]) -> Path:
    relative = Path(str(record["path"]))
    if relative.is_absolute():
        raise ValueError("Expected source artifact paths to be relative.")
    path = (run_dir / relative).resolve()
    try:
        path.relative_to(run_dir)
    except ValueError as error:
        raise ValueError("Source artifact path escapes the source run.") from error
    if not path.is_file() or path.stat().st_size != int(record.get("size_bytes", -1)):
        raise ValueError("Source deployment artifact size/path mismatch.")
    return path


def run(
    *,
    config_path: Path,
    teacher_weights_path: Path,
    source_run: Path,
    output_dir: Path,
    command: Sequence[str],
    sample_limit: int | None,
    maximum_endpoints: int | None,
) -> Path:
    if sample_limit is not None and sample_limit < 1:
        raise ValueError("Expected --sample-limit to be positive when provided.")
    if maximum_endpoints is not None and maximum_endpoints < 1:
        raise ValueError("Expected --maximum-endpoints to be positive when provided.")
    config = _require_file(config_path, label="--config")
    teacher_weights = _require_file(teacher_weights_path, label="--teacher-weights")
    run_dir, source_manifest, spec, deployment_records = _source_bundle(
        source_run, config_path=config
    )
    if sha256_file(teacher_weights) != spec.protocol.expected_teacher_weights_sha256:
        raise ValueError("Teacher checkpoint SHA-256 does not match the source protocol.")
    ordered_records = tuple(sorted(deployment_records, key=lambda item: str(item["path"])))
    selected_records = (
        ordered_records
        if maximum_endpoints is None
        else ordered_records[:maximum_endpoints]
    )
    full_coverage = sample_limit is None and maximum_endpoints is None
    resolved_config = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "evidence_tier": EVIDENCE_TIER,
        "operation": "read_only_saved_endpoint_replay",
        "source_run": str(run_dir),
        "source_run_id": source_manifest.get("run_id"),
        "source_experiment_id": SOURCE_EXPERIMENT_ID,
        "forward_policy": FORWARD_POLICY,
        "literal_apparent_state_retained_as_diagnostic": True,
        "literal_apparent_network_forward": False,
        "programming_or_resampling_invoked": False,
        "inference_read_noise": False,
        "sample_limit": sample_limit,
        "maximum_endpoints": maximum_endpoints,
        "expected_source_endpoints": len(ordered_records),
        "selected_endpoints": len(selected_records),
        "coverage": "complete" if full_coverage else "reduced",
        "primary_readout": (
            "apparent_projected_accuracy_and_paired_corruption_penalty_on_B_D"
        ),
    }
    store = RunStore.create(
        output_root=output_dir,
        experiment_id=EXPERIMENT_ID,
        resolved_config=resolved_config,
        command=command,
        repo_root=_ROOT,
        input_artifacts=(
            _input("source_manifest", run_dir / "manifest.json"),
            _input("source_resolved_config", run_dir / "config.resolved.json"),
            _input("source_result", run_dir / "result.json"),
            _input("declared_source_config", config),
            _input("teacher_weights", teacher_weights),
        ),
        resume_capability="unsupported",
    )
    try:
        if spec.student.runtime.device != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("Apparent-state replay requires local CUDA.")
        torch.manual_seed(spec.student.runtime.seed)
        torch.cuda.manual_seed_all(spec.student.runtime.seed)
        device = torch.device("cuda")
        loaders = build_mnist_loaders(
            spec.student.data,
            data_seed=spec.student.runtime.data_seed,
            calibration_examples=spec.student.mapping.calibration_examples,
            calibration_batch_size=spec.student.mapping.calibration_batch_size,
        )
        teacher, teacher_metadata = _load_teacher(
            teacher_weights, device=device, spec=spec.student
        )
        stack = build_student_stack(spec.student, enable_measured=False)
        stack.cost.gain = float(spec.protocol.deployment["fixed_logit_gain"])
        bindings = tuple(stack.bundle.catalog.trainable)
        binding_keys = tuple(binding.key for binding in bindings)
        binding_shapes = tuple(tuple(int(item) for item in binding.state.shape) for binding in bindings)

        endpoint_reports: list[Mapping[str, Any]] = []
        observed_ids: set[tuple[Any, ...]] = set()
        persistent_replay_canary: Mapping[str, Any] | None = None
        for index, record in enumerate(selected_records, start=1):
            path = _artifact_path(run_dir, record)
            observed_sha256 = sha256_file(path)
            if observed_sha256 != record.get("sha256"):
                raise ValueError("Source deployment artifact SHA-256 mismatch.")
            payload = _strict_torch_load(path)
            endpoint_id = _endpoint_id(payload)
            if endpoint_id in observed_ids:
                raise ValueError("Duplicate source deployment endpoint identity.")
            observed_ids.add(endpoint_id)
            if endpoint_id not in _expected_endpoint_ids(spec):
                raise ValueError(f"Unexpected source endpoint identity: {endpoint_id!r}.")
            persistent_g, apparent_g, state_report = prepare_apparent_endpoint(
                payload, binding_shapes=binding_shapes
            )
            saved_test = payload.get("test")
            saved_prediction_sha256 = payload.get("test_prediction_sha256")
            if not isinstance(saved_test, Mapping) or not isinstance(
                saved_prediction_sha256, str
            ):
                raise ValueError("Expected saved persistent test evidence.")

            persistent_comparison_metrics = dict(saved_test)
            persistent_comparison_prediction_sha256 = saved_prediction_sha256
            persistent_comparison_source = "saved_full_test"
            if sample_limit is not None:
                _apply_full_g(
                    stack,
                    split_flat_physical(
                        persistent_g.to(device=device), binding_shapes
                    ),
                )
                (
                    persistent_comparison_metrics,
                    persistent_comparison_prediction,
                ) = _evaluate_detailed(
                    stack,
                    teacher,
                    loaders.test,
                    sample_limit=sample_limit,
                )
                persistent_comparison_prediction_sha256 = _tensor_sha256(
                    persistent_comparison_prediction.to(torch.int64)
                )
                persistent_comparison_source = "same_subset_replay"

            # One full-coverage endpoint proves that the current model, loader,
            # and binding order replay the source persistent forward exactly.
            if persistent_replay_canary is None and sample_limit is None:
                _apply_full_g(
                    stack,
                    split_flat_physical(
                        persistent_g.to(device=device), binding_shapes
                    ),
                )
                replay_metrics, replay_prediction = _evaluate_detailed(
                    stack, teacher, loaders.test, sample_limit=None
                )
                replay_sha256 = _tensor_sha256(replay_prediction.to(torch.int64))
                if (
                    replay_sha256 != saved_prediction_sha256
                    or int(replay_metrics["examples"]) != int(saved_test["examples"])
                    or int(replay_metrics["student_correct"])
                    != int(saved_test["student_correct"])
                ):
                    raise RuntimeError(
                        "Saved persistent endpoint did not replay exactly before "
                        "the apparent-state intervention."
                    )
                persistent_replay_canary = {
                    "endpoint_id": list(endpoint_id),
                    "source_artifact": str(path),
                    "prediction_sha256": replay_sha256,
                    "examples": int(replay_metrics["examples"]),
                    "student_correct": int(replay_metrics["student_correct"]),
                    "passed": True,
                }

            _apply_full_g(
                stack,
                split_flat_physical(apparent_g.to(device=device), binding_shapes),
            )
            apparent_metrics, apparent_prediction = _evaluate_detailed(
                stack,
                teacher,
                loaders.test,
                sample_limit=sample_limit,
            )
            report = {
                "mode": "apparent_state_endpoint_replay",
                "array_label": payload["array_label"],
                "assignment_seed": int(payload["assignment_seed"]),
                "state_label": payload["state_label"],
                "population_role": payload["population_role"],
                "population_fingerprint": payload["population_fingerprint"],
                "endpoint_seed": int(payload["endpoint_seed"]),
                "source_artifact": {
                    "path": str(path),
                    "sha256": observed_sha256,
                },
                "persistent_saved_test": dict(saved_test),
                "persistent_saved_prediction_sha256": saved_prediction_sha256,
                "persistent_comparison_test": dict(
                    persistent_comparison_metrics
                ),
                "persistent_comparison_prediction_sha256": (
                    persistent_comparison_prediction_sha256
                ),
                "persistent_comparison_source": persistent_comparison_source,
                "apparent_projected_test": dict(apparent_metrics),
                "apparent_projected_prediction_sha256": _tensor_sha256(
                    apparent_prediction.to(torch.int64)
                ),
                "apparent_projected_minus_persistent_accuracy": (
                    float(apparent_metrics["student_accuracy"])
                    - float(persistent_comparison_metrics["student_accuracy"])
                ),
                "state_diagnostics": state_report,
                "programming_or_resampling_invoked": False,
            }
            endpoint_reports.append(report)
            store.append_metric(report)
            print(
                f"[{index}/{len(selected_records)}] array={payload['array_label']} "
                f"state={payload['state_label']} population={payload['population_role']} "
                f"seed={payload['endpoint_seed']} "
                f"persistent_comparison="
                f"{100.0*float(persistent_comparison_metrics['student_accuracy']):.2f}% "
                f"apparent_projected={100.0*float(apparent_metrics['student_accuracy']):.2f}% "
                f"literal_oob="
                f"{100.0*float(state_report['out_of_range_fraction']):.2f}%",
                flush=True,
            )
            gc.collect()
            torch.cuda.empty_cache()

        if full_coverage and observed_ids != _expected_endpoint_ids(spec):
            raise RuntimeError("Full apparent replay did not cover every source endpoint.")
        aggregates = summarize_endpoints(endpoint_reports)
        summary = {
            "schema": SUMMARY_SCHEMA,
            "schema_version": SUMMARY_SCHEMA_VERSION,
            "status": "complete",
            "evidence_tier": EVIDENCE_TIER,
            "operation": "read_only_saved_endpoint_replay",
            "source": {
                "run_dir": str(run_dir),
                "run_id": source_manifest.get("run_id"),
                "source_revision": source_manifest.get("source"),
                "manifest_sha256": sha256_file(run_dir / "manifest.json"),
                "result_sha256": sha256_file(run_dir / "result.json"),
                "resolved_config_sha256": sha256_file(
                    run_dir / "config.resolved.json"
                ),
            },
            "teacher": {
                "path": str(teacher_weights),
                "sha256": sha256_file(teacher_weights),
                "metadata": teacher_metadata,
            },
            "contract": {
                "forward_policy": FORWARD_POLICY,
                "held_apparent_sample_reused_for_entire_test_forward": True,
                "fresh_inference_read_noise": False,
                "programming_invoked": False,
                "resampling_invoked": False,
                "optimizer_updates": 0,
                "literal_apparent_network_forward": False,
                "literal_apparent_state_retained_as_diagnostic": True,
                "persistent_state_role": "saved_continuation_and_comparison_only",
                "binding_keys": list(binding_keys),
                "binding_shapes": [list(shape) for shape in binding_shapes],
            },
            "coverage": {
                "kind": "complete" if full_coverage else "reduced",
                "source_endpoints": len(ordered_records),
                "evaluated_endpoints": len(endpoint_reports),
                "test_sample_limit": sample_limit,
                "observed_endpoint_ids_unique": True,
            },
            "persistent_replay_canary": persistent_replay_canary,
            "aggregates": aggregates,
            "endpoints": endpoint_reports,
            "limitations": [
                "exploratory_noncanonical",
                "model_based_aihwkit_1.1.0_OM_preset_not_raw_measured_traces",
                "apparent_projection_to_positive_G_is_a_DRN_specific_diagnostic",
                "projection_is_not_literal_unbounded_aihwkit_apparent_forward",
                "no_inference_read_noise_retention_drift_or_peripheral_nonideality",
            ],
        }
        summary_path = store.run_dir / "artifacts" / "scientific_summary.json"
        atomic_write_json(summary_path, summary)
        terminal = {
            "evidence_tier": EVIDENCE_TIER,
            "coverage": "complete" if full_coverage else "reduced",
            "evaluated_endpoints": len(endpoint_reports),
            "source_endpoints": len(ordered_records),
            "forward_policy": FORWARD_POLICY,
            "transfer_B_D": aggregates["transfer_B_D"],
            "persistent_replay_canary_passed": (
                None
                if persistent_replay_canary is None
                else bool(persistent_replay_canary["passed"])
            ),
        }
        store.append_metric({"mode": "apparent_state_terminal", **terminal})
        result_path = store.complete(
            metrics=terminal,
            artifacts=(
                store.artifact_record(summary_path, kind="scientific_summary"),
            ),
        )
        print(f"run_dir={store.run_dir}", flush=True)
        return result_path
    except BaseException as error:
        store.fail(error)
        raise


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    command = tuple(sys.argv if argv is None else ("apparent-replay", *argv))
    run(
        config_path=args.config,
        teacher_weights_path=args.teacher_weights,
        source_run=args.source_run,
        output_dir=args.output_dir,
        command=command,
        sample_limit=args.sample_limit,
        maximum_endpoints=args.maximum_endpoints,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through main
    raise SystemExit(main())
