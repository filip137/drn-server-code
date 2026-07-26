"""CUDA memory preflight for the immutable v5 candidate manifest.

The 36 candidates contain only six distinct execution geometries: one for each
architecture/amplification-scheme pair.  This module measures one conservative
per-process peak for each geometry and expands it to every immutable candidate
entry.  It deliberately builds only the MNIST-train loader used by the LR
engine; the official test split is never instantiated.
"""

from __future__ import annotations

import gc
import hashlib
import math
from pathlib import Path
from typing import Any, Mapping

from .io import atomic_write_json, read_json
from .lr_artifacts import load_stage_manifest
from .lr_engine import build_loader_bundle, build_model_runtime, training_step
from .lr_study import load_study
from .lr_v5_packing import (
    MEMORY_PREFLIGHT_SCHEMA_VERSION,
    _stage_entries,
    validate_memory_preflight,
)


ARCHITECTURES = ("conv1", "conv2")
SCHEMES = ("baseline", "ours", "legacy")
RUNTIME_KEYS = tuple(
    f"{architecture}:{scheme}"
    for architecture in ARCHITECTURES
    for scheme in SCHEMES
)


def _error(expected: str, provided: Any) -> ValueError:
    return ValueError(f"Expected {expected}. Provided value: {provided!r}.")


def _positive_finite(value: Any, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) <= 0.0
    ):
        raise _error(f"{label} to be a positive finite number", value)
    return float(value)


def _nonnegative_finite(value: Any, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) < 0.0
    ):
        raise _error(f"{label} to be a non-negative finite number", value)
    return float(value)


def _runtime_key(architecture: str, scheme: str) -> str:
    return f"{architecture}:{scheme}"


def v5_runtime_groups(stage_manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Validate and collapse 36 candidates into six runtime-equivalent groups.

    The representative is the entry with the smallest maximum raw parameter
    LR.  LR values do not change tensor geometry, and the smallest vector
    minimizes the chance that a deliberately short memory measurement stops
    on a numerical gate before the execution graph has been exercised.
    """

    entries = _stage_entries(stage_manifest)
    grouped: dict[str, list[dict[str, Any]]] = {key: [] for key in RUNTIME_KEYS}
    for entry in entries:
        payload = entry["payload"]
        architecture = payload["architecture"]
        scheme = payload.get("scheme")
        if scheme not in SCHEMES:
            raise _error(
                f"entry {entry['entry_index']} scheme to be one of {SCHEMES!r}",
                scheme,
            )
        rates = payload.get("learning_rates_by_parameter")
        if not isinstance(rates, Mapping) or not rates:
            raise _error(
                f"entry {entry['entry_index']} learning_rates_by_parameter to be a non-empty object",
                rates,
            )
        normalized_rates = {
            str(name): _positive_finite(
                value,
                f"entry {entry['entry_index']} learning rate for {name!r}",
            )
            for name, value in rates.items()
        }
        normalized = dict(entry)
        normalized["payload"] = dict(payload)
        normalized["payload"]["learning_rates_by_parameter"] = normalized_rates
        grouped[_runtime_key(architecture, scheme)].append(normalized)

    result: list[dict[str, Any]] = []
    for key in RUNTIME_KEYS:
        members = grouped[key]
        if len(members) != 6:
            raise _error(
                f"runtime group {key!r} to contain exactly six arm/alpha candidates",
                len(members),
            )
        architecture, scheme = key.split(":", maxsplit=1)
        row_ids = {member["payload"]["row_id"] for member in members}
        if len(row_ids) != 1:
            raise _error(
                f"runtime group {key!r} to use one frozen row_id",
                sorted(row_ids),
            )
        arm_roles = {
            (member["payload"]["arm"], member["payload"]["alpha_role"])
            for member in members
        }
        expected_arm_roles = {
            (arm, role)
            for arm in ("strict_equal", "historical_profile")
            for role in ("lower", "center", "upper")
        }
        if arm_roles != expected_arm_roles:
            raise _error(
                f"runtime group {key!r} to cover every arm/alpha role once",
                sorted(arm_roles),
            )

        def selection_key(member: Mapping[str, Any]) -> tuple[float, int]:
            rates = member["payload"]["learning_rates_by_parameter"]
            return max(float(value) for value in rates.values()), int(
                member["entry_index"]
            )

        representative = min(members, key=selection_key)
        result.append(
            {
                "runtime_key": key,
                "architecture": architecture,
                "scheme": scheme,
                "row_id": representative["payload"]["row_id"],
                "representative_entry_index": representative["entry_index"],
                "member_entry_indices": sorted(
                    int(member["entry_index"]) for member in members
                ),
                "learning_rates_by_parameter": dict(
                    representative["payload"]["learning_rates_by_parameter"]
                ),
                "maximum_representative_parameter_lr": max(
                    representative["payload"][
                        "learning_rates_by_parameter"
                    ].values()
                ),
            }
        )
    return result


def build_v5_memory_preflight(
    stage_manifest: Mapping[str, Any],
    *,
    stage_manifest_sha256: str,
    measurements_by_runtime_key: Mapping[str, Mapping[str, Any]],
    steps_per_runtime: int,
    device: Mapping[str, Any],
) -> dict[str, Any]:
    """Expand six measured runtime peaks to the 36-entry packing input."""

    if (
        not isinstance(stage_manifest_sha256, str)
        or len(stage_manifest_sha256) != 64
        or any(character not in "0123456789abcdef" for character in stage_manifest_sha256)
    ):
        raise _error(
            "stage_manifest_sha256 to be a lowercase SHA-256 digest",
            stage_manifest_sha256,
        )
    if type(steps_per_runtime) is not int or steps_per_runtime <= 0:
        raise _error("steps_per_runtime to be a positive integer", steps_per_runtime)
    if not isinstance(device, Mapping):
        raise _error("device metadata to be an object", device)

    groups = v5_runtime_groups(stage_manifest)
    if set(measurements_by_runtime_key) != set(RUNTIME_KEYS):
        raise _error(
            f"measurements_by_runtime_key to contain exactly {list(RUNTIME_KEYS)!r}",
            sorted(measurements_by_runtime_key),
        )

    expanded: dict[str, float] = {}
    records: list[dict[str, Any]] = []
    required_measurement_fields = {
        "torch_peak_allocated_mib",
        "torch_peak_reserved_mib",
        "maximum_sampled_device_used_mib",
        "maximum_sampled_non_torch_overhead_mib",
        "conservative_peak_gpu_memory_mib",
    }
    for group in groups:
        key = group["runtime_key"]
        raw_measurement = measurements_by_runtime_key[key]
        if not isinstance(raw_measurement, Mapping):
            raise _error(f"measurement for runtime {key!r} to be an object", raw_measurement)
        missing = sorted(required_measurement_fields - set(raw_measurement))
        if missing:
            raise _error(
                f"measurement for runtime {key!r} to contain {sorted(required_measurement_fields)!r}",
                {"missing": missing},
            )
        measurement = {
            field: (
                _positive_finite
                if field == "conservative_peak_gpu_memory_mib"
                else _nonnegative_finite
            )(raw_measurement[field], f"runtime {key!r} {field}")
            for field in sorted(required_measurement_fields)
        }
        conservative = measurement["conservative_peak_gpu_memory_mib"]
        if conservative + 1e-12 < max(
            measurement["torch_peak_reserved_mib"],
            measurement["maximum_sampled_device_used_mib"],
            measurement["torch_peak_reserved_mib"]
            + measurement["maximum_sampled_non_torch_overhead_mib"],
        ):
            raise _error(
                f"runtime {key!r} conservative peak to cover allocator and sampled device peaks",
                measurement,
            )
        for entry_index in group["member_entry_indices"]:
            expanded[str(entry_index)] = conservative
        records.append(
            {
                **group,
                "measurement": measurement,
            }
        )

    if set(expanded) != {str(index) for index in range(36)}:
        raise RuntimeError(
            "Expected runtime measurements to expand to every candidate entry exactly once. "
            f"Provided value: {sorted(expanded)!r}."
        )
    return {
        "schema_version": MEMORY_PREFLIGHT_SCHEMA_VERSION,
        "study_id": stage_manifest["study_id"],
        "stage_name": "candidates",
        "stage_manifest_sha256": stage_manifest_sha256,
        "measurement_contract": {
            "source_split": "mnist_train",
            "official_test_read": False,
            "validation_read": False,
            "steps_per_runtime": steps_per_runtime,
            "distinct_runtime_count": 6,
            "candidate_entry_count": 36,
            "representative_selection": (
                "smallest_maximum_parameter_lr_then_lowest_entry_index"
            ),
            "memory_statistic": (
                "ceil(max(sampled_device_used, torch_peak_reserved + "
                "sampled_non_torch_overhead))"
            ),
            "bytes_per_mib": 1_048_576,
        },
        "device": dict(device),
        "representative_measurements": records,
        "peak_gpu_memory_mib_by_entry_index": expanded,
    }


def _sample_cuda_memory(torch: Any, device: Any) -> dict[str, int]:
    torch.cuda.synchronize(device)
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    reserved_bytes = int(torch.cuda.memory_reserved(device))
    used_bytes = int(total_bytes) - int(free_bytes)
    return {
        "device_used_bytes": used_bytes,
        "non_torch_overhead_bytes": max(0, used_bytes - reserved_bytes),
    }


def _measure_runtime(
    study: Mapping[str, Any],
    row: Mapping[str, Any],
    *,
    checkpoint: Path,
    learning_rates_by_parameter: Mapping[str, float],
    bundle: Any,
    device: str,
    steps: int,
) -> dict[str, float]:
    import torch

    requested = torch.device(device)
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize(requested)
    torch.cuda.reset_peak_memory_stats(requested)
    samples = [_sample_cuda_memory(torch, requested)]
    runtime = build_model_runtime(
        study,
        row,
        device=device,
        initialization_checkpoint=checkpoint,
        learning_rate=learning_rates_by_parameter,
    )
    samples.append(_sample_cuda_memory(torch, requested))
    bundle.reset_train_shuffle()
    iterator = iter(bundle.train_loader)
    result = None
    batch = None
    try:
        for _step in range(steps):
            batch = next(iterator)
            result = training_step(
                runtime,
                batch,
                learning_rate=learning_rates_by_parameter,
                restore=False,
                zero_proposal_epsilon=float(
                    study["range_test"]["gates"]["projection_efficiency"][
                        "zero_update_epsilon"
                    ]
                ),
            )
            samples.append(_sample_cuda_memory(torch, requested))
        torch.cuda.synchronize(requested)
        peak_allocated = int(torch.cuda.max_memory_allocated(requested))
        peak_reserved = int(torch.cuda.max_memory_reserved(requested))
        maximum_used = max(sample["device_used_bytes"] for sample in samples)
        maximum_non_torch = max(
            sample["non_torch_overhead_bytes"] for sample in samples
        )
        conservative = max(maximum_used, peak_reserved + maximum_non_torch)
        to_mib = lambda value: float(value) / 1_048_576.0
        return {
            "torch_peak_allocated_mib": to_mib(peak_allocated),
            "torch_peak_reserved_mib": to_mib(peak_reserved),
            "maximum_sampled_device_used_mib": to_mib(maximum_used),
            "maximum_sampled_non_torch_overhead_mib": to_mib(maximum_non_torch),
            "conservative_peak_gpu_memory_mib": float(
                math.ceil(to_mib(conservative))
            ),
        }
    finally:
        del result
        del batch
        del iterator
        del runtime
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize(requested)


def run_v5_memory_preflight(
    *,
    study_dir: str | Path,
    candidate_manifest_path: str | Path,
    data_root: str | Path,
    output_path: str | Path,
    device: str = "cuda",
    steps_per_runtime: int = 3,
) -> dict[str, Any]:
    """Measure six v5 CUDA runtimes and publish an immutable preflight JSON."""

    import torch

    root, spec = load_study(study_dir)
    if spec.data.get("schema_version") != "mnist-conv-lr-study/v5":
        raise _error(
            "study schema_version to be 'mnist-conv-lr-study/v5'",
            spec.data.get("schema_version"),
        )
    official_test = spec.data["dataset"].get("official_test")
    if official_test != {"enabled": False, "read_allowed": False}:
        raise _error(
            "v5 official-test contract to disable and prohibit reads",
            official_test,
        )
    if type(steps_per_runtime) is not int or steps_per_runtime <= 0:
        raise _error("steps_per_runtime to be a positive integer", steps_per_runtime)
    requested = torch.device(device)
    if requested.type != "cuda":
        raise _error("a CUDA device", device)

    manifest_supplied = Path(candidate_manifest_path).expanduser().absolute()
    output_supplied = Path(output_path).expanduser().absolute()
    for label, path in (
        ("candidate manifest", manifest_supplied),
        ("output", output_supplied),
    ):
        if path.is_symlink():
            raise _error(f"{label} path not to be a symlink", path)
    manifest_path = manifest_supplied.resolve()
    output = output_supplied.resolve()
    for label, path in (("candidate manifest", manifest_path), ("output", output)):
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise _error(f"{label} path to be contained by the v5 study", path) from exc
    stage_bytes = manifest_path.read_bytes()
    stage_sha256 = hashlib.sha256(stage_bytes).hexdigest()
    stage_manifest = load_stage_manifest(
        manifest_path,
        study_dir=root,
        expected_study_id=spec.study_id,
        expected_stage_name="candidates",
    )
    if output.exists():
        existing = read_json(output)
        validate_memory_preflight(
            existing,
            stage_manifest_sha256=stage_sha256,
            entry_count=36,
        )
        if existing.get("study_id") != spec.study_id:
            raise _error(
                "existing memory preflight study_id to match the v5 study",
                existing.get("study_id"),
            )
        existing_contract = existing.get("measurement_contract")
        if (
            not isinstance(existing_contract, Mapping)
            or existing_contract.get("steps_per_runtime") != steps_per_runtime
        ):
            raise _error(
                "existing memory preflight steps_per_runtime to match the request",
                None
                if not isinstance(existing_contract, Mapping)
                else existing_contract.get("steps_per_runtime"),
            )
        existing_device = existing.get("device")
        if (
            not isinstance(existing_device, Mapping)
            or existing_device.get("requested") != device
        ):
            raise _error(
                "existing memory preflight requested device to match the request",
                None
                if not isinstance(existing_device, Mapping)
                else existing_device.get("requested"),
            )
        return existing
    if not torch.cuda.is_available():
        raise _error("an available CUDA device", device)
    groups = v5_runtime_groups(stage_manifest)
    rows_by_id = {row["row_id"]: row for row in spec.rows}
    bundle = build_loader_bundle(
        spec.data,
        data_root=data_root,
        download=False,
        return_source_indices=True,
    )
    measurements: dict[str, dict[str, float]] = {}
    for group in groups:
        row_id = group["row_id"]
        if row_id not in rows_by_id:
            raise _error("candidate runtime row_id to exist in the v5 study", row_id)
        checkpoint = root / "initialization" / f"{group['architecture']}.pt"
        if not checkpoint.is_file():
            raise FileNotFoundError(
                "Expected the shared architecture initialization checkpoint to exist. "
                f"Provided value: {checkpoint}."
            )
        measurements[group["runtime_key"]] = _measure_runtime(
            spec.data,
            rows_by_id[row_id],
            checkpoint=checkpoint,
            learning_rates_by_parameter=group["learning_rates_by_parameter"],
            bundle=bundle,
            device=device,
            steps=steps_per_runtime,
        )

    index = requested.index
    if index is None:
        index = torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(index)
    result = build_v5_memory_preflight(
        stage_manifest,
        stage_manifest_sha256=stage_sha256,
        measurements_by_runtime_key=measurements,
        steps_per_runtime=steps_per_runtime,
        device={
            "requested": device,
            "cuda_index": int(index),
            "name": str(properties.name),
            "total_memory_mib": float(properties.total_memory) / 1_048_576.0,
            "torch_version": str(torch.__version__),
            "cuda_runtime_version": (
                None if torch.version.cuda is None else str(torch.version.cuda)
            ),
        },
    )
    atomic_write_json(output, result, canonical=True)
    return result


__all__ = [
    "ARCHITECTURES",
    "RUNTIME_KEYS",
    "SCHEMES",
    "build_v5_memory_preflight",
    "run_v5_memory_preflight",
    "v5_runtime_groups",
]
