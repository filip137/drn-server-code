"""Jean Zay concurrency evidence and deterministic one-GPU waves for v6.

This module is intentionally independent of the v5 measured-memory packer.  A
v6 pack manifest binds both the immutable stage manifest and, for training
stages, the immutable concurrency benchmark that justified the wave width.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from .io import atomic_write_json, read_json


BENCHMARK_SCHEMA_VERSION = "mnist-conv-lr-concurrency-benchmark/v1"
PACK_SCHEMA_VERSION = "mnist-conv-lr-pack-manifest/v2"
R3_ALLOCATION_ID = "AD010913993R3"
R3_AUTHORIZATION_TOKEN = (
    "AD010913993R3:fmu@v100:gpu_p13:qos_gpu-t3:v100-32g:gpu1"
)
IDRENV_PROJECT = "fmu"
SLURM_ACCOUNT = "fmu@v100"
PARTITION = "gpu_p13"
QOS = "qos_gpu-t3"
GPU_CONSTRAINT = "v100-32g"
GPU_CAPACITY_MIB = 32_768
HEADROOM_FRACTION = 0.10
CPU_COUNT = 16
WALLTIME_HOURS = 20
TARGET_SWEEP_HOURS = 18
CONCURRENCY_LEVELS = (1, 2, 4, 8, 12, 16)
WARMUP_STEPS = 32
MEASURED_STEPS = 256
CANONICAL_STEPS_PER_ENTRY = 17_190
THROUGHPUT_FRACTION = 0.90
SUPPORTED_STAGES = ("audit", "probe", "baseline_candidates", "confirmations")


def _error(expected: str, provided: Any) -> ValueError:
    return ValueError(f"Expected {expected}. Provided value: {provided!r}.")


def _sha256_digest(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise _error(f"{label} to be a lowercase SHA-256 digest", value)
    return value


def _positive(value: Any, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) <= 0.0
    ):
        raise _error(f"{label} to be a positive finite number", value)
    return float(value)


def _nonnegative(value: Any, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) < 0.0
    ):
        raise _error(f"{label} to be a non-negative finite number", value)
    return float(value)


def r3_allocation_contract() -> dict[str, Any]:
    return {
        "allocation_id": R3_ALLOCATION_ID,
        "allocation_suffix": "R3",
        "idrenv_project": IDRENV_PROJECT,
        "slurm_account": SLURM_ACCOUNT,
        "partition": PARTITION,
        "qos": QOS,
        "constraint": GPU_CONSTRAINT,
        "gpus": 1,
        "cpus": CPU_COUNT,
        "hint": "nomultithread",
        "walltime_hours": WALLTIME_HOURS,
    }


def validate_r3_allocation_contract(value: Any) -> dict[str, Any]:
    expected = r3_allocation_contract()
    if value != expected:
        raise _error(
            "the Jean Zay allocation contract to be the frozen R3/fmu V100 contract",
            value,
        )
    serialized = json.dumps(value, sort_keys=True).lower()
    if "umg" in serialized or not value["allocation_id"].endswith("R3"):
        raise _error("an R3 allocation with no historical umg account", value)
    return dict(value)


def validate_stage_manifest(
    stage_manifest: Mapping[str, Any],
    *,
    expected_stage: str | None = None,
) -> list[dict[str, Any]]:
    if not isinstance(stage_manifest, Mapping):
        raise _error("stage manifest to be a JSON object", stage_manifest)
    stage_name = stage_manifest.get("stage_name")
    if stage_name not in SUPPORTED_STAGES:
        raise _error(f"stage_name to be one of {SUPPORTED_STAGES!r}", stage_name)
    if expected_stage is not None and stage_name != expected_stage:
        raise _error(f"stage_name to be {expected_stage!r}", stage_name)
    study_id = stage_manifest.get("study_id")
    study_digest = (
        study_id.removeprefix("lrstudy_") if isinstance(study_id, str) else ""
    )
    if (
        not isinstance(study_id, str)
        or not study_id.startswith("lrstudy_")
        or len(study_digest) != 64
        or any(character not in "0123456789abcdef" for character in study_digest)
    ):
        raise _error(
            "stage manifest study_id to be 'lrstudy_' plus 64 lowercase hex characters",
            study_id,
        )
    entries = stage_manifest.get("entries")
    expected_counts = {
        "audit": 1,
        "probe": 3,
        "baseline_candidates": 16,
        "confirmations": 2,
    }
    expected_count = expected_counts[stage_name]
    if not isinstance(entries, list) or len(entries) != expected_count:
        raise _error(
            f"{stage_name} manifest to contain exactly {expected_count} entries",
            entries,
        )
    normalized: list[dict[str, Any]] = []
    for index, entry in enumerate(entries):
        if not isinstance(entry, Mapping) or entry.get("entry_index") != index:
            raise _error("entry indices to be contiguous from zero", entry)
        payload = entry.get("payload")
        if not isinstance(payload, Mapping):
            raise _error(f"entry {index} payload to be an object", payload)
        if stage_name in {"probe", "baseline_candidates", "confirmations"}:
            architecture = payload.get("architecture")
            if architecture is None and isinstance(payload.get("row"), Mapping):
                architecture = payload["row"].get("architecture")
            if architecture != "conv2":
                raise _error(f"entry {index} architecture to be conv2", architecture)
        normalized.append(dict(entry))
    return normalized


def find_safe_baseline_entry(stage_manifest: Mapping[str, Any]) -> int:
    """Return the unique v6 baseline entry at rho=(3e-3, 3e-3)."""

    entries = validate_stage_manifest(
        stage_manifest, expected_stage="baseline_candidates"
    )
    matches = []
    for entry in entries:
        payload = entry["payload"]
        try:
            rho_conv = float(payload.get("rho_conv"))
            rho_dense = float(payload.get("rho_dense"))
        except (TypeError, ValueError):
            continue
        if (
            payload.get("scheme") == "baseline"
            and math.isclose(rho_conv, 0.003, rel_tol=0.0, abs_tol=1e-15)
            and math.isclose(rho_dense, 0.003, rel_tol=0.0, abs_tol=1e-15)
        ):
            matches.append(int(entry["entry_index"]))
    if len(matches) != 1:
        raise _error(
            "exactly one baseline benchmark entry at rho_conv=rho_dense=0.003",
            matches,
        )
    return matches[0]


def _normalize_level(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise _error("benchmark level to be an object", value)
    concurrency = value.get("concurrency")
    if type(concurrency) is not int or concurrency not in CONCURRENCY_LEVELS:
        raise _error(
            f"benchmark concurrency to be one of {CONCURRENCY_LEVELS!r}", concurrency
        )
    completed = value.get("completed_children")
    failed = value.get("failed_children")
    if type(completed) is not int or completed < 0 or completed > concurrency:
        raise _error("completed_children to be within the concurrency", completed)
    if type(failed) is not int or failed < 0 or failed > concurrency:
        raise _error("failed_children to be within the concurrency", failed)
    if completed + failed != concurrency:
        raise _error(
            "completed_children plus failed_children to equal concurrency",
            {"completed": completed, "failed": failed, "concurrency": concurrency},
        )
    normalized = {
        "concurrency": concurrency,
        "completed_children": completed,
        "failed_children": failed,
        "combined_peak_gpu_memory_mib": _nonnegative(
            value.get("combined_peak_gpu_memory_mib"),
            "combined_peak_gpu_memory_mib",
        ),
        "peak_host_rss_mib": _nonnegative(
            value.get("peak_host_rss_mib"), "peak_host_rss_mib"
        ),
        "mean_gpu_utilization_percent": _nonnegative(
            value.get("mean_gpu_utilization_percent"),
            "mean_gpu_utilization_percent",
        ),
        "aggregate_successful_steps_per_second": _nonnegative(
            value.get("aggregate_successful_steps_per_second"),
            "aggregate_successful_steps_per_second",
        ),
        "successful_measured_steps": value.get("successful_measured_steps"),
        "wall_seconds": _positive(value.get("wall_seconds"), "wall_seconds"),
        "child_failures": value.get("child_failures", []),
    }
    expected_steps = completed * MEASURED_STEPS
    if (
        type(normalized["successful_measured_steps"]) is not int
        or normalized["successful_measured_steps"] != expected_steps
    ):
        raise _error(
            f"successful_measured_steps to equal completed_children*{MEASURED_STEPS}",
            normalized["successful_measured_steps"],
        )
    if not isinstance(normalized["child_failures"], list) or len(
        normalized["child_failures"]
    ) != failed:
        raise _error("child_failures to record every failed child", normalized)
    if normalized["mean_gpu_utilization_percent"] > 100.0:
        raise _error("mean GPU utilization to be at most 100 percent", normalized)
    return normalized


def select_concurrency(
    levels: Sequence[Mapping[str, Any]],
    *,
    candidate_count: int = 16,
) -> dict[str, Any]:
    """Apply the frozen memory, throughput, and projected-duration gates."""

    if type(candidate_count) is not int or candidate_count <= 0:
        raise _error("candidate_count to be a positive integer", candidate_count)
    normalized = [_normalize_level(level) for level in levels]
    if [level["concurrency"] for level in normalized] != list(CONCURRENCY_LEVELS):
        raise _error(
            f"benchmark levels in exact order {list(CONCURRENCY_LEVELS)!r}",
            [level["concurrency"] for level in normalized],
        )
    safe = [
        level
        for level in normalized
        if level["failed_children"] == 0
        and level["completed_children"] == level["concurrency"]
        and level["combined_peak_gpu_memory_mib"] * (1.0 + HEADROOM_FRACTION)
        <= GPU_CAPACITY_MIB
        and level["aggregate_successful_steps_per_second"] > 0.0
    ]
    if not safe:
        raise RuntimeError(
            "Expected at least one zero-failure benchmark level to fit the V100-32GB "
            "memory contract with 10% headroom."
        )
    best_throughput = max(
        level["aggregate_successful_steps_per_second"] for level in safe
    )
    threshold = THROUGHPUT_FRACTION * best_throughput
    safe_by_concurrency = {level["concurrency"]: level for level in safe}
    evaluated = []
    for level in normalized:
        throughput = level["aggregate_successful_steps_per_second"]
        concurrency = level["concurrency"]
        full_waves, remainder = divmod(candidate_count, concurrency)
        wave_widths = [concurrency] * full_waves
        if remainder:
            wave_widths.append(remainder)
        projected: float | None = 0.0
        for width in wave_widths:
            measured = safe_by_concurrency.get(width)
            if measured is None:
                projected = None
                break
            projected += (
                width
                * CANONICAL_STEPS_PER_ENTRY
                / measured["aggregate_successful_steps_per_second"]
            )
        memory_with_headroom = level["combined_peak_gpu_memory_mib"] * (
            1.0 + HEADROOM_FRACTION
        )
        qualified = (
            level["failed_children"] == 0
            and level["completed_children"] == level["concurrency"]
            and memory_with_headroom <= GPU_CAPACITY_MIB
            and throughput >= threshold
            and projected is not None
            and projected <= TARGET_SWEEP_HOURS * 3600
        )
        evaluated.append(
            {
                "concurrency": level["concurrency"],
                "memory_with_headroom_mib": memory_with_headroom,
                "throughput_fraction_of_best": (
                    0.0 if best_throughput == 0.0 else throughput / best_throughput
                ),
                "deterministic_wave_widths": wave_widths,
                "projected_full_sweep_seconds": projected,
                "qualified": qualified,
            }
        )
    qualified = [item for item in evaluated if item["qualified"]]
    if not qualified:
        raise RuntimeError(
            "Expected at least one concurrency level to satisfy memory, throughput, "
            "and the 18-hour projected sweep limit."
        )
    selected = max(qualified, key=lambda item: item["concurrency"])
    return {
        "selected_concurrency": selected["concurrency"],
        "best_aggregate_successful_steps_per_second": best_throughput,
        "minimum_accepted_steps_per_second": threshold,
        "projected_full_sweep_seconds": selected[
            "projected_full_sweep_seconds"
        ],
        "evaluated_levels": evaluated,
    }


def build_benchmark_report(
    stage_manifest: Mapping[str, Any],
    *,
    stage_manifest_sha256: str,
    levels: Sequence[Mapping[str, Any]],
    device: Mapping[str, Any],
) -> dict[str, Any]:
    entries = validate_stage_manifest(
        stage_manifest, expected_stage="baseline_candidates"
    )
    digest = _sha256_digest(stage_manifest_sha256, "stage_manifest_sha256")
    safe_entry_index = find_safe_baseline_entry(stage_manifest)
    if not isinstance(device, Mapping):
        raise _error("device metadata to be an object", device)
    name = device.get("name")
    total_memory = _positive(device.get("total_memory_mib"), "device memory")
    if not isinstance(name, str) or "V100" not in name:
        raise _error("device name to identify a V100", name)
    if not 32_000 <= total_memory <= 33_000:
        raise _error("V100 memory to be approximately 32,768 MiB", total_memory)
    normalized = [_normalize_level(level) for level in levels]
    selection = select_concurrency(normalized, candidate_count=len(entries))
    return {
        "schema_version": BENCHMARK_SCHEMA_VERSION,
        "study_id": stage_manifest["study_id"],
        "stage_name": "baseline_candidates",
        "stage_manifest_sha256": digest,
        "r3_allocation": r3_allocation_contract(),
        "gpu_contract": {
            "name": name,
            "total_memory_mib": total_memory,
            "capacity_mib": GPU_CAPACITY_MIB,
            "headroom_fraction": HEADROOM_FRACTION,
        },
        "benchmark_contract": {
            "concurrency_levels": list(CONCURRENCY_LEVELS),
            "warmup_steps_per_child": WARMUP_STEPS,
            "measured_steps_per_child": MEASURED_STEPS,
            "synchronized_start": True,
            "shadow_only": True,
            "production_candidate_step_loop": True,
            "official_test_read": False,
            "safe_entry_index": safe_entry_index,
            "safe_rho_conv": 0.003,
            "safe_rho_dense": 0.003,
            "canonical_steps_per_entry": CANONICAL_STEPS_PER_ENTRY,
            "candidate_count": len(entries),
            "target_sweep_hours": TARGET_SWEEP_HOURS,
            "throughput_fraction": THROUGHPUT_FRACTION,
        },
        "levels": normalized,
        "selection": selection,
    }


def validate_benchmark_report(
    benchmark: Mapping[str, Any],
    *,
    expected_study_id: str | None = None,
) -> dict[str, Any]:
    if not isinstance(benchmark, Mapping):
        raise _error("benchmark report to be a JSON object", benchmark)
    if benchmark.get("schema_version") != BENCHMARK_SCHEMA_VERSION:
        raise _error(
            f"benchmark schema_version to be {BENCHMARK_SCHEMA_VERSION!r}",
            benchmark.get("schema_version"),
        )
    if benchmark.get("stage_name") != "baseline_candidates":
        raise _error("benchmark stage_name to be baseline_candidates", benchmark)
    study_id = benchmark.get("study_id")
    if expected_study_id is not None and study_id != expected_study_id:
        raise _error("benchmark study_id to match the stage manifest", study_id)
    _sha256_digest(benchmark.get("stage_manifest_sha256"), "stage_manifest_sha256")
    validate_r3_allocation_contract(benchmark.get("r3_allocation"))
    contract = benchmark.get("benchmark_contract")
    expected_contract_subset = {
        "concurrency_levels": list(CONCURRENCY_LEVELS),
        "warmup_steps_per_child": WARMUP_STEPS,
        "measured_steps_per_child": MEASURED_STEPS,
        "synchronized_start": True,
        "shadow_only": True,
        "production_candidate_step_loop": True,
        "official_test_read": False,
        "safe_entry_index": 8,
        "safe_rho_conv": 0.003,
        "safe_rho_dense": 0.003,
        "canonical_steps_per_entry": CANONICAL_STEPS_PER_ENTRY,
        "candidate_count": 16,
        "target_sweep_hours": TARGET_SWEEP_HOURS,
        "throughput_fraction": THROUGHPUT_FRACTION,
    }
    if not isinstance(contract, Mapping) or any(
        contract.get(key) != value for key, value in expected_contract_subset.items()
    ):
        raise _error("benchmark contract to match the frozen v6 protocol", contract)
    levels = benchmark.get("levels")
    if not isinstance(levels, list):
        raise _error("benchmark levels to be a list", levels)
    recomputed = select_concurrency(levels, candidate_count=16)
    if benchmark.get("selection") != recomputed:
        raise _error("benchmark selection to match recomputed gates", benchmark.get("selection"))
    gpu = benchmark.get("gpu_contract")
    if (
        not isinstance(gpu, Mapping)
        or gpu.get("capacity_mib") != GPU_CAPACITY_MIB
        or gpu.get("headroom_fraction") != HEADROOM_FRACTION
        or not isinstance(gpu.get("name"), str)
        or "V100" not in gpu["name"]
        or not 32_000 <= _positive(gpu.get("total_memory_mib"), "GPU memory") <= 33_000
    ):
        raise _error("benchmark GPU contract to describe one V100-32GB", gpu)
    return dict(benchmark)


def _deterministic_waves(entry_count: int, concurrency: int) -> list[dict[str, Any]]:
    waves = []
    for start in range(0, entry_count, concurrency):
        indices = list(range(start, min(start + concurrency, entry_count)))
        waves.append(
            {
                "wave_index": len(waves),
                "entry_indices": indices,
                "concurrency": len(indices),
            }
        )
    return waves


def build_v6_pack_manifest(
    stage_manifest: Mapping[str, Any],
    *,
    stage_manifest_sha256: str,
    benchmark: Mapping[str, Any] | None = None,
    benchmark_sha256: str | None = None,
) -> dict[str, Any]:
    entries = validate_stage_manifest(stage_manifest)
    stage_name = str(stage_manifest["stage_name"])
    stage_digest = _sha256_digest(
        stage_manifest_sha256, "stage_manifest_sha256"
    )
    if stage_name == "audit":
        concurrency = 1
        benchmark_binding = None
    elif stage_name == "probe":
        concurrency = 3
        benchmark_binding = None
    else:
        if benchmark is None or benchmark_sha256 is None:
            raise _error(
                f"{stage_name} pack creation to receive concurrency benchmark evidence",
                benchmark,
            )
        validated = validate_benchmark_report(
            benchmark, expected_study_id=stage_manifest["study_id"]
        )
        if (
            stage_name == "baseline_candidates"
            and validated["stage_manifest_sha256"] != stage_digest
        ):
            raise _error(
                "baseline_candidates benchmark to bind the current stage manifest",
                {
                    "benchmark": validated["stage_manifest_sha256"],
                    "current": stage_digest,
                },
            )
        concurrency = int(validated["selection"]["selected_concurrency"])
        benchmark_binding = {
            "schema_version": BENCHMARK_SCHEMA_VERSION,
            "sha256": _sha256_digest(benchmark_sha256, "benchmark_sha256"),
            "source_stage_name": validated["stage_name"],
            "source_stage_manifest_sha256": validated[
                "stage_manifest_sha256"
            ],
            "selected_concurrency": concurrency,
            "projected_full_sweep_seconds": validated["selection"][
                "projected_full_sweep_seconds"
            ],
        }
        if stage_name == "confirmations" and concurrency < 2:
            raise RuntimeError(
                "Expected the measured v6 concurrency to support both confirmation "
                "rows together. Provided selected concurrency: 1."
            )
        concurrency = min(concurrency, len(entries))
    waves = _deterministic_waves(len(entries), concurrency)
    flattened = [index for wave in waves for index in wave["entry_indices"]]
    if flattened != list(range(len(entries))):
        raise RuntimeError(
            "Expected deterministic waves to cover every entry exactly once. "
            f"Provided value: {flattened!r}."
        )
    return {
        "schema_version": PACK_SCHEMA_VERSION,
        "study_id": stage_manifest["study_id"],
        "stage_name": stage_name,
        "stage_manifest_sha256": stage_digest,
        "benchmark_binding": benchmark_binding,
        "r3_allocation": r3_allocation_contract(),
        "gpu_contract": {
            "constraint": GPU_CONSTRAINT,
            "gpus": 1,
            "capacity_mib": GPU_CAPACITY_MIB,
            "headroom_fraction": HEADROOM_FRACTION,
        },
        "execution_contract": {
            "one_slurm_job": True,
            "sequential_waves": True,
            "concurrent_within_wave": True,
            "selected_concurrency": concurrency,
            "entry_count": len(entries),
            "wave_count": len(waves),
            "login_node_finalization": True,
        },
        "waves": waves,
    }


def create_v6_pack_manifest(
    stage_manifest_path: str | Path,
    output_path: str | Path,
    *,
    benchmark_path: str | Path | None = None,
) -> dict[str, Any]:
    stage_path = Path(stage_manifest_path).expanduser().resolve()
    output = Path(output_path).expanduser().resolve()
    stage_bytes = stage_path.read_bytes()
    stage_manifest = json.loads(stage_bytes)
    benchmark = None
    benchmark_digest = None
    if benchmark_path is not None:
        benchmark_file = Path(benchmark_path).expanduser().resolve()
        benchmark_bytes = benchmark_file.read_bytes()
        benchmark = json.loads(benchmark_bytes)
        benchmark_digest = hashlib.sha256(benchmark_bytes).hexdigest()
    manifest = build_v6_pack_manifest(
        stage_manifest,
        stage_manifest_sha256=hashlib.sha256(stage_bytes).hexdigest(),
        benchmark=benchmark,
        benchmark_sha256=benchmark_digest,
    )
    if output.exists():
        if read_json(output) != manifest:
            raise RuntimeError(
                "Expected an existing content-bound v6 pack manifest to be identical. "
                f"Provided value: {output}."
            )
    else:
        atomic_write_json(output, manifest, canonical=True)
    return manifest


def validate_v6_pack_manifest_files(
    stage_manifest_path: str | Path,
    pack_manifest_path: str | Path,
    *,
    benchmark_path: str | Path | None = None,
) -> dict[str, Any]:
    """Rebuild a v2 pack from its bound inputs and require byte-meaning equality."""

    stage_path = Path(stage_manifest_path).expanduser().resolve()
    pack_path = Path(pack_manifest_path).expanduser().resolve()
    stage_bytes = stage_path.read_bytes()
    stage = json.loads(stage_bytes)
    pack = read_json(pack_path)
    if not isinstance(pack, Mapping) or pack.get("schema_version") != PACK_SCHEMA_VERSION:
        raise _error(
            f"pack schema_version to be {PACK_SCHEMA_VERSION!r}",
            None if not isinstance(pack, Mapping) else pack.get("schema_version"),
        )
    binding = pack.get("benchmark_binding")
    benchmark = None
    benchmark_digest = None
    if binding is not None:
        if benchmark_path is None:
            raise _error(
                "a benchmark path for a training-stage pack manifest", benchmark_path
            )
        benchmark_file = Path(benchmark_path).expanduser().resolve()
        benchmark_bytes = benchmark_file.read_bytes()
        benchmark = json.loads(benchmark_bytes)
        benchmark_digest = hashlib.sha256(benchmark_bytes).hexdigest()
    elif benchmark_path is not None:
        raise _error(
            "no benchmark path for audit/probe pack manifests", benchmark_path
        )
    expected = build_v6_pack_manifest(
        stage,
        stage_manifest_sha256=hashlib.sha256(stage_bytes).hexdigest(),
        benchmark=benchmark,
        benchmark_sha256=benchmark_digest,
    )
    if dict(pack) != expected:
        raise RuntimeError(
            "Expected the v6 pack manifest to match its immutable stage and benchmark "
            f"inputs. Provided value: {pack_path}."
        )
    return dict(pack)


__all__ = [
    "BENCHMARK_SCHEMA_VERSION",
    "CANONICAL_STEPS_PER_ENTRY",
    "CONCURRENCY_LEVELS",
    "GPU_CAPACITY_MIB",
    "MEASURED_STEPS",
    "PACK_SCHEMA_VERSION",
    "R3_ALLOCATION_ID",
    "R3_AUTHORIZATION_TOKEN",
    "WARMUP_STEPS",
    "build_benchmark_report",
    "build_v6_pack_manifest",
    "create_v6_pack_manifest",
    "find_safe_baseline_entry",
    "r3_allocation_contract",
    "select_concurrency",
    "validate_benchmark_report",
    "validate_r3_allocation_contract",
    "validate_stage_manifest",
    "validate_v6_pack_manifest_files",
]
