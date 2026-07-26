"""Deterministic routing policy for Conv3 v7 candidate manifests."""

from __future__ import annotations

from typing import Any, Mapping, Sequence


LOCAL_AVAILABILITY_CHECK_ORDER = ("main", "akibscomputer", "trex")
CONV3_DISPATCH_PRIORITY = ("trex", "jean_zay_r3", "main", "akibscomputer")


def _integer(value: Any, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(
            f"Expected {label} to be an integer >= {minimum}. "
            f"Provided value: {value!r}."
        )
    return value


def normalize_local_gpu_availability(
    value: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    """Require a measured availability record for every local tmux target."""

    if not isinstance(value, Mapping):
        raise ValueError(
            "Expected local GPU availability to be a mapping. "
            f"Provided value: {value!r}."
        )
    if set(value) != set(LOCAL_AVAILABILITY_CHECK_ORDER):
        raise ValueError(
            "Expected local GPU availability for exactly "
            f"{list(LOCAL_AVAILABILITY_CHECK_ORDER)!r}. "
            f"Provided value: {sorted(value)!r}."
        )
    normalized: dict[str, dict[str, Any]] = {}
    for target in LOCAL_AVAILABILITY_CHECK_ORDER:
        record = value[target]
        if not isinstance(record, Mapping):
            raise ValueError(
                f"Expected availability.{target} to be a mapping. "
                f"Provided value: {record!r}."
            )
        if record.get("checked") is not True:
            raise ValueError(
                f"Expected availability.{target}.checked to be true. "
                f"Provided value: {record.get('checked')!r}."
            )
        reachable = record.get("reachable")
        if not isinstance(reachable, bool):
            raise ValueError(
                f"Expected availability.{target}.reachable to be a boolean. "
                f"Provided value: {reachable!r}."
            )
        slots = _integer(
            record.get("free_gpu_slots"),
            f"availability.{target}.free_gpu_slots",
        )
        if not reachable and slots:
            raise ValueError(
                f"Expected unreachable target {target!r} to have zero free GPU "
                f"slots. Provided value: {slots!r}."
            )
        normalized[target] = {
            "checked": True,
            "reachable": reachable,
            "free_gpu_slots": slots,
        }
    return normalized


def route_v7_candidate_entries(
    entry_ids: Sequence[str],
    *,
    local_availability: Mapping[str, Any],
    packing_benchmark: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Route Trex first and every remaining candidate to Jean Zay R3.

    Main and Akib are still required availability checks, but the frozen Conv3
    preference keeps Jean Zay ahead of those targets. Packing remains disabled
    unless a separate benchmark explicitly passes both memory and throughput.
    """

    if not isinstance(entry_ids, Sequence) or isinstance(entry_ids, (str, bytes)):
        raise ValueError(
            "Expected entry_ids to be a sequence of unique non-empty strings. "
            f"Provided value: {entry_ids!r}."
        )
    ids = [str(value).strip() for value in entry_ids]
    if any(not value for value in ids) or len(ids) != len(set(ids)):
        raise ValueError(
            "Expected entry_ids to be unique non-empty strings. "
            f"Provided value: {entry_ids!r}."
        )
    availability = normalize_local_gpu_availability(local_availability)
    processes_per_gpu = 1
    packing_enabled = False
    if packing_benchmark is not None:
        if not isinstance(packing_benchmark, Mapping):
            raise ValueError(
                "Expected packing_benchmark to be a mapping or null. "
                f"Provided value: {packing_benchmark!r}."
            )
        requested = _integer(
            packing_benchmark.get("processes_per_gpu"),
            "packing_benchmark.processes_per_gpu",
            minimum=1,
        )
        memory_passed = packing_benchmark.get("memory_passed") is True
        throughput_passed = packing_benchmark.get("throughput_passed") is True
        if requested > 1 and not (memory_passed and throughput_passed):
            raise ValueError(
                "Expected multi-process packing to have passed measured memory "
                "and throughput benchmarks. "
                f"Provided value: {dict(packing_benchmark)!r}."
            )
        processes_per_gpu = requested
        packing_enabled = requested > 1

    trex_slots = (
        availability["trex"]["free_gpu_slots"]
        if availability["trex"]["reachable"]
        else 0
    )
    trex_count = min(len(ids), trex_slots * processes_per_gpu)
    trex_entries = ids[:trex_count]
    jean_zay_entries = ids[trex_count:]
    return {
        "schema_version": "mnist-conv-lr-v7-routing/v1",
        "availability_check_order": list(LOCAL_AVAILABILITY_CHECK_ORDER),
        "dispatch_priority": list(CONV3_DISPATCH_PRIORITY),
        "local_availability": availability,
        "processes_per_gpu": processes_per_gpu,
        "packing_enabled": packing_enabled,
        "routes": [
            {
                "target": "trex",
                "entry_ids": trex_entries,
                "entry_count": len(trex_entries),
            },
            {
                "target": "jean_zay_r3",
                "account": "fmu@v100",
                "constraint": "v100-32g",
                "entry_ids": jean_zay_entries,
                "entry_count": len(jean_zay_entries),
            },
        ],
        "secondary_local_targets_checked_but_not_preferred": [
            "main",
            "akibscomputer",
        ],
    }


__all__ = [
    "CONV3_DISPATCH_PRIORITY",
    "LOCAL_AVAILABILITY_CHECK_ORDER",
    "normalize_local_gpu_availability",
    "route_v7_candidate_entries",
]
