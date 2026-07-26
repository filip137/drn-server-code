from __future__ import annotations

import pytest

from experiments.mnist_conv.lr_v5_memory_preflight import (
    RUNTIME_KEYS,
    build_v5_memory_preflight,
    v5_runtime_groups,
)
from experiments.mnist_conv.lr_v5_packing import validate_memory_preflight


def _candidate_manifest() -> dict:
    entries = []
    role_scale = {"lower": 1.0, "center": 3.0, "upper": 9.0}
    scheme_scale = {"baseline": 1.0, "ours": 2.0, "legacy": 3.0}
    index = 0
    for architecture in ("conv1", "conv2"):
        for arm in ("strict_equal", "historical_profile"):
            for alpha_role in ("lower", "center", "upper"):
                for scheme in ("baseline", "ours", "legacy"):
                    scale = role_scale[alpha_role] * scheme_scale[scheme]
                    if arm == "historical_profile":
                        scale *= 10.0
                    rates = {
                        "ConvWeight_0": scale,
                        "Bias_0": scale,
                        "DenseWeight_0": scale * 2.0,
                    }
                    if architecture == "conv2":
                        rates.update({"ConvWeight_1": scale / 2.0, "Bias_1": scale / 2.0})
                    entries.append(
                        {
                            "entry_index": index,
                            "entry_id": (
                                f"{architecture}_{scheme}--{arm}--{alpha_role}"
                            ),
                            "payload": {
                                "architecture": architecture,
                                "scheme": scheme,
                                "row_id": f"{architecture}_{scheme}_row",
                                "arm": arm,
                                "alpha_role": alpha_role,
                                "learning_rates_by_parameter": rates,
                            },
                        }
                    )
                    index += 1
    return {
        "schema_version": "mnist-conv-lr-stage-manifest/v1",
        "study_id": "lrstudy_" + "a" * 64,
        "stage_name": "candidates",
        "entries": entries,
    }


def _measurements() -> dict[str, dict[str, float]]:
    return {
        key: {
            "torch_peak_allocated_mib": 7000.0 + index,
            "torch_peak_reserved_mib": 8000.0 + index,
            "maximum_sampled_device_used_mib": 8500.0 + index,
            "maximum_sampled_non_torch_overhead_mib": 500.0,
            "conservative_peak_gpu_memory_mib": 8500.0 + index,
        }
        for index, key in enumerate(RUNTIME_KEYS)
    }


def test_runtime_groups_cover_six_candidates_and_choose_stable_smallest_lr() -> None:
    stage = _candidate_manifest()
    groups = v5_runtime_groups(stage)

    assert [group["runtime_key"] for group in groups] == list(RUNTIME_KEYS)
    assert all(len(group["member_entry_indices"]) == 6 for group in groups)
    for group in groups:
        representative = stage["entries"][group["representative_entry_index"]]
        assert representative["payload"]["arm"] == "strict_equal"
        assert representative["payload"]["alpha_role"] == "lower"
        assert group["maximum_representative_parameter_lr"] == max(
            representative["payload"]["learning_rates_by_parameter"].values()
        )


def test_memory_measurements_expand_to_all_36_pack_entries() -> None:
    stage = _candidate_manifest()
    digest = "b" * 64
    result = build_v5_memory_preflight(
        stage,
        stage_manifest_sha256=digest,
        measurements_by_runtime_key=_measurements(),
        steps_per_runtime=3,
        device={"name": "Tesla V100-SXM2-32GB", "total_memory_mib": 32510.5},
    )

    assert result["measurement_contract"]["official_test_read"] is False
    assert result["measurement_contract"]["validation_read"] is False
    assert len(result["representative_measurements"]) == 6
    peaks = validate_memory_preflight(
        result,
        stage_manifest_sha256=digest,
        entry_count=36,
    )
    assert set(peaks) == set(range(36))
    for group in v5_runtime_groups(stage):
        expected = _measurements()[group["runtime_key"]][
            "conservative_peak_gpu_memory_mib"
        ]
        assert {
            peaks[index] for index in group["member_entry_indices"]
        } == {expected}


def test_memory_preflight_rejects_missing_or_underreported_runtime_measurement() -> None:
    stage = _candidate_manifest()
    missing = _measurements()
    missing.pop("conv2:legacy")
    with pytest.raises(ValueError, match="measurements_by_runtime_key"):
        build_v5_memory_preflight(
            stage,
            stage_manifest_sha256="b" * 64,
            measurements_by_runtime_key=missing,
            steps_per_runtime=3,
            device={"name": "V100"},
        )

    underreported = _measurements()
    underreported["conv1:baseline"]["conservative_peak_gpu_memory_mib"] = 8000.0
    with pytest.raises(ValueError, match="conservative peak to cover"):
        build_v5_memory_preflight(
            stage,
            stage_manifest_sha256="b" * 64,
            measurements_by_runtime_key=underreported,
            steps_per_runtime=3,
            device={"name": "V100"},
        )
