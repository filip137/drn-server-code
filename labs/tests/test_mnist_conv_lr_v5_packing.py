from __future__ import annotations

from experiments.mnist_conv.lr_v5_packing import build_v5_pack_manifest


def _manifest() -> dict:
    entries = []
    schemes = ("baseline", "ours", "legacy")
    index = 0
    for architecture in ("conv1", "conv2"):
        for arm in ("strict_equal", "historical_profile"):
            for alpha_role in ("lower", "center", "upper"):
                for scheme in schemes:
                    entries.append(
                        {
                            "entry_index": index,
                            "entry_id": f"{architecture}_{scheme}--{arm}--{alpha_role}",
                            "payload": {
                                "architecture": architecture,
                                "row_id": f"{architecture}_{scheme}_row",
                                "arm": arm,
                                "alpha_role": alpha_role,
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


def test_v5_packing_prefers_three_runs_when_headroom_fits() -> None:
    manifest = build_v5_pack_manifest(
        _manifest(),
        stage_manifest_sha256="b" * 64,
        peak_gpu_memory_mib_by_entry_index={index: 9000.0 for index in range(36)},
    )

    assert len(manifest["packs"]) == 12
    assert {len(pack["entry_indices"]) for pack in manifest["packs"]} == {3}
    assert [pack["pack_index"] for pack in manifest["packs"]] == list(range(12))
    assert sorted(
        index for pack in manifest["packs"] for index in pack["entry_indices"]
    ) == list(range(36))
    assert all(
        pack["measured_combined_peak_gpu_memory_mib"] * 1.1 <= 32768
        for pack in manifest["packs"]
    )


def test_v5_packing_falls_back_to_two_runs_and_never_mixes_architectures() -> None:
    stage = _manifest()
    manifest = build_v5_pack_manifest(
        stage,
        stage_manifest_sha256="b" * 64,
        peak_gpu_memory_mib_by_entry_index={index: 11000.0 for index in range(36)},
    )

    assert len(manifest["packs"]) == 18
    assert {len(pack["entry_indices"]) for pack in manifest["packs"]} == {2}
    for pack in manifest["packs"]:
        assert {
            stage["entries"][index]["payload"]["architecture"]
            for index in pack["entry_indices"]
        } == {pack["architecture"]}
