from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path

import pytest

from experiments.exact_run import load_exact_config
from experiments.prepare_bounded_wmax_sweep import (
    ARCHITECTURE_TARGETS,
    EVIDENCE_CLASS,
    PARENT_COMMIT,
    STUDY_ID,
    WEIGHT_MIN,
    WMAX_CASES,
    load_parent_config,
    prepare,
)
from experiments.validate_bounded_wmax_run import validate_config_set


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _parent_equivalent(generated: dict, parent: dict) -> None:
    restored = deepcopy(generated)
    restored.pop("wmax_sweep")
    restored["model_base"]["weight_max"] = parent["model_base"]["weight_max"]
    for key in ("arm_id", "study_id", "reporting"):
        restored[key] = deepcopy(parent[key])
    assert restored == parent


def test_prepare_bounded_wmax_sweep_preserves_parent_science(
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "configs"
    manifest = prepare(output_root)

    assert manifest["study_id"] == STUDY_ID
    assert manifest["run_count"] == 90
    assert manifest["weight_min"] == WEIGHT_MIN
    assert manifest["weight_max_values"] == [case.value for case in WMAX_CASES]
    assert manifest["parent"]["source_commit"] == PARENT_COMMIT
    assert manifest["parent"]["run_count"] == 36
    assert manifest["target_split"] == ARCHITECTURE_TARGETS
    assert manifest["no_scientific_safety_or_rho_runs"] is True
    assert [row["global_array_index"] for row in manifest["runs"]] == list(
        range(90)
    )

    generated_json = {
        path.relative_to(output_root).as_posix()
        for path in output_root.rglob("*.json")
    }
    expected_json = {row["config"] for row in manifest["runs"]} | {
        "study_manifest.json"
    }
    assert generated_json == expected_json
    assert len(generated_json) == 91
    assert prepare(output_root, check=True) == manifest

    rows_by_architecture = {
        architecture: [
            row
            for row in manifest["runs"]
            if row["architecture"] == architecture
        ]
        for architecture in ARCHITECTURE_TARGETS
    }
    assert {key: len(value) for key, value in rows_by_architecture.items()} == {
        "conv1": 30,
        "conv2": 30,
        "conv3": 30,
    }

    for row in manifest["runs"]:
        path = output_root / row["config"]
        assert _sha256(path) == row["config_sha256"]
        generated = load_exact_config(path)
        parent_filename = Path(row["parent_config"]).name
        parent, parent_bytes = load_parent_config(
            row["architecture"], parent_filename
        )

        assert hashlib.sha256(parent_bytes).hexdigest() == row[
            "parent_config_sha256"
        ]
        _parent_equivalent(generated, parent)
        assert generated["study_id"] == STUDY_ID
        assert generated["reporting"]["evidence_class"] == EVIDENCE_CLASS
        assert generated["reporting"]["paper_facing"] is False
        assert generated["model_base"]["weight_min"] == WEIGHT_MIN
        assert generated["model_base"]["weight_max"] == row["weight_max"]
        assert generated["model_base"]["weight_init_mode"] == "bounded_uniform"
        assert generated.get("init_checkpoint_path") in (None, "")
        assert generated["seed"] == 0
        assert generated["lr"] == parent["lr"]
        assert generated["optimizer"]["learning_rate"] == parent["optimizer"][
            "learning_rate"
        ]
        assert generated["learning_rates_by_parameter"] == parent[
            "learning_rates_by_parameter"
        ]
        assert generated["parameter_order"] == parent["parameter_order"]
        assert generated["evaluation"]["official_test"]["policy"] == "disabled"
        assert generated["wmax_sweep"]["changed_scientific_fields"] == [
            "model_base.weight_max"
        ]
        assert generated["wmax_sweep"]["initialization_policy"][
            "matched_random_quantiles"
        ] is True


def test_bounded_wmax_sweep_keeps_matched_surface_contracts(
    tmp_path: Path,
) -> None:
    manifest = prepare(tmp_path / "configs")

    for architecture in ARCHITECTURE_TARGETS:
        rows = [
            row
            for row in manifest["runs"]
            if row["architecture"] == architecture
        ]
        for scheme in ("baseline", "ours", "legacy"):
            for optimizer in ("sgd", "adam"):
                surface = [
                    row
                    for row in rows
                    if row["scheme"] == scheme and row["optimizer"] == optimizer
                ]
                assert len(surface) == 5
                assert [row["weight_max"] for row in surface] == [
                    case.value for case in WMAX_CASES
                ]
                assert len(
                    {
                        json.dumps(
                            row["learning_rates_by_parameter"],
                            sort_keys=True,
                        )
                        for row in surface
                    }
                ) == 1

    conv3_baseline = [
        row
        for row in manifest["runs"]
        if row["architecture"] == "conv3" and row["scheme"] == "baseline"
    ]
    conv3_amplified = [
        row
        for row in manifest["runs"]
        if row["architecture"] == "conv3" and row["scheme"] != "baseline"
    ]
    assert {(row["T"], row["K"]) for row in conv3_baseline} == {(12, 8)}
    assert {(row["T"], row["K"]) for row in conv3_amplified} == {(8, 8)}


def test_static_validator_binds_each_architecture_config_set(
    tmp_path: Path,
) -> None:
    config_root = tmp_path / "configs"
    manifest = prepare(config_root)

    for architecture in ARCHITECTURE_TARGETS:
        receipt = validate_config_set(
            config_root,
            architecture=architecture,
            expected_config_set_sha256=manifest[
                "ordered_config_set_sha256_by_architecture"
            ][architecture],
        )
        assert receipt["semantic_status"] == "pass"
        assert receipt["architecture"] == architecture
        assert receipt["config_count"] == 30


def test_prepare_check_rejects_changed_and_stale_files(tmp_path: Path) -> None:
    output_root = tmp_path / "configs"
    manifest = prepare(output_root)
    first_config = output_root / manifest["runs"][0]["config"]
    original = first_config.read_bytes()
    first_config.write_text("{}\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match='"changed"'):
        prepare(output_root, check=True)

    first_config.write_bytes(original)
    stale = output_root / "conv1" / "orphan.json"
    stale.write_text("{}\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match='"stale"'):
        prepare(output_root, check=True)
    with pytest.raises(RuntimeError, match="Refusing to leave stale files"):
        prepare(output_root)
