from __future__ import annotations

import json
from pathlib import Path

import pytest

import experiments.run_conv12_directed_exploratory_lr_search as directed


STUDY_PATH = (
    Path(__file__).resolve().parents[2]
    / "configs/conv/perfectdiode_conv12_directed_exploratory_lr_search_seed0_20260812_v1.json"
)


def _loaded():
    return directed.load_study(STUDY_PATH)


def test_contract_has_all_surfaces_257_new_cells_and_no_prior_numeric_overlap() -> None:
    _path, study, _parent_path, parent = _loaded()
    surfaces = directed.surface_specs(study)

    assert parent["study_id"] == directed.EXPECTED_PARENT_STUDY_ID
    assert len(surfaces) == 12
    assert sum(len(directed.surface_cells(study, row)) for row in surfaces) == 257
    assert [row["expected_cells"] for row in surfaces] == [
        39,
        30,
        18,
        16,
        20,
        6,
        30,
        40,
        17,
        17,
        14,
        10,
    ]
    for surface in surfaces:
        new_pairs = {
            (row["rho_conv_ladder_index"], row["rho_dense_ladder_index"])
            for row in directed.surface_cells(study, surface)
        }
        prior_pairs = set(
            directed._block_pairs(
                study["prior_numeric_blocks"][directed._surface_key(surface)],
                conv_count=11,
                dense_count=10,
                label="test-prior",
            )
        )
        assert new_pairs.isdisjoint(prior_pairs)


def test_sparse_cartesian_indices_match_each_surface_axes_without_collisions() -> None:
    _path, study, _parent_path, _parent = _loaded()
    for surface in directed.surface_specs(study):
        cells = directed.surface_cells(study, surface)
        conv, dense = directed.surface_axes(study, surface)
        assert len({row["index"] for row in cells}) == len(cells)
        assert all(0 <= row["index"] < len(conv) * len(dense) for row in cells)
        for row in cells:
            conv_position = conv.index(row["rho_conv"])
            dense_position = dense.index(row["rho_dense"])
            assert row["index"] == conv_position * len(dense) + dense_position


def test_only_three_deliberate_retries_lack_prior_numeric_results() -> None:
    _path, study, _parent_path, _parent = _loaded()
    assert study["retry_non_numeric_coordinates"] == {
        "conv2__baseline__adam": [[5, 5], [5, 6], [5, 7]]
    }
    surface = directed.surface_specs(study)[9]
    pairs = {
        (row["rho_conv_ladder_index"], row["rho_dense_ladder_index"])
        for row in directed.surface_cells(study, surface)
    }
    assert {(5, 5), (5, 6), (5, 7)}.issubset(pairs)


def test_production_args_keep_zero_bias_and_disable_exploratory_gates(tmp_path: Path) -> None:
    _path, study, _parent_path, _parent = _loaded()
    surface = directed.surface_specs(study)[9]
    first = directed.surface_cells(study, surface)[0]
    args = directed._rho_args(
        study,
        surface,
        tmp_path / "source.json",
        tmp_path / "rho",
        device="cuda",
        target="jean-zay",
        index=first["index"],
    )

    assert args.probe_batches == [128]
    assert args.stability_tolerance == 0.5
    assert args.epochs == 3
    assert args.evidence_class == "ordinary_mnist_exploratory"
    assert args.bias_policy == "zero"
    assert args.skip_canary is True
    assert args.canary_steps == 0
    assert args.restart_interrupted_cells is True
    assert args.disable_safety_rejections is True
    assert args.post_candidate_gate_name is None
    assert args.expected_candidate_steps == 10314
    assert args.minimum_validation_accuracy == 0.0
    assert args.index == first["index"]


def test_contract_rejects_a_prior_numeric_duplicate(tmp_path: Path) -> None:
    study = json.loads(STUDY_PATH.read_text(encoding="utf-8"))
    study["surfaces"][0]["blocks"][1]["rho_dense_indices"][-1] = 5
    duplicate = tmp_path / "study.json"
    duplicate.write_text(json.dumps(study), encoding="utf-8")

    with pytest.raises(ValueError, match="repeats prior numeric pairs"):
        directed.load_study(duplicate)
