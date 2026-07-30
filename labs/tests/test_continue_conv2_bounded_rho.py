from __future__ import annotations

import copy
import json
from argparse import Namespace
from pathlib import Path

import pytest

from experiments.continue_conv2_bounded_rho import (
    DEFAULT_STUDY,
    _seed_imported_probe,
    continuation_axes_and_pairs,
    load_study,
    surface_specs,
)


def test_continuation_config_declares_seven_expansions_and_35_cells() -> None:
    _path, study, _base_path, _base = load_study(DEFAULT_STUDY)
    surfaces = surface_specs(study)

    assert len(surfaces) == 8
    assert sum(bool(surface["expand"]) for surface in surfaces) == 7
    assert sum(surface["expected_new_cells"] for surface in surfaces) == 35
    assert {surface["target"] for surface in surfaces} == {"main", "akib", "trex"}
    assert all("legacy" not in surface["surface_id"] for surface in surfaces)


@pytest.mark.parametrize(
    ("expand", "expected"),
    [
        ({"rho_conv": "upper"}, 4),
        ({"rho_dense": "lower"}, 4),
        ({"rho_conv": "lower", "rho_dense": "lower"}, 9),
    ],
)
def test_continuation_adds_only_the_new_cartesian_border(
    expand: dict[str, str],
    expected: int,
) -> None:
    parent = {
        "expansion": {
            "expanded_axes": {
                "rho_conv": [0.001, 0.003, 0.009, 0.027],
                "rho_dense": [0.001111111111, 0.003333333333, 0.01, 0.03],
            }
        }
    }

    axes, pairs = continuation_axes_and_pairs(parent, expand, 3.0)

    assert len(pairs) == expected
    if expand.get("rho_conv") == "upper":
        assert axes["rho_conv"][-1] == pytest.approx(0.081)
    if expand.get("rho_conv") == "lower":
        assert axes["rho_conv"][0] == pytest.approx(0.001 / 3.0)
    if expand.get("rho_dense") == "lower":
        assert axes["rho_dense"][0] == pytest.approx(0.001111111111 / 3.0)


def test_imported_probe_is_rebound_to_continuation_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source.json"
    source.write_text("{}\n", encoding="utf-8")
    contract = {
        "source_config_sha256": "source",
        "optimizer": "Adam",
        "bias_policy": "q90_cap",
        "probe_batches": [32, 64, 128],
        "stability_tolerance": 0.1,
        "split_seed": 0,
        "shuffle_seed": 0,
        "validation_batch_size": 64,
        "rho_conv": [0.001],
        "rho_dense": [0.01],
    }
    parent = {
        "probe_stable": True,
        "normalization_unit_by_weight": {"ConvWeight_0": 1.0},
        "search_signature": copy.deepcopy(contract),
    }
    parent_path = tmp_path / "parent_probe.json"
    parent_path.write_text(json.dumps(parent), encoding="utf-8")
    args = Namespace()
    monkeypatch.setattr(
        "experiments.continue_conv2_bounded_rho._search_contract",
        lambda _args: copy.deepcopy(contract),
    )

    imported = _seed_imported_probe(
        args,
        parent_path,
        tmp_path / "continuation",
        surface_id="surface",
    )

    assert imported["search_signature"] == contract
    assert imported["import_provenance"]["sha256"]
    assert json.loads(
        (tmp_path / "continuation" / "resolved.json").read_text()
    ) == contract
