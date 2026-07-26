from __future__ import annotations

import copy
import csv
import hashlib
import json
from pathlib import Path
from typing import Callable

import pytest

from experiments.mnist_conv.lr_protocol import (
    parameter_relative_update_from_rms,
    probe_parameter_relative_rho_unit,
    range_normalized_update_schedule,
    target_learning_rates,
)
from experiments.mnist_conv.lr_study import plan_only_result
from experiments.mnist_conv.lr_study_spec import (
    FROZEN_RELATIVE_RHO_ROWS,
    LRStudySpec,
)
from experiments.mnist_conv.specs import SpecValidationError


REPO_ROOT = Path(__file__).resolve().parents[2]
V1_STUDY_CONFIG = REPO_ROOT / "configs/conv/hardsigmoid_lr_study_sgd_bs16_v1.json"
V2_STUDY_CONFIG = (
    REPO_ROOT / "configs/conv/hardsigmoid_lr_rescue_warmin_sgd_bs16_v2.json"
)
V3_STUDY_CONFIG = (
    REPO_ROOT / "configs/conv/hardsigmoid_lr_relative_rho_sgd_bs16_v3.json"
)

V1_STUDY_ID = (
    "lrstudy_2103c850c10540d5628cd20cbebc0af5"
    "19bd871116062e8b80e50ad0c068a9a4"
)
V2_STUDY_ID = (
    "lrstudy_aa0a5d954056594099cf76ed49a8ef15"
    "9f9e707d207b0784fc11f44212f9b0a6"
)
V3_STUDY_ID = (
    "lrstudy_b2cffbb9c9976c58338141fc179fba2f"
    "f4f37b3d02009ef9255e3d5a21d650d2"
)

V3_ROW_IDS = (
    "conv1_ours_v4_c1",
    "conv1_legacy_v4_c0p25",
    "conv2_ours_v4_c1",
    "conv2_legacy_v4_c0p25",
)
V3_RHO_TARGETS = (1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0)
V3_TARGET_CROSSING_STEPS = {
    1e-3: 106,
    3e-3: 201,
    1e-2: 306,
    3e-2: 401,
    1e-1: 505,
    3e-1: 601,
    1.0: 705,
}

V2_RESULT_ROOT = (
    REPO_ROOT
    / "simulation_results/conv_lr_protocol_20260719/lr_studies"
    / (
        "conv1-hard-sigmoid-amplified-lr-warm-in-rescue--"
        "lrstudy_aa0a5d954056594099cf76ed49a8ef159f9e707d207b0784fc11f44212f9b0a6"
    )
)
V2_CONV1_ORACLES = {
    "conv1_ours_v4_c1": {
        "probe_sha256": (
            "960f3e023adfd0e29ddf6bee7cc9301ef03fa73d300d850e0291146aa1fc1d19"
        ),
        "range_summary_sha256": (
            "a3383bf58dd9ccc79857f81adbc99dd89d420747b07515702ec0096949b90594"
        ),
        "rho_unit": 16.318411279982637,
        "by_parameter": {
            "ConvWeight_0": 0.43859869364286863,
            "DenseWeight_0": 16.318411279982637,
        },
    },
    "conv1_legacy_v4_c0p25": {
        "probe_sha256": (
            "f630024329865ce470d12159e5c7b2dc8594e37ac06c2a4d976f4267f88d0144"
        ),
        "range_summary_sha256": (
            "0461c3648891ee10a54c6e9ad4531a70dca8552b4c66250c523139d2954dc7c7"
        ),
        "rho_unit": 215.5583558791094,
        "by_parameter": {
            "ConvWeight_0": 5.521915379423545,
            "DenseWeight_0": 215.5583558791094,
        },
    },
}


def _v3_value() -> dict:
    return json.loads(V3_STUDY_CONFIG.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_v3_has_an_exact_new_identity_and_does_not_change_v1_or_v2() -> None:
    v1 = LRStudySpec.from_path(V1_STUDY_CONFIG)
    v2 = LRStudySpec.from_path(V2_STUDY_CONFIG)
    v3 = LRStudySpec.from_path(V3_STUDY_CONFIG)

    assert (v1.data["schema_version"], v1.study_id) == (
        "mnist-conv-lr-study/v1",
        V1_STUDY_ID,
    )
    assert (v2.data["schema_version"], v2.study_id) == (
        "mnist-conv-lr-study/v2",
        V2_STUDY_ID,
    )
    assert (v3.data["schema_version"], v3.study_id) == (
        "mnist-conv-lr-study/v3",
        V3_STUDY_ID,
    )
    assert v3.data["protocol_id"] == "conv-hardsigmoid-lr-sgd-bs16-relative-rho-v3"


def test_v3_scope_is_exactly_the_four_unresolved_amplified_rows() -> None:
    spec = LRStudySpec.from_path(V3_STUDY_CONFIG)

    assert tuple(row["row_id"] for row in spec.rows) == V3_ROW_IDS
    assert spec.rows == list(FROZEN_RELATIVE_RHO_ROWS)
    assert [(row["architecture"], row["scheme"]) for row in spec.rows] == [
        ("conv1", "ours"),
        ("conv1", "legacy"),
        ("conv2", "ours"),
        ("conv2", "legacy"),
    ]
    assert all(row["scheme"] != "baseline" for row in spec.rows)
    assert spec.data["rescue"]["eligibility"]["included_row_ids"] == list(
        V3_ROW_IDS
    )
    assert spec.data["rescue"]["eligibility"]["baseline_rows_reused_not_rerun"]


def test_v3_relative_rho_target_grid_range_endpoints_and_crossings() -> None:
    spec = LRStudySpec.from_path(V3_STUDY_CONFIG)
    probe = spec.data["probe"]
    range_test = spec.data["range_test"]

    assert probe["rho_definition"] == "initial_parameter_rms"
    assert probe["aggregation"] == {
        "within_parameter_statistic": "quantile",
        "within_parameter_q": 0.9,
        "within_parameter_interpolation": "linear",
        "across_parameters": "maximum",
        "architecture_semantics": "worst_bounded_parameter_q90",
    }
    assert tuple(probe["rho_targets"]) == V3_RHO_TARGETS
    assert (range_test["steps"], range_test["start_rho_target"], range_test["end_rho_target"]) == (
        800,
        3e-4,
        3.0,
    )

    main = range_normalized_update_schedule(
        main_steps=range_test["steps"],
        start_rho=range_test["start_rho_target"],
        end_rho=range_test["end_rho_target"],
    )
    assert len(main) == 800
    assert main[0] == 3e-4
    assert main[-1] == 3.0
    assert main[31] == pytest.approx(0.0004288597096388198, rel=1e-15)

    crossings = {
        target: next(index for index, rho in enumerate(main, start=1) if rho >= target)
        for target in V3_RHO_TARGETS
    }
    assert crossings == V3_TARGET_CROSSING_STEPS
    for target, step in crossings.items():
        assert main[step - 1] >= target
        assert main[step - 2] < target

    extension = range_test["extension"]
    full = range_normalized_update_schedule(
        include_extension=True,
        main_steps=range_test["steps"],
        start_rho=range_test["start_rho_target"],
        end_rho=range_test["end_rho_target"],
        extension_steps=extension["steps"],
        extension_start_rho=extension["start_rho_target"],
        extension_end_rho=extension["end_rho_target"],
    )
    assert len(full) == 928
    assert full[799:801] == (3.0, 3.0)
    assert full[-1] == 9.0


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value.update({"schema_version": "mnist-conv-lr-study/v2"}),
        lambda value: value.update({"protocol_id": "relative-rho-v3-drift"}),
        lambda value: value["probe"].update(
            {"rho_definition": "conductance_bound_span"}
        ),
        lambda value: value["probe"]["initial_parameter_scale"].update(
            {"frozen_for_all_steps": False}
        ),
        lambda value: value["probe"]["aggregation"].update(
            {"across_parameters": "quantile"}
        ),
        lambda value: value["probe"].update(
            {"rho_targets": [1e-4, *value["probe"]["rho_targets"]]}
        ),
        lambda value: value["range_test"].update({"steps": 799}),
        lambda value: value["range_test"].update({"start_rho_target": 1e-4}),
        lambda value: value["range_test"].update({"end_rho_target": 1.0}),
        lambda value: value["range_test"]["extension"].update(
            {"end_rho_target": 10.0}
        ),
        lambda value: value["rows"][2].update({"input_gain": 716.0}),
        lambda value: value["rows"].append(copy.deepcopy(value["rows"][0])),
    ],
)
def test_v3_rejects_scientific_contract_drift(
    mutation: Callable[[dict], None],
) -> None:
    value = _v3_value()
    mutation(value)

    with pytest.raises(SpecValidationError):
        LRStudySpec.from_dict(value)


def test_v3_plan_counts_are_four_probes_four_ranges_and_twelve_candidates(
    tmp_path: Path,
) -> None:
    spec = LRStudySpec.from_path(V3_STUDY_CONFIG)
    root = tmp_path / f"study--{spec.study_id}"

    probe = plan_only_result(spec, root, "probe")
    range_test = plan_only_result(spec, root, "range")
    candidates = plan_only_result(spec, root, "candidates")
    selection = plan_only_result(spec, root, "select")

    assert (probe["entry_count"], tuple(probe["entry_ids"])) == (4, V3_ROW_IDS)
    assert (range_test["entry_count"], tuple(range_test["entry_ids"])) == (
        4,
        V3_ROW_IDS,
    )
    assert candidates["entry_count"] == 12
    assert tuple(candidates["entry_ids"]) == tuple(
        f"{row_id}--{role}"
        for row_id in V3_ROW_IDS
        for role in ("fast", "middle", "high")
    )
    assert (selection["entry_count"], selection["entry_ids"]) == (1, ["selection"])


@pytest.mark.parametrize("row_id", tuple(V2_CONV1_ORACLES))
def test_v3_conv1_relative_rho_oracle_from_immutable_v2_probe(
    row_id: str,
) -> None:
    """Re-derive the v3 Conv1 oracle without modifying the v2 evidence."""

    probe_path = (
        V2_RESULT_ROOT
        / "stages/probe/entries"
        / row_id
        / "parameter_diagnostics.csv"
    )
    summary_path = V2_RESULT_ROOT / "stages/range/entries" / row_id / "summary.json"
    if not probe_path.is_file() or not summary_path.is_file():
        pytest.skip("immutable local v2 result bundle is not present")

    oracle = V2_CONV1_ORACLES[row_id]
    assert _sha256(probe_path) == oracle["probe_sha256"]
    assert _sha256(summary_path) == oracle["range_summary_sha256"]

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    initial = summary["initial_parameter_diagnostics"]
    relative_updates: dict[str, list[float]] = {}
    with probe_path.open(newline="", encoding="utf-8") as handle:
        for record in csv.DictReader(handle):
            if record["bounded_gate"] != "True":
                continue
            name = record["parameter"]
            relative_updates.setdefault(name, []).append(
                parameter_relative_update_from_rms(
                    float(record["proposed_update_rms"]),
                    initial_parameter_rms=float(initial[name]["rms"]),
                )
            )

    assert set(relative_updates) == {"ConvWeight_0", "DenseWeight_0"}
    assert all(len(values) == 32 for values in relative_updates.values())
    rho_unit, by_parameter = probe_parameter_relative_rho_unit(relative_updates)
    assert rho_unit == pytest.approx(oracle["rho_unit"], rel=1e-14)
    assert by_parameter == pytest.approx(oracle["by_parameter"], rel=1e-14)
    assert max(by_parameter, key=by_parameter.__getitem__) == "DenseWeight_0"

    rates = target_learning_rates(rho_unit, V3_RHO_TARGETS)
    assert rates[1e-3] == pytest.approx(1e-3 / oracle["rho_unit"], rel=1e-14)
    assert rates[1.0] == pytest.approx(1.0 / oracle["rho_unit"], rel=1e-14)
