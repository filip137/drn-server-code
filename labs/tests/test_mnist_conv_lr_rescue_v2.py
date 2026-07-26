from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from experiments.mnist_conv.io import atomic_write_json
from experiments.mnist_conv.lr_study import _entries, plan_only_result
from experiments.mnist_conv.lr_study_spec import LRStudySpec
from experiments.mnist_conv.specs import SpecValidationError


REPO_ROOT = Path(__file__).resolve().parents[2]
V1_STUDY_CONFIG = REPO_ROOT / "configs/conv/hardsigmoid_lr_study_sgd_bs16_v1.json"
RESCUE_STUDY_CONFIG = (
    REPO_ROOT / "configs/conv/hardsigmoid_lr_rescue_warmin_sgd_bs16_v2.json"
)

PARENT_STUDY_ID = (
    "lrstudy_2103c850c10540d5628cd20cbebc0af5"
    "19bd871116062e8b80e50ad0c068a9a4"
)
RESCUE_ROW_IDS = (
    "conv1_ours_v4_c1",
    "conv1_legacy_v4_c0p25",
)
RESCUE_RHO_UNITS = {
    "conv1_ours_v4_c1": 0.0005425719918788645,
    "conv1_legacy_v4_c0p25": 0.007345373341757793,
}


def _range_summary(root: Path, row_id: str, *, resolved: bool) -> None:
    directory = root / "stages" / "range" / "entries" / row_id
    directory.mkdir(parents=True, exist_ok=True)
    if resolved:
        candidates = {
            "status": "resolved",
            "reason": None,
            "fast": {"rho_target": 1e-5, "learning_rate": 0.01},
            "middle": {"rho_target": 1e-4, "learning_rate": 0.1},
            "high": {"rho_target": 1e-3, "learning_rate": 1.0},
        }
    else:
        candidates = {
            "status": "unresolved",
            "reason": "fewer_than_three_stable_targets",
            "fast": None,
            "middle": None,
            "high": None,
        }
    atomic_write_json(
        directory / "summary.json",
        {"candidates": candidates},
        canonical=True,
    )


def test_rescue_v2_has_only_the_two_frozen_conv1_rows_and_v1_is_unchanged() -> None:
    original = LRStudySpec.from_path(V1_STUDY_CONFIG)
    rescue = LRStudySpec.from_path(RESCUE_STUDY_CONFIG)

    assert original.data["schema_version"] == "mnist-conv-lr-study/v1"
    assert original.data["protocol_id"] == "conv-hardsigmoid-lr-sgd-bs16-v1"
    assert len(original.rows) == 6
    assert [row["row_id"] for row in original.rows] == [
        "conv1_baseline_v1_c1",
        "conv1_ours_v4_c1",
        "conv1_legacy_v4_c0p25",
        "conv2_baseline_v1_c1",
        "conv2_ours_v4_c1",
        "conv2_legacy_v4_c0p25",
    ]

    assert rescue.data["schema_version"] == "mnist-conv-lr-study/v2"
    assert (
        rescue.data["protocol_id"]
        == "conv-hardsigmoid-lr-sgd-bs16-warm-in-rescue-v2"
    )
    assert tuple(row["row_id"] for row in rescue.rows) == RESCUE_ROW_IDS
    assert [(row["architecture"], row["scheme"]) for row in rescue.rows] == [
        ("conv1", "ours"),
        ("conv1", "legacy"),
    ]
    # Rows are the original frozen dictionaries, not hand-edited rescue variants.
    original_by_id = {row["row_id"]: row for row in original.rows}
    assert rescue.rows == [original_by_id[row_id] for row_id in RESCUE_ROW_IDS]


def test_rescue_v2_freezes_parent_provenance_and_measured_rho_units() -> None:
    rescue = LRStudySpec.from_path(RESCUE_STUDY_CONFIG).data["rescue"]

    assert rescue["kind"] == "unresolved_row_geometric_warm_in"
    assert rescue["import_mode"] == "copy_and_verify_parent_probe"
    assert rescue["parent_study"] == {
        "name": "conv1-conv2-hard-sigmoid-sgd-lr-study",
        "study_id": PARENT_STUDY_ID,
        "resolved_config_sha256": (
            "8bbe06ca62ed622233ab467cc1eb1124a85e4081eeea8c2e2cf0ee33bc7cb896"
        ),
        "selection_sha256": (
            "fe6fa3b744d48c9de0951bd2e71d0d09887c9972cc6006d3d72874888d3d08a7"
        ),
    }
    assert rescue["eligibility"] == {
        "included_row_ids": list(RESCUE_ROW_IDS),
        "parent_row_status": "unresolved",
        "parent_unresolved_reason": "fewer_than_three_stable_targets",
        "parent_failure_kind": "loss_ema_explosion",
        "parent_failure_onset_step": 33,
        "parent_failure_confirmed_step": 40,
        "baseline_rows_reused_not_rerun": True,
    }

    provenance = rescue["frozen_provenance"]
    assert provenance == {
        "split_indices_sha256": (
            "0cb0af3c4f63d4338f68081c8484268ae3f79bd3c7310c2fa37653f407dc0737"
        ),
        "split_provenance_sha256": (
            "e9ae8b1a9b16a0067380afa90cf25285ad7ed406c38d632eb15818b4627161a2"
        ),
        "train_indices_sha256": (
            "c0940cfdde9fb2a87f846a4dd234a6ff634eed32a98b9f5d72dfed3e2119a809"
        ),
        "validation_indices_sha256": (
            "4801b7805d54dbe2197b98320cfe8a34d5d853c4ab568bcf1d7f69e136c94bb4"
        ),
        "conv1_checkpoint_sha256": (
            "204282e999a10306cc15f6095d1ce6971dcc137f8a473ff8f6481eed5921177c"
        ),
        "conv1_metadata_sha256": (
            "05c63549d15d5c315113a4a54af52458362a2316eff35a3af0caac2bed0a745b"
        ),
        "conv1_parameter_tensor_sha256": (
            "99f6ee133f72aae48205e58bb5d9a5568f6d7a2f3d3a801a0f4e07d16463fc1d"
        ),
        "parent_probe_manifest_sha256": (
            "7d9a6cd431bcab5138eaf4936d9c75cb232122da46035773c8d09597d402f7b9"
        ),
        "parent_probe_completion_sha256": (
            "132602568e1b3a7bc70a45b192a44efea67aa49f2e6640e4d6f29bf7e10abcd2"
        ),
        "probe_minibatch_order_sha256": (
            "1c65076487a87e0badfe19b80edc8263f50c4b765dc317299dec4a8f1fad460b"
        ),
        "rows": {
            "conv1_ours_v4_c1": {
                "rho_unit": RESCUE_RHO_UNITS["conv1_ours_v4_c1"],
                "parent_range_summary_sha256": (
                    "b275b153e87b8cc6534fba86ed709f880c5f7893086576056c9fb236571ac990"
                ),
                "probe_outputs": {
                    "minibatches.json": (
                        "21e79df150b007d9275d9a69ac71c62d04a280ecfeff36ad3e65455c5686dce1"
                    ),
                    "parameter_diagnostics.csv": (
                        "960f3e023adfd0e29ddf6bee7cc9301ef03fa73d300d850e0291146aa1fc1d19"
                    ),
                    "step_log.csv": (
                        "853db55d1e9549f9595816640185db918280377029bc4d1ca516ae3a12c93ed3"
                    ),
                    "summary.json": (
                        "bbfb9e7fe4d049f4bda5ad467273f300fafcca67b94ca535dbc3e00d992aa94f"
                    ),
                },
            },
            "conv1_legacy_v4_c0p25": {
                "rho_unit": RESCUE_RHO_UNITS["conv1_legacy_v4_c0p25"],
                "parent_range_summary_sha256": (
                    "793bb444404936ba4a67ea60422b35a100a0032ce6259aa9155f99d94b4e29eb"
                ),
                "probe_outputs": {
                    "minibatches.json": (
                        "bb2289df9680d87afeefdbf479f0a9d1030881a142ead4e68816b4a1ffca6ad5"
                    ),
                    "parameter_diagnostics.csv": (
                        "f630024329865ce470d12159e5c7b2dc8594e37ac06c2a4d976f4267f88d0144"
                    ),
                    "step_log.csv": (
                        "fbd81e0a4c8ba348f4a6e78b5715656d8f864b2fc45a6ae05b65e445357fba9a"
                    ),
                    "summary.json": (
                        "c666df537e0f59827c6131fc5fca99a7fad62c9deef02830f9cfaa36f89cb1ad"
                    ),
                },
            },
        },
    }


def test_rescue_v2_rejects_parent_or_rho_unit_drift() -> None:
    value = json.loads(RESCUE_STUDY_CONFIG.read_text())
    changed_parent = copy.deepcopy(value)
    changed_parent["rescue"]["parent_study"]["study_id"] = "lrstudy_" + "0" * 64
    with pytest.raises(SpecValidationError):
        LRStudySpec.from_dict(changed_parent)

    changed_rho = copy.deepcopy(value)
    changed_rho["rescue"]["frozen_provenance"]["rows"]["conv1_ours_v4_c1"][
        "rho_unit"
    ] *= 2.0
    with pytest.raises(SpecValidationError):
        LRStudySpec.from_dict(changed_rho)


def test_rescue_plan_counts_two_probe_two_range_and_at_most_six_candidates(
    tmp_path: Path,
) -> None:
    rescue = LRStudySpec.from_path(RESCUE_STUDY_CONFIG)
    root = tmp_path / f"study--{rescue.study_id}"

    probe_plan = plan_only_result(rescue, root, "probe")
    range_plan = plan_only_result(rescue, root, "range")
    candidate_plan = plan_only_result(rescue, root, "candidates")
    assert probe_plan["entry_count"] == 2
    assert tuple(probe_plan["entry_ids"]) == RESCUE_ROW_IDS
    assert range_plan["entry_count"] == 2
    assert tuple(range_plan["entry_ids"]) == RESCUE_ROW_IDS
    assert candidate_plan["entry_count"] == 6
    assert tuple(candidate_plan["entry_ids"]) == tuple(
        f"{row_id}--{role}"
        for row_id in RESCUE_ROW_IDS
        for role in ("fast", "middle", "high")
    )
    assert plan_only_result(rescue, root, "select")["entry_count"] == 1


@pytest.mark.parametrize(
    ("ours_resolved", "legacy_resolved", "expected_ids"),
    [
        (False, False, ("no-resolved-candidates",)),
        (
            True,
            False,
            tuple(f"conv1_ours_v4_c1--{role}" for role in ("fast", "middle", "high")),
        ),
        (
            False,
            True,
            tuple(
                f"conv1_legacy_v4_c0p25--{role}"
                for role in ("fast", "middle", "high")
            ),
        ),
        (
            True,
            True,
            tuple(
                f"{row_id}--{role}"
                for row_id in RESCUE_ROW_IDS
                for role in ("fast", "middle", "high")
            ),
        ),
    ],
)
def test_rescue_candidate_entries_are_conditional_zero_three_or_six(
    tmp_path: Path,
    ours_resolved: bool,
    legacy_resolved: bool,
    expected_ids: tuple[str, ...],
) -> None:
    rescue = LRStudySpec.from_path(RESCUE_STUDY_CONFIG)
    root = tmp_path / f"study--{rescue.study_id}"
    _range_summary(root, RESCUE_ROW_IDS[0], resolved=ours_resolved)
    _range_summary(root, RESCUE_ROW_IDS[1], resolved=legacy_resolved)

    entries = _entries(rescue, root, "candidates")

    assert tuple(entry["entry_id"] for entry in entries) == expected_ids
    if not ours_resolved and not legacy_resolved:
        assert entries[0]["payload"] == {
            "no_op": True,
            "reason": "all_range_rows_unresolved",
        }
    else:
        assert all("baseline" not in entry["entry_id"] for entry in entries)
        assert all("conv2" not in entry["entry_id"] for entry in entries)
