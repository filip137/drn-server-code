from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.mnist_conv.identity import code_provenance
from experiments.mnist_conv.io import atomic_write_json, read_json
from experiments.mnist_conv.lr_conv3_legacy_rescue import (
    DIAGNOSTIC_STAGE,
    PARENT_V7_STUDY_ID,
    PROMOTION_RETRY_STAGE,
    LegacyRescueSpec,
    _publish_entry_completion,
    finalize_diagnostics,
    plan_rescue,
    replan_promotions,
)
from experiments.mnist_conv.lr_protocol import two_rho_learning_rates
from experiments.mnist_conv.lr_stages import candidate_run_spec_v7
from experiments.mnist_conv.lr_study_spec import LRStudySpec
from experiments.mnist_conv.specs import SpecValidationError


REPO_ROOT = Path(__file__).resolve().parents[2]
RESCUE_CONFIG = (
    REPO_ROOT
    / "configs/conv/hardsigmoid_lr_conv3_legacy_low_rho_rescue_bs16_v1.json"
)
V7_CONFIG = (
    REPO_ROOT
    / "configs/conv/hardsigmoid_lr_conv3_scheme_two_rho_constant_sgd_bs16_v7.json"
)


def _parent_tree(tmp_path: Path) -> Path:
    parent = tmp_path / "parent-v7"
    (parent / "split").mkdir(parents=True)
    (parent / "initialization").mkdir()
    probe_dir = (
        parent
        / "stages/probe/entries/conv3_legacy_v4_c0p25"
    )
    probe_dir.mkdir(parents=True)
    (parent / "study.resolved.json").write_bytes(V7_CONFIG.read_bytes())
    (parent / "split/indices.json").write_text("{}")
    (parent / "split/provenance.json").write_text("{}")
    (parent / "initialization/conv3.pt").write_bytes(b"checkpoint")
    (parent / "initialization/conv3.json").write_text("{}")
    atomic_write_json(
        probe_dir / "summary.json",
        {
            "row": {"row_id": "conv3_legacy_v4_c0p25"},
            "median_units_by_weight": {
                "ConvWeight_0": 2.0,
                "ConvWeight_1": 4.0,
                "ConvWeight_2": 5.0,
                "DenseWeight_0": 10.0,
            },
            "bias_weight_lr_groups": {
                "ConvWeight_0": ["ConvWeight_0", "Bias_0"],
                "ConvWeight_1": ["ConvWeight_1", "Bias_1"],
                "ConvWeight_2": ["ConvWeight_2", "Bias_2"],
                "DenseWeight_0": ["DenseWeight_0"],
            },
        },
        canonical=True,
    )
    return parent


def test_rescue_has_separate_stable_identity_and_does_not_change_v7() -> None:
    rescue = LegacyRescueSpec.from_path(RESCUE_CONFIG)
    assert (
        rescue.rescue_id
        == "lrrescue_e9c1d08dc339f723f61d3d73b8059fe5f82fe95952a5d1fa06107ed877141a04"
    )
    assert rescue.data["parent"]["mutate_parent_study"] is False
    assert LRStudySpec.from_path(V7_CONFIG).study_id == PARENT_V7_STUDY_ID


def test_rescue_contract_rejects_an_added_automatic_target() -> None:
    value = json.loads(RESCUE_CONFIG.read_text())
    value["diagnostic"]["targets"].append(
        {"role": "unfrozen", "rho_conv": 1e-6, "rho_dense": 1e-6}
    )
    with pytest.raises(ValueError, match="rescue.diagnostic.targets"):
        LegacyRescueSpec.from_dict(value)


def test_plan_has_three_exact_vectors_with_tied_biases(tmp_path: Path) -> None:
    parent = _parent_tree(tmp_path)
    planned = plan_rescue(
        rescue_config=RESCUE_CONFIG,
        parent_study_dir=parent,
        results_root=tmp_path / "results",
        provenance=code_provenance(),
    )
    assert planned["entry_count"] == 3
    assert [entry["entry_id"] for entry in planned["entries"]] == [
        "lower-10x",
        "lower-33x",
        "lower-100x",
    ]
    rates = planned["entries"][0]["learning_rates_by_parameter"]
    assert rates == pytest.approx({
        "ConvWeight_0": 2.5e-5,
        "Bias_0": 2.5e-5,
        "ConvWeight_1": 1.25e-5,
        "Bias_1": 1.25e-5,
        "ConvWeight_2": 1e-5,
        "Bias_2": 1e-5,
        "DenseWeight_0": 3e-5,
    })
    manifest = read_json(Path(planned["manifest_path"]))
    assert manifest["parent_study_id"] == PARENT_V7_STUDY_ID
    assert manifest["official_test_read"] is False


def test_only_safety_clean_diagnostics_are_promoted(tmp_path: Path) -> None:
    parent = _parent_tree(tmp_path)
    planned = plan_rescue(
        rescue_config=RESCUE_CONFIG,
        parent_study_dir=parent,
        results_root=tmp_path / "results",
        provenance=code_provenance(),
    )
    root = Path(planned["rescue_dir"])
    for entry_id, safety_clean in (
        ("lower-10x", True),
        ("lower-33x", False),
        ("lower-100x", True),
    ):
        entry_dir = root / f"stages/{DIAGNOSTIC_STAGE}/entries/{entry_id}"
        entry_dir.mkdir(parents=True)
        atomic_write_json(
            entry_dir / "summary.json",
            {
                "entry_id": entry_id,
                "safety_clean": safety_clean,
            },
            canonical=True,
        )
        _publish_entry_completion(
            root,
            stage=DIAGNOSTIC_STAGE,
            entry_id=entry_id,
            required_outputs=("summary.json",),
        )

    finalized = finalize_diagnostics(rescue_dir=root)
    assert finalized["safety_clean_count"] == 2
    promotion = read_json(Path(finalized["promotion_manifest_path"]))
    assert [entry["source_diagnostic_entry_id"] for entry in promotion["entries"]] == [
        "lower-10x",
        "lower-100x",
    ]
    assert all(
        entry["payload"]["candidate_stage"] == "legacy_rescue_promotion"
        for entry in promotion["entries"]
    )
    assert promotion["zero_work"] is False

    retried = replan_promotions(rescue_dir=root)
    assert retried["stage"] == PROMOTION_RETRY_STAGE
    retry_manifest = read_json(Path(retried["manifest_path"]))
    assert retry_manifest["retry_of"] == {
        "stage": "promotion",
        "manifest_sha256": retry_manifest["retry_of"]["manifest_sha256"],
        "failure_phase": "pre_training_run_spec_validation",
        "candidate_outputs_published": False,
    }
    assert all(
        entry["output_dir"].startswith(
            f"stages/{PROMOTION_RETRY_STAGE}/entries/"
        )
        for entry in retry_manifest["entries"]
    )


class _FakeBundle:
    validation_indices_hash = "a" * 64

    def train_batch_indices(self, *, num_epochs: int = 1):
        return tuple(((epoch,),) for epoch in range(num_epochs))


def test_rescue_promotion_passes_the_complete_v7_run_spec_validator(
    tmp_path: Path,
) -> None:
    parent = _parent_tree(tmp_path)
    atomic_write_json(
        parent / "initialization/conv3.json",
        {"parameter_tensor_sha256": "b" * 64},
        canonical=True,
    )
    study = LRStudySpec.from_path(V7_CONFIG)
    row = next(
        item
        for item in study.rows
        if item["row_id"] == "conv3_legacy_v4_c0p25"
    )
    probe = read_json(
        parent
        / "stages/probe/entries/conv3_legacy_v4_c0p25/summary.json"
    )
    rates = two_rho_learning_rates(
        probe["median_units_by_weight"],
        rho_conv=5e-5,
        rho_dense=3e-4,
        bias_weight_lr_groups=probe["bias_weight_lr_groups"],
    )
    run = candidate_run_spec_v7(
        study.data,
        parent,
        row,
        "legacy-rescue-lower-10x",
        candidate_stage="legacy_rescue_promotion",
        rho_conv=5e-5,
        rho_dense=3e-4,
        median_units_by_weight=probe["median_units_by_weight"],
        learning_rates_by_parameter=rates,
        bundle=_FakeBundle(),
    ).data
    assert (
        run["run"]["lr_provenance"]["candidate_stage"]
        == "legacy_rescue_promotion"
    )

    with pytest.raises(SpecValidationError, match="legacy rescue rho pairs"):
        candidate_run_spec_v7(
            study.data,
            parent,
            row,
            "unfrozen-rescue",
            candidate_stage="legacy_rescue_promotion",
            rho_conv=1e-6,
            rho_dense=1e-6,
            median_units_by_weight=probe["median_units_by_weight"],
            learning_rates_by_parameter=rates,
            bundle=_FakeBundle(),
        )
