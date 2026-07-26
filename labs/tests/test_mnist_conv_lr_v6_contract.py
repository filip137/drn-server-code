from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from experiments.mnist_conv.io import atomic_write_json
from experiments.mnist_conv.lr_protocol import (
    architecture_relative_learning_rates,
    two_rho_learning_rates,
)
from experiments.mnist_conv.lr_stages import candidate_run_spec_v4
from experiments.mnist_conv.lr_study_spec import LRStudySpec
from experiments.mnist_conv.specs import RunSpec, SpecValidationError


REPO_ROOT = Path(__file__).resolve().parents[2]
V5_CONFIG = (
    REPO_ROOT
    / "configs/conv/hardsigmoid_lr_architecture_relative_rho_constant_sgd_bs16_v5.json"
)
V6_CONFIG = (
    REPO_ROOT
    / "configs/conv/hardsigmoid_lr_conv2_two_rho_constant_sgd_bs16_v6.json"
)


class _FakeBundle:
    validation_indices_hash = "a" * 64

    def train_batch_indices(self, *, num_epochs: int = 1):
        return tuple(((epoch, epoch + 1),) for epoch in range(num_epochs))


def _all_keys(value: object) -> set[str]:
    if isinstance(value, dict):
        return set(value) | set().union(*(_all_keys(item) for item in value.values()))
    if isinstance(value, list):
        return set().union(*(_all_keys(item) for item in value)) if value else set()
    return set()


def _valid_run_v5(tmp_path: Path) -> dict[str, object]:
    """Derive a valid common run body through the established v4 builder."""

    parent = LRStudySpec.from_path(V5_CONFIG)
    v6 = LRStudySpec.from_path(V6_CONFIG)
    row = next(item for item in parent.rows if item["row_id"] == "conv2_baseline_v1_c1")
    root = tmp_path / "results/lr_studies" / f"study--{parent.study_id}"
    (root / "initialization").mkdir(parents=True)
    (root / "initialization/conv2.pt").write_bytes(b"verified-conv2-checkpoint")
    atomic_write_json(
        root / "initialization/conv2.json",
        {"parameter_tensor_sha256": "b" * 64},
        canonical=True,
    )
    for stage, entry_id in (
        ("probe", row["row_id"]),
        ("audit", "anchor-audit"),
    ):
        directory = root / "stages" / stage / "entries" / entry_id
        directory.mkdir(parents=True)
        atomic_write_json(
            directory / "summary.json", {"stage": stage}, canonical=True
        )

    units = {
        "ConvWeight_0": 0.1,
        "ConvWeight_1": 0.2,
        "DenseWeight_0": 0.5,
    }
    old_rates = architecture_relative_learning_rates(
        units,
        0.001,
        ("ConvWeight_0", "Bias_0", "ConvWeight_1", "Bias_1", "DenseWeight_0"),
    )
    value = candidate_run_spec_v4(
        parent.data,
        root,
        row,
        "strict_equal--center",
        arm="strict_equal",
        alpha_role="center",
        alpha=0.001,
        median_units_by_weight=units,
        target_multipliers_by_weight={name: 1.0 for name in units},
        learning_rates_by_parameter=old_rates,
        bundle=_FakeBundle(),
    ).to_dict()

    rho_conv, rho_dense = 0.003, 0.03
    value["schema_version"] = "mnist-conv-run/v5"
    value["label"] = "conv2_baseline_v1_c1--rho-conv-0p003--rho-dense-0p03"
    value["protocol_id"] = v6.data["protocol_id"]
    value["run"]["training"]["learning_rates_by_parameter"] = two_rho_learning_rates(
        units,
        rho_conv,
        rho_dense,
        value["run"]["training"]["learning_rates_by_parameter"],
    )
    checkpoint_sha256 = value["run"]["initialization"]["checkpoint"]["sha256"]
    value["run"]["lr_provenance"] = {
        "study_id": v6.study_id,
        "row_id": row["row_id"],
        "candidate_stage": "baseline_grid",
        "rho_conv": rho_conv,
        "rho_dense": rho_dense,
        "median_unit_by_weight": units,
        "bias_weight_mapping": {
            "Bias_0": "ConvWeight_0",
            "Bias_1": "ConvWeight_1",
        },
        "probe_sha256": "c" * 64,
        "split_sha256": _FakeBundle.validation_indices_hash,
        "batch_order_sha256": "d" * 64,
        "initialization_checkpoint_sha256": checkpoint_sha256,
        "initialization_tensor_sha256": "b" * 64,
    }
    return value


def test_v6_is_strict_content_addressed_conv2_only_contract() -> None:
    v6 = LRStudySpec.from_path(V6_CONFIG)
    v5 = LRStudySpec.from_path(V5_CONFIG)

    assert v6.study_id == "lrstudy_c735d2beead9fcbadda89a65ffe86257da53f15e6cbfab7b4f04b14b08ff6c2f"
    assert v5.study_id == "lrstudy_5da3a7452c00d42325bfe930d80f037791125d3ce7b5ddd79be13ff8b96b49e2"
    assert [row["row_id"] for row in v6.rows] == [
        "conv2_baseline_v1_c1",
        "conv2_ours_v4_c1",
        "conv2_legacy_v4_c0p25",
    ]
    assert v6.data["conv1_settled"] == {
        "architecture": "conv1",
        "rho_conv": 0.001,
        "rho_dense": 0.03,
        "source": "v5_selected_two_target_interpretation",
        "rerun": False,
        "included_in_rows": False,
        "included_in_candidate_count": False,
    }
    assert v6.data["rho_grid"]["rho_conv"] == [0.0005, 0.001, 0.003, 0.01]
    assert v6.data["rho_grid"]["rho_dense"] == [0.003, 0.01, 0.03, 0.1]
    assert v6.data["rho_grid"]["baseline_candidate_count"] == 16
    assert v6.data["rho_grid"]["maximum_canonical_training_runs"] == 18
    assert v6.data["candidate_training"]["total_steps"] == 17190
    assert v6.data["dataset"]["official_test"]["read_allowed"] is False
    assert {"alpha", "alpha_arch", "candidate_arm", "target_profile"}.isdisjoint(
        _all_keys(v6.data)
    )

    changed = copy.deepcopy(v6.data)
    changed["rho_grid"]["rho_dense"][-1] = 0.2
    with pytest.raises(SpecValidationError, match="frozen rho_grid contract"):
        LRStudySpec.from_dict(changed)

    renamed = copy.deepcopy(v6.data)
    renamed["name"] = "display-name-only"
    assert LRStudySpec.from_dict(renamed).study_id == v6.study_id


def test_v6_reuse_contract_binds_verified_v5_assets_and_fallback_policy() -> None:
    reuse = LRStudySpec.from_path(V6_CONFIG).data["reuse"]

    assert reuse["source_study_id"] == (
        "lrstudy_5da3a7452c00d42325bfe930d80f037791125d3ce7b5ddd79be13ff8b96b49e2"
    )
    assert reuse["expected_assets"]["conv2_checkpoint_sha256"] == (
        "dc16ef2bebd2c9e6afe307b41555683c3eec29b9481ab64ea3f412382c837c7e"
    )
    assert set(reuse["expected_assets"]["probe_summary_sha256_by_row"]) == {
        "conv2_baseline_v1_c1",
        "conv2_ours_v4_c1",
        "conv2_legacy_v4_c0p25",
    }
    assert reuse["mismatch_policy"] == "never_accept_mismatched_bytes_as_reused"
    assert reuse["regeneration"]["split_invalid"] == ["split", "all_conv2_probes"]
    assert reuse["regeneration"]["initialization_invalid"] == [
        "conv2_initialization",
        "all_conv2_probes",
    ]
    assert reuse["official_test_read"] is False


def test_run_v5_records_direct_targets_and_rejects_retired_coordinates(
    tmp_path: Path,
) -> None:
    value = _valid_run_v5(tmp_path)
    spec = RunSpec.from_dict(value)

    provenance = spec.data["run"]["lr_provenance"]
    assert provenance["candidate_stage"] == "baseline_grid"
    assert provenance["rho_conv"] == 0.003
    assert provenance["rho_dense"] == 0.03
    assert "candidate_arm" not in provenance
    assert "alpha_arch" not in provenance
    rates = spec.data["run"]["training"]["learning_rates_by_parameter"]
    assert rates["Bias_0"] == rates["ConvWeight_0"]
    assert rates["Bias_1"] == rates["ConvWeight_1"]

    retired = copy.deepcopy(value)
    retired["run"]["lr_provenance"]["alpha_arch"] = 0.001
    with pytest.raises(SpecValidationError, match="exactly keys"):
        RunSpec.from_dict(retired)

    wrong_stage = copy.deepcopy(value)
    wrong_stage["run"]["lr_provenance"]["candidate_stage"] = "confirmation"
    with pytest.raises(SpecValidationError, match="baseline_grid"):
        RunSpec.from_dict(wrong_stage)


def test_run_v5_rejects_target_bias_split_and_checkpoint_drift(tmp_path: Path) -> None:
    value = _valid_run_v5(tmp_path)

    dense_drift = copy.deepcopy(value)
    dense_drift["run"]["training"]["learning_rates_by_parameter"][
        "DenseWeight_0"
    ] *= 2.0
    with pytest.raises(SpecValidationError, match="direct target"):
        RunSpec.from_dict(dense_drift)

    bias_drift = copy.deepcopy(value)
    bias_drift["run"]["training"]["learning_rates_by_parameter"]["Bias_1"] *= 2.0
    with pytest.raises(SpecValidationError, match="associated ConvWeight_1 rate"):
        RunSpec.from_dict(bias_drift)

    split_drift = copy.deepcopy(value)
    split_drift["run"]["lr_provenance"]["split_sha256"] = "e" * 64
    with pytest.raises(SpecValidationError, match="validation.indices_sha256"):
        RunSpec.from_dict(split_drift)

    checkpoint_drift = copy.deepcopy(value)
    checkpoint_drift["run"]["lr_provenance"][
        "initialization_checkpoint_sha256"
    ] = "f" * 64
    with pytest.raises(SpecValidationError, match="checkpoint.sha256"):
        RunSpec.from_dict(checkpoint_drift)
