from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from experiments.mnist_conv.io import atomic_write_json
from experiments.mnist_conv.lr_engine import LRModelRuntime
from experiments.mnist_conv.lr_protocol import (
    CandidateRunResult,
    layerwise_learning_rate_report,
    layerwise_parameter_groups,
    layerwise_target_learning_rates,
    select_final_candidate,
)
from experiments.mnist_conv.lr_stages import candidate_run_spec_v3
from experiments.mnist_conv.lr_study import plan_only_result
from experiments.mnist_conv.lr_study_spec import LRStudySpec
from experiments.mnist_conv.specs import RunSpec, SpecValidationError


REPO_ROOT = Path(__file__).resolve().parents[2]
V4_CONFIG = (
    REPO_ROOT
    / "configs/conv/hardsigmoid_lr_layerwise_relative_rho_constant_sgd_bs16_v4.json"
)


class _FakeBundle:
    validation_indices_hash = "a" * 64

    def train_batch_indices(self, *, num_epochs=1):
        return tuple(((epoch, epoch + 1),) for epoch in range(num_epochs))


def _study_tree(tmp_path: Path, study: LRStudySpec, row: dict) -> Path:
    root = tmp_path / "results" / "lr_studies" / f"study--{study.study_id}"
    (root / "initialization").mkdir(parents=True)
    (root / "initialization" / f"{row['architecture']}.pt").write_bytes(
        b"checkpoint"
    )
    atomic_write_json(
        root / "initialization" / f"{row['architecture']}.json",
        {"parameter_tensor_sha256": "b" * 64},
        canonical=True,
    )
    for stage in ("probe", "range"):
        directory = root / "stages" / stage / "entries" / row["row_id"]
        directory.mkdir(parents=True)
        atomic_write_json(directory / "summary.json", {"stage": stage}, canonical=True)
    return root


def test_v4_contract_has_distinct_identity_and_all_six_rows(tmp_path: Path) -> None:
    study = LRStudySpec.from_path(V4_CONFIG)
    assert study.study_id == (
        "lrstudy_b345477de5ff64c828a9d31602314a00046bc9728039562d3e5011132805e14d"
    )
    assert len(study.rows) == 6
    assert study.data["optimizer"]["bias_lr_policy"] == "match_associated_conv_weight"
    assert study.data["candidate_training"]["schedule"] == {
        "type": "constant_layerwise",
        "scheduler_enabled": False,
        "first_step": 1,
        "final_step": 17190,
        "first_lr_factor": 1.0,
        "final_lr_factor": 1.0,
        "warmup_steps": 0,
        "decay": "none",
        "restarts": False,
    }
    root = tmp_path / f"study--{study.study_id}"
    assert plan_only_result(study, root, "probe")["entry_count"] == 6
    assert plan_only_result(study, root, "range")["entry_count"] == 6
    assert plan_only_result(study, root, "candidates")["entry_count"] == 18


def test_layerwise_groups_use_artifact_backed_bias_topology() -> None:
    assert layerwise_parameter_groups(
        ["Bias_0", "DenseWeight_0", "ConvWeight_0"]
    ) == {
        "ConvWeight_0": ("ConvWeight_0", "Bias_0"),
        "DenseWeight_0": ("DenseWeight_0",),
    }
    assert layerwise_parameter_groups(
        [
            "DenseWeight_0",
            "Bias_1",
            "ConvWeight_0",
            "Bias_0",
            "ConvWeight_1",
        ]
    ) == {
        "ConvWeight_0": ("ConvWeight_0", "Bias_0"),
        "ConvWeight_1": ("ConvWeight_1", "Bias_1"),
        "DenseWeight_0": ("DenseWeight_0",),
    }
    with pytest.raises(ValueError, match="one Bias_i for every ConvWeight_i"):
        layerwise_parameter_groups(["ConvWeight_0", "DenseWeight_0"])
    with pytest.raises(ValueError, match="no Dense output bias"):
        layerwise_parameter_groups(
            ["ConvWeight_0", "DenseWeight_0", "Bias_0", "Bias_1"]
        )


def test_layerwise_rates_hit_common_target_and_bias_matches_conv() -> None:
    units = {"ConvWeight_0": 2.0, "ConvWeight_1": 4.0, "DenseWeight_0": 8.0}
    names = [
        "DenseWeight_0",
        "Bias_0",
        "ConvWeight_1",
        "ConvWeight_0",
        "Bias_1",
    ]
    rates = layerwise_target_learning_rates(units, 0.3, names)
    assert rates == {
        "ConvWeight_0": 0.15,
        "Bias_0": 0.15,
        "ConvWeight_1": 0.075,
        "Bias_1": 0.075,
        "DenseWeight_0": 0.0375,
    }
    for weight, unit in units.items():
        assert rates[weight] * unit == pytest.approx(0.3)


def test_large_raw_rates_are_reported_without_cap_or_gate() -> None:
    rates = {
        "ConvWeight_0": 20.0,
        "Bias_0": 20.0,
        "DenseWeight_0": 0.5,
    }
    report = layerwise_learning_rate_report(rates, thresholds=(1.0, 10.0))
    assert report["reporting_only_thresholds"] == {
        "1.0": ["ConvWeight_0"],
        "10.0": ["ConvWeight_0"],
    }
    assert report["weight_learning_rates"]["ConvWeight_0"] == 20.0
    assert report["raw_lr_is_gate"] is False
    assert report["raw_lr_is_capped"] is False


def test_runtime_assigns_named_rates_by_tensor_identity_not_group_order() -> None:
    first = SimpleNamespace(name="ConvWeight_0", state=torch.tensor([1.0]))
    second = SimpleNamespace(name="DenseWeight_0", state=torch.tensor([2.0]))
    bias = SimpleNamespace(name="Bias_0", state=torch.tensor([0.0]))
    optimizer = torch.optim.SGD(
        [
            {"params": [bias.state], "lr": 9.0},
            {"params": [second.state], "lr": 9.0},
            {"params": [first.state], "lr": 9.0},
        ]
    )
    runtime = LRModelRuntime(
        row={},
        device=torch.device("cpu"),
        energy_fn=None,
        network=None,
        free_layers=[],
        cost_fn=None,
        minimizer_inference=None,
        minimizer_training=None,
        estimator=None,
        parameters=[first, second, bias],
        optimizer=optimizer,
        unbounded_normalization_scales={},
    )
    runtime.set_learning_rate(
        {"ConvWeight_0": 0.2, "DenseWeight_0": 0.03, "Bias_0": 0.2}
    )
    assert [group["lr"] for group in optimizer.param_groups] == [0.2, 0.03, 0.2]


def test_v3_run_spec_records_constant_vector_and_rejects_bias_drift(
    tmp_path: Path,
) -> None:
    study = LRStudySpec.from_path(V4_CONFIG)
    row = study.rows[0]
    root = _study_tree(tmp_path, study, row)
    units = {"ConvWeight_0": 2.0, "DenseWeight_0": 8.0}
    rates = {"ConvWeight_0": 0.15, "Bias_0": 0.15, "DenseWeight_0": 0.0375}
    spec = candidate_run_spec_v3(
        study.data,
        root,
        row,
        "middle",
        rho_target=0.3,
        rho_unit_by_weight=units,
        learning_rates_by_parameter=rates,
        bundle=_FakeBundle(),
    )
    assert spec.data["schema_version"] == "mnist-conv-run/v3"
    assert spec.data["run"]["training"]["schedule"] == {
        "name": "constant",
        "interval": "optimizer_step",
        "total_steps": 17190,
        "scheduler_enabled": False,
    }
    assert spec.data["run"]["training"]["learning_rates_by_parameter"] == rates
    assert spec.data["run"]["lr_provenance"]["bias_weight_mapping"] == {
        "Bias_0": "ConvWeight_0"
    }

    value = spec.to_dict()
    value["run"]["training"]["learning_rates_by_parameter"]["Bias_0"] = 0.01
    with pytest.raises(SpecValidationError, match="associated ConvWeight_0 rate"):
        RunSpec.from_dict(value)

    value = spec.to_dict()
    value["run"]["training"]["learning_rates_by_parameter"]["DenseWeight_0"] = 0.1
    with pytest.raises(SpecValidationError, match="rho_target"):
        RunSpec.from_dict(value)

    value = spec.to_dict()
    value["run"]["training"]["learning_rates_by_parameter"]["ConvWeight_0"] = 0.1
    value["run"]["training"]["learning_rates_by_parameter"]["Bias_0"] = 0.1
    with pytest.raises(SpecValidationError, match="rho_target"):
        RunSpec.from_dict(value)

    value = spec.to_dict()
    value["category"] = "final"
    with pytest.raises(SpecValidationError, match="exactly 'diagnostic'"):
        RunSpec.from_dict(value)


def test_plateau_center_can_use_common_rho_instead_of_max_raw_lr() -> None:
    candidates = [
        CandidateRunResult("fast", 0.5, True, 1.0, 0.8, 0.9, selection_coordinate=0.01),
        CandidateRunResult("middle", 9.0, True, 1.0, 0.8, 0.9, selection_coordinate=0.1),
        CandidateRunResult("high", 10.0, True, 1.0, 0.8, 0.9, selection_coordinate=1.0),
    ]
    selected = select_final_candidate(candidates)
    assert selected.selected is not None
    assert selected.selected.candidate_id == "middle"
