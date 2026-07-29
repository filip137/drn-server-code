from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

import pytest
import torch

from experiments.rho_search import (
    OptimizerProbe,
    SafetyRejection,
    TrainingSafetyMonitor,
    _cell_signature,
    _git_state,
    _indexed_cells,
    derive_learning_rates,
    linear_quantile,
    nominal_proposal,
    parameter_topology,
    prepare_training_config,
    run,
)


class FakeParameter:
    def __init__(
        self,
        name: str,
        state: torch.Tensor,
        min_cond: float | None = None,
        max_cond: float | None = None,
    ):
        self.name = name
        self.state = state
        self.min_cond = min_cond
        self.max_cond = max_cond


def _safety_batch(
    parameter: FakeParameter,
    *,
    loss: float = 1.0,
    gradient: float = 1.0,
    proposed: float = 0.0,
    applied: float = 0.0,
) -> dict:
    initial = torch.full_like(parameter.state, 5.5e-5)
    return {
        "loss": loss,
        "parameters": (parameter,),
        "gradients": (torch.full_like(parameter.state, gradient),),
        "pre_optimizer_states": {parameter.name: initial},
        "post_optimizer_states": {
            parameter.name: torch.full_like(parameter.state, proposed)
        },
        "post_projection_states": {
            parameter.name: torch.full_like(parameter.state, applied)
        },
    }


def _base_config() -> dict:
    return {
        "lab": {"model_key": "mnist_bp_conv_amp", "epochs": 3},
        "lr": [1.0, 1.0, 1.0],
        "optimizer": {
            "name": "SGD",
            "learning_rate": [1.0, 1.0, 1.0],
            "lr_decay": 1.0,
            "momentum": 0.0,
            "weight_decay": 0.0,
        },
        "datasets": {
            "mnist": {
                "factory": "labs.datasets.AffineMnistDataset",
                "params": {
                    "name": "mnist",
                    "batch_size": 16,
                    "root": "~/datasets/mnist",
                    "train": True,
                    "download": False,
                    "normalize": True,
                    "normalize_std": 0.3081,
                },
            }
        },
    }


def test_linear_quantile_interpolates() -> None:
    assert linear_quantile([0.0, 10.0], 0.5) == 5.0
    assert linear_quantile([0.0, 10.0], 0.9) == 9.0


def test_parameter_topology_uses_names_not_order() -> None:
    topology = parameter_topology(
        ["Bias_1", "DenseWeight_0", "ConvWeight_0 ", "Bias_0", "ConvWeight_1"]
    )

    assert topology["parameter_names"] == [
        "Bias_1",
        "DenseWeight_0",
        "ConvWeight_0",
        "Bias_0",
        "ConvWeight_1",
    ]
    assert topology["bias_to_weight"] == {
        "Bias_1": "ConvWeight_1",
        "Bias_0": "ConvWeight_0",
    }


def test_nominal_adam_proposal_matches_fresh_pytorch_step() -> None:
    initial = torch.tensor([-2.0, 0.5, 4.0], dtype=torch.float64)
    gradient = torch.tensor([0.25, -2.0, 0.0], dtype=torch.float64)
    parameter = initial.clone().requires_grad_(True)
    optimizer = torch.optim.Adam(
        [parameter],
        lr=1.0,
        betas=(0.9, 0.999),
        eps=1e-8,
        foreach=False,
        fused=False,
    )
    parameter.grad = gradient.clone()
    optimizer.step()

    assert torch.allclose(
        parameter.detach() - initial,
        nominal_proposal(gradient, "Adam"),
        atol=1e-12,
        rtol=1e-12,
    )


def test_probe_collects_units_and_leaves_parameters_unchanged() -> None:
    parameters = (
        FakeParameter("ConvWeight_0 ", torch.ones(4)),
        FakeParameter("DenseWeight_0", torch.full((4,), 2.0)),
        FakeParameter("Bias_0", torch.zeros(4)),
    )
    before = [parameter.state.clone() for parameter in parameters]
    gradients = (
        torch.full((4,), 0.5),
        torch.full((4,), 1.0),
        torch.full((4,), 2.0),
    )
    probe = OptimizerProbe("SGD", batch_counts=(2,))

    assert probe({"parameters": parameters, "gradients": gradients}) is False
    assert probe({"parameters": parameters, "gradients": gradients}) is True
    result = probe.result()

    assert result["probe_stable"] is True
    assert result["normalization_unit_by_weight"] == {
        "ConvWeight_0": 0.5,
        "DenseWeight_0": 0.5,
    }
    assert result["bias_q90_unit_by_parameter"]["Bias_0"] == 2.0
    for parameter, expected in zip(parameters, before):
        assert torch.equal(parameter.state, expected)


def test_bound_occupancy_and_projection_efficiency_are_report_only() -> None:
    parameter = FakeParameter(
        "ConvWeight_0",
        torch.full((4,), 5.5e-5),
        min_cond=1e-5,
        max_cond=1e-4,
    )
    monitor = TrainingSafetyMonitor()

    for _ in range(64):
        monitor(
            _safety_batch(
                parameter,
                proposed=1.0,
                applied=1e-4,
            )
        )

    result = monitor.summary()
    assert result["safety_failure"] is None
    assert result["maximum_bound_occupancy_by_parameter"]["ConvWeight_0"] == 1.0
    assert result["median_projection_efficiency"] < 1e-3
    assert result["report_only_diagnostics"]["used_for_rejection"] is False


def test_sustained_gradient_growth_rejects_after_warmup() -> None:
    parameter = FakeParameter(
        "ConvWeight_0",
        torch.full((4,), 5.5e-5),
        min_cond=1e-5,
        max_cond=1e-4,
    )
    monitor = TrainingSafetyMonitor()
    for _ in range(32):
        monitor(_safety_batch(parameter, gradient=1.0, proposed=5.5e-5, applied=5.5e-5))

    with pytest.raises(SafetyRejection) as caught:
        for _ in range(8):
            monitor(
                _safety_batch(
                    parameter,
                    gradient=101.0,
                    proposed=5.5e-5,
                    applied=5.5e-5,
                )
            )

    assert caught.value.failure == {
        "kind": "gradient_rms_explosion",
        "parameter": "ConvWeight_0",
        "onset_step": 33,
        "confirmed_step": 40,
    }


def test_derive_learning_rates_applies_bias_q90_cap() -> None:
    probe = {
        "status": "complete",
        "probe_stable": True,
        "parameter_names": ["ConvWeight_0", "DenseWeight_0", "Bias_0"],
        "normalization_unit_by_weight": {
            "ConvWeight_0": 0.5,
            "DenseWeight_0": 2.0,
        },
        "bias_q90_unit_by_parameter": {"Bias_0": 4.0},
    }

    capped = derive_learning_rates(probe, rho_conv=0.1, rho_dense=0.2)
    tied = derive_learning_rates(
        probe, rho_conv=0.1, rho_dense=0.2, bias_policy="tied"
    )

    assert capped == {
        "ConvWeight_0": 0.2,
        "DenseWeight_0": 0.1,
        "Bias_0": 0.025,
    }
    assert tied["Bias_0"] == tied["ConvWeight_0"] == 0.2


def test_zero_bias_unit_falls_back_to_attached_conv_rate() -> None:
    probe = {
        "status": "complete",
        "probe_stable": True,
        "parameter_names": ["Bias_0", "ConvWeight_0", "DenseWeight_0"],
        "normalization_unit_by_weight": {
            "ConvWeight_0": 1.0,
            "DenseWeight_0": 1.0,
        },
        "bias_q90_unit_by_parameter": {"Bias_0": 0.0},
    }

    rates = derive_learning_rates(probe, rho_conv=0.01, rho_dense=0.02)

    assert list(rates) == probe["parameter_names"]
    assert rates["Bias_0"] == rates["ConvWeight_0"] == 0.01


def test_prepare_training_config_uses_train_validation_and_adam() -> None:
    config = prepare_training_config(
        _base_config(),
        "Adam",
        [0.1, 0.2, 0.3],
        epochs=4,
        max_batches=5,
        max_validation_batches=2,
        split_seed=7,
        shuffle_seed=9,
        validation_batch_size=64,
    )

    assert config["datasets"]["mnist"]["factory"].endswith(
        "MnistTrainValidationDataset"
    )
    assert config["datasets"]["mnist"]["params"]["split_seed"] == 7
    assert config["optimizer"] == {
        "name": "Adam",
        "learning_rate": [0.1, 0.2, 0.3],
        "lr_decay": 1.0,
        "momentum": 0.0,
        "weight_decay": 0.0,
        "betas": [0.9, 0.999],
        "eps": 1e-8,
    }
    assert config["lab"]["epochs"] == 4


def test_prepare_training_config_rejects_non_mnist_source() -> None:
    base = _base_config()
    base["lab"]["dataset_key"] = "fashion_mnist"
    base["datasets"]["fashion_mnist"] = base["datasets"].pop("mnist")

    with pytest.raises(ValueError, match="dataset_key 'mnist'"):
        prepare_training_config(
            base,
            "SGD",
            [0.1, 0.2, 0.3],
            epochs=1,
            max_batches=1,
            max_validation_batches=1,
            split_seed=0,
            shuffle_seed=0,
            validation_batch_size=128,
        )


def test_cell_signature_changes_with_science_and_probe() -> None:
    resolved = {
        "source_config_sha256": "source-a",
        "code": {"commit": "abc", "dirty": False, "working_tree_sha256": "tree"},
        "optimizer": "SGD",
        "bias_policy": "q90_cap",
        "epochs": 3,
        "max_batches": None,
        "max_validation_batches": None,
        "split_seed": 0,
        "shuffle_seed": 0,
    }
    probe = {"normalization_unit_by_weight": {"ConvWeight_0": 1.0}}
    signature = _cell_signature(resolved, probe, 0.001, 0.01)

    changed = dict(resolved, optimizer="Adam")
    assert _cell_signature(changed, probe, 0.001, 0.01) != signature
    assert _cell_signature(
        resolved,
        {"normalization_unit_by_weight": {"ConvWeight_0": 2.0}},
        0.001,
        0.01,
    ) != signature


def test_dry_run_does_not_create_output_or_import_trainer(tmp_path: Path) -> None:
    config_path = tmp_path / "source.json"
    config_path.write_text(json.dumps(_base_config()), encoding="utf-8")
    output = tmp_path / "results"
    args = Namespace(
        config=str(config_path),
        output_root=str(output),
        rho_conv=[0.001, 0.003],
        rho_dense=[0.01, 0.03],
        optimizer=None,
        bias_policy="q90_cap",
        probe_batches=[32, 64, 128],
        stability_tolerance=0.1,
        epochs=None,
        max_batches=None,
        max_validation_batches=None,
        validation_batch_size=128,
        split_seed=0,
        shuffle_seed=0,
        device=None,
        probe_only=False,
        index=None,
        collect_only=False,
        force=False,
        dry_run=True,
    )

    result = run(args)

    assert result["cells"] == 4
    assert not output.exists()


def test_indexed_cells_selects_one_cartesian_entry() -> None:
    all_pairs, selected = _indexed_cells(
        [0.001, 0.003],
        [0.01, 0.03],
        index=2,
    )

    assert all_pairs == [
        (0.001, 0.01),
        (0.001, 0.03),
        (0.003, 0.01),
        (0.003, 0.03),
    ]
    assert selected == [(2, 0.003, 0.01)]


def test_indexed_cells_rejects_out_of_range_index() -> None:
    with pytest.raises(ValueError, match="--index"):
        _indexed_cells([0.001], [0.01], index=1)


def test_git_state_survives_missing_git(monkeypatch: pytest.MonkeyPatch) -> None:
    def missing_git(*args, **kwargs):
        raise FileNotFoundError("git")

    monkeypatch.setattr("experiments.rho_search.subprocess.run", missing_git)

    state = _git_state()

    assert state["commit"] is None
    assert state["dirty"] is None
    assert len(state["working_tree_sha256"]) == 64


@pytest.mark.parametrize("value", [0.0, -1.0, float("nan")])
def test_invalid_rho_is_rejected(value: float) -> None:
    probe = {
        "status": "complete",
        "probe_stable": True,
        "parameter_names": ["ConvWeight_0", "DenseWeight_0"],
        "normalization_unit_by_weight": {
            "ConvWeight_0": 1.0,
            "DenseWeight_0": 1.0,
        },
        "bias_q90_unit_by_parameter": {},
    }

    with pytest.raises(ValueError, match="rho_conv"):
        derive_learning_rates(probe, rho_conv=value, rho_dense=0.1)
