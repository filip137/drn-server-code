from __future__ import annotations

import argparse

import pytest
import torch

from labs.tools.evaluate_conv1_hardsigmoid_finite_conductance import (
    EvaluationInputError,
    _build_replay_engine_config,
    _run_spec_filename,
    _saturation_per_example,
    _source_checkpoint_dataset,
    apply_conductance_bounds,
    build_bound_grid,
    parse_optional_bound,
)
from model.variable.parameter import Bias, ConvWeight, DenseWeight


def _parameters() -> tuple[list[object], list[torch.Tensor]]:
    conv = ConvWeight(
        shape=(1, 1, 1, 3),
        gain=1.0,
        device="cpu",
        clamp=True,
        clamp_min=0.0,
        clamp_max=100.0,
    )
    dense = DenseWeight(
        (3,),
        (1,),
        gain=1.0,
        device="cpu",
        clamp=True,
        clamp_min=0.0,
        clamp_max=100.0,
    )
    bias = Bias((1,), gain=0.0, device="cpu")
    conv.state = torch.tensor([[[[0.0, 0.2, 1.0]]]], dtype=torch.float32)
    dense.state = torch.tensor([[0.05], [0.5], [0.9]], dtype=torch.float32)
    bias.state = torch.tensor([-0.75], dtype=torch.float32)
    params = [conv, dense, bias]
    return params, [param.state.detach().clone() for param in params]


def test_grid_includes_reference_and_open_slices() -> None:
    assert build_bound_grid([0.1], [0.8]) == [
        (None, None),
        (None, 0.8),
        (0.1, None),
        (0.1, 0.8),
    ]
    assert build_bound_grid([0.9], [0.8]) == [
        (None, None),
        (None, 0.8),
        (0.9, None),
    ]


def test_bounds_restore_pristine_and_never_clip_bias() -> None:
    params, pristine = _parameters()
    rows, aggregate = apply_conductance_bounds(
        params,
        pristine,
        gmin=0.1,
        gmax=0.8,
    )
    assert [row["parameter_type"] for row in rows] == [
        "ConvWeight",
        "DenseWeight",
    ]
    torch.testing.assert_close(
        params[0].state,
        torch.tensor([[[[0.1, 0.2, 0.8]]]], dtype=torch.float32),
    )
    torch.testing.assert_close(
        params[1].state,
        torch.tensor([[0.1], [0.5], [0.8]], dtype=torch.float32),
    )
    assert torch.equal(params[2].state, pristine[2])
    assert aggregate["lower_clipped_fraction"] == pytest.approx(2 / 6)
    assert aggregate["upper_clipped_fraction"] == pytest.approx(2 / 6)
    assert aggregate["clipped_fraction"] == pytest.approx(4 / 6)

    params[0].state.fill_(99.0)
    params[1].state.fill_(99.0)
    params[2].state.fill_(99.0)
    _, aggregate = apply_conductance_bounds(
        params,
        pristine,
        gmin=None,
        gmax=None,
    )
    for parameter, original in zip(params, pristine):
        assert torch.equal(parameter.state, original)
    assert aggregate["clipped_fraction"] == 0.0


def test_invalid_bound_fails_with_expected_format_first() -> None:
    params, pristine = _parameters()
    with pytest.raises(
        EvaluationInputError,
        match=r"Expected gmin to be less than or equal to gmax",
    ):
        apply_conductance_bounds(
            params,
            pristine,
            gmin=0.9,
            gmax=0.8,
        )
    with pytest.raises(
        argparse.ArgumentTypeError,
        match="Expected a non-negative finite number or 'none'",
    ):
        parse_optional_bound("-1")


def test_hard_sigmoid_saturation_uses_strict_outside_interval() -> None:
    class Layer:
        state = torch.tensor(
            [
                [[[-4.1, -4.0], [4.0, 4.1]]],
                [[[-3.0, 0.0], [3.0, 4.0]]],
            ],
            dtype=torch.float32,
        )

    result = _saturation_per_example(Layer(), v_off=4.0)
    torch.testing.assert_close(
        result,
        torch.tensor([0.5, 0.0], dtype=torch.float64),
    )


def test_ordinary_mnist_override_only_changes_dataset_factory() -> None:
    class Spec:
        data = {
            "seed": 0,
            "run": {
                "dataset": {
                    "name": "mnist",
                    "batch_size": 16,
                    "normalization": {
                        "mean": 0.1307,
                        "std": 0.3081,
                        "scale": 0.3,
                    },
                    "affine": {
                        "enabled": True,
                        "preset": "medium",
                        "degrees": 25.0,
                        "translate": [0.2, 0.2],
                        "scale": [0.8, 1.2],
                        "shear": 0.0,
                        "seed": 1729,
                        "interpolation": "bilinear",
                        "fill": 0.0,
                    },
                },
                "architecture": {
                    "channels": [64],
                    "kernel_sizes": [3],
                    "strides": [2],
                    "paddings": [1],
                    "output_dim": 20,
                },
                "model": {
                    "weight_min": 0.0,
                    "weight_max": 100.0,
                    "weight_init_mode": "kaiming_uniform",
                    "input_gain": 75.0,
                    "voltage_amp": 1.0,
                    "current_amp": 1.0,
                    "trainable_parameters": {"amplification": False},
                    "amplification_min": 1e-6,
                    "amplification_max": None,
                    "non_linearity": "hard_sigmoid",
                    "quadratic_diode_param": {},
                    "exponential_diode_param": {},
                    "hard_sigmoid_param": {
                        "g_on": 100.0,
                        "g_off": 0.0,
                        "v_off": 4.0,
                    },
                    "weight_gains": [1.0, 1.0],
                },
                "solver": {
                    "inference_iterations": 4,
                    "training_iterations": 4,
                    "minimizer": {"adaptive_equilibrium": False},
                    "energy_mode": "asynchronous",
                },
                "training": {"batch_state_policy": "reset_each_batch"},
            },
        }

    source = _build_replay_engine_config(
        Spec(),
        dataset_root="/tmp/mnist",
        device="cpu",
        download=False,
        ordinary_mnist=False,
    )
    ordinary = _build_replay_engine_config(
        Spec(),
        dataset_root="/tmp/mnist",
        device="cpu",
        download=False,
        ordinary_mnist=True,
    )
    assert source["datasets"]["mnist"]["factory"] == "labs.datasets.AffineMnistDataset"
    assert ordinary["datasets"]["mnist"]["factory"] == "labs.datasets.MnistDataset"
    assert "affine_config" in source["datasets"]["mnist"]["params"]
    assert "affine_config" not in ordinary["datasets"]["mnist"]["params"]
    assert source["model_base"] == ordinary["model_base"]


def test_supported_source_protocols_and_run_spec_versions() -> None:
    class MediumSpec:
        data = {"protocol_id": "conv-hardsigmoid-lr-sgd-bs16-v1"}

    class OrdinarySpec:
        data = {
            "protocol_id": (
                "conv-hardsigmoid-lr-architecture-relative-median-"
                "constant-sgd-bs16-v5"
            )
        }

    assert (
        _source_checkpoint_dataset(MediumSpec())
        == "deterministic medium affine MNIST"
    )
    assert _source_checkpoint_dataset(OrdinarySpec()) == "ordinary MNIST"
    assert (
        _run_spec_filename(
            {"outputs": [{"path": "candidate/run_spec.v4.json"}]}
        )
        == "run_spec.v4.json"
    )
    with pytest.raises(
        EvaluationInputError,
        match="Expected complete.json run-spec outputs",
    ):
        _run_spec_filename(
            {
                "outputs": [
                    {"path": "candidate/run_spec.v2.json"},
                    {"path": "candidate/run_spec.v4.json"},
                ]
            }
        )
