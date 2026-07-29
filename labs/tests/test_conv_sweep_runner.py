from __future__ import annotations

import sys

from experiments.train_mnist_bp_conv_amplification_sweep import (
    RunSpec,
    _build_config,
    parse_args,
)


def test_generated_config_has_explicit_sgd(monkeypatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "conv-sweep",
            "--minimizer-config",
            "labs/configs/mnist_minimizer_fixed_iterations.json",
        ],
    )
    args = parse_args()
    config = _build_config(
        args,
        RunSpec("hard_sigmoid", "mnist_bp_amp_v1_c1", 0, 1.0, 1.0),
    )

    assert config["optimizer"] == {
        "name": "SGD",
        "learning_rate": config["lr"],
        "lr_decay": config["lr_decay"],
        "momentum": 0.0,
        "weight_decay": 0.0,
    }
