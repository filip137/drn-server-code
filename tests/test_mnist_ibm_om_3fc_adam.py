from __future__ import annotations

import json
from pathlib import Path

import torch

from experiments.definitions import EXPERIMENT_REGISTRY, resolve_experiment_config
from experiments.mnist_om_3fc_adam.config import ThreeFcAdamSpec
from experiments.mnist_om_3fc_adam.runtime import (
    _PulseAdam,
    _aihwkit_default_matrices,
    _build_layout,
    _flatten_matrices,
    _forward_matrices,
    _matrices_from_state,
)
from experiments.schema import RunMode
from experiments.study_workflow import load_study_plan


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "examples" / "mnist_om_3fc_adam" / "short_5ep_published_cuda.json"
STUDY = ROOT / "studies" / "mnist-ibm-om-3fc-adam-short-cuda-20260904-v1.json"


def test_short_three_fc_adam_config_and_study_are_frozen() -> None:
    definition, spec = resolve_experiment_config(CONFIG, RunMode.TRAIN)
    plan = load_study_plan(STUDY)

    assert definition is EXPERIMENT_REGISTRY["mnist_ibm_om_3fc_adam.v1"]
    assert isinstance(spec, ThreeFcAdamSpec)
    assert spec.runtime.device == "cuda"
    assert spec.model.dims == (784, 256, 128, 10)
    assert spec.model.bias is True
    assert spec.model.hidden_activation == "sigmoid"
    assert spec.model.weight_mapping == "none_direct_q"
    assert spec.source.initial_requested_range == (-1.0, 1.0)
    assert spec.device.corruption_policy == "published"
    assert spec.device.corrupt_devices_probability == 0.1348
    assert spec.device.reference_std == 0.05
    assert spec.optimizer.epochs == 5
    assert spec.optimizer.learning_rates_q == (0.001, 0.001, 0.001)
    assert len(plan["arms"]) == 1
    assert Path(plan["arms"][0]["configs"][0]["resolved_path"]) == CONFIG


def test_aihwkit_default_three_fc_initialization_matches_torch_linear() -> None:
    torch.manual_seed(42)
    native = (
        torch.nn.Linear(784, 256, bias=True),
        torch.nn.Linear(256, 128, bias=True),
        torch.nn.Linear(128, 10, bias=True),
    )
    actual = _aihwkit_default_matrices(
        (784, 256, 128, 10),
        bias=True,
        seed=42,
    )

    for matrix, layer in zip(actual, native, strict=True):
        expected = torch.cat(
            (layer.weight.detach().transpose(0, 1), layer.bias.detach().unsqueeze(0)),
            dim=0,
        )
        assert torch.equal(matrix, expected)


def test_three_fc_layout_roundtrip_and_forward_match_direct_matrices() -> None:
    layout = _build_layout(
        (784, 256, 128, 10),
        bias=True,
        maximum_input_size=512,
    )
    matrices = _aihwkit_default_matrices(
        (784, 256, 128, 10),
        bias=True,
        seed=7,
    )
    flat = _flatten_matrices(matrices, layout)
    rebuilt = _matrices_from_state(flat, layout)
    inputs = torch.randn((3, 784), generator=torch.Generator().manual_seed(9))

    assert [tile.shape for tile in layout] == [
        (393, 256),
        (392, 256),
        (257, 128),
        (129, 10),
    ]
    assert flat.numel() == 235_146
    assert all(
        torch.equal(left, right)
        for left, right in zip(matrices, rebuilt, strict=True)
    )
    assert torch.equal(
        _forward_matrices(inputs, matrices, bias=True),
        _forward_matrices(inputs, rebuilt, bias=True),
    )


def test_pulse_adam_reads_apparent_gradient_and_mutates_persistent_state() -> None:
    layout = _build_layout((2, 2, 2, 2), bias=False, maximum_input_size=512)
    size = sum(tile.cells for tile in layout)

    class Plant:
        def __init__(self) -> None:
            self.persistent = torch.zeros(size)
            self.apparent = torch.full((size,), 0.25)
            self.last_direction = torch.zeros(size, dtype=torch.int8)

        def pulse(self, direction: torch.Tensor) -> None:
            self.last_direction = direction.detach().clone()
            self.persistent.add_(0.1 * direction)
            touched = direction != 0
            self.apparent[touched] = self.persistent[touched] + 0.01

    plant = Plant()
    optimizer = _PulseAdam(
        size=size,
        layout=layout,
        learning_rates=(0.2, 0.2, 0.2),
        betas=(0.9, 0.999),
        epsilon=1e-8,
        nominal_dw_min=0.1,
        maximum_updates_per_cell=1,
        generator=torch.Generator().manual_seed(4),
        device=torch.device("cpu"),
    )
    optimizer.step(torch.ones(size), plant)  # type: ignore[arg-type]

    assert torch.equal(plant.last_direction, -torch.ones(size, dtype=torch.int8))
    assert torch.equal(plant.persistent, torch.full((size,), -0.1))
    assert torch.allclose(plant.apparent, torch.full((size,), -0.09))
    assert optimizer.report()["commanded_pulses"] == size


def test_config_rejects_a_cpu_training_fallback() -> None:
    payload = json.loads(CONFIG.read_text(encoding="utf-8"))
    payload["runtime"]["device"] = "cpu"

    try:
        EXPERIMENT_REGISTRY["mnist_ibm_om_3fc_adam.v1"].parser(payload)
    except ValueError as error:
        assert "config.runtime.device" in str(error)
    else:  # pragma: no cover - fail-closed assertion
        raise AssertionError("CPU fallback unexpectedly parsed")
