import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from experiments import RunMode, resolve_experiment_config
from experiments.schema import ConfigError
from experiments.small_network.components import build_model_stack, seed_runtime
from experiments.small_network.config import parse_small_drn_config
from experiments.small_network.runtime import _selected_model_metadata
from model.variable.parameter import Bias


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT
    / "examples"
    / "small_drn"
    / "mnist_bounded_memristor_teacher_10ep.json"
)
CMO_FLOOR_RATIO = 9.0 / 88.199997


def _train_spec():
    return resolve_experiment_config(CONFIG, RunMode.TRAIN)[1]


def _cpu_common(spec):
    return replace(
        spec.common,
        runtime=replace(spec.common.runtime, device="cpu"),
    )


def test_bounded_teacher_config_resolves_physical_protocol() -> None:
    train = _train_spec()
    validate = resolve_experiment_config(CONFIG, RunMode.VALIDATE)[1]

    assert train.settings.num_epochs == 10
    assert train.common.data.validation_points == 5000
    assert train.common.model.dims == (1568, 100, 20)
    assert train.common.model.input_gain == 100.0
    assert train.common.model.weight_min == pytest.approx(CMO_FLOOR_RATIO)
    assert train.common.model.weight_max == 1.0
    assert (
        train.common.model.weight_init_mode
        == "floor_shifted_kaiming_uniform"
    )
    assert train.common.model.include_biases is False
    assert train.common.model.voltage_amp == 4.0
    assert train.common.model.current_amp == 0.25
    assert train.settings.bias_learning_rates == ()
    assert validate.common.model == train.common.model


def test_floor_shifted_initialization_is_bounded_and_rebuild_stable() -> None:
    spec = _train_spec()
    common = _cpu_common(spec)

    seed_runtime(common.runtime.seed)
    first = build_model_stack(common)
    first_states = {
        binding.key: binding.state.detach().clone()
        for binding in first.bundle.catalog.trainable
    }
    seed_runtime(common.runtime.seed)
    second = build_model_stack(common)
    second_states = {
        binding.key: binding.state.detach().clone()
        for binding in second.bundle.catalog.trainable
    }

    assert tuple(first_states) == (
        "base.dense_weight.0",
        "base.dense_weight.1",
    )
    assert not any(
        isinstance(binding.parameter, Bias)
        for binding in first.bundle.catalog.all
    )
    for key, state in first_states.items():
        assert torch.equal(state, second_states[key])
        assert float(state.min()) >= CMO_FLOOR_RATIO
        assert float(state.max()) <= 1.0
        floor_fraction = float(
            state.eq(spec.common.model.weight_min).float().mean()
        )
        assert 0.45 < floor_fraction < 0.55


def test_teacher_checkpoint_metadata_records_model_local_semantics() -> None:
    spec = _train_spec()
    common = _cpu_common(spec)
    seed_runtime(common.runtime.seed)
    stack = build_model_stack(common)

    metadata = _selected_model_metadata(
        spec,
        SimpleNamespace(stack=stack),
    )

    assert metadata["checkpoint_role"] == "supervised_drn"
    assert metadata["objective"] == "supervised_paired_squared_error"
    assert metadata["output_semantics"] == "adjacent_pair_difference"
    assert metadata["model"]["weight_bounds"] == pytest.approx(
        [CMO_FLOOR_RATIO, 1.0]
    )
    assert metadata["model"]["include_biases"] is False
    topology = metadata["model"]["amplification_indices"]
    assert [item["resolved_pre_index"] for item in topology] == [0, 1]
    assert [item["resolved_post_index"] for item in topology] == [1, 2]
    assert [item["forward_gain"] for item in topology] == [1.0, 4.0]
    assert [item["post_layer_metric"] for item in topology] == [
        1.0,
        0.0625,
    ]


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("include_biases", "false", "include_biases.*boolean"),
        ("weight_init_mode", "unknown", "weight_init_mode"),
    ),
)
def test_bounded_teacher_model_fields_fail_closed(
    field: str,
    value: object,
    message: str,
) -> None:
    payload = json.loads(CONFIG.read_text())
    payload["model"][field] = value

    with pytest.raises(ConfigError, match=message):
        parse_small_drn_config(payload)
