from __future__ import annotations

import json
import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
LABS_ROOT = REPO_ROOT / "labs"
if str(LABS_ROOT) not in sys.path:
    sys.path.insert(0, str(LABS_ROOT))

from custom_classes import FlexibleDeepResistiveEnergy  # noqa: E402
from labs.mnist_train import _reset_name_counters, _set_seed  # noqa: E402


CONFIG_ROOT = (
    REPO_ROOT
    / "configs/conv/"
    "perfectdiode_bounded_wmax_sweep_signed_scaled_bias_conv123_seed0_20260809_v1"
)
WEIGHT_PREFIXES = ("ConvWeight_", "DenseWeight_")
SCHEME_OFFSETS = {"baseline": 0, "ours": 2, "legacy": 4}
WMAX_BLOCKS = ("1em4", "3em4", "1em3", "3em3", "1em2")


def _config(architecture: str, scheme: str, block: int) -> dict:
    index = 6 * block + SCHEME_OFFSETS[scheme]
    path = (
        CONFIG_ROOT
        / architecture
        / f"{index:02d}_wmax_{WMAX_BLOCKS[block]}_{scheme}_sgd_seed0.json"
    )
    return json.loads(path.read_text(encoding="utf-8"))


def _initial_weights(config: dict) -> tuple[list[str], list[torch.Tensor]]:
    _reset_name_counters()
    _set_seed(int(config["seed"]))
    model_key = config["lab"]["model_key"]
    model = {**config["model_base"], **config["model_overrides"][model_key]}
    energy = FlexibleDeepResistiveEnergy(
        layer_shapes=[tuple(shape) for shape in model["layer_shapes"]],
        conv_pipeline=model.get("conv_pipeline") or [],
        pooling_mode=model.get("pooling_mode"),
        weight_gains=model["weight_gains"],
        input_gain=model["input_gain"],
        non_linearity=model["non_linearity"],
        exponential_diode_param=model["exponential_diode_param"],
        quadratic_diode_param=model["quadratic_diode_param"],
        hard_sigmoid_param=model["hard_sigmoid_param"],
        voltage_amp=model["voltage_amp"],
        current_amp=model["current_amp"],
        weight_min=model["weight_min"],
        weight_max=model["weight_max"],
        weight_init_mode=model["weight_init_mode"],
        input_mode=config["input_mode"],
        trainable_amplification=model["trainable_amplification"],
        amplification_min=model["amplification_min"],
        amplification_max=model["amplification_max"],
    )
    energy.set_device("cpu")
    names = []
    states = []
    for parameter in energy.params():
        name = str(parameter.name).strip()
        if name.startswith(WEIGHT_PREFIXES):
            names.append(name)
            states.append(parameter.state.detach().cpu().clone())
    return names, states


def test_bounded_wmax_initializers_reuse_matched_uniform_quantiles() -> None:
    for architecture in ("conv1", "conv2", "conv3"):
        reference_normalized = None
        raw_by_block: dict[int, list[torch.Tensor]] = {}
        for block in range(5):
            config = _config(architecture, "baseline", block)
            names, states = _initial_weights(config)
            weight_min = float(config["model_base"]["weight_min"])
            weight_max = float(config["model_base"]["weight_max"])
            normalized = [
                (state - weight_min) / (weight_max - weight_min)
                for state in states
            ]
            assert names
            assert all(float(state.min()) >= weight_min for state in states)
            assert all(float(state.max()) <= weight_max for state in states)
            if reference_normalized is None:
                reference_normalized = normalized
            else:
                for observed, expected in zip(normalized, reference_normalized):
                    torch.testing.assert_close(
                        observed,
                        expected,
                        rtol=3e-6,
                        atol=3e-6,
                    )
            raw_by_block[block] = states

            for scheme in ("ours", "legacy"):
                matched_names, matched_states = _initial_weights(
                    _config(architecture, scheme, block)
                )
                assert matched_names == names
                for observed, expected in zip(matched_states, states):
                    torch.testing.assert_close(observed, expected, rtol=0.0, atol=0.0)

        for smallest, largest in zip(raw_by_block[0], raw_by_block[4]):
            assert not torch.equal(smallest, largest)
