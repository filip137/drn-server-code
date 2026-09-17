#!/usr/bin/env python3
"""Sample native HfO2 tile.reset -> 200 RESET/SET endpoint populations."""

from __future__ import annotations

import argparse
from collections import OrderedDict
import hashlib
import json
from pathlib import Path
import platform
from typing import Any, Mapping

import aihwkit
import numpy as np
import torch
from aihwkit.simulator.configs import SingleRPUConfig
from aihwkit.simulator.parameters import PulseType
from aihwkit.simulator.presets.devices import ReRamArrayHfO2PresetDevice
from aihwkit.simulator.tiles import AnalogTile


SCHEMA = "ebl.lab.hfo2_native_tile_reset_figure6"
RESULT_SCHEMA = "ebl.lab.hfo2_native_tile_reset_figure6.samples"
PUBLISHED_HFO2_CORRUPT_PROBABILITY = 0.0977


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_keys(value: Mapping[str, Any], required: set[str], path: str) -> None:
    missing = sorted(required - set(value))
    unknown = sorted(set(value) - required)
    if missing or unknown:
        raise ValueError(
            f"Expected {path} keys {sorted(required)!r}; "
            f"missing={missing!r}, unknown={unknown!r}."
        )


def _load_config(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Expected the probe config to be a JSON object.")
    _require_keys(
        payload,
        {
            "schema",
            "schema_version",
            "probe_id",
            "evidence_class",
            "source",
            "population",
            "initialization",
            "update",
            "figure6_target_user_supplied_us",
            "comparison",
        },
        "config",
    )
    if payload["schema"] != SCHEMA or payload["schema_version"] != 1:
        raise ValueError("Expected the version-1 native HfO2 reset probe schema.")
    if payload["evidence_class"] != "model_based_aihwkit_preset":
        raise ValueError("Expected model_based_aihwkit_preset evidence.")

    source = payload["source"]
    _require_keys(
        source,
        {
            "preset",
            "required_aihwkit_version",
            "construction_seed",
            "enable_published_corruption",
        },
        "config.source",
    )
    if source["preset"] != "ReRamArrayHfO2PresetDevice":
        raise ValueError("Expected the baseline-HfO2 array preset.")
    if source["required_aihwkit_version"] != "1.1.0":
        raise ValueError("Expected AIHWKit 1.1.0.")
    if (
        isinstance(source["construction_seed"], bool)
        or not isinstance(source["construction_seed"], int)
        or source["construction_seed"] <= 0
    ):
        raise ValueError("Expected a positive construction seed.")
    if not isinstance(source["enable_published_corruption"], bool):
        raise ValueError("Expected a Boolean corruption policy.")

    population = payload["population"]
    _require_keys(population, {"num_devices", "tile_out_size"}, "config.population")
    if (
        isinstance(population["num_devices"], bool)
        or not isinstance(population["num_devices"], int)
        or population["num_devices"] < 1
        or population["tile_out_size"] != 1
    ):
        raise ValueError("Expected one row and a positive device count.")

    initialization = payload["initialization"]
    _require_keys(
        initialization,
        {"operation", "reset_probability", "branching"},
        "config.initialization",
    )
    if (
        initialization["operation"] != "tile.reset"
        or initialization["reset_probability"] != 1.0
        or initialization["branching"]
        != "same_native_tile_and_device_population_with_one_independent_reset_call_per_arm"
    ):
        raise ValueError("Expected the frozen native tile.reset branching protocol.")

    update = payload["update"]
    _require_keys(
        update,
        {
            "pulses_per_arm",
            "pulse_type",
            "desired_bl",
            "fixed_bl",
            "update_bl_management",
            "update_management",
            "learning_rate",
            "unit_x",
            "set_d",
            "reset_d",
        },
        "config.update",
    )
    expected_update = {
        "pulses_per_arm": 200,
        "pulse_type": "DeterministicImplicit",
        "desired_bl": 1,
        "fixed_bl": True,
        "update_bl_management": False,
        "update_management": False,
        "learning_rate": "preset_dw_min",
        "unit_x": 1.0,
        "set_d": -1.0,
        "reset_d": 1.0,
    }
    if update != expected_update:
        raise ValueError(f"Expected the frozen 200-pulse update contract: {expected_update!r}.")
    return payload


def _hidden_snapshot(tile: AnalogTile) -> OrderedDict[str, torch.Tensor]:
    return OrderedDict(
        (name, value.detach().clone())
        for name, value in tile.get_hidden_parameters().items()
    )


def _visible_weight(tile: AnalogTile) -> np.ndarray:
    weight = tile.get_weights()[0]
    return weight.detach().cpu().numpy().astype(np.float64, copy=True).reshape(-1)


def _hidden_array(
    hidden: Mapping[str, torch.Tensor], name: str
) -> np.ndarray:
    if name not in hidden:
        raise RuntimeError(f"Expected native hidden parameter {name!r}.")
    return (
        hidden[name]
        .detach()
        .cpu()
        .numpy()
        .astype(np.float64, copy=True)
        .reshape(-1)
    )


def _apply_pulses(
    tile: AnalogTile,
    *,
    x: torch.Tensor,
    d_value: float,
    pulses: int,
) -> None:
    d = torch.full((1, 1), float(d_value), dtype=torch.float32)
    for _ in range(pulses):
        tile.update(x, d)


def run(config_path: Path, output_dir: Path) -> None:
    config = _load_config(config_path)
    required_version = config["source"]["required_aihwkit_version"]
    if aihwkit.__version__ != required_version:
        raise RuntimeError(
            "Expected the native AIHWKit version to match the config exactly. "
            f"required={required_version!r}, loaded={aihwkit.__version__!r}."
        )

    torch.set_num_threads(1)
    device = ReRamArrayHfO2PresetDevice()
    device.construction_seed = int(config["source"]["construction_seed"])
    if config["source"]["enable_published_corruption"]:
        device.corrupt_devices_prob = PUBLISHED_HFO2_CORRUPT_PROBABILITY

    rpu_config = SingleRPUConfig(device=device)
    rpu_config.update.pulse_type = PulseType.DETERMINISTIC_IMPLICIT
    rpu_config.update.desired_bl = 1
    rpu_config.update.fixed_bl = True
    rpu_config.update.update_bl_management = False
    rpu_config.update.update_management = False

    num_devices = int(config["population"]["num_devices"])
    tile = AnalogTile(1, num_devices, rpu_config, bias=False)
    tile.set_learning_rate(float(device.dw_min))
    reset_probability = float(config["initialization"]["reset_probability"])
    tile.reset(reset_prob=reset_probability)
    reset_start_hidden = _hidden_snapshot(tile)
    if "persistent_weights" not in reset_start_hidden:
        raise RuntimeError(
            "Expected write-noise-enabled HfO2 tiles to expose persistent_weights."
        )
    reset_start_visible = _visible_weight(tile)

    x = torch.full(
        (1, num_devices),
        float(config["update"]["unit_x"]),
        dtype=torch.float32,
    )
    pulses = int(config["update"]["pulses_per_arm"])
    _apply_pulses(
        tile,
        x=x,
        d_value=float(config["update"]["reset_d"]),
        pulses=pulses,
    )
    reset_hidden = _hidden_snapshot(tile)
    reset_visible = _visible_weight(tile)

    tile.reset(reset_prob=reset_probability)
    set_start_hidden = _hidden_snapshot(tile)
    set_start_visible = _visible_weight(tile)
    fixed_names = (
        "max_bound",
        "min_bound",
        "dwmin_up",
        "dwmin_down",
        "reference",
        "reset_bias",
    )
    for name in fixed_names:
        if not torch.equal(reset_start_hidden[name], set_start_hidden[name]):
            raise RuntimeError(
                f"Native tile.reset unexpectedly changed fixed device parameter {name!r}."
            )
    _apply_pulses(
        tile,
        x=x,
        d_value=float(config["update"]["set_d"]),
        pulses=pulses,
    )
    set_hidden = _hidden_snapshot(tile)
    set_visible = _visible_weight(tile)

    reference = _hidden_array(reset_start_hidden, "reference")
    min_bound = _hidden_array(reset_start_hidden, "min_bound")
    max_bound = _hidden_array(reset_start_hidden, "max_bound")
    if not np.array_equal(min_bound, _hidden_array(reset_hidden, "min_bound")):
        raise RuntimeError("RESET-arm bounds changed unexpectedly.")
    if not np.array_equal(max_bound, _hidden_array(set_hidden, "max_bound")):
        raise RuntimeError("SET-arm bounds changed unexpectedly.")

    output_dir.mkdir(parents=True, exist_ok=False)
    samples_path = output_dir / "native_hfo2_reset_200_samples.npz"
    np.savez_compressed(
        samples_path,
        schema=np.asarray(RESULT_SCHEMA),
        schema_version=np.asarray(1, dtype=np.int64),
        reset_start_apparent_w=reset_start_visible,
        reset_start_persistent_w=_hidden_array(
            reset_start_hidden, "persistent_weights"
        ),
        set_start_apparent_w=set_start_visible,
        set_start_persistent_w=_hidden_array(set_start_hidden, "persistent_weights"),
        reset_200_apparent_w=reset_visible,
        reset_200_persistent_w=_hidden_array(reset_hidden, "persistent_weights"),
        set_200_apparent_w=set_visible,
        set_200_persistent_w=_hidden_array(set_hidden, "persistent_weights"),
        reference=reference,
        min_bound=min_bound,
        max_bound=max_bound,
        reset_bias=_hidden_array(reset_start_hidden, "reset_bias"),
        dwmin_up=_hidden_array(reset_start_hidden, "dwmin_up"),
        dwmin_down=_hidden_array(reset_start_hidden, "dwmin_down"),
    )
    resolved_config_path = output_dir / "config.json"
    resolved_config_path.write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    receipt = {
        "schema": "ebl.lab.hfo2_native_tile_reset_figure6.receipt",
        "schema_version": 1,
        "probe_id": config["probe_id"],
        "status": "sampled",
        "evidence_class": config["evidence_class"],
        "python_executable": str(Path(__import__("sys").executable).resolve()),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "aihwkit_version": aihwkit.__version__,
        "execution_device": "cpu",
        "num_devices": num_devices,
        "construction_seed": device.construction_seed,
        "native_operational_rng": (
            "AIHWKit CPU RealWorldRNG is not exposed for exact replay; the complete "
            "sampled initial state and realized endpoints are preserved in the NPZ"
        ),
        "config_sha256": _sha256(resolved_config_path),
        "samples_sha256": _sha256(samples_path),
    }
    (output_dir / "sampling_receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    arguments = parser.parse_args()
    run(arguments.config.resolve(), arguments.output_dir.resolve())


if __name__ == "__main__":
    main()
