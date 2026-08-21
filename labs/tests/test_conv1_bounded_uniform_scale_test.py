from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path

import torch

from experiments.exact_run import load_exact_config
from experiments.prepare_conv1_bounded_uniform_scale_test import (
    CASES,
    DEFAULT_CONFIG_ROOT,
    REPO_ROOT,
    RESULT_ASSET,
    SCALE_FACTOR,
    SCALED_WEIGHT_MAX,
    SCALED_WEIGHT_MIN,
    SOURCE_ASSET,
    SOURCE_ASSET_SHA256,
    SOURCE_WEIGHT_MAX,
    SOURCE_WEIGHT_MIN,
    STUDY_ID,
    _scaled_checkpoint,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _restore_source(config: dict) -> dict:
    restored = deepcopy(config)
    transfer = restored.pop("absolute_scale_transfer")
    source = json.loads((REPO_ROOT / transfer["source_config"]).read_text(encoding="utf-8"))
    restored.pop("study_id")
    restored.pop("arm_id")
    parameter_order = restored.pop("parameter_order")
    restored.pop("learning_rates_by_parameter")
    restored.pop("evaluation")
    restored.pop("reporting")
    restored["init_checkpoint_path"] = source["init_checkpoint_path"]
    restored["model_base"]["weight_min"] = SOURCE_WEIGHT_MIN
    restored["model_base"]["weight_max"] = SOURCE_WEIGHT_MAX
    restored["lr"] = [
        transfer["source_learning_rates_by_parameter"][name]
        for name in parameter_order
    ]
    restored["optimizer"]["learning_rate"] = restored["lr"]
    return restored


def test_scaled_checkpoint_preserves_source_quantiles_and_zero_bias() -> None:
    assert _sha256(REPO_ROOT / SOURCE_ASSET) == SOURCE_ASSET_SHA256
    source = torch.load(REPO_ROOT / SOURCE_ASSET, map_location="cpu", weights_only=True)
    payload = _scaled_checkpoint()
    scaled = payload["checkpoint"]

    assert source["schema"] == scaled["schema"]
    for descriptor, source_state, scaled_state in zip(
        source["schema"], source["states"], scaled["states"], strict=True
    ):
        parameter_type = descriptor["type"]
        if parameter_type.endswith((".ConvWeight", ".DenseWeight")):
            assert torch.equal(scaled_state, source_state * SCALE_FACTOR)
            assert float(scaled_state.min()) >= SCALED_WEIGHT_MIN
            assert float(scaled_state.max()) < SCALED_WEIGHT_MAX
        else:
            assert parameter_type.endswith(".Bias")
            assert torch.count_nonzero(source_state).item() == 0
            assert torch.equal(scaled_state, source_state)


def test_frozen_configs_change_only_declared_absolute_scale_fields() -> None:
    manifest = json.loads(
        (DEFAULT_CONFIG_ROOT / "study_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["study_id"] == STUDY_ID
    assert manifest["run_count"] == 3
    assert manifest["canonical_lr_handoff"] is False
    assert manifest["initializer"]["scale_factor"] == SCALE_FACTOR
    assert _sha256(REPO_ROOT / RESULT_ASSET) == manifest["initializer"][
        "scaled_checkpoint_sha256"
    ]

    for case, row in zip(CASES, manifest["runs"], strict=True):
        config_path = DEFAULT_CONFIG_ROOT / row["config"]
        config = load_exact_config(config_path)
        source = json.loads((REPO_ROOT / case.source_config).read_text(encoding="utf-8"))
        assert _restore_source(config) == source
        assert config["model_base"]["weight_init_mode"] == "bounded_uniform"
        assert config["model_base"]["weight_min"] == SCALED_WEIGHT_MIN
        assert config["model_base"]["weight_max"] == SCALED_WEIGHT_MAX
        assert config["lab"]["epochs"] == 3
        assert config["optimizer"]["name"] == "Adam"
        assert config["evaluation"]["official_test"]["policy"] == "disabled"
        assert config["lr"] == [rate * SCALE_FACTOR for rate in case.rates]
        assert config["learning_rates_by_parameter"]["Bias_0"] == 0.0
        assert _sha256(config_path) == row["config_sha256"]


def test_local_wrapper_runs_three_exact_configs_with_epoch_checkpoints() -> None:
    wrapper = (
        REPO_ROOT / "experiments/run_conv1_bounded_uniform_scale1e4_local.sh"
    ).read_text(encoding="utf-8")

    assert wrapper.count("conv1_") >= 3
    assert wrapper.count("_adam.json\"") == 3
    assert "-m experiments.exact_run" in wrapper
    assert "--checkpoint-every-epoch" in wrapper
    assert "--target main" in wrapper
