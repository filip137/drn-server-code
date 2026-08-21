from __future__ import annotations

import hashlib
import json
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import torch

from experiments.analyze_conv2_fixed_uniform_init_wmax_sweep import _weight_audit
from experiments.exact_run import load_exact_config, selected_configs
from experiments.prepare_conv2_fixed_uniform_init_wmax_sweep import (
    INITIALIZER_SHA256,
    INITIALIZER_SOURCE,
    PARAMETER_ORDER,
    REPO_ROOT,
    STUDY_ID,
    SURFACES,
    WMAX_CASES,
    prepare,
)


LABS_ROOT = REPO_ROOT / "labs"
if str(LABS_ROOT) not in sys.path:
    sys.path.insert(0, str(LABS_ROOT))

from custom_classes import FlexibleDeepResistiveEnergy  # noqa: E402
from labs.mnist_train import _reset_name_counters, _set_seed  # noqa: E402


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _restore_source(generated: dict, source: dict) -> dict:
    restored = deepcopy(generated)
    for key in ("study_id", "arm_id", "reporting", "weight_ceiling_sweep"):
        if key in source:
            restored[key] = deepcopy(source[key])
        else:
            restored.pop(key, None)
    restored["init_checkpoint_path"] = source["init_checkpoint_path"]
    restored["lab"]["epochs"] = source["lab"]["epochs"]
    restored["model_base"]["weight_max"] = source["model_base"]["weight_max"]
    if "evaluation" in source:
        restored["evaluation"] = deepcopy(source["evaluation"])
    else:
        restored.pop("evaluation", None)
    for key in ("parameter_order", "learning_rates_by_parameter"):
        if key in source:
            restored[key] = deepcopy(source[key])
        else:
            restored.pop(key, None)
    return restored


def _build_and_load(config: dict) -> tuple[list[str], list[torch.Tensor], list[tuple[float, float]]]:
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
    energy.load(INITIALIZER_SOURCE)
    names = []
    states = []
    bounds = []
    for parameter in energy.params():
        name = str(parameter.name)
        names.append(name)
        states.append(parameter.state.detach().cpu().clone())
        if name.startswith(("ConvWeight_", "DenseWeight_")):
            bounds.append((float(parameter.min_cond), float(parameter.max_cond)))
    return names, states, bounds


def test_prepared_configs_freeze_twenty_four_fixed_initializer_cases(tmp_path: Path) -> None:
    manifest = prepare(tmp_path)

    assert manifest["study_id"] == STUDY_ID
    assert manifest["run_count"] == 24
    assert manifest["execution"]["array"] == "0-23%24"
    assert manifest["canonical_lr_handoff"] is False
    assert [row["index"] for row in manifest["runs"]] == list(range(24))
    assert {row["weight_max"] for row in manifest["runs"]} == {
        case.value for case in WMAX_CASES
    }

    checkpoint_paths = set()
    for row in manifest["runs"]:
        path = tmp_path / row["config"]
        config = load_exact_config(path)
        surface = next(
            item
            for item in SURFACES
            if item.scheme == row["scheme"] and item.optimizer == row["optimizer"]
        )
        source = json.loads((REPO_ROOT / surface.source_config).read_text(encoding="utf-8"))
        assert _restore_source(config, source) == source
        assert config["lr"] == list(surface.rates)
        assert config["parameter_order"] == list(PARAMETER_ORDER)
        assert config["lab"]["epochs"] == 10
        assert config["model_base"]["weight_min"] == 1e-5
        assert config["model_base"]["weight_max"] == row["weight_max"]
        assert config["model_base"]["weight_init_mode"] == "bounded_uniform"
        assert config["model_base"]["num_iterations_inference"] == 6
        assert config["model_base"]["num_iterations_training"] == 6
        assert config["evaluation"]["official_test"]["policy"] == "disabled"
        assert config["weight_ceiling_sweep"]["changed_scientific_fields"] == [
            "model_base.weight_max"
        ]
        assert config["weight_ceiling_sweep"]["initializer_checkpoint_sha256"] == (
            INITIALIZER_SHA256
        )
        assert config["learning_rates_by_parameter"]["Bias_0"] == 0.0
        assert config["learning_rates_by_parameter"]["Bias_1"] == 0.0
        checkpoint_paths.add(config["init_checkpoint_path"])
    assert len(checkpoint_paths) == 1


def test_shared_checkpoint_loads_identical_states_with_each_runtime_ceiling(
    tmp_path: Path,
) -> None:
    manifest = prepare(tmp_path)
    reference_names = None
    reference_states = None
    for offset, wmax in enumerate(WMAX_CASES):
        row = manifest["runs"][offset * len(SURFACES)]
        config = json.loads((tmp_path / row["config"]).read_text(encoding="utf-8"))
        names, states, bounds = _build_and_load(config)
        assert bounds and all(bound == (1e-5, wmax.value) for bound in bounds)
        if reference_names is None:
            reference_names = names
            reference_states = states
        else:
            assert names == reference_names
            for observed, expected in zip(states, reference_states, strict=True):
                torch.testing.assert_close(observed, expected, rtol=0.0, atol=0.0)


def test_frozen_uniform_initializer_has_expected_digest() -> None:
    assert _sha256(INITIALIZER_SOURCE) == INITIALIZER_SHA256


def test_jean_zay_wrapper_is_twenty_four_way_exact_run_array() -> None:
    wrapper = (
        REPO_ROOT
        / "experiments/run_conv2_fixed_uniform_init_wmax_sweep_jeanzay.slurm"
    ).read_text(encoding="utf-8")

    assert "#SBATCH --account=umg@v100" in wrapper
    assert "#SBATCH --qos=qos_gpu-t3" in wrapper
    assert "#SBATCH --constraint=v100-16g" in wrapper
    assert "#SBATCH --array=0-23%24" in wrapper
    config_lines = [
        line
        for line in wrapper.splitlines()
        if line.strip().startswith('"${CONFIG_ROOT}/') and line.strip().endswith('.json"')
    ]
    assert len(config_lines) == 24
    assert "TASK_INDEX=$((10#${SLURM_ARRAY_TASK_ID}))" in wrapper
    assert (
        'WMAX_SWEEP_SOURCE_ARCHIVE="${WMAX_SWEEP_SOURCE_ROOT}.tar.gz"'
        in wrapper
    )
    assert "Set the frozen source archive path" not in wrapper
    assert (
        'python -m experiments.exact_run \\\n'
        '  "${CONFIGS[@]}" \\\n'
    ) in wrapper
    assert 'WMAX_SWEEP_CANARY_SMOKE="${WMAX_SWEEP_CANARY_SMOKE:-0}"' in wrapper
    assert "EXACT_RUN_MODE+=(--smoke)" in wrapper
    assert "--checkpoint-every-epoch" in wrapper
    assert "--skip-terminal-official-test" in wrapper
    assert "WMAX_FIXED_INIT_SEMANTIC_PASS" in wrapper

    config_paths = [Path(f"case_{index}.json") for index in range(24)]
    for index in (0, 1, 23):
        assert selected_configs(
            config_paths,
            environ={"SLURM_ARRAY_TASK_ID": str(index)},
        ) == [(index, config_paths[index].resolve())]


def test_weight_audit_measures_endpoints_and_rejects_outside_values(tmp_path: Path) -> None:
    valid = tmp_path / "valid.npz"
    np.savez(
        valid,
        param_names=np.asarray(["ConvWeight_0", "DenseWeight_0", "Bias_0"]),
        ConvWeight_0=np.asarray([1e-5, 2e-4, 5e-5], dtype=np.float32),
        DenseWeight_0=np.asarray([1e-5, 2e-4], dtype=np.float32),
        Bias_0=np.asarray([0.0], dtype=np.float32),
    )
    pooled, layers = _weight_audit(valid, weight_min=1e-5, weight_max=2e-4)
    assert pooled["final_weight_count"] == 5
    assert pooled["final_lower_endpoint_count"] == 2
    assert pooled["final_upper_endpoint_count"] == 2
    assert len(layers) == 2

    invalid = tmp_path / "invalid.npz"
    np.savez(
        invalid,
        param_names=np.asarray(["ConvWeight_0", "Bias_0"]),
        ConvWeight_0=np.asarray([1e-6], dtype=np.float32),
        Bias_0=np.asarray([0.0], dtype=np.float32),
    )
    with pytest.raises(ValueError, match="Invalid bounded weights"):
        _weight_audit(invalid, weight_min=1e-5, weight_max=2e-4)
