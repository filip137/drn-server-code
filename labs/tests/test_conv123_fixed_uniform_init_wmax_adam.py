from __future__ import annotations

import hashlib
import json
import sys
from copy import deepcopy
from pathlib import Path

import torch

from experiments.exact_run import load_exact_config, selected_configs
from experiments.prepare_conv123_fixed_uniform_init_wmax_adam import (
    ARCHITECTURES,
    ARCHITECTURE_BY_NAME,
    REPO_ROOT,
    STUDY_ID,
    SURFACES,
    WMAX_CASES,
    Architecture,
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


def _build_and_load(
    config: dict,
    architecture: Architecture,
) -> tuple[list[str], list[torch.Tensor], list[tuple[float, float]]]:
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
    energy.load(REPO_ROOT / architecture.initializer_source)
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


def test_prepared_configs_freeze_twenty_seven_adam_cases(tmp_path: Path) -> None:
    manifest = prepare(tmp_path)

    assert manifest["study_id"] == STUDY_ID
    assert manifest["run_count"] == 27
    assert manifest["run_count_by_architecture"] == {
        "conv1": 9,
        "conv2": 9,
        "conv3": 9,
    }
    assert manifest["canonical_lr_handoff"] is False
    assert manifest["execution"]["account"] == "fmu@v100"
    assert manifest["execution"]["dossier"] == "AD010913993R3"
    assert manifest["execution"]["maximum_concurrent_gpus"] == 27
    assert set(manifest["ordered_config_set_sha256_by_architecture"]) == {
        "conv1",
        "conv2",
        "conv3",
    }

    seen_config_hashes = set()
    checkpoint_paths: dict[str, set[str]] = {
        architecture.name: set() for architecture in ARCHITECTURES
    }
    rows_by_architecture = {
        architecture.name: [
            row for row in manifest["runs"] if row["architecture"] == architecture.name
        ]
        for architecture in ARCHITECTURES
    }
    for architecture in ARCHITECTURES:
        rows = rows_by_architecture[architecture.name]
        assert [row["index"] for row in rows] == list(range(9))
        assert [row["scheme"] for row in rows] == [
            "baseline",
            "ours",
            "legacy",
        ] * 3
        assert {row["weight_max"] for row in rows} == {
            case.value for case in WMAX_CASES
        }
        assert {row["optimizer"] for row in rows} == {"Adam"}

        for row in rows:
            path = tmp_path / row["config"]
            config = load_exact_config(path)
            surface = next(
                item
                for item in SURFACES
                if item.architecture == architecture.name
                and item.scheme == row["scheme"]
            )
            source = json.loads(
                (REPO_ROOT / surface.source_config).read_text(encoding="utf-8")
            )
            assert _restore_source(config, source) == source
            assert config["optimizer"]["name"] == "Adam"
            assert config["lr"] == list(surface.rates)
            assert config["parameter_order"] == list(architecture.parameter_order)
            assert config["lab"]["epochs"] == architecture.epochs
            assert config["model_base"]["weight_min"] == 1e-5
            assert config["model_base"]["weight_max"] == row["weight_max"]
            assert config["model_base"]["weight_init_mode"] == "bounded_uniform"
            assert config["model_base"]["num_iterations_inference"] == (
                architecture.iterations
            )
            assert config["model_base"]["num_iterations_training"] == (
                architecture.iterations
            )
            assert config["model_base"]["input_gain"] == architecture.input_gain
            assert config["evaluation"]["official_test"]["policy"] == "disabled"
            assert config["weight_ceiling_sweep"]["source_transform_fields"] == [
                "lab.epochs",
                "model_base.weight_max",
            ]
            assert (
                config["weight_ceiling_sweep"]["initializer_checkpoint_sha256"]
                == architecture.initializer_sha256
            )
            assert all(
                config["learning_rates_by_parameter"][name] == 0.0
                for name in architecture.bias_names
            )
            checkpoint_paths[architecture.name].add(config["init_checkpoint_path"])
            seen_config_hashes.add(row["config_sha256"])

    assert all(len(paths) == 1 for paths in checkpoint_paths.values())
    assert len(seen_config_hashes) == 27


def test_shared_checkpoint_loads_identical_states_for_each_runtime_ceiling(
    tmp_path: Path,
) -> None:
    manifest = prepare(tmp_path)
    for architecture in ARCHITECTURES:
        rows = [
            row
            for row in manifest["runs"]
            if row["architecture"] == architecture.name and row["scheme"] == "baseline"
        ]
        reference_names = None
        reference_states = None
        for row in rows:
            config = json.loads((tmp_path / row["config"]).read_text(encoding="utf-8"))
            names, states, bounds = _build_and_load(config, architecture)
            assert bounds and all(
                bound == (1e-5, row["weight_max"]) for bound in bounds
            )
            if reference_names is None:
                reference_names = names
                reference_states = states
            else:
                assert names == reference_names
                for observed, expected in zip(states, reference_states, strict=True):
                    torch.testing.assert_close(observed, expected, rtol=0.0, atol=0.0)


def test_frozen_uniform_initializers_have_expected_digests() -> None:
    for architecture in ARCHITECTURES:
        assert _sha256(REPO_ROOT / architecture.initializer_source) == (
            architecture.initializer_sha256
        )


def test_jean_zay_wrapper_is_nine_way_architecture_selected_array() -> None:
    wrapper = (
        REPO_ROOT
        / "experiments/run_conv123_fixed_uniform_init_wmax_adam_jeanzay.slurm"
    ).read_text(encoding="utf-8")

    assert "#SBATCH --account=fmu@v100" in wrapper
    assert "#SBATCH --partition=gpu_p13" in wrapper
    assert "#SBATCH --qos=qos_gpu-t3" in wrapper
    assert "#SBATCH --constraint=v100-16g" in wrapper
    assert "#SBATCH --array=0-8%9" in wrapper
    assert ': "${WMAX_SWEEP_ARCH:?Set WMAX_SWEEP_ARCH' in wrapper
    assert 'case "${WMAX_SWEEP_ARCH}" in' in wrapper
    config_lines = [
        line
        for line in wrapper.splitlines()
        if line.strip().startswith('"${CONFIG_ROOT}/') and line.strip().endswith('.json"')
    ]
    assert len(config_lines) == 9
    assert "TASK_INDEX=$((10#${SLURM_ARRAY_TASK_ID}))" in wrapper
    assert 'WMAX_SWEEP_SOURCE_ARCHIVE="${WMAX_SWEEP_SOURCE_ROOT}.tar.gz"' in wrapper
    assert 'python -m experiments.exact_run \\\n  "${CONFIGS[@]}" \\' in wrapper
    assert 'WMAX_SWEEP_REMOTE_SMOKE="${WMAX_SWEEP_REMOTE_SMOKE:-0}"' in wrapper
    assert "EXACT_RUN_MODE+=(--smoke)" in wrapper
    assert "--checkpoint-every-epoch" in wrapper
    assert "--skip-terminal-official-test" in wrapper
    assert 'task_summaries/production/${WMAX_SWEEP_ARCH}' in wrapper
    assert "WMAX_ADAM_SEMANTIC_PASS" in wrapper

    config_paths = [Path(f"case_{index}.json") for index in range(9)]
    assert selected_configs(config_paths, environ={}) == [
        (index, path.resolve()) for index, path in enumerate(config_paths)
    ]
    for index in (0, 1, 8):
        assert selected_configs(
            config_paths,
            environ={"SLURM_ARRAY_TASK_ID": str(index)},
        ) == [(index, config_paths[index].resolve())]


def test_architecture_lookup_is_complete() -> None:
    assert ARCHITECTURE_BY_NAME == {
        architecture.name: architecture for architecture in ARCHITECTURES
    }
