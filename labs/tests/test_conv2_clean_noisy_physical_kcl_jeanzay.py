from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
import re
import subprocess

from experiments import run_conv123_zero_bias_adam_eqprop_one_decade as clean
from experiments import run_conv2_zero_bias_adam_eqprop_read_noise_5em4 as noisy


ROOT = Path(__file__).resolve().parents[2]
WRAPPER = (
    ROOT
    / "experiments/run_conv2_clean_noisy_legacy_adam_physical_kcl_jeanzay.sh"
)
CLEAN_STUDY = (
    ROOT
    / "configs/conv/"
    "perfectdiode_conv123_zero_bias_adam_eqprop_one_decade_ordinary_mnist_"
    "10_30_30ep_seed0_20260816_v1.json"
)
NOISY_STUDY = (
    ROOT
    / "configs/conv/"
    "perfectdiode_conv2_zero_bias_adam_eqprop_one_decade_sigma_5em4_"
    "ordinary_mnist_30ep_seed0_20260817_v1.json"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _legacy_config(runner: object, study_path: Path) -> dict:
    study = runner.load_and_validate_study(study_path)
    spec = next(
        row
        for row in runner.CASE_SPECS
        if row.architecture == "conv2" and row.scheme == "legacy"
    )
    return runner._resolved_case_config(study, spec)


def test_wrapper_syntax_inline_guard_and_terminal_validation() -> None:
    result = subprocess.run(
        ["bash", "-n", str(WRAPPER)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr

    text = WRAPPER.read_text(encoding="utf-8")
    required = (
        "clean_eqprop)",
        "noisy_eqprop_5em4)",
        'RUNNER_MODULE="experiments.run_conv123_zero_bias_adam_eqprop_one_decade"',
        'RUNNER_MODULE="experiments.run_conv2_zero_bias_adam_eqprop_read_noise_5em4"',
        "EXPECTED_CONFIG_COUNT=9",
        "EXPECTED_CONFIG_COUNT=3",
        "EXACT_INDEX=8",
        "EXACT_INDEX=2",
        'MODULE_ID="pytorch-gpu/py3/2.5.0"',
        'model["num_iterations_inference"] == 6',
        'model["num_iterations_training"] == 6',
        'float(config["eqprop"]["injected_beta_B"])',
        "0.0001171875",
        "experiments.exact_run",
        "experiments.reporting validate-run",
        "--skip-terminal-official-test",
        "PHYSKCL_C2_SOURCE_PASS",
        "PHYSKCL_C2_SEMANTIC_PASS",
        "Refusing duplicate completed case",
        "Refusing concurrent duplicate case",
    )
    for value in required:
        assert value in text

    programs = re.findall(r"<<'PY'\n(.*?)\nPY", text, flags=re.DOTALL)
    assert len(programs) == 1
    ast.parse(programs[0])


def test_exact_legacy_configs_preserve_same_beta_lr_and_operating_point() -> None:
    clean_config = _legacy_config(clean, CLEAN_STUDY)
    noisy_config = _legacy_config(noisy, NOISY_STUDY)

    assert clean_config["arm_id"].startswith("conv2_legacy_")
    assert noisy_config["arm_id"].startswith("conv2_legacy_")
    for config, noise_std in ((clean_config, 0.0), (noisy_config, 5e-4)):
        model = config["model_base"]
        assert config["training_algorithm"] == "EP"
        assert config["runtime_dtype"] == "float64"
        assert config["optimizer"]["name"] == "Adam"
        assert config["lab"]["epochs"] == 30
        assert model["num_iterations_inference"] == 6
        assert model["num_iterations_training"] == 6
        assert model["voltage_amp"] == 4.0
        assert model["current_amp"] == 0.25
        assert model["input_gain"] == 100.0
        assert model["non_linearity"] == "perfect_diode"
        assert isinstance(model["exponential_diode_param"], dict)
        assert isinstance(model["quadratic_diode_param"], dict)
        assert config["beta"] == 0.0001171875
        assert config["eqprop"]["injected_beta_B"] == 0.03
        assert config["eqprop"]["endpoint_read_noise_std"] == noise_std
        assert config["evaluation"]["official_test"]["policy"] == "disabled"
        assert all(
            config["learning_rates_by_parameter"][name] == 0.0
            for name in config["parameter_order"]
            if name.startswith("Bias_")
        )

    assert clean_config["parameter_order"] == noisy_config["parameter_order"]
    assert clean_config["learning_rates_by_parameter"] == noisy_config[
        "learning_rates_by_parameter"
    ]
    assert clean_config["optimizer"] == noisy_config["optimizer"]


def test_materialized_config_hashes_match_frozen_wrapper_contract(
    tmp_path: Path,
) -> None:
    clean_paths = clean.materialize_configs(
        clean.load_and_validate_study(CLEAN_STUDY), tmp_path / "clean"
    )
    noisy_paths = noisy.materialize_configs(
        noisy.load_and_validate_study(NOISY_STUDY), tmp_path / "noisy"
    )

    assert len(clean_paths) == 9
    assert len(noisy_paths) == 3
    assert _sha256(clean_paths[8]) == (
        "83609bea33d5ad85f4def238ccb8d98a7a7bbcf1f54c58cfca910cf3aa404c2a"
    )
    assert _sha256(noisy_paths[2]) == (
        "863f0ee3aab5b103253492db68fbd85e10c17f11580f3bc3b45725843ccfe1ea"
    )
    assert json.loads(clean_paths[8].read_text(encoding="utf-8"))[
        "learning_rates_by_parameter"
    ] == json.loads(noisy_paths[2].read_text(encoding="utf-8"))[
        "learning_rates_by_parameter"
    ]
