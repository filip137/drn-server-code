from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from experiments.run_conv2_zero_bias_adam_eqprop_read_noise_5em4 import (
    CASE_SPECS,
    DEFAULT_STUDY,
    NOISE_STD,
    PACKS,
    load_and_validate_study,
    materialize_configs,
    ordered_config_set_sha256,
    plan,
)
from experiments.validate_conv2_zero_bias_adam_eqprop_read_noise_5em4_pack import (
    runtime_beta_values,
    scientific_outcome,
)


ROOT = Path(__file__).resolve().parents[2]
WRAPPER = (
    ROOT
    / "experiments/run_conv2_zero_bias_adam_eqprop_read_noise_5em4_jeanzay.slurm"
)


def test_frozen_plan_has_three_conv2_cases_in_two_packs() -> None:
    study = load_and_validate_study(DEFAULT_STUDY)
    payload = plan(study)

    assert payload["case_count"] == 3
    assert payload["pack_count"] == 2
    assert payload["packs"] == [[0, 1], [2]]
    assert PACKS == ((0, 1), (2,))
    assert [row["index"] for row in payload["cases"]] == list(range(3))
    assert [row["architecture"] for row in payload["cases"]] == ["conv2"] * 3
    assert [row["scheme"] for row in payload["cases"]] == [
        "baseline",
        "ours",
        "legacy",
    ]
    assert {row["endpoint_read_noise_std"] for row in payload["cases"]} == {
        NOISE_STD
    }
    assert payload["canary_array"] == "0-0"
    assert payload["production_array"] == "0-1%2"


def test_materialized_configs_only_add_noise_to_clean_scientific_contract(
    tmp_path: Path,
) -> None:
    study = load_and_validate_study(DEFAULT_STUDY)
    paths = materialize_configs(study, tmp_path / "configs")

    assert len(paths) == 3
    assert len(ordered_config_set_sha256(paths)) == 64
    contract = study["scientific_contract"]
    for spec, path in zip(CASE_SPECS, paths):
        config = json.loads(path.read_text(encoding="utf-8"))
        source_row = study["source"]["configs"][spec.architecture][spec.scheme]
        source = json.loads((ROOT / source_row["path"]).read_text(encoding="utf-8"))

        assert config["training_algorithm"] == "EP"
        assert config["runtime_dtype"] == "float64"
        assert config["optimizer"] == source["optimizer"]
        assert config["lr"] == source["lr"]
        assert config["learning_rates_by_parameter"] == source[
            "learning_rates_by_parameter"
        ]
        assert config["parameter_order"] == source["parameter_order"]
        assert config["lab"]["epochs"] == contract["epochs"][spec.architecture]
        assert (
            config["model_base"]["num_iterations_inference"]
            == contract["T"][spec.architecture]
        )
        assert (
            config["model_base"]["num_iterations_training"]
            == contract["K"][spec.architecture]
        )
        assert config["beta"] == contract["base_beta"][spec.architecture][spec.scheme]
        assert (
            config["eqprop"]["injected_beta_B"]
            == contract["injected_beta_B"][spec.architecture][spec.scheme]
        )
        assert config["eqprop"]["endpoint_read_noise_std"] == NOISE_STD
        assert (
            config["eqprop"]["endpoint_read_noise_seed"]
            == contract["endpoint_read_noise_seed"]
        )
        assert config["eqprop"]["input_read_noise"] is False
        assert config["evaluation"]["official_test"]["policy"] == "disabled"
        assert config["qualification_source"]["paper_ready"] is False
        assert config["qualification_source"]["gradient_gate_passed_rows"] == 72
        assert config["qualification_source"]["gradient_gate_total_rows"] == 72
        assert (
            config["qualification_source"]["clean_control_result_sha256"]
            == study["source"]["clean_controls"]["result_sha256"][
                spec.architecture
            ][spec.scheme]
        )
        for name in config["parameter_order"]:
            if name.startswith("Bias_"):
                assert config["learning_rates_by_parameter"][name] == 0.0


def test_declared_noise_draws_follow_batches_epochs_and_state_layers() -> None:
    study = load_and_validate_study(DEFAULT_STUDY)
    contract = study["scientific_contract"]
    draws = contract["expected_endpoint_read_noise_draws"]

    assert draws["smoke"] == {"conv2": 6}
    assert draws["production"]["conv2"] == 3438 * 30 * 6


def test_study_rejects_changed_source_and_parent_hashes(tmp_path: Path) -> None:
    original = json.loads(DEFAULT_STUDY.read_text(encoding="utf-8"))

    changed_source = copy.deepcopy(original)
    changed_source["source"]["configs"]["conv2"]["baseline"]["sha256"] = "0" * 64
    source_path = tmp_path / "changed-source.json"
    source_path.write_text(json.dumps(changed_source), encoding="utf-8")
    with pytest.raises(ValueError, match="Source config bytes changed"):
        load_and_validate_study(source_path)

    changed_parent = copy.deepcopy(original)
    changed_parent["source"]["parent_clean_study_config"]["sha256"] = "0" * 64
    parent_path = tmp_path / "changed-parent.json"
    parent_path.write_text(json.dumps(changed_parent), encoding="utf-8")
    with pytest.raises(ValueError, match="parent clean study bytes changed"):
        load_and_validate_study(parent_path)


def test_runtime_beta_scaling_uses_output_row_depth() -> None:
    run_config = {
        "training": {"beta": 0.625},
        "amplification": {"voltage_amp": 4.0, "current_amp": 1.0},
    }
    base, injected = runtime_beta_values(run_config, depth=2)
    assert base == 0.625
    assert injected == 10.0


def test_scientific_outcome_retains_nonfinite_and_uses_strict_stability(
    tmp_path: Path,
) -> None:
    complete_log = tmp_path / "complete.log"
    complete_log.write_text("complete\n", encoding="utf-8")
    stable = scientific_outcome(
        state="complete",
        summary_status="complete",
        exact_returncode=0,
        log_path=complete_log,
        epoch_rows=[
            {"validation_accuracy": 0.90},
            {"validation_accuracy": 0.851},
        ],
        expected_epochs=2,
    )
    boundary = scientific_outcome(
        state="complete",
        summary_status="complete",
        exact_returncode=0,
        log_path=complete_log,
        epoch_rows=[
            {"validation_accuracy": 0.90},
            {"validation_accuracy": 0.85},
        ],
        expected_epochs=2,
    )
    assert stable["stable"] is True
    assert boundary["stable"] is False

    failed_log = tmp_path / "failed.log"
    failed_log.write_text(
        "NonFiniteTrainingError: gradient failed at epoch=4, batch=17 with "
        "12 non-finite elements\n",
        encoding="utf-8",
    )
    failed = scientific_outcome(
        state="failed",
        summary_status="failed",
        exact_returncode=1,
        log_path=failed_log,
        epoch_rows=[{"validation_accuracy": 0.75}],
        expected_epochs=30,
    )
    assert failed["terminal_kind"] == "nonfinite_training"
    assert failed["scientific_terminal"] is True
    assert failed["stable"] is False


def test_slurm_wrapper_freezes_umg_two_pack_production_contract() -> None:
    text = WRAPPER.read_text(encoding="utf-8")

    required = (
        "#SBATCH --account=umg@v100",
        "#SBATCH --partition=gpu_p13",
        "#SBATCH --qos=qos_gpu-t3",
        "#SBATCH --constraint=v100-32g",
        "#SBATCH --cpus-per-task=16",
        "#SBATCH --time=08:00:00",
        "#SBATCH --array=0-1%2",
        'PACK_MAP=("0 1" "2")',
        "C2RN_RUN_MODE",
        "EXPECTED_TASK_COUNT=1",
        "EXPECTED_TASK_COUNT=2",
        "EXACT_MODE_ARGS=(--smoke)",
        'PARENT_JOB_ID="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID}}"',
        '[[ -n "${SLURM_ARRAY_TASK_COUNT:-}" ]]',
        "experiments.exact_run",
        "experiments.validate_conv2_zero_bias_adam_eqprop_read_noise_5em4_pack",
        "C2RN_ENVIRONMENT_PASS",
        "C2RN_SOURCE_PASS",
        "C2RN_SEMANTIC_PASS",
        'PACK_EXECUTION_MODE="singleton"',
        "--skip-terminal-official-test",
        "Refusing duplicate",
    )
    for value in required:
        assert value in text
    assert "--gres=gpu:1" in text
    assert ">&quot;" not in text
    assert " &\n" in text


def test_canary_summary_is_complete_but_production_is_scientific_terminal() -> None:
    validator = (
        ROOT
        / "experiments/validate_conv2_zero_bias_adam_eqprop_read_noise_5em4_pack.py"
    ).read_text(encoding="utf-8")
    assert 'summary_status = "complete" if expected_smoke else "scientific_terminal"' in validator
    assert "Canary logical run" in validator
