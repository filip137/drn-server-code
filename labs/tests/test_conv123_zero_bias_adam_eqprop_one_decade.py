from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from experiments.run_conv123_zero_bias_adam_eqprop_one_decade import (
    CASE_SPECS,
    DEFAULT_STUDY,
    PACKS,
    load_and_validate_study,
    materialize_configs,
    ordered_config_set_sha256,
    plan,
)
from experiments.validate_conv123_zero_bias_adam_eqprop_one_decade_pack import (
    runtime_beta_values,
    scientific_outcome,
)


ROOT = Path(__file__).resolve().parents[2]
WRAPPER = ROOT / "experiments/run_conv123_zero_bias_adam_eqprop_one_decade_jeanzay.slurm"


def test_frozen_plan_has_nine_cases_in_five_requested_packs() -> None:
    study = load_and_validate_study(DEFAULT_STUDY)
    payload = plan(study)

    assert payload["case_count"] == 9
    assert payload["pack_count"] == 5
    assert payload["packs"] == [[0, 1], [2, 3], [4, 5], [6, 7], [8]]
    assert PACKS == ((0, 1), (2, 3), (4, 5), (6, 7), (8,))
    assert [row["index"] for row in payload["cases"]] == list(range(9))
    assert [row["architecture"] for row in payload["cases"]] == [
        "conv1",
        "conv3",
        "conv1",
        "conv3",
        "conv1",
        "conv3",
        "conv2",
        "conv2",
        "conv2",
    ]
    assert payload["protocol_deviations"]["one_decade_gradient_gate"][
        "passing_layer_batch_rows"
    ] == 213
    assert payload["protocol_deviations"]["paper_ready"] is False


def test_materialized_configs_preserve_exact_lrs_zero_bias_and_contract(
    tmp_path: Path,
) -> None:
    study = load_and_validate_study(DEFAULT_STUDY)
    paths = materialize_configs(study, tmp_path / "configs")

    assert len(paths) == 9
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
        assert config["eqprop"]["endpoint_read_noise_std"] == 0.0
        assert config["evaluation"]["official_test"]["policy"] == "disabled"
        assert config["qualification_source"]["paper_ready"] is False
        assert config["qualification_source"]["gradient_gate_passed_rows"] == 213
        for name in config["parameter_order"]:
            if name.startswith("Bias_"):
                assert config["learning_rates_by_parameter"][name] == 0.0


def test_study_rejects_changed_source_hash(tmp_path: Path) -> None:
    study = copy.deepcopy(json.loads(DEFAULT_STUDY.read_text(encoding="utf-8")))
    study["source"]["configs"]["conv1"]["baseline"]["sha256"] = "0" * 64
    path = tmp_path / "changed-study.json"
    path.write_text(json.dumps(study), encoding="utf-8")

    with pytest.raises(ValueError, match="Source config bytes changed"):
        load_and_validate_study(path)


def test_runtime_beta_scaling_uses_output_row_depth() -> None:
    run_config = {
        "training": {"beta": 0.046875},
        "amplification": {"voltage_amp": 4.0, "current_amp": 1.0},
    }
    base, injected = runtime_beta_values(run_config, depth=3)
    assert base == 0.046875
    assert injected == 3.0


def test_scientific_outcome_uses_strict_five_point_stability_rule(
    tmp_path: Path,
) -> None:
    log = tmp_path / "complete.log"
    log.write_text("complete\n", encoding="utf-8")
    stable = scientific_outcome(
        state="complete",
        summary_status="complete",
        exact_returncode=0,
        log_path=log,
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
        log_path=log,
        epoch_rows=[
            {"validation_accuracy": 0.90},
            {"validation_accuracy": 0.85},
        ],
        expected_epochs=2,
    )
    assert stable["stable"] is True
    assert boundary["stable"] is False


def test_nonfinite_training_is_retained_as_scientific_terminal(tmp_path: Path) -> None:
    log = tmp_path / "failed.log"
    log.write_text(
        "NonFiniteTrainingError: gradient failed at epoch=4, batch=17 with "
        "12 non-finite elements\n",
        encoding="utf-8",
    )
    outcome = scientific_outcome(
        state="failed",
        summary_status="failed",
        exact_returncode=1,
        log_path=log,
        epoch_rows=[{"validation_accuracy": 0.75}],
        expected_epochs=30,
    )
    assert outcome["terminal_kind"] == "nonfinite_training"
    assert outcome["scientific_terminal"] is True
    assert outcome["stable"] is False


def test_slurm_wrapper_freezes_umg_packing_and_semantic_validation() -> None:
    text = WRAPPER.read_text(encoding="utf-8")

    required = (
        "#SBATCH --account=umg@v100",
        "#SBATCH --constraint=v100-32g",
        "#SBATCH --cpus-per-task=16",
        "#SBATCH --time=08:00:00",
        "#SBATCH --array=0-4%5",
        'PACK_MAP=("0 1" "2 3" "4 5" "6 7" "8")',
        "experiments.exact_run",
        "experiments.validate_conv123_zero_bias_adam_eqprop_one_decade_pack",
        "C123OD_ENVIRONMENT_PASS",
        "C123OD_SOURCE_PASS",
        "C123OD_SEMANTIC_PASS",
        "--skip-terminal-official-test",
    )
    for value in required:
        assert value in text
    assert "--gres=gpu:1" in text
    assert "&\n" in text
