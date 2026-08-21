from __future__ import annotations

import ast
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess

import pytest

from experiments.analyze_conv123_centered_eqprop_beta_collapse import classify_pair
from experiments.run_conv123_centered_eqprop_beta_collapse import (
    BETA_TIERS,
    CASE_SPECS,
    DEFAULT_STUDY,
    PACKS,
    _resolved_case_config,
    load_and_validate_study,
    materialize_configs,
    ordered_config_set_sha256,
    plan,
)
from experiments.validate_conv123_eqprop_beta_collapse_pack import (
    _scientific_outcome,
    runtime_beta_values,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
WRAPPER = REPO_ROOT / "experiments/run_conv123_centered_eqprop_beta_collapse_jeanzay.slurm"
STUDY_RELATIVE = Path(
    "configs/conv/perfectdiode_conv123_centered_float64_eqprop_"
    "beta_collapse_boundary_vs_1decade_sgd_adam_10ep_seed0_20260815_v1.json"
)


def _read_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _base_environment(tmp_path: Path, *, task: int) -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "C123BC_SOURCE_ROOT": str(tmp_path / "missing-source"),
            "C123BC_SOURCE_ARCHIVE": str(tmp_path / "missing-source.tar.gz"),
            "C123BC_CONFIG_ROOT": str(tmp_path / "missing-configs"),
            "C123BC_RESULT_ROOT": str(tmp_path / "results"),
            "C123BC_DATASET_ROOT": str(tmp_path / "mnist"),
            "C123BC_ENVIRONMENT_ID": "test-jean-zay-umg-v100-32g",
            "C123BC_STUDY_CONFIG_SHA256": "0" * 64,
            "C123BC_CONFIG_SET_SHA256": "0" * 64,
            "EXPERIMENT_SOURCE_COMMIT": "0" * 40,
            "EXPERIMENT_SOURCE_ARCHIVE_SHA256": "0" * 64,
            "SLURM_JOB_ID": "123",
            "SLURM_ARRAY_JOB_ID": "123",
            "SLURM_ARRAY_TASK_ID": str(task),
        }
    )
    return env


def _run_wrapper(env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", "-x", str(WRAPPER)],
        cwd=REPO_ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )


def _stage_minimal_guard_inputs(
    tmp_path: Path, *, task: int
) -> tuple[dict[str, str], Path]:
    source_root = tmp_path / "source"
    study_path = source_root / STUDY_RELATIVE
    study_path.parent.mkdir(parents=True)
    study_path.write_text("{}\n", encoding="utf-8")
    archive = tmp_path / "source.tar.gz"
    archive.write_bytes(b"frozen source\n")
    config_root = tmp_path / "configs"
    config_root.mkdir()
    digests: list[str] = []
    for index in range(36):
        path = config_root / f"{index:02d}.json"
        path.write_text(f'{{"index": {index}}}\n', encoding="utf-8")
        digests.append(hashlib.sha256(path.read_bytes()).hexdigest())
    config_set_sha = hashlib.sha256(
        "".join(f"{digest}\n" for digest in digests).encode("ascii")
    ).hexdigest()
    dataset_root = tmp_path / "mnist"
    dataset_root.mkdir()
    result_root = tmp_path / "results"
    env = _base_environment(tmp_path, task=task)
    env.update(
        {
            "C123BC_SOURCE_ROOT": str(source_root),
            "C123BC_SOURCE_ARCHIVE": str(archive),
            "C123BC_CONFIG_ROOT": str(config_root),
            "C123BC_RESULT_ROOT": str(result_root),
            "C123BC_DATASET_ROOT": str(dataset_root),
            "C123BC_STUDY_CONFIG_SHA256": hashlib.sha256(
                study_path.read_bytes()
            ).hexdigest(),
            "C123BC_CONFIG_SET_SHA256": config_set_sha,
            "EXPERIMENT_SOURCE_ARCHIVE_SHA256": hashlib.sha256(
                archive.read_bytes()
            ).hexdigest(),
            "SLURM_ARRAY_TASK_COUNT": "18",
        }
    )
    return env, result_root


def test_frozen_matrix_and_pack_plan() -> None:
    study = load_and_validate_study(DEFAULT_STUDY)
    payload = plan(study)

    assert len(CASE_SPECS) == 36
    assert PACKS == tuple((index, index + 1) for index in range(0, 36, 2))
    assert payload["array"] == "0-17%18"
    assert payload["packs"] == [list(pack) for pack in PACKS]
    assert payload["collapse_definition"] == {
        "nonfinite_training": True,
        "final_validation_drop_from_best_pp_at_least": 5.0,
    }

    for pack_index, (left_index, right_index) in enumerate(PACKS):
        left = CASE_SPECS[left_index]
        right = CASE_SPECS[right_index]
        if pack_index < 12:
            assert (left.architecture, right.architecture) == ("conv1", "conv3")
            assert (left.scheme, left.optimizer, left.beta_tier) == (
                right.scheme,
                right.optimizer,
                right.beta_tier,
            )
        else:
            assert left.architecture == right.architecture == "conv2"
            assert (left.scheme, left.optimizer) == (right.scheme, right.optimizer)
            assert (left.beta_tier, right.beta_tier) == BETA_TIERS

    observed = {
        (row["architecture"], row["scheme"], row["beta_tier"]): row[
            "injected_beta_B"
        ]
        for row in payload["cases"]
    }
    assert observed == {
        ("conv1", "baseline", "boundary"): 1000.0,
        ("conv1", "baseline", "one_decade_lower"): 100.0,
        ("conv1", "ours", "boundary"): 300.0,
        ("conv1", "ours", "one_decade_lower"): 30.0,
        ("conv1", "legacy", "boundary"): 30.0,
        ("conv1", "legacy", "one_decade_lower"): 3.0,
        ("conv2", "baseline", "boundary"): 1000.0,
        ("conv2", "baseline", "one_decade_lower"): 100.0,
        ("conv2", "ours", "boundary"): 100.0,
        ("conv2", "ours", "one_decade_lower"): 10.0,
        ("conv2", "legacy", "boundary"): 0.3,
        ("conv2", "legacy", "one_decade_lower"): 0.03,
        ("conv3", "baseline", "boundary"): 1000.0,
        ("conv3", "baseline", "one_decade_lower"): 100.0,
        ("conv3", "ours", "boundary"): 30.0,
        ("conv3", "ours", "one_decade_lower"): 3.0,
        ("conv3", "legacy", "boundary"): 0.01,
        ("conv3", "legacy", "one_decade_lower"): 0.001,
    }


def test_materialized_configs_preserve_exact_lrs_and_science(tmp_path: Path) -> None:
    study = load_and_validate_study(DEFAULT_STUDY)
    paths = materialize_configs(study, tmp_path / "resolved")

    assert len(paths) == 36
    assert len(ordered_config_set_sha256(paths)) == 64
    for spec, path in zip(CASE_SPECS, paths, strict=True):
        resolved = _read_json(path)
        source_row = study["source"]["scheme_configs"][spec.architecture][
            spec.scheme
        ][spec.optimizer]
        source = _read_json(REPO_ROOT / source_row["path"])
        assert resolved["learning_rates_by_parameter"] == source[
            "learning_rates_by_parameter"
        ]
        assert resolved["lr"] == source["lr"]
        assert resolved["optimizer"] == source["optimizer"]
        assert resolved["training_algorithm"] == "EP"
        assert resolved["runtime_dtype"] == "float64"
        assert resolved["lab"]["epochs"] == 10
        assert resolved["model_base"]["num_iterations_inference"] == 8
        assert resolved["model_base"]["num_iterations_training"] == 8
        assert resolved["model_base"]["weight_min"] == 0.0
        assert resolved["model_base"]["weight_max"] == 100.0
        assert resolved["model_base"]["weight_init_mode"] == "kaiming_uniform"
        assert resolved["model_base"]["exponential_diode_param"] == {
            "I_s": 1.0e-6,
            "V_off": 0.0,
            "V_t": 0.025,
        }
        assert resolved["model_base"]["quadratic_diode_param"] == {
            "diode_conductance": 1.0,
            "v_max": 1.0e6,
            "v_min": -1.0e6,
        }
        assert resolved["datasets"]["mnist"]["params"]["affine_config"] is None
        assert resolved["evaluation"]["official_test"]["policy"] == "disabled"
        assert resolved["eqprop"]["endpoint_read_noise_std"] == 0.0
        assert resolved["eqprop"]["input_read_noise"] is False


def test_resolved_beta_scaling() -> None:
    study = load_and_validate_study(DEFAULT_STUDY)
    for spec in CASE_SPECS:
        config = _resolved_case_config(study, spec)
        factor = float(config["eqprop"]["amplification_factor"])
        assert float(config["beta"]) * factor == pytest.approx(
            float(config["eqprop"]["injected_beta_B"])
        )


def test_wrapper_has_umg_parallel_resource_contract_and_valid_syntax() -> None:
    result = subprocess.run(
        ["bash", "-n", str(WRAPPER)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr

    text = WRAPPER.read_text(encoding="utf-8")
    expected_directives = {
        "#SBATCH --account=umg@v100",
        "#SBATCH --partition=gpu_p13",
        "#SBATCH --qos=qos_gpu-t3",
        "#SBATCH --constraint=v100-32g",
        "#SBATCH --gres=gpu:1",
        "#SBATCH --cpus-per-task=16",
        "#SBATCH --time=04:00:00",
        "#SBATCH --array=0-17%18",
    }
    assert expected_directives <= set(text.splitlines())
    assert '"34 35"' in text
    assert 'child_pid="$!"' in text
    assert 'pids+=("${child_pid}")' in text
    assert 'wait "${pids[offset]}"' in text
    assert "experiments.validate_conv123_eqprop_beta_collapse_pack" in text
    assert "--execution-mode concurrent" in text
    assert "C123BC_SEMANTIC_PASS" in text
    assert "if (( child_status" not in text

    heredocs = re.findall(r"<<'PY'\n(.*?)\nPY", text, flags=re.DOTALL)
    assert len(heredocs) == 1
    ast.parse(heredocs[0])


def test_array_count_and_task_range_fail_closed(tmp_path: Path) -> None:
    wrong_count = _base_environment(tmp_path, task=0)
    wrong_count["SLURM_ARRAY_TASK_COUNT"] = "17"
    count_result = _run_wrapper(wrong_count)
    assert count_result.returncode == 3
    assert "Expected 18 array tasks; got 17." in count_result.stderr

    out_of_range = _base_environment(tmp_path, task=18)
    out_of_range["SLURM_ARRAY_TASK_COUNT"] = "18"
    range_result = _run_wrapper(out_of_range)
    assert range_result.returncode == 3
    assert "Array task 18 is outside [0,17]." in range_result.stderr


def test_missing_optional_array_count_reaches_source_guard(tmp_path: Path) -> None:
    env = _base_environment(tmp_path, task=0)
    env.pop("SLURM_ARRAY_TASK_COUNT", None)
    result = _run_wrapper(env)
    assert result.returncode == 3
    assert "Frozen source directory or archive is missing." in result.stderr
    assert "array tasks" not in result.stderr


def test_terminal_path_and_lock_prevent_duplicate_pack(tmp_path: Path) -> None:
    env, result_root = _stage_minimal_guard_inputs(tmp_path, task=17)
    summary = result_root / "task_summaries/production/packs/17.json"
    summary.parent.mkdir(parents=True)
    summary.write_text("[]\n", encoding="utf-8")
    terminal_result = _run_wrapper(env)
    assert terminal_result.returncode == 17
    assert "Refusing duplicate production task 17" in terminal_result.stderr

    summary.unlink()
    lock = result_root / "task_locks/production/17.lock"
    lock.mkdir(parents=True)
    lock_result = _run_wrapper(env)
    assert lock_result.returncode == 17
    assert "Refusing concurrent duplicate production task 17" in lock_result.stderr


def test_nonfinite_is_scientific_terminal_but_other_failures_are_operational(
    tmp_path: Path,
) -> None:
    nonfinite_log = tmp_path / "nonfinite.log"
    nonfinite_log.write_text(
        "Traceback (most recent call last):\n"
        "labs.mnist_train.NonFiniteTrainingError: Expected parameter gradient "
        "at epoch=6, batch=17 to contain only finite values. Provided value: "
        "tensor with 3 non-finite element(s).\n",
        encoding="utf-8",
    )
    outcome = _scientific_outcome(
        state="failed",
        summary_status="failed",
        exact_returncode=1,
        log_path=nonfinite_log,
        epoch_rows=[{"validation_accuracy": 0.95}],
        expected_epochs=10,
    )
    assert outcome["scientific_terminal"] is True
    assert outcome["terminal_kind"] == "nonfinite_training"
    assert outcome["collapsed"] is True
    assert outcome["failure"]["epoch"] == 6

    operational_log = tmp_path / "operational.log"
    operational_log.write_text("ModuleNotFoundError: no module named torch\n")
    with pytest.raises(ValueError, match="operational"):
        _scientific_outcome(
            state="failed",
            summary_status="failed",
            exact_returncode=1,
            log_path=operational_log,
            epoch_rows=[],
            expected_epochs=10,
        )


def test_finite_collapse_threshold_is_five_percentage_points(tmp_path: Path) -> None:
    clean_log = tmp_path / "clean.log"
    clean_log.write_text("complete\n", encoding="utf-8")
    collapsed = _scientific_outcome(
        state="complete",
        summary_status="complete",
        exact_returncode=0,
        log_path=clean_log,
        epoch_rows=[
            {"validation_accuracy": 0.96},
            {"validation_accuracy": 0.90},
        ],
        expected_epochs=2,
    )
    stable = _scientific_outcome(
        state="complete",
        summary_status="complete",
        exact_returncode=0,
        log_path=clean_log,
        epoch_rows=[
            {"validation_accuracy": 0.96},
            {"validation_accuracy": 0.92},
        ],
        expected_epochs=2,
    )
    assert collapsed["collapsed"] is True
    assert collapsed["final_drop_from_best_pp"] == pytest.approx(6.0)
    assert stable["collapsed"] is False
    assert stable["final_drop_from_best_pp"] == pytest.approx(4.0)


def test_runtime_beta_recovery() -> None:
    base, injected = runtime_beta_values(
        {
            "training": {"beta": 0.001171875},
            "amplification": {"voltage_amp": 4.0, "current_amp": 0.25},
        },
        depth=2,
    )
    assert base == pytest.approx(0.001171875)
    assert injected == pytest.approx(0.3)


@pytest.mark.parametrize(
    ("boundary", "control", "expected"),
    [
        (
            {"collapsed": True, "final_validation_accuracy": None},
            {"collapsed": False, "final_validation_accuracy": 0.96},
            "confirmed_beta_collapse",
        ),
        (
            {"collapsed": True, "final_validation_accuracy": None},
            {"collapsed": True, "final_validation_accuracy": None},
            "inconclusive_shared_instability",
        ),
        (
            {"collapsed": False, "final_validation_accuracy": 0.80},
            {"collapsed": False, "final_validation_accuracy": 0.96},
            "degradation_without_internal_collapse",
        ),
        (
            {"collapsed": False, "final_validation_accuracy": 0.959},
            {"collapsed": False, "final_validation_accuracy": 0.96},
            "not_confirmed",
        ),
    ],
)
def test_pair_classification(
    boundary: dict[str, object], control: dict[str, object], expected: str
) -> None:
    assert classify_pair(boundary, control) == expected
