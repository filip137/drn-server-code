from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess

import pytest

from experiments.run_conv13_centered_eqprop_beta_qualification import (
    CASE_SPECS,
    DEFAULT_STUDY,
    PACKS,
    _resolved_case_config,
    load_and_validate_study,
    materialize_configs,
    ordered_config_set_sha256,
    plan,
)
from experiments.validate_conv13_eqprop_beta_pack import (
    runtime_beta_values,
    validate_pack,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
WRAPPER = (
    REPO_ROOT
    / "experiments/run_conv13_centered_eqprop_beta_qualification_jeanzay.slurm"
)
STUDY_RELATIVE = Path(
    "configs/conv/"
    "perfectdiode_conv13_centered_float64_eqprop_"
    "beta_qualification_1to2decades_10ep_seed0_20260814_v1.json"
)


def _read_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _base_environment(tmp_path: Path, *, mode: str, task: int) -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "C13BQ_SOURCE_ROOT": str(tmp_path / "missing-source"),
            "C13BQ_SOURCE_ARCHIVE": str(tmp_path / "missing-source.tar.gz"),
            "C13BQ_CONFIG_ROOT": str(tmp_path / "missing-configs"),
            "C13BQ_RESULT_ROOT": str(tmp_path / "results"),
            "C13BQ_DATASET_ROOT": str(tmp_path / "mnist"),
            "C13BQ_ENVIRONMENT_ID": "test-jean-zay-umg-v100-32g",
            "C13BQ_RUN_MODE": mode,
            "C13BQ_STUDY_CONFIG_SHA256": "0" * 64,
            "C13BQ_CONFIG_SET_SHA256": "0" * 64,
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
    tmp_path: Path, *, mode: str, task: int
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
    for index in range(12):
        path = config_root / f"{index:02d}.json"
        path.write_text(f'{{"index": {index}}}\n', encoding="utf-8")
        digests.append(hashlib.sha256(path.read_bytes()).hexdigest())
    config_set_sha = hashlib.sha256(
        "".join(f"{digest}\n" for digest in digests).encode("ascii")
    ).hexdigest()
    dataset_root = tmp_path / "mnist"
    dataset_root.mkdir()
    result_root = tmp_path / "results"
    env = _base_environment(tmp_path, mode=mode, task=task)
    env.update(
        {
            "C13BQ_SOURCE_ROOT": str(source_root),
            "C13BQ_SOURCE_ARCHIVE": str(archive),
            "C13BQ_CONFIG_ROOT": str(config_root),
            "C13BQ_RESULT_ROOT": str(result_root),
            "C13BQ_DATASET_ROOT": str(dataset_root),
            "C13BQ_STUDY_CONFIG_SHA256": hashlib.sha256(
                study_path.read_bytes()
            ).hexdigest(),
            "C13BQ_CONFIG_SET_SHA256": config_set_sha,
            "EXPERIMENT_SOURCE_ARCHIVE_SHA256": hashlib.sha256(
                archive.read_bytes()
            ).hexdigest(),
            "SLURM_ARRAY_TASK_COUNT": "1" if mode == "canary" else "6",
        }
    )
    return env, result_root


def test_frozen_matrix_and_two_runs_per_gpu_plan() -> None:
    study = load_and_validate_study(DEFAULT_STUDY)
    payload = plan(study)

    assert len(CASE_SPECS) == 12
    assert PACKS == ((0, 1), (2, 3), (4, 5), (6, 7), (8, 9), (10, 11))
    assert payload["array"] == "0-5%6"
    assert payload["packs"] == [list(pack) for pack in PACKS]
    observed = {
        (row["architecture"], row["scheme"], row["beta_tier"]): (
            row["injected_beta_B"],
            row["base_beta"],
        )
        for row in payload["cases"]
    }
    assert observed == {
        ("conv1", "baseline", "one_decade_lower"): (100.0, 100.0),
        ("conv1", "ours", "one_decade_lower"): (30.0, 7.5),
        ("conv1", "legacy", "one_decade_lower"): (3.0, 0.1875),
        ("conv3", "baseline", "one_decade_lower"): (100.0, 100.0),
        ("conv3", "ours", "one_decade_lower"): (3.0, 0.046875),
        ("conv3", "legacy", "one_decade_lower"): (0.001, 2.44140625e-7),
        ("conv1", "baseline", "two_decades_lower"): (10.0, 10.0),
        ("conv1", "ours", "two_decades_lower"): (3.0, 0.75),
        ("conv1", "legacy", "two_decades_lower"): (0.3, 0.01875),
        ("conv3", "baseline", "two_decades_lower"): (10.0, 10.0),
        ("conv3", "ours", "two_decades_lower"): (0.3, 0.0046875),
        ("conv3", "legacy", "two_decades_lower"): (1.0e-4, 2.44140625e-8),
    }


def test_materialized_configs_preserve_exact_lrs_and_science(tmp_path: Path) -> None:
    study = load_and_validate_study(DEFAULT_STUDY)
    paths = materialize_configs(study, tmp_path / "resolved")

    assert len(paths) == 12
    assert len(ordered_config_set_sha256(paths)) == 64
    for spec, path in zip(CASE_SPECS, paths, strict=True):
        resolved = _read_json(path)
        source_row = study["source"][spec.architecture]["scheme_configs"][spec.scheme]
        source = _read_json(REPO_ROOT / source_row["path"])
        assert resolved["learning_rates_by_parameter"] == source[
            "learning_rates_by_parameter"
        ]
        assert resolved["lr"] == source["lr"]
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
        "#SBATCH --time=08:00:00",
        "#SBATCH --array=0-5%6",
    }
    assert expected_directives <= set(text.splitlines())
    assert 'PACKS=("0 1" "2 3" "4 5" "6 7" "8 9" "10 11")' in text
    assert "PACK_INDEX=5" in text
    assert "EXPECTED_ARRAY_TASKS=1" in text
    assert "EXPECTED_ARRAY_TASKS=6" in text
    assert 'child_pid="$!"' in text
    assert 'pids+=("${child_pid}")' in text
    assert 'wait "${pids[offset]}"' in text
    assert "experiments.validate_conv13_eqprop_beta_pack" in text
    assert "C13BQ_SEMANTIC_PASS" in text

    heredocs = re.findall(r"<<'PY'\n(.*?)\nPY", text, flags=re.DOTALL)
    assert len(heredocs) == 1
    ast.parse(heredocs[0])


def test_canary_accepts_missing_count_and_maps_task_zero_to_pack_five(
    tmp_path: Path,
) -> None:
    env = _base_environment(tmp_path, mode="canary", task=0)
    env.pop("SLURM_ARRAY_TASK_COUNT", None)
    result = _run_wrapper(env)

    assert result.returncode == 3
    assert "+ EXPECTED_ARRAY_TASKS=1" in result.stderr
    assert "+ PACK_INDEX=5" in result.stderr
    assert "Frozen source directory or archive is missing." in result.stderr
    assert "array task(s)" not in result.stderr


def test_array_count_and_task_range_fail_closed(tmp_path: Path) -> None:
    wrong_count = _base_environment(tmp_path, mode="canary", task=0)
    wrong_count["SLURM_ARRAY_TASK_COUNT"] = "6"
    count_result = _run_wrapper(wrong_count)
    assert count_result.returncode == 3
    assert "Expected 1 array task(s); got 6." in count_result.stderr

    out_of_range = _base_environment(tmp_path, mode="production", task=6)
    out_of_range["SLURM_ARRAY_TASK_COUNT"] = "6"
    range_result = _run_wrapper(out_of_range)
    assert range_result.returncode == 3
    assert "Array task 6 is outside [0,5]." in range_result.stderr


def test_terminal_path_and_lock_prevent_duplicate_pack(tmp_path: Path) -> None:
    env, result_root = _stage_minimal_guard_inputs(
        tmp_path, mode="production", task=4
    )
    summary = result_root / "task_summaries/production/packs/4.json"
    summary.parent.mkdir(parents=True)
    summary.write_text("[]\n", encoding="utf-8")
    terminal_result = _run_wrapper(env)
    assert terminal_result.returncode == 17
    assert "Refusing duplicate production task 4" in terminal_result.stderr

    summary.unlink()
    lock = result_root / "task_locks/production/4.lock"
    lock.mkdir(parents=True)
    lock_result = _run_wrapper(env)
    assert lock_result.returncode == 17
    assert "Refusing concurrent duplicate production task 4" in lock_result.stderr


def test_pack_validator_rejects_wrong_logical_pair(tmp_path: Path) -> None:
    args = argparse.Namespace(
        study=DEFAULT_STUDY,
        pack_index=0,
        logical_indices=[0, 3],
        run_mode="canary",
        config_root=tmp_path / "configs",
        run_root=tmp_path / "runs",
        summary_root=tmp_path / "summaries",
        receipt_root=tmp_path / "receipts",
        pack_receipt=tmp_path / "pack.json",
        semantic_summary=tmp_path / "semantic.json",
        job_id="1",
        parent_job_id="1",
        environment_id="test",
        source_commit="0" * 40,
        source_archive_sha256="0" * 64,
        config_set_sha256="0" * 64,
    )
    with pytest.raises(ValueError, match="must contain"):
        validate_pack(args)


@pytest.mark.parametrize(
    ("depth", "voltage_amp", "current_amp", "base_beta", "injected_beta"),
    [
        (1, 4.0, 0.25, 0.01875, 0.3),
        (3, 4.0, 0.25, 2.44140625e-8, 1.0e-4),
    ],
)
def test_pack_validator_recovers_beta_from_trainer_runtime_config(
    depth: int,
    voltage_amp: float,
    current_amp: float,
    base_beta: float,
    injected_beta: float,
) -> None:
    runtime_config = {
        "training": {"beta": base_beta},
        "amplification": {
            "voltage_amp": voltage_amp,
            "current_amp": current_amp,
        },
    }

    observed_base, observed_injected = runtime_beta_values(
        runtime_config, depth=depth
    )

    assert observed_base == pytest.approx(base_beta)
    assert observed_injected == pytest.approx(injected_beta)
