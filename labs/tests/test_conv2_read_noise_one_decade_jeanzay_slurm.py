from __future__ import annotations

import ast
import hashlib
import os
from pathlib import Path
import re
import subprocess


REPO_ROOT = Path(__file__).resolve().parents[2]
WRAPPER = (
    REPO_ROOT
    / "experiments/run_conv2_centered_eqprop_read_noise_one_decade_jeanzay.slurm"
)
STUDY_RELATIVE = Path(
    "configs/conv/"
    "perfectdiode_conv2_centered_float64_eqprop_read_noise_"
    "one_decade_sigma_5em4_sgd_10ep_seed0_20260816_v1.json"
)


def _base_environment(tmp_path: Path, *, mode: str, task: int) -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "C2RN_SOURCE_ROOT": str(tmp_path / "missing-source"),
            "C2RN_SOURCE_ARCHIVE": str(tmp_path / "missing-source.tar.gz"),
            "C2RN_CONFIG_ROOT": str(tmp_path / "missing-configs"),
            "C2RN_RESULT_ROOT": str(tmp_path / "results"),
            "C2RN_DATASET_ROOT": str(tmp_path / "mnist"),
            "C2RN_ENVIRONMENT_ID": "test-jean-zay-v100-16g",
            "C2RN_RUN_MODE": mode,
            "C2RN_STUDY_CONFIG_SHA256": "0" * 64,
            "C2RN_CONFIG_SET_SHA256": "0" * 64,
            "C2RN_EXPECTED_CONFIGS": "3",
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
    tmp_path: Path,
    *,
    mode: str,
    task: int,
) -> tuple[dict[str, str], Path]:
    source_root = tmp_path / "source"
    study_path = source_root / STUDY_RELATIVE
    study_path.parent.mkdir(parents=True)
    study_path.write_text("{}\n", encoding="utf-8")

    archive = tmp_path / "frozen-source.tar.gz"
    archive.write_bytes(b"frozen source test archive\n")
    config_root = tmp_path / "exact-configs"
    config_root.mkdir()
    config_digests: list[str] = []
    for index in range(3):
        path = config_root / f"{index:03d}.json"
        path.write_text(f'{{"index": {index}}}\n', encoding="utf-8")
        config_digests.append(hashlib.sha256(path.read_bytes()).hexdigest())
    config_set_sha = hashlib.sha256(
        "".join(f"{digest}\n" for digest in config_digests).encode()
    ).hexdigest()

    dataset_root = tmp_path / "mnist"
    dataset_root.mkdir()
    result_root = tmp_path / "results"
    env = _base_environment(tmp_path, mode=mode, task=task)
    env.update(
        {
            "C2RN_SOURCE_ROOT": str(source_root),
            "C2RN_SOURCE_ARCHIVE": str(archive),
            "C2RN_CONFIG_ROOT": str(config_root),
            "C2RN_RESULT_ROOT": str(result_root),
            "C2RN_DATASET_ROOT": str(dataset_root),
            "C2RN_STUDY_CONFIG_SHA256": hashlib.sha256(
                study_path.read_bytes()
            ).hexdigest(),
            "C2RN_CONFIG_SET_SHA256": config_set_sha,
            "EXPERIMENT_SOURCE_ARCHIVE_SHA256": hashlib.sha256(
                archive.read_bytes()
            ).hexdigest(),
            "SLURM_ARRAY_TASK_COUNT": "1" if mode == "canary" else "3",
        }
    )
    return env, result_root


def test_wrapper_has_exact_parallel_resource_contract_and_valid_syntax() -> None:
    result = subprocess.run(
        ["bash", "-n", str(WRAPPER)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr

    text = WRAPPER.read_text(encoding="utf-8")
    expected_directives = {
        "#SBATCH --account=fmu@v100",
        "#SBATCH --partition=gpu_p13",
        "#SBATCH --qos=qos_gpu-t3",
        "#SBATCH --constraint=v100-16g",
        "#SBATCH --gres=gpu:1",
        "#SBATCH --cpus-per-task=2",
        "#SBATCH --time=03:00:00",
        "#SBATCH --array=0-2%3",
    }
    assert expected_directives <= set(text.splitlines())
    assert 'MODULE_ID="pytorch-gpu/py3/2.5.0"' in text
    assert "EXPECTED_ARRAY_TASKS=1" in text
    assert 'EXPECTED_ARRAY_TASKS="${EXPECTED_CONFIGS}"' in text
    assert "EXPECTED_NOISE_DRAWS=6" in text
    assert "EXPECTED_NOISE_DRAWS=206280" in text
    assert "validate_run(run_dir)" in text
    assert "C2RN_SEMANTIC_PASS" in text

    heredocs = re.findall(r"<<'PY'\n(.*?)\nPY", text, flags=re.DOTALL)
    assert len(heredocs) == 2
    for program in heredocs:
        ast.parse(program)


def test_canary_accepts_missing_optional_slurm_count_and_maps_task_zero(
    tmp_path: Path,
) -> None:
    env = _base_environment(tmp_path, mode="canary", task=0)
    env.pop("SLURM_ARRAY_TASK_COUNT", None)

    result = _run_wrapper(env)

    assert result.returncode == 3
    assert "+ EXPECTED_ARRAY_TASKS=1" in result.stderr
    assert "+ TASK_ID=0" in result.stderr
    assert "Frozen source directory or archive is missing." in result.stderr
    assert "array task(s)" not in result.stderr


def test_array_count_and_task_range_fail_closed(tmp_path: Path) -> None:
    wrong_count = _base_environment(tmp_path, mode="canary", task=0)
    wrong_count["SLURM_ARRAY_TASK_COUNT"] = "3"
    count_result = _run_wrapper(wrong_count)

    assert count_result.returncode == 3
    assert "Expected 1 array task(s); got 3." in count_result.stderr

    out_of_range = _base_environment(tmp_path, mode="production", task=3)
    out_of_range["SLURM_ARRAY_TASK_COUNT"] = "3"
    range_result = _run_wrapper(out_of_range)

    assert range_result.returncode == 3
    assert "Array task 3 is outside [0,2]." in range_result.stderr


def test_existing_terminal_path_prevents_duplicate_task_resume(tmp_path: Path) -> None:
    env, result_root = _stage_minimal_guard_inputs(
        tmp_path,
        mode="production",
        task=2,
    )
    summary = result_root / "task_summaries/production/2.json"
    summary.parent.mkdir(parents=True)
    summary.write_text("[]\n", encoding="utf-8")

    result = _run_wrapper(env)

    assert result.returncode == 17
    assert "Refusing duplicate production task 2: terminal path exists." in result.stderr


def test_existing_lock_prevents_concurrent_duplicate_task(tmp_path: Path) -> None:
    env, result_root = _stage_minimal_guard_inputs(
        tmp_path,
        mode="canary",
        task=0,
    )
    lock = result_root / "task_locks/canary/0.lock"
    lock.mkdir(parents=True)

    result = _run_wrapper(env)

    assert result.returncode == 17
    assert "Refusing concurrent duplicate canary task 0." in result.stderr
