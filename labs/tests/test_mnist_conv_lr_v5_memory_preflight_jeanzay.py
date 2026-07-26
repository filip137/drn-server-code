from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from experiments.mnist_conv.io import atomic_write_json
from experiments.submit_mnist_conv_lr_v5_memory_preflight_jeanzay import (
    _command,
    submit,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
WRAPPER = (
    REPO_ROOT
    / "experiments/run_mnist_conv_lr_v5_memory_preflight_jeanzay.slurm"
)


def _fixture(tmp_path: Path) -> argparse.Namespace:
    study = tmp_path / ("v5-study--lrstudy_" + "a" * 64)
    candidate_dir = study / "stages/candidates"
    candidate_dir.mkdir(parents=True)
    data_root = tmp_path / "data"
    data_root.mkdir()
    candidate_manifest = candidate_dir / "manifest.json"
    atomic_write_json(
        candidate_manifest,
        {
            "schema_version": "mnist-conv-lr-stage-manifest/v1",
            "study_id": "lrstudy_" + "a" * 64,
            "stage_name": "candidates",
            "entries": [
                {"entry_index": index, "payload": {"architecture": "conv1"}}
                for index in range(36)
            ],
        },
        canonical=True,
    )
    return argparse.Namespace(
        study=str(study),
        candidate_manifest=str(candidate_manifest),
        output=str(candidate_dir / "memory_preflight.json"),
        repo_root=str(REPO_ROOT),
        python="python",
        data_root=str(data_root),
        module="pytorch-gpu/py3/2.5.0",
        steps_per_runtime=3,
        submit=False,
    )


def test_preflight_submission_defaults_to_dry_run_with_exact_v100_contract(
    tmp_path: Path,
) -> None:
    args = _fixture(tmp_path)
    result = submit(args)
    command = result["command"]

    assert result["submitted"] is False
    assert command[:2] == ["sbatch", "--parsable"]
    assert "--account=fmu@v100" in command
    assert "--partition=gpu_p13" in command
    assert "--qos=qos_gpu-t3" in command
    assert "--constraint=v100-32g" in command
    assert "--gres=gpu:1" in command
    assert not any(item.startswith("--array=") for item in command)
    exports = next(item for item in command if item.startswith("--export="))
    assert "MNIST_CONV_LR_STAGE=candidates" in exports
    assert "MNIST_CONV_DEVICE=cuda" in exports
    assert "MNIST_CONV_LR_MEMORY_PREFLIGHT_STEPS=3" in exports
    assert "MNIST_CONV_SLURM_ACCOUNT=fmu@v100" in exports
    assert "MNIST_CONV_SLURM_CONSTRAINT=v100-32g" in exports
    assert command[-1].endswith(
        "experiments/run_mnist_conv_lr_v5_memory_preflight_jeanzay.slurm"
    )


def test_preflight_submission_rejects_output_outside_study(tmp_path: Path) -> None:
    args = _fixture(tmp_path)
    args.output = str(tmp_path / "outside.json")

    with pytest.raises(ValueError, match="contained by the v5 study"):
        _command(args)


def test_preflight_submission_rejects_non_36_entry_manifest(tmp_path: Path) -> None:
    args = _fixture(tmp_path)
    manifest = Path(args.candidate_manifest)
    value = {
        "schema_version": "mnist-conv-lr-stage-manifest/v1",
        "study_id": "lrstudy_" + "a" * 64,
        "stage_name": "candidates",
        "entries": [{"entry_index": 0}],
    }
    atomic_write_json(manifest, value, canonical=True)

    with pytest.raises(ValueError, match="exactly 36 entries"):
        _command(args)


def test_preflight_wrapper_checks_exact_gpu_and_train_only_contract() -> None:
    source = WRAPPER.read_text()

    assert "#SBATCH --account=fmu@v100" in source
    assert "#SBATCH --partition=gpu_p13" in source
    assert "#SBATCH --constraint=v100-32g" in source
    assert "#SBATCH --gres=gpu:1" in source
    assert "--query-gpu=name,memory.total" in source
    assert "gpu_name" in source and "*V100*" in source
    assert "gpu_memory_mib < 32000" in source
    assert "gpu_memory_mib > 33000" in source
    assert 'official_test != {"enabled": False, "read_allowed": False}' in source
    assert "build_loader_bundle(download=False)" in source
    assert "--data-root" in source
    assert "--device" in source
