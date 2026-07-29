from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

from experiments.exact_run import load_exact_config


REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_ROOT = (
    REPO_ROOT
    / "configs/conv/paper_medium_affine_perfectdiode_wide_seed0_20260729_v1"
)


def test_study_manifest_freezes_exact_eighteen_run_matrix() -> None:
    manifest = json.loads(
        (CONFIG_ROOT / "study_manifest.json").read_text(encoding="utf-8")
    )

    assert manifest["weight_contract"] == [0.0, 100.0]
    assert manifest["seed"] == 0
    assert len(manifest["runs"]) == 18
    assert Counter(row["architecture"] for row in manifest["runs"]) == {
        "conv1": 6,
        "conv2": 6,
        "conv3": 6,
    }
    assert manifest["architectures"]["conv1"]["epochs"] == 10
    assert manifest["architectures"]["conv2"]["epochs"] == 30
    assert manifest["architectures"]["conv3"]["epochs"] == 30


def test_every_paper_config_is_exact_and_uses_the_frozen_dataset_contract() -> None:
    manifest = json.loads(
        (CONFIG_ROOT / "study_manifest.json").read_text(encoding="utf-8")
    )

    for row in manifest["runs"]:
        path = REPO_ROOT / row["config"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == row["config_sha256"]
        config = load_exact_config(path)
        params = config["datasets"]["mnist"]["params"]
        affine = params["affine_config"]
        model = config["model_base"]

        assert config["seed"] == 0
        assert config["training_algorithm"] == "BP"
        assert config["batch_state_policy"] == "reset_each_batch"
        assert config["optimizer"]["lr_decay"] == 1.0
        assert config["lr"] == [
            config["learning_rates_by_parameter"][name]
            for name in config["parameter_order"]
        ]
        assert params["batch_size"] == 16
        assert params["split_seed"] == 0
        assert params["shuffle_seed"] == 0
        assert affine == {
            "degrees": 25.0,
            "deterministic_by_original_index_and_split": True,
            "enabled": True,
            "fill": 0,
            "interpolation": "bilinear",
            "preset": "medium",
            "scale": [0.8, 1.2],
            "seed": 1729,
            "shear": 0.0,
            "translate": [0.2, 0.2],
        }
        assert (model["weight_min"], model["weight_max"]) == (0.0, 100.0)
        assert model["weight_init_mode"] == "kaiming_uniform"
        assert model["non_linearity"] == "perfect_diode"
        assert config["evaluation"]["checkpoint_selection"] == (
            "maximum_validation_accuracy"
        )
        assert config["evaluation"]["official_test"] == {
            "checkpoint": "best_validation",
            "expected_examples": 10_000,
            "policy": "terminal_once",
        }


def test_architecture_and_optimizer_surface_counts_are_complete() -> None:
    expected_epochs = {"conv1": 10, "conv2": 30, "conv3": 30}
    expected_iterations = {"conv1": 4, "conv2": 6, "conv3": 8}
    expected_strides = {"conv1": [2], "conv2": [2, 2], "conv3": [2, 2, 1]}
    expected_surfaces = {
        ("baseline", "SGD"),
        ("baseline", "Adam"),
        ("ours", "SGD"),
        ("ours", "Adam"),
        ("legacy", "SGD"),
        ("legacy", "Adam"),
    }

    for arch in ("conv1", "conv2", "conv3"):
        configs = [
            load_exact_config(path)
            for path in sorted((CONFIG_ROOT / arch).glob("*.json"))
        ]
        assert len(configs) == 6
        assert {
            (config["arm_id"].split("_")[1], config["optimizer"]["name"])
            for config in configs
        } == expected_surfaces
        for config in configs:
            assert config["lab"]["epochs"] == expected_epochs[arch]
            assert config["model_base"]["num_iterations_inference"] == (
                expected_iterations[arch]
            )
            assert config["model_base"]["num_iterations_training"] == (
                expected_iterations[arch]
            )
            assert [
                stage["stride"]
                for stage in config["model_overrides"]["mnist_bp_conv_amp"][
                    "conv_pipeline"
                ]
            ] == expected_strides[arch]


def test_slurm_wrapper_freezes_the_reviewed_resource_contract() -> None:
    wrapper = (
        REPO_ROOT / "experiments/run_paper_medium_affine_seed0_jeanzay.slurm"
    ).read_text(encoding="utf-8")

    for directive in (
        "#SBATCH --account=fmu@v100",
        "#SBATCH --partition=gpu_p13",
        "#SBATCH --qos=qos_gpu-t3",
        "#SBATCH --constraint=v100-16g",
        "#SBATCH --gres=gpu:1",
        "#SBATCH --cpus-per-task=10",
    ):
        assert directive in wrapper
    assert "#SBATCH --mem" not in wrapper
    assert "pytorch-gpu/py3/2.5.0" in wrapper
    assert "--summary-json" in wrapper
    assert "PAPER_MODE" in wrapper
    assert "--smoke" in wrapper
