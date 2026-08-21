from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path

from experiments.exact_run import load_exact_config
from experiments.prepare_conv12_bounded_kaiming_adam_lr_transfer import (
    CASES,
    INITIALIZER_SHA256,
    REPO_ROOT,
    STUDY_ID,
    prepare,
)


SOURCE_ASSETS = {
    architecture: REPO_ROOT
    / "results/perfectdiode-conv12-bounded-rho-baseline-ours-tk46-seed0-v1"
    / "assets/bounded_kaiming_uniform"
    / architecture
    / "final_model.pt"
    for architecture in ("conv1", "conv2")
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _scientific_parent(config: dict) -> dict:
    restored = deepcopy(config)
    transfer = restored.pop("initializer_transfer")
    restored.pop("study_id")
    restored.pop("arm_id")
    restored.pop("parameter_order")
    restored.pop("learning_rates_by_parameter")
    restored.pop("evaluation")
    restored.pop("reporting")
    restored["init_checkpoint_path"] = json.loads(
        (REPO_ROOT / transfer["source_config"]).read_text(encoding="utf-8")
    )["init_checkpoint_path"]
    restored["model_base"]["weight_init_mode"] = "bounded_uniform"
    return restored


def test_prepared_configs_are_exact_raw_lr_initializer_transfers(tmp_path: Path) -> None:
    manifest = prepare(tmp_path)

    assert manifest["run_count"] == 6
    assert manifest["canonical_lr_handoff"] is False
    assert manifest["execution"]["array"] == "0-5%6"
    assert [row["index"] for row in manifest["runs"]] == list(range(6))

    initial_paths = {"conv1": set(), "conv2": set()}
    for case, row in zip(CASES, manifest["runs"], strict=True):
        path = tmp_path / row["config"]
        config = load_exact_config(path)
        source = json.loads((REPO_ROOT / case.source_config).read_text(encoding="utf-8"))

        assert _scientific_parent(config) == source
        assert config["lr"] == list(case.rates)
        assert config["optimizer"]["name"] == "Adam"
        assert config["lab"]["epochs"] == 3
        assert config["model_base"]["weight_init_mode"] == "bounded_kaiming_uniform"
        assert config["model_base"]["weight_min"] == 1e-5
        assert config["model_base"]["weight_max"] == 1e-4
        assert config["evaluation"]["official_test"]["policy"] == "disabled"
        assert config["initializer_transfer"]["canonical_lr_handoff"] is False
        assert config["initializer_transfer"]["initializer_checkpoint_sha256"] == (
            INITIALIZER_SHA256[case.architecture]
        )
        assert all(
            config["learning_rates_by_parameter"][name] == 0.0
            for name in case.parameter_order
            if name.startswith("Bias_")
        )
        initial_paths[case.architecture].add(config["init_checkpoint_path"])

    assert all(len(paths) == 1 for paths in initial_paths.values())


def test_frozen_kaiming_assets_have_expected_digests() -> None:
    for architecture, path in SOURCE_ASSETS.items():
        assert _sha256(path) == INITIALIZER_SHA256[architecture]


def test_jean_zay_wrapper_is_six_way_exact_run_array() -> None:
    wrapper = (
        REPO_ROOT
        / "experiments/run_conv12_bounded_kaiming_adam_lr_transfer_jeanzay.slurm"
    ).read_text(encoding="utf-8")

    assert "#SBATCH --account=umg@v100" in wrapper
    assert "#SBATCH --qos=qos_gpu-dev" in wrapper
    assert "#SBATCH --array=0-5%6" in wrapper
    assert wrapper.count("_adam.json\"") == 6
    assert "python -m experiments.exact_run" in wrapper
    assert "--checkpoint-every-epoch" in wrapper
    assert "--skip-terminal-official-test" in wrapper
    assert "SLURM_ARRAY_TASK_ID:?" in wrapper
