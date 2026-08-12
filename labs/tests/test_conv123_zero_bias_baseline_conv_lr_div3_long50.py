from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess

import pytest
import torch

import experiments.validate_zero_bias_exact_summary as zero_bias_validator
from experiments.prepare_conv123_zero_bias_baseline_conv_lr_div3_long50 import (
    ARCHITECTURES,
    CONV_LR_DIVISOR,
    EPOCHS,
    EVIDENCE_CLASS,
    OPERATING_POINTS,
    OUTPUT_ROOT,
    PARENT_FILENAME,
    PARENT_ROOT,
    REPO_ROOT,
    STUDY_ID,
    prepare,
)
from experiments.validate_zero_bias_exact_summary import _verify_biases


WRAPPER = (
    REPO_ROOT
    / "experiments/"
    "run_conv123_zero_bias_baseline_conv_lr_div3_long50_jeanzay.slurm"
)
CONFIG_FILENAME = "00_baseline_sgd_conv_lr_div3_long50.json"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_materialized_files_match_generator() -> None:
    prepare(check=True)


def test_manifest_and_configs_encode_only_the_declared_interventions() -> None:
    manifest = _load(OUTPUT_ROOT / "study_manifest.json")
    assert manifest["study_id"] == STUDY_ID
    assert manifest["evidence_class"] == EVIDENCE_CLASS
    assert manifest["paper_facing"] is False
    assert manifest["axes"] == {
        "architectures": list(ARCHITECTURES),
        "optimizer": ["SGD"],
        "scheme": ["baseline"],
        "seed": [0],
    }
    assert manifest["epoch_budget"] == {
        architecture: EPOCHS for architecture in ARCHITECTURES
    }
    assert manifest["operating_points"] == OPERATING_POINTS
    assert manifest["diagnostics"]["expected_checkpoint_epochs"] == list(
        range(EPOCHS + 1)
    )
    assert manifest["diagnostics"]["expected_gradient_trace_transitions"] == (
        EPOCHS * 5
    )
    assert manifest["dataset"]["official_test_read"] is False
    assert len(manifest["configs"]) == 3

    digests: list[str] = []
    for index, architecture in enumerate(ARCHITECTURES):
        path = OUTPUT_ROOT / architecture / CONFIG_FILENAME
        parent_path = PARENT_ROOT / architecture / PARENT_FILENAME
        config = _load(path)
        parent = _load(parent_path)
        row = manifest["configs"][index]
        digests.append(_sha256(path))

        assert row["global_index"] == index
        assert row["architecture"] == architecture
        assert row["config_sha256"] == digests[-1]
        assert row["parent_config_sha256"] == _sha256(parent_path)
        assert config["study_id"] == STUDY_ID
        assert config["reporting"] == {
            "arm_id": config["arm_id"],
            "evidence_class": EVIDENCE_CLASS,
            "paper_facing": False,
            "study_id": STUDY_ID,
        }
        assert config["lab"]["epochs"] == EPOCHS
        assert config["seed"] == parent["seed"] == 0
        assert config["datasets"] == parent["datasets"]
        assert config["model_base"] == parent["model_base"]
        assert config["model_overrides"] == parent["model_overrides"]
        assert config["batch_state_policy"] == parent["batch_state_policy"]
        assert config["beta"] == parent["beta"]
        assert config["training_algorithm"] == parent["training_algorithm"]
        assert config["evaluation"]["official_test"]["policy"] == "disabled"

        point = OPERATING_POINTS[architecture]
        assert config["model_base"]["input_gain"] == point["input_gain"]
        assert config["model_base"]["num_iterations_inference"] == point["T"]
        assert config["model_base"]["num_iterations_training"] == point["K"]
        assert config["model_base"]["non_linearity"] == "perfect_diode"
        assert config["model_base"]["weight_min"] == 0.0
        assert config["model_base"]["weight_max"] == 100.0
        for key in (
            "exponential_diode_param",
            "hard_sigmoid_param",
            "quadratic_diode_param",
        ):
            assert isinstance(config["model_base"][key], dict)
            assert config["model_base"][key]

        order = config["parameter_order"]
        rates = config["learning_rates_by_parameter"]
        parent_rates = parent["learning_rates_by_parameter"]
        assert config["lr"] == [rates[name] for name in order]
        assert config["optimizer"]["learning_rate"] == config["lr"]
        assert config["optimizer"]["name"] == "SGD"
        assert config["optimizer"]["momentum"] == 0.0
        assert config["optimizer"]["weight_decay"] == 0.0
        for name in order:
            if name.startswith("ConvWeight_"):
                assert rates[name] == pytest.approx(
                    parent_rates[name] / CONV_LR_DIVISOR,
                    rel=0,
                    abs=0,
                )
            else:
                assert rates[name] == parent_rates[name]
            if name.startswith("Bias_"):
                assert rates[name] == 0.0

        intervention = config["handoff"]["lr_duration_counterfactual"]
        assert intervention["parent_config_sha256"] == _sha256(parent_path)
        assert intervention["duration_intervention"] == {
            "parent_epochs": 30,
            "target_epochs": EPOCHS,
        }
        assert (
            intervention["rate_intervention"]["conv_learning_rate_divisor"]
            == CONV_LR_DIVISOR
        )
        assert "lr_counterfactual" not in config["handoff"]

    digest_payload = "".join(f"{digest}\n" for digest in digests).encode()
    assert manifest["ordered_config_set_sha256"] == hashlib.sha256(
        digest_payload
    ).hexdigest()


def test_wrapper_is_the_exact_three_task_jean_zay_contract() -> None:
    subprocess.run(["bash", "-n", str(WRAPPER)], check=True)
    text = WRAPPER.read_text(encoding="utf-8")
    assert "#SBATCH --job-name=pd-c123-zb-div3-e50" in text
    assert "#SBATCH --account=fmu@v100" in text
    assert "#SBATCH --partition=gpu_p13" in text
    assert "#SBATCH --qos=qos_gpu-t3" in text
    assert "#SBATCH --constraint=v100-16g" in text
    assert "#SBATCH --array=0-2%3" in text
    assert "#SBATCH --time=10:00:00" in text
    assert "pytorch-gpu/py3/2.5.0" in text
    assert "python -m experiments.exact_run" in text
    assert "--gradient-trace-samples-per-epoch 5" in text
    assert "--checkpoint-every-epoch" in text
    assert "--skip-terminal-official-test" in text
    assert "python -m experiments.validate_zero_bias_exact_summary" in text
    assert text.count(CONFIG_FILENAME) == 3


def test_zero_bias_checkpoint_validator_is_fail_closed(tmp_path: Path) -> None:
    zero = tmp_path / "zero.pt"
    nonzero = tmp_path / "nonzero.pt"
    schema = [{"name": "Weight_0"}, {"name": "Bias_0"}]
    torch.save(
        {
            "schema": schema,
            "states": [torch.ones(2), torch.zeros(2)],
        },
        zero,
    )
    torch.save(
        {
            "schema": schema,
            "states": [torch.ones(2), torch.tensor([0.0, 1.0])],
        },
        nonzero,
    )
    _verify_biases(zero, ["Bias_0"])
    with pytest.raises(RuntimeError, match="Nonzero frozen bias"):
        _verify_biases(nonzero, ["Bias_0"])


def test_summary_validator_matches_digest_keys_not_path_values(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config_paths = [tmp_path / "a.json", tmp_path / "b.json"]
    for index, path in enumerate(config_paths):
        path.write_text(json.dumps({"index": index}), encoding="utf-8")
    config_digests = [_sha256(path) for path in config_paths]
    manifest_path = tmp_path / "study_manifest.json"
    manifest_path.write_text(
        json.dumps({"ordered_config_set_sha256": "set-digest"}),
        encoding="utf-8",
    )
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(
        json.dumps(
            [
                {"config_sha256": digest, "status": "complete"}
                for digest in config_digests
            ]
        ),
        encoding="utf-8",
    )
    expected = [tmp_path / "run-a", tmp_path / "run-b"]
    observed: list[str] = []

    def fake_validate_one(**kwargs: object) -> Path:
        row = kwargs["row"]
        assert isinstance(row, dict)
        observed.append(str(row["config_sha256"]))
        return expected[len(observed) - 1]

    monkeypatch.setattr(
        zero_bias_validator, "_validate_one", fake_validate_one
    )
    assert zero_bias_validator.validate_summary(
        summary_path=summary_path,
        study_manifest_path=manifest_path,
        config_paths=config_paths,
        config_set_sha256="set-digest",
        run_mode="smoke",
    ) == expected
    assert observed == config_digests
