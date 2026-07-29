from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from labs import mnist_train
from experiments.exact_run import (
    build_train_command,
    load_exact_config,
    run,
    selected_configs,
)


def _config(*, optimizer: str = "SGD") -> dict:
    value = {
        "lab": {"epochs": 3},
        "lr": [0.1, 0.02],
        "optimizer": {
            "name": optimizer,
            "learning_rate": [0.1, 0.02],
            "lr_decay": 1.0,
            "momentum": 0.0,
            "weight_decay": 0.0,
        },
    }
    if optimizer == "Adam":
        value["optimizer"].update(
            {
                "betas": [0.9, 0.999],
                "eps": 1e-8,
                "amsgrad": False,
                "foreach": False,
                "fused": False,
                "maximize": False,
                "capturable": False,
                "differentiable": False,
            }
        )
    return value


def _write_config(path: Path, *, optimizer: str = "SGD") -> Path:
    path.write_text(json.dumps(_config(optimizer=optimizer)), encoding="utf-8")
    return path


def test_load_exact_config_requires_matching_explicit_rates(tmp_path: Path) -> None:
    path = _write_config(tmp_path / "run.json")
    assert load_exact_config(path)["lr"] == [0.1, 0.02]

    value = _config()
    value["optimizer"]["learning_rate"] = [0.3]
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="to equal config\\['lr'\\]"):
        load_exact_config(path)


def test_load_exact_config_rejects_unsupported_adam_mode(tmp_path: Path) -> None:
    value = _config(optimizer="Adam")
    value["optimizer"]["fused"] = True
    path = tmp_path / "adam.json"
    path.write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises(ValueError, match="\\['fused'\\].*false"):
        load_exact_config(path)


def test_slurm_array_environment_selects_one_config(tmp_path: Path) -> None:
    paths = [tmp_path / "a.json", tmp_path / "b.json"]
    selected = selected_configs(
        paths,
        environ={"SLURM_ARRAY_TASK_ID": "1"},
    )

    assert selected == [(1, paths[1].resolve())]


def test_smoke_command_caps_training_and_validation(tmp_path: Path) -> None:
    command = build_train_command(
        tmp_path / "config.json",
        tmp_path / "output",
        device="cuda",
        smoke=True,
    )

    assert command[0] == sys.executable
    assert command[-8:] == [
        "--device",
        "cuda",
        "--epochs",
        "1",
        "--max-batches",
        "1",
        "--max-test-batches",
        "1",
    ]


def test_dry_run_is_indexed_and_has_no_output_side_effect(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configs = [
        _write_config(tmp_path / "sgd.json"),
        _write_config(tmp_path / "adam.json", optimizer="Adam"),
    ]
    output_root = tmp_path / "results"
    monkeypatch.setattr(
        "experiments.exact_run._git_state",
        lambda: {"commit": "abc", "dirty": False},
    )

    result = run(
        configs,
        output_root=output_root,
        device="cuda",
        smoke=True,
        dry_run=True,
        environ={"SLURM_ARRAY_TASK_ID": "1"},
    )

    assert result["status"] == "planned"
    assert [item["index"] for item in result["runs"]] == [1]
    assert result["runs"][0]["learning_rates"] == [0.1, 0.02]
    assert not output_root.exists()


def test_run_writes_per_case_record_and_propagates_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _write_config(tmp_path / "sgd.json")
    output_root = tmp_path / "results"
    monkeypatch.setattr(
        "experiments.exact_run._git_state",
        lambda: {"commit": "abc", "dirty": False},
    )

    def failed(*args: object, **kwargs: object) -> object:
        return type("Completed", (), {"returncode": 7})()

    monkeypatch.setattr("experiments.exact_run.subprocess.run", failed)
    result = run([config], output_root=output_root)

    assert result["status"] == "failed"
    record_path = Path(result["runs"][0]["output_dir"]) / "exact_run.json"
    record = json.loads(record_path.read_text(encoding="utf-8"))
    assert record["returncode"] == 7
    assert record["status"] == "failed"


def test_existing_case_directory_is_not_reused(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _write_config(tmp_path / "sgd.json")
    monkeypatch.setattr(
        "experiments.exact_run._git_state",
        lambda: {"commit": "abc", "dirty": False},
    )
    planned = run([config], output_root=tmp_path / "results", dry_run=True)
    case_dir = Path(planned["runs"][0]["output_dir"])
    case_dir.mkdir(parents=True)
    (case_dir / "old.txt").write_text("old", encoding="utf-8")

    with pytest.raises(FileExistsError, match="new or empty"):
        run([config], output_root=tmp_path / "results")


def test_training_cli_forwards_bounded_validation_and_optimizer_rates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config()
    del config["lr"]
    config.update(
        {
            "lab": {
                "epochs": 3,
                "model_key": "mnist_bp_conv_amp",
                "dataset_key": "mnist",
            },
            "log_interval": 10,
            "training_algorithm": "BP",
            "seed": 0,
        }
    )
    captured = {}
    monkeypatch.setattr(mnist_train, "load_config", lambda path: config)

    def fake_train(**kwargs: object) -> dict:
        captured.update(kwargs)
        return {"summary": {}}

    monkeypatch.setattr(mnist_train, "train_mnist_conv", fake_train)
    config_path = tmp_path / "config.json"
    assert mnist_train.main(
        [
            "--config",
            str(config_path),
            "--output-dir",
            str(tmp_path / "out"),
            "--max-batches",
            "1",
            "--max-validation-batches",
            "1",
        ]
    ) == 0

    assert captured["config_path"] == config_path.resolve()
    assert captured["lr"] == [0.1, 0.02]
    assert captured["max_batches"] == 1
    assert captured["max_test_batches"] == 1
