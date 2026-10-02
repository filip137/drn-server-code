from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from labs import mnist_train
from experiments.exact_run import (
    _completion_errors,
    _git_state,
    _terminal_metrics,
    build_train_command,
    load_exact_config,
    run,
    selected_configs,
)
from experiments import reporting


def _config(*, optimizer: str = "SGD") -> dict:
    value = {
        "lab": {"epochs": 3, "dataset_key": "mnist"},
        "datasets": {"mnist": {"factory": "MnistTrainValidation"}},
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


def _training_outputs(output, epochs=3):
    metrics = dict(best_epoch=1, final_train_loss=.3, final_train_accuracy=.8,
                   best_train_accuracy=.8, final_test_loss=.2, final_test_accuracy=.9,
                   best_test_accuracy=.9,
                   dataset_provenance={"schema": "mnist-train-validation-split/v1"})
    (output / "metrics.json").write_text(json.dumps(metrics))
    for name in ("best_model.pt", "final_model.pt"):
        (output / name).write_bytes(b"checkpoint")
    (output / "metrics.jsonl").write_text("")
    for epoch in range(1, epochs + 1):
        reporting.append_metric(output / "metrics.jsonl", dict(
            kind="epoch", epoch=epoch, evaluation_split="validation",
            metrics=dict(train_loss=.3, train_accuracy=.8, validation_loss=.2, validation_accuracy=.9)))


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


def test_load_exact_config_binds_named_rates_to_runtime_order(tmp_path: Path) -> None:
    value = _config()
    value.update(
        {
            "parameter_order": ["ConvWeight_0", "DenseWeight_0"],
            "learning_rates_by_parameter": {
                "ConvWeight_0": 0.1,
                "DenseWeight_0": 0.02,
            },
        }
    )
    path = tmp_path / "ordered.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    assert load_exact_config(path)["lr"] == [0.1, 0.02]

    value["parameter_order"] = ["DenseWeight_0", "ConvWeight_0"]
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="derived from the named mapping"):
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
    assert command[-9:] == [
        "--device",
        "cuda",
        "--epochs",
        "1",
        "--max-batches",
        "1",
        "--max-test-batches",
        "1",
        "--skip-terminal-official-test",
    ]


def test_train_command_records_transport_dataset_root(tmp_path: Path) -> None:
    dataset_root = tmp_path / "mnist"
    command = build_train_command(
        tmp_path / "config.json",
        tmp_path / "output",
        device="cpu",
        dataset_root=dataset_root,
        smoke=True,
    )

    assert command[command.index("--dataset-root") + 1] == str(
        dataset_root.resolve()
    )


def test_train_command_records_limited_gradient_diagnostics(tmp_path: Path) -> None:
    command = build_train_command(
        tmp_path / "config.json",
        tmp_path / "output",
        device="cuda",
        epoch_override=10,
        gradient_trace_samples_per_epoch=5,
        checkpoint_every_epoch=True,
        skip_terminal_official_test=True,
    )

    assert command[command.index("--epochs") + 1] == "10"
    assert command[command.index("--gradient-trace-samples-per-epoch") + 1] == "5"
    assert "--checkpoint-every-epoch" in command
    assert "--skip-terminal-official-test" in command


def test_paper_completion_requires_one_full_official_test() -> None:
    config = {
        "evaluation": {
            "official_test": {
                "policy": "terminal_once",
            }
        }
    }
    valid = {
        "official_test_evaluations": 1,
        "official_test_examples": 10_000,
        "official_test_checkpoint": "best_validation",
        "official_test_accuracy": 0.9,
        "official_test_loss": 0.2,
    }
    assert _completion_errors(config, valid, smoke=False) == []
    assert _completion_errors(config, {}, smoke=True) == []
    assert "exactly one" in _completion_errors(config, {}, smoke=False)[0]


def test_git_state_uses_frozen_archive_identity_when_git_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("EXPERIMENT_SOURCE_COMMIT", "abc123")
    monkeypatch.setenv("EXPERIMENT_SOURCE_ARCHIVE_SHA256", "deadbeef")

    def unavailable(*args: object, **kwargs: object) -> object:
        raise FileNotFoundError("git")

    monkeypatch.setattr("experiments.exact_run.subprocess.run", unavailable)

    assert _git_state() == {
        "commit": "abc123",
        "dirty": None,
        "source_archive_sha256": "deadbeef",
    }


def test_terminal_metrics_separates_validation_from_official_test() -> None:
    metrics = {
        "best_epoch": 3,
        "final_train_loss": 0.3,
        "final_train_accuracy": 0.8,
        "best_train_accuracy": 0.85,
        "final_test_loss": 0.25,
        "final_test_accuracy": 0.86,
        "best_test_accuracy": 0.88,
        "official_test_loss": 0.27,
        "official_test_accuracy": 0.84,
        "official_test_checkpoint": "best_validation",
        "official_test_examples": 10_000,
        "official_test_evaluations": 1,
        "dataset_provenance": {
            "schema": "mnist-train-validation-split/v1",
        },
    }

    terminal = _terminal_metrics(metrics)

    assert terminal["validation"]["final_accuracy"] == 0.86
    assert terminal["test"] == {
        "loss": 0.27,
        "accuracy": 0.84,
        "checkpoint": "best_validation",
        "examples": 10_000,
        "evaluations": 1,
    }


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


def test_dry_run_records_diagnostic_prefix_without_mutating_source_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _write_config(tmp_path / "sgd.json")
    source_bytes = config.read_bytes()
    monkeypatch.setattr(
        "experiments.exact_run._git_state",
        lambda: {"commit": "abc", "dirty": False},
    )

    result = run(
        [config],
        output_root=tmp_path / "results",
        dry_run=True,
        epoch_override=2,
        gradient_trace_samples_per_epoch=5,
        checkpoint_every_epoch=True,
        skip_terminal_official_test=True,
    )

    record = result["runs"][0]
    assert record["configured_epochs"] == 3
    assert record["epochs"] == 2
    assert record["diagnostics"] == {
        "gradient_trace_samples_per_epoch": 5,
        "checkpoint_every_epoch": True,
        "skip_terminal_official_test": True,
    }
    assert config.read_bytes() == source_bytes
    assert not (tmp_path / "results").exists()


def test_diagnostic_prefix_cannot_exceed_frozen_config_budget(tmp_path: Path) -> None:
    config = _write_config(tmp_path / "sgd.json")
    with pytest.raises(ValueError, match="no larger than the configured budget"):
        run([config], output_root=tmp_path / "results", dry_run=True, epoch_override=4)


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
    reporting_status = json.loads(
        (record_path.parent / "status.json").read_text(encoding="utf-8")
    )
    assert reporting_status["state"] == "failed"
    assert reporting_status["error"]["returncode"] == 7
    assert not (record_path.parent / "result.json").exists()


def test_successful_run_writes_canonical_result_bundle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _write_config(tmp_path / "sgd.json")
    output_root = tmp_path / "results"
    monkeypatch.setattr(
        "experiments.exact_run._git_state",
        lambda: {"commit": "abc", "dirty": False},
    )

    def successful(command: list[str], **kwargs: object) -> object:
        output = Path(command[command.index("--output-dir") + 1])
        _training_outputs(output)
        return type("Completed", (), {"returncode": 0})()

    monkeypatch.setattr("experiments.exact_run.subprocess.run", successful)
    result = run(
        [config],
        output_root=output_root,
        study_id="paper-conv1",
        evidence_class="medium_affine_experiment",
    )

    run_dir = Path(result["runs"][0]["output_dir"])
    reporting_result = json.loads(
        (run_dir / "result.json").read_text(encoding="utf-8")
    )
    assert reporting_result["study_id"] == "paper-conv1"
    assert reporting_result["terminal_metrics"]["validation"][
        "final_accuracy"
    ] == 0.9
    assert (run_dir / "checkpoints" / "final_model.pt").is_symlink()
    assert reporting.validate_run(run_dir) == []


@pytest.mark.parametrize("defect", ["no_artifacts", "short_epochs", "missing_checkpoint", "empty_checkpoint",
                                  "invalid_metrics", "nonfinite_metrics", "wrong_split", "missing_measurements"])
def test_zero_exit_without_complete_training_evidence_fails(tmp_path, monkeypatch, defect):
    config = _write_config(tmp_path / "config.json")
    monkeypatch.setattr("experiments.exact_run._git_state", lambda: {"commit": "abc"})

    def trainer(command, **kwargs):
        output = Path(command[command.index("--output-dir") + 1])
        if defect != "no_artifacts":
            _training_outputs(output, epochs=2 if defect == "short_epochs" else 3)
            if defect == "missing_checkpoint":
                (output / "best_model.pt").unlink()
            elif defect == "empty_checkpoint":
                (output / "final_model.pt").write_bytes(b"")
            elif defect == "invalid_metrics":
                (output / "metrics.json").write_text("not json")
            elif defect == "nonfinite_metrics":
                (output / "metrics.jsonl").write_text('{"epoch":1,"loss":NaN}\n')
            elif defect in {"wrong_split", "missing_measurements"}:
                rows = list(reporting.iter_metrics(output / "metrics.jsonl"))
                for row in rows:
                    if defect == "wrong_split": row["evaluation_split"] = "test"
                    else: row["metrics"] = {}
                (output / "metrics.jsonl").write_text(''.join(json.dumps(row)+'\n' for row in rows))
        return type("Completed", (), {"returncode": 0})()

    monkeypatch.setattr("experiments.exact_run.subprocess.run", trainer)
    result = run([config], output_root=tmp_path / "output")
    assert result["status"] == "failed"
    directory = Path(result["runs"][0]["output_dir"])
    assert not (directory / "result.json").exists()
    assert json.loads((directory / "status.json").read_text())["state"] == "failed"


@pytest.mark.parametrize("options,epochs", [({"smoke": True}, 1), ({"epoch_override": 2}, 2),
                                          ({"skip_terminal_official_test": True}, 3)])
def test_smoke_and_diagnostic_overrides_still_require_training_evidence(tmp_path, monkeypatch, options, epochs):
    config = _write_config(tmp_path / "config.json")
    monkeypatch.setattr("experiments.exact_run._git_state", lambda: {"commit": "abc"})
    def trainer(command, **kwargs):
        _training_outputs(Path(command[command.index("--output-dir") + 1]), epochs=epochs)
        return type("Completed", (), {"returncode": 0})()
    monkeypatch.setattr("experiments.exact_run.subprocess.run", trainer)
    assert run([config], output_root=tmp_path / "good", **options)["status"] == "complete"
    monkeypatch.setattr("experiments.exact_run.subprocess.run",
                        lambda *a, **k: type("Completed", (), {"returncode": 0})())
    assert run([config], output_root=tmp_path / "bad", **options)["status"] == "failed"


@pytest.mark.parametrize("legacy_pause", [False, True])
def test_exact_pause_resume_and_final_coverage(tmp_path, monkeypatch, legacy_pause):
    config = _write_config(tmp_path / "config.json")
    monkeypatch.setattr("experiments.exact_run._git_state", lambda: {"commit": "abc"})
    calls = []
    def trainer(command, **kwargs):
        output = Path(command[command.index("--output-dir") + 1])
        assert json.loads((output / "status.json").read_text())["state"] == "running"
        calls.append(output)
        if len(calls) < 3:
            (output / "continuation.pt").write_bytes(f"checkpoint-{len(calls)}".encode())
            reporting.atomic_write_json(output / "continuation.json", dict(
                completed_epochs=len(calls), checkpoint_sha256=reporting.sha256_file(output / "continuation.pt")))
            return type("Completed", (), {"returncode": 75})()
        _training_outputs(output)
        return type("Completed", (), {"returncode": 0})()
    monkeypatch.setattr("experiments.exact_run.subprocess.run", trainer)
    for expected in ("paused", "paused", "complete"):
        result = run([config], output_root=tmp_path / "output", epoch_chunk_size=1)
        assert result["status"] == expected
        assert json.loads((calls[-1] / "status.json").read_text())["state"] == expected
        assert reporting.validate_run(calls[-1]) == []
        if legacy_pause and expected == "paused":
            path = calls[-1] / "status.json"
            status = json.loads(path.read_text())
            status["state"] = "running"
            status.pop("pause")
            reporting.atomic_write_json(path, status)


def test_exact_rejects_a_corrupt_pause_checkpoint(tmp_path, monkeypatch):
    config = _write_config(tmp_path / "config.json")
    monkeypatch.setattr("experiments.exact_run._git_state", lambda: {"commit": "abc"})
    def trainer(command, **kwargs):
        output = Path(command[command.index("--output-dir") + 1])
        (output / "continuation.pt").write_bytes(b"damaged")
        reporting.atomic_write_json(output / "continuation.json",
                                    {"completed_epochs": 1, "checkpoint_sha256": "wrong"})
        return type("Completed", (), {"returncode": 75})()
    monkeypatch.setattr("experiments.exact_run.subprocess.run", trainer)
    result = run([config], output_root=tmp_path / "output", epoch_chunk_size=1)
    assert result["status"] == "failed"
    path = Path(result["runs"][0]["output_dir"])
    assert not (path / "result.json").exists()
    assert json.loads((path / "status.json").read_text())["state"] == "failed"


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
            "datasets": {
                "mnist": {
                    "factory": "labs.datasets.MnistTrainValidationDataset",
                    "params": {"root": "/remote/mnist"},
                }
            },
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
            "--dataset-root",
            str(tmp_path / "mnist"),
        ]
    ) == 0

    assert captured["config_path"] == config_path.resolve()
    assert captured["lr"] == [0.1, 0.02]
    assert captured["max_batches"] == 1
    assert captured["max_test_batches"] == 1
    assert captured["dataset_root_override"] == str(tmp_path / "mnist")
