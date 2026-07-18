from types import SimpleNamespace

import pytest
import torch

from labs.mnist_train import (
    NonFiniteTrainingError,
    _batch_reset_flag,
    _checkpoint_best_then_callback,
    _require_finite_tensor,
    _resolve_batch_state_policy,
    _resolve_dataset_config,
    _write_json,
)


def test_batch_state_policy_defaults_to_independent_batches_with_legacy_opt_in():
    assert _resolve_batch_state_policy({}) == "reset_each_batch"
    assert [_batch_reset_flag("reset_each_batch", index) for index in range(3)] == [True] * 3
    assert _resolve_batch_state_policy({"batch_state_policy": "carry_within_epoch"}) == (
        "carry_within_epoch"
    )
    assert _resolve_batch_state_policy(
        {"training": {"batch_state_policy": "carry_within_epoch"}}
    ) == "carry_within_epoch"
    assert [_batch_reset_flag("carry_within_epoch", index) for index in range(3)] == [
        True,
        False,
        False,
    ]


def test_derived_fashion_mnist_replaces_inherited_mnist_statistics():
    config = {
        "datasets": {
            "mnist": {
                "factory": "labs.datasets.MnistDataset",
                "params": {
                    "root": "~/datasets/mnist",
                    "normalize_mean": 0.1307,
                    "normalize_std": 0.3081,
                },
            }
        }
    }

    key, resolved = _resolve_dataset_config(config, "fmnist")

    assert key == "fmnist"
    assert resolved["params"]["normalize_mean"] == pytest.approx(0.2860)
    assert resolved["params"]["normalize_std"] == pytest.approx(0.3530)
    assert config["datasets"]["mnist"]["params"]["normalize_mean"] == 0.1307


def test_explicit_fashion_mnist_config_remains_authoritative():
    config = {
        "datasets": {
            "fmnist": {
                "factory": "custom.Factory",
                "params": {"normalize_mean": 0.5, "normalize_std": 0.25},
            }
        }
    }

    _, resolved = _resolve_dataset_config(config, "fashion-mnist")

    assert resolved["factory"] == "custom.Factory"
    assert resolved["params"] == {"normalize_mean": 0.5, "normalize_std": 0.25}


def test_nonfinite_guard_and_strict_json_fail_before_artifact_write(tmp_path):
    with pytest.raises(NonFiniteTrainingError, match="only finite"):
        _require_finite_tensor(torch.tensor([1.0, float("nan")]), "gradient")

    target = tmp_path / "metrics.json"
    with pytest.raises(ValueError, match="Out of range float values"):
        _write_json(target, {"loss": float("nan")})
    assert not target.exists()


def test_best_checkpoint_is_saved_before_early_stop_callback(tmp_path):
    events = []

    class Energy:
        _params = []

        def save(self, path):
            events.append(("save", path))
            torch.save(
                {
                    "format": "drn.function.parameters",
                    "version": 1,
                    "schema": [],
                    "states": [],
                },
                path,
            )

    def callback(epoch_info, history):
        events.append(("callback", epoch_info["is_best"], len(history["learning_rate"])))
        return True

    history = {"learning_rate": []}
    optimizer = SimpleNamespace(param_groups=[{"lr": 0.25}])
    best_accuracy, best_epoch, should_stop = _checkpoint_best_then_callback(
        energy_fn=Energy(),
        best_model_path=tmp_path / "best.pt",
        epoch=1,
        train_loss=0.7,
        train_accuracy=0.6,
        test_loss=0.5,
        test_accuracy=0.8,
        best_accuracy=float("-inf"),
        best_epoch=None,
        optimizer=optimizer,
        history=history,
        epoch_callback=callback,
    )

    assert events == [
        ("save", tmp_path / "best.pt"),
        ("callback", True, 1),
    ]
    assert (best_accuracy, best_epoch, should_stop) == (0.8, 1, True)
