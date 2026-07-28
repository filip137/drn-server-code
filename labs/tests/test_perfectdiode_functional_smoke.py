from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from experiments.mnist_conv import perfectdiode_functional_smoke as smoke


@dataclass
class _Parameter:
    name: str
    state: torch.Tensor
    min_cond: float = 0.0
    max_cond: float = 100.0


class _Runtime:
    def __init__(self) -> None:
        self.parameters = [
            _Parameter("ConvWeight_0", torch.tensor([1.0, 2.0])),
            _Parameter("Bias_0", torch.tensor([0.5])),
        ]
        self.optimizer = torch.optim.SGD(
            [parameter.state.requires_grad_() for parameter in self.parameters],
            lr=1.0,
        )

    def save(self, path: Path) -> None:
        path.write_bytes(b"disposable-checkpoint")


class _Bundle:
    train_loader = [("images", "labels", (7, 8))]

    def __init__(self) -> None:
        self.reset_count = 0

    def reset_train_shuffle(self) -> None:
        self.reset_count += 1

    def train_batch_indices(self, *, num_epochs: int):
        assert num_epochs == 1
        return (((7, 8),),)


def _item(name: str, bounded: bool) -> SimpleNamespace:
    return SimpleNamespace(
        name=name,
        bounded_gate=bounded,
        gradient_rms=0.2,
        proposed_update_rms=0.1,
        normalized_update=0.05,
        projection_efficiency=1.0,
    )


def _step(runtime, batch, *, learning_rate, restore, zero_proposal_epsilon):
    assert batch == ("images", "labels", (7, 8))
    assert learning_rate == {"ConvWeight_0": 0.2, "Bias_0": 0.1}
    assert restore is False
    assert zero_proposal_epsilon == 1.0e-12
    with torch.no_grad():
        runtime.parameters[0].state.add_(0.1)
    return SimpleNamespace(
        loss=1.25,
        accuracy=0.5,
        sample_count=2,
        source_indices=(7, 8),
        transition=SimpleNamespace(
            parameters=(
                _item("ConvWeight_0", True),
                _item("Bias_0", False),
            )
        ),
    )


def _save(runtime, path: Path) -> Path:
    runtime.save(path)
    return path


def test_execute_one_batch_runs_exactly_one_step_and_checkpoint(tmp_path) -> None:
    runtime = _Runtime()
    bundle = _Bundle()
    result = smoke.execute_one_batch(
        runtime,
        bundle,
        learning_rates_by_parameter={
            "ConvWeight_0": 0.2,
            "Bias_0": 0.1,
        },
        checkpoint_path=tmp_path / "disposable_checkpoint.pt",
        step_function=_step,
        save_checkpoint=_save,
    )

    assert bundle.reset_count == 1
    assert result["kind"] == "perfectdiode-conv-one-batch/v1"
    assert result["status"] == "passed"
    assert result["completed_steps"] == 1
    assert result["sample_count"] == 2
    assert result["source_indices"] == [7, 8]
    assert result["parameter_tensor_sha256_before"] != (
        result["parameter_tensor_sha256_after"]
    )
    assert result["checkpoint"] == {
        "path": "disposable_checkpoint.pt",
        "sha256": hashlib.sha256(b"disposable-checkpoint").hexdigest(),
        "bytes": len(b"disposable-checkpoint"),
        "disposable": True,
    }


def test_execute_one_batch_rejects_incomplete_rate_vector(tmp_path) -> None:
    with pytest.raises(ValueError, match="cover every runtime parameter exactly"):
        smoke.execute_one_batch(
            _Runtime(),
            _Bundle(),
            learning_rates_by_parameter={"ConvWeight_0": 0.2},
            checkpoint_path=tmp_path / "checkpoint.pt",
            step_function=_step,
            save_checkpoint=_save,
        )


def test_execute_one_batch_rejects_wrong_minibatch(tmp_path) -> None:
    def wrong_step(*args, **kwargs):
        value = _step(*args, **kwargs)
        value.source_indices = (99, 100)
        return value

    with pytest.raises(smoke.FunctionalSmokeError, match="first deterministic"):
        smoke.execute_one_batch(
            _Runtime(),
            _Bundle(),
            learning_rates_by_parameter={
                "ConvWeight_0": 0.2,
                "Bias_0": 0.1,
            },
            checkpoint_path=tmp_path / "checkpoint.pt",
            step_function=wrong_step,
            save_checkpoint=_save,
        )


def test_functional_smoke_rejects_controller_files_in_output(tmp_path) -> None:
    output = tmp_path / "smoke"
    output.mkdir()
    (output / "worker.log").write_text("controller-owned")

    with pytest.raises(
        FileExistsError,
        match="fresh attempt-specific functional-smoke output directory",
    ):
        smoke.run_perfectdiode_one_batch_smoke(
            {},
            tmp_path / "bundle",
            {"entry_id": "job-a"},
            attempt_id="attempt-a",
            target="local",
            data_root=tmp_path / "data",
            device="cuda",
            output_dir=output,
        )


def test_compact_functional_preflight_schema() -> None:
    value = {
        "schema_version": "experiment-functional-preflight/v1",
        "status": "passed",
        "passed": True,
        "official_test_read": False,
        "smoke": {
            "kind": "perfectdiode-conv-one-batch/v1",
            "status": "passed",
            "completed_steps": 1,
            "loss": 1.0,
            "accuracy": 0.25,
            "checkpoint": {
                "disposable": True,
                "sha256": "a" * 64,
            },
        },
    }
    assert smoke.validate_functional_preflight(value) == value


def test_cli_selects_exact_entry() -> None:
    from experiments.run_mnist_conv_perfectdiode_functional_smoke import _entry

    manifest = {
        "entries": [
            {"entry_id": "job-a", "entry_index": 0},
            {"entry_id": "job-b", "entry_index": 1},
        ]
    }
    assert _entry(manifest, entry_id="job-b", entry_index=None)["entry_index"] == 1
    with pytest.raises(ValueError, match="0 matches"):
        _entry(manifest, entry_id="missing", entry_index=None)
