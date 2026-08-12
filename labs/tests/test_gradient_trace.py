from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import torch

from experiments.gradient_trace import GradientTraceRecorder, sampled_batch_positions


def _parameter(name: str, values: list[float], *, lower: float | None = None):
    return SimpleNamespace(
        name=name,
        state=torch.nn.Parameter(torch.tensor(values, dtype=torch.float32)),
        min_cond=lower,
        max_cond=1.0 if lower is not None else None,
        _non_negative=lower == 0.0,
    )


def _record_step(recorder, optimizer, parameters, gradients, *, batch: int) -> None:
    pre = {
        parameter.name.strip(): parameter.state.detach().clone()
        for parameter in parameters
    }
    optimizer.zero_grad()
    for parameter, gradient in zip(parameters, gradients):
        parameter.state.grad = gradient.detach().clone()
    optimizer.step()
    post = {
        parameter.name.strip(): parameter.state.detach().clone()
        for parameter in parameters
    }
    with torch.no_grad():
        for parameter in parameters:
            if parameter.min_cond is not None:
                parameter.state.clamp_(parameter.min_cond, parameter.max_cond)
    projected = {
        parameter.name.strip(): parameter.state.detach().clone()
        for parameter in parameters
    }
    recorder(
        {
            "epoch": 1,
            "batch": batch,
            "total_batches": 2,
            "loss": 0.25,
            "optimizer": optimizer,
            "parameters": tuple(parameters),
            "gradients": tuple(gradients),
            "pre_optimizer_states": pre,
            "post_optimizer_states": post,
            "post_projection_states": projected,
        }
    )


def test_sampled_batch_positions_include_epoch_endpoints() -> None:
    assert sampled_batch_positions(3438, 5) == (1, 860, 1719, 2579, 3438)
    assert sampled_batch_positions(2, 5) == (1, 2)
    assert sampled_batch_positions(10, 1) == (1,)


@pytest.mark.parametrize("optimizer_name", ["SGD", "Adam"])
def test_gradient_trace_records_biases_projection_and_optimizer_state(
    tmp_path, optimizer_name: str
) -> None:
    weight = _parameter("ConvWeight_0 ", [0.01, 0.4], lower=0.0)
    bias = _parameter("Bias_0", [0.2, 0.3], lower=0.0)
    parameters = (weight, bias)
    groups = [
        {"params": [weight.state], "lr": 0.1},
        {"params": [bias.state], "lr": 0.05},
    ]
    if optimizer_name == "SGD":
        optimizer = torch.optim.SGD(groups, momentum=0.0, weight_decay=0.0)
    else:
        optimizer = torch.optim.Adam(
            groups,
            betas=(0.9, 0.999),
            eps=1.0e-8,
            weight_decay=0.0,
            foreach=False,
            fused=False,
        )

    recorder = GradientTraceRecorder(
        output_path=tmp_path / "gradient_trace.jsonl",
        metadata_path=tmp_path / "gradient_trace_metadata.json",
        parameters=parameters,
        optimizer=optimizer,
        batch_indices_by_epoch=(((0, 1), (2, 3)),),
        effective_batches_per_epoch=2,
        samples_per_epoch=2,
        dataset_provenance={"train_indices_sha256": "abc"},
    )
    assert recorder.track_all_parameters is True

    _record_step(
        recorder,
        optimizer,
        parameters,
        (torch.tensor([2.0, -1.0]), torch.tensor([0.5, -0.25])),
        batch=1,
    )
    _record_step(
        recorder,
        optimizer,
        parameters,
        (torch.tensor([1.0, 0.5]), torch.tensor([-0.5, 0.25])),
        batch=2,
    )
    metadata = recorder.finalize()

    rows = [
        json.loads(line)
        for line in (tmp_path / "gradient_trace.jsonl").read_text().splitlines()
    ]
    assert metadata["status"] == "complete"
    assert metadata["recorded_steps"] == 2
    assert metadata["recorded_rows"] == 4
    assert {row["parameter_name"] for row in rows} == {"ConvWeight_0", "Bias_0"}
    assert all(row["source_indices_sha256"] for row in rows)
    assert all(row["parameter_initial_sha256"] for row in rows)
    assert rows[0]["projection_changed_fraction"] > 0.0
    if optimizer_name == "SGD":
        assert all(
            row["raw_optimizer_over_fresh_shadow_rms"] == pytest.approx(1.0)
            for row in rows
        )
        assert all(row["optimizer_step"] is None for row in rows)
    else:
        assert [row["optimizer_step"] for row in rows[::2]] == [1.0, 2.0]
        assert all(row["adam_exp_avg_rms"] is not None for row in rows)
        assert all(
            row["adam_bias_corrected_exp_avg_rms"] is not None for row in rows
        )
        assert all(row["gradient_momentum_cosine"] is not None for row in rows)
