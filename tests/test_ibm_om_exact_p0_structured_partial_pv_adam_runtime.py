from __future__ import annotations

from pathlib import Path

import pytest
import torch

from experiments.definitions import resolve_experiment_config
from experiments.mnist_relu_drn.ibm_om_baseline_selection import quad_stack
from experiments.mnist_relu_drn import (
    ibm_om_exact_p0_structured_partial_pv_adam_runtime as runtime,
)
from experiments.schema import RunMode


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT
    / "examples/mnist_relu_drn/ibm_om_exact_p0_structured_partial_pv_adam/partial_sweep.json"
)


def test_runtime_strict_protocol_and_exact_full_parity_gate() -> None:
    _definition, spec = resolve_experiment_config(CONFIG, RunMode.TRAIN)
    runtime._strict_protocol(spec.protocol)
    rows = []
    for epoch, expected in enumerate(runtime._FULL_VALIDATION, start=1):
        rows.append(
            {
                "epoch": epoch,
                "validation": {
                    "student_correct": expected["correct"],
                    "kl_teacher_student": expected["kl"],
                },
                "validation_prediction_sha256": expected["prediction"],
                "issued_commands_cumulative": expected["issued"],
                "effective_state_change_events_cumulative": expected["effective"],
                "train_generator_state_sha256": expected["train_state"],
            }
        )
    report = {"selected_epoch": 3, "epoch_reports": rows}
    runtime._assert_full_historical_epoch_parity(report)
    drifted = {**report, "epoch_reports": [*rows[:-1], {**rows[-1], "issued_commands_cumulative": 1}]}
    with pytest.raises(RuntimeError, match="prior evidence"):
        runtime._assert_full_historical_epoch_parity(drifted)


def test_runtime_uses_train_only_selector_and_existing_masked_arm_path() -> None:
    assert runtime.CLOSED_WRITER_ARM_ID == (
        "incremental_one_pulse_closed_loop_target_tracking"
    )
    assert callable(runtime._run_arm)
    assert callable(runtime._rank_partial_masks)
    source = runtime._rank_partial_masks.__doc__ or ""
    assert "train-only" in source


@pytest.mark.parametrize("layout", ("halves", "paired"))
def test_calibration_score_is_quad_mean_absolute_raw_x_gradient(layout: str) -> None:
    gradient = torch.tensor(
        [[-1.0, 2.0, -3.0, 4.0], [5.0, -6.0, 7.0, -8.0],
         [9.0, -10.0, 11.0, -12.0], [-13.0, 14.0, -15.0, 16.0]],
        dtype=torch.float32,
    )
    observed = runtime._quad_score_batch(gradient, layout=layout)
    expected = quad_stack(2.0 * gradient, layout=layout).abs().mean(dim=-1)
    torch.testing.assert_close(observed, expected.to(torch.float64))
