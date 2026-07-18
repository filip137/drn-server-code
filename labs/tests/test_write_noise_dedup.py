from __future__ import annotations

import pytest

from experiments.evaluate_mnist_bp_write_noise_sweep import (
    _deduplicate_result_rows,
    _noise_cases,
)


def _row(**updates):
    row = {
        "run_name": "mnist_bp_amp_v1_c1",
        "voltage_amp": 1.0,
        "current_amp": 1.0,
        "training_seed": 0,
        "checkpoint_kind": "best",
        "checkpoint_path": "run/checkpoints/best.pt",
        "sigma": 0.1,
        "noise_seed": 7,
        "noisy_accuracy": 0.9,
    }
    row.update(updates)
    return row


def test_noise_cases_deduplicate_inputs_and_evaluate_sigma_zero_once():
    assert _noise_cases([0.0, 0.0, 0.1, 0.1], [7, 7, 8]) == [
        (0.0, 7),
        (0.1, 7),
        (0.1, 8),
    ]


def test_duplicate_write_noise_rows_are_counted_once():
    row = _row()
    assert _deduplicate_result_rows([row, dict(row)]) == [row]


def test_conflicting_duplicate_write_noise_rows_fail_fast():
    with pytest.raises(ValueError, match="conflicting key"):
        _deduplicate_result_rows([_row(), _row(noisy_accuracy=0.8)])
