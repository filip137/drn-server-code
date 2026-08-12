from pathlib import Path

import pytest

from experiments import analyze_conv3_eqprop_high_beta_high_tk as analysis


def test_gradient_status_preserves_exact_zero_as_categorical() -> None:
    row = {
        "float32_eqprop_vs_bptt_cosine": "",
        "float32_eqprop_l2": "0",
        "float32_exact_zero_fraction": "1",
    }
    assert analysis._gradient_status(row, "float32") == "exact_zero"
    row["float32_eqprop_l2"] = "1e-9"
    assert analysis._gradient_status(row, "float32") == "undefined"


def test_scaling_context_does_not_require_batch_columns(tmp_path: Path) -> None:
    row = {
        "architecture": "conv3",
        "scheme": "ours",
        "checkpoint_role": "best_validation",
        "T": "64",
        "K": "64",
    }
    analysis._validate_context_fields([row], path=tmp_path / "beta_scaling.csv")


def test_cosine_range_is_fail_closed() -> None:
    row = {
        "float64_eqprop_vs_bptt_cosine": "1.01",
        "float64_eqprop_l2": "1",
        "float64_exact_zero_fraction": "0",
    }
    with pytest.raises(ValueError, match="Cosine outside"):
        analysis._gradient_status(row, "float64")
