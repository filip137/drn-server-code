from __future__ import annotations

import pytest

from experiments.summarize_conv2_eqprop_read_noise_terminal import (
    BETA_TIERS,
    EXPECTED_INDICES,
    SCHEMES,
    SIGMAS,
    TOTAL_STEPS,
    _case_identity,
    make_penalty_plot,
    summarize,
)


def test_user_truncated_coverage_stops_at_sigma_1em5():
    assert SIGMAS == (0.0, 1.0e-5, 1.0e-4)
    assert EXPECTED_INDICES == {
        0,
        2,
        3,
        4,
        6,
        7,
        8,
        10,
        11,
        12,
        14,
        15,
        16,
        18,
        19,
        20,
        22,
        23,
    }


def test_case_identity_includes_beta_tier():
    config = {
        "arm_id": "conv2_ours_adam_beta_two_decades_lower_sigma_1em4",
        "eqprop": {
            "beta_tier": "two_decades_lower",
            "endpoint_read_noise_std": 1.0e-4,
        },
    }

    assert _case_identity(config) == ("two_decades_lower", "ours", 1.0e-4)


def test_summary_anchors_noise_to_clean_within_each_beta_tier():
    cases = []
    for beta_tier in BETA_TIERS:
        clean_accuracy = 0.90 if beta_tier == "one_decade_lower" else 0.60
        for scheme in SCHEMES:
            for sigma in SIGMAS:
                accuracy = clean_accuracy if sigma == 0.0 else clean_accuracy - sigma * 100
                cases.append(
                    {
                        "index": len(cases),
                        "beta_tier": beta_tier,
                        "scheme": scheme,
                        "sigma": sigma,
                        "state": "complete",
                        "completed_epochs": 2,
                        "terminal_steps": TOTAL_STEPS,
                        "failure_epoch": None,
                        "failure_batch": None,
                        "nonfinite_element_count": None,
                        "injected_beta_B": 1.0,
                        "base_beta": 1.0,
                        "run_dir": "/tmp/run",
                        "log_path": "/tmp/log",
                        "epochs": [
                            {
                                "epoch": 1,
                                "validation_accuracy": accuracy - 0.01,
                            },
                            {"epoch": 2, "validation_accuracy": accuracy},
                        ],
                    }
                )

    rows = summarize(cases)
    by_identity = {
        (row["beta_tier"], row["scheme"], float(row["sigma"])): row
        for row in rows
    }
    assert by_identity[("one_decade_lower", "ours", 0.0)][
        "validation_change_pp_at_last_common_epoch"
    ] == pytest.approx(0.0)
    assert by_identity[("two_decades_lower", "ours", 1.0e-4)][
        "validation_change_pp_at_last_common_epoch"
    ] == pytest.approx(-1.0)


def test_penalty_plot_covers_retained_grid(tmp_path):
    rows = [
        {
            "beta_tier": beta_tier,
            "scheme": scheme,
            "sigma": f"{sigma:.12g}",
            "validation_change_pp_at_last_common_epoch": -sigma * 10_000,
        }
        for beta_tier in BETA_TIERS
        for scheme in SCHEMES
        for sigma in SIGMAS
    ]
    output = tmp_path / "penalty.png"

    make_penalty_plot(rows, output)

    assert output.stat().st_size > 0
