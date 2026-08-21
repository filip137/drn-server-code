from __future__ import annotations

import pytest

from experiments.analyze_conv2_eqprop_read_noise_high_extension import (
    EXPECTED_CASES,
    SCHEMES,
    SIGMAS,
    _scheme,
    make_plot,
    summarize,
)


def test_expected_surface_is_scheme_major_six_case_grid() -> None:
    assert EXPECTED_CASES == [
        (scheme, sigma) for scheme in SCHEMES for sigma in SIGMAS
    ]


def test_scheme_is_resolved_from_exact_arm_id() -> None:
    assert _scheme({"arm_id": "conv2_legacy_adam_beta_one_decade_lower_sigma"}) == (
        "legacy"
    )


def test_summary_compares_high_noise_to_clean_and_1em4() -> None:
    cases = [
        {
            "task": 0,
            "scheme": "ours",
            "sigma": 5.0e-4,
            "best_validation_accuracy": 0.96,
            "final_validation_accuracy": 0.95,
        }
    ]
    anchors = {
        "ours": {
            "clean_best": 0.98,
            "clean_final": 0.97,
            "sigma_1e4_best": 0.975,
            "sigma_1e4_final": 0.965,
        }
    }

    row = summarize(cases, anchors)[0]

    assert row["best_change_pp_vs_clean"] == pytest.approx(-2.0)
    assert row["final_change_pp_vs_clean"] == pytest.approx(-2.0)
    assert row["best_change_pp_vs_sigma_1e4"] == pytest.approx(-1.5)
    assert row["final_change_pp_vs_sigma_1e4"] == pytest.approx(-1.5)
    assert row["stable_through_ten_epochs"] is True


def test_plot_covers_clean_parent_and_high_noise_points(tmp_path) -> None:
    rows = [
        {
            "scheme": scheme,
            "sigma": sigma,
            "final_validation_accuracy": 0.97 - sigma,
        }
        for scheme in SCHEMES
        for sigma in SIGMAS
    ]
    anchors = {
        scheme: {
            "clean_final": 0.98,
            "sigma_1e4_final": 0.975,
        }
        for scheme in SCHEMES
    }
    output = tmp_path / "plot.png"

    make_plot(rows, anchors, output)

    assert output.stat().st_size > 0
