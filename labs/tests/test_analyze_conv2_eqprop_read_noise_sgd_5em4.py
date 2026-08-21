from __future__ import annotations

import numpy as np
import pytest

from experiments.analyze_conv2_eqprop_read_noise_sgd_5em4 import (
    SCHEMES,
    _scheme,
    make_plot,
    summarize,
)


def _case(final: float, best: float, *, optimizer: str) -> dict[str, object]:
    return {
        "optimizer": optimizer,
        "learning_rates": [1.0],
        "best_validation_accuracy": best,
        "final_validation_accuracy": final,
        "validation_accuracy": np.linspace(final - 0.09, final, 10),
        "run_dir": f"/{optimizer}/{final}",
    }


def test_scheme_is_resolved_from_arm_id() -> None:
    assert _scheme({"arm_id": "conv2_legacy_sgd_beta_one_decade_lower"}) == "legacy"


def test_summary_separates_noise_penalty_from_noisy_optimizer_gap() -> None:
    sgd_noise = {scheme: _case(0.94, 0.95, optimizer="SGD") for scheme in SCHEMES}
    sgd_clean = {scheme: _case(0.96, 0.965, optimizer="SGD") for scheme in SCHEMES}
    adam_noise = {scheme: _case(0.97, 0.975, optimizer="Adam") for scheme in SCHEMES}
    adam_clean = {scheme: _case(0.975, 0.98, optimizer="Adam") for scheme in SCHEMES}

    row = summarize(sgd_noise, sgd_clean, adam_noise, adam_clean)[0]

    assert row["sgd_final_change_pp_vs_clean"] == pytest.approx(-2.0)
    assert row["adam_final_change_pp_vs_clean"] == pytest.approx(-0.5)
    assert row["sgd_minus_adam_noise_final_pp"] == pytest.approx(-3.0)
    assert row["sgd_minus_adam_noise_penalty_pp"] == pytest.approx(-1.5)
    assert row["sgd_stable_through_ten_epochs"] is True


def test_plot_contains_all_four_contract_trajectories(tmp_path) -> None:
    groups = {
        name: {
            scheme: _case(0.95, 0.96, optimizer="SGD" if "sgd" in name else "Adam")
            for scheme in SCHEMES
        }
        for name in ("sgd_clean", "sgd_noise", "adam_clean", "adam_noise")
    }
    output = tmp_path / "plot.png"

    make_plot(groups, output)

    assert output.stat().st_size > 0
