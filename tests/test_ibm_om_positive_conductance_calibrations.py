from pathlib import Path

import pytest
import torch

from experiments.mnist_relu_drn.ibm_om_positive_conductance import (
    DESTINATION_PAIR_BASELINE,
    EMPIRICAL_QUANTILES,
    QUAD_BASELINE,
    load_positive_conductance_calibration,
)
from experiments.mnist_relu_drn.ibm_om_positive_conductance_diagnostics import (
    DIAGNOSTIC_SCHEMA,
    diagnose_positive_conductance_population,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
CALIBRATION_ROOT = (
    REPO_ROOT / "experiments/mnist_relu_drn/calibrations"
)
RANK_MATCHED_PATH = (
    CALIBRATION_ROOT
    / "iedm2022_fig6_baseline_hfo2_rank_matched_v1.json"
)
INDEPENDENT_PATH = (
    CALIBRATION_ROOT
    / "iedm2022_fig6_baseline_hfo2_independent_v1.json"
)
PDF_SHA256 = (
    "cd0f1e44f14b1b44a51447e477a217df7aa9f4e6fa3ea00805c3240bfd9e51d2"
)
CROP_SHA256 = (
    "0389c6c99cf28ac034c7ec8dede0230e86b83ab1a78092a1a121482ec691e07f"
)


@pytest.mark.parametrize(
    ("path", "pairing_policy"),
    (
        (RANK_MATCHED_PATH, "rank_matched"),
        (INDEPENDENT_PATH, "independent"),
    ),
)
def test_audited_figure6_calibration_loads_with_endpoint_extended_quantiles(
    path: Path,
    pairing_policy: str,
) -> None:
    calibration = load_positive_conductance_calibration(path)

    assert calibration.representation == EMPIRICAL_QUANTILES
    assert calibration.pairing_policy == pairing_policy
    assert calibration.source_sha256 == PDF_SHA256
    assert calibration.provenance["figure_crop_sha256"] == CROP_SHA256
    assert calibration.reset_probabilities is not None
    assert calibration.set_probabilities is not None
    assert calibration.reset_probabilities.numel() == 65
    assert calibration.set_probabilities.numel() == 65
    expected_probabilities = torch.arange(65, dtype=torch.float64) / 64.0
    assert torch.equal(calibration.reset_probabilities, expected_probabilities)
    assert torch.equal(calibration.set_probabilities, expected_probabilities)
    assert calibration.reset_values_s[0].item() == pytest.approx(
        4.997164643e-6
    )
    assert calibration.reset_values_s[-1].item() == pytest.approx(
        32.340669063e-6
    )
    assert calibration.set_values_s[0].item() == pytest.approx(
        153.912376363e-6
    )
    assert calibration.set_values_s[-1].item() == pytest.approx(
        320.995142453e-6
    )
    assert calibration.reset_values_s[0] == calibration.reset_values_s[1]
    assert calibration.reset_values_s[-1] == calibration.reset_values_s[-2]
    assert calibration.set_values_s[0] == calibration.set_values_s[1]
    assert calibration.set_values_s[-1] == calibration.set_values_s[-2]
    assert "Synthetic joint control" in calibration.provenance["notes"]
    assert "not raw measured device identities" in calibration.provenance["notes"]


def test_audited_controls_have_identical_marginals_and_only_change_copula() -> None:
    rank = load_positive_conductance_calibration(RANK_MATCHED_PATH)
    independent = load_positive_conductance_calibration(INDEPENDENT_PATH)

    assert torch.equal(rank.reset_probabilities, independent.reset_probabilities)
    assert torch.equal(rank.set_probabilities, independent.set_probabilities)
    assert torch.equal(rank.reset_values_s, independent.reset_values_s)
    assert torch.equal(rank.set_values_s, independent.set_values_s)

    rank_population = rank.sample_population(devices=128, assignment_seed=2022)
    rank_replay = rank.sample_population(devices=128, assignment_seed=2022)
    independent_population = independent.sample_population(
        devices=128, assignment_seed=2022
    )
    assert torch.equal(
        rank_population.reset_conductance_s,
        independent_population.reset_conductance_s,
    )
    assert not torch.equal(
        rank_population.set_conductance_s,
        independent_population.set_conductance_s,
    )
    assert rank_population.population_sha256 == rank_replay.population_sha256
    assert rank_population.report["identity_semantics"] == (
        "model_sample_not_raw_measured_device_identity"
    )
    assert independent_population.report["identity_semantics"] == (
        "model_sample_not_raw_measured_device_identity"
    )
    assert rank_population.report["joint_model_assumption"].endswith(
        "rank_matched"
    )
    assert independent_population.report["joint_model_assumption"].endswith(
        "independent"
    )


@pytest.mark.parametrize("path", (RANK_MATCHED_PATH, INDEPENDENT_PATH))
def test_real_calibration_read_only_diagnostic_reports_positive_windows_and_loads(
    path: Path,
) -> None:
    result = diagnose_positive_conductance_population(
        path,
        devices=64,
        assignment_seed=2022,
    )

    assert result["schema"] == DIAGNOSTIC_SCHEMA
    assert result["population"]["devices"] == 64
    assert result["population"]["assignment_seed"] == 2022
    assert result["population"]["identity_semantics"] == (
        "model_sample_not_raw_measured_device_identity"
    )
    endpoints = result["endpoints"]
    assert endpoints["reset"]["minimum"] > 0.0
    assert endpoints["set"]["minimum"] > endpoints["reset"]["maximum"]
    assert endpoints["absolute_RESET_to_SET_headroom"]["minimum"] > 0.0
    assert endpoints["dynamic_range_G_high_over_G_low"]["minimum"] > 1.0

    pair = result["baselines"][DESTINATION_PAIR_BASELINE]
    quad = result["baselines"][QUAD_BASELINE]
    assert pair["all_groups_feasible"] is True
    assert pair["group_count"] == 32
    assert pair["feasible_group_count"] == 32
    assert quad["all_groups_feasible"] is True
    assert quad["group_count"] == 16
    assert quad["feasible_group_count"] == 16
    assert pair["full_quad_zero_state_loading"]["minimum"] > 0.0
    assert quad["full_quad_zero_state_loading"]["minimum"] > 0.0
    assert (
        quad["full_quad_zero_state_loading"]["mean"]
        >= pair["full_quad_zero_state_loading"]["mean"]
    )
    assert pair["target_projection"] == "none"
    assert quad["post_handoff_clipping"] is False


def test_diagnostic_requires_complete_quads() -> None:
    with pytest.raises(ValueError, match="multiple of four"):
        diagnose_positive_conductance_population(
            RANK_MATCHED_PATH,
            devices=6,
            assignment_seed=1,
        )
