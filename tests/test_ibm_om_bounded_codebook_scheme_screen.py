from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from experiments.mnist_relu_drn.ibm_om_bounded_codebook_scheme_screen import (
    build_baseline_state,
    build_bounded_scheme_targets,
    build_deterministic_set_codebook,
    load_initial_reset_calibration_receipt,
    load_screen_contract,
    project_to_nearest_code,
)
from experiments.mnist_relu_drn.ibm_om_ideal_mapping_scheme_screen import Scheme
from training.ibm_reram_hwa import IbmReramArrayPopulation


_ROOT = Path(__file__).resolve().parents[1]
_SCREEN = (
    _ROOT
    / "examples"
    / "mnist_relu_drn"
    / "ibm_om_bounded_codebook_scheme_screen"
    / "screen.json"
)
_INITIAL_RESET_RECEIPT = _ROOT / "data" / "ibm_om_cell_aware_full_span_v1.receipt.json"
_TEACHER = _ROOT / "data" / "mnist_relu_teacher_fixed_init_20260816.pt"


def _population(*, devices: int, reference: float = 0.0) -> IbmReramArrayPopulation:
    if devices == 4:
        keys = ("base.dense_weight.0", "base.dense_weight.1")
    elif devices == 8:
        keys = (
            "base.conductance_plus.0",
            "base.conductance_minus.0",
            "base.conductance_plus.1",
            "base.conductance_minus.1",
        )
    else:  # pragma: no cover - helper guard
        raise ValueError
    shapes = tuple((2, 2) for _ in keys)
    size = 4 * len(keys)
    return IbmReramArrayPopulation(
        assignment_seed=1,
        corruption_policy="counterfactual_repaired",
        binding_keys=keys,
        binding_shapes=shapes,
        binding_sampling_seeds=tuple(range(10, 10 + len(keys))),
        donor_sampling_seeds=tuple(range(20, 20 + len(keys))),
        nominal_dw_min=0.5,
        dw_min_std=0.4,
        write_noise_std=1.0,
        max_bound=torch.ones(size, dtype=torch.float32),
        min_bound=-torch.ones(size, dtype=torch.float32),
        dwmin_up=torch.full((size,), 0.5, dtype=torch.float32),
        dwmin_down=torch.full((size,), 0.5, dtype=torch.float32),
        reference=torch.full((size,), reference, dtype=torch.float32),
        corrupt=torch.zeros(size, dtype=torch.bool),
        published_corrupt=torch.zeros(size, dtype=torch.bool),
        fingerprint=f"test-{devices}-{reference}",
        aihwkit_version="1.1.0",
    )


def test_tracked_contract_pins_no_noise_device_specific_codebook() -> None:
    contract = load_screen_contract(_SCREEN)
    assert contract.maximum_pulses == 128
    assert contract.development_assignment_seed == 86001
    assert contract.heldout_assignment_seeds == (87001, 87002, 87003)
    assert contract.corruption_policy == "counterfactual_repaired"
    assert contract.shared_scale_fractions == (1.0, 1.0)
    assert contract.shared_fixed_logit_gain == pytest.approx(4.46683592150963)
    assert contract.raw["deterministic_codebook"]["cycle_to_cycle_random_term"] == 0.0
    assert contract.raw["deterministic_codebook"]["apparent_write_noise"] == 0.0


def test_shared_calibration_is_validated_against_initial_reset_receipt() -> None:
    contract = load_screen_contract(_SCREEN)
    report = load_initial_reset_calibration_receipt(
        _INITIAL_RESET_RECEIPT,
        contract=contract,
        teacher_weights_path=_TEACHER,
    )
    assert report["target_scale_fractions"] == [1.0, 1.0]
    assert report["fixed_logit_gain"] == pytest.approx(4.46683592150963)
    assert report["optimizer_updates"] == 0
    assert report["per_scheme_refit_performed"] is False


def test_contract_rejects_calling_nominal_step_a_universal_level_count(
    tmp_path: Path,
) -> None:
    payload = json.loads(_SCREEN.read_text(encoding="utf-8"))
    payload["deterministic_codebook"]["target_projection"] = "uniform_9_level"
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="deterministic no-noise codebook"):
        load_screen_contract(path)


def test_deterministic_codebook_uses_bounds_and_pulse_resolution_without_noise() -> None:
    population = _population(devices=4)
    codebook = build_deterministic_set_codebook(population, maximum_pulses=2)
    expected = torch.tensor((0.0, 0.5, 0.75), dtype=torch.float32)
    torch.testing.assert_close(codebook.values[:, 0], expected)
    assert torch.equal(
        codebook.effective_level_counts,
        torch.full((population.size,), 3, dtype=torch.int64),
    )


def test_nearest_code_exact_tie_uses_lower_pulse_index() -> None:
    codebook = torch.tensor(((0.0,), (0.5,), (0.75,)), dtype=torch.float32)
    realized, index = project_to_nearest_code(
        codebook,
        torch.tensor((0.625,), dtype=torch.float32),
        chunk_size=1,
    )
    torch.testing.assert_close(realized, torch.tensor((0.5,)))
    assert index.item() == 1


def test_out_of_bound_reference_is_projected_and_reported() -> None:
    population = _population(devices=4, reference=2.0)
    codebook = build_deterministic_set_codebook(population, maximum_pulses=2)
    baseline = build_baseline_state(
        population,
        codebook,
        use_fixed_reference=True,
        projection_chunk_size=8,
    )
    assert baseline.report["raw_reference_outside_sampled_bounds_count"] == population.size
    torch.testing.assert_close(
        baseline.values,
        torch.full((population.size,), 0.75),
    )


def test_four_device_fixed_r_retains_a_r_pattern_and_physical_loading() -> None:
    population = _population(devices=4)
    codebook = build_deterministic_set_codebook(population, maximum_pulses=2)
    exact, continuous, report = build_bounded_scheme_targets(
        (torch.tensor([[1.0]]), torch.tensor([[1.0]])),
        population,
        codebook,
        scheme=Scheme("four_with_fixed_r", 4, True, 0.0),
        scale_fractions=(1.0, 1.0),
        conductance_min=0.0,
        conductance_max=1.0,
        projection_chunk_size=8,
    )
    expected = torch.tensor(((0.75, 0.5), (0.5, 0.75)))
    for target in exact:
        torch.testing.assert_close(target, expected)
    for target in continuous:
        torch.testing.assert_close(target, expected)
    layer = report["layers"][0]
    assert layer["logical_sign_flip_count"] == 0
    assert layer["continuous_logical_sign_flip_count"] == 0
    assert layer["codebook_incremental_sign_flip_count"] == 0
    assert layer["continuous_incremental_sign_flip_count"] == 0
    assert layer["baseline_logical_contrast"]["rms"] == pytest.approx(0.0)
    assert layer["mean_edge_denominator_loading"] == pytest.approx(0.625)


def test_eight_device_fixed_r_uses_difference_for_transfer_and_sum_for_loading() -> None:
    population = _population(devices=8)
    codebook = build_deterministic_set_codebook(population, maximum_pulses=2)
    exact, _continuous, report = build_bounded_scheme_targets(
        (torch.tensor([[1.0]]), torch.tensor([[1.0]])),
        population,
        codebook,
        scheme=Scheme("eight_with_fixed_r", 8, True, 0.0),
        scale_fractions=(1.0, 1.0),
        conductance_min=0.0,
        conductance_max=1.0,
        projection_chunk_size=8,
    )
    expected_plus = torch.tensor(((0.75, 0.5), (0.5, 0.75)))
    expected_minus = torch.full((2, 2), 0.5)
    for plus, minus in ((exact[0], exact[1]), (exact[2], exact[3])):
        torch.testing.assert_close(plus, expected_plus)
        torch.testing.assert_close(minus, expected_minus)
        torch.testing.assert_close(
            plus - minus,
            torch.tensor(((0.25, 0.0), (0.0, 0.25))),
        )
        torch.testing.assert_close(
            plus + minus,
            torch.tensor(((1.25, 1.0), (1.0, 1.25))),
        )
    layer = report["layers"][0]
    assert layer["continuous_logical_sign_flip_count"] == 0
    assert layer["codebook_incremental_sign_flip_count"] == 0
    assert layer["continuous_incremental_sign_flip_count"] == 0
    assert layer["baseline_logical_contrast"]["rms"] == pytest.approx(0.0)
    assert layer["difference_rms"] == pytest.approx(0.1767766953)
    assert layer["mean_edge_denominator_loading"] == pytest.approx(1.125)
