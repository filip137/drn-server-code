from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from experiments.mnist_relu_drn.ibm_om_ideal_mapping_scheme_screen import Scheme
from experiments.mnist_relu_drn.ibm_om_standard_level_scheme_screen import (
    _commission_reset_origin_for_population,
    build_standard_level_grid,
    build_standard_scheme_targets,
    load_standard_level_contract,
)
from training.ibm_reram_hwa import IbmReramArrayPopulation


_ROOT = Path(__file__).resolve().parents[1]
_SCREEN = (
    _ROOT
    / "examples"
    / "mnist_relu_drn"
    / "ibm_om_standard_level_scheme_screen"
    / "screen.json"
)
_SCREEN_V2 = _SCREEN.with_name("screen_v2.json")


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
        nominal_dw_min=0.1,
        dw_min_std=0.4,
        write_noise_std=1.0,
        max_bound=torch.ones(size, dtype=torch.float32),
        min_bound=-torch.ones(size, dtype=torch.float32),
        dwmin_up=torch.full((size,), 0.1, dtype=torch.float32),
        dwmin_down=torch.full((size,), 0.1, dtype=torch.float32),
        reference=torch.full((size,), reference, dtype=torch.float32),
        corrupt=torch.zeros(size, dtype=torch.bool),
        published_corrupt=torch.zeros(size, dtype=torch.bool),
        fingerprint=f"test-{devices}-{reference}",
        aihwkit_version="1.1.0",
    )


def test_contract_pins_reset_centered_four_delta_and_per_scheme_refit() -> None:
    contract = load_standard_level_contract(_SCREEN)
    assert contract.spacing_delta_multiples == 4
    assert contract.development_assignment_seed == 86001
    assert contract.heldout_assignment_seeds == (87001, 87002, 87003)
    assert contract.scale_fractions == (0.125, 0.25, 0.5, 1.0)
    assert (
        contract.raw["standard_levels"]["without_fixed_r_origin"]
        == "sampled_min_bound_reset"
    )
    assert (
        contract.raw["standard_levels"]["with_fixed_r_origin"]
        == "exact_sampled_reference_r"
    )
    assert (
        contract.raw["per_scheme_calibration"]["logit_gain"]
        == "per_scheme_positive_kl_fit"
    )


def test_v2_contract_pins_per_cell_raw_a_reset_mean_commissioning() -> None:
    contract = load_standard_level_contract(_SCREEN_V2)
    assert contract.contract_schema_version == 2
    assert contract.reset_read_samples == 8
    reset = contract.raw["reset_baseline_commissioning"]
    assert reset["baseline_estimator"] == "per_cell_arithmetic_mean"
    assert reset["cross_cell_pooling"] == "none"
    assert reset["standard_error_guard"] == 0.0
    assert (
        contract.raw["standard_levels"]["without_fixed_r_origin"]
        == "bounded_per_cell_mean_of_repeated_apparent_raw_a_reset_reads"
    )


def test_v2_contract_rejects_a_different_reset_sample_count(
    tmp_path: Path,
) -> None:
    payload = json.loads(_SCREEN_V2.read_text(encoding="utf-8"))
    payload["reset_baseline_commissioning"]["samples_per_cell"] = 7
    path = tmp_path / "invalid-v2.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="eight RESET/read"):
        load_standard_level_contract(path)


def test_contract_rejects_spacing_below_four_delta(tmp_path: Path) -> None:
    payload = json.loads(_SCREEN.read_text(encoding="utf-8"))
    payload["standard_levels"]["minimum_spacing_delta_multiples"] = 3
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="four delta apart"):
        load_standard_level_contract(path)


def test_reset_grid_uses_exact_uniform_four_delta_levels() -> None:
    population = _population(devices=4)
    grid = build_standard_level_grid(
        population,
        use_fixed_reference=False,
        spacing_delta_multiples=4,
    )
    assert grid.delta == pytest.approx(0.05)
    assert grid.spacing == pytest.approx(0.2)
    torch.testing.assert_close(grid.origin, torch.zeros(population.size))
    torch.testing.assert_close(grid.active_zero, torch.zeros(population.size))
    assert torch.equal(
        grid.maximum_positive_steps,
        torch.full((population.size,), 5, dtype=torch.int64),
    )
    torch.testing.assert_close(grid.upper_values, torch.ones(population.size))


def test_reset_mean_grid_preserves_per_cell_means_and_bounds_them() -> None:
    population = _population(devices=4)
    raw_mean = torch.linspace(-1.2, 1.2, population.size)
    grid = build_standard_level_grid(
        population,
        use_fixed_reference=False,
        no_r_reset_mean_raw_a=raw_mean,
        spacing_delta_multiples=4,
    )
    expected = torch.clamp((raw_mean + 1.0) / 2.0, 0.0, 1.0)
    torch.testing.assert_close(grid.origin, expected)
    torch.testing.assert_close(grid.active_zero, expected)
    assert torch.unique(grid.origin).numel() > 1
    assert grid.report["policy"] == (
        "bounded_per_cell_mean_apparent_raw_a_reset_origin"
    )
    assert grid.report["reset_mean_commissioning"]["projected_cell_count"] == 2
    positive = grid.maximum_positive_steps > 0
    torch.testing.assert_close(
        grid.upper_values[positive] - grid.origin[positive],
        grid.maximum_positive_steps[positive].to(torch.float32) * grid.spacing,
    )


def test_fixed_r_grid_rejects_reset_mean_origin() -> None:
    population = _population(devices=4)
    with pytest.raises(ValueError, match="only for schemes without fixed r"):
        build_standard_level_grid(
            population,
            use_fixed_reference=True,
            no_r_reset_mean_raw_a=population.min_bound,
        )


def test_reset_commissioning_receipt_fails_closed_on_policy_tamper(
    tmp_path: Path,
) -> None:
    population = _population(devices=4)
    commissioning, report = _commission_reset_origin_for_population(
        population,
        topology=4,
        read_samples=8,
        output_dir=tmp_path,
    )
    assert commissioning.size == population.size
    receipt_path = Path(report["receipt"])
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["baseline_estimator"] = "tampered"
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(RuntimeError, match="exact RESET commissioning receipt"):
        _commission_reset_origin_for_population(
            population,
            topology=4,
            read_samples=8,
            output_dir=tmp_path,
        )


def test_reset_commissioning_artifact_rejects_noncanonical_tensor_dtype(
    tmp_path: Path,
) -> None:
    population = _population(devices=4)
    _commissioning, report = _commission_reset_origin_for_population(
        population,
        topology=4,
        read_samples=8,
        output_dir=tmp_path,
    )
    artifact_path = Path(report["path"])
    with np.load(artifact_path, allow_pickle=False) as source:
        payload = {name: source[name].copy() for name in source.files}
    payload["reset_mean_raw_a"] = payload["reset_mean_raw_a"].astype(
        np.float64
    )
    with artifact_path.open("wb") as stream:
        np.savez_compressed(stream, **payload)
    with pytest.raises(RuntimeError, match="float32 RESET field"):
        _commission_reset_origin_for_population(
            population,
            topology=4,
            read_samples=8,
            output_dir=tmp_path,
        )


def test_fixed_r_grid_keeps_r_exact_and_reports_both_centered_sides() -> None:
    population = _population(devices=4, reference=0.0)
    grid = build_standard_level_grid(
        population,
        use_fixed_reference=True,
        spacing_delta_multiples=4,
    )
    torch.testing.assert_close(grid.origin, torch.full((population.size,), 0.5))
    torch.testing.assert_close(
        grid.active_zero, torch.full((population.size,), 0.5)
    )
    assert torch.equal(
        grid.maximum_positive_steps,
        torch.full((population.size,), 2, dtype=torch.int64),
    )
    assert torch.equal(
        grid.maximum_negative_steps,
        torch.full((population.size,), 2, dtype=torch.int64),
    )
    torch.testing.assert_close(grid.upper_values, torch.full((population.size,), 0.9))
    assert grid.report["sampled_reference_outside_active_bounds_count"] == 0


def test_out_of_bound_r_is_not_moved_but_active_a_uses_nearest_grid_level() -> None:
    population = _population(devices=4, reference=0.8)
    population = replace(
        population,
        max_bound=torch.full((population.size,), 0.5, dtype=torch.float32),
    )
    grid = build_standard_level_grid(
        population,
        use_fixed_reference=True,
        spacing_delta_multiples=4,
    )
    torch.testing.assert_close(grid.origin, torch.full((population.size,), 0.9))
    torch.testing.assert_close(
        grid.active_zero, torch.full((population.size,), 0.7)
    )
    assert torch.equal(
        grid.zero_indices,
        torch.full((population.size,), -1, dtype=torch.int64),
    )
    assert grid.report["sampled_reference_outside_active_bounds_count"] == population.size
    assert grid.report["active_zero_minus_origin"]["mean"] == pytest.approx(-0.2)


def test_narrow_displaced_range_is_explicit_reference_only_zero_level() -> None:
    population = _population(devices=4, reference=0.8)
    population = replace(
        population,
        min_bound=torch.full((population.size,), 0.2, dtype=torch.float32),
        max_bound=torch.full((population.size,), 0.3, dtype=torch.float32),
    )
    grid = build_standard_level_grid(
        population,
        use_fixed_reference=True,
        spacing_delta_multiples=4,
    )
    torch.testing.assert_close(grid.origin, torch.full((population.size,), 0.9))
    torch.testing.assert_close(
        grid.active_zero, torch.full((population.size,), 0.9)
    )
    assert torch.count_nonzero(grid.maximum_positive_steps) == 0
    assert grid.report["no_whole_in_bounds_standard_level_count"] == population.size
    assert grid.report["active_zero_outside_active_bounds_count"] == population.size


def test_four_device_no_r_uses_reset_origin_and_standard_magnitude_levels() -> None:
    population = _population(devices=4)
    grid = build_standard_level_grid(population, use_fixed_reference=False)
    exact, continuous, report = build_standard_scheme_targets(
        (torch.tensor([[1.0]]), torch.tensor([[1.0]])),
        population,
        grid,
        scheme=Scheme("four_without_fixed_r", 4, False, 0.0),
        scale_fractions=(1.0, 1.0),
        conductance_min=0.0,
        conductance_max=1.0,
    )
    expected = torch.tensor(((1.0, 0.0), (0.0, 1.0)))
    for target in exact:
        torch.testing.assert_close(target, expected)
    for target in continuous:
        torch.testing.assert_close(target, expected)
    layer = report["layers"][0]
    assert layer["per_quad_common_positive_step_capacity"]["mean"] == 5
    assert layer["per_quad_common_signed_logical_level_count"]["mean"] == 11
    assert layer["logical_sign_flip_count"] == 0


def test_four_device_fixed_r_retains_a_r_pattern_with_standard_levels() -> None:
    population = _population(devices=4)
    grid = build_standard_level_grid(population, use_fixed_reference=True)
    exact, continuous, report = build_standard_scheme_targets(
        (torch.tensor([[1.0]]), torch.tensor([[1.0]])),
        population,
        grid,
        scheme=Scheme("four_with_fixed_r", 4, True, 0.0),
        scale_fractions=(1.0, 1.0),
        conductance_min=0.0,
        conductance_max=1.0,
    )
    expected = torch.tensor(((0.9, 0.5), (0.5, 0.9)))
    for target in exact:
        torch.testing.assert_close(target, expected)
    for target in continuous:
        torch.testing.assert_close(target, expected)
    layer = report["layers"][0]
    assert layer["per_quad_common_positive_step_capacity"]["mean"] == 2
    assert layer["per_quad_common_signed_logical_level_count"]["mean"] == 5
    assert layer["logical_sign_flip_count"] == 0
    assert layer["active_target_outside_bounds_count"] == 0
    assert layer["mean_edge_denominator_loading"] == pytest.approx(0.7)


def test_four_device_negative_weight_swaps_a_and_r_roles() -> None:
    population = _population(devices=4)
    grid = build_standard_level_grid(population, use_fixed_reference=True)
    exact, _continuous, report = build_standard_scheme_targets(
        (torch.tensor([[-1.0]]), torch.tensor([[-1.0]])),
        population,
        grid,
        scheme=Scheme("four_with_fixed_r", 4, True, 0.0),
        scale_fractions=(1.0, 1.0),
        conductance_min=0.0,
        conductance_max=1.0,
    )
    expected = torch.tensor(((0.5, 0.9), (0.9, 0.5)))
    for target in exact:
        torch.testing.assert_close(target, expected)
    assert report["layers"][0]["logical_sign_flip_count"] == 0


def test_eight_device_fixed_r_uses_difference_and_sum_with_standard_levels() -> None:
    population = _population(devices=8)
    grid = build_standard_level_grid(population, use_fixed_reference=True)
    exact, continuous, report = build_standard_scheme_targets(
        (torch.tensor([[1.0]]), torch.tensor([[1.0]])),
        population,
        grid,
        scheme=Scheme("eight_with_fixed_r", 8, True, 0.0),
        scale_fractions=(1.0, 1.0),
        conductance_min=0.0,
        conductance_max=1.0,
    )
    expected_plus = torch.tensor(((0.9, 0.5), (0.5, 0.9)))
    expected_minus = torch.full((2, 2), 0.5)
    for plus, minus in ((exact[0], exact[1]), (exact[2], exact[3])):
        torch.testing.assert_close(plus, expected_plus)
        torch.testing.assert_close(minus, expected_minus)
        torch.testing.assert_close(
            plus - minus,
            torch.tensor(((0.4, 0.0), (0.0, 0.4))),
        )
        torch.testing.assert_close(
            plus + minus,
            torch.tensor(((1.4, 1.0), (1.0, 1.4))),
        )
    for plus, minus in ((continuous[0], continuous[1]), (continuous[2], continuous[3])):
        torch.testing.assert_close(plus, expected_plus)
        torch.testing.assert_close(minus, expected_minus)
    layer = report["layers"][0]
    assert layer["logical_sign_flip_count"] == 0
    assert layer["difference_rms"] == pytest.approx(0.2828427125)
    assert layer["mean_edge_denominator_loading"] == pytest.approx(1.2)
