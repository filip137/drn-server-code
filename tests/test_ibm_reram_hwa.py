from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path

import numpy as np
import pytest
import torch

from experiments.reram_program_verify.hwa_model import CONDITION_KEY
from model.resistive.builders import ParameterBinding
from model.variable.parameter import DenseWeight
from training.ibm_reram_hwa import (
    IBM_RERAM_ENDPOINT_APPLICATION_POLICY,
    IbmReramArrayPopulation,
    IbmReramHwaConfig,
    IbmReramHwaParameterModifier,
    _map_cell_aware_exact_bounds_quad,
    build_ibm_reram_cell_aware_exact_bounds_codebook,
    load_om_array_population,
    map_ibm_reram_array_targets,
    sample_om_array_population,
    save_om_array_population,
    validate_ibm_reram_target_mapping_preflight,
)
from training.ibm_reram_program_verify import PopulationStepEstimator


def _binding(shape: tuple[int, int] = (2, 2)) -> ParameterBinding:
    parameter = DenseWeight(
        (shape[0],),
        (shape[1],),
        1.0,
        "cpu",
        clamp=True,
        clamp_min=0.1,
        clamp_max=1.0,
    )
    return ParameterBinding("base.dense_weight.0", parameter)


def _differential_bindings(
    shapes: tuple[tuple[int, int], ...] = ((2, 3),),
) -> tuple[ParameterBinding, ...]:
    bindings = []
    for pair_index, shape in enumerate(shapes):
        for role in ("conductance_plus", "conductance_minus"):
            parameter = DenseWeight(
                (shape[0],),
                (shape[1],),
                1.0,
                "cpu",
                clamp=True,
                clamp_min=0.1,
                clamp_max=1.0,
            )
            bindings.append(
                ParameterBinding(
                    f"base.{role}.{pair_index}",
                    parameter,
                    role=role,
                )
            )
    return tuple(bindings)


def _population(
    binding: ParameterBinding,
    *,
    corruption_policy: str,
) -> IbmReramArrayPopulation:
    size = binding.state.numel()
    published = torch.zeros(size, dtype=torch.bool)
    published[1] = True
    corrupt = published.clone() if corruption_policy == "published" else torch.zeros_like(published)
    max_bound = torch.ones(size)
    min_bound = -torch.ones(size)
    dwmin_up = torch.full((size,), 0.25)
    dwmin_down = torch.full((size,), 0.25)
    reference = torch.zeros(size)
    if corruption_policy == "published":
        max_bound[1] = 0.0
        min_bound[1] = 0.0
        dwmin_up[1] = 0.0
        dwmin_down[1] = 0.0
    return IbmReramArrayPopulation(
        assignment_seed=83001,
        corruption_policy=corruption_policy,
        binding_keys=(binding.key,),
        binding_shapes=(tuple(binding.state.shape),),
        binding_sampling_seeds=(11,),
        donor_sampling_seeds=(12,),
        nominal_dw_min=0.0949,
        dw_min_std=0.0,
        write_noise_std=0.0,
        max_bound=max_bound,
        min_bound=min_bound,
        dwmin_up=dwmin_up,
        dwmin_down=dwmin_down,
        reference=reference,
        corrupt=corrupt,
        published_corrupt=published,
        fingerprint=f"fixture-{corruption_policy}",
        aihwkit_version="1.1.0",
    )


def _quantiles(value: float) -> dict[str, object]:
    return {
        "count": 10,
        "probabilities": [0.0, 0.5, 1.0],
        "values": [value, value, value],
    }


def _endpoint_model(
    *,
    corrupt: bool,
    persistent_terminal: float | None = None,
) -> dict[str, object]:
    records = []
    for target in (0.0, 1.0):
        classes = {}
        for name in (
            "target_below_lower_bound",
            "target_inside_bounds",
            "target_above_upper_bound",
        ):
            classes[name] = {
                "probability": 1.0 / 3.0,
                "acceptance_window_reachable_probability": 1.0,
                "success_probability": 1.0,
                "accepted_terminal": {
                    "apparent_endpoint": _quantiles(target),
                    "persistent_endpoint": _quantiles(
                        target
                        if persistent_terminal is None
                        else persistent_terminal
                    ),
                },
                "failed_terminal": {
                    "apparent_endpoint": _quantiles(target),
                    "persistent_endpoint": _quantiles(
                        target
                        if persistent_terminal is None
                        else persistent_terminal
                    ),
                },
            }
        records.append(
            {
                "target": target,
                "tolerance": 0.04745,
                "accepted_noncorrupt_residual": {
                    "fit_count": 10,
                    "bin_edges": [-0.04745, 0.0, 0.04745],
                    "bin_probabilities": [0.5, 0.5],
                },
                "outcome_model": {
                    "corrupt_identity_fraction": 0.25,
                    "noncorrupt_reachability": {"classes": classes},
                    "corrupt_terminal": {
                        "apparent_endpoint": _quantiles(0.5),
                        "persistent_endpoint": _quantiles(0.5),
                    },
                },
            }
        )
    return {
        "schema": "ebl.ibm_reram.bounded_piecewise_uniform_endpoint_model",
        "schema_version": 2,
        "metadata": {
            "preset": "reram_array_om",
            "execution_profile": "hwa_production_cap128",
            "enable_published_corruption": corrupt,
        },
        "conditions": {
            CONDITION_KEY: {
                "fit_status": "fit",
                "reachability_fit_status": "fit",
                "adequate": True,
                "validation": {"per_target": records},
            }
        },
    }


def _device_model(
    path: Path,
    *,
    persistent_terminal: float | None = None,
) -> Path:
    estimator = PopulationStepEstimator(
        bins=4,
        fallback_step=0.0949 / 2.0,
    ).to_mapping()
    payload = {
        "schema": "ebl.ibm_reram.om_hwa_device_model",
        "schema_version": 1,
        "preset": "reram_array_om",
        "programming": {
            "controller": "adaptive",
            "start_protocol": "lower_to_target",
            "condition_key": CONDITION_KEY,
            "tolerance_step_ratio": 0.5,
            "maximum_program_pulses": 128,
            "nominal_dw_min": 0.0949,
            "adaptive": {
                "eta": 0.75,
                "maximum_batch": 32,
                "epsilon": 1e-8,
                "force_one_within_steps": 2.0,
            },
        },
        "endpoint_models": {
            "continuous": _endpoint_model(
                corrupt=False,
                persistent_terminal=persistent_terminal,
            ),
            "published_corruption": _endpoint_model(
                corrupt=True,
                persistent_terminal=persistent_terminal,
            ),
        },
        "step_estimators": {
            "continuous": estimator,
            "published_corruption": estimator,
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _config(
    *,
    execution: str,
    corruption_policy: str,
    noisy_evaluation: bool,
    target_mapping: str = "literal_global",
    dual_rail_layout_by_parameter: dict[str, str] | None = None,
    common_window_margin_fraction: float = 0.0,
    cell_aware_mode: str | None = None,
    cell_aware_signed_levels: int | None = None,
    forward_logit_gain: float | None = None,
) -> IbmReramHwaConfig:
    return IbmReramHwaConfig(
        execution=execution,
        assignment_seed=83001,
        endpoint_seed=83002,
        corruption_policy=corruption_policy,
        noisy_evaluation=noisy_evaluation,
        target_mapping=target_mapping,
        dual_rail_layout_by_parameter=dual_rail_layout_by_parameter,
        common_window_margin_fraction=common_window_margin_fraction,
        cell_aware_mode=cell_aware_mode,
        cell_aware_signed_levels=cell_aware_signed_levels,
        forward_logit_gain=forward_logit_gain,
    )


def _population_from_x_bounds(
    binding: ParameterBinding,
    lower: torch.Tensor,
    upper: torch.Tensor,
    *,
    corrupt: torch.Tensor | None = None,
    corruption_policy: str = "published",
) -> IbmReramArrayPopulation:
    size = binding.state.numel()
    lower = lower.reshape(-1).to(dtype=torch.float32)
    upper = upper.reshape(-1).to(dtype=torch.float32)
    corrupt = (
        torch.zeros(size, dtype=torch.bool)
        if corrupt is None
        else corrupt.reshape(-1).to(dtype=torch.bool)
    )
    step = torch.full((size,), 0.2, dtype=torch.float32)
    step[corrupt] = 0.0
    return IbmReramArrayPopulation(
        assignment_seed=83001,
        corruption_policy=corruption_policy,
        binding_keys=(binding.key,),
        binding_shapes=(tuple(binding.state.shape),),
        binding_sampling_seeds=(11,),
        donor_sampling_seeds=(12,),
        nominal_dw_min=0.0949,
        dw_min_std=0.0,
        write_noise_std=0.0,
        max_bound=2.0 * upper - 1.0,
        min_bound=2.0 * lower - 1.0,
        dwmin_up=step.clone(),
        dwmin_down=step.clone(),
        reference=torch.zeros(size, dtype=torch.float32),
        corrupt=corrupt,
        published_corrupt=corrupt.clone(),
        fingerprint=f"mapping-{binding.key}",
        aihwkit_version="1.1.0",
    )


def _differential_population_from_x_bounds(
    bindings: tuple[ParameterBinding, ...],
    lower_by_binding: tuple[torch.Tensor, ...],
    upper_by_binding: tuple[torch.Tensor, ...],
    *,
    corrupt_by_binding: tuple[torch.Tensor, ...] | None = None,
    corruption_policy: str = "counterfactual_repaired",
) -> IbmReramArrayPopulation:
    if len(bindings) != len(lower_by_binding) or len(bindings) != len(
        upper_by_binding
    ):
        raise AssertionError("Expected one lower and upper tensor per binding.")
    lowers = tuple(value.reshape(-1).to(torch.float32) for value in lower_by_binding)
    uppers = tuple(value.reshape(-1).to(torch.float32) for value in upper_by_binding)
    if corrupt_by_binding is None:
        corrupts = tuple(torch.zeros_like(value, dtype=torch.bool) for value in lowers)
    else:
        corrupts = tuple(
            value.reshape(-1).to(torch.bool) for value in corrupt_by_binding
        )
    lower = torch.cat(lowers)
    upper = torch.cat(uppers)
    corrupt = torch.cat(corrupts)
    step = torch.full_like(lower, 0.2)
    step[corrupt] = 0.0
    return IbmReramArrayPopulation(
        assignment_seed=83001,
        corruption_policy=corruption_policy,
        binding_keys=tuple(binding.key for binding in bindings),
        binding_shapes=tuple(tuple(binding.state.shape) for binding in bindings),
        binding_sampling_seeds=tuple(range(11, 11 + len(bindings))),
        donor_sampling_seeds=tuple(range(101, 101 + len(bindings))),
        nominal_dw_min=0.0949,
        dw_min_std=0.0,
        write_noise_std=0.0,
        max_bound=2.0 * upper - 1.0,
        min_bound=2.0 * lower - 1.0,
        dwmin_up=step.clone(),
        dwmin_down=step.clone(),
        reference=torch.zeros_like(lower),
        corrupt=corrupt,
        published_corrupt=corrupt.clone(),
        fingerprint="mapping-differential-pairs",
        aihwkit_version="1.1.0",
    )


@pytest.mark.parametrize("layout", ["halves", "paired"])
def test_quad_mapping_uses_exact_four_cell_layout_and_inner_margin(
    layout: str,
) -> None:
    binding = _binding((4, 4))
    fractions = torch.tensor(
        [
            [0.00, 0.10, 0.20, 0.30],
            [0.40, 0.50, 0.60, 0.70],
            [0.80, 0.90, 1.00, 0.75],
            [0.25, 0.35, 0.45, 0.55],
        ]
    )
    lower = 0.10 + torch.arange(16, dtype=torch.float32).reshape(4, 4) / 400.0
    upper = 0.90 - torch.arange(16, dtype=torch.float32).reshape(4, 4) / 500.0
    population = _population_from_x_bounds(binding, lower, upper)

    mapped, report = map_ibm_reram_array_targets(
        fractions.reshape(-1),
        population,
        target_mapping="dual_rail_quad_common_window",
        dual_rail_layout_by_parameter={binding.key: layout},
        common_window_margin_fraction=0.25,
    )
    mapped = mapped.reshape(4, 4)
    for logical_input in range(2):
        rows = (logical_input, logical_input + 2)
        for logical_output in range(2):
            columns = (
                (logical_output, logical_output + 2)
                if layout == "halves"
                else (2 * logical_output, 2 * logical_output + 1)
            )
            indices = tuple(
                (row, column) for row in rows for column in columns
            )
            common_low = max(float(lower[index]) for index in indices)
            common_high = min(float(upper[index]) for index in indices)
            raw_span = common_high - common_low
            inner_low = common_low + 0.25 * raw_span
            inner_span = 0.5 * raw_span
            for index in indices:
                assert mapped[index].item() == pytest.approx(
                    inner_low + float(fractions[index]) * inner_span,
                    abs=1e-7,
                )

    assert report["quad_count"] == 4
    assert report["common_window_empty_quad_count"] == 0
    assert report["corrupt_quad_count"] == 0
    assert report["inner_common_span"]["mean"] == pytest.approx(
        0.5 * report["raw_common_span"]["mean"]
    )
    assert report["mapped_target_support"] == {
        "below_lower_bound": 0,
        "above_upper_bound": 0,
        "inside_bounds": 16,
    }
    assert report["global_target_support"]["below_lower_bound"] > 0


def test_quad_mapping_retains_published_corrupt_identity_in_common_window() -> None:
    binding = _binding((2, 2))
    lower = torch.full((2, 2), 0.2)
    upper = torch.full((2, 2), 0.8)
    corrupt = torch.tensor([[False, True], [False, False]])
    lower[corrupt] = 0.6
    upper[corrupt] = 0.6
    population = _population_from_x_bounds(
        binding,
        lower,
        upper,
        corrupt=corrupt,
    )

    mapped, report = map_ibm_reram_array_targets(
        torch.tensor([0.0, 0.25, 0.75, 1.0]),
        population,
        target_mapping="dual_rail_quad_common_window",
        dual_rail_layout_by_parameter={binding.key: "halves"},
        common_window_margin_fraction=0.25,
    )

    torch.testing.assert_close(mapped, torch.full((4,), 0.6))
    assert report["quad_count"] == 1
    assert report["corrupt_quad_count"] == 1
    assert report["published_corrupt_quad_count"] == 1
    assert report["common_window_empty_quad_count"] == 1
    assert report["inner_common_span"] == {
        "minimum": 0.0,
        "mean": 0.0,
        "maximum": 0.0,
    }


def test_quad_preflight_separates_empty_window_failures_and_caps_them() -> None:
    binding = _binding((2, 2))
    lower = torch.tensor([[0.9, 0.2], [0.2, 0.2]])
    upper = torch.tensor([[1.0, 0.8], [0.8, 0.8]])
    population = _population_from_x_bounds(binding, lower, upper)
    _mapped, report = map_ibm_reram_array_targets(
        torch.tensor([0.0, 0.25, 0.75, 1.0]),
        population,
        target_mapping="dual_rail_quad_common_window",
        dual_rail_layout_by_parameter={binding.key: "halves"},
        common_window_margin_fraction=0.25,
    )

    assert report["common_window_empty_quad_count"] == 1
    assert report["mapped_target_below_lower_bound_nonempty_quad"] == 0
    assert report["mapped_target_above_upper_bound_nonempty_quad"] == 0
    assert report["mapped_target_below_lower_bound_empty_quad"] == 1
    assert report["mapped_target_above_upper_bound_empty_quad"] == 3
    validate_ibm_reram_target_mapping_preflight(report)

    too_many = dict(report)
    too_many["common_window_empty_quad_count"] = 21
    with pytest.raises(ValueError, match="no more than 20"):
        validate_ibm_reram_target_mapping_preflight(too_many)

    outside_nonempty = dict(report)
    outside_nonempty["mapped_target_below_lower_bound_nonempty_quad"] = 1
    with pytest.raises(ValueError, match="zero mapped targets"):
        validate_ibm_reram_target_mapping_preflight(outside_nonempty)


def test_cell_aware_quantized_mapper_uses_each_exact_bound_and_half_away_rounding() -> None:
    binding = _binding((2, 4))
    lower = torch.tensor(
        [[-0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]],
        dtype=torch.float32,
    )
    upper = torch.tensor(
        [[1.1, 0.9, 0.8, 0.7], [0.6, 0.7, 0.9, 1.2]],
        dtype=torch.float32,
    )
    population = _population_from_x_bounds(
        binding,
        lower,
        upper,
        corruption_policy="counterfactual_repaired",
    )
    # In paired layout these two quads have 4u = +0.5 and -0.5. Exact
    # half-away rounding therefore selects signed codes +1 and -1.
    source = torch.tensor(
        [[0.25, 0.0, 0.0, 0.25], [0.0, 0.0, 0.0, 0.0]],
        dtype=torch.float32,
    )
    mapped, report = map_ibm_reram_array_targets(
        source.reshape(-1),
        population,
        target_mapping="cell_aware_exact_bounds_quad",
        dual_rail_layout_by_parameter={binding.key: "paired"},
        common_window_margin_fraction=0.0,
        cell_aware_mode="quantized_9_level",
        cell_aware_signed_levels=9,
    )

    expected = torch.tensor(
        [[0.25, 0.2, 0.3, 0.475], [0.5, 0.625, 0.75, 0.8]],
        dtype=torch.float32,
    )
    torch.testing.assert_close(mapped.reshape(2, 4), expected)
    assert report["oracle"] is True
    assert report["hidden_device_bounds_consumed_by_target_mapper"] is True
    assert report["post_mapping_clipping"] is False
    assert report["code_index_counts"] == {
        "0": 0,
        "1": 0,
        "2": 0,
        "3": 1,
        "4": 0,
        "5": 1,
        "6": 0,
        "7": 0,
        "8": 0,
    }
    assert report["mapped_target_support"] == {
        "below_lower_bound": 0,
        "above_upper_bound": 0,
        "inside_bounds": 8,
    }
    assert report["hashes"]["mapped_target"] == (
        "8a11647f50c2c96f8309a2db4d90415bbf35b6245ad0bff63e22f4513c37fbd6"
    )
    assert report["code_index_sha256"] == (
        "aa7c9e574f8acc19902fa6abeab4256916695b58790de278f5fc70383a8fbb7b"
    )
    validate_ibm_reram_target_mapping_preflight(report)

    artifact = build_ibm_reram_cell_aware_exact_bounds_codebook(
        source.reshape(-1),
        population,
        dual_rail_layout_by_parameter={binding.key: "paired"},
        cell_aware_mode="quantized_9_level",
    )
    torch.testing.assert_close(
        artifact["cell_logical_min"].reshape(2, 4),
        population.logical_min.reshape(2, 4),
    )
    torch.testing.assert_close(
        artifact["cell_logical_max"].reshape(2, 4),
        population.logical_max.reshape(2, 4),
    )
    torch.testing.assert_close(
        artifact["normalized_exact_lower"].reshape(2, 4), lower
    )
    torch.testing.assert_close(
        artifact["normalized_exact_upper"].reshape(2, 4), upper
    )
    torch.testing.assert_close(artifact["usable_lower"].reshape(2, 4), lower.clamp_min(0.0))
    torch.testing.assert_close(artifact["usable_upper"].reshape(2, 4), upper.clamp_max(1.0))
    torch.testing.assert_close(artifact["requested_target"], mapped)
    assert artifact["signed_code"].tolist() == [1, -1]
    assert artifact["code_index"].tolist() == [5, 3]
    assert artifact["report"]["hashes"] == report["hashes"]


def test_cell_aware_continuous_mapper_does_not_quantize_or_compensate_baselines() -> None:
    binding = _binding((2, 2))
    lower = torch.tensor([[0.1, 0.2], [0.3, 0.4]])
    upper = torch.tensor([[0.9, 0.8], [0.7, 0.6]])
    population = _population_from_x_bounds(
        binding,
        lower,
        upper,
        corruption_policy="counterfactual_repaired",
    )
    source = torch.tensor([[0.4, 0.0], [0.0, 0.0]])
    mapped, report = map_ibm_reram_array_targets(
        source.reshape(-1),
        population,
        target_mapping="cell_aware_exact_bounds_quad",
        dual_rail_layout_by_parameter={binding.key: "halves"},
        common_window_margin_fraction=0.0,
        cell_aware_mode="continuous",
        cell_aware_signed_levels=9,
    )

    # u=0.2 and n=0.8, so positive rails receive exactly 0.2 of their own
    # spans. Unselected rails remain at their distinct per-cell floors.
    torch.testing.assert_close(
        mapped.reshape(2, 2),
        torch.tensor([[0.26, 0.2], [0.3, 0.44]]),
    )
    assert report["rounding"] is None
    assert "code_index_counts" not in report
    assert report["cell_specific_baseline_contrast"]["mean"] == pytest.approx(0.0)
    validate_ibm_reram_target_mapping_preflight(report)


@pytest.mark.parametrize("mode", ["continuous", "quantized_9_level"])
def test_cell_aware_training_mapper_defers_evidence_without_changing_targets(
    mode: str,
) -> None:
    binding = _binding((2, 4))
    lower = torch.tensor(
        [[0.10, 0.20, 0.30, 0.40], [0.50, 0.60, 0.70, 0.80]]
    )
    upper = torch.tensor(
        [[0.90, 0.85, 0.80, 0.75], [0.70, 0.75, 0.90, 0.95]]
    )
    population = _population_from_x_bounds(
        binding,
        lower,
        upper,
        corruption_policy="counterfactual_repaired",
    )
    source = torch.tensor(
        [[0.625, 0.125, 0.25, 0.75], [0.125, 0.5, 0.625, 0.25]]
    ).reshape(-1)
    layouts = ((binding.key, "paired"),)

    detailed, detailed_report, artifact = _map_cell_aware_exact_bounds_quad(
        source,
        population,
        layouts=layouts,
        mode=mode,
        signed_levels=9,
    )
    deferred, deferred_report, deferred_artifact = (
        _map_cell_aware_exact_bounds_quad(
            source,
            population,
            layouts=layouts,
            mode=mode,
            signed_levels=9,
            materialize_evidence=False,
        )
    )

    assert torch.equal(deferred, detailed)
    assert artifact is not None
    assert deferred_artifact is None
    assert deferred_report["mapped_target_support"] == detailed_report[
        "mapped_target_support"
    ]
    assert deferred_report["evidence_materialization"] == (
        "deferred_to_evaluation_stream"
    )
    if mode == "quantized_9_level":
        assert deferred_report["code_index_counts"] == detailed_report[
            "code_index_counts"
        ]
    validate_ibm_reram_target_mapping_preflight(deferred_report)


def test_cell_aware_config_is_strict_and_records_separate_forward_gain() -> None:
    config = _config(
        execution="compact_endpoint",
        corruption_policy="counterfactual_repaired",
        noisy_evaluation=False,
        target_mapping="cell_aware_exact_bounds_quad",
        dual_rail_layout_by_parameter={"base.dense_weight.0": "halves"},
        cell_aware_mode="quantized_9_level",
        cell_aware_signed_levels=9,
        forward_logit_gain=2.75,
    )
    assert config.cell_aware_mode == "quantized_9_level"
    assert config.cell_aware_signed_levels == 9
    assert config.forward_logit_gain == 2.75

    with pytest.raises(ValueError, match="cell_aware_signed_levels"):
        _config(
            execution="compact_endpoint",
            corruption_policy="counterfactual_repaired",
            noisy_evaluation=False,
            target_mapping="cell_aware_exact_bounds_quad",
            dual_rail_layout_by_parameter={"base.dense_weight.0": "halves"},
            cell_aware_mode="quantized_9_level",
            cell_aware_signed_levels=7,
        )


def test_differential_pair_config_requires_null_layout_and_allows_margin() -> None:
    config = _config(
        execution="compact_endpoint",
        corruption_policy="counterfactual_repaired",
        noisy_evaluation=False,
        target_mapping="differential_pair_common_window",
        common_window_margin_fraction=0.25,
    )
    assert config.dual_rail_layout_by_parameter is None
    assert config.common_window_margin_fraction == 0.25

    with pytest.raises(ValueError, match="canonical adjacent plus/minus"):
        _config(
            execution="compact_endpoint",
            corruption_policy="counterfactual_repaired",
            noisy_evaluation=False,
            target_mapping="differential_pair_common_window",
            dual_rail_layout_by_parameter={
                "base.conductance_plus.0": "halves"
            },
            common_window_margin_fraction=0.25,
        )
    with pytest.raises(ValueError, match="literal_global"):
        _config(
            execution="compact_endpoint",
            corruption_policy="counterfactual_repaired",
            noisy_evaluation=False,
            common_window_margin_fraction=0.25,
        )


def test_differential_pair_mapping_uses_same_coordinates_and_inner_margin() -> None:
    plus, minus = _differential_bindings(((2, 3),))
    plus_target = torch.tensor([[0.0, 0.2, 0.4], [0.6, 0.8, 1.0]])
    minus_target = torch.tensor([[1.0, 0.8, 0.6], [0.4, 0.2, 0.0]])
    plus_lower = torch.tensor([[0.10, 0.20, 0.30], [0.10, 0.20, 0.30]])
    minus_lower = torch.tensor([[0.20, 0.10, 0.25], [0.20, 0.10, 0.25]])
    plus_upper = torch.tensor([[0.90, 0.80, 0.70], [0.90, 0.80, 0.70]])
    minus_upper = torch.tensor([[0.80, 0.90, 0.75], [0.80, 0.90, 0.75]])
    population = _differential_population_from_x_bounds(
        (plus, minus),
        (plus_lower, minus_lower),
        (plus_upper, minus_upper),
    )

    mapped, report = map_ibm_reram_array_targets(
        torch.cat((plus_target.reshape(-1), minus_target.reshape(-1))),
        population,
        target_mapping="differential_pair_common_window",
        dual_rail_layout_by_parameter=None,
        common_window_margin_fraction=0.25,
    )

    common_low = torch.maximum(plus_lower, minus_lower)
    common_high = torch.minimum(plus_upper, minus_upper)
    raw_span = common_high - common_low
    inner_low = common_low + 0.25 * raw_span
    inner_span = 0.5 * raw_span
    expected = torch.cat(
        (
            (inner_low + plus_target * inner_span).reshape(-1),
            (inner_low + minus_target * inner_span).reshape(-1),
        )
    )
    torch.testing.assert_close(mapped, expected)
    assert report["common_window_grouping"] == "differential_pair"
    assert report["common_window_group_size"] == 2
    assert report["common_window_group_count"] == 6
    assert report["pair_count"] == 6
    assert report["common_window_empty_pair_count"] == 0
    assert report["raw_common_span"]["mean"] == pytest.approx(
        raw_span.mean().item()
    )
    assert report["inner_common_span"]["mean"] == pytest.approx(
        inner_span.mean().item()
    )
    assert report["nonempty_inner_common_span"]["minimum"] == pytest.approx(
        inner_span.min().item()
    )
    assert report["nonempty_inner_common_span"]["minimum"] > 0.0
    assert (
        report["nonempty_pair_inner_common_span"]
        == report["nonempty_inner_common_span"]
    )
    assert report["mapped_target_support"] == {
        "below_lower_bound": 0,
        "above_upper_bound": 0,
        "inside_bounds": 12,
    }
    assert report["differential_pair_binding"] == {
        "policy": "canonical_adjacent_plus_minus_same_coordinate",
        "catalog_order": [plus.key, minus.key],
        "pairs": [
            {
                "pair_index": 0,
                "conductance_plus_key": plus.key,
                "conductance_minus_key": minus.key,
                "shape": [2, 3],
            }
        ],
    }
    pair_report = report["parameter_pairs"]["base.differential_pair.0"]
    assert pair_report["pair_count"] == 6
    assert pair_report["conductance_plus_key"] == plus.key
    assert pair_report["conductance_minus_key"] == minus.key
    validate_ibm_reram_target_mapping_preflight(report)


def test_differential_pair_mapping_clamps_shared_support_to_global_coordinate() -> None:
    plus, minus = _differential_bindings(((1, 2),))
    population = _differential_population_from_x_bounds(
        (plus, minus),
        (
            torch.tensor([[-0.2, -0.1]]),
            torch.tensor([[-0.1, -0.2]]),
        ),
        (
            torch.tensor([[1.2, 1.1]]),
            torch.tensor([[1.1, 1.2]]),
        ),
    )

    mapped, report = map_ibm_reram_array_targets(
        torch.tensor([0.0, 1.0, 0.25, 0.75]),
        population,
        target_mapping="differential_pair_common_window",
        dual_rail_layout_by_parameter=None,
        common_window_margin_fraction=0.1,
    )

    torch.testing.assert_close(mapped, torch.tensor([0.1, 0.9, 0.3, 0.7]))
    assert report["raw_common_span"] == {
        "minimum": 1.0,
        "mean": 1.0,
        "maximum": 1.0,
    }
    assert report["nonempty_inner_common_span"] == {
        "minimum": pytest.approx(0.8),
        "mean": pytest.approx(0.8),
        "maximum": pytest.approx(0.8),
    }


@pytest.mark.parametrize(
    ("bindings", "match"),
    [
        (
            tuple(reversed(_differential_bindings(((2, 3),)))),
            "canonical adjacent plus/minus key and suffix order",
        ),
        (
            (
                _differential_bindings(((2, 3),))[0],
                ParameterBinding(
                    "base.conductance_minus.1",
                    _differential_bindings(((2, 3),))[1].parameter,
                    role="conductance_minus",
                ),
            ),
            "canonical adjacent plus/minus key and suffix order",
        ),
        (
            (
                _differential_bindings(((2, 3),))[0],
                _differential_bindings(((3, 2),))[1],
            ),
            "identical non-empty rank-2 shapes",
        ),
    ],
)
def test_differential_pair_mapper_rejects_noncanonical_order_suffix_and_shape(
    bindings: tuple[ParameterBinding, ...],
    match: str,
) -> None:
    lower = tuple(torch.full(tuple(binding.state.shape), 0.2) for binding in bindings)
    upper = tuple(torch.full(tuple(binding.state.shape), 0.8) for binding in bindings)
    population = _differential_population_from_x_bounds(
        bindings,
        lower,
        upper,
    )
    with pytest.raises(ValueError, match=match):
        map_ibm_reram_array_targets(
            torch.full((population.size,), 0.5),
            population,
            target_mapping="differential_pair_common_window",
            dual_rail_layout_by_parameter=None,
            common_window_margin_fraction=0.25,
        )


def test_repaired_assignment_changes_only_published_corrupt_sites(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    binding = _binding((1, 4))

    def sample_hidden(shape, *, construction_seed, published_corruption):
        size = shape[0] * shape[1]
        values = {
            "max_bound": torch.full((size,), 1.0 if published_corruption else 0.8),
            "min_bound": torch.full((size,), -1.0 if published_corruption else -0.8),
            "dwmin_up": torch.full((size,), 0.2 if published_corruption else 0.3),
            "dwmin_down": torch.full((size,), 0.2 if published_corruption else 0.3),
            "reference": torch.zeros(size),
        }
        if published_corruption:
            for name in ("max_bound", "min_bound"):
                values[name][1] = 0.1
            values["dwmin_up"][1] = 0.0
            values["dwmin_down"][1] = 0.0
        return values, {
            "aihwkit_version": "1.1.0",
            "nominal_dw_min": 0.0949,
            "dw_min_std": 0.0,
            "write_noise_std": 0.0,
        }

    monkeypatch.setattr(
        "training.ibm_reram_hwa._sample_tile_hidden",
        sample_hidden,
    )
    published = sample_om_array_population(
        (binding,), assignment_seed=83001, corruption_policy="published"
    )
    repaired = sample_om_array_population(
        (binding,),
        assignment_seed=83001,
        corruption_policy="counterfactual_repaired",
    )

    healthy = ~published.published_corrupt
    assert torch.equal(published.max_bound[healthy], repaired.max_bound[healthy])
    assert torch.equal(published.dwmin_up[healthy], repaired.dwmin_up[healthy])
    assert published.published_corrupt.tolist() == [False, True, False, False]
    assert torch.equal(repaired.published_corrupt, published.published_corrupt)
    assert not bool(torch.any(repaired.corrupt))
    assert repaired.max_bound[1].item() == pytest.approx(0.8)


def test_compact_modifier_restores_clean_master_and_rng_exactly(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    binding = _binding()
    binding.state.copy_(torch.tensor([[0.2, 0.4], [0.6, 0.8]]))
    population = _population(binding, corruption_policy="published")
    monkeypatch.setattr(
        "training.ibm_reram_hwa.sample_om_array_population",
        lambda *args, **kwargs: population,
    )
    modifier = IbmReramHwaParameterModifier(
        (binding,),
        _config(
            execution="compact_endpoint",
            corruption_policy="published",
            noisy_evaluation=False,
        ),
        device_model_path=_device_model(tmp_path / "device.json"),
        conductance_min=0.1,
        conductance_max=1.0,
    )
    clean = binding.state.clone()
    snapshot = modifier.state_dict()
    assert snapshot["version"] == 2
    assert (
        snapshot["endpoint_application_policy"]
        == IBM_RERAM_ENDPOINT_APPLICATION_POLICY
    )
    with modifier.training_context():
        first = binding.state.clone()
        assert first[0, 1].item() == pytest.approx(0.55)
    assert torch.equal(binding.state, clean)
    modifier.load_state_dict(snapshot)
    with modifier.training_context():
        replay = binding.state.clone()
    assert torch.equal(first, replay)
    assert torch.equal(binding.state, clean)

    wrong_policy = dict(snapshot)
    wrong_policy["endpoint_application_policy"] = "persistent_forward"
    with pytest.raises(ValueError, match="endpoint application policy"):
        modifier.load_state_dict(wrong_policy)


def test_compact_forward_applies_apparent_and_retains_persistent_endpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    binding = _binding()
    binding.state.copy_(torch.tensor([[0.2, 0.4], [0.6, 0.8]]))
    population = _population(
        binding,
        corruption_policy="counterfactual_repaired",
    )
    monkeypatch.setattr(
        "training.ibm_reram_hwa.sample_om_array_population",
        lambda *args, **kwargs: population,
    )
    modifier = IbmReramHwaParameterModifier(
        (binding,),
        _config(
            execution="compact_endpoint",
            corruption_policy="counterfactual_repaired",
            noisy_evaluation=False,
        ),
        device_model_path=_device_model(
            tmp_path / "device.json",
            persistent_terminal=0.25,
        ),
        conductance_min=0.1,
        conductance_max=1.0,
    )

    clean = binding.state.clone()
    with modifier.training_context():
        applied_forward = ((binding.state - 0.1) / 0.9).reshape(-1).clone()
    bundle = modifier.last_deployment_bundle
    assert bundle is not None
    torch.testing.assert_close(applied_forward, bundle["apparent_endpoint"])
    torch.testing.assert_close(
        bundle["persistent_endpoint"],
        torch.full((4,), 0.25),
    )
    assert not torch.equal(
        bundle["apparent_endpoint"], bundle["persistent_endpoint"]
    )
    assert torch.equal(binding.state, clean)


def test_modifier_preflight_and_compact_context_share_quad_mapping(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    binding = _binding()
    binding.state.copy_(torch.tensor([[0.1, 0.325], [0.775, 1.0]]))
    population = _population(binding, corruption_policy="published")
    monkeypatch.setattr(
        "training.ibm_reram_hwa.sample_om_array_population",
        lambda *args, **kwargs: population,
    )
    modifier = IbmReramHwaParameterModifier(
        (binding,),
        _config(
            execution="compact_endpoint",
            corruption_policy="published",
            noisy_evaluation=False,
            target_mapping="dual_rail_quad_common_window",
            dual_rail_layout_by_parameter={binding.key: "halves"},
            common_window_margin_fraction=0.25,
        ),
        device_model_path=_device_model(tmp_path / "device.json"),
        conductance_min=0.1,
        conductance_max=1.0,
    )

    preflight_targets, preflight = modifier.preflight_target_mapping()
    torch.testing.assert_close(preflight_targets, torch.full((4,), 0.5))
    assert preflight["common_window_empty_quad_count"] == 1
    clean = binding.state.clone()
    with modifier.training_context():
        pass
    assert torch.equal(binding.state, clean)
    report = modifier.programming_report
    assert report is not None
    assert report["target_mapping"] == "dual_rail_quad_common_window"
    assert report["target_mapping_report"] == preflight


def test_cell_aware_compact_context_has_no_fallback_and_restores_fp32_shadow(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    binding = _binding((2, 2))
    source = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    binding.state.copy_(0.1 + 0.9 * source)
    lower = torch.tensor([[0.1, 0.2], [0.3, 0.4]])
    upper = torch.tensor([[0.9, 0.8], [0.7, 0.6]])
    population = _population_from_x_bounds(
        binding,
        lower,
        upper,
        corruption_policy="counterfactual_repaired",
    )
    monkeypatch.setattr(
        "training.ibm_reram_hwa.sample_om_array_population",
        lambda *args, **kwargs: population,
    )
    modifier = IbmReramHwaParameterModifier(
        (binding,),
        _config(
            execution="compact_endpoint",
            corruption_policy="counterfactual_repaired",
            noisy_evaluation=True,
            target_mapping="cell_aware_exact_bounds_quad",
            dual_rail_layout_by_parameter={binding.key: "halves"},
            cell_aware_mode="quantized_9_level",
            cell_aware_signed_levels=9,
            forward_logit_gain=2.75,
        ),
        device_model_path=_device_model(tmp_path / "device.json"),
        conductance_min=0.1,
        conductance_max=1.0,
    )

    expected, preflight = modifier.preflight_target_mapping()
    torch.testing.assert_close(
        expected.reshape(2, 2),
        torch.tensor([[0.9, 0.2], [0.3, 0.6]]),
    )
    validate_ibm_reram_target_mapping_preflight(preflight)
    clean = binding.state.clone()
    snapshot = modifier.state_dict()
    with modifier.training_context():
        assert not torch.equal(binding.state, clean)
    assert torch.equal(binding.state, clean)
    report = modifier.programming_report
    deployment = modifier.last_deployment_bundle
    assert report is not None and deployment is not None
    assert report["pulse_resolved_fallback_devices"] == 0
    assert report["execution_detail"] == "compact_endpoint_exact_bounds_in_support"
    assert deployment["endpoint_generation_policy"] == (
        "compact_exact_bounds_in_support_only"
    )
    assert not bool(torch.any(deployment["pulse_resolved_fallback_mask"]))
    torch.testing.assert_close(deployment["requested_target"], expected)
    assert deployment["oracle_codebook"] is None
    assert deployment["target_mapping_report"]["evidence_materialization"] == (
        "deferred_to_evaluation_stream"
    )

    with modifier.evaluation_context():
        assert not torch.equal(binding.state, clean)
    assert torch.equal(binding.state, clean)
    evaluation = modifier.last_deployment_bundle
    assert evaluation is not None
    codebook = evaluation["oracle_codebook"]
    assert codebook["schema"] == (
        "ebl.ibm_reram.om_cell_aware_exact_bounds_codebook"
    )
    assert codebook["signed_code"].tolist() == [4]
    assert codebook["code_index"].tolist() == [8]
    assert "hashes" in evaluation["target_mapping_report"]

    first = deployment["apparent_endpoint"].clone()
    modifier.load_state_dict(snapshot)
    with modifier.training_context():
        pass
    replay = modifier.last_deployment_bundle
    assert replay is not None
    assert torch.equal(replay["requested_target"], expected)
    assert torch.equal(replay["apparent_endpoint"], first)
    assert replay["oracle_codebook"] is None
    assert torch.equal(binding.state, clean)


def test_compact_quad_mapping_uses_exact_fallback_for_unsupported_empty_window_cells(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    binding = _binding((2, 4))
    binding.state.fill_(0.55)
    lower = torch.full((2, 4), 0.2)
    upper = torch.full((2, 4), 0.8)
    # The first logical quad has no common interval: one cell starts at 0.84,
    # while the other three saturate at 0.80. Its midpoint is 0.82.
    lower[0, 0] = 0.84
    upper[0, 0] = 1.0
    population = _population_from_x_bounds(
        binding,
        lower,
        upper,
        corruption_policy="counterfactual_repaired",
    )
    monkeypatch.setattr(
        "training.ibm_reram_hwa.sample_om_array_population",
        lambda *args, **kwargs: population,
    )

    device_model = _device_model(tmp_path / "device.json")
    payload = json.loads(device_model.read_text(encoding="utf-8"))
    for endpoint_model in payload["endpoint_models"].values():
        records = endpoint_model["conditions"][CONDITION_KEY]["validation"][
            "per_target"
        ]
        for record in records:
            classes = record["outcome_model"]["noncorrupt_reachability"][
                "classes"
            ]
            for name in (
                "target_below_lower_bound",
                "target_above_upper_bound",
            ):
                classes[name]["success_probability"] = None
                classes[name][
                    "acceptance_window_reachable_probability"
                ] = None
    device_model.write_text(json.dumps(payload), encoding="utf-8")

    modifier = IbmReramHwaParameterModifier(
        (binding,),
        _config(
            execution="compact_endpoint",
            corruption_policy="counterfactual_repaired",
            noisy_evaluation=False,
            target_mapping="dual_rail_quad_common_window",
            dual_rail_layout_by_parameter={binding.key: "halves"},
            common_window_margin_fraction=0.25,
        ),
        device_model_path=device_model,
        conductance_min=0.1,
        conductance_max=1.0,
    )

    clean = binding.state.clone()
    initial_modifier_state = modifier.state_dict()
    with modifier.training_context():
        assert bool(torch.all(torch.isfinite(binding.state)))
    assert torch.equal(binding.state, clean)
    report = modifier.programming_report
    deployment = modifier.last_deployment_bundle
    assert report is not None
    assert report["execution"] == "compact_endpoint"
    assert (
        report["execution_detail"]
        == "compact_endpoint_with_exact_empty_quad_fallback"
    )
    assert report["compact_endpoint_devices"] == 4
    assert report["pulse_resolved_fallback_devices"] == 4
    fallback_report = report["pulse_resolved_fallback"]
    assert fallback_report["selection_indices"] == [0, 2, 4, 6]
    assert fallback_report["target_below_lower_bound"] == 1
    assert fallback_report["target_above_upper_bound"] == 3
    assert fallback_report["accepted"] == 4
    assert fallback_report["pulse_count"]["maximum"] > 0
    assert deployment is not None
    assert deployment["pulse_resolved_fallback_mask"].tolist() == [
        True,
        False,
        True,
        False,
        True,
        False,
        True,
        False,
    ]
    fallback = deployment["pulse_resolved_fallback"]
    assert fallback["selection_indices"].tolist() == [0, 2, 4, 6]
    assert fallback["total_pulses"].shape == (4,)
    assert fallback["total_pulses"][0].item() == 0

    first_apparent = deployment["apparent_endpoint"].clone()
    first_persistent = deployment["persistent_endpoint"].clone()
    first_pulses = fallback["total_pulses"].clone()
    modifier.load_state_dict(initial_modifier_state)
    with modifier.training_context():
        pass
    replay = modifier.last_deployment_bundle
    assert replay is not None
    assert torch.equal(replay["apparent_endpoint"], first_apparent)
    assert torch.equal(replay["persistent_endpoint"], first_persistent)
    assert torch.equal(
        replay["pulse_resolved_fallback"]["total_pulses"],
        first_pulses,
    )

    # Outside the quad-mapping contract, an absent class fit remains a hard
    # error; the hybrid does not silently turn into a general extrapolator.
    binding.state.fill_(0.1 + 0.9 * 0.82)
    literal = IbmReramHwaParameterModifier(
        (binding,),
        _config(
            execution="compact_endpoint",
            corruption_policy="counterfactual_repaired",
            noisy_evaluation=False,
        ),
        device_model_path=device_model,
        conductance_min=0.1,
        conductance_max=1.0,
    )
    with pytest.raises(
        ValueError,
        match="fitted class-conditional non-corrupt success probability",
    ):
        with literal.training_context():
            pass


def test_compact_differential_pair_mapping_replays_exact_16_cell_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plus, minus = _differential_bindings(((1, 10),))
    plus.state.fill_(0.55)
    minus.state.fill_(0.55)
    plus_lower = torch.full((1, 10), 0.2)
    plus_upper = torch.full((1, 10), 0.8)
    minus_lower = torch.full((1, 10), 0.2)
    minus_upper = torch.full((1, 10), 0.8)
    # Eight coordinates have disjoint pair support. Their clamped midpoint is
    # 0.82: below the G+ lower bound and above the G- upper bound. The final
    # two coordinates remain covered by the compact endpoint surrogate.
    plus_lower[:, :8] = 0.84
    plus_upper[:, :8] = 1.0
    population = _differential_population_from_x_bounds(
        (plus, minus),
        (plus_lower, minus_lower),
        (plus_upper, minus_upper),
    )
    monkeypatch.setattr(
        "training.ibm_reram_hwa.sample_om_array_population",
        lambda *args, **kwargs: population,
    )

    device_model = _device_model(tmp_path / "device.json")
    payload = json.loads(device_model.read_text(encoding="utf-8"))
    for endpoint_model in payload["endpoint_models"].values():
        records = endpoint_model["conditions"][CONDITION_KEY]["validation"][
            "per_target"
        ]
        for record in records:
            classes = record["outcome_model"]["noncorrupt_reachability"][
                "classes"
            ]
            for name in (
                "target_below_lower_bound",
                "target_above_upper_bound",
            ):
                classes[name]["success_probability"] = None
                classes[name][
                    "acceptance_window_reachable_probability"
                ] = None
    device_model.write_text(json.dumps(payload), encoding="utf-8")

    modifier = IbmReramHwaParameterModifier(
        (plus, minus),
        _config(
            execution="compact_endpoint",
            corruption_policy="counterfactual_repaired",
            noisy_evaluation=False,
            target_mapping="differential_pair_common_window",
            common_window_margin_fraction=0.25,
        ),
        device_model_path=device_model,
        conductance_min=0.1,
        conductance_max=1.0,
    )

    mapped, mapping_report = modifier.preflight_target_mapping()
    assert mapping_report["pair_count"] == 10
    assert mapping_report["common_window_empty_pair_count"] == 8
    assert mapping_report["mapped_target_below_lower_bound_empty_pair"] == 8
    assert mapping_report["mapped_target_above_upper_bound_empty_pair"] == 8
    assert mapping_report["mapped_target_below_lower_bound_nonempty_pair"] == 0
    assert mapping_report["mapped_target_above_upper_bound_nonempty_pair"] == 0
    assert mapping_report["nonempty_inner_common_span"]["minimum"] > 0.0
    with pytest.raises(ValueError, match="no more than 7 empty differential-pair"):
        validate_ibm_reram_target_mapping_preflight(
            mapping_report,
            maximum_empty_pairs=7,
        )
    torch.testing.assert_close(mapped[:8], torch.full((8,), 0.82))
    torch.testing.assert_close(mapped[10:18], torch.full((8,), 0.82))

    clean_plus = plus.state.clone()
    clean_minus = minus.state.clone()
    initial_modifier_state = modifier.state_dict()
    with modifier.training_context():
        assert bool(torch.all(torch.isfinite(plus.state)))
        assert bool(torch.all(torch.isfinite(minus.state)))
    assert torch.equal(plus.state, clean_plus)
    assert torch.equal(minus.state, clean_minus)

    report = modifier.programming_report
    deployment = modifier.last_deployment_bundle
    assert report is not None
    assert deployment is not None
    assert (
        report["execution_detail"]
        == "compact_endpoint_with_exact_empty_pair_fallback"
    )
    assert report["compact_endpoint_devices"] == 4
    assert report["pulse_resolved_fallback_devices"] == 16
    fallback_report = report["pulse_resolved_fallback"]
    assert (
        fallback_report["policy"]
        == "pulse_resolved_noncorrupt_out_of_bound_empty_pair_only"
    )
    expected_indices = list(range(8)) + list(range(10, 18))
    expected_digest = sha256(
        np.asarray(expected_indices, dtype=np.int64).tobytes()
    ).hexdigest()
    assert fallback_report["selection_indices"] == expected_indices
    assert fallback_report["selection_indices_sha256"] == expected_digest
    assert fallback_report["target_below_lower_bound"] == 8
    assert fallback_report["target_above_upper_bound"] == 8
    assert fallback_report["devices"] == 16
    assert fallback_report["accepted"] == 16
    assert fallback_report["pulse_count"]["maximum"] <= 128
    assert (
        deployment["endpoint_generation_policy"]
        == "compact_covered_exact_out_of_bound_empty_pair_fallback"
    )
    expected_mask = torch.zeros(20, dtype=torch.bool)
    expected_mask[expected_indices] = True
    assert torch.equal(deployment["pulse_resolved_fallback_mask"], expected_mask)
    assert torch.equal(deployment["compact_endpoint_mask"], ~expected_mask)
    fallback = deployment["pulse_resolved_fallback"]
    assert fallback["selection_indices_sha256"] == expected_digest
    assert fallback["selection_indices"].tolist() == expected_indices
    for name in (
        "requested_target",
        "raw_apparent_endpoint",
        "apparent_endpoint",
        "persistent_endpoint",
        "accepted",
        "budget_exhausted",
        "set_count",
        "reset_count",
        "total_pulses",
        "verify_count",
        "reversals",
    ):
        assert fallback[name].shape == (16,)

    first_bundle = {
        name: fallback[name].clone()
        for name in (
            "raw_apparent_endpoint",
            "apparent_endpoint",
            "persistent_endpoint",
            "accepted",
            "budget_exhausted",
            "set_count",
            "reset_count",
            "total_pulses",
            "verify_count",
            "reversals",
        )
    }
    modifier.load_state_dict(initial_modifier_state)
    assert (
        modifier.config.target_mapping
        == "differential_pair_common_window"
    )
    with modifier.training_context():
        pass
    replay = modifier.last_deployment_bundle
    assert replay is not None
    replay_fallback = replay["pulse_resolved_fallback"]
    assert torch.equal(replay["pulse_resolved_fallback_mask"], expected_mask)
    assert replay_fallback["selection_indices_sha256"] == expected_digest
    for name, expected in first_bundle.items():
        assert torch.equal(replay_fallback[name], expected)


def test_differential_pair_modifier_rejects_incorrect_binding_roles(
    tmp_path: Path,
) -> None:
    plus, minus = _differential_bindings(((1, 2),))
    wrong_plus = ParameterBinding(plus.key, plus.parameter)
    with pytest.raises(ValueError, match="conductance_plus/conductance_minus roles"):
        IbmReramHwaParameterModifier(
            (wrong_plus, minus),
            _config(
                execution="compact_endpoint",
                corruption_policy="counterfactual_repaired",
                noisy_evaluation=False,
                target_mapping="differential_pair_common_window",
                common_window_margin_fraction=0.25,
            ),
            device_model_path=tmp_path / "unread.json",
            conductance_min=0.1,
            conductance_max=1.0,
        )


def test_differential_pair_modifier_rejects_mismatched_conductance_bounds(
    tmp_path: Path,
) -> None:
    plus, _minus = _differential_bindings(((1, 2),))
    minus_parameter = DenseWeight(
        (1,),
        (2,),
        1.0,
        "cpu",
        clamp=True,
        clamp_min=0.1,
        clamp_max=0.9,
    )
    minus = ParameterBinding(
        "base.conductance_minus.0",
        minus_parameter,
        role="conductance_minus",
    )
    with pytest.raises(ValueError, match="exactly matching conductance bounds"):
        IbmReramHwaParameterModifier(
            (plus, minus),
            _config(
                execution="compact_endpoint",
                corruption_policy="counterfactual_repaired",
                noisy_evaluation=False,
                target_mapping="differential_pair_common_window",
                common_window_margin_fraction=0.25,
            ),
            device_model_path=tmp_path / "unread.json",
            conductance_min=0.1,
            conductance_max=1.0,
        )


def test_pulse_resolved_modifier_exports_persistent_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    binding = _binding()
    binding.state.fill_(0.1)
    binding.state.reshape(-1)[1] = 0.55
    population = _population(binding, corruption_policy="published")
    monkeypatch.setattr(
        "training.ibm_reram_hwa.sample_om_array_population",
        lambda *args, **kwargs: population,
    )
    modifier = IbmReramHwaParameterModifier(
        (binding,),
        _config(
            execution="pulse_resolved",
            corruption_policy="published",
            noisy_evaluation=True,
        ),
        device_model_path=_device_model(tmp_path / "device.json"),
        conductance_min=0.1,
        conductance_max=1.0,
    )
    clean = binding.state.clone()
    with modifier.evaluation_context():
        assert torch.equal(binding.state, clean)
    assert torch.equal(binding.state, clean)
    report = modifier.programming_report
    bundle = modifier.last_deployment_bundle
    assert report is not None and report["accepted"] == binding.state.numel()
    assert (
        report["endpoint_application_policy"]
        == IBM_RERAM_ENDPOINT_APPLICATION_POLICY
    )
    assert bundle is not None
    assert bundle["schema"] == "ebl.ibm_reram.om_pulse_resolved_deployment"
    assert (
        bundle["endpoint_application_policy"]
        == IBM_RERAM_ENDPOINT_APPLICATION_POLICY
    )
    assert torch.equal(
        bundle["persistent_endpoint"],
        torch.tensor([0.0, 0.5, 0.0, 0.0]),
    )
    assert torch.equal(
        bundle["apparent_endpoint"],
        torch.tensor([0.0, 0.5, 0.0, 0.0]),
    )
    assert torch.equal(bundle["accepted"], torch.ones(4, dtype=torch.bool))


def test_om_array_population_npz_round_trip_and_fingerprint_gate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    binding = _binding((2, 3))

    def sample_hidden(shape, *, construction_seed, published_corruption):
        size = shape[0] * shape[1]
        offset = float(construction_seed % 17) / 1000.0
        return {
            "max_bound": torch.full((size,), 1.0 - offset),
            "min_bound": torch.full((size,), -1.0 + offset),
            "dwmin_up": torch.full((size,), 0.2 + offset),
            "dwmin_down": torch.full((size,), 0.21 + offset),
            "reference": torch.full((size,), offset),
        }, {
            "aihwkit_version": "1.1.0",
            "nominal_dw_min": 0.0949,
            "dw_min_std": 0.3,
            "write_noise_std": 0.1,
        }

    monkeypatch.setattr(
        "training.ibm_reram_hwa._sample_tile_hidden",
        sample_hidden,
    )
    population = sample_om_array_population(
        (binding,), assignment_seed=83001, corruption_policy="published"
    )
    path = tmp_path / "population.npz"
    save_om_array_population(path, population)
    loaded = load_om_array_population(path)
    assert loaded.fingerprint == population.fingerprint
    assert loaded.binding_keys == population.binding_keys
    assert loaded.binding_shapes == population.binding_shapes
    assert loaded.binding_sampling_seeds == population.binding_sampling_seeds
    assert loaded.donor_sampling_seeds == population.donor_sampling_seeds
    for name, expected in population.tensor_state().items():
        assert torch.equal(loaded.tensor_state()[name], expected)

    with np.load(path, allow_pickle=False) as raw:
        tampered = {name: raw[name].copy() for name in raw.files}
    tampered["fingerprint"] = np.asarray("0" * 64)
    bad = tmp_path / "tampered.npz"
    with bad.open("wb") as handle:
        np.savez_compressed(handle, **tampered)
    with pytest.raises(ValueError, match="recomputed OM population fingerprint"):
        load_om_array_population(bad)
