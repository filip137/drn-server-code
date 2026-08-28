from __future__ import annotations

from dataclasses import replace
import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from experiments.mnist_relu_drn.ibm_om_four_reference_balance import (
    BALANCED_POLICY,
    MAPPING_SCHEMA,
    MAPPING_SCHEMA_VERSION,
    PLAN_SCHEMA,
    PLAN_SCHEMA_VERSION,
    RANDOM_POLICY,
    build_continuous_reference_targets,
    build_reference_binding_plan,
    save_continuous_mapping,
    save_reference_binding_plan,
)
from experiments.mnist_relu_drn.ibm_om_ideal_mapping_scheme_screen import (
    _quad_contrast,
)
from training.ibm_reram_hwa import IbmReramArrayPopulation


_KEYS = ("base.dense_weight.0", "base.dense_weight.1")
_LAYOUTS = ("halves", "paired")


def _native(unit: torch.Tensor) -> torch.Tensor:
    return (2.0 * unit - 1.0).to(torch.float32)


def _scatter_quads(
    quads: torch.Tensor,
    *,
    shape: tuple[int, int],
    layout: str,
) -> torch.Tensor:
    rows, columns = shape
    logical_rows = rows // 2
    logical_columns = columns // 2
    plus = (
        torch.arange(logical_columns)
        if layout == "halves"
        else 2 * torch.arange(logical_columns)
    )
    minus = plus + logical_columns if layout == "halves" else plus + 1
    result = torch.empty(shape, dtype=quads.dtype)
    grid = quads.reshape(logical_rows, logical_columns, 4)
    result[:logical_rows][:, plus] = grid[..., 0]
    result[:logical_rows][:, minus] = grid[..., 1]
    result[logical_rows:][:, plus] = grid[..., 2]
    result[logical_rows:][:, minus] = grid[..., 3]
    return result


def _population(
    *,
    assignment_seed: int = 17,
    shapes: tuple[tuple[int, int], ...] = ((4, 4), (4, 4)),
    reference_unit: tuple[torch.Tensor, ...] | None = None,
    lower_unit: tuple[torch.Tensor, ...] | None = None,
    upper_unit: tuple[torch.Tensor, ...] | None = None,
    fingerprint: str | None = None,
) -> IbmReramArrayPopulation:
    if reference_unit is None:
        reference_unit = tuple(
            torch.linspace(0.125, 0.875, math.prod(shape)).reshape(shape)
            for shape in shapes
        )
    if lower_unit is None:
        lower_unit = tuple(torch.zeros(shape) for shape in shapes)
    if upper_unit is None:
        upper_unit = tuple(torch.ones(shape) for shape in shapes)
    if not (
        len(shapes)
        == len(reference_unit)
        == len(lower_unit)
        == len(upper_unit)
    ):
        raise ValueError("Test population fields must have matching bindings.")
    size = sum(math.prod(shape) for shape in shapes)
    reference = torch.cat([value.reshape(-1) for value in reference_unit])
    lower = torch.cat([value.reshape(-1) for value in lower_unit])
    upper = torch.cat([value.reshape(-1) for value in upper_unit])
    return IbmReramArrayPopulation(
        assignment_seed=assignment_seed,
        corruption_policy="counterfactual_repaired",
        binding_keys=_KEYS[: len(shapes)],
        binding_shapes=shapes,
        binding_sampling_seeds=tuple(range(100, 100 + len(shapes))),
        donor_sampling_seeds=tuple(range(200, 200 + len(shapes))),
        nominal_dw_min=0.1,
        dw_min_std=0.4,
        write_noise_std=1.0,
        max_bound=_native(upper),
        min_bound=_native(lower),
        dwmin_up=torch.full((size,), 0.1, dtype=torch.float32),
        dwmin_down=torch.full((size,), 0.1, dtype=torch.float32),
        reference=_native(reference),
        corrupt=torch.zeros(size, dtype=torch.bool),
        published_corrupt=torch.zeros(size, dtype=torch.bool),
        fingerprint=(
            fingerprint
            if fingerprint is not None
            else f"test-four-reference-{assignment_seed}-{shapes!r}"
        ),
        aihwkit_version="1.1.0",
    )


def _imbalanced_reference(shape: tuple[int, int], layout: str) -> torch.Tensor:
    quads = torch.tensor(
        (
            (0.10, 0.90, 0.80, 0.20),
            (0.11, 0.91, 0.81, 0.21),
            (0.12, 0.92, 0.82, 0.22),
            (0.13, 0.93, 0.83, 0.23),
        ),
        dtype=torch.float32,
    )
    return _scatter_quads(quads, shape=shape, layout=layout)


def _binding_slices(population: IbmReramArrayPopulation) -> tuple[slice, ...]:
    result = []
    offset = 0
    for shape in population.binding_shapes:
        count = math.prod(shape)
        result.append(slice(offset, offset + count))
        offset += count
    return tuple(result)


def test_balanced_plan_is_canonical_bijective_and_within_each_layer() -> None:
    population = _population()
    plan = build_reference_binding_plan(population, policy=BALANCED_POLICY)

    assert plan.policy == BALANCED_POLICY
    assert plan.destination_to_source.dtype == torch.int64
    assert plan.destination_to_source.device.type == "cpu"
    assert torch.equal(
        torch.sort(plan.destination_to_source).values,
        torch.arange(population.size, dtype=torch.int64),
    )
    for binding_slice in _binding_slices(population):
        start = int(binding_slice.start)
        stop = int(binding_slice.stop)
        assert torch.equal(
            torch.sort(plan.destination_to_source[binding_slice]).values,
            torch.arange(start, stop, dtype=torch.int64),
        )

    assert plan.report["schema"] == PLAN_SCHEMA
    assert plan.report["schema_version"] == PLAN_SCHEMA_VERSION
    assert plan.report["uses_weights"] is False
    assert plan.report["uses_labels"] is False
    assert [layer["layout"] for layer in plan.report["layers"]] == list(
        _LAYOUTS
    )
    assert [layer["shape"] for layer in plan.report["layers"]] == [
        [4, 4],
        [4, 4],
    ]


def test_balanced_plan_preserves_reference_multisets_and_is_deterministic() -> None:
    population = _population()
    first = build_reference_binding_plan(population, policy=BALANCED_POLICY)
    second = build_reference_binding_plan(population, policy=BALANCED_POLICY)

    assert torch.equal(first.destination_to_source, second.destination_to_source)
    assert first.report == second.report
    mapped_reference = torch.clamp((population.reference + 1.0) / 2.0, 0.0, 1.0)
    for binding_slice, layer in zip(
        _binding_slices(population), first.report["layers"]
    ):
        source = mapped_reference[binding_slice]
        realized = mapped_reference[first.destination_to_source[binding_slice]]
        torch.testing.assert_close(
            torch.sort(realized).values,
            torch.sort(source).values,
            rtol=0.0,
            atol=0.0,
        )
        assert layer["reference_multiset_preserved"] is True


def test_balanced_quad_address_shuffle_is_assignment_seed_sensitive() -> None:
    first_population = _population(assignment_seed=17)
    second_population = _population(assignment_seed=18)
    first = build_reference_binding_plan(
        first_population, policy=BALANCED_POLICY
    )
    second = build_reference_binding_plan(
        second_population, policy=BALANCED_POLICY
    )

    assert not torch.equal(first.destination_to_source, second.destination_to_source)
    assert [layer["quad_shuffle_seed"] for layer in first.report["layers"]] != [
        layer["quad_shuffle_seed"] for layer in second.report["layers"]
    ]


def test_known_heterogeneous_reference_imbalance_is_reduced() -> None:
    shapes = ((4, 4), (4, 4))
    references = tuple(
        _imbalanced_reference(shape, layout)
        for shape, layout in zip(shapes, _LAYOUTS)
    )
    population = _population(shapes=shapes, reference_unit=references)
    plan = build_reference_binding_plan(population, policy=BALANCED_POLICY)

    for layer in plan.report["layers"]:
        before = layer["baseline_contrast_before"]["rms"]
        after = layer["baseline_contrast_after"]["rms"]
        assert before > 0.25
        assert after < 0.02
        assert after < before
        assert layer["baseline_contrast_rms_ratio_after_over_before"] < 0.1


def _heterogeneous_two_by_two_population(
    *,
    assignment_seed: int = 17,
    upper: torch.Tensor | None = None,
) -> IbmReramArrayPopulation:
    reference = torch.tensor(
        ((0.125, 0.25), (0.375, 0.625)), dtype=torch.float32
    )
    upper_value = torch.ones((2, 2)) if upper is None else upper
    return _population(
        assignment_seed=assignment_seed,
        shapes=((2, 2), (2, 2)),
        reference_unit=(reference, reference.clone()),
        upper_unit=(upper_value, upper_value.clone()),
    )


def test_continuous_mapping_keeps_explicit_g_equals_baseline_plus_offset() -> None:
    population = _heterogeneous_two_by_two_population()
    plan = build_reference_binding_plan(population, policy=RANDOM_POLICY)
    targets, report, components = build_continuous_reference_targets(
        (torch.tensor([[1.0]]), torch.tensor([[1.0]])),
        population,
        plan,
        scale_fractions=(0.5, 0.5),
        conductance_min=0.0,
        conductance_max=1.0,
    )

    expected_baseline = torch.tensor(
        ((0.125, 0.25), (0.375, 0.625)), dtype=torch.float32
    )
    expected_offset = torch.tensor(
        ((0.1875, 0.0), (0.0, 0.1875)), dtype=torch.float32
    )
    expected_conductance = expected_baseline + expected_offset
    for target, layer in zip(targets, components):
        torch.testing.assert_close(layer["baseline"], expected_baseline)
        torch.testing.assert_close(layer["offset"], expected_offset)
        torch.testing.assert_close(layer["conductance"], expected_conductance)
        torch.testing.assert_close(
            layer["conductance"],
            layer["baseline"] + layer["offset"],
            rtol=0.0,
            atol=0.0,
        )
        torch.testing.assert_close(target, layer["conductance"])

    assert report["decomposition"] == "every physical branch G=B+d"
    assert report["circuit_accounting"] == (
        "full G enters numerator and denominator"
    )
    assert all(
        layer["full_g_decomposition_max_abs_residual"] == 0.0
        for layer in report["layers"]
    )


def test_full_contrast_and_loading_retain_heterogeneous_baseline() -> None:
    population = _heterogeneous_two_by_two_population()
    plan = build_reference_binding_plan(population, policy=RANDOM_POLICY)
    _targets, report, components = build_continuous_reference_targets(
        (torch.tensor([[1.0]]), torch.tensor([[1.0]])),
        population,
        plan,
        scale_fractions=(0.5, 0.5),
        conductance_min=0.0,
        conductance_max=1.0,
    )

    for layout, layer_report, layer in zip(
        _LAYOUTS, report["layers"], components
    ):
        manual_baseline_contrast = _quad_contrast(
            layer["baseline"], layout=layout
        ) / 2.0
        manual_offset_contrast = _quad_contrast(
            layer["offset"], layout=layout
        ) / 2.0
        manual_realized_contrast = _quad_contrast(
            layer["conductance"], layout=layout
        ) / 2.0
        torch.testing.assert_close(
            layer["baseline_contrast"], manual_baseline_contrast
        )
        torch.testing.assert_close(
            layer["offset_contrast"], manual_offset_contrast
        )
        torch.testing.assert_close(
            layer["realized_contrast"], manual_realized_contrast
        )
        torch.testing.assert_close(
            layer["realized_contrast"],
            layer["baseline_contrast"] + layer["offset_contrast"],
        )
        torch.testing.assert_close(
            layer["baseline_contrast"], torch.tensor([[0.0625]])
        )
        torch.testing.assert_close(
            layer["offset_contrast"], torch.tensor([[0.1875]])
        )
        torch.testing.assert_close(
            layer["realized_contrast"], torch.tensor([[0.25]])
        )
        torch.testing.assert_close(layer["loading"], torch.tensor([[1.75]]))
        assert layer_report["baseline_contrast"]["rms"] == pytest.approx(
            0.0625
        )
        assert layer_report["quad_loading"]["mean"] == pytest.approx(1.75)


def test_continuous_active_targets_stay_inside_heterogeneous_bounds() -> None:
    upper = torch.tensor(((0.5, 0.9), (0.8, 0.75)), dtype=torch.float32)
    population = _heterogeneous_two_by_two_population(upper=upper)
    plan = build_reference_binding_plan(population, policy=RANDOM_POLICY)
    _targets, report, components = build_continuous_reference_targets(
        (torch.tensor([[1.0]]), torch.tensor([[-1.0]])),
        population,
        plan,
        scale_fractions=(1.0, 1.0),
        conductance_min=0.0,
        conductance_max=1.0,
    )

    for layer_report, layer in zip(report["layers"], components):
        active = layer["active_mask"]
        assert torch.all(layer["conductance"][active] >= 0.0)
        assert torch.all(layer["conductance"][active] <= upper[active])
        assert layer_report["active_bound_violation_count"] == 0
        assert layer_report["finite_conductance"] is True
        assert layer_report["nonnegative_conductance"] is True


def test_float32_deployed_state_stays_strictly_inside_native_bounds() -> None:
    upper = torch.full((2, 2), 0.500001, dtype=torch.float32)
    population = _heterogeneous_two_by_two_population(upper=upper)
    plan = build_reference_binding_plan(population, policy=RANDOM_POLICY)
    _targets, report, components = build_continuous_reference_targets(
        (torch.tensor([[1.0]]), torch.tensor([[-1.0]])),
        population,
        plan,
        scale_fractions=(1.0, 1.0),
        conductance_min=0.0,
        conductance_max=0.00011,
    )

    physical_upper = (
        0.00011
        * torch.clamp(
            (population.max_bound.to(torch.float64) + 1.0) / 2.0,
            0.0,
            1.0,
        )
    )
    offset = 0
    inward_adjustments = 0
    for layer_report, layer, shape in zip(
        report["layers"], components, population.binding_shapes
    ):
        count = math.prod(shape)
        assigned_upper = physical_upper[
            plan.destination_to_source[offset : offset + count]
        ].reshape(shape)
        active = layer["active_mask"]
        assert torch.all(
            layer["conductance"][active].to(torch.float64)
            <= assigned_upper[active]
        )
        assert torch.equal(
            layer["conductance"], layer["baseline"] + layer["offset"]
        )
        assert layer_report["active_bound_violation_count"] == 0
        inward_adjustments += (
            layer_report["float32_inward_baseline_adjustment_count"]
            + layer_report["float32_inward_conductance_adjustment_count"]
        )
        assert (
            layer_report["float32_inward_conductance_max_abs_adjustment"]
            >= 0.0
        )
        offset += count
    assert inward_adjustments > 0


def test_continuous_mapping_declares_that_no_level_quantization_was_used() -> None:
    population = _heterogeneous_two_by_two_population()
    plan = build_reference_binding_plan(population, policy=RANDOM_POLICY)
    _targets, report, _components = build_continuous_reference_targets(
        (torch.tensor([[1.0]]), torch.tensor([[1.0]])),
        population,
        plan,
        scale_fractions=(1.0, 1.0),
        conductance_min=0.0,
        conductance_max=1.0,
    )

    for layer in report["layers"]:
        assert layer["mapping"] == "continuous_positive_only_local_common_headroom"
        assert layer["level_rounding"] is False
        assert layer["level_spacing"] is None
        assert layer["level_count_cap"] is None
        assert not any("pulse_index" in key for key in layer)


def test_plan_rejects_malformed_policy_layout_and_population() -> None:
    population = _population()
    with pytest.raises(ValueError, match="binding policy"):
        build_reference_binding_plan(population, policy="unknown")
    with pytest.raises(ValueError, match="canonical W1-halves/W2-paired"):
        build_reference_binding_plan(
            population, policy=BALANCED_POLICY, layouts=("paired", "halves")
        )

    odd_population = _population(shapes=((2, 3), (2, 2)))
    with pytest.raises(
        ValueError, match="multiple of four|even dual-rail matrix"
    ):
        build_reference_binding_plan(odd_population, policy=BALANCED_POLICY)

    with pytest.raises(ValueError, match="reference"):
        replace(
            population,
            reference=torch.full_like(population.reference, float("nan")),
        )


def test_mapping_rejects_plan_population_mismatch_and_invalid_controls() -> None:
    population = _heterogeneous_two_by_two_population()
    plan = build_reference_binding_plan(population, policy=RANDOM_POLICY)
    mismatched = replace(population, fingerprint="different-population")
    with pytest.raises(ValueError, match="provenance"):
        build_continuous_reference_targets(
            (torch.tensor([[1.0]]), torch.tensor([[1.0]])),
            mismatched,
            plan,
            scale_fractions=(1.0, 1.0),
            conductance_min=0.0,
            conductance_max=1.0,
        )
    with pytest.raises(ValueError, match="scale fraction"):
        build_continuous_reference_targets(
            (torch.tensor([[1.0]]), torch.tensor([[1.0]])),
            population,
            plan,
            scale_fractions=(0.0, 1.0),
            conductance_min=0.0,
            conductance_max=1.0,
        )
    with pytest.raises(ValueError, match="increasing conductance bounds"):
        build_continuous_reference_targets(
            (torch.tensor([[1.0]]), torch.tensor([[1.0]])),
            population,
            plan,
            scale_fractions=(1.0, 1.0),
            conductance_min=1.0,
            conductance_max=1.0,
        )


def test_plan_artifact_is_pickle_free_and_records_exact_structure(
    tmp_path: Path,
) -> None:
    population = _population()
    plan = build_reference_binding_plan(population, policy=BALANCED_POLICY)
    record = save_reference_binding_plan(tmp_path, population, plan)

    with np.load(record["path"], allow_pickle=False) as artifact:
        assert set(artifact.files) == {
            "schema",
            "schema_version",
            "policy",
            "assignment_seed",
            "population_fingerprint",
            "binding_keys_json",
            "binding_shapes_json",
            "destination_to_source",
        }
        assert artifact["schema"].item() == PLAN_SCHEMA
        assert artifact["schema_version"].item() == PLAN_SCHEMA_VERSION
        assert artifact["schema_version"].dtype == np.dtype(np.int64)
        assert artifact["destination_to_source"].dtype == np.dtype(np.int64)
        np.testing.assert_array_equal(
            artifact["destination_to_source"], plan.destination_to_source.numpy()
        )
    receipt = json.loads(Path(record["receipt"]).read_text(encoding="utf-8"))
    assert receipt["schema"] == PLAN_SCHEMA
    assert receipt["schema_version"] == PLAN_SCHEMA_VERSION
    assert receipt["artifact_sha256"] == record["sha256"]
    assert receipt["destination_to_source_sha256"] == record[
        "destination_to_source_sha256"
    ]


def test_mapping_artifact_records_only_full_continuous_components(
    tmp_path: Path,
) -> None:
    population = _heterogeneous_two_by_two_population()
    plan = build_reference_binding_plan(population, policy=RANDOM_POLICY)
    _targets, report, components = build_continuous_reference_targets(
        (torch.tensor([[1.0]]), torch.tensor([[1.0]])),
        population,
        plan,
        scale_fractions=(0.5, 0.5),
        conductance_min=0.0,
        conductance_max=1.0,
    )
    record = save_continuous_mapping(
        tmp_path,
        assignment_seed=population.assignment_seed,
        binding_policy=RANDOM_POLICY,
        calibration_policy=RANDOM_POLICY,
        mapping_report=report,
        components=components,
    )

    component_names = {
        "baseline",
        "offset",
        "conductance",
        "baseline_contrast",
        "offset_contrast",
        "realized_contrast",
        "loading",
        "active_mask",
        "common_headroom",
    }
    expected = {
        "schema",
        "schema_version",
        "assignment_seed",
        "binding_policy",
        "calibration_policy",
    } | {
        f"layer_{layer}_{name}"
        for layer in range(2)
        for name in component_names
    }
    with np.load(record["path"], allow_pickle=False) as artifact:
        assert set(artifact.files) == expected
        assert artifact["schema"].item() == MAPPING_SCHEMA
        assert artifact["schema_version"].item() == MAPPING_SCHEMA_VERSION
        assert artifact["schema_version"].dtype == np.dtype(np.int64)
        assert not any(
            "level" in name or "quant" in name or "pulse" in name
            for name in artifact.files
        )
        for layer in range(2):
            np.testing.assert_array_equal(
                artifact[f"layer_{layer}_conductance"],
                artifact[f"layer_{layer}_baseline"]
                + artifact[f"layer_{layer}_offset"],
            )
    receipt = json.loads(Path(record["receipt"]).read_text(encoding="utf-8"))
    assert receipt["schema"] == MAPPING_SCHEMA
    assert receipt["schema_version"] == MAPPING_SCHEMA_VERSION
    assert receipt["artifact_sha256"] == record["sha256"]
    assert receipt["mapping"]["decomposition"] == "every physical branch G=B+d"
