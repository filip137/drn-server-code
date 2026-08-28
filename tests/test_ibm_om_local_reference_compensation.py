from __future__ import annotations

from dataclasses import replace
import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from experiments.mnist_relu_drn.ibm_om_bounded_codebook_scheme_screen import (
    _tensor_sha256,
)
from experiments.mnist_relu_drn.ibm_om_local_reference_compensation import (
    COMPENSATED_POLICY,
    CONTROL_POLICY,
    MAPPING_SCHEMA,
    MAPPING_SCHEMA_VERSION,
    PLAN_SCHEMA,
    PLAN_SCHEMA_VERSION,
    build_local_baseline_plan,
    build_matched_continuous_targets,
    project_local_exact_zero,
    save_continuous_mapping,
    save_local_baseline_plan,
)
from training.ibm_reram_hwa import IbmReramArrayPopulation


_KEYS = ("base.dense_weight.0", "base.dense_weight.1")


def _native(unit: torch.Tensor) -> torch.Tensor:
    return (2.0 * unit - 1.0).to(torch.float32)


def _population(
    *,
    assignment_seed: int = 91,
    reference_unit: tuple[torch.Tensor, torch.Tensor] | None = None,
    lower_unit: tuple[torch.Tensor, torch.Tensor] | None = None,
    upper_unit: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> IbmReramArrayPopulation:
    shapes = ((2, 2), (2, 2))
    if reference_unit is None:
        reference_unit = (
            torch.tensor(((0.10, 0.80), (0.70, 0.20))),
            torch.tensor(((0.15, 0.75), (0.65, 0.25))),
        )
    if lower_unit is None:
        lower_unit = tuple(torch.zeros(shape) for shape in shapes)
    if upper_unit is None:
        upper_unit = tuple(torch.ones(shape) for shape in shapes)
    size = sum(math.prod(shape) for shape in shapes)
    reference = torch.cat([value.reshape(-1) for value in reference_unit])
    lower = torch.cat([value.reshape(-1) for value in lower_unit])
    upper = torch.cat([value.reshape(-1) for value in upper_unit])
    return IbmReramArrayPopulation(
        assignment_seed=assignment_seed,
        corruption_policy="counterfactual_repaired",
        binding_keys=_KEYS,
        binding_shapes=shapes,
        binding_sampling_seeds=(101, 102),
        donor_sampling_seeds=(201, 202),
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
        fingerprint=f"local-compensation-test-{assignment_seed}",
        aihwkit_version="1.1.0",
    )


def test_unconstrained_checkerboard_projection_is_exact_and_load_preserving() -> None:
    reference = torch.tensor([[[0.10, 0.80, 0.70, 0.20]]], dtype=torch.float64)
    lower = torch.zeros_like(reference)
    upper = torch.ones_like(reference)

    baseline, diagnostics = project_local_exact_zero(reference, lower, upper)

    expected = torch.tensor([[[0.40, 0.50, 0.40, 0.50]]], dtype=torch.float64)
    torch.testing.assert_close(baseline, expected, rtol=0.0, atol=2e-16)
    assert diagnostics["unconstrained_in_bounds"].item() is True
    assert diagnostics["bound_active_quad"].item() is False
    assert diagnostics["zero_residual"].abs().max().item() <= 2e-16
    # The analytic checkerboard correction leaves both row sums, both column
    # sums, and the total zero-state loading unchanged.
    torch.testing.assert_close(
        baseline[..., 0] + baseline[..., 1],
        reference[..., 0] + reference[..., 1],
    )
    torch.testing.assert_close(
        baseline[..., 2] + baseline[..., 3],
        reference[..., 2] + reference[..., 3],
    )
    torch.testing.assert_close(
        baseline[..., 0] + baseline[..., 2],
        reference[..., 0] + reference[..., 2],
    )
    torch.testing.assert_close(
        baseline[..., 1] + baseline[..., 3],
        reference[..., 1] + reference[..., 3],
    )
    torch.testing.assert_close(baseline.sum(dim=-1), reference.sum(dim=-1))


def test_bound_active_projection_is_the_expected_min_l2_solution() -> None:
    reference = torch.tensor([[[0.90, 0.10, 0.10, 0.90]]], dtype=torch.float64)
    lower = torch.zeros_like(reference)
    upper = torch.ones_like(reference)
    upper[..., 0] = 0.20

    baseline, diagnostics = project_local_exact_zero(reference, lower, upper)

    expected = torch.tensor([[[0.20, 0.40, 0.40, 0.60]]], dtype=torch.float64)
    torch.testing.assert_close(baseline, expected, rtol=0.0, atol=5e-15)
    assert diagnostics["bound_active_quad"].item() is True
    assert diagnostics["at_upper"][..., 0].item() is True
    assert diagnostics["zero_residual"].abs().max().item() <= 2e-16
    alternative = torch.tensor([[[0.20, 0.45, 0.35, 0.60]]], dtype=torch.float64)
    assert torch.sum((baseline - reference) ** 2) < torch.sum(
        (alternative - reference) ** 2
    )


def test_projection_fails_closed_when_exact_zero_is_outside_the_box() -> None:
    reference = torch.full((1, 1, 4), 0.5, dtype=torch.float64)
    lower = torch.tensor([[[0.0, 1.0, 1.0, 0.0]]], dtype=torch.float64)
    upper = lower.clone()

    with pytest.raises(ValueError, match="infeasible"):
        project_local_exact_zero(reference, lower, upper)


def test_projection_randomized_kkt_bounds_and_zero_invariants() -> None:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(8173)
    reference = torch.rand((2048, 4), generator=generator, dtype=torch.float64)
    lower = 0.2 * torch.rand(
        (2048, 4), generator=generator, dtype=torch.float64
    )
    upper = 0.8 + 0.2 * torch.rand(
        (2048, 4), generator=generator, dtype=torch.float64
    )

    baseline, diagnostics = project_local_exact_zero(reference, lower, upper)

    sign = torch.tensor((1.0, -1.0, -1.0, 1.0), dtype=torch.float64)
    kkt_clipped = torch.minimum(
        torch.maximum(
            reference - diagnostics["multiplier"].unsqueeze(-1) * sign,
            lower,
        ),
        upper,
    )
    assert torch.all(baseline >= lower)
    assert torch.all(baseline <= upper)
    assert torch.max(torch.abs((baseline * sign).sum(dim=-1))).item() <= 1e-12
    torch.testing.assert_close(baseline, kkt_clipped, rtol=0.0, atol=2e-15)


def test_plans_keep_sampled_identity_order_and_intrinsic_reference_immutable() -> None:
    population = _population()
    original_reference = population.reference.clone()

    control = build_local_baseline_plan(population, policy=CONTROL_POLICY)
    treatment = build_local_baseline_plan(population, policy=COMPENSATED_POLICY)

    assert torch.equal(population.reference, original_reference)
    for plan in (control, treatment):
        assert plan.assignment_seed == population.assignment_seed
        assert plan.population_fingerprint == population.fingerprint
        assert plan.report["identity_binding"] == "sampled_order_no_permutation"
        assert plan.report["intrinsic_reference_is_immutable"] is True
        assert all(
            layer["identity_binding"] == "sampled_order_no_permutation"
            for layer in plan.report["layers"]
        )
    mapped = ((population.reference.to(torch.float64) + 1.0) / 2.0).clamp(0.0, 1.0)
    torch.testing.assert_close(control.baseline_unit, mapped, rtol=0.0, atol=0.0)
    assert all(
        layer["exact_zero_max_abs_residual"] <= 1e-12
        for layer in treatment.report["layers"]
    )
    assert treatment.report["layers"][0][
        "loading_change_from_intrinsic_reference"
    ]["total"]["absolute"]["maximum"] <= 1e-12


def test_control_clips_only_the_stored_baseline_not_intrinsic_r() -> None:
    reference = (
        torch.tensor(((0.10, 0.80), (0.70, 0.20))),
        torch.tensor(((0.10, 0.80), (0.70, 0.20))),
    )
    lower = (
        torch.tensor(((0.20, 0.00), (0.00, 0.00))),
        torch.tensor(((0.20, 0.00), (0.00, 0.00))),
    )
    upper = tuple(torch.ones((2, 2)) for _ in range(2))
    population = _population(
        reference_unit=reference, lower_unit=lower, upper_unit=upper
    )
    raw = population.reference.clone()

    control = build_local_baseline_plan(population, policy=CONTROL_POLICY)

    assert control.components[0]["baseline"][0, 0].item() == pytest.approx(0.2)
    assert control.components[0]["intrinsic_reference"][0, 0].item() == pytest.approx(
        0.1
    )
    assert torch.equal(population.reference, raw)
    assert control.report["layers"][0]["reference_outside_active_bounds_count"] == 1


def test_matched_mapping_uses_identical_d_and_full_g_in_both_arms() -> None:
    population = _population()
    plans = {
        CONTROL_POLICY: build_local_baseline_plan(
            population, policy=CONTROL_POLICY
        ),
        COMPENSATED_POLICY: build_local_baseline_plan(
            population, policy=COMPENSATED_POLICY
        ),
    }
    logical = (torch.tensor([[1.0]]), torch.tensor([[-1.0]]))

    matched = build_matched_continuous_targets(
        logical,
        population,
        plans,
        scale_fractions=(0.5, 0.5),
        conductance_min=0.0,
        conductance_max=1.0,
    )

    assert matched.shared_report["offset_hashes_equal"] is True
    for layer in range(2):
        control = matched.components_by_policy[CONTROL_POLICY][layer]
        treatment = matched.components_by_policy[COMPENSATED_POLICY][layer]
        assert torch.equal(control["offset"], treatment["offset"])
        for component in (control, treatment):
            assert torch.equal(
                component["conductance"],
                component["baseline"] + component["offset"],
            )
            assert torch.equal(
                component["loading"],
                component["conductance"].sum().reshape(1, 1),
            )
        # Positive W1 raises ++/--; negative W2 raises +-/ -+.
        active = control["active_mask"]
        expected = (
            torch.tensor(((True, False), (False, True)))
            if layer == 0
            else torch.tensor(((False, True), (True, False)))
        )
        assert torch.equal(active, expected)
    for policy in (CONTROL_POLICY, COMPENSATED_POLICY):
        assert matched.reports_by_policy[policy]["decomposition"] == (
            "every physical branch G=B+d"
        )
        assert matched.reports_by_policy[policy]["circuit_accounting"] == (
            "full G enters numerator and denominator"
        )
        assert all(
            layer["bound_violation_count"] == 0
            for layer in matched.reports_by_policy[policy]["layers"]
        )


def test_policy_selected_mapping_has_prior_style_return_and_shared_receipt() -> None:
    population = _population()
    plans = {
        policy: build_local_baseline_plan(population, policy=policy)
        for policy in (CONTROL_POLICY, COMPENSATED_POLICY)
    }
    targets, report, components = build_matched_continuous_targets(
        (torch.tensor([[1.0]]), torch.tensor([[1.0]])),
        population,
        plans,
        policy=COMPENSATED_POLICY,
        scale_fractions=(0.5, 0.5),
        conductance_min=0.0,
        conductance_max=1.0,
    )

    assert len(targets) == len(components) == 2
    assert report["baseline_policy"] == COMPENSATED_POLICY
    assert report["shared_matching"]["offset_hashes_equal"] is True


@pytest.mark.parametrize(
    ("conductance_min", "conductance_max"),
    ((0.0, 1.0), (0.0, 0.0001103), (1.0e-7, 1.0003e-7)),
)
def test_matched_float32_offset_stays_in_both_boxes_for_awkward_spans(
    conductance_min: float,
    conductance_max: float,
) -> None:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(931)
    references = tuple(
        torch.rand((2, 2), generator=generator) for _ in range(2)
    )
    lowers = tuple(
        0.2 * torch.rand((2, 2), generator=generator) for _ in range(2)
    )
    uppers = tuple(
        0.8 + 0.2 * torch.rand((2, 2), generator=generator) for _ in range(2)
    )
    population = _population(
        reference_unit=references,
        lower_unit=lowers,
        upper_unit=uppers,
    )
    plans = {
        policy: build_local_baseline_plan(population, policy=policy)
        for policy in (CONTROL_POLICY, COMPENSATED_POLICY)
    }

    matched = build_matched_continuous_targets(
        (torch.tensor([[1.0]]), torch.tensor([[-1.0]])),
        population,
        plans,
        scale_fractions=(1.0, 1.0),
        conductance_min=conductance_min,
        conductance_max=conductance_max,
    )

    assert matched.shared_report["offset_hashes_equal"] is True
    for layer in range(2):
        assert torch.equal(
            matched.components_by_policy[CONTROL_POLICY][layer]["offset"],
            matched.components_by_policy[COMPENSATED_POLICY][layer]["offset"],
        )
    for policy in (CONTROL_POLICY, COMPENSATED_POLICY):
        assert all(
            layer["bound_violation_count"] == 0
            for layer in matched.reports_by_policy[policy]["layers"]
        )


def test_plan_and_mapping_artifacts_are_pickle_free_and_complete(tmp_path: Path) -> None:
    population = _population()
    plans = {
        policy: build_local_baseline_plan(population, policy=policy)
        for policy in (CONTROL_POLICY, COMPENSATED_POLICY)
    }
    plan_record = save_local_baseline_plan(
        tmp_path / "plans", population, plans[COMPENSATED_POLICY]
    )
    with np.load(plan_record["path"], allow_pickle=False) as artifact:
        assert artifact["schema"].item() == PLAN_SCHEMA
        assert artifact["schema_version"].item() == PLAN_SCHEMA_VERSION
        np.testing.assert_array_equal(
            artifact["identity_order"], np.arange(population.size, dtype=np.int64)
        )
        for name in (
            "raw_intrinsic_reference",
            "intrinsic_reference",
            "lower_bound",
            "upper_bound",
            "baseline",
            "correction",
            "bound_active_quad",
        ):
            assert f"layer_0_{name}" in artifact.files
    plan_receipt = json.loads(
        Path(plan_record["receipt"]).read_text(encoding="utf-8")
    )
    assert plan_receipt["artifact_sha256"] == plan_record["sha256"]
    assert plan_receipt["identity_binding"] == "sampled_order_no_permutation"

    targets, report, components = build_matched_continuous_targets(
        (torch.tensor([[1.0]]), torch.tensor([[1.0]])),
        population,
        plans,
        policy=COMPENSATED_POLICY,
        scale_fractions=(0.5, 0.5),
        conductance_min=0.0,
        conductance_max=1.0,
    )
    assert len(targets) == 2
    mapping_record = save_continuous_mapping(
        tmp_path / "mappings",
        assignment_seed=population.assignment_seed,
        baseline_policy=COMPENSATED_POLICY,
        calibration_policy=CONTROL_POLICY,
        mapping_report=report,
        components=components,
    )
    with np.load(mapping_record["path"], allow_pickle=False) as artifact:
        assert artifact["schema"].item() == MAPPING_SCHEMA
        assert artifact["schema_version"].item() == MAPPING_SCHEMA_VERSION
        for layer in range(2):
            np.testing.assert_array_equal(
                artifact[f"layer_{layer}_conductance"],
                artifact[f"layer_{layer}_baseline"]
                + artifact[f"layer_{layer}_offset"],
            )
            assert report["layers"][layer]["hashes"]["common_headroom"] == (
                _tensor_sha256(
                    torch.from_numpy(
                        artifact[f"layer_{layer}_common_headroom"].copy()
                    )
                )
            )
    mapping_receipt = json.loads(
        Path(mapping_record["receipt"]).read_text(encoding="utf-8")
    )
    assert mapping_receipt["artifact_sha256"] == mapping_record["sha256"]
    assert mapping_receipt["mapping"]["shared_matching"][
        "offset_hashes_equal"
    ] is True


def test_rejects_plan_provenance_policy_and_mapping_controls() -> None:
    population = _population()
    plans = {
        policy: build_local_baseline_plan(population, policy=policy)
        for policy in (CONTROL_POLICY, COMPENSATED_POLICY)
    }
    mismatched = replace(population, fingerprint="different")
    with pytest.raises(ValueError, match="provenance"):
        build_matched_continuous_targets(
            (torch.tensor([[1.0]]), torch.tensor([[1.0]])),
            mismatched,
            plans,
            scale_fractions=(1.0, 1.0),
            conductance_min=0.0,
            conductance_max=1.0,
        )
    with pytest.raises(ValueError, match="scale fraction"):
        build_matched_continuous_targets(
            (torch.tensor([[1.0]]), torch.tensor([[1.0]])),
            population,
            plans,
            scale_fractions=(0.0, 1.0),
            conductance_min=0.0,
            conductance_max=1.0,
        )
    with pytest.raises(ValueError, match="baseline policy"):
        build_local_baseline_plan(population, policy="global_reassignment")
