from types import SimpleNamespace

import pytest
import torch

from experiments.mnist_relu_drn.ibm_om_baseline_selection import quad_contrast
from experiments.mnist_relu_drn.ibm_om_baseline_selection import (
    quad_stack,
    scatter_quads,
)
from experiments.mnist_relu_drn.ibm_om_baseline_spacing_pv_truncated_nominal import (
    TruncatedNominalOmEmbedding,
    build_truncated_nominal_baseline_spacing_mapping,
    project_shared_destination_reset_baselines,
)
from experiments.mnist_relu_drn import (
    ibm_om_baseline_spacing_pv_truncated_nominal_runtime as runtime,
)
from training.ibm_reram_hwa import IbmReramArrayPopulation


def _population() -> IbmReramArrayPopulation:
    return IbmReramArrayPopulation(
        assignment_seed=17,
        corruption_policy="counterfactual_repaired",
        binding_keys=("layer0", "layer1"),
        binding_shapes=((1, 2), (1, 2)),
        binding_sampling_seeds=(101, 102),
        donor_sampling_seeds=(201, 202),
        nominal_dw_min=0.1,
        dw_min_std=0.0,
        write_noise_std=0.0,
        min_bound=torch.tensor([-2.0, -0.8, -1.5, -0.2]),
        max_bound=torch.tensor([2.0, 2.0, 0.8, 1.7]),
        dwmin_up=torch.full((4,), 0.1),
        dwmin_down=torch.full((4,), 0.1),
        reference=torch.zeros(4),
        corrupt=torch.zeros(4, dtype=torch.bool),
        published_corrupt=torch.zeros(4, dtype=torch.bool),
        fingerprint="source-population",
        aihwkit_version="1.1.0",
    )


def test_nominal_embedding_is_direct_G_equals_a_plus_one_without_clipping() -> None:
    embedding = TruncatedNominalOmEmbedding()
    assert torch.equal(
        embedding.to_full_conductance(torch.tensor([0.0, 0.25, 1.0])),
        torch.tensor([0.0, 0.5, 2.0]),
    )
    with pytest.raises(ValueError, match="left x in"):
        embedding.to_full_conductance(torch.tensor([-1e-8, 0.5]))
    with pytest.raises(ValueError, match="left x in"):
        embedding.to_full_conductance(torch.tensor([0.5, 1.000001]))


def test_population_bounds_are_winsorized_before_fresh_reset_commissioning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _population()
    monkeypatch.setattr(runtime, "_source_joint_population", lambda _hardware: source)

    native = runtime._native_assignment_from_joint_artifact(
        {"hardware_instance_id": "joint-hardware"}
    )
    winsorized = native["population"]

    assert torch.equal(
        winsorized.min_bound, torch.tensor([-1.0, -0.8, -1.0, -0.2])
    )
    assert torch.equal(
        winsorized.max_bound, torch.tensor([1.0, 1.0, 0.8, 1.0])
    )
    assert winsorized.fingerprint != source.fingerprint
    assert native["commissioning"].population_fingerprint == winsorized.fingerprint
    assert native["report"]["operation_order"].endswith(
        "winsorize_bounds_then_repeat_RESET_commissioning"
    )
    intervention = native["report"]["winsorization"]
    assert intervention["lower_bound_clipped_count"] == 2
    assert intervention["upper_bound_clipped_count"] == 3
    assert intervention["both_bounds_clipped_count"] == 1
    assert intervention["any_bound_clipped_count"] == 4
    assert intervention["identity_resampled"] is False
    assert all(
        bool(torch.all(reset >= lower)) and bool(torch.all(reset <= upper))
        for reset, lower, upper in zip(
            native["reset_baseline_raw_x"],
            native["cell_lower_raw_x"],
            native["cell_upper_raw_x"],
        )
    )


def test_truncated_mapping_keeps_full_G_and_exact_shared_destination_zero() -> None:
    weights = (
        torch.tensor([[0.8]], dtype=torch.float32),
        torch.tensor([[-0.6]], dtype=torch.float32),
    )
    lower = (
        torch.tensor([[0.0, 0.1], [0.2, 0.0]]),
        torch.tensor([[0.05, 0.15], [0.1, 0.0]]),
    )
    upper = (
        torch.tensor([[0.9, 1.0], [0.8, 0.95]]),
        torch.tensor([[1.0, 0.85], [0.9, 0.95]]),
    )
    reset = tuple(value + 0.1 for value in lower)
    references = tuple(torch.zeros_like(value) for value in lower)

    mapping = build_truncated_nominal_baseline_spacing_mapping(
        weights,
        baseline_position_fraction=0.0,
        spacing_delta_multiples=1,
        scale_fractions=(1.0, 1.0),
        nominal_dw_min=0.1,
        cell_lower_raw_x=lower,
        cell_upper_raw_x=upper,
        reset_baseline_raw_x=reset,
        intrinsic_references_native=references,
    )

    for layer, target_x, target_g in zip(
        mapping.physical.layers,
        mapping.ideal_target_units,
        mapping.ideal_targets,
    ):
        assert torch.count_nonzero(
            quad_contrast(layer.baseline, layout=layer.layout)
        ) == 0
        assert torch.allclose(target_g, 2.0 * target_x, rtol=0.0, atol=2e-6)
        assert bool(torch.all(target_g >= 0.0))
        assert bool(torch.all(target_g <= 2.0))
    assert mapping.report["conductance_embedding"]["reference_device"] == "absent"


def test_requested_reset_max_is_projected_once_into_pair_common_support() -> None:
    lower_quad = torch.tensor([[[0.1, 0.2, 0.3, 0.4]]])
    upper_quad = torch.tensor([[[0.9, 0.8, 0.5, 0.7]]])
    # Both RESET estimates are valid for their own cells, but each pair's max
    # exceeds the other cell's exact upper bound.
    reset_quad = torch.tensor([[[0.85, 0.75, 0.4, 0.6]]])
    shapes = ((2, 2), (2, 2))
    layouts = ("halves", "paired")
    lower = tuple(
        scatter_quads(lower_quad, shape=shape, layout=layout)
        for shape, layout in zip(shapes, layouts)
    )
    upper = tuple(
        scatter_quads(upper_quad, shape=shape, layout=layout)
        for shape, layout in zip(shapes, layouts)
    )
    reset = tuple(
        scatter_quads(reset_quad, shape=shape, layout=layout)
        for shape, layout in zip(shapes, layouts)
    )

    projected, report = project_shared_destination_reset_baselines(
        lower, upper, reset
    )

    expected_quad = torch.tensor([[[0.5, 0.7, 0.5, 0.7]]])
    for value, layout in zip(projected, layouts):
        assert torch.equal(quad_stack(value, layout=layout), expected_quad)
    assert report["projected_pair_count"] == 4
    assert report["destination_pair_count"] == 4
    assert report["absolute_projection_sum_raw_x"] == pytest.approx(0.8)
    assert report["absolute_projection_maximum_raw_x"] == pytest.approx(0.35)
    assert report["persistent_endpoint_projection"] is False

    mapping = build_truncated_nominal_baseline_spacing_mapping(
        (torch.tensor([[1.0]]), torch.tensor([[-1.0]])),
        baseline_position_fraction=0.0,
        spacing_delta_multiples=1,
        scale_fractions=(1.0, 1.0),
        nominal_dw_min=0.1,
        cell_lower_raw_x=lower,
        cell_upper_raw_x=upper,
        reset_baseline_raw_x=reset,
        intrinsic_references_native=tuple(torch.zeros_like(value) for value in lower),
    )
    assert mapping.report["shared_destination_baseline_projection"][
        "projected_pair_count"
    ] == 4
    for layer in mapping.physical.layers:
        assert torch.count_nonzero(
            quad_contrast(layer.baseline, layout=layer.layout)
        ) == 0


def test_empty_destination_common_support_fails_before_mapping() -> None:
    lower = (torch.tensor([[0.1, 0.8], [0.7, 0.2]]),) * 2
    upper = (torch.tensor([[0.6, 0.9], [0.75, 0.7]]),) * 2
    reset = (torch.tensor([[0.2, 0.85], [0.72, 0.3]]),) * 2
    with pytest.raises(ValueError, match="empty exact common support"):
        project_shared_destination_reset_baselines(lower, upper, reset)


def test_persistent_handoff_uses_winsorized_endpoint_without_projection() -> None:
    population = _population()
    # This helper only needs layout metadata; use endpoints inside [0,1].
    outcome = SimpleNamespace(
        persistent_endpoint_unit=torch.tensor([0.0, 0.25, 0.75, 1.0]),
        raw_lower_unit=torch.zeros(4),
        raw_upper_unit=torch.ones(4),
        coordinate="x=(native_raw_active_a+1)/2",
    )
    layers, report = runtime._persistent_affine_handoff(
        outcome, population, TruncatedNominalOmEmbedding()
    )
    assert torch.equal(layers[0], torch.tensor([[0.0, 0.5]]))
    assert torch.equal(layers[1], torch.tensor([[1.5, 2.0]]))
    assert report["projected"] == 0
    assert report["formula"] == "G=a_truncated+1=2*x"
