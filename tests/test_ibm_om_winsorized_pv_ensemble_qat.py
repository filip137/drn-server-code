from dataclasses import replace

import pytest
import torch

from experiments.mnist_relu_drn.ibm_om_winsorized_pv_ensemble_qat import (
    EndpointGradientMixtureWeights,
    PersistentEndpointCodebookEnsemble,
    build_persistent_endpoint_codebook_ensemble,
    gradient_mixture_weights,
    mix_logical_gradients,
)
from experiments.mnist_relu_drn.ibm_om_winsorized_qat import (
    WinsorizedQatLayerTemplate,
)
from training.ibm_reram_hwa import IbmReramArrayPopulation
from training.ibm_reram_raw_active_program_verify import (
    run_raw_active_program_verify,
)


def _strict_table() -> PersistentEndpointCodebookEnsemble:
    maximum = torch.tensor([0, 1, 2, 1, 2], dtype=torch.int64)
    table = torch.full((4, 5, 3), torch.nan, dtype=torch.float32)
    for sample in range(4):
        for cell, capacity in enumerate(maximum.tolist()):
            for level in range(capacity + 1):
                table[sample, cell, level] = (
                    0.05 + 0.1 * sample + 0.01 * cell + 0.02 * level
                )
    return PersistentEndpointCodebookEnsemble(
        assignment_seed=87001,
        population_fingerprint="strict-test-population",
        binding_shapes=((1, 2), (1, 3)),
        endpoint_seeds=(91001, 91002, 91003, 91004),
        spacing_raw_x=0.1,
        tolerance_raw_x=0.05,
        maximum_program_pulses=16,
        baseline_raw_x=torch.full((5,), 0.05),
        maximum_index=maximum,
        persistent_raw_x=table,
    )


def test_strict_table_gathers_current_codes_and_splits_literal_full_g() -> None:
    ensemble = _strict_table()
    requested_layers = (
        torch.tensor([[0, 1]], dtype=torch.int64),
        torch.tensor([[2, 0, 1]], dtype=torch.int64),
    )

    raw = ensemble.lookup_raw_x(
        torch.cat(tuple(value.reshape(-1) for value in requested_layers)),
        sample_indices=(1, 3),
    )
    views = ensemble.full_conductance_views(
        requested_layers, sample_indices=(1, 3)
    )

    assert raw.shape == (2, 5)
    assert len(views) == 2
    assert tuple(value.shape for value in views[0]) == ((1, 2), (1, 3))
    assert torch.equal(torch.cat(tuple(value.reshape(-1) for value in views[0])), 2.0 * raw[0])
    assert torch.equal(torch.cat(tuple(value.reshape(-1) for value in views[1])), 2.0 * raw[1])
    # This is multiplication only: zero and every non-boundary value are kept.
    zero_table = ensemble.persistent_raw_x.clone()
    zero_table[0, 0, 0] = 0.0
    zero_ensemble = replace(ensemble, persistent_raw_x=zero_table)
    zero_full = zero_ensemble.lookup_full_conductance(
        torch.zeros(5, dtype=torch.int64), sample_indices=(0,)
    )
    assert zero_full[0, 0].item() == 0.0


def test_strict_table_rejects_unsupported_codes_and_nonphysical_entries() -> None:
    ensemble = _strict_table()
    unsupported = torch.tensor([1, 0, 0, 0, 0], dtype=torch.int64)
    with pytest.raises(ValueError, match="exceeds commissioned support"):
        ensemble.lookup_raw_x(unsupported)

    finite_invalid = ensemble.persistent_raw_x.clone()
    finite_invalid[0, 0, 1] = 0.2
    with pytest.raises(ValueError, match="NaN everywhere else"):
        replace(ensemble, persistent_raw_x=finite_invalid)

    clipped_candidate = ensemble.persistent_raw_x.clone()
    clipped_candidate[0, 0, 0] = 1.01
    with pytest.raises(ValueError, match="in-range persistent endpoints"):
        replace(ensemble, persistent_raw_x=clipped_candidate)


def test_gradient_mixture_weights_are_exact_and_tail_ties_are_stable() -> None:
    deterministic = gradient_mixture_weights(
        "deterministic", (), ideal_loss=torch.tensor(2.0)
    )
    mean = gradient_mixture_weights("mean2", (1.0, 3.0), ideal_loss=2.0)
    tail = gradient_mixture_weights("tail4", (1.0, 4.0, 4.0, 2.0), ideal_loss=2.0)

    assert deterministic == EndpointGradientMixtureWeights(
        "deterministic", 1.0, (), None
    )
    assert mean == EndpointGradientMixtureWeights(
        "mean2", 0.25, (0.375, 0.375), None
    )
    # Python max resolves a tie to the first occurrence, making replay stable.
    assert tail.tail_sample_index == 1
    assert tail.ideal_weight == 0.25
    assert tail.persistent_weights == (0.125, 0.375, 0.125, 0.125)
    assert sum(tail.persistent_weights) + tail.ideal_weight == 1.0


def test_gradient_mixture_combines_deterministic_mean2_and_tail4() -> None:
    ideal = (torch.tensor([2.0, 4.0]),)
    persistent = tuple(
        (torch.full((2,), float(value)),) for value in (10, 20, 30, 40)
    )

    deterministic = mix_logical_gradients(
        "deterministic",
        (),
        (),
        ideal_loss=1.0,
        ideal_gradients=ideal,
    )
    mean = mix_logical_gradients(
        "mean2",
        (1.0, 2.0),
        persistent[:2],
        ideal_loss=2.0,
        ideal_gradients=ideal,
    )
    tail = mix_logical_gradients(
        "tail4",
        (1.0, 4.0, 3.0, 2.0),
        persistent,
        ideal_loss=2.0,
        ideal_gradients=ideal,
    )

    assert torch.equal(deterministic[0], ideal[0])
    assert torch.equal(mean[0], torch.tensor([11.75, 12.25]))
    assert torch.equal(tail[0], torch.tensor([18.0, 18.5]))
    assert all(value.grad_fn is None for value in tail)


def _templates_and_population() -> tuple[
    tuple[WinsorizedQatLayerTemplate, WinsorizedQatLayerTemplate],
    IbmReramArrayPopulation,
]:
    shapes = ((2, 2), (2, 2))
    baselines = (
        torch.tensor([[0.20, 0.20], [0.25, 0.25]], dtype=torch.float32),
        torch.tensor([[0.15, 0.15], [0.20, 0.20]], dtype=torch.float32),
    )
    uppers = (
        torch.tensor([[0.50, 0.40], [0.55, 0.45]], dtype=torch.float32),
        torch.tensor([[0.45, 0.35], [0.50, 0.40]], dtype=torch.float32),
    )
    templates = []
    for layer, (shape, baseline, upper, layout) in enumerate(
        zip(shapes, baselines, uppers, ("halves", "paired"))
    ):
        templates.append(
            WinsorizedQatLayerTemplate(
                layer_index=layer,
                layout=layout,
                baseline_full_g=2.0 * baseline,
                baseline_raw_x=baseline,
                cell_upper_raw_x=upper,
                positive_headroom_raw_x=torch.tensor([[0.2]]),
                negative_headroom_raw_x=torch.tensor([[0.2]]),
                positive_capacity=torch.tensor([[2]], dtype=torch.int64),
                negative_capacity=torch.tensor([[2]], dtype=torch.int64),
                initial_normalized_weight=torch.zeros((1, 1)),
                level_spacing_raw_x=0.1,
            )
        )
    upper_flat = torch.cat(tuple(value.reshape(-1) for value in uppers))
    population = IbmReramArrayPopulation(
        assignment_seed=87001,
        corruption_policy="counterfactual_repaired",
        binding_keys=("base.dense_weight.0", "base.dense_weight.1"),
        binding_shapes=shapes,
        binding_sampling_seeds=(12345, 12346),
        donor_sampling_seeds=(54321, 54322),
        nominal_dw_min=0.2,
        dw_min_std=0.15,
        write_noise_std=0.3,
        max_bound=2.0 * upper_flat - 1.0,
        min_bound=torch.full((8,), -0.8, dtype=torch.float32),
        dwmin_up=torch.full((8,), 0.2, dtype=torch.float32),
        dwmin_down=torch.full((8,), 0.2, dtype=torch.float32),
        reference=torch.zeros(8, dtype=torch.float32),
        corrupt=torch.zeros(8, dtype=torch.bool),
        published_corrupt=torch.zeros(8, dtype=torch.bool),
        fingerprint="tiny-exact-persistent-ensemble-population",
        aihwkit_version="1.1.0",
    )
    return (templates[0], templates[1]), population


def test_cpu_table_matches_direct_exact_pv_for_mixed_current_codes() -> None:
    templates, population = _templates_and_population()
    endpoint_seeds = (92001, 92002, 92003, 92004)
    ensemble = build_persistent_endpoint_codebook_ensemble(
        population,
        templates,
        endpoint_seeds=endpoint_seeds,
        tolerance_raw_x=0.05,
        maximum_program_pulses=16,
        device="cpu",
    )
    requested = torch.tensor([0, 1, 2, 0, 1, 0, 3, 2], dtype=torch.int64)
    assert bool(torch.all(requested <= ensemble.maximum_index))
    target = ensemble.baseline_raw_x + requested.to(torch.float32) * 0.1
    direct = run_raw_active_program_verify(
        population,
        targets_unit=target,
        endpoint_seed=endpoint_seeds[2],
        tolerance_unit=0.05,
        maximum_program_pulses=16,
        device="cpu",
    )
    gathered = ensemble.lookup_raw_x(requested, sample_indices=(2,))

    assert ensemble.persistent_raw_x.shape == (4, 8, 4)
    assert ensemble.build_report["diagnostics_complete"] is True
    assert len(ensemble.build_report["per_endpoint_level"]) == 16
    assert torch.equal(gathered[0], direct.persistent_endpoint_unit)
    valid = (
        torch.arange(ensemble.num_levels).reshape(1, -1)
        <= ensemble.maximum_index.reshape(-1, 1)
    )
    assert bool(torch.all(torch.isfinite(ensemble.persistent_raw_x[:, valid])))
    assert bool(torch.all(torch.isnan(ensemble.persistent_raw_x[:, ~valid])))


def test_builder_rejects_population_template_support_drift() -> None:
    templates, population = _templates_and_population()
    drifted = replace(
        templates[0], cell_upper_raw_x=templates[0].cell_upper_raw_x - 0.01
    )
    with pytest.raises(ValueError, match="do not match population raw-x support"):
        build_persistent_endpoint_codebook_ensemble(
            population,
            (drifted, templates[1]),
            endpoint_seeds=(92001, 92002, 92003, 92004),
            tolerance_raw_x=0.05,
            maximum_program_pulses=16,
            device="cpu",
        )
