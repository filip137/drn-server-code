from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from experiments.mnist_relu_drn.ibm_om_corrupt_source_hwa_cross_array_pulse_adam import (
    build_corrupt_persistent_endpoint_codebook_ensemble,
    build_published_persistent_targets,
    build_repaired_mapping_templates,
    clamp_targets_to_published_corrupt_singletons,
    pair_and_winsorize_om_populations,
)
from training.ibm_reram_hwa import (
    IbmReramArrayPopulation,
    load_om_array_population,
    save_om_array_population,
)
from training.ibm_reram_program_verify import OM_PRESET, derive_seed


SHAPES = ((2, 2), (2, 2))
KEYS = ("base.dense_weight.0", "base.dense_weight.1")


def _native_seed(value: int) -> int:
    return int(value % (2**31 - 2) + 1)


def _native_pair() -> tuple[IbmReramArrayPopulation, IbmReramArrayPopulation]:
    # Cell zero is the literal published corrupt singleton.  Healthy cells
    # deliberately have heterogeneous upper bounds and two supports extend
    # beyond nominal raw-a, exercising the Winsorization intervention.
    published_corrupt = torch.tensor(
        [True, False, False, False, False, False, False, False]
    )
    healthy_lower = torch.tensor(
        [0.005, -1.20, -0.80, -0.70, -0.90, -0.75, -0.65, -0.85],
        dtype=torch.float32,
    )
    healthy_upper = torch.tensor(
        [0.005, 1.20, 0.50, 0.90, 0.40, 0.85, 0.55, 0.95],
        dtype=torch.float32,
    )
    published_up = torch.full((8,), 0.2, dtype=torch.float32)
    published_down = torch.full((8,), 0.2, dtype=torch.float32)
    published_up[0] = 0.0
    published_down[0] = 0.0
    common = dict(
        assignment_seed=93001,
        binding_keys=KEYS,
        binding_shapes=SHAPES,
        binding_sampling_seeds=tuple(
            _native_seed(derive_seed(93001, OM_PRESET, key, "published"))
            for key in KEYS
        ),
        donor_sampling_seeds=tuple(
            _native_seed(derive_seed(93001, OM_PRESET, key, "repair_donor"))
            for key in KEYS
        ),
        nominal_dw_min=0.2,
        dw_min_std=0.1,
        write_noise_std=0.1,
        reference=torch.zeros(8, dtype=torch.float32),
        published_corrupt=published_corrupt,
        aihwkit_version="1.1.0",
    )
    published = IbmReramArrayPopulation(
        **common,
        corruption_policy="published",
        max_bound=healthy_upper,
        min_bound=healthy_lower,
        dwmin_up=published_up,
        dwmin_down=published_down,
        corrupt=published_corrupt,
        fingerprint="native-published",
    )
    repaired_lower = healthy_lower.clone()
    repaired_upper = healthy_upper.clone()
    repaired_lower[0] = -0.80
    repaired_upper[0] = 0.60
    repaired_up = published_up.clone()
    repaired_down = published_down.clone()
    repaired_up[0] = 0.2
    repaired_down[0] = 0.2
    repaired = IbmReramArrayPopulation(
        **common,
        corruption_policy="counterfactual_repaired",
        max_bound=repaired_upper,
        min_bound=repaired_lower,
        dwmin_up=repaired_up,
        dwmin_down=repaired_down,
        corrupt=torch.zeros(8, dtype=torch.bool),
        fingerprint="native-repaired",
    )
    return published, repaired


def test_pair_preserves_native_fault_and_uses_positive_conductances() -> None:
    pair = pair_and_winsorize_om_populations(*_native_pair())

    assert pair.published.min_bound[0].item() == pytest.approx(0.005)
    assert pair.published.max_bound[0].item() == pytest.approx(0.005)
    assert pair.published.dwmin_up[0].item() == 0.0
    assert pair.published.dwmin_down[0].item() == 0.0
    assert pair.repaired.min_bound[0].item() == pytest.approx(-0.8)
    assert pair.published.min_bound[1].item() == -1.0
    assert pair.published.max_bound[1].item() == 1.0
    assert bool(torch.all(pair.published.min_bound + 1.0 >= 0.0))
    assert bool(torch.all(pair.published.max_bound + 1.0 <= 2.0))
    assert pair.report["corrupt_cells"] == 1
    assert pair.report["negative_full_conductance_count"] == 0


def test_pair_rejects_an_independently_changed_healthy_draw() -> None:
    published, repaired = _native_pair()
    drifted = replace(repaired, max_bound=repaired.max_bound.clone())
    drifted.max_bound[1] -= 0.01
    with pytest.raises(ValueError, match="healthy parameters differ"):
        pair_and_winsorize_om_populations(published, drifted)


def test_winsorized_pair_is_strict_npz_replayable(tmp_path) -> None:
    pair = pair_and_winsorize_om_populations(*_native_pair())
    for name, population in (
        ("published", pair.published),
        ("repaired", pair.repaired),
    ):
        path = tmp_path / f"{name}.npz"
        save_om_array_population(path, population)
        replay = load_om_array_population(path)
        assert replay.fingerprint == population.fingerprint
        assert torch.equal(replay.min_bound, population.min_bound)
        assert torch.equal(replay.max_bound, population.max_bound)
        assert torch.equal(replay.corrupt, population.corrupt)


def test_repaired_mapping_then_published_target_audit_is_explicit() -> None:
    pair = pair_and_winsorize_om_populations(*_native_pair())
    masters = (torch.tensor([[0.75]]), torch.tensor([[-0.50]]))
    templates = build_repaired_mapping_templates(
        masters, pair, level_spacing_raw_x=0.1
    )
    targets = build_published_persistent_targets(masters, templates, pair)

    assert len(templates) == 2
    assert targets.report["mapping_population"] == "paired_counterfactual_repaired"
    assert targets.report["programming_population"] == "literal_published"
    assert targets.report["target_clipping"] is False
    assert targets.report["unsupported_healthy_cells"] == 0
    assert all(bool(torch.all(value >= 0.0)) for value in targets.target_full_conductance)
    assert all(bool(torch.all(value <= 2.0)) for value in targets.target_full_conductance)


def test_fault_only_no_write_rung_clamps_only_native_corrupt_cells() -> None:
    pair = pair_and_winsorize_om_populations(*_native_pair())
    masters = (torch.tensor([[0.75]]), torch.tensor([[-0.50]]))
    templates = build_repaired_mapping_templates(
        masters, pair, level_spacing_raw_x=0.1
    )
    targets = build_published_persistent_targets(masters, templates, pair)
    faulted = clamp_targets_to_published_corrupt_singletons(
        targets.target_full_conductance, pair
    )

    for requested, realized, mask in zip(
        faulted.requested_full_conductance,
        faulted.realized_full_conductance,
        faulted.corrupt_mask,
    ):
        assert torch.equal(realized[~mask], requested[~mask])
    flat_realized = torch.cat(
        tuple(value.reshape(-1) for value in faulted.realized_full_conductance)
    )
    corrupt = pair.published.corrupt
    assert torch.equal(
        flat_realized[corrupt], pair.published.min_bound[corrupt] + 1.0
    )
    assert faulted.report["program_verify"] is False
    assert faulted.report["verify_reads"] == 0
    assert faulted.report["programming_noise"] is False
    assert faulted.report["healthy_requested_values_preserved"] is True


def test_corrupt_endpoint_table_is_stuck_across_heterogeneous_capacities() -> None:
    pair = pair_and_winsorize_om_populations(*_native_pair())
    masters = (torch.tensor([[0.75]]), torch.tensor([[-0.50]]))
    templates = build_repaired_mapping_templates(
        masters, pair, level_spacing_raw_x=0.1
    )
    ensemble = build_corrupt_persistent_endpoint_codebook_ensemble(
        pair,
        templates,
        endpoint_seeds=(93201, 93202, 93203, 93204),
        tolerance_raw_x=0.05,
        maximum_program_pulses=16,
        device="cpu",
    )

    assert ensemble.maximum_index.unique().numel() > 1
    assert ensemble.population_fingerprint == pair.published.fingerprint
    corrupt = pair.published.corrupt
    stuck_x = ((pair.published.min_bound[corrupt] + 1.0) / 2.0).item()
    observed = ensemble.persistent_raw_x[:, corrupt, :]
    valid = (
        torch.arange(ensemble.num_levels).reshape(1, 1, -1)
        <= ensemble.maximum_index[corrupt].reshape(1, -1, 1)
    ).expand_as(observed)
    assert torch.allclose(
        observed[valid], torch.full_like(observed[valid], stuck_x)
    )
    assert bool(torch.all(torch.isnan(observed[~valid])))

    # A mixed request at each cell's own capacity exercises the gathered API.
    gathered = ensemble.lookup_full_conductance(
        ensemble.maximum_index.clone(), sample_indices=(0, 1)
    )
    assert gathered.shape == (2, pair.published.size)
    assert bool(torch.all(gathered >= 0.0))
    assert bool(torch.all(gathered <= 2.0))
