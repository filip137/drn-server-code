"""Regression for roundoff in the frozen analyzer's reconstructed-noise hash."""
import hashlib

import pytest
import torch

from experiments.replay_conv3_trained_beta_noise import (
    beta_points, cell_identity, check_noise_record, replay_contexts, vector_comparison,
)


def test_explicit_beta_and_t_grid_preserves_checkpoint_role_and_epoch():
    cfg = dict(cases=[dict(name='ours_epoch27', checkpoint_role='best_validation',
                           checkpoint_epoch=27, beta=.987333678708)],
               T_values=[8, 16], K=8, injected_betas=[.001, .1, .987333678708, 10],
               smoke_injected_betas=[.001, 10])
    contexts = replay_contexts(cfg)
    assert [c['replay_T'] for c in contexts] == [8, 16]
    assert all(c['checkpoint_role'] == 'best_validation' and c['checkpoint_epoch'] == 27
               for c in contexts)
    assert beta_points(cfg) == [.001, .1, .987333678708, 10]
    assert beta_points(cfg, smoke=True) == [.001, 10]
    names = {cell_identity(c, b, absolute=True) for c in contexts for b in beta_points(cfg)}
    assert len(names) == 8
    assert 'ours_epoch27_T16K8_beta_0p987333678708' in names


def test_historical_factor_config_keeps_cell_names():
    cfg = dict(cases=[dict(name='legacy')], T=64, K=8,
               beta_factors=[.1, 1., 10.], smoke_beta_factors=[1.])
    case, = replay_contexts(cfg)
    assert case['replay_T'] == 64
    assert cell_identity(case, 1.) == 'legacy_factor_1'
    assert beta_points(cfg) == [.1, 1., 10.]


def test_scheme_specific_betas_do_not_form_a_cross_product():
    cfg = dict(injected_betas=[.02173, 1.385], smoke_injected_betas=[.02173, 1.385])
    legacy, ours = dict(injected_betas=[.02173]), dict(injected_betas=[1.385])
    assert beta_points(cfg, case=legacy) == [.02173]
    assert beta_points(cfg, case=ours, smoke=True) == [1.385]
    with pytest.raises(AssertionError, match='outside declared grid'):
        beta_points(cfg, case=dict(injected_betas=[2.]))


def test_k_sweep_computes_reference_first_without_mixing_checkpoints():
    contexts = replay_contexts(dict(cases=[dict(name='legacy'), dict(name='ours')],
                                   T_values=[16], K_values=[8, 16, 32, 64]))
    assert [(c['name'], c['replay_K']) for c in contexts] == [
        (name, k) for name in ['legacy', 'ours'] for k in [64, 8, 16, 32]]
    assert len({c['context_name'] for c in contexts}) == 8


def test_fixed_reference_separates_direction_and_scale_and_handles_zero():
    ref = torch.tensor([3., 4.], dtype=torch.float64)
    assert vector_comparison(ref * 2, ref) == dict(cosine=1., relative_l2=1., norm_ratio=2.)
    assert vector_comparison(ref, ref * 0) == dict(cosine=None, relative_l2=None, norm_ratio=None)


def test_matched_draws_allow_sigma_dependent_reconstruction_roundoff():
    seed, batch, layer = 2026092101, 2, 1
    derived_seed = seed + 1000 * batch + 100 + layer
    generator = torch.Generator().manual_seed(derived_seed)
    standard = torch.randn((16, 64, 14, 14), generator=generator, dtype=torch.float64)
    seen, records = {}, []
    for sigma in (3e-4, 5e-4):
        reconstructed = (standard * sigma) / sigma
        record = dict(phase='negative', state_layer_index=layer, seed=derived_seed,
                      element_count=standard.numel(),
                      standard_normal_sha256=hashlib.sha256(reconstructed.numpy().tobytes()).hexdigest())
        check_noise_record(seen, record, sigma=sigma, seed=seed, batch_index=batch)
        check_noise_record(seen, record, sigma=sigma, seed=seed, batch_index=batch)
        records.append(record)
    assert records[0]['standard_normal_sha256'] != records[1]['standard_normal_sha256']
    with pytest.raises(AssertionError, match='within sigma'):
        check_noise_record(seen, records[1], sigma=3e-4, seed=seed, batch_index=batch)
    with pytest.raises(AssertionError, match='RNG seed'):
        check_noise_record(seen, {**records[0], 'seed': derived_seed + 1},
                           sigma=3e-4, seed=seed, batch_index=batch)
