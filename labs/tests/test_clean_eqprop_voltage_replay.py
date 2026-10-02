"""Count-weighted voltage moments and fail-closed scientific replay guards."""
from types import SimpleNamespace

import pytest
import torch

from experiments.replay_clean_eqprop_voltages import (
    aggregate_profiles, assert_parameter_values, check_initial_profile,
    pool_statistics, tensor_statistics,
)


def test_pooled_statistics_match_concatenation_with_unequal_batch_sizes():
    batches = [torch.tensor([-8., 0.], dtype=torch.float64),
               torch.tensor([1., 2., 4., 0., 13.], dtype=torch.float64)]
    pooled = pool_statistics([tensor_statistics(b) for b in batches])
    values = torch.cat(batches)
    assert pooled['rms'] == pytest.approx(float(values.square().mean().sqrt()))
    assert pooled['signed_mean'] == pytest.approx(float(values.mean()))
    assert pooled['mean_absolute'] == pytest.approx(float(values.abs().mean()))
    assert pooled['std'] == pytest.approx(float(values.std(correction=0)))
    assert pooled['element_count'] == 7
    assert pooled['zero_fraction'] == 2 / 7
    assert (pooled['minimum'], pooled['maximum']) == (-8., 13.)


def test_centered_and_pooled_endpoint_rms_keep_distinct_definitions():
    rows = []
    for phase, value in [('free', [2., 2.]), ('positive', [6., 0.]),
                         ('negative', [2., 4.]), ('centered_half_difference', [2., -2.])]:
        rows.append(dict(state_layer_index=0, phase=phase, **tensor_statistics(torch.tensor(value))))
    profile = aggregate_profiles(rows, depth=0)[0]
    assert profile['endpoint_rms_P'] == pytest.approx(14 ** .5)
    assert profile['centered_half_difference_rms_D'] == 2
    assert profile['D_over_F'] == 1
    assert profile['D_over_P'] == pytest.approx(2 / 14 ** .5)


def test_nonfinite_voltage_rejected_before_aggregation():
    with pytest.raises(FloatingPointError):
        tensor_statistics(torch.tensor([float('nan')]))


def test_initialization_reference_checks_depth_and_drift():
    check_initial_profile([2. + 1e-11, 0.], [2., 0.])
    with pytest.raises(ValueError, match='layer count'):
        check_initial_profile([2.], [2., 3.])
    with pytest.raises(ValueError, match='layer 1'):
        check_initial_profile([2., 3.01], [2., 3.])


def test_parameter_restore_uses_exact_values_and_rejects_mutations_and_bias():
    saved = [torch.tensor([1.23456789], dtype=torch.float64), torch.zeros(1, dtype=torch.float64)]
    params = [SimpleNamespace(name='Weight_0', state=saved[0].clone()),
              SimpleNamespace(name='Bias_0', state=saved[1].clone())]
    assert_parameter_values(params, saved)
    params[0].state += 1e-12
    with pytest.raises(ValueError, match='changed'):
        assert_parameter_values(params, saved)
    params[0].state = saved[0].clone()
    params[1].state += 1
    with pytest.raises(ValueError, match='Nonzero frozen bias'):
        assert_parameter_values(params, [saved[0], params[1].state.clone()])


def test_voltage_replay_restores_common_start_and_keeps_training_injected_beta():
    from experiments.replay_clean_eqprop_voltages import replay_batch

    layer = SimpleNamespace(state=torch.zeros(2, dtype=torch.float64))
    nudging = SimpleNamespace(_force=None)
    augmented = SimpleNamespace(_nudging=nudging, nudging=0.)
    starts, signed_injections = [], []
    cost = SimpleNamespace(set_target=lambda _: None, layers=lambda: [layer],
                           _grad=lambda *_args, **_kwargs: torch.tensor([-3., 2.], dtype=torch.float64))
    def free():
        assert not torch.is_grad_enabled()
        layer.state = torch.tensor([4., 5.], dtype=torch.float64)
    def prepare():
        nudging._force = -cost._grad()
    augmented.prepare_nudging = prepare
    def endpoint():
        assert not torch.is_grad_enabled()
        starts.append(layer.state.clone())
        signed_injections.append(float(augmented.nudging) * 16)
        layer.state = layer.state + augmented.nudging * 16 * nudging._force
    def restore(layers, values):
        for target, value in zip(layers, values):
            target.state = value.clone()
    base = SimpleNamespace(_clone_states=lambda layers: [v.state.clone() for v in layers],
                           _restore_states=restore, _finite=lambda *_: None)
    gate = SimpleNamespace(torch=torch, base=base,
                           shadow=SimpleNamespace(_set_input_and_reset=lambda *_a, **_kw: None))
    runtime = dict(free_layers=[layer], cost_fn=cost, augmented_fn=augmented,
                   minimizer_inference=SimpleNamespace(compute_equilibrium=free),
                   minimizer_augmented=SimpleNamespace(compute_equilibrium=endpoint),
                   nudging_current_scale=16.)
    batch = dict(images=torch.zeros(2, 1), labels=torch.zeros(2, dtype=torch.long), batch_index=0)
    rows = replay_batch(gate, runtime, batch, injected_beta=2., device=torch.device('cpu'))
    assert torch.equal(starts[0], starts[1])
    assert signed_injections == [2., -2.]
    assert len(rows) == 4
    assert aggregate_profiles(rows, depth=0)[0]['centered_half_difference_rms_D'] == pytest.approx(26 ** .5)
