"""Precision conversions and matched-endpoint controls on real tiny circuits."""
import torch
from experiments import replay_cifar_precision_response as replay
from experiments.calibrate_cifar_block_beta import capture_batches
from experiments.compare_cifar_block_beta_gradients import eqprop_gradients
from experiments.train_cifar_l8_analog import deterministic_setup
from labs.cifar_l8_analog import CifarL8Analog, batchnorm_minibatch_stats_only
from labs.tests.test_cifar_beta_feedback import config


def model_and_batch():
    deterministic_setup(17)
    c = config()
    m = CifarL8Analog(c)
    x = torch.randn(2, 3, 8, 8)
    y = torch.tensor([1, 3])
    return c, m, x, y


def test_full_conversion_preserves_checkpoint_values_and_survives_state_reset():
    c, m, x, y = model_and_batch()
    before = m.snapshot()
    replay.convert_model(m, torch.float64)
    replay.check_snapshot(before, m.snapshot())
    promoted = m.snapshot()
    with batchnorm_minibatch_stats_only(m):
        _, output = m.bptt(x.double(), c['iterations'])
        gradients = torch.autograd.grad(output.square().mean(), [p for _, p in m.named_conductances()])
    assert output.dtype == torch.float64
    assert all(g.dtype == torch.float64 for g in gradients)
    replay.verify_dtype(m, torch.float64)
    replay.check_snapshot(promoted, m.snapshot())


def test_float32_phase_pair_matches_existing_estimator_with_exact_same_displacement(monkeypatch):
    c, m, x, y = model_and_batch()
    before = m.snapshot()
    saved = capture_batches(m, [(x, y)], c['iterations'], 'cpu')[0]
    original_gradient = replay.stable_centered_conv_gradient
    def checked(interaction, minus, plus, beta):
        assert interaction._amp_factor().dtype == next(iter(plus.values())).dtype
        return original_gradient(interaction, minus, plus, beta)
    monkeypatch.setattr(replay, 'stable_centered_conv_gradient', checked)
    for i, block in enumerate(m.analog_blocks):
        original, stats = eqprop_gradients(block, saved[i], .001, c['iterations'][i], 'cpu',
                                          measure_layer_displacement=True)
        observed, promoted, measured = replay.precision_pair(block, saved[i], .001, c['iterations'][i], 'cpu', True)
        for actual, expected in zip(observed, original):
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        for a, b in zip(measured, stats['layer_displacements']):
            for key in ['elements', 'free_squares', 'centered_squares', 'unchanged_elements']:
                assert a[key] == b[key]
        assert all(g.dtype == torch.float64 and torch.isfinite(g).all() for g in promoted)
    replay.check_snapshot(before, m.snapshot())


def test_float64_replay_computes_nonzero_changes_below_float32_spacing():
    free = torch.tensor([1., 1., 0., -2.], dtype=torch.float64)
    minus = free - torch.tensor([1e-10, 0., 0., 1e-10], dtype=torch.float64)
    plus = free + torch.tensor([1e-10, 0., 0., 1e-10], dtype=torch.float64)
    row = replay.finish_measurement(replay.state_measurement(minus, plus, free))
    assert row['active_elements'] == 3
    assert row['active_sub_ulp_elements'] == 2
    assert row['active_unchanged_elements'] == 1
    assert row['unchanged_elements'] == 2
    assert 0 < row['absolute_rms'] < 1e-9
    rounded = replay.state_measurement(minus.float(), plus.float(), free.float())
    assert rounded['centered_squares'] == 0


def test_float64_phase_solver_and_gradients_keep_promoted_model_unchanged():
    c, m, x, y = model_and_batch()
    replay.convert_model(m, torch.float64)
    before = m.snapshot()
    saved = capture_batches(m, [(x.double(), y)], c['iterations'], 'cpu')[0]
    for i, block in enumerate(m.analog_blocks):
        gradients, promoted, measured = replay.precision_pair(block, saved[i], 1e-7,
            c['iterations'][i], 'cpu', False)
        assert not promoted
        assert all(g.dtype == torch.float64 and torch.isfinite(g).all() for g in gradients)
        assert any(v['centered_squares'] > 0 for v in measured)
    replay.verify_dtype(m, torch.float64)
    replay.check_snapshot(before, m.snapshot())
