"""Split precision must promote before both phase solves and leave callers intact."""
import copy

import pytest
import torch

from experiments import compare_cifar_block_beta_gradients as comparison
from experiments import compare_cifar_split_precision as split
from experiments.calibrate_cifar_block_beta import capture_batches
from experiments.replay_cifar_precision_response import convert_model, check_snapshot, verify_dtype
from experiments.train_cifar_l8_analog import deterministic_setup
from labs.cifar_l8_analog import CifarL8Analog
from labs.tests.test_cifar_beta_feedback import config, exact


def setup_models():
    deterministic_setup(31)
    c = config()
    source = CifarL8Analog(c)
    donor = CifarL8Analog(c)
    convert_model(donor, torch.float64)
    donor.restore(source.snapshot())
    x, y = torch.randn(2, 3, 8, 8), torch.tensor([1, 3])
    saved = capture_batches(source, [(x, y)], c['iterations'], 'cpu')[0]
    return c, source, donor, saved


def test_split_promotes_before_phase_solves_and_preserves_source_and_donor(monkeypatch):
    c, source, donor, saved = setup_models()
    source_before, donor_before = source.snapshot(), donor.snapshot()
    saved_before = copy.deepcopy(saved)
    source_states = [(v, v.state, v.state.clone()) for b in source.analog_blocks
                     for v in [b.boundary_layer, *b.free_layers()]]
    donor_states = [(v, v.state, v.state.clone()) for b in donor.analog_blocks
                    for v in [b.boundary_layer, *b.free_layers()]]
    solves, endpoints = [], []
    original_factory = comparison.nudged_minimizer
    original_gradient = comparison.stable_centered_conv_gradient

    def factory(block, augmented, iterations):
        assert augmented._nudging._function.gradient.dtype == torch.float64
        minimizer = original_factory(block, augmented, iterations)
        original_solve = minimizer.compute_equilibrium
        def solve(*args, **kwargs):
            assert all(v.state.dtype == torch.float64 for v in [block.boundary_layer, *block.free_layers(), *block.weights])
            solves.append(iterations)
            return original_solve(*args, **kwargs)
        minimizer.compute_equilibrium = solve
        return minimizer

    def gradient(interaction, minus, plus, beta):
        assert all(v.dtype == torch.float64 for v in [*minus.values(), *plus.values()])
        endpoints.append((minus, plus))
        result = original_gradient(interaction, minus, plus, beta)
        assert result.dtype == torch.float64
        return result

    monkeypatch.setattr(comparison, 'nudged_minimizer', factory)
    monkeypatch.setattr(comparison, 'stable_centered_conv_gradient', gradient)
    actual = []
    for i, block in enumerate(donor.analog_blocks):
        gradients, displacement = split.eqprop_from_float32(block, saved[i], 1e-7,
            c['iterations'][i], 'cpu', cpu_output=False, measure_layer_displacement=True)
        assert all(g.dtype == torch.float64 and torch.isfinite(g).all() for g in gradients)
        assert any(row['centered_squares'] > 0 for row in displacement['layer_displacements'])
        actual.append(gradients)
    assert len(solves) == 6 and len(endpoints) == 8
    assert any(torch.any((p[k]-m[k]) != (p[k].float()-m[k].float()).double())
               for m,p in endpoints for k in p)
    for variable, original, values in source_states + donor_states:
        assert variable.state is original
        torch.testing.assert_close(variable.state, values, rtol=0, atol=0)
    exact(saved, saved_before)
    check_snapshot(source_before, source.snapshot())
    check_snapshot(donor_before, donor.snapshot())
    verify_dtype(source, torch.float32)
    verify_dtype(donor, torch.float64)
    for i, block in enumerate(donor.analog_blocks):
        expected, _ = comparison.eqprop_gradients(block, split.promote_saved(saved[i]), 1e-7,
                                                c['iterations'][i], 'cpu')
        for a,b in zip(actual[i], expected):
            torch.testing.assert_close(a, b, atol=0, rtol=0)


def test_split_restores_donor_states_when_estimator_fails(monkeypatch):
    c, source, donor, saved = setup_models()
    block = donor.analog_blocks[0]
    states = [(v, v.state) for v in [block.boundary_layer, *block.free_layers()]]
    before = donor.snapshot()
    def broken(block, saved, *args, **kwargs):
        comparison.restore_block(block, saved, 'cpu')
        raise RuntimeError('deliberate failure')
    monkeypatch.setattr(split, 'eqprop_gradients', broken)
    with pytest.raises(RuntimeError, match='deliberate failure'):
        split.eqprop_from_float32(block, saved[0], .001, c['iterations'][0], 'cpu')
    assert all(v.state is original for v, original in states)
    check_snapshot(before, donor.snapshot())
    with pytest.raises(ValueError, match='entirely float64'):
        split.eqprop_from_float32(source.analog_blocks[0], saved[0], .001, 6, 'cpu')
    with pytest.raises(ValueError, match='float32 free/forcing'):
        split.promote_saved(split.promote_saved(saved[0]))


def test_checkpoint_restore_preserves_full64_and_exact_promoted_float32():
    _, source, donor, _ = setup_models()
    # Values deliberately lie between float32 representable numbers.
    with torch.no_grad():
        for _, weight in donor.named_conductances():
            weight.add_(1e-10)
    saved = donor.snapshot()
    converted_first = CifarL8Analog(config())
    convert_model(converted_first, torch.float64)
    converted_first.restore(saved)
    exact(saved, converted_first.snapshot())
    source.restore(saved)
    donor.restore(source.snapshot())
    for group, values in source.snapshot().items():
        for key, value in values.items():
            assert torch.equal(donor.snapshot()[group][key], value.to(donor.snapshot()[group][key].dtype))


def test_precision_difference_pools_squared_errors_not_batch_rms():
    first = split.difference_sums(torch.tensor([2., 3.]), torch.tensor([1., 3.]))
    second = split.difference_sums(torch.tensor([10.]), torch.tensor([6.]))
    pooled = split.finish_difference({k:first[k]+second[k] for k in first})
    assert pooled['difference_rms'] == pytest.approx((17/3)**.5)
    assert pooled['relative_l2'] == pytest.approx((17/46)**.5)


def test_tiny_replay_seals_complete_bundle_and_preserves_gradient_sums(tmp_path, monkeypatch):
    """Exercise full coverage/reporting on CPU with a synthetic training cohort."""
    import argparse
    import json
    import numpy as np

    c = config()
    c.update(batch_size=32, conv_eqprop={'cohort_indices': list(range(256))})
    config_path = tmp_path/'config.json'
    config_path.write_text(json.dumps(c))
    model = CifarL8Analog(c)
    convert_model(model, torch.float64)
    checkpoint_path = tmp_path/'checkpoint.pt'
    torch.save({'epoch': 0, 'model': model.snapshot()}, checkpoint_path)
    dataset_root = tmp_path/'dataset'
    dataset_root.mkdir()
    for i in range(1, 6): (dataset_root/f'data_batch_{i}').write_bytes(b'synthetic-training-input')
    images = np.random.default_rng(41).integers(0, 256, size=(256, 8, 8, 3), dtype=np.uint8)
    labels = np.arange(256) % 10
    monkeypatch.setattr(split, 'read_training_data', lambda _: (images, labels))
    monkeypatch.setattr(split, 'make_split', lambda *_: {'train': list(range(256))})
    out = tmp_path/'run'
    split.run(argparse.Namespace(config=config_path, checkpoint=checkpoint_path,
        dataset_root=dataset_root, run_dir=out, device='cpu', target='test',
        betas=[.0005857333362799141, 5.431465723731432e-05, 2.204854001994856e-06],
        budget_seconds=120.))
    assert not split.reporting.validate_run(out)
    comparison = json.loads((out/'artifacts/comparison.json').read_text())
    assert len(comparison['batches']) == 192 and len(comparison['layers']) == 24
    tensors = torch.load(out/'artifacts/pooled_gradients.pt', weights_only=False)
    assert tensors['examples'] == 256 and len(tensors['example_weighted_gradient_sums']) == 48
    result = json.loads((out/'result.json').read_text())
    assert result['completion']['all_three_modes'] and not result['completion']['official_test_read']
    assert all(torch.isfinite(t).all() for t in tensors['example_weighted_gradient_sums'].values())
