"""Actual perfect-diode trajectory and causal mask replay CPU checks."""
import argparse
import copy
import json

import numpy as np
import pytest
import torch

from experiments import diagnose_cifar_block_switching as diagnostic
from experiments.calibrate_cifar_block_beta import capture_batches
from experiments.compare_cifar_block_beta_gradients import eqprop_gradients
from experiments.compare_cifar_selective_precision import block_scope
from experiments.replay_cifar_precision_response import convert_model, check_snapshot
from experiments.train_cifar_l8_analog import deterministic_setup
from labs.cifar_l8_analog import CifarL8Analog
from labs.tests.test_cifar_beta_feedback import config, exact


def fixture():
    deterministic_setup(31)
    c = config()
    model = CifarL8Analog(c)
    convert_model(model, torch.float64)
    saved = capture_batches(model, [(torch.randn(2, 3, 8, 8).double(), torch.tensor([1, 3]))],
                            c['iterations'], 'cpu')[0][2]
    return model, model.analog_blocks[2], saved


def test_tracing_preserves_trajectory_and_existing_ep_formula():
    model, block, saved = fixture()
    before = model.snapshot()
    untouched = copy.deepcopy(saved)
    zero, trace = diagnostic.phase(block, saved, 0., 'cpu')
    assert len(trace) == 8
    assert [r['state_name'] for r in trace] == [l.name for l in block.free_layers()] * 4
    for beta in [0., .01]:
        endpoint, recorded = diagnostic.phase(block, saved, beta, 'cpu', zero_trace=trace)
        ordinary, _ = diagnostic.phase(block, saved, beta, 'cpu', instrument=False)
        exact(endpoint, ordinary)
        if beta == 0:
            assert all(r['switch_count'] == 0 for r in recorded)
            replayed, _ = diagnostic.phase(block, saved, beta, 'cpu', zero_trace=trace, fixed=True)
            exact(replayed, zero)
    estimate, *_ = diagnostic.centered(block, saved, .01, 'cpu', trace)
    with block_scope(block, torch.float64):
        baseline, _ = eqprop_gradients(block, saved, .01, 4, 'cpu')
    for got, expected in zip(estimate, baseline):
        torch.testing.assert_close(got, expected, atol=1e-12, rtol=1e-12)
    check_snapshot(before, model.snapshot())
    exact(untouched, saved)
    assert all('activate' not in l.__dict__ for l in block.free_layers())


def test_fixed_mask_causal_effect_and_order_guard():
    _, block, saved = fixture()
    _, trace = diagnostic.phase(block, saved, 0., 'cpu')
    # Deliberately strong frozen forcing makes this real network cross boundaries.
    saved['gradient'] = torch.randn_like(saved['gradient']) * 100
    native, rows = diagnostic.phase(block, saved, 10., 'cpu', zero_trace=trace)
    fixed, _ = diagnostic.phase(block, saved, 10., 'cpu', zero_trace=trace, fixed=True)
    assert sum(r['switch_count'] for r in rows) > 0
    assert any(not torch.equal(native[k], fixed[k]) for k in native)
    # A reversed mask sequence must be rejected, even when all tensor shapes match.
    with pytest.raises(AssertionError, match='Trace identity mismatch'):
        diagnostic.phase(block, saved, 1., 'cpu', zero_trace=list(reversed(trace)), fixed=True)
    with pytest.raises(AssertionError, match='Trace count mismatch'):
        diagnostic.phase(block, saved, 1., 'cpu', zero_trace=trace[:-1], fixed=True)
    assert all('activate' not in l.__dict__ for l in block.free_layers())
    # Fixed decision trajectory is affine in forcing, including crossed free units.
    low_minus, _ = diagnostic.phase(block, saved, -.001, 'cpu', zero_trace=trace, fixed=True)
    low_plus, _ = diagnostic.phase(block, saved, .001, 'cpu', zero_trace=trace, fixed=True)
    high_minus, _ = diagnostic.phase(block, saved, -10., 'cpu', zero_trace=trace, fixed=True)
    for name in fixed:
        torch.testing.assert_close((fixed[name]-high_minus[name])/20.,
                                   (low_plus[name]-low_minus[name])/.002, rtol=1e-9, atol=1e-9)


def test_margin_is_preclamp_distance_and_exact_zero_is_free():
    _, block, saved = fixture()
    _, trace = diagnostic.phase(block, saved, 0., 'cpu')
    assert any(row['margins']['signed_min'] < 0 for row in trace)
    assert any(row['margins']['absolute_min'] > 0 for row in trace)
    # Post-clamp states would have zeros for all these clamped coordinates.
    assert any(row['clamped'].any() for row in trace)
    summary = diagnostic.margin_summary(torch.tensor([-2., 0., 3.], dtype=torch.float64))
    assert summary['exact_boundary_count'] == 1 and summary['signed_min'] == -2.


def test_tiny_end_to_end_float64_fixed_reference_and_validation(tmp_path, monkeypatch):
    c = config()
    c.update(batch_size=32, conv_eqprop={'cohort_indices': list(range(256))})
    config_path = tmp_path/'config.json'
    config_path.write_text(json.dumps(c))
    deterministic_setup(31)
    model = CifarL8Analog(c)
    convert_model(model, torch.float64)
    with torch.no_grad():
        for _, weight in model.named_conductances():
            weight.add_(1e-10)
    checkpoint_path = tmp_path/'checkpoint.pt'
    checkpoint = dict(epoch=1, model=model.snapshot())
    torch.save(checkpoint, checkpoint_path)
    dataset_root = tmp_path/'dataset'
    dataset_root.mkdir()
    for i in range(1, 6):
        (dataset_root/f'data_batch_{i}').write_bytes(b'synthetic-training-only')
    images = np.random.default_rng(41).integers(0, 256, size=(256, 8, 8, 3), dtype=np.uint8)
    labels = np.arange(256) % 10
    monkeypatch.setattr(diagnostic, 'read_training_data', lambda _: (images, labels))
    monkeypatch.setattr(diagnostic, 'make_split', lambda *_: {'train': list(range(256))})
    calls = []
    original = diagnostic.local_bptt_gradients
    def reference(block, saved, iterations, device):
        assert saved['gradient'].dtype == saved['input'].dtype == torch.float64
        calls.append(iterations)
        return original(block, saved, iterations, device)
    monkeypatch.setattr(diagnostic, 'local_bptt_gradients', reference)
    out = tmp_path/'run'
    diagnostic.run(argparse.Namespace(config=config_path, checkpoint=checkpoint_path,
        dataset_root=dataset_root, run_dir=out, device='cpu', target='test', budget_seconds=180.))
    diagnostic.validate(out)
    assert calls == [4]*8
    data = json.loads((out/'artifacts/comparison.json').read_text())
    assert len(data['native']) == 5 and len(data['zero_trace_rows']) == 64
    for cell in data['native']:
        assert abs(cell['layers'][-1]['state']['relative_rms']/cell['target_relative_rms']-1) <= .03
        for row in cell['updates']:
            assert row['switch_count'] == row['clamp_to_free_count'] + row['free_to_clamp_count']
    exact(torch.load(checkpoint_path, weights_only=False), checkpoint)
