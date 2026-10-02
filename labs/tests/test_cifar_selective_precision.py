"""Exercise selective precision, fixed reference K, calibration and immutability."""
import argparse
import copy
import json

import numpy as np
import pytest
import torch

from experiments import compare_cifar_selective_precision as selective
from experiments import compare_cifar_block_beta_gradients as comparison
from experiments.calibrate_cifar_block_beta import capture_batches
from experiments.replay_cifar_precision_response import convert_model, check_snapshot
from experiments.train_cifar_l8_analog import deterministic_setup
from labs.cifar_l8_analog import CifarL8Analog
from labs.tests.test_cifar_beta_feedback import config, exact


def test_candidate_routes_precision_and_restores_live_states(monkeypatch):
    deterministic_setup(31)
    c = config()
    models = {'full32': CifarL8Analog(c), 'donor64': CifarL8Analog(c)}
    convert_model(models['donor64'], torch.float64)
    models['donor64'].restore(models['full32'].snapshot())
    saved = capture_batches(models['full32'], [(torch.randn(2, 3, 8, 8), torch.tensor([1, 3]))],
                            c['iterations'], 'cpu')[0]
    original = copy.deepcopy(saved)
    before = {name: model.snapshot() for name, model in models.items()}
    calls = []
    factory = comparison.nudged_minimizer
    def spy(block, augmented, iterations):
        dtype = block.weights[0].state.dtype
        assert augmented._nudging._function.gradient.dtype == dtype
        calls.append((dtype, iterations))
        assert all(interaction._a_col2im_cache is not None for interaction in block.interactions)
        return factory(block, augmented, iterations)
    monkeypatch.setattr(comparison, 'nudged_minimizer', spy)
    for i, k in enumerate([4, 4, 3]):
        selective.candidate_gradients(models, saved[i], i, .001, k, [3], 'cpu')
    assert calls == [(torch.float32, 4), (torch.float32, 4), (torch.float64, 3)]
    exact(saved, original)
    for name, model in models.items():
        check_snapshot(before[name], model.snapshot())
        for block in model.analog_blocks:
            assert all(interaction._a_col2im_cache is None for interaction in block.interactions)
    block = models['donor64'].analog_blocks[2]
    states = [(v, v.state) for v in [block.boundary_layer, *block.free_layers()]]
    def broken(block, saved, *args, **kwargs):
        comparison.restore_block(block, saved, 'cpu')
        raise RuntimeError('deliberate failure')
    monkeypatch.setattr(selective, 'eqprop_gradients', broken)
    with pytest.raises(RuntimeError, match='deliberate'):
        selective.candidate_gradients(models, saved[2], 2, .001, 3, [3], 'cpu')
    assert all(v.state is state for v, state in states)


def test_tiny_256_replay_calibrates_and_keeps_full64_reference_fixed(tmp_path, monkeypatch):
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
    checkpoint = {'epoch': 0, 'model': model.snapshot()}
    torch.save(checkpoint, checkpoint_path)
    digest = selective.reporting.sha256_file(checkpoint_path)
    dataset_root = tmp_path/'dataset'
    dataset_root.mkdir()
    for i in range(1, 6):
        (dataset_root/f'data_batch_{i}').write_bytes(b'synthetic-training-input')
    images = np.random.default_rng(41).integers(0, 256, size=(256, 8, 8, 3), dtype=np.uint8)
    labels = np.arange(256) % 10
    monkeypatch.setattr(selective, 'read_training_data', lambda _: (images, labels))
    monkeypatch.setattr(selective, 'make_split', lambda *_: {'train': list(range(256))})
    captures, references, calibration_calls, formulas = [], [], [], []
    capture = selective.capture_batches
    local = selective.local_bptt_gradients
    measure = selective.measure_block
    formula = comparison.stable_centered_conv_gradient
    def captured(model, batches, iterations, device):
        captures.append((model.analog_blocks[0].weights[0].state.dtype, tuple(iterations)))
        return capture(model, batches, iterations, device)
    def reference(block, saved, iterations, device):
        assert block.weights[0].state.dtype == saved['gradient'].dtype == torch.float64
        references.append(iterations)
        return local(block, saved, iterations, device)
    def measured(block, minimizer, augmented, cost, records, index, beta, device):
        calibration_calls.append((index, block.weights[0].state.dtype, minimizer.num_iterations))
        return measure(block, minimizer, augmented, cost, records, index, beta, device)
    def gradient(interaction, minus, plus, beta):
        formulas.append(interaction._weight.state.dtype)
        return formula(interaction, minus, plus, beta)
    monkeypatch.setattr(selective, 'capture_batches', captured)
    monkeypatch.setattr(selective, 'local_bptt_gradients', reference)
    monkeypatch.setattr(selective, 'measure_block', measured)
    monkeypatch.setattr(comparison, 'stable_centered_conv_gradient', gradient)
    out = tmp_path/'run'
    selective.run(argparse.Namespace(config=config_path, checkpoint=checkpoint_path,
        dataset_root=dataset_root, run_dir=out, device='cpu', target='test',
        betas=[.0006, .00006, .000002], rms_targets=[.01, .005, .00001],
        nudged_iterations=[4, 4, 3], float64_blocks=[3], budget_seconds=180.))
    selective.validate(out)
    assert references == [6, 6, 4]*8
    assert len(captures) == 16 and all(k == (6, 6, 4) for _, k in captures)
    assert all((dtype, k) == [(torch.float32, 4), (torch.float32, 4), (torch.float64, 3)][i]
               for i, dtype, k in calibration_calls)
    # Endpoint-only calibration must never execute the parameter-gradient formula.
    # Block3's identical double candidate/control is computed once and reused.
    assert len(formulas) == 176 and formulas.count(torch.float32) == 8*6
    data = json.loads((out/'artifacts/comparison.json').read_text())
    assert len(data['batches']) == 192 and len(data['layers']) == 24
    for match in data['calibration']:
        assert match['status'] == 'matched' and 1 <= len(match['history']) <= 12
        assert match['achieved']['relative_target_error'] <= .03
    for row in data['layers']:
        if row['mode'] == 'candidate':
            assert abs(row['output_displacement']['relative_rms']/row['target_relative_rms']-1) <= .03
        if row['mode'] == 'candidate' and row['block'] == 3:
            assert row['same_k_all64_nudged_control']['cosine'] == pytest.approx(1.)
            assert row['same_k_all64_nudged_control']['norm_ratio'] == pytest.approx(1.)
        assert 'native_bptt' not in row
    sums = torch.load(out/'artifacts/pooled_gradients.pt', weights_only=False)
    assert sums['examples'] == 256 and len(sums['example_weighted_gradient_sums']) == 32
    assert all(torch.isfinite(t).all() for t in sums['example_weighted_gradient_sums'].values())
    result = json.loads((out/'result.json').read_text())
    assert result['completion']['all_three_modes'] and result['completion']['all_calibration_targets_met']
    assert not result['completion']['official_test_read']
    assert result['terminal_metrics']['parameters_and_bn_unchanged']
    assert selective.reporting.sha256_file(checkpoint_path) == digest
    exact(torch.load(checkpoint_path, weights_only=False), checkpoint)


def test_numerical_thresholds_are_strict_for_cosine_and_inclusive_for_norm():
    assert not selective.numerical_pass(dict(cosine=.95, norm_ratio=1.))
    assert not selective.numerical_pass(dict(cosine=.99, norm_ratio=1.10001))
    assert selective.numerical_pass(dict(cosine=.99, norm_ratio=.9))


def test_calibration_failure_preserves_all_bounded_evaluations(tmp_path, monkeypatch):
    model = CifarL8Analog(config())
    out = tmp_path/'failed'
    selective.reporting.start_run(out, dict(study_id='test', run_id='failed', arm_id='test',
                                            seed=0, smoke=True, evidence_class='diagnostic'))
    monkeypatch.setattr(selective, 'measure_block', lambda *a, **kw:
        dict(absolute_rms=0., free_rms=1., relative_rms=0., elements=1))
    args = argparse.Namespace(rms_targets=[.01], betas=[1e-6], float64_blocks=[3],
                              nudged_iterations=[4, 4, 3], device='cpu')
    with pytest.raises(RuntimeError, match='No target match within 12'):
        selective.calibrate({'full32': model}, [], args, out, lambda: None)
    history = json.loads((out/'artifacts/calibration.json').read_text())
    assert history[0]['status'] == 'failed' and len(history[0]['history']) == 12
    assert not (out/'result.json').exists()
