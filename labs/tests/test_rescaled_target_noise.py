"""Protect exp025's actual-loss rematching and unchanged training contract."""
import copy
import importlib.util
import json
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from experiments import prepare_rescaled_target_noise as prep
from experiments.exact_run import load_exact_config

ROOT = Path(__file__).resolve().parents[2]


def source_config(depth, scheme):
    source_scheme = 'ours' if scheme == 'amp2' else scheme
    target = prep.TARGETS[depth]
    path = ROOT/f'configs/conv/relative_noise_eta5em4_20260928/conv{depth}_{source_scheme}_relative{target}_eta5em4_seed0.json'
    return json.loads(path.read_text())


@pytest.mark.parametrize('depth', [1, 2, 3])
@pytest.mark.parametrize('scheme', prep.SCHEMES)
def test_finalized_grid_preserves_rates_and_matched_context(tmp_path, depth, scheme):
    source = source_config(depth, scheme)
    untouched = copy.deepcopy(source)
    calibration = dict(injected_beta=.137, target=prep.TARGETS[depth],
                       target_scale=4.**(-depth), actual_loss_target=True)
    configs = [prep.training_config(source, depth, scheme, eta, label,
                                   Path('accepted.pt'), calibration)
               for eta, label in prep.ETAS]
    assert source == untouched
    factor = 2 if scheme == 'amp2' else 1
    amplification = {'baseline': (1., 1.), 'ours': (4., 1.),
                     'legacy': (4., .25), 'amp2': (4., 2.)}[scheme]
    for config, (eta, label) in zip(configs, prep.ETAS):
        path = tmp_path/(config['arm_id']+'.json')
        path.write_text(json.dumps(config))
        load_exact_config(path)
        assert config['arm_id'] == f'conv{depth}_{scheme}_target{4**depth}_{label}_seed0'
        assert config['target_scale'] == 4.**(-depth)
        assert config['lr'] == [factor*x for x in source['lr']]
        assert config['optimizer'] == dict(source['optimizer'], learning_rate=config['lr'])
        expected_datasets = copy.deepcopy(source['datasets'])
        expected_datasets['mnist']['params']['root'] = prep.TRAINING_DATA_ROOT
        assert config['datasets'] == expected_datasets
        assert config['rescaled_target_noise']['checkpoint_recovery'] == 'full_state_epoch_boundary_only'
        assert config['parameter_order'] == source['parameter_order']
        assert config['initialization']['checkpoint_sha256'] == source['initialization']['checkpoint_sha256']
        assert config['init_checkpoint_path'] == 'accepted.pt'
        assert config['relative_output_calibration'] == calibration
        assert config['eqprop']['endpoint_read_noise_eta'] == eta
        assert config['eqprop']['endpoint_read_noise_seed'] == 2026081601
        assert config['eqprop']['injected_beta_B'] == .137
        assert config['beta']*config['eqprop']['amplification_factor'] == pytest.approx(.137)
        assert (config['model_base']['voltage_amp'], config['model_base']['current_amp']) == amplification
        assert config['runtime_dtype'] == 'float64' and config['seed'] == 0
        assert config['lab']['epochs'] == (10 if depth == 1 else 30)
        assert config['evaluation']['official_test']['policy'] == 'disabled'
        assert prep.cohort_digest(source) == '95af3eebd9416ec643b7d698d061b7dac3fe0d52311dd888293609bada629acd'
    assert len({c['beta'] for c in configs}) == 1


@pytest.fixture
def replay_base(tmp_path, monkeypatch):
    config_path = tmp_path/'bootstrap.json'
    config_path.write_text(json.dumps({'source_contract': {'runtime_source_root': str(ROOT)}}))
    monkeypatch.setattr(sys, 'argv', ['test', '--config', str(config_path)])
    spec = importlib.util.spec_from_file_location('rescaled_target_test_base',
        ROOT/'experiments/analyze_conv_eqprop_bptt_checkpoint_gradients.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize('target_scale', [None, 1/4, 1/16, 1/64])
def test_replay_actual_loss_and_frozen_teaching_derivative(replay_base, target_scale):
    config = source_config(1, 'ours')
    config.pop('init_checkpoint_path')
    config['model_overrides']['mnist_bp_conv_amp']['layer_shapes'][1][0] = 2
    if target_scale is not None:
        config['target_scale'] = target_scale
    runtime = replay_base._build_runtime(config, device=torch.device('cpu'),
        gradient_iterations=4, nudging_mode='current', current_scale='auto')
    layer = runtime['cost_fn'].layers()[-1]
    layer.state = torch.linspace(-.2, .3, 40, dtype=torch.float64).reshape(2, 20).requires_grad_()
    labels = torch.tensor([2, 7])
    runtime['cost_fn'].set_target(labels)
    target = (1. if target_scale is None else target_scale)*F.one_hot(labels, 10).to(layer.state)
    expected = .5*((layer.state[:, 0::2]-layer.state[:, 1::2]-target)**2).sum(1)
    derivative = torch.autograd.grad(expected.sum(), layer.state)[0]
    torch.testing.assert_close(runtime['cost_fn'].eval(), expected, rtol=0, atol=0)
    torch.testing.assert_close(runtime['cost_fn']._grad(layer, mean=False), derivative, rtol=0, atol=0)
    runtime['augmented_fn'].prepare_nudging()
    torch.testing.assert_close(runtime['augmented_fn']._nudging._force, -derivative, rtol=0, atol=0)
    assert runtime['nudging_current_scale'] == 4.


def test_replay_rejects_scaled_unpaired_outputs(replay_base):
    config = source_config(1, 'baseline')
    config.pop('init_checkpoint_path')
    config['model_overrides']['mnist_bp_conv_amp']['layer_shapes'][-1] = [10]
    config['target_scale'] = .25
    with pytest.raises(ValueError, match='requires paired outputs'):
        replay_base._build_runtime(config, device=torch.device('cpu'), gradient_iterations=4)
