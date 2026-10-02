"""Numerical contracts for the new block-current calibration diagnostic."""
import json
from pathlib import Path
import pytest
import torch
from experiments.calibrate_cifar_block_beta import (FrozenOutputCost, measure_block, nudged_minimizer, pooled_rms, search_beta)
from labs.cifar_l8_analog import AnalogSolverConfig, VoltageClampedConvBlock
from training.sgd import AugmentedFunction


def test_log_search_handles_nonlinear_response_and_zero_resolution():
    beta, rows = search_beta(lambda b: {'relative_rms': 0. if b < 10 else b**.5/1000}, 4.,metric='relative_rms',tolerance=.01)
    assert abs(beta**.5/1000/4-1) <= .01
    assert rows[0]['relative_rms']==0
    result=pooled_rms(16,4,4)
    assert result['absolute_rms']==2 and result['free_rms']==1 and result['relative_rms']==2


def test_physical_current_coefficient_and_centered_zero_beta():
    torch.manual_seed(7)
    c=json.loads((Path(__file__).resolve().parents[2]/'configs/cifar/fresh_readout_lr_20260928/ours.json').read_text())
    b=VoltageClampedConvBlock(logical_input_channels=1,analog_channels=[2,2],spatial_size=3,
        solver=AnalogSolverConfig.from_mapping(c['solver']),input_gain_init=1.,weight_gain=1.,
        weight_init_mode='kaiming_uniform',weight_min=1e-7,weight_max=10.,device='cpu')
    with torch.no_grad():b(torch.ones(2,1,3,3),num_iterations=2,reset=True)
    layer=b.free_layers()[-1]
    cost=FrozenOutputCost(layer);cost.gradient=torch.full_like(layer.state,.3)
    aug=AugmentedFunction(b.energy,cost,nudging_mode='current',current_scale=1.)
    aug.prepare_nudging();aug.nudging=7.
    original=b.energy.b_coef_fn(layer)();nudged=aug.b_coef_fn(layer)()
    torch.testing.assert_close(nudged-original,torch.full_like(layer.state,2.1))
    record=[[dict(input=b.boundary_layer.state.detach().clone(),states=[l.state.detach().clone() for l in b.free_layers()],gradient=cost.gradient.clone())]]
    minimizer=nudged_minimizer(b,aug,2)
    zero=measure_block(b,minimizer,aug,cost,record,0,0.,torch.device('cpu'))
    assert zero['relative_rms']==zero['absolute_rms']==0.
    positive=measure_block(b,minimizer,aug,cost,record,0,.1,torch.device('cpu'))
    assert positive['absolute_rms']>0


def test_centered_gradient_sign_and_batch_mean_on_single_layer():
    from experiments.compare_cifar_block_beta_gradients import local_gradients, agreement
    torch.manual_seed(7)
    c=json.loads((Path(__file__).resolve().parents[2]/'configs/cifar/fresh_readout_lr_20260928/ours.json').read_text())
    c['solver']['voltage_amp']=c['solver']['current_amp']=1.
    b=VoltageClampedConvBlock(logical_input_channels=1,analog_channels=[2],spatial_size=3,
        solver=AnalogSolverConfig.from_mapping(c['solver']),input_gain_init=1.,weight_gain=1.,
        weight_init_mode='kaiming_uniform',weight_min=1e-7,weight_max=10.,device='cpu')
    before=[w.state.detach().clone() for w in b.weights]
    measurements=[]
    for batch_size in (2,4):
        with torch.no_grad():b(torch.ones(batch_size,1,3,3),num_iterations=10,reset=True)
        saved=dict(input=b.boundary_layer.state.detach().clone(),
            states=[l.state.detach().clone() for l in b.free_layers()],
            gradient=torch.full_like(b.free_layers()[-1].state,.001))
        ep,ref=local_gradients(b,saved,.01,10,torch.device('cpu'))
        result=agreement(ep[0],ref[0])
        assert result['cosine']>.999 and abs(result['norm_ratio']-1)<.01
        measurements.append((ep[0],ref[0]))
    for first,second in zip(*measurements):
        torch.testing.assert_close(first,second,rtol=.01,atol=1e-6)
    assert all(torch.equal(w.state,s) for w,s in zip(b.weights,before))


@pytest.mark.parametrize('voltage_amp',[1.,4.])
def test_bounded_conv_derivative_matches_original_formula(voltage_amp):
    from experiments.compare_cifar_block_beta_gradients import bounded_conv_energy_gradient
    from model.resistive.interaction import ConvResistive
    torch.manual_seed(17)
    c=json.loads((Path(__file__).resolve().parents[2]/'configs/cifar/fresh_readout_lr_20260928/ours.json').read_text())
    c['solver']['voltage_amp']=voltage_amp
    b=VoltageClampedConvBlock(logical_input_channels=1,analog_channels=[2,4],spatial_size=3,
        solver=AnalogSolverConfig.from_mapping(c['solver']),input_gain_init=1.,weight_gain=1.,
        weight_init_mode='kaiming_uniform',weight_min=1e-7,weight_max=10.,device='cpu')
    with torch.no_grad():
        b(torch.randn(3,1,3,3),num_iterations=2,reset=True)
        for layer in b.free_layers():layer.state=torch.randn_like(layer.state)
        for fn in b.energy._interactions:
            if isinstance(fn,ConvResistive):
                torch.testing.assert_close(bounded_conv_energy_gradient(fn),fn._grad_weight(),rtol=2e-6,atol=2e-6)


def test_centered_voltage_product_avoids_large_boundary_cancellation():
    from experiments.compare_cifar_block_beta_gradients import stable_centered_conv_gradient
    from model.resistive.interaction import ConvResistive
    torch.manual_seed(19)
    c=json.loads((Path(__file__).resolve().parents[2]/'configs/cifar/fresh_readout_lr_20260928/ours.json').read_text())
    b=VoltageClampedConvBlock(logical_input_channels=1,analog_channels=[2,4],spatial_size=3,
        solver=AnalogSolverConfig.from_mapping(c['solver']),input_gain_init=100.,weight_gain=1.,
        weight_init_mode='kaiming_uniform',weight_min=1e-7,weight_max=10.,device='cpu')
    with torch.no_grad():
        b(torch.randn(3,1,3,3),num_iterations=2,reset=True)
        minus={l.name:torch.randn_like(l.state)*.01 for l in b.free_layers()}
        plus={k:s+torch.randn_like(s)*.001 for k,s in minus.items()}
        beta=.2
        for fn in b.energy._interactions:
            if not isinstance(fn,ConvResistive):continue
            stable=stable_centered_conv_gradient(fn,minus,plus,beta)
            phase_gradients=[]
            for states in (minus,plus):
                for l in (fn._layer_pre,fn._layer_post):l.state=states.get(l.name,l.state).double()
                phase_gradients.append(fn._grad_weight())
            reference=(phase_gradients[1]-phase_gradients[0])/(2*beta)
            torch.testing.assert_close(stable.double(),reference,rtol=5e-5,atol=1e-7)


def test_internal_displacements_preserve_gradients_and_endpoint_measurement(monkeypatch):
    from experiments.compare_cifar_block_beta_gradients import eqprop_gradients
    from labs.custom_minimizer import CustomQuadraticMinimizer
    torch.manual_seed(23)
    c=json.loads((Path(__file__).resolve().parents[2]/'configs/cifar/fresh_readout_lr_20260928/ours.json').read_text())
    block=VoltageClampedConvBlock(logical_input_channels=1,analog_channels=[2,4],spatial_size=3,
        solver=AnalogSolverConfig.from_mapping(c['solver']),input_gain_init=1.,weight_gain=1.,
        weight_init_mode='kaiming_uniform',weight_min=1e-7,weight_max=10.,device='cpu')
    with torch.no_grad():block(torch.ones(2,1,3,3),num_iterations=6,reset=True)
    saved=dict(input=block.boundary_layer.state.detach().clone(),
        states=[layer.state.detach().clone() for layer in block.free_layers()],
        gradient=torch.full_like(block.free_layers()[-1].state,.01))
    weights=[w.state.detach().clone() for w in block.weights]
    rng=torch.get_rng_state().clone()
    original=CustomQuadraticMinimizer.compute_equilibrium
    calls=[]
    def counted(self,*args,**kwargs):
        calls.append(self.num_iterations)
        return original(self,*args,**kwargs)
    monkeypatch.setattr(CustomQuadraticMinimizer,'compute_equilibrium',counted)
    reference,output=eqprop_gradients(block,saved,.1,6,torch.device('cpu'))
    plain_calls=list(calls);calls.clear()
    measured,details=eqprop_gradients(block,saved,.1,6,torch.device('cpu'),measure_layer_displacement=True)
    assert calls==plain_calls==[6,6]
    assert all(torch.equal(a,b) for a,b in zip(reference,measured))
    assert {key:details[key] for key in output}==output
    assert len(details['layer_displacements'])==2
    last=details['layer_displacements'][-1]
    assert {key:last[key] for key in output}==output
    for row in details['layer_displacements']:
        assert row['absolute_rms']**2==pytest.approx(row['centered_squares']/row['elements'])
        assert 0<=row['unchanged_fraction']<=1
    assert details['output_force_rms']==pytest.approx(.01)
    assert details['injected_force_rms']==pytest.approx(.001)
    assert all(torch.equal(w.state,before) for w,before in zip(block.weights,weights))
    assert torch.equal(torch.get_rng_state(),rng)


def test_state_and_force_pooling_matches_concatenated_unequal_batches():
    from experiments.compare_cifar_block_beta_gradients import (
        accumulate_displacement, state_measurement_fields)
    # Different internal/output sizes ensure force denominators cannot be confused.
    internal_deltas = [torch.tensor([0., 2.]), torch.tensor([3., 0., 4., 0.])]
    output_deltas = [torch.tensor([1.]), torch.tensor([0., 2.])]
    internal_free = [torch.tensor([1., 1.]), torch.tensor([2., 2., 2., 2.])]
    output_free = [torch.tensor([2.]), torch.tensor([3., 3.])]
    forces = [torch.tensor([2.]), torch.tensor([4., 6.])]
    def raw(delta, free, layer):
        return dict(layer=layer, state_name=f'Layer_{layer}',
            centered_squares=float(delta.double().square().sum()),
            free_squares=float(free.double().square().sum()), elements=free.numel(),
            unchanged_elements=int((delta == 0).sum()))
    pooled = {}
    beta = .3
    for internal_delta, output_delta, free, output, force in zip(
            internal_deltas, output_deltas, internal_free, output_free, forces):
        layers = [raw(internal_delta, free, 1), raw(output_delta, output, 2)]
        force_rms = float(force.double().square().mean().sqrt())
        measured = dict(layers[-1], layer_displacements=layers,
            output_force_rms=force_rms, injected_force_rms=beta * force_rms)
        batch = state_measurement_fields(measured, 'Layer_1')
        assert batch['state_displacement']['elements'] == internal_delta.numel()
        assert batch['force_elements'] == output_delta.numel()
        assert batch['output_force_rms'] == pytest.approx(force_rms)
        accumulate_displacement(pooled, measured)
    result = state_measurement_fields(pooled, 'Layer_1')
    state = result['state_displacement']
    delta, free, force = map(torch.cat, (internal_deltas, internal_free, forces))
    assert state['absolute_rms'] == pytest.approx(float(delta.double().square().mean().sqrt()))
    assert state['free_rms'] == pytest.approx(float(free.double().square().mean().sqrt()))
    assert state['relative_rms'] == pytest.approx(float(delta.norm() / free.norm()))
    assert state['unchanged_fraction'] == .5
    assert state['elements'] == 6 and result['force_elements'] == 3
    assert result['output_force_rms'] == pytest.approx(float(force.double().square().mean().sqrt()))
    assert result['injected_force_rms'] == pytest.approx(beta * result['output_force_rms'])
    final = state_measurement_fields(pooled, 'Layer_2')['state_displacement']
    assert final['absolute_rms'] == pytest.approx(float(torch.cat(output_deltas).double().square().mean().sqrt()))
    assert final['centered_squares'] == pooled['centered_squares']
    assert final['unchanged_elements'] == pooled['unchanged_elements']


@pytest.mark.parametrize('centered_squares', [0., 4.])
def test_zero_internal_free_rms_is_undefined_relative_displacement(centered_squares):
    from experiments.compare_cifar_block_beta_gradients import (
        accumulate_displacement, state_measurement_fields)
    measured = dict(centered_squares=1., free_squares=4., elements=2, unchanged_elements=0,
        layer_displacements=[dict(layer=1, state_name='Layer_1', centered_squares=centered_squares,
            free_squares=0., elements=4, unchanged_elements=4 if not centered_squares else 0)],
        output_force_rms=2., injected_force_rms=.2)
    pooled = {}
    accumulate_displacement(pooled, measured)
    for source in (measured, pooled):
        state = state_measurement_fields(source, 'Layer_1')['state_displacement']
        assert state['free_rms'] == 0.
        assert state['relative_rms'] is None
        assert state['absolute_rms'] == (centered_squares / 4) ** .5
        assert json.loads(json.dumps(state, allow_nan=False))['relative_rms'] is None


def test_output_only_displacement_pooling_remains_compatible():
    from experiments.compare_cifar_block_beta_gradients import accumulate_displacement
    measured = dict(centered_squares=4., free_squares=16., elements=8, unchanged_elements=2)
    pooled = {}
    accumulate_displacement(pooled, measured)
    accumulate_displacement(pooled, measured)
    assert pooled == {key: value * 2 for key, value in measured.items()}


def test_declared_environment_checked_before_replay_inputs(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from experiments.compare_cifar_block_beta_gradients import check_environment, run
    monkeypatch.setattr(torch, '__version__', '2.11.0+cu128')
    monkeypatch.setattr(torch.version, 'cuda', '12.8')
    expected = dict(torch='2.11.0+cu128', cuda='12.8')
    assert check_environment(expected) == expected
    assert check_environment(None) == expected
    for key in expected:
        mismatch = dict(expected, **{key: 'wrong-version'})
        with pytest.raises(RuntimeError, match=f'Environment mismatch for {key}'):
            check_environment(mismatch)
    with pytest.raises(ValueError, match='Unsupported expected_environment key'):
        check_environment({'torch_version': '2.11.0+cu128'})
    config = tmp_path / 'config.json'
    config.write_text(json.dumps({'expected_environment': dict(expected, cuda='12.1')}))
    with pytest.raises(RuntimeError, match='Environment mismatch for cuda'):
        run(SimpleNamespace(config=config, input_root=tmp_path / 'missing-inputs'))
