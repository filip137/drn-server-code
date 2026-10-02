"""Focused CPU contracts for the hybrid learning update and beta state."""
import copy
import json
import random
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from experiments.cifar_conv_eqprop import (
    EpochBeta, METHOD, calibration_updates, hybrid_backward, initial_beta_state, replay_without_training_effects,
)
from experiments.compare_cifar_block_beta_gradients import eqprop_gradients
from experiments.train_cifar_l8_analog import (
    configure_training_precision, make_optimizer, save_checkpoint, restore_checkpoint, verify_training_precision,
)
from labs.cifar_l8_analog import CifarL8Analog


def config(policy='epoch'):
    c=json.loads(Path('configs/cifar/cifar10_l8_analog_bs16_seed0.json').read_text())
    c['solver'].update(voltage_amp=4.,current_amp=1.)
    c['scheme']='ours'
    c.update(blocks=[[2,2,2],[3,3,3],[4,4]],input_size=8,gradient_method=METHOD)
    c['conv_eqprop']=dict(policy=policy,initial_betas=[.003,.001,.01],rms_targets=[.09]*3,
        cohort_indices=list(range(256)),relative_tolerance=.03,max_evaluations=24,cosine_epochs=[0,1,5,10],read_noise=0.)
    if policy=='periodic':c['conv_eqprop']['calibrations_per_epoch']=4
    return c


def examples():
    torch.manual_seed(7)
    return torch.randn(3,3,8,8),torch.tensor([0,1,2])


def test_hybrid_replaces_only_conv_gradients_and_updates_bn_once():
    torch.manual_seed(13);c=config();a=CifarL8Analog(c);b=CifarL8Analog(c);b.restore(a.snapshot())
    x,y=examples();saved=[];physical=[]
    def hook(block,args,out):
        physical.append(block.free_layers()[-1].state)
        saved.append(dict(input=block.boundary_layer.state.detach().clone(),states=[l.state.detach().clone() for l in block.free_layers()]))
    handles=[block.register_forward_hook(hook) for block in a.analog_blocks]
    logits=a(x,c['iterations']);summed=F.cross_entropy(logits,y,reduction='sum')
    forces=torch.autograd.grad(summed,physical,retain_graph=True)
    summed.backward()
    reference={n:p.grad.detach().clone()/len(y) for n,p in a.trainable_tensors().items()}
    for h in handles:h.remove()
    expected=[]
    for i,block in enumerate(a.analog_blocks):
        saved[i]['gradient']=forces[i].detach()
        gradients,_=eqprop_gradients(block,saved[i],c['conv_eqprop']['initial_betas'][i],c['iterations'][i],torch.device('cpu'))
        expected.extend(gradients)
    prediction,loss=hybrid_backward(b,x,y,c['iterations'],c['conv_eqprop']['initial_betas'])
    torch.testing.assert_close(prediction,logits.detach(),rtol=0,atol=0)
    torch.testing.assert_close(loss,summed.detach()/len(y))
    for n,p in b.trainable_tensors().items():
        if not n.startswith('blocks.'):
            torch.testing.assert_close(p.grad,reference[n],rtol=1e-5,atol=1e-7)
    actual=[p.grad for n,p in b.named_conductances() if n.startswith('blocks.')]
    for observed,wanted in zip(actual,expected):torch.testing.assert_close(observed.double(),wanted,rtol=1e-4,atol=1e-6)
    assert all(int(bridge[1].num_batches_tracked)==1 for bridge in b.bridges)
    assert all(not layer.state.requires_grad for block in b.analog_blocks for layer in block.free_layers())


def test_mean_gradient_does_not_scale_with_duplicate_batch():
    torch.manual_seed(15);c=config();a=CifarL8Analog(c).double();b=CifarL8Analog(c).double();b.restore(a.snapshot())
    for model in (a,b):
        model.head.weight.state=model.head.weight.state.detach().double().requires_grad_()
    x,y=examples();x=x.double()
    hybrid_backward(a,x,y,c['iterations'],c['conv_eqprop']['initial_betas'])
    hybrid_backward(b,x.repeat(2,1,1,1),y.repeat(2),c['iterations'],c['conv_eqprop']['initial_betas'])
    for (n,p),(m,q) in zip(a.named_conductances(),b.named_conductances()):
        torch.testing.assert_close(p.grad,q.grad,rtol=.01,atol=2e-5,msg=n)


@pytest.mark.parametrize('gain_mode,input_grad', [
    ('softplus', False), ('log', False), ('fixed', False), ('fixed', True),
])
def test_skip_conv_autograd_matches_float64_reference(monkeypatch, gain_mode, input_grad):
    import experiments.cifar_conv_eqprop as module
    torch.manual_seed(31)
    c = config()
    c.update(precision='float64', conv_input_gain_mode=gain_mode)
    a, b = CifarL8Analog(c), CifarL8Analog(c)
    for model in (a, b):
        configure_training_precision(model, c)
        # A pre-existing frozen tensor must keep its original flag.
        model.analog_blocks[-1].weights[-1].state.requires_grad_(False)
    b.restore(a.snapshot())
    x, y = examples()
    xa = x.double().requires_grad_(input_grad)
    xb = xa.detach().clone().requires_grad_(input_grad)
    records, conv_autograd_present = [], []
    original = module.eqprop_gradients
    def capture(block, saved, *args, **kwargs):
        records.append(copy.deepcopy(saved))
        conv_autograd_present.append(block.weights[0].state.grad is not None)
        assert block.weights[0].state.requires_grad
        return original(block, saved, *args, **kwargs)
    monkeypatch.setattr(module, 'eqprop_gradients', capture)
    oa, _ = make_optimizer(a, c)
    ob, _ = make_optimizer(b, c)
    flags = [p.requires_grad for _, p in a.named_conductances()]
    prediction, loss, rms = hybrid_backward(a, xa, y, c['iterations'],
        c['conv_eqprop']['initial_betas'], measure_displacement=True, skip_conv_autograd=False)
    actual_prediction, actual_loss, actual_rms = hybrid_backward(b, xb, y, c['iterations'],
        c['conv_eqprop']['initial_betas'], measure_displacement=True, skip_conv_autograd=True)
    # Constant weights can select different backend reductions. Require tight
    # float64 agreement, rather than bitwise equality across autograd graphs.
    torch.testing.assert_close(actual_prediction, prediction, rtol=1e-12, atol=1e-13)
    torch.testing.assert_close(actual_loss, loss, rtol=1e-12, atol=1e-13)
    assert conv_autograd_present[:3] == [True] * 3
    assert conv_autograd_present[3:] == [gain_mode == 'fixed' and not input_grad] * 3
    for wanted, actual in zip(records[:3], records[3:]):
        for field in ('input', 'gradient'):
            torch.testing.assert_close(actual[field], wanted[field], rtol=1e-12, atol=1e-14)
        for value, reference in zip(actual['states'], wanted['states']):
            torch.testing.assert_close(value, reference, rtol=1e-12, atol=1e-13)
    for wanted, actual in zip(rms, actual_rms):
        assert actual == pytest.approx(wanted, rel=1e-12, abs=1e-14)
    assert a.trainable_tensors().keys() == b.trainable_tensors().keys()
    for name, parameter in b.trainable_tensors().items():
        torch.testing.assert_close(parameter.grad, a.trainable_tensors()[name].grad,
                                   rtol=1e-10, atol=1e-12, msg=name)
    if input_grad:
        torch.testing.assert_close(xb.grad, xa.grad, rtol=1e-12, atol=1e-14)
    for (name, value), (_, reference) in zip(b.named_buffers(), a.named_buffers()):
        torch.testing.assert_close(value, reference, rtol=1e-12, atol=1e-13, msg=name)
    assert all(int(bridge[1].num_batches_tracked) == 1 for bridge in b.bridges)
    for model, optimizer in ((a, oa), (b, ob)):
        assert [p.requires_grad for _, p in model.named_conductances()] == flags
        assert all(not block._forward_hooks for block in model.analog_blocks)
        assert all(not layer.state.requires_grad
                   for block in model.analog_blocks for layer in block.free_layers())
        optimizer.step()
        model.project_()
    for group, values in a.snapshot().items():
        for name, value in values.items():
            torch.testing.assert_close(value, b.snapshot()[group][name], rtol=1e-12, atol=1e-14)


@pytest.mark.parametrize('failure_stage', ['forward', 'backward', 'eqprop'])
def test_skip_conv_autograd_restores_flags_and_hooks_on_failure(monkeypatch, failure_stage):
    import experiments.cifar_conv_eqprop as module
    c = config()
    c['precision'] = 'float64'
    model = CifarL8Analog(c)
    configure_training_precision(model, c)
    model.analog_blocks[-1].weights[-1].state.requires_grad_(False)
    parameters = list(model.named_parameters()) + list(model.named_conductances())
    flags = {name: p.requires_grad for name, p in parameters}
    x, y = examples()
    def fail(*args, **kwargs):
        if failure_stage != 'eqprop':
            assert not any(p.requires_grad for name, p in model.named_conductances()
                           if name.startswith('blocks.'))
        else:
            assert {name: p.requires_grad for name, p in parameters} == flags
        raise RuntimeError('injected failure')
    with monkeypatch.context() as patch:
        if failure_stage == 'forward':
            patch.setattr(model.head, 'forward', fail)
        elif failure_stage == 'backward':
            patch.setattr(torch.Tensor, 'backward', fail)
        else:
            patch.setattr(module, 'eqprop_gradients', fail)
        with pytest.raises(RuntimeError, match='injected failure'):
            hybrid_backward(model, x.double(), y, c['iterations'], c['conv_eqprop']['initial_betas'])
    assert {name: p.requires_grad for name, p in parameters} == flags
    assert all(not block._forward_hooks for block in model.analog_blocks)
    assert all(interaction._a_col2im_cache is None
               for block in model.analog_blocks for interaction in block.interactions)
    variables = [v for block in model.analog_blocks for v in (block.boundary_layer, *block.free_layers())]
    variables.extend((model.head.boundary, model.head.output))
    assert all(not variable.state.requires_grad for variable in variables)
    optimizer, _ = make_optimizer(model, c)
    optimizer.zero_grad(set_to_none=True)
    hybrid_backward(model, x.double(), y, c['iterations'], c['conv_eqprop']['initial_betas'])
    optimizer.step()


def test_scoped_coefficient_cache_matches_two_uncached_updates_exactly(monkeypatch):
    torch.manual_seed(53)
    c = config()
    c['precision'] = 'float64'
    a, b = CifarL8Analog(c), CifarL8Analog(c)
    for model in (a, b):
        configure_training_precision(model, c)
    b.restore(a.snapshot())
    oa, _ = make_optimizer(a, c)
    ob, _ = make_optimizer(b, c)
    x, y = examples()
    calls = [0, 0]
    for index, model in enumerate((a, b)):
        for block in model.analog_blocks:
            for interaction in block.interactions:
                original = interaction._a_col2im_uncached
                def count(original=original, index=index):
                    calls[index] += 1
                    return original()
                monkeypatch.setattr(interaction, '_a_col2im_uncached', count)
    for batch_size in (3, 2):
        outputs = []
        for model, optimizer, cache in ((a, oa, False), (b, ob, True)):
            optimizer.zero_grad(set_to_none=True)
            outputs.append(hybrid_backward(model, x[:batch_size].double(), y[:batch_size],
                c['iterations'], [1e-7, 1e-6, 1e-5], skip_conv_autograd=True,
                cache_conv_coefficients=cache))
            assert all(interaction._a_col2im_cache is None
                       for block in model.analog_blocks for interaction in block.interactions)
            optimizer.step()
            model.project_()
        for actual, expected in zip(outputs[1], outputs[0]):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        for name, parameter in b.trainable_tensors().items():
            torch.testing.assert_close(parameter.grad, a.trainable_tensors()[name].grad,
                                       rtol=0, atol=0, msg=name)
        for group, values in a.snapshot().items():
            for name, value in values.items():
                torch.testing.assert_close(value, b.snapshot()[group][name], rtol=0, atol=0)
    assert 0 < calls[1] < calls[0]


def test_replay_preserves_buffers_rng_mode_and_parameters():
    torch.manual_seed(1);c=config();m=CifarL8Analog(c);m.eval();before=m.snapshot()
    rng=torch.get_rng_state();np_rng=np.random.get_state();py_rng=random.getstate()
    x,y=examples();rng=torch.get_rng_state()
    variables=[v for block in m.analog_blocks for v in (block.boundary_layer,*block.free_layers())]
    variables.extend((m.head.boundary,m.head.output))
    states=[v.state.clone() for v in variables]
    with replay_without_training_effects(m):
        from experiments.calibrate_cifar_block_beta import capture_batches
        capture_batches(m,[(x,y)],c['iterations'],torch.device('cpu'))
        torch.rand(2);np.random.rand();random.random()
    assert not m.training and torch.equal(rng,torch.get_rng_state())
    assert np.array_equal(np_rng[1],np.random.get_state()[1]) and py_rng==random.getstate()
    for group,values in before.items():
        for name,value in values.items():assert torch.equal(value,m.snapshot()[group][name])
    assert all(torch.equal(v.state,state) for v,state in zip(variables,states))


@pytest.mark.parametrize('policy',['fixed','epoch'])
def test_calibration_cadence_and_atomic_failure(tmp_path,monkeypatch,policy):
    import experiments.cifar_conv_eqprop as module
    c=config(policy);m=CifarL8Analog(c);x,y=examples()
    controller=EpochBeta(m,c,[(x,y)],torch.device('cpu'),tmp_path,float('inf'))
    monkeypatch.setattr(module.reporting,'update_status_progress',lambda *a,**k:None)
    calls=[]
    def search(measure,target,**kw):
        calls.append(kw['initial_beta']);return kw['initial_beta']*2,[{'relative_rms':target}]
    monkeypatch.setattr(module,'search_beta',search)
    assert controller.before_epoch(1)==0 and not calls
    controller.before_epoch(2)
    assert len(calls)==(3 if policy=='epoch' else 0)
    state=copy.deepcopy(controller.state);controller.before_epoch(2);assert controller.state==state
    if policy=='epoch':
        assert state['values']==[2*v for v in c['conv_eqprop']['initial_betas']]
        def failure(*a,**kw):raise RuntimeError('unattainable')
        monkeypatch.setattr(module,'search_beta',failure)
        with pytest.raises(RuntimeError,match='unattainable'):controller.before_epoch(3)
        assert controller.state==state


@pytest.mark.parametrize('precision',['float32','float64'])
@pytest.mark.parametrize('policy',['fixed','epoch','periodic'])
def test_checkpoint_resume_restores_beta_and_next_update(tmp_path,policy,precision):
    torch.manual_seed(23);c=config(policy);a=CifarL8Analog(c);a.conv_beta_state=initial_beta_state(c)
    c['precision']=precision;configure_training_precision(a,c)
    oa,sa=make_optimizer(a,c);x,y=examples();x=x.to(getattr(torch,precision))
    hybrid_backward(a,x,y,c['iterations'],a.conv_beta_state['values']);oa.step();a.project_();sa.step()
    if policy=='epoch':a.conv_beta_state.update(values=[.004,.002,.02],calibrated_for_epoch=1)
    if policy=='periodic':
        a.conv_beta_state.update(values=[.004,.002,.02],calibrated_for_epoch=1,
            calibration_count=4,last_calibration_completed_updates=3,batches_in_calibrated_epoch=4)
    path=tmp_path/'checkpoint.pt';save_checkpoint(path,a,oa,sa,1,c)
    state=torch.load(path,map_location='cpu',weights_only=False)
    b=CifarL8Analog(c);configure_training_precision(b,c)
    b.conv_beta_state=initial_beta_state(c);ob,sb=make_optimizer(b,c)
    assert restore_checkpoint(state,b,ob,sb,c)==1
    assert a.conv_beta_state==b.conv_beta_state
    verify_training_precision(a,c,oa);verify_training_precision(b,c,ob)
    for m,o in [(a,oa),(b,ob)]:
        o.zero_grad(set_to_none=True);hybrid_backward(m,x,y,c['iterations'],m.conv_beta_state['values']);o.step();m.project_()
        assert all(p.grad.dtype==getattr(torch,precision) for p in m.trainable_tensors().values())
    for group,values in a.snapshot().items():
        for name,value in values.items():torch.testing.assert_close(value,b.snapshot()[group][name],rtol=0,atol=0)
    bad=copy.deepcopy(state);bad['conv_beta_state']['calibrated_for_epoch']=99
    with pytest.raises(ValueError,match='calibration'):restore_checkpoint(bad,b,ob,sb,c)


@pytest.mark.parametrize('batches,expected',[(1407,[0,352,704,1056]),(8,[0,2,4,6]),(5,[0,2,3,4])])
def test_periodic_schedule_uses_optimizer_progress(batches,expected):
    assert calibration_updates(batches,4)==expected
    with pytest.raises(ValueError):calibration_updates(3,4)


def test_periodic_five_epoch_cadence_warm_start_and_precision(tmp_path,monkeypatch):
    import experiments.cifar_conv_eqprop as module
    c=config('periodic');c['precision']='float64'
    c['conv_eqprop'].update(cosine_epochs=[],observation_epochs=[],calibration_batch_size=32)
    m=CifarL8Analog(c);configure_training_precision(m,c);x,y=examples()
    controller=EpochBeta(m,c,[(x,y)],torch.device('cpu'),tmp_path,float('inf'))
    monkeypatch.setattr(module.reporting,'update_status_progress',lambda *a,**k:None)
    calls=[]
    original=module.capture_batches
    def capture(model,batches,*args):
        batches=list(batches)
        assert all(values.dtype==torch.float64 for values,_ in batches)
        return original(model,batches,*args)
    monkeypatch.setattr(module,'capture_batches',capture)
    def search(measure,target,**kw):
        calls.append(kw['initial_beta'])
        assert kw['max_evaluations']==24 and kw['tolerance']==.03
        return kw['initial_beta']*1.1,[{'relative_rms':target}]
    monkeypatch.setattr(module,'search_beta',search)
    for epoch in range(1,6):
        for completed in range(7):
            prior=copy.deepcopy(controller.state)
            seconds=controller.before_update(epoch,completed,7)
            if completed in [0,2,4,6]:
                assert seconds>0
                assert calls[-3:]==prior['values']
                assert controller.last_calibration['completed_updates']==completed
                assert controller.before_update(epoch,completed,7)==0
            else:
                assert seconds==0 and controller.state==prior
        assert controller.state['calibration_count']==4*epoch
        assert controller.observe(epoch) is None
    assert len(calls)==60
    assert len(list((tmp_path/'artifacts').glob('beta_calibration_*.json')))==20
    before=copy.deepcopy(controller.state)
    def failure(*a,**kw):raise RuntimeError('unattainable')
    monkeypatch.setattr(module,'search_beta',failure)
    with pytest.raises(RuntimeError,match='unattainable'):controller.before_update(6,0,7)
    assert controller.state==before


def test_float64_checkpoint_rejects_silent_downcast(tmp_path):
    c=config('fixed');c['precision']='float64';m=CifarL8Analog(c)
    configure_training_precision(m,c);m.conv_beta_state=initial_beta_state(c)
    optimizer,scheduler=make_optimizer(m,c)
    path=tmp_path/'state.pt';save_checkpoint(path,m,optimizer,scheduler,0,c)
    state=torch.load(path,weights_only=False)
    name=next(iter(state['model']['conductances']))
    state['model']['conductances'][name]=state['model']['conductances'][name].float()
    with pytest.raises(ValueError,match='tensor precision'):
        restore_checkpoint(state,m,optimizer,scheduler,c)


def test_fixed_policy_has_no_calibration_and_optional_observations(tmp_path):
    c=config('fixed');c['conv_eqprop']['observation_epochs']=[]
    controller=EpochBeta(CifarL8Analog(c),c,[],torch.device('cpu'),tmp_path,float('inf'))
    for epoch in range(1,6):
        for completed in range(8):assert controller.before_update(epoch,completed,8)==0
        assert controller.observe(epoch) is None
    assert controller.state['calibration_count']==0
    assert controller.state['values']==c['conv_eqprop']['initial_betas']
    assert not (tmp_path/'artifacts').exists()


def test_float64_real_periodic_calibration_preserves_training_state(tmp_path,monkeypatch):
    import experiments.cifar_conv_eqprop as module
    c=config('periodic');c['precision']='float64';c['conv_eqprop']['cosine_epochs']=[]
    m=CifarL8Analog(c);configure_training_precision(m,c);x,y=examples()
    hybrid_backward(m,x.double(),y,c['iterations'],c['conv_eqprop']['initial_betas'])
    monkeypatch.setattr(module.reporting,'update_status_progress',lambda *a,**k:None)
    controller=EpochBeta(m,c,[(x,y)],torch.device('cpu'),tmp_path,float('inf'))
    observed=controller.observe(0)
    c['conv_eqprop']['rms_targets']=[block['relative_rms'] for block in observed['blocks']]
    before=m.snapshot();rng=torch.get_rng_state()
    gradients={name:p.grad.clone() for name,p in m.trainable_tensors().items()}
    assert controller.before_update(1,0,4)>0
    artifact=json.loads((tmp_path/'artifacts/beta_calibration_e001_u00000.json').read_text())
    assert artifact['complete'] and artifact['beta_state']['calibration_count']==1
    for block in artifact['blocks']:
        assert len(block['trials'])==1 and block['trials'][0]['relative_target_error']<=.03
    assert m.training and torch.equal(rng,torch.get_rng_state())
    for group,values in before.items():
        for name,value in values.items():assert torch.equal(value,m.snapshot()[group][name])
    for name,p in m.trainable_tensors().items():assert torch.equal(p.grad,gradients[name])
    verify_training_precision(m,c)


def test_boundary_observations_preserve_state_and_match_rms_paths(tmp_path,monkeypatch):
    import experiments.cifar_conv_eqprop as module
    torch.manual_seed(31);c=config();c['conv_eqprop']['cosine_epochs']=[0]
    m=CifarL8Analog(c);x,y=examples();before=m.snapshot()
    monkeypatch.setattr(module.reporting,'update_status_progress',lambda *a,**k:None)
    controller=EpochBeta(m,c,[(x,y)],torch.device('cpu'),tmp_path,float('inf'))
    with_cosine=controller.observe(0);rms_only=controller.observe(1)
    assert len(with_cosine['layers'])==8 and not rms_only['layers']
    for a,b in zip(with_cosine['blocks'],rms_only['blocks']):
        assert a['relative_rms']==pytest.approx(b['relative_rms'],rel=1e-10)
    for group,values in before.items():
        for name,value in values.items():assert torch.equal(value,m.snapshot()[group][name])
