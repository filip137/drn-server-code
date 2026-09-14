"""Independent derivatives and physical-access gates for the paper protocol."""

import copy
import math

import numpy as np
import pytest
import torch

from labs.fmnist_homeostasis import (PaperNet, MeasuredFeedback, activation, slope,
    free_state, forward_tangent, measured_responses, homeostasis, feedback, training_gradient)
from labs.directed_eqprop import local_gradient


@pytest.fixture
def case():
    torch.set_num_threads(1)
    model=PaperNet(inputs=4,hidden=3,outputs=2,seed=0,dtype=torch.float64)
    x=torch.tensor([[.2,.8,.4,.1],[.9,.2,.3,.6]],dtype=torch.float64)
    y=torch.tensor([0,1])
    return model,x,y


def explicit_trajectory(model, u, drive, steps, beta=None, labels=None, current=None):
    for _ in range(steps):
        r=activation(u)
        h=model.hidden
        a=r[:,:h]; b=r[:,h:2*h]; o=r[:,2*h:]
        new=torch.cat((b@model.backward1.T,
            a@model.forward1.T+o@model.backward2.T,b@model.forward2.T),dim=-1)+drive
        if beta is not None:
            new=new-beta*model.cost(u,labels)[2]
        if current is not None:
            new=new-current
        u=new
    return activation(u)


def test_tangent_matches_autodiff_of_finite_nonstationary_trajectory(case):
    model,x,y=case
    drive=model.drive(x).detach()
    # Intentionally start away from equilibrium: tangent must follow the states.
    u=torch.zeros_like(drive)
    f=lambda beta:explicit_trajectory(model,u,drive,7,beta=beta,labels=y)
    expected=torch.autograd.functional.jacobian(f,torch.tensor(0.,dtype=u.dtype))
    actual=forward_tangent(u,drive,torch.nn.functional.one_hot(y,2).double(),
        *model.weights(),model.readout,model.readout_bias,7)
    torch.testing.assert_close(actual,expected,rtol=1e-11,atol=1e-12)


def test_complete_physical_coordinate_probes_recover_finite_response_adjoint(case):
    model,x,y=case
    x,y=x[:1],y[:1]
    u,drive=free_state(model,x)
    c=model.cost(u,y)[2].detach()
    f=lambda current:explicit_trajectory(model,u,drive,20,current=current.unsqueeze(0))[0]
    response=torch.autograd.functional.jacobian(f,torch.zeros(model.n,dtype=u.dtype))
    z=torch.eye(model.n,dtype=u.dtype)[:,None,:]
    measured=measured_responses(model,drive,u,z,1e-4,20)
    ell=(measured*c).sum(-1).squeeze(-1)
    torch.testing.assert_close(ell,response.T@c[0],rtol=2e-6,atol=1e-8)


def test_true_local_gradient_matches_parameter_directional_finite_difference(case):
    model,x,y=case
    with torch.no_grad():
        u,drive=free_state(model,x,300)
        assert float(model.force(u,drive).abs().max())<1e-11
        c=model.cost(u,y)[2]
        ju=model.dense_w()*slope(u).unsqueeze(-2)-torch.eye(model.n,dtype=u.dtype)
        ell=torch.linalg.solve(ju.transpose(-1,-2),(slope(u)*c).unsqueeze(-1)).squeeze(-1)
        grads=local_gradient(model,x,u,y,ell)
        gen=torch.Generator().manual_seed(400)
        for name,p in model.named_parameters():
            direction=torch.randn(p.shape,dtype=p.dtype,generator=gen)
            direction/=direction.norm()
            saved=p.clone()
            values=[]
            for sign in (1,-1):
                p.copy_(saved+sign*1e-5*direction)
                changed,_=free_state(model,x,300)
                values.append(float(model.cost(changed,y)[0].mean()))
            p.copy_(saved)
            assert float((grads[name]*direction).sum())==pytest.approx((values[0]-values[1])/2e-5,abs=1e-9,rel=1e-5)


def test_homeostasis_matches_exact_penalty_and_parameter_derivatives(case):
    model,_,_=case
    reg,value=homeostasis(model,1,None,eps=math.sqrt(model.n)*torch.eye(model.n,dtype=model.bias.dtype))
    w=model.dense_w()
    exact=(w-w.T).square().sum()/(2*model.n)
    assert value==pytest.approx(float(exact),abs=1e-12)
    params=dict(model.named_parameters())
    derivatives=torch.autograd.grad(exact,tuple(params.values()),allow_unused=True)
    for (name,p),g in zip(params.items(),derivatives):
        torch.testing.assert_close(reg[name],torch.zeros_like(p) if g is None else g)


@pytest.mark.parametrize('method',['vf_central','zo_learned4'])
def test_measured_training_uses_no_derivative_or_transpose_or_projection(case,monkeypatch,method):
    model,x,y=case
    def forbidden(*a,**k):
        raise AssertionError('Physical path used derivative/oracle/projection')
    monkeypatch.setattr(torch.autograd,'grad',forbidden)
    monkeypatch.setattr(torch.linalg,'solve',forbidden)
    monkeypatch.setattr(model,'dense_w',forbidden)
    monkeypatch.setattr(model,'recurrent_transpose',forbidden)
    monkeypatch.setattr(model,'project',forbidden)
    learner=MeasuredFeedback(model.n,model.outputs) if method=='zo_learned4' else None
    grads,work,_=training_gradient(model,x,y,method,torch.Generator().manual_seed(1),learner=learner)
    assert all(torch.isfinite(g).all() for g in grads.values())
    assert work.tangent_jvps==work.homeostasis_jvps==0
    assert work.measured_nudge_phases==len(x)*(8 if learner else 2)
    if learner:
        assert learner.observations==4*len(x)


def test_predictor_updates_follow_current_correction_and_readonly_does_not_update(case):
    model,x,y=case
    predictor=MeasuredFeedback(model.n,model.outputs)
    frozen=copy.deepcopy(predictor)
    a=feedback(model,x,y,'zo_learned4',torch.Generator().manual_seed(3),learner=predictor)
    b=feedback(model,x,y,'zo_learned4',torch.Generator().manual_seed(3),learner=frozen,update_learner=False)
    torch.testing.assert_close(a[0],b[0],rtol=0,atol=0)
    assert np.any(predictor.matrix!=0)
    assert np.all(frozen.matrix==0)


def test_paper_initialization_untied_zero_bias_and_reverse_fan_in():
    model=PaperNet()
    assert float(model.bias.norm())==0
    assert model.n==522
    assert model.forward1.data_ptr()!=model.backward1.data_ptr()
    assert not torch.allclose(model.forward1,model.backward1.T)
    assert float(model.backward2.std())==pytest.approx(1/math.sqrt(10),rel=.05)
    assert float(model.forward2.std())==pytest.approx(1/math.sqrt(256),rel=.05)
    assert float(torch.linalg.matrix_norm(model.dense_w(),ord=2))>1


def test_fashion_loader_rejects_mnist_or_corrupt_files(monkeypatch):
    from labs import fashion_eqprop_data as data
    monkeypatch.setattr(data,'read_idx',lambda p:(np.zeros(1,dtype=np.uint8),'incorrect'))
    with pytest.raises(ValueError,match='Expected verified Fashion-MNIST'):
        data.load_fashion_mnist('/unused')


def test_replay_detects_two_cycle_even_when_even_horizons_agree():
    from labs.tools.report_fmnist_homeostasis import replay
    # Strong negative reciprocal feedback produces an attracting period-two
    # cycle. Comparing 150 with 300 alone would mistakenly report convergence.
    w=dict(input=np.zeros((1,2)),forward1=np.array([[-4.]]),backward1=np.array([[-4.]]),
           forward2=np.array([[4.],[-4.]]),backward2=np.zeros((1,2)),
           bias=np.array([2.,2.,-2.,2.]),readout=np.eye(2),readout_bias=np.zeros(2))
    x=np.zeros((2,2));y=np.array([0,1])
    even=replay(w,x,y,150)
    doubled=replay(w,x,y,300)
    assert even['accuracy']==doubled['accuracy']
    assert even['nonconverged_examples']==2
    assert even['period_two_examples']==2
    assert even['prediction_changes_after_one_step']==2
