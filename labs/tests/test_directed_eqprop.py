"""Scientific gates for directed weights, response coordinates and homeostasis."""

import math

import pytest
import torch

from labs.directed_eqprop import (Counts, DirectedNet, LearnedFeedback, activation, adjoint,
    audit, feedback, homeostasis, local_gradient, paired_response, settle, slope,
    training_gradient)


@pytest.fixture
def case():
    torch.set_num_threads(1)
    model = DirectedNet(inputs=4, hidden=3, outputs=2, seed=31, alpha=90)
    x = torch.tensor([[.2, -.4, .7, 1.1], [-.5, .3, -.2, .1]], dtype=torch.float64)
    y = torch.tensor([0, 1])
    return model, x, y


def test_angle_sweep_preserves_forward_draws_and_keeps_reverse_parameters_untied():
    models = [DirectedNet(inputs=4, hidden=3, outputs=2, seed=7, alpha=a) for a in (0,45,90)]
    first = models[0]
    assert first.forward1.data_ptr() != first.backward1.data_ptr()
    torch.testing.assert_close(first.forward1, first.backward1.T, rtol=0, atol=0)
    for model in models[1:]:
        torch.testing.assert_close(model.forward1, first.forward1, rtol=0, atol=0)
        torch.testing.assert_close(model.forward2, first.forward2, rtol=0, atol=0)
        torch.testing.assert_close(model.input, first.input, rtol=0, atol=0)
    assert not torch.allclose(models[-1].forward1, models[-1].backward1.T)
    for model in models:
        assert float(torch.linalg.matrix_norm(model.dense_w(), ord=2)) <= model.cap+1e-12


def test_activity_and_membrane_adjoint_coordinates_give_same_parameter_gradient(case):
    model, x, y = case
    with torch.no_grad():
        u = settle(model, model.drive(x), tolerance=1e-13)
        c = model.cost(u,y)[2]
        ju, jr = model.jacobian(u), model.jacobian(u,"activity")
        lam_u = torch.linalg.solve(ju.transpose(-1,-2), (slope(u)*c).unsqueeze(-1)).squeeze(-1)
        lam_r = torch.linalg.solve(jr.transpose(-1,-2), c.unsqueeze(-1)).squeeze(-1)
        torch.testing.assert_close(lam_u, lam_r, rtol=1e-11, atol=1e-12)
        torch.testing.assert_close(adjoint(model,u,c), lam_r, rtol=1e-9, atol=1e-11)
        torch.testing.assert_close(ju, jr*slope(u).unsqueeze(-2))
        ja = (jr-jr.transpose(-1,-2))/2
        jc = model.jacobian(u,"code")
        torch.testing.assert_close(ja, ((jc-jc.T)/2).expand_as(ja))


@pytest.mark.parametrize("name", ["input", "forward1", "forward2", "backward1", "backward2", "bias", "readout", "readout_bias"])
def test_exact_local_parameter_gradient_matches_cost_finite_difference(case, name):
    model, x, y = case
    with torch.no_grad():
        u = settle(model, model.drive(x), tolerance=1e-13)
        c = model.cost(u,y)[2]
        gradient = local_gradient(model,x,u,y,adjoint(model,u,c))
        p = dict(model.named_parameters())[name]
        index = (0,)*(p.ndim)
        original = float(p[index])
        costs = []
        for sign in (1,-1):
            p[index] = original+sign*1e-5
            changed = settle(model,model.drive(x),tolerance=1e-13)
            costs.append(float(model.cost(changed,y)[0].mean()))
        p[index] = original
        assert float(gradient[name][index]) == pytest.approx((costs[0]-costs[1])/2e-5, abs=2e-9, rel=2e-5)


@pytest.mark.parametrize("method,coordinate", [("vf_membrane","membrane"),("ep_activity","activity")])
def test_finite_eqprop_responses_match_their_respective_jacobians(case, method, coordinate):
    model, x, y = case
    ell,u,_,_ = feedback(model,x,y,method,torch.Generator().manual_seed(4),beta=1e-4,tolerance=1e-13)
    with torch.no_grad():
        c = model.cost(u,y)[2]
        if coordinate == "membrane":
            c = c*slope(u)
        expected = torch.linalg.solve(model.jacobian(u,coordinate),c.unsqueeze(-1)).squeeze(-1)
        torch.testing.assert_close(ell,expected,atol=2e-8,rtol=2e-6)


def test_symmetric_weights_make_activity_ep_exact_but_not_raw_membrane_ep(case):
    _,x,y = case
    model = DirectedNet(inputs=4,hidden=3,outputs=2,seed=1,alpha=0)
    with torch.no_grad():
        model.input.mul_(8)
    r,u,_,_ = feedback(model,x,y,"ep_activity",torch.Generator().manual_seed(2),beta=1e-4,tolerance=1e-13)
    raw,_,_,_ = feedback(model,x,y,"vf_membrane",torch.Generator().manual_seed(2),beta=1e-4,tolerance=1e-13)
    truth = adjoint(model,u,model.cost(u,y)[2])
    torch.testing.assert_close(r,truth,atol=2e-8,rtol=2e-6)
    assert float((raw-truth).norm()/truth.norm()) > .01


def test_full_coordinate_nudges_recover_adjoint_and_random_projection_identity(case):
    model,x,y = case
    with torch.no_grad():
        x,y=x[:1],y[:1]
        drive=model.drive(x)
        u=settle(model,drive,tolerance=1e-13)
        c=model.cost(u,y)[2]
        truth=adjoint(model,u,c)
        z=torch.eye(model.n,dtype=torch.float64)
        counts=Counts()
        rz=paired_response(model,drive.repeat(model.n,1),u.repeat(model.n,1),z,
            beta=1e-4,tolerance=1e-13,counts=counts)
        measured=(rz*c).sum(-1)
        torch.testing.assert_close(measured,truth[0],atol=2e-8,rtol=2e-6)
        q=torch.linalg.solve(model.jacobian(u,"activity"),c.unsqueeze(-1)).squeeze(-1)[0]
        torch.testing.assert_close(measured-z@q,z@(truth[0]-q),atol=2e-8,rtol=2e-6)
        assert counts.equilibrations==2*model.n
        assert counts.excitation_sq==pytest.approx(2*model.n*1e-8)


def test_homeostasis_jvp_expectation_and_parameter_gradient_and_descent(case):
    model,_,_=case
    eps=math.sqrt(model.n)*torch.eye(model.n,dtype=torch.float64)
    grads,loss=homeostasis(model,1,torch.Generator(),eps=eps)
    w=model.dense_w()
    expected=(w-w.T).square().sum()/(2*model.n)
    assert loss==pytest.approx(float(expected),abs=1e-12)
    names,params=zip(*model.named_parameters())
    exact=torch.autograd.grad(expected,params,allow_unused=True)
    for name,p,g in zip(names,params,exact):
        torch.testing.assert_close(grads[name],torch.zeros_like(p) if g is None else g)
    assert float(grads["backward1"].norm())>0
    with torch.no_grad():
        for name,p in model.named_parameters():
            p.add_(grads[name],alpha=-.1)
        after=model.dense_w()
        assert float((after-after.T).norm())<float((w-w.T).norm())


@pytest.mark.parametrize("method", ["vf_membrane","ep_activity","ep_probe4","learned_probe4"])
def test_physical_training_never_accesses_oracle_or_autograd(case,monkeypatch,method):
    model,x,y=case
    def forbidden(*args,**kwargs):
        raise AssertionError("Oracle or autodiff used by physical learning rule")
    monkeypatch.setattr(model,"jacobian",forbidden)
    monkeypatch.setattr(model,"dense_w",forbidden)
    monkeypatch.setattr(model,"recurrent_transpose",forbidden)
    monkeypatch.setattr(torch.linalg,"solve",forbidden)
    monkeypatch.setattr(torch.autograd,"grad",forbidden)
    learner=LearnedFeedback(model.n,model.outputs) if method.startswith("learned") else None
    grads,counts,_=training_gradient(model,x,y,method,torch.Generator().manual_seed(7),learner=learner)
    assert all(torch.isfinite(g).all() for g in grads.values())
    assert counts.equilibrations==len(x)*(9 if learner is not None else 11 if "probe" in method else 3)
    assert counts.homeostasis_jvps==0
    assert counts.oracle_adjoint_iterations==0


def test_learned_baseline_uses_past_measurements_and_audit_does_not_mutate_it(case):
    import copy
    import numpy as np
    model,x,y=case
    learner=LearnedFeedback(model.n,model.outputs)
    frozen=copy.deepcopy(learner)
    result=feedback(model,x,y,"learned_probe4",torch.Generator().manual_seed(4),learner=learner)
    reference=feedback(model,x,y,"learned_probe4",torch.Generator().manual_seed(4),learner=frozen,update_learner=False)
    torch.testing.assert_close(result[0],reference[0],rtol=0,atol=0)
    assert learner.observations==4*len(x)
    assert np.linalg.norm(learner.matrix)>0
    before=learner.matrix.copy()
    audit(model,x,y,"learned_probe4",learner=learner)
    np.testing.assert_array_equal(before,learner.matrix)
    assert learner.observations==4*len(x)


def test_homeostasis_control_records_digital_work_and_symmetry_can_evolve(case):
    _,x,y=case
    model=DirectedNet(inputs=4,hidden=3,outputs=2,seed=7,alpha=0)
    gradient,counts,_=training_gradient(model,x,y,"ep_homeo",torch.Generator().manual_seed(7))
    assert counts.homeostasis_gaussian_vectors==5*len(x)
    assert counts.homeostasis_jvps==10*len(x)
    with torch.no_grad():
        for name,p in model.named_parameters():
            p.add_(gradient[name],alpha=-.01)
        model.project()
        assert not torch.allclose(model.forward1,model.backward1.T,atol=1e-10,rtol=1e-10)


def test_checkpoint_replay_uses_independent_numpy_equations(case):
    import numpy as np
    from labs.tools.report_directed_eqprop_mnist import numpy_predict
    model,x,y=case
    with torch.no_grad():
        u=settle(model,model.drive(x),tolerance=1e-12)
        logits=model.cost(u,y)[1].numpy()
        weights={k:v.detach().numpy() for k,v in model.state_dict().items()}
    replay=numpy_predict(weights,x.numpy(),tolerance=1e-12)
    np.testing.assert_allclose(replay,logits,rtol=1e-12,atol=1e-12)
