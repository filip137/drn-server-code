"""Check control-variate independence, physical budgets and online learning."""

import copy
from itertools import product

import numpy as np
import pytest

from labs.adjoint_baselines import AveragedMomentum, MeasuredBaseline, local_slope_baseline
from labs.adjoint_estimators import estimate_adjoint
from labs.random_nudge_hopfield import Hopfield
from labs.recurrent_eqprop import flatten_gradient, make_classifier
from labs.tools import train_recurrent_eqprop_digits as digits
from labs.tools.test_nudge_improvements import LEARNING_RATES, MOMENTA, select_configurations


def test_arbitrary_frozen_baseline_keeps_mc_mean_and_variance():
    rng = np.random.default_rng(50)
    truth, baseline = rng.normal(size=(2, 4))
    probes = np.array(list(product([-0.5, 0.5], repeat=4)))[:, None, :]
    estimates = estimate_adjoint(baseline, probes, (probes @ truth[..., None])[..., 0], "mc")
    np.testing.assert_allclose(estimates.mean(axis=0), truth, atol=1e-14)
    assert np.mean(np.sum((estimates-truth)**2, axis=1)) == pytest.approx(3*np.sum((baseline-truth)**2))


@pytest.mark.parametrize("method,m", [("local_mc4",4),("local_mc8",8),("learned_mc4",4),("learned_mc8",8)])
def test_local_and_learned_steps_need_no_error_nudge_or_jacobian(monkeypatch, method, m):
    model = make_classifier(51, size=16, outputs=2, input_size=3)
    x = np.random.default_rng(52).normal(size=(3, 3))

    def forbidden(*args, **kwargs):
        raise AssertionError("Physical method accessed an error pair or exact Jacobian")

    monkeypatch.setattr(digits, "error_pair", forbidden)
    monkeypatch.setattr(digits, "oracle_adjoint", forbidden)
    monkeypatch.setattr(Hopfield, "jacobian", forbidden)
    monkeypatch.setattr(np.linalg, "solve", forbidden)
    learner = MeasuredBaseline.zeros(16, 2)
    gradient, meta = digits.gradient_step(model, x, np.array([0,1,0]), method,
                                          np.random.default_rng(53), beta=0.01, sigma=1e-5,
                                          learner=learner, audit=False)
    assert np.isfinite(flatten_gradient(gradient)).all()
    assert meta["equilibrations"] == meta["state_reads"] == len(x)*(1+2*m)
    assert meta.get("total_error_excitation_sq", 0) == 0
    assert meta["total_probe_excitation_sq"] == pytest.approx(2*len(x)*m*0.01**2)
    assert meta["max_residual"] <= 1e-9


def test_predictor_learns_measured_equations_and_has_no_oracle_targets():
    rng = np.random.default_rng(54)
    n, o, batch, m = 8, 2, 64, 8
    truth = rng.normal(scale=0.1, size=(n,o))
    states = rng.normal(size=(batch,n))
    cost = np.zeros_like(states)
    cost[:, -o:] = rng.normal(size=(batch,o))
    probes = (2*rng.integers(0,2,size=(batch,m,n))-1)/np.sqrt(n)
    reference = (-cost + cost[:,-o:] @ truth.T)/(1+0.75*states**2)
    readings = np.einsum("bmn,bn->bm", probes, reference)
    learner = MeasuredBaseline.zeros(n,o)
    before = np.linalg.norm(learner.predict(states,cost,0.25)-reference)
    learner.observe(states,cost,0.25,probes,readings)
    after = np.linalg.norm(learner.predict(states,cost,0.25)-reference)
    assert after < before*0.03
    assert learner.observations == batch*m


def test_current_gradient_uses_frozen_baseline_then_updates_predictor():
    model = make_classifier(55,size=16,outputs=2,input_size=3)
    x = np.random.default_rng(56).normal(size=(4,3))
    labels = np.array([0,1,0,1])
    learner = MeasuredBaseline.zeros(16,2)
    # Its first prediction equals the local baseline, so identical new probes
    # must produce identical gradients even though observe changes H afterward.
    local, _ = digits.gradient_step(model,x,labels,"local_mc4",np.random.default_rng(57),beta=0.01,sigma=1e-5)
    learned, meta = digits.gradient_step(model,x,labels,"learned_mc4",np.random.default_rng(57),
                                         beta=0.01,sigma=1e-5,learner=learner,audit=True)
    np.testing.assert_array_equal(flatten_gradient(local),flatten_gradient(learned))
    assert meta["predictor_previous_observations"] == 0
    assert learner.observations == 16
    assert np.linalg.norm(learner.matrix) > 0
    clones = [copy.deepcopy(learner),copy.deepcopy(learner)]
    results = [digits.gradient_step(model,x,labels,"learned_mc4",np.random.default_rng(58),
                                    beta=0.01,sigma=1e-5,learner=p,audit=a)
               for p,a in zip(clones,[False,True])]
    np.testing.assert_array_equal(flatten_gradient(results[0][0]),flatten_gradient(results[1][0]))
    np.testing.assert_array_equal(clones[0].matrix,clones[1].matrix)


def test_bias_corrected_momentum_matches_weighted_history_and_preserves_constant_gradient():
    opt = AveragedMomentum(0.9)
    gradients = [np.array([1.,2.]),np.array([3.,-1.]),np.array([-2.,5.])]
    for i,g in enumerate(gradients):
        actual, = opt.direction((g,))
        weights = 0.9**np.arange(i,-1,-1)
        expected = np.average(gradients[:i+1],axis=0,weights=weights)
        np.testing.assert_allclose(actual,expected,atol=1e-14)
    for mu in (0.0,0.9):
        opt = AveragedMomentum(mu)
        for _ in range(20):
            np.testing.assert_allclose(opt.direction((gradients[0],))[0],gradients[0],atol=1e-14)


def test_screen_selection_uses_validation_and_requires_complete_grid():
    rows = [dict(method="local_mc4",epoch=8,learning_rate=lr,momentum=mu,
                 validation_accuracy=0.9,validation_loss=lr+mu)
            for lr in LEARNING_RATES for mu in MOMENTA]
    chosen = select_configurations(rows,8,["local_mc4"])["selections"][0]
    assert chosen["learning_rate"] == 0.03 and chosen["momentum"] == 0
    with pytest.raises(ValueError,match="complete"):
        select_configurations(rows[:-1],8,["local_mc4"])
    rows[0]["test_accuracy"] = 1.0
    with pytest.raises(ValueError,match="validation-only"):
        select_configurations(rows,8,["local_mc4"])


def test_learned_training_screen_does_not_evaluate_test_and_saves_predictor(monkeypatch,tmp_path):
    rng = np.random.default_rng(59)
    data = dict(train_x=rng.normal(size=(4,3)), train_y=np.array([0,1,0,1]),
                val_x=rng.normal(size=(2,3)), val_y=np.array([1,0]),
                test_x=rng.normal(size=(2,3)),test_y=np.array([0,1]))
    original = digits.evaluate

    def checked(model,x,y):
        assert x is not data["test_x"] and y is not data["test_y"]
        return original(model,x,y)

    monkeypatch.setattr(digits,"evaluate",checked)
    cfg = dict(size=16,outputs=2,asymmetry=1.,epochs=2,learning_rate=.1,momentum=.9,
               batch_size=4,beta=.01,read_noise=1e-5,audit=True)
    rows = digits.train_one(data,"learned_mc4",0,cfg,tmp_path,lambda *a,**k:None,calibration=True)
    assert rows[-1]["equilibrations"] == 2*4*9
    assert all("test_accuracy" not in r for r in rows)
    assert "applied_update_cosine" in rows[0]
    with np.load(tmp_path/"learned_mc4_seed0_lr0.1.npz") as stored:
        assert stored["predictor_observations"] == 2*4*4
        assert stored["predictor_matrix"].shape == (16,2)
        assert stored["momentum"] == 0.9
