"""Check the shared-baseline noise term by exact finite enumeration."""

from itertools import product

import numpy as np

from labs.adjoint_estimators import estimate_adjoint
from labs.tools.measure_mc_scaling import expected_mc_mse
from labs.recurrent_eqprop import make_classifier
from labs.tools import train_recurrent_eqprop_digits as digits
from labs.tools.summarize_mc_scaling import plots


def test_mc_noise_formula_accounts_for_one_shared_baseline():
    n,m = 3,2
    truth = np.array([0.7,-0.2,0.4])
    baseline = np.array([0.1,0.3,-0.1])
    vq,vy = 0.03,0.07
    designs = np.array(list(product((-1,1),repeat=n*m))).reshape(-1,m,n)/np.sqrt(n)
    q_noise = np.array(list(product((-1,1),repeat=n)))*np.sqrt(vq)
    y_noise = np.array(list(product((-1,1),repeat=m)))*np.sqrt(vy)
    q = baseline+q_noise[None,:,None,:]
    u = designs[:,None,None,:,:]
    y = np.einsum("...mn,n->...m",u,truth)+y_noise[None,None,:,:]
    estimate = estimate_adjoint(q,u,y,"mc")
    measured = np.mean(np.sum((estimate-truth)**2,axis=-1))
    expected = expected_mc_mse(n,m,np.sum((truth-baseline)**2),vq,vy)
    np.testing.assert_allclose(measured,expected,rtol=1e-14,atol=1e-14)
    error = (estimate-truth).reshape(-1,n)
    delta = truth-baseline
    covariance = (np.sum(delta**2)*np.eye(n)+np.outer(delta,delta)-2*np.diag(delta**2)
                  +((n-1)*vq+n*vy)*np.eye(n))/m
    np.testing.assert_allclose(error.mean(axis=0),0,atol=1e-14)
    np.testing.assert_allclose(error.T@error/len(error),covariance,rtol=1e-14,atol=1e-14)


def test_effectively_zero_noise_control_preserves_probe_sequence(monkeypatch):
    original = digits.make_probes
    captured = []

    def record(*args,**kwargs):
        probes = original(*args,**kwargs)
        captured.append(probes.copy())
        return probes

    monkeypatch.setattr(digits,"make_probes",record)
    x = np.random.default_rng(81).normal(size=(3,3))
    labels = np.array([0,1,0])
    streams = []
    for sigma in (1e-5,1e-30):
        model = make_classifier(80,size=16,outputs=2,input_size=3)
        rng = np.random.default_rng(82)
        for _ in range(2):
            gradient,_ = digits.gradient_step(model,x,labels,"mc8",rng,beta=0.01,sigma=sigma)
            model.update(gradient,0.1)
        streams.append(rng.bit_generator.state)
    assert streams[0] == streams[1]
    for noisy,clean in zip(captured[:6],captured[6:]):
        np.testing.assert_array_equal(noisy,clean)


def test_plot_regeneration_accepts_float_dimensions_from_csv(tmp_path):
    # read_metrics intentionally converts numeric CSV columns to floats.
    rows = [dict(size=64.0,method=method,epoch=15.0,validation_accuracy=0.9)
            for method in ("adjoint","mc8","mc16")]
    summary = [dict(size=64,method=method,test_accuracy_mean=0.9,test_accuracy_std=0.01,
                    equilibrations_per_example=phases)
               for method,phases in (("adjoint",1),("mc8",19),("mc16",35))]
    plots(tmp_path,rows,summary)
    assert (tmp_path/"scaling.png").stat().st_size > 0
    assert (tmp_path/"scaling.svg").stat().st_size > 0
