"""Physical sign routing must match the constrained nonlinear task gradient."""

from dataclasses import replace

import numpy as np
import pytest

from labs.random_nudge_hopfield import Hopfield
from labs.recurrent_eqprop import (
    cost_gradient, error_pair, flatten_gradient, loss_accuracy,
    parameter_gradient, settle,
)
from labs.twisted_reciprocity import (
    make_twisted_classifier, twisted_contrastive_gradient, twisted_gradient_step,
)


def _case(seed=12, asymmetry=1.5):
    model = make_twisted_classifier(seed, size=12, outputs=3, input_size=4,
                                    asymmetry=asymmetry)
    model.bias = np.linspace(-0.2, 0.15, model.size)
    x = np.random.default_rng(seed+1).normal(scale=0.8, size=(5, 4))
    labels = np.array([0, 1, 2, 1, 0])
    network = model.network()
    free = settle(network, model.drive(x), tolerance=1e-13).state
    c = cost_gradient(free, labels, model.hidden, model.logit_scale)
    adjoint = np.linalg.solve(network.jacobian(free).swapaxes(-1, -2), c[..., None])[..., 0]
    gradient = parameter_gradient(model, x, free, adjoint)
    reference = (gradient[0]*model.symmetric_mask, gradient[1], gradient[2])
    return model, network, x, labels, free, adjoint, reference


@pytest.mark.parametrize("seed", [2, 12, 29])
def test_global_nonlinear_tangent_identity_and_output_routing(seed):
    model = make_twisted_classifier(seed, size=16, outputs=3, input_size=4,
                                    asymmetry=2.0)
    network = model.network()
    states = np.random.default_rng(seed).normal(size=(7, model.size))
    jacobians = network.jacobian(states)
    signs = model.signs
    np.testing.assert_allclose(jacobians*signs[:, None]*signs[None, :],
                               jacobians.swapaxes(-1, -2), atol=1e-14)
    assert np.linalg.norm(jacobians-jacobians.swapaxes(-1, -2)) > 1
    c = cost_gradient(states, np.arange(len(states)) % 3, model.hidden)
    q = np.linalg.solve(jacobians, c[..., None])[..., 0]
    adjoint = np.linalg.solve(jacobians.swapaxes(-1, -2), c[..., None])[..., 0]
    np.testing.assert_allclose(q*signs, adjoint, atol=2e-14, rtol=2e-13)
    assert np.linalg.norm(q-adjoint)/np.linalg.norm(adjoint) > 0.1


@pytest.mark.parametrize("asymmetry", [0.7, 2.0])
def test_actual_nonlinear_cost_phases_and_all_signed_contrastive_gradients(asymmetry):
    model, network, x, labels, free, adjoint, reference = _case(asymmetry=asymmetry)
    pair = error_pair(model, network, x, labels, free, 1e-4, tolerance=1e-13)
    np.testing.assert_allclose(pair.response*model.signs, adjoint, atol=3e-8, rtol=3e-7)
    for state, sign in ((pair.positive, 1), (pair.negative, -1)):
        residual = network.force(state, model.drive(x))
        residual -= sign*pair.effective_beta[:, None]*cost_gradient(
            state, labels, model.hidden, model.logit_scale)
        assert np.max(np.abs(residual)) <= 1.01e-13
    gradient = twisted_contrastive_gradient(model, x, pair)
    for actual, expected in zip(gradient, reference):
        np.testing.assert_allclose(actual, expected, atol=3e-8, rtol=3e-7)
    np.testing.assert_array_equal(gradient[0][~model.symmetric_mask], 0)
    np.testing.assert_allclose(gradient[0], gradient[0].T, atol=1e-14)


def test_signed_rule_matches_task_parameter_finite_differences():
    model, _, x, labels, _, _, _ = _case()
    gradient, _ = twisted_gradient_step(model, x, labels, beta=1e-4, tolerance=1e-13)
    rng = np.random.default_rng(31)
    raw = rng.normal(size=model.symmetric.shape)
    recurrent = (raw+raw.T)/2*model.symmetric_mask
    np.fill_diagonal(recurrent, 0)
    directions = (recurrent, rng.normal(size=model.inputs.shape), rng.normal(size=model.bias.shape))

    def loss(candidate):
        state = settle(candidate.network(), candidate.drive(x), tolerance=1e-13).state
        return loss_accuracy(state, labels, candidate.hidden, candidate.logit_scale)[0]

    epsilon = 1e-5
    for index, name in enumerate(("symmetric", "inputs", "bias")):
        direction = directions[index]/np.linalg.norm(directions[index])
        value = getattr(model, name)
        measured = (loss(replace(model, **{name: value+epsilon*direction}))
                    - loss(replace(model, **{name: value-epsilon*direction})))/(2*epsilon)
        np.testing.assert_allclose(np.sum(gradient[index]*direction), measured,
                                   atol=3e-8, rtol=3e-6)


def test_centered_signed_contrast_has_quadratic_finite_nudge_bias():
    model, _, x, labels, _, _, reference = _case()
    errors = []
    for beta in (0.008, 0.004):
        gradient, _ = twisted_gradient_step(model, x, labels, beta=beta, tolerance=1e-13)
        errors.append(np.linalg.norm(flatten_gradient(gradient)-flatten_gradient(reference)))
    assert errors[0]/errors[1] == pytest.approx(4.0, rel=0.05)


def test_physical_step_uses_three_phases_without_derivative_or_probe_access(monkeypatch):
    model, _, x, labels, _, _, reference = _case()

    def forbidden(*args, **kwargs):
        raise AssertionError("Physical sign routing used a forbidden derivative, solve, or random probe")

    monkeypatch.setattr(Hopfield, "jacobian", forbidden)
    for name in ("solve", "inv", "pinv", "lstsq"):
        monkeypatch.setattr(np.linalg, name, forbidden)
    # With no read noise, even an RNG whose every operation fails is sufficient.
    class NoRandomness:
        def __getattr__(self, name):
            return forbidden

    gradient, meta = twisted_gradient_step(model, x, labels, NoRandomness(),
                                           beta=1e-4, tolerance=1e-13)
    for actual, expected in zip(gradient, reference):
        np.testing.assert_allclose(actual, expected, atol=3e-8, rtol=3e-7)
    assert meta["equilibrations"] == meta["state_reads"] == 3*len(x)
    assert meta["probe_count"] == meta["total_probe_excitation_sq"] == 0
    assert meta["physical_phases_per_example"] == 3
    assert meta["max_residual"] <= 1e-13


def test_updates_preserve_architecture_stability_and_reduce_local_task_loss():
    model, _, x, labels, free, _, _ = _case()
    initial_loss = loss_accuracy(free, labels, model.hidden, model.logit_scale)[0]
    initial_inputs, initial_skew = model.inputs.copy(), model.skew.copy()
    for _ in range(4):
        gradient, _ = twisted_gradient_step(model, x, labels, beta=1e-4, tolerance=1e-12)
        model.update(gradient, learning_rate=0.01)
        np.testing.assert_array_equal(model.symmetric[~model.symmetric_mask], 0)
        np.testing.assert_array_equal(model.skew, initial_skew)
        assert model.network().margin >= 1-model.symmetric_cap-1e-14
    state = settle(model.network(), model.drive(x), tolerance=1e-12).state
    assert loss_accuracy(state, labels, model.hidden, model.logit_scale)[0] < initial_loss
    assert np.linalg.norm(model.inputs-initial_inputs) > 1e-5
    # Stress the architectural and spectral projection with forbidden entries.
    rng = np.random.default_rng(32)
    for _ in range(3):
        assert model.update((rng.normal(size=model.symmetric.shape),
                             np.zeros_like(model.inputs), np.zeros_like(model.bias)), 10.0)
        np.testing.assert_array_equal(model.symmetric[~model.symmetric_mask], 0)
        np.testing.assert_array_equal(np.diag(model.symmetric), 0)
        assert np.linalg.norm(model.symmetric, 2) <= model.symmetric_cap+1e-14
        network = model.network()
        assert network.margin >= 1-model.symmetric_cap-1e-14
        j = network.jacobian(state)
        np.testing.assert_allclose(j*model.signs[:, None]*model.signs[None, :],
                                   j.swapaxes(-1, -2), atol=1e-14)


def test_constructor_rejects_couplings_outside_the_sign_architecture():
    model = make_twisted_classifier(1, size=8, outputs=2, input_size=3)
    invalid_s = model.symmetric.copy()
    invalid_s[0, -1] = invalid_s[-1, 0] = 0.01
    with pytest.raises(ValueError, match="same-sign blocks"):
        replace(model, symmetric=invalid_s)
    invalid_k = model.skew.copy()
    invalid_k[0, 1], invalid_k[1, 0] = 0.01, -0.01
    with pytest.raises(ValueError, match="opposite-sign blocks"):
        replace(model, skew=invalid_k)
