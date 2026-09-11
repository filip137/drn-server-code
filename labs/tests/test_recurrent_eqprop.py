"""Scientific checks for recurrent classification and actual EqProp phases."""

from dataclasses import replace

import numpy as np
import pytest

from labs.adjoint_estimators import estimate_adjoint, make_probes
from labs.random_nudge_hopfield import Hopfield, relax
from labs.recurrent_eqprop import (
    contrastive_gradient, cost_gradient, error_pair, flatten_gradient,
    loss_accuracy, make_classifier, parameter_gradient, probe_responses, settle,
)


def _case(asymmetry=0.0, cubic=0.25):
    model = make_classifier(12, size=8, outputs=2, input_size=3, asymmetry=asymmetry)
    model.cubic = cubic
    model.bias = np.linspace(-0.12, 0.13, model.size)
    x = np.random.default_rng(13).normal(scale=0.6, size=(4, 3))
    labels = np.array([0, 1, 1, 0])
    network = model.network()
    free = settle(network, model.drive(x), tolerance=1e-13).state
    return model, network, x, labels, free


def _exact(model, network, x, labels, free):
    c = cost_gradient(free, labels, model.hidden, model.logit_scale)
    jacobian = network.jacobian(free)
    adjoint = np.linalg.solve(np.swapaxes(jacobian, -1, -2), c[..., None])[..., 0]
    return c, adjoint, parameter_gradient(model, x, free, adjoint)


@pytest.mark.parametrize("asymmetry", [0.0, 1.0])
@pytest.mark.parametrize("cubic", [0.0, 0.25])
def test_imex_forward_settle_matches_explicit_relaxation_equilibrium(asymmetry, cubic):
    model, network, x, _, free = _case(asymmetry, cubic)
    reference = relax(network, model.drive(x), tolerance=1e-13).state
    np.testing.assert_allclose(free, reference, atol=5e-13, rtol=1e-12)
    assert np.max(np.abs(network.force(free, model.drive(x)))) <= 1.01e-13
    # The solver must keep the probe and example axes independent.
    broadcast_drive = model.drive(x)[:, None, :] + np.linspace(-0.1, 0.1, 3)[None, :, None]
    batched = settle(network, broadcast_drive, free[:, None, :], tolerance=1e-13)
    for example in range(len(x)):
        for probe in range(3):
            individual = settle(network, broadcast_drive[example, probe], tolerance=1e-13)
            np.testing.assert_allclose(batched.state[example, probe], individual.state, atol=5e-13)
    assert batched.equilibrations == len(x) * 3


def test_actual_cost_nudges_use_state_dependent_gradient_at_both_signs():
    model, network, x, labels, free = _case()
    pair = error_pair(model, network, x, labels, free, 0.015, tolerance=1e-13)
    c0 = cost_gradient(free, labels, model.hidden, model.logit_scale)
    expected_effective = 0.015 / np.maximum(np.linalg.norm(c0, axis=-1), 1)
    np.testing.assert_allclose(pair.effective_beta, expected_effective)
    for state, sign in ((pair.positive, 1), (pair.negative, -1)):
        c_state = cost_gradient(state, labels, model.hidden, model.logit_scale)
        actual_residual = network.force(state, model.drive(x)) - sign * expected_effective[:, None] * c_state
        frozen_residual = network.force(state, model.drive(x)) - sign * expected_effective[:, None] * c0
        assert np.max(np.abs(actual_residual)) <= 1.01e-13
        assert np.max(np.abs(frozen_residual)) > 1e-5
    assert pair.equilibrations == 2 * len(x)


def test_softmax_cost_gradient_matches_state_directional_derivative():
    model, _, _, labels, free = _case()
    c = cost_gradient(free, labels, model.hidden, model.logit_scale)
    direction = np.random.default_rng(14).normal(size=free.shape)
    direction /= np.linalg.norm(direction)
    epsilon = 1e-5
    positive = loss_accuracy(free + epsilon * direction, labels, model.hidden, model.logit_scale)[0]
    negative = loss_accuracy(free - epsilon * direction, labels, model.hidden, model.logit_scale)[0]
    np.testing.assert_allclose((positive - negative)/(2*epsilon), np.sum(c*direction)/len(free), atol=1e-10)
    np.testing.assert_array_equal(c[:, :model.hidden], 0.0)


@pytest.mark.parametrize("asymmetry", [0.0, 1.0])
def test_implicit_gradients_match_parameter_directional_finite_differences(asymmetry):
    model, network, x, labels, free = _case(asymmetry)
    _, _, gradient = _exact(model, network, x, labels, free)
    rng = np.random.default_rng(15)
    raw = rng.normal(size=model.symmetric.shape)
    ds = (raw + raw.T)/2
    np.fill_diagonal(ds, 0)
    directions = (ds, rng.normal(size=model.inputs.shape), rng.normal(size=model.bias.shape))
    directions = tuple(direction / np.linalg.norm(direction) for direction in directions)
    epsilon = 1e-5

    def loss(candidate):
        state = settle(candidate.network(), candidate.drive(x), tolerance=1e-13).state
        return loss_accuracy(state, labels, candidate.hidden, candidate.logit_scale)[0]

    # A full symmetric matrix direction tests the full-matrix convention:
    # both off-diagonal entries participate, without introducing a factor two.
    for index, attribute in enumerate(("symmetric", "inputs", "bias")):
        direction = directions[index]
        value = getattr(model, attribute)
        positive = replace(model, **{attribute: value + epsilon*direction})
        negative = replace(model, **{attribute: value - epsilon*direction})
        fd = (loss(positive) - loss(negative))/(2*epsilon)
        np.testing.assert_allclose(np.sum(gradient[index]*direction), fd, atol=2e-8, rtol=2e-6)
    np.testing.assert_allclose(gradient[0], gradient[0].T, atol=1e-15)
    np.testing.assert_array_equal(np.diag(gradient[0]), 0.0)


def test_symmetric_actual_contrastive_eqprop_matches_all_implicit_gradients():
    model, network, x, labels, free = _case()
    _, adjoint, exact_gradient = _exact(model, network, x, labels, free)
    pair = error_pair(model, network, x, labels, free, 1e-4, tolerance=1e-13)
    measured_gradient = contrastive_gradient(model, x, pair)
    np.testing.assert_allclose(pair.response, adjoint, atol=3e-8, rtol=3e-7)
    for measured, exact in zip(measured_gradient, exact_gradient):
        np.testing.assert_allclose(measured, exact, atol=3e-8, rtol=3e-7)
    np.testing.assert_array_equal(np.diag(measured_gradient[0]), 0.0)


@pytest.mark.parametrize("known_skew", [False, True])
def test_centered_contrastive_gradient_has_quadratic_nudge_bias(known_skew):
    model, network, x, labels, free = _case(float(known_skew))
    _, _, exact_gradient = _exact(model, network, x, labels, free)
    exact = flatten_gradient(exact_gradient)
    errors = []
    for beta in (0.008, 0.004):
        pair = error_pair(model, network, x, labels, free, beta, known_skew=known_skew, tolerance=1e-13)
        errors.append(np.linalg.norm(flatten_gradient(contrastive_gradient(model, x, pair)) - exact))
    assert errors[0]/errors[1] == pytest.approx(4.0, rel=0.04)


def test_known_skew_asymep_recovers_adjoint_and_all_parameter_gradients():
    model, network, x, labels, free = _case(1.0)
    c, adjoint, exact_gradient = _exact(model, network, x, labels, free)
    ordinary = np.linalg.solve(network.jacobian(free), c[..., None])[..., 0]
    assert np.linalg.norm(ordinary-adjoint)/np.linalg.norm(adjoint) > 0.2
    pair = error_pair(model, network, x, labels, free, 1e-4, known_skew=True, tolerance=1e-13)
    np.testing.assert_allclose(pair.response, adjoint, atol=3e-8, rtol=3e-7)
    for state, sign in ((pair.positive, 1), (pair.negative, -1)):
        force = network.force(state, model.drive(x)) - 2*(state-free) @ model.skew.T
        force -= sign * pair.effective_beta[:, None] * cost_gradient(state, labels, model.hidden, model.logit_scale)
        assert np.max(np.abs(force)) <= 1.01e-13
    for measured, exact in zip(contrastive_gradient(model, x, pair), exact_gradient):
        np.testing.assert_allclose(measured, exact, atol=3e-8, rtol=3e-7)


def test_projected_sgd_trains_all_parameters_and_preserves_stability():
    model, network, x, labels, free = _case(1.0)
    _, _, gradient = _exact(model, network, x, labels, free)
    initial_loss = loss_accuracy(free, labels, model.hidden, model.logit_scale)[0]
    initial_inputs, initial_bias, initial_skew = model.inputs.copy(), model.bias.copy(), model.skew.copy()
    assert not model.update(gradient, learning_rate=0.01)
    new_free = settle(model.network(), model.drive(x), tolerance=1e-13).state
    assert loss_accuracy(new_free, labels, model.hidden, model.logit_scale)[0] < initial_loss
    np.testing.assert_allclose(model.inputs, initial_inputs - 0.01*gradient[1])
    np.testing.assert_allclose(model.bias, initial_bias - 0.01*gradient[2])
    np.testing.assert_array_equal(model.skew, initial_skew)

    rng = np.random.default_rng(16)
    for _ in range(3):
        # Deliberately include nonsymmetric entries and a nonzero diagonal in
        # the proposed update; the parameter projection must remove them.
        proposal = (rng.normal(size=(model.size, model.size)), np.zeros_like(model.inputs), np.zeros_like(model.bias))
        assert model.update(proposal, learning_rate=10.0)
        np.testing.assert_allclose(model.symmetric, model.symmetric.T, atol=1e-15)
        np.testing.assert_array_equal(np.diag(model.symmetric), 0.0)
        assert np.linalg.norm(model.symmetric, 2) <= model.symmetric_cap + 1e-14
        assert model.network().margin >= 1-model.symmetric_cap-1e-14
        settled = settle(model.network(), model.drive(x), tolerance=1e-11)
        assert settled.residual <= 1e-11


def test_measurement_path_does_not_access_jacobian_or_dense_linear_solve(monkeypatch):
    model, network, x, labels, free = _case(1.0)
    c, adjoint, _ = _exact(model, network, x, labels, free)
    probes = make_probes(np.random.default_rng(17), model.size, model.size, "hadamard")

    def forbidden(*args, **kwargs):
        raise AssertionError("The physical measurement path accessed a Jacobian or dense solve")

    monkeypatch.setattr(Hopfield, "jacobian", forbidden)
    monkeypatch.setattr(np.linalg, "solve", forbidden)
    independent_free = settle(network, model.drive(x), tolerance=1e-13).state
    pair = error_pair(model, network, x, labels, independent_free, 1e-4, tolerance=1e-13)
    responses, equilibrations, _, residual = probe_responses(
        network, model.drive(x), independent_free, probes, 1e-4, tolerance=1e-13,
    )
    readings = np.einsum("bmn,bn->bm", responses, c)
    estimated = estimate_adjoint(pair.response, probes, readings, "orthogonal")
    np.testing.assert_allclose(estimated, adjoint, atol=3e-8, rtol=3e-7)
    assert equilibrations == 2*len(x)*model.size
    assert residual <= 1e-13
    # The known-skew control also requires only forward local dynamics.
    corrected_pair = error_pair(model, network, x, labels, independent_free, 1e-4,
                                known_skew=True, tolerance=1e-13)
    np.testing.assert_allclose(corrected_pair.response, adjoint, atol=3e-8, rtol=3e-7)


def test_settle_rejects_unstable_repulsive_cost_and_incomplete_relaxation():
    model, network, x, labels, free = _case()
    with pytest.raises(ValueError, match="cost-force bound"):
        settle(network, model.drive(x), free, labels=labels, hidden=model.hidden,
               logit_scale=model.logit_scale, cost_scale=-1.0)
    with pytest.raises(RuntimeError, match="Forward settle failed"):
        settle(network, model.drive(x), max_steps=1, tolerance=1e-13)
