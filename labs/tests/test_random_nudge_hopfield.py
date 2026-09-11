"""Scientific checks for the random-nudge toy; no GPU or datasets required."""

from itertools import product

import numpy as np
import pytest

from labs.random_nudge_hopfield import (
    Hopfield, add_read_noise, corrected_feedback, drive_parameter_gradient,
    exact_feedback, hadamard_probes, make_network, measure_response,
    mismatch_projections, rademacher, relax,
)


def test_projection_identity_unbiasedness_and_exact_rademacher_variance():
    net = make_network(4, 1.1, seed=2, cubic=0)
    c = np.array([0.2, -0.4, 0.3, 0.8])
    q, adjoint = exact_feedback(net, np.zeros(4), c)
    z = np.array(list(product((-1.0, 1.0), repeat=4)))
    r = np.linalg.solve(net.jacobian(np.zeros(4)), z.T).T
    d = mismatch_projections(c, q, z, r)
    delta = adjoint - q
    np.testing.assert_allclose(d, z @ delta, atol=1e-15)
    np.testing.assert_allclose(np.mean(d**2), delta @ delta, rtol=1e-14)
    np.testing.assert_allclose(corrected_feedback(q, z, d), adjoint, atol=1e-15)
    errors = q + z * d[:, None] - adjoint
    np.testing.assert_allclose(np.mean(np.sum(errors**2, axis=-1)), 3 * (delta @ delta), rtol=1e-14)


def test_independent_probe_mean_square_error_and_shrinkage():
    rng = np.random.default_rng(10)
    delta = np.array([0.2, -0.5, 0.8, -0.1])
    m = 16
    z = rademacher(rng, (12000, m, 4))
    d = z @ delta
    for scale in (1.0, m / (m + 3)):
        estimate = corrected_feedback(np.zeros(4), z, d, scale)
        mse = np.mean(np.sum((estimate - delta)**2, axis=-1))
        theory = ((1 - scale)**2 + scale**2 * 3 / m) * (delta @ delta)
        assert mse == pytest.approx(theory, rel=0.025)


@pytest.mark.parametrize("asymmetry", [0.0, 1.2])
def test_measured_linear_responses_recover_adjoint_without_oracle(monkeypatch, asymmetry):
    net = make_network(8, asymmetry, seed=3, cubic=0)
    drive = np.linspace(-0.4, 0.3, 8)
    free = relax(net, drive).state
    c = np.zeros(8)
    c[-2:] = [0.2, -0.7]
    q, adjoint = exact_feedback(net, free, c)
    z = hadamard_probes(8)

    def forbidden(*args, **kwargs):
        raise AssertionError("Measurement path accessed a Jacobian or inverse")

    monkeypatch.setattr(Hopfield, "jacobian", forbidden)
    monkeypatch.setattr(np.linalg, "solve", forbidden)
    qm = measure_response(net, drive, free, c, 0.01, tolerance=1e-14).value
    r = measure_response(net, drive, free, z, 0.01, tolerance=1e-14).value
    d = mismatch_projections(c, qm, z, r)
    np.testing.assert_allclose(qm, q, atol=2e-10)
    np.testing.assert_allclose(corrected_feedback(qm, z, d), adjoint, atol=3e-10)
    if asymmetry == 0:
        np.testing.assert_allclose(d, 0, atol=3e-10)


@pytest.mark.parametrize("scheme, expected_ratio", [("central", 4.0), ("forward", 2.0)])
def test_nonlinear_nudge_convergence_order(scheme, expected_ratio):
    net = make_network(4, 0.8, seed=4, cubic=0.5)
    drive = np.array([0.3, -0.5, 0.7, -0.2])
    free = relax(net, drive, tolerance=1e-14).state
    c = np.array([0.5, -0.8, 0.1, 0.7])
    q, _ = exact_feedback(net, free, c)
    errors = [np.linalg.norm(measure_response(net, drive, free, c, beta, scheme=scheme,
                                            tolerance=1e-14).value - q)
              for beta in (0.02, 0.01)]
    assert errors[0] / errors[1] == pytest.approx(expected_ratio, rel=0.04)


def test_nonlinear_jacobian_and_negative_parameter_gradient_by_finite_differences():
    rng = np.random.default_rng(22)
    net = make_network(4, 1.0, seed=1)
    inputs = rng.normal(size=(5, 2))
    b = rng.normal(scale=0.3, size=(3, 2))
    targets = rng.normal(scale=0.2, size=5)

    def evaluate(weights):
        drive = np.zeros((len(inputs), 4))
        drive[:, :3] = inputs @ weights.T
        state = relax(net, drive, tolerance=1e-14).state
        loss = np.mean((state[:, -1] - targets)**2) / 2
        return loss, state

    loss, free = evaluate(b)
    c = np.zeros_like(free)
    c[:, -1] = free[:, -1] - targets
    _, adjoint = exact_feedback(net, free, c)
    grad = drive_parameter_gradient(adjoint, inputs, 3)
    epsilon = 1e-5
    for i, j in product(range(3), range(2)):
        delta = np.zeros_like(b)
        delta[i, j] = epsilon
        fd = (evaluate(b + delta)[0] - evaluate(b - delta)[0]) / (2 * epsilon)
        assert grad[i, j] == pytest.approx(fd, abs=2e-10, rel=2e-6)
    assert evaluate(b - 0.1 * grad)[0] < loss


@pytest.mark.parametrize("scheme, variance_factor", [("central", 0.5), ("forward", 2.0)])
def test_read_noise_matches_two_independent_state_measurements(scheme, variance_factor):
    sigma, beta = 1e-4, 1e-2
    noisy = add_read_noise(np.zeros(100000), sigma, beta, np.random.default_rng(0), scheme)
    assert np.var(noisy) == pytest.approx(variance_factor * sigma**2 / beta**2, rel=0.015)


def test_shared_error_response_noise_cancels_with_full_orthogonal_basis():
    # q is read ONCE per estimator and reused in every d(z).
    rng = np.random.default_rng(1)
    net = make_network(8, 1.0, cubic=0)
    c = rng.normal(size=8)
    q, adjoint = exact_feedback(net, np.zeros(8), c)
    q_noisy = q + rng.normal(size=8)
    z = hadamard_probes(8)
    r = np.linalg.solve(net.jacobian(np.zeros(8)), z.T).T
    d = mismatch_projections(c, q_noisy, z, r)
    np.testing.assert_allclose(corrected_feedback(q_noisy, z, d), adjoint, atol=1e-14)


@pytest.mark.parametrize("scale", [1.0, 0.6])
def test_noisy_correction_variance_accounts_for_reusing_one_error_read(scale):
    rng = np.random.default_rng(37)
    trials, m, size = 20000, 8, 4
    net = make_network(size, 1.1, cubic=0)
    c = np.array([0.0, 0.0, 0.3, -0.8])
    q, adjoint = exact_feedback(net, np.zeros(size), c)
    z = rademacher(rng, (trials, m, size))
    r = np.linalg.solve(net.jacobian(np.zeros(size)), z.reshape(-1, size).T).T.reshape(z.shape)
    sigma, beta = 0.03, 0.1
    q_noisy = add_read_noise(np.broadcast_to(q, (trials, size)), sigma, beta, rng)
    r_noisy = add_read_noise(r, sigma, beta, rng)
    d = mismatch_projections(c, q_noisy, z, r_noisy)
    estimate = corrected_feedback(q_noisy, z, d, scale)
    delta_sq = np.sum((adjoint-q)**2)
    v = sigma**2 / (2*beta**2)
    factor = (1-scale)**2 + scale**2*(size-1)/m
    theory = factor*delta_sq + v*(size*factor + scale**2*size*(c@c)/m)
    observed = np.mean(np.sum((estimate-adjoint)**2, axis=-1))
    assert observed == pytest.approx(theory, rel=0.025)
    assert np.mean(d**2) == pytest.approx(delta_sq + v*(c@c+size), rel=0.025)


def test_asymmetry_preserves_stable_symmetric_part():
    for alpha in (0.0, 0.5, 1.5):
        net = make_network(8, alpha, seed=9)
        settled = relax(net, np.ones(8))
        j = net.jacobian(settled.state)
        assert np.linalg.eigvalsh((j + j.T) / 2)[-1] < -0.59
        assert settled.residual <= 1e-12


def test_solver_fails_loudly_if_not_converged():
    with pytest.raises(RuntimeError, match="Relaxation failed"):
        relax(make_network(4, 1.0), np.ones(4), max_steps=1)
    with pytest.raises(ValueError, match="lambda_max"):
        Hopfield(1.1 * np.eye(4))
    with pytest.raises(ValueError, match="power-of-two"):
        hadamard_probes(6)
