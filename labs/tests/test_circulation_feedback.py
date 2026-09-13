"""Signs, physical access, and independent oracles for circulation feedback.

Dense Jacobians, covariance solves, and adjoints appear only in test oracles.
The controller receives an opaque force function and measured trajectories.
"""

import numpy as np
import pytest
from scipy.linalg import solve_continuous_lyapunov

from labs.circulation_feedback import area_rate, calibrate_controller
from labs.random_nudge_hopfield import Hopfield
from labs.recurrent_eqprop import (
    NudgedPair, contrastive_gradient, cost_gradient, flatten_gradient,
    make_classifier, parameter_gradient, settle,
)
from labs.tools import train_recurrent_eqprop_digits as digits


def test_measured_area_has_rotation_sign_time_units_and_batch_normalization():
    # Uniform points on a circle: anticlockwise angular velocity omega gives
    # <dot{x} x^T - x dot{x}^T> = omega * [[0, -1], [1, 0]].
    previous = np.array([[1., 0.], [0., 1.], [-1., 0.], [0., -1.]])
    rotation = np.array([[0., -1.], [1., 0.]])
    omega, dt = 0.7, 0.013
    current = previous + dt * omega * previous @ rotation.T
    expected = omega * rotation
    np.testing.assert_allclose(area_rate(previous, current, dt), expected, atol=2e-15)
    np.testing.assert_allclose(area_rate(previous, current, 2*dt), expected/2, atol=2e-15)
    np.testing.assert_allclose(area_rate(3*previous, 3*current, dt), 9*expected, atol=2e-14)
    # Splitting a batch over replica and time axes must not change its scale.
    np.testing.assert_allclose(
        area_rate(previous.reshape(2, 2, 2), current.reshape(2, 2, 2), dt),
        expected, atol=2e-15,
    )
    np.testing.assert_allclose(area_rate(current, previous, dt), -expected, atol=2e-15)


def test_nonisotropic_ou_circulation_identifies_skew_and_descent_direction():
    rng = np.random.default_rng(701)
    raw = rng.normal(size=(4, 4))
    damping = raw @ raw.T/4 + np.diag([0.5, 1., 1.5, 2.])
    raw = rng.normal(size=(4, 4))
    skew = 0.4*(raw-raw.T)
    drift = -damping + skew
    temperature = 0.17
    covariance = solve_continuous_lyapunov(drift, -2*temperature*np.eye(4))
    expected_area = drift @ covariance - covariance @ drift.T

    # Exact covariance cubature avoids Monte Carlo tolerance hiding sign errors.
    root = np.linalg.cholesky(covariance)
    previous = np.concatenate([2*root.T, -2*root.T])
    dt = 0.01
    current = previous + dt*previous @ drift.T
    measured_area = area_rate(previous, current, dt)
    np.testing.assert_allclose(measured_area, expected_area, atol=2e-14)

    # For isotropic white forcing, stationarity implies this identity even
    # when the damping and skew matrices do not commute.
    inverse_covariance = np.linalg.inv(covariance)
    recovered_skew = (measured_area @ inverse_covariance
                      + inverse_covariance @ measured_area)/4
    np.testing.assert_allclose(recovered_skew, skew, atol=2e-13)
    assert np.sum(skew*measured_area) > 0
    epsilon = 1e-3
    descended_skew = skew - epsilon*measured_area/(2*temperature)
    assert np.linalg.norm(descended_skew) < np.linalg.norm(skew)

    reciprocal_covariance = solve_continuous_lyapunov(
        -damping, -2*temperature*np.eye(4),
    )
    reciprocal_area = -damping @ reciprocal_covariance + reciprocal_covariance @ damping
    np.testing.assert_allclose(reciprocal_area, 0, atol=2e-15)


def test_doubled_anchored_controller_yields_adjoint_and_contrastive_gradient():
    model = make_classifier(702, size=8, outputs=2, input_size=3, asymmetry=1.)
    model.bias = np.linspace(-0.2, 0.15, model.size)
    x = np.random.default_rng(703).normal(scale=0.6, size=(4, 3))
    labels = np.array([0, 1, 1, 0])
    network, drive = model.network(), model.drive(x)
    free = settle(network, drive, tolerance=1e-13).state
    controller = -model.skew
    jacobian = network.jacobian(free)
    error = cost_gradient(free, labels, model.hidden, model.logit_scale)
    adjoint = np.linalg.solve(np.swapaxes(jacobian, -1, -2), error[..., None])[..., 0]
    exact_gradient = parameter_gradient(model, x, free, adjoint)
    # The learned C cancels K during calibration. Using 2C during nudging
    # changes +K to -K, and anchoring preserves every original free state.
    anchored = settle(network, drive, free, skew_correction=-controller,
                      center=free, tolerance=1e-13)
    np.testing.assert_allclose(anchored.state, free, atol=1e-13)
    np.testing.assert_allclose(jacobian + 2*controller,
                               np.swapaxes(jacobian, -1, -2), atol=1e-15)

    effective = 1e-4/np.maximum(np.linalg.norm(error, axis=-1), 1.)
    kwargs = dict(labels=labels, hidden=model.hidden, logit_scale=model.logit_scale,
                  center=free, tolerance=1e-13)
    positive = settle(network, drive, free, cost_scale=effective,
                      skew_correction=-controller, **kwargs)
    negative = settle(network, drive, free, cost_scale=-effective,
                      skew_correction=-controller, **kwargs)
    pair = NudgedPair(positive.state, negative.state, effective,
                      positive.equilibrations+negative.equilibrations,
                      positive.iterations+negative.iterations,
                      max(positive.residual, negative.residual))
    np.testing.assert_allclose(pair.response, adjoint, atol=5e-8, rtol=5e-7)
    np.testing.assert_allclose(flatten_gradient(contrastive_gradient(model, x, pair)),
                               flatten_gradient(exact_gradient), atol=5e-8, rtol=5e-7)

    # Merely making the nudged dynamics reciprocal solves a different response.
    single_positive = settle(network, drive, free, cost_scale=effective,
                             skew_correction=-controller/2, **kwargs)
    single_negative = settle(network, drive, free, cost_scale=-effective,
                             skew_correction=-controller/2, **kwargs)
    single_response = (single_positive.state-single_negative.state)/(2*effective[:, None])
    assert np.linalg.norm(single_response-adjoint)/np.linalg.norm(adjoint) > 0.1


def test_stochastic_calibration_improves_feedback_without_operator_access(monkeypatch):
    center = np.array([0.3, -0.4])
    skew = np.array([[0., -0.7], [0.7, 0.]])
    drift = -np.diag([0.8, 1.3]) + skew
    replicas = 32
    force_calls = []

    def physical_force(states):
        # The only public interface to this system is simultaneous forward
        # force evaluation while integrating its physical noisy trajectories.
        assert states.shape == (replicas, 2)
        force_calls.append(None)
        return (states-center) @ drift.T

    def forbidden(*args, **kwargs):
        raise AssertionError("Calibration attempted a dense operator reconstruction or solve")

    kwargs = dict(duration=60., dt=0.02, temperature=0.05, replicas=replicas,
                  learning_rate=0.15, update_interval=10, burn_in=4.,
                  average_after=20., seed=704, read_noise=0.005)
    with monkeypatch.context() as blocked:
        for name in ("solve", "inv", "pinv", "lstsq"):
            blocked.setattr(np.linalg, name, forbidden)
        result = calibrate_controller(physical_force, center, **kwargs)
        repeated = calibrate_controller(physical_force, center, **kwargs)

    np.testing.assert_array_equal(result.matrix, repeated.matrix)
    np.testing.assert_array_equal(result.final_matrix, repeated.final_matrix)
    np.testing.assert_allclose(result.matrix, -result.matrix.T, atol=0)
    np.testing.assert_array_equal(np.diag(result.matrix), 0.)
    # A broad scientific threshold, rather than a particular random output,
    # requires useful compensation of a non-reciprocal drift under read noise.
    assert np.linalg.norm(skew+result.matrix)/np.linalg.norm(skew) < 0.3
    error = np.array([1., -0.5])
    adjoint = np.linalg.solve(drift.T, error)
    ordinary = np.linalg.solve(drift, error)
    corrected = np.linalg.solve(drift+2*result.matrix, error)
    assert np.linalg.norm(corrected-adjoint) < 0.4*np.linalg.norm(ordinary-adjoint)

    # Independent trajectories cost real observation time, even if computed
    # in a vectorized batch. Include the unadapted warm-up in that accounting.
    steps = int(np.ceil(kwargs["duration"]/kwargs["dt"])) + int(np.ceil(kwargs["burn_in"]/kwargs["dt"]))
    assert len(force_calls) == 2*2*steps  # two Heun calls, two repeated calibrations
    assert result.stats["state_reads"] == replicas*(steps+1)
    assert result.stats["total_record_time"] == pytest.approx(replicas*steps*kwargs["dt"])


@pytest.mark.parametrize("controller_fraction", [0., 1.])
def test_training_route_uses_supplied_controller_and_no_random_probe_or_adjoint(
        monkeypatch, controller_fraction):
    model = make_classifier(705, size=8, outputs=2, input_size=3, asymmetry=1.)
    x = np.random.default_rng(706).normal(scale=0.6, size=(4, 3))
    labels = np.array([0, 1, 0, 1])
    expected_method = "known_skew_asymep" if controller_fraction else "contrastive_ep"
    kwargs = dict(beta=1e-4, sigma=0., tolerance=1e-13, audit=False)
    expected, _ = digits.gradient_step(model, x, labels, expected_method,
                                      np.random.default_rng(707), **kwargs)

    def forbidden(*args, **kwargs):
        raise AssertionError("Circulation-calibrated EqProp accessed probes or an adjoint oracle")

    monkeypatch.setattr(digits, "oracle_adjoint", forbidden)
    monkeypatch.setattr(digits, "probe_responses", forbidden)
    monkeypatch.setattr(digits, "make_probes", forbidden)
    monkeypatch.setattr(Hopfield, "jacobian", forbidden)
    monkeypatch.setattr(np.linalg, "solve", forbidden)
    measured, metadata = digits.gradient_step(
        model, x, labels, "circulation_asymep", np.random.default_rng(707),
        feedback_controller=-controller_fraction*model.skew, **kwargs,
    )
    # Zero supplied compensation must reduce to ordinary EP. In particular,
    # the new route must not secretly substitute the known physical skew.
    for actual, reference in zip(measured, expected):
        np.testing.assert_array_equal(actual, reference)
    assert metadata["probe_count"] == 0
    assert metadata["total_probe_excitation_sq"] == 0
    assert metadata["equilibrations"] == 3*len(x)
    assert metadata["state_reads"] == 3*len(x)
    assert metadata["max_residual"] <= 1e-13
    assert "gradient_relative_error" not in metadata
