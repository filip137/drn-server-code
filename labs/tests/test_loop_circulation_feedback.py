"""Independent physical-access and stochastic-area checks for known loops."""

import numpy as np
import pytest

from labs.circulation_feedback import area_rate
from labs.loop_circulation_feedback import calibrate_loop_controller, loop_area_rate


def known_wiring(seed, states, loops):
    # Wiring is chosen independently of the unknown gains, never extracted
    # from a force Jacobian or from the hidden system's skew decomposition.
    patterns, _ = np.linalg.qr(np.random.default_rng(seed).normal(size=(states, 2*loops)))
    return patterns[:, :loops], patterns[:, loops:]


def test_projection_area_equals_dense_oracle_for_arbitrary_trajectory_pairs():
    states, loops, dt = 12, 3, 0.037
    left, right = known_wiring(810, states, loops)
    rng = np.random.default_rng(811)
    previous = rng.normal(size=(3, 5, states))
    current = previous+0.2*rng.normal(size=previous.shape)
    wiring = np.concatenate((left, right), axis=1)
    expected_area = area_rate(previous, current, dt)
    basis = np.array([(np.outer(u, v)-np.outer(v, u))/np.sqrt(2)
                      for u, v in zip(left.T, right.T)])
    expected = np.einsum("lij,ij->l", basis, expected_area)
    measured = loop_area_rate(previous @ wiring, current @ wiring, dt)
    np.testing.assert_allclose(measured, expected, rtol=2e-14, atol=2e-14)
    np.testing.assert_allclose(
        loop_area_rate(current @ wiring, previous @ wiring, dt), -expected,
        rtol=2e-14, atol=2e-14)
    np.testing.assert_allclose(
        loop_area_rate(previous @ wiring, current @ wiring, 2*dt), expected/2,
        rtol=2e-14, atol=2e-14)


def test_projected_noisy_calibration_learns_hidden_gains_without_operator_access(monkeypatch):
    states, loops, replicas = 8, 2, 32
    left, right = known_wiring(812, states, loops)
    # Hidden gains are selected separately from the supplied physical wiring.
    gains = np.array([0.8, -0.6])
    hidden = (left*gains) @ right.T
    skew = (hidden-hidden.T)/np.sqrt(2)
    center = np.linspace(-0.3, 0.4, states)
    damping = np.linspace(0.8, 1.4, states)
    calls = []

    def opaque_force(state):
        assert state.shape == (replicas, states)
        calls.append(None)
        x = state-center
        return -damping*x-0.2*x**3+x @ skew.T

    def forbidden(*args, **kwargs):
        raise AssertionError("Loop calibration attempted operator identification or a solve")

    kwargs = dict(duration=80., dt=0.02, temperature=0.05, replicas=replicas,
                  learning_rate=0.15, update_interval=10, burn_in=4.,
                  average_after=30., seed=813, read_noise=0.005)
    callback_rows = []

    def callback(info, current, average):
        assert current.shape == average.shape == (loops,)
        callback_rows.append(info.copy())
        # Diagnostic callbacks cannot alter controller state through aliases.
        current[:] = 1234
        average[:] = -1234
        info["updates"] = -1

    with monkeypatch.context() as blocked:
        for name in ("solve", "inv", "pinv", "lstsq", "eig", "eigh", "svd", "qr"):
            blocked.setattr(np.linalg, name, forbidden)
        result = calibrate_loop_controller(
            opaque_force, center, left, right, callback=callback, **kwargs)
        repeated = calibrate_loop_controller(opaque_force, center, left, right, **kwargs)

    np.testing.assert_array_equal(result.coefficients, repeated.coefficients)
    np.testing.assert_array_equal(result.final_coefficients, repeated.final_coefficients)
    np.testing.assert_array_equal(result.matrix, repeated.matrix)
    assert np.linalg.norm(result.coefficients+gains)/np.linalg.norm(gains) < 0.3
    np.testing.assert_array_equal(result.matrix, -result.matrix.T)
    np.testing.assert_array_equal(np.diag(result.matrix), 0.)
    assert np.linalg.norm(result.matrix+skew)/np.linalg.norm(skew) < 0.3
    assert len(callback_rows) == len(result.history)
    assert all(row["updates"] > 0 for row in result.history)

    # Read noise and counting concern projection channels, not all n voltages.
    steps = int(np.ceil(kwargs["duration"]/kwargs["dt"])
                + np.ceil(kwargs["burn_in"]/kwargs["dt"]))
    assert len(calls) == 4*steps
    assert result.stats["learned_parameters"] == loops
    assert result.stats["fixed_wiring_coefficients"] == 2*states*loops
    assert result.stats["readout_channels"] == 2*loops
    assert result.stats["state_reads"] == replicas*(steps+1)
    assert result.stats["scalar_measurement_reads"] == 2*loops*replicas*(steps+1)
    assert result.stats["total_record_time"] == pytest.approx(replicas*steps*kwargs["dt"])


def test_unresolved_physical_modes_are_not_filled_in_by_the_calibrator():
    # One controlled two-node loop, with an independent hidden loop beyond
    # its sensor/actuator support. The returned matrix must respect the wiring.
    left, right = np.eye(4)[:, :1], np.eye(4)[:, 1:2]
    skew = np.zeros((4, 4))
    skew[2, 3], skew[3, 2] = 0.8, -0.8
    result = calibrate_loop_controller(
        lambda s: -s+s @ skew.T, np.zeros(4), left, right,
        duration=3., burn_in=1., average_after=1., replicas=8, seed=814)
    np.testing.assert_array_equal(result.matrix[2:, :], 0.)
    np.testing.assert_array_equal(result.matrix[:, 2:], 0.)
    assert np.linalg.norm(skew+result.matrix) >= np.linalg.norm(skew)


@pytest.mark.parametrize("left,right", [
    (np.eye(4)[:, :1], np.eye(4)[:, :1]),
    (2*np.eye(4)[:, :1], np.eye(4)[:, 1:2]),
    (np.eye(4)[:, :2], np.eye(4)[:, 2:3]),
])
def test_invalid_known_wiring_is_rejected_before_running_physics(left, right):
    def forbidden(state):
        raise AssertionError("Invalid wiring reached the physical simulation")
    with pytest.raises(ValueError, match="Expected"):
        calibrate_loop_controller(forbidden, np.zeros(4), left, right)


@pytest.mark.parametrize("previous,current,dt", [
    (np.zeros((2, 3)), np.zeros((2, 3)), 0.1),
    (np.zeros((2, 4)), np.zeros((3, 4)), 0.1),
    (np.zeros((0, 4)), np.zeros((0, 4)), 0.1),
    (np.zeros((2, 4)), np.zeros((2, 4)), 0),
])
def test_invalid_projection_readings_are_rejected(previous, current, dt):
    with pytest.raises(ValueError, match="Expected"):
        loop_area_rate(previous, current, dt)
