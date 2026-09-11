"""Check physical measurements and the diagnostic parameter-force mapping."""

import numpy as np
import pytest

from labs.adjoint_estimators import estimate_adjoint, fit_response, make_probes
from labs.random_nudge_hopfield import Hopfield, make_network, relax
from labs.tools.compare_adjoint_measurements import design_statistics, physical_measurements


def test_full_physical_coordinate_measurements_recover_adjoint_without_jacobian(monkeypatch):
    net = make_network(8, 1.5, seed=2, cubic=0)
    drive = np.linspace(-0.5, 0.6, 8)
    free = relax(net, drive, tolerance=1e-13)
    c = np.array([0, 0, 0, 0, 0.1, -0.2, 0.3, 0.4])
    reference_r = np.linalg.solve(net.jacobian(free.state), np.eye(8))
    z = make_probes(np.random.default_rng(3), 8, 8, "coordinate")
    heldout = make_probes(np.random.default_rng(4), 8, 16, "random_sign")

    def forbidden(*args, **kwargs):
        raise AssertionError("Physical path accessed a Jacobian or its inverse")

    monkeypatch.setattr(Hopfield, "jacobian", forbidden)
    monkeypatch.setattr(np.linalg, "solve", forbidden)
    measured = physical_measurements(net, drive, free.state, c, z, heldout, 0.03, 1e-13)
    np.testing.assert_allclose(measured["q_unit"] * measured["norm_c"], reference_r @ c, atol=2e-11)
    np.testing.assert_allclose(measured["response"], z @ reference_r.T, atol=2e-11)
    estimate = estimate_adjoint(np.ones(8), z, measured["response"] @ c, "orthogonal")
    np.testing.assert_allclose(estimate, reference_r.T @ c, atol=2e-11)
    fitted = fit_response(z, measured["response"])
    np.testing.assert_allclose(heldout @ fitted.T, measured["heldout"], atol=2e-11)
    assert measured["actual_equilibrations"] == 2 * (1 + 8 + 16)
    assert measured["residual"] <= 1e-13


def test_rank_statistics_distinguish_full_response_from_measured_subspace():
    rng = np.random.default_rng(9)
    z = make_probes(rng, 8, 16, "hadamard")
    partial, full = design_statistics(z[:4]), design_statistics(z)
    assert partial["design_rank"] == 4
    assert not partial["identifiable_full_response"]
    assert partial["full_design_condition"] is None
    assert full["design_rank"] == 8
    assert full["identifiable_full_response"]
    assert full["full_design_condition"] == pytest.approx(1)


def test_directed_recurrent_local_gradient_matches_re_equilibrated_cost():
    net = make_network(8, 1.1, seed=7)
    drive = np.linspace(-0.5, 0.3, 8)
    free = relax(net, drive, tolerance=1e-13).state
    target = np.array([0.5, -0.4])
    c = np.zeros(8)
    c[-2:] = free[-2:] - target
    adjoint = np.linalg.solve(net.jacobian(free).T, c)
    exact_gradient = -np.outer(adjoint, free)
    epsilon = 1e-5
    for i, j in [(0, 3), (3, 0), (6, 1), (1, 6)]:
        costs = []
        for sign in (-1, 1):
            w = net.weights.copy()
            w[i, j] += sign * epsilon
            s = relax(Hopfield(w, net.cubic), drive, tolerance=1e-13).state
            costs.append(float(np.sum((s[-2:] - target)**2) / 2))
        finite_difference = (costs[1] - costs[0]) / (2 * epsilon)
        assert finite_difference == pytest.approx(exact_gradient[i, j], abs=2e-10, rel=2e-7)
