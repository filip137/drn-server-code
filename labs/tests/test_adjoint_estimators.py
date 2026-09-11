"""Identification, geometry and noise checks for measurement-only estimators."""

from itertools import product

import numpy as np
import pytest

from labs.adjoint_estimators import estimate_adjoint, fit_response, make_probes


@pytest.mark.parametrize("design", ["coordinate", "canonical", "hadamard", "orthogonal", "random_sign"])
def test_probe_designs_match_total_excitation_and_block_orthogonality(design):
    probes = make_probes(np.random.default_rng(1), 8, 19, design)
    assert probes.shape == (19, 8)
    np.testing.assert_allclose(np.linalg.norm(probes, axis=-1), 1.0, atol=1e-14)
    if design != "random_sign":
        for start in range(0, 19, 8):
            block = probes[start:start + 8]
            np.testing.assert_allclose(block @ block.T, np.eye(len(block)), atol=1e-14)
    if design == "hadamard":
        np.testing.assert_allclose(np.abs(probes), 1 / np.sqrt(8))


def test_mc_unit_sign_convention_has_exact_unbiasedness_and_variance():
    size = 4
    probes = np.array(list(product((-1.0, 1.0), repeat=size))) / np.sqrt(size)
    q = np.array([0.3, -0.1, 0.5, 0.2])
    truth = np.array([0.8, 0.2, -0.7, 0.4])
    readings = probes @ truth
    np.testing.assert_allclose(estimate_adjoint(q, probes, readings, "mc"), truth, atol=1e-15)
    individual = estimate_adjoint(q, probes[:, None, :], readings[:, None], "mc")
    np.testing.assert_allclose(np.mean(individual, axis=0), truth, atol=1e-15)
    np.testing.assert_allclose(
        np.mean(np.sum((individual - truth)**2, axis=-1)),
        (size - 1) * np.sum((truth - q)**2), rtol=1e-14,
    )
    with pytest.raises(ValueError, match="unit-L2"):
        estimate_adjoint(q, probes * np.sqrt(size), readings * np.sqrt(size), "mc")


def test_each_exact_kaczmarz_projection_decreases_error_by_measured_component():
    rng = np.random.default_rng(2)
    truth, q = rng.normal(size=(2, 7))
    probes = rng.normal(size=(15, 7))  # deliberately non-unit rows
    readings = probes @ truth
    previous = q
    for end in range(1, len(probes) + 1):
        estimate = estimate_adjoint(q, probes[:end], readings[:end], "kaczmarz")
        z, error = probes[end - 1], previous - truth
        prediction = error @ error - (z @ error)**2 / (z @ z)
        np.testing.assert_allclose(np.sum((estimate - truth)**2), prediction, atol=1e-14)
        previous = estimate


def test_expected_single_sign_kaczmarz_error_factor():
    size = 4
    probes = np.array(list(product((-1.0, 1.0), repeat=size))) / np.sqrt(size)
    truth = np.array([0.4, -0.3, 0.9, -0.2])
    q = np.zeros(size)
    estimate = estimate_adjoint(q, probes[:, None], (probes @ truth)[:, None], "kaczmarz")
    np.testing.assert_allclose(
        np.mean(np.sum((estimate - truth)**2, axis=-1)),
        (1 - 1 / size) * (truth @ truth), rtol=1e-14,
    )


@pytest.mark.parametrize("method", ["lstsq", "ridge"])
def test_underdetermined_regression_retains_baseline_nullspace(method):
    rng = np.random.default_rng(3)
    probes = rng.normal(size=(3, 7))
    q, truth = rng.normal(size=(2, 7))
    estimate = estimate_adjoint(q, probes, probes @ truth, method)
    np.testing.assert_allclose(probes @ estimate, probes @ truth, atol=1e-14)
    _, _, vh = np.linalg.svd(probes, full_matrices=True)
    nullspace = vh[3:]
    np.testing.assert_allclose(nullspace @ estimate, nullspace @ q, atol=1e-14)


@pytest.mark.parametrize("design", ["coordinate", "hadamard", "orthogonal"])
def test_full_orthogonal_design_recovers_exact_adjoint_and_cancels_baseline(design):
    rng = np.random.default_rng(4)
    probes = make_probes(rng, 8, 8, design)
    truth = rng.normal(size=(3, 8))
    q = 100 * rng.normal(size=(3, 8))
    readings = np.einsum("mn,bn->bm", probes, truth)
    for method in ("mc", "kaczmarz", "orthogonal", "lstsq"):
        estimate = estimate_adjoint(q, probes, readings, method)
        np.testing.assert_allclose(estimate, truth, atol=3e-13)


def test_partial_orthogonal_correction_only_changes_measured_subspace():
    rng = np.random.default_rng(5)
    full = make_probes(rng, 8, 8, "orthogonal")
    probes, complement = full[:3], full[3:]
    q, truth = rng.normal(size=(2, 8))
    estimate = estimate_adjoint(q, probes, probes @ truth, "orthogonal")
    np.testing.assert_allclose(probes @ estimate, probes @ truth, atol=1e-14)
    np.testing.assert_allclose(complement @ estimate, complement @ q, atol=1e-14)
    np.testing.assert_allclose(estimate, estimate_adjoint(q, probes, probes @ truth, "lstsq"), atol=1e-14)
    with pytest.raises(ValueError, match="orthonormal"):
        estimate_adjoint(q, np.stack([full[0], full[0]]), np.zeros(2), "orthogonal")


def test_weighted_ridge_matches_positive_definite_reference_and_broadcasts():
    rng = np.random.default_rng(6)
    probes = rng.normal(size=(5, 4))
    q = rng.normal(size=(3, 4))
    readings = rng.normal(size=(3, 5))
    weights = rng.uniform(0.1, 2.0, size=(3, 5))
    weights[0, 0] = 0.0
    ridge = 0.7
    estimates = estimate_adjoint(q, probes, readings, "ridge", ridge=ridge, weights=weights)
    for batch in range(3):
        normal = probes.T @ (weights[batch, :, None] * probes) + ridge * np.eye(4)
        rhs = probes.T @ (weights[batch] * readings[batch]) + ridge * q[batch]
        np.testing.assert_allclose(estimates[batch], np.linalg.solve(normal, rhs), atol=1e-14)


def test_ridge_bias_variance_matches_orthogonal_noise_model():
    rng = np.random.default_rng(7)
    size, trials, sigma, ridge = 8, 12000, 0.5, 2.0
    probes = make_probes(rng, size, size, "hadamard")
    truth = rng.normal(size=size)
    q = truth + np.linspace(-0.15, 0.15, size)
    readings = probes @ truth + rng.normal(scale=sigma, size=(trials, size))
    regularized = estimate_adjoint(q, probes, readings, "ridge", ridge=ridge)
    unregularized = estimate_adjoint(q, probes, readings, "lstsq")
    mse = np.mean(np.sum((regularized - truth)**2, axis=-1))
    theory = (ridge**2 * np.sum((q - truth)**2) + size * sigma**2) / (1 + ridge)**2
    assert mse == pytest.approx(theory, rel=0.025)
    assert mse < np.mean(np.sum((unregularized - truth)**2, axis=-1))


def test_fit_response_orientation_rank_and_weighted_regularization():
    rng = np.random.default_rng(8)
    probes = rng.normal(size=(9, 4))
    truth = rng.normal(size=(2, 3, 4))
    responses = probes @ np.swapaxes(truth, -1, -2)
    np.testing.assert_allclose(fit_response(probes, responses), truth, atol=1e-14)
    prior = rng.normal(size=(3, 4))
    weights = np.linspace(0.2, 1.2, 9)
    ridge = 0.3
    fitted = fit_response(probes, responses, ridge=ridge, prior=prior, weights=weights)
    for batch in range(2):
        normal = probes.T @ (weights[:, None] * probes) + ridge * np.eye(4)
        rhs = probes.T @ (weights[:, None] * responses[batch]) + ridge * prior.T
        np.testing.assert_allclose(fitted[batch], np.linalg.solve(normal, rhs).T, atol=1e-14)
    for reg in (0.0, 1.0):
        with pytest.raises(ValueError, match="full-column-rank"):
            fit_response(probes[:3], responses[..., :3, :], ridge=reg)
    with pytest.raises(ValueError, match="full-column-rank"):
        fit_response(probes, responses, weights=np.r_[np.ones(3), np.zeros(6)])


def test_response_fit_predicts_unseen_directions_and_equivalent_adjoint():
    rng = np.random.default_rng(9)
    size = 8
    jacobian = -2 * np.eye(size) + rng.normal(scale=0.12, size=(size, size))
    probes = make_probes(rng, size, size, "hadamard")
    responses = np.linalg.solve(jacobian, probes.T).T
    fitted = fit_response(probes, responses)
    unseen = rng.normal(size=(11, size))
    np.testing.assert_allclose(unseen @ fitted.T, np.linalg.solve(jacobian, unseen.T).T, atol=1e-14)
    c, q = rng.normal(size=(2, size))
    estimated = estimate_adjoint(q, probes, responses @ c, "orthogonal")
    np.testing.assert_allclose(estimated, fitted.T @ c, atol=1e-14)


def test_invalid_weights_and_degenerate_designs_fail_explicitly():
    q, probes, y = np.zeros(3), np.eye(3), np.zeros(3)
    with pytest.raises(ValueError, match="nonnegative weights"):
        estimate_adjoint(q, probes, y, "ridge", weights=[1, -1, 1])
    with pytest.raises(ValueError, match="nonnegative weights"):
        estimate_adjoint(q, probes, y, "ridge", weights=[0, 0, 0])
    with pytest.raises(ValueError, match="nonzero probe"):
        estimate_adjoint(q, np.zeros((1, 3)), np.zeros(1), "kaczmarz")
    with pytest.raises(ValueError, match="power-of-two"):
        make_probes(np.random.default_rng(10), 7, 3, "hadamard")
