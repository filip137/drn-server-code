"""Estimate equilibrium adjoints from controlled, measured responses.

With ``F(s)-b=0``, a paired nudge in direction z measures ``r = R z``,
where R is the equilibrium response. The scalar reading ``y = c.T r`` is
therefore an equation ``z.T lambda = y`` for ``lambda = R.T c``. These
estimators use only those equations and a baseline q, typically ordinary
EqProp's measured ``R c``; they never evaluate a force Jacobian.

Probe directions are ROWS: probes (..., m, n), readings (..., m), and
baseline (..., n). Leading batch dimensions broadcast. All generated probes
have total L2 norm one, so a common physical nudge amplitude specifies a
common total excitation, including for dense sign patterns.
"""

import numpy as np


def make_probes(rng, size, count, design="random_sign"):
    """Return ``(count, size)`` unit directions with randomized ordering.

    ``coordinate`` (alias ``canonical``), ``hadamard`` and ``orthogonal``
    draw independent orthonormal blocks, each truncated only at its end.
    The first uses signed coordinate vectors; Hadamard uses dense signed
    patterns and requires a power-of-two size; orthogonal uses a Haar-random
    real basis. ``random_sign`` draws independent Rademacher rows / sqrt(n).

    Every row is isotropic in expectation: E[z z.T] = I/n. Rows in an
    orthonormal block are dependent, so the independent-probe Monte Carlo
    variance formula must not be applied to those blocks. Counts above n
    repeat the orthonormal-block procedure and are valid for regression or
    sequential projections, but not a single orthogonal projection call.
    """
    for label, value in (("size", size), ("count", count)):
        if not isinstance(value, (int, np.integer)) or isinstance(value, bool) or value < 1:
            raise ValueError(f"Expected positive integer {label}; got {value!r}")
    allowed = ("coordinate", "canonical", "hadamard", "orthogonal", "random_sign")
    if design not in allowed:
        raise ValueError(f"Expected probe design in {allowed}; got {design!r}")
    if design == "random_sign":
        return (2 * rng.integers(0, 2, size=(count, size)) - 1) / np.sqrt(size)
    if design == "hadamard":
        if size & (size - 1):
            raise ValueError(f"Expected a positive power-of-two Hadamard size; got {size}")
        basis = np.ones((1, 1))
        while basis.shape[0] < size:
            basis = np.block([[basis, basis], [basis, -basis]])
        basis /= np.sqrt(size)
    elif design in ("coordinate", "canonical"):
        basis = np.eye(size)

    blocks = []
    remaining = count
    while remaining:
        if design == "orthogonal":
            q, r = np.linalg.qr(rng.normal(size=(size, size)))
            signs = np.where(np.diag(r) < 0, -1.0, 1.0)
            block = (q * signs).T
        else:
            signs = 2 * rng.integers(0, 2, size=size) - 1
            block = basis[rng.permutation(size)] * signs
        take = min(remaining, size)
        blocks.append(block[:take])
        remaining -= take
    return np.concatenate(blocks, axis=0)


def _finite_array(value, name):
    result = np.asarray(value, dtype=np.float64)
    if not np.isfinite(result).all():
        raise ValueError(f"Expected finite {name}; got non-finite values")
    return result


def _check_ridge(ridge):
    if np.ndim(ridge) != 0 or not np.isfinite(ridge) or ridge < 0:
        raise ValueError(f"Expected finite ridge >= 0; got {ridge!r}")


def _weight_design(probes, target, weights):
    if weights is None:
        return probes, target
    weights = _finite_array(weights, "weights")
    if np.any(weights < 0) or np.any(np.sum(weights, axis=-1) <= 0):
        raise ValueError("Expected nonnegative weights with positive sum per fit; got invalid weights")
    # Weight broadcasting has already been checked by the caller. A weight
    # multiplies a squared residual, hence sqrt(weight) multiplies its row.
    root = np.sqrt(weights)[..., :, None]
    return probes * root, target * root


def _svd_solve(probes, target, ridge, require_full_rank=False):
    """Stable batched least squares, without forming the normal equations."""
    u, singular, vh = np.linalg.svd(probes, full_matrices=False)
    cutoff = np.finfo(np.float64).eps * max(probes.shape[-2:]) * singular[..., :1]
    identifiable = singular > cutoff
    if require_full_rank and (
        singular.shape[-1] < probes.shape[-1]
        or np.any(np.sum(identifiable, axis=-1) < probes.shape[-1])
    ):
        raise ValueError(
            "Expected a full-column-rank weighted probe design to identify the complete response; "
            f"got design shape {probes.shape} with deficient rank"
        )
    if ridge > 0:
        factors = singular / (singular**2 + ridge)
    else:
        factors = np.zeros_like(singular)
        np.divide(1.0, singular, out=factors, where=identifiable)
    projected = np.swapaxes(u, -1, -2) @ target
    return np.swapaxes(vh, -1, -2) @ (factors[..., :, None] * projected)


def estimate_adjoint(q, probes, y, method, *, ridge=0.0, weights=None):
    """Correct baseline q using the measured equations ``probes @ lambda=y``.

    Methods:
      * ``mc``: q + (n/m) sum z (y-z.T q). Requires unit probes and a
        randomized isotropic design. It is unbiased over the design in exact
        linear response; an individual correction can increase error.
      * ``kaczmarz``: one ordered pass of exact row projections, starting at
        q. Arbitrary nonzero row norms are allowed. Each noiseless consistent
        equation decreases squared Euclidean adjoint error. Noisy equations
        need not improve the estimate.
      * ``lstsq``: fit the minimum-norm correction to q using an SVD. This
        retains q in the unmeasured nullspace, including underdetermined fits.
      * ``ridge``: minimize sum w_k (y_k-z_k.T l)^2 + ridge * ||l-q||^2.
        SVD filtering avoids squaring the design condition number. At zero
        ridge it has the same minimum-norm correction as ``lstsq``.
      * ``orthogonal``: q + sum z (y-z.T q), after checking orthonormal rows.
        With exact readings it corrects the measured subspace exactly and
        retains q in its orthogonal complement.

    Weights apply only to regression methods and multiply squared residuals.
    Their absolute scale matters relative to ridge. Only ``ridge`` accepts a
    nonzero ridge parameter. No noise level or optimal regularizer is inferred
    from an exact Jacobian or reference adjoint.
    """
    allowed = ("mc", "kaczmarz", "lstsq", "ridge", "orthogonal")
    if method not in allowed:
        raise ValueError(f"Expected estimator method in {allowed}; got {method!r}")
    _check_ridge(ridge)
    if method != "ridge" and ridge != 0:
        raise ValueError(f"Expected ridge=0 for estimator {method!r}; got {ridge}")
    if method not in ("lstsq", "ridge") and weights is not None:
        raise ValueError(f"Expected weights only for lstsq or ridge; got method {method!r}")
    q, probes, y = (_finite_array(value, label)
                    for value, label in ((q, "q"), (probes, "probes"), (y, "readings")))
    if (q.ndim < 1 or probes.ndim < 2 or y.ndim < 1
            or probes.shape[-2] < 1 or probes.shape[-1] < 1
            or q.shape[-1] != probes.shape[-1] or y.shape[-1] != probes.shape[-2]):
        raise ValueError(
            "Expected q (...,n), probes (...,m,n), readings (...,m), n,m >= 1; "
            f"got {q.shape}, {probes.shape}, {y.shape}"
        )
    shapes = [q.shape[:-1], probes.shape[:-2], y.shape[:-1]]
    if weights is not None:
        weights = _finite_array(weights, "weights")
        if weights.ndim < 1 or weights.shape[-1] != probes.shape[-2]:
            raise ValueError(f"Expected weights (...,{probes.shape[-2]}); got {weights.shape}")
        shapes.append(weights.shape[:-1])
    try:
        batch = np.broadcast_shapes(*shapes)
    except ValueError as exc:
        raise ValueError(f"Expected broadcastable batch dimensions; got {shapes}") from exc
    size, count = probes.shape[-1], probes.shape[-2]
    q = np.broadcast_to(q, batch + (size,))
    probes = np.broadcast_to(probes, batch + (count, size))
    y = np.broadcast_to(y, batch + (count,))
    if weights is not None:
        weights = np.broadcast_to(weights, batch + (count,))
    residual = y - np.einsum("...mn,...n->...m", probes, q)

    if method in ("mc", "orthogonal"):
        norms = np.sum(probes**2, axis=-1)
        if not np.allclose(norms, 1.0, rtol=1e-10, atol=1e-12):
            raise ValueError(f"Expected unit-L2 probe rows for {method}; got squared norms {norms}")
        if method == "orthogonal":
            if count > size or not np.allclose(
                probes @ np.swapaxes(probes, -1, -2), np.eye(count), rtol=1e-10, atol=1e-10
            ):
                raise ValueError("Expected mutually orthonormal probe rows; got a nonorthogonal design")
        scale = size / count if method == "mc" else 1.0
        return q + scale * np.einsum("...mn,...m->...n", probes, residual)
    if method == "kaczmarz":
        norms = np.sum(probes**2, axis=-1)
        if np.any(norms <= 0):
            raise ValueError("Expected nonzero probe rows for kaczmarz; got a zero row")
        estimate = q.copy()
        for index in range(count):
            z = probes[..., index, :]
            error = y[..., index] - np.sum(z * estimate, axis=-1)
            estimate += z * (error / norms[..., index])[..., None]
        return estimate
    design, target = _weight_design(probes, residual[..., None], weights)
    return q + _svd_solve(design, target, ridge)[..., 0]


def fit_response(probes, responses, *, ridge=0.0, prior=None, weights=None):
    """Fit a complete response R from rows ``responses[k] = R @ probes[k]``.

    Inputs have shapes (..., m, n_inputs) and (..., m, n_outputs); the return
    shape is (..., n_outputs, n_inputs). The fit minimizes the weighted sum
    of squared response residuals plus ridge * ||R-prior||_F^2. A missing
    prior is zero. Batch dimensions broadcast; the calculation uses an SVD.

    A full-column-rank *weighted* design is required even with a prior or
    ridge. A deficient design identifies only a subspace; returning it as a
    complete experimentally identified response would conceal that limitation.
    Use ``estimate_adjoint`` when only the current adjoint is needed.
    """
    _check_ridge(ridge)
    probes = _finite_array(probes, "probes")
    responses = _finite_array(responses, "responses")
    if (probes.ndim < 2 or responses.ndim < 2 or min(probes.shape[-2:]) < 1
            or responses.shape[-1] < 1 or responses.shape[-2] != probes.shape[-2]):
        raise ValueError(
            "Expected probes (...,m,n_inputs), responses (...,m,n_outputs), dimensions >= 1; "
            f"got {probes.shape}, {responses.shape}"
        )
    count, size = probes.shape[-2:]
    outputs = responses.shape[-1]
    shapes = [probes.shape[:-2], responses.shape[:-2]]
    if prior is not None:
        prior = _finite_array(prior, "response prior")
        if prior.ndim < 2 or prior.shape[-2:] != (outputs, size):
            raise ValueError(f"Expected prior (...,{outputs},{size}); got {prior.shape}")
        shapes.append(prior.shape[:-2])
    if weights is not None:
        weights = _finite_array(weights, "weights")
        if weights.ndim < 1 or weights.shape[-1] != count:
            raise ValueError(f"Expected weights (...,{count}); got {weights.shape}")
        shapes.append(weights.shape[:-1])
    try:
        batch = np.broadcast_shapes(*shapes)
    except ValueError as exc:
        raise ValueError(f"Expected broadcastable batch dimensions; got {shapes}") from exc
    probes = np.broadcast_to(probes, batch + (count, size))
    responses = np.broadcast_to(responses, batch + (count, outputs))
    prior = (np.zeros(batch + (outputs, size)) if prior is None
             else np.broadcast_to(prior, batch + (outputs, size)))
    if weights is not None:
        weights = np.broadcast_to(weights, batch + (count,))
    target = responses - probes @ np.swapaxes(prior, -1, -2)
    design, target = _weight_design(probes, target, weights)
    correction = _svd_solve(design, target, ridge, require_full_rank=True)
    return prior + np.swapaxes(correction, -1, -2)
