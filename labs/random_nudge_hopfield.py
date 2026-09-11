"""Random-nudge adjoints in a small, dissipative continuous Hopfield network.

The measurement path uses forward relaxation only. Dense Jacobian solves in
``exact_feedback`` are reference calculations, never inputs to the estimator.
All arrays use float64 and the last axis indexes the free state variables.
"""

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class Hopfield:
    """F(s) = -s - cubic*s**3 + W@s + drive.

    For symmetric W this is -grad(E), with E = |s|²/2 + cubic*sum(s**4)/4
    - s.T@W@s/2 - drive.T@s. A skew coupling makes it non-conservative.
    Requiring lambda_max(sym(W)) < 1 guarantees a unique globally attracting
    equilibrium for every constant drive, including constant nudges.
    """

    weights: np.ndarray
    cubic: float = 0.25
    margin: float = field(init=False)
    weight_norm: float = field(init=False)

    def __post_init__(self):
        w = np.array(self.weights, dtype=np.float64, copy=True)
        if w.ndim != 2 or w.shape[0] != w.shape[1] or not np.isfinite(w).all():
            raise ValueError(f"Expected a finite square weight matrix; got shape {w.shape}")
        if w.shape[0] < 1 or not np.isfinite(self.cubic) or self.cubic < 0:
            raise ValueError(f"Expected nonempty weights and cubic >= 0; got {self.cubic}")
        margin = 1.0 - np.linalg.eigvalsh((w + w.T) / 2)[-1]
        if margin <= 0:
            raise ValueError(f"Expected lambda_max(sym(W)) < 1; got {1 - margin}")
        w.setflags(write=False)
        object.__setattr__(self, "weights", w)
        object.__setattr__(self, "margin", float(margin))
        object.__setattr__(self, "weight_norm", float(np.linalg.norm(w, 2)))

    @property
    def size(self):
        return self.weights.shape[0]

    def force(self, state, drive):
        return -state - self.cubic * state**3 + state @ self.weights.T + drive

    def jacobian(self, state):
        """Evaluation oracle; supports batches of free equilibria."""
        state = np.asarray(state, dtype=np.float64)
        return self.weights - (1 + 3 * self.cubic * state[..., :, None] ** 2) * np.eye(self.size)


def make_network(size, asymmetry, seed=0, cubic=0.25, symmetric_norm=0.4):
    """Matched symmetric/skew draws across asymmetry and cubic settings."""
    if size < 2 or not np.isfinite(asymmetry) or asymmetry < 0:
        raise ValueError(f"Expected size >= 2 and finite asymmetry >= 0; got {size}, {asymmetry}")
    if not 0 <= symmetric_norm < 1:
        raise ValueError(f"Expected 0 <= symmetric_norm < 1; got {symmetric_norm}")
    rng = np.random.default_rng(seed)
    raw = rng.normal(size=(size, size))
    symmetric = (raw + raw.T) / 2
    np.fill_diagonal(symmetric, 0)
    symmetric *= symmetric_norm / np.linalg.norm(symmetric, 2)
    raw = rng.normal(size=(size, size))
    skew = (raw - raw.T) / 2
    skew /= np.linalg.norm(skew, 2)
    return Hopfield(symmetric + asymmetry * skew, cubic)


@dataclass
class Relaxation:
    state: np.ndarray
    iterations: int
    residual: float
    equilibrations: int


def relax(network, drive, initial=None, *, tolerance=1e-12, max_steps=20000, dt=0.2):
    """Forward Euler relaxation, checked against the actual force residual.

    A conservative step cap accounts for the skew coupling and local cubic
    stiffness. No Newton solver, inverse, transpose dynamics or autodiff is used.
    Batched trajectories all finish only when the largest residual is small.
    """
    drive = np.asarray(drive, dtype=np.float64)
    if drive.ndim < 1 or drive.shape[-1] != network.size or not np.isfinite(drive).all():
        raise ValueError(f"Expected finite drive with last dimension {network.size}; got {drive.shape}")
    if not np.isfinite(tolerance) or tolerance <= 0 or not np.isfinite(dt) or dt <= 0 or max_steps < 1:
        raise ValueError(f"Expected positive tolerance, dt and max_steps; got {tolerance}, {dt}, {max_steps}")
    state = np.zeros_like(drive) if initial is None else np.broadcast_to(np.asarray(initial, dtype=np.float64), drive.shape).copy()
    if not np.isfinite(state).all():
        raise ValueError("Expected finite initial state")
    for iteration in range(max_steps + 1):
        force = network.force(state, drive)
        residual = float(np.max(np.abs(force)))
        if not np.isfinite(residual):
            raise RuntimeError(f"Relaxation became non-finite at iteration {iteration}")
        if residual <= tolerance:
            return Relaxation(state, iteration, residual, int(np.prod(drive.shape[:-1])))
        if iteration == max_steps:
            break
        lipschitz = 1 + network.weight_norm + 3 * network.cubic * np.max(state**2)
        step = min(dt, network.margin / lipschitz**2)
        state += step * force
    raise RuntimeError(f"Relaxation failed after {max_steps} iterations: residual {residual:.3e} > {tolerance:.3e}")


@dataclass
class Response:
    value: np.ndarray
    equilibrations: int
    iterations: int
    residual: float


def measure_response(network, drive, free, direction, beta, *, scheme="central", tolerance=1e-12):
    """Measure ds/dbeta for F(s)-beta*direction=0 by actual equilibrations.

    ``direction`` is held fixed at its free-phase value. For a cost gradient
    this is the frozen-error nudge (the linearized cost), whose infinitesimal
    response agrees with ordinary EqProp. Random signs are not normalized.
    Read noise can be added later using ``add_read_noise``.
    """
    if not np.isfinite(beta) or beta <= 0:
        raise ValueError(f"Expected finite beta > 0; got {beta}")
    if scheme not in ("central", "forward"):
        raise ValueError(f"Expected scheme 'central' or 'forward'; got {scheme!r}")
    direction = np.asarray(direction, dtype=np.float64)
    positive = relax(network, np.asarray(drive) - beta * direction, free, tolerance=tolerance)
    if scheme == "forward":
        return Response((positive.state - free) / beta, positive.equilibrations,
                        positive.iterations, positive.residual)
    negative = relax(network, np.asarray(drive) + beta * direction, free, tolerance=tolerance)
    return Response((positive.state - negative.state) / (2 * beta),
                    positive.equilibrations + negative.equilibrations,
                    positive.iterations + negative.iterations,
                    max(positive.residual, negative.residual))


def add_read_noise(response, sigma, beta, rng, scheme="central"):
    """Independent Gaussian noise on each read of a settled state.

    Central differences use two independent reads: Var(eta)=sigma²/(2 beta²).
    Forward differences use an independent free-state read for EACH response:
    Var(eta)=2 sigma²/beta². Sharing a noisy free-state read across directions
    would be a different correlated-noise protocol and is not modeled here.
    """
    if not np.isfinite(sigma) or sigma < 0 or not np.isfinite(beta) or beta <= 0:
        raise ValueError(f"Expected finite sigma >= 0 and beta > 0; got {sigma}, {beta}")
    if scheme not in ("central", "forward"):
        raise ValueError(f"Expected scheme 'central' or 'forward'; got {scheme!r}")
    scale = sigma / (np.sqrt(2) * beta) if scheme == "central" else np.sqrt(2) * sigma / beta
    return np.asarray(response) + rng.normal(scale=scale, size=np.shape(response))


def rademacher(rng, shape):
    return (2 * rng.integers(0, 2, size=shape) - 1).astype(np.float64)


def hadamard_probes(size, rng=None):
    """A full orthogonal sign basis (m=n), an explicitly structured control."""
    if size < 1 or size & (size - 1):
        raise ValueError(f"Expected a positive power-of-two size; got {size}")
    h = np.ones((1, 1))
    while h.shape[0] < size:
        h = np.block([[h, h], [h, -h]])
    if rng is not None:
        h = h[rng.permutation(size)] * rademacher(rng, (size,))
    return h


def mismatch_projections(c, q, probes, responses):
    """d(z)=c.T r_z-z.T q; probe axis is second-to-last."""
    return np.sum(np.asarray(responses) * np.asarray(c)[..., None, :], axis=-1) - np.sum(
        np.asarray(probes) * np.asarray(q)[..., None, :], axis=-1)


def corrected_feedback(q, probes, projections, shrinkage=1.0):
    if not np.isfinite(shrinkage) or not 0 <= shrinkage <= 1:
        raise ValueError(f"Expected shrinkage in [0, 1]; got {shrinkage}")
    probes, projections = np.asarray(probes), np.asarray(projections)
    if probes.ndim < 2 or probes.shape[-2] < 1 or probes.shape[:-1] != projections.shape:
        raise ValueError(f"Expected probes (..., m, n) and projections (..., m), m >= 1; got {probes.shape}, {projections.shape}")
    return np.asarray(q) + shrinkage * np.mean(probes * projections[..., None], axis=-2)


def exact_feedback(network, free, c):
    """Dense reference q and adjoint, for validation or an oracle baseline."""
    j = network.jacobian(free)
    q = np.linalg.solve(j, np.asarray(c)[..., None])[..., 0]
    adjoint = np.linalg.solve(np.swapaxes(j, -1, -2), np.asarray(c)[..., None])[..., 0]
    return q, adjoint


def drive_parameter_gradient(feedback, inputs, trainable_nodes):
    """For drive_hidden = B@x, grad_B(C) = -lambda_hidden outer x."""
    return -np.einsum("bi,bj->ij", feedback[:, :trainable_nodes], inputs) / len(inputs)
