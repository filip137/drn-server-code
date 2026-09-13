"""Recurrent soft-spin Hopfield classification and actual contrastive EqProp.

The symmetric recurrent component, hidden input couplings and biases are trained.
A fixed skew component supplies controlled non-conservative coupling. The forward
solver uses local implicit leak/cubic updates, never a state Jacobian or adjoint.
"""

from dataclasses import dataclass

import numpy as np

from labs.random_nudge_hopfield import Hopfield, Relaxation


@dataclass
class Classifier:
    symmetric: np.ndarray
    skew: np.ndarray
    inputs: np.ndarray
    bias: np.ndarray
    outputs: int = 10
    cubic: float = 0.25
    logit_scale: float = 4.0
    symmetric_cap: float = 0.7

    @property
    def size(self):
        return len(self.bias)

    @property
    def hidden(self):
        return self.size - self.outputs

    def network(self):
        return Hopfield(self.symmetric + self.skew, self.cubic)

    def drive(self, x):
        drive = np.broadcast_to(self.bias, (len(x), self.size)).copy()
        drive[:, :self.hidden] += x @ self.inputs.T
        return drive

    def update(self, gradient, learning_rate):
        """Projected SGD, with the same stability constraint for every method."""
        gs, gb, gbias = gradient
        self.symmetric -= learning_rate * gs
        self.inputs -= learning_rate * gb
        self.bias -= learning_rate * gbias
        self.symmetric = (self.symmetric + self.symmetric.T) / 2
        np.fill_diagonal(self.symmetric, 0)
        norm = np.max(np.abs(np.linalg.eigvalsh(self.symmetric)))
        clipped = norm > self.symmetric_cap
        if clipped:
            self.symmetric *= self.symmetric_cap / norm
        return bool(clipped)


def make_classifier(seed, *, size=32, outputs=10, input_size=64, asymmetry=1.0):
    if not size > outputs >= 2:
        raise ValueError(f"Expected size > outputs >= 2; got {size}, {outputs}")
    if input_size < 1 or not np.isfinite(asymmetry) or asymmetry < 0:
        raise ValueError(f"Expected input_size >= 1 and finite asymmetry >= 0; got {input_size}, {asymmetry}")
    rng = np.random.default_rng(seed)
    raw = rng.normal(size=(size, size))
    symmetric = (raw + raw.T) / 2
    np.fill_diagonal(symmetric, 0)
    symmetric *= 0.5 / np.linalg.norm(symmetric, 2)
    raw = rng.normal(size=(size, size))
    skew = (raw - raw.T) / 2
    skew *= asymmetry / np.linalg.norm(skew, 2)
    inputs = rng.normal(scale=0.5 / np.sqrt(input_size), size=(size-outputs, input_size))
    return Classifier(symmetric, skew, inputs, np.zeros(size), outputs=outputs)


def probabilities(state, hidden, logit_scale):
    logits = logit_scale * state[..., hidden:]
    logits = logits - logits.max(axis=-1, keepdims=True)
    exponent = np.exp(logits)
    return exponent / exponent.sum(axis=-1, keepdims=True)


def cost_gradient(state, labels, hidden, logit_scale=4.0):
    p = probabilities(state, hidden, logit_scale)
    gradient = np.zeros_like(state)
    gradient[..., hidden:] = logit_scale * (p - np.eye(p.shape[-1])[labels])
    return gradient


def loss_accuracy(state, labels, hidden, logit_scale=4.0):
    logits = logit_scale * state[..., hidden:]
    maximum = logits.max(axis=-1)
    normalizer = maximum + np.log(np.exp(logits - maximum[..., None]).sum(axis=-1))
    chosen = np.take_along_axis(logits, np.asarray(labels)[..., None], axis=-1)[..., 0]
    return float(np.mean(normalizer - chosen)), float(np.mean(logits.argmax(axis=-1) == labels))


def settle(network, drive, initial=None, *, labels=None, hidden=None,
           logit_scale=4.0, cost_scale=0.0, skew_correction=None,
           center=None, tolerance=1e-9, max_steps=10000):
    """Settle F-beta*grad(C), optionally with known-skew AsymEP correction.

    IMEX Euler treats leak/cubic forces implicitly with a scalar closed-form
    inverse and treats coupling and the cost force explicitly. A contraction
    bound chooses dt from coupling norms; only the equilibrium is measured.
    Signed cost_scale implements the actual state-dependent cost during settling.
    """
    drive = np.asarray(drive, dtype=np.float64)
    if drive.ndim < 1 or drive.shape[-1] != network.size or not np.isfinite(drive).all():
        raise ValueError(f"Expected finite drive (..., {network.size}); got {drive.shape}")
    if not np.isfinite(tolerance) or tolerance <= 0 or max_steps < 1:
        raise ValueError(f"Expected positive tolerance/max_steps; got {tolerance}, {max_steps}")
    w = network.weights
    if skew_correction is not None:
        skew_correction = np.asarray(skew_correction)
        if center is None or not np.allclose(skew_correction, -skew_correction.T, atol=1e-12):
            raise ValueError("Expected an antisymmetric correction and a free-state center")
        w = w - 2 * skew_correction
        drive = drive + 2 * np.asarray(center) @ skew_correction.T
    cost_scale = np.asarray(cost_scale)
    if not np.isfinite(cost_scale).all():
        raise ValueError("Expected finite cost_scale")
    cost_bound = float(np.max(np.abs(cost_scale))) * logit_scale**2 / 2
    if cost_bound and (labels is None or hidden is None):
        raise ValueError("Expected labels and hidden boundary when applying a cost nudge")
    margin = network.margin - cost_bound
    if margin <= 0:
        raise ValueError(f"Expected cost-force bound below stability margin; got {cost_bound} >= {network.margin}")
    coupling_norm = (network.weight_norm if skew_correction is None
                     else float(np.linalg.norm(w, 2))) + cost_bound
    # ||I+dt*G'||/(1+dt)<1 if dt < 2*margin/(||G'||²-1).
    dt = min(0.8, margin / max(coupling_norm**2 - 1, 1e-12))
    a, b = 1 + dt, dt * network.cubic
    k = np.sqrt(3*b/a) if b else 0.0
    state = np.zeros_like(drive) if initial is None else np.broadcast_to(initial, drive.shape).astype(np.float64).copy()
    for iteration in range(max_steps + 1):
        coupled = state @ w.T + drive
        if cost_bound:
            coupled -= cost_scale[..., None] * cost_gradient(state, labels, hidden, logit_scale)
        force = coupled - state - network.cubic*state**3
        residual = float(np.max(np.abs(force)))
        if not np.isfinite(residual):
            raise RuntimeError(f"Non-finite forward force at iteration {iteration}")
        if residual <= tolerance:
            return Relaxation(state, iteration, residual, int(np.prod(drive.shape[:-1])))
        value = state + dt*coupled
        state = 2/k*np.sinh(np.arcsinh(1.5*k*value/a)/3) if b else value/a
    raise RuntimeError(f"Forward settle failed: residual {residual:.3e} after {max_steps} steps")


@dataclass
class NudgedPair:
    positive: np.ndarray
    negative: np.ndarray
    effective_beta: np.ndarray
    equilibrations: int
    iterations: int
    residual: float

    @property
    def response(self):
        return (self.positive - self.negative) / (2*self.effective_beta[..., None])


def error_pair(model, network, x, labels, free, beta, *, known_skew=False,
               skew_estimate=None, sigma=0.0, rng=None, tolerance=1e-9):
    """Actual +/- cost nudging, with normalized free-phase error force.

    beta_eff=beta/max(||c0||,1) keeps the initial error force <= beta and prevents
    a vanishing error from making the repulsive-phase cost curvature unbounded.
    Read noise affects settled +/- states; free state and target remain clean.
    """
    if not np.isfinite(beta) or beta <= 0 or not np.isfinite(sigma) or sigma < 0:
        raise ValueError(f"Expected finite beta > 0 and sigma >= 0; got {beta}, {sigma}")
    if known_skew and skew_estimate is not None:
        raise ValueError("Expected either known skew or a measured estimate, not both")
    correction = model.skew if known_skew else skew_estimate
    c = cost_gradient(free, labels, model.hidden, model.logit_scale)
    effective = beta / np.maximum(np.linalg.norm(c, axis=-1), 1.0)
    kw = dict(labels=labels, hidden=model.hidden, logit_scale=model.logit_scale,
              skew_correction=correction,
              center=free if correction is not None else None, tolerance=tolerance)
    drive = model.drive(x)
    plus = settle(network, drive, free, cost_scale=effective, **kw)
    minus = settle(network, drive, free, cost_scale=-effective, **kw)
    positive, negative = plus.state, minus.state
    if sigma:
        if rng is None:
            raise ValueError("Expected rng for nonzero read noise")
        positive = positive + rng.normal(scale=sigma, size=positive.shape)
        negative = negative + rng.normal(scale=sigma, size=negative.shape)
    return NudgedPair(positive, negative, effective, plus.equilibrations+minus.equilibrations,
                      plus.iterations+minus.iterations, max(plus.residual, minus.residual))


def probe_responses(network, drive, free, probes, beta, *, sigma=0.0, rng=None, tolerance=1e-9):
    """Physical paired constant-current probes, all with unit L2 directions."""
    if not np.isfinite(beta) or beta <= 0 or not np.isfinite(sigma) or sigma < 0:
        raise ValueError(f"Expected finite beta > 0 and sigma >= 0; got {beta}, {sigma}")
    if not np.allclose(np.linalg.norm(probes, axis=-1), 1.0, atol=1e-12):
        raise ValueError("Expected unit-L2 probe directions")
    plus = settle(network, drive[:, None, :] - beta*probes, free[:, None, :], tolerance=tolerance)
    minus = settle(network, drive[:, None, :] + beta*probes, free[:, None, :], tolerance=tolerance)
    response = (plus.state-minus.state)/(2*beta)
    if sigma:
        if rng is None:
            raise ValueError("Expected rng for nonzero read noise")
        response += rng.normal(scale=sigma/(np.sqrt(2)*beta), size=response.shape)
    return response, plus.equilibrations+minus.equilibrations, plus.iterations+minus.iterations, max(plus.residual,minus.residual)


def parameter_gradient(model, x, free, feedback):
    """Projected full-matrix convention for symmetric recurrent parameters."""
    recurrent = -np.einsum("bi,bj->ij", feedback, free)/len(x)
    recurrent = (recurrent+recurrent.T)/2
    np.fill_diagonal(recurrent, 0)
    inputs = -feedback[:, :model.hidden].T @ x / len(x)
    bias = -feedback.mean(axis=0)
    return recurrent, inputs, bias


def contrastive_gradient(model, x, pair):
    """Centered energy-partial contrast for S, B and b, including recurrent S.

    E contains -.5*s.T@S@s - s_hidden.T@B@x - b.T@s. This expression is
    directly applicable as the EP control and to known-skew AsymEP phases.
    """
    coefficient = 1 / (4*pair.effective_beta)
    recurrent = -(np.einsum("b,bi,bj->ij", coefficient, pair.positive, pair.positive)
                  - np.einsum("b,bi,bj->ij", coefficient, pair.negative, pair.negative))/len(x)
    np.fill_diagonal(recurrent, 0)
    inputs = -pair.response[:, :model.hidden].T @ x / len(x)
    bias = -pair.response.mean(axis=0)
    return recurrent, inputs, bias


def flatten_gradient(gradient):
    return np.concatenate([part.ravel() for part in gradient])
