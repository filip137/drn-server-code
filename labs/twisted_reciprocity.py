"""Experimental sign-routing adaptation of twisted reciprocity, with no probes.

The constant intertwining principle is established in Section 10 of
https://arxiv.org/html/2608.11585, not a novelty claim for this experiment.
With hidden signs -1 and output signs +1, same-sign symmetric couplings and
opposite-sign skew couplings give T J(s) T = J(s).T for every nonlinear state.
Ordinary output cost nudges therefore measure q with exact adjoint T q in the
infinitesimal limit. Only local signs change the contrastive learning rule.
This restriction changes the hypothesis class; it is not a correction for an
arbitrary existing asymmetric network. K is fixed and only S, inputs, bias train.
"""

from dataclasses import dataclass

import numpy as np

from labs.recurrent_eqprop import (
    Classifier, contrastive_gradient, cost_gradient, error_pair,
    make_classifier, settle,
)


@dataclass
class TwistedClassifier(Classifier):
    """Classifier with known diagonal sign involution and constrained weights."""

    @property
    def signs(self):
        return np.concatenate((-np.ones(self.hidden), np.ones(self.outputs)))

    @property
    def symmetric_mask(self):
        signs = self.signs
        return signs[:, None] == signs[None, :]

    def __post_init__(self):
        if not self.size > self.outputs >= 2:
            raise ValueError(f"Expected size > outputs >= 2; got {self.size}, {self.outputs}")
        for name in ("symmetric", "skew"):
            value = np.asarray(getattr(self, name))
            if value.shape != (self.size, self.size) or not np.isfinite(value).all():
                raise ValueError(f"Expected finite {name} shape {(self.size, self.size)}; got {value.shape}")
        if (not np.allclose(self.symmetric, self.symmetric.T, atol=1e-12, rtol=0)
                or not np.allclose(self.symmetric[~self.symmetric_mask], 0, atol=1e-12)
                or not np.allclose(np.diag(self.symmetric), 0, atol=1e-12)):
            raise ValueError("Expected symmetric zero-diagonal S on same-sign blocks; got incompatible couplings")
        if (not np.allclose(self.skew, -self.skew.T, atol=1e-12, rtol=0)
                or not np.allclose(self.skew[self.symmetric_mask], 0, atol=1e-12)):
            raise ValueError("Expected skew K on opposite-sign blocks; got incompatible couplings")
        if not np.isfinite(self.symmetric_cap) or not 0 < self.symmetric_cap < 1:
            raise ValueError(f"Expected 0 < symmetric_cap < 1; got {self.symmetric_cap}")

    def update(self, gradient, learning_rate):
        """Project weight proposals onto the sign architecture and stability cap."""
        recurrent, inputs, bias = gradient
        return super().update((recurrent*self.symmetric_mask, inputs, bias), learning_rate)


def make_twisted_classifier(seed, *, size=64, outputs=10, input_size=64,
                            asymmetry=1.0, cubic=0.25):
    """Draw a nonlinear classifier with fixed skew hidden/output connections."""
    base = make_classifier(seed, size=size, outputs=outputs,
                           input_size=input_size, asymmetry=asymmetry)
    signs = np.concatenate((-np.ones(base.hidden), np.ones(outputs)))
    mask = signs[:, None] == signs[None, :]
    symmetric = base.symmetric*mask
    symmetric *= 0.5/np.linalg.norm(symmetric, 2)
    skew = base.skew*(~mask)
    if asymmetry:
        skew *= asymmetry/np.linalg.norm(skew, 2)
    if not np.isfinite(cubic) or cubic < 0:
        raise ValueError(f"Expected finite cubic >= 0; got {cubic}")
    return TwistedClassifier(symmetric, skew, base.inputs, base.bias,
                             outputs=outputs, cubic=cubic)


def twisted_contrastive_gradient(model, x, pair):
    """Signed local correlations for trainable same-sign S blocks, inputs, bias.

    On an allowed S edge both endpoint signs coincide, so multiplying its
    ordinary centered correlation contrast by either sign produces the
    constrained parameter gradient. Cross-sign S edges are not parameters.
    Finite paired cost nudges retain O(beta**2) smooth-nudge bias.
    """
    recurrent, inputs, bias = contrastive_gradient(model, x, pair)
    signs = model.signs
    return (recurrent*signs[:, None]*model.symmetric_mask,
            inputs*signs[:model.hidden, None], bias*signs)


def twisted_gradient_step(model, x, labels, rng=None, *, beta=0.01, sigma=0.0,
                          tolerance=1e-9):
    """Free equilibrium plus two actual cost nudges; no random response probes.

    rng is used only for optional read noise. This physical path has no audit,
    Jacobian, adjoint oracle, surrogate fit, or dense linear-system solve.
    Counts include every example: a batch of B uses 3*B equilibrations.
    """
    network = model.network()
    free = settle(network, model.drive(x), tolerance=tolerance)
    c = cost_gradient(free.state, labels, model.hidden, model.logit_scale)
    pair = error_pair(model, network, x, labels, free.state, beta,
                      sigma=sigma, rng=rng, tolerance=tolerance)
    metadata = dict(
        equilibrations=free.equilibrations+pair.equilibrations,
        state_reads=free.equilibrations+pair.equilibrations,
        relaxation_iterations=free.iterations+pair.iterations,
        max_residual=max(free.residual, pair.residual),
        probe_count=0, total_probe_excitation_sq=0.0,
        total_error_excitation_sq=float(2*np.sum(
            (pair.effective_beta*np.linalg.norm(c, axis=-1))**2)),
        mean_free_error_force_norm=float(np.linalg.norm(c, axis=-1).mean()),
        physical_phases_per_example=3,
    )
    return twisted_contrastive_gradient(model, x, pair), metadata
