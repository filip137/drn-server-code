"""Control variates learned from local slopes and past physical probe readings."""

from dataclasses import dataclass

import numpy as np


def local_slope_baseline(state, cost, cubic):
    """Isolated-node response; needs state and known local nonlinearity only."""
    return -cost / (1 + 3*cubic*state**2)


@dataclass
class MeasuredBaseline:
    """Predict a residual feedback map from previous measured scalar equations.

    H has shape (states, outputs). No exact feedback targets are used. The
    caller must predict before measuring a minibatch, form its MC correction,
    then call observe to prepare the baseline for the following minibatch.
    """

    matrix: np.ndarray
    relaxation: float = 0.25
    observations: int = 0

    def __post_init__(self):
        if not np.isfinite(self.relaxation) or not 0 < self.relaxation <= 1:
            raise ValueError(f"Expected relaxation in (0, 1]; got {self.relaxation}")
        if self.matrix.ndim != 2 or not np.isfinite(self.matrix).all():
            raise ValueError("Expected a finite (states, outputs) predictor matrix")

    @classmethod
    def zeros(cls, size, outputs, relaxation=0.25):
        return cls(np.zeros((size, outputs)), relaxation)

    def predict(self, state, cost, cubic):
        outputs = self.matrix.shape[1]
        diagonal = 1 + 3*cubic*state**2
        return (-cost + cost[:, -outputs:] @ self.matrix.T) / diagonal

    def observe(self, state, cost, cubic, probes, readings):
        """Relaxed normalized least squares, using each scalar reading once.

        Each equation is a.T H e = y - u.T(-c/D), a=u/D and e=c_output.
        The update is a rank-one Kaczmarz step in H. This fits a shared
        approximate response; changing states and parameters limit its accuracy.
        """
        diagonal = 1 + 3*cubic*state**2
        local = -cost / diagonal
        target = readings - np.einsum("bmn,bn->bm", probes, local)
        outputs = self.matrix.shape[1]
        for i in range(len(state)):
            error = cost[i, -outputs:]
            error_sq = float(error @ error)
            for k in range(probes.shape[1]):
                direction = probes[i, k] / diagonal[i]
                denominator = float(direction @ direction) * error_sq
                if denominator <= 1e-20:
                    continue
                residual = target[i, k] - direction @ self.matrix @ error
                self.matrix += (self.relaxation * residual / denominator) * np.outer(direction, error)
                self.observations += 1


@dataclass
class AveragedMomentum:
    """Bias-corrected EMA; mu=0 reproduces the existing projected SGD."""

    momentum: float = 0.0
    moments: tuple | None = None
    steps: int = 0

    def __post_init__(self):
        if not np.isfinite(self.momentum) or not 0 <= self.momentum < 1:
            raise ValueError(f"Expected momentum in [0, 1); got {self.momentum}")

    def direction(self, gradient):
        if self.moments is None:
            self.moments = tuple(np.zeros_like(g) for g in gradient)
        self.steps += 1
        self.moments = tuple(self.momentum*v + (1-self.momentum)*g
                             for v, g in zip(self.moments, gradient))
        denominator = 1 - self.momentum**self.steps
        return tuple(v / denominator for v in self.moments)
