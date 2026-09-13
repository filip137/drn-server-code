"""Adjust known skew-loop gains from exchanged physical response measurements.

The learner receives only one measured reciprocity defect per loop. The
measurement callback owns equilibrium experiments and never supplies a force
Jacobian, adjoint, unknown gains, or full response matrix to the learner.
"""

import numpy as np


def calibrate_response_loops(measure, loops, *, steps=60, learning_rate=0.2,
                             callback=None):
    """Use h_j=<B_j,R_C.T-R_C> to update c_j <- c_j - rate*h_j.

    ``measure(coefficients)`` returns (defects, physical_counts). Coefficients
    and callback arguments are copies so diagnostics cannot steer calibration.
    The fixed measurement budget is independent of diagnostic reference errors.
    Continuous stationary descent does not guarantee finite noisy-step descent.
    """
    if not isinstance(loops, (int, np.integer)) or loops < 1:
        raise ValueError(f"Expected a positive loop count; got {loops}")
    if not isinstance(steps, (int, np.integer)) or steps < 1:
        raise ValueError(f"Expected a positive step count; got {steps}")
    if not np.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError(f"Expected finite positive learning_rate; got {learning_rate}")
    coefficients, history, counts = np.zeros(loops), [], {}
    for step in range(steps):
        defect, measurement_counts = measure(coefficients.copy())
        defect = np.asarray(defect, dtype=float)
        if defect.shape != (loops,) or not np.isfinite(defect).all():
            raise ValueError(f"Expected {loops} finite measured defects; got {defect}")
        coefficients -= learning_rate*defect
        for name, value in measurement_counts.items():
            counts[name] = counts.get(name, 0)+value
        row = dict(step=step+1, defect_norm=float(np.linalg.norm(defect)),
                   coefficient_norm=float(np.linalg.norm(coefficients)))
        history.append(row)
        if callback is not None:
            callback(dict(row), coefficients.copy())
    return coefficients, counts, history
