"""Calibrate skew feedback from noisy physical trajectories, without derivatives.

The simulator integrates the explicitly specified additive-noise physical SDE
with stochastic Heun steps. Noise is not inserted into the equilibrium solver.
An opaque force callable is the only access to the unmodified system. The
controller sees state readings and learns from their antisymmetric time order.
"""

from dataclasses import dataclass
import time

import numpy as np


def area_rate(previous, current, dt):
    """Estimate E[dx x.T - x dx.T]/dt from simultaneous trajectory pairs."""
    previous = np.asarray(previous, dtype=float)
    current = np.asarray(current, dtype=float)
    if previous.shape != current.shape or previous.ndim < 1:
        raise ValueError(f"Expected matching (..., states) arrays; got {previous.shape}, {current.shape}")
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError(f"Expected finite positive dt; got {dt}")
    if not np.isfinite(previous).all() or not np.isfinite(current).all():
        raise ValueError("Expected finite state readings")
    old = previous.reshape(-1, previous.shape[-1])
    difference = (current-previous).reshape(old.shape)
    product = difference.T @ old / (len(old)*dt)
    return product-product.T


@dataclass
class CalibrationResult:
    matrix: np.ndarray
    final_matrix: np.ndarray
    stats: dict
    history: list


def calibrate_controller(force, center, *, duration=200.0, dt=0.02,
                         temperature=0.05, replicas=32, learning_rate=0.1,
                         update_interval=10, burn_in=10.0, average_after=50.0,
                         seed=0, read_noise=0.0, callback=None):
    """Learn C approximately equal to minus the constant physical skew part.

    Calibration drift is force(s) + C @ (s-center), C skew. A window update is
    C <- C - learning_rate/(2*temperature) * area_rate * window_duration.
    The reported matrix averages controller iterates after average_after units
    of adaptive calibration time; this average is not fed back during fitting.

    replicas are independent noise records of the SAME physical system. They
    represent repeated/parallel calibration experiments and their complete
    observation time is charged, not treated as free hardware parallelism.
    force accepts an array (replicas, states). It must not include C itself.
    callback(info, current_C, averaged_C) is read-only diagnostic output.
    """
    center = np.asarray(center, dtype=float)
    if center.ndim != 1 or len(center) < 2 or not np.isfinite(center).all():
        raise ValueError(f"Expected finite center vector of at least two states; got {center.shape}")
    for name, value in dict(duration=duration, dt=dt, temperature=temperature,
                            learning_rate=learning_rate).items():
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"Expected finite positive {name}; got {value}")
    for name, value in dict(replicas=replicas, update_interval=update_interval).items():
        if not isinstance(value, (int, np.integer)) or value < 1:
            raise ValueError(f"Expected positive integer {name}; got {value}")
    if not np.isfinite(burn_in) or burn_in < 0 or not 0 <= average_after < duration:
        raise ValueError(f"Expected burn_in >= 0 and 0 <= average_after < duration; got {burn_in}, {average_after}")
    if not np.isfinite(read_noise) or read_noise < 0:
        raise ValueError(f"Expected finite read_noise >= 0; got {read_noise}")
    size = len(center)
    rng = np.random.default_rng(seed)
    noise_rng = np.random.default_rng(seed+300000)
    warm_steps = int(np.ceil(burn_in/dt))
    adaptive_steps = int(np.ceil(duration/dt))
    state = np.broadcast_to(center, (replicas, size)).copy()
    controller = np.zeros((size, size))
    averaged = np.zeros_like(controller)
    averaged_time = 0.0
    previous_read = state-center
    if read_noise:
        previous_read = previous_read+noise_rng.normal(scale=read_noise, size=state.shape)
    old_window, new_window = [], []
    history = []
    started = time.perf_counter()
    updates = 0
    max_abs_state = float(np.max(np.abs(state)))
    last_report = -np.inf
    scale = np.sqrt(2*temperature*dt)
    for step in range(warm_steps+adaptive_steps):
        # Two force evaluations implement stochastic Heun for the physical
        # additive-noise SDE. No Jacobian, decomposition, solve or force samples
        # are provided to the controller update below.
        increment = rng.normal(scale=scale, size=state.shape)
        drift = force(state)+(state-center) @ controller.T
        predictor = state+dt*drift+increment
        second_drift = force(predictor)+(predictor-center) @ controller.T
        state = state+0.5*dt*(drift+second_drift)+increment
        if not np.isfinite(state).all():
            raise RuntimeError(f"Nonfinite stochastic physical state at step {step}; reduce integration dt")
        max_abs_state = max(max_abs_state, float(np.max(np.abs(state))))
        current_read = state-center
        if read_noise:
            current_read = current_read+noise_rng.normal(scale=read_noise, size=state.shape)
        if step >= warm_steps:
            old_window.append(previous_read)
            new_window.append(current_read)
            if len(old_window) == update_interval or step == warm_steps+adaptive_steps-1:
                elapsed = (step-warm_steps+1)*dt
                window_time = len(old_window)*dt
                omega = area_rate(np.concatenate(old_window), np.concatenate(new_window), dt)
                controller -= learning_rate/(2*temperature)*window_time*omega
                controller = (controller-controller.T)/2
                updates += 1
                weight = max(0.0, elapsed-max(average_after, elapsed-window_time))
                if weight:
                    averaged_time += weight
                    averaged += weight/averaged_time*(controller-averaged)
                if elapsed-last_report >= max(1.0, duration/40) or step == warm_steps+adaptive_steps-1:
                    row = dict(adaptive_time=elapsed, total_time=elapsed+warm_steps*dt,
                               updates=updates, circulation_rms=float(np.sqrt(np.mean(omega**2))),
                               controller_rms=float(np.sqrt(np.mean(controller**2))),
                               averaged_time=averaged_time,
                               state_rms=float(np.sqrt(np.mean((state-center)**2))),
                               elapsed_seconds=time.perf_counter()-started)
                    history.append(row)
                    if callback is not None:
                        callback(dict(row), controller.copy(), averaged.copy() if averaged_time else controller.copy())
                    last_report = elapsed
                old_window.clear()
                new_window.clear()
        previous_read = current_read
    if not averaged_time:
        raise RuntimeError("Expected at least one averaged controller update")
    total_steps = warm_steps+adaptive_steps
    stats = dict(states=size, replicas=replicas, adaptive_time=adaptive_steps*dt,
                 burn_in_time=warm_steps*dt, dt=dt, temperature=temperature,
                 learning_rate=learning_rate, update_interval=update_interval,
                 average_after=average_after, averaging_time=averaged_time,
                 read_noise=read_noise, seed=seed, updates=updates,
                 state_reads=replicas*(total_steps+1),
                 scalar_state_reads=size*replicas*(total_steps+1),
                 physical_time_per_record=total_steps*dt,
                 total_record_time=replicas*total_steps*dt,
                 force_evaluations=2*replicas*total_steps,
                 independent_controller_coefficients=size*(size-1)//2,
                 max_abs_state=max_abs_state, elapsed_seconds=time.perf_counter()-started,
                 integration="stochastic Heun, additive isotropic physical white noise",
                 measurement="successive state readings; no force/Jacobian/adjoint given to plasticity",
                 averaged_controller_used_during_calibration=False)
    return CalibrationResult(averaged, controller, stats, history)
