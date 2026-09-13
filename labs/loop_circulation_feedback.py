"""Calibrate known skew feedback loops from their projected stochastic areas.

Only the loop gains are learned. The left/right sensor and actuator patterns
are supplied physical wiring, independent of the unknown force coefficients.
The simulated physical SDE evolves all states, but calibration observes only
the two projection channels associated with each loop.
"""

from dataclasses import dataclass
import time

import numpy as np


def loop_area_rate(previous, current, dt):
    """Return <B_j, Omega> from (..., 2*loops) projection readings.

    Channels are ordered [left projections, right projections]. For
    B_j = (u_j v_j.T - v_j u_j.T)/sqrt(2), this is
    sqrt(2) * E[right * dleft - left * dright] / dt.
    """
    previous = np.asarray(previous, dtype=float)
    current = np.asarray(current, dtype=float)
    if (previous.shape != current.shape or previous.ndim < 1
            or previous.shape[-1] < 2 or previous.shape[-1] % 2
            or previous.size == 0):
        raise ValueError(
            "Expected matching nonempty (..., 2*loops) readings; "
            f"got {previous.shape}, {current.shape}")
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError(f"Expected finite positive dt; got {dt}")
    if not np.isfinite(previous).all() or not np.isfinite(current).all():
        raise ValueError("Expected finite projection readings")
    old = previous.reshape(-1, previous.shape[-1])
    difference = (current-previous).reshape(old.shape)
    loops = old.shape[-1]//2
    return np.sqrt(2)*np.mean(
        old[:, loops:]*difference[:, :loops]
        - old[:, :loops]*difference[:, loops:], axis=0)/dt


@dataclass
class LoopCalibrationResult:
    coefficients: np.ndarray
    final_coefficients: np.ndarray
    matrix: np.ndarray
    stats: dict
    history: list


def calibrate_loop_controller(force, center, left, right, *, duration=100.0,
                              dt=0.02, temperature=0.05, replicas=16,
                              learning_rate=0.1, update_interval=10,
                              burn_in=10.0, average_after=30.0, seed=0,
                              read_noise=0.0, callback=None):
    """Learn L gains for known orthonormal rank-two skew feedback loops.

    ``left`` and ``right`` have shape (states, L); all 2L columns together
    must be orthonormal. The force callable accepts (replicas, states) and
    describes the unmodified physical system. No derivatives or measured
    force values enter the learning rule.

    Calibration applies Cx = sum_j c_j (u_j(v_j.T x)-v_j(u_j.T x))/sqrt(2),
    with x = s-center. Each window updates c_j by its measured stochastic
    area, normalized by learning_rate/(2*temperature). Read noise is added
    only to the 2L measured projections, not to the controller's physical
    sensing/injection paths. ``replicas`` are separately charged records.

    The returned coefficients average iterates after ``average_after``;
    this average is not used during calibration. A dense matrix is assembled
    once, on return, for compatibility with existing EqProp simulations.
    callback(info, current_coefficients, averaged_coefficients) is diagnostic.
    """
    center = np.asarray(center, dtype=float)
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    if center.ndim != 1 or len(center) < 2 or not np.isfinite(center).all():
        raise ValueError(f"Expected finite center vector of at least two states; got {center.shape}")
    if (left.ndim != 2 or right.shape != left.shape
            or left.shape[0] != len(center) or left.shape[1] < 1
            or not np.isfinite(left).all() or not np.isfinite(right).all()):
        raise ValueError(
            "Expected finite matching (states, loops) wiring arrays; "
            f"got {left.shape}, {right.shape} for {len(center)} states")
    wiring = np.concatenate((left, right), axis=1)
    if (wiring.shape[1] > len(center)
            or not np.allclose(wiring.T @ wiring, np.eye(wiring.shape[1]),
                               atol=1e-10, rtol=1e-8)):
        raise ValueError("Expected all 2*loops left/right wiring columns to be orthonormal")
    for name, value in dict(duration=duration, dt=dt, temperature=temperature,
                            learning_rate=learning_rate).items():
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"Expected finite positive {name}; got {value}")
    for name, value in dict(replicas=replicas, update_interval=update_interval).items():
        if not isinstance(value, (int, np.integer)) or value < 1:
            raise ValueError(f"Expected positive integer {name}; got {value}")
    if (not np.isfinite(burn_in) or burn_in < 0
            or not np.isfinite(average_after) or not 0 <= average_after < duration):
        raise ValueError(
            "Expected burn_in >= 0 and 0 <= average_after < duration; "
            f"got {burn_in}, {average_after}")
    if not np.isfinite(read_noise) or read_noise < 0:
        raise ValueError(f"Expected finite read_noise >= 0; got {read_noise}")

    size, loops = left.shape
    channels = 2*loops
    rng = np.random.default_rng(seed)
    read_rng = np.random.default_rng(seed+300000)
    warm_steps = int(np.ceil(burn_in/dt))
    adaptive_steps = int(np.ceil(duration/dt))
    total_steps = warm_steps+adaptive_steps
    state = np.broadcast_to(center, (replicas, size)).copy()
    coefficients = np.zeros(loops)
    averaged = np.zeros(loops)
    averaged_time = 0.0
    area_integral = np.zeros(loops)
    window_steps = 0
    updates = 0
    history = []
    last_report = -np.inf
    started = time.perf_counter()
    sqrt_two = np.sqrt(2)

    def controller_force(states):
        displacement = states-center
        return (((displacement @ right)*coefficients) @ left.T
                - ((displacement @ left)*coefficients) @ right.T)/sqrt_two

    def read_projections(states):
        reading = (states-center) @ wiring
        if read_noise:
            reading += read_rng.normal(scale=read_noise, size=reading.shape)
        return reading

    previous_read = read_projections(state)
    scale = np.sqrt(2*temperature*dt)
    for step in range(total_steps):
        # Stochastic Heun evolves the physical force and the known loop
        # circuit. Only read_projections supplies the plasticity measurements.
        increment = rng.normal(scale=scale, size=state.shape)
        drift = force(state)+controller_force(state)
        predictor = state+dt*drift+increment
        second_drift = force(predictor)+controller_force(predictor)
        state = state+0.5*dt*(drift+second_drift)+increment
        if not np.isfinite(state).all():
            raise RuntimeError(f"Nonfinite stochastic physical state at step {step}; reduce integration dt")
        current_read = read_projections(state)
        if step >= warm_steps:
            area_integral += dt*loop_area_rate(previous_read, current_read, dt)
            window_steps += 1
            if window_steps == update_interval or step == total_steps-1:
                elapsed = (step-warm_steps+1)*dt
                window_time = window_steps*dt
                measured_rate = area_integral/window_time
                coefficients -= learning_rate/(2*temperature)*area_integral
                updates += 1
                weight = max(0.0, elapsed-max(average_after, elapsed-window_time))
                if weight:
                    averaged_time += weight
                    averaged += weight/averaged_time*(coefficients-averaged)
                if elapsed-last_report >= max(1.0, duration/40) or step == total_steps-1:
                    row = dict(
                        adaptive_time=elapsed, total_time=elapsed+warm_steps*dt,
                        updates=updates,
                        projected_circulation_rms=float(np.sqrt(np.mean(measured_rate**2))),
                        coefficient_rms=float(np.sqrt(np.mean(coefficients**2))),
                        projection_rms=float(np.sqrt(np.mean(current_read**2))),
                        averaged_time=averaged_time,
                        elapsed_seconds=time.perf_counter()-started)
                    history.append(row)
                    if callback is not None:
                        callback(dict(row), coefficients.copy(),
                                 averaged.copy() if averaged_time else coefficients.copy())
                    last_report = elapsed
                area_integral.fill(0)
                window_steps = 0
        previous_read = current_read
    if not averaged_time:
        raise RuntimeError("Expected at least one averaged loop-controller update")

    reads = replicas*(total_steps+1)
    stats = dict(
        states=size, loops=loops, replicas=replicas,
        adaptive_time=adaptive_steps*dt, burn_in_time=warm_steps*dt, dt=dt,
        temperature=temperature, learning_rate=learning_rate,
        update_interval=update_interval, average_after=average_after,
        averaging_time=averaged_time, read_noise=read_noise, seed=seed,
        updates=updates, readout_channels=channels,
        state_reads=reads, scalar_measurement_reads=channels*reads,
        scalar_state_reads=channels*reads,
        process_noise_channels=size,
        process_noise_scalar_increments=replicas*total_steps*size,
        physical_time_per_record=total_steps*dt,
        total_record_time=replicas*total_steps*dt,
        force_evaluations=2*replicas*total_steps,
        learned_parameters=loops, fixed_wiring_coefficients=2*size*loops,
        elapsed_seconds=time.perf_counter()-started,
        integration="stochastic Heun, additive isotropic physical white noise",
        measurement="2L wiring projections; no state covariance/Jacobian/adjoint given to plasticity",
        read_noise_location="projection readout only; physical feedback paths ideal",
        averaged_controller_used_during_calibration=False)
    # This compatibility output is the sole dense controller allocation.
    forward = (left*averaged) @ right.T
    matrix = (forward-forward.T)/sqrt_two
    return LoopCalibrationResult(averaged, coefficients, matrix, stats, history)
