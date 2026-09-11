#!/usr/bin/env python3
"""Exploratory, matched-excitation DC measurements of Hopfield adjoints.

Run ``python -m labs.tools.compare_adjoint_measurements --quick`` first.
Each direction has unit L2 norm. Measurements use paired forward relaxation;
Jacobian solves appear only in the reference/evaluation path. Noise trials
repeat voltage reads of the same clean equilibria, not network realizations.
"""

import argparse
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import shlex
import subprocess
import sys
import time

import numpy as np

from labs.adjoint_estimators import estimate_adjoint, fit_response, make_probes
from labs.random_nudge_hopfield import add_read_noise, make_network, measure_response, relax


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def write_csv(path, rows):
    keys = list(dict.fromkeys(key for row in rows for key in row))
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def physical_measurements(network, drive, free, c, probes, heldout, amplitude, tolerance):
    """Obtain normalized paired responses with no oracle or response inversion.

    The error nudge is c/||c|| so its total current norm equals every probe's.
    Rescaling its response by ||c|| recovers q. One q is reused by all probes.
    """
    norm_c = float(np.linalg.norm(c))
    if not norm_c > 0:
        raise ValueError(f"Expected a nonzero cost gradient; got norm {norm_c}")
    directions = np.concatenate([(c / norm_c)[None], probes, heldout], axis=0)
    response = measure_response(network, drive, free, directions, amplitude,
                                tolerance=tolerance)
    m = len(probes)
    return dict(q_unit=response.value[0], response=response.value[1:1 + m],
                heldout=response.value[1 + m:], norm_c=norm_c,
                residual=response.residual, batched_iterations=response.iterations,
                actual_equilibrations=response.equilibrations)


def design_statistics(probes):
    singular = np.linalg.svd(probes, compute_uv=False)
    threshold = np.finfo(float).eps * max(probes.shape) * singular[0]
    rank = int(np.sum(singular > threshold))
    return dict(design_rank=rank,
                identifiable_full_response=rank == probes.shape[-1],
                nonzero_design_condition=float(singular[0] / singular[rank - 1]),
                full_design_condition=(float(singular[0] / singular[-1])
                                       if rank == probes.shape[-1] else None))


def relative_error(estimate, reference):
    return float(np.linalg.norm(estimate - reference) / max(np.linalg.norm(reference), 1e-30))


def cosine(estimate, reference):
    denominator = np.linalg.norm(estimate) * np.linalg.norm(reference)
    return float(np.clip(np.sum(estimate * reference) / max(denominator, 1e-30), -1, 1))


def gradient_metrics(estimate, adjoint, free, inputs):
    """Local derivatives of directed off-diagonal W and hidden input matrix B.

    These parameters are hypothetical diagnostic mappings; this sweep performs
    no training. For F=-s-cubic*s^3+W@s+B@x, grad_W=-lambda outer s.
    """
    trainable = len(free) - 4
    grad_w = -np.outer(estimate, free)
    exact_w = -np.outer(adjoint, free)
    np.fill_diagonal(grad_w, 0)
    np.fill_diagonal(exact_w, 0)
    grad_b = -np.outer(estimate[:trainable], inputs)
    exact_b = -np.outer(adjoint[:trainable], inputs)
    return dict(recurrent_gradient_relative_error=relative_error(grad_w, exact_w),
                recurrent_gradient_cosine=cosine(grad_w, exact_w),
                input_gradient_relative_error=relative_error(grad_b, exact_b),
                input_gradient_cosine=cosine(grad_b, exact_b))


def summarize_samples(samples):
    result = {}
    for key in samples[0]:
        values = np.asarray([s[key] for s in samples], dtype=float)
        result[key + "_mean"] = float(np.mean(values))
        result[key + "_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
    return result


def method_specs(design, count, size):
    if design == "random_sign":
        return [("mc", 0.0), ("kaczmarz", 0.0), ("lstsq", 0.0),
                ("ridge", 0.1 * count / size)]
    return [("orthogonal", 0.0)] if count <= size else [("lstsq", 0.0)]


def compare_case(config, size, asymmetry, seed, progress):
    net = make_network(size, asymmetry, seed=seed, cubic=config["cubic"])
    rng = np.random.default_rng(72000 + size * 100 + seed)
    drive = rng.normal(scale=0.55, size=size)
    inputs = rng.uniform(-1, 1, size=5)
    free = relax(net, drive, tolerance=config["tolerance"])
    c = np.zeros(size)
    c[-4:] = free.state[-4:] - np.array([-0.8, 0.2, 0.6, -0.4])
    # Evaluation-only oracle: never passed into measurement or estimator calls.
    jacobian = net.jacobian(free.state)
    exact_response = np.linalg.solve(jacobian, np.eye(size))
    exact_q, adjoint = exact_response @ c, exact_response.T @ c
    hold_rng = np.random.default_rng(82000 + size * 100 + seed)
    heldout = make_probes(hold_rng, size, config["heldout_probes"], "random_sign")
    rows = []
    actual_equilibrations = 1
    actual_reads = 1
    # Share each random/orthogonal design across amplitudes and asymmetries.
    for design_index, design in enumerate(config["designs"]):
        probe_rng = np.random.default_rng(92000 + size * 100 + seed * 10 + design_index)
        probes = make_probes(probe_rng, size, max(config["budgets"]), design)
        for amplitude in config["amplitudes"]:
            measured = physical_measurements(net, drive, free.state, c, probes,
                                             heldout, amplitude, config["tolerance"])
            actual_equilibrations += measured["actual_equilibrations"]
            # Baseline/q/heldout equilibrium values reused computationally; each
            # noisy trial represents independent repeats of the two state reads.
            actual_reads += measured["actual_equilibrations"]
            for sigma_index, sigma in enumerate(config["read_noise"]):
                trials = config["noise_trials"] if sigma else 1
                noise_rng = np.random.default_rng(102000 + size * 100 + seed * 10 + sigma_index)
                # Same noise draw for q across designs and amplitudes. Probe
                # channels share standard-normal draws across designs as a CRN
                # comparison; within a design all states and signs are independent.
                q = add_read_noise(np.broadcast_to(measured["q_unit"], (trials, size)),
                                   sigma, amplitude, noise_rng) * measured["norm_c"]
                responses = add_read_noise(np.broadcast_to(measured["response"],
                                                           (trials,) + measured["response"].shape),
                                           sigma, amplitude, noise_rng)
                validation = add_read_noise(np.broadcast_to(measured["heldout"],
                                                           (trials,) + measured["heldout"].shape),
                                           sigma, amplitude, noise_rng)
                actual_reads += 2 * trials * (1 + len(probes) + len(heldout)) if sigma else 0
                base = dict(size=size, asymmetry=asymmetry, seed=seed, cubic=config["cubic"],
                            design=design, amplitude=amplitude, read_noise=sigma,
                            noise_trials=trials, oracle_eqprop_relative_error=relative_error(exact_q, adjoint),
                            q_central_relative_bias=relative_error(measured["q_unit"] * measured["norm_c"], exact_q),
                            probe_response_relative_bias=relative_error(measured["response"], probes @ exact_response.T),
                            max_force_residual=max(free.residual, measured["residual"]),
                            physical_batch_iterations=measured["batched_iterations"],
                            per_direction_l2_norm=1.0,
                            max_probe_node_current=float(amplitude * np.max(np.abs(probes))),
                            heldout_probe_count=len(heldout),
                            heldout_extra_equilibrations=2 * len(heldout),
                            heldout_extra_state_reads=2 * len(heldout),
                            heldout_extra_scalar_voltage_reads=2 * len(heldout) * size,
                            heldout_extra_squared_current_sum=2 * len(heldout) * amplitude**2)
                for count in config["budgets"]:
                    z = probes[:count]
                    statistics = design_statistics(z)
                    costs = dict(probes=count,
                                 equilibrations_per_estimate=1 + 2 * (count + 1),
                                 state_reads_per_estimate=1 + 2 * (count + 1),
                                 scalar_voltage_reads_per_estimate=size * (1 + 2 * (count + 1)),
                                 squared_current_sum=2 * (count + 1) * amplitude**2,
                                 # Iteration count is the batched slowest stopping
                                 # trajectory; wall time is a simulator diagnostic.
                                 nudge_pairs_per_estimate=count + 1)
                    response_samples = []
                    if statistics["identifiable_full_response"]:
                        for trial in range(trials):
                            fitted = fit_response(z, responses[trial, :count])
                            predicted = heldout @ fitted.T
                            response_samples.append(dict(
                                response_matrix_relative_error=relative_error(fitted, exact_response),
                                heldout_response_prediction_relative_error=relative_error(predicted, measured["heldout"]),
                                heldout_noisy_response_prediction_relative_error=relative_error(predicted, validation[trial])))
                    response_metrics = summarize_samples(response_samples) if response_samples else {}
                    for method, ridge in method_specs(design, count, size):
                        samples = []
                        for trial in range(trials):
                            y = responses[trial, :count] @ c
                            estimate = estimate_adjoint(q[trial], z, y, method, ridge=ridge)
                            clean_y = measured["heldout"] @ c
                            noisy_y = validation[trial] @ c
                            sample = dict(adjoint_relative_error=relative_error(estimate, adjoint),
                                          adjoint_squared_error=float(np.sum((estimate - adjoint)**2)),
                                          adjoint_relative_squared_error=relative_error(estimate, adjoint)**2,
                                          adjoint_cosine=cosine(estimate, adjoint),
                                          measured_eqprop_relative_error=relative_error(q[trial], adjoint),
                                          measured_eqprop_squared_error=float(np.sum((q[trial] - adjoint)**2)),
                                          adjoint_error_over_measured_eqprop=(np.linalg.norm(estimate - adjoint)
                                                / max(np.linalg.norm(q[trial] - adjoint), 1e-30)),
                                          heldout_adjoint_projection_relative_error=relative_error(heldout @ estimate, clean_y),
                                          heldout_noisy_adjoint_projection_relative_error=relative_error(heldout @ estimate, noisy_y))
                            sample.update(gradient_metrics(estimate, adjoint, free.state, inputs))
                            samples.append(sample)
                        rows.append(dict(base, **statistics, **costs, method=method,
                                         ridge=ridge, **summarize_samples(samples), **response_metrics))
            progress(f"n={size} alpha={asymmetry} seed={seed} {design} h={amplitude:g}")
    return rows, dict(clean_forward_equilibrations=actual_equilibrations,
                      modeled_state_reads=actual_reads)


def plot_results(rows, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    size = max(row["size"] for row in rows)
    amplitude = max(row["amplitude"] for row in rows)
    asymmetry = max(row["asymmetry"] for row in rows)
    styles = [("random_sign", "mc", "Random MC"),
              ("random_sign", "kaczmarz", "Random Kaczmarz"),
              ("random_sign", "lstsq", "Random LS"),
              ("random_sign", "ridge", "Random ridge"),
              ("coordinate", None, "Coordinate"),
              ("hadamard", None, "Hadamard"),
              ("orthogonal", None, "Random orthogonal")]
    for row_index, sigma in enumerate(sorted({row["read_noise"] for row in rows})):
        for style_index, (design, method, label) in enumerate(styles):
            subset = [r for r in rows if r["size"] == size and r["amplitude"] == amplitude
                      and r["asymmetry"] == asymmetry and r["read_noise"] == sigma
                      and r["design"] == design and (method is None or r["method"] == method)]
            budgets = sorted({r["probes"] for r in subset})
            for column, metric in enumerate(("adjoint_relative_error_mean",
                                             "input_gradient_relative_error_mean",
                                             "heldout_response_prediction_relative_error_mean")):
                # Response reconstruction is its own unregularized LS control,
                # independent of the chosen adjoint estimator. Plot it once.
                if column == 2 and design == "random_sign" and method != "lstsq":
                    continue
                available = [m for m in budgets if any(r["probes"] == m and metric in r for r in subset)]
                values = [np.mean([r[metric] for r in subset if r["probes"] == m and metric in r]) for m in available]
                axes[row_index, column].plot(available, values, "o-", label=label,
                                              color=f"C{style_index}", linewidth=1.3, markersize=3)
        relevant = [r for r in rows if r["size"] == size and r["amplitude"] == amplitude
                    and r["asymmetry"] == asymmetry and r["read_noise"] == sigma]
        axes[row_index, 0].axhline(np.mean([r["measured_eqprop_relative_error_mean"] for r in relevant]),
                                  color="gray", ls="--", label="Measured EqProp")
        for column, title in enumerate(("Adjoint error", "Hidden input-gradient error", "Held-out response prediction")):
            ax = axes[row_index, column]
            ax.set(xlabel="Probe pairs (one additional pair for q)", ylabel="Relative L2 error",
                   title=f"{title}; read σ={sigma:g}", yscale="log", xscale="log")
            ax.grid(alpha=0.2)
    axes[0, 0].legend(fontsize=8)
    axes[0, 2].legend(fontsize=8)
    figure.suptitle(f"Matched unit-L2 DC nudges: n={size:g}, asymmetry={asymmetry}, h={amplitude:g}\n"
                   "Means over independent networks; held-out response uses only full-rank designs")
    figure.savefig(output / "measurement_comparison.png", dpi=170)
    figure.savefig(output / "measurement_comparison.svg")
    plt.close(figure)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="Smoke grid: one 8-state problem, budgets 4/8/16")
    parser.add_argument("--output", type=Path, default=Path("simulation_results/adjoint_measurement_comparison"))
    args = parser.parse_args(argv)
    config = dict(tier="exploratory_noncanonical", sizes=[8] if args.quick else [16, 32],
                  asymmetries=[1.5] if args.quick else [0.0, 1.5],
                  seeds=[0] if args.quick else [0, 1, 2], cubic=0.25,
                  budgets=[4, 8, 16] if args.quick else [4, 8, 16, 32, 64],
                  amplitudes=[0.003, 0.03], read_noise=[0.0, 1e-4],
                  designs=["random_sign", "coordinate", "hadamard", "orthogonal"],
                  heldout_probes=16, noise_trials=2 if args.quick else 8,
                  ridge_rule="alpha = 0.1 * probe_count / state_count; not tuned using oracle",
                  tolerance=1e-12, command=shlex.join([sys.executable, "-m", "labs.tools.compare_adjoint_measurements", *sys.argv[1:]]),
                  python=sys.version, numpy=np.__version__, platform=platform.platform(),
                  source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                  source_status=subprocess.check_output(["git", "status", "--short"], text=True),
                  started_utc=datetime.now(timezone.utc).isoformat(), pid=os.getpid(),
                  interpretation="Local fixed-equilibrium comparison only; no training in this sweep.",
                  budget_convention="Each central pair uses two unit-L2 currents of amplitude h. Full state readout. Costs per estimator exclude 16 held-out diagnostic pairs.",
                  stochastic_repeats="Eight independent voltage read trials per noisy setting; clean equilibria reused. Trials are not extra networks.",
                  noise_scope="Independent additive voltage read noise on q, probe responses and held-out responses; exact free state, cost gradient and parameter-force factors. No process noise, drift, actuator error or noisy-state-gradient correction.",
                  control_reuse="Clean equilibria reused across noise levels, nested budgets, and estimators. Reported per-estimate costs model separate physical execution.")
    config["expected_problem_cases"] = len(config["sizes"]) * len(config["asymmetries"]) * len(config["seeds"])
    config["expected_rows"] = config["expected_problem_cases"] * len(config["amplitudes"]) * len(config["read_noise"]) * len(config["budgets"]) * 7
    output = args.output.resolve()
    if output.exists():
        raise ValueError(f"Expected a new output directory to protect earlier runs; got {output}")
    output.mkdir(parents=True)
    write_json(output / "config.json", config)
    started = time.monotonic()
    rows, completed_cases, totals = [], 0, dict(clean_forward_equilibrations=0, modeled_state_reads=0)
    log = (output / "run.log").open("w", buffering=1)

    def progress(message):
        entry = f"{datetime.now(timezone.utc).isoformat()} {message}"
        print(entry, flush=True)
        log.write(entry + "\n")
        write_json(output / "status.json", dict(state="running", pid=os.getpid(), completed_problem_cases=completed_cases,
                   expected_problem_cases=config["expected_problem_cases"], completed_rows=len(rows),
                   latest=message, elapsed_seconds=time.monotonic() - started))

    progress("starting; expected heartbeat after every amplitude/design batch, normally seconds")
    try:
        for size in config["sizes"]:
            for asymmetry in config["asymmetries"]:
                for seed in config["seeds"]:
                    case, cost = compare_case(config, size, asymmetry, seed, progress)
                    rows.extend(case)
                    completed_cases += 1
                    for key in totals:
                        totals[key] += cost[key]
                    write_csv(output / "measurements.csv", rows)
                    progress(f"completed problem {completed_cases}/{config['expected_problem_cases']}; rows={len(rows)}")
        if len(rows) != config["expected_rows"]:
            raise RuntimeError(f"Expected {config['expected_rows']} comparison rows; got {len(rows)}")
        residual = max(row["max_force_residual"] for row in rows)
        if residual > config["tolerance"]:
            raise RuntimeError(f"Expected force residual <= {config['tolerance']}; got {residual}")
        plot_results(rows, output)
        status = dict(state="complete", pid=os.getpid(), completed_problem_cases=completed_cases,
                      expected_problem_cases=config["expected_problem_cases"], completed_rows=len(rows),
                      expected_rows=config["expected_rows"], max_force_residual=residual,
                      elapsed_seconds=time.monotonic() - started, totals=totals,
                      completed_utc=datetime.now(timezone.utc).isoformat())
        write_json(output / "status.json", status)
        write_json(output / "completion.json", status)
        log.write(json.dumps(status) + "\n")
        print(json.dumps(status, indent=2), flush=True)
    except Exception as error:
        write_json(output / "status.json", dict(state="failed", pid=os.getpid(), error=repr(error),
                   completed_problem_cases=completed_cases, completed_rows=len(rows),
                   elapsed_seconds=time.monotonic() - started))
        log.write(f"FAILED {error!r}\n")
        raise
    finally:
        log.close()


if __name__ == "__main__":
    main()
