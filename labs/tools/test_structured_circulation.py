#!/usr/bin/env python3
"""Explore few-loop physical calibration with independent known wiring/gains.

The unknown physical skew is restricted to a known loop basis. Calibration gets
only that wiring, an opaque physical force, and measured trajectories; reference
gains and dense gradients are used exclusively in read-only diagnostics. The
optional dense comparison is additional study work with separately charged cost.
"""

import argparse
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np

from labs.circulation_feedback import calibrate_controller
from labs.loop_circulation_feedback import calibrate_loop_controller
from labs.recurrent_eqprop import make_classifier, settle
from labs.tools.run_random_nudge_hopfield import write_csv, write_json
from labs.tools.test_circulation_feedback import audit_controller
from labs.tools.train_recurrent_eqprop_digits import dataset, train_one


def make_loop_case(seed, size, loops, *, outputs=10, input_size=64):
    """Known orthonormal wiring and independently sampled unknown physical gains."""
    if not 1 <= loops <= min(outputs, size-outputs):
        raise ValueError(f"Expected 1 <= loops <= min(outputs, hidden); got {loops}, {outputs}, {size-outputs}")
    wiring_rng = np.random.default_rng(900000+seed)
    hidden = size-outputs
    q, _ = np.linalg.qr(wiring_rng.normal(size=(hidden, loops)), mode="reduced")
    left, right = np.zeros((size, loops)), np.zeros((size, loops))
    left[:hidden] = q
    right[hidden+np.arange(loops), np.arange(loops)] = 1.0
    raw_gains = np.random.default_rng(910000+seed).uniform(0.7, 1.3, loops)
    product = (left*raw_gains) @ right.T
    raw_skew = (product-product.T)/np.sqrt(2)
    normalization = 1/np.linalg.norm(raw_skew, 2)
    gains = raw_gains*normalization
    model = make_classifier(seed, size=size, outputs=outputs, input_size=input_size)
    model.skew = raw_skew*normalization
    reference = dict(raw_gains=raw_gains, normalized_gains=gains,
                     skew_normalization=normalization, skew=model.skew)
    return model, left, right, reference


def recording_cost(args, size, *, structured):
    """Count every record and observed scalar; parallel replicas are not free."""
    steps = int(np.ceil(args.burn_in/args.dt))+int(np.ceil(args.duration/args.dt))
    channels = 2*args.loops if structured else size
    snapshots = args.replicas*(steps+1)
    return dict(projection_readout_channels=channels if structured else 0,
                observed_scalar_channels=channels,
                scalar_measurement_reads=channels*snapshots,
                measurement_snapshots=snapshots,
                process_noise_channels=size,
                process_noise_scalar_increments=args.replicas*steps*size,
                total_recording_time=args.replicas*steps*args.dt,
                learned_controller_coefficients=args.loops if structured else size*(size-1)//2,
                known_wiring_coefficients=2*size*args.loops if structured else 0,
                random_adjoint_probe_pairs=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sizes", type=int, nargs="+", default=[64])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--loops", type=int, default=4)
    parser.add_argument("--duration", type=float, default=100)
    parser.add_argument("--dt", type=float, default=0.02)
    parser.add_argument("--temperature", type=float, default=0.05)
    parser.add_argument("--replicas", type=int, default=8)
    parser.add_argument("--calibration-rate", type=float, default=0.1)
    parser.add_argument("--update-interval", type=int, default=10)
    parser.add_argument("--burn-in", type=float, default=10)
    parser.add_argument("--average-after", type=float, default=30)
    parser.add_argument("--calibration-read-noise", type=float, default=0)
    parser.add_argument("--epochs", type=int, default=0)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--dense-control", action="store_true")
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--read-noise", type=float, default=1e-5)
    supported = ["contrastive_ep", "known_skew_asymep", "circulation_asymep", "learned_mc4", "adjoint"]
    parser.add_argument("--methods", choices=supported, nargs="+", default=supported[:-1])
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"Expected a fresh output directory; got {args.output}")
    if args.loops < 1 or args.loops > 10 or any(n-10 < args.loops for n in args.sizes):
        parser.error(f"Expected 1 <= loops <= 10 and sizes >= 10+loops; got {args.loops}, {args.sizes}")
    if args.epochs < 0 or args.replicas < 1 or args.update_interval < 1:
        parser.error(f"Expected nonnegative epochs and positive replicas/update_interval; got {args.epochs}, {args.replicas}, {args.update_interval}")
    for name in ("duration", "dt", "temperature", "calibration_rate", "learning_rate"):
        value = getattr(args, name)
        if not np.isfinite(value) or value <= 0:
            parser.error(f"Expected finite positive {name}; got {value}")
    for name in ("burn_in", "calibration_read_noise", "read_noise"):
        value = getattr(args, name)
        if not np.isfinite(value) or value < 0:
            parser.error(f"Expected finite nonnegative {name}; got {value}")
    if not 0 <= args.average_after < args.duration or not 0 <= args.momentum < 1:
        parser.error(f"Expected 0 <= average_after < duration and 0 <= momentum < 1; got {args.average_after}, {args.duration}, {args.momentum}")
    if any(len(values) != len(set(values)) for values in (args.sizes, args.seeds, args.methods)) or min(args.seeds) < 0:
        parser.error(f"Expected distinct sizes/methods and distinct nonnegative seeds; got {args.sizes}, {args.methods}, {args.seeds}")
    output = args.output
    output.mkdir(parents=True)
    config = vars(args).copy()
    config["output"] = str(output)
    root, started = Path(__file__).resolve().parents[2], time.time()
    case_count = len(args.sizes)*len(args.seeds)
    run = dict(config=config, evidence_tier="exploratory, known low-dimensional skew wiring",
               pid=os.getpid(), command=[sys.executable, "-m", "labs.tools.test_structured_circulation", *sys.argv[1:]],
               commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
               source_status=subprocess.check_output(["git", "status", "--short"], cwd=root, text=True),
               python=sys.version, numpy=np.__version__, platform=platform.platform(),
               expected_structured_calibrations=case_count,
               expected_dense_calibrations=case_count*int(args.dense_control),
               expected_training_trajectories=case_count*len(args.methods) if args.epochs else 0,
               plasticity_access="known projected wiring and opaque force; no gains/skew/Jacobian/adjoint",
               physical_skew="independent 0.7--1.3 loop gains, then fixed global normalization to operator norm one",
               dense_comparison="optional extra study cost; not part of structured learner's required calibration",
               measurement_budget="all replicas and their observation time charged; known wiring also reported",
               stochastic_integrator="physical additive-noise Heun, distinct from equilibrium solver")
    write_json(output/"run.json", run)
    write_json(output/"config.json", config)
    status = dict(status="running", completed_structured_calibrations=0,
                  completed_dense_calibrations=0, completed_training=0, pid=os.getpid())
    summaries, metrics = [], []

    def progress(message, advance=False):
        status["completed_training"] += int(advance)
        status.update(detail=message, elapsed_seconds=time.time()-started)
        write_json(output/"status.json", status)
        line = f"[{time.time()-started:.1f}s] {message}"
        print(line, flush=True)
        with (output/"run.log").open("a") as stream:
            stream.write(line+"\n")

    try:
        data = dataset(args.quick)
        progress("dataset ready; independent loop wiring and unknown gains pending")
        for size in args.sizes:
            for seed in args.seeds:
                case = output/f"n{size}_seed{seed}"
                case.mkdir()
                model, left, right, reference = make_loop_case(seed, size, args.loops,
                                                              input_size=data["train_x"].shape[1])
                np.savez_compressed(case/"wiring.npz", left=left, right=right)
                np.savez_compressed(case/"reference.npz", **reference, symmetric=model.symmetric,
                                    inputs=model.inputs, bias=model.bias)
                network = model.network()
                drive = model.drive(data["train_x"][:1])[0]
                free = settle(network, drive).state
                gain_norm = np.linalg.norm(reference["normalized_gains"])
                skew_norm = np.linalg.norm(model.skew)
                structured_history = []

                def structured_progress(info, current, averaged):
                    # Evaluation only: values do not affect updates or duration.
                    row = dict(info)
                    row["current_coefficient_relative_error"] = float(np.linalg.norm(current+reference["normalized_gains"])/gain_norm)
                    row["averaged_coefficient_relative_error"] = float(np.linalg.norm(averaged+reference["normalized_gains"])/gain_norm)
                    structured_history.append(row)
                    write_csv(case/"structured_calibration.csv", structured_history)
                    progress(f"n={size} seed={seed} structured t={info['adaptive_time']:.1f}/{args.duration:g} "
                             f"gain_error={row['averaged_coefficient_relative_error']:.3f}")

                kwargs = dict(duration=args.duration, dt=args.dt, temperature=args.temperature,
                              replicas=args.replicas, learning_rate=args.calibration_rate,
                              update_interval=args.update_interval, burn_in=args.burn_in,
                              average_after=args.average_after, seed=64000+seed,
                              read_noise=args.calibration_read_noise)
                # The learning API receives known wiring, never reference gains.
                result = calibrate_loop_controller(lambda states: network.force(states, drive),
                                                    free, left, right, callback=structured_progress, **kwargs)
                np.savez_compressed(case/"controller.npz", controller=result.matrix,
                                    coefficients=result.coefficients, final_coefficients=result.final_coefficients,
                                    calibration_center=free)
                structured_cost = recording_cost(args, size, structured=True)
                summary = dict(size=size, seed=seed, loops=args.loops,
                               controller_relative_error=float(np.linalg.norm(result.matrix+model.skew)/skew_norm),
                               calibration=result.stats, measurement_cost=structured_cost,
                               gradient_audit=audit_controller(model, data, result.matrix),
                               controller_frozen_after_calibration=True,
                               known_gains_used_for_calibration=False,
                               known_basis_is_an_exact_structural_prior=True,
                               training_uses_identical_initial_physical_model_for_all_methods=True)
                status["completed_structured_calibrations"] += 1
                write_json(case/"calibration_summary.json", summary)
                if args.dense_control:
                    dense_history = []

                    def dense_progress(info, current, averaged):
                        row = dict(info)
                        row["current_controller_relative_error"] = float(np.linalg.norm(current+model.skew)/skew_norm)
                        row["averaged_controller_relative_error"] = float(np.linalg.norm(averaged+model.skew)/skew_norm)
                        dense_history.append(row)
                        write_csv(case/"dense_calibration.csv", dense_history)
                        progress(f"n={size} seed={seed} dense comparison t={info['adaptive_time']:.1f}/{args.duration:g} "
                                 f"controller_error={row['averaged_controller_relative_error']:.3f}")

                    dense = calibrate_controller(lambda states: network.force(states, drive), free,
                                                  callback=dense_progress, **kwargs)
                    np.savez_compressed(case/"dense_controller.npz", controller=dense.matrix,
                                        final_controller=dense.final_matrix, calibration_center=free)
                    summary["dense_comparison"] = dict(
                        calibration=dense.stats, extra_study_cost=recording_cost(args, size, structured=False),
                        controller_relative_error=float(np.linalg.norm(dense.matrix+model.skew)/skew_norm),
                        gradient_audit=audit_controller(model, data, dense.matrix),
                        matched_duration_replicas_physical_model_and_noise_seed=True,
                        used_for_training=False)
                    status["completed_dense_calibrations"] += 1
                    write_json(case/"calibration_summary.json", summary)
                summaries.append(summary)
                progress(f"n={size} seed={seed} all requested calibration/audits complete")
                if args.epochs:
                    train_config = dict(size=size, outputs=10, asymmetry=1.0, epochs=args.epochs,
                                        batch_size=96, beta=.01, read_noise=args.read_noise, audit=True,
                                        learning_rate=args.learning_rate, momentum=args.momentum)
                    for method in args.methods:
                        training_output = case/"training"/method
                        training_output.mkdir(parents=True)
                        rows = train_one(data, method, seed, train_config, training_output, progress,
                                         model_override=model,
                                         feedback_controller=result.matrix if method == "circulation_asymep" else None)
                        calibrated = method == "circulation_asymep"
                        for row in rows:
                            row.update(size=size, loops=args.loops,
                                       calibration_total_record_time=structured_cost["total_recording_time"] if calibrated else 0.0,
                                       calibration_scalar_measurement_reads=structured_cost["scalar_measurement_reads"] if calibrated else 0,
                                       calibration_projection_readout_channels=2*args.loops if calibrated else 0,
                                       learned_controller_coefficients=args.loops if calibrated else 0)
                        metrics.extend(rows)
                        write_csv(output/"metrics.csv", metrics)
                write_json(output/"summary.json", dict(
                    cases=summaries, training_final=[r for r in metrics if r["epoch"] == args.epochs],
                    total_calibration_recording_time=sum(
                        item["measurement_cost"]["total_recording_time"]
                        + item.get("dense_comparison", {}).get("extra_study_cost", {}).get("total_recording_time", 0)
                        for item in summaries)))
        for actual, expected in (("completed_structured_calibrations", "expected_structured_calibrations"),
                                 ("completed_dense_calibrations", "expected_dense_calibrations"),
                                 ("completed_training", "expected_training_trajectories")):
            assert status[actual] == run[expected], (actual, status[actual], run[expected])
        status["status"] = "complete"
        progress("all structured calibration, comparisons, and selected training trajectories complete")
    except BaseException as error:
        status.update(status="failed", error=f"{type(error).__name__}: {error}")
        progress(status["error"])
        raise


if __name__ == "__main__":
    main()
