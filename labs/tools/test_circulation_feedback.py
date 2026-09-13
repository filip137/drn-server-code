#!/usr/bin/env python3
"""Exploratory calibration and training with a circulation-learned controller."""

import argparse
import copy
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np

from labs.circulation_feedback import calibrate_controller
from labs.recurrent_eqprop import flatten_gradient, make_classifier, settle
from labs.tools.run_random_nudge_hopfield import cosine, write_csv, write_json
from labs.tools.train_recurrent_eqprop_digits import dataset, gradient_step, train_one


def audit_controller(model, data, controller):
    """Exact gradients are read-only evaluation targets, never calibration data."""
    rows = []
    x, y = data["val_x"][:96], data["val_y"][:96]
    for name, method, feedback in (
        ("ordinary_ep", "contrastive_ep", None),
        ("known_skew", "known_skew_asymep", None),
        ("calibrated_double_gain", "circulation_asymep", controller),
        ("calibrated_single_gain", "circulation_asymep", controller/2),
    ):
        sink = {}
        gradient, meta = gradient_step(model, x, y, method, np.random.default_rng(74000),
                                       beta=0.01, sigma=0.0, audit=True, audit_sink=sink,
                                       feedback_controller=feedback)
        meta["method"] = name
        for label, g, ref in zip(("recurrent", "input", "bias"), gradient, sink["reference_gradient"]):
            meta[label+"_gradient_cosine"] = cosine(g.ravel(), ref.ravel())
            meta[label+"_gradient_relative_error"] = float(np.linalg.norm(g-ref)/max(np.linalg.norm(ref), 1e-20))
        rows.append(meta)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sizes", type=int, nargs="+", default=[32])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--duration", type=float, default=200)
    parser.add_argument("--dt", type=float, default=0.02)
    parser.add_argument("--temperature", type=float, default=0.05)
    parser.add_argument("--replicas", type=int, default=32)
    parser.add_argument("--calibration-rate", type=float, default=0.1)
    parser.add_argument("--update-interval", type=int, default=10)
    parser.add_argument("--burn-in", type=float, default=10)
    parser.add_argument("--average-after", type=float, default=50)
    parser.add_argument("--calibration-read-noise", type=float, default=0)
    parser.add_argument("--epochs", type=int, default=0)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--read-noise", type=float, default=1e-5)
    parser.add_argument("--asymmetry", type=float, default=1.0)
    parser.add_argument("--methods", nargs="+", default=["contrastive_ep", "known_skew_asymep", "circulation_asymep"])
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"Expected a new output directory; got {args.output}")
    if any(n <= 10 for n in args.sizes) or args.epochs < 0:
        parser.error(f"Expected sizes > 10 and epochs >= 0; got {args.sizes}, {args.epochs}")
    supported = {"adjoint", "contrastive_ep", "known_skew_asymep", "circulation_asymep", "learned_mc4"}
    if set(args.methods)-supported:
        parser.error(f"Expected methods in {sorted(supported)}; got {args.methods}")
    output = args.output
    output.mkdir(parents=True)
    config = vars(args).copy()
    config["output"] = str(output)
    root = Path(__file__).resolve().parents[2]
    started = time.time()
    run = dict(config=config, evidence_tier="exploratory, non-canonical", pid=os.getpid(),
               command=[sys.executable, "-m", "labs.tools.test_circulation_feedback", *sys.argv[1:]],
               commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
               source_status=subprocess.check_output(["git", "status", "--short"], cwd=root, text=True),
               python=sys.version, numpy=np.__version__, platform=platform.platform(),
               calibration_measurement="physical noisy state trajectories, no Jacobian or known skew given to plasticity",
               stochastic_integrator="Heun; not noisy coordinate descent",
               expected_calibrations=len(args.sizes)*len(args.seeds),
               expected_training_trajectories=len(args.sizes)*len(args.seeds)*len(args.methods) if args.epochs else 0)
    write_json(output/"run.json", run)
    status = dict(status="running", completed_calibrations=0, completed_training=0, pid=os.getpid())

    def progress(message, advance=False):
        status["completed_training"] += int(advance)
        status.update(detail=message, elapsed_seconds=time.time()-started)
        write_json(output/"status.json", status)
        line = f"[{time.time()-started:.1f}s] {message}"
        print(line, flush=True)
        with (output/"run.log").open("a") as stream:
            stream.write(line+"\n")

    data = dataset(args.quick)
    summaries, metrics = [], []
    try:
        progress("dataset ready; calibration and all selected training cases pending")
        for size in args.sizes:
            for seed in args.seeds:
                case = output/f"n{size}_seed{seed}"
                case.mkdir()
                model = make_classifier(seed, size=size, asymmetry=args.asymmetry)
                network = model.network()
                drive = model.drive(data["train_x"][:1])[0]
                free = settle(network, drive).state
                history = []

                def calibration_progress(info, current, averaged):
                    # Known skew is used ONLY for this read-only audit. Neither
                    # these values nor a stopping decision return to plasticity.
                    denominator = max(np.linalg.norm(model.skew), 1e-20)
                    info["current_controller_relative_error"] = float(np.linalg.norm(current+model.skew)/denominator)
                    info["averaged_controller_relative_error"] = float(np.linalg.norm(averaged+model.skew)/denominator)
                    history.append(info)
                    write_csv(case/"calibration.csv", history)
                    progress(f"n={size} seed={seed} calibration t={info['adaptive_time']:.1f}/{args.duration:g} "
                             f"controller_error={info['averaged_controller_relative_error']:.3f}")

                result = calibrate_controller(
                    lambda states: network.force(states, drive), free,
                    duration=args.duration, dt=args.dt, temperature=args.temperature,
                    replicas=args.replicas, learning_rate=args.calibration_rate,
                    update_interval=args.update_interval, burn_in=args.burn_in,
                    average_after=args.average_after, seed=64000+seed,
                    read_noise=args.calibration_read_noise, callback=calibration_progress)
                np.savez_compressed(case/"controller.npz", controller=result.matrix,
                                    final_controller=result.final_matrix, calibration_center=free)
                audit = audit_controller(model, data, result.matrix)
                # Input transfer: validation differs from the single training
                # input used for calibration. A changed symmetric model tests
                # transfer through later changes without recalibrating C.
                changed = copy.deepcopy(model)
                changed.symmetric *= 0.4
                changed.inputs *= 1.8
                changed.bias += np.linspace(-0.1, 0.1, size)
                transferred = audit_controller(changed, data, result.matrix)
                summary = dict(size=size, seed=seed, calibration=result.stats,
                               controller_relative_error=float(np.linalg.norm(result.matrix+model.skew)/max(np.linalg.norm(model.skew), 1e-20)),
                               initial_validation_gradient_audit=audit,
                               changed_symmetric_model_gradient_audit=transferred,
                               controller_frozen_after_calibration=True,
                               reference_skew_used_for_calibration=False)
                write_json(case/"calibration_summary.json", summary)
                status["completed_calibrations"] += 1
                summaries.append(summary)
                progress(f"n={size} seed={seed} calibration complete; error={summary['controller_relative_error']:.3f}")
                if args.epochs:
                    train_config = dict(size=size, outputs=10, asymmetry=args.asymmetry,
                                        epochs=args.epochs, batch_size=96, beta=.01,
                                        read_noise=args.read_noise, audit=True,
                                        learning_rate=args.learning_rate, momentum=args.momentum)
                    for method in args.methods:
                        rows = train_one(data, method, seed, train_config, case, progress,
                                         feedback_controller=result.matrix if method == "circulation_asymep" else None)
                        for row in rows:
                            row["size"] = size
                            row["calibration_total_record_time"] = result.stats["total_record_time"] if method == "circulation_asymep" else 0.0
                            row["calibration_state_reads"] = result.stats["state_reads"] if method == "circulation_asymep" else 0
                        metrics.extend(rows)
                        write_csv(output/"metrics.csv", metrics)
                write_json(output/"summary.json", dict(cases=summaries, training_final=[r for r in metrics if r["epoch"] == args.epochs]))
        assert status["completed_calibrations"] == run["expected_calibrations"]
        assert status["completed_training"] == run["expected_training_trajectories"]
        status["status"] = "complete"
        progress("all calibration, gradient audit, and selected training cases complete")
    except BaseException as error:
        status.update(status="failed", error=f"{type(error).__name__}: {error}")
        progress(status["error"])
        raise


if __name__ == "__main__":
    main()
