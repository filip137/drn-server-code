#!/usr/bin/env python3
"""Train EqProp using a frozen controller from completed paired-DC calibration.

Calibration is reused without modification. The physical model, data split,
optimizer, and 15-epoch default match the structured-circulation controls.
No fresh random adjoint probes or process-noise calibration are used here.
"""

import argparse
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np

from labs.tools.run_random_nudge_hopfield import write_csv, write_json
from labs.tools.test_structured_circulation import make_loop_case
from labs.tools.train_recurrent_eqprop_digits import dataset, train_one


METHOD = "response_calibrated_asymep"
IMPLEMENTATION_METHOD = "circulation_asymep"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_calibration(source, sizes, seeds):
    """Check declared source coverage and requested wiring/controller artifacts."""
    source = source.resolve()
    run = json.loads((source/"run.json").read_text())
    status = json.loads((source/"status.json").read_text())
    summaries = json.loads((source/"summary.json").read_text())
    config = run["config"]
    expected = {(int(n), int(s)) for n in config["sizes"] for s in config["seeds"]}
    cases = {(int(c["size"]), int(c["seed"])): c for c in summaries}
    require(status["status"] == "complete" and status["completed"] == run["expected_cases"] == len(expected)
            and len(summaries) == len(cases) == len(expected) and set(cases) == expected,
            f"Expected complete unique declared calibration coverage; got {source}: {status}")
    sizes = list(config["sizes"]) if sizes is None else sizes
    seeds = list(config["seeds"]) if seeds is None else seeds
    requested = {(n, s) for n in sizes for s in seeds}
    require(len(requested) == len(sizes)*len(seeds) and requested <= expected,
            f"Expected distinct requested sizes/seeds present in calibration source; got {sizes}, {seeds}")
    loaded = []
    loops, steps = int(config["loops"]), int(config["steps"])
    require(loops > 0 and steps > 0, f"Expected positive source loop/step counts; got {loops}, {steps}")
    for size in sizes:
        for seed in seeds:
            directory = source/f"n{size}_seed{seed}"
            summary = json.loads((directory/"summary.json").read_text())
            require(summary == cases[size, seed] and summary["loops"] == loops,
                    f"Expected matching case/run calibration summaries; got {directory}")
            counts = summary["calibration_counts"]
            require(counts["equilibrations"] == counts["scalar_reads"] == 4*loops*steps
                    and summary["free_anchor_equilibrations"] == 1
                    and summary["anchor_projection_reads"] == 2*loops
                    and summary["process_noise_channels"] == 0
                    and not summary["task_training_performed"],
                    f"Expected paired-DC calibration counts and a single free anchor; got {directory}")
            require(np.isclose(counts["total_excitation_sq"], 4*loops*steps*config["amplitude"]**2),
                    f"Expected source excitation accounting; got {directory}")
            model, left, right, _ = make_loop_case(seed, size, loops)
            with np.load(directory/"controller.npz") as saved:
                controller, coefficients = saved["controller"].copy(), saved["coefficients"].copy()
                require(controller.shape == (size, size) and coefficients.shape == (loops,)
                        and np.isfinite(controller).all() and np.isfinite(coefficients).all(),
                        f"Expected finite controller and loop-gain arrays; got {directory}")
                require(np.array_equal(saved["left"], left) and np.array_equal(saved["right"], right),
                        f"Expected identical independently generated physical wiring; got {directory}")
                forward = (left*coefficients) @ right.T
                require(np.allclose(controller, (forward-forward.T)/np.sqrt(2), atol=1e-13, rtol=1e-13),
                        f"Expected controller assembled from saved gains and wiring; got {directory}")
            measured_error = np.linalg.norm(controller+model.skew)/np.linalg.norm(model.skew)
            require(np.isclose(measured_error, summary["controller_relative_error"], atol=1e-12, rtol=1e-12),
                    f"Expected calibration audit consistent with the training model; got {directory}")
            loaded.append(dict(size=size, seed=seed, model=model, controller=controller,
                               summary=summary, source_controller=str(directory/"controller.npz"),
                               source_summary=str(directory/"summary.json")))
    return run, status, sizes, seeds, loaded


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sizes", type=int, nargs="+")
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"Expected a fresh output directory; got {args.output}")
    if args.epochs < 1:
        parser.error(f"Expected positive training epochs; got {args.epochs}")
    source_run, source_status, sizes, seeds, cases = read_calibration(args.calibration_source, args.sizes, args.seeds)
    config = dict(calibration_source=str(args.calibration_source.resolve()), sizes=sizes, seeds=seeds,
                  epochs=args.epochs, quick=args.quick, methods=[METHOD], loops=source_run["config"]["loops"],
                  learning_rate=.1, momentum=.9, read_noise=1e-5, beta=.01, batch_size=96,
                  source_calibration_uses_full_training_split=True,
                  smoke_training_uses_reduced_split=args.quick)
    output = args.output.resolve()
    output.mkdir(parents=True)
    started = time.time()
    run = dict(config=config, command=[sys.executable, "-m", "labs.tools.train_response_loop_feedback", *sys.argv[1:]],
               expected_training_trajectories=len(cases), implementation_method=IMPLEMENTATION_METHOD,
               reused_calibration_run=source_run, reused_calibration_status=source_status,
               evidence_tier="exploratory training with previously measured paired-DC loop feedback",
               pid=os.getpid(), python=sys.version, numpy=np.__version__, platform=platform.platform(),
               commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
               source_status=subprocess.check_output(["git", "status", "--short"], text=True),
               calibration_paths=[dict(size=c["size"], seed=c["seed"], controller=c["source_controller"],
                                       summary=c["source_summary"]) for c in cases])
    write_json(output/"run.json", run)
    write_json(output/"config.json", config)
    status = dict(status="running", completed_training=0, expected_training=len(cases), pid=os.getpid())

    def progress(message, advance=False):
        status["completed_training"] += int(advance)
        status.update(detail=message.replace(IMPLEMENTATION_METHOD, METHOD), elapsed_seconds=time.time()-started)
        write_json(output/"status.json", status)
        line = f"[{status['elapsed_seconds']:.1f}s] {status['completed_training']}/{len(cases)} {status['detail']}"
        print(line, flush=True)
        with (output/"run.log").open("a") as stream:
            stream.write(line+"\n")

    metrics, completed = [], []
    try:
        data = dataset(args.quick)
        progress("validated frozen calibration sources; training data ready")
        for case in cases:
            size, seed, summary = case["size"], case["seed"], case["summary"]
            directory = output/f"n{size}_seed{seed}"
            directory.mkdir()
            write_json(directory/"calibration_source.json", dict(source_controller=case["source_controller"],
                       source_summary=case["source_summary"], source_config=source_run["config"], summary=summary))
            train_config = dict(config, size=size, outputs=10, asymmetry=1., audit=True)
            rows = train_one(data, IMPLEMENTATION_METHOD, seed, train_config, directory, progress,
                             model_override=case["model"], feedback_controller=case["controller"])
            original = directory/f"{IMPLEMENTATION_METHOD}_seed{seed}_lr0.1.npz"
            checkpoint = directory/f"{METHOD}_seed{seed}_lr0.1.npz"
            original.rename(checkpoint)
            for row in rows:
                row.update(method=METHOD, implementation_method=IMPLEMENTATION_METHOD, size=size,
                           loops=config["loops"], checkpoint=str(checkpoint),
                           calibration_perturbed_equilibrations=summary["calibration_counts"]["equilibrations"],
                           calibration_anchor_equilibrations=summary["free_anchor_equilibrations"],
                           calibration_scalar_projection_reads=summary["calibration_counts"]["scalar_reads"],
                           calibration_anchor_projection_reads=summary["anchor_projection_reads"],
                           training_random_adjoint_probe_pairs=0,
                           calibration_process_noise_channels=0,
                           learned_controller_coefficients=summary["learned_coefficients"],
                           known_wiring_coefficients=summary["known_wiring_coefficients"])
                require(row["equilibrations"] == row["state_reads"] == 3*len(data["train_y"])*row["epoch"],
                        f"Expected three training equilibrations per example; got {row}")
                require(row["total_probe_excitation_sq"] == 0, f"Expected no training adjoint probes; got {row}")
            write_csv(directory/"metrics.csv", rows)
            write_csv(directory/"current.csv", rows)
            metrics.extend(rows)
            completed.append(dict(size=size, seed=seed, checkpoint=str(checkpoint),
                                  calibration_summary=summary, final_training=rows[-1],
                                  controller_frozen_after_calibration=True))
            write_csv(output/"metrics.csv", metrics)
            write_json(output/"summary.json", dict(cases=completed, training_final=[c["final_training"] for c in completed],
                       new_calibration_performed=False,
                       calibration_cost="Charge source perturbed equilibrations and its free anchor once per trained model, separately from training phases."))
        require(status["completed_training"] == len(cases), "Expected complete declared training coverage")
        status["status"] = "complete"
        progress("all frozen response-calibrated training trajectories complete")
    except BaseException as error:
        status.update(status="failed", error=f"{type(error).__name__}: {error}")
        progress(status["error"])
        raise


if __name__ == "__main__":
    main()
