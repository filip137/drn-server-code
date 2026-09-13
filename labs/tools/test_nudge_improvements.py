#!/usr/bin/env python3
"""Validation screens and confirmations for measured-baseline EqProp corrections.

Run independent method shards in fresh directories. A selected configuration is
read from a screen's selection.json; screening never evaluates held-out test data.
"""

import argparse
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

from labs.tools.run_random_nudge_hopfield import write_csv, write_json
from labs.tools.train_recurrent_eqprop_digits import dataset, probe_specification, train_one


METHODS = ("mc8", "local_mc4", "local_mc8", "learned_mc4", "learned_mc8")
LEARNING_RATES = (0.03, 0.1, 0.3)
MOMENTA = (0.0, 0.9)


def select_configurations(rows, epochs, methods):
    """One choice per method, using the final validation epoch only."""
    selected = []
    for method in methods:
        candidates = [r for r in rows if r["method"] == method and r["epoch"] == epochs]
        expected = {(lr, mu) for lr in LEARNING_RATES for mu in MOMENTA}
        actual = {(r["learning_rate"], r["momentum"]) for r in candidates}
        if actual != expected or len(candidates) != len(expected):
            raise ValueError(f"Expected the complete six-setting screen for {method}; got {actual}")
        if any(any(key.startswith("test_") for key in r) for r in candidates):
            raise ValueError("Expected validation-only screening rows; got test metrics")
        best = max(candidates, key=lambda r: (r["validation_accuracy"], -r["validation_loss"]))
        selected.append({key: best[key] for key in
                         ("method", "learning_rate", "momentum", "validation_accuracy", "validation_loss")})
    return dict(criterion="final validation accuracy; tie: lower validation cross entropy",
                selection_seed=0, epochs=epochs, selections=selected)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("screen", "run"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--selection", type=Path)
    parser.add_argument("--learning-rate", type=float, default=0.3)
    parser.add_argument("--momentum", type=float, default=0.0)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--validation-only", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"Expected a new output directory; got existing {args.output}")
    if not np.isfinite(args.learning_rate) or args.learning_rate <= 0 or not 0 <= args.momentum < 1:
        parser.error("Expected finite learning rate > 0 and momentum in [0, 1)")
    if args.stage == "screen" and args.selection is not None:
        parser.error("Expected --selection only for stage run")
    epochs = args.epochs if args.epochs is not None else (8 if args.stage == "screen" else 15)
    if epochs < 1 or args.size <= 10:
        parser.error("Expected epochs >= 1 and size > 10")
    for method in args.methods:
        probe_specification(method, args.size)
    selection = json.loads(args.selection.read_text()) if args.selection else None
    cases = []
    for method in args.methods:
        if args.stage == "screen":
            settings = [(lr, mu) for lr in LEARNING_RATES for mu in MOMENTA]
        elif selection:
            matches = [r for r in selection["selections"] if r["method"] == method]
            if len(matches) != 1:
                parser.error(f"Expected exactly one selected configuration for {method}")
            settings = [(matches[0]["learning_rate"], matches[0]["momentum"])]
        else:
            settings = [(args.learning_rate, args.momentum)]
        for lr, mu in settings:
            for seed in ([0] if args.stage == "screen" else args.seeds):
                cases.append(dict(method=method, seed=seed, learning_rate=lr, momentum=mu))
    if len({tuple(c.values()) for c in cases}) != len(cases):
        parser.error("Expected unique methods and seeds")
    data = dataset(args.quick)
    config = dict(size=args.size, outputs=10, epochs=epochs, batch_size=96,
                  beta=0.01, read_noise=1e-5, asymmetry=1.0, audit=True,
                  predictor_relaxation=0.25, stage=args.stage, quick=args.quick,
                  momentum_rule="bias-corrected EMA; v=mu*v+(1-mu)*g",
                  validation_only=args.stage == "screen" or args.validation_only,
                  evidence_tier="exploratory, non-canonical", probe_norm="L2=1",
                  train_samples=len(data["train_y"]), validation_samples=len(data["val_y"]),
                  test_samples=len(data["test_y"]))
    args.output.mkdir(parents=True)
    root = Path(__file__).resolve().parents[2]
    run = dict(config=config, cases=cases, selection=selection, pid=os.getpid(),
               command=shlex.join([sys.executable, "-m", "labs.tools.test_nudge_improvements", *sys.argv[1:]]),
               commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
               source_status=subprocess.check_output(["git", "status", "--short"], cwd=root, text=True),
               started_utc=datetime.now(timezone.utc).isoformat(), python=sys.version,
               numpy=np.__version__, platform=platform.platform(),
               heartbeat="each epoch and trajectory; CPU foreground exec handle",
               recovery="preserve failed directory; diagnose and rerun unchanged config in a fresh directory")
    write_json(args.output / "run.json", run)
    started = time.time()
    state = dict(status="running", pid=os.getpid(), expected_trajectories=len(cases), completed_trajectories=0)

    def progress(detail, advance):
        state["completed_trajectories"] += int(advance)
        state.update(detail=detail, heartbeat_utc=datetime.now(timezone.utc).isoformat(),
                     elapsed_seconds=time.time()-started)
        write_json(args.output / "status.json", state)
        line = f"[{time.time()-started:.1f}s] {state['completed_trajectories']}/{len(cases)} {detail}"
        print(line, flush=True)
        with (args.output / "run.log").open("a") as stream:
            stream.write(line+"\n")

    progress("starting", False)
    rows = []
    try:
        for case in cases:
            name = "{method}_seed{seed}_lr{learning_rate:g}_mu{momentum:g}".format(**case)
            case_path = args.output / name
            case_path.mkdir()
            progress(f"starting {name}", False)
            case_rows = train_one(data, case["method"], case["seed"], dict(config, **case),
                                  case_path, progress, calibration=config["validation_only"])
            for row in case_rows:
                row["case"] = name
                row["checkpoint"] = str(case_path / f"{case['method']}_seed{case['seed']}_lr{case['learning_rate']:g}.npz")
            write_csv(case_path / "metrics.csv", case_rows)
            rows.extend(case_rows)
            write_csv(args.output / "metrics.csv", rows)
        assert len(rows) == len(cases)*(epochs+1)
        assert state["completed_trajectories"] == len(cases)
        if args.stage == "screen":
            write_json(args.output / "selection.json", select_configurations(rows, epochs, args.methods))
        final = [r for r in rows if r["epoch"] == epochs]
        write_json(args.output / "summary.json", dict(final_rows=final, config=config))
        state.update(status="complete", terminal_result="summary.json", elapsed_seconds=time.time()-started)
        write_json(args.output / "status.json", state)
        print(f"Complete: {args.output}", flush=True)
    except BaseException as error:
        state.update(status="failed", error=f"{type(error).__name__}: {error}", elapsed_seconds=time.time()-started)
        write_json(args.output / "status.json", state)
        raise


if __name__ == "__main__":
    main()
