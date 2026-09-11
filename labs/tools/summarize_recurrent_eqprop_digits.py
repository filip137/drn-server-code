#!/usr/bin/env python3
"""Validate/replay completed digits comparisons and merge their reports.

Primary runs and exploratory follow-ups retain explicit group labels. This
command never resumes or changes training, and never overwrites a result.
"""

import argparse
import csv
import json
from pathlib import Path
import re

import numpy as np

from labs.recurrent_eqprop import Classifier
from labs.tools.train_recurrent_eqprop_digits import evaluate, report
from labs.tools.run_random_nudge_hopfield import write_csv, write_json


def read_metrics(path):
    rows = []
    for source in csv.DictReader(path.open()):
        row = {}
        for key, value in source.items():
            if value == "":
                continue
            try:
                number = float(value)
            except ValueError:
                row[key] = value
            else:
                if not np.isfinite(number):
                    raise ValueError(f"Expected finite numeric metrics; got {key}={value}")
                row[key] = number
        rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--follow-up-runs", type=Path, nargs="*", default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"Expected a new output directory; exists: {args.output}")
    rows, sources, checks = [], [], []
    shared_config = None
    initial_by_seed = {}
    comparable = ("epochs", "learning_rate", "batch_size", "beta", "read_noise", "asymmetry",
                  "size", "hidden", "outputs", "train_samples", "validation_samples", "test_samples")
    for group, paths in (("primary",args.runs),("exploratory_followup",args.follow_up_runs)):
        for path in paths:
            status = json.loads((path/"status.json").read_text())
            run = json.loads((path/"run.json").read_text())
            config = run["config"]
            if status["status"] != "complete" or status["completed_trajectories"] != status["expected_trajectories"]:
                raise ValueError(f"Expected complete declared coverage; got {path}: {status}")
            if shared_config is None:
                shared_config = config
            for key in comparable:
                if config[key] != shared_config[key]:
                    raise ValueError(f"Expected matched {key}; got {config[key]} vs {shared_config[key]} at {path}")
            measured = read_metrics(path/"metrics.csv")
            if len(measured) != status["expected_trajectories"]*(config["epochs"]+1):
                raise ValueError(f"Expected one row per epoch and trajectory; got {len(measured)} at {path}")
            for row in measured:
                if row["max_force_residual"] > 1e-9:
                    raise ValueError(f"Expected force residual <= 1e-9; got {row['max_force_residual']}")
                row.update(comparison_group=group,source_run=str(path.resolve()))
                if row["epoch"] != config["epochs"]:
                    if "test_accuracy" in row or "test_loss" in row:
                        raise ValueError(f"Expected test evaluation only at terminal epoch; got {row}")
                    continue
                method,seed = row["method"],int(row["seed"])
                checkpoint = path/f"{method}_seed{seed}_lr{config['learning_rate']:g}.npz"
                with np.load(checkpoint) as values:
                    model = Classifier(values["symmetric"],values["skew"],values["inputs"],values["bias"],
                                       outputs=int(values["outputs"]),cubic=float(values["cubic"]),
                                       logit_scale=float(values["logit_scale"]),symmetric_cap=float(values["symmetric_cap"]))
                    if not np.allclose(model.symmetric,model.symmetric.T) or np.any(np.diag(model.symmetric)):
                        raise ValueError(f"Expected symmetric zero-diagonal recurrent parameters; got {checkpoint}")
                    if np.linalg.norm(model.symmetric,2) > model.symmetric_cap+1e-12:
                        raise ValueError(f"Expected projected recurrent spectral norm; got {checkpoint}")
                    split_sets=[set(values[key].tolist()) for key in ("train_indices","val_indices","test_indices")]
                    if any(split_sets[i]&split_sets[j] for i,j in ((0,1),(0,2),(1,2))):
                        raise ValueError(f"Expected disjoint dataset splits; got {checkpoint}")
                    reference = {key:values[key].copy() for key in ("initial_symmetric","initial_inputs","skew","train_indices","val_indices","test_indices","feature_mean","feature_std")}
                    if seed in initial_by_seed:
                        for key,value in reference.items():
                            if not np.array_equal(value,initial_by_seed[seed][key]):
                                raise ValueError(f"Expected matched initialization/data for seed {seed}, {key}; got {checkpoint}")
                    else:
                        initial_by_seed[seed] = reference
                    result = evaluate(model,values["test_x"],values["test_y"])
                    error = abs(result["loss"]-row["test_loss"])
                    if error > 1e-12 or result["accuracy"] != row["test_accuracy"]:
                        raise ValueError(f"Checkpoint replay mismatch: {checkpoint}, {result}, {row}")
                    changed = float(np.linalg.norm(model.symmetric-values["initial_symmetric"]))
                    if changed < 1e-6:
                        raise ValueError(f"Expected trained recurrent parameters; unchanged at {checkpoint}")
                    suffix = re.search(r"(\d+)$",method)
                    m = int(suffix.group(1)) if suffix else 0
                    phases = 1 if method == "adjoint" else 3+2*m
                    expected_cost = config["epochs"]*len(values["train_x"])*phases
                    if row["equilibrations"] != expected_cost or row["state_reads"] != expected_cost:
                        raise ValueError(f"Expected {expected_cost} training states; got {row}")
                    checks.append(dict(method=method,seed=seed,loss_replay_abs_error=error,
                                       test_accuracy=result["accuracy"],recurrent_change_norm=changed,
                                       training_equilibrations=expected_cost,comparison_group=group))
            rows.extend(measured)
            sources.append(dict(path=str(path.resolve()),group=group,run=run,status=status))
    identities=[(r["method"],r["seed"],r["epoch"]) for r in rows]
    if len(identities) != len(set(identities)):
        raise ValueError("Expected distinct method/seed/epoch tuples across source runs")
    args.output.mkdir(parents=True)
    write_csv(args.output/"metrics.csv",rows)
    write_json(args.output/"sources.json",sources)
    write_json(args.output/"verification.json",dict(state="complete",rows=len(rows),
               checkpoint_replays=checks,max_force_residual=max(r["max_force_residual"] for r in rows)))
    report(args.output,rows,shared_config,False)
    with (args.output/"report.md").open("a") as stream:
        stream.write("\nThe orthogonal_mc8 rows belong to a separately declared exploratory follow-up; sources.json preserves the original run groups.\n")
    print(f"Validated {len(checks)} checkpoints and {len(rows)} epoch rows: {args.output.resolve()}")


if __name__ == "__main__":
    main()
