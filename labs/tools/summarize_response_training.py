#!/usr/bin/env python3
"""Replay DC-calibrated EqProp training and compare verified matched controls.

Control replay is reused only from a completed structured verification artifact;
the comparison independently checks checkpoint initialization and dataset pairing.
No source run, calibration, or checkpoint is modified.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from labs.tools.run_random_nudge_hopfield import write_csv, write_json
from labs.tools.summarize_recurrent_eqprop_digits import read_metrics
from labs.tools.summarize_structured_circulation import DATA_KEYS, replay_checkpoint
from labs.tools.train_recurrent_eqprop_digits import dataset, evaluate
from labs.tools.train_response_loop_feedback import METHOD, read_calibration, require


PHASES = {METHOD: 3, "contrastive_ep": 3, "known_skew_asymep": 3,
          "circulation_asymep": 3, "learned_mc4": 9}
LABELS = {METHOD: "DC-calibrated feedback", "contrastive_ep": "Ordinary EP",
          "known_skew_asymep": "Known-skew AsymEP", "circulation_asymep": "Noise-calibrated loops",
          "learned_mc4": "Learned baseline + MC4"}


def close(actual, expected, label):
    require(np.allclose(actual, expected, atol=1e-12, rtol=1e-12),
            f"Expected matching {label}; got {actual} vs {expected}")


def mean_sd(values):
    return dict(mean=float(np.mean(values)), std=float(np.std(values)))


def collect_dc(paths):
    records, rows, checks, sources = {}, [], [], []
    for path in paths:
        path = path.resolve()
        run = json.loads((path/"run.json").read_text())
        config = run["config"]
        require(json.loads((path/"config.json").read_text()) == config, "Expected matching saved training config")
        status = json.loads((path/"status.json").read_text())
        expected = {(n,s) for n in config["sizes"] for s in config["seeds"]}
        require(status["status"] == "complete"
                and status["completed_training"] == status["expected_training"] == run["expected_training_trajectories"] == len(expected),
                f"Expected complete declared DC training; got {path}: {status}")
        source_run, source_status, _, _, cases = read_calibration(
            Path(config["calibration_source"]), config["sizes"], config["seeds"])
        require(run["reused_calibration_run"] == source_run and run["reused_calibration_status"] == source_status,
                f"Expected unchanged source-calibration records; got {path}")
        require(source_run["config"]["steps"] == 10 and source_run["config"]["loops"] == 4,
                "Expected the selected four-loop, ten-update calibration budget")
        summary = json.loads((path/"summary.json").read_text())
        declared = {(c["size"],c["seed"]): c for c in summary["cases"]}
        require(set(declared) == expected and len(declared) == len(summary["cases"])
                and not summary["new_calibration_performed"], "Expected exact reused-calibration case coverage")
        metric_rows = read_metrics(path/"metrics.csv")
        require(len(metric_rows) == len(expected)*(config["epochs"]+1), "Expected all DC trajectory epoch rows")
        require({(int(r["size"]),int(r["seed"])) for r in metric_rows} == expected,
                "Expected only declared DC trajectory identities")
        data = dataset(config["quick"])
        expected_checkpoints = set()
        for case in cases:
            identity = case["size"], case["seed"]
            require(identity not in records, f"Expected unique width/seed across supplied DC runs; got {identity}")
            directory = path/f"n{identity[0]}_seed{identity[1]}"
            saved_source = json.loads((directory/"calibration_source.json").read_text())
            require(saved_source["summary"] == case["summary"]
                    and saved_source["source_config"] == source_run["config"], "Expected saved calibration summary/config equality")
            series = sorted([r for r in metric_rows if (r["size"],r["seed"]) == identity], key=lambda r:r["epoch"])
            require([r["epoch"] for r in series] == list(range(config["epochs"]+1)), "Expected unique complete DC epoch coverage")
            for row in series:
                require(row["method"] == METHOD and row["implementation_method"] == "circulation_asymep",
                        "Expected explicit external and implementation method labels")
                count = 3*len(data["train_y"])*row["epoch"]
                require(row["equilibrations"] == row["state_reads"] == count
                        and row["max_force_residual"] <= 1e-9
                        and row["total_probe_excitation_sq"] == row["training_random_adjoint_probe_pairs"] == 0,
                        f"Expected converged three-phase DC training without new random probes; got {row}")
                for key,value in dict(calibration_perturbed_equilibrations=160, calibration_anchor_equilibrations=1,
                                      calibration_scalar_projection_reads=160, calibration_anchor_projection_reads=8,
                                      calibration_process_noise_channels=0, learned_controller_coefficients=4,
                                      known_wiring_coefficients=8*case["size"]).items():
                    require(row[key] == value, f"Expected {key}={value}; got {row[key]}")
                close(row["learning_rate"], config["learning_rate"], "training learning rate")
                close(row["momentum"], config["momentum"], "training momentum")
                require((row["epoch"] == config["epochs"]) == ("test_accuracy" in row and "test_loss" in row),
                        "Expected held-out test only at final epoch")
            checkpoint = directory/f"{METHOD}_seed{identity[1]}_lr0.1.npz"
            expected_checkpoints.add(checkpoint)
            require(Path(series[-1]["checkpoint"]).resolve() == checkpoint, "Expected checkpoint in declared case directory")
            for key,value in declared[identity]["final_training"].items():
                require(key in series[-1], f"Expected summary field {key}")
                if isinstance(value,(int,float)):
                    close(series[-1][key],value,f"final summary {key}")
                else:
                    require(series[-1][key] == value, f"Expected matching final summary {key}")
            require(declared[identity]["calibration_summary"] == case["summary"]
                    and declared[identity]["controller_frozen_after_calibration"], "Expected frozen source calibration")
            for split,label in (("train","train"),("val","validation")):
                result = evaluate(case["model"],data[split+"_x"],data[split+"_y"])
                close(series[0][label+"_loss"],result["loss"],f"initial {label} loss")
                close(series[0][label+"_accuracy"],result["accuracy"],f"initial {label} accuracy")
            replay = replay_checkpoint(checkpoint,case["model"],case["controller"],data,config,"circulation_asymep",series)
            replay.update(method=METHOD,size=identity[0],seed=identity[1],source_run=str(path))
            checks.append(replay)
            for row in series:
                row.update(source_run=str(path), declared_epochs=config["epochs"], calibration_record_time=0.,
                           calibration_dc_equilibrations=161, calibration_scalar_reads=168)
            rows.extend(series)
            records[identity] = dict(case,config=config,data=data,initial_metrics=series[0],source_run=str(path))
        require(set(path.glob("n*_seed*/*.npz")) == expected_checkpoints, "Expected exactly the declared DC checkpoints")
        sources.append(dict(path=str(path),run=run,status=status))
    return records,rows,checks,sources


def matched_controls(directory, records):
    """Reuse terminal replay evidence, then independently check same-case pairing."""
    directory = directory.resolve()
    verification = json.loads((directory/"verification.json").read_text())
    require(verification["declared_coverage_verified"] and verification["gradient_audits_recomputed"],
            "Expected completed structured-control verification")
    inventory = {Path(c["checkpoint"]).resolve():c for c in verification["checkpoint_replays_details"]}
    require(len(inventory) == verification["checkpoint_replays"], "Expected unique verified control checkpoints")
    sources = json.loads((directory/"sources.json").read_text())
    source_configs = {}
    for source in sources:
        path = Path(source["path"]).resolve()
        current = json.loads((path/"run.json").read_text())
        require(current == source["run"], f"Expected unchanged verified control run; got {path}")
        status = json.loads((path/"status.json").read_text())
        require(status["status"] == "complete"
                and status["completed_training"] == current["expected_training_trajectories"],
                f"Expected terminal declared control coverage; got {path}")
        source_configs[str(path)] = current["config"]
    rows = read_metrics(directory/"metrics.csv")
    selected, pairing = [], []
    for identity, case in records.items():
        candidates = [r for r in rows if (int(r["size"]),int(r["seed"])) == identity]
        expected_methods = {"contrastive_ep","known_skew_asymep","circulation_asymep","learned_mc4"}
        require({r["method"] for r in candidates} == expected_methods,
                f"Expected all four matched control methods for {identity}")
        for method in sorted(expected_methods):
            series = sorted([r for r in candidates if r["method"] == method],key=lambda r:r["epoch"])
            config = source_configs[series[0]["source_run"]]
            require([r["epoch"] for r in series] == list(range(case["config"]["epochs"]+1)),
                    f"Expected matching epoch coverage for {identity}/{method}")
            for key in ("epochs","quick","loops","learning_rate","momentum","read_noise"):
                require(config[key] == case["config"][key], f"Expected matched {key} across DC and controls")
            checkpoint = Path(series[0]["source_run"])/f"n{identity[0]}_seed{identity[1]}"/"training"/method/f"{method}_seed{identity[1]}_lr{config['learning_rate']:g}.npz"
            require(checkpoint in inventory, f"Expected control checkpoint in verified inventory; got {checkpoint}")
            with np.load(checkpoint) as saved:
                for key in DATA_KEYS:
                    require(np.array_equal(saved[key],case["data"][key]), f"Expected identical DC/control data: {key}")
                for key,value in (("initial_symmetric",case["model"].symmetric),("initial_inputs",case["model"].inputs),
                                  ("skew",case["model"].skew)):
                    require(np.array_equal(saved[key],value), f"Expected identical DC/control physical initialization: {key}")
            for label in ("train","validation"):
                for metric in ("loss","accuracy"):
                    close(series[0][label+"_"+metric],case["initial_metrics"][label+"_"+metric],"matched initial metrics")
            replay = inventory[checkpoint]
            for label in ("validation","test"):
                close(series[-1][label+"_loss"], replay[label]["loss"], "verified control loss")
                close(series[-1][label+"_accuracy"], replay[label]["accuracy"], "verified control accuracy")
            for row in series:
                count = PHASES[method]*len(case["data"]["train_y"])*row["epoch"]
                require(row["equilibrations"] == row["state_reads"] == count, "Expected matched-control physical budgets")
                row.update(calibration_dc_equilibrations=0,
                           calibration_record_time=row["calibration_total_record_time"],
                           calibration_scalar_reads=row["calibration_scalar_measurement_reads"])
            selected.extend(series)
            pairing.append(dict(size=identity[0],seed=identity[1],method=method,checkpoint=str(checkpoint),
                                initialization_and_data_match=True,replay_source=str(directory)))
    return selected,pairing,verification


def aggregate(rows):
    results = []
    for size,method in sorted({(int(r["size"]),r["method"]) for r in rows}):
        series = [r for r in rows if int(r["size"]) == size and r["method"] == method]
        finals = [r for r in series if r["epoch"] == r["declared_epochs"]]
        require(len(finals) == len({r["seed"] for r in finals}), "Expected one final row per seed")
        record = dict(size=size,method=method,seeds=sorted(int(r["seed"]) for r in finals),
                      epochs=int(finals[0]["epoch"]),equilibrations_per_example=PHASES[method])
        for key in ("test_accuracy","validation_accuracy","test_loss","equilibrations","calibration_dc_equilibrations",
                    "calibration_record_time","calibration_scalar_reads"):
            record[key] = mean_sd([r[key] for r in finals])
        last = [r for r in series if r["epoch"] == record["epochs"]-1]
        for key in ("gradient_cosine","gradient_relative_error","applied_update_cosine"):
            record[key+"_last_audited_epoch"] = mean_sd([r[key] for r in last])
        results.append(record)
    return results


def report(output,results,verification):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,axes = plt.subplots(1,2,figsize=(11,4.5),constrained_layout=True)
    for method in PHASES:
        selected = sorted([r for r in results if r["method"] == method],key=lambda r:r["size"])
        if not selected:
            continue
        for axis,key,scale in ((axes[0],"test_accuracy",100),(axes[1],"gradient_cosine_last_audited_epoch",1)):
            axis.errorbar([r["size"] for r in selected],[scale*r[key]["mean"] for r in selected],
                          yerr=[scale*r[key]["std"] for r in selected],marker="o",capsize=3,label=LABELS[method])
    axes[0].set(xlabel="State count",ylabel="Final test accuracy (%)",title="Matched physical model and optimizer",ylim=(0,101))
    axes[1].set(xlabel="State count",ylabel="Raw parameter-gradient cosine",title="First minibatch of final training epoch",ylim=(-1.05,1.05))
    for axis in axes:
        axis.set_xticks(sorted({r["size"] for r in results}))
        axis.spines[["top","right"]].set_visible(False)
        axis.legend(fontsize=7)
    fig.savefig(output/"response_training.png",dpi=180)
    fig.savefig(output/"response_training.svg")
    plt.close(fig)
    lines = ["# Training with response-calibrated physical feedback", "",
             f"Independently replayed {verification['dc_checkpoint_replays']} DC checkpoints. "
             f"Matched, previously verified control checkpoints: {verification['matched_control_checkpoints']}. "
             f"Complete six-case DC grid: {verification['dc_main_grid_complete']}. "
             f"Complete matched comparison: {verification['matched_comparison_complete']}.","",
             "Each DC controller uses four known physical loops, ten calibration updates, 160 perturbed equilibrations, "
             "one free anchor, and 168 scalar projection readings including anchor projections. Calibration is paid once; "
             "training uses the frozen controller, actual contrastive EqProp, three equilibrations per example, and zero fresh random adjoint probes.","",
             "| States | Method | Seeds | Test accuracy (%) | Training equilibrations | DC calibration equilibrations | Noise-calibration record time | Last gradient cosine |",
             "|---:|---|---|---:|---:|---:|---:|---:|"]
    for row in results:
        accuracy,cosine = row["test_accuracy"],row["gradient_cosine_last_audited_epoch"]
        lines.append(f"| {row['size']} | {LABELS[row['method']]} | {row['seeds']} | {100*accuracy['mean']:.2f} ± {100*accuracy['std']:.2f} | "
                     f"{row['equilibrations']['mean']:,.0f} | {row['calibration_dc_equilibrations']['mean']:,.0f} | "
                     f"{row['calibration_record_time']['mean']:,.0f} | {cosine['mean']:.4f} ± {cosine['std']:.4f} |")
    lines += ["", "Mean ± population SD across seeds. Shared LR 0.1, weight-gradient EMA 0.9, and nudged read noise 1e-5; "
              "15 epochs for the main grid. Controls use the exact same initial symmetric weights, input weights, fixed skew, and data. "
              "Control final validation/test replay and gradient-audit evidence come from the linked structured verification; "
              "this report independently checks same-case pairing.","",
              "Noise-calibration record time is not converted into equilibrium counts. DC calibration and training counts can be added as equilibrations "
              "but differing settling times, actuation, and readout costs remain. All oracle audit work is excluded from physical learning counts.","",
              "These results test a known four-loop structural prior and fixed non-reciprocity. They do not establish a four-parameter correction "
              "for arbitrary asymmetric systems, hardware speedups, or superiority from small accuracy differences."]
    (output/"report.md").write_text("\n".join(lines)+"\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs",type=Path,nargs="+",required=True)
    parser.add_argument("--control-verification",type=Path)
    parser.add_argument("--output",type=Path,required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"Expected fresh output directory; got {args.output}")
    records,rows,checks,sources = collect_dc(args.runs)
    pairing,control_evidence = [],None
    if args.control_verification is not None:
        control_rows,pairing,control_evidence = matched_controls(args.control_verification,records)
        rows.extend(control_rows)
    expected = {(n,s) for n in (64,256) for s in (0,1,2)}
    complete = set(records) == expected and all(c["config"]["epochs"] == 15 and not c["config"]["quick"] for c in records.values())
    verification = dict(declared_dc_coverage_complete=True,dc_checkpoint_replays=len(checks),
                        dc_main_grid_complete=complete,matched_control_checkpoints=len(pairing),
                        matched_comparison_complete=complete and len(pairing) == 24,
                        control_verification=None if args.control_verification is None else str(args.control_verification.resolve()),
                        calibration_reused_without_refitting=True)
    results = aggregate(rows)
    args.output.mkdir(parents=True)
    write_json(args.output/"summary.json",dict(verification=verification,results=results))
    write_json(args.output/"verification.json",dict(**verification,dc_replays=checks,control_pairing=pairing,
                                                   reused_control_verification=control_evidence))
    write_json(args.output/"sources.json",sources)
    write_csv(args.output/"metrics.csv",rows)
    report(args.output,results,verification)
    print(json.dumps(verification,indent=2),flush=True)


if __name__ == "__main__":
    main()
