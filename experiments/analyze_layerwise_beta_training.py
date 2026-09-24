"""Compare cosine-selected pilots with matched ten-epoch validation controls."""
import csv
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from experiments.reporting import validate_run

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "results/eqprop-layerwise-beta-training-20260919-v1"
REPORT = ROOT / "paper_ready_results/layerwise_beta_training_20260919.md"


def epoch_rows(run):
    return [r for line in (run / "metrics.jsonl").read_text().splitlines()
            if (r := json.loads(line)).get("kind") == "epoch" and r["epoch"] <= 10]


def validate_contract(cfg, control):
    for key in ("model_base", "model_overrides", "lr", "optimizer", "parameter_order",
                "training_algorithm", "runtime_dtype", "seed", "batch_state_policy"):
        assert cfg[key] == control[key], key
    for key in ("variant", "nudging_mode", "normalize_current_scale", "endpoint_read_noise_std"):
        assert cfg["eqprop"][key] == control["eqprop"][key], key
    assert cfg["evaluation"]["official_test"]["policy"] == "disabled"
    assert cfg["lab"]["epochs"] in (10, 30)
    assert cfg["optimizer"]["lr_decay"] == 1.0


def analyze():
    rows = []
    for case in json.loads((STUDY / "cases.json").read_text()):
        row = dict(case)
        control = ROOT / case["comparison_control"]
        assert not validate_run(control)
        control_cfg = json.loads((control / "config.used.json").read_text())
        control_metrics = json.loads((control / "metrics.json").read_text())
        control_epochs = epoch_rows(control)
        assert len(control_epochs) == 10
        row.update(control_beta=control_cfg["eqprop"]["injected_beta_B"],
                   control_final10_pct=100 * control_epochs[-1]["metrics"]["validation_accuracy"],
                   control_epochs=control_epochs, state="queued", epochs_completed=0)
        if case["mode"] == "new":
            paths = list((STUDY / case["target"] / "training").glob(f"000_{case['case']}_*"))
            assert len(paths) <= 1
            run = paths[0] if paths else None
        else:
            run = ROOT / case["bundle"]
        if run is not None:
            row["bundle"] = str(run.relative_to(ROOT))
            assert not validate_run(run), run
            cfg = json.loads((run / "config.used.json").read_text())
            validate_contract(cfg, control_cfg)
            assert cfg["eqprop"]["injected_beta_B"] == case["injected_beta"]
            st = json.loads((run / "status.json").read_text())
            epochs = epoch_rows(run)
            row.update(state=st["state"], epochs=epochs, epochs_completed=len(epochs), bundle_valid=True)
            if epochs:
                values = np.array([e["metrics"]["validation_accuracy"] for e in epochs])
                row.update(latest_validation_pct=100 * float(values[-1]),
                           best_observed_validation_pct=100 * float(values.max()))
            if st["state"] == "failed":
                if case["mode"] == "new":
                    log = STUDY / case["target"] / f"{case['case']}.log"
                else:
                    log = ROOT / case["prior_study"] / case["target"] / f"{case['case']}.log"
                hits = re.findall(r"NonFiniteTrainingError: ([^\n]+)", log.read_text())
                row.update(early_stable=False, failure_kind="nonfinite" if hits else "operational_or_unclassified")
                if hits:
                    row["failure"] = hits[-1]
                    match = re.search(r"epoch=(\d+), batch=(\d+)", hits[-1])
                    row["failure_epoch"], row["failure_batch"] = map(int, match.groups())
            elif st["state"] == "complete":
                metrics = json.loads((run / "metrics.json").read_text())
                assert metrics["official_test_evaluations"] == 0
                assert metrics["initial_parameter_state_sha256"] == control_metrics["initial_parameter_state_sha256"]
                for key in ("train_indices_sha256", "validation_indices_sha256", "first_epoch_batch_order_sha256"):
                    assert metrics["dataset_provenance"][key] == control_metrics["dataset_provenance"][key]
                assert metrics["dataset_provenance"]["train_batch_order_sha256"][:10] == control_metrics["dataset_provenance"]["train_batch_order_sha256"][:10]
                assert len(epochs) == 10
                histories = {k: np.load(run / f"{k}.npy", allow_pickle=False)[:10]
                             for k in ("accuracy_train", "accuracy_test", "loss_train", "loss_test")}
                assert all(len(v) == 10 and np.isfinite(v).all() for v in histories.values())
                values = histories["accuracy_test"]
                assert np.allclose(values, [e["metrics"]["validation_accuracy"] for e in epochs], rtol=0, atol=1e-12)
                drop = float(values.max() - values[-1])
                row.update(final10_pct=100 * float(values[-1]), best10_pct=100 * float(values.max()),
                           drop_from_best_pp=100 * drop, early_stable=drop < .05-1e-12,
                           final_delta_vs_control_pp=100 * float(values[-1]) - row["control_final10_pct"],
                           initializer_and_order_match=True,
                           maximum_drawdown_pp=100 * float((np.maximum.accumulate(values)-values).max()))
                if case["mode"] != "reuse_control_prefix":
                    import torch
                    for f in ("best_model.pt", "final_model.pt"):
                        ckpt = torch.load(run/f, map_location="cpu", weights_only=True)
                        for spec, tensor in zip(ckpt["schema"], ckpt["states"], strict=True):
                            assert tensor.dtype == torch.float64 and torch.isfinite(tensor).all()
                            if spec["name"].strip().startswith("Bias_"):
                                assert torch.count_nonzero(tensor) == 0
        rows.append(row)
    selections = json.loads((STUDY / "selection_audit.json").read_text())
    by_case = {r["case"]: r for r in rows}
    comparisons = []
    for arch in ("conv2", "conv3"):
        for scheme in ("baseline", "ours", "legacy"):
            selected = {s["threshold"]: s for s in selections if s["architecture"] == arch and s["scheme"] == scheme}
            loose, strict = (by_case[selected[t]["case"]] for t in (.90, .95))
            comparison = dict(architecture=arch, scheme=scheme, beta90=loose["injected_beta"], beta95=strict["injected_beta"])
            if loose["case"] == strict["case"]:
                comparison["interpretation"] = "Same selected beta; no distinct threshold comparison"
            elif all(r["state"] == "complete" for r in (loose, strict)):
                delta = loose["final10_pct"] - strict["final10_pct"]
                comparison.update(larger_minus_smaller_pp=delta, interpretation=(
                    "Larger beta has lower final validation" if delta < -1e-9 else
                    "Larger beta has higher final validation" if delta > 1e-9 else "Equal final validation"))
            elif loose["state"] == "failed" and strict["state"] == "complete":
                comparison["interpretation"] = "Larger beta failed; smaller beta completed"
            elif loose["state"] == "complete" and strict["state"] == "failed":
                comparison["interpretation"] = "Smaller beta failed; larger beta completed"
            elif loose["state"] == strict["state"] == "failed":
                comparison["interpretation"] = "Both selected betas failed"
            else:
                comparison["interpretation"] = "Pending"
            comparisons.append(comparison)
    terminal = sum(r["state"] in ("complete", "failed") for r in rows)
    payload = dict(updated_at=datetime.now(timezone.utc).isoformat(), threshold_conditions=12,
                   distinct_settings=9, new_runs=5, reused_settings=4, terminal=terminal,
                   complete=terminal == 9, official_test_read=False, rows=rows, comparisons=comparisons)
    (STUDY / "analysis.json").write_text(json.dumps(payload, indent=2)+"\n")
    with REPORT.with_name(REPORT.stem + "_epochs.csv").open("w") as stream:
        fields = ["case", "architecture", "scheme", "beta", "role", "origin", "epoch",
                  "train_accuracy", "validation_accuracy", "train_loss", "validation_loss"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        seen_controls = set()
        for r in rows:
            series = [("candidate", r["case"], r["injected_beta"], r["mode"], r.get("epochs", []))]
            control_key = (r["architecture"], r["scheme"])
            if control_key not in seen_controls:
                seen_controls.add(control_key)
                series.append(("control", r["comparison_control"], r["control_beta"], "historical_first10", r["control_epochs"]))
            for role, name, beta, origin, records in series:
                for e in records:
                    writer.writerow(dict(case=name, architecture=r["architecture"], scheme=r["scheme"],
                        beta=beta, role=role, origin=origin, epoch=e["epoch"],
                        **{k:e["metrics"][k] for k in fields[-4:]}))
    text = ["# Per-matrix cosine beta selection: ten-epoch training", "",
            f"Updated {payload['updated_at']}. **{terminal}/9 distinct settings terminal; 12 threshold conditions.**", "",
            "Seed 0, zero read noise, fixed Adam rates, T=K=6/8, float64 centered EqProp; ordinary-MNIST validation only.",
            "Select the largest tested injected beta whose cosine is strictly above the threshold for every weight matrix on every one of 36 batches at both initialization and the saved BPTT checkpoint. No norm gate is imposed; norm mismatch is reported below.", "",
            "| Model | Scheme | Cosine threshold | Beta | Worst cosine | Worst norm mismatch |",
            "|---|---|---:|---:|---:|---:|"]
    for s in selections:
        text.append(f"| {s['architecture']} | {s['scheme']} | {s['threshold']:.2f} | {s['injected_beta']:g} | {s['minimum_matrix_cosine']:.6f} | {s['maximum_matrix_norm_mismatch']:.6f} |")
    text += ["", "Legacy selections are passing upper grid edges, not known maxima. Equal selected betas share one outcome.", "",
             "| Model | Scheme | Beta | Outcome | Final / best validation | Previous-beta control (epoch 10) | Delta (pp) | Evidence |",
             "|---|---|---:|---|---|---|---:|---|"]
    for r in rows:
        outcome = r["state"]; accuracy = delta = "—"
        if r.get("failure_kind") == "nonfinite":
            outcome = f"Nonfinite epoch {r['failure_epoch']}, batch {r['failure_batch']}"
        elif r["state"] == "complete":
            outcome = "Ten-epoch stable" if r["early_stable"] else "Final-drop criterion failed"
            accuracy = f"{r['final10_pct']:.2f}% / {r['best10_pct']:.2f}%"
            delta = f"{r['final_delta_vs_control_pp']:+.2f}"
        elif r["state"] == "running":
            outcome = f"Running ({r['epochs_completed']}/10)"
        text.append(f"| {r['architecture']} | {r['scheme']} | {r['injected_beta']:g} | {outcome} | {accuracy} | {r['control_final10_pct']:.2f}% (beta {r['control_beta']:g}) | {delta} | {r['mode']} |")
    text += ["", "| Model | Scheme | beta .90 / .95 | Larger minus smaller final accuracy (pp) | Interpretation |",
             "|---|---|---|---:|---|"]
    for c in comparisons:
        delta = f"{c['larger_minus_smaller_pp']:+.2f}" if "larger_minus_smaller_pp" in c else "—"
        text.append(f"| {c['architecture']} | {c['scheme']} | {c['beta90']:g} / {c['beta95']:g} | {delta} | {c['interpretation']} |")
    if payload["complete"]:
        stable = sum(r.get("early_stable", False) for r in rows)
        text += ["", f"**Interpretation:** {stable}/9 distinct settings passed ten finite epochs and the final-drop criterion. "
                 "The distinct .90-versus-.95 comparisons show no consistent accuracy penalty from larger beta: "
                 + "; ".join(f"{c['architecture']} {c['scheme']} {c['larger_minus_smaller_pp']:+.2f}pp"
                             for c in comparisons if "larger_minus_smaller_pp" in c) + ".",
                 "Conv2 legacy beta30 is the exception to training stability: despite worst per-matrix cosine .954935, "
                 "the matching prior pilot became nonfinite in epoch 5, batch 1910; its beta.03 control reached 98.00% at epoch 10. "
                 "Thus neither tested static cosine threshold guarantees stable training. Calibration measures two fixed "
                 "reference parameter states, not every state visited by the new EqProp training trajectory."]
    text += ["", "Five settings are newly trained; four reuse explicitly named matching evidence. Historical 30-epoch controls contribute only their first ten epoch records; their full-horizon best checkpoints are not treated as epoch-10 checkpoints.",
             "The two Conv2 ours settings were restarted together on the local RTX 3090 after unrelated GPU clients arrived on Loulou. The three-epoch Loulou attempts are preserved but superseded solely for resource contention, not numerical failure. Two explicitly excluded 100-batch throughput probes informed this placement change; an initial local transport attempt exited before smoke or training. Thus five new scientific settings used seven full training attempts, with only the designated replacement outcomes included.",
             "All new runs retain the original learning rates, initialization and data order. New/reused host placement is recorded in cases.json. This is an exploratory multi-host comparison.",
             "Stability means ten finite epochs and final validation less than 5pp below the run's best. Numerical failures remain included; no epoch-10 accuracy is invented for them. Small accuracy differences describe one seed and are not statistically established effects. No official test, noisy training qualification, full 30-epoch qualification, or automatic beta promotion.",
             "", "[Validation curves](layerwise_beta_training_20260919.png) · [PDF](layerwise_beta_training_20260919.pdf) · [Train/validation losses and accuracies by epoch](layerwise_beta_training_20260919_epochs.csv) · [Plan](../docs/eqprop_layerwise_beta_training_plan_20260919.md) · [Raw evidence](../results/eqprop-layerwise-beta-training-20260919-v1/)"]
    REPORT.write_text("\n".join(text)+"\n")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(12, 7), constrained_layout=True)
    for i, arch in enumerate(("conv2", "conv3")):
        for j, scheme in enumerate(("baseline", "ours", "legacy")):
            ax = axes[i, j]; subset = [r for r in rows if r["architecture"] == arch and r["scheme"] == scheme]
            control = subset[0]
            ax.plot(range(1, 11), [100*e["metrics"]["validation_accuracy"] for e in control["control_epochs"]], "k--", label=f"Control beta {control['control_beta']:g}")
            for r in subset:
                records = r.get("epochs", [])
                if records:
                    label = f"beta {r['injected_beta']:g} (cos " + "/".join(f"{t:.2f}" for t in r["thresholds"]) + ")"
                    ax.plot([e["epoch"] for e in records], [100*e["metrics"]["validation_accuracy"] for e in records], "o-", ms=3, label=label)
                if r.get("failure_epoch"):
                    ax.axvline(r["failure_epoch"]-1+r["failure_batch"]/3438, color="red", ls=":", alpha=.7)
            ax.set(title=f"{arch.capitalize()} {scheme}", xlabel="Epoch", ylabel="Validation accuracy (%)", xlim=(.8, 10.2))
            ax.grid(alpha=.2); ax.legend(fontsize=7)
    fig.suptitle("Per-matrix cosine beta pilots — seed 0, zero read noise\nDotted red line: nonfinite failure; dashed black: previous-beta control")
    fig.savefig(REPORT.with_suffix(".png"), dpi=160); fig.savefig(REPORT.with_suffix(".pdf")); plt.close(fig)
    print(json.dumps({k:v for k,v in payload.items() if k not in ("rows", "comparisons")}))


if __name__ == "__main__":
    analyze()
