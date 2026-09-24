"""Validate the three sigma=1e-3 outcomes against previously completed controls."""
import csv
import json
import re
from datetime import datetime, timezone

from experiments.analyze_conv3_p90_read_noise import ROOT, STUDY as PARENT, collect_case, read

STUDY = ROOT / "results/eqprop-conv3-p90-read-noise-1em3-20260920-v1"
REPORT = ROOT / "paper_ready_results/conv3_p90_read_noise_1em3_20260920.md"


def plot_trajectories(rows, controls):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(10, 3.5), sharey=True)
    for ax, row in zip(axes, rows, strict=True):
        clean = controls[row["scheme"]]["epochs_data"]
        ax.plot([e["epoch"] for e in clean],
                [100 * e["metrics"]["validation_accuracy"] for e in clean],
                color="0.45", label="Reused clean control")
        observed = row["epochs_data"]
        if observed:
            ax.plot([e["epoch"] for e in observed],
                    [100 * e["metrics"]["validation_accuracy"] for e in observed],
                    color="tab:blue", linestyle="-" if row["state"] == "complete" else "--",
                    marker=".", markersize=4,
                    label=r"Read noise $\sigma=10^{-3}$")
        if row.get("failure_kind") == "scientific_nonfinite":
            match = re.search(r"epoch=(\d+)", row["failure_detail"])
            if match:
                epoch = int(match.group(1))
                ax.axvline(epoch, color="tab:red", linestyle=":", alpha=.8)
                ax.text(.98, .05, f"Non-finite in epoch {epoch}", transform=ax.transAxes,
                        ha="right", color="tab:red", fontsize=9)
        ax.set(title=f"{row['scheme'].capitalize()} (beta={row['beta']:.4g})", xlabel="Epoch", xlim=(1, 30))
        ax.set_xticks([1, 10, 20, 30])
        ax.grid(alpha=.2)
    axes[0].set_ylabel("Validation accuracy (%)")
    axes[0].legend(loc="lower right", fontsize=8)
    fig.text(.5, .01, "MNIST validation, seed 0. Partial/failed curves end at the last completed epoch.",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .04, 1, 1))
    for suffix in ("png", "pdf"):
        fig.savefig(REPORT.with_name(REPORT.stem + f"_epochs.{suffix}"), dpi=180)
    plt.close(fig)


def analyze():
    cases = read(STUDY / "cases.json")
    assert len(cases) == 3 and {c["scheme"] for c in cases} == {"baseline", "legacy", "ours"}
    assert all(c["sigma"] == 0.001 and c["epochs"] == 30 and c["seed"] == 0 for c in cases)
    controls = {c["scheme"]: collect_case(c) for c in read(PARENT / "cases.json")
                if c["gpu"] == "RTX5090" and c["sigma"] == 0}
    rows = [collect_case(c, study=STUDY) for c in cases]
    for row in rows:
        clean = controls[row["scheme"]]
        assert clean["state"] == "complete" and clean["bundle"] == row["clean_reference_bundle"]
        row["clean_validation_pct"] = clean["final_validation_pct"]
        if row["state"] == "complete":
            for key in ("dataset_provenance", "initial_parameter_state_sha256"):
                assert row[key] == clean[key], key
            for key in ("torch", "cuda", "name"):
                assert row["gpu_environment"][key] == clean["gpu_environment"][key], key
            row["final_drop_pp"] = clean["final_validation_pct"] - row["final_validation_pct"]
        elif row["epochs_data"]:
            row["latest_validation_pct"] = 100 * row["epochs_data"][-1]["metrics"]["validation_accuracy"]
            row["best_observed_validation_pct"] = 100 * max(
                epoch["metrics"]["validation_accuracy"] for epoch in row["epochs_data"])
        if row["state"] == "failed":
            assert not (ROOT / row["bundle"] / "result.json").exists()
            log = (STUDY / row["local_relative_root"] / f"{row['case']}.log").read_text()
            errors = [line for line in log.splitlines() if line.startswith("NonFiniteTrainingError:")]
            row["failure_kind"] = "scientific_nonfinite" if errors else "other_failure"
            row["failure_detail"] = errors[-1] if errors else str(row["failure"])
    complete = sum(r["state"] == "complete" for r in rows)
    terminal = sum(r["state"] in ("complete", "failed", "startup-failed") for r in rows)
    payload = dict(updated_at=datetime.now(timezone.utc).isoformat(), expected=3,
                   completed=complete, terminal=terminal, rows=rows)
    (STUDY / "analysis.json").write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    fields = ["scheme", "beta", "sigma", "target", "state", "epochs_completed",
              "clean_validation_pct", "final_validation_pct", "best_validation_pct",
              "final_drop_pp", "latest_validation_pct", "best_observed_validation_pct", "best_to_final_drop_pp",
              "max_running_best_drawdown_pp", "max_running_best_drawdown_epoch",
              "passes_final_drop_screen", "bundle", "clean_reference_bundle", "failure",
              "failure_kind", "failure_detail"]
    with REPORT.with_suffix(".csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    text = ["# Conv3 p90 read noise: sigma 1e-3 extension", "",
            f"Updated {payload['updated_at']}. {complete}/3 completed; {terminal}/3 terminal.", "",
            "Three new seed-0 RTX5090 runs, each with a planned 30-epoch budget, frozen p90 betas, T=K=8, "
            "float64 centered EqProp and unchanged Adam learning rates. Ordinary MNIST "
            "55,000/5,000 train/validation; official test disabled. Independent Gaussian "
            "noise perturbs copied non-input endpoint voltages during gradient readout; "
            "relaxation and validation remain clean. Previous RTX5090 clean controls "
            "are reused; no prior noisy case is repeated. These are single-seed validation diagnostics.", "",
            "| Scheme | Beta | State / completed epochs | Clean (%) | New final / best (%) | Clean-relative drop (pp) |",
            "|---|---:|---|---:|---:|---:|"]
    for row in rows:
        accuracy = (f"{row['final_validation_pct']:.2f} / {row['best_validation_pct']:.2f}"
                    if row["state"] == "complete" else
                    "failed; no epoch-30 result" if row["state"] == "failed" else "pending")
        drop = f"{row['final_drop_pp']:+.2f}" if "final_drop_pp" in row else "—"
        text.append(f"| {row['scheme']} | {row['beta']:.6g} | {row['state']} / "
                    f"{row['epochs_completed']} | {row['clean_validation_pct']:.2f} | {accuracy} | {drop} |")
    for row in rows:
        if row["state"] == "failed":
            text += ["", f"**{row['scheme']} failure:** {row['failure_detail']}"]
            if "latest_validation_pct" in row:
                text.append(f"Last complete validation epoch: {row['epochs_completed']}, "
                            f"{row['latest_validation_pct']:.2f}%; best observed: "
                            f"{row['best_observed_validation_pct']:.2f}%. "
                            "These partial values are not thirty-epoch results. "
                            "The failure and saved best checkpoint are retained; no repeat is scheduled.")
    if terminal == 3:
        text += ["", f"All three planned cases have terminal outcomes; {complete} reached epoch 30. "
                 "Scientific failures are included in coverage and are not treated as missing runs."]
        if any(r.get("failure_kind") == "scientific_nonfinite" for r in rows):
            text += ["", "The zero-noise per-matrix cosine >0.90 selection and successful clean "
                     "training do not guarantee finite training at sigma=1e-3. "
                     "These observations compare fixed scheme-specific beta/LR operating points "
                     "in one seed; they do not establish an intrinsic or universal amplification ranking."]
    text += ["", "Positive drop means worse accuracy with noise. All scientific outcomes are retained. "
             "The endpoint screen requires the final accuracy to be strictly less than 5pp "
             "below its own best; the CSV also records temporary running-best drawdown. "
             "A passing endpoint screen does not establish smooth training or noise robustness.", "",
             "[Previous completed sweep](conv3_p90_read_noise_20260919.md) · "
             "[Launch plan](../docs/eqprop_conv3_p90_read_noise_1em3_plan_20260920.md)"]
    plot_trajectories(rows, controls)
    text += ["", f"![Epoch trajectories]({REPORT.stem}_epochs.png)"]
    REPORT.write_text("\n".join(text) + "\n")
    print(json.dumps({k: v for k, v in payload.items() if k != "rows"}))


if __name__ == "__main__":
    analyze()
