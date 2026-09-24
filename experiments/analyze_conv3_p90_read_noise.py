"""Collect the split V100/A100 Conv3 p90 noise outcomes and matched controls."""
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from experiments.analyze_conv3_refined_beta_training import without_dataset_root
from experiments.reporting import validate_run

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "results/eqprop-conv3-p90-read-noise-20260919-v1"
REPORT = ROOT / "paper_ready_results/conv3_p90_read_noise_20260919.md"
SOURCE_SHA = "74f41a03ce4718eebcd227bb38e4bbd2309e6b30bb9dd5515c099bb880068819"


def read(path):
    return json.loads(path.read_text())


def epoch_rows(run):
    path = run / "metrics.jsonl"
    if not path.exists():
        return []
    return [row for line in path.read_text().splitlines()
            if (row := json.loads(line)).get("kind") == "epoch"]


def historical_rows():
    rows = []
    for filename in ("read_noise_run_status_20260914.csv", "baseline_read_noise_run_status_20260916.csv"):
        with (REPORT.parent / filename).open() as stream:
            for old in csv.DictReader(stream):
                if old["architecture"] != "conv3" or not float(old["sigma"]):
                    continue
                assert int(old["epochs"]) == 30
                bundle = REPORT.parent / old["collected_result"]
                metrics = read(bundle.with_name("metrics.json"))
                assert metrics["official_test_evaluations"] == 0
                assert abs(100*metrics["final_validation_accuracy"] - float(old["final_validation_percent"])) < 1e-9
                rows.append(dict(scheme=old.get("scheme", "baseline"),
                                 beta=float(old.get("injected_beta", old.get("beta"))),
                                 sigma=float(old["sigma"]), final_validation_pct=float(old["final_validation_percent"]),
                                 final_drop_pp=float(old["final_drop_pp"]),
                                 environment=old["noise_stream_group"], bundle=str(bundle.relative_to(ROOT))))
    assert len(rows) == 15
    return rows


def collect_case(case, study=None):
    study = STUDY if study is None else study
    row = dict(case, state="pending", epochs_completed=0, epochs_data=[])
    config = ROOT / case["config"]
    assert hashlib.sha256(config.read_bytes()).hexdigest() == case["config_sha256"]
    cfg = read(config)
    assert cfg["lab"]["epochs"] == case["epochs"] == 30
    single = case.get("layout") == "single"
    task = (study / case["local_relative_root"] if single else
            study / "jz" / case["production_root"] / f"task_{case['index']}")
    bundles = list((task / "runs").glob(f"000_{case['case']}_*"))
    assert len(bundles) <= 1, f"Ambiguous outcome: {case['case']}"
    if not bundles:
        exits = ([task / f"{case['case']}.exit_code"] if single else
                 list(task.glob("segments/chunk_*/worker_started/exit_code")))
        exits = [path for path in exits if path.exists()]
        if any(path.read_text().strip() != "0" for path in exits):
            row.update(state="startup-failed", failure="Worker exited without a canonical bundle; inspect segment receipts")
        return row
    run = bundles[0]
    assert not validate_run(run), run
    status = read(run / "status.json")
    epochs = epoch_rows(run)
    assert [e["epoch"] for e in epochs] == list(range(1, len(epochs) + 1))
    row.update(state=status["state"], epochs_completed=len(epochs), epochs_data=epochs,
               bundle=str(run.relative_to(ROOT)))
    if status.get("progress", {}).get("stage") == "checkpointed_pause":
        row["state"] = "paused"
    if status["state"] == "failed":
        row["failure"] = status
    used_config = run / "config.used.json"
    if not used_config.exists():
        # exact_run publishes the starting bundle before the trainer resolves
        # its config. A live collection can legitimately land in that window.
        assert not epochs and status["state"] in ("running", "failed"), run
        assert status["state"] == "failed" or status.get("progress", {}).get("stage") == "starting", run
        row["collection_note"] = "Resolved config not yet available; startup only"
        return row
    assert without_dataset_root(read(used_config)) == without_dataset_root(cfg)
    if status["state"] != "complete":
        return row
    assert len(epochs) == 30
    gpu = read(task / ("gpu.json" if single else "segments/chunk_0/gpu.json"))
    assert case.get("gpu_model", case["gpu"]) in gpu["name"], gpu
    if single:
        summary = read(task / f"{case['case']}.summary.json")
        assert len(summary) == 1 and summary[0]["returncode"] == 0 and not summary[0]["smoke"]
        assert summary[0]["config_sha256"] == case["config_sha256"]
        assert summary[0]["epochs"] == 30
        assert (task / f"{case['case']}.exit_code").read_text().strip() == "0"
    for chunk, returncode in ([] if single else enumerate((75, 75, 0))):
        segment = task / f"segments/chunk_{chunk}"
        summary = read(segment / "summary.json")
        assert len(summary) == 1 and summary[0]["returncode"] == returncode and not summary[0]["smoke"]
        assert summary[0]["config_sha256"] == case["config_sha256"]
        assert summary[0]["epochs"] == 30
        assert (segment / "worker_started/exit_code").read_text().strip() == "0"
        segment_gpu = read(segment / "gpu.json")
        assert all(segment_gpu[key] == gpu[key] for key in ("name", "torch", "cuda"))
    manifest, result, metrics = (read(run / f"{name}.json") for name in ("manifest", "result", "metrics"))
    assert not manifest["smoke"] and not result["smoke"]
    assert result["completion"]["criteria_met"]
    assert manifest["git"]["source_archive_sha256"] == SOURCE_SHA
    assert manifest["runtime"]["target"] == case.get("target", "jean-zay")
    assert metrics["official_test_evaluations"] == 0 and not result["dataset"]["official_test_read"]
    assert cfg["max_batches"] is None and cfg["max_test_batches"] is None
    assert metrics["eqprop_endpoint_read_noise_draw_count"] == (825120 if case["sigma"] else 0)
    assert metrics["eqprop"]["endpoint_read_noise_std"] == case["sigma"]
    assert metrics["eqprop"]["endpoint_read_noise_seed"] == 2026081601
    assert cfg["eqprop"]["injected_beta_B"] == case["beta"]
    parent = ROOT / cfg["read_noise_study"]["parent_config"]
    assert hashlib.sha256(parent.read_bytes()).hexdigest() == cfg["read_noise_study"]["parent_config_sha256"]
    qualification = ROOT / cfg["read_noise_study"]["qualification_clean_bundle"]
    qm = read(qualification / "metrics.json")
    assert metrics["initial_parameter_state_sha256"] == qm["initial_parameter_state_sha256"]
    for key in ("train_indices_sha256", "validation_indices_sha256", "first_epoch_batch_order_sha256"):
        assert metrics["dataset_provenance"][key] == qm["dataset_provenance"][key]
    assert metrics["dataset_provenance"]["train_batch_order_sha256"][:10] == qm["dataset_provenance"]["train_batch_order_sha256"]
    histories = {name: np.load(run / f"{name}.npy", allow_pickle=False)
                 for name in ("accuracy_train", "accuracy_test", "loss_train", "loss_test")}
    assert all(len(h) == 30 and np.isfinite(h).all() for h in histories.values())
    values = histories["accuracy_test"]
    assert np.allclose(values, [e["metrics"]["validation_accuracy"] for e in epochs], rtol=0, atol=1e-12)
    import torch
    for role in ("best", "final"):
        ck = torch.load(run / f"{role}_model.pt", map_location="cpu", weights_only=True)
        with np.load(run / f"weights_{role}.npz", allow_pickle=False) as weights:
            for spec, tensor in zip(ck["schema"], ck["states"], strict=True):
                name = spec["name"].strip()
                assert tensor.dtype == torch.float64 and torch.isfinite(tensor).all()
                assert np.array_equal(tensor.numpy(), weights[name])
                if name.startswith("Bias_"):
                    assert torch.count_nonzero(tensor) == 0
                else:
                    assert ((tensor >= 0) & (tensor <= 100)).all()
    drawdown = 100 * (np.maximum.accumulate(values) - values)
    row.update(final_validation_pct=100*float(values[-1]), best_validation_pct=100*float(values.max()),
               best_to_final_drop_pp=100*float(values.max()-values[-1]),
               max_running_best_drawdown_pp=float(drawdown.max()),
               max_running_best_drawdown_epoch=int(drawdown.argmax()) + 1,
               passes_final_drop_screen=bool(100*float(values.max()-values[-1]) <
                                            cfg["stability_pilot"]["stable_final_drop_strictly_less_than_pp"]),
               dataset_provenance=metrics["dataset_provenance"],
               initial_parameter_state_sha256=metrics["initial_parameter_state_sha256"], gpu_environment=gpu)
    return row


def analyze():
    cases = read(STUDY / "cases.json")
    expected = len(cases)
    assert expected in (21, 24)
    assert len({c["row_id"] for c in cases}) == expected
    assert len({(c["gpu"], c["index"]) for c in cases}) == expected
    gpu_classes = [g for g in ("V100", "A100", "RTX3090", "RTX5090") if any(c["gpu"] == g for c in cases)]
    for scheme in ("baseline", "ours", "legacy"):
        coverage = (("V100", [0, 1e-5, 3e-5]), ("A100", [0, 1e-4, 3e-4, 5e-4])) if expected == 21 else (
            ("V100", [0, 1e-5]), ("A100", [0, 1e-4, 3e-4, 5e-4]),
            ("RTX5090", [0, 3e-5]))
        for gpu, sigmas in coverage:
            assert sorted(c["sigma"] for c in cases if c["scheme"] == scheme and c["gpu"] == gpu) == sigmas
    rows = [collect_case(c) for c in cases]
    by_name = {(r["gpu"], r["case"]): r for r in rows}
    for row in rows:
        clean = by_name[(row["gpu"], row["clean_reference_case"])]
        if row["state"] == clean["state"] == "complete":
            assert row["dataset_provenance"] == clean["dataset_provenance"]
            assert row["initial_parameter_state_sha256"] == clean["initial_parameter_state_sha256"]
            # CUDA/PyTorch must agree; V100 memory size may differ.
            for key in ("torch", "cuda"):
                assert row["gpu_environment"][key] == clean["gpu_environment"][key]
            row["final_drop_pp"] = clean["final_validation_pct"] - row["final_validation_pct"]
            row["best_drop_pp"] = clean["best_validation_pct"] - row["best_validation_pct"]
    completed = sum(r["state"] == "complete" for r in rows)
    terminal = sum(r["state"] in ("complete", "failed", "startup-failed") for r in rows)
    history = historical_rows()
    payload = dict(updated_at=datetime.now(timezone.utc).isoformat(), expected=expected,
                   completed=completed, terminal=terminal, rows=rows, historical_rows=history)
    (STUDY / "analysis.json").write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    fields = ["gpu", "target", "comparison_role", "scheme", "beta", "sigma", "state", "epochs_completed", "final_validation_pct",
              "best_validation_pct", "final_drop_pp", "best_drop_pp", "best_to_final_drop_pp",
              "passes_final_drop_screen", "max_running_best_drawdown_pp",
              "max_running_best_drawdown_epoch", "bundle", "failure"]
    with REPORT.with_suffix(".csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)
    text = ["# Conv3 p90 beta: thirty-epoch read-noise sweep", "",
            f"Updated {payload['updated_at']}. **{completed}/{expected} complete; {terminal}/{expected} terminal.**", "",
            "Seed 0, ordinary MNIST with a deterministic 55,000/5,000 train/validation split, T=K=8, float64 centered EqProp. "
            "All cases train from fresh matched initialization for 30 epochs. Jean Zay uses three exact ten-epoch "
            "continuations; local workers retain their in-memory training state, including scheduling pauses. "
            "Official test remains unread; these are validation diagnostics.", "",
            "Injected beta is fixed from the zero-noise per-matrix cosine >.90 calibration. "
            "Gaussian noise perturbs copied positive/negative non-input endpoint voltages "
            "only during gradient readout; relaxation and validation are noise-free. "
            "Each scheme/GPU pair uses its own matched clean control. "
            "Positive drop means lower validation accuracy with noise. Failures remain listed. "
            + ("V100 covers sigma0/1e-5/3e-5; A100 covers0/1e-4/3e-4/5e-4." if expected == 21 else
               "After the user-requested local acceleration, V100 covers sigma0/1e-5, RTX5090 covers "
               "0/3e-5, and A100 covers0/1e-4/3e-4/5e-4. "
               "Matched controls share GPU/PyTorch/CUDA class; physical hosts are recorded in the CSV. "
               "There are 15 noisy runs and nine matched clean controls. "
               "Fifi/Riri clean workers were temporarily suspended in memory to reduce GPU contention "
               "and resumed automatically after their noisy peers finished, preserving their live training states. "
               "Matched-control drops remain blank until both outcomes finish."), "",
            "| GPU | Scheme | Injected beta | Noise sigma | State / epochs | Final / best validation (%) | Final drop (pp) |",
            "|---|---|---:|---:|---|---:|---:|"]
    for r in sorted(rows, key=lambda x: (x["scheme"], x["sigma"])):
        acc = f"{r['final_validation_pct']:.2f} / {r['best_validation_pct']:.2f}" if r["state"] == "complete" else "—"
        drop = f"{r['final_drop_pp']:+.2f}" if "final_drop_pp" in r else "—"
        text.append(f"| {r['gpu']} | {r['scheme']} | {r['beta']:.6g} | {r['sigma']:g} | {r['state']} / {r['epochs_completed']} | {acc} | {drop} |")
    screened = [r for r in rows if r["state"] == "complete"]
    failed_screen = [r for r in screened if not r["passes_final_drop_screen"]]
    text += ["", "Completion means all thirty finite epochs were collected and validated. "
             "The separate predeclared stability screen requires epoch30 accuracy to finish "
             "strictly less than 5 percentage points below that run's own best; "
             "passing this screen does not establish robustness to noise.", "",
             f"Final-drop screen: {len(screened)-len(failed_screen)}/{len(screened)} completed runs pass. "
             "The CSV records each run's best-to-final drop."]
    if failed_screen:
        text.append("")
    for row in failed_screen:
        text.append(f"- {row['gpu']} {row['scheme']}, σ={row['sigma']:g}: "
                    f"{row['best_to_final_drop_pp']:.2f}pp best-to-final drop; retained as an outcome.")
    if screened:
        largest = max(screened, key=lambda r: r["max_running_best_drawdown_pp"])
        text += ["", "A final-epoch screen can miss temporary deterioration. "
                 "The CSV also records the largest decline from any earlier validation best "
                 "within each completed run, and the first epoch attaining it. "
                 f"The largest observed decline is {largest['max_running_best_drawdown_pp']:.2f}pp "
                 f"for {largest['scheme']} on {largest['gpu']} at σ={largest['sigma']:g}, "
                 f"epoch {largest['max_running_best_drawdown_epoch']}; "
                 f"its final best-to-final drop is {largest['best_to_final_drop_pp']:.2f}pp. "
                 "This is a descriptive trajectory diagnostic, not a new selection or exclusion rule."]
    text += ["", "Previous thirty-epoch validation curves use baseline beta 10, ours beta 3 and legacy beta 0.001. "
             "They ran on the recorded RTX 3080/3090/5090 environments. Differences against the new "
             "curves combine the beta change with environment changes; they are not a controlled beta-only effect.", "",
             "| Scheme | Old injected beta | Sigma | Old final validation (%) | Old drop (pp) | New minus old final (pp) |",
             "|---|---:|---:|---:|---:|---:|"]
    for old in sorted(history, key=lambda r: (r["scheme"], r["sigma"])):
        new = next(r for r in rows if r["scheme"] == old["scheme"] and r["sigma"] == old["sigma"]
                   and r.get("comparison_role") != "additional")
        delta = f"{new['final_validation_pct'] - old['final_validation_pct']:+.2f}" if new["state"] == "complete" else "—"
        text.append(f"| {old['scheme']} | {old['beta']:g} | {old['sigma']:g} | {old['final_validation_pct']:.2f} | {old['final_drop_pp']:.2f} | {delta} |")
    text += ["", "Single-seed differences are descriptive. The prior ten-epoch clean pilots qualify "
             "admission only; they are not controls for this full-horizon sweep.", "",
             "[Epoch-by-epoch validation curves](conv3_p90_read_noise_20260919_epochs.png) "
             + ("cover all thirty epochs for every declared case. " if completed == expected else
                "include partial runs, shown as dashed lines ending at their latest collected epoch. ") +
             "The tables above use completed thirty-epoch outcomes only.", "",
             "[Plan](../docs/eqprop_conv3_p90_read_noise_plan_20260919.md) · "
             "[CSV](conv3_p90_read_noise_20260919.csv) · "
             "[Local evidence](../results/eqprop-conv3-p90-read-noise-20260919-v1/)"]
    REPORT.write_text("\n".join(text) + "\n")
    if completed:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
        for scheme in ("baseline", "ours", "legacy"):
            for gpu in gpu_classes:
                subset = sorted((r for r in rows if r["scheme"] == scheme and r["gpu"] == gpu and r["state"] == "complete"), key=lambda r: r["sigma"])
                axes[0].plot([r["sigma"] for r in subset], [r["final_validation_pct"] for r in subset], "o-", label=f"{scheme} {gpu}")
                subset = [r for r in subset if "final_drop_pp" in r]
                axes[1].plot([r["sigma"] for r in subset], [r["final_drop_pp"] for r in subset], "o-", label=f"{scheme} {gpu}")
        for ax, ylabel in zip(axes, ("Epoch30 validation accuracy (%)", "Drop from matched clean (pp)")):
            ax.set_xscale("symlog", linthresh=1e-5); ax.set(xlabel="Endpoint read-noise sigma", ylabel=ylabel)
            ax.grid(alpha=.2); ax.legend()
        fig.savefig(REPORT.with_suffix(".png"), dpi=170)
        fig.savefig(REPORT.with_suffix(".pdf")); plt.close(fig)
        fig, axes = plt.subplots(3, len(gpu_classes), figsize=(6*len(gpu_classes), 10), constrained_layout=True,
                                 sharex=True, sharey=True)
        colors = dict(zip((0, 1e-5, 3e-5, 1e-4, 3e-4, 5e-4),
                          ("black", "tab:blue", "tab:cyan", "tab:orange", "tab:red", "tab:purple")))
        for i, scheme in enumerate(("baseline", "legacy", "ours")):
            for j, gpu in enumerate(gpu_classes):
                ax = axes[i, j]
                subset = sorted((r for r in rows if r["scheme"] == scheme and r["gpu"] == gpu
                                 and r["epochs_data"]), key=lambda r: r["sigma"])
                for row in subset:
                    curve = row["epochs_data"]
                    partial = row["state"] != "complete"
                    ax.plot([e["epoch"] for e in curve],
                            [100*e["metrics"]["validation_accuracy"] for e in curve],
                            "--" if partial else "-", color=colors[row["sigma"]], marker=".", markersize=3,
                            label=f"σ={row['sigma']:g}" + (" (partial)" if partial else ""))
                ax.set(title=f"{scheme} · {gpu}", xlim=(1, 30),
                       xlabel="Epoch", ylabel="Validation accuracy (%)")
                ax.grid(alpha=.2)
                if subset:
                    ax.legend(fontsize=8)
        fig.suptitle(f"Conv3 p90 read noise · seed 0 · {completed}/{expected} completed · validation only")
        fig.savefig(REPORT.with_name(REPORT.stem + "_epochs.png"), dpi=170)
        fig.savefig(REPORT.with_name(REPORT.stem + "_epochs.pdf")); plt.close(fig)
    print(json.dumps({k: payload[k] for k in ("expected", "completed", "terminal")}))


if __name__ == "__main__":
    analyze()
