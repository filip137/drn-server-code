"""Collect completed new-reference beta checks and report partial/full coverage."""
from __future__ import annotations

import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil

from experiments.collect_paper_training_completion import tree_hashes
from experiments.reporting import sha256_file, validate_run

ROOT = Path(__file__).resolve().parents[1]
CHECKS = ROOT / "results/paper-training-completion-20260911-v1/checks/baseline_tk_revision_best_gate"
PAPER = ROOT / "paper_ready_results"


def summarize():
    destination_root = PAPER / "provenance/baseline_tk_revision_best_gate"
    rows, provenance = [], []
    for terminal in sorted((CHECKS / "runs").glob("*/result.json")):
        run = terminal.parent
        if validate_run(run):
            raise ValueError(f"Invalid completed replay: {run}")
        manifest = json.loads((run / "manifest.json").read_text())
        config_path = Path(manifest["configuration"]["path"])
        if sha256_file(config_path) != manifest["configuration"]["sha256"]:
            raise ValueError("Replay configuration changed")
        config = json.loads(config_path.read_text())
        if config.get("operating_point_revision") != "baseline_tk_20260913_v1":
            raise ValueError("Wrong operating-point evidence in revised gate collection")
        case = config["cases"][0]
        if not Path(case["run_dir"]).resolve().is_relative_to(PAPER / "baseline_tk_revision/bundles"):
            raise ValueError("Beta gate did not use a new collected baseline reference")
        metrics = json.loads(terminal.read_text())["terminal_metrics"]
        if metrics["official_test_read"] or metrics["optimizer_steps_applied"] or not metrics["source_bytes_unchanged"]:
            raise ValueError("Read-only contract failed")
        destination = destination_root / "runs" / run.name
        hashes = tree_hashes(run)
        if destination.exists():
            if tree_hashes(destination) != hashes:
                raise FileExistsError(f"Different collected evidence exists: {destination}")
        else:
            shutil.copytree(run, destination, symlinks=True)
        if tree_hashes(destination) != hashes or validate_run(destination):
            raise ValueError("Local replay copy failed hash/canonical validation")
        config_copy = destination_root / "configs" / config_path.name
        config_copy.parent.mkdir(parents=True, exist_ok=True)
        if config_copy.exists() and sha256_file(config_copy) != sha256_file(config_path):
            raise FileExistsError("Collected config differs")
        if not config_copy.exists():
            shutil.copy2(config_path, config_copy)
        log = run.parent / f"{run.name}.log"
        log_copy = destination_root / "logs" / log.name
        log_copy.parent.mkdir(parents=True, exist_ok=True)
        if log_copy.exists() and sha256_file(log_copy) != sha256_file(log):
            raise FileExistsError("Collected replay log differs")
        if not log_copy.exists():
            shutil.copy2(log, log_copy)
        provenance.append({"run": str(run.relative_to(ROOT)), "collected_run": str(destination.relative_to(PAPER)),
                           "hashes": hashes, "canonical_valid": True, "smoke": manifest["smoke"],
                           "log_sha256": sha256_file(log_copy),
                           "config_sha256": sha256_file(config_copy)})
        if manifest["smoke"]:
            continue
        architecture = case["architecture"]
        if metrics["replay_count"] != 8 or metrics["layer_comparison_count"] != (24 if architecture == "conv2" else 32):
            raise ValueError("Full replay coverage is incomplete")
        rows.append({"architecture": architecture, "ceiling": run.name.split("_")[1],
                     "T": case["T"], "K": case["K"], "beta": case["injected_beta"],
                     "minimum_cosine": metrics["minimum_cosine"],
                     "maximum_symmetric_norm_delta": metrics["maximum_symmetric_norm_delta"],
                     "residual_gate_passed": metrics["equilibrium_residual_all_passed"],
                     "combined_gate_passed": metrics["unqualified_launch_gate_all_passed"],
                     "collected_result": str((destination / "result.json").relative_to(PAPER))})
    if not rows:
        raise ValueError("No full new-reference checks are complete")
    destination_root.mkdir(parents=True, exist_ok=True)
    (destination_root / "collection_validation.json").write_text(json.dumps(provenance, indent=2) + "\n")
    out = PAPER / "baseline_tk_revision"
    with (out / "beta_qualification.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = ["# Beta recheck on newly trained baseline references", "",
             f"Updated {datetime.now(timezone.utc).isoformat()}. {len(rows)} full ceiling/beta checks completed and collected. Smoke checks are excluded from qualification coverage.", "",
             "Each full check covers initialization and the newly trained seed-0 BPTT best checkpoint on four fixed validation batches. Every weight layer must have cosine ≥ .99 and symmetric norm difference ≤ .10, with every projected-residual p90 ≤ .01. Source files and parameters remain unchanged; official-test evaluations and optimizer steps are zero.", "",
             "| Architecture | Passing common candidates | Coverage and next gate |", "|---|---|---|"]
    for architecture in ("conv2", "conv3"):
        selected = [r for r in rows if r["architecture"] == architecture]
        common = [beta for beta in (.1, .01) if {r["ceiling"] for r in selected if r["beta"] == beta and r["combined_gate_passed"]} == {"1em4", "5em4", "1em3"}]
        lines.append(f"| {architecture} | {', '.join(map(str, common)) or 'none yet'} | " +
                     ("Numerically qualified; all three full seed-0 EqProp pilots still required" if common else
                      f"{len({r['ceiling'] for r in selected})}/3 ceilings checked; hold EqProp launch") + " |")
    lines += ["", "| Architecture | Ceiling | T/K | Beta | Minimum cosine | Maximum norm difference | Residual | Combined gate | Result |",
              "|---|---|---:|---:|---:|---:|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['architecture']} | {r['ceiling']} | {r['T']}/{r['K']} | {r['beta']:g} | {r['minimum_cosine']:.6f} | {r['maximum_symmetric_norm_delta']:.6f} | {'pass' if r['residual_gate_passed'] else 'fail'} | {'pass' if r['combined_gate_passed'] else 'fail'} | [bundle](../{r['collected_result']}) |")
    lines += ["", "[Training revision tracker](README.md) · [All measurements](beta_qualification.csv) · [Earlier T/K diagnostics on native-trained checkpoints](../baseline_tk_beta_qualification.md)"]
    (out / "beta_qualification.md").write_text("\n".join(lines) + "\n")
    print(f"Collected {len(rows)} full beta checks and {len(provenance)-len(rows)} smoke checks")


if __name__ == "__main__":
    summarize()
