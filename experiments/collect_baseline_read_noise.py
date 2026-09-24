"""Validate and collect the declared baseline read-noise training curves."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil

import numpy as np
import torch

from experiments.collect_eqprop_read_noise import config_matches_recorded_transport
from experiments.collect_paper_training_completion import tree_hashes, validate_initialization
from experiments.reporting import sha256_file, validate_run

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "results/eqprop-read-noise-seed0-baseline-20260916-v1"
PAPER = ROOT / "paper_ready_results"


def read(path):
    return json.loads(Path(path).read_text())


def csv_write(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def validate(path, row):
    pack = STUDY / row["pack_directory"]
    receipt = read(pack / "pack_status.json")
    case = next(c for c in receipt["cases"] if c["case"] == row["case"])
    if case["returncode"] != 0:
        raise ValueError("Successful process receipt required")
    summaries = read(pack / f"{row['case']}.summary.json")
    if len(summaries) != 1 or summaries[0]["status"] != "complete":
        raise ValueError("Exact-run semantic completion required")
    errors = validate_run(path)
    if errors:
        raise ValueError(f"Invalid canonical bundle: {errors}")
    cfg, metrics, manifest, result = (read(path / name) for name in
        ("config.used.json", "metrics.json", "manifest.json", "result.json"))
    config_path = STUDY / "source" / row["config"]
    expected = read(config_path)
    if sha256_file(config_path) != row["config_sha256"] or not config_matches_recorded_transport(
        cfg, expected, summaries[0].get("dataset_root_override")
    ):
        raise ValueError("Scientific configuration changed")
    if manifest["smoke"] or result["smoke"] or not result["completion"]["criteria_met"]:
        raise ValueError("Incomplete/smoke result cannot count as full training")
    if manifest["runtime"]["target"] != row["target"]:
        raise ValueError("Target differs from declared placement")
    identity = (STUDY / "source/SOURCE_ARCHIVE_SHA256").read_text().strip()
    if manifest["git"]["source_archive_sha256"] != identity:
        raise ValueError("Scientific source identity differs")
    epochs = int(row["epochs"])
    events = [json.loads(x) for x in (path / "metrics.jsonl").read_text().splitlines()]
    if [e["epoch"] for e in events] != list(range(1, epochs + 1)):
        raise ValueError("Missing full epoch coverage")
    if metrics["official_test_evaluations"] != 0 or result["dataset"]["official_test_read"]:
        raise ValueError("Unexpected official-test access")
    if cfg["max_batches"] is not None or cfg["max_test_batches"] is not None:
        raise ValueError("Shortened data budget")
    validate_initialization(cfg, metrics, "EP")
    parent = ROOT / cfg["read_noise_study"]["parent_config"]
    for p, key in ((parent, "parent_config_sha256"),
                   (parent.with_name("result.json"), "parent_result_sha256")):
        if sha256_file(p) != cfg["read_noise_study"][key]:
            raise ValueError("Historical reference bytes changed")
    parent_metrics = read(parent.with_name("metrics.json"))
    if metrics["dataset_provenance"] != parent_metrics["dataset_provenance"]:
        raise ValueError("Dataset cohort/minibatch order differs from parent")
    if metrics["initial_parameter_state_sha256"] != parent_metrics["initial_parameter_state_sha256"]:
        raise ValueError("Initial parameters differ from parent")
    sigma = float(row["sigma"])
    draws = 2 * (int(row["architecture"][-1]) + 1) * 3438 * epochs if sigma else 0
    if metrics["eqprop_endpoint_read_noise_draw_count"] != draws:
        raise ValueError("Unexpected endpoint-noise draw count")
    if (metrics["eqprop"]["endpoint_read_noise_std"] != sigma
            or metrics["eqprop"]["endpoint_read_noise_seed"] != 2026081601
            or cfg["beta"] != float(row["beta"])
            or cfg["eqprop"]["injected_beta_B"] != float(row["beta"])):
        raise ValueError("Wrong beta/noise contract")
    for name in ("accuracy_train", "accuracy_test", "loss_train", "loss_test"):
        history = np.load(path / f"{name}.npy", allow_pickle=False)
        if len(history) != epochs or not np.isfinite(history).all():
            raise ValueError("Incomplete or nonfinite history")
    for role in ("best", "final"):
        checkpoint = torch.load(path / f"{role}_model.pt", map_location="cpu", weights_only=True)
        with np.load(path / f"weights_{role}.npz", allow_pickle=False) as npz:
            for spec, tensor in zip(checkpoint["schema"], checkpoint["states"], strict=True):
                name = spec["name"].strip()
                if tensor.dtype != torch.float64 or not torch.isfinite(tensor).all():
                    raise ValueError("Nonfinite or non-float64 checkpoint")
                if not np.array_equal(tensor.numpy(), npz[name]):
                    raise ValueError("PT/NPZ checkpoint mismatch")
                if name.startswith("Bias_"):
                    if torch.count_nonzero(tensor):
                        raise ValueError("Nonzero frozen bias")
                elif not ((tensor >= 0).all() and (tensor <= 100).all()):
                    raise ValueError("Conductance projection violation")
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revalidate", action="store_true",
                        help="Recheck all previously collected source and destination bundles")
    args = parser.parse_args()
    rows = list(csv.DictReader((STUDY / "run_plan.csv").open()))
    if len(rows) != 22 or len({r["case"] for r in rows}) != 22:
        raise ValueError("Declared coverage must contain exactly 22 unique cases")
    now = datetime.now(timezone.utc).isoformat()
    proof_path = PAPER / "provenance/baseline_read_noise_collection_20260916.json"
    proofs = read(proof_path) if proof_path.exists() else {}
    for row in rows:
        for key in ("collected_result", "best_validation_percent", "final_validation_percent",
                    "best_drop_pp", "final_drop_pp", "completed_epochs"):
            row.setdefault(key, "")
        row["noise_stream_group"] = {
            "local": "rtx3090_torch2.5.1_cuda12.1",
            "nom-cool-1": "rtx3090_torch2.5.1_cuda12.1",
            "akibscomputer": "rtx3080_torch2.5.1_cuda12.1",
            "fifi": "rtx5090_torch2.11_cuda12.8",
        }[row["target"]]
        pack = STUDY / row["pack_directory"]
        case_root = pack / row["case"]
        receipt = read(pack / "pack_status.json") if (pack / "pack_status.json").exists() else None
        case = next((c for c in receipt["cases"] if c["case"] == row["case"]), None) if receipt else None
        results = list(case_root.rglob("result.json")) if case_root.exists() else []
        if row["case"] not in proofs:
            statuses = list(case_root.rglob("status.json")) if case_root.exists() else []
            if statuses:
                if len(statuses) != 1:
                    raise ValueError("Ambiguous training status")
                status = read(statuses[0])
                row["status"] = status["state"]
                metric_path = statuses[0].with_name("metrics.jsonl")
                if metric_path.exists():
                    row["completed_epochs"] = str(len(metric_path.read_text().splitlines()))
            elif case:
                row["status"] = "running" if case["returncode"] is None else "failed"
            if not results or not case or case["returncode"] is None:
                continue
            if len(results) != 1:
                raise ValueError("Ambiguous result directory")
            path = results[0].parent
            metrics = validate(path, row)
            relative = (Path("bundles/baseline_read_noise_20260916") / row["architecture"]
                        / f"beta_{float(row['beta']):g}" / f"sigma_{float(row['sigma']):g}" / "seed0")
            destination = PAPER / relative
            hashes = tree_hashes(path)
            if destination.exists():
                if tree_hashes(destination) != hashes:
                    raise FileExistsError(f"Different existing evidence: {destination}")
            else:
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copytree(path, destination, symlinks=True)
            if tree_hashes(destination) != hashes or validate_run(destination):
                raise ValueError("Collected copy failed validation")
            proofs[row["case"]] = dict(collected_at=now, source=str(path.relative_to(ROOT)),
                collected_result=str(relative / "result.json"), file_and_link_hashes=hashes,
                canonical_valid=True, full_epochs_and_noise_draw_count=True,
                matched_initializer_cohort_order=True, finite_float64_zero_bias_bounded_pt_npz_equal=True,
                noise_stream_group=row["noise_stream_group"], official_test_read=False)
        proof = proofs[row["case"]]
        if args.revalidate:
            source = ROOT / proof["source"]
            destination = (PAPER / proof["collected_result"]).parent
            validate(source, row)
            validate(destination, row)
            if any(tree_hashes(path) != proof["file_and_link_hashes"]
                   for path in (source, destination)):
                raise ValueError(f"Collected evidence changed: {row['case']}")
            proof["revalidated_at"] = now
        metrics = read((PAPER / proof["collected_result"]).with_name("metrics.json"))
        row.update(status="complete_collected_validated", collected_result=proof["collected_result"],
            completed_epochs=row["epochs"],
            best_validation_percent=f"{100 * metrics['best_validation_accuracy']:.2f}",
            final_validation_percent=f"{100 * metrics['final_validation_accuracy']:.2f}")

    for row in rows:
        if row["status"] != "complete_collected_validated":
            continue
        if row["beta"] == "100":
            clean = ROOT / "paper_ready_results/bundles/table2_wide_ep" / row["architecture"] / "baseline/seed0/metrics.json"
        else:
            match = next(r for r in rows if r["architecture"] == row["architecture"]
                         and r["beta"] == row["beta"] and float(r["sigma"]) == 0)
            if not match["collected_result"]:
                continue
            clean = (PAPER / match["collected_result"]).with_name("metrics.json")
        control = read(clean)
        row["best_drop_pp"] = f"{100 * control['best_validation_accuracy'] - float(row['best_validation_percent']):.2f}"
        row["final_drop_pp"] = f"{100 * control['final_validation_accuracy'] - float(row['final_validation_percent']):.2f}"
    proof_path.parent.mkdir(parents=True, exist_ok=True)
    proof_path.write_text(json.dumps(proofs, indent=2, sort_keys=True) + "\n")
    for path in (STUDY / "run_plan.csv", PAPER / "baseline_read_noise_run_status_20260916.csv"):
        csv_write(path, rows)
    complete = sum(r["status"] == "complete_collected_validated" for r in rows)
    table = ["| Model | Beta | Sigma | Host | State | Epochs | Best val. % | Final val. % | Final clean loss, pp |",
             "|---|---:|---:|---|---|---:|---:|---:|---:|"]
    for r in rows:
        table.append(f"| {r['architecture']} | {r['beta']} | {float(r['sigma']):g} | {r['target']} | {r['status']} | "
                     f"{r['completed_epochs'] or '0'}/{r['epochs']} | {r['best_validation_percent'] or '—'} | "
                     f"{r['final_validation_percent'] or '—'} | {r['final_drop_pp'] or '—'} |")
    report = ("# Baseline EqProp read-noise training progress\n\n"
              f"Updated {now}. **{complete}/22 new trainings collected and validated.** "
              "The declared scope is 20 noisy runs plus two new clean controls, seed 0 only.\n\n"
              "T=K=4/6/8. Conv1 beta100/200; Conv2 beta100; Conv3 beta10. "
              "Conv3 beta10 remains unconfirmed across seeds and retains its equilibrium caveat. "
              "These are validation diagnostics; no official-test evaluation.\n\n"
              + "\n".join(table) + "\n\n"
              "Clean losses use matching-beta controls; a positive loss is degradation. "
              "Conv1/Conv2 beta100 reuse audited historical controls. Conv1 beta200 and "
              "Conv3 beta10 use new controls when complete. GPU/software noise-stream groups "
              "remain separate: shared seeds do not imply identical noise across these environments. "
              "The Conv1 beta comparison also crosses host/noise-stream groups.\n\n"
              "[Launch plan](../docs/eqprop_baseline_read_noise_launch_plan_20260916.md) · "
              "[Per-run CSV](baseline_read_noise_run_status_20260916.csv) · "
              "[Collection proof](provenance/baseline_read_noise_collection_20260916.json)\n")
    (PAPER / "baseline_read_noise_run_status_20260916.md").write_text(report)
    print(f"Collected and validated {complete}/22; "
          f"running={sum(r['status'] == 'running' for r in rows)}, "
          f"failed={sum(r['status'] == 'failed' for r in rows)}")


if __name__ == "__main__":
    main()
