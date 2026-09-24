"""Validate locally collected read-noise runs and refresh their paper-folder ledger."""
from __future__ import annotations

import copy
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil

import numpy as np
import torch

from experiments.collect_paper_training_completion import tree_hashes, validate_initialization
from experiments.reporting import sha256_file, validate_run

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "results/eqprop-read-noise-seed0-ours-legacy-20260914-v1"
PAPER = ROOT / "paper_ready_results"


def read(path):
    return json.loads(Path(path).read_text())


def config_matches_recorded_transport(config, expected, dataset_root_override):
    resolved = copy.deepcopy(expected)
    if dataset_root_override is not None:
        key = resolved["lab"]["dataset_key"]
        resolved["datasets"][key]["params"]["root"] = str(dataset_root_override)
    return config == resolved


def validate(path, row):
    pack = STUDY / row.get("pack_directory", f"production/{row['target']}")
    receipt = read(pack / "pack_status.json")
    case_receipt = next(item for item in receipt["cases"] if item["case"] == row["case"])
    if case_receipt["returncode"] != 0:
        raise ValueError("A successful training exit receipt is required")
    summaries = read(pack / f"{row['case']}.summary.json")
    if len(summaries) != 1 or summaries[0]["status"] != "complete":
        raise ValueError("The exact runner did not report successful completion")
    errors = validate_run(path)
    if errors:
        raise ValueError(f"{path}: {errors}")
    cfg, metrics, manifest, result = (read(path / name) for name in
        ("config.used.json", "metrics.json", "manifest.json", "result.json"))
    expected_path = STUDY / "source-v2" / row["config"]
    expected = read(expected_path)
    if sha256_file(expected_path) != row["config_sha256"] or not config_matches_recorded_transport(
            cfg, expected, summaries[0].get("dataset_root_override")):
        raise ValueError(f"Changed scientific config: {row['case']}")
    if manifest["smoke"] or result["smoke"] or not result["completion"]["criteria_met"]:
        raise ValueError("Smoke or incomplete result cannot count as training")
    if manifest["runtime"]["target"] != row["target"]:
        raise ValueError("Training target differs from the declared host")
    if manifest["git"]["source_archive_sha256"] != (STUDY / "source-v2/SOURCE_ARCHIVE_SHA256").read_text().strip():
        raise ValueError("Unexpected source archive")
    epochs = int(row["epochs"])
    lines = [json.loads(line) for line in (path / "metrics.jsonl").read_text().splitlines()]
    if [line["epoch"] for line in lines] != list(range(1, epochs + 1)):
        raise ValueError("Missing full epoch coverage")
    if metrics["official_test_evaluations"] != 0 or result["dataset"]["official_test_read"]:
        raise ValueError("Unexpected official-test read")
    if cfg["max_batches"] is not None or cfg["max_test_batches"] is not None:
        raise ValueError("Shortened data budget")
    init_config = copy.deepcopy(cfg)
    init_config["initialization"]["checkpoint_path"] = str(
        STUDY / "source-v2" / cfg["initialization"]["checkpoint_path"])
    validate_initialization(init_config, metrics, "EP")
    clean = ROOT / cfg["read_noise_study"]["clean_parent"]
    if sha256_file(clean) != cfg["read_noise_study"]["clean_parent_sha256"]:
        raise ValueError("Clean reference config changed")
    if sha256_file(clean.with_name("result.json")) != cfg["read_noise_study"]["clean_result_sha256"]:
        raise ValueError("Clean reference result changed")
    clean_metrics = read(clean.with_name("metrics.json"))
    if metrics["initial_parameter_state_sha256"] != clean_metrics["initial_parameter_state_sha256"]:
        raise ValueError("Noise and clean initial parameters differ")
    if metrics["dataset_provenance"] != clean_metrics["dataset_provenance"]:
        raise ValueError("Noise and clean cohorts or minibatch orders differ")
    depth = int(row["architecture"][-1])
    if metrics["eqprop_endpoint_read_noise_draw_count"] != 2 * (depth + 1) * 3438 * epochs:
        raise ValueError("Unexpected number of endpoint-noise layer draws")
    if metrics["eqprop"]["endpoint_read_noise_std"] != float(row["sigma"]):
        raise ValueError("Wrong measured read noise")
    if metrics["eqprop"]["endpoint_read_noise_seed"] != 2026081601:
        raise ValueError("Wrong noise seed")
    for name in ("accuracy_train", "accuracy_test", "loss_train", "loss_test"):
        history = np.load(path / f"{name}.npy", allow_pickle=False)
        if len(history) != epochs or not np.isfinite(history).all():
            raise ValueError("Short or nonfinite history")
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
                    raise ValueError("Weight projection bounds violated")
    return metrics, clean_metrics


def main():
    rows = list(csv.DictReader((STUDY / "run_plan.csv").open()))
    now = datetime.now(timezone.utc).isoformat()
    provenance_path = PAPER / "provenance/read_noise_collection_20260914.json"
    provenance = read(provenance_path) if provenance_path.exists() else {}
    for row in rows:
        row["noise_stream_group"] = ("rtx3090_torch2.5.1_cuda12.1" if row["target"] in ("local", "nom-cool-1")
            else "rtx3080_torch2.5.1_cuda12.1" if row["target"] == "akibscomputer"
            else "rtx5090_torch2.11_cuda12.8" if row["target"] in ("trex", "fifi", "loulou") else "unassigned")
        row.setdefault("collected_result", "")
        for field in ("best_validation_percent", "final_validation_percent", "best_drop_pp", "final_drop_pp"):
            row.setdefault(field, "")
        pack = STUDY / row.get("pack_directory", f"production/{row['target']}")
        root = pack / row["case"]
        if row["status"] != "complete_collected_validated" and root.exists():
            statuses = list(root.rglob("status.json"))
            if any(read(p).get("state") == "running" for p in statuses):
                row["status"] = "running"
        results = list(root.rglob("result.json")) if row["target"] and root.exists() else []
        if not results:
            continue
        receipt = read(pack / "pack_status.json")
        case_receipt = next(item for item in receipt["cases"] if item["case"] == row["case"])
        if case_receipt["returncode"] is None:
            continue
        if len(results) != 1:
            raise ValueError(f"Ambiguous result for {row['case']}")
        path = results[0].parent
        metrics, clean = validate(path, row)
        relative = Path("bundles/read_noise_20260914") / row["architecture"] / row["scheme"] / f"sigma_{float(row['sigma']):g}" / "seed0"
        destination = PAPER / relative
        hashes = tree_hashes(path)
        if destination.exists():
            if tree_hashes(destination) != hashes:
                raise FileExistsError(f"Different evidence already at {destination}")
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(path, destination, symlinks=True)
        if tree_hashes(destination) != hashes or validate_run(destination):
            raise ValueError("Paper-folder copy failed validation")
        row.update(status="complete_collected_validated", collected_result=str(relative / "result.json"),
            best_validation_percent=f"{100 * metrics['best_validation_accuracy']:.2f}",
            final_validation_percent=f"{100 * metrics['final_validation_accuracy']:.2f}",
            best_drop_pp=f"{100 * (clean['best_validation_accuracy'] - metrics['best_validation_accuracy']):.2f}",
            final_drop_pp=f"{100 * (clean['final_validation_accuracy'] - metrics['final_validation_accuracy']):.2f}")
        if row["case"] not in provenance:
            provenance[row["case"]] = dict(collected_at=now, source=str(path.relative_to(ROOT)),
                collected_result=row["collected_result"], file_and_link_hashes=hashes,
                canonical_valid=True, full_epochs_and_noise_draw_count=True,
                matched_clean_initializer_cohort_and_order=True,
                noise_stream_group=row["noise_stream_group"],
                finite_float64_zero_bias_bounded_pt_npz_equal=True,
                official_test_read=False)
    for path in (STUDY / "run_plan.csv", PAPER / "read_noise_run_status_20260914.csv"):
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
    provenance_path.parent.mkdir(parents=True, exist_ok=True)
    provenance_path.write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")
    complete = sum(row["status"] == "complete_collected_validated" for row in rows)
    running = sum(row["status"] == "running" for row in rows)
    queued = sum(row["status"] in ("queued", "prepared_continuation") for row in rows)
    held = sum(row["status"] == "held_next_window" for row in rows)
    lines = ["# Single-seed EqProp read-noise results and remaining runs", "", f"Updated: {now}.", "",
        f"**{complete}/30 collected and validated; {running} running; {queued} queued/prepared; {held} held.**", "",
        "Wide Conv1/2/3, ours/legacy, seed 0; no new zero-noise or 1e-3 runs. "
        "Ordinary-MNIST validation only; official test disabled. Conv3 ours retains "
        "the recorded clean-gradient qualification exception. The first 12 runs finished "
        "before the September 15 08:00 deadline. The remaining 18 were authorized "
        "on local, Akib, and Nom on September 15; the two queued Conv3 legacy cases "
        "at 3e-5/3e-4 moved to Trex/Fifi for the September 16 08:00 deadline.", "",
        "[Continuation plan](../docs/eqprop_read_noise_continuation_plan_20260915.md) · "
        "[Overnight plan](../docs/eqprop_read_noise_overnight_plan_20260914.md) · "
        "[CSV](read_noise_run_status_20260914.csv)", "",
        "Noise-stream audit: local and Nom reproduce the same sampled arrays. "
        "The 5090 group and Akib's 3080 form two further, distinct groups. Conv1/2 "
        "pairs stay on one host. Conv3 at 1e-5, 1e-4, and 5e-4 compares ours/5090 "
        "with legacy/3090 runs; at 3e-5 and 3e-4 it compares ours/3090 or ours/3080 "
        "with the deadline-transferred legacy/5090 runs. All five Conv3 pairs "
        "therefore differ in GPU/software environment and noise realization. "
        "Historical clean references also use a different GPU/software environment.", "",
        "| Architecture | Sigma | Scheme | Host | State | Best / final validation (%) | Drop from clean, best / final (pp) |",
        "|---|---:|---|---|---|---|---|"]
    for row in sorted(rows, key=lambda x: (x["architecture"], float(x["sigma"]), x["scheme"])):
        values = f"{row['best_validation_percent']} / {row['final_validation_percent']}" if row["collected_result"] else "—"
        drops = f"{row['best_drop_pp']} / {row['final_drop_pp']}" if row["collected_result"] else "—"
        state = row["status"].replace("complete_collected_validated", "complete").replace("held_next_window", "held")
        if row["collected_result"]:
            state = f"[{state}]({row['collected_result']})"
        lines.append(f"| {row['architecture']} | {float(row['sigma']):g} | {row['scheme']} | {row['target'] or 'unassigned'} | {state} | {values} | {drops} |")
    lines += ["", "Positive drops mean lower accuracy than the matching clean seed-0 reference. "
        "Only full, locally validated trainings enter the result table; smoke/timing runs "
        "and deadline-truncated runs are excluded. A single seed measures this trajectory, "
        "not seed-to-seed uncertainty.", ""]
    (PAPER / "read_noise_run_status_20260914.md").write_text("\n".join(lines))
    print(json.dumps(dict(complete=complete, running=running, queued=queued, held=held)))


if __name__ == "__main__":
    main()
