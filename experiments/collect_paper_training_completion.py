"""Validate completed backlog bundles, copy them, and refresh the paper CSV ledger."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shutil

from experiments.prepare_paper_training_completion import CEILINGS
from experiments.reporting import sha256_file, validate_run

ROOT = Path(__file__).resolve().parents[1]


def validate_initialization(config, metrics, algorithm):
    import torch
    from types import SimpleNamespace
    from labs.mnist_train import _param_schema, _parameter_state_sha256
    asset = ROOT / config["initialization"]["checkpoint_path"]
    if sha256_file(asset) != config["initialization"]["checkpoint_sha256"]:
        raise ValueError("Initializer asset hash mismatch")
    checkpoint = torch.load(asset, map_location="cpu", weights_only=True)
    dtype = torch.float64 if algorithm == "EP" else torch.float32
    params = [SimpleNamespace(name=spec["name"], state=tensor.to(dtype))
              for spec, tensor in zip(checkpoint["schema"], checkpoint["states"], strict=True)]
    expected = _parameter_state_sha256(_param_schema(params))
    if metrics["initial_parameter_state_sha256"] != expected:
        raise ValueError("Recorded initial state differs from the shared initializer")


def tree_hashes(root):
    output = {}
    for path in sorted(Path(root).rglob("*")):
        relative = str(path.relative_to(root))
        if path.is_symlink():
            target = os.readlink(path)
            if not path.resolve().is_relative_to(Path(root).resolve()):
                raise ValueError(f"Bundle link escapes its directory: {path}")
            output[relative] = {"symlink": target}
        elif path.is_file():
            output[relative] = {"sha256": sha256_file(path)}
    return output


def validate_training(path, row, *, expected_config_path=None):
    import numpy as np
    import torch
    errors = validate_run(path)
    if errors:
        raise ValueError(f"Invalid run {path}: {errors}")
    config = json.loads((path / "config.used.json").read_text())
    metrics = json.loads((path / "metrics.json").read_text())
    manifest = json.loads((path / "manifest.json").read_text())
    if manifest["smoke"] or config["completion_plan"]["cell_id"] != row["cell_id"]:
        raise ValueError("Unexpected training cell")
    expected_path = (Path(expected_config_path) if expected_config_path is not None else
                     ROOT / "configs/conv/paper_training_completion_20260911_v1/training" / f"{row['cell_id']}.json")
    expected = json.loads(expected_path.read_text())
    expected["datasets"]["mnist"]["params"]["root"] = config["datasets"]["mnist"]["params"]["root"]
    if config != expected:
        raise ValueError("Collected scientific config differs from the frozen training config")
    if config["seed"] != int(row["model_seed"]) or config["lab"]["epochs"] != int(row["epochs"]):
        raise ValueError("Seed or full epoch budget mismatch")
    if metrics["official_test_evaluations"] != 0:
        raise ValueError("Official test read during training")
    validate_initialization(config, metrics, row["algorithm"])
    for name in ("loss_train", "loss_test", "accuracy_train", "accuracy_test"):
        values = np.load(path / f"{name}.npy", allow_pickle=False)
        if len(values) != int(row["epochs"]) or not np.isfinite(values).all():
            raise ValueError("Missing or nonfinite training history")
    expected_dtype = torch.float64 if row["algorithm"] == "EP" else torch.float32
    for role in ("best", "final"):
        checkpoint = torch.load(path / f"{role}_model.pt", map_location="cpu", weights_only=True)
        with np.load(path / f"weights_{role}.npz", allow_pickle=False) as npz:
            for spec, tensor in zip(checkpoint["schema"], checkpoint["states"], strict=True):
                name = spec["name"].strip()
                if tensor.dtype != expected_dtype or not torch.isfinite(tensor).all():
                    raise ValueError("Checkpoint precision/finiteness failure")
                if not np.array_equal(tensor.numpy(), npz[name]):
                    raise ValueError("Checkpoint PT/NPZ mismatch")
                if name.startswith("Bias_"):
                    if torch.count_nonzero(tensor):
                        raise ValueError("Bias is nonzero")
                elif not ((tensor >= float(row["G_min"])).all() and (tensor <= float(row["G_max"])).all()):
                    raise ValueError("Weights escaped projection bounds")
    return config, metrics, manifest


def collect(paths):
    collection = ROOT / "paper_ready_results"
    with (collection / "run_status.csv").open() as stream:
        reader = csv.DictReader(stream); fields = reader.fieldnames; rows = list(reader)
    indexed = {row["cell_id"]: row for row in rows}
    provenance_path = collection / "provenance/training_completion_collection.json"
    provenance = json.loads(provenance_path.read_text()) if provenance_path.exists() else {}
    now = datetime.now(timezone.utc).isoformat()
    for path in paths:
        path = path.resolve()
        config = json.loads((path / "config.used.json").read_text())
        cell = config["completion_plan"]["cell_id"]
        row = indexed[cell]
        config, metrics, manifest = validate_training(path, row)
        block = {"T1_BPTT":"table1_wide_bptt", "T2_EP":"table2_wide_ep", "T3_BPTT":"table3_bounded_bptt", "T3_EP":"table3_bounded_ep"}[row["block"]]
        relative = Path("bundles") / block / row["architecture"] / row["scheme"]
        if row["block"].startswith("T3"):
            relative /= f"gmax_{CEILINGS[float(row['G_max'])]}"
        relative /= f"seed{row['model_seed']}"
        destination = collection / relative
        original_hashes = tree_hashes(path)
        if destination.exists():
            if tree_hashes(destination) != original_hashes:
                raise FileExistsError(f"Different evidence already occupies {destination}")
        else:
            shutil.copytree(path, destination, symlinks=True)
        if tree_hashes(destination) != original_hashes or validate_run(destination):
            raise ValueError("Collected copy failed equality/canonical validation")
        source_asset = ROOT / config["initialization"]["checkpoint_path"]
        if sha256_file(source_asset) != config["initialization"]["checkpoint_sha256"]:
            raise ValueError("Initializer asset hash mismatch")
        family = "bounded_uniform" if row["block"].startswith("T3") else "wide_kaiming"
        asset_dest = collection / "assets" / family / row["architecture"] / f"seed{row['model_seed']}" / "final_model.pt"
        asset_dest.parent.mkdir(parents=True, exist_ok=True)
        if asset_dest.exists() and sha256_file(asset_dest) != sha256_file(source_asset):
            raise ValueError("A different initializer is already collected")
        if not asset_dest.exists():
            shutil.copy2(source_asset, asset_dest)
        provenance[cell] = {"source_local_path": str(path), "collected_path": str(relative),
            "collected_at": now, "file_and_link_hashes": original_hashes,
            "canonical_valid": True, "checkpoints_finite_zero_bias_bounded_and_npz_equal": True,
            "recorded_initial_state_matches_shared_asset": True,
            "initialization_asset": str(asset_dest.relative_to(collection)),
            "initialization_sha256": sha256_file(asset_dest)}
        row.update(training_status="complete_collected_validated", collected_result=str(relative / "result.json"),
            source_result_sha256=sha256_file(path / "result.json"),
            best_validation_accuracy_percent=str(100*metrics["best_validation_accuracy"]),
            final_validation_accuracy_percent=str(100*metrics["final_validation_accuracy"]),
            completed_epochs=str(config["lab"]["epochs"]), source_evidence_class=manifest["evidence_class"],
            source_target=manifest["runtime"]["target"], source_commit=manifest["git"]["commit"],
            source_archive_sha256=manifest["git"].get("source_archive_sha256", ""),
            official_test_read="false", last_checked=now,
            source_result_or_completion_record=str((path / "result.json").relative_to(ROOT)),
            paper_status="training_collected_paper_gates_pending")
        print(f"COLLECTED {cell}: {row['best_validation_accuracy_percent']}% best validation")
    for name, selected in (("run_status.csv", rows), ("remaining_runs.csv", [r for r in rows if not r["collected_result"]])):
        temporary = collection / f"{name}.tmp"
        with temporary.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields);writer.writeheader();writer.writerows(selected)
        temporary.replace(collection / name)
    provenance_path.write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")
    collected = [r for r in rows if r["collected_result"]]
    remaining = [r for r in rows if not r["collected_result"]]
    with (collection / "collected_results.csv").open() as stream:
        collected_fields = csv.DictReader(stream).fieldnames
    with (collection / "collected_results.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=collected_fields, extrasaction="ignore")
        writer.writeheader(); writer.writerows(collected)
    summary_path = collection / "collection_summary.json"
    summary = json.loads(summary_path.read_text())
    summary.update(updated=now, collected_validated_training_runs=len(collected),
        remaining_training_runs=len(remaining), remaining_training_epochs=sum(int(r["epochs"]) for r in remaining),
        training_launched_by_completion_campaign=True)
    for block, counts in summary["blocks"].items():
        counts["collected"] = sum(r["block"] == block for r in collected)
        counts["training_remaining"] = sum(r["block"] == block for r in remaining)
    summary["remaining_by_architecture"] = {arch:sum(r["architecture"] == arch for r in remaining) for arch in ("conv1","conv2","conv3")}
    # Preserve the original collection-size audit as an explicit dated snapshot.
    original_fields = ("bundle_regular_files", "bundle_symlinks", "bundle_bytes", "checkpoint_tensor_audits_passed")
    if "original_collection_20260911" not in summary:
        summary["original_collection_20260911"] = {key:summary.pop(key) for key in original_fields}
    summary["new_campaign_bundles_collected"] = len(provenance)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    table = ["# Newly completed training results", "", f"Updated {now}. Validation only; official test remains unread.", "",
             "| Cell | Best validation | Final validation | Result |", "|---|---:|---:|---|"]
    for row in collected:
        if row["cell_id"] in provenance:
            table.append(f"| {row['cell_id']} | {float(row['best_validation_accuracy_percent']):.2f}% | {float(row['final_validation_accuracy_percent']):.2f}% | [bundle]({row['collected_result']}) |")
    (collection / "training_completion_results.md").write_text("\n".join(table) + "\n")
    tracker_path = ROOT / "docs/paper_ready_results_manifest.md"
    tracker = tracker_path.read_text()
    tracker = re.sub(r"^Updated: \d{4}-\d{2}-\d{2}", f"Updated: {now[:10]}", tracker, flags=re.M)
    tracker = re.sub(r"\*\*\d+ of 216 training results are collected and validated; \d+ full training\nruns remain\. No official-test results are available yet\.\*\*",
        f"**{len(collected)} of 216 training results are collected and validated; {len(remaining)} full training\nruns remain. No official-test results are available yet.**", tracker)
    labels = {"T1_BPTT":"Table 1: wide BPTT", "T2_EP":"Table 2: wide centered EP", "T3_BPTT":"Table 3: bounded BPTT", "T3_EP":"Table 3: bounded centered EP"}
    for block, label in labels.items():
        count = summary["blocks"][block]
        tracker = re.sub(r"^\| " + re.escape(label) + r" \|.*$",
            f"| {label} | {count['required']} | {count['collected']} | {count['training_remaining']} | 0 / {count['required']} |", tracker, flags=re.M)
    tracker = re.sub(r"^\| \*\*Total\*\* \| \*\*216\*\*.*$",
        f"| **Total** | **216** | **{len(collected)}** | **{len(remaining)}** | **0 / 216** |", tracker, flags=re.M)
    tracker_path.write_text(tracker)
    readme_path = collection / "README.md"
    readme = re.sub(r"\*\*\d+ / 216 training runs collected and validated; \d+\ntraining runs remain\.",
        f"**{len(collected)} / 216 training runs collected and validated; {len(remaining)}\ntraining runs remain.", readme_path.read_text())
    readme = re.sub(r"^Updated: \d{4}-\d{2}-\d{2}", f"Updated: {now[:10]}", readme, flags=re.M)
    readme_path.write_text(readme)
    print(f"Coverage {len(collected)}/{len(rows)}; official test remains unread")
    from experiments.summarize_paper_training_completion import summarize
    summarize(collection, ROOT / "results/paper-training-completion-20260911-v1/analysis/validation_tables")
    revision_ledger = collection / "baseline_tk_revision/run_status.csv"
    if revision_ledger.exists():
        from experiments.collect_baseline_tk_revision import summarize as summarize_revision
        with revision_ledger.open() as stream:
            summarize_revision(list(csv.DictReader(stream)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path)
    collect(parser.parse_args().paths)
