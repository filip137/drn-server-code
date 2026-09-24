#!/usr/bin/env python3
"""Validate complete digital-reference coverage and summarize validation accuracy."""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import statistics

import torch

from experiments import reporting
from experiments.train_digital_relu_reference import (
    build_model, cached_data, check_state, evaluate, state_hash,
)


def summarize(root, config_path, replay_device=None, dataset_root=None):
    config = json.loads(config_path.read_text())
    expected = {(d, loss, seed) for d in (1, 2, 3)
                for loss in config["losses"] for seed in config["seeds"]}
    rows, manifests = {}, {}
    for lane, depth in (("local", 1), ("akib", 2), ("trex", 3)):
        paths = sorted((root / lane / "training").glob("*/result.json"))
        for path in paths:
            directory = path.parent
            errors = reporting.validate_run(directory)
            if errors:
                raise RuntimeError((directory, errors))
            result = json.loads(path.read_text())
            manifest = json.loads((directory / "manifest.json").read_text())
            resolved = manifest["config"]
            key = (resolved["depth"], resolved["loss"], resolved["seed"])
            assert key in expected and key not in rows, key
            assert resolved["depth"] == depth and not manifest["smoke"]
            assert result["completion"]["criteria_met"]
            assert manifest["config_sha256"] == reporting.sha256_file(config_path)
            assert manifest["evidence_class"] == "ordinary_mnist_selection"
            assert manifest["dataset"]["official_test_read"] is False
            assert manifest["dataset"]["train_size"] == 55000
            assert manifest["dataset"]["validation_size"] == 5000
            for name, digest in manifest["source_sha256"].items():
                assert reporting.sha256_file(directory / "artifacts/source" / name) == digest
            metrics = [json.loads(line) for line in (directory / "metrics.jsonl").read_text().splitlines()]
            epochs = config["epochs_by_depth"][str(depth)]
            assert len(metrics) == epochs
            assert [m["epoch"] for m in metrics] == list(range(1, epochs + 1))
            assert all(m["train_examples"] == 55000 and m["validation_examples"] == 5000 for m in metrics)
            assert all(m["steps"] == m["epoch"] * 3438 for m in metrics)
            best = max(metrics, key=lambda m: m["validation_accuracy"])
            terminal = result["terminal_metrics"]
            assert terminal["optimizer_steps"] == epochs * 3438
            assert terminal["best_validation_accuracy"] == best["validation_accuracy"]
            assert terminal["best_epoch"] == best["epoch"]
            assert terminal["final_validation_accuracy"] == metrics[-1]["validation_accuracy"]
            assert all(v == (0 if n.endswith("bias") else .001)
                       for n, v in manifest["learning_rates_by_parameter"].items())
            model = build_model(config, depth)
            for name in ("initial_model.pt", "best_model.pt", "final_model.pt"):
                state = torch.load(directory / "checkpoints" / name, map_location="cpu", weights_only=True)
                model.load_state_dict(state, strict=True)
                check_state(model)
                if name == "initial_model.pt":
                    assert state_hash(model) == manifest["initial_state_sha256"]
            optimizer = torch.load(directory / "checkpoints/final_optimizer.pt", map_location="cpu", weights_only=True)
            for state in optimizer["state"].values():
                assert state["step"].item() == epochs * 3438
                assert all(torch.isfinite(v).all() for v in state.values() if torch.is_tensor(v))
            drn_path = Path("paper_ready_results/bundles/table1_wide_bptt") / f"conv{depth}/baseline/seed{key[2]}/metrics.json"
            drn = json.loads(drn_path.read_text())["dataset_provenance"]
            for field in ("train_indices_sha256", "validation_indices_sha256"):
                assert drn[field] == manifest["dataset"][field]
            assert drn["train_batch_order_sha256"] == manifest["dataset"]["epoch_batch_order_sha256"]
            rows[key] = {
                "architecture": f"Conv{depth}", "depth": depth, "loss": key[1], "seed": key[2],
                "best_validation_percent": terminal["best_validation_accuracy"] * 100,
                "final_validation_percent": terminal["final_validation_accuracy"] * 100,
                "best_epoch": terminal["best_epoch"], "epochs": epochs,
                "optimizer_steps": terminal["optimizer_steps"],
                "parameter_count": manifest["parameter_count"],
                "elapsed_seconds": terminal["elapsed_seconds"],
                "target": manifest["runtime"]["target"],
                "result_path": str(path), "result_sha256": reporting.sha256_file(path),
            }
            manifests[key] = manifest
    assert set(rows) == expected, f"Missing cases: {expected - set(rows)}"
    for depth in (1, 2, 3):
        for seed in config["seeds"]:
            left, right = (manifests[depth, loss, seed] for loss in config["losses"])
            for field in ("initial_state_sha256", "dataset", "environment", "source_sha256"):
                assert left[field] == right[field], (depth, seed, field)
    dataset_records = [m["dataset"] for m in manifests.values()]
    for field in ("train_indices_sha256", "validation_indices_sha256", "raw_training_files"):
        assert all(d[field] == dataset_records[0][field] for d in dataset_records)
    for seed in config["seeds"]:
        orders = [manifests[d, "mse", seed]["dataset"]["epoch_batch_order_sha256"] for d in (1, 2, 3)]
        assert orders[0] == orders[1][:10] == orders[2][:10]
        assert orders[1] == orders[2]
    summary = []
    for depth in (1, 2, 3):
        for loss in config["losses"]:
            selected = [rows[depth, loss, seed] for seed in config["seeds"]]
            best = [r["best_validation_percent"] for r in selected]
            final = [r["final_validation_percent"] for r in selected]
            summary.append({"architecture": f"Conv{depth}", "loss": loss, "n": 3,
                            "best_mean_percent": statistics.mean(best),
                            "best_sample_sd_percent": statistics.stdev(best),
                            "final_mean_percent": statistics.mean(final),
                            "final_sample_sd_percent": statistics.stdev(final),
                            "parameter_count": selected[0]["parameter_count"]})
    analysis = root / "analysis"
    analysis.mkdir(exist_ok=True)
    if replay_device:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
        images, labels, _, validation = cached_data(config, dataset_root, replay_device)
        batches = torch.tensor(validation, device=replay_device).split(config["validation_batch_size"])
        replays = []
        for key, row in sorted(rows.items()):
            model = build_model(config, key[0]).to(replay_device)
            directory = Path(row["result_path"]).parent
            model.load_state_dict(torch.load(directory / "checkpoints/best_model.pt",
                                            map_location=replay_device, weights_only=True))
            measurement = evaluate(model, images, labels, batches, key[1])
            delta = measurement["accuracy"] * 100 - row["best_validation_percent"]
            replays.append({"run": str(directory), "validation_percent": measurement["accuracy"] * 100,
                            "difference_percentage_points": delta})
        reporting.atomic_write_json(analysis / "checkpoint_replay.json", {
            "device": replay_device, "torch": torch.__version__, "official_test_read": False,
            "records": replays,
            "all_accuracies_reproduced": all(r["difference_percentage_points"] == 0 for r in replays),
        })
        assert all(r["difference_percentage_points"] == 0 for r in replays), "Checkpoint replay differs; inspect checkpoint_replay.json"
    for name, values in (("per_seed.csv", [rows[k] for k in sorted(rows)]), ("summary.csv", summary)):
        with (analysis / name).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(values[0]))
            writer.writeheader()
            writer.writerows(values)
    reporting.atomic_write_json(analysis / "summary.json", {
        "study_id": config["study_id"], "coverage": "18/18", "official_test_read": False,
        "config_sha256": reporting.sha256_file(config_path), "summary": summary,
        "runs": [rows[k] for k in sorted(rows)],
        "training_gpu_hours": sum(r["elapsed_seconds"] for r in rows.values()) / 3600,
        "checks": ["canonical bundle hashes", "complete predeclared coverage", "full epoch/step counts",
                   "finite initial/best/final checkpoints", "exact-zero biases",
                   "finite Adam state and exact parameter step counts",
                   "split and all epoch orders match corresponding DRN table runs",
                   "matched loss-pair initialization, environment, source, split and order",
                   "all-depth matching dataset files, split and shared epoch-order prefix"],
    })
    print(json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--replay-device")
    parser.add_argument("--dataset-root", type=Path)
    args = parser.parse_args()
    if args.replay_device and args.dataset_root is None:
        parser.error("--replay-device requires --dataset-root")
    summarize(args.root, args.config, args.replay_device, args.dataset_root)


if __name__ == "__main__":
    main()
