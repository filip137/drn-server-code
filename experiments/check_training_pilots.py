"""Validate full-budget EqProp pilots before block or qualified-group replication."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from experiments.reporting import validate_run, sha256_file


def stable_metrics(best, final, epochs, expected_epochs):
    return (epochs == expected_epochs and all(math.isfinite(x) for x in (best, final))
            and 0 <= final <= best <= 1 and best - final < .05 - 1e-12)


def check_pilot(path):
    import numpy as np
    import torch
    path = Path(path)
    errors = validate_run(path)
    if errors:
        raise ValueError(f"Invalid canonical pilot {path}: {errors}")
    config = json.loads((path / "config.used.json").read_text())
    manifest = json.loads((path / "manifest.json").read_text())
    metrics = json.loads((path / "metrics.json").read_text())
    if manifest["smoke"] or config["training_algorithm"] != "EP" or config["seed"] != 0:
        raise ValueError("Expected a full seed-0 EP pilot")
    if metrics["official_test_evaluations"] != 0 or config["evaluation"]["official_test"]["policy"] != "disabled":
        raise ValueError("Official test must remain unread")
    epochs = config["lab"]["epochs"]
    depth = len(config["model_overrides"][config["lab"]["model_key"]]["conv_pipeline"])
    expected_epochs = {1: 10, 2: 30, 3: 30}[depth]
    if epochs != expected_epochs:
        raise ValueError("Pilot does not use the complete architecture epoch budget")
    for name in ("loss_train", "loss_test", "accuracy_train", "accuracy_test"):
        history = np.load(path / f"{name}.npy", allow_pickle=False)
        if len(history) != epochs or not np.isfinite(history).all():
            raise ValueError("Incomplete or nonfinite full-budget history")
    for name in ("best_model.pt", "final_model.pt"):
        checkpoint = torch.load(path / name, map_location="cpu", weights_only=True)
        for spec, tensor in zip(checkpoint["schema"], checkpoint["states"], strict=True):
            if tensor.dtype != torch.float64 or not torch.isfinite(tensor).all():
                raise ValueError("Pilot checkpoint precision/finiteness failure")
            if spec["name"].strip().startswith("Bias_") and torch.count_nonzero(tensor):
                raise ValueError("Pilot bias is nonzero")
    best, final = metrics["best_validation_accuracy"], metrics["final_validation_accuracy"]
    return {"path": str(path.resolve()), "result_sha256": sha256_file(path / "result.json"),
        "epochs": epochs, "best_validation_accuracy": best, "final_validation_accuracy": final,
        "drop_percentage_points": 100*(best-final), "stable": stable_metrics(best, final, epochs, expected_epochs),
        "architecture": f"conv{depth}",
        "voltage_amp": config["model_base"]["voltage_amp"],
        "current_amp": config["model_base"]["current_amp"],
        "weight_max": config["model_base"]["weight_max"], "base_beta": config["beta"],
        "T": config["model_base"]["num_iterations_inference"],
        "K": config["model_base"]["num_iterations_training"]}


def check_block(paths, bounded):
    rows = [check_pilot(path) for path in paths]
    expected = {(f"conv{arch}", av, ai, ceiling)
        for arch in (1,2,3) for av, ai in ((1.,1.),(4.,1.),(4.,.25))
        for ceiling in ((1e-4,5e-4,1e-3) if bounded else (100.,))}
    keys = [(r["architecture"], r["voltage_amp"], r["current_amp"], r["weight_max"]) for r in rows]
    if len(keys) != len(expected) or set(keys) != expected:
        raise ValueError("Pilot block has missing, duplicate, or unexpected conditions")
    if not all(row["stable"] for row in rows):
        raise ValueError("At least one pilot fails the strict <5 pp stability rule")
    if bounded:
        for arch, av, ai, _ in expected:
            betas = {r["base_beta"] for r in rows if (r["architecture"], r["voltage_amp"], r["current_amp"]) == (arch,av,ai)}
            if len(betas) != 1:
                raise ValueError("Bounded ceilings do not share beta")
    return {"stable": True, "bounded": bounded, "pilots": rows,
            "budget_review_required": bounded, "official_test_read": False}


def validate_bounded_groups(rows):
    """Require all three stable, same-beta ceilings for every included group."""
    allowed = {(f"conv{a}", av, ai) for a in (1, 2, 3)
               for av, ai in ((1., 1.), (4., 1.), (4., .25))}
    keys = [(r["architecture"], r["voltage_amp"], r["current_amp"], r["weight_max"])
            for r in rows]
    if not rows or len(keys) != len(set(keys)):
        raise ValueError("Bounded pilot groups are empty or duplicated")
    groups = {key[:3] for key in keys}
    if not groups <= allowed:
        raise ValueError("Unexpected bounded pilot group")
    for group in groups:
        selected = [r for r, key in zip(rows, keys, strict=True) if key[:3] == group]
        if {r["weight_max"] for r in selected} != {1e-4, 5e-4, 1e-3}:
            raise ValueError("Bounded pilot group is missing a ceiling")
        if not all(r["stable"] is True for r in selected):
            raise ValueError("Bounded pilot group fails the stability rule")
        betas = {r["base_beta"] for r in selected}
        if len(betas) != 1 or not all(math.isfinite(b) and b > 0 for b in betas):
            raise ValueError("Bounded pilot group does not share one finite positive beta")


def check_bounded_groups(paths):
    rows = [check_pilot(path) for path in paths]
    validate_bounded_groups(rows)
    return {"stable": True, "bounded": True, "scope": "complete_bounded_groups",
            "pilots": rows, "official_test_read": False}


def validate_bounded_replication(config, gate):
    """Enforce Filip's September 12 release after a group's three pilots pass."""
    if (config["completion_plan"]["qualification"] != "bounded_beta_qualified"
            or config["training_algorithm"] != "EP" or config["seed"] not in (1, 2)):
        raise ValueError("Expected a numerically qualified bounded EP replication")
    if (gate.get("scope") != "complete_bounded_groups" or gate.get("stable") is not True
            or gate.get("bounded") is not True or gate.get("official_test_read") is not False):
        raise ValueError("Missing qualified-group pilot gate")
    validate_bounded_groups(gate["pilots"])
    depth = len(config["model_overrides"][config["lab"]["model_key"]]["conv_pipeline"])
    if config["lab"]["epochs"] != {1: 10, 2: 30, 3: 30}[depth]:
        raise ValueError("Replication epoch budget differs from the pilot contract")
    group = (f"conv{depth}", config["model_base"]["voltage_amp"], config["model_base"]["current_amp"])
    rows = [r for r in gate["pilots"]
            if (r["architecture"], r["voltage_amp"], r["current_amp"]) == group]
    if len(rows) != 3:
        raise ValueError("This bounded group has no complete stable pilot coverage")
    if config["model_base"]["weight_max"] not in {r["weight_max"] for r in rows}:
        raise ValueError("Replication ceiling differs from the pilot contract")
    if any(r["base_beta"] != config["beta"] for r in rows):
        raise ValueError("Replication beta differs from the stable pilot beta")
    if config["completion_plan"].get("operating_point_revision"):
        expected_tk = (config["model_base"]["num_iterations_inference"],
                       config["model_base"]["num_iterations_training"])
        if any((r.get("T"), r.get("K")) != expected_tk for r in rows):
            raise ValueError("Revised T/K differs from the complete stable pilot group")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--bounded", action="store_true")
    parser.add_argument("--bounded-groups", action="store_true",
                        help="Validate complete architecture/scheme groups for the approved seed-1/2 release")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.bounded and args.bounded_groups:
        parser.error("Choose a whole bounded block or complete bounded groups")
    result = check_bounded_groups(args.paths) if args.bounded_groups else check_block(args.paths, args.bounded)
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Validated {len(result['pilots'])} stable pilots")
