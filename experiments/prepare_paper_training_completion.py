"""Materialize the September paper backlog as ordinary exact-run configs.

No jobs are launched. Unqualified bounded EP configs are written separately
under candidates/ and are never included in runnable lists.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
STUDY = "paper-training-completion-20260911-v1"
CONFIG_ROOT = Path("configs/conv/paper_training_completion_20260911_v1")
RESULT_ROOT = Path("results") / STUDY
BETAS = {"conv1": [100., 30., 3.], "conv2": [100., 10., .03],
         "conv3": [100., 3., .001]}
SCHEMES = ("baseline", "ours", "legacy")
CEILINGS = {1e-4: "1em4", 5e-4: "5em4", 1e-3: "1em3"}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    content = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if path.exists():
        if path.read_text() != content:
            raise FileExistsError(f"Refusing to replace different content: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def build_energy(config):
    # Identical constructor and RNG order to mnist_train, before data creation.
    from labs.mnist_train import _set_seed, _reset_name_counters, FlexibleDeepResistiveEnergy
    _set_seed(config["seed"])
    _reset_name_counters()
    model = {**config["model_base"],
             **config["model_overrides"][config["lab"]["model_key"]]}
    return FlexibleDeepResistiveEnergy(
        layer_shapes=[tuple(s) for s in model["layer_shapes"]],
        conv_pipeline=model.get("conv_pipeline") or [],
        pooling_mode=model.get("pooling_mode"),
        **{key: model[key] for key in (
            "weight_gains", "input_gain", "non_linearity", "exponential_diode_param",
            "quadratic_diode_param", "hard_sigmoid_param", "voltage_amp", "current_amp",
            "weight_min", "weight_max")},
        weight_init_mode=model.get("weight_init_mode", "kaiming_uniform"),
        input_mode=config.get("input_mode", "train"),
        trainable_amplification=bool(model.get("trainable_amplification", False)),
        amplification_min=model.get("amplification_min", 1e-6),
        amplification_max=model.get("amplification_max"),
    )


def state_hash(energy):
    from labs.mnist_train import _param_schema, _parameter_state_sha256
    return _parameter_state_sha256(_param_schema(energy.params()))


def prepare_assets(root):
    import torch
    assets = {}
    for architecture in BETAS:
        for bounded in (False, True):
            family = "bounded_uniform" if bounded else "wide_kaiming"
            reference = root / "paper_ready_results/bundles" / (
                f"table3_bounded_bptt/{architecture}/baseline/gmax_1em4/seed0/config.used.json"
                if bounded else f"table1_wide_bptt/{architecture}/baseline/seed0/config.used.json")
            parent = json.loads(reference.read_text())
            hashes = []
            for seed in range(3):
                config = copy.deepcopy(parent)
                config["seed"] = seed
                if bounded:
                    config["model_base"].update(weight_min=1e-5, weight_max=1e-4)
                energy = build_energy(config)
                path = RESULT_ROOT / "assets" / family / architecture / f"seed{seed}.pt"
                destination = root / path
                destination.parent.mkdir(parents=True, exist_ok=True)
                if bounded and seed == 0:
                    original = root / f"paper_ready_results/assets/bounded_uniform/{architecture}/seed0/final_model.pt"
                    if destination.exists() and digest(destination) != digest(original):
                        raise ValueError("Preserved initializer changed")
                    if not destination.exists():
                        shutil.copy2(original, destination)
                    energy.load(destination)
                elif not destination.exists():
                    energy.save(destination)
                else:
                    expected = state_hash(energy)
                    energy.load(destination)
                    if state_hash(energy) != expected:
                        raise ValueError("Saved initializer differs from normal seed draw")
                tensor_hash = state_hash(energy)
                hashes.append(tensor_hash)
                for parameter in energy.params():
                    state = parameter.state
                    if state.dtype != torch.float32 or not torch.isfinite(state).all():
                        raise ValueError("Initializer precision/finiteness failure")
                    if parameter.__class__.__name__ == "Bias":
                        if torch.count_nonzero(state):
                            raise ValueError("Initializer bias is nonzero")
                    elif bounded and not ((state >= 1e-5).all() and (state < 1e-4).all()):
                        raise ValueError("Initializer escaped fixed uniform support")
                # Amplification must not change the normal numerical initializer.
                if not (bounded and seed == 0):
                    for av, ai in ((4., 1.), (4., .25)):
                        config["model_base"].update(voltage_amp=av, current_amp=ai)
                        if state_hash(build_energy(config)) != tensor_hash:
                            raise ValueError("Scheme-dependent initialization")
                assets[f"{family}/{architecture}/seed{seed}"] = {
                    "checkpoint_path": str(path), "checkpoint_sha256": digest(destination),
                    "float32_parameter_state_sha256": tensor_hash,
                    "recipe": "preserved source seed0" if bounded and seed == 0 else family,
                    "seed": seed, "ep_conversion": "load float32 then cast every runtime variable to float64",
                    "optimizer_state": "fresh; no optimizer state loaded",
                }
            if len(set(hashes)) != 3:
                raise ValueError("Independent seeds have identical initialization")
    return assets


def resolved_config(row, parent, asset, ep_template):
    config = copy.deepcopy(parent)
    seed = int(row["model_seed"])
    config["seed"] = seed
    config["datasets"]["mnist"]["params"].update(shuffle_seed=seed, split_seed=0,
        root="/home/filip/datasets/mnist", download=False)
    config["lab"]["epochs"] = int(row["epochs"])
    config["max_batches"] = config["max_test_batches"] = None
    config["evaluation"] = {"checkpoint_selection": "maximum_validation_accuracy",
        "epoch_split": "validation", "official_test": {"policy": "disabled"}}
    config["init_checkpoint_path"] = asset["checkpoint_path"]
    config["initialization"] = copy.deepcopy(asset)
    if "weight_ceiling_sweep" in config:
        config["weight_ceiling_sweep"].update(
            initializer_checkpoint_path=asset["checkpoint_path"],
            initializer_checkpoint_sha256=asset["checkpoint_sha256"])
    config["study_id"] = STUDY
    config["arm_id"] = row["cell_id"]
    config["reporting"] = {"study_id": STUDY, "arm_id": row["cell_id"],
        "evidence_class": "ordinary_mnist_training_completion", "paper_facing": False}
    config["training_algorithm"] = "EP" if row["algorithm"] == "EP" else "BP"
    config["runtime_dtype"] = "float64" if row["algorithm"] == "EP" else "float32"
    if row["algorithm"] == "EP":
        config["eqprop"] = copy.deepcopy(ep_template["eqprop"])
        arch, scheme = row["architecture"], row["scheme"]
        beta = BETAS[arch][SCHEMES.index(scheme)]
        model = config["model_base"]
        amplification = (model["voltage_amp"] / model["current_amp"]) ** int(arch[-1])
        config["beta"] = beta / amplification
        config["eqprop"].update(injected_beta_B=beta, amplification_factor=amplification,
            endpoint_read_noise_std=0., input_read_noise=False,
            beta_tier="bounded_candidate_unqualified" if row["block"] == "T3_EP" else "one_decade_lower")
    config["completion_plan"] = {
        "cell_id": row["cell_id"], "block": row["block"],
        "parent_config": row["template_or_parent_config"],
        "source_learning_rates_unchanged": True,
        "accepted_T_K": [int(row["T"]), int(row["K"])],
        "qualification": "bounded_beta_pending" if row["block"] == "T3_EP" else "accepted_operating_point",
        "replication_gate": "qualified_group_seed0_pilots" if row["block"] == "T3_EP" and seed else
            "all_9_wide_pilots" if row["block"] == "T2_EP" and seed else "seed0_pilot" if row["algorithm"] == "EP" else "none",
    }
    rates = [config["learning_rates_by_parameter"][name] for name in config["parameter_order"]]
    if rates != parent["lr"] or rates != config["optimizer"]["learning_rate"]:
        raise ValueError("Exact parent learning-rate vector mismatch")
    if any(rate != 0 for name, rate in zip(config["parameter_order"], rates) if name.startswith("Bias_")):
        raise ValueError("Bias learning rates must be zero")
    return config


def prepare(root=ROOT):
    if STUDY not in (root / "docs/current_simulations.md").read_text():
        raise ValueError("Register the planned result directory first")
    assets = prepare_assets(root)
    write_json(root / RESULT_ROOT / "assets/initializers.json", assets)
    rows = list(csv.DictReader((root / "paper_ready_results/remaining_runs.csv").open()))
    ep_template = json.loads((root / "paper_ready_results/bundles/table2_wide_ep/conv1/baseline/seed0/config.used.json").read_text())
    lists = {}
    coverage = []
    for row in rows:
        bounded = row["block"].startswith("T3")
        if row["block"] == "T2_EP":
            source = root / f"paper_ready_results/bundles/table1_wide_bptt/{row['architecture']}/{row['scheme']}/seed0/config.used.json"
        else:
            source = root / row["template_or_parent_config"]
        parent = json.loads(source.read_text())
        family = "bounded_uniform" if bounded else "wide_kaiming"
        asset = assets[f"{family}/{row['architecture']}/seed{row['model_seed']}"]
        config = resolved_config(row, parent, asset, ep_template)
        config["completion_plan"]["resolved_parent_path"] = str(source.relative_to(root))
        config["completion_plan"]["resolved_parent_sha256"] = digest(source)
        branch = "candidates" if row["block"] == "T3_EP" else "training"
        path = CONFIG_ROOT / branch / f"{row['cell_id']}.json"
        write_json(root / path, config)
        stage = ("bounded_ep_held" if row["model_seed"] != "0" else "bounded_ep_pilots") if row["block"] == "T3_EP" else (
            "wide_ep_pilot" if row["model_seed"] == "0" else "wide_ep_replications") if row["block"] == "T2_EP" else (
            "bounded_bptt_seed0" if bounded and row["model_seed"] == "0" else "bounded_bptt" if bounded else "wide_bptt")
        lists.setdefault(f"{stage}_{row['architecture']}", []).append(str(path))
        coverage.append({"cell_id": row["cell_id"], "stage": stage, "config": str(path),
                         "sha256": digest(root / path)})
    for name, paths in lists.items():
        destination = root / CONFIG_ROOT / "lists" / f"{name}.txt"
        destination.parent.mkdir(parents=True, exist_ok=True)
        content = "\n".join(sorted(paths)) + "\n"
        if destination.exists() and destination.read_text() != content:
            raise FileExistsError(destination)
        destination.write_text(content)
    write_json(root / RESULT_ROOT / "preparation.json", {"cells": coverage,
        "counts": {name: len(paths) for name, paths in lists.items()},
        "official_test_read": False})
    print(json.dumps({"configs": len(coverage), "initializers": len(assets),
                      "counts": {name: len(paths) for name, paths in lists.items()}}, indent=2))


if __name__ == "__main__":
    argparse.ArgumentParser(description=__doc__).parse_args()
    prepare()
