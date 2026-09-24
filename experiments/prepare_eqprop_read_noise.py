"""Prepare the five-value, single-seed read-noise sweep without launching it."""
from __future__ import annotations

import copy
import csv
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
STUDY = "eqprop-read-noise-seed0-ours-legacy-20260914-v1"
OUT = ROOT / "results" / STUDY
CONFIGS = Path("configs/conv/eqprop_read_noise_20260914_v1")
SIGMAS = ((1e-5, "1em5"), (3e-5, "3em5"), (1e-4, "1em4"),
          (3e-4, "3em4"), (5e-4, "5em4"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path = Path(path)
    content = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if path.exists() and path.read_text() != content:
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def main():
    from experiments.exact_run import load_exact_config

    assert STUDY in (ROOT / "docs/current_simulations.md").read_text()
    base = ROOT / "results/paper-training-completion-20260911-v1/source-v11"
    # Use the already accepted trainer snapshot, isolating unrelated current work.
    for line in (base / "INPUT_SHA256SUMS").read_text().splitlines():
        digest, name = line.split("  ", 1)
        assert sha(base / name) == digest, name
    source = OUT / "source"
    if source.exists():
        raise FileExistsError(source)
    shutil.copytree(base, source, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    assets = json.loads((ROOT / "results/paper-training-completion-20260911-v1/assets/initializers.json").read_text())
    rows = []
    for depth in (1, 2, 3):
        architecture = f"conv{depth}"
        asset = copy.deepcopy(assets[f"wide_kaiming/{architecture}/seed0"])
        original_asset = ROOT / asset["checkpoint_path"]
        assert sha(original_asset) == asset["checkpoint_sha256"]
        asset["checkpoint_path"] = f"assets/wide_kaiming/{architecture}/seed0.pt"
        destination = source / asset["checkpoint_path"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(original_asset, destination)
        for scheme in ("ours", "legacy"):
            parent_dir = ROOT / f"paper_ready_results/bundles/table2_wide_ep/{architecture}/{scheme}/seed0"
            parent_path = parent_dir / "config.used.json"
            parent = json.loads(parent_path.read_text())
            assert parent["seed"] == 0 and parent["runtime_dtype"] == "float64"
            assert parent["training_algorithm"] == "EP"
            parent_model = {**parent["model_base"], **parent["model_overrides"][parent["lab"]["model_key"]]}
            assert parent_model["weight_min"] == 0 and parent_model["weight_max"] == 100
            assert parent_model["num_iterations_inference"] == (4, 6, 8)[depth-1]
            assert parent_model["num_iterations_training"] == (4, 6, 8)[depth-1]
            for sigma, tag in SIGMAS:
                name = f"RN_{architecture}_{scheme}_sigma{tag}_seed0"
                config = copy.deepcopy(parent)
                config["study_id"] = STUDY
                config["arm_id"] = name
                config["reporting"] = dict(study_id=STUDY, arm_id=name,
                    evidence_class="ordinary_mnist_eqprop_read_noise_training_diagnostic", paper_facing=False)
                config["evaluation"] = dict(checkpoint_selection="maximum_validation_accuracy",
                    epoch_split="validation", official_test={"policy": "disabled"})
                config["max_batches"] = config["max_test_batches"] = None
                config["init_checkpoint_path"] = asset["checkpoint_path"]
                config["initialization"] = copy.deepcopy(asset)
                config["datasets"]["mnist"]["params"].update(root="/home/filip/datasets/mnist",
                    download=False, shuffle_seed=0, split_seed=0)
                config["eqprop"].update(endpoint_read_noise_std=sigma,
                    endpoint_read_noise_seed=2026081601, input_read_noise=False)
                config.pop("completion_plan", None)
                config["read_noise_study"] = dict(
                    architecture=architecture, scheme=scheme, sigma=sigma,
                    model_seed=0, noise_seed=2026081601,
                    clean_parent=str(parent_path.relative_to(ROOT)), clean_parent_sha256=sha(parent_path),
                    clean_result_sha256=sha(parent_dir / "result.json"),
                    current_corrected_physical_kcl=True, beta_unchanged=True,
                    known_clean_gradient_exception=(architecture == "conv3" and scheme == "ours"),
                    qualification="inherited_training_stability; noise robustness diagnostic",
                    official_test_read=False)
                rates = [config["learning_rates_by_parameter"][p] for p in config["parameter_order"]]
                assert rates == parent["lr"] == config["optimizer"]["learning_rate"]
                assert all(v == 0 for p, v in zip(config["parameter_order"], rates) if p.startswith("Bias_"))
                assert config["beta"] == parent["beta"]
                path = CONFIGS / f"{name}.json"
                write(ROOT / path, config)
                load_exact_config(ROOT / path)
                write(source / path, config)
                # Separate short timing configs; full configs remain unchanged.
                timing = copy.deepcopy(config)
                timing["lab"]["epochs"] = 1
                timing["max_batches"] = 256
                timing["max_test_batches"] = 1
                timing["read_noise_study"]["timing_only"] = True
                timing["reporting"]["evidence_class"] = "ordinary_mnist_runtime_diagnostic"
                write(source / "timing_configs" / f"{name}.json", timing)
                rows.append(dict(case=name, architecture=architecture, scheme=scheme,
                    sigma=sigma, seed=0, epochs=config["lab"]["epochs"],
                    injected_beta=config["eqprop"]["injected_beta_B"], base_beta=config["beta"],
                    config=str(path), config_sha256=sha(ROOT / path),
                    clean_result=str((parent_dir / "result.json").relative_to(ROOT)),
                    status="prepared_not_admitted", target=""))
    assert len(rows) == 30
    (source / "SOURCE_PARENT_ARCHIVE_SHA256").write_text((base / "SOURCE_ARCHIVE_SHA256").read_text())
    # The final identity is regenerated after adding the generic pack transport.
    write(OUT / "prepared_cases.json", rows)
    with (OUT / "run_plan.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(dict(configs=len(rows), source=str(source), training_started=False)))


if __name__ == "__main__":
    main()
