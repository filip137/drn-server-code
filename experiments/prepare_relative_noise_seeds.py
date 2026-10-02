"""Prepare exp022's fixed-contract seed replications; never launch training."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
STUDY_ID = "conv123-relative-noise-seeds12-jeanzay-20260930-v1"
STUDY = ROOT / "results" / STUDY_ID
CONFIGS = ROOT / "configs/conv/relative_noise_seeds12_20260930"
ETAS = (("1em5", 1e-5), ("1em4", 1e-4), ("3em4", 3e-4), ("5em4", 5e-4))
SCHEMES = ("baseline", "ours", "legacy")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_path(depth, scheme, seed):
    target = {1: 1, 2: 4, 3: 6}[depth]
    folder = ("conv1_clean_baseline_noise_20260927" if depth == 1 else
              "noiseless_baseline_three_seed_20260927" if scheme == "baseline" else
              "noiseless_three_seed_20260927")
    return ROOT / f"configs/conv/{folder}/conv{depth}_{scheme}_relative{target}_eta0_seed{seed}.json"


def make_config(depth, scheme, seed, tag, eta):
    source = source_path(depth, scheme, seed)
    template = json.loads(source.read_text())
    config = copy.deepcopy(template)
    target = {1: 1, 2: 4, 3: 6}[depth]
    epochs = 10 if depth == 1 else 30
    arm = f"conv{depth}_{scheme}_relative{target}_eta{tag}_seed{seed}"
    config.update(study_id=STUDY_ID, arm_id=arm)
    config["reporting"].update(study_id=STUDY_ID, arm_id=arm)
    config["eqprop"].update(endpoint_read_noise_eta=eta, endpoint_read_noise_seed=2026081601 + seed)
    config["relative_noise_training"].update(eta=eta, primary_metric=f"epoch_{epochs}_validation_accuracy")
    config["relative_noise_training"].pop("tests_and_canaries_skipped_by_user", None)
    for key in ("noiseless_replication", "conv1_control_contract", "execution_waiver", "canary_waiver"):
        config.pop(key, None)
    config["evaluation"]["inclusion_rule"] = (
        "Retain all outcomes. Primary comparison uses paired seeds 1/2 at the fixed final epoch; "
        "historical seed0 optimizer/RNG-reset trajectories remain separate.")
    config["read_noise_seed_replication"] = dict(
        source_config=str(source.relative_to(ROOT)), source_config_sha256=sha(source),
        model_seed=seed, shuffle_seed=seed, split_seed=0,
        acquisition_seed=2026081601 + seed, eta=eta, uninterrupted_epochs=epochs,
        optimizer_reset_at_epoch10=False, rng_reset_at_epoch10=False,
        checkpoint_recovery="full_state_epoch_boundary_only", fixed_seed0_beta=True,
        displacement_label=f"seed0-reference D/F={target}",
    )
    assert config["seed"] == config["initialization"]["seed"] == seed
    assert config["lab"]["epochs"] == epochs
    params = config["datasets"]["mnist"]["params"]
    assert (params["shuffle_seed"], params["split_seed"], params["batch_size"], params["validation_batch_size"]) == (seed, 0, 16, 64)
    assert config["runtime_dtype"] == "float64"
    assert config["eqprop"]["variant"] == "centered"
    assert config["eqprop"]["endpoint_read_noise_mode"] == "nodewise_relative"
    assert config["eqprop"]["endpoint_read_noise_std"] == config["eqprop"]["endpoint_read_noise_floor"] == 0
    assert config["evaluation"]["official_test"]["policy"] == "disabled"
    assert config["model_base"]["num_iterations_inference"] == config["model_base"]["num_iterations_training"] == {1: 4, 2: 6, 3: 8}[depth]
    assert isinstance(config["model_base"]["quadratic_diode_param"], dict)
    assert all(v == 0 for k, v in config["learning_rates_by_parameter"].items() if k.startswith("Bias"))
    return config


def main():
    CONFIGS.mkdir(parents=True, exist_ok=True)
    records, assets = [], []
    for depth in (1, 2, 3):
        hashes = []
        for seed in (1, 2):
            original = ROOT / f"results/paper-training-completion-20260911-v1/assets/wide_kaiming/conv{depth}/seed{seed}.pt"
            digest = sha(original)
            hashes.append(digest)
            staged = f"inputs/initializers/conv{depth}_seed{seed}.pt"
            checkpoint = STUDY / staged
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(original, checkpoint)
            assets.append(dict(depth=depth, seed=seed, source=str(original.relative_to(ROOT)),
                               checkpoint=str(checkpoint.relative_to(ROOT)), staged_checkpoint=staged,
                               checkpoint_sha256=digest))
            for scheme in SCHEMES:
                for tag, eta in ETAS:
                    config = make_config(depth, scheme, seed, tag, eta)
                    assert config["initialization"]["checkpoint_sha256"] == digest
                    assert config["init_checkpoint_path"] == staged
                    dest = CONFIGS / f"{config['arm_id']}.json"
                    contents = json.dumps(config, indent=2, sort_keys=True) + "\n"
                    if dest.exists() and dest.read_text() != contents:
                        raise ValueError(f"Refusing to overwrite changed config {dest}")
                    dest.write_text(contents)
                    records.append(dict(arm_id=config["arm_id"], config=str(dest.relative_to(ROOT)),
                        config_sha256=sha(dest), checkpoint_sha256=digest, staged_checkpoint=staged,
                        seed=seed, depth=depth, scheme=scheme, eta=eta, epochs=config["lab"]["epochs"],
                        account="fmu@v100" if seed == 1 else "umg@v100",
                        wall_seconds={1: 1800, 2: 10800, 3: 21600}[depth]))
        assert len(set(hashes)) == 2, "Seed initializers must differ"
    assert len(records) == len({r["arm_id"] for r in records}) == 72
    report = dict(study_id=STUDY_ID, expected_cases=72, ready_cases=72,
                  total_epochs=sum(r["epochs"] for r in records), assets=assets, records=records)
    (STUDY / "preparation.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(dict(configs=len(records), assets=len(assets), epochs=report["total_epochs"])))


if __name__ == "__main__":
    main()
