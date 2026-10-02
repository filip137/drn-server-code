"""Freeze the nine seed-zero clean EqProp checkpoints for voltage-only replay."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
STUDY = "conv123-clean-eqprop-trained-voltage-20260928-v1"
CONFIGS = Path("configs/conv/clean_eqprop_trained_voltage_20260928")
SOURCES = ("conv1-clean-and-baseline-noise-20260927-v1",
           "conv23-noiseless-three-seed-20260927-v1",
           "conv23-noiseless-baseline-three-seed-20260927-v1")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path):
    return json.loads(path.read_text())


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def main():
    study = ROOT / "results" / STUDY
    prior = ROOT / "results/conv123-reduced-current-layerwise-20260928-v1/analysis/state_profiles.csv"
    with prior.open() as handle:
        profiles = list(csv.DictReader(handle))
    inventory = {}
    for name in SOURCES:
        with (ROOT / "results" / name / "analysis/accuracy.csv").open() as handle:
            for row in csv.DictReader(handle):
                if row["seed"] == "0" and float(row.get("eta", 0)) == 0:
                    architecture = row.get("architecture", "conv1")
                    key = (architecture, row["scheme"])
                    assert key not in inventory, f"Duplicate source: {key}"
                    inventory[key] = row
    assert len(inventory) == 9
    receipts = []
    for depth, seconds in ((1, 600), (2, 900), (3, 1200)):
        architecture = f"conv{depth}"
        reference = read_json(ROOT / f"configs/conv/reduced_current_layerwise_20260928/{architecture}.json")
        cases = []
        for scheme in ("baseline", "ours", "legacy"):
            row = inventory[(architecture, scheme)]
            source = Path(row["path"])
            result = read_json(source / "result.json")
            manifest = read_json(source / "manifest.json")
            status = read_json(source / "status.json")
            config = read_json(source / "config.used.json")
            assert status["state"] == "complete" and result["completion"]["criteria_met"]
            assert result["manifest_sha256"] == digest(source / "manifest.json")
            assert not manifest.get("official_test_read", False)
            assert config["training_algorithm"] == "EP" and config["runtime_dtype"] == "float64"
            assert config["seed"] == 0 and config["eqprop"]["variant"] == "centered"
            assert config["eqprop"]["endpoint_read_noise_eta"] == config["eqprop"]["endpoint_read_noise_std"] == 0
            epoch = 10 if depth == 1 else 30
            assert config["lab"]["epochs"] == epoch
            index = [json.loads(line) for line in (source / "epoch_checkpoint_index.jsonl").read_text().splitlines()]
            assert [r["epoch"] for r in index] == list(range(epoch + 1))
            indexed = {r["epoch"]: r for r in index}
            selected = {"source_config.json": source / "config.used.json",
                        "initialization.pt": source / indexed[0]["model_path"],
                        "final.pt": source / "final_model.pt"}
            hashes = {}
            destination = study / "inputs" / f"{architecture}_{scheme}"
            destination.mkdir(parents=True, exist_ok=True)
            for name, path in selected.items():
                actual = digest(path)
                artifacts = [a for a in result["artifacts"] if (source / a["path"]).resolve() == path.resolve()]
                assert len(artifacts) == 1 and artifacts[0]["sha256"] == actual, (path, artifacts)
                target = destination / name
                if target.exists():
                    assert digest(target) == actual, f"Refusing to overwrite changed input {target}"
                else:
                    shutil.copy2(path, target)
                hashes[name] = actual
            assert hashes["initialization.pt"] == indexed[0]["model_sha256"]
            receipt = dict(architecture=architecture, scheme=scheme, source_bundle=str(source.relative_to(ROOT)),
                           source_result_sha256=digest(source / "result.json"), source_manifest_sha256=digest(source / "manifest.json"),
                           checkpoint_epoch=epoch, source_hashes=hashes, seed=0, training_algorithm="EP",
                           training_noise_eta=0, checkpoint_selection="fixed_final_epoch", source_final_validation=float(row[f"epoch{epoch}"]))
            receipts.append(receipt)
            rel = destination.relative_to(ROOT)
            for role, selected_epoch in (("initialization", 0), ("final", epoch)):
                filename = "initialization.pt" if selected_epoch == 0 else "final.pt"
                case = dict(name=f"{architecture}_{scheme}_{role}", architecture=architecture, scheme=scheme,
                            source_config=str(rel / "source_config.json"), source_config_sha256=hashes["source_config.json"],
                            checkpoint_path=str(rel / filename), checkpoint_sha256=hashes[filename], checkpoint_epoch=selected_epoch,
                            checkpoint_role=role, initialization_checkpoint_path=str(rel / "initialization.pt"),
                            initialization_checkpoint_sha256=hashes["initialization.pt"],
                            injected_beta=config["eqprop"]["injected_beta_B"], T=2*depth+2, K=2*depth+2,
                            input_gain={1:40, 2:100, 3:360}[depth], source_provenance=receipt)
                if selected_epoch == 0:
                    selected_rows = sorted((r for r in profiles if r["architecture"] == architecture and r["scheme"] == scheme),
                                           key=lambda r: int(r["layer_index"]))
                    assert len(selected_rows) == depth+1
                    case["expected_initial_free_rms"] = [float(r["F"]) for r in selected_rows]
                cases.append(case)
        cfg = dict(study_id=STUDY, architecture=architecture, target="akib", budget_seconds=seconds,
                   source_contract=reference["source_contract"],
                   analyzer_source_root=reference["analyzer_source_root"], cohort_config=reference["cohort_config"],
                   expected_cohort_sha256=reference["expected_cohort_sha256"], cuda_allocator_cap_bytes=reference["cuda_allocator_cap_bytes"],
                   output_root=f"results/{STUDY}/production/{architecture}", cases=cases)
        write_json(ROOT / CONFIGS / f"{architecture}.json", cfg)
    write_json(study / "inputs/source_inventory.json", dict(cases=receipts, source_count=9, replay_contexts=18,
                                                           prior_state_profiles_sha256=digest(prior)))
    print(f"Prepared nine sources / 18 replay contexts under {study}")


if __name__ == "__main__":
    main()
