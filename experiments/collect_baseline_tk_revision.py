"""Collect the matched baseline revision without replacing native-T/K evidence."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import shutil

from experiments.collect_paper_training_completion import tree_hashes, validate_training
from experiments.reporting import sha256_file

ROOT = Path(__file__).resolve().parents[1]
COLLECTION = ROOT / "paper_ready_results/baseline_tk_revision"
CONFIGS = ROOT / "configs/conv/paper_baseline_tk_revision_20260913_v1/training"
REVISION = "baseline_tk_20260913_v1"


def summarize(rows):
    complete = [r for r in rows if r["collected_result"]]
    original_path = COLLECTION.parent / "run_status.csv"
    with original_path.open() as stream:
        reader = csv.DictReader(stream)
        fields, original = reader.fieldnames, list(reader)
    revised = {r["cell_id"]: r for r in rows}
    contract = []
    for old in original:
        row = dict(revised.get(old["cell_id"], old))
        if row["cell_id"] in revised and row["collected_result"]:
            row["collected_result"] = "baseline_tk_revision/" + row["collected_result"]
        contract.append(row)
    if len(contract) != 216 or len({r["cell_id"] for r in contract}) != 216:
        raise ValueError("Current paper contract must contain exactly 216 distinct cells")
    remaining = [r for r in contract if not r["collected_result"]]
    for name, selected in (("current_contract_run_status.csv", contract),
                           ("current_contract_remaining_runs.csv", remaining)):
        path = COLLECTION.parent / name
        temporary = path.with_suffix(".csv.tmp")
        with temporary.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(selected)
        temporary.replace(path)
    original_count = sum(bool(r["collected_result"]) for r in original)
    current_count = len(contract) - len(remaining)
    summary = {"updated_at": datetime.now(timezone.utc).isoformat(), "required": len(contract),
               "original_collected_retained": original_count, "revised_baseline_required": len(rows),
               "revised_baseline_collected": len(complete), "current_contract_collected": current_count,
               "current_contract_remaining": len(remaining), "official_test_evaluations": 0,
               "native_tk_baseline_bptt_runs_preserved_as_earlier_evidence": 18}
    (COLLECTION.parent / "current_contract_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    missing_text = ["# Remaining training for the current paper contract", "",
                    f"Updated {summary['updated_at']}. **{current_count}/216 training results collected and validated; {len(remaining)} full runs remain.**", "",
                    "This list is regenerated from the current contract after each collection. All remaining cases use 30 epochs. These are training-completion requirements; official-test evaluations remain zero and require the later contract/checkpoint seal.", "",
                    "| Architecture | Algorithm | Scheme | T/K | Missing seeds by weight ceiling | Runs |",
                    "|---|---|---|---:|---|---:|"]
    groups = {}
    for row in remaining:
        key = (row["architecture"], row["algorithm"], row["scheme"], row["T"], row["K"])
        groups.setdefault(key, []).append(row)
    for (architecture, algorithm, scheme, t, k), group in sorted(groups.items()):
        ceilings = {}
        for row in group:
            ceilings.setdefault(float(row["G_max"]), []).append(int(row["model_seed"]))
        seeds = "; ".join(f"{ceiling:g}: {', '.join(map(str, sorted(values)))}"
                          for ceiling, values in sorted(ceilings.items()))
        missing_text.append(f"| {architecture} | {algorithm} | {scheme} | {t}/{k} | {seeds} | {len(group)} |")
    missing_text += ["", "The revised baseline EqProp groups require a common beta to pass on every new seed-0 BPTT reference, followed by all three full seed-0 EqProp pilots passing stability before seeds 1/2. See the [new-reference beta report](baseline_tk_revision/beta_qualification.md) and [revision manifest](baseline_tk_revision/README.md) for gate status.", "",
                     "[Live targets and launch records](../docs/paper_ready_results_manifest.md) · [Per-run CSV](current_contract_remaining_runs.csv) · [Current validation tables](current_contract_validation_tables.md)"]
    (COLLECTION.parent / "current_contract_remaining_runs.md").write_text("\n".join(missing_text) + "\n")
    from experiments.check_training_pilots import stable_metrics
    pilots = [r for r in rows if r["algorithm"] == "EP" and r["model_seed"] == "0"]
    pilot_text = ["# Revised baseline EqProp pilot stability", "",
                  "Only complete, collected and validated 30-epoch seed-0 runs count. Collection verifies finite histories, float64 checkpoints, exact-zero biases, bounds, shared initialization and PT/NPZ equality. Stability additionally requires a best-to-final validation drop strictly below five percentage points. Official-test evaluations remain zero.", "",
                  "| Architecture | Gmax | T/K | Beta | Best/final validation | Drop (pp) | Full pilot | Result |",
                  "|---|---:|---:|---:|---:|---:|---|---|"]
    passed = {"conv2": 0, "conv3": 0}
    for row in pilots:
        if row["collected_result"]:
            best, final = (float(row[key]) for key in
                           ("best_validation_accuracy_percent", "final_validation_accuracy_percent"))
            stable = stable_metrics(best / 100, final / 100, int(row["completed_epochs"]), 30)
            config = json.loads((COLLECTION / row["collected_result"]).with_name("config.used.json").read_text())
            passed[row["architecture"]] += int(stable)
            values = f"{config['beta']:g} | {best:.2f}/{final:.2f}% | {best-final:.2f} | {'pass' if stable else 'fail'} | [bundle]({row['collected_result']})"
        else:
            config = json.loads((ROOT / row["template_or_parent_config"]).read_text())
            beta = (f"{config['beta']:g}" if config["completion_plan"]["qualification"] == "bounded_beta_qualified"
                    else "pending")
            values = f"{beta} | pending | pending | {row['training_status']} | pending"
        pilot_text.append(f"| {row['architecture']} | {float(row['G_max']):g} | {row['T']}/{row['K']} | {values} |")
    pilot_text += ["", f"Full stable pilot coverage: **Conv2 {passed['conv2']}/3; Conv3 {passed['conv3']}/3.** Each architecture needs all three ceilings at the same qualified beta before its seeds 1/2. Passing coverage still requires frozen pilot proofs, exact smokes and budget admission before a repetition launches.", "",
                   "[Numerical beta qualification](beta_qualification.md) · [Training revision ledger](README.md)"]
    (COLLECTION / "pilot_stability.md").write_text("\n".join(pilot_text) + "\n")
    readme = COLLECTION.parent / "README.md"
    content = re.sub(r"\*\*\d+ / 216 results satisfy the revised training contract;\n\d+ full training completions remain\.",
                     f"**{current_count} / 216 results satisfy the revised training contract;\n{len(remaining)} full training completions remain.", readme.read_text())
    content = re.sub(r"All \*\*\d+ original collected bundles\*\*", f"All **{original_count} original collected bundles**", content)
    readme.write_text(content)
    tracker = ROOT / "docs/paper_ready_results_manifest.md"
    content = re.sub(r"\*\*\d+ of 216 results satisfy the revised training contract; \d+ full training\ncompletions remain\. All \d+ original collected bundles are preserved\.",
                     f"**{current_count} of 216 results satisfy the revised training contract; {len(remaining)} full training\ncompletions remain. All {original_count} original collected bundles are preserved.", tracker.read_text())
    tracker.write_text(content)
    text = ["# Matched baseline T/K revision", "",
            f"Updated {datetime.now(timezone.utc).isoformat()}. **{len(complete)}/36 revised training results collected and validated.** Official-test evaluations: zero.", "",
            "Filip authorized revised T/K for both BPTT and EqProp. Conv2 uses T=12, K=6; Conv3 uses T=24, K=8. The three weight ceilings and seeds 0/1/2, exact Adam vectors, initializers, minibatch order and 30-epoch budgets are unchanged.", "",
            "The common candidate beta is 0.1, supported by the [61 completed numerical replays](../baseline_tk_beta_qualification.md). New seed-0 BPTT best checkpoints must pass the same numerical gate. Each architecture then requires all three full seed-0 EqProp pilots to pass before its seeds 1/2 are released.", "",
            "The [new-reference beta report](beta_qualification.md) tracks the collected checkpoint rechecks and distinguishes partial ceiling coverage from a qualified common beta.", "",
            "The [full-pilot stability report](pilot_stability.md) tracks all six revised seed-0 EqProp pilots and the gate before seeds 1/2.", "",
            "The 18 earlier baseline BPTT runs remain in the original collection as native-T/K evidence. They do not satisfy this revision. This directory has its own 36-cell ledger and bundles; it does not overwrite the original 216-cell ledger. Eight original non-baseline repetitions were still missing when this revision started, giving 44 further full training completions in total.", "",
            f"Current campaign coverage is **{current_count}/216**, with **{len(remaining)}** full training completions still missing. See the [current contract ledger](../current_contract_run_status.csv) and [remaining runs](../current_contract_remaining_runs.csv), which overlay this revision onto the unchanged other cells.", "",
            "Preferred resources are local/Main, Akib and nom-cool-1. Loulou is the sole weekday campaign RTX 5090. Packing requires measured throughput improvement; the completed two-EqProp benchmark was slower. Each full wave must fit the unchanged 300 GPU-hour ceiling, with an eight-hour maximum per training case. Explicit shorter operational ceilings and reservations are recorded in the continuing campaign manifest.", "",
            "| Cell | T/K | Status | Target | Best/final validation | Result |",
            "|---|---:|---|---|---:|---|"]
    for r in rows:
        accuracy = (f"{float(r['best_validation_accuracy_percent']):.2f}/{float(r['final_validation_accuracy_percent']):.2f}%"
                    if r["collected_result"] else "pending")
        result = f"[bundle]({r['collected_result']})" if r["collected_result"] else "pending"
        text.append(f"| {r['cell_id']} | {r['T']}/{r['K']} | {r['training_status']} | {r['source_target'] or r['future_target_and_job'] or 'not admitted'} | {accuracy} | {result} |")
    text += ["", "[Machine-readable ledger](run_status.csv) · [Original baseline evidence snapshot](original_native_tk_rows.json) · [Continuing campaign manifest](../../docs/paper_ready_results_manifest.md)"]
    (COLLECTION / "README.md").write_text("\n".join(text) + "\n")
    from experiments.summarize_paper_training_completion import summarize as summarize_tables
    summarize_tables(COLLECTION.parent, current_contract=True)


def collect(paths):
    with (COLLECTION / "run_status.csv").open() as stream:
        reader = csv.DictReader(stream)
        fields, rows = reader.fieldnames, list(reader)
    indexed = {r["cell_id"]: r for r in rows}
    assert len(indexed) == len(rows) == 36
    provenance_path = COLLECTION / "collection_validation.json"
    provenance = json.loads(provenance_path.read_text()) if provenance_path.exists() else {}
    now = datetime.now(timezone.utc).isoformat()
    for path in paths:
        path = path.resolve()
        config = json.loads((path / "config.used.json").read_text())
        if config.get("completion_plan", {}).get("operating_point_revision") != REVISION:
            raise ValueError("Only revised baseline runs belong in this collection")
        cell = config["completion_plan"]["cell_id"]
        row = indexed[cell]
        config, metrics, manifest = validate_training(path, row, expected_config_path=CONFIGS / f"{cell}.json")
        if config["completion_plan"]["accepted_T_K"] != [int(row["T"]), int(row["K"])]:
            raise ValueError("Revision ledger T/K does not match the accepted contract")
        if config["completion_plan"]["qualification"] == "bounded_beta_pending":
            raise ValueError("Unqualified EqProp cannot enter the revised collection")
        asset = (COLLECTION.parent / "assets/bounded_uniform" / row["architecture"]
                 / f"seed{row['model_seed']}" / "final_model.pt")
        if sha256_file(asset) != config["initialization"]["checkpoint_sha256"]:
            raise ValueError("Collected shared initializer differs from the revised training initializer")
        relative = Path("bundles") / cell
        destination = COLLECTION / relative
        hashes = tree_hashes(path)
        if destination.exists():
            if tree_hashes(destination) != hashes:
                raise FileExistsError(f"Different evidence already occupies {destination}")
        else:
            shutil.copytree(path, destination, symlinks=True)
        if tree_hashes(destination) != hashes:
            raise ValueError("Collected revision bundle differs from source")
        validate_training(destination, row, expected_config_path=CONFIGS / f"{cell}.json")
        provenance[cell] = {"source_local_path": str(path), "collected_path": str(relative),
                            "collected_at": now, "file_and_link_hashes": hashes,
                            "canonical_and_semantic_validation_passed": True,
                            "shared_initializer_in_parent_collection": str(asset.relative_to(COLLECTION.parent)),
                            "shared_initializer_sha256": sha256_file(asset),
                            "configuration_sha256": sha256_file(CONFIGS / f"{cell}.json")}
        row.update(training_status="complete_collected_validated", collected_result=str(relative / "result.json"),
                   source_result_sha256=sha256_file(path / "result.json"),
                   best_validation_accuracy_percent=str(100 * metrics["best_validation_accuracy"]),
                   final_validation_accuracy_percent=str(100 * metrics["final_validation_accuracy"]),
                   completed_epochs=str(config["lab"]["epochs"]), source_evidence_class=manifest["evidence_class"],
                   source_target=manifest["runtime"]["target"], source_commit=manifest["git"]["commit"],
                   source_archive_sha256=manifest["git"].get("source_archive_sha256", ""),
                   official_test_read="false", last_checked=now,
                   source_result_or_completion_record=str((path / "result.json").relative_to(ROOT)),
                   paper_status="revised_training_collected_paper_gates_pending")
        print(f"COLLECTED REVISION {cell}")
    if paths:
        provenance_path.write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")
        temporary = COLLECTION / "run_status.csv.tmp"
        with temporary.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        temporary.replace(COLLECTION / "run_status.csv")
    summarize(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", type=Path)
    collect(parser.parse_args().paths)
