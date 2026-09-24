"""Summarize collected three-seed validation results without opening MNIST."""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
from statistics import mean, stdev


COMPLETE = "complete_collected_validated"
CONTROLS = ("algorithm", "optimizer", "epochs", "T", "K", "G_min",
            "voltage_amp", "current_amp", "split_seed")
BLOCKS = {
    "T1_BPTT": "Table 1: wide BPTT",
    "T2_EP": "Table 2: wide EqProp",
    "T3_BPTT": "Table 3: bounded BPTT",
    "T3_EP": "Table 3: bounded EqProp",
}


def summarize_group(rows):
    seeds = [int(row["model_seed"]) for row in rows]
    if len(rows) != 3 or set(seeds) != {0, 1, 2}:
        raise ValueError("Each planned group must contain seeds 0, 1, 2 exactly once")
    if any(int(row["shuffle_seed"]) != int(row["model_seed"]) for row in rows):
        raise ValueError("Model and shuffle seeds differ")
    if any(len({row[field] for row in rows}) != 1 for field in CONTROLS):
        raise ValueError("Three-seed group contains mixed scientific controls")
    values = {}
    for row in rows:
        if row["training_status"] != COMPLETE:
            continue
        if row["official_test_read"] != "false":
            raise ValueError("Expected validation-only evidence")
        best = float(row["best_validation_accuracy_percent"])
        final = float(row["final_validation_accuracy_percent"])
        if not all(math.isfinite(x) for x in (best, final)) or not 0 <= final <= best <= 100:
            raise ValueError("Invalid validation accuracy")
        values[int(row["model_seed"])] = (best, final)
    first = rows[0]
    output = {field: first[field] for field in ("block", "architecture", "scheme", "G_max", "T", "K")}
    output.update(collected_count=len(values), collected_seeds=",".join(map(str, sorted(values))))
    for role, index in (("best", 0), ("final", 1)):
        scores = [values[seed][index] for seed in sorted(values)]
        output[f"mean_{role}_validation_percent"] = mean(scores) if len(scores) == 3 else None
        output[f"sample_sd_{role}_validation_pp"] = stdev(scores) if len(scores) == 3 else None
        for seed in range(3):
            output[f"seed{seed}_{role}_validation_percent"] = values.get(seed, (None, None))[index]
    return output


def summarize(collection, analysis_dir=None, *, current_contract=False):
    collection = Path(collection)
    prefix_name = "current_contract_" if current_contract else ""
    ledger = collection / f"{prefix_name}run_status.csv"
    summary_name = f"{prefix_name}validation_table_summary.csv"
    table_name = f"{prefix_name}validation_tables.md"
    audit_name = f"{prefix_name}validation_table_aggregation.json"
    seeds_name = "current_contract_run_status.csv" if current_contract else "collected_results.csv"
    remaining_name = f"{prefix_name}remaining_runs.csv"
    with ledger.open() as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 216 or len({row["cell_id"] for row in rows}) != 216:
        raise ValueError("Expected the complete unique 216-cell plan")
    groups = defaultdict(list)
    included = []
    for row in rows:
        key = (row["block"], row["architecture"], row["scheme"], float(row["G_max"]))
        groups[key].append(row)
        if row["training_status"] != COMPLETE:
            continue
        result = collection / row["collected_result"]
        digest = hashlib.sha256(result.read_bytes()).hexdigest()
        if digest != row["source_result_sha256"]:
            raise ValueError(f"Result hash differs from collection ledger: {row['cell_id']}")
        metrics = json.loads(result.with_name("metrics.json").read_text())
        for role in ("best", "final"):
            if not math.isclose(float(row[f"{role}_validation_accuracy_percent"]),
                                100 * metrics[f"{role}_validation_accuracy"], abs_tol=1e-8):
                raise ValueError(f"Ledger metric differs from result: {row['cell_id']}")
        included.append({"cell_id": row["cell_id"], "result": row["collected_result"], "sha256": digest})
    summary = [summarize_group(groups[key]) for key in sorted(groups)]
    if len(summary) != 72:
        raise ValueError("Expected 72 three-seed conditions")
    with (collection / summary_name).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=summary[0].keys())
        writer.writeheader()
        writer.writerows(summary)
    lines = ["# Three-seed validation tables" + (" — revised contract" if current_contract else " — original inventory"), "",
             f"Collected and validated: {len(included)}/216 training runs. Official-test results: 0/216.", "",
             "Values are mean best-validation accuracy ± sample standard deviation over model/shuffle seeds 0, 1, 2.",
             "The fixed 5,000-example validation split was used for selection; these are not official-test accuracies.",
             "Incomplete groups show their collected seed count and have no aggregate. All individual seed outcomes remain in the run ledger.", "",
             f"[Per-seed results]({seeds_name}) · [Best and final aggregates]({summary_name}) · [Remaining runs]({remaining_name})", ""]
    if current_contract:
        lines += ["Bounded baseline uses T/K=12/6 for Conv2 and 24/8 for Conv3, for both algorithms. Other groups retain T/K=4/4, 6/6 and 8/8 by architecture. Cross-scheme comparisons therefore use different baseline relaxation counts; report training cost alongside accuracy. Earlier native-T/K baseline BPTT runs are excluded from these revised tables.", ""]
    else:
        lines += ["This table preserves the original inventory. The matched baseline T/K revision is tracked in the [current-contract tables](current_contract_validation_tables.md).", ""]
    indexed = {(row["block"], row["architecture"], row["scheme"], float(row["G_max"])): row for row in summary}
    for block, title in BLOCKS.items():
        lines += ["## " + title, "", "| Architecture | Gmax | Baseline | Ours | Legacy |",
                  "|---|---:|---:|---:|---:|"]
        ceilings = (100.,) if block.startswith(("T1", "T2")) else (1e-4, 5e-4, 1e-3)
        for arch in ("conv1", "conv2", "conv3"):
            for ceiling in ceilings:
                cells = []
                for scheme in ("baseline", "ours", "legacy"):
                    row = indexed[(block, arch, scheme, ceiling)]
                    if row["collected_count"] == 3:
                        cells.append(f"{row['mean_best_validation_percent']:.3f} ± {row['sample_sd_best_validation_pp']:.3f}%")
                    else:
                        cells.append(f"{row['collected_count']}/3 seeds")
                lines.append(f"| {arch} | {ceiling:g} | " + " | ".join(cells) + " |")
        lines.append("")
    (collection / table_name).write_text("\n".join(lines))
    audit = {"updated": datetime.now(timezone.utc).isoformat(),
             "ledger_sha256": hashlib.sha256(ledger.read_bytes()).hexdigest(),
             "evidence": "ordinary_MNIST_validation_only", "official_test_read": False,
             "current_contract": current_contract,
             "sample_standard_deviation_ddof": 1,
             "complete_three_seed_conditions": sum(row["collected_count"] == 3 for row in summary),
             "included_results": included}
    (collection / "provenance" / audit_name).write_text(json.dumps(audit, indent=2) + "\n")
    if analysis_dir is not None:
        analysis_dir = Path(analysis_dir)
        analysis_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(collection / summary_name, analysis_dir / "summary.csv")
        shutil.copy2(collection / "provenance" / audit_name, analysis_dir / "summary.json")
        prefix = os.path.relpath(collection.resolve(), analysis_dir.resolve())
        report = "\n".join(lines)
        for name in (seeds_name, summary_name, remaining_name, "current_contract_validation_tables.md"):
            report = report.replace(f"({name})", f"({prefix}/{name})")
        (analysis_dir / "report.md").write_text(report)
    print(f"Summarized {len(included)}/216 results; {audit['complete_three_seed_conditions']}/72 complete three-seed conditions")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection", type=Path, default=Path("paper_ready_results"))
    parser.add_argument("--analysis-output", type=Path,
                        default=Path("results/paper-training-completion-20260911-v1/analysis/validation_tables"))
    parser.add_argument("--current-contract", action="store_true")
    args = parser.parse_args()
    analysis = args.analysis_output
    if args.current_contract and analysis == Path("results/paper-training-completion-20260911-v1/analysis/validation_tables"):
        analysis = analysis.with_name("current_contract_validation_tables")
    summarize(args.collection, analysis, current_contract=args.current_contract)
