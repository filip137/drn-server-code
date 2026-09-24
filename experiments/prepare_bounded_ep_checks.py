"""Run the declared small bounded-EP beta ladder on collected seed-0 BPTT references.

One existing scientific replay per ceiling avoids mixing checkpoint identities.
Missing corrected references leave their complete architecture/scheme group pending.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from experiments.prepare_paper_training_completion import BETAS, SCHEMES, CEILINGS, CONFIG_ROOT, RESULT_ROOT, STUDY, write_json
from experiments.reporting import validate_run

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def freeze_qualified_configs(selection_path=None):
    """Freeze passing beta choices; replications still need their full pilot group."""
    selection_path = Path(selection_path) if selection_path else ROOT / RESULT_ROOT / "checks/bounded_beta_selection.json"
    selection = json.loads(selection_path.read_text())
    written = []
    for group in selection["groups"]:
        if group["status"] != "qualified":
            continue
        selected = group["selected_injected_beta"]
        attempt = next(a for a in group["attempts"] if a["injected_beta"] == selected)
        if {row["weight_max"] for row in attempt["ceilings"]} != set(CEILINGS):
            raise ValueError("Beta selection does not cover all three ceilings")
        for row in attempt["ceilings"]:
            path = ROOT / row["run"]
            if validate_run(path) or sha(path / "result.json") != row["result_sha256"]:
                raise ValueError("Selected beta evidence changed")
            if not row["unqualified_launch_gate_all_passed"]:
                raise ValueError("Selected beta did not pass both numerical gates")
        for source in sorted((ROOT / CONFIG_ROOT / "candidates").glob(
                f"T3_EP_{group['architecture']}_{group['scheme']}_*.json")):
            config = json.loads(source.read_text())
            config["beta"] = selected / config["eqprop"]["amplification_factor"]
            config["eqprop"].update(injected_beta_B=selected, beta_tier="qualified_bounded_ladder")
            config["completion_plan"]["qualification"] = "bounded_beta_qualified"
            if config["seed"] in (1, 2):
                config["completion_plan"]["replication_gate"] = "qualified_group_seed0_pilots"
            config["bounded_beta_qualification"] = {
                "common_beta_across_three_ceilings": True, "selected_injected_beta": selected,
                "largest_tested_passing_beta": True, "runtime_dtype": "float64",
                "references": [{"run": r["run"], "result_sha256": r["result_sha256"]} for r in attempt["ceilings"]]}
            destination = ROOT / CONFIG_ROOT / "training" / source.name
            write_json(destination, config)
            written.append(str(destination.relative_to(ROOT)))
    return written


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--group", action="append", choices=[f"{a}/{s}" for a in BETAS for s in SCHEMES],
                        help="Check only these groups; requires a separate report path")
    parser.add_argument("--report", type=Path, help="Independent summary for a selected group")
    args = parser.parse_args()
    if args.group and args.report is None:
        parser.error("--group requires --report to preserve the full ladder summary")
    study = ROOT / RESULT_ROOT
    report_path = args.report or study / "checks/bounded_beta_selection.json"
    if args.group and report_path.resolve() == (study / "checks/bounded_beta_selection.json").resolve():
        parser.error("A selected group must not overwrite the full ladder summary")
    frozen = study / "source"
    template = json.loads((study / "checks/configs/wide_conv1_legacy.json").read_text())
    template["source_contract"]["runtime_source_root"] = str(frozen)
    # The source snapshot contains the exact code qualified by the wide check.
    for name, expected in template["source_contract"]["runtime_code_sha256"].items():
        if sha(frozen / name) != expected:
            raise ValueError(f"Frozen runtime differs: {name}")
    report = {"study_id": STUDY, "device": args.device, "official_test_read": False,
              "ladder_multipliers": [1., .1, .01, .001, .0001], "groups": []}
    for arch in BETAS:
        for scheme in SCHEMES:
            if args.group and f"{arch}/{scheme}" not in args.group:
                continue
            sources = {ceiling: ROOT / f"paper_ready_results/bundles/table3_bounded_bptt/{arch}/{scheme}/gmax_{tag}/seed0"
                       for ceiling, tag in CEILINGS.items()}
            group = {"architecture": arch, "scheme": scheme, "attempts": [],
                     "status": "pending_corrected_bptt_references", "selected_injected_beta": None}
            report["groups"].append(group)
            if not all((path / "result.json").exists() for path in sources.values()):
                print(f"PENDING {arch}/{scheme}: corrected BPTT references", flush=True)
                continue
            for decade, multiplier in enumerate(report["ladder_multipliers"]):
                beta = BETAS[arch][SCHEMES.index(scheme)] * multiplier
                attempt = {"injected_beta": beta, "ceilings": []}
                group["attempts"].append(attempt)
                for ceiling, source in sources.items():
                    depth = int(arch[-1]); tk = {1:4, 2:6, 3:8}[depth]
                    source_manifest = json.loads((source / "manifest.json").read_text())
                    original = json.loads((source / "config.used.json").read_text())
                    av, ai = original["model_base"]["voltage_amp"], original["model_base"]["current_amp"]
                    asset = study / f"assets/bounded_uniform/{arch}/seed0.pt"
                    config = copy.deepcopy(template)
                    config["scientific_question"] = "Qualify bounded centered EP against matched BPTT at initialization and best validation."
                    config["source_contract"]["source_study_root"] = str(source.parent)
                    config["cases"] = [{"architecture": arch, "scheme": scheme,
                        "run_id": source.name, "run_dir": str(source), "T": tk, "K": tk,
                        "output_row_exponent_L": depth, "base_beta": beta / (av/ai)**depth,
                        "injected_beta": beta,
                        "production_source_commit": source_manifest["git"]["commit"],
                        "production_source_archive_sha256": source_manifest["git"].get("source_archive_sha256"),
                        "source_file_sha256": {name:sha(source/name) for name in config["source_contract"]["required_source_files"]},
                        "initializer_checkpoint_path": str(asset), "initializer_checkpoint_sha256": sha(asset)}]
                    config["completion"].update(production_replay_count=8, production_layer_comparison_count=8*(depth+1))
                    run_id = f"{arch}_{scheme}_gmax_{CEILINGS[ceiling]}_decade{decade}"
                    config_path = study / "checks/bounded/configs" / f"{run_id}.json"
                    write_json(config_path, config)
                    output = study / "checks/bounded" / run_id
                    if not (output / "result.json").exists():
                        command = [sys.executable, str(frozen / "experiments/analyze_conv123_zero_bias_eqprop_bptt_gradient_gate.py"),
                            "--config", str(config_path), "--output-root", str(output.parent), "--run-id", run_id,
                            "--device", args.device, "--target", f"main:{args.device}"]
                        print(f"CHECK {run_id} injected_beta={beta:g}", flush=True)
                        with (output.parent / f"{run_id}.log").open("x") as log:
                            subprocess.run(command, cwd=frozen, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=1200)
                    errors = validate_run(output)
                    if errors:
                        raise ValueError(f"Invalid replay {output}: {errors}")
                    manifest = json.loads((output / "manifest.json").read_text())
                    if manifest["configuration"]["sha256"] != sha(config_path):
                        raise ValueError("Existing replay uses a different config")
                    result = json.loads((output / "result.json").read_text())
                    terminal = result["terminal_metrics"]
                    attempt["ceilings"].append({"weight_max": ceiling, "run": str(output.relative_to(ROOT)),
                        "result_sha256": sha(output / "result.json"), **terminal})
                if all(row["unqualified_launch_gate_all_passed"] for row in attempt["ceilings"]):
                    group.update(status="qualified", selected_injected_beta=beta)
                    break
            else:
                group["status"] = "held_numerical_or_equilibrium_gate"
            print(f"{group['status'].upper()} {arch}/{scheme} beta={group['selected_injected_beta']}", flush=True)
            # This is a mutable summary; all underlying configs/bundles are immutable.
            report_path.write_text(json.dumps(report, indent=2) + "\n")
    report_path.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
