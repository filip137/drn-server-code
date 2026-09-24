"""Recheck a declared baseline beta on a collected revised BPTT reference."""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from experiments.collect_paper_training_completion import validate_training
from experiments.reporting import sha256_file, validate_run

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "results/paper-training-completion-20260911-v1"
COLLECTION = ROOT / "paper_ready_results/baseline_tk_revision"
SOURCE = STUDY / "source-v9"
CHECKS = STUDY / "checks/baseline_tk_revision_best_gate"
CEILINGS = {"1em4": "0.0001", "5em4": "0.0005", "1em3": "0.001"}
CONFIGS = ROOT / "configs/conv/paper_baseline_tk_revision_20260913_v1"


def prepare(architecture, ceiling, beta, smoke=False):
    cell = f"T3_BPTT_{architecture}_baseline_gmax{CEILINGS[ceiling]}_seed0"
    rows = list(csv.DictReader((COLLECTION / "run_status.csv").open()))
    row = next(r for r in rows if r["cell_id"] == cell)
    if row["training_status"] != "complete_collected_validated" or not row["collected_result"]:
        raise ValueError(f"New full BPTT reference is not collected and validated: {cell}")
    bundle = (COLLECTION / row["collected_result"]).parent
    expected = ROOT / row["template_or_parent_config"]
    training, _, manifest = validate_training(bundle, row, expected_config_path=expected)
    if training["completion_plan"].get("operating_point_revision") != "baseline_tk_20260913_v1":
        raise ValueError("An earlier native-T/K reference cannot qualify the revision")
    template = STUDY / f"checks/bounded/configs/{architecture}_baseline_gmax_{ceiling}_decade4.json"
    config = json.loads(template.read_text())
    contract = config["source_contract"]
    contract.update(runtime_source_root=str(SOURCE), source_study_root=str(bundle.parent))
    contract["runtime_code_sha256"] = {name: sha256_file(SOURCE / name) for name in contract["runtime_code_files"]}
    t, k = training["completion_plan"]["accepted_T_K"]
    case = config["cases"][0]
    case.update(run_dir=str(bundle), run_id=bundle.name, T=t, K=k,
                base_beta=beta, injected_beta=beta,
                production_source_commit=manifest["git"]["commit"],
                production_source_archive_sha256=manifest["git"]["source_archive_sha256"],
                source_file_sha256={name: sha256_file(bundle / name) for name in contract["required_source_files"]},
                initializer_checkpoint_path=str(ROOT / training["initialization"]["checkpoint_path"]),
                initializer_checkpoint_sha256=training["initialization"]["checkpoint_sha256"])
    config.update(scientific_question="Does the common baseline beta pass at initialization and the newly trained matched BPTT best checkpoint?",
                  evidence_class="ordinary_mnist_revised_baseline_gradient_qualification",
                  operating_point_revision="baseline_tk_20260913_v1",
                  preparation_script_sha256=sha256_file(Path(__file__)))
    tag = f"{architecture}_{ceiling}_beta{beta:g}" + ("_smoke" if smoke else "")
    path = CHECKS / "configs" / f"{tag}.json"
    payload = json.dumps(config, indent=2, sort_keys=True) + "\n"
    if path.exists() and path.read_text() != payload:
        raise FileExistsError(f"A different frozen replay already occupies {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text(payload)
    return path, CHECKS / "runs" / tag


def freeze_qualified(architecture, beta):
    """Release nine configs only after all three new-reference gates pass."""
    if beta not in (.1, .01):
        raise ValueError("Beta is outside the declared revision recheck candidates")
    references = []
    for ceiling in CEILINGS:
        config_path, output = prepare(architecture, ceiling, beta)
        if not (output / "result.json").exists() or validate_run(output):
            raise ValueError(f"Missing valid full beta replay: {output}")
        manifest = json.loads((output / "manifest.json").read_text())
        metrics = json.loads((output / "result.json").read_text())["terminal_metrics"]
        if manifest["smoke"] or manifest["configuration"]["sha256"] != sha256_file(config_path):
            raise ValueError("Smoke or mismatched replay cannot qualify full training")
        required_true = ("unqualified_launch_gate_all_passed", "gradient_fidelity_all_passed",
                         "equilibrium_residual_all_passed", "all_bias_tensors_exact_zero", "source_bytes_unchanged")
        if any(metrics.get(key) is not True for key in required_true):
            raise ValueError("A revised ceiling fails numerical qualification")
        if (metrics["official_test_read"] or metrics["optimizer_steps_applied"]
                or metrics["replay_count"] != 8
                or metrics["layer_comparison_count"] != (24 if architecture == "conv2" else 32)):
            raise ValueError("Revised gate has incomplete coverage or violates the read-only contract")
        references.append({"run": str(output.relative_to(ROOT)), "result_sha256": sha256_file(output / "result.json"),
                           "ceiling": ceiling, "minimum_cosine": metrics["minimum_cosine"],
                           "maximum_symmetric_norm_delta": metrics["maximum_symmetric_norm_delta"]})
    if beta == .01:
        rejected_point_one = False
        for ceiling in CEILINGS:
            path, run = prepare(architecture, ceiling, .1)
            if not (run / "result.json").exists() or validate_run(run):
                continue
            manifest = json.loads((run / "manifest.json").read_text())
            terminal = json.loads((run / "result.json").read_text())["terminal_metrics"]
            if (not manifest["smoke"] and manifest["configuration"]["sha256"] == sha256_file(path)
                    and terminal["unqualified_launch_gate_all_passed"] is False):
                rejected_point_one = True
        if not rejected_point_one:
            raise ValueError("The selected .1 candidate must fail before .01 is released")
    candidates = sorted((CONFIGS / "candidates").glob(f"T3_EP_{architecture}_baseline_*.json"))
    if len(candidates) != 9:
        raise ValueError("Expected exactly nine revision candidates")
    outputs = []
    selected = {"architecture": architecture, "common_beta_across_three_ceilings": True,
                "selected_injected_beta": beta, "runtime_dtype": "float64", "references": references,
                "selection_rule": "recheck selected .1 on new references; .01 only after .1 fails"}
    for path in candidates:
        config = json.loads(path.read_text())
        config["beta"] = beta
        config["eqprop"].update(injected_beta_B=beta, beta_tier="qualified_matched_baseline_tk_recheck")
        config["completion_plan"]["qualification"] = "bounded_beta_qualified"
        config["completion_plan"]["replication_gate"] = "seed0_pilot" if config["seed"] == 0 else "qualified_group_seed0_pilots"
        config["bounded_beta_qualification"] = selected
        destination = CONFIGS / "training" / path.name
        payload = json.dumps(config, indent=2, sort_keys=True) + "\n"
        if destination.exists() and destination.read_text() != payload:
            raise FileExistsError(f"Different qualified config already exists: {destination}")
        outputs.append((destination, payload))
    selection_path = CONFIGS / f"qualified_{architecture}_selection.json"
    payload = json.dumps(selected, indent=2, sort_keys=True) + "\n"
    if selection_path.exists() and selection_path.read_text() != payload:
        raise FileExistsError("A different revised beta is already frozen")
    for destination, config_payload in outputs:
        if not destination.exists():
            destination.write_text(config_payload)
    selection_path.write_text(payload)
    for seed in (0, 1, 2):
        paths = [str(path.relative_to(ROOT)) for path, _ in outputs if path.name.endswith(f"_seed{seed}.json")]
        (CONFIGS / "lists" / f"{architecture}_ep_seed{seed}.txt").write_text("\n".join(paths) + "\n")
    print(f"QUALIFIED {architecture}: beta {beta:g}; three pilots still required before seeds 1/2")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("architecture", choices=("conv2", "conv3"))
    parser.add_argument("ceiling", choices=tuple(CEILINGS), nargs="?")
    parser.add_argument("--beta", type=float, choices=(.1, .01), default=.1,
                        help="Selected .1 or the remaining lower candidate in the declared baseline ladder")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--freeze-qualified", action="store_true")
    args = parser.parse_args()
    if args.freeze_qualified:
        if args.ceiling or args.smoke or args.prepare_only:
            parser.error("Freezing checks the full architecture; omit ceiling and smoke/prepare flags")
        freeze_qualified(args.architecture, args.beta)
        return
    if args.ceiling is None:
        parser.error("A ceiling is required for a replay")
    config, output = prepare(args.architecture, args.ceiling, args.beta, args.smoke)
    command = [sys.executable, str(SOURCE / "experiments/analyze_conv123_zero_bias_eqprop_bptt_gradient_gate.py"),
               "--config", str(config), "--output-root", str(output.parent), "--run-id", output.name,
               "--device", args.device, "--target", f"main:{args.device}"]
    if args.smoke:
        command.append("--smoke")
    print(json.dumps(command), flush=True)
    if args.prepare_only:
        return
    if args.device == "cuda" and subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True).strip():
        raise RuntimeError("GPU occupied; use CPU or wait for a free allocation")
    env = os.environ.copy()
    env.update(KMP_DISABLE_SHM="1", KMP_SHM_DISABLE="1", OMP_NUM_THREADS="1",
               MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1",
               PYTHONDONTWRITEBYTECODE="1", MPLCONFIGDIR="/tmp/paper-completion-mpl", TF_CPP_MIN_LOG_LEVEL="3")
    started = time.monotonic()
    if not (output / "result.json").exists():
        output.parent.mkdir(parents=True, exist_ok=True)
        with (output.parent / f"{output.name}.log").open("x") as log:
            subprocess.run(command, cwd=SOURCE, env=env, stdout=log, stderr=subprocess.STDOUT,
                           check=True, timeout=900)
    if validate_run(output):
        raise ValueError("Revised beta replay failed canonical validation")
    manifest = json.loads((output / "manifest.json").read_text())
    if manifest["configuration"]["sha256"] != sha256_file(config):
        raise ValueError("Revised beta replay used a different frozen config")
    metrics = json.loads((output / "result.json").read_text())["terminal_metrics"]
    if metrics["official_test_read"] or not metrics["source_bytes_unchanged"]:
        raise ValueError("Read-only replay contract failed")
    print(json.dumps({"run": str(output), "wall_seconds_this_invocation": time.monotonic() - started,
                      "smoke": args.smoke, **metrics}), flush=True)


if __name__ == "__main__":
    main()
