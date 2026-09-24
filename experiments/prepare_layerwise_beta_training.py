"""Freeze the requested cosine-only selections and five missing training pilots."""
import csv
import hashlib
import json
import shutil
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STUDY = "eqprop-layerwise-beta-training-20260919-v1"


def main():
    out = ROOT / "results" / STUDY
    out.mkdir(exist_ok=False)
    old = ROOT / "results/eqprop-beta-training-stability-20260918-v1"
    raw = defaultdict(list)
    with (ROOT / "paper_ready_results/beta_rule_comparison_20260918_layer_metrics.csv").open() as stream:
        for r in csv.DictReader(stream):
            if r["architecture"] in ("conv2", "conv3"):
                raw[r["architecture"], r["scheme"], float(r["injected_beta"])].append(r)
    selections, cases = [], {}
    prior = json.loads((old / "analysis.json").read_text())["rows"]
    placement = {("conv2", "baseline"): "fifi", ("conv2", "ours"): "loulou",
                 ("conv3", "baseline"): "riri", ("conv3", "ours"): "trex"}
    configs = ROOT / "configs/conv/eqprop_layerwise_beta_training_20260919_v1"
    configs.mkdir(exist_ok=False)
    for arch in ("conv2", "conv3"):
        for scheme in ("baseline", "ours", "legacy"):
            candidates = sorted(b for a, s, b in raw if (a, s) == (arch, scheme))
            for threshold in (.90, .95):
                beta = max(b for b in candidates if min(float(r["cosine"]) for r in raw[arch, scheme, b]) > threshold)
                measurements = raw[arch, scheme, beta]
                layers = sorted({r["parameter_name"] for r in measurements})
                assert len(layers) == int(arch[-1]) + 1
                assert all(not layer.startswith("Bias_") for layer in layers)
                assert len(measurements) == 72 * len(layers)
                assert {r["checkpoint_role"] for r in measurements} == {"reconstructed_initialization", "best_validation"}
                assert all(float(r["endpoint_read_noise_std"]) == 0 for r in measurements)
                name = f"{arch}_{scheme}_beta{beta:g}_seed0"
                selections.append(dict(architecture=arch, scheme=scheme, threshold=threshold,
                    case=name, injected_beta=beta, minimum_matrix_cosine=min(float(r["cosine"]) for r in measurements),
                    maximum_matrix_norm_mismatch=max(float(r["symmetric_norm_delta"]) for r in measurements),
                    comparisons=len(measurements), upper_grid_edge=beta == max(candidates)))
                if name in cases:
                    cases[name]["thresholds"].append(threshold)
                    continue
                control = f"paper_ready_results/bundles/table2_wide_ep/{arch}/{scheme}/seed0"
                control_cfg = json.loads((ROOT / control / "config.used.json").read_text())
                existing = next((r for r in prior if r["architecture"] == arch and r["scheme"] == scheme and r["injected_beta"] == beta), None)
                row = dict(case=name, architecture=arch, scheme=scheme, injected_beta=beta,
                           thresholds=[threshold], comparison_control=control)
                if existing:
                    row.update(mode="reuse_pilot", bundle=existing["bundle"], target=existing["target"],
                               prior_study=str(old.relative_to(ROOT)))
                elif control_cfg["eqprop"]["injected_beta_B"] == beta:
                    row.update(mode="reuse_control_prefix", bundle=control, target="historical_control",
                               reused_epochs=list(range(1, 11)))
                else:
                    source = next((ROOT / "configs/conv/eqprop_beta_training_stability_20260918_v1").glob(f"{arch}_{scheme}_*.json"))
                    cfg = json.loads(source.read_text())
                    cfg.update(arm_id=name, study_id=STUDY, beta=beta / cfg["eqprop"]["amplification_factor"])
                    cfg["eqprop"].update(injected_beta_B=beta, beta_tier="per_matrix_cosine_only")
                    cfg["reporting"].update(arm_id=name, study_id=STUDY)
                    cfg["qualification_source"].update(rule="every weight matrix cosine strictly > threshold on all 72 replays; cosine only, no norm gate", gradient_gate_override="user_authorized_per_matrix_cos90_cos95_training_pilot")
                    cfg["stability_pilot"]["cosine_only_thresholds"] = [threshold]
                    filename = configs / f"{name}.json"
                    filename.write_text(json.dumps(cfg, indent=2) + "\n")
                    row.update(mode="new", target=placement[arch, scheme], config=str(filename.relative_to(ROOT)))
                cases[name] = row
    for row in cases.values():
        if row["mode"] == "new":
            p = ROOT / row["config"]
            cfg = json.loads(p.read_text())
            cfg["stability_pilot"]["cosine_only_thresholds"] = row["thresholds"]
            p.write_text(json.dumps(cfg, indent=2) + "\n")
    assert len(cases) == 9 and sum(r["mode"] == "new" for r in cases.values()) == 5
    (out / "cases.json").write_text(json.dumps(list(cases.values()), indent=2) + "\n")
    (out / "selection_audit.json").write_text(json.dumps(selections, indent=2) + "\n")
    shutil.copytree(old / "source", out / "source", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    staged = out / "source/configs/layerwise_beta"
    staged.mkdir()
    for p in configs.glob("*.json"):
        shutil.copy2(p, staged / p.name)
    snapshot = out / "source"
    paths = sorted(p for p in snapshot.rglob("*") if p.is_file() and not p.is_symlink())
    (snapshot / "LAYERWISE_SHA256SUMS").write_text("".join(
        f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(snapshot)}\n" for p in paths))
    print(json.dumps({"study": STUDY, "rules": 12, "distinct_settings": 9, "new_runs": 5}, indent=2))


if __name__ == "__main__":
    main()
