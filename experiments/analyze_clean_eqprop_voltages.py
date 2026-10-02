"""Validate and plot matched initialization/final clean EqProp voltage replays."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

from experiments.reporting import validate_run

ROOT = Path(__file__).resolve().parents[1]
STUDY = "conv123-clean-eqprop-trained-voltage-20260928-v1"
SCHEMES = {"baseline": (1, 1), "ours": (4, 1), "legacy": (4, .25)}
COLORS = {"baseline": "#333333", "ours": "#d95f02", "legacy": "#9b59b6"}
MARKERS = {"baseline": "o", "ours": "D", "legacy": "x"}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def plot(rows, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fields = (("free_rms", "Free-state RMS voltage (V)", "RMS voltage magnitude"),
              ("free_mean_absolute", "Mean |free-state voltage| (V)", "Mean absolute voltage"),
              ("free_signed_mean", "Mean free-state voltage (V)", "Signed mean voltage"),
              ("endpoint_rms_P", "Nudged endpoint RMS voltage P (V)", "Nudged endpoint voltage magnitude"))

    def panel(ax, architecture, role, field, limits):
        depth = int(architecture[-1])
        for scheme, (a, b) in SCHEMES.items():
            selected = sorted((r for r in rows if r["architecture"] == architecture and r["scheme"] == scheme
                               and r["checkpoint_role"] == role), key=lambda r: int(r["state_layer_index"]))
            ax.plot(range(depth+1), [float(r[field]) for r in selected], color=COLORS[scheme], marker=MARKERS[scheme],
                    markerfacecolor="none" if scheme == "baseline" else COLORS[scheme], label=f"{scheme} ({a:g}, {b:g})", lw=1.8)
        ax.set_xticks(range(depth+1), [f"Conv{i+1}" for i in range(depth)]+["Readout"])
        ax.set_xlabel("Layer"); ax.grid(alpha=.2)
        if field == "free_signed_mean":
            ax.set_yscale("symlog", linthresh=1e-4); ax.axhline(0, color="gray", lw=.6)
        else:
            ax.set_yscale("log")
        ax.set_ylim(*limits)
        epoch = 0 if role == "initialization" else (10 if depth == 1 else 30)
        ax.set_title(f"{architecture.upper()} · epoch {epoch}")

    for field, label, title in fields:
        values = [float(r[field]) for r in rows]
        if field == "free_signed_mean":
            peak = max(abs(v) for v in values)
            limits = (-1.5*max(peak,1e-4), 1.5*max(peak,1e-4))
        else:
            assert min(values) > 0, f"Zero voltage cannot be silently omitted from log plot: {field}"
            limits = (min(values)/1.5, max(values)*1.5)
        subtitle = "Clean EqProp · seed 0 · 576 paired validation examples · pooled over all nodes"
        if field == "endpoint_rms_P":
            subtitle += "\nSaved training beta held fixed; trained output displacement is not rematched"
        for roles, suffix in ((("final",), "trained"), (("initialization", "final"), "initial_vs_trained")):
            fig, axes = plt.subplots(len(roles), 3, figsize=(13.6, 4.6 if len(roles)==1 else 8), squeeze=False)
            for i, role in enumerate(roles):
                for j, architecture in enumerate(("conv1", "conv2", "conv3")):
                    panel(axes[i,j], architecture, role, field, limits)
                    if j == 0: axes[i,j].set_ylabel(label)
            axes[0,-1].legend(title="(voltage gain, current gain)", fontsize=8)
            fig.suptitle(f"{title}\n{subtitle}", fontsize=12)
            fig.tight_layout()
            fig.savefig(output/f"{field}_{suffix}.jpg", dpi=180, bbox_inches="tight")
            plt.close(fig)


def analyze(study, output):
    rows, provenance = [], []
    for depth in (1,2,3):
        architecture = f"conv{depth}"
        lane = study/"production"/architecture
        errors = validate_run(lane)
        if errors: raise ValueError(errors)
        manifest = json.loads((lane/"manifest.json").read_text())
        assert not manifest.get("smoke", False)
        assert not manifest["official_test_read"] and not manifest["optimizer_steps_applied"]
        cohort = json.loads((lane/"cohort.json").read_text())
        assert cohort["cohort_sha256"] == "95af3eebd9416ec643b7d698d061b7dac3fe0d52311dd888293609bada629acd"
        assert cohort["example_count"] == 576 and cohort["batch_count"] == 36
        cases = sorted((lane/"cases").iterdir())
        assert len(cases) == 6
        for case in cases:
            errors = validate_run(case)
            if errors: raise ValueError(errors)
            data = read_csv(case/"voltage_profiles.csv")
            assert len(data) == depth+1
            assert {int(r["state_layer_index"]) for r in data} == set(range(depth+1))
            guard = json.loads((case/"read_only_guards.json").read_text())
            for field in ("checkpoint_restore_exact", "parameters_unchanged", "source_checkpoint_bytes_unchanged",
                          "source_config_bytes_unchanged", "zero_biases", "frozen_current_force_unchanged",
                          "centered_endpoints_share_post_T_start", "native_float64"):
                assert guard[field] is True, (case, field)
            assert guard["cohort_sha256"] == cohort["cohort_sha256"]
            assert guard["initialization_reference_checked"] == (data[0]["checkpoint_role"] == "initialization")
            for row in data:
                assert row["architecture"] == architecture
                assert row["scheme"] in SCHEMES and row["checkpoint_role"] in ("initialization", "final")
                expected_epoch = 0 if row["checkpoint_role"] == "initialization" else (10 if depth == 1 else 30)
                assert int(row["checkpoint_epoch"]) == expected_epoch
                assert int(row["T"]) == int(row["K"]) == depth*2+2
                assert float(row["input_gain"]) == {1:40,2:100,3:360}[depth]
                config_path = ROOT/f"configs/conv/clean_eqprop_trained_voltage_20260928/{architecture}.json"
                cfg = json.loads(config_path.read_text())
                declared = next(c for c in cfg["cases"] if c["name"] == row["name"])
                source = json.loads((ROOT/declared["source_config"]).read_text())
                layer = int(row["state_layer_index"])
                shape = source["model_overrides"][source["lab"]["model_key"]]["layer_shapes"][layer+1]
                assert int(row["free_element_count"]) == 576*math.prod(shape)
                for field in ("free_rms", "free_mean_absolute", "free_signed_mean", "free_std", "endpoint_rms_P", "centered_half_difference_rms_D", "D_over_F", "D_over_P"):
                    assert math.isfinite(float(row[field]))
                assert 0 <= float(row["free_mean_absolute"]) <= float(row["free_rms"])*(1+1e-12)
                assert abs(float(row["free_signed_mean"])) <= float(row["free_mean_absolute"])*(1+1e-12)
                assert math.isclose(float(row["free_rms"])**2, float(row["free_signed_mean"])**2+float(row["free_std"])**2, rel_tol=1e-10, abs_tol=1e-15)
                assert math.isclose(float(row["D_over_F"]),float(row["centered_half_difference_rms_D"])/float(row["free_rms"]),rel_tol=1e-10)
            rows.extend(data)
            provenance.append(dict(case=str(case), result_sha256=sha(case/"result.json"), voltage_profiles_sha256=sha(case/"voltage_profiles.csv")))
    expected={(f"conv{d}",s,role,i) for d in (1,2,3) for s in SCHEMES for role in ("initialization","final") for i in range(d+1)}
    identities=[(r["architecture"],r["scheme"],r["checkpoint_role"],int(r["state_layer_index"])) for r in rows]
    assert len(identities)==len(set(identities)) and set(identities)==expected
    lookup={key:r for key,r in zip(identities,rows)}
    comparison=[]
    for r in rows:
        if r["checkpoint_role"] != "final": continue
        key=(r["architecture"],r["scheme"],"initialization",int(r["state_layer_index"]))
        initial=lookup[key]
        result={key:r[key] for key in ("architecture","scheme","checkpoint_epoch","state_layer_index","state_layer_name")}
        for field in ("free_rms","free_mean_absolute","free_signed_mean","endpoint_rms_P"):
            first,last=float(initial[field]),float(r[field])
            result.update({f"initial_{field}":first,f"trained_{field}":last,f"delta_{field}":last-first})
            if field != "free_signed_mean": result[f"trained_over_initial_{field}"]=last/first if first else None
        result["initial_D_over_F"]=float(initial["D_over_F"]); result["trained_D_over_F"]=float(r["D_over_F"])
        comparison.append(result)
    output.mkdir(parents=True,exist_ok=True)
    write_csv(output/"voltage_profiles.csv",rows)
    write_csv(output/"voltage_epoch_comparison.csv",comparison)
    plot(rows,output)
    (output/"validation.json").write_text(json.dumps(dict(complete=True,contexts=18,layer_rows=len(rows),examples_per_context=576,
        checkpoint_epochs={"conv1":[0,10],"conv2":[0,30],"conv3":[0,30]}, seed=0,
        training_algorithm="clean centered EqProp", beta="saved training beta; no rematching",
        analysis_source_sha256=sha(Path(__file__)), sources=provenance),indent=2)+"\n")
    print(f"Validated 18 contexts / {len(rows)} layer profiles: {output}")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-root",type=Path,default=ROOT/"results"/STUDY)
    parser.add_argument("--output-dir",type=Path)
    args=parser.parse_args()
    analyze(args.study_root,args.output_dir or args.study_root/"analysis")


if __name__ == "__main__":
    main()
