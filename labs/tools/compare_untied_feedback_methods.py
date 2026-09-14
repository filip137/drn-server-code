"""Focus the verified untied-weight MNIST study on VF, homeostasis and zero order.

Reuses completed trajectories; it does not train, tune, interpolate accuracy,
or combine the zero-order method with homeostasis. Measurement-matched results
use the largest exactly shared stored equilibrium count within every method.
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np


METHODS = ("ep_activity", "ep_homeo", "learned_probe4")
LABELS = {"ep_activity": "VF EqProp (activity)",
          "ep_homeo": "VF EqProp + homeostasis",
          "learned_probe4": "Zero-order feedback (learned baseline, 4 probes)"}
PARAMETERS = ("input", "forward1", "forward2", "backward1", "backward2",
              "bias", "readout", "readout_bias")


def load_cases(root):
    root = Path(root)
    verification = json.loads((root/"verification.json").read_text())
    assert verification["status"] == "passed"
    run = json.loads((root/"run.json").read_text())
    cases = []
    for name, cfg in run["cases"].items():
        if cfg["method"] not in METHODS:
            continue
        path = root/name
        verified = json.loads((path/"independent_verification.json").read_text())
        result = json.loads((path/"result.json").read_text())
        assert verified["status"] == "passed" and result["status"] == "complete"
        assert result["config"] == cfg
        initial = np.load(path/"initial.npz")
        final = np.load(path/"final.npz")
        assert set(PARAMETERS).issubset(initial.files)
        # Independently updated directed matrices, with no fixed-skew parameters.
        assert not any("skew" in key for key in initial.files)
        for key in ("forward1", "forward2", "backward1", "backward2"):
            assert not np.array_equal(initial[key], final[key]), (name, key)
        if cfg["method"] != "ep_homeo":
            assert result["training_counts"]["homeostasis_jvps"] == 0
        cases.append(dict(name=name, config=cfg, result=result,
                          history=json.loads((path/"metrics.json").read_text()),
                          initial={key: initial[key] for key in PARAMETERS}))
    groups = {(c["config"]["alpha"], c["config"]["seed"]) for c in cases}
    assert groups
    common = {k:v for k,v in cases[0]["config"].items() if k not in ("method", "alpha", "seed")}
    assert all({k:v for k,v in c["config"].items() if k not in ("method", "alpha", "seed")} == common for c in cases)
    for alpha, seed in groups:
        group = [c for c in cases if (c["config"]["alpha"], c["config"]["seed"]) == (alpha, seed)]
        assert {c["config"]["method"] for c in group} == set(METHODS)
        baseline = next(c for c in group if c["config"]["method"] == "ep_activity")
        for case in group:
            assert {k:v for k,v in case["config"].items() if k != "method"} == {
                k:v for k,v in baseline["config"].items() if k != "method"}
            for key in PARAMETERS:
                np.testing.assert_array_equal(case["initial"][key], baseline["initial"][key])
    return cases, run


def aggregate(cases):
    # Only exact stored epoch endpoints; no interpolated or extrapolated metrics.
    count_sets = [{e["training_counts"]["equilibrations"] for e in c["history"]} for c in cases]
    shared = set.intersection(*count_sets)
    if not shared:
        raise ValueError("Expected an exactly shared saved equilibrium budget; found none")
    budget = max(shared)
    rows = []
    for alpha in sorted({c["config"]["alpha"] for c in cases}):
        for method in METHODS:
            group = [c for c in cases if c["config"]["alpha"] == alpha and c["config"]["method"] == method]
            matched = [next(e for e in c["history"] if e["training_counts"]["equilibrations"] == budget) for c in group]
            assert len({e["epoch"] for e in matched}) == 1
            row = dict(alpha=alpha, method=method, label=LABELS[method], seeds=len(group),
                       final_epoch=group[0]["config"]["epochs"], matched_epoch=matched[0]["epoch"],
                       matched_equilibrations=budget,
                       final_equilibrations=group[0]["result"]["training_counts"]["equilibrations"],
                       matched_digital_jvps=matched[0]["training_counts"]["homeostasis_jvps"])
            values = dict(test_accuracy=[c["result"]["final_test"]["accuracy"] for c in group],
                          matched_validation_accuracy=[e["validation"]["accuracy"] for e in matched],
                          hidden_feedback_cosine=[c["result"]["final_audit"]["hidden1_feedback_cosine"] for c in group],
                          first_pair_weight_angle=[c["result"]["final_audit"]["pair1_weight_angle_degrees"] for c in group])
            for key, numbers in values.items():
                row[key+"_values"] = numbers
                row[key+"_mean"] = float(np.mean(numbers))
                row[key+"_sd"] = float(np.std(numbers))
            rows.append(row)
    return rows, budget


def report(root, output):
    root, output = Path(root), Path(output)
    cases, run = load_cases(root)
    rows, budget = aggregate(cases)
    cfg = cases[0]["config"]
    dataset = json.loads((root/"dataset.json").read_text())
    seeds = len({c["config"]["seed"] for c in cases})
    projections = [sum(e["projection_batches"] for e in c["history"])/(
        cfg["epochs"]*math.ceil(dataset["train_examples"]/cfg["batch_size"])) for c in cases]
    matched_epochs = ", ".join(f"{LABELS[m]}: epoch {next(r['matched_epoch'] for r in rows if r['method']==m)}"
                               for m in METHODS)
    output.mkdir(parents=True, exist_ok=True)
    payload = dict(source=str(root.resolve()), source_commit=run["git_head"],
                   scope="untied trained forward/backward weights; three uncombined learning methods",
                   trajectories_reused=len(cases), new_training_runs=0,
                   equal_initial_parameters_verified=True, rows=rows,
                   matched_budget_scope="Equilibrium count only; homeostasis AD/JVP costs are reported separately")
    (output/"comparison.json").write_text(json.dumps(payload, indent=2)+"\n")
    angles = sorted({r["alpha"] for r in rows})
    lines = ["# Zero-order feedback versus VF EqProp and Jacobian homeostasis", "",
        "The network has independently trainable forward and backward weights. "
        "This comparison uses the completed untied-weight MNIST study. Initial network parameters "
        "are identical across the three methods at each seed/mixing angle. No new training or selection was performed.", "",
        "The primary comparison uses a common activity-response VF rule, matching the homeostasis "
        "implementation's convention. The original membrane-state VF control remains in the full "
        "study; changing coordinates as well as adding a regularizer would confound the homeostasis comparison.", "",
        "| Method | What changes in learning | Network equilibrations per example |",
        "|---|---|---:|",
        "| VF EqProp | Local directed-synapse update using the error-nudged activity response | 3 |",
        "| VF EqProp + homeostasis | Same task update, plus the AD gradient of the Jacobian-symmetry penalty | 3 |",
        "| Zero-order feedback | Four random state-nudge pairs estimate adjoint projections; a learned baseline reduces residual variance | 9 |", "",
        "Homeostasis uses five Gaussian vectors and ten forward JVPs per example, plus parameter AD. "
        f"The zero-order arm contains no homeostasis. It learns a {2*cfg['hidden']+10}-by-10 feedback predictor from past measurements. "
        "Fresh probes correct that predictor before the predictor is updated for later batches.", "",
        "Zero order refers to estimating equilibrium feedback through perturbations and observations. "
        "It does not reconstruct the entire Jacobian. It still uses the known output-cost gradient "
        "and local parameter-force derivatives in the directed-synapse update; it is not parameter-space SPSA.", "",
        "## Same number of training epochs", "",
        f"Test accuracy (%), fixed epoch {cfg['epochs']}, mean ± population SD over {seeds} seeds. "
        "The zero-order arm uses three times as many training equilibrations.", "",
        "| Method | "+" | ".join(f"Initial mixing {a:g}°" for a in angles)+" |",
        "|---|"+"---:|"*len(angles)]
    for method in METHODS:
        group = [next(r for r in rows if r["method"] == method and r["alpha"] == a) for a in angles]
        lines.append("| "+LABELS[method]+" | "+" | ".join(
            f"{100*r['test_accuracy_mean']:.2f} ± {100*r['test_accuracy_sd']:.2f}" for r in group)+" |")
    lines += ["", "## Same number of measured equilibrations", "",
        f"Validation accuracy at exactly {budget:,} training equilibrations ({matched_epochs}). "
        "These are actual stored epoch endpoints, "
        "without interpolation. No intermediate test-set evaluation was used.", "",
        "| Method | "+" | ".join(f"Initial mixing {a:g}°" for a in angles)+" | Additional digital JVPs |",
        "|---|"+"---:|"*(len(angles)+1)]
    for method in METHODS:
        group = [next(r for r in rows if r["method"] == method and r["alpha"] == a) for a in angles]
        lines.append("| "+LABELS[method]+" | "+" | ".join(
            f"{100*r['matched_validation_accuracy_mean']:.2f} ± {100*r['matched_validation_accuracy_sd']:.2f}" for r in group)+
            f" | {group[0]['matched_digital_jvps']:,} |")
    lines += ["", "Equal equilibrium counts do not equal wall time, energy, or total digital work. "
        "Homeostasis also uses parameter AD; physical relaxation counts/iterations and read counts remain in the original records.", "",
        "![Focused accuracy comparison](accuracy_comparison.png)", "",
        "![Validation trajectories](learning_curves.png)", "",
        "## Interpretation", "",
        "Single-example neuronal-error alignment and eventual task accuracy measure different outcomes. "
        "A four-probe adjoint estimate remains noisy even when its averaged parameter updates learn. "
        "These measurements do not isolate the reason for that optimization behavior.", "",
        f"Limits: {dataset['features']}-feature MNIST, "
        f"{dataset['train_examples']:,}/{dataset['validation_examples']:,}/{dataset['test_examples']:,} "
        f"train/validation/test split, {cfg['hidden']}/{cfg['hidden']}/10 dynamical states, "
        f"{cfg['epochs']} epochs, {seeds} seeds, common Adam settings, read noise {cfg['noise']:g}. "
        f"The recurrent norm cap acts on {100*min(projections):.1f}–{100*max(projections):.1f}% of updates. "
        "This is preliminary evidence under a strong "
        "stability constraint, not a reproduction of published performance or a general algorithm ranking.", "",
        "Definitions and parameter updates follow the general "
        "[VF EqProp construction](https://arxiv.org/html/1808.04873) and the "
        "[Jacobian-homeostasis paper](https://arxiv.org/html/2309.02214v2), with the coordinate and "
        "implementation adaptations documented in docs/directed_eqprop_mnist.md.", ""]
    (output/"report.md").write_text("\n".join(lines))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = ("#3564a1", "#35865d", "#c05b27")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), layout="constrained")
    width = .23
    for ax, key, title in zip(axes, ("test_accuracy", "matched_validation_accuracy"),
            (f"{cfg['epochs']} epochs · test accuracy", f"{budget:,} equilibrations · validation accuracy")):
        for index, (method, color) in enumerate(zip(METHODS, colors)):
            group = [next(r for r in rows if r["method"] == method and r["alpha"] == a) for a in angles]
            positions = np.arange(len(angles))+(index-1)*width
            ax.bar(positions, [100*r[key+"_mean"] for r in group], width,
                   yerr=[100*r[key+"_sd"] for r in group], capsize=3, color=color, alpha=.8,
                   label=LABELS[method])
            for pos, row in zip(positions, group):
                ax.scatter(np.full(len(row[key+"_values"]), pos), np.array(row[key+"_values"])*100,
                           c="black", s=11, zorder=4)
        ax.set_xticks(range(len(angles)), [f"{a:g}°" for a in angles])
        ax.set_xlabel("Initial backward-weight mixing angle")
        ax.set_ylabel("Accuracy (%)")
        ax.set_ylim(0, 100)
        ax.set_title(title, fontsize=11)
        ax.grid(axis="y", alpha=.2)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="outside lower center", ncol=1, fontsize=9)
    fig.savefig(output/"accuracy_comparison.png", dpi=180)
    fig.savefig(output/"accuracy_comparison.pdf")
    plt.close(fig)
    fig, axes = plt.subplots(1, len(angles), figsize=(13, 4), layout="constrained", squeeze=False)
    for ax, angle in zip(axes.flat, angles):
        for method, color in zip(METHODS, colors):
            group = [c for c in cases if c["config"]["method"] == method and c["config"]["alpha"] == angle]
            values = np.array([[e["validation"]["accuracy"] for e in c["history"]] for c in group])*100
            epochs = np.arange(1, values.shape[1]+1)
            ax.plot(epochs, values.mean(0), color=color, label=LABELS[method], marker="o", markersize=3)
            ax.fill_between(epochs, values.mean(0)-values.std(0), values.mean(0)+values.std(0), color=color, alpha=.12)
        ax.set_title(f"Initial mixing {angle:g}°")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Validation accuracy (%)")
        ax.set_ylim(0, 100)
        ax.grid(alpha=.2)
    fig.legend(*axes[0,0].get_legend_handles_labels(), loc="outside lower center", ncol=1, fontsize=9)
    fig.savefig(output/"learning_curves.png", dpi=180)
    plt.close(fig)
    print(json.dumps(dict(status="complete", trajectories_reused=len(cases),
                          exactly_matched_equilibrations=budget, output=str(output))))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report(args.root, args.output)
