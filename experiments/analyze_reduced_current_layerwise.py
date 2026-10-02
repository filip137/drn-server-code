"""Validate and plot the bounded seed-zero reduced-current initialization replay.

This is diagnostic finite-K evidence, not an equilibrium or accuracy claim.
State RMS values pool element-weighted squares over 36 batches. Gradient cosines
use only batches 0..11, average within each draw, then across three draws. Bands
are draw minima/maxima, not confidence intervals. Undefined cosines remain gaps.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STUDY = "conv123-reduced-current-layerwise-20260928-v1"
REFERENCE = "conv123-relative-noise-advantage-map-20260925-v1"
SCHEMES = {"baseline": (1, 1), "v1_c0p5": (1, .5), "v1_c0p25": (1, .25),
           "ours": (4, 1), "legacy": (4, .25)}
TARGETS = {"conv1": 1, "conv2": 4, "conv3": 6}
LEVELS = (0., 1e-6, 1e-5, 1e-4, 3e-4)
SEEDS = (111000000, 112000000, 113000000)
COHORT = "95af3eebd9416ec643b7d698d061b7dac3fe0d52311dd888293609bada629acd"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path):
    return json.loads(path.read_text())


def read_csv(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def number(value, *, optional=False):
    if optional and value in (None, ""):
        return None
    result = float(value)
    require(math.isfinite(result), f"Nonfinite numeric input: {value!r}")
    return result


def pooled_signal(rows):
    """Pool RMS squares, never ratios; zero denominators are undefined."""
    counts = [int(row["element_count"]) for row in rows]
    require(bool(counts) and all(n > 0 for n in counts), "Invalid element counts")
    result = {"element_count": sum(counts), "batch_count": len(rows)}
    for label, field in (("D", "centered_half_difference_rms"),
                         ("P", "endpoint_rms_P"), ("F", "free_rms")):
        values = [number(row[field]) for row in rows]
        require(all(v >= 0 for v in values), f"Negative RMS: {field}")
        result[label] = math.sqrt(math.fsum(n*v*v for n, v in zip(counts, values))/sum(counts))
    result["D_over_P"] = result["D"]/result["P"] if result["P"] else None
    result["D_over_F"] = result["D"]/result["F"] if result["F"] else None
    return result


def cosine_summary(rows, field, draws):
    """A missing cosine makes its plotted cell a gap, with explicit counts."""
    means, defined = [], []
    for draw in draws:
        selected = [r for r in rows if int(r["noise_draw_index"]) == draw]
        require(sorted(int(r["batch_index"]) for r in selected) == list(range(12)),
                f"Missing/duplicate gradient batches for draw {draw}")
        values = [number(r[field], optional=True) for r in selected]
        require(all(v is None or -1-1e-10 <= v <= 1+1e-10 for v in values), "Invalid cosine")
        finite = [v for v in values if v is not None]
        defined.append(len(finite))
        means.append(math.fsum(finite)/12 if len(finite) == 12 else None)
    complete = all(v is not None for v in means)
    return dict(mean=math.fsum(means)/len(means) if complete else None,
                draw_min=min(means) if complete else None,
                draw_max=max(means) if complete else None,
                defined_count=sum(defined), undefined_count=12*len(draws)-sum(defined),
                expected_count=12*len(draws), draw_count=len(draws),
                draw_means=json.dumps(means), defined_counts_by_draw=json.dumps(defined))


def exact_cells(rows, expected, key, label):
    observed = [key(row) for row in rows]
    require(len(observed) == len(set(observed)), f"Duplicate {label} cells")
    missing, extra = expected-set(observed), set(observed)-expected
    require(not missing and not extra,
            f"Incomplete {label}: {len(missing)} missing, {len(extra)} extra cells")


def validate_cosine(row, field, reference):
    norm, refnorm = number(row["eqprop_rms"]), number(row[reference])
    require(norm >= 0 and refnorm >= 0, "Negative gradient RMS")
    value = number(row[field], optional=True)
    require((value is None) == (norm == 0 or refnorm == 0), "Undefined cosine/zero norm mismatch")
    require(value is None or -1-1e-10 <= value <= 1+1e-10, "Invalid cosine")


def validate_case(path, architecture, scheme, pairing):
    depth = int(architecture[-1]); target = TARGETS[architecture]; iterations = 2*depth+2
    manifest = read_json(path/"manifest.json")
    cfg, declared = manifest["configuration"]["resolved"], manifest["inputs"]
    result, status = read_json(path/"result.json"), read_json(path/"status.json")
    require(status["state"] == "complete" and result["completion"]["criteria_met"], f"Incomplete {path}")
    require(result["manifest_sha256"] == sha256(path/"manifest.json"), "Manifest hash mismatch")
    hashes = {}
    for name in ("state_signal.csv", "layer_metrics.csv", "matching.json", "read_only_guards.json", "endpoint_read_noise.csv"):
        digest = sha256(path/name); hashes[name] = digest
        artifact = [a for a in result["artifacts"] if Path(a["path"]).name == name]
        require(len(artifact) == 1 and artifact[0]["sha256"] == digest, f"Artifact hash mismatch: {path/name}")
    require(declared["architecture"] == architecture and declared["scheme"] == scheme and
            declared["relative_target"] == target, "Case identity mismatch")
    require(cfg["checkpoint_epoch"] == 0 and cfg["checkpoint_role"] == "reconstructed_initialization",
            "Not an initialization replay")
    require(cfg["expected_batches"] == 36 and cfg["noisy_batches"] == 12 and
            tuple(cfg["read_noise_seeds"]) == SEEDS, "Cohort/noise contract mismatch")
    require(cfg["expected_cohort_sha256"] == COHORT, "Unexpected cohort")
    require(not manifest["official_test_read"] and not manifest["optimizer_steps_applied"], "Not read-only validation")
    source_path = ROOT/declared["source_config"]
    require(sha256(source_path) == declared["source_config_sha256"], "Source config hash mismatch")
    source = read_json(source_path); model = source["model_base"]
    require(source["seed"] == 0 and (model["voltage_amp"], model["current_amp"]) == SCHEMES[scheme],
            "Seed/amplification mismatch")
    require(model["num_iterations_inference"] == model["num_iterations_training"] == iterations and
            model["input_gain"] == {1:40, 2:100, 3:360}[depth] and
            model["non_linearity"] == "perfect_diode", "Model contract mismatch")
    cohort_path = path.parent.parent/"cohort.json"
    cohort = read_json(cohort_path)
    require(cohort["cohort_sha256"] == COHORT and cohort["example_count"] == 576 and
            cohort["batch_count"] == 36 and cohort["batch_size"] == 16 and not cohort["official_test_read"],
            "Recorded cohort mismatch")
    require([b["batch_index"] for b in cohort["batches"]] == list(range(36)) and
            all(len(b["source_indices"]) == 16 for b in cohort["batches"]), "Cohort batch coverage mismatch")
    guards = read_json(path/"read_only_guards.json")
    require(len(guards) == 36, "Incomplete read-only guards")
    tensor_hash = declared["reconstructed_initialization_tensor_sha256"]
    for guard in guards:
        for field in ("parameter_values_unchanged_after_cast", "frozen_current_force_matches_post_T_cost_gradient",
                      "frozen_current_force_unchanged_across_phases", "iteration_contract_passed"):
            require(guard[field] is True, f"Failed guard: {field}")
        require(guard["reconstructed_initialization_tensor_sha256"] == tensor_hash and
                guard["precision"] == "float64" and guard["eqprop_variant"] == "centered" and
                guard["nudging_mode"] == "current" and guard["runtime_gradient_iterations_K"] == iterations,
                "Initialization/gradient contract mismatch")
        require(guard["bptt_start_state_sha256"] == guard["common_post_T_state_sha256"] ==
                guard["positive_start_state_sha256"] == guard["negative_start_state_sha256"],
                "Unpaired BPTT/EqProp start state")
    matching = read_json(path/"matching.json")
    require(matching["matched"] and matching["relative_target"] == target and
            abs(number(matching["actual_relative_output_rms"])/target-1) <= .01 and
            matching["matching_examples"] == 576 and matching["cohort_sha256"] == COHORT, "Calibration failed")
    states = read_csv(path/"state_signal.csv")
    overrides = source["model_overrides"][source["lab"]["model_key"]]
    shapes = overrides["layer_shapes"][1:]
    exact_cells(states, {(b,l) for b in range(36) for l in range(depth+1)},
                lambda r: (int(r["batch_index"]),int(r["state_layer_index"])), "state")
    for row in states:
        require(int(row["element_count"]) == 16*math.prod(shapes[int(row["state_layer_index"])]),
                "State element count does not match the 16-example batch")
        positive, negative = number(row["positive_rms"]), number(row["negative_rms"])
        endpoint, displacement = number(row["endpoint_rms_P"]), number(row["centered_half_difference_rms"])
        require(positive >= 0 and negative >= 0 and
                math.isclose(endpoint, math.hypot(positive,negative)/math.sqrt(2),rel_tol=1e-10,abs_tol=1e-15),
                "Endpoint pooling mismatch")
        require(0 <= displacement <= endpoint*(1+1e-10), "Unphysical centered displacement")
    gradients = [r for r in read_csv(path/"layer_metrics.csv")
                 if number(r["noise_level"]) in LEVELS and int(r["batch_index"]) < 12]
    expected = {(b,l,eta,d) for b in range(12) for l in range(depth+1)
                for eta in LEVELS for d in ((-1,) if eta == 0 else range(3))}
    exact_cells(gradients, expected,
                lambda r: (int(r["batch_index"]),int(r["state_layer_index"]),number(r["noise_level"]),int(r["noise_draw_index"])), "gradient")
    for row in states+gradients:
        require(row["architecture"] == architecture and row["scheme"] == scheme and
                number(row["relative_target"]) == target and row["matched"] == "True", "CSV identity mismatch")
    for row in gradients:
        clean = number(row["noise_level"]) == 0
        require(row["noise_model"] == ("clean" if clean else "nodewise_relative") and
                (row["noise_seed"] == "" if clean else int(row["noise_seed"]) == SEEDS[int(row["noise_draw_index"])]),
                "Noise seed/model mismatch")
        layer = int(row["state_layer_index"])
        require(row["parameter_name"] == (f"ConvWeight_{layer}" if layer < depth else "DenseWeight_0"),
                "Parameter order mismatch")
        for field, reference in (("cosine", "bptt_rms"), ("noisy_clean_eqprop_cosine", "clean_eqprop_rms")):
            validate_cosine(row, field, reference)
    noise = [r for r in read_csv(path/"endpoint_read_noise.csv")
             if number(r["noise_level"]) in LEVELS[1:] and int(r["batch_index"]) < 12]
    exact_cells(noise, {(b,l,e,d,p) for b in range(12) for l in range(depth+1)
                       for e in LEVELS[1:] for d in range(3) for p in ("positive","negative")},
                lambda r: (int(r["batch_index"]),int(r["state_layer_index"]),number(r["noise_level"]),
                           int(r["noise_draw_index"]),r["phase"]), "endpoint-noise")
    for row in noise:
        require(int(row["noise_seed"]) == SEEDS[int(row["noise_draw_index"])], "Endpoint seed mismatch")
        key = (architecture, int(row["batch_index"]), int(row["state_layer_index"]), row["phase"], int(row["noise_seed"]))
        digest = row["standard_normal_sha256"]
        require(pairing.setdefault(key, digest) == digest, "Unpaired Gaussian noise")
    summary = []
    for layer in range(depth+1):
        pooled = pooled_signal([r for r in states if int(r["state_layer_index"]) == layer])
        summary.append(dict(architecture=architecture, scheme=scheme, relative_target=target,
                            layer_index=layer, layer_name=f"Conv{layer+1}" if layer < depth else "Readout", **pooled))
    require(math.isclose(summary[-1]["D_over_F"], matching["actual_relative_output_rms"], rel_tol=1e-10),
            "Recomputed calibration mismatch")
    provenance = dict(architecture=architecture, scheme=scheme, case_path=str(path), resolved_path=str(path.resolve()),
                      reused_reference=scheme in ("baseline","ours","legacy"),
                      file_sha256={**hashes, "manifest.json":sha256(path/"manifest.json"), "result.json":sha256(path/"result.json"),
                                   "cohort.json":sha256(cohort_path)},
                      source_config=str(source_path), source_config_sha256=declared["source_config_sha256"],
                      cohort_batches_sha256=hashlib.sha256(json.dumps(cohort["batches"],sort_keys=True).encode()).hexdigest(),
                      source_contract=cfg["source_contract"], gradient_contract=cfg["gradient_contract"],
                      initialization_tensor_sha256=tensor_hash, initializer_checkpoint_sha256=declared["initializer_checkpoint_sha256"],
                      runtime=manifest["runtime"], runner_identity=manifest["git"])
    return summary, gradients, provenance


def write_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def plot(states, gradients, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = dict(zip(SCHEMES, ("#333333", "#2878b5", "#39a077", "#d95f02", "#9b59b6")))
    markers = dict(zip(SCHEMES, ("o", "s", "^", "D", "x")))
    cosine_lower = min([-.08] + [r["draw_min"]-.03 for r in gradients if r["draw_min"] is not None])
    def panel(ax, architecture, rows, field, eta=None):
        depth = int(architecture[-1])
        for scheme, (v,c) in SCHEMES.items():
            subset = sorted([r for r in rows if r["architecture"] == architecture and r["scheme"] == scheme and
                             (eta is None or r["noise_level"] == eta)], key=lambda r:r["layer_index"])
            x = [r["layer_index"] for r in subset]
            y = [r[field] if r[field] is not None and (eta is not None or r[field] > 0) else math.nan for r in subset]
            ax.plot(x, y, linestyle="-", marker=markers[scheme], color=colors[scheme],
                    markerfacecolor="none" if scheme == "baseline" else colors[scheme],
                    label=f"({v:g}, {c:g})", markersize=5)
            if eta is not None:
                ax.fill_between(x, [r["draw_min"] if r["draw_min"] is not None else math.nan for r in subset],
                                [r["draw_max"] if r["draw_max"] is not None else math.nan for r in subset],
                                color=colors[scheme], alpha=.14)
                missing = sum(r["undefined_count"] for r in subset)
                if missing:
                    ax.text(.02, .04+.07*list(SCHEMES).index(scheme), f"({v:g},{c:g}): {missing} undefined",
                            transform=ax.transAxes, fontsize=7, color=colors[scheme])
            else:
                omitted = sum(r[field] is None or r[field] <= 0 for r in subset)
                if omitted:
                    ax.text(.02,.04+.07*list(SCHEMES).index(scheme), f"({v:g},{c:g}): {omitted} zero/undefined",
                            transform=ax.transAxes, fontsize=7, color=colors[scheme])
        ax.set_xticks(range(depth+1), [f"Conv{i+1}" for i in range(depth)]+["Readout"])
        ax.grid(alpha=.2)
        if eta is None:
            ax.set_yscale("log")
        else:
            ax.set_ylim(cosine_lower, 1.035); ax.axhline(0, color="grey", lw=.5)
    def save(fig, name):
        fig.savefig(output/name, dpi=180, bbox_inches="tight"); plt.close(fig)
    for field, label in (("D_over_P", "Normalized displacement D/P"), ("D", "Absolute displacement D (V)"),
                         ("F", "Free-state RMS voltage F (V)"), ("P", "Nudged endpoint RMS voltage P (V)")):
        voltage = field in ("F", "P")
        heading = {"F": "Free-state voltage magnitude: F = sqrt(mean(v_free²))",
                   "P": "Nudged voltage magnitude: P = sqrt(mean((v_plus² + v_minus²)/2))"}.get(field)
        fig, axes = plt.subplots(1,3,figsize=(13,4.4))
        if voltage:
            # Identical scales permit comparisons across architectures as well as schemes.
            values = [r[field] for r in states if r[field] > 0]
            limits = (min(values)/1.5, max(values)*1.5)
        for ax, architecture in zip(axes, TARGETS):
            panel(ax,architecture,states,field); ax.set_title(f"{architecture.upper()} — output D/F={TARGETS[architecture]}")
            ax.set_ylabel(label)
            one, single = plt.subplots(figsize=(5.3,4.4)); panel(single,architecture,states,field)
            single.set_title(f"{architecture.upper()} — output D/F={TARGETS[architecture]}"); single.set_ylabel(label)
            if voltage:
                ax.set_ylim(*limits); single.set_ylim(*limits)
                ax.set_xlabel("Layer"); single.set_xlabel("Layer")
                one.suptitle("Seed 0 initialization · 576 examples", fontsize=10)
            single.legend(title="(voltage, current)", fontsize=8); one.tight_layout(); save(one,f"{field}_{architecture}.jpg")
        axes[-1].legend(title="(voltage, current)", fontsize=8)
        subtitle = "Seed 0 initialization · pooled over all nodes and 576 examples · logarithmic voltage axis" if voltage else "Seed 0 initialization · pooled over 576 examples"
        fig.suptitle(f"{heading}\n{subtitle}" if heading else subtitle, fontsize=12)
        fig.tight_layout(); save(fig,f"{field}_combined.jpg")
    for reference in ("finite_K_BPTT", "clean_EqProp"):
        rows = [r for r in gradients if r["reference"] == reference]
        fig, axes = plt.subplots(3,5,figsize=(19,10),squeeze=False)
        for i, architecture in enumerate(TARGETS):
            one, singles = plt.subplots(1,5,figsize=(19,3.8))
            for j, eta in enumerate(LEVELS):
                for ax in (axes[i,j],singles[j]):
                    panel(ax,architecture,rows,"mean",eta); ax.set_title(f"{architecture.upper()} · η={eta:g}")
                    if j == 0: ax.set_ylabel(f"Cosine to own {reference.replace('_',' ')}")
            singles[-1].legend(title="(voltage, current)",fontsize=8)
            one.suptitle(f"Output D/F={TARGETS[architecture]} · 192 paired examples · band: min–max of three draw means (not CI)")
            one.tight_layout(); save(one,f"cosine_{reference}_{architecture}.jpg")
        axes[0,-1].legend(title="(voltage, current)",fontsize=8)
        fig.suptitle("Seed 0 initialization · output D/F=1/4/6 for Conv1/2/3 · 192 paired examples · band: three-draw range (not CI)")
        fig.tight_layout(); save(fig,f"cosine_{reference}_combined.jpg")


def analyze(study_root, output):
    states, gradients, provenance, pairing = [], [], [], {}
    reference_root = study_root.parent/REFERENCE
    for architecture, target in TARGETS.items():
        identities = set()
        for scheme in SCHEMES:
            lane = reference_root/"stage_a"/architecture if scheme in ("baseline","ours","legacy") else study_root/"production"/architecture
            path = lane/"cases"/f"{architecture}_{scheme}_relative{target}"
            signal, rows, source = validate_case(path, architecture, scheme, pairing)
            states.extend(signal); provenance.append(source)
            identities.add((source["initialization_tensor_sha256"], source["initializer_checkpoint_sha256"]))
            for layer in range(int(architecture[-1])+1):
                for eta in LEVELS:
                    selected = [r for r in rows if int(r["state_layer_index"]) == layer and number(r["noise_level"]) == eta]
                    for ref, field in (("finite_K_BPTT","cosine"), ("clean_EqProp","noisy_clean_eqprop_cosine")):
                        gradients.append(dict(architecture=architecture, scheme=scheme, relative_target=target,
                                              layer_index=layer, noise_level=eta, reference=ref,
                                              **cosine_summary(selected,field,(-1,) if eta == 0 else range(3))))
        require(len(identities) == 1, f"Unpaired initialization: {architecture}")
    require(len({p["cohort_batches_sha256"] for p in provenance}) == 1, "Unpaired cohort payloads/source indices")
    for field in ("source_contract", "gradient_contract"):
        require(len({json.dumps(p[field],sort_keys=True) for p in provenance}) == 1, f"Unlike {field} across cases")
    # Publish only after every required case passes. Never report partial success.
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output/"state_profiles.csv", states); write_csv(output/"gradient_profiles.csv", gradients)
    (output/"reference_provenance.json").write_text(json.dumps(dict(
        cases=provenance, cohort_sha256=COHORT, historical_cross_device_equivalence_checked=False,
        analysis_source_sha256=sha256(Path(__file__))), indent=2)+"\n")
    plot(states, gradients, output)
    (output/"validation.json").write_text(json.dumps(dict(coverage_complete=True, validated_cases=len(provenance),
        state_examples=576, gradient_examples=192, selected_noise_levels=LEVELS, noise_seeds=SEEDS,
        paired_gaussian_keys=len(pairing), undefined_cosines=sum(r["undefined_count"] for r in gradients),
        bands="minimum and maximum of three batch-mean draw values; not confidence intervals",
        undefined_policy="any undefined batch/draw makes the corresponding plotted summary a gap; counts in CSV",
        evidence_class="diagnostic initialization replay, own finite-K BPTT reference",
        historical_cross_device_equivalence_checked=False), indent=2)+"\n")
    return len(provenance)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-root", type=Path, default=ROOT/"results"/STUDY)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    output = args.output_dir or args.study_root/"analysis"
    count = analyze(args.study_root.resolve(),output.resolve())
    print(f"Validated {count}/15 cases; plots and CSV: {output}")


if __name__ == "__main__":
    main()
