"""Replot saved free-state voltage RMS; keep EqProp and BPTT trajectories separate."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


EQPROP_ROOT = Path("results/cifar-ours-epoch-beta-gradients-20260930-v1")
BPTT_ROOT = Path("results/cifar-block-rms-intermediate-20260929-v1")
EQPROP_EPOCHS = (0, 1, 5)
BPTT_EPOCHS = (0, 10, 30, 50)
SCHEMES = ("baseline", "ours", "legacy")
DEPTHS = {1: 3, 2: 3, 3: 2}
COLORS = ("#0072B2", "#D55E00", "#009E73")


def read(path):
    return json.loads(path.read_text())


def close(a, b):
    return math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-14)


def base_row(trajectory, scheme, epoch, block, scope, rms, elements, source, checkpoint, layer=None):
    assert math.isfinite(rms) and rms > 0 and elements > 0
    return dict(trajectory=trajectory, scheme=scheme, epoch=epoch, block=block, scope=scope,
                layer=layer, state_rms=rms, elements=elements, state_mean=None, state_std=None,
                source=str(source), checkpoint=str(checkpoint))


def add_changes(rows):
    initial = {(r["scheme"], r["block"], r["scope"], r["layer"]): r["state_rms"]
               for r in rows if r["epoch"] == 0}
    for row in rows:
        zero = initial[row["scheme"], row["block"], row["scope"], row["layer"]]
        row["delta_from_initial"] = row["state_rms"] - zero
        row["ratio_to_initial"] = row["state_rms"] / zero
    return rows


def load_eqprop(root):
    indices = read(root / "inputs/cohort.json")["indices"]
    assert len(indices) == len(set(indices)) == 256
    provenance = read(root / "analysis/input_provenance.json")
    rows, batches = [], []
    for epoch in EQPROP_EPOCHS:
        path = root / f"cells/e{epoch}/artifacts/comparison.json"
        data = read(path)
        assert len(data["layers"]) == 136 and len(data["batches"]) == 1088
        checkpoint = provenance[str(epoch)]["source"]
        selected = {}
        for block, depth in DEPTHS.items():
            for layer in range(1, depth + 1):
                cases = sorted((r for r in data["layers"] if r["block"] == block and r["layer"] == layer),
                               key=lambda r: r["case_index"])
                assert len(cases) == 17 and {r["case_index"] for r in cases} == set(range(17))
                state = cases[0]["state_displacement"]
                assert all(close(r["state_displacement"][field], state[field])
                           for r in cases for field in ("free_squares", "elements", "free_rms"))
                assert close(state["free_rms"], math.sqrt(state["free_squares"] / state["elements"]))
                batch_rows = [r for r in data["batches"] if r["case_index"] == 0
                              and r["block"] == block and r["layer"] == layer]
                assert len(batch_rows) == 8 and {r["batch"] for r in batch_rows} == set(range(8))
                assert state["elements"] == sum(r["state_displacement"]["elements"] for r in batch_rows)
                assert close(state["free_squares"], math.fsum(r["state_displacement"]["free_squares"] for r in batch_rows))
                selected[block, layer] = state
                rows.append(base_row("EqProp", "ours", epoch, block, "layer", state["free_rms"],
                                     state["elements"], path, checkpoint, layer))
                batches.append(dict(epoch=epoch, block=block, layer=layer,
                                    batch_rms=[r["state_displacement"]["free_rms"] for r in sorted(batch_rows, key=lambda r:r["batch"])]))
            states = [selected[block, layer] for layer in range(1, depth + 1)]
            count = sum(s["elements"] for s in states)
            all_rms = math.sqrt(math.fsum(s["free_squares"] for s in states) / count)
            rows.append(base_row("EqProp", "ours", epoch, block, "block_all_layers", all_rms,
                                 count, path, checkpoint))
            last = states[-1]
            output = next(r["output_displacement"] for r in data["layers"]
                          if r["case_index"] == 0 and r["block"] == block)
            assert close(last["free_rms"], output["free_rms"])
            rows.append(base_row("EqProp", "ours", epoch, block, "block_output", last["free_rms"],
                                 last["elements"], path, checkpoint))
    assert len(rows) == 42
    return add_changes(rows), indices, provenance, batches


def load_bptt(root, indices):
    rows, provenance = [], []
    for scheme in SCHEMES:
        for epoch in BPTT_EPOCHS:
            source = (root.parent / "cifar-block-rms-expanded-20260929-v1" / scheme if epoch == 0 else
                      root.parent / "cifar-block-rms-e50-20260929-v1" / scheme if epoch == 50 else
                      root / f"e{epoch}" / scheme)
            assert read(source / "inputs/cohort.json")["indices"] == indices
            config = read(source / "inputs/model_config.json")
            assert config["iterations"] == [6, 6, 4] and config["batch_size"] == 32
            summary = read(source / "analysis/summary.json")
            manifest = read(source / "run/manifest.json")
            assert manifest["environment"] == {"cuda": "12.8", "torch": "2.11.0+cu128"}
            checkpoint = source / manifest.get("checkpoint_file", "inputs/initial_model.pt")
            provenance.append(dict(scheme=scheme, epoch=epoch, checkpoint=str(checkpoint),
                                   recorded_checkpoint_sha256=manifest.get("checkpoint_sha256"), source=str(source)))
            for block in DEPTHS:
                cases = [r for r in summary["layers"] if r["block"] == block]
                output = cases[0]["output_displacement"]
                assert all(close(r["output_displacement"]["free_rms"], output["free_rms"]) for r in cases)
                rows.append(base_row("BPTT", scheme, epoch, block, "block_output", output["free_rms"],
                                     output["elements"], source / "analysis/summary.json", checkpoint))
    assert len(rows) == 36
    return add_changes(rows), provenance


def draw_lines(ax, points, label, color, metric):
    points = sorted(points, key=lambda r: r["epoch"])
    ax.plot([r["epoch"] for r in points], [r[metric] for r in points], "o-", color=color,
            markersize=5, linewidth=1.9, label=label)
    ax.set_yscale("log")
    ax.grid(alpha=.2, which="major")


def layer_plot(rows, out):
    fig, axes = plt.subplots(2, 3, figsize=(14.5, 8.5), sharex=True, sharey="row")
    for col, (block, depth) in enumerate(DEPTHS.items()):
        for layer in range(1, depth + 1):
            points = [r for r in rows if r["scope"] == "layer" and r["block"] == block and r["layer"] == layer]
            label = f"Conv {layer}" + (" = block output" if layer == depth else "")
            for row, metric in enumerate(("state_rms", "ratio_to_initial")):
                draw_lines(axes[row, col], points, label, COLORS[layer - 1], metric)
        axes[0, col].set_title(f"Block {block}", fontsize=14)
        axes[0, col].legend(frameon=False, fontsize=10)
        for row in range(2):
            axes[row, col].set_xticks(EQPROP_EPOCHS)
        axes[1, col].axhline(1, color="#777777", linestyle=":", linewidth=.8)
        axes[1, col].set_xlabel("EqProp training epoch")
    axes[0, 0].set_ylabel("Free-state voltage RMS (simulation units)")
    axes[1, 0].set_ylabel("Voltage RMS / initialization RMS")
    fig.suptitle("How every layer's voltage magnitude changes during EqProp training\nOurs · saved checkpoints 0, 1 and 5", fontsize=16, y=.985)
    fig.text(.5, .025, "Raw physical free states before differential decoding, pooling and BN.\n"
             "Same 256 images · count-weighted RMS · T = [6, 6, 4] · dots are measured checkpoints; lines only guide the eye.",
             ha="center", fontsize=10, linespacing=1.5)
    fig.subplots_adjust(top=.855, bottom=.14, left=.075, right=.985, hspace=.22, wspace=.15)
    fig.savefig(out / "eqprop_layer_voltage_rms.jpg", dpi=200)
    plt.close(fig)


def block_plot(rows, out):
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.4), sharey=True)
    for ax, scope, title in zip(axes, ("block_output", "block_all_layers"),
                               ("Block output (last convolution)", "All internal nodes in each block")):
        for block in DEPTHS:
            points = [r for r in rows if r["scope"] == scope and r["block"] == block]
            draw_lines(ax, points, f"Block {block}", COLORS[block - 1], "state_rms")
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("EqProp training epoch")
        ax.set_xticks(EQPROP_EPOCHS)
        ax.legend(frameon=False)
    axes[0].set_ylabel("Free-state voltage RMS (simulation units)")
    fig.suptitle("Ours · voltage magnitude by block", fontsize=16, y=.98)
    fig.text(.5, .025, "All-node block RMS = sqrt(sum of layer squared voltages / total node count).\n"
             "Same 256 images at each saved checkpoint; physical states before decoding and BN.", ha="center", fontsize=10)
    fig.subplots_adjust(top=.82, bottom=.19, left=.085, right=.98, wspace=.2)
    fig.savefig(out / "eqprop_block_voltage_rms.jpg", dpi=200)
    plt.close(fig)


def bptt_plot(rows, out):
    fig, axes = plt.subplots(2, 3, figsize=(14.5, 8.3), sharex=True, sharey="row")
    for col, scheme in enumerate(SCHEMES):
        for block in DEPTHS:
            points = [r for r in rows if r["scheme"] == scheme and r["block"] == block]
            for row, metric in enumerate(("state_rms", "ratio_to_initial")):
                draw_lines(axes[row, col], points, f"Block {block}", COLORS[block - 1], metric)
        axes[0, col].set_title(scheme.capitalize(), fontsize=14)
        axes[0, col].legend(frameon=False)
        for row in range(2):
            axes[row, col].set_xticks(BPTT_EPOCHS)
        axes[1, col].set_xlabel("BPTT training epoch")
        axes[1, col].axhline(1, color="#777777", linestyle=":", linewidth=.8)
    axes[0, 0].set_ylabel("Block-output free voltage RMS")
    axes[1, 0].set_ylabel("Voltage RMS / initialization RMS")
    fig.suptitle("Earlier BPTT trajectories · block-output voltage magnitudes\nSeparate evidence from the EqProp epoch-0/1/5 trajectory", fontsize=16, y=.985)
    fig.text(.5, .025, "Same 256 images · T = [6, 6, 4] · physical output before decoding and BN.\n"
             "Internal-layer voltages were not recorded in these older sweeps; only epochs 0, 10, 30 and 50 are shown.",
             ha="center", fontsize=10, linespacing=1.5)
    fig.subplots_adjust(top=.85, bottom=.145, left=.075, right=.985, hspace=.23, wspace=.16)
    fig.savefig(out / "bptt_block_output_voltage_rms.jpg", dpi=200)
    plt.close(fig)


def write_evidence(out, rows, metadata):
    with (out / "voltage_epoch_comparison.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (out / "summary.json").write_text(json.dumps(dict(metadata, measurements=rows), indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eqprop-root", type=Path, default=EQPROP_ROOT)
    parser.add_argument("--bptt-root", type=Path, default=BPTT_ROOT)
    args = parser.parse_args()
    eqroot, bproot = args.eqprop_root.resolve(), args.bptt_root.resolve()
    eq, indices, provenance, batches = load_eqprop(eqroot)
    bp, bp_provenance = load_bptt(bproot, indices)
    for block in DEPTHS:
        a = next(r["state_rms"] for r in eq if r["epoch"] == 0 and r["block"] == block and r["scope"] == "block_output")
        b = next(r["state_rms"] for r in bp if r["epoch"] == 0 and r["block"] == block and r["scheme"] == "ours")
        assert close(a, b)
    common = dict(
        magnitude_definition="sqrt(sum of free-state voltage squares / state-element count); raw physical state before differential decoding, pooling and BN",
        signed_statistics="Signed means, standard deviations and extrema were not saved. They cannot be reconstructed from squared sums and are not inferred from RMS.",
        protocol=dict(examples=256, batch_size=32, free_iterations=[6, 6, 4], solver="asynchronous perfect-diode finite-iteration endpoint",
                      dtype="float32", replay_environment="torch 2.11.0+cu128 / CUDA 12.8", noise=False, augmentation=False,
                      bn="minibatch statistics with running buffers frozen"),
        ordered_cohort_sha256=hashlib.sha256(json.dumps(indices, separators=(",", ":")).encode()).hexdigest(),
        collection="Reuse previously validated replay artifacts. No checkpoint reads, new replay, training or official test access.",
    )
    eqout, bpout = eqroot / "analysis/voltage_evolution", bproot / "analysis/voltage_evolution"
    eqout.mkdir(parents=True, exist_ok=True)
    bpout.mkdir(parents=True, exist_ok=True)
    write_evidence(eqout, eq, dict(common, epochs=list(EQPROP_EPOCHS), missing_epochs="No measurements at epochs 2–4; no later checkpoint claimed.",
                                  checkpoint_provenance=provenance, batch_layer_rms=batches,
                                  validation="24 layers; eight-batch squared-sum pooling and 17-beta invariance verified; nine block aggregates and nine block outputs derived."))
    write_evidence(bpout, bp, dict(common, epochs=list(BPTT_EPOCHS), missing_epochs="Only 0/10/30/50; internal layer statistics unavailable.",
                                  checkpoint_provenance=bp_provenance,
                                  validation="36 block-output values; identical ordered cohort, iteration counts and runtime checked; repeated beta/layer free RMS invariant."))
    layer_plot(eq, eqout)
    block_plot(eq, eqout)
    bptt_plot(bp, bpout)
    for block, depth in DEPTHS.items():
        for layer in range(1, depth + 1):
            points = [r for r in eq if r["scope"] == "layer" and r["block"] == block and r["layer"] == layer]
            print(f"B{block} C{layer}: " + "; ".join(f"e{r['epoch']} {r['state_rms']:.6g} ({r['ratio_to_initial']:.4g}x init)" for r in points))
    print(f"Saved 42 EqProp and 36 BPTT measurements plus 3 JPGs under {eqout} and {bpout}")


if __name__ == "__main__":
    main()
