#!/usr/bin/env python3
"""Reproducible exploratory random-nudge diagnostics and toy learning.

Run from the repository root with ``python -m labs.tools.run_random_nudge_hopfield``.
Uses NumPy/Matplotlib only; outputs live in ignored run directories.
"""

import argparse
import csv
import json
import os
from pathlib import Path
import platform
import shlex
import subprocess
import sys
import time
from datetime import datetime, timezone

import numpy as np

from labs.random_nudge_hopfield import (
    add_read_noise, corrected_feedback, drive_parameter_gradient, exact_feedback,
    hadamard_probes, make_network, measure_response, mismatch_projections,
    rademacher, relax,
)


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def mean_se(values):
    values = np.asarray(values)
    return float(values.mean()), float(values.std(ddof=1) / np.sqrt(len(values)))


def ratio(numerator, denominator):
    return float(numerator / denominator) if denominator > 1e-24 else None


def cosine(a, b):
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.sum(a * b) / denominator) if denominator > 1e-24 else None


def problem(size, alpha, seed, cubic):
    network = make_network(size, alpha, seed=seed, cubic=cubic)
    rng = np.random.default_rng(1000 + seed)
    drive = rng.normal(scale=0.35, size=size)
    free = relax(network, drive)
    c = np.zeros(size)
    c[-2:] = free.state[-2:] - np.array([0.6, -0.4])
    q, adjoint = exact_feedback(network, free.state, c)
    return network, drive, free, c, q, adjoint


def ideal_sweep(config, output, progress):
    """Oracle linear responses isolate sampling variance from finite nudging."""
    rows = []
    for size in config["sizes"]:
        for cubic in (0.0, 0.25):
            for alpha in config["asymmetries"]:
                for seed in config["seeds"]:
                    net, _, free, c, q, adjoint = problem(size, alpha, seed, cubic)
                    j = net.jacobian(free.state)
                    rng = np.random.default_rng(2000 + seed + size)
                    z = rademacher(rng, (config["ideal_trials"], max(config["ideal_probes"]), size))
                    # Explicitly an ORACLE control: actual equilibrations are in finite_sweep.
                    r = np.linalg.solve(j, z.reshape(-1, size).T).T.reshape(z.shape)
                    d = mismatch_projections(c, q, z, r)
                    delta = adjoint - q
                    delta_sq = float(delta @ delta)
                    base = dict(size=size, cubic=cubic, asymmetry=alpha, seed=seed,
                                trials=config["ideal_trials"], delta_sq=delta_sq,
                                eqprop_relative_error=float(np.linalg.norm(delta) / np.linalg.norm(adjoint)),
                                jacobian_asymmetry=float(np.linalg.norm(j - j.T) / np.linalg.norm(j)),
                                stability_max_real=float(np.linalg.eigvals(j).real.max()),
                                projection_max_abs_error=float(np.max(np.abs(d - z @ delta))))
                    for m in config["ideal_probes"]:
                        diagnostic, diagnostic_se = mean_se(np.mean(d[:, :m]**2, axis=-1))
                        for method, scale in (("iid", 1.0), ("iid_shrunk", m / (m + size - 1))):
                            estimate = corrected_feedback(q, z[:, :m], d[:, :m], scale)
                            mse, mse_se = mean_se(np.sum((estimate - adjoint)**2, axis=-1))
                            theory = ((1 - scale)**2 + scale**2 * (size - 1) / m) * delta_sq
                            rows.append(dict(base, method=method, probes=m, shrinkage=scale,
                                             diagnostic=diagnostic, diagnostic_se=diagnostic_se,
                                             diagnostic_ratio=ratio(diagnostic, delta_sq), mse=mse,
                                             mse_se=mse_se, theory_mse=theory, mse_theory_ratio=ratio(mse, theory),
                                             mse_over_eqprop=ratio(mse, delta_sq)))
                    h = hadamard_probes(size)
                    rh = np.linalg.solve(j, h.T).T
                    dh = mismatch_projections(c, q, h, rh)
                    estimate = corrected_feedback(q, h, dh)
                    mse = float(np.sum((estimate - adjoint)**2))
                    rows.append(dict(base, method="orthogonal", probes=size, shrinkage=1.0,
                                     diagnostic=float(np.mean(dh**2)), mse=mse, theory_mse=0.0,
                                     mse_over_eqprop=ratio(mse, delta_sq)))
                    progress("ideal", f"n={size}, cubic={cubic}, alpha={alpha}, seed={seed}")
    write_csv(output / "ideal.csv", rows)
    return rows


def finite_sweep(config, output, progress):
    """Physical forward relaxations; noise is independent across repeated reads."""
    rows = []
    size = 8
    trials, max_m = config["finite_trials"], max(config["finite_probes"])
    for alpha in (0.0, 1.5):
        for seed in config["seeds"]:
            net, drive, free, c, q_exact, adjoint = problem(size, alpha, seed, 0.25)
            delta = adjoint - q_exact
            delta_sq = float(delta @ delta)
            rng = np.random.default_rng(3000 + seed)
            z = rademacher(rng, (trials, max_m, size))
            h = hadamard_probes(size)
            for scheme in ("central", "forward"):
                for beta in config["betas"]:
                    qm = measure_response(net, drive, free.state, c, beta, scheme=scheme)
                    rm = measure_response(net, drive, free.state, z, beta, scheme=scheme)
                    hm = measure_response(net, drive, free.state, h, beta, scheme=scheme)
                    d_clean = mismatch_projections(c, qm.value, z, rm.value)
                    base = dict(size=size, cubic=0.25, asymmetry=alpha, seed=seed, scheme=scheme,
                                beta=beta, trials=trials, delta_sq=delta_sq,
                                q_relative_bias=float(np.linalg.norm(qm.value - q_exact) / np.linalg.norm(q_exact)),
                                clean_projection_rmse=float(np.sqrt(np.mean((d_clean - z @ delta)**2))),
                                max_force_residual=max(free.residual, qm.residual, rm.residual, hm.residual),
                                batch_relaxation_iterations=qm.iterations + rm.iterations + hm.iterations)
                    for noise_index, sigma in enumerate(config["read_noise"]):
                        # Same random numbers across beta isolate the amplitude/noise tradeoff.
                        noise_rng = np.random.default_rng(4000 + 10 * seed + noise_index)
                        q = add_read_noise(np.broadcast_to(qm.value, (trials, size)), sigma, beta, noise_rng, scheme)
                        r = add_read_noise(rm.value, sigma, beta, noise_rng, scheme)
                        d = mismatch_projections(c, q, z, r)
                        q_mse = float(np.mean(np.sum((q - adjoint)**2, axis=-1)))
                        response_variance = sigma**2 / beta**2 * (0.5 if scheme == "central" else 2)
                        noise_floor = response_variance * (float(c @ c) + size)
                        for m in config["finite_probes"]:
                            diagnostic, diagnostic_se = mean_se(np.mean(d[:, :m]**2, axis=-1))
                            for method, scale in (("iid", 1.0), ("iid_shrunk", m / (m + size - 1))):
                                estimate = corrected_feedback(q, z[:, :m], d[:, :m], scale)
                                mse, mse_se = mean_se(np.sum((estimate - adjoint)**2, axis=-1))
                                # Exact noise contribution for ONE q read reused across probes.
                                # M=mean(zz.T); E|| (I-aM) eta_q ||² plus probe-read noise.
                                ideal_mse = ((1-scale)**2 + scale**2*(size-1)/m)*delta_sq
                                noise_mse = response_variance * (
                                    size*((1-scale)**2 + scale**2*(size-1)/m)
                                    + scale**2*size*float(c @ c)/m)
                                rows.append(dict(base, read_noise=sigma, method=method, probes=m,
                                                 diagnostic=diagnostic, diagnostic_se=diagnostic_se,
                                                 diagnostic_ratio=ratio(diagnostic, delta_sq),
                                                 diagnostic_noise_floor=noise_floor,
                                                 diagnostic_debiased=diagnostic-noise_floor,
                                                 mse=mse, mse_se=mse_se, eqprop_measured_mse=q_mse,
                                                 linear_theory_mse=ideal_mse+noise_mse,
                                                 mse_over_eqprop=ratio(mse, q_mse),
                                                 physical_equilibrations_per_estimate=1+(2 if scheme == "central" else 1)*(m+1)))
                        hr = add_read_noise(np.broadcast_to(hm.value, (trials, size, size)), sigma, beta, noise_rng, scheme)
                        hz = np.broadcast_to(h, hr.shape)
                        hd = mismatch_projections(c, q, hz, hr)
                        estimate = corrected_feedback(q, hz, hd)
                        mse, mse_se = mean_se(np.sum((estimate - adjoint)**2, axis=-1))
                        rows.append(dict(base, read_noise=sigma, method="orthogonal", probes=size,
                                         diagnostic=float(np.mean(hd**2)), diagnostic_noise_floor=noise_floor,
                                         diagnostic_debiased=float(np.mean(hd**2))-noise_floor,
                                         mse=mse, mse_se=mse_se, eqprop_measured_mse=q_mse,
                                         linear_theory_mse=response_variance*float(c @ c),
                                         mse_over_eqprop=ratio(mse, q_mse),
                                         physical_equilibrations_per_estimate=1+(2 if scheme == "central" else 1)*(size+1)))
                    progress("finite", f"alpha={alpha}, seed={seed}, {scheme}, beta={beta:g}")
    write_csv(output / "finite.csv", rows)
    return rows


def training_sweep(config, output, progress):
    """Learn only hidden input couplings; outputs cannot bypass feedback.

    Teacher and student share a fixed recurrent W. Targets are teacher free
    equilibria, using disjoint synthetic train/test inputs. All feedback methods
    receive identical starting parameters, inputs and fixed asymmetry per seed.
    Dense adjoints are recorded as audits at each trajectory's current state;
    only the explicitly named 'adjoint' baseline uses them for its updates.
    """
    rows = []
    size, hidden = 8, 6
    checkpoints = output / "checkpoints"
    checkpoints.mkdir()
    for alpha in (0.0, 1.5):
        for seed in config["seeds"]:
            net = make_network(size, alpha, seed=seed)
            data_rng = np.random.default_rng(5000 + seed)
            train_x = data_rng.uniform(-1, 1, size=(config["train_samples"], 3))
            test_x = data_rng.uniform(-1, 1, size=(config["test_samples"], 3))
            teacher = data_rng.normal(scale=0.5, size=(hidden, 3))
            initial = data_rng.normal(scale=0.1, size=(hidden, 3))

            def states(weights, inputs):
                drive = np.zeros((len(inputs), size))
                drive[:, :hidden] = inputs @ weights.T
                return drive, relax(net, drive)

            _, teacher_train = states(teacher, train_x)
            _, teacher_test = states(teacher, test_x)
            train_y = teacher_train.state[:, hidden:]
            test_y = teacher_test.state[:, hidden:]
            for method in config["training_methods"]:
                weights = initial.copy()
                probe_rng = np.random.default_rng(6000 + seed)
                train_equilibrations = 0
                max_residual = 0.0
                for epoch in range(config["epochs"] + 1):
                    drive, free = states(weights, train_x)
                    _, test_free = states(weights, test_x)
                    c = np.zeros_like(free.state)
                    c[:, hidden:] = free.state[:, hidden:] - train_y
                    _, adjoint = exact_feedback(net, free.state, c)
                    exact_gradient = drive_parameter_gradient(adjoint, train_x, hidden)
                    record = dict(asymmetry=alpha, seed=seed, method=method, epoch=epoch,
                                  train_loss=float(np.mean(np.sum(c**2, axis=-1)) / 2),
                                  test_loss=float(np.mean(np.sum((test_free.state[:, hidden:]-test_y)**2, axis=-1)) / 2),
                                  train_equilibrations=train_equilibrations)
                    if epoch == config["epochs"]:
                        record["max_force_residual"] = max(max_residual, free.residual, test_free.residual)
                        rows.append(record)
                        break
                    train_equilibrations += free.equilibrations
                    max_residual = max(max_residual, free.residual, test_free.residual)
                    if method == "adjoint":
                        feedback, m = adjoint, 0
                    else:
                        qm = measure_response(net, drive, free.state, c, config["training_beta"])
                        q = qm.value
                        max_residual = max(max_residual, qm.residual)
                        train_equilibrations += qm.equilibrations
                        if method == "eqprop":
                            feedback, m = q, 0
                        else:
                            m = size if method == "orthogonal" else int(method.removeprefix("mc"))
                            if method == "orthogonal":
                                z = np.broadcast_to(hadamard_probes(size, probe_rng), (len(train_x), size, size))
                            else:
                                z = rademacher(probe_rng, (len(train_x), m, size))
                            rm = measure_response(net, drive[:, None, :], free.state[:, None, :], z,
                                                  config["training_beta"])
                            d = mismatch_projections(c, q, z, rm.value)
                            feedback = corrected_feedback(q, z, d)
                            max_residual = max(max_residual, rm.residual)
                            train_equilibrations += rm.equilibrations
                    gradient = drive_parameter_gradient(feedback, train_x, hidden)
                    record.update(probes=m, gradient_cosine=cosine(gradient, exact_gradient),
                                  gradient_relative_error=ratio(np.linalg.norm(gradient-exact_gradient),
                                                                np.linalg.norm(exact_gradient)),
                                  max_force_residual=max_residual)
                    rows.append(record)
                    weights -= config["learning_rate"] * gradient
                    if not np.isfinite(weights).all():
                        raise RuntimeError(f"Non-finite learning parameters: {alpha}, {seed}, {method}, {epoch}")
                    if epoch % 10 == 0:
                        progress("train", f"alpha={alpha}, seed={seed}, {method}, epoch={epoch}, loss={record['train_loss']:.6g}", advance=False)
                np.savez_compressed(checkpoints / f"alpha{alpha:g}_seed{seed}_{method}.npz",
                                    weights=weights, initial_weights=initial, teacher_weights=teacher,
                                    recurrent=net.weights, cubic=net.cubic, train_x=train_x, train_y=train_y,
                                    test_x=test_x, test_y=test_y)
                write_csv(output / "training.csv", rows)
                progress("train", f"alpha={alpha}, seed={seed}, {method}, final test loss={record['test_loss']:.6g}")
    return rows


def make_plots(output, ideal, finite, training):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    ax = axes[0, 0]
    if ideal:
        m = max(r["probes"] for r in ideal if r["method"] == "iid")
        selected = [r for r in ideal if r["method"] == "iid" and r["probes"] == m and r["asymmetry"] > 0]
        ax.scatter([r["delta_sq"] for r in selected], [r["diagnostic"] for r in selected], s=14, alpha=0.7)
        limits = [min(r["delta_sq"] for r in selected), max(r["delta_sq"] for r in selected)]
        ax.plot(limits, limits, "k--", label="identity")
        ax.set(xscale="log", yscale="log", xlabel="Exact squared adjoint mismatch", ylabel="Mean d(z)²",
               title=f"Linear-response diagnostic ({m} probes per trial)")
        ax.legend()
    ax = axes[0, 1]
    if ideal:
        for size in sorted({r["size"] for r in ideal}):
            selected = [r for r in ideal if r["method"] == "iid" and r["size"] == size and r["asymmetry"] == max(x["asymmetry"] for x in ideal) and r["cubic"] == 0.25]
            ms = sorted({r["probes"] for r in selected})
            measured = [np.mean([r["mse_over_eqprop"] for r in selected if r["probes"] == m]) for m in ms]
            line, = ax.loglog(ms, measured, "o-", label=f"n={size}")
            ax.loglog(ms, [(size-1)/m for m in ms], "--", color=line.get_color(), alpha=0.7)
        ax.axhline(1, color="gray", ls=":", label="uncorrected feedback")
        ax.set(xlabel="Independent sign probes", ylabel="Correction MSE / EqProp squared error",
               title="Monte Carlo variance (dashed: theory)")
        ax.legend()
    ax = axes[1, 0]
    if finite:
        selected = [r for r in finite if r["asymmetry"] == 1.5 and r["scheme"] == "central" and r["method"] == "orthogonal"]
        for sigma in sorted({r["read_noise"] for r in selected}):
            betas = sorted({r["beta"] for r in selected})
            values = [np.mean([r["mse"] for r in selected if r["beta"] == b and r["read_noise"] == sigma]) for b in betas]
            ax.loglog(betas, values, "o-", label=f"read σ={sigma:g}")
        baseline = np.mean([r["delta_sq"] for r in selected])
        ax.axhline(baseline, color="gray", ls=":", label="exact uncorrected feedback")
        ax.set(xlabel="Nudge amplitude β", ylabel="Adjoint MSE", title="Finite nudges: orthogonal 8-probe control")
        ax.legend(fontsize=8)
    ax = axes[1, 1]
    if training:
        for method in dict.fromkeys(r["method"] for r in training):
            selected = [r for r in training if r["method"] == method and r["asymmetry"] == 1.5]
            epochs = sorted({r["epoch"] for r in selected})
            values = [np.mean([r["test_loss"] for r in selected if r["epoch"] == ep]) for ep in epochs]
            ax.semilogy(epochs, values, label=method)
        ax.set(xlabel="Full-batch update", ylabel="Mean held-out loss", title="Toy learning with fixed asymmetric coupling")
        ax.legend(fontsize=8)
    fig.savefig(output / "overview.png", dpi=180)
    fig.savefig(output / "overview.svg")
    plt.close(fig)
    if finite:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
        for alpha, ax in zip((0.0, 1.5), axes):
            selected = [r for r in finite if r["asymmetry"] == alpha and r["read_noise"] == 0 and r["method"] == "orthogonal"]
            for scheme in ("forward", "central"):
                betas = sorted({r["beta"] for r in selected})
                errors = [np.mean([r["clean_projection_rmse"] for r in selected if r["beta"] == b and r["scheme"] == scheme]) for b in betas]
                ax.loglog(betas, errors, "o-", label=scheme)
            ax.set(xlabel="Nudge amplitude β", ylabel="RMS error in measured d(z)", title=f"Asymmetry α={alpha:g}")
            ax.legend()
        fig.savefig(output / "finite_nudges.png", dpi=180)
        fig.savefig(output / "finite_nudges.svg")
        plt.close(fig)


def summarize(output, ideal, finite, training, config):
    summary = {"evidence_tier": "exploratory, non-canonical", "ideal_rows": len(ideal),
               "finite_rows": len(finite), "training_rows": len(training)}
    if ideal:
        selected = [r for r in ideal if r["method"] == "iid" and r["delta_sq"] > 1e-24]
        summary["ideal"] = {
            "max_projection_identity_error": max(r["projection_max_abs_error"] for r in ideal),
            "mean_mse_theory_ratio": float(np.mean([r["mse_theory_ratio"] for r in selected])),
            "max_orthogonal_mse": max(r["mse"] for r in ideal if r["method"] == "orthogonal"),
            "mean_diagnostic_ratio_at_max_probes": float(np.mean([r["diagnostic_ratio"] for r in selected if r["probes"] == max(config["ideal_probes"])])),
        }
    if finite:
        summary["finite"] = {"max_force_residual": max(r["max_force_residual"] for r in finite)}
    final = []
    if training:
        for alpha in (0.0, 1.5):
            for method in config["training_methods"]:
                selected = [r for r in training if r["method"] == method and r["asymmetry"] == alpha]
                initial = [r for r in selected if r["epoch"] == 0]
                last = [r for r in selected if r["epoch"] == config["epochs"]]
                final.append(dict(asymmetry=alpha, method=method, seeds=len(last),
                                  initial_test_loss=float(np.mean([r["test_loss"] for r in initial])),
                                  final_test_loss=float(np.mean([r["test_loss"] for r in last])),
                                  final_test_loss_std=float(np.std([r["test_loss"] for r in last])),
                                  initial_gradient_cosine=float(np.mean([r["gradient_cosine"] for r in initial])),
                                  mean_train_equilibrations=float(np.mean([r["train_equilibrations"] for r in last]))))
        summary["training"] = final
    write_json(output / "summary.json", summary)
    lines = ["# Random-nudge Hopfield experiment", "", "Exploratory toy results; not a hardware or scale-efficiency claim.", "",
             "The measurement estimator uses forward relaxation only. Dense solves are evaluation oracles.", "",
             "See `run.json` for the configuration/environment, CSVs for all cases, and `overview.png` for the plots.", ""]
    if ideal:
        lines += [f"Maximum projection identity error: {summary['ideal']['max_projection_identity_error']:.3e}.",
                  f"Mean independent-probe MSE / theoretical MSE: {summary['ideal']['mean_mse_theory_ratio']:.4f}.", ""]
    if final:
        lines += ["| Asymmetry | Feedback | Initial test loss | Final test loss (mean ± SD) | Initial gradient cosine | Training equilibrations |",
                  "|---:|---|---:|---:|---:|---:|"]
        for row in final:
            lines.append(f"| {row['asymmetry']:g} | {row['method']} | {row['initial_test_loss']:.6g} | {row['final_test_loss']:.6g} ± {row['final_test_loss_std']:.3g} | {row['initial_gradient_cosine']:.4f} | {row['mean_train_equilibrations']:.0f} |")
        lines += ["", "Matched update counts and initialization; the perturbation methods use different equilibration budgets.",
                  "Counts cover training free/nudged states, excluding held-out evaluation and dense diagnostic solves."]
    (output / "report.md").write_text("\n".join(lines) + "\n")
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("simulation_results/random_nudge_hopfield") / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    parser.add_argument("--mode", choices=("all", "ideal", "finite", "train"), default="all")
    parser.add_argument("--quick", action="store_true", help="Small end-to-end smoke run")
    parser.add_argument("--epochs", type=int, help="Override the toy learning update count")
    args = parser.parse_args(argv)
    if args.epochs is not None and args.epochs < 1:
        parser.error(f"Expected --epochs >= 1; got {args.epochs}")
    if args.output.exists():
        parser.error(f"Expected a new output directory; already exists: {args.output}")
    config = dict(
        sizes=[8, 16] if args.quick else [8, 16, 32],
        asymmetries=[0.0, 1.5] if args.quick else [0.0, 0.25, 0.75, 1.5],
        seeds=[0] if args.quick else [0, 1, 2],
        ideal_trials=32 if args.quick else 128,
        ideal_probes=[1, 8, 32] if args.quick else [1, 4, 16, 64, 256],
        finite_trials=8 if args.quick else 32,
        finite_probes=[1, 8, 32] if args.quick else [1, 8, 32, 128],
        betas=[0.01, 0.001] if args.quick else [0.1, 0.03, 0.01, 0.003, 0.001, 0.0001, 0.00001],
        read_noise=[0.0, 1e-6] if args.quick else [0.0, 1e-8, 1e-6, 1e-4],
        training_methods=["adjoint", "eqprop", "mc8", "orthogonal"] if args.quick else ["adjoint", "eqprop", "mc1", "mc8", "mc32", "orthogonal"],
        epochs=args.epochs if args.epochs is not None else (8 if args.quick else 80),
        train_samples=16 if args.quick else 64, test_samples=32 if args.quick else 128,
        learning_rate=1.0, training_beta=1e-3, solver_tolerance=1e-12,
    )
    modes = ("ideal", "finite", "train") if args.mode == "all" else (args.mode,)
    expected = {
        "ideal": len(config["sizes"])*2*len(config["asymmetries"])*len(config["seeds"]),
        "finite": 2*len(config["seeds"])*2*len(config["betas"]),
        "train": 2*len(config["seeds"])*len(config["training_methods"]),
    }
    expected = {mode: expected[mode] for mode in modes}
    args.output.mkdir(parents=True)
    root = Path(__file__).resolve().parents[2]

    def git(*parts):
        return subprocess.check_output(["git", *parts], cwd=root, text=True).strip()

    started = time.time()
    run = dict(config=config, modes=list(modes), expected_cases=expected, evidence_tier="exploratory, non-canonical",
               pid=os.getpid(), launcher="local foreground Python", source_root=str(root),
               commit=git("rev-parse", "HEAD"), branch=git("branch", "--show-current"),
               source_status=git("status", "--short"), python=sys.version, numpy=np.__version__,
               platform=platform.platform(), command=shlex.join([sys.executable, "-m", "labs.tools.run_random_nudge_hopfield", *sys.argv[1:]]),
               started_utc=datetime.now(timezone.utc).isoformat(), output=str(args.output.resolve()),
               heartbeat="after each diagnostic case and every 10 learning updates",
               expected_runtime="a few seconds for quick; several minutes for full CPU run",
               recovery="diagnose failures; preserve this output; retry unchanged config in a new directory")
    write_json(args.output / "run.json", run)
    status = dict(state="running", pid=os.getpid(), expected_cases=expected, completed_cases={mode: 0 for mode in modes})

    def progress(stage, detail, advance=True):
        if advance:
            status["completed_cases"][stage] += 1
        status.update(stage=stage, detail=detail, elapsed_seconds=time.time()-started,
                      heartbeat_utc=datetime.now(timezone.utc).isoformat())
        write_json(args.output / "status.json", status)
        message = f"[{status['elapsed_seconds']:.1f}s] {stage} {status['completed_cases'][stage]}/{expected[stage]}: {detail}"
        print(message, flush=True)
        with (args.output / "run.log").open("a") as stream:
            stream.write(message + "\n")

    write_json(args.output / "status.json", status)
    try:
        ideal = ideal_sweep(config, args.output, progress) if "ideal" in modes else []
        finite = finite_sweep(config, args.output, progress) if "finite" in modes else []
        training = training_sweep(config, args.output, progress) if "train" in modes else []
        if status["completed_cases"] != expected:
            raise RuntimeError(f"Incomplete case coverage: {status['completed_cases']} != {expected}")
        summarize(args.output, ideal, finite, training, config)
        make_plots(args.output, ideal, finite, training)
        status.update(state="complete", elapsed_seconds=time.time()-started,
                      heartbeat_utc=datetime.now(timezone.utc).isoformat(),
                      terminal_result="summary.json", report="report.md")
        write_json(args.output / "status.json", status)
        print(f"Complete: {args.output.resolve()} ({time.time()-started:.1f}s)", flush=True)
    except BaseException as error:
        status.update(state="failed", error=f"{type(error).__name__}: {error}", elapsed_seconds=time.time()-started)
        write_json(args.output / "status.json", status)
        raise


if __name__ == "__main__":
    main()
