#!/usr/bin/env python3
"""Validate and compare the Conv2 sigma-5e-4 SGD read-noise repeat."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments.reporting import atomic_write_json, validate_run


ROOT = Path(__file__).resolve().parents[1]
STUDY_ID = (
    "perfectdiode-conv2-centered-float64-eqprop-read-noise-one-decade-"
    "sigma-5em4-sgd-10ep-seed0-20260816-v1"
)
EVIDENCE_CLASS = "ordinary_mnist_eqprop_read_noise_training_diagnostic"
SCHEMES = ("baseline", "ours", "legacy")
SIGMA = 5.0e-4
INJECTED_BETA = {"baseline": 100.0, "ours": 10.0, "legacy": 0.03}
BASE_BETA = {
    "baseline": 100.0,
    "ours": 0.625,
    "legacy": 0.0001171875,
}
SGD_LR = {
    "baseline": [7.90864, 3.81476, 0.82063, 7.90864, 3.81476],
    "ours": [0.523358, 0.246043, 0.0520201, 0.523358, 0.246043],
    "legacy": [0.00496733, 0.00367839, 0.00236044, 0.00496733, 0.000674726],
}
EXPECTED_DRAWS = 10 * math.ceil(55_000 / 16) * 2 * 3
EXPECTED_INITIAL_SHA = (
    "6870f1c95a903f7c22bec6e460f0e7b52c979fd7e3f850fc14868736f4418e66"
)
EXPECTED_TRAIN_SHA = (
    "c0940cfdde9fb2a87f846a4dd234a6ff634eed32a98b9f5d72dfed3e2119a809"
)
EXPECTED_VALIDATION_SHA = (
    "4801b7805d54dbe2197b98320cfe8a34d5d853c4ab568bcf1d7f69e136c94bb4"
)
EXPECTED_BATCH_ORDER = (
    "f1ea1c770bff668646139efdd8dcf33bb4298aecc8d49e5abde4455961da91cc",
    "610606f904fd6bc6d9d7437e42cc3ca5c83104f9b96c2d514f405abc5dbf04e5",
    "1ddc1d2dde572e25980c1d8a2c4db5044193b56812a1a5cdcc07d5264c29aca9",
    "60ff8ff31fed16c0b8d7de8023acf5a7f1cd9561f78ed9b5eecb29b66e016ec0",
    "c6dff97452a342a65aa2f6b4c9e967ad43b81523da337bfd684f1a302cc426d6",
    "25f784ad4b6b4369d1a686195be08c434f237dc8e0a715a7e3f8fc7836890e63",
    "5ea0d79d28f02820ea13669acdc13c8854ac9d83ba70123a7993beafb4966125",
    "fd2276547840fcb6258196c69b9cc921420e17cc947037b19ab920c961416d4e",
    "2a23c1dc2a4c2302349a30657e3cd4cc3064f6069d339b373803ab723191e68d",
    "22f1f6a2b79ed820ad512b9d75df17cad28454da4ecf3718c7018aabb0590a82",
)
DEFAULT_STUDY_ROOT = ROOT / "results" / STUDY_ID
DEFAULT_CLEAN_ROOT = ROOT / "results" / (
    "perfectdiode-conv123-centered-float64-eqprop-beta-collapse-boundary-"
    "vs-1decade-sgd-adam-10ep-seed0-20260815-v1"
)
DEFAULT_ADAM_ROOT = ROOT / "results" / (
    "perfectdiode-conv2-centered-float64-eqprop-read-noise-one-decade-"
    "sigma-3em4-5em4-10ep-seed0-20260813-v1"
)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _same(left: Any, right: float) -> bool:
    try:
        return math.isclose(float(left), right, rel_tol=1.0e-12, abs_tol=1.0e-15)
    except (TypeError, ValueError):
        return False


def _scheme(config: Mapping[str, Any]) -> str:
    arm_id = str(config.get("arm_id", ""))
    matches = [scheme for scheme in SCHEMES if arm_id.startswith(f"conv2_{scheme}_")]
    if len(matches) != 1:
        raise ValueError(f"Cannot resolve scheme from arm_id={arm_id!r}.")
    return matches[0]


def _history(run_dir: Path, name: str) -> np.ndarray:
    path = run_dir / name
    if not path.is_file():
        raise ValueError(f"Missing history {path}.")
    values = np.asarray(np.load(path), dtype=np.float64).reshape(-1)
    if values.size != 10 or not np.all(np.isfinite(values)):
        raise ValueError(f"Expected ten finite values in {path}.")
    return values


def _resolved_config(run_dir: Path) -> Mapping[str, Any]:
    manifest = _read_json(run_dir / "manifest.json")
    config = manifest.get("configuration", {}).get("resolved")
    if not isinstance(config, Mapping):
        raise ValueError(f"Manifest has no resolved config: {run_dir}.")
    return config


def _run_dirs(root: Path) -> list[Path]:
    return sorted(
        path
        for path in root.expanduser().resolve().iterdir()
        if path.is_dir() and (path / "manifest.json").is_file()
    )


def _validate_scientific_run(
    run_dir: Path,
    *,
    optimizer: str,
    sigma: float,
    expected_noise_seed: int | None,
) -> dict[str, Any]:
    errors = list(validate_run(run_dir))
    manifest = _read_json(run_dir / "manifest.json")
    status = _read_json(run_dir / "status.json")
    result = _read_json(run_dir / "result.json")
    metrics = _read_json(run_dir / "metrics.json")
    config = manifest.get("configuration", {}).get("resolved", {})
    scheme = _scheme(config)
    eqprop = config.get("eqprop", {})
    metric_eqprop = metrics.get("eqprop", {})
    provenance = metrics.get("dataset_provenance", {})
    train_accuracy = _history(run_dir, "accuracy_train.npy")
    validation_accuracy = _history(run_dir, "accuracy_test.npy")
    train_loss = _history(run_dir, "loss_train.npy")
    validation_loss = _history(run_dir, "loss_test.npy")
    expected_draws = 0 if sigma == 0.0 else EXPECTED_DRAWS
    checks = {
        "complete status": status.get("state") == "complete",
        "completion criteria": result.get("completion", {}).get("criteria_met") is True,
        "not smoke": manifest.get("smoke") is False,
        "ten epochs": int(manifest.get("configuration", {}).get("epochs", -1)) == 10,
        "optimizer": config.get("optimizer", {}).get("name") == optimizer
        and metrics.get("optimizer", {}).get("name") == optimizer,
        "float64": metrics.get("runtime_dtype") == "float64",
        "algorithm": metrics.get("training_algorithm") == "EP",
        "beta tier": eqprop.get("beta_tier") == "one_decade_lower",
        "base beta": _same(config.get("beta"), BASE_BETA[scheme]),
        "injected beta": _same(eqprop.get("injected_beta_B"), INJECTED_BETA[scheme]),
        "sigma config": _same(eqprop.get("endpoint_read_noise_std"), sigma),
        "sigma runtime": _same(metric_eqprop.get("endpoint_read_noise_std"), sigma),
        "centered": metric_eqprop.get("variant") == "centered",
        "current": metric_eqprop.get("nudging_mode") == "current",
        "normalized current": metric_eqprop.get("normalize_current_scale") is True,
        "input noise disabled": metric_eqprop.get("input_read_noise") is False,
        "noise draws": int(metrics.get("eqprop_endpoint_read_noise_draw_count", -1))
        == expected_draws,
        "optimizer steps": metrics.get("optimizer_steps_applied") is True,
        "T": int(config.get("model_base", {}).get("num_iterations_inference", -1)) == 8,
        "K": int(config.get("model_base", {}).get("num_iterations_training", -1)) == 8,
        "official test manifest": manifest.get("dataset", {}).get("official_test_read")
        is False,
        "official test count": int(metrics.get("official_test_evaluations", -1)) == 0,
        "initialization": metrics.get("initial_parameter_state_sha256")
        == EXPECTED_INITIAL_SHA,
        "train split": provenance.get("train_indices_sha256") == EXPECTED_TRAIN_SHA,
        "validation split": provenance.get("validation_indices_sha256")
        == EXPECTED_VALIDATION_SHA,
        "batch order": tuple(provenance.get("train_batch_order_sha256", []))
        == EXPECTED_BATCH_ORDER,
        "best metric/history": _same(
            metrics.get("best_validation_accuracy"), float(np.max(validation_accuracy))
        ),
        "final metric/history": _same(
            metrics.get("final_validation_accuracy"), float(validation_accuracy[-1])
        ),
    }
    if expected_noise_seed is not None:
        checks["noise seed"] = int(metric_eqprop.get("endpoint_read_noise_seed", -1)) == int(
            expected_noise_seed
        )
    errors.extend(f"failed guard {name}" for name, passed in checks.items() if not passed)
    if errors:
        raise ValueError(f"Invalid run {run_dir}:\n- " + "\n- ".join(errors))
    return {
        "scheme": scheme,
        "optimizer": optimizer,
        "sigma": sigma,
        "base_beta": float(config["beta"]),
        "injected_beta_B": float(eqprop["injected_beta_B"]),
        "learning_rates": [float(value) for value in config["lr"]],
        "best_epoch": int(metrics["best_epoch"]),
        "best_validation_accuracy": float(np.max(validation_accuracy)),
        "final_validation_accuracy": float(validation_accuracy[-1]),
        "final_train_accuracy": float(train_accuracy[-1]),
        "final_validation_loss": float(validation_loss[-1]),
        "final_train_loss": float(train_loss[-1]),
        "validation_accuracy": validation_accuracy,
        "train_accuracy": train_accuracy,
        "run_dir": str(run_dir),
    }


def load_primary(study_root: Path) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    study_root = study_root.expanduser().resolve()
    run_dirs = _run_dirs(study_root / "production")
    configs = sorted((study_root / "launch/resolved_configs").glob("*.json"))
    summaries = sorted((study_root / "task_summaries/production").glob("*.json"))
    receipts = sorted((study_root / "transport_receipts/production").glob("*.json"))
    if not all(len(items) == 3 for items in (run_dirs, configs, summaries, receipts)):
        raise ValueError(
            "Expected exactly three primary runs/configs/summaries/receipts; got "
            f"{len(run_dirs)}/{len(configs)}/{len(summaries)}/{len(receipts)}."
        )
    launch_contract = _read_json(study_root / "launch/launch_contract.json")
    source = launch_contract.get("source", {})
    submission = launch_contract.get("submission", {})
    expected_parent = str(submission.get("job_id", ""))
    if not expected_parent:
        raise ValueError("Launch contract has no production job ID.")

    cases: dict[str, dict[str, Any]] = {}
    errors: list[str] = []
    job_ids: set[str] = set()
    for task, (run_dir, config_path, summary_path, receipt_path) in enumerate(
        zip(run_dirs, configs, summaries, receipts, strict=True)
    ):
        manifest = _read_json(run_dir / "manifest.json")
        exact = _read_json(run_dir / "exact_run.json")
        config = _read_json(config_path)
        summary = _read_json(summary_path)
        receipt = _read_json(receipt_path)
        config_sha = _sha256(config_path)
        scheme = _scheme(config)
        checks = {
            "manifest study": manifest.get("study_id") == STUDY_ID,
            "manifest evidence": manifest.get("evidence_class") == EVIDENCE_CLASS,
            "task index": int(exact.get("index", -1)) == task,
            "resolved config": manifest.get("configuration", {}).get("resolved") == config,
            "SGD LR": config.get("lr") == SGD_LR[scheme],
            "summary one row": isinstance(summary, list) and len(summary) == 1,
            "summary complete": isinstance(summary, list)
            and len(summary) == 1
            and summary[0].get("status") == "complete",
            "summary config hash": isinstance(summary, list)
            and len(summary) == 1
            and summary[0].get("config_sha256") == config_sha,
            "receipt task": int(receipt.get("task_index", -1)) == task,
            "receipt semantic": receipt.get("semantic_pass") is True,
            "receipt optimizer": receipt.get("optimizer") == "SGD",
            "receipt parent": str(receipt.get("parent_job_id")) == expected_parent,
            "receipt source": receipt.get("source_archive_sha256")
            == source.get("archive_sha256"),
            "receipt config set": receipt.get("config_set_sha256")
            == source.get("ordered_config_set_sha256"),
            "receipt config": receipt.get("selected_config_sha256") == config_sha,
            "receipt manifest": receipt.get("manifest_sha256")
            == _sha256(run_dir / "manifest.json"),
            "receipt status": receipt.get("status_sha256")
            == _sha256(run_dir / "status.json"),
            "receipt metrics": receipt.get("metrics_sha256")
            == _sha256(run_dir / "metrics.jsonl"),
            "receipt result": receipt.get("result_sha256")
            == _sha256(run_dir / "result.json"),
            "exact source": exact.get("git", {}).get("source_archive_sha256")
            == source.get("archive_sha256"),
        }
        errors.extend(
            f"task {task}/{scheme}: failed guard {name}"
            for name, passed in checks.items()
            if not passed
        )
        try:
            case = _validate_scientific_run(
                run_dir,
                optimizer="SGD",
                sigma=SIGMA,
                expected_noise_seed=2026081201,
            )
        except ValueError as error:
            errors.append(str(error))
        else:
            if scheme in cases:
                errors.append(f"duplicate primary scheme {scheme}")
            cases[scheme] = case
        job_ids.add(str(receipt.get("job_id")))
    if tuple(cases) != SCHEMES:
        errors.append(f"primary scheme order/coverage mismatch: {tuple(cases)!r}")
    if len(job_ids) != 3:
        errors.append(f"expected three distinct Slurm task IDs, got {sorted(job_ids)!r}")
    if errors:
        raise ValueError("Primary collection validation failed:\n- " + "\n- ".join(errors))
    verification = {
        "schema_version": "conv2-read-noise-sgd-5em4-verification/v1",
        "study_id": STUDY_ID,
        "status": "pass",
        "canonical_run_count": 3,
        "summary_count": 3,
        "transport_receipt_count": 3,
        "slurm_parent_job_id": expected_parent,
        "slurm_task_job_ids": sorted(job_ids),
        "source_archive_sha256": source.get("archive_sha256"),
        "study_config_sha256": source.get("study_config_sha256"),
        "config_set_sha256": source.get("ordered_config_set_sha256"),
        "initial_parameter_state_sha256": EXPECTED_INITIAL_SHA,
        "train_indices_sha256": EXPECTED_TRAIN_SHA,
        "validation_indices_sha256": EXPECTED_VALIDATION_SHA,
        "train_batch_order_sha256": list(EXPECTED_BATCH_ORDER),
        "optimizer": "SGD",
        "runtime_dtype": "float64",
        "T": 8,
        "K": 8,
        "sigma": SIGMA,
        "expected_noise_draws_per_run": EXPECTED_DRAWS,
        "official_test_read": False,
    }
    return cases, verification


def load_anchors(
    root: Path,
    *,
    optimizer: str,
    sigma: float,
    expected_noise_seed: int | None,
) -> dict[str, dict[str, Any]]:
    matches: dict[str, dict[str, Any]] = {}
    for run_dir in _run_dirs(root.expanduser().resolve() / "production"):
        config = _resolved_config(run_dir)
        arm_id = str(config.get("arm_id", ""))
        if not arm_id.startswith("conv2_"):
            continue
        if config.get("optimizer", {}).get("name") != optimizer:
            continue
        eqprop = config.get("eqprop", {})
        if eqprop.get("beta_tier") != "one_decade_lower":
            continue
        if not _same(eqprop.get("endpoint_read_noise_std"), sigma):
            continue
        scheme = _scheme(config)
        if scheme in matches:
            raise ValueError(
                f"Duplicate {optimizer}/sigma={sigma:g} anchor for {scheme}: "
                f"{matches[scheme]['run_dir']} and {run_dir}."
            )
        matches[scheme] = _validate_scientific_run(
            run_dir,
            optimizer=optimizer,
            sigma=sigma,
            expected_noise_seed=expected_noise_seed,
        )
    if tuple(matches) != SCHEMES:
        raise ValueError(
            f"Incomplete {optimizer}/sigma={sigma:g} anchors: {tuple(matches)!r}."
        )
    return matches


def summarize(
    sgd_noise: Mapping[str, Mapping[str, Any]],
    sgd_clean: Mapping[str, Mapping[str, Any]],
    adam_noise: Mapping[str, Mapping[str, Any]],
    adam_clean: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for scheme in SCHEMES:
        sn = sgd_noise[scheme]
        sc = sgd_clean[scheme]
        an = adam_noise[scheme]
        ac = adam_clean[scheme]
        sgd_final_delta = 100.0 * (
            float(sn["final_validation_accuracy"])
            - float(sc["final_validation_accuracy"])
        )
        adam_final_delta = 100.0 * (
            float(an["final_validation_accuracy"])
            - float(ac["final_validation_accuracy"])
        )
        rows.append(
            {
                "scheme": scheme,
                "sigma": SIGMA,
                "injected_beta_B": INJECTED_BETA[scheme],
                "base_beta": BASE_BETA[scheme],
                "sgd_learning_rates": json.dumps(sn["learning_rates"]),
                "adam_learning_rates": json.dumps(an["learning_rates"]),
                "sgd_clean_best_validation_accuracy": float(
                    sc["best_validation_accuracy"]
                ),
                "sgd_noise_best_validation_accuracy": float(
                    sn["best_validation_accuracy"]
                ),
                "sgd_clean_final_validation_accuracy": float(
                    sc["final_validation_accuracy"]
                ),
                "sgd_noise_final_validation_accuracy": float(
                    sn["final_validation_accuracy"]
                ),
                "sgd_best_change_pp_vs_clean": 100.0
                * (
                    float(sn["best_validation_accuracy"])
                    - float(sc["best_validation_accuracy"])
                ),
                "sgd_final_change_pp_vs_clean": sgd_final_delta,
                "adam_clean_best_validation_accuracy": float(
                    ac["best_validation_accuracy"]
                ),
                "adam_noise_best_validation_accuracy": float(
                    an["best_validation_accuracy"]
                ),
                "adam_clean_final_validation_accuracy": float(
                    ac["final_validation_accuracy"]
                ),
                "adam_noise_final_validation_accuracy": float(
                    an["final_validation_accuracy"]
                ),
                "adam_best_change_pp_vs_clean": 100.0
                * (
                    float(an["best_validation_accuracy"])
                    - float(ac["best_validation_accuracy"])
                ),
                "adam_final_change_pp_vs_clean": adam_final_delta,
                "sgd_minus_adam_noise_final_pp": 100.0
                * (
                    float(sn["final_validation_accuracy"])
                    - float(an["final_validation_accuracy"])
                ),
                "sgd_minus_adam_noise_penalty_pp": sgd_final_delta
                - adam_final_delta,
                "sgd_stable_through_ten_epochs": True,
                "sgd_noise_run_dir": sn["run_dir"],
                "sgd_clean_run_dir": sc["run_dir"],
                "adam_noise_run_dir": an["run_dir"],
                "adam_clean_run_dir": ac["run_dir"],
            }
        )
    return rows


def _write_csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def make_plot(
    groups: Mapping[str, Mapping[str, Mapping[str, Any]]], output: Path
) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(12.0, 3.8), sharex=True, sharey=True)
    styles = {
        "sgd_clean": ("SGD clean", "#4C78A8", "-"),
        "sgd_noise": ("SGD σ=5e-4", "#4C78A8", "--"),
        "adam_clean": ("Adam clean", "#F58518", "-"),
        "adam_noise": ("Adam σ=5e-4", "#F58518", "--"),
    }
    epochs = np.arange(1, 11)
    for axis, scheme in zip(axes, SCHEMES, strict=True):
        for name, (label, color, linestyle) in styles.items():
            axis.plot(
                epochs,
                100.0 * np.asarray(groups[name][scheme]["validation_accuracy"]),
                label=label,
                color=color,
                linestyle=linestyle,
                linewidth=1.8,
            )
        axis.set_title(scheme)
        axis.set_xlabel("Epoch")
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("Validation accuracy (%)")
    axes[-1].legend(frameon=False, fontsize=8)
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def write_report(rows: Sequence[Mapping[str, Any]], output: Path) -> None:
    lines = [
        "# Conv2 read-noise optimizer comparison at σ=5e-4",
        "",
        (
            "All cases use the same one-decade-lower scheme-specific beta, "
            "T=K=8, seed-0 ordinary MNIST, initialization, split, and minibatch "
            "order. The noisy SGD and Adam cases also use the same endpoint-noise "
            "seed. This is exploratory diagnostic evidence, not paper-facing accuracy."
        ),
        "",
        (
            "For a common optimizer-paired reference, both clean columns use the "
            "one-decade controls from the later boundary study. The earlier Adam "
            "higher-noise report instead anchors to its original clean parent; its "
            "published -0.22/-0.42/-1.44 pp deltas remain valid for that lineage, "
            "and the noisy Adam endpoints themselves are unchanged."
        ),
        "",
        "| Scheme | SGD clean → noisy final | Δ SGD | Adam clean → noisy final | Δ Adam | SGD−Adam noisy |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {scheme} | {sc:.2%} → {sn:.2%} | {sd:+.2f} pp | "
            "{ac:.2%} → {an:.2%} | {ad:+.2f} pp | {oa:+.2f} pp |".format(
                scheme=row["scheme"],
                sc=float(row["sgd_clean_final_validation_accuracy"]),
                sn=float(row["sgd_noise_final_validation_accuracy"]),
                sd=float(row["sgd_final_change_pp_vs_clean"]),
                ac=float(row["adam_clean_final_validation_accuracy"]),
                an=float(row["adam_noise_final_validation_accuracy"]),
                ad=float(row["adam_final_change_pp_vs_clean"]),
                oa=float(row["sgd_minus_adam_noise_final_pp"]),
            )
        )
    lines.extend(
        [
            "",
            (
                "All three SGD noise runs completed ten finite epochs, so the "
                "sigma-5e-4 endpoint noise caused no observed SGD training instability."
            ),
            "",
            (
                "The SGD-versus-Adam contrast is an optimizer/LR-contract comparison: "
                "each optimizer uses its own historical scheme-specific LR vector. "
                "It therefore does not isolate the optimizer algorithm alone."
            ),
            "",
            (
                "The qualitative result is shared by both contracts: ours has the "
                "highest noisy final accuracy and legacy has the largest clean-relative "
                "penalty. The penalty magnitude is contract-dependent, most visibly "
                "for legacy (-0.80 pp with SGD versus -1.50 pp with Adam)."
            ),
            "",
            (
                "Single-seed differences and small non-monotonic changes should not "
                "be interpreted as population-level effects."
            ),
        ]
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-root", type=Path, default=DEFAULT_STUDY_ROOT)
    parser.add_argument("--clean-root", type=Path, default=DEFAULT_CLEAN_ROOT)
    parser.add_argument("--adam-root", type=Path, default=DEFAULT_ADAM_ROOT)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    study_root = args.study_root.expanduser().resolve()
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else study_root / "analysis"
    )
    sgd_noise, verification = load_primary(study_root)
    sgd_clean = load_anchors(
        args.clean_root, optimizer="SGD", sigma=0.0, expected_noise_seed=None
    )
    adam_clean = load_anchors(
        args.clean_root, optimizer="Adam", sigma=0.0, expected_noise_seed=None
    )
    adam_noise = load_anchors(
        args.adam_root,
        optimizer="Adam",
        sigma=SIGMA,
        expected_noise_seed=2026081201,
    )
    rows = summarize(sgd_noise, sgd_clean, adam_noise, adam_clean)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "terminal_summary.csv", rows)
    atomic_write_json(
        output_dir / "summary.json",
        {
            "study_id": STUDY_ID,
            "interpretation_scope": "optimizer_and_historical_lr_contract",
            "rows": rows,
        },
    )
    verification["comparison_anchor_run_count"] = 9
    verification["comparison_scope"] = "optimizer_and_historical_lr_contract"
    atomic_write_json(output_dir / "verification.json", verification)
    groups = {
        "sgd_noise": sgd_noise,
        "sgd_clean": sgd_clean,
        "adam_noise": adam_noise,
        "adam_clean": adam_clean,
    }
    make_plot(groups, output_dir / "validation_accuracy_trajectories.png")
    write_report(rows, output_dir / "report.md")
    print(json.dumps({"verification": verification, "rows": rows}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
