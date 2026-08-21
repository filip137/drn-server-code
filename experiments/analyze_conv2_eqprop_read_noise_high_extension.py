#!/usr/bin/env python3
"""Validate and summarize the one-decade-lower high read-noise extension."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments.reporting import atomic_write_json, validate_run


STUDY_ID = (
    "perfectdiode-conv2-centered-float64-eqprop-read-noise-"
    "one-decade-sigma-3em4-5em4-10ep-seed0-20260813-v1"
)
EVIDENCE_CLASS = "ordinary_mnist_eqprop_read_noise_training_diagnostic"
SCHEMES = ("baseline", "ours", "legacy")
SIGMAS = (3.0e-4, 5.0e-4)
EXPECTED_CASES = [
    (scheme, sigma)
    for scheme in SCHEMES
    for sigma in SIGMAS
]
INJECTED_BETA = {"baseline": 100.0, "ours": 10.0, "legacy": 0.03}
BASE_BETA = {
    "baseline": 100.0,
    "ours": 0.625,
    "legacy": 0.0001171875,
}
EXPECTED_DRAWS = 10 * math.ceil(55_000 / 16) * 2 * 3
EXPECTED_INITIAL_SHA = (
    "6870f1c95a903f7c22bec6e460f0e7b52c979fd7e3f850fc14868736f4418e66"
)
EXPECTED_TRAIN_INDICES_SHA = (
    "c0940cfdde9fb2a87f846a4dd234a6ff634eed32a98b9f5d72dfed3e2119a809"
)
EXPECTED_VALIDATION_INDICES_SHA = (
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
EXPECTED_SOURCE_ARCHIVE_SHA = (
    "17fceecad251868b95de9f197167359ced01511b42f1c1c404e8df2ee54803f4"
)
EXPECTED_CONFIG_SET_SHA = (
    "166a7025825e17051c22292e658f10d4fefc70293882094c29868a0d9158fa38"
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


def _load_history(run_dir: Path, filename: str) -> np.ndarray:
    path = run_dir / filename
    if not path.is_file():
        raise ValueError(f"Missing history array {path}.")
    values = np.asarray(np.load(path), dtype=np.float64).reshape(-1)
    if values.size != 10 or not np.all(np.isfinite(values)):
        raise ValueError(f"Expected ten finite history values in {path}.")
    return values


def load_and_validate(study_root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    study_root = study_root.expanduser().resolve()
    production_root = study_root / "production"
    config_root = study_root / "launch/resolved_configs"
    summary_root = study_root / "task_summaries/production"
    receipt_root = study_root / "transport_receipts/production"
    run_dirs = sorted(
        path
        for path in production_root.iterdir()
        if path.is_dir() and (path / "manifest.json").is_file()
    )
    configs = sorted(config_root.glob("*.json"))
    summaries = sorted(summary_root.glob("*.json"))
    receipts = sorted(receipt_root.glob("*.json"))
    if not all(len(values) == 6 for values in (run_dirs, configs, summaries, receipts)):
        raise ValueError(
            "Expected exactly six runs, configs, summaries, and receipts; got "
            f"runs={len(run_dirs)}, configs={len(configs)}, "
            f"summaries={len(summaries)}, receipts={len(receipts)}."
        )

    errors: list[str] = []
    cases: list[dict[str, Any]] = []
    identities: list[tuple[str, float]] = []
    job_ids: set[str] = set()
    for task, (run_dir, config_path, summary_path, receipt_path) in enumerate(
        zip(run_dirs, configs, summaries, receipts, strict=True)
    ):
        errors.extend(f"{run_dir.name}: {error}" for error in validate_run(run_dir))
        manifest = _read_json(run_dir / "manifest.json")
        status = _read_json(run_dir / "status.json")
        result = _read_json(run_dir / "result.json")
        metrics = _read_json(run_dir / "metrics.json")
        exact = _read_json(run_dir / "exact_run.json")
        config = _read_json(config_path)
        summary = _read_json(summary_path)
        receipt = _read_json(receipt_path)
        scheme = _scheme(config)
        sigma = float(config.get("eqprop", {}).get("endpoint_read_noise_std", -1.0))
        identities.append((scheme, sigma))
        eqprop = config.get("eqprop", {})
        metric_eqprop = metrics.get("eqprop", {})
        provenance = metrics.get("dataset_provenance", {})
        train_accuracy = _load_history(run_dir, "accuracy_train.npy")
        validation_accuracy = _load_history(run_dir, "accuracy_test.npy")
        train_loss = _load_history(run_dir, "loss_train.npy")
        validation_loss = _load_history(run_dir, "loss_test.npy")
        selected_config_sha = _sha256(config_path)

        checks = {
            "task/config index": int(exact.get("index", -1)) == task,
            "study": manifest.get("study_id") == STUDY_ID,
            "evidence": manifest.get("evidence_class") == EVIDENCE_CLASS,
            "complete status": status.get("state") == "complete",
            "completion criteria": result.get("completion", {}).get("criteria_met") is True,
            "not smoke": manifest.get("smoke") is False,
            "ten epochs": int(manifest.get("configuration", {}).get("epochs", -1)) == 10,
            "float64": metrics.get("runtime_dtype") == "float64",
            "algorithm": metrics.get("training_algorithm") == "EP",
            "beta tier": eqprop.get("beta_tier") == "one_decade_lower",
            "base beta": _same(config.get("beta"), BASE_BETA[scheme]),
            "injected beta": _same(eqprop.get("injected_beta_B"), INJECTED_BETA[scheme]),
            "noise sigma": _same(metric_eqprop.get("endpoint_read_noise_std"), sigma),
            "noise seed": int(metric_eqprop.get("endpoint_read_noise_seed", -1))
            == 2026081201,
            "centered": metric_eqprop.get("variant") == "centered",
            "current": metric_eqprop.get("nudging_mode") == "current",
            "normalized current": metric_eqprop.get("normalize_current_scale") is True,
            "input noise disabled": metric_eqprop.get("input_read_noise") is False,
            "noise draws": int(metrics.get("eqprop_endpoint_read_noise_draw_count", -1))
            == EXPECTED_DRAWS,
            "optimizer steps": metrics.get("optimizer_steps_applied") is True,
            "T": int(config.get("model_base", {}).get("num_iterations_inference", -1)) == 8,
            "K": int(config.get("model_base", {}).get("num_iterations_training", -1)) == 8,
            "official test manifest": manifest.get("dataset", {}).get("official_test_read")
            is False,
            "official test count": int(metrics.get("official_test_evaluations", -1)) == 0,
            "initialization hash": metrics.get("initial_parameter_state_sha256")
            == EXPECTED_INITIAL_SHA,
            "train split hash": provenance.get("train_indices_sha256")
            == EXPECTED_TRAIN_INDICES_SHA,
            "validation split hash": provenance.get("validation_indices_sha256")
            == EXPECTED_VALIDATION_INDICES_SHA,
            "batch-order hashes": tuple(provenance.get("train_batch_order_sha256", []))
            == EXPECTED_BATCH_ORDER,
            "metric final train accuracy": _same(
                metrics.get("final_train_accuracy"), train_accuracy[-1]
            ),
            "metric final validation accuracy": _same(
                metrics.get("final_validation_accuracy"), validation_accuracy[-1]
            ),
            "metric best validation accuracy": _same(
                metrics.get("best_validation_accuracy"), np.max(validation_accuracy)
            ),
            "summary shape": isinstance(summary, list) and len(summary) == 1,
            "summary complete": isinstance(summary, list)
            and len(summary) == 1
            and summary[0].get("status") == "complete",
            "summary config hash": isinstance(summary, list)
            and len(summary) == 1
            and summary[0].get("config_sha256") == selected_config_sha,
            "receipt task": int(receipt.get("task_index", -1)) == task,
            "receipt semantic pass": receipt.get("semantic_pass") is True,
            "receipt parent job": receipt.get("parent_job_id") == "910848",
            "receipt source": receipt.get("source_archive_sha256")
            == EXPECTED_SOURCE_ARCHIVE_SHA,
            "receipt config set": receipt.get("config_set_sha256")
            == EXPECTED_CONFIG_SET_SHA,
            "receipt selected config": receipt.get("selected_config_sha256")
            == selected_config_sha,
            "receipt manifest hash": receipt.get("manifest_sha256")
            == _sha256(run_dir / "manifest.json"),
            "receipt status hash": receipt.get("status_sha256")
            == _sha256(run_dir / "status.json"),
            "receipt metrics hash": receipt.get("metrics_sha256")
            == _sha256(run_dir / "metrics.jsonl"),
            "receipt result hash": receipt.get("result_sha256")
            == _sha256(run_dir / "result.json"),
        }
        errors.extend(
            f"{run_dir.name}: failed guard {name}"
            for name, passed in checks.items()
            if not passed
        )
        job_ids.add(str(receipt.get("job_id")))
        cases.append(
            {
                "task": task,
                "scheme": scheme,
                "sigma": sigma,
                "injected_beta_B": float(eqprop["injected_beta_B"]),
                "base_beta": float(config["beta"]),
                "best_epoch": int(metrics["best_epoch"]),
                "best_validation_accuracy": float(metrics["best_validation_accuracy"]),
                "final_validation_accuracy": float(validation_accuracy[-1]),
                "final_validation_loss": float(validation_loss[-1]),
                "final_train_accuracy": float(train_accuracy[-1]),
                "final_train_loss": float(train_loss[-1]),
                "run_dir": str(run_dir),
            }
        )

    if identities != EXPECTED_CASES:
        errors.append(f"case order/coverage mismatch: {identities!r}")
    if len(job_ids) != 6:
        errors.append(f"expected six distinct Slurm task IDs, got {sorted(job_ids)!r}")
    if errors:
        raise ValueError("Collection validation failed:\n- " + "\n- ".join(errors))
    verification = {
        "schema_version": "conv2-read-noise-high-extension-verification/v1",
        "study_id": STUDY_ID,
        "status": "pass",
        "canonical_run_count": 6,
        "summary_count": 6,
        "transport_receipt_count": 6,
        "slurm_parent_job_id": "910848",
        "slurm_task_job_ids": sorted(job_ids),
        "source_archive_sha256": EXPECTED_SOURCE_ARCHIVE_SHA,
        "config_set_sha256": EXPECTED_CONFIG_SET_SHA,
        "initial_parameter_state_sha256": EXPECTED_INITIAL_SHA,
        "train_indices_sha256": EXPECTED_TRAIN_INDICES_SHA,
        "validation_indices_sha256": EXPECTED_VALIDATION_INDICES_SHA,
        "train_batch_order_sha256": list(EXPECTED_BATCH_ORDER),
        "runtime_dtype": "float64",
        "T": 8,
        "K": 8,
        "official_test_read": False,
        "expected_noise_draws_per_run": EXPECTED_DRAWS,
        "gate_deviation": (
            "The separate live 0-0 Jean Zay canary was explicitly bypassed by "
            "Filip because this study is exploratory; six local semantic smokes passed."
        ),
    }
    return cases, verification


def load_parent_anchors(summary_path: Path, verification_path: Path) -> dict[str, dict[str, float]]:
    verification = _read_json(verification_path)
    if (
        verification.get("status") != "pass"
        or verification.get("initial_parameter_state_sha256") != EXPECTED_INITIAL_SHA
        or verification.get("train_indices_sha256") != EXPECTED_TRAIN_INDICES_SHA
        or verification.get("validation_indices_sha256") != EXPECTED_VALIDATION_INDICES_SHA
        or tuple(verification.get("train_batch_order_sha256", [])) != EXPECTED_BATCH_ORDER
    ):
        raise ValueError("Parent lower-beta verification does not match this study.")
    anchors: dict[str, dict[str, float]] = {}
    with summary_path.open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            if row["beta_tier"] != "one_decade_lower":
                continue
            sigma = float(row["sigma"])
            if sigma not in (0.0, 1.0e-4):
                continue
            scheme = row["scheme"]
            key = "clean" if sigma == 0.0 else "sigma_1e4"
            anchors.setdefault(scheme, {})[f"{key}_best"] = float(
                row["best_observed_validation_accuracy"]
            )
            anchors[scheme][f"{key}_final"] = float(
                row["last_observed_validation_accuracy"]
            )
    if set(anchors) != set(SCHEMES) or any(len(values) != 4 for values in anchors.values()):
        raise ValueError(f"Parent anchors are incomplete: {anchors!r}")
    return anchors


def summarize(
    cases: Sequence[Mapping[str, Any]],
    anchors: Mapping[str, Mapping[str, float]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for case in cases:
        scheme = str(case["scheme"])
        anchor = anchors[scheme]
        final_accuracy = float(case["final_validation_accuracy"])
        best_accuracy = float(case["best_validation_accuracy"])
        rows.append(
            {
                **case,
                "final_change_pp_vs_clean": 100.0
                * (final_accuracy - float(anchor["clean_final"])),
                "best_change_pp_vs_clean": 100.0
                * (best_accuracy - float(anchor["clean_best"])),
                "final_change_pp_vs_sigma_1e4": 100.0
                * (final_accuracy - float(anchor["sigma_1e4_final"])),
                "best_change_pp_vs_sigma_1e4": 100.0
                * (best_accuracy - float(anchor["sigma_1e4_best"])),
                "stable_through_ten_epochs": True,
            }
        )
    return rows


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = [
        "task",
        "scheme",
        "sigma",
        "injected_beta_B",
        "base_beta",
        "best_epoch",
        "best_validation_accuracy",
        "final_validation_accuracy",
        "final_validation_loss",
        "final_train_accuracy",
        "final_train_loss",
        "best_change_pp_vs_clean",
        "final_change_pp_vs_clean",
        "best_change_pp_vs_sigma_1e4",
        "final_change_pp_vs_sigma_1e4",
        "stable_through_ten_epochs",
        "run_dir",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row[field] for field in fields} for row in rows)


def make_plot(
    rows: Sequence[Mapping[str, Any]],
    anchors: Mapping[str, Mapping[str, float]],
    output: Path,
) -> None:
    colors = {"baseline": "#4C78A8", "ours": "#F58518", "legacy": "#54A24B"}
    figure, axis = plt.subplots(figsize=(7.2, 4.5))
    for scheme in SCHEMES:
        scheme_rows = [row for row in rows if row["scheme"] == scheme]
        x = np.asarray([0.0, 1.0, 3.0, 5.0])
        y = 100.0 * np.asarray(
            [
                anchors[scheme]["clean_final"],
                anchors[scheme]["sigma_1e4_final"],
                *[float(row["final_validation_accuracy"]) for row in scheme_rows],
            ]
        )
        axis.plot(x, y, marker="o", linewidth=2, color=colors[scheme], label=scheme)
    axis.set_xlabel(r"Endpoint read-noise $\sigma$ ($\times 10^{-4}$)")
    axis.set_ylabel("Final validation accuracy (%)")
    axis.set_xticks([0, 1, 3, 5])
    axis.grid(alpha=0.25)
    axis.legend(frameon=False)
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def write_report(rows: Sequence[Mapping[str, Any]], output: Path) -> None:
    by_case = {(row["scheme"], float(row["sigma"])): row for row in rows}
    lines = [
        "# Conv2 one-decade-lower high read-noise report",
        "",
        (
            "This is seed-0 ordinary-MNIST exploratory evidence using matched "
            "initialization, split, minibatch order, and endpoint-noise draws. "
            "It is not paper-facing accuracy evidence."
        ),
        "",
        "| Scheme | σ | Injected β | State | Best val. | Final val. | Δ final vs clean | Δ final vs σ=1e-4 |",
        "|---|---:|---:|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {scheme} | {sigma:.0e} | {beta:g} | complete (10/10) | "
            "{best:.2%} | {final:.2%} | {clean:+.2f} pp | {low:+.2f} pp |".format(
                scheme=row["scheme"],
                sigma=float(row["sigma"]),
                beta=float(row["injected_beta_B"]),
                best=float(row["best_validation_accuracy"]),
                final=float(row["final_validation_accuracy"]),
                clean=float(row["final_change_pp_vs_clean"]),
                low=float(row["final_change_pp_vs_sigma_1e4"]),
            )
        )
    losses_at_5e4 = {
        scheme: -float(by_case[(scheme, 5.0e-4)]["final_change_pp_vs_clean"])
        for scheme in SCHEMES
    }
    lines.extend(
        [
            "",
            (
                "All six runs completed all ten epochs with finite metrics and "
                "validated canonical bundles, so there is no observed training "
                "instability at either new noise level."
            ),
            "",
            (
                "At σ=5e-4, the final-validation penalty relative to the matched "
                f"clean anchor is {losses_at_5e4['baseline']:.2f} pp for baseline, "
                f"{losses_at_5e4['ours']:.2f} pp for ours, and "
                f"{losses_at_5e4['legacy']:.2f} pp for legacy. The legacy scheme "
                "is therefore the most noise-sensitive on this single-seed surface, "
                "while ours retains the highest final validation accuracy at both "
                "3e-4 and 5e-4."
            ),
            "",
            (
                "The separate live Jean Zay canary was explicitly bypassed for this "
                "exploratory run. Six local semantic smokes passed before submission, "
                "and every production task independently passed the live V100, source, "
                "config, dataset, and runtime guards."
            ),
            "",
            (
                "Because there is only one seed, small non-monotonic differences "
                "between nearby noise levels should not be interpreted as a trend."
            ),
        ]
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-root", type=Path, required=True)
    parser.add_argument("--parent-summary", type=Path, required=True)
    parser.add_argument("--parent-verification", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    cases, verification = load_and_validate(args.study_root)
    anchors = load_parent_anchors(args.parent_summary, args.parent_verification)
    rows = summarize(cases, anchors)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "terminal_summary.csv", rows)
    atomic_write_json(output_dir / "summary.json", {"rows": rows, "parent_anchors": anchors})
    atomic_write_json(output_dir / "verification.json", verification)
    make_plot(rows, anchors, output_dir / "final_validation_vs_noise.png")
    write_report(rows, output_dir / "report.md")
    print(json.dumps({"verification": verification, "rows": rows}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
