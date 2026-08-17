"""Summarize and plot teacher-initialized MNIST DRN distillation runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _completed_run(label: str, run_dir: Path) -> dict[str, Any]:
    result_path = run_dir / "result.json"
    metrics_path = run_dir / "metrics.jsonl"
    if not result_path.is_file() or not metrics_path.is_file():
        raise FileNotFoundError(
            "Expected each run directory to contain result.json and "
            f"metrics.jsonl. Provided value: label={label!r}, "
            f"run={str(run_dir)!r}."
        )
    result = _json(result_path)
    if result.get("status") != "complete" or result.get("error") is not None:
        raise ValueError(
            "Expected every distillation result to be semantically complete. "
            f"Provided value: label={label!r}, status={result.get('status')!r}, "
            f"error={result.get('error')!r}."
        )
    rows = [
        json.loads(line)
        for line in metrics_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    initialization = [row for row in rows if row.get("mode") == "initialization"]
    training = sorted(
        (row for row in rows if row.get("mode") == "train"),
        key=lambda row: int(row["completed_epochs"]),
    )
    if len(initialization) != 1 or not training:
        raise ValueError(
            "Expected exactly one initialization row and at least one train "
            f"row. Provided value: label={label!r}, "
            f"initialization_rows={len(initialization)}, train_rows={len(training)}."
        )
    return {
        "result": result,
        "initialization": initialization[0],
        "training": training,
    }


def _test_metrics(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    result = _json(path)
    if result.get("status") != "complete" or result.get("error") is not None:
        raise ValueError(
            "Expected every supplied test result to be semantically complete. "
            f"Provided value: path={str(path)!r}, status={result.get('status')!r}."
        )
    return dict(result["metrics"])


def _arm_summary(
    label: str,
    run_dir: Path,
    run: dict[str, Any],
    test_path: Path | None,
) -> dict[str, Any]:
    metrics = run["result"]["metrics"]
    training = run["training"]
    initial = dict(metrics["initial_validation"])
    selected = dict(metrics["selected"])
    last = dict(metrics["last_validation"])
    programming = metrics.get("measured_programming")
    return {
        "label": label,
        "run_dir": str(run_dir),
        "encoding": str(metrics["encoding"]),
        "update_backend": str(metrics["update_backend"]),
        "initial_target_mapping": (
            None
            if programming is None
            else programming.get("initial_target_mapping")
        ),
        "completed_epochs": int(training[-1]["completed_epochs"]),
        "initial": initial,
        "selected": selected,
        "last": last,
        "relative_kl_improvement": float(
            metrics["relative_kl_recovery_from_initialization"]
        ),
        "acceptance_gate": dict(metrics["acceptance_gate"]),
        "test": _test_metrics(test_path),
    }


def _encoding_comparisons(
    arms: Mapping[str, dict[str, Any]],
) -> dict[str, Any]:
    by_backend: dict[str, dict[str, dict[str, Any]]] = {}
    for arm in arms.values():
        by_backend.setdefault(arm["update_backend"], {})[arm["encoding"]] = arm
    comparisons: dict[str, Any] = {}
    for backend, encodings in sorted(by_backend.items()):
        if set(encodings) != {"single", "differential"}:
            continue
        single = encodings["single"]["selected"]
        differential = encodings["differential"]["selected"]
        single_kl = float(single["kl_teacher_student"])
        differential_kl = float(differential["kl_teacher_student"])
        kl_reduction = (single_kl - differential_kl) / single_kl
        accuracy_delta = float(differential["student_accuracy"]) - float(
            single["student_accuracy"]
        )
        agreement_delta = float(differential["teacher_agreement"]) - float(
            single["teacher_agreement"]
        )
        comparisons[backend] = {
            "single_selected_kl": single_kl,
            "differential_selected_kl": differential_kl,
            "relative_kl_reduction_differential_vs_single": kl_reduction,
            "student_accuracy_delta": accuracy_delta,
            "teacher_agreement_delta": agreement_delta,
            "criteria": {
                "kl_at_least_5pct_lower": kl_reduction >= 0.05,
                "accuracy_not_worse_by_more_than_0_2pp": accuracy_delta >= -0.002,
                "agreement_not_worse_by_more_than_0_2pp": agreement_delta >= -0.002,
            },
        }
        comparisons[backend]["two_device_better"] = all(
            comparisons[backend]["criteria"].values()
        )
    return comparisons


def compare(
    run_dirs: Mapping[str, Path],
    *,
    test_results: Mapping[str, Path] | None,
    output_dir: Path,
) -> dict[str, Any]:
    if len(run_dirs) < 2:
        raise ValueError(
            "Expected at least two labeled distillation runs. "
            f"Provided value: {sorted(run_dirs)!r}."
        )
    unknown_tests = set(test_results or {}) - set(run_dirs)
    if unknown_tests:
        raise ValueError(
            "Expected every test-result label to name a supplied run. "
            f"Provided value: {sorted(unknown_tests)!r}."
        )
    loaded = {
        label: _completed_run(label, path)
        for label, path in sorted(run_dirs.items())
    }
    arms = {
        label: _arm_summary(
            label,
            run_dirs[label],
            loaded[label],
            (test_results or {}).get(label),
        )
        for label in loaded
    }
    summary = {
        "schema": "mnist-relu-drn-kd-comparison/v1",
        "arms": arms,
        "encoding_comparisons": _encoding_comparisons(arms),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    figure, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for label, run in loaded.items():
        rows = run["training"]
        epochs = [0] + [int(row["completed_epochs"]) for row in rows]
        validation = [run["initialization"]["validation"]] + [
            row["validation"] for row in rows
        ]
        axes[0].plot(
            epochs,
            [float(item["kl_teacher_student"]) for item in validation],
            marker="o",
            markersize=3,
            label=label,
        )
        axes[1].plot(
            epochs,
            [100.0 * float(item["student_accuracy"]) for item in validation],
            marker="o",
            markersize=3,
            label=f"{label}: accuracy",
        )
        axes[1].plot(
            epochs,
            [100.0 * float(item["teacher_agreement"]) for item in validation],
            linestyle="--",
            linewidth=1.2,
            label=f"{label}: agreement",
        )
    axes[0].set_yscale("log")
    axes[0].set(xlabel="Completed epochs", ylabel="Validation KL(teacher || DRN)")
    axes[1].set(xlabel="Completed epochs", ylabel="Validation metric (%)")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend(fontsize=7)
    figure.savefig(output_dir / "validation_comparison.png", dpi=180)
    plt.close(figure)
    return summary


def _labeled_path(value: str) -> tuple[str, Path]:
    label, separator, raw_path = value.partition("=")
    if not separator or not label or not raw_path:
        raise argparse.ArgumentTypeError(
            "expected LABEL=PATH with non-empty label and path"
        )
    return label, Path(raw_path).expanduser().resolve()


def _mapping(values: Sequence[tuple[str, Path]]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for label, path in values:
        if label in result:
            raise ValueError(
                "Expected each label exactly once. "
                f"Provided value: duplicated label={label!r}."
            )
        result[label] = path
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", type=_labeled_path, required=True)
    parser.add_argument("--test-result", action="append", type=_labeled_path, default=[])
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    compare(
        _mapping(args.run),
        test_results=_mapping(args.test_result),
        output_dir=args.output_dir.expanduser().resolve(),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
