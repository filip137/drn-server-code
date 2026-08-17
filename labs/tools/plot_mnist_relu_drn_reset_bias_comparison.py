"""Compare the three legacy-index, bias-inclusive RESET objectives."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import matplotlib.pyplot as plt


OBJECTIVES = (
    "paired_squared_error",
    "cross_entropy",
    "teacher_kl",
)
LABELS = {
    "paired_squared_error": "paired squared error",
    "cross_entropy": "cross-entropy",
    "teacher_kl": "teacher KL",
}
COLORS = {
    "paired_squared_error": "tab:blue",
    "cross_entropy": "tab:orange",
    "teacher_kl": "tab:green",
}
LEGACY_TRAIN_INDICES = ((3, 4), (4, 5))
FRESH_TEST_INDICES = ((0, 1), (1, 2))


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _complete(path: Path, *, kind: str) -> dict[str, Any]:
    result = _json(path)
    if result.get("status") != "complete" or result.get("error") is not None:
        raise ValueError(
            f"Expected {kind} result to be semantically complete. "
            f"Provided value: path={str(path)!r}, "
            f"status={result.get('status')!r}, error={result.get('error')!r}."
        )
    return result


def _indices(metrics: Mapping[str, Any]) -> tuple[tuple[int, int], ...]:
    return tuple(
        (
            int(item["resolved_pre_index"]),
            int(item["resolved_post_index"]),
        )
        for item in metrics.get("amplification_indices", ())
    )


def _training_run(run_dir: Path, *, objective: str) -> dict[str, Any]:
    required = (
        run_dir / "result.json",
        run_dir / "metrics.jsonl",
        run_dir / "artifacts" / "learning_rate_selection.json",
    )
    missing = tuple(str(path) for path in required if not path.is_file())
    if missing:
        raise FileNotFoundError(
            "Expected a training run with result.json, metrics.jsonl, and "
            "artifacts/learning_rate_selection.json. "
            f"Provided value: missing={missing!r}."
        )
    result = _complete(required[0], kind="training")
    metrics = result["metrics"]
    expected = {
        "objective": objective,
        "include_biases": True,
        "amplification_indexing": "legacy_process_global",
    }
    mismatches = {
        key: {"expected": value, "provided": metrics.get(key)}
        for key, value in expected.items()
        if metrics.get(key) != value
    }
    if mismatches:
        raise ValueError(
            "Expected a bias-inclusive legacy-index training result. "
            f"Provided value: {mismatches!r}."
        )
    rows = tuple(
        json.loads(line)
        for line in required[1].read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    initialization = tuple(
        row for row in rows if row.get("mode") == "initialization"
    )
    training = tuple(
        sorted(
            (row for row in rows if row.get("mode") == "train"),
            key=lambda row: int(row["completed_epochs"]),
        )
    )
    if len(initialization) != 1 or len(training) != 20:
        raise ValueError(
            "Expected one initialization row and 20 production epochs. "
            f"Provided value: initialization={len(initialization)}, "
            f"training={len(training)}."
        )
    return {
        "result": result,
        "selection": _json(required[2]),
        "initialization": initialization[0],
        "training": training,
    }


def _test_result(path: Path, *, objective: str) -> dict[str, Any]:
    metrics = _complete(path, kind="test")["metrics"]
    expected = {
        "objective": objective,
        "include_biases": True,
        "amplification_indexing": "legacy_process_global",
        "split": "test",
    }
    mismatches = {
        key: {"expected": value, "provided": metrics.get(key)}
        for key, value in expected.items()
        if metrics.get(key) != value
    }
    if mismatches:
        raise ValueError(
            "Expected a fresh-process legacy-index test result. "
            f"Provided value: {mismatches!r}."
        )
    return metrics


def _pulse_changes(programming: Mapping[str, Any]) -> dict[str, float]:
    return {
        key: float(value["pulse_changed_fraction"])
        for key, value in sorted(programming["parameters"].items())
    }


def compare(
    *,
    training_runs: Mapping[str, Path],
    test_results: Mapping[str, Path],
    historical_test_result: Path,
    output_dir: Path,
) -> dict[str, Any]:
    expected = set(OBJECTIVES)
    if set(training_runs) != expected or set(test_results) != expected:
        raise ValueError(
            f"Expected training and test mappings for {OBJECTIVES!r}. "
            f"Provided value: training={tuple(training_runs)!r}, "
            f"tests={tuple(test_results)!r}."
        )
    runs = {
        objective: _training_run(training_runs[objective], objective=objective)
        for objective in OBJECTIVES
    }
    tests = {
        objective: _test_result(test_results[objective], objective=objective)
        for objective in OBJECTIVES
    }
    historical = _complete(
        historical_test_result,
        kind="historical test",
    )["metrics"]
    historical_accuracy = float(
        historical.get("student_accuracy", historical.get("accuracy"))
    )

    arms: dict[str, Any] = {}
    for objective in OBJECTIVES:
        run = runs[objective]
        metrics = run["result"]["metrics"]
        arms[objective] = {
            "run_dir": str(training_runs[objective]),
            "initial": metrics["initial_validation"],
            "selected": metrics["selected"],
            "last": metrics["last_validation"],
            "test": tests[objective],
            "selected_learning_rates": metrics["selected_learning_rates"],
            "reset_state_fingerprint": run["selection"][
                "reset_state_fingerprint"
            ],
            "pulse_changed_fraction_by_parameter": _pulse_changes(
                metrics["measured_programming"]
            ),
            "assignment_sha256_by_parameter": metrics[
                "measured_programming"
            ]["assignment_sha256_by_parameter"],
            "teacher_sha256": metrics["teacher"]["sha256"],
            "training_amplification_indices": _indices(metrics),
            "test_amplification_indices": _indices(tests[objective]),
        }

    assignments = {
        json.dumps(arm["assignment_sha256_by_parameter"], sort_keys=True)
        for arm in arms.values()
    }
    teachers = {arm["teacher_sha256"] for arm in arms.values()}
    reset_states = {arm["reset_state_fingerprint"] for arm in arms.values()}
    initial_signatures = {
        (
            float(arm["initial"]["student_accuracy"]),
            float(arm["initial"]["paired_squared_error"]),
        )
        for arm in arms.values()
    }
    accuracies = {
        objective: float(tests[objective]["student_accuracy"])
        for objective in OBJECTIVES
    }
    all_dense_parameters_written = all(
        all(value > 0.0 for value in arm["pulse_changed_fraction_by_parameter"].values())
        for arm in arms.values()
    )
    invariants = {
        "same_device_assignments": len(assignments) == 1,
        "same_teacher": len(teachers) == 1,
        "same_reset_state": len(reset_states) == 1,
        "same_initial_metrics": len(initial_signatures) == 1,
        "all_dense_parameters_written": all_dense_parameters_written,
        "all_training_indices_are_legacy_3_4_5": all(
            arm["training_amplification_indices"] == LEGACY_TRAIN_INDICES
            for arm in arms.values()
        ),
        "all_test_indices_are_fresh_0_1_2": all(
            arm["test_amplification_indices"] == FRESH_TEST_INDICES
            for arm in arms.values()
        ),
    }
    paired_accuracy = accuracies["paired_squared_error"]
    paired_parity = abs(paired_accuracy - historical_accuracy) <= 0.005
    ce_recovery = (
        accuracies["cross_entropy"] >= 0.94
        and accuracies["cross_entropy"] >= paired_accuracy - 0.01
    )
    kl_recovery = (
        accuracies["teacher_kl"] >= 0.94
        and accuracies["teacher_kl"] >= paired_accuracy - 0.01
    )
    summary = {
        "schema": "mnist-relu-drn-reset-bias-legacy-loss-comparison/v1",
        "historical_paired_squared_error_test": {
            "result": str(historical_test_result),
            "accuracy": historical_accuracy,
        },
        "arms": arms,
        "comparison": {
            "test_accuracy_by_objective": accuracies,
            "paired_minus_historical_test_accuracy": (
                paired_accuracy - historical_accuracy
            ),
            "cross_entropy_minus_paired_test_accuracy": (
                accuracies["cross_entropy"] - paired_accuracy
            ),
            "teacher_kl_minus_paired_test_accuracy": (
                accuracies["teacher_kl"] - paired_accuracy
            ),
            "invariants": invariants,
            "legacy_provenance_control_valid": (
                all(invariants.values()) and paired_parity
            ),
            "paired_reproduces_historical_within_0p5pp": paired_parity,
            "cross_entropy_recovers_at_least_94pct_and_within_1pp_of_paired": (
                ce_recovery
            ),
            "teacher_kl_recovers_at_least_94pct_and_within_1pp_of_paired": (
                kl_recovery
            ),
            "supervision_objective_is_material_under_legacy_mismatch": (
                paired_parity and not ce_recovery and not kl_recovery
            ),
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    figure, axes = plt.subplots(1, 4, figsize=(18, 4), constrained_layout=True)
    fields = (
        ("student_accuracy", "Validation accuracy (%)", 100.0, False),
        ("kl_teacher_student", "KL(teacher || DRN)", 1.0, True),
        ("paired_squared_error", "Paired squared error", 1.0, True),
    )
    for objective in OBJECTIVES:
        run = runs[objective]
        epochs = [0] + [
            int(row["completed_epochs"]) for row in run["training"]
        ]
        validation = [run["initialization"]["validation"]] + [
            row["validation"] for row in run["training"]
        ]
        for axis, (field, _ylabel, scale, _log) in zip(axes[:3], fields):
            axis.plot(
                epochs,
                [scale * float(item[field]) for item in validation],
                color=COLORS[objective],
                marker="o",
                markersize=2.5,
                label=LABELS[objective],
            )
    for axis, (_field, ylabel, _scale, log_scale) in zip(axes[:3], fields):
        if log_scale:
            axis.set_yscale("log")
        axis.set(xlabel="Completed epochs", ylabel=ylabel)
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    axes[0].axhline(94.0, color="black", linestyle=":", linewidth=1.0)

    labels = [LABELS[objective] for objective in OBJECTIVES]
    values = [100.0 * accuracies[objective] for objective in OBJECTIVES]
    axes[3].bar(
        labels,
        values,
        color=[COLORS[objective] for objective in OBJECTIVES],
    )
    axes[3].axhline(
        100.0 * historical_accuracy,
        color="black",
        linestyle="--",
        linewidth=1.0,
        label="archived paired test",
    )
    axes[3].set(ylabel="Held-out accuracy (%)", ylim=(75.0, 100.0))
    axes[3].tick_params(axis="x", rotation=20)
    axes[3].grid(axis="y", alpha=0.25)
    axes[3].legend(fontsize=8)
    for index, value in enumerate(values):
        axes[3].text(index, value + 0.35, f"{value:.2f}", ha="center", fontsize=8)
    figure.savefig(output_dir / "validation_and_test_comparison.png", dpi=180)
    plt.close(figure)
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    for objective in OBJECTIVES:
        flag = objective.replace("_", "-")
        parser.add_argument(f"--{flag}-run", type=Path, required=True)
        parser.add_argument(f"--{flag}-test-result", type=Path, required=True)
    parser.add_argument("--historical-test-result", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    training_runs = {
        objective: getattr(args, f"{objective}_run").expanduser().resolve()
        for objective in OBJECTIVES
    }
    test_results = {
        objective: getattr(
            args, f"{objective}_test_result"
        ).expanduser().resolve()
        for objective in OBJECTIVES
    }
    compare(
        training_runs=training_runs,
        test_results=test_results,
        historical_test_result=(
            args.historical_test_result.expanduser().resolve()
        ),
        output_dir=args.output_dir.expanduser().resolve(),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
