"""Analyze the controlled RESET bias/loss/indexing MNIST factorial."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Mapping

import matplotlib.pyplot as plt


EXPECTED_ARMS = {
    "bias_free_logical_mse": (False, "logical", "paired_squared_error"),
    "bias_free_logical_kl": (False, "logical", "teacher_kl"),
    "bias_free_legacy_mse": (
        False,
        "legacy_process_global",
        "paired_squared_error",
    ),
    "bias_free_legacy_kl": (False, "legacy_process_global", "teacher_kl"),
    "bias_logical_mse": (True, "logical", "paired_squared_error"),
    "bias_logical_kl": (True, "logical", "teacher_kl"),
    "bias_legacy_mse": (
        True,
        "legacy_process_global",
        "paired_squared_error",
    ),
    "bias_legacy_kl": (True, "legacy_process_global", "teacher_kl"),
}
LOGICAL_PAIRS = ((0, 1), (1, 2))
LEGACY_PRODUCTION_PAIRS = ((3, 4), (4, 5))


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


def _collected_run_result(
    record: Mapping[str, Any], *, campaign_root: Path
) -> Path:
    recorded = Path(record["run_result"])
    if recorded.is_file():
        return recorded
    stage_root = (
        campaign_root
        / "stages"
        / record["case_id"]
        / record["target_id"]
        / record["stage_id"]
    )
    candidates = tuple(sorted(stage_root.glob("attempt-*/runs/*/result.json")))
    if len(candidates) != 1:
        raise FileNotFoundError(
            "Expected one collected run result when the recorded remote path "
            "is unavailable locally. "
            f"Provided value: recorded={str(recorded)!r}, "
            f"candidates={tuple(str(path) for path in candidates)!r}."
        )
    return candidates[0]


def _pairs(metrics: Mapping[str, Any]) -> tuple[tuple[int, int], ...]:
    return tuple(
        (
            int(item["resolved_pre_index"]),
            int(item["resolved_post_index"]),
        )
        for item in metrics.get("amplification_indices", ())
    )


def _assert_metadata(
    metrics: Mapping[str, Any],
    *,
    include_biases: bool,
    indexing: str,
    objective: str,
    split: str | None = None,
) -> None:
    expected = {
        "include_biases": include_biases,
        "amplification_indexing": indexing,
        "objective": objective,
    }
    if split is not None:
        expected["split"] = split
    mismatches = {
        key: {"expected": value, "provided": metrics.get(key)}
        for key, value in expected.items()
        if metrics.get(key) != value
    }
    if mismatches:
        raise ValueError(
            "Expected factorial result metadata to match its declared arm. "
            f"Provided value: {mismatches!r}."
        )
    if metrics.get("reset_input_between_batches", True) is not True:
        raise ValueError(
            "Expected every factorial arm to reset input between batches. "
            f"Provided value: {metrics.get('reset_input_between_batches')!r}."
        )


def _train_arm(
    result_path: Path,
    *,
    include_biases: bool,
    indexing: str,
    objective: str,
) -> dict[str, Any]:
    result = _complete(result_path, kind="training")
    metrics = result["metrics"]
    _assert_metadata(
        metrics,
        include_biases=include_biases,
        indexing=indexing,
        objective=objective,
    )
    run_dir = result_path.parent
    metrics_path = run_dir / "metrics.jsonl"
    selection_path = run_dir / "artifacts/learning_rate_selection.json"
    missing = [
        str(path)
        for path in (metrics_path, selection_path)
        if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError(
            "Expected training metrics and LR-selection artifacts. "
            f"Provided value: missing={missing!r}."
        )
    rows = tuple(
        json.loads(line)
        for line in metrics_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    restarts = [row for row in rows if row.get("mode") == "production_restart"]
    epochs = [row for row in rows if row.get("mode") == "train"]
    if len(restarts) != 1 or len(epochs) != 20:
        raise ValueError(
            "Expected one controlled production restart and 20 epochs. "
            f"Provided value: restarts={len(restarts)}, epochs={len(epochs)}."
        )
    if restarts[0].get("semantics") != "controlled_exact_reseed_and_rebuild":
        raise ValueError(
            "Expected controlled reseed/rebuild lifecycle provenance. "
            f"Provided value: {restarts[0].get('semantics')!r}."
        )
    expected_pairs = (
        LOGICAL_PAIRS if indexing == "logical" else LEGACY_PRODUCTION_PAIRS
    )
    if _pairs(metrics) != expected_pairs:
        raise ValueError(
            "Expected production amplifier indices to match the factorial arm. "
            f"Provided value: expected={expected_pairs!r}, actual={_pairs(metrics)!r}."
        )
    selection = _json(selection_path)
    programming = metrics["measured_programming"]
    return {
        "result_path": str(result_path),
        "duration_seconds": float(result["duration_seconds"]),
        "initial_validation_accuracy": float(
            metrics["initial_validation"]["student_accuracy"]
        ),
        "selected_validation_accuracy": float(
            metrics["selected"]["student_accuracy"]
        ),
        "selected_epoch": int(metrics["selected"]["epoch"]),
        "selected_learning_rates": list(metrics["selected_learning_rates"]),
        "teacher_sha256": metrics["teacher"]["sha256"],
        "device_source_sha256": programming["source_sha256"],
        "assignment_sha256_by_parameter": programming[
            "assignment_sha256_by_parameter"
        ],
        "reset_state_fingerprint": selection["reset_state_fingerprint"],
        "production_amplification_indices": _pairs(metrics),
    }


def _test_arm(
    result_path: Path,
    *,
    include_biases: bool,
    indexing: str,
    objective: str,
) -> dict[str, Any]:
    result = _complete(result_path, kind="test")
    metrics = result["metrics"]
    _assert_metadata(
        metrics,
        include_biases=include_biases,
        indexing=indexing,
        objective=objective,
        split="test",
    )
    if _pairs(metrics) != LOGICAL_PAIRS:
        raise ValueError(
            "Expected every fresh-process test to resolve layer indices 0/1/2. "
            f"Provided value: {_pairs(metrics)!r}."
        )
    return {
        "result_path": str(result_path),
        "duration_seconds": float(result["duration_seconds"]),
        "accuracy": float(metrics["student_accuracy"]),
        "teacher_agreement": float(metrics["teacher_agreement"]),
        "kl_teacher_student": float(metrics["kl_teacher_student"]),
        "paired_squared_error": float(metrics["paired_squared_error"]),
        "cross_entropy": float(metrics["cross_entropy"]),
        "fresh_process_amplification_indices": _pairs(metrics),
        "teacher_sha256": metrics["teacher"]["sha256"],
    }


def _factorial_effects(
    accuracies: Mapping[tuple[bool, str, str], float],
) -> dict[str, float]:
    if set(accuracies) != set(EXPECTED_ARMS.values()):
        raise ValueError(
            "Expected one accuracy for every 2x2x2 factorial cell. "
            f"Provided value: {tuple(accuracies)!r}."
        )
    codes = {
        key: (
            1 if key[0] else -1,
            1 if key[1] == "legacy_process_global" else -1,
            1 if key[2] == "teacher_kl" else -1,
        )
        for key in accuracies
    }

    def contrast(indices: tuple[int, ...]) -> float:
        return 100.0 * sum(
            accuracies[key]
            * _product(codes[key][index] for index in indices)
            for key in accuracies
        ) / 4.0

    return {
        "bias_enabled_minus_disabled_pp": contrast((0,)),
        "legacy_minus_logical_indexing_pp": contrast((1,)),
        "teacher_kl_minus_paired_mse_pp": contrast((2,)),
        "bias_x_indexing_interaction_pp": contrast((0, 1)),
        "bias_x_loss_interaction_pp": contrast((0, 2)),
        "indexing_x_loss_interaction_pp": contrast((1, 2)),
        "bias_x_indexing_x_loss_interaction_pp": contrast((0, 1, 2)),
    }


def _product(values) -> int:
    result = 1
    for value in values:
        result *= int(value)
    return result


def _conditional_contrasts(
    accuracies: Mapping[tuple[bool, str, str], float],
) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {
        "bias_enabled_minus_disabled_pp": {},
        "legacy_minus_logical_indexing_pp": {},
        "teacher_kl_minus_paired_mse_pp": {},
    }
    for indexing in ("logical", "legacy_process_global"):
        for objective in ("paired_squared_error", "teacher_kl"):
            key = f"indexing={indexing},objective={objective}"
            result["bias_enabled_minus_disabled_pp"][key] = 100.0 * (
                accuracies[(True, indexing, objective)]
                - accuracies[(False, indexing, objective)]
            )
    for include_biases in (False, True):
        for objective in ("paired_squared_error", "teacher_kl"):
            key = f"bias={include_biases},objective={objective}"
            result["legacy_minus_logical_indexing_pp"][key] = 100.0 * (
                accuracies[(include_biases, "legacy_process_global", objective)]
                - accuracies[(include_biases, "logical", objective)]
            )
    for include_biases in (False, True):
        for indexing in ("logical", "legacy_process_global"):
            key = f"bias={include_biases},indexing={indexing}"
            result["teacher_kl_minus_paired_mse_pp"][key] = 100.0 * (
                accuracies[(include_biases, indexing, "teacher_kl")]
                - accuracies[(include_biases, indexing, "paired_squared_error")]
            )
    return result


def _plot(arms: Mapping[str, Any], path: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(9.0, 4.2), sharey=True)
    width = 0.34
    x = (0.0, 1.0)
    for axis, objective in zip(
        axes, ("paired_squared_error", "teacher_kl")
    ):
        for offset, indexing, label, color in (
            (-width / 2, "logical", "logical indexing", "tab:blue"),
            (
                width / 2,
                "legacy_process_global",
                "legacy process-global",
                "tab:orange",
            ),
        ):
            values = [
                100.0
                * arms[
                    next(
                        case
                        for case, factors in EXPECTED_ARMS.items()
                        if factors == (bias, indexing, objective)
                    )
                ]["test"]["accuracy"]
                for bias in (False, True)
            ]
            axis.bar(
                [value + offset for value in x],
                values,
                width=width,
                label=label,
                color=color,
            )
        axis.set_title(
            "paired MSE" if objective == "paired_squared_error" else "teacher KL"
        )
        axis.set_xticks(x, ("bias off", "bias on"))
        axis.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("MNIST test accuracy (%)")
    axes[1].legend(frameon=False, fontsize=8)
    figure.suptitle("Controlled RESET factorial (single seed; no error bars)")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def analyze(
    *, campaign_roots: tuple[Path, ...], output_dir: Path
) -> dict[str, Any]:
    if not campaign_roots:
        raise ValueError(
            "Expected at least one campaign root. Provided value: ()."
        )
    records: dict[str, tuple[dict[str, Any], Path]] = {}
    for campaign_root in campaign_roots:
        aggregate_path = campaign_root / "aggregate/results.json"
        aggregate = _json(aggregate_path)
        for item in aggregate["stages"]:
            stage_id = item["stage_id"]
            if stage_id in records:
                raise ValueError(
                    "Expected factorial shard stage IDs to be disjoint. "
                    f"Provided value: duplicate={stage_id!r}."
                )
            records[stage_id] = (item, campaign_root)
    arms: dict[str, Any] = {}
    accuracies: dict[tuple[bool, str, str], float] = {}
    for case_id, factors in EXPECTED_ARMS.items():
        include_biases, indexing, objective = factors
        train_entry = records.get(f"{case_id}_train")
        test_entry = records.get(f"{case_id}_test")
        if train_entry is None or test_entry is None:
            raise ValueError(
                "Expected train and test campaign records for every arm. "
                f"Provided value: missing case={case_id!r}."
            )
        train_record, train_root = train_entry
        test_record, test_root = test_entry
        if train_record.get("status") != "complete" or test_record.get("status") != "complete":
            raise ValueError(
                "Expected complete campaign stages before factorial analysis. "
                f"Provided value: case={case_id!r}, "
                f"train={train_record.get('status')!r}, "
                f"test={test_record.get('status')!r}."
            )
        if train_record["target_id"] != test_record["target_id"]:
            raise ValueError(
                "Expected each factorial cell to train and test on one host. "
                f"Provided value: case={case_id!r}, "
                f"train={train_record['target_id']!r}, "
                f"test={test_record['target_id']!r}."
            )
        train = _train_arm(
            _collected_run_result(train_record, campaign_root=train_root),
            include_biases=include_biases,
            indexing=indexing,
            objective=objective,
        )
        test = _test_arm(
            _collected_run_result(test_record, campaign_root=test_root),
            include_biases=include_biases,
            indexing=indexing,
            objective=objective,
        )
        arms[case_id] = {
            "execution_target": train_record["target_id"],
            "include_biases": include_biases,
            "amplification_indexing": indexing,
            "objective": objective,
            "train": train,
            "test": test,
        }
        accuracies[factors] = test["accuracy"]

    assignments = {
        json.dumps(arm["train"]["assignment_sha256_by_parameter"], sort_keys=True)
        for arm in arms.values()
    }
    factor_codes_by_target: dict[str, list[tuple[int, int, int]]] = {}
    for arm in arms.values():
        factor_codes_by_target.setdefault(arm["execution_target"], []).append(
            (
                1 if arm["include_biases"] else -1,
                (
                    1
                    if arm["amplification_indexing"]
                    == "legacy_process_global"
                    else -1
                ),
                1 if arm["objective"] == "teacher_kl" else -1,
            )
        )
    host_main_effect_balance = all(
        all(sum(codes[index] for codes in cells) == 0 for index in range(3))
        for cells in factor_codes_by_target.values()
    )
    invariants = {
        "same_teacher": len(
            {arm["train"]["teacher_sha256"] for arm in arms.values()}
            | {arm["test"]["teacher_sha256"] for arm in arms.values()}
        )
        == 1,
        "same_device_source": len(
            {arm["train"]["device_source_sha256"] for arm in arms.values()}
        )
        == 1,
        "same_dense_device_assignments": len(assignments) == 1,
        "all_reset_input_between_batches": True,
        "all_production_restarts_verified": True,
        "all_amplification_indices_verified": True,
        "main_factor_levels_balanced_within_each_execution_host": (
            host_main_effect_balance
        ),
    }
    if not all(invariants.values()):
        raise ValueError(
            "Expected every controlled-factorial invariant to hold. "
            f"Provided value: {invariants!r}."
        )

    summary = {
        "schema": "mnist_relu_drn_reset.controlled-factorial-analysis",
        "schema_version": 1,
        "campaign_roots": [str(path.resolve()) for path in campaign_roots],
        "replication": {
            "seeds": [42],
            "independent_runs_per_cell": 1,
            "uncertainty_estimated": False,
        },
        "execution_blocking": {
            "targets": {
                target: len(cells)
                for target, cells in sorted(factor_codes_by_target.items())
            },
            "main_factor_levels_balanced_within_each_target": (
                host_main_effect_balance
            ),
            "limitation": (
                "host-by-interaction effects are not separately estimable "
                "with one run per factorial cell"
            ),
        },
        "invariants": invariants,
        "factorial_effects": _factorial_effects(accuracies),
        "conditional_contrasts": _conditional_contrasts(accuracies),
        "arms": arms,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "factorial_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with (output_dir / "factorial_arms.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "case_id",
                "execution_target",
                "include_biases",
                "amplification_indexing",
                "objective",
                "initial_validation_accuracy",
                "selected_validation_accuracy",
                "test_accuracy",
                "teacher_agreement",
                "kl_teacher_student",
                "paired_squared_error",
                "selected_epoch",
                "train_duration_seconds",
            ),
        )
        writer.writeheader()
        for case_id, arm in arms.items():
            writer.writerow(
                {
                    "case_id": case_id,
                    "execution_target": arm["execution_target"],
                    "include_biases": arm["include_biases"],
                    "amplification_indexing": arm["amplification_indexing"],
                    "objective": arm["objective"],
                    "initial_validation_accuracy": arm["train"][
                        "initial_validation_accuracy"
                    ],
                    "selected_validation_accuracy": arm["train"][
                        "selected_validation_accuracy"
                    ],
                    "test_accuracy": arm["test"]["accuracy"],
                    "teacher_agreement": arm["test"]["teacher_agreement"],
                    "kl_teacher_student": arm["test"]["kl_teacher_student"],
                    "paired_squared_error": arm["test"][
                        "paired_squared_error"
                    ],
                    "selected_epoch": arm["train"]["selected_epoch"],
                    "train_duration_seconds": arm["train"][
                        "duration_seconds"
                    ],
                }
            )
    _plot(arms, output_dir / "factorial_accuracy.png")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--campaign-root",
        type=Path,
        required=True,
        action="append",
        help="campaign root; repeat once per disjoint host shard",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    summary = analyze(
        campaign_roots=tuple(
            path.expanduser().resolve() for path in args.campaign_root
        ),
        output_dir=args.output_dir.expanduser().resolve(),
    )
    print(json.dumps(summary["factorial_effects"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
