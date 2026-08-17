"""Compare matched cohort-B deterministic and probabilistic write policies."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt


_POLICY_CONFIG_KEYS = (
    "programming_deadband_mode",
    "programming_deadband_relative",
    "probabilistic_write_mode",
    "probabilistic_write_probability",
    "probabilistic_write_scale_relative",
    "probabilistic_write_seed",
)


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256_json(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _complete_training_result(run_dir: Path) -> dict[str, Any]:
    result = _json(run_dir / "result.json")
    if result.get("status") != "complete" or result.get("error") is not None:
        raise ValueError(
            "Expected a semantically complete cohort-B training result. "
            f"Provided value: run={str(run_dir)!r}, "
            f"status={result.get('status')!r}, error={result.get('error')!r}."
        )
    return result


def _held_out_test_metrics(path: Path) -> dict[str, Any]:
    result = _json(path)
    metrics = dict(result.get("metrics", {}))
    if (
        result.get("status") != "complete"
        or result.get("error") is not None
        or metrics.get("split_protocol", {}).get("effective")
        != "held_out_test"
        or int(metrics.get("examples", 0)) != 10000
    ):
        raise ValueError(
            "Expected a complete 10,000-example held_out_test result. "
            f"Provided value: path={str(path)!r}, result={result!r}."
        )
    return metrics


def _weighted_fraction(
    parameters: dict[str, dict[str, Any]],
    field: str,
) -> float:
    weighted = 0.0
    elements = 0
    for key, report in sorted(parameters.items()):
        shape = report.get("shape")
        if not isinstance(shape, list) or not shape:
            raise ValueError(
                "Expected every programming report to contain a non-empty "
                f"shape array. Provided value: key={key!r}, shape={shape!r}."
            )
        count = math.prod(int(value) for value in shape)
        value = report.get(field)
        if value is None or not math.isfinite(float(value)):
            raise ValueError(
                f"Expected {field} to be finite for every parameter. "
                f"Provided value: key={key!r}, value={value!r}."
            )
        weighted += count * float(value)
        elements += count
    if elements == 0:
        raise ValueError(
            "Expected at least one measured dense-weight programming report. "
            "Provided value: none."
        )
    return weighted / elements


def _manifest_inputs(manifest: dict[str, Any]) -> dict[str, str]:
    inputs: dict[str, str] = {}
    for item in manifest.get("inputs", []):
        role = str(item.get("role"))
        digest = str(item.get("sha256"))
        if role in inputs:
            raise ValueError(
                "Expected one manifest input per role. "
                f"Provided value: duplicated role={role!r}."
            )
        inputs[role] = digest
    return inputs


def _comparison_identity(
    run_dir: Path,
    result: dict[str, Any],
) -> dict[str, Any]:
    config = copy.deepcopy(_json(run_dir / "config.resolved.json"))
    try:
        if "settings" in config:
            backend_parameters = config["settings"]["update_backend"][
                "parameters"
            ]
        else:
            backend_parameters = config["modes"]["train"][
                "update_backend"
            ]["parameters"]
    except (KeyError, TypeError) as exc:
        raise ValueError(
            "Expected a cohort-B training config with either persisted "
            "settings.update_backend.parameters or source "
            "modes.train.update_backend.parameters. "
            f"Provided value: run={str(run_dir)!r}."
        ) from exc
    for key in _POLICY_CONFIG_KEYS:
        backend_parameters.pop(key, None)

    manifest = _json(run_dir / "manifest.json")
    source = dict(manifest.get("source", {}))
    programming = result["metrics"]["update_programming"]
    return {
        "normalized_config_sha256": _sha256_json(config),
        "input_sha256_by_role": _manifest_inputs(manifest),
        "source_revision": {
            "commit": source.get("commit"),
            "dirty": source.get("dirty"),
            "dirty_hash": source.get("dirty_hash"),
        },
        "completed_epochs": int(result["metrics"]["completed_epochs"]),
        "global_step": int(result["metrics"]["global_step"]),
        "device_source_sha256": programming.get("source_sha256"),
        "assignment_sha256_by_parameter": programming.get(
            "assignment_sha256_by_parameter"
        ),
    }


def _classify_policy(
    programming: dict[str, Any],
    *,
    is_baseline: bool,
) -> dict[str, Any]:
    deadband_mode = str(
        programming.get("programming_deadband_mode", "none")
    )
    deadband_relative = float(
        programming.get("programming_deadband_relative", 0.0)
    )
    probabilistic_mode = str(
        programming.get("probabilistic_write_mode", "none")
    )
    probability = float(
        programming.get("probabilistic_write_probability", 1.0)
    )
    scale_relative = float(
        programming.get("probabilistic_write_scale_relative", 0.0)
    )
    seed = int(programming.get("probabilistic_write_seed", 0))

    deadband_active = deadband_mode != "none" or deadband_relative != 0.0
    probabilistic_active = probabilistic_mode != "none"
    if deadband_active and probabilistic_active:
        raise ValueError(
            "Expected deterministic deadband and probabilistic writing to be "
            "mutually exclusive. Provided value: "
            f"deadband_mode={deadband_mode!r}, "
            f"probabilistic_mode={probabilistic_mode!r}."
        )
    if is_baseline != (not deadband_active and not probabilistic_active):
        raise ValueError(
            "Expected exactly the baseline run to program every update. "
            f"Provided value: is_baseline={is_baseline!r}, "
            f"deadband_mode={deadband_mode!r}, "
            f"deadband_relative={deadband_relative!r}, "
            f"probabilistic_mode={probabilistic_mode!r}."
        )

    if is_baseline:
        return {
            "family": "every_update",
            "mode": "none",
            "configured_value": 1.0,
            "probabilistic_write_seed": None,
        }
    if deadband_active:
        if (
            deadband_mode != "accumulated_shadow_relative_rms"
            or deadband_relative <= 0.0
        ):
            raise ValueError(
                "Expected an accumulated_shadow_relative_rms deadband with a "
                "positive relative threshold. Provided value: "
                f"mode={deadband_mode!r}, relative={deadband_relative!r}."
            )
        return {
            "family": "deterministic_deadband",
            "mode": deadband_mode,
            "configured_value": deadband_relative,
            "probabilistic_write_seed": None,
        }
    if probabilistic_mode == "uniform_bernoulli":
        if not 0.0 < probability < 1.0:
            raise ValueError(
                "Expected uniform Bernoulli probability in (0, 1). "
                f"Provided value: {probability!r}."
            )
        return {
            "family": "probabilistic",
            "mode": probabilistic_mode,
            "configured_value": probability,
            "probabilistic_write_seed": seed,
        }
    if probabilistic_mode == "displacement_proportional":
        if scale_relative <= 0.0:
            raise ValueError(
                "Expected a positive displacement-proportional relative "
                f"scale. Provided value: {scale_relative!r}."
            )
        return {
            "family": "probabilistic",
            "mode": probabilistic_mode,
            "configured_value": scale_relative,
            "probabilistic_write_seed": seed,
        }
    raise ValueError(
        "Expected probabilistic_write_mode to be uniform_bernoulli or "
        "displacement_proportional. "
        f"Provided value: {probabilistic_mode!r}."
    )


def _run_summary(
    label: str,
    run_dir: Path,
    test_result: Path,
    *,
    is_baseline: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    result = _complete_training_result(run_dir)
    metrics = result["metrics"]
    programming = metrics["update_programming"]
    if programming.get("active_cohort") != "B":
        raise ValueError(
            "Expected every comparison run to use cohort B exclusively. "
            f"Provided value: run={str(run_dir)!r}, "
            f"active_cohort={programming.get('active_cohort')!r}."
        )
    policy = _classify_policy(programming, is_baseline=is_baseline)
    parameters = dict(programming["parameters"])
    test = _held_out_test_metrics(test_result)
    selected = metrics["selected"]
    last = metrics["last_validation"]

    parameter_activity = {
        key: {
            "shape": report["shape"],
            "programming_event_fraction": report[
                "programming_event_fraction"
            ],
            "pulse_changed_fraction": report["pulse_changed_fraction"],
            "mean_write_probability": report.get("mean_write_probability"),
        }
        for key, report in sorted(parameters.items())
    }
    row = {
        "label": label,
        "run_dir": str(run_dir),
        "test_result": str(test_result),
        **policy,
        "selected_epoch": int(selected["epoch"]),
        "selected_validation_accuracy": float(selected["accuracy"]),
        "selected_validation_mean_cost": float(selected["value"]),
        "last_validation_accuracy": float(last["accuracy"]),
        "official_test_accuracy": float(test["accuracy"]),
        "official_test_mean_cost": float(test["mean_cost"]),
        "programming_event_fraction": _weighted_fraction(
            parameters,
            "programming_event_fraction",
        ),
        "pulse_changed_fraction": _weighted_fraction(
            parameters,
            "pulse_changed_fraction",
        ),
        "mean_write_probability": (
            None
            if policy["family"] != "probabilistic"
            else _weighted_fraction(parameters, "mean_write_probability")
        ),
        "parameters": parameter_activity,
    }
    return row, _comparison_identity(run_dir, result)


def summarize_write_policies(
    baseline_run_dir: Path,
    policy_runs: Sequence[tuple[str, Path, Path]],
    *,
    baseline_test_result: Path,
    output_dir: Path,
    baseline_label: str = "Every update",
) -> dict[str, Any]:
    if not policy_runs:
        raise ValueError(
            "Expected at least one non-baseline write policy. "
            "Provided value: none."
        )
    labels = [baseline_label, *(label for label, _, _ in policy_runs)]
    if len(set(labels)) != len(labels):
        raise ValueError(
            "Expected unique write-policy labels. "
            f"Provided value: {labels!r}."
        )

    baseline, identity = _run_summary(
        baseline_label,
        baseline_run_dir,
        baseline_test_result,
        is_baseline=True,
    )
    policies: list[dict[str, Any]] = []
    for label, run_dir, test_result in policy_runs:
        row, row_identity = _run_summary(
            label,
            run_dir,
            test_result,
            is_baseline=False,
        )
        if row_identity != identity:
            raise ValueError(
                "Expected policy runs to differ only in write-policy fields. "
                f"Provided value: label={label!r}, "
                f"baseline_identity={identity!r}, "
                f"policy_identity={row_identity!r}."
            )
        policies.append(row)

    accuracy_tolerance = 0.001
    for item in policies:
        item["selected_validation_accuracy_delta"] = (
            item["selected_validation_accuracy"]
            - baseline["selected_validation_accuracy"]
        )
        item["selected_validation_mean_cost_delta"] = (
            item["selected_validation_mean_cost"]
            - baseline["selected_validation_mean_cost"]
        )
        item["official_test_accuracy_delta"] = (
            item["official_test_accuracy"] - baseline["official_test_accuracy"]
        )
        item["official_test_mean_cost_delta"] = (
            item["official_test_mean_cost"]
            - baseline["official_test_mean_cost"]
        )
        item["official_test_mean_cost_relative_change"] = (
            item["official_test_mean_cost"]
            / baseline["official_test_mean_cost"]
            - 1.0
        )
        item["programming_event_reduction"] = (
            1.0
            - item["programming_event_fraction"]
            / baseline["programming_event_fraction"]
        )
        item["pulse_change_reduction"] = (
            1.0
            - item["pulse_changed_fraction"]
            / baseline["pulse_changed_fraction"]
        )
        item["retains_test_accuracy"] = (
            item["official_test_accuracy"]
            >= baseline["official_test_accuracy"] - accuracy_tolerance
        )
        item["retains_selected_validation_accuracy"] = (
            item["selected_validation_accuracy"]
            >= baseline["selected_validation_accuracy"] - accuracy_tolerance
        )

    rows = [baseline, *policies]
    probabilistic = [
        item for item in policies if item["family"] == "probabilistic"
    ]
    retained_probabilistic = [
        item
        for item in probabilistic
        if item["retains_selected_validation_accuracy"]
    ]
    recommended = (
        None
        if not retained_probabilistic
        else min(
            retained_probabilistic,
            key=lambda item: (
                item["selected_validation_mean_cost"],
                -item["selected_validation_accuracy"],
                item["programming_event_fraction"],
            ),
        )["label"]
    )
    conclusion = {
        "accuracy_retention_tolerance": accuracy_tolerance,
        "any_policy_official_test_accuracy_improvement": any(
            item["official_test_accuracy_delta"] > 0.0 for item in policies
        ),
        "any_probabilistic_official_test_accuracy_improvement": any(
            item["official_test_accuracy_delta"] > 0.0
            for item in probabilistic
        ),
        "any_probabilistic_official_test_cost_improvement": any(
            item["official_test_mean_cost_delta"] < 0.0
            for item in probabilistic
        ),
        "best_official_test_accuracy_label": max(
            rows,
            key=lambda item: item["official_test_accuracy"],
        )["label"],
        "best_official_test_mean_cost_label": min(
            rows,
            key=lambda item: item["official_test_mean_cost"],
        )["label"],
        "recommended_probabilistic_policy": recommended,
        "recommendation_basis": (
            "lowest selected-validation mean cost among probabilistic "
            "policies within 0.1 percentage points of baseline selected "
            "validation accuracy"
        ),
        "exploratory_single_training_seed_per_policy": True,
    }
    summary = {
        "schema": "measured-cohort-b-write-policy-comparison/v1",
        "comparison_identity": identity,
        "baseline": baseline,
        "policies": policies,
        "conclusion": conclusion,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    positions = list(range(len(rows)))
    width = 0.36
    figure, axes = plt.subplots(
        1,
        3,
        figsize=(14, 4.4),
        constrained_layout=True,
    )
    validation_accuracy_deltas = [
        100.0
        * (
            item["selected_validation_accuracy"]
            - baseline["selected_validation_accuracy"]
        )
        for item in rows
    ]
    test_accuracy_deltas = [
        100.0
        * (item["official_test_accuracy"] - baseline["official_test_accuracy"])
        for item in rows
    ]
    axes[0].bar(
        [position - width / 2 for position in positions],
        validation_accuracy_deltas,
        width,
        label="Selected validation",
    )
    axes[0].bar(
        [position + width / 2 for position in positions],
        test_accuracy_deltas,
        width,
        label="Held-out test",
    )
    axes[0].axhline(0.0, color="black", linewidth=0.8)
    axes[0].set_ylabel("Accuracy delta vs every update (pp)")
    axes[0].legend()

    validation_cost_deltas = [
        1000.0
        * (
            item["selected_validation_mean_cost"]
            - baseline["selected_validation_mean_cost"]
        )
        for item in rows
    ]
    test_cost_deltas = [
        1000.0
        * (item["official_test_mean_cost"] - baseline["official_test_mean_cost"])
        for item in rows
    ]
    axes[1].bar(
        [position - width / 2 for position in positions],
        validation_cost_deltas,
        width,
        label="Selected validation",
    )
    axes[1].bar(
        [position + width / 2 for position in positions],
        test_cost_deltas,
        width,
        label="Held-out test",
    )
    axes[1].axhline(0.0, color="black", linewidth=0.8)
    axes[1].set_ylabel("Mean-cost delta x 1000 (lower is better)")
    axes[1].legend()

    axes[2].bar(
        [position - width / 2 for position in positions],
        [100.0 * item["programming_event_fraction"] for item in rows],
        width,
        label="Programming requests",
    )
    axes[2].bar(
        [position + width / 2 for position in positions],
        [100.0 * item["pulse_changed_fraction"] for item in rows],
        width,
        label="Pulse-index changes",
    )
    axes[2].set_ylabel("Fraction of element-steps (%)")
    axes[2].legend()

    for axis in axes:
        axis.set_xticks(positions, [item["label"] for item in rows])
        axis.tick_params(axis="x", rotation=20)
        axis.grid(axis="y", alpha=0.25)
    figure.savefig(
        output_dir / "cohort_b_write_policy_comparison.png",
        dpi=180,
    )
    plt.close(figure)
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-run-dir", type=Path, required=True)
    parser.add_argument("--baseline-test-result", type=Path, required=True)
    parser.add_argument("--baseline-label", default="Every update")
    parser.add_argument(
        "--policy",
        action="append",
        nargs=3,
        metavar=("LABEL", "RUN_DIR", "TEST_RESULT"),
        required=True,
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    summarize_write_policies(
        args.baseline_run_dir.expanduser().resolve(),
        [
            (
                label,
                Path(run_dir).expanduser().resolve(),
                Path(test_result).expanduser().resolve(),
            )
            for label, run_dir, test_result in args.policy
        ],
        baseline_test_result=args.baseline_test_result.expanduser().resolve(),
        output_dir=args.output_dir.expanduser().resolve(),
        baseline_label=args.baseline_label,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
