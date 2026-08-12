#!/usr/bin/env python3
"""Analyze collected canonical runs from the bounded-weight wmax sweep.

The analyzer is intentionally read-only with respect to run bundles.  It joins
completed production bundles to the immutable study manifest by ``arm_id`` and
config digest, reports incomplete or ambiguous coverage, and writes lightweight
tables and figures suitable for reviewing a partially collected study.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

try:
    from experiments.reporting import (
        MANIFEST_SCHEMA,
        RESULT_SCHEMA,
        STATUS_SCHEMA,
        validate_run,
    )
except ModuleNotFoundError:  # Support direct execution from the repository root.
    from reporting import (  # type: ignore[no-redef]
        MANIFEST_SCHEMA,
        RESULT_SCHEMA,
        STATUS_SCHEMA,
        validate_run,
    )


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STUDY_MANIFEST = (
    REPO_ROOT
    / "configs/conv/"
    "perfectdiode_bounded_wmax_sweep_signed_scaled_bias_conv123_seed0_20260809_v1/"
    "study_manifest.json"
)
ANALYSIS_SCHEMA = "perfectdiode-bounded-wmax-analysis/v1"
ANCHOR_WEIGHT_MAX = 1e-4
EXPECTED_SOURCE_ARCHIVE_SHA256 = (
    "2d7f5e5e2c5728fae6fede391d0941637fbb112b1bdb3e57f346d0b289e99416"
)
SCHEME_COMPARISONS = (
    ("ours", "baseline"),
    ("legacy", "baseline"),
    ("ours", "legacy"),
)
SCHEME_COLORS = {
    "baseline": "#4c78a8",
    "ours": "#f58518",
    "legacy": "#54a24b",
}


class IncompleteCoverageError(RuntimeError):
    """Raised after outputs are written when complete coverage was required."""

    def __init__(self, report: dict[str, Any]):
        coverage = report["coverage"]
        super().__init__(
            "Bounded-wmax coverage is incomplete or ambiguous: "
            f"usable={coverage['usable_arm_count']}/"
            f"{coverage['expected_arm_count']}, "
            f"missing={len(coverage['missing_arm_ids'])}, "
            f"duplicates={len(coverage['duplicate_arm_ids'])}, "
            f"invalid_runs={len(coverage['invalid_runs'])}."
        )
        self.report = report


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Could not read JSON object {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _ordered_set_sha256(rows: Sequence[Mapping[str, Any]]) -> str:
    return hashlib.sha256(
        "".join(f"{row['config_sha256']}\n" for row in rows).encode("ascii")
    ).hexdigest()


def _finite_accuracy(value: Any, *, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or not 0.0 <= float(value) <= 1.0
    ):
        raise ValueError(f"Expected finite {label} in [0,1], got {value!r}.")
    return float(value)


def _validate_study_manifest(
    study_manifest: Mapping[str, Any], *, path: Path
) -> list[dict[str, Any]]:
    if study_manifest.get("schema_version") != (
        "perfectdiode-bounded-wmax-sweep-config-set/v1"
    ):
        raise ValueError(f"Unexpected bounded-wmax manifest schema in {path}.")
    study_id = study_manifest.get("study_id")
    if not isinstance(study_id, str) or not study_id:
        raise ValueError(f"Missing study_id in {path}.")
    rows = study_manifest.get("runs")
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"Expected a non-empty runs list in {path}.")
    if study_manifest.get("run_count") != len(rows):
        raise ValueError(f"run_count does not match runs in {path}.")
    if study_manifest.get("varied_field") != "model_base.weight_max":
        raise ValueError(f"Unexpected varied field in {path}.")

    expected_indices = list(range(len(rows)))
    indices = [row.get("global_array_index") for row in rows]
    if indices != expected_indices:
        raise ValueError(f"Expected contiguous ordered global indices in {path}.")
    arm_ids = [row.get("arm_id") for row in rows]
    if any(not isinstance(value, str) or not value for value in arm_ids):
        raise ValueError(f"Every row must have a non-empty arm_id in {path}.")
    if len(set(arm_ids)) != len(arm_ids):
        raise ValueError(f"Duplicate arm_id values in {path}.")

    group_keys: set[tuple[str, str, str, float]] = set()
    observed_wmax: list[float] = []
    seen_wmax: set[float] = set()
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError(f"Every manifest run row must be an object in {path}.")
        try:
            architecture = str(row["architecture"])
            scheme = str(row["scheme"])
            optimizer = str(row["optimizer"]).lower()
            weight_min = float(row["weight_min"])
            weight_max = float(row["weight_max"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"Invalid scientific fields in row {row!r}.") from error
        if not architecture or scheme not in {"baseline", "ours", "legacy"}:
            raise ValueError(f"Invalid architecture or scheme in row {row!r}.")
        if optimizer not in {"sgd", "adam"}:
            raise ValueError(f"Invalid optimizer in row {row!r}.")
        if not (0.0 < weight_min < weight_max):
            raise ValueError(f"Invalid weight interval in row {row!r}.")
        key = (architecture, scheme, optimizer, weight_max)
        if key in group_keys:
            raise ValueError(f"Duplicate scientific surface point {key!r} in {path}.")
        group_keys.add(key)
        if weight_max not in seen_wmax:
            observed_wmax.append(weight_max)
            seen_wmax.add(weight_max)
        if not isinstance(row.get("config_sha256"), str):
            raise ValueError(f"Missing config digest in row {row!r}.")
        if not isinstance(row.get("target"), str) or not row["target"]:
            raise ValueError(f"Missing target in row {row!r}.")

    declared_wmax = [float(value) for value in study_manifest.get("weight_max_values", [])]
    if declared_wmax != observed_wmax:
        raise ValueError(
            f"weight_max_values does not match ordered run coverage in {path}."
        )
    if ANCHOR_WEIGHT_MAX not in seen_wmax:
        raise ValueError(f"The required 1e-4 anchor is absent from {path}.")

    surface_keys = {
        (str(row["architecture"]), str(row["scheme"]), str(row["optimizer"]).lower())
        for row in rows
    }
    for surface in surface_keys:
        available = {
            float(row["weight_max"])
            for row in rows
            if (
                str(row["architecture"]),
                str(row["scheme"]),
                str(row["optimizer"]).lower(),
            )
            == surface
        }
        if available != set(declared_wmax):
            raise ValueError(
                f"Surface {surface!r} does not cover the declared wmax grid in {path}."
            )

    recorded_digest = study_manifest.get("ordered_config_set_sha256")
    if recorded_digest != _ordered_set_sha256(rows):
        raise ValueError(f"Ordered config-set digest mismatch in {path}.")
    return [dict(row) for row in rows]


def _candidate_errors(
    run_dir: Path,
    *,
    manifest: Mapping[str, Any],
    row: Mapping[str, Any],
    study_manifest: Mapping[str, Any],
) -> tuple[list[str], dict[str, Any] | None]:
    try:
        errors = list(validate_run(run_dir))
    except Exception as error:  # Malformed collected evidence is a coverage issue.
        errors = [f"canonical bundle validation failed: {error}"]
    status_path = run_dir / "status.json"
    result_path = run_dir / "result.json"
    try:
        status = _load_json(status_path)
    except ValueError as error:
        errors.append(str(error))
        status = {}
    try:
        result = _load_json(result_path)
    except ValueError as error:
        errors.append(str(error))
        result = None

    study_id = study_manifest["study_id"]
    arm_id = row["arm_id"]
    expected_evidence = study_manifest.get("evidence_class")
    if manifest.get("schema_version") != MANIFEST_SCHEMA:
        errors.append("unexpected run manifest schema")
    if manifest.get("study_id") != study_id:
        errors.append("run manifest study_id mismatch")
    if manifest.get("arm_id") != arm_id:
        errors.append("run manifest arm_id mismatch")
    if manifest.get("smoke") is not False:
        errors.append("production evidence is marked smoke")
    if expected_evidence and manifest.get("evidence_class") != expected_evidence:
        errors.append("run manifest evidence_class mismatch")
    if status.get("schema_version") != STATUS_SCHEMA:
        errors.append("unexpected status schema")
    if status.get("state") != "complete":
        errors.append(f"run status is not complete: {status.get('state')!r}")

    configuration = manifest.get("configuration", {})
    if not isinstance(configuration, Mapping):
        errors.append("missing manifest configuration")
        configuration = {}
    if configuration.get("sha256") != row["config_sha256"]:
        errors.append("scientific config digest does not match study row")
    if configuration.get("epochs") != row.get("epochs"):
        errors.append("executed epoch budget does not match study row")
    resolved = configuration.get("resolved", {})
    if not isinstance(resolved, Mapping):
        errors.append("missing resolved scientific config")
        resolved = {}
    if resolved.get("study_id") != study_id or resolved.get("arm_id") != arm_id:
        errors.append("resolved config identity does not match study row")
    lab = resolved.get("lab", {})
    if not isinstance(lab, Mapping) or lab.get("epochs") != row.get("epochs"):
        errors.append("resolved configured epoch budget does not match study row")
    model = resolved.get("model_base", {})
    if not isinstance(model, Mapping):
        errors.append("missing resolved model_base")
        model = {}
    try:
        resolved_weight_min = float(model.get("weight_min"))
        resolved_weight_max = float(model.get("weight_max"))
    except (TypeError, ValueError):
        errors.append("invalid resolved weight interval")
    else:
        if resolved_weight_min != float(row["weight_min"]):
            errors.append("resolved weight_min does not match study row")
        if resolved_weight_max != float(row["weight_max"]):
            errors.append("resolved weight_max does not match study row")
    optimizer = resolved.get("optimizer", {})
    optimizer_name = (
        optimizer.get("name") if isinstance(optimizer, Mapping) else None
    )
    if str(optimizer_name).lower() != str(row["optimizer"]).lower():
        errors.append("resolved optimizer does not match study row")
    runtime = manifest.get("runtime", {})
    if not isinstance(runtime, Mapping) or runtime.get("target") != row["target"]:
        errors.append("runtime target does not match study row")
    expected_commit = study_manifest.get("parent", {}).get("source_commit")
    git = manifest.get("git", {})
    if expected_commit and (
        not isinstance(git, Mapping) or git.get("commit") != expected_commit
    ):
        errors.append("run source commit does not match frozen parent source")
    if (
        not isinstance(git, Mapping)
        or git.get("source_archive_sha256") != EXPECTED_SOURCE_ARCHIVE_SHA256
    ):
        errors.append("run source archive does not match the sealed sweep source")

    if result is not None:
        if result.get("schema_version") != RESULT_SCHEMA:
            errors.append("unexpected result schema")
        if result.get("study_id") != study_id or result.get("arm_id") != arm_id:
            errors.append("result identity does not match study row")
        if result.get("smoke") is not False:
            errors.append("result is marked smoke")
        if expected_evidence and result.get("evidence_class") != expected_evidence:
            errors.append("result evidence_class mismatch")
        completion = result.get("completion", {})
        if not isinstance(completion, Mapping) or completion.get("criteria_met") is not True:
            errors.append("result completion criteria were not met")
        terminal = result.get("terminal_metrics", {})
        validation = terminal.get("validation", {}) if isinstance(terminal, Mapping) else {}
        try:
            best_accuracy = _finite_accuracy(
                validation.get("best_accuracy"), label="best validation accuracy"
            )
            final_accuracy = _finite_accuracy(
                validation.get("final_accuracy"), label="final validation accuracy"
            )
        except ValueError as error:
            errors.append(str(error))
        else:
            metrics = {
                "best_epoch": terminal.get("best_epoch"),
                "best_validation_accuracy": best_accuracy,
                "final_validation_accuracy": final_accuracy,
            }
            return sorted(set(errors)), metrics
    return sorted(set(errors)), None


def _discover_and_join(
    result_roots: Sequence[Path],
    *,
    study_manifest: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    study_id = str(study_manifest["study_id"])
    rows_by_arm = {str(row["arm_id"]): row for row in rows}
    discovered_dirs: set[Path] = set()
    ignored_runs: list[dict[str, str]] = []
    invalid_runs: list[dict[str, Any]] = []
    valid_by_arm: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for root in result_roots:
        root = Path(root).expanduser().resolve()
        if not root.is_dir():
            raise ValueError(f"Result root is not a directory: {root}.")
        for manifest_path in sorted(root.rglob("manifest.json")):
            run_dir = manifest_path.parent.resolve()
            if run_dir in discovered_dirs:
                continue
            discovered_dirs.add(run_dir)
            try:
                manifest = _load_json(manifest_path)
            except ValueError as error:
                ignored_runs.append(
                    {"run_dir": str(run_dir), "reason": str(error)}
                )
                continue
            if manifest.get("study_id") != study_id:
                ignored_runs.append(
                    {"run_dir": str(run_dir), "reason": "different study_id"}
                )
                continue
            if manifest.get("smoke") is True:
                ignored_runs.append(
                    {"run_dir": str(run_dir), "reason": "smoke/canary"}
                )
                continue
            arm_id = manifest.get("arm_id")
            row = rows_by_arm.get(str(arm_id))
            if row is None:
                invalid_runs.append(
                    {
                        "arm_id": arm_id,
                        "errors": ["arm_id is absent from study manifest"],
                        "run_dir": str(run_dir),
                    }
                )
                continue
            errors, metrics = _candidate_errors(
                run_dir,
                manifest=manifest,
                row=row,
                study_manifest=study_manifest,
            )
            if errors or metrics is None:
                invalid_runs.append(
                    {
                        "arm_id": arm_id,
                        "errors": errors or ["terminal metrics are unavailable"],
                        "run_dir": str(run_dir),
                    }
                )
                continue
            valid_by_arm[str(arm_id)].append(
                {
                    "manifest": manifest,
                    "metrics": metrics,
                    "run_dir": str(run_dir),
                }
            )

    duplicate_arm_ids = sorted(
        arm_id for arm_id, candidates in valid_by_arm.items() if len(candidates) > 1
    )
    usable = {
        arm_id: candidates[0]
        for arm_id, candidates in valid_by_arm.items()
        if len(candidates) == 1
    }
    missing_arm_ids = sorted(
        set(rows_by_arm) - set(valid_by_arm),
        key=lambda arm_id: int(rows_by_arm[arm_id]["global_array_index"]),
    )
    duplicate_runs = {
        arm_id: [candidate["run_dir"] for candidate in valid_by_arm[arm_id]]
        for arm_id in duplicate_arm_ids
    }
    coverage = {
        "complete": (
            len(usable) == len(rows)
            and not missing_arm_ids
            and not duplicate_arm_ids
            and not invalid_runs
        ),
        "discovered_manifest_count": len(discovered_dirs),
        "duplicate_arm_ids": duplicate_arm_ids,
        "duplicate_runs": duplicate_runs,
        "expected_arm_count": len(rows),
        "ignored_runs": sorted(ignored_runs, key=lambda item: item["run_dir"]),
        "invalid_runs": sorted(invalid_runs, key=lambda item: item["run_dir"]),
        "missing_arm_ids": missing_arm_ids,
        "usable_arm_count": len(usable),
    }
    return usable, coverage


def _accuracy_rows(
    rows: Sequence[Mapping[str, Any]],
    usable: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    accepted_by_key: dict[tuple[str, str, str, float], Mapping[str, Any]] = {}
    for row in rows:
        candidate = usable.get(str(row["arm_id"]))
        if candidate is not None:
            key = (
                str(row["architecture"]),
                str(row["scheme"]),
                str(row["optimizer"]).lower(),
                float(row["weight_max"]),
            )
            accepted_by_key[key] = candidate

    output = []
    for row in rows:
        architecture = str(row["architecture"])
        scheme = str(row["scheme"])
        optimizer = str(row["optimizer"]).lower()
        weight_max = float(row["weight_max"])
        candidate = usable.get(str(row["arm_id"]))
        anchor = accepted_by_key.get(
            (architecture, scheme, optimizer, ANCHOR_WEIGHT_MAX)
        )
        metrics = candidate["metrics"] if candidate else None
        anchor_metrics = anchor["metrics"] if anchor else None
        best = metrics["best_validation_accuracy"] if metrics else None
        final = metrics["final_validation_accuracy"] if metrics else None
        anchor_best = (
            anchor_metrics["best_validation_accuracy"] if anchor_metrics else None
        )
        anchor_final = (
            anchor_metrics["final_validation_accuracy"] if anchor_metrics else None
        )
        delta_best = (
            best - anchor_best if best is not None and anchor_best is not None else None
        )
        delta_final = (
            final - anchor_final
            if final is not None and anchor_final is not None
            else None
        )
        output.append(
            {
                "global_array_index": int(row["global_array_index"]),
                "arm_id": row["arm_id"],
                "architecture": architecture,
                "scheme": scheme,
                "optimizer": optimizer,
                "target": row["target"],
                "weight_min": float(row["weight_min"]),
                "weight_max": weight_max,
                "weight_range_ratio": float(row["weight_range_ratio"]),
                "available": candidate is not None,
                "anchor_available": anchor is not None,
                "run_dir": candidate["run_dir"] if candidate else None,
                "best_epoch": metrics["best_epoch"] if metrics else None,
                "best_validation_accuracy": best,
                "final_validation_accuracy": final,
                "delta_best_vs_1e-4": delta_best,
                "delta_best_vs_1e-4_percentage_points": (
                    100.0 * delta_best if delta_best is not None else None
                ),
                "delta_final_vs_1e-4": delta_final,
                "delta_final_vs_1e-4_percentage_points": (
                    100.0 * delta_final if delta_final is not None else None
                ),
            }
        )
    return output


def _scheme_gap_rows(
    study_rows: Sequence[Mapping[str, Any]],
    accuracy_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    by_key = {
        (
            str(row["architecture"]),
            str(row["scheme"]),
            str(row["optimizer"]),
            float(row["weight_max"]),
        ): row
        for row in accuracy_rows
    }
    architecture_optimizer_wmax = sorted(
        {
            (
                str(row["architecture"]),
                str(row["optimizer"]).lower(),
                float(row["weight_max"]),
            )
            for row in study_rows
        }
    )
    output = []
    for architecture, optimizer, weight_max in architecture_optimizer_wmax:
        for lhs_scheme, rhs_scheme in SCHEME_COMPARISONS:
            lhs = by_key.get((architecture, lhs_scheme, optimizer, weight_max))
            rhs = by_key.get((architecture, rhs_scheme, optimizer, weight_max))
            lhs_best = lhs.get("best_validation_accuracy") if lhs else None
            rhs_best = rhs.get("best_validation_accuracy") if rhs else None
            lhs_final = lhs.get("final_validation_accuracy") if lhs else None
            rhs_final = rhs.get("final_validation_accuracy") if rhs else None
            best_gap = (
                lhs_best - rhs_best
                if lhs_best is not None and rhs_best is not None
                else None
            )
            final_gap = (
                lhs_final - rhs_final
                if lhs_final is not None and rhs_final is not None
                else None
            )
            output.append(
                {
                    "architecture": architecture,
                    "optimizer": optimizer,
                    "weight_max": weight_max,
                    "lhs_scheme": lhs_scheme,
                    "rhs_scheme": rhs_scheme,
                    "gap_definition": f"{lhs_scheme}_minus_{rhs_scheme}",
                    "lhs_available": bool(lhs and lhs["available"]),
                    "rhs_available": bool(rhs and rhs["available"]),
                    "best_validation_accuracy_gap": best_gap,
                    "best_validation_accuracy_gap_percentage_points": (
                        100.0 * best_gap if best_gap is not None else None
                    ),
                    "final_validation_accuracy_gap": final_gap,
                    "final_validation_accuracy_gap_percentage_points": (
                        100.0 * final_gap if final_gap is not None else None
                    ),
                }
            )
    return output


def _coverage_by_architecture(
    rows: Sequence[Mapping[str, Any]], accuracy_rows: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    expected = Counter(str(row["architecture"]) for row in rows)
    usable = Counter(
        str(row["architecture"]) for row in accuracy_rows if row["available"]
    )
    return [
        {
            "architecture": architecture,
            "expected_arm_count": expected[architecture],
            "usable_arm_count": usable[architecture],
            "complete": expected[architecture] == usable[architecture],
        }
        for architecture in sorted(expected)
    ]


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write a fieldless CSV: {path}.")
    fieldnames = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def _ordered_unique(values: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(values))


def _save_plots(
    analysis_dir: Path,
    *,
    study_rows: Sequence[Mapping[str, Any]],
    accuracy_rows: Sequence[Mapping[str, Any]],
    scheme_gap_rows: Sequence[Mapping[str, Any]],
) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    outputs: list[str] = []
    architectures = _ordered_unique(str(row["architecture"]) for row in study_rows)
    for architecture in architectures:
        architecture_rows = [
            row for row in accuracy_rows if row["architecture"] == architecture
        ]
        optimizers = _ordered_unique(str(row["optimizer"]) for row in architecture_rows)
        for kind, metric_names, ylabels, filename_prefix in (
            (
                "accuracy",
                ("best_validation_accuracy", "final_validation_accuracy"),
                ("Best validation accuracy (%)", "Final validation accuracy (%)"),
                "accuracy_vs_wmax",
            ),
            (
                "delta",
                ("delta_best_vs_1e-4", "delta_final_vs_1e-4"),
                (
                    "Best accuracy change vs 1e-4 (pp)",
                    "Final accuracy change vs 1e-4 (pp)",
                ),
                "delta_vs_anchor",
            ),
        ):
            fig, axes = plt.subplots(
                len(optimizers), 2, figsize=(10.5, 3.5 * len(optimizers)), squeeze=False
            )
            for optimizer_index, optimizer in enumerate(optimizers):
                for metric_index, (metric_name, ylabel) in enumerate(
                    zip(metric_names, ylabels)
                ):
                    axis = axes[optimizer_index][metric_index]
                    for scheme in ("baseline", "ours", "legacy"):
                        points = sorted(
                            (
                                (float(row["weight_max"]), row[metric_name])
                                for row in architecture_rows
                                if row["optimizer"] == optimizer
                                and row["scheme"] == scheme
                                and row[metric_name] is not None
                            ),
                            key=lambda item: item[0],
                        )
                        if not points:
                            continue
                        axis.plot(
                            [point[0] for point in points],
                            [100.0 * point[1] for point in points],
                            color=SCHEME_COLORS[scheme],
                            marker="o",
                            label=scheme,
                        )
                    axis.set_xscale("log")
                    axis.set_xlabel("Weight maximum")
                    axis.set_ylabel(ylabel)
                    axis.set_title(f"{architecture.upper()} · {optimizer.upper()}")
                    axis.grid(True, alpha=0.25)
                    if kind == "delta":
                        axis.axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
                    handles, labels = axis.get_legend_handles_labels()
                    if handles:
                        axis.legend(handles, labels, frameon=False)
            fig.suptitle(
                f"{architecture.upper()} bounded-weight {kind} sensitivity",
                fontsize=13,
            )
            fig.tight_layout()
            filename = f"{filename_prefix}_{architecture}.png"
            fig.savefig(analysis_dir / filename, dpi=180, bbox_inches="tight")
            plt.close(fig)
            outputs.append(filename)

        architecture_gaps = [
            row for row in scheme_gap_rows if row["architecture"] == architecture
        ]
        fig, axes = plt.subplots(
            len(optimizers), 2, figsize=(10.5, 3.5 * len(optimizers)), squeeze=False
        )
        for optimizer_index, optimizer in enumerate(optimizers):
            for metric_index, (metric_name, ylabel) in enumerate(
                (
                    (
                        "best_validation_accuracy_gap",
                        "Best validation scheme gap (pp)",
                    ),
                    (
                        "final_validation_accuracy_gap",
                        "Final validation scheme gap (pp)",
                    ),
                )
            ):
                axis = axes[optimizer_index][metric_index]
                for comparison in (
                    "ours_minus_baseline",
                    "legacy_minus_baseline",
                    "ours_minus_legacy",
                ):
                    points = sorted(
                        (
                            (float(row["weight_max"]), row[metric_name])
                            for row in architecture_gaps
                            if row["optimizer"] == optimizer
                            and row["gap_definition"] == comparison
                            and row[metric_name] is not None
                        ),
                        key=lambda item: item[0],
                    )
                    if points:
                        axis.plot(
                            [point[0] for point in points],
                            [100.0 * point[1] for point in points],
                            marker="o",
                            label=comparison.replace("_", " "),
                        )
                axis.set_xscale("log")
                axis.set_xlabel("Weight maximum")
                axis.set_ylabel(ylabel)
                axis.set_title(f"{architecture.upper()} · {optimizer.upper()}")
                axis.axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
                axis.grid(True, alpha=0.25)
                handles, labels = axis.get_legend_handles_labels()
                if handles:
                    axis.legend(handles, labels, frameon=False, fontsize=8)
        fig.suptitle(f"{architecture.upper()} amplification-scheme gaps", fontsize=13)
        fig.tight_layout()
        filename = f"scheme_gaps_vs_wmax_{architecture}.png"
        fig.savefig(analysis_dir / filename, dpi=180, bbox_inches="tight")
        plt.close(fig)
        outputs.append(filename)
    return outputs


def analyze(
    *,
    study_manifest_path: Path,
    result_roots: Sequence[Path],
    analysis_dir: Path,
    require_complete: bool = False,
) -> dict[str, Any]:
    study_manifest_path = Path(study_manifest_path).expanduser().resolve()
    study_manifest = _load_json(study_manifest_path)
    study_rows = _validate_study_manifest(
        study_manifest, path=study_manifest_path
    )
    if not result_roots:
        raise ValueError("Expected at least one local result root.")
    resolved_roots = [Path(path).expanduser().resolve() for path in result_roots]
    usable, coverage = _discover_and_join(
        resolved_roots,
        study_manifest=study_manifest,
        rows=study_rows,
    )
    accuracy_rows = _accuracy_rows(study_rows, usable)
    scheme_gap_rows = _scheme_gap_rows(study_rows, accuracy_rows)
    coverage["by_architecture"] = _coverage_by_architecture(
        study_rows, accuracy_rows
    )

    analysis_dir = Path(analysis_dir).expanduser().resolve()
    analysis_dir.mkdir(parents=True, exist_ok=True)
    accuracy_csv = analysis_dir / "accuracy_by_surface.csv"
    anchor_csv = analysis_dir / "anchor_deltas.csv"
    gap_csv = analysis_dir / "scheme_gaps.csv"
    _write_csv(accuracy_csv, accuracy_rows)
    anchor_fields = (
        "global_array_index",
        "arm_id",
        "architecture",
        "scheme",
        "optimizer",
        "weight_max",
        "available",
        "anchor_available",
        "delta_best_vs_1e-4",
        "delta_best_vs_1e-4_percentage_points",
        "delta_final_vs_1e-4",
        "delta_final_vs_1e-4_percentage_points",
    )
    anchor_rows = [
        {field: row[field] for field in anchor_fields} for row in accuracy_rows
    ]
    _write_csv(anchor_csv, anchor_rows)
    _write_csv(gap_csv, scheme_gap_rows)
    plot_files = _save_plots(
        analysis_dir,
        study_rows=study_rows,
        accuracy_rows=accuracy_rows,
        scheme_gap_rows=scheme_gap_rows,
    )
    output_files = [
        accuracy_csv.name,
        anchor_csv.name,
        gap_csv.name,
        *plot_files,
        "analysis_summary.json",
    ]
    report = {
        "schema_version": ANALYSIS_SCHEMA,
        "study_id": study_manifest["study_id"],
        "evidence_class": study_manifest.get("evidence_class"),
        "paper_facing": bool(study_manifest.get("paper_facing", False)),
        "anchor_weight_max": ANCHOR_WEIGHT_MAX,
        "study_manifest": {
            "path": str(study_manifest_path),
            "sha256": _sha256(study_manifest_path),
        },
        "result_roots": [str(path) for path in resolved_roots],
        "coverage": coverage,
        "accuracy_by_surface": accuracy_rows,
        "scheme_gaps": scheme_gap_rows,
        "outputs": output_files,
    }
    summary_path = analysis_dir / "analysis_summary.json"
    summary_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    if require_complete and not coverage["complete"]:
        raise IncompleteCoverageError(report)
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--study-manifest", type=Path, default=DEFAULT_STUDY_MANIFEST
    )
    parser.add_argument(
        "--result-root",
        type=Path,
        action="append",
        required=True,
        help="Local collected result root; repeat for multiple roots.",
    )
    parser.add_argument("--analysis-dir", type=Path, required=True)
    parser.add_argument(
        "--require-complete",
        action="store_true",
        help="Write diagnostics, then exit nonzero unless all arms are uniquely valid.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        report = analyze(
            study_manifest_path=args.study_manifest,
            result_roots=args.result_root,
            analysis_dir=args.analysis_dir,
            require_complete=args.require_complete,
        )
    except IncompleteCoverageError as error:
        print(str(error), file=sys.stderr)
        return 2
    summary = {
        "analysis_dir": str(Path(args.analysis_dir).expanduser().resolve()),
        "complete": report["coverage"]["complete"],
        "expected_arm_count": report["coverage"]["expected_arm_count"],
        "study_id": report["study_id"],
        "usable_arm_count": report["coverage"]["usable_arm_count"],
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
