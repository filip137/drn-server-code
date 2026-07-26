#!/usr/bin/env python3
"""Curate the active six-row Conv LR handoff from frozen v1 and v3 studies.

The immutable v3 selection contains only the four amplified rows that v3
actually screens.  This post-selection tool leaves that bundle untouched and
combines it with the two frozen v1 baselines.  Baseline observed relative rho
is derived from the immutable v1 candidate diagnostics using the v3
normalization and aggregation rule; it is not substituted for v1's native
span-normalized selection coordinate.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping

# Support both ``python -m labs.tools...`` and direct script execution.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.mnist_conv.identity import sha256_file
from experiments.mnist_conv.io import atomic_write_csv, atomic_write_json, read_json
from experiments.mnist_conv.lr_artifacts import validate_stage_completion
from experiments.mnist_conv.lr_protocol import linear_quantile
from experiments.mnist_conv.lr_study_spec import LRStudySpec


CURATION_SCHEMA_VERSION = "mnist-conv-lr-active-handoff-curation/v1"
BASELINE_WINDOW_START = 829
BASELINE_WINDOW_END = 860
BASELINE_WINDOW_STEPS = tuple(range(BASELINE_WINDOW_START, BASELINE_WINDOW_END + 1))

RAW_LR_PLOT = "selected_raw_peak_lr_vs_input_gain_active_six_row.png"
RELATIVE_RHO_PLOT = "observed_peak_rho_relative_vs_input_gain_active_six_row.png"
PROVENANCE_FILE = "provenance.json"

CSV_FIELDS = [
    "architecture",
    "scheme",
    "row_id",
    "input_gain",
    "inference_iterations",
    "training_iterations",
    "source_study_id",
    "source_selection_sha256",
    "source_selection_schema_version",
    "selection_status",
    "unresolved_reason",
    "selected_candidate_role",
    "selected_peak_learning_rate",
    "selected_rho_target_native",
    "selected_rho_target_coordinate",
    "observed_peak_rho_relative",
    "observed_peak_rho_relative_source",
    "observed_peak_rho_span",
    "final_validation_loss",
    "final_validation_accuracy",
    "official_test_read",
]


def _error(expected: str, provided: Any, path: str) -> ValueError:
    return ValueError(f"Expected {path} to be {expected}. Provided value: {provided!r}.")


def _finite(value: Any, path: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise _error("a finite number", value, path)
    result = float(value)
    if not math.isfinite(result) or (positive and result <= 0.0):
        qualifier = "a positive finite number" if positive else "a finite number"
        raise _error(qualifier, value, path)
    return result


def _artifact(path: Path, *, name: str | None = None) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise _error("an existing regular file", str(path), "artifact")
    return {
        "name": path.name if name is None else name,
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }


def _selection_paths(study_dir: Path) -> tuple[Path, Path, Path]:
    stage = study_dir / "stages/select"
    return (
        stage / "manifest.json",
        stage / "complete.json",
        stage / "entries/selection/selection.json",
    )


def _load_completed_selection(
    study_dir: Path,
    *,
    expected_schema: str,
    expected_selection_schema: str,
) -> tuple[LRStudySpec, dict[str, Any], dict[str, Any]]:
    contract_path = study_dir / "study.resolved.json"
    spec = LRStudySpec.from_path(contract_path)
    if spec.data["schema_version"] != expected_schema:
        raise _error(
            f"exactly {expected_schema!r}",
            spec.data["schema_version"],
            f"{study_dir.name}.schema_version",
        )
    manifest_path, completion_path, selection_path = _selection_paths(study_dir)
    validate_stage_completion(
        study_dir=study_dir,
        manifest_path=manifest_path,
        completion_path=completion_path,
    )
    selection = read_json(selection_path)
    if not isinstance(selection, dict):
        raise _error("a JSON object", selection, str(selection_path))
    if selection.get("schema_version") != expected_selection_schema:
        raise _error(
            f"exactly {expected_selection_schema!r}",
            selection.get("schema_version"),
            f"{selection_path}.schema_version",
        )
    if selection.get("final_paper_training_authorized") is not False:
        raise _error(
            "exactly false",
            selection.get("final_paper_training_authorized"),
            f"{selection_path}.final_paper_training_authorized",
        )
    rows = selection.get("rows")
    expected_row_ids = [row["row_id"] for row in spec.rows]
    if not isinstance(rows, list) or [row.get("row_id") for row in rows] != expected_row_ids:
        raise _error(
            f"rows in frozen study order {expected_row_ids!r}",
            rows,
            f"{selection_path}.rows",
        )
    source = {
        "directory_name": study_dir.name,
        "study_id": spec.study_id,
        "resolved_config": _artifact(contract_path),
        "selection_manifest": _artifact(manifest_path),
        "selection_completion": _artifact(completion_path),
        "selection": _artifact(selection_path),
        "selection_schema_version": selection["schema_version"],
    }
    return spec, selection, source


def derive_baseline_relative_rho(
    v1_study: Path,
    *,
    row_id: str,
    architecture: str,
    candidate_role: str,
) -> dict[str, Any]:
    """Derive v3-coordinate observed rho from one selected v1 candidate."""

    metadata_path = v1_study / f"initialization/{architecture}.json"
    diagnostics_path = (
        v1_study
        / "stages/candidates/entries"
        / f"{row_id}--{candidate_role}"
        / "parameter_diagnostics.csv"
    )
    metadata = read_json(metadata_path)
    parameter_diagnostics = metadata.get("parameter_diagnostics")
    if not isinstance(parameter_diagnostics, dict):
        raise _error("a parameter diagnostics object", parameter_diagnostics, str(metadata_path))
    initial_rms = {
        name: _finite(record.get("rms"), f"{metadata_path}:{name}.rms", positive=True)
        for name, record in parameter_diagnostics.items()
        if record.get("bounded_gate") is True
    }
    if not initial_rms:
        raise _error("at least one bounded tensor RMS", initial_rms, str(metadata_path))

    relative: dict[str, dict[int, float]] = {name: {} for name in initial_rms}
    span_values: list[float] = []
    if diagnostics_path.is_symlink() or not diagnostics_path.is_file():
        raise _error("an existing regular candidate diagnostics CSV", str(diagnostics_path), "diagnostics")
    with diagnostics_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"step", "parameter", "bounded_gate", "proposed_update_rms"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise _error(
                f"CSV columns including {sorted(required)!r}",
                reader.fieldnames,
                str(diagnostics_path),
            )
        for record in reader:
            try:
                step = int(record["step"])
            except (TypeError, ValueError) as exc:
                raise _error("an integer", record.get("step"), f"{diagnostics_path}:step") from exc
            if step not in BASELINE_WINDOW_STEPS or record["bounded_gate"] != "True":
                continue
            name = record["parameter"]
            if name not in initial_rms:
                raise _error(
                    f"one of the initialized bounded tensors {sorted(initial_rms)!r}",
                    name,
                    f"{diagnostics_path}:parameter",
                )
            if step in relative[name]:
                raise _error(
                    "one record per bounded tensor and warm-up step",
                    {"parameter": name, "step": step},
                    str(diagnostics_path),
                )
            proposal = _finite(
                float(record["proposed_update_rms"]),
                f"{diagnostics_path}:{name}:step={step}:proposed_update_rms",
            )
            if proposal < 0.0:
                raise _error("a non-negative finite number", proposal, "proposed_update_rms")
            relative[name][step] = proposal / initial_rms[name]
            span_values.append(proposal / 100.0)

    expected_steps = set(BASELINE_WINDOW_STEPS)
    for name, values in relative.items():
        if set(values) != expected_steps:
            raise _error(
                f"exactly warm-up steps {BASELINE_WINDOW_START}-{BASELINE_WINDOW_END}",
                sorted(values),
                f"{diagnostics_path}:{name}",
            )
    by_parameter = {
        name: linear_quantile((values[step] for step in BASELINE_WINDOW_STEPS), 0.9)
        for name, values in relative.items()
    }
    limiting = max(by_parameter, key=by_parameter.__getitem__)
    return {
        "rho_relative": by_parameter[limiting],
        "rho_relative_by_parameter": by_parameter,
        "limiting_parameter": limiting,
        "rho_span_rederived": linear_quantile(span_values, 0.9),
        "window_start_step": BASELINE_WINDOW_START,
        "window_end_step": BASELINE_WINDOW_END,
        "window_steps_per_parameter": len(BASELINE_WINDOW_STEPS),
        "aggregation": "max_parameter(linear_Q90_over_steps(parameter_update_rms/initial_parameter_rms))",
        "candidate_diagnostics": {
            "path": diagnostics_path.relative_to(v1_study).as_posix(),
            **_artifact(diagnostics_path),
        },
        "initialization_metadata": {
            "path": metadata_path.relative_to(v1_study).as_posix(),
            **_artifact(metadata_path),
        },
    }


def _selection_row_map(selection: Mapping[str, Any], path: str) -> dict[str, dict[str, Any]]:
    rows = selection.get("rows")
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        raise _error("a list of row objects", rows, path)
    mapped = {row.get("row_id"): row for row in rows}
    if len(mapped) != len(rows) or None in mapped:
        raise _error("rows with unique non-null row_id values", rows, path)
    return mapped


def compose_active_rows(
    *,
    frozen_six_rows: Iterable[Mapping[str, Any]],
    v1_selection: Mapping[str, Any],
    v3_selection: Mapping[str, Any],
    v1_study_id: str,
    v3_study_id: str,
    v1_selection_sha256: str,
    v3_selection_sha256: str,
    baseline_derivations: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Return the active six rows with coordinate semantics made explicit."""

    v1_rows = _selection_row_map(v1_selection, "v1_selection.rows")
    v3_rows = _selection_row_map(v3_selection, "v3_selection.rows")
    output: list[dict[str, Any]] = []
    for frozen in frozen_six_rows:
        row_id = str(frozen["row_id"])
        scheme = str(frozen["scheme"])
        if scheme == "baseline":
            selected = v1_rows.get(row_id)
            if selected is None:
                raise _error("a frozen v1 baseline selection row", row_id, "v1_selection")
            if selected.get("status") != "frozen_seed0_screen":
                raise _error(
                    "a frozen_seed0_screen baseline",
                    selected.get("status"),
                    f"v1_selection.rows[{row_id}].status",
                )
            derivation = baseline_derivations.get(row_id)
            if derivation is None:
                raise _error("a baseline relative-rho derivation", row_id, "baseline_derivations")
            native_span = _finite(
                selected.get("observed_peak_rho"),
                f"v1_selection.rows[{row_id}].observed_peak_rho",
                positive=True,
            )
            rederived_span = _finite(
                derivation.get("rho_span_rederived"),
                f"baseline_derivations[{row_id}].rho_span_rederived",
                positive=True,
            )
            if not math.isclose(native_span, rederived_span, rel_tol=1e-12, abs_tol=1e-15):
                raise _error(
                    f"the native v1 span rho {native_span!r}",
                    rederived_span,
                    f"baseline_derivations[{row_id}].rho_span_rederived",
                )
            source_study_id = v1_study_id
            source_selection_sha256 = v1_selection_sha256
            source_schema = str(v1_selection["schema_version"])
            target_coordinate = "conductance_bound_span"
            relative_rho = _finite(
                derivation.get("rho_relative"),
                f"baseline_derivations[{row_id}].rho_relative",
                positive=True,
            )
            relative_source = "rederived_from_v1_candidate_log"
            span_rho = native_span
        else:
            selected = v3_rows.get(row_id)
            if selected is None:
                raise _error("a native v3 amplified selection row", row_id, "v3_selection")
            source_study_id = v3_study_id
            source_selection_sha256 = v3_selection_sha256
            source_schema = str(v3_selection["schema_version"])
            target_coordinate = "initial_parameter_rms"
            relative_rho = selected.get("observed_peak_rho_relative")
            relative_source = "native_v3_selection"
            span_rho = selected.get("observed_peak_rho_span")
            if selected.get("status") == "frozen_seed0_screen":
                _finite(relative_rho, f"v3_selection.rows[{row_id}].observed_peak_rho_relative", positive=True)
                _finite(span_rho, f"v3_selection.rows[{row_id}].observed_peak_rho_span", positive=True)

        for key in ("architecture", "scheme", "input_gain", "inference_iterations", "training_iterations"):
            if selected.get(key) != frozen[key]:
                raise _error(
                    f"the frozen row value {frozen[key]!r}",
                    selected.get(key),
                    f"selection.rows[{row_id}].{key}",
                )
        output.append(
            {
                "architecture": frozen["architecture"],
                "scheme": scheme,
                "row_id": row_id,
                "input_gain": frozen["input_gain"],
                "inference_iterations": frozen["inference_iterations"],
                "training_iterations": frozen["training_iterations"],
                "source_study_id": source_study_id,
                "source_selection_sha256": source_selection_sha256,
                "source_selection_schema_version": source_schema,
                "selection_status": selected.get("status"),
                "unresolved_reason": selected.get("reason"),
                "selected_candidate_role": selected.get("selected_candidate_role"),
                "selected_peak_learning_rate": selected.get("selected_peak_learning_rate"),
                "selected_rho_target_native": selected.get("selected_rho_target"),
                "selected_rho_target_coordinate": target_coordinate,
                "observed_peak_rho_relative": relative_rho,
                "observed_peak_rho_relative_source": relative_source,
                "observed_peak_rho_span": span_rho,
                "final_validation_loss": selected.get("final_validation_loss"),
                "final_validation_accuracy": selected.get("final_validation_accuracy"),
                "official_test_read": False,
            }
        )
    return output


def plot_active_rows(
    rows: list[Mapping[str, Any]],
    path: Path,
    *,
    field: str,
    ylabel: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"baseline": "#1f77b4", "ours": "#ff7f0e", "legacy": "#2ca02c"}
    architectures = ("conv1", "conv2")
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.2), squeeze=False)
    for axis, architecture in zip(axes[0], architectures):
        for scheme in ("baseline", "ours", "legacy"):
            points = [
                row
                for row in rows
                if row["architecture"] == architecture
                and row["scheme"] == scheme
                and row.get(field) is not None
            ]
            if points:
                x = [_finite(row["input_gain"], f"{row['row_id']}.input_gain", positive=True) for row in points]
                y = [_finite(row[field], f"{row['row_id']}.{field}", positive=True) for row in points]
                axis.scatter(x, y, color=colors[scheme], label=scheme, s=55)
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_title(architecture.capitalize())
        axis.set_xlabel("Frozen input_gain")
        axis.grid(True, which="both", alpha=0.25)
    axes[0, 0].set_ylabel(ylabel)
    legend: dict[str, Any] = {}
    for axis in axes[0]:
        handles, labels = axis.get_legend_handles_labels()
        for handle, label in zip(handles, labels):
            legend.setdefault(label, handle)
    if legend:
        fig.legend(list(legend.values()), list(legend), loc="upper center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def curate_active_handoff(
    *,
    v1_study: str | Path,
    v3_study: str | Path,
    output_dir: str | Path,
    output_csv: str | Path,
) -> dict[str, Any]:
    v1_root = Path(v1_study).expanduser().resolve()
    v3_root = Path(v3_study).expanduser().resolve()
    destination = Path(output_dir).expanduser().resolve()
    csv_path = Path(output_csv).expanduser().resolve()

    v1_spec, v1_selection, v1_source = _load_completed_selection(
        v1_root,
        expected_schema="mnist-conv-lr-study/v1",
        expected_selection_schema="mnist-conv-lr-selection/v1",
    )
    # Baseline derivation reads candidate diagnostics, so validate the entire
    # immutable v1 candidate stage rather than trusting selection summaries alone.
    validate_stage_completion(
        study_dir=v1_root,
        manifest_path=v1_root / "stages/candidates/manifest.json",
        completion_path=v1_root / "stages/candidates/complete.json",
    )
    v3_spec, v3_selection, v3_source = _load_completed_selection(
        v3_root,
        expected_schema="mnist-conv-lr-study/v3",
        expected_selection_schema="mnist-conv-lr-selection/v2",
    )
    if v3_selection.get("rho_definition") != "initial_parameter_rms":
        raise _error(
            "exactly 'initial_parameter_rms'",
            v3_selection.get("rho_definition"),
            "v3_selection.rho_definition",
        )

    v1_split_path = v1_root / "split/provenance.json"
    v3_split_path = v3_root / "split/provenance.json"
    v1_split = read_json(v1_split_path)
    v3_split = read_json(v3_split_path)
    for name, split in (("v1", v1_split), ("v3", v3_split)):
        if split.get("official_test_read") is not False:
            raise _error("exactly false", split.get("official_test_read"), f"{name}_split.official_test_read")
    for key in ("train_indices_sha256", "validation_indices_sha256"):
        if v1_split.get(key) != v3_split.get(key):
            raise _error(
                f"the v1 value {v1_split.get(key)!r}",
                v3_split.get(key),
                f"v3_split.{key}",
            )

    v1_rows = _selection_row_map(v1_selection, "v1_selection.rows")
    baseline_derivations: dict[str, dict[str, Any]] = {}
    for frozen in v1_spec.rows:
        if frozen["scheme"] != "baseline":
            continue
        selected = v1_rows[frozen["row_id"]]
        role = selected.get("selected_candidate_role")
        if not isinstance(role, str) or not role:
            raise _error("a selected candidate role", role, f"v1_selection.rows[{frozen['row_id']}]")
        baseline_derivations[frozen["row_id"]] = derive_baseline_relative_rho(
            v1_root,
            row_id=frozen["row_id"],
            architecture=frozen["architecture"],
            candidate_role=role,
        )

    expected_v3_ids = [row["row_id"] for row in v1_spec.rows if row["scheme"] != "baseline"]
    if [row["row_id"] for row in v3_spec.rows] != expected_v3_ids:
        raise _error(
            f"the four unresolved amplified v1 rows {expected_v3_ids!r}",
            [row["row_id"] for row in v3_spec.rows],
            "v3_study.rows",
        )
    rows = compose_active_rows(
        frozen_six_rows=v1_spec.rows,
        v1_selection=v1_selection,
        v3_selection=v3_selection,
        v1_study_id=v1_spec.study_id,
        v3_study_id=v3_spec.study_id,
        v1_selection_sha256=v1_source["selection"]["sha256"],
        v3_selection_sha256=v3_source["selection"]["sha256"],
        baseline_derivations=baseline_derivations,
    )

    destination.mkdir(parents=True, exist_ok=True)
    atomic_write_csv(csv_path, CSV_FIELDS, rows)
    raw_plot = destination / RAW_LR_PLOT
    relative_plot = destination / RELATIVE_RHO_PLOT
    plot_active_rows(
        rows,
        raw_plot,
        field="selected_peak_learning_rate",
        ylabel="Selected raw peak learning rate",
    )
    plot_active_rows(
        rows,
        relative_plot,
        field="observed_peak_rho_relative",
        ylabel="Observed peak update / initial parameter RMS",
    )

    provenance = {
        "schema_version": CURATION_SCHEMA_VERSION,
        "status": (
            "complete"
            if all(row["selection_status"] == "frozen_seed0_screen" for row in rows)
            else "complete_with_unresolved_rows"
        ),
        "row_count": len(rows),
        "rho_relative_definition": "proposed_update_rms_over_frozen_initial_parameter_rms",
        "rho_relative_aggregation": "max_parameter(linear_Q90_over_final_32_warmup_steps)",
        "native_selection_coordinates_are_not_numerically_interchangeable": True,
        "official_test_read": False,
        "sources": {
            "v1": {
                **v1_source,
                "candidate_manifest": _artifact(v1_root / "stages/candidates/manifest.json"),
                "candidate_completion": _artifact(v1_root / "stages/candidates/complete.json"),
                "split_provenance": _artifact(v1_split_path),
            },
            "v3": {**v3_source, "split_provenance": _artifact(v3_split_path)},
        },
        "baseline_relative_rho_derivations": baseline_derivations,
        "outputs": {
            "csv": _artifact(csv_path),
            "selected_raw_peak_lr_plot": _artifact(raw_plot),
            "observed_peak_rho_relative_plot": _artifact(relative_plot),
        },
        "curation_script": _artifact(Path(__file__).resolve()),
    }
    provenance_path = destination / PROVENANCE_FILE
    atomic_write_json(provenance_path, provenance, canonical=True)
    return {
        "status": provenance["status"],
        "row_count": len(rows),
        "output_csv": str(csv_path),
        "output_dir": str(destination),
        "provenance": str(provenance_path),
        "plots": [str(raw_plot), str(relative_plot)],
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v1-study", required=True)
    parser.add_argument("--v3-study", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--output-csv", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = curate_active_handoff(
        v1_study=args.v1_study,
        v3_study=args.v3_study,
        output_dir=args.output_dir,
        output_csv=args.output_csv,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
