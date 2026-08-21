#!/usr/bin/env python3
"""Summarize the Conv2/Conv3 one-decade endpoint-read-noise gradient replay."""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import median
import sys
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.reporting import sha256_file, validate_run  # noqa: E402


DEFAULT_STUDY_ROOT = (
    ROOT
    / "results/perfectdiode-conv23-zero-bias-adam-centered-float64-eqprop-bptt-gradient-gate-one-decade-sigma-5em4-seed0-20260817-v1"
)
DEFAULT_CLEAN_RUN = (
    ROOT
    / "results/perfectdiode-conv123-zero-bias-adam-centered-float64-eqprop-bptt-gradient-gate-tk4-6-8-seed0-20260816-v1/production"
)
ARCHITECTURES = ("conv2", "conv3")
SCHEMES = ("baseline", "ours", "legacy")
ROLES = ("reconstructed_initialization", "best_validation")
PARAMETER_ORDER = {
    "conv2": ("ConvWeight_0", "ConvWeight_1", "DenseWeight_0"),
    "conv3": (
        "ConvWeight_0",
        "ConvWeight_1",
        "ConvWeight_2",
        "DenseWeight_0",
    ),
}
PARAMETER_LABEL = {
    "ConvWeight_0": "C0",
    "ConvWeight_1": "C1",
    "ConvWeight_2": "C2",
    "DenseWeight_0": "Dense",
}
SCHEME_COLOR = {
    "baseline": "#1f77b4",
    "ours": "#2ca02c",
    "legacy": "#d62728",
}


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object in {path}.")
    return value


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return [dict(row) for row in csv.DictReader(stream)]


def _bool(value: Any) -> bool:
    normalized = str(value).strip().lower()
    if normalized in {"true", "1"}:
        return True
    if normalized in {"false", "0"}:
        return False
    raise ValueError(f"Expected boolean, got {value!r}.")


def _semantic_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty CSV {path}.")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        row["architecture"],
        row["scheme"],
        row["checkpoint_role"],
        int(row["batch_index"]),
        row["parameter_name"],
    )


def _clean_replay_guard(
    noisy_rows: Sequence[Mapping[str, str]],
    clean_rows: Sequence[Mapping[str, str]],
) -> dict[str, Any]:
    clean = {
        _key(row): row
        for row in clean_rows
        if row["architecture"] in ARCHITECTURES
    }
    if len(clean) != len(noisy_rows):
        raise ValueError("Clean and noisy layer-row coverage differs.")
    cosine_deltas = []
    norm_deltas = []
    for row in noisy_rows:
        reference = clean.get(_key(row))
        if reference is None:
            raise ValueError(f"Missing clean parent row {_key(row)}.")
        cosine_deltas.append(
            abs(
                float(row["clean_eqprop_bptt_cosine"])
                - float(reference["cosine"])
            )
        )
        norm_deltas.append(
            abs(
                float(row["clean_eqprop_bptt_symmetric_norm_delta"])
                - float(reference["symmetric_norm_delta"])
            )
        )
    maximum_cosine_delta = max(cosine_deltas)
    maximum_norm_delta = max(norm_deltas)
    passed = maximum_cosine_delta <= 1.0e-12 and maximum_norm_delta <= 1.0e-12
    if not passed:
        raise RuntimeError("The embedded clean replay differs from its accepted parent.")
    return {
        "row_count": len(noisy_rows),
        "maximum_absolute_cosine_delta": maximum_cosine_delta,
        "maximum_absolute_symmetric_norm_delta_delta": maximum_norm_delta,
        "passed": True,
    }


def _layer_summary(rows: Sequence[Mapping[str, str]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str], list[Mapping[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[
            (
                row["architecture"],
                row["scheme"],
                row["checkpoint_role"],
                row["parameter_name"],
            )
        ].append(row)
    output = []
    for architecture in ARCHITECTURES:
        for scheme in SCHEMES:
            for role in ROLES:
                for parameter in PARAMETER_ORDER[architecture]:
                    values = grouped[(architecture, scheme, role, parameter)]
                    if len(values) != 4:
                        raise ValueError(
                            f"Expected four batches for {architecture}/{scheme}/{role}/{parameter}."
                        )
                    noisy_cosines = [float(row["cosine"]) for row in values]
                    clean_cosines = [
                        float(row["clean_eqprop_bptt_cosine"]) for row in values
                    ]
                    acquisition_cosines = [
                        float(row["noisy_clean_eqprop_cosine"]) for row in values
                    ]
                    output.append(
                        {
                            "architecture": architecture,
                            "scheme": scheme,
                            "checkpoint_role": role,
                            "parameter_name": parameter,
                            "layer": PARAMETER_LABEL[parameter],
                            "injected_beta": float(values[0]["injected_beta"]),
                            "endpoint_read_noise_std": float(
                                values[0]["endpoint_read_noise_std"]
                            ),
                            "batch_count": len(values),
                            "minimum_clean_eqprop_bptt_cosine": min(clean_cosines),
                            "minimum_noisy_clean_eqprop_cosine": min(
                                acquisition_cosines
                            ),
                            "minimum_noisy_eqprop_bptt_cosine": min(noisy_cosines),
                            "median_noisy_eqprop_bptt_cosine": median(noisy_cosines),
                            "maximum_noisy_eqprop_bptt_norm_delta": max(
                                float(row["symmetric_norm_delta"]) for row in values
                            ),
                            "acquisition_passed_batches": sum(
                                _bool(row["read_noise_acquisition_passed"])
                                for row in values
                            ),
                            "task_passed_batches": sum(
                                _bool(row["gradient_fidelity_passed"])
                                for row in values
                            ),
                            "usable_passed_batches": sum(
                                _bool(row["read_noise_usable_gate_passed"])
                                for row in values
                            ),
                        }
                    )
    return output


def _plot(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    lookup = {
        (
            row["architecture"],
            row["scheme"],
            row["checkpoint_role"],
            row["parameter_name"],
        ): row
        for row in rows
    }
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.2), sharey=True)
    for row_index, architecture in enumerate(ARCHITECTURES):
        parameters = PARAMETER_ORDER[architecture]
        x = list(range(len(parameters)))
        for column_index, role in enumerate(ROLES):
            axis = axes[row_index][column_index]
            for scheme in SCHEMES:
                values = [
                    lookup[(architecture, scheme, role, parameter)]
                    for parameter in parameters
                ]
                color = SCHEME_COLOR[scheme]
                clean = [
                    float(value["minimum_clean_eqprop_bptt_cosine"])
                    for value in values
                ]
                noisy = [
                    float(value["minimum_noisy_eqprop_bptt_cosine"])
                    for value in values
                ]
                beta = float(values[0]["injected_beta"])
                axis.plot(
                    x,
                    clean,
                    linestyle="--",
                    linewidth=1.1,
                    color=color,
                    alpha=0.6,
                )
                axis.plot(
                    x,
                    noisy,
                    marker="o",
                    linewidth=2.0,
                    color=color,
                    label=f"{scheme}, B={beta:g}",
                )
            axis.axhline(0.99, color="black", linestyle=":", linewidth=1.0)
            axis.axhline(0.0, color="#777777", linestyle="-", linewidth=0.7)
            axis.set_xticks(x, [PARAMETER_LABEL[value] for value in parameters])
            axis.set_ylim(-0.25, 1.035)
            axis.grid(axis="y", alpha=0.25)
            role_label = "initialization" if role.startswith("reconstructed") else "best validation"
            axis.set_title(f"{architecture.title()} — {role_label}")
            if column_index == 0:
                axis.set_ylabel("minimum cosine across 4 batches")
            if row_index == 1:
                axis.set_xlabel("weight layer")
            axis.legend(fontsize=8, loc="lower right")
    fig.suptitle(
        "One-decade EqProp gradient fidelity: clean (dashed) vs endpoint read noise σ=5e-4 (solid)",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _format(value: Any) -> str:
    return f"{float(value):.6g}"


def _write_report(
    path: Path,
    *,
    case_rows: Sequence[Mapping[str, str]],
    layer_rows: Sequence[Mapping[str, Any]],
    verification: Mapping[str, Any],
) -> None:
    lines = [
        "# Conv2/Conv3 one-decade gradient replay with endpoint read noise",
        "",
        "Status: reviewed read-only ordinary-MNIST diagnostic; not paper-facing accuracy evidence.",
        "",
        "Independent Gaussian noise with sigma `5e-4` is applied to every negative and positive free-layer endpoint voltage after equilibrium and before the unchanged local energy-gradient readout. The input and BPTT reference remain exact.",
        "",
        "| Network | Scheme | Checkpoint | Injected beta | Worst clean cosine | Worst noisy cosine | Max noisy norm delta | Acquisition gate | Task gate |",
        "|---|---|---|---:|---:|---:|---:|---|---|",
    ]
    for row in case_rows:
        lines.append(
            "| {architecture} | {scheme} | {role} | {beta} | {clean} | {noisy} | {norm} | {acquisition} | {task} |".format(
                architecture=row["architecture"],
                scheme=row["scheme"],
                role=row["checkpoint_role"],
                beta=_format(row["injected_beta"]),
                clean=_format(row["minimum_clean_eqprop_bptt_cosine"]),
                noisy=_format(row["minimum_cosine"]),
                norm=_format(row["maximum_symmetric_norm_delta"]),
                acquisition=(
                    "pass" if _bool(row["read_noise_acquisition_passed"]) else "fail"
                ),
                task="pass" if _bool(row["gradient_fidelity_passed"]) else "fail",
            )
        )
    acquisition_passes = sum(
        int(row["acquisition_passed_batches"]) for row in layer_rows
    )
    task_passes = sum(int(row["task_passed_batches"]) for row in layer_rows)
    usable_passes = sum(int(row["usable_passed_batches"]) for row in layer_rows)
    lines.extend(
        [
            "",
            "## Result",
            "",
            "None of the 12 architecture x scheme x checkpoint contexts passes the all-layer acquisition or task gate. At layer-batch resolution, acquisition passes `{}/168`, task passes `{}/168`, and the combined usable gate passes `{}/168`.".format(
                acquisition_passes, task_passes, usable_passes
            ),
            "",
            "All six Dense-layer groups pass all eight initialization/best batch rows. The convolutional gradients carry the failure: only Conv2 baseline C1 retains `2/8` task-passing rows; every other convolutional layer/scheme group has `0/8` task passes.",
            "",
            "The result is a gradient-fidelity failure, not an operational or training-instability result. A single noisy two-phase estimate is strongly corrupted at sigma `5e-4`, especially when divided by the small legacy base beta. Separate long-training studies can remain finite because stochastic errors may average over many minibatches and optimizer steps.",
            "",
            "The embedded clean replay matches the accepted one-decade parent over all 168 rows, with maximum absolute cosine delta `{}` and norm-delta delta `{}`.".format(
                _format(verification["clean_parent_guard"]["maximum_absolute_cosine_delta"]),
                _format(
                    verification["clean_parent_guard"][
                        "maximum_absolute_symmetric_norm_delta_delta"
                    ]
                ),
            ),
            "",
            "## Limits",
            "",
            "One model seed, one noise seed, four validation batches, and an absolute uncalibrated voltage-noise scale. The strict `.99/.10` gate is evaluated per layer and batch; this diagnostic does not estimate multi-step optimizer averaging or paper accuracy.",
            "",
            "Artifacts: `layer_summary.csv`, `summary.json`, and `plots/layerwise_clean_vs_noisy_cosine.png`.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def run(study_root: Path, clean_run: Path) -> dict[str, Any]:
    study_root = study_root.expanduser().resolve()
    clean_run = clean_run.expanduser().resolve()
    production = study_root / "production"
    smoke = study_root / "smoke"
    for run_dir in (smoke, production):
        errors = validate_run(run_dir)
        if errors:
            raise ValueError(f"Invalid bundle {run_dir}: {'; '.join(errors)}")
    noisy_rows = _read_csv(production / "layer_metrics.csv")
    case_rows = _read_csv(production / "case_summary.csv")
    clean_rows = _read_csv(clean_run / "layer_metrics.csv")
    if len(noisy_rows) != 168 or len(case_rows) != 12:
        raise ValueError("Noisy production coverage is incomplete.")
    clean_guard = _clean_replay_guard(noisy_rows, clean_rows)
    layer_rows = _layer_summary(noisy_rows)
    if len(layer_rows) != 42:
        raise ValueError("Layer-summary coverage is incomplete.")
    result = _read_json(production / "result.json")
    guards = _read_json(production / "read_only_guards.json")
    completion = result["completion"]
    if not bool(completion.get("criteria_met")):
        raise RuntimeError("Production completion criteria did not pass.")
    verification = {
        "schema": "conv23-zero-bias-gradient-read-noise-analysis/v1",
        "study_root": str(study_root),
        "production_result_sha256": sha256_file(production / "result.json"),
        "smoke_result_sha256": sha256_file(smoke / "result.json"),
        "clean_parent_run": str(clean_run),
        "clean_parent_result_sha256": sha256_file(clean_run / "result.json"),
        "clean_parent_guard": clean_guard,
        "production_completion": completion,
        "matched_noise_draw_guard": guards[
            "matched_endpoint_read_noise_draw_guard"
        ],
        "layer_row_count": len(noisy_rows),
        "case_row_count": len(case_rows),
        "layer_summary_row_count": len(layer_rows),
        "acquisition_passed_layer_batches": sum(
            _bool(row["read_noise_acquisition_passed"]) for row in noisy_rows
        ),
        "task_passed_layer_batches": sum(
            _bool(row["gradient_fidelity_passed"]) for row in noisy_rows
        ),
        "usable_passed_layer_batches": sum(
            _bool(row["read_noise_usable_gate_passed"]) for row in noisy_rows
        ),
        "semantic_sha256": _semantic_sha256(
            {
                "production_result_sha256": sha256_file(production / "result.json"),
                "clean_parent_result_sha256": sha256_file(clean_run / "result.json"),
                "layer_summary": layer_rows,
            }
        ),
    }
    analysis = study_root / "analysis"
    _write_csv(analysis / "layer_summary.csv", layer_rows)
    _write_csv(analysis / "case_summary.csv", case_rows)
    _write_json(analysis / "summary.json", verification)
    _plot(layer_rows, analysis / "plots/layerwise_clean_vs_noisy_cosine.png")
    _write_report(
        analysis / "report.md",
        case_rows=case_rows,
        layer_rows=layer_rows,
        verification=verification,
    )
    return verification


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-root", type=Path, default=DEFAULT_STUDY_ROOT)
    parser.add_argument("--clean-run", type=Path, default=DEFAULT_CLEAN_RUN)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    print(json.dumps(run(args.study_root, args.clean_run), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
