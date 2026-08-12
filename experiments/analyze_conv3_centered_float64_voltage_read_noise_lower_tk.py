#!/usr/bin/env python3
"""Replay Conv3 centered-float64 voltage read noise at T=K=8 or 12.

The six T=K=64-selected beta values are held fixed.  Each lower-T/K run
recomputes its own free state, BPTT gradient, signed EqProp endpoints, clean
EqProp gradient, residuals, and noisy gradients.  Accepted T=K=64 tensors are
used only as immutable drift references, never as lower-T/K equality targets.
Scientific failures (under-relaxed residuals or a bad clean EqProp gradient)
are retained in a successful diagnostic bundle.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import importlib.util
import json
import math
from pathlib import Path
import socket
import sys
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
PARENT_SCRIPT = ROOT / "experiments/analyze_conv3_centered_float64_voltage_read_noise.py"
DEFAULT_STUDY = ROOT / "configs/conv/perfectdiode_conv3_centered_float64_voltage_read_noise_tk8_tk12_20260812_v1.json"
DEFAULT_PARENT_CONFIG = ROOT / "configs/conv/perfectdiode_conv3_centered_float64_voltage_read_noise_20260811_v1.json"
DEFAULT_OUTPUT = ROOT / "results/perfectdiode-conv3-centered-float64-voltage-read-noise-tk8-tk12-20260812-v1"


def _load_parent() -> Any:
    spec = importlib.util.spec_from_file_location("_conv3_vnoise_parent", PARENT_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {PARENT_SCRIPT}.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


parent = _load_parent()
base = parent.base
extended = parent.extended
audit = parent.audit
torch = parent.torch
np = parent.np
plt = parent.plt
SCHEMA = "perfectdiode-conv3-centered-float64-voltage-read-noise-lower-tk/v1"


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _validate_configs(study: Mapping[str, Any], parent_study: Mapping[str, Any]) -> None:
    if study.get("schema_version") != "perfectdiode-conv3-centered-float64-voltage-read-noise-tk-study/v1":
        raise ValueError("Unexpected lower-T/K study schema.")
    if list(study["replay"]["tested_equal_T_K"]) != [8, 12]:
        raise ValueError("Lower-T/K grid must be exactly [8, 12].")
    if int(study["replay"]["reference_equal_T_K"]) != 64:
        raise ValueError("Reference T/K must remain 64.")
    if study["replay"]["beta_policy"] != "hold_parent_tk64_selected_injected_B_and_base_beta_fixed":
        raise ValueError("Beta policy changed.")
    expected_replay = {
        "architecture": "conv3",
        "batch_index": 0,
        "batch_size": 16,
        "runtime_dtype": "float64",
        "eqprop_variant": "centered",
        "nudging_mode": "current",
        "same_tk_bptt_reference": True,
        "compare_clean_eqprop_and_bptt_to_tk64": True,
        "input_read_noise": False,
        "biases_active_in_dynamics": True,
        "bias_gradients_excluded": True,
        "optimizer_steps_applied": False,
        "official_test_read": False,
    }
    if any(study["replay"].get(key) != value for key, value in expected_replay.items()):
        raise ValueError("Lower-T/K replay contract changed.")
    expected_cases = {
        (scheme, role)
        for scheme in ("baseline", "ours", "legacy")
        for role in ("reconstructed_initialization", "best_validation")
    }
    cases = list(parent_study.get("cases", ()))
    if len(cases) != 6 or {
        (str(row["scheme"]), str(row["checkpoint_role"])) for row in cases
    } != expected_cases:
        raise ValueError("Parent study must retain the six scheme/checkpoint cases.")


def _with_tk(rows: Sequence[Mapping[str, Any]], tk: int) -> list[dict[str, Any]]:
    return [{**dict(row), "schema": SCHEMA, "T": tk, "K": tk} for row in rows]


def _drift_metrics(candidate: torch.Tensor, reference: torch.Tensor) -> dict[str, Any]:
    metrics = parent._gradient_metrics(
        candidate,
        reference,
        cosine_minimum=0.99,
        symmetric_norm_delta_maximum=0.1,
    )
    left = candidate.detach().cpu().to(torch.float64)
    right = reference.detach().cpu().to(torch.float64)
    metrics["candidate_zero_fraction"] = float((left.abs() <= 1.0e-12).to(torch.float64).mean())
    metrics["reference_zero_fraction"] = float((right.abs() <= 1.0e-12).to(torch.float64).mean())
    metrics["absolute_zero_fraction_delta"] = abs(
        metrics["candidate_zero_fraction"] - metrics["reference_zero_fraction"]
    )
    return metrics


def _same_tk_gradient_gate(
    parent_gate: Mapping[str, Any], tk: int
) -> dict[str, Any]:
    gate = dict(parent_gate)
    gate["task_reference"] = f"bptt_same_post_T_state_K{int(tk)}"
    return gate


def _aggregate_trials_allowing_undefined_cosine(
    rows: Sequence[Mapping[str, Any]],
    *,
    cosine_minimum: float,
    norm_delta_maximum: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Retain zero-norm cosine outcomes as explicit gate failures.

    The parent aggregator predates the scientific-failure-retention contract
    and casts every cosine to float.  A temporary -1 sentinel lets it perform
    gate/quantile bookkeeping; the published aggregate fields are restored to
    null wherever at least one contributing cosine was undefined.
    """

    counts: dict[tuple[Any, ...], dict[str, int]] = defaultdict(
        lambda: {"acquisition": 0, "task": 0}
    )
    compatible: list[dict[str, Any]] = []
    for source in rows:
        row = dict(source)
        key = (
            row["scheme"],
            row["checkpoint_role"],
            float(row["sigma_voltage"]),
            row["parameter_name"],
        )
        for label in ("acquisition", "task"):
            field = f"{label}_cosine"
            if row[field] is None:
                counts[key][label] += 1
                row[field] = -1.0
        compatible.append(row)
    layer_rows, configuration_rows = parent._aggregate_trials(
        compatible,
        cosine_minimum=cosine_minimum,
        norm_delta_maximum=norm_delta_maximum,
    )
    for row in layer_rows:
        key = (
            row["scheme"],
            row["checkpoint_role"],
            float(row["sigma_voltage"]),
            row["parameter_name"],
        )
        for label in ("acquisition", "task"):
            count = counts[key][label]
            row[f"undefined_{label}_cosine_trial_count"] = count
            if count:
                for suffix in ("p05", "median", "minimum"):
                    row[f"{label}_cosine_{suffix}"] = None
    layer_by_config: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in layer_rows:
        layer_by_config[
            (row["scheme"], row["checkpoint_role"], float(row["sigma_voltage"]))
        ].append(row)
    for row in configuration_rows:
        key = (row["scheme"], row["checkpoint_role"], float(row["sigma_voltage"]))
        values = layer_by_config[key]
        for label in ("acquisition", "task"):
            count = sum(
                int(value[f"undefined_{label}_cosine_trial_count"])
                for value in values
            )
            row[f"undefined_{label}_cosine_trial_count"] = count
            if count:
                row[f"minimum_layer_{label}_cosine_p05"] = None
    return layer_rows, configuration_rows


def _plot_rows_with_undefined_cosines(
    rows: Sequence[Mapping[str, Any]], fields: Sequence[str]
) -> list[dict[str, Any]]:
    return [
        {
            **dict(row),
            **{
                field: (-1.0 if row.get(field) is None else row[field])
                for field in fields
            },
        }
        for row in rows
    ]


def _reference_tensors(
    parent_study: Mapping[str, Any],
    case: Mapping[str, Any],
) -> tuple[dict[str, dict[str, torch.Tensor]], dict[str, Any]]:
    accepted_root = parent._resolve_repository_path(parent_study["source"]["accepted_sweep_root"])
    run_dir = accepted_root / str(case["accepted_reference_run"])
    result_path = run_dir / "result.json"
    archive = run_dir / "artifacts/eqprop_gradients" / f"conv3__{case['scheme']}__{case['checkpoint_role']}.npz"
    errors = base.validate_run(run_dir)
    if errors:
        raise ValueError("Invalid accepted T64 reference bundle: " + "; ".join(errors))
    if base.sha256_file(result_path) != case["accepted_reference_result_sha256"]:
        raise ValueError("Accepted T64 reference result changed.")
    if base.sha256_file(archive) != case["accepted_reference_archive_sha256"]:
        raise ValueError("Accepted T64 gradient archive changed.")
    tensors: dict[str, dict[str, torch.Tensor]] = {"eqprop": {}, "bptt": {}}
    with np.load(archive, allow_pickle=False) as loaded:
        metadata = json.loads(str(loaded["metadata_json"].item()))
        expected_metadata = {
            "T": 64,
            "K": 64,
            "scheme": case["scheme"],
            "checkpoint_role": case["checkpoint_role"],
            "eqprop_variant": "centered",
        }
        if any(metadata.get(key) != value for key, value in expected_metadata.items()):
            raise ValueError("Accepted reference archive metadata is misassociated.")
        if not math.isclose(
            float(metadata["actual_beta"]),
            float(case["actual_beta"]),
            rel_tol=1.0e-15,
            abs_tol=0.0,
        ):
            raise ValueError("Accepted reference beta differs from the fixed case beta.")
        for name in parent.LAYER_LABELS:
            for quantity in ("eqprop", "bptt"):
                key = f"{quantity}_float64__{name}"
                if key not in loaded:
                    raise ValueError(f"Accepted reference archive lacks {key}.")
                array = np.asarray(loaded[key])
                if array.dtype != np.dtype(np.float64) or not bool(np.isfinite(array).all()):
                    raise ValueError(f"Accepted reference tensor {key} is not finite float64.")
                tensors[quantity][name] = torch.from_numpy(array.copy())
    return tensors, {
        "reference_result": str(result_path),
        "reference_result_sha256": base.sha256_file(result_path),
        "reference_archive": str(archive),
        "reference_archive_sha256": base.sha256_file(archive),
    }


def _thresholds_allowing_clean_failure(
    configuration_rows: Sequence[Mapping[str, Any]],
    clean_state_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows = parent._threshold_rows(configuration_rows, clean_state_rows)
    by_key = {
        (str(row["scheme"]), str(row["checkpoint_role"])): row
        for row in configuration_rows
        if float(row["sigma_voltage"]) == 0.0
    }
    for row in rows:
        zero = by_key[(str(row["scheme"]), str(row["checkpoint_role"]))]
        clean_pass = bool(zero["all_layers_task_gate_passed"])
        row["clean_eqprop_vs_same_tk_bptt_gate_passed"] = clean_pass
        if not clean_pass:
            row["usable_maximum_sustained_sigma"] = None
            row["usable_first_failing_sigma"] = 0.0
            row["usable_sigma_over_minimum_state_delta_rms"] = None
            row["usable_phase_difference_noise_rms_over_minimum_state_delta_rms"] = None
            row["usable_threshold_status"] = "clean_task_gate_failed_at_sigma_zero"
        else:
            row["usable_threshold_status"] = "resolved_zero_anchored"
    return rows


def _annotate_equilibrium_validity(
    thresholds: Sequence[Mapping[str, Any]],
    case_guards: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    guards = {
        (str(row["scheme"]), str(row["checkpoint_role"])): row
        for row in case_guards
    }
    result: list[dict[str, Any]] = []
    for source in thresholds:
        row = dict(source)
        guard = guards[(str(row["scheme"]), str(row["checkpoint_role"]))]
        residual_valid = bool(guard["all_residual_gates_passed"])
        clean_valid = bool(row["clean_eqprop_vs_same_tk_bptt_gate_passed"])
        row["all_residual_gates_passed"] = residual_valid
        row["under_relaxed_diagnostic"] = not residual_valid
        row["equilibrium_valid_acquisition_only_maximum_sustained_sigma"] = (
            row["acquisition_only_maximum_sustained_sigma"]
            if residual_valid
            else None
        )
        row["equilibrium_valid_usable_maximum_sustained_sigma"] = (
            row["usable_maximum_sustained_sigma"]
            if residual_valid and clean_valid
            else None
        )
        if not residual_valid:
            row["equilibrium_valid_threshold_status"] = (
                "under_relaxed_residual_gate_failed"
            )
        elif not clean_valid:
            row["equilibrium_valid_threshold_status"] = (
                "clean_task_gate_failed_at_sigma_zero"
            )
        else:
            row["equilibrium_valid_threshold_status"] = (
                "equilibrium_valid_resolved_zero_anchored"
            )
        result.append(row)
    return result


def _write_report(
    path: Path,
    *,
    tk: int,
    thresholds: Sequence[Mapping[str, Any]],
    case_summary: Sequence[Mapping[str, Any]],
    smoke: bool,
) -> None:
    lines = [
        f"# Conv3 centered float64 voltage-read noise at T=K={tk}",
        "",
        "Exploratory ordinary-MNIST read-only diagnostic; fixed T64-selected beta values and one fixed 16-example batch.",
        "",
        "Residual failures and sigma-zero gradient failures are scientific outcomes, not operational bundle failures. A usable threshold is null when clean EqProp does not pass against same-T/K BPTT.",
        "",
        "| scheme | checkpoint | residual-valid | clean EqProp valid | acquisition sigma | gradient-only usable sigma | equilibrium-valid usable sigma | first usable fail | min phase RMS |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    summaries = {(row["scheme"], row["checkpoint_role"]): row for row in case_summary}
    for row in thresholds:
        summary = summaries[(row["scheme"], row["checkpoint_role"])]
        lines.append(
            "| {scheme} | {role} | {residual} | {clean} | {acq} | {usable} | {equilibrium_usable} | {fail} | {phase:.6g} |".format(
                scheme=row["scheme"], role=row["checkpoint_role"],
                residual=summary["all_residual_gates_passed"],
                clean=row["clean_eqprop_vs_same_tk_bptt_gate_passed"],
                acq=row["acquisition_only_maximum_sustained_sigma"],
                usable=row["usable_maximum_sustained_sigma"],
                equilibrium_usable=row[
                    "equilibrium_valid_usable_maximum_sustained_sigma"
                ],
                fail=row["usable_first_failing_sigma"],
                phase=float(row["minimum_clean_positive_minus_negative_state_delta_rms"]),
            )
        )
    if smoke:
        lines.extend(["", "Smoke scope only; no scientific threshold conclusion."])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    tk = int(args.tk)
    if tk not in (8, 12):
        raise ValueError("--tk must be 8 or 12.")
    study_path = args.study_config.expanduser().resolve()
    parent_path = args.parent_config.expanduser().resolve()
    study = base._read_json(study_path)
    parent_study = base._read_json(parent_path)
    _validate_configs(study, parent_study)
    if parent_path != parent._resolve_repository_path(study["parent_study"]["config"]):
        raise ValueError("--parent-config differs from the frozen lower-T/K study.")
    if base.sha256_file(parent_path) != study["parent_study"]["config_sha256"]:
        raise ValueError("Parent study config bytes changed.")
    parent._validate_study(base._read_json(parent.DEFAULT_SOURCE_CONFIG), parent_study)

    source_config = base._read_json(parent.DEFAULT_SOURCE_CONFIG)
    source_run = parent.DEFAULT_SOURCE_RUN.resolve()
    parent_run = parent._resolve_repository_path(study["parent_study"]["production_run"])
    if base.sha256_file(parent_run / "result.json") != study["parent_study"]["production_result_sha256"]:
        raise ValueError("Accepted T64 production result changed.")
    errors = base.validate_run(parent_run)
    if errors:
        raise ValueError("Invalid accepted T64 production bundle: " + "; ".join(errors))

    runtime_source = base._validate_runtime_source(source_config)
    inventory, source_hashes_before = extended._source_inventory(source_config)
    inventory_by_scheme = {str(row["scheme"]): row for row in inventory if row["architecture"] == "conv3"}
    batch, materialized, source_cohort = parent._materialize_batch(
        source_run=source_run,
        dataset_config=inventory[0]["source_config"],
        dataset_root=args.dataset_root.expanduser().resolve(),
        study=parent_study,
    )
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable.")
    cases = list(parent_study["cases"][:1] if args.smoke else parent_study["cases"])
    sigma_values = [0.0, 1.0e-7] if args.smoke else [float(x) for x in parent_study["noise"]["sigma_values"]]
    trial_count = 2 if args.smoke else int(parent_study["noise"]["trial_count"])
    gate = _same_tk_gradient_gate(parent_study["gradient_fidelity_gate"], tk)
    cosine_minimum = float(gate["cosine_minimum"])
    norm_delta_maximum = float(gate["symmetric_norm_delta_maximum"])
    residual_contract = source_config["equilibrium_residual_contract"]
    residual_threshold = float(residual_contract["per_layer_sample_max_p90_threshold"])
    if not math.isclose(
        float(residual_contract["hard_failure_maximum_threshold"]),
        1.0e-1,
        rel_tol=0.0,
        abs_tol=0.0,
    ):
        raise ValueError("Audit hard-residual threshold differs from the source contract.")
    run_id = args.run_id or (f"smoke-tk{tk}" if args.smoke else f"tk{tk}")
    run_dir = args.output_root.expanduser().resolve() / run_id
    sources = {
        "lower_tk_analyzer": Path(__file__).resolve(),
        "parent_analyzer": PARENT_SCRIPT,
        "float64_audit": Path(audit.__file__).resolve(),
        "extended_eqprop_analysis": Path(extended.__file__).resolve(),
        "base_eqprop_analysis": Path(base.__file__).resolve(),
    }
    manifest = {
        "study_id": study["study_id"], "run_id": run_id,
        "arm_id": f"conv3_centered_float64_endpoint_voltage_read_noise_tk{tk}",
        "evidence_class": study["evidence_class"], "dataset": source_config["dataset"]["name"],
        "smoke": bool(args.smoke), "command": [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]],
        "configuration": {"study_config_path": str(study_path), "study_config_sha256": base.sha256_file(study_path), "resolved": {"T": tk, "K": tk, "cases": cases, "sigma_values": sigma_values, "trial_count_per_nonzero_sigma": trial_count, "beta_policy": study["replay"]["beta_policy"], "gradient_fidelity_gate": gate}},
        "runtime": {**base.runtime_context(target=args.target), "device": str(device), "hostname": socket.gethostname(), "torch_version": torch.__version__, "cuda_version": torch.version.cuda, "runtime_source": runtime_source},
        "source": {"parent_t64_run": str(parent_run), "parent_t64_result_sha256": base.sha256_file(parent_run / "result.json"), "source_config": str(parent.DEFAULT_SOURCE_CONFIG), "source_config_sha256": base.sha256_file(parent.DEFAULT_SOURCE_CONFIG)},
        "replay": {"same_tk_bptt_reference": True, "t64_reference_used_for_drift_only": True, "endpoint_noise_location": "after_equilibrium_before_local_squared_drop_gradient", "bias_gradients_excluded": True, "optimizer_steps_applied": False, "official_test_read": False},
        "analyzer": {"path": str(Path(__file__).resolve()), "sha256": base.sha256_file(Path(__file__).resolve()), "repository_head": base._git_output(ROOT, "rev-parse", "HEAD"), "source_files": {label: {"path": str(path), "sha256": base.sha256_file(path)} for label, path in sources.items()}},
    }
    base.start_run(run_dir, manifest)
    try:
        snapshots = parent._write_source_snapshots(run_dir, sources)
        base._write_json(run_dir / "study_config.resolved.json", study)
        base._write_json(run_dir / "parent_study_config.resolved.json", parent_study)
        base._write_json(run_dir / "source_cohort.json", source_cohort)
        base._write_json(run_dir / "materialized_cohort.json", materialized)
        gradient_rows: list[dict[str, Any]] = []
        state_noise_rows: list[dict[str, Any]] = []
        clean_state_rows: list[dict[str, Any]] = []
        residual_rows: list[dict[str, Any]] = []
        drift_rows: list[dict[str, Any]] = []
        reference_files: list[dict[str, Any]] = []
        case_guards: list[dict[str, Any]] = []
        for case_index, case in enumerate(cases):
            base.update_status_progress(run_dir, {"stage": "case_replay", "case_index": case_index, "case_count": len(cases), "T": tk, "K": tk, "scheme": case["scheme"], "checkpoint_role": case["checkpoint_role"]})
            output = parent._capture_case(
                inventory_case=inventory_by_scheme[str(case["scheme"])], case=case,
                batch=batch, device=device, residual_threshold=residual_threshold, T=tk, K=tk,
            )
            refs, ref_files = _reference_tensors(parent_study, case)
            reference_files.append({"T": 64, "K": 64, **dict(case), **ref_files})
            for quantity in ("eqprop", "bptt"):
                candidate_set = output["gradients"] if quantity == "eqprop" else output["bptt_gradients"]
                for name, candidate in candidate_set.items():
                    drift_rows.append({"schema": SCHEMA, "T": tk, "K": tk, "scheme": case["scheme"], "checkpoint_role": case["checkpoint_role"], "parameter_name": name, "layer": parent.LAYER_LABELS[name], "quantity": quantity, "reference_T": 64, "reference_K": 64, **_drift_metrics(candidate, refs[quantity][name])})
            clean_state_rows.extend(_with_tk(parent._clean_state_rows(case, output), tk))
            residual_rows.extend([{**dict(row), "schema": SCHEMA} for row in output["residual_rows"]])
            parameter_hash_before = base._parameter_state_sha256(output["captured_runtime"]["parameters"])
            rows, noise_rows, noise_guards = parent._evaluate_case_noise(
                case=case, output=output, sigma_values=sigma_values, trial_count=trial_count,
                base_seed=int(parent_study["noise"]["base_seed"]), device=device,
                cosine_minimum=cosine_minimum, norm_delta_maximum=norm_delta_maximum,
            )
            gradient_rows.extend(_with_tk(rows, tk))
            state_noise_rows.extend(_with_tk(noise_rows, tk))
            parameter_hash_after = base._parameter_state_sha256(output["captured_runtime"]["parameters"])
            clean_rows = [row for row in rows if float(row["sigma_voltage"]) == 0.0]
            case_guards.append({"T": tk, "K": tk, "scheme": case["scheme"], "checkpoint_role": case["checkpoint_role"], "clean_all_layer_eqprop_vs_same_tk_bptt_gate_passed": all(bool(row["task_gate_passed"]) for row in clean_rows), "all_residual_gates_passed": all(bool(row["gate_passed"]) for row in output["residual_rows"]), "residual_hard_failure": any(bool(row["hard_failure"]) for row in output["residual_rows"]), "parameters_unchanged": parameter_hash_before == parameter_hash_after, **noise_guards})
            del output
            if device.type == "cuda":
                torch.cuda.empty_cache()

        layer_rows, configuration_rows = _aggregate_trials_allowing_undefined_cosine(
            gradient_rows,
            cosine_minimum=cosine_minimum,
            norm_delta_maximum=norm_delta_maximum,
        )
        layer_rows = _with_tk(layer_rows, tk)
        configuration_rows = _with_tk(configuration_rows, tk)
        thresholds = _with_tk(
            _annotate_equilibrium_validity(
                _thresholds_allowing_clean_failure(
                    configuration_rows, clean_state_rows
                ),
                case_guards,
            ),
            tk,
        )
        source_hashes_after = extended._verify_source_hashes(inventory, source_hashes_before)
        source_hashes_before_json = {"/".join(key): value for key, value in source_hashes_before.items()}
        operational_guards = {
            "case_guards": case_guards,
            "source_hashes_before": source_hashes_before_json,
            "source_hashes_after": source_hashes_after,
            "source_checkpoint_bytes_unchanged": source_hashes_after == source_hashes_before_json,
            "analysis_sources_archived_exactly": all(bool(row["matches"]) for row in snapshots),
            "optimizer_steps_applied": False,
            "official_test_read": False,
        }
        for filename, rows in (("gradient_trials.csv", gradient_rows), ("state_noise_trials.csv", state_noise_rows), ("clean_state_displacement.csv", clean_state_rows), ("layer_quantiles.csv", layer_rows), ("configuration_summary.csv", configuration_rows), ("noise_thresholds.csv", thresholds), ("phase_residuals.csv", residual_rows), ("tk64_gradient_drift.csv", drift_rows)):
            base._write_csv(run_dir / filename, rows)
        base._write_json(run_dir / "accepted_tk64_reference_files.json", reference_files)
        base._write_json(run_dir / "read_only_guards.json", operational_guards)
        parent._plot_summary(
            run_dir / "noise_fidelity_vs_sigma.png",
            _plot_rows_with_undefined_cosines(
                configuration_rows,
                ("minimum_layer_task_cosine_p05",),
            ),
            cosine_minimum=cosine_minimum,
            norm_delta_maximum=norm_delta_maximum,
        )
        parent._plot_layerwise(
            run_dir / "layerwise_cosine_vs_sigma.png",
            _plot_rows_with_undefined_cosines(
                layer_rows,
                ("task_cosine_p05",),
            ),
            cosine_minimum=cosine_minimum,
        )
        _write_report(run_dir / "report.md", tk=tk, thresholds=thresholds, case_summary=case_guards, smoke=bool(args.smoke))
        for row in thresholds:
            base.append_metric(run_dir / "metrics.jsonl", {"stage": "lower_tk_voltage_read_noise_threshold", "split": "ordinary_mnist_validation", **row, "official_test_read": False})
        expected_gradient_rows = len(cases) * (1 + (len(sigma_values) - 1) * trial_count) * 4
        expected_state_rows = len(cases) * (1 + (len(sigma_values) - 1) * trial_count) * 2 * 4
        operational_completion = {
            "declared_case_coverage_complete": len(case_guards) == len(cases),
            "gradient_trial_coverage_complete": len(gradient_rows) == expected_gradient_rows,
            "state_noise_coverage_complete": len(state_noise_rows) == expected_state_rows,
            "all_noise_zero_identity_guards_passed": all(bool(row["noise_zero_clean_identity_passed"]) for row in case_guards),
            "all_inputs_remained_exact": all(bool(row["input_remained_exact"]) for row in case_guards),
            "all_parameters_unchanged": all(bool(row["parameters_unchanged"]) for row in case_guards),
            "source_checkpoint_bytes_unchanged": operational_guards["source_checkpoint_bytes_unchanged"],
            "analysis_sources_archived_exactly": operational_guards["analysis_sources_archived_exactly"],
            "tk64_reference_hashes_validated": len(reference_files) == len(cases),
            "float64_endpoint_and_subtraction_arithmetic": True,
            "bias_gradients_excluded": True,
            "optimizer_steps_applied": False,
            "official_test_read": False,
        }
        positive_required = [key for key in operational_completion if key not in {"optimizer_steps_applied", "official_test_read"}]
        operational_completion["criteria_met"] = all(bool(operational_completion[key]) for key in positive_required)
        if not operational_completion["criteria_met"]:
            raise RuntimeError("Operational completion guards failed: " + ", ".join(key for key in positive_required if not bool(operational_completion[key])))
        diagnostic_usable = [float(row["usable_maximum_sustained_sigma"]) for row in thresholds if row["usable_maximum_sustained_sigma"] is not None]
        equilibrium_usable = [float(row["equilibrium_valid_usable_maximum_sustained_sigma"]) for row in thresholds if row["equilibrium_valid_usable_maximum_sustained_sigma"] is not None]
        terminal = {"T": tk, "K": tk, "case_count": len(cases), "gradient_trial_row_count": len(gradient_rows), "state_noise_trial_row_count": len(state_noise_rows), "clean_task_gate_pass_count": sum(bool(row["clean_all_layer_eqprop_vs_same_tk_bptt_gate_passed"]) for row in case_guards), "residual_valid_case_count": sum(bool(row["all_residual_gates_passed"]) for row in case_guards), "minimum_diagnostic_gradient_usable_sigma": min(diagnostic_usable) if diagnostic_usable else None, "maximum_diagnostic_gradient_usable_sigma": max(diagnostic_usable) if diagnostic_usable else None, "minimum_equilibrium_valid_usable_sigma": min(equilibrium_usable) if equilibrium_usable else None, "maximum_equilibrium_valid_usable_sigma": max(equilibrium_usable) if equilibrium_usable else None, "optimizer_steps_applied": False, "official_test_read": False}
        running_errors = base.validate_run(run_dir)
        if running_errors:
            raise RuntimeError("Running bundle validation failed: " + "; ".join(running_errors))
        result = base.complete_run(run_dir, terminal_metrics=terminal, completion=operational_completion)
        final_errors = base.validate_run(run_dir)
        if final_errors:
            raise RuntimeError("Completed bundle validation failed: " + "; ".join(final_errors))
        return result
    except BaseException as error:
        if (run_dir / "status.json").is_file() and base._read_json(run_dir / "status.json").get("state") == "running":
            if (run_dir / "result.json").is_file():
                (run_dir / "result.json").unlink()
            base.fail_run(run_dir, error=error)
        raise


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tk", type=int, choices=(8, 12), required=True)
    parser.add_argument("--study-config", type=Path, default=DEFAULT_STUDY)
    parser.add_argument("--parent-config", type=Path, default=DEFAULT_PARENT_CONFIG)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--run-id")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--dataset-root", type=Path, default=Path("/home/filip/datasets/mnist"))
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--target", default="local:RTX3090")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = run(args)
    print(json.dumps({"state": "complete", "terminal_metrics": result["terminal_metrics"]}, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
