from __future__ import annotations

import csv
import json
from pathlib import Path
import shutil

import pytest

from experiments import analyze_conv3_eqprop_complete_beta_sweep as analysis
from experiments import reporting


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    assert rows
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _source_fixture(tmp_path: Path) -> tuple[Path, Path, str, str]:
    source_run = tmp_path / "source-run"
    source_run.mkdir()
    payload_sha = "a" * 64
    indices_sha = "b" * 64
    source_config = {
        "study_id": "source-current-beta-study",
        "gradient_contract": {"nudging_mode": "current"},
        "equilibrium_residual_contract": {
            "comparison": "strictly_less_than",
            "per_layer_sample_max_p90_threshold": 0.01,
            "uniform_endpoint_maximum_threshold": 0.01,
            "hard_failure_maximum_threshold": 0.1,
        },
    }
    cohort = {
        "batch_size": 16,
        "batches": [
            {
                "batch_index": 0,
                "payload_sha256": payload_sha,
                "source_indices_sha256": indices_sha,
            }
        ],
        "official_test_read": False,
    }
    _write_json(source_run / "config.resolved.json", source_config)
    _write_json(source_run / "cohort.json", cohort)
    _write_json(source_run / "manifest.json", {"source": True})
    _write_json(source_run / "result.json", {"complete": True})
    (source_run / "context_beta_selection.csv").write_text(
        "architecture,scheme\nconv3,baseline\n", encoding="utf-8"
    )
    (source_run / "parameter_summary.csv").write_text(
        "architecture,parameter\nconv3,ConvWeight_0\n", encoding="utf-8"
    )
    return source_run, source_run / "config.resolved.json", payload_sha, indices_sha


def _plan_fixture(
    tmp_path: Path,
    *,
    source_run: Path,
    source_config_path: Path,
    payload_sha: str,
    indices_sha: str,
) -> tuple[Path, Path, Path]:
    study_root = tmp_path / "study"
    study_root.mkdir()
    launcher = tmp_path / "run_complete_sweep.sh"
    launcher.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    plan = {
        "schema_version": "perfectdiode-conv3-current-eqprop-complete-beta-sweep/v1",
        "study_id": "planned-complete-beta-study",
        "evidence_tier": "exploratory_noncanonical",
        "evidence_class": "ordinary_mnist_learning_algorithm_state_diagnostic",
        "source": {
            "config": str(source_config_path),
            "run": str(source_run),
            "runtime_source_head": "runtime-head",
            "schemes": list(analysis.SCHEMES),
            "checkpoint_roles": list(analysis.CHECKPOINT_ROLES),
        },
        "replay": {
            "T": 64,
            "K": 64,
            "batch_indices": [0],
            "batch_size": 16,
            "batch_payload_sha256": payload_sha,
            "batch_source_indices_sha256": indices_sha,
            "precisions": list(analysis.PRECISIONS),
            "eqprop_variants": list(analysis.VARIANTS),
        },
        "beta_contract": {
            "injected_betas": list(analysis.EXPECTED_BETAS),
            "base_beta_divisor": {
                "baseline": 1.0,
                "ours": 64.0,
                "legacy": 4096.0,
            },
        },
        "completion": {
            "expected_run_bundles": 22,
            "expected_scheme_checkpoint_beta_variant_contexts": 132,
            "expected_true_dtype_contexts": 264,
        },
        "execution": {"output_root": str(study_root)},
    }
    config_path = tmp_path / "plan.json"
    _write_json(config_path, plan)
    return config_path, launcher, study_root


def _source_file_hashes(source_run: Path) -> dict[str, str]:
    names = (
        "result.json",
        "manifest.json",
        "config.resolved.json",
        "cohort.json",
        "context_beta_selection.csv",
        "parameter_summary.csv",
    )
    return {name: reporting.sha256_file(source_run / name) for name in names}


def _selected_cases(
    *, variant: str, injected_beta: float, divisors: dict[str, float]
) -> list[dict[str, object]]:
    rows = []
    for scheme in analysis.SCHEMES:
        for role in analysis.CHECKPOINT_ROLES:
            actual = injected_beta / divisors[scheme]
            rows.append(
                {
                    "architecture": "conv3",
                    "scheme": scheme,
                    "checkpoint_role": role,
                    "T": 64,
                    "K": 64,
                    "actual_beta": actual,
                    "requested_actual_beta": actual,
                    "beta_explicit_cli": True,
                    "resolved_beta_row": {
                        "beta": actual,
                        "beta_effective": injected_beta,
                    },
                }
            )
    return rows


def _scaling_rows(
    *, variant: str, injected_beta: float, divisors: dict[str, float]
) -> list[dict[str, object]]:
    rows = []
    for scheme in analysis.SCHEMES:
        for role in analysis.CHECKPOINT_ROLES:
            actual = injected_beta / divisors[scheme]
            rows.append(
                {
                    "architecture": "conv3",
                    "scheme": scheme,
                    "checkpoint_role": role,
                    "eqprop_variant": variant,
                    "T": 64,
                    "K": 64,
                    "actual_beta": actual,
                    "requested_actual_beta": actual,
                    "effective_beta": injected_beta,
                    "amplification_factor": divisors[scheme],
                    "amplification_depth_L": 3,
                    "amplification_exponent_convention": "output_bias_current_row",
                    "scored_interaction_count": 4,
                    "beta_capped": False,
                }
            )
    return rows


def _parameter_rows(
    *,
    variant: str,
    beta_index: int,
    injected_beta: float,
    divisors: dict[str, float],
    payload_sha: str,
) -> list[dict[str, object]]:
    rows = []
    for scheme in analysis.SCHEMES:
        for role in analysis.CHECKPOINT_ROLES:
            for parameter in analysis.PARAMETERS:
                undefined_float32 = (
                    variant == "positive_one_sided"
                    and beta_index == 0
                    and scheme == "baseline"
                    and role == "reconstructed_initialization"
                    and parameter == "ConvWeight_0"
                )
                row: dict[str, object] = {
                    "schema": analysis.BUNDLE_PARAMETER_SCHEMA,
                    "architecture": "conv3",
                    "scheme": scheme,
                    "checkpoint_role": role,
                    "eqprop_variant": variant,
                    "T": 64,
                    "K": 64,
                    "batch_index": 0,
                    "batch_payload_sha256": payload_sha,
                    "parameter_name": parameter,
                    "parameter_type": (
                        "DenseWeight" if parameter == "DenseWeight_0" else "ConvWeight"
                    ),
                    "bias_excluded": True,
                    "actual_beta": injected_beta / divisors[scheme],
                    "effective_beta": injected_beta,
                    "amplification_factor": divisors[scheme],
                    "residual_limited": False,
                    "outcome": "ok",
                }
                for precision in analysis.PRECISIONS:
                    is_undefined = precision == "float32" and undefined_float32
                    row[f"{precision}_eqprop_gradient_dtype"] = f"torch.{precision}"
                    row[f"{precision}_bptt_gradient_dtype"] = f"torch.{precision}"
                    row[f"{precision}_eqprop_l2"] = 0.0 if is_undefined else 1.0
                    row[f"{precision}_bptt_l2"] = 1.0
                    row[f"{precision}_exact_zero_fraction"] = 1.0 if is_undefined else 0.0
                    row[f"{precision}_bptt_exact_zero_fraction"] = 0.0
                    row[f"{precision}_eqprop_vs_bptt_cosine"] = "" if is_undefined else 0.9
                    row[f"{precision}_eqprop_over_bptt_norm_ratio"] = (
                        "" if is_undefined else 1.0
                    )
                    row[f"{precision}_eqprop_vs_bptt_symmetric_norm_delta"] = (
                        "" if is_undefined else 0.0
                    )
                rows.append(row)
    return rows


def _displacement_rows(
    *,
    variant: str,
    beta_index: int,
    injected_beta: float,
    divisors: dict[str, float],
    payload_sha: str,
    indices_sha: str,
) -> list[dict[str, object]]:
    phase_references = (
        (("positive", "post_T_free"), ("positive", "matched_zero_K"))
        if variant == "positive_one_sided"
        else (
            ("negative", "post_T_free"),
            ("negative", "matched_zero_K"),
            ("positive", "post_T_free"),
            ("positive", "matched_zero_K"),
            ("positive_minus_negative", "negative_phase"),
        )
    )
    rows = []
    for scheme_index, scheme in enumerate(analysis.SCHEMES):
        for role_index, role in enumerate(analysis.CHECKPOINT_ROLES):
            for precision_index, precision in enumerate(analysis.PRECISIONS):
                for phase, reference in phase_references:
                    for state_index, state in enumerate(analysis.ALL_STATE_NAMES):
                        base_value = (
                            (beta_index + 1)
                            * (scheme_index + 1)
                            * (role_index + 1)
                            * (precision_index + 1)
                            * (state_index + 1)
                            * 1.0e-8
                        )
                        value = base_value * (1.2 if phase == "negative" else 1.0)
                        actual = injected_beta / divisors[scheme]
                        rows.append(
                            {
                                "architecture": "conv3",
                                "scheme": scheme,
                                "checkpoint_role": role,
                                "precision": precision,
                                "eqprop_variant": variant,
                                "T": 64,
                                "K": 64,
                                "batch_index": 0,
                                "batch_payload_sha256": payload_sha,
                                "batch_source_indices_sha256": indices_sha,
                                "phase": phase,
                                "actual_beta": -actual if phase == "negative" else actual,
                                "reference_kind": reference,
                                "outcome": "ok",
                                "state_layer_name": state,
                                "relative_displacement": value,
                                "delta_rms": value * 10.0,
                            }
                        )
    return rows


def _residual_rows(
    *,
    variant: str,
    beta_index: int,
    injected_beta: float,
    divisors: dict[str, float],
    payload_sha: str,
    indices_sha: str,
) -> list[dict[str, object]]:
    phases = (
        ("post_T_free", "zero", "positive")
        if variant == "positive_one_sided"
        else ("post_T_free", "zero", "negative", "positive")
    )
    rows: list[dict[str, object]] = []
    for scheme in analysis.SCHEMES:
        for role in analysis.CHECKPOINT_ROLES:
            for precision in analysis.PRECISIONS:
                for phase in phases:
                    for layer, layer_contract in analysis.RESIDUAL_LAYER_CONTRACT.items():
                        gate_failure = (
                            scheme == "baseline"
                            and role == "best_validation"
                            and precision == "float32"
                            and layer == "Layer_2"
                        )
                        p90 = 0.011 if gate_failure else 0.001
                        maximum = 0.02 if gate_failure else 0.002
                        base_beta = injected_beta / divisors[scheme]
                        phase_beta = (
                            -base_beta
                            if phase == "negative"
                            else base_beta
                            if phase == "positive"
                            else 0.0
                        )
                        selected = {
                            "mean": p90 * 0.5,
                            "median": p90 * 0.8,
                            "p90": p90,
                            "p99": (p90 + maximum) / 2.0,
                            "maximum": maximum,
                        }
                        raw = dict(selected)
                        occupancy = {
                            "mean": 0.4,
                            "median": 0.4,
                            "p90": 0.5,
                            "p99": 0.55,
                            "maximum": 0.6,
                        }
                        row: dict[str, object] = {
                            "schema": analysis.RESIDUAL_SCHEMA,
                            "architecture": "conv3",
                            "scheme": scheme,
                            "checkpoint_role": role,
                            "precision": precision,
                            "eqprop_variant": variant,
                            "T": 64,
                            "K": 64,
                            "batch_index": 0,
                            "batch_payload_sha256": payload_sha,
                            "batch_source_indices_sha256": indices_sha,
                            "phase": phase,
                            "beta": phase_beta,
                            "layer_index": layer_contract["layer_index"],
                            "state_layer_name": layer,
                            "state_layer_type": layer_contract["state_layer_type"],
                            "layer_role": layer_contract["layer_role"],
                            "residual_mode": layer_contract["residual_mode"],
                            "example_count": 16,
                            "expected_example_count": 16,
                            "unique_source_index_count": 16,
                            "coverage_complete": True,
                            "outcome": "ok",
                            "per_sample_selected_max_sha256": "c" * 64,
                            **{
                                (
                                    "selected_maximum"
                                    if name == "maximum"
                                    else f"selected_max_{name}"
                                ): value
                                for name, value in selected.items()
                            },
                            **{
                                (
                                    "raw_maximum"
                                    if name == "maximum"
                                    else f"raw_max_{name}"
                                ): value
                                for name, value in raw.items()
                            },
                            "gate_threshold": 0.01,
                            "gate_comparison": "strictly_less_than",
                            "gate_passed": not gate_failure,
                            "uniform_equilibrium_max_threshold": 0.01,
                            "uniform_equilibrium": not gate_failure,
                            "hard_failure_maximum_threshold": 0.1,
                            "hard_failure": False,
                        }
                        for name in ("mean", "median", "p90", "p99", "maximum"):
                            occupancy_field = (
                                "clamp_occupancy_maximum"
                                if name == "maximum"
                                else f"clamp_occupancy_{name}"
                            )
                            row[occupancy_field] = (
                                ""
                                if layer_contract["residual_mode"] == "raw"
                                else occupancy[name]
                            )
                        rows.append(row)
    return rows


def _make_bundle(
    *,
    study_root: Path,
    source_run: Path,
    source_config_path: Path,
    payload_sha: str,
    indices_sha: str,
    variant: str,
    beta_index: int,
) -> Path:
    injected_beta = analysis.EXPECTED_BETAS[beta_index]
    divisors = {"baseline": 1.0, "ours": 64.0, "legacy": 4096.0}
    run_id = analysis._expected_run_id(variant, beta_index)
    run_dir = study_root / run_id
    source_config = json.loads(source_config_path.read_text(encoding="utf-8"))
    source_files = _source_file_hashes(source_run)
    manifest = {
        "study_id": f"{source_config['study_id']}--float64-shadow",
        "run_id": run_id,
        "arm_id": f"conv_{variant}_eqprop_float32_float64_shadow",
        "evidence_class": "ordinary_mnist_learning_algorithm_gradient_diagnostic",
        "dataset": "ordinary_mnist",
        "smoke": False,
        "configuration": {
            "path": str(source_config_path),
            "sha256": reporting.sha256_file(source_config_path),
            "resolved": {
                "source_beta_config": source_config,
                "source_eqprop_variant": "positive_one_sided",
                "eqprop_variant": variant,
                "selected_cases": _selected_cases(
                    variant=variant,
                    injected_beta=injected_beta,
                    divisors=divisors,
                ),
                "batch_index": 0,
                "tk_override": [64, 64],
            },
        },
        "source_native_beta_run": {
            "run_id": source_run.name,
            "result_sha256": source_files["result.json"],
            "source_files": source_files,
        },
        "runtime": {"runtime_source": {"head": "runtime-head"}},
    }
    reporting.start_run(run_dir, manifest)
    shutil.copyfile(source_run / "cohort.json", run_dir / "source_cohort.json")
    shutil.copyfile(source_run / "cohort.json", run_dir / "materialized_cohort.json")
    _write_csv(
        run_dir / "beta_scaling.csv",
        _scaling_rows(
            variant=variant,
            injected_beta=injected_beta,
            divisors=divisors,
        ),
    )
    _write_csv(
        run_dir / "parameter_precision_comparison.csv",
        _parameter_rows(
            variant=variant,
            beta_index=beta_index,
            injected_beta=injected_beta,
            divisors=divisors,
            payload_sha=payload_sha,
        ),
    )
    _write_csv(
        run_dir / "state_displacement.csv",
        _displacement_rows(
            variant=variant,
            beta_index=beta_index,
            injected_beta=injected_beta,
            divisors=divisors,
            payload_sha=payload_sha,
            indices_sha=indices_sha,
        ),
    )
    _write_csv(
        run_dir / "equilibrium_residual_summary.csv",
        _residual_rows(
            variant=variant,
            beta_index=beta_index,
            injected_beta=injected_beta,
            divisors=divisors,
            payload_sha=payload_sha,
            indices_sha=indices_sha,
        ),
    )
    guards = {
        key: True for key in analysis.TRUE_READ_ONLY_GUARDS
    }
    guards.update(
        {
            "official_test_read": False,
            "optimizer_constructed": False,
            "optimizer_steps_applied": False,
            "source_bundle_validation_errors": [],
            "bptt_replay": {
                "all_bptt_and_configured_eqprop_phases_started_from_common_post_T": True
            },
        }
    )
    _write_json(run_dir / "read_only_guards.json", guards)
    completion = {key: True for key in analysis.TRUE_COMPLETION_GUARDS}
    completion.update({"official_test_read": False, "optimizer_steps_applied": False})
    reporting.complete_run(
        run_dir,
        terminal_metrics={
            "eqprop_variant": variant,
            "checkpoint_cases": 6,
            "precision_comparisons": 24,
            "batch_index": 0,
            "batch_examples": 16,
            "bias_gradients_excluded": True,
        },
        completion=completion,
    )
    return run_dir


def _complete_study(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    source_run, source_config, payload_sha, indices_sha = _source_fixture(tmp_path)
    config_path, launcher, study_root = _plan_fixture(
        tmp_path,
        source_run=source_run,
        source_config_path=source_config,
        payload_sha=payload_sha,
        indices_sha=indices_sha,
    )
    for variant in analysis.VARIANTS:
        for beta_index in range(len(analysis.EXPECTED_BETAS)):
            _make_bundle(
                study_root=study_root,
                source_run=source_run,
                source_config_path=source_config,
                payload_sha=payload_sha,
                indices_sha=indices_sha,
                variant=variant,
                beta_index=beta_index,
            )
    return config_path, launcher, study_root, source_run


def test_complete_sweep_validates_hashes_coverage_and_preserves_undefined_cosine(
    tmp_path: Path,
) -> None:
    config_path, launcher, study_root, _ = _complete_study(tmp_path)
    summary = analysis.analyze(
        config_path=config_path,
        launcher_path=launcher,
        study_root=study_root,
    )

    assert summary["coverage"]["production_bundle_count"] == 22
    assert summary["coverage"]["joined_layer_row_count"] == 1056
    assert summary["coverage"]["undefined_exact_zero_cosine_count"] == 1
    assert summary["coverage"]["equilibrium_residual_row_count"] == 3696
    assert summary["coverage"]["equilibrium_residual_gate_failure_count"] == 77
    assert summary["coverage"]["equilibrium_residual_hard_failure_count"] == 0
    residual = summary["equilibrium_residual_audit"]
    assert residual["row_count_by_variant"] == {
        "positive_one_sided": 1584,
        "centered": 2112,
    }
    assert residual["all_float64_gates_passed"] is True
    assert {
        (
            row["scheme"],
            row["checkpoint_role"],
            row["precision"],
            tuple(row["failed_state_layers"]),
            row["gate_failure_count"],
        )
        for row in residual["context_summaries"]
        if not row["all_gates_passed"]
    } == {("baseline", "best_validation", "float32", ("H1",), 77)}
    assert summary["primary_beta_coordinate"].startswith("effective_beta")
    assert summary["manifest_study_id_deviation"] == {
        "present": True,
        "plan_study_id": "planned-complete-beta-study",
        "observed_bundle_manifest_study_id": "source-current-beta-study--float64-shadow",
        "reason": "audit runner inherited its generic source-compatible float64-shadow study ID; plan and launcher are bound here by SHA-256",
    }
    assert summary["input_contract"]["plan_config"]["sha256"] == reporting.sha256_file(
        config_path
    )
    assert summary["input_contract"]["launcher"]["sha256"] == reporting.sha256_file(
        launcher
    )

    rows = list(csv.DictReader((study_root / "analysis/summary.csv").open()))
    undefined = [row for row in rows if row["cosine_defined"] == "False"]
    assert len(undefined) == 1
    assert undefined[0]["eqprop_vs_bptt_cosine"] == ""
    assert undefined[0]["cosine_undefined_reason"] == "eqprop_exact_zero"
    assert float(undefined[0]["injected_beta"]) == pytest.approx(0.001)
    one_sided = next(
        row
        for row in rows
        if row["eqprop_variant"] == "positive_one_sided"
        and row["scheme"] == "ours"
        and row["gradient_layer"] == "C1"
    )
    centered = next(
        row
        for row in rows
        if row["eqprop_variant"] == "centered"
        and row["scheme"] == "ours"
        and row["gradient_layer"] == "C1"
        and row["injected_beta"] == one_sided["injected_beta"]
        and row["precision"] == one_sided["precision"]
        and row["checkpoint_role"] == one_sided["checkpoint_role"]
    )
    assert one_sided["matched_zero_negative_delta_rms"] == ""
    assert centered["matched_zero_negative_delta_rms"] != ""
    assert len(list((study_root / "analysis/plots").glob("*.png"))) == 6
    assert (study_root / "analysis/report.md").is_file()

    shutil.rmtree(study_root / analysis._expected_run_id("centered", 10))
    with pytest.raises(ValueError, match="exactly 22 production bundles"):
        analysis.analyze(
            config_path=config_path,
            launcher_path=launcher,
            study_root=study_root,
        )


def test_undefined_cosine_requires_an_exact_zero_vector() -> None:
    assert analysis._cosine_status(
        cosine=None,
        eqprop_l2=0.0,
        bptt_l2=1.0,
        eqprop_zero_fraction=1.0,
        bptt_zero_fraction=0.0,
    ) == (False, True, False, "eqprop_exact_zero")
    with pytest.raises(ValueError, match="not explained"):
        analysis._cosine_status(
            cosine=None,
            eqprop_l2=1.0,
            bptt_l2=1.0,
            eqprop_zero_fraction=0.0,
            bptt_zero_fraction=0.0,
        )


def test_residual_gate_boolean_must_match_strict_p90_semantics(tmp_path: Path) -> None:
    source_run, source_config_path, payload_sha, indices_sha = _source_fixture(tmp_path)
    plan_path, _, _ = _plan_fixture(
        tmp_path,
        source_run=source_run,
        source_config_path=source_config_path,
        payload_sha=payload_sha,
        indices_sha=indices_sha,
    )
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    source_config = json.loads(source_config_path.read_text(encoding="utf-8"))
    rows = _residual_rows(
        variant="positive_one_sided",
        beta_index=0,
        injected_beta=0.001,
        divisors={"baseline": 1.0, "ours": 64.0, "legacy": 4096.0},
        payload_sha=payload_sha,
        indices_sha=indices_sha,
    )
    passing = next(row for row in rows if row["gate_passed"] is True)
    passing["gate_passed"] = False
    with pytest.raises(ValueError, match="p90 gate semantics mismatch"):
        analysis._validate_residual_rows(
            rows,
            plan=plan,
            source_config=source_config,
            variant="positive_one_sided",
            beta_index=0,
            injected_beta=0.001,
            payload_sha=payload_sha,
            indices_sha=indices_sha,
            run_id="one-sided-B1em3-tk64",
            run_dir=tmp_path,
        )
