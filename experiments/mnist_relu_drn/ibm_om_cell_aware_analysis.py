"""Artifact-verified analysis for the cell-aware exact-bounds study."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from experiments.artifacts import atomic_write_json, sha256_file
from experiments.definitions import resolve_experiment_config
from experiments.mnist_relu_drn.ibm_om_cell_aware_launcher import (
    CONFIG_ROOT,
    ENDPOINT_SEEDS,
    HELDOUT_ARM_MODES,
    HELDOUT_SOURCE_ARMS,
    STUDY_ID,
    TEACHER,
    _completed_run,
    _load_development_freeze,
)
from experiments.mnist_relu_drn.ibm_om_deployment_decomposition import (
    _binding_layout,
    _cost_summary,
    _exact_tree_equal,
    _layer_decomposition,
    _output_comparison,
    _output_state_summary,
    _population_from_deployment,
    _strict_torch_load,
    canonical_layer_geometries,
    collect_evaluation_trace,
    evaluate_reversible_states,
)
from experiments.mnist_relu_drn.components import build_student_stack
from experiments.mnist_relu_drn.runtime import (
    _amplification_index_report,
    _load_teacher,
    _validate_checkpoint_metadata,
)
from experiments.mnist_shared import build_mnist_loaders
from experiments.schema import RunMode
from experiments.study_workflow import summarize_study
from training.checkpoint import load_named_weights
from training.ibm_reram_hwa import (
    IbmReramHwaConfig,
    build_ibm_reram_cell_aware_exact_bounds_codebook,
    map_ibm_reram_array_targets,
    validate_ibm_reram_target_mapping_preflight,
)


SCHEMA = "ebl.mnist_relu_drn.ibm_om_cell_aware_exact_bounds_analysis"
SCHEMA_VERSION = 1
BOOTSTRAP_SEED = 20260824
BOOTSTRAP_REPLICATES = 100_000


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--study-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    return parser


def _read_json(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Expected readable JSON artifact: {path}.") from error
    if not isinstance(value, Mapping):
        raise ValueError(f"Expected JSON object artifact: {path}.")
    return value


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("Expected at least one scalar.")
    return float(sum(values) / len(values))


def _sample_std(values: Sequence[float]) -> float:
    return float(np.asarray(values, dtype=np.float64).std(ddof=1))


def _paired_bootstrap(values: Sequence[float]) -> dict[str, Any]:
    observed = np.asarray(values, dtype=np.float64)
    if observed.shape != (5,) or not np.isfinite(observed).all():
        raise ValueError("Expected five finite paired endpoint differences.")
    generator = np.random.default_rng(BOOTSTRAP_SEED)
    indices = generator.integers(
        0,
        observed.size,
        size=(BOOTSTRAP_REPLICATES, observed.size),
    )
    means = observed[indices].mean(axis=1)
    low, high = np.quantile(means, (0.025, 0.975))
    return {
        "seed": BOOTSTRAP_SEED,
        "replicates": BOOTSTRAP_REPLICATES,
        "method": "paired_nonparametric_percentile",
        "lower": float(low),
        "upper": float(high),
    }


def _flat_selected(bindings: Sequence[Any], *, low: float, high: float) -> torch.Tensor:
    span = high - low
    return torch.cat(
        tuple(
            ((binding.state.detach() - low) / span).cpu().reshape(-1)
            for binding in bindings
        )
    )


def _validate_and_replay(
    *,
    deployment: Mapping[str, Any],
    modifier_parameters: Mapping[str, Any],
    selected_normalized: torch.Tensor,
    selected_sha256: str,
    binding_keys: Sequence[str],
    binding_shapes: Sequence[tuple[int, ...]],
) -> tuple[dict[str, torch.Tensor], IbmReramHwaConfig, Any, Mapping[str, Any]]:
    if (
        deployment.get("schema") != "ebl.ibm_reram.om_pulse_resolved_deployment"
        or deployment.get("schema_version") != 1
        or deployment.get("source_weights_sha256") != selected_sha256
    ):
        raise ValueError("Expected a matching held-out pulse deployment sidecar.")
    config = IbmReramHwaConfig(**dict(deployment.get("config", {})))
    configured = IbmReramHwaConfig(**dict(modifier_parameters))
    if asdict(config) != asdict(configured):
        raise ValueError("Expected deployment config to match held-out config exactly.")
    keys = tuple(deployment.get("binding_keys", ()))
    shapes = tuple(tuple(shape) for shape in deployment.get("binding_shapes", ()))
    if keys != tuple(binding_keys) or shapes != tuple(binding_shapes):
        raise ValueError("Expected held-out deployment binding parity.")
    size = int(selected_normalized.numel())
    tensors = {}
    for name, dtype in (
        ("global_requested_target", torch.float32),
        ("requested_target", torch.float32),
        ("apparent_endpoint", torch.float32),
        ("persistent_endpoint", torch.float32),
        ("accepted", torch.bool),
    ):
        value = deployment.get(name)
        if not isinstance(value, torch.Tensor) or value.shape != (size,):
            raise ValueError(f"Expected flat deployment tensor {name!r}.")
        tensors[name] = value.detach().cpu().to(dtype=dtype)
    if not torch.equal(selected_normalized, tensors["global_requested_target"]):
        raise ValueError("Expected checkpoint to reproduce global requested targets.")
    population = _population_from_deployment(deployment, config=config)
    remapped, mapping_report = map_ibm_reram_array_targets(
        tensors["global_requested_target"],
        population,
        target_mapping=config.target_mapping,
        dual_rail_layout_by_parameter=config.dual_rail_layout_by_parameter,
        common_window_margin_fraction=config.common_window_margin_fraction,
        cell_aware_mode=config.cell_aware_mode,
        cell_aware_signed_levels=config.cell_aware_signed_levels,
    )
    validate_ibm_reram_target_mapping_preflight(mapping_report)
    if not torch.equal(remapped, tensors["requested_target"]):
        raise ValueError("Expected bit-exact held-out target reconstruction.")
    codebook = build_ibm_reram_cell_aware_exact_bounds_codebook(
        tensors["global_requested_target"],
        population,
        dual_rail_layout_by_parameter=(
            config.dual_rail_layout_by_parameter or ()
        ),
        cell_aware_mode=config.cell_aware_mode or "",
        cell_aware_signed_levels=config.cell_aware_signed_levels or 0,
    )
    if not _exact_tree_equal(deployment.get("oracle_codebook"), codebook):
        raise ValueError("Expected bit-exact held-out oracle codebook replay.")
    if deployment.get("target_mapping_report") != mapping_report:
        raise ValueError("Expected held-out mapping-report replay parity.")
    return tensors, config, population, mapping_report


def _analyze_deployment(
    *,
    arm_id: str,
    endpoint_seed: int,
    config_path: Path,
    run_dir: Path,
    weights_path: Path,
) -> dict[str, Any]:
    definition, spec = resolve_experiment_config(config_path, RunMode.VALIDATE)
    if definition.experiment_id != "mnist_relu_drn_kd.v1":
        raise ValueError("Expected MNIST DRN validation config.")
    # Analysis is read-only but uses the same configured CUDA equations.
    stack = build_student_stack(spec, enable_measured=False)
    loaded = load_named_weights(weights_path, stack.bundle.catalog)
    teacher_sha = sha256_file(TEACHER)
    _validate_checkpoint_metadata(
        loaded.metadata,
        spec=spec,
        teacher_sha256=teacher_sha,
        expected_amplification_indices=_amplification_index_report(stack),
    )
    shadow_gain = float(loaded.metadata["fixed_logit_gain"])
    stack.cost.gain = shadow_gain
    teacher, _metadata = _load_teacher(
        TEACHER,
        device=stack.device,
        spec=spec,
    )
    bindings = tuple(stack.bundle.catalog.trainable)
    keys, shapes, sections = _binding_layout(bindings)
    selected = _flat_selected(
        bindings,
        low=float(spec.model.conductance_min),
        high=float(spec.model.conductance_max),
    )
    deployment_path = run_dir / "artifacts" / "ibm_om_deployment.pt"
    deployment = _strict_torch_load(deployment_path)
    if not isinstance(deployment, Mapping):
        raise ValueError("Expected held-out deployment mapping.")
    parameters = spec.settings.weight_modifier.parameters
    tensors, config, _population, mapping_report = _validate_and_replay(
        deployment=deployment,
        modifier_parameters=parameters,
        selected_normalized=selected,
        selected_sha256=sha256_file(weights_path),
        binding_keys=keys,
        binding_shapes=shapes,
    )
    data = build_mnist_loaders(
        spec.data,
        data_seed=spec.runtime.data_seed,
        calibration_examples=spec.mapping.calibration_examples,
        calibration_batch_size=spec.mapping.calibration_batch_size,
    )
    clean_trace = collect_evaluation_trace(
        stack,
        teacher,
        data.test,
        maximum_batches=None,
    )
    device_gain = config.forward_logit_gain
    if device_gain is None:
        raise ValueError("Expected frozen device-forward gain.")
    stack.cost.gain = float(device_gain)
    traces = evaluate_reversible_states(
        bindings,
        {
            "ideal_mapped_target": tensors["requested_target"],
            "apparent_programmed": tensors["apparent_endpoint"],
            "persistent_counterfactual": tensors["persistent_endpoint"],
        },
        conductance_min=float(spec.model.conductance_min),
        conductance_max=float(spec.model.conductance_max),
        evaluator=lambda: collect_evaluation_trace(
            stack,
            teacher,
            data.test,
            maximum_batches=None,
        ),
    )
    if len({clean_trace.cohort_sha256, *(trace.cohort_sha256 for trace in traces.values())}) != 1:
        raise RuntimeError("Expected identical full-test replay cohorts.")
    result = _read_json(run_dir / "result.json")
    result_metrics = result.get("metrics")
    if not isinstance(result_metrics, Mapping):
        raise ValueError("Expected held-out result metrics.")
    apparent_accuracy = traces["apparent_programmed"].metrics["student_accuracy"]
    if not math.isclose(
        float(result_metrics["student_accuracy"]),
        apparent_accuracy,
        rel_tol=0.0,
        abs_tol=1e-8,
    ):
        raise ValueError("Expected analyzer apparent accuracy to match native result.")
    layouts = dict(config.dual_rail_layout_by_parameter or ())
    layers = canonical_layer_geometries(
        keys,
        shapes,
        sections,
        encoding=spec.model.encoding,
        dual_rail_layout_by_parameter=layouts,
    )
    size = int(selected.numel())
    saturated = deployment.get("saturated")
    saturated_tensor = saturated if isinstance(saturated, torch.Tensor) else None
    raw_apparent = deployment.get("raw_apparent_endpoint")
    clipped = (
        raw_apparent != tensors["apparent_endpoint"]
        if isinstance(raw_apparent, torch.Tensor)
        else torch.zeros(size, dtype=torch.bool)
    )
    layer_reports = {
        layer.name: _layer_decomposition(
            layer=layer,
            size=size,
            clean=selected,
            global_requested=tensors["global_requested_target"],
            mapped=tensors["requested_target"],
            apparent=tensors["apparent_endpoint"],
            persistent=tensors["persistent_endpoint"],
            accepted=tensors["accepted"],
            saturated=saturated_tensor,
            clipped=clipped,
            fallback=None,
            conductance_min=float(spec.model.conductance_min),
            conductance_max=float(spec.model.conductance_max),
            deployment=deployment,
        )
        for layer in layers
    }
    all_devices = torch.ones(size, dtype=torch.bool)
    return {
        "arm_id": arm_id,
        "endpoint_seed": endpoint_seed,
        "mode": config.cell_aware_mode,
        "run_dir": str(run_dir),
        "config": {"path": str(config_path), "sha256": sha256_file(config_path)},
        "weights": {"path": str(weights_path), "sha256": sha256_file(weights_path)},
        "deployment": {
            "path": str(deployment_path),
            "sha256": sha256_file(deployment_path),
        },
        "population_fingerprint": deployment.get("population_fingerprint"),
        "shadow_logit_gain": shadow_gain,
        "device_forward_logit_gain": float(device_gain),
        "cohort_sha256": clean_trace.cohort_sha256,
        "states": {
            "clean_fp32_shadow": _output_state_summary(clean_trace),
            **{
                name: _output_state_summary(trace)
                for name, trace in traces.items()
            },
        },
        "apparent_versus_ideal": _output_comparison(
            traces["apparent_programmed"],
            traces["ideal_mapped_target"],
        ),
        "persistent_versus_ideal": _output_comparison(
            traces["persistent_counterfactual"],
            traces["ideal_mapped_target"],
        ),
        "mapping_report": mapping_report,
        "programming": _cost_summary(deployment, all_devices),
        "layers": layer_reports,
    }


def _report_markdown(analysis: Mapping[str, Any]) -> str:
    primary = analysis["primary_hypothesis"]
    rows = [
        "# Cell-aware exact-bounds study analysis",
        "",
        f"Study: `{analysis['study_id']}`",
        "",
        "## Primary result",
        "",
        (
            "Quantized-QAT minus continuous-HWA mean programmed accuracy: "
            f"{100.0 * primary['mean_accuracy_difference']:.3f} percentage points."
        ),
        "",
        (
            "Quantized-QAT ideal-retention ratio: "
            f"{100.0 * primary['quantized_qat_ideal_retention_ratio']:.3f}%."
        ),
        "",
        f"Two-point improvement gate: **{primary['improvement_gate_passed']}**.",
        "",
        f"Ninety-percent ideal-retention gate: **{primary['retention_gate_passed']}**.",
        "",
        "## Held-out programmed accuracy",
        "",
        "| Pipeline | Mean | Std. dev. |",
        "| --- | ---: | ---: |",
    ]
    for arm_id, summary in analysis["heldout_accuracy_summary"].items():
        rows.append(
            f"| {arm_id} | {100.0 * summary['mean']:.3f}% | "
            f"{100.0 * summary['sample_standard_deviation']:.3f} pp |"
        )
    rows.extend(
        [
            "",
            "The cell-aware mapper is an exact-bounds oracle and consumes hidden "
            "per-cell ranges. This result is not a no-characterization deployment claim.",
            "",
        ]
    )
    return "\n".join(rows)


def run_analysis(
    *,
    study_dir: Path,
    output: Path,
    report: Path,
    source_commit: str,
) -> dict[str, Any]:
    study_dir = study_dir.expanduser().resolve()
    summary = summarize_study(study_dir, verify_artifacts=True)
    if summary.get("state") != "ready_for_review":
        raise RuntimeError(
            "Expected complete artifact-verified study coverage before analysis. "
            f"Provided state: {summary.get('state')!r}."
        )
    freeze = _load_development_freeze(
        study_dir,
        source_commit=source_commit,
    )
    checkpoints = freeze["checkpoints"]
    records = []
    for arm_id, mode in HELDOUT_ARM_MODES.items():
        source_arm = HELDOUT_SOURCE_ARMS[arm_id]
        weights = Path(checkpoints[source_arm]["weights"])
        for endpoint_seed in ENDPOINT_SEEDS:
            config = CONFIG_ROOT / f"heldout_{mode}_seed_{endpoint_seed}.json"
            run_dir = _completed_run(study_dir / "runs" / arm_id, config)
            if run_dir is None:
                raise RuntimeError(
                    f"Missing completed held-out run for {arm_id}/{endpoint_seed}."
                )
            records.append(
                _analyze_deployment(
                    arm_id=arm_id,
                    endpoint_seed=endpoint_seed,
                    config_path=config,
                    run_dir=run_dir,
                    weights_path=weights,
                )
            )
    by_arm: dict[str, list[dict[str, Any]]] = {
        arm_id: [] for arm_id in HELDOUT_ARM_MODES
    }
    for record in records:
        by_arm[record["arm_id"]].append(record)
    for values in by_arm.values():
        values.sort(key=lambda item: item["endpoint_seed"])
    accuracy_summary = {}
    for arm_id, values in by_arm.items():
        accuracies = [
            float(value["states"]["apparent_programmed"]["metrics"]["student_accuracy"])
            for value in values
        ]
        accuracy_summary[arm_id] = {
            "endpoint_seeds": list(ENDPOINT_SEEDS),
            "values": accuracies,
            "mean": _mean(accuracies),
            "sample_standard_deviation": _sample_std(accuracies),
        }
    quantized = by_arm["deploy-quantized-qat-heldout"]
    continuous = by_arm["deploy-continuous-hwa-heldout"]
    paired = [
        float(q["states"]["apparent_programmed"]["metrics"]["student_accuracy"])
        - float(c["states"]["apparent_programmed"]["metrics"]["student_accuracy"])
        for q, c in zip(quantized, continuous)
    ]
    quantized_apparent = [
        float(value["states"]["apparent_programmed"]["metrics"]["student_accuracy"])
        for value in quantized
    ]
    quantized_ideal = [
        float(value["states"]["ideal_mapped_target"]["metrics"]["student_accuracy"])
        for value in quantized
    ]
    retention = _mean(quantized_apparent) / _mean(quantized_ideal)
    primary = {
        "paired_endpoint_seeds": list(ENDPOINT_SEEDS),
        "paired_accuracy_differences": paired,
        "mean_accuracy_difference": _mean(paired),
        "sample_standard_deviation": _sample_std(paired),
        "paired_bootstrap_95_percent_interval": _paired_bootstrap(paired),
        "minimum_required_accuracy_difference": 0.02,
        "improvement_gate_passed": _mean(paired) >= 0.02,
        "quantized_qat_mean_apparent_accuracy": _mean(quantized_apparent),
        "quantized_qat_mean_ideal_accuracy": _mean(quantized_ideal),
        "quantized_qat_ideal_retention_ratio": retention,
        "minimum_required_ideal_retention_ratio": 0.90,
        "retention_gate_passed": retention >= 0.90,
        "hypothesis_supported": _mean(paired) >= 0.02 and retention >= 0.90,
    }
    analysis = {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "study_id": STUDY_ID,
        "source_commit": source_commit,
        "study_summary_sha256": sha256_file(
            study_dir / "analysis" / "summary.json"
        ),
        "oracle_information_policy": (
            "exact_hidden_per_cell_bounds_consumed_by_target_generation"
        ),
        "primary_hypothesis": primary,
        "heldout_accuracy_summary": accuracy_summary,
        "development_freeze": freeze,
        "deployments": records,
    }
    atomic_write_json(output.expanduser().resolve(), analysis)
    report_path = report.expanduser().resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(_report_markdown(analysis), encoding="utf-8")
    return analysis


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    run_analysis(
        study_dir=args.study_dir,
        output=args.output,
        report=args.report,
        source_commit=args.source_commit,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - operational entry point
    raise SystemExit(main())
