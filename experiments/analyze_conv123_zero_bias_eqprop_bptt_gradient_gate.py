#!/usr/bin/env python3
"""Qualify zero-bias centered frozen-current EqProp against BPTT.

This is a read-only ordinary-MNIST replay.  It loads the accepted Adam
zero-bias Conv1/2/3 checkpoints, reconstructs their initializations, and
compares true-float64 EqProp and BPTT weight gradients from identical post-T
states.  No optimizer is constructed and the official test split is not read.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import socket
import subprocess
import sys
from typing import Any, Mapping, Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = (
    REPOSITORY_ROOT
    / "configs/conv/perfectdiode_conv123_zero_bias_adam_eqprop_bptt_gradient_gate_tk4_6_8_seed0_20260816_v1.json"
)
SHADOW_RUNNER = REPOSITORY_ROOT / "experiments/audit_eqprop_float64_shadow.py"
SCHEMA = "perfectdiode-conv123-zero-bias-eqprop-bptt-gradient-gate/v1"
ROLE_ORDER = {"reconstructed_initialization": 0, "best_validation": 1}
ARCH_ORDER = {"conv1": 0, "conv2": 1, "conv3": 2}
SCHEME_ORDER = {"baseline": 0, "ours": 1, "legacy": 2}
NORM_EPSILON = 1.0e-30


def _ensure_bootstrap_config() -> None:
    if "--config" not in sys.argv:
        sys.argv.extend(("--config", str(DEFAULT_CONFIG)))


def _load_shadow_runner() -> Any:
    _ensure_bootstrap_config()
    spec = importlib.util.spec_from_file_location(
        "_zero_bias_eqprop_float64_shadow", SHADOW_RUNNER
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {SHADOW_RUNNER}.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


shadow = _load_shadow_runner()
base = shadow.base
torch = base.torch


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    return value


def _semantic_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _git(root: Path, *arguments: str, check: bool = True) -> str:
    result = subprocess.run(
        ["git", "-C", str(root), *arguments],
        check=False,
        capture_output=True,
        text=True,
    )
    if check and result.returncode:
        raise RuntimeError(
            f"git {' '.join(arguments)} failed in {root}: {result.stderr.strip()}"
        )
    return result.stdout.strip()


def _artifact_map(result: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for row in result.get("artifacts", []):
        path = str(row["path"])
        if path in output:
            raise ValueError(f"Duplicate source artifact record {path}.")
        output[path] = dict(row)
    return output


def _source_file_hashes(run_dir: Path, names: Sequence[str]) -> dict[str, str]:
    output: dict[str, str] = {}
    for name in names:
        path = run_dir / name
        if not path.is_file():
            raise FileNotFoundError(path)
        output[name] = base.sha256_file(path)
    return output


def _bias_lr_proof(source_config: Mapping[str, Any]) -> dict[str, Any]:
    names = [str(value) for value in source_config["parameter_order"]]
    rates = [float(value) for value in source_config["optimizer"]["learning_rate"]]
    if len(names) != len(rates):
        raise ValueError("Source parameter order and learning-rate vector differ in length.")
    mapping = dict(zip(names, rates, strict=True))
    bias = {name: rate for name, rate in mapping.items() if name.startswith("Bias_")}
    if not bias or any(rate != 0.0 for rate in bias.values()):
        raise ValueError(f"Source bias learning-rate contract failed: {bias}.")
    if str(source_config["optimizer"]["name"]).lower() != "adam":
        raise ValueError("The Adam-first gradient gate requires Adam source checkpoints.")
    return {
        "parameter_order": names,
        "learning_rates_by_parameter": mapping,
        "bias_learning_rates": bias,
        "all_bias_learning_rates_exact_zero": True,
    }


def _runtime_bias_proof(
    case: Mapping[str, Any], *, role: str, device: torch.device
) -> dict[str, Any]:
    runtime, checkpoint_guard = shadow._checkpoint_runtime(
        case,
        role=role,
        device=device,
        gradient_iterations=int(case["native_K"]),
        nudging_mode="current",
    )
    biases = [
        parameter
        for parameter in shadow._all_runtime_parameters(runtime)
        if parameter.__class__.__name__ == "Bias"
    ]
    if not biases:
        raise RuntimeError(f"No bias tensors found for {case['architecture']}/{case['scheme']}.")
    rows = []
    for parameter in biases:
        state = parameter.state.detach().cpu()
        nonzero = int(torch.count_nonzero(state))
        rows.append(
            {
                "parameter_name": str(parameter.name).strip(),
                "element_count": int(state.numel()),
                "nonzero_count": nonzero,
                "maximum_absolute_value": float(state.abs().max()) if state.numel() else 0.0,
                "sha256": base._tensor_sha256(state),
            }
        )
    if any(row["nonzero_count"] != 0 for row in rows):
        raise ValueError(
            f"Nonzero bias tensor in {case['architecture']}/{case['scheme']}/{role}."
        )
    return {
        "architecture": case["architecture"],
        "scheme": case["scheme"],
        "checkpoint_role": role,
        "all_bias_tensors_exact_zero": True,
        "biases": rows,
        "checkpoint_guard": checkpoint_guard,
    }


def _runtime_source_proof(config: Mapping[str, Any]) -> dict[str, Any]:
    contract = config["source_contract"]
    root = Path(contract["runtime_source_root"]).expanduser().resolve()
    if root != base.RUNTIME_SOURCE_ROOT:
        raise ValueError(
            f"Bootstrapped runtime source {base.RUNTIME_SOURCE_ROOT} differs from {root}."
        )
    head = _git(root, "rev-parse", "HEAD")
    if head != contract["runtime_source_expected_head"]:
        raise ValueError(f"Runtime source HEAD mismatch: {head}.")
    changed_python = [
        line
        for line in _git(root, "status", "--porcelain", "--untracked-files=no").splitlines()
        if line.strip().endswith(".py")
    ]
    if changed_python:
        raise ValueError(
            "The exact runtime source has changed tracked Python files: "
            + "; ".join(changed_python[:20])
        )
    file_hashes = {}
    for relative in contract["runtime_code_files"]:
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        file_hashes[str(relative)] = base.sha256_file(path)
    return {
        "root": str(root),
        "head": head,
        "tracked_python_files_clean": True,
        "relevant_code_sha256": file_hashes,
    }


def _lineage_proof(config: Mapping[str, Any]) -> dict[str, Any] | None:
    lineage = config.get("lineage")
    if not isinstance(lineage, Mapping):
        return None
    path = Path(str(lineage["clean_gradient_gate_config"]))
    if not path.is_absolute():
        path = REPOSITORY_ROOT / path
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    observed_sha256 = base.sha256_file(path)
    if observed_sha256 != str(lineage["clean_gradient_gate_config_sha256"]):
        raise ValueError("The declared clean gradient-gate parent config changed.")
    parent = _read_json(path)
    for key in (
        "schema_version",
        "source_contract",
        "dataset",
        "gradient_contract",
        "checkpoint_roles",
    ):
        if config[key] != parent[key]:
            raise ValueError(f"Noisy replay differs from its clean parent in {key}.")
    architectures = {str(row["architecture"]) for row in config["cases"]}
    expected_cases = [
        row for row in parent["cases"] if str(row["architecture"]) in architectures
    ]
    if list(config["cases"]) != expected_cases:
        raise ValueError("Noisy replay cases differ from the filtered clean parent cases.")
    return {
        "clean_gradient_gate_config": str(path),
        "clean_gradient_gate_config_sha256": observed_sha256,
        "selected_architectures": sorted(architectures),
        "scientific_contract_matches_parent_except_read_noise": True,
    }


def _inventory(config: Mapping[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    source_contract = config["source_contract"]
    source_root = Path(source_contract["source_study_root"])
    if not source_root.is_absolute():
        source_root = REPOSITORY_ROOT / source_root
    source_root = source_root.resolve()
    required_files = [str(value) for value in source_contract["required_source_files"]]
    inventory: list[dict[str, Any]] = []
    source_hashes: dict[str, dict[str, str]] = {}
    bias_lr_proofs: list[dict[str, Any]] = []
    dataset_provenance: list[dict[str, Any]] = []

    for declared in config["cases"]:
        architecture = str(declared["architecture"])
        scheme = str(declared["scheme"])
        run_dir = source_root / "runs" / str(declared["run_id"])
        validation_errors = base.validate_run(run_dir)
        if validation_errors:
            raise ValueError(
                f"Invalid source bundle {run_dir}: {'; '.join(validation_errors)}"
            )
        status = _read_json(run_dir / "status.json")
        manifest = _read_json(run_dir / "manifest.json")
        result = _read_json(run_dir / "result.json")
        metrics = _read_json(run_dir / "metrics.json")
        source_config = _read_json(run_dir / "config.used.json")
        if status.get("state") != "complete" or bool(manifest.get("smoke")):
            raise ValueError(f"Source run is not a completed production bundle: {run_dir}.")
        source_git = manifest.get("git", {})
        if source_git.get("commit") != source_contract["production_source_commit"]:
            raise ValueError(f"Source commit mismatch for {run_dir.name}.")
        if (
            source_git.get("source_archive_sha256")
            != source_contract["production_source_archive_sha256"]
        ):
            raise ValueError(f"Source archive mismatch for {run_dir.name}.")
        if source_config.get("training_algorithm") != "BP" or int(source_config["seed"]) != 0:
            raise ValueError(f"Source algorithm/seed mismatch for {run_dir.name}.")
        model = base._model_config(source_config)
        if (
            int(model["num_iterations_inference"]) != int(declared["T"])
            or int(model["num_iterations_training"]) != int(declared["K"])
        ):
            raise ValueError(f"Source T/K mismatch for {architecture}/{scheme}.")
        expected_amp = {
            "baseline": (1.0, 1.0),
            "ours": (4.0, 1.0),
            "legacy": (4.0, 0.25),
        }[scheme]
        observed_amp = (float(model["voltage_amp"]), float(model["current_amp"]))
        if observed_amp != expected_amp:
            raise ValueError(
                f"Amplification mismatch for {architecture}/{scheme}: {observed_amp}."
            )
        current_scale = (observed_amp[0] / observed_amp[1]) ** int(
            declared["output_row_exponent_L"]
        )
        if not math.isclose(
            float(declared["base_beta"]) * current_scale,
            float(declared["injected_beta"]),
            rel_tol=1.0e-12,
            abs_tol=1.0e-15,
        ):
            raise ValueError(f"Base/injected beta mismatch for {architecture}/{scheme}.")

        artifacts = _artifact_map(result)
        artifact_pairs = {
            "config.used.json": "artifacts/config.used.json",
            "metrics.json": "artifacts/metrics.json",
            "best_model.pt": "checkpoints/best_model.pt",
            "weights_best.npz": "checkpoints/weights_best.npz",
        }
        for local_name, artifact_name in artifact_pairs.items():
            if artifact_name not in artifacts:
                raise ValueError(f"Missing source artifact record {artifact_name}.")
            observed_hash = base.sha256_file(run_dir / local_name)
            if observed_hash != artifacts[artifact_name]["sha256"]:
                raise ValueError(f"Source artifact hash mismatch for {run_dir.name}/{local_name}.")

        lr_proof = _bias_lr_proof(source_config)
        bias_lr_proofs.append(
            {"architecture": architecture, "scheme": scheme, **lr_proof}
        )
        provenance = dict(metrics.get("dataset_provenance", {}))
        if provenance.get("validation_indices_sha256") != config["dataset"][
            "validation_indices_sha256"
        ]:
            raise ValueError(f"Validation split mismatch for {architecture}/{scheme}.")
        dataset_provenance.append(
            {
                "architecture": architecture,
                "scheme": scheme,
                **provenance,
            }
        )
        hashes = _source_file_hashes(run_dir, required_files)
        key = f"{architecture}/{scheme}"
        source_hashes[key] = hashes

        runtime = base._build_runtime(
            source_config,
            device=torch.device("cpu"),
            gradient_iterations=int(declared["K"]),
            nudging_mode="current",
            current_scale="auto",
        )
        reconstructed_hash = base._historical_initialization_sha256(
            runtime["parameters"]
        )
        weight_count = len(runtime["weight_indices"])
        if int(declared["output_row_exponent_L"]) != weight_count - 1:
            raise ValueError(
                f"Output-row exponent mismatch for {architecture}/{scheme}."
            )
        inventory.append(
            {
                **dict(declared),
                "native_T": int(declared["T"]),
                "native_K": int(declared["K"]),
                "voltage_amp": observed_amp[0],
                "current_amp": observed_amp[1],
                "voltage_amplification": observed_amp[0],
                "current_amplification": observed_amp[1],
                "scored_interaction_count_L": weight_count,
                "reconstructed_initialization_tensor_sha256": reconstructed_hash,
                "source_config": source_config,
                "run_dir": run_dir,
                "best_checkpoint_path": run_dir / "best_model.pt",
                "weights_best_path": run_dir / "weights_best.npz",
                "best_checkpoint_sha256": hashes["best_model.pt"],
                "best_epoch": int(result["terminal_metrics"]["best_epoch"]),
                "best_validation_accuracy": float(
                    result["terminal_metrics"]["validation"]["best_accuracy"]
                ),
            }
        )

    expected_pairs = {
        (str(case["architecture"]), str(case["scheme"]))
        for case in config["cases"]
    }
    observed_pairs = {
        (str(case["architecture"]), str(case["scheme"])) for case in inventory
    }
    if observed_pairs != expected_pairs:
        raise ValueError("Source inventory does not cover every declared case.")
    architectures = sorted({architecture for architecture, _scheme in expected_pairs})
    for architecture in architectures:
        schemes = {
            scheme for candidate_architecture, scheme in expected_pairs
            if candidate_architecture == architecture
        }
        if schemes != {"baseline", "ours", "legacy"}:
            raise ValueError(
                f"Declared architecture {architecture} does not cover all schemes."
            )
        hashes = {
            str(case["reconstructed_initialization_tensor_sha256"])
            for case in inventory
            if case["architecture"] == architecture
        }
        if len(hashes) != 1:
            raise ValueError(f"Schemes do not share initialization for {architecture}.")
    return inventory, {
        "source_study_root": str(source_root),
        "source_file_hashes_before": source_hashes,
        "bias_learning_rate_proofs": bias_lr_proofs,
        "dataset_provenance": dataset_provenance,
    }


def _build_cohort(
    config: Mapping[str, Any], inventory: Sequence[Mapping[str, Any]]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    dataset = config["dataset"]
    batches, cohort = base._build_validation_cohort(
        inventory[0]["source_config"],
        data_root=Path(dataset["root"]).expanduser().resolve(),
        batch_size=int(dataset["batch_size"]),
        example_count=int(dataset["example_count"]),
    )
    observed = [
        {
            "batch_index": int(row["batch_index"]),
            "source_indices_sha256": row["source_indices_sha256"],
            "payload_sha256": row["payload_sha256"],
        }
        for row in batches
    ]
    if observed != list(dataset["expected_batch_guards"]):
        raise ValueError("Materialized cohort differs from the predeclared batch guards.")
    if cohort["validation_indices_sha256"] != dataset["validation_indices_sha256"]:
        raise ValueError("Materialized validation partition hash mismatch.")
    return batches, cohort


def _selected(case: Mapping[str, Any], role: str) -> dict[str, Any]:
    return {
        "architecture": case["architecture"],
        "scheme": case["scheme"],
        "checkpoint_role": role,
        "T": int(case["T"]),
        "K": int(case["K"]),
        "source_native_T": int(case["native_T"]),
        "source_native_K": int(case["native_K"]),
        "native_context": True,
        "tk_source": "paper_shared_contract",
        "actual_beta": float(case["base_beta"]),
        "base_beta": float(case["base_beta"]),
        "injected_beta": float(case["injected_beta"]),
    }


def _symmetric_norm_delta(left: float, right: float) -> float:
    return 2.0 * abs(left - right) / max(left + right, NORM_EPSILON)


def _read_noise_contract(config: Mapping[str, Any]) -> dict[str, Any]:
    raw = dict(config.get("read_noise_contract", {}))
    sigma = float(raw.get("endpoint_read_noise_std", 0.0))
    if not math.isfinite(sigma) or sigma < 0.0:
        raise ValueError("endpoint_read_noise_std must be finite and non-negative.")
    seed = raw.get("endpoint_read_noise_seed")
    if sigma > 0.0 and seed is None:
        raise ValueError("Nonzero endpoint read noise requires a deterministic seed.")
    if bool(raw.get("input_read_noise", False)):
        raise ValueError("This diagnostic does not support input read noise.")
    acquisition_model = str(
        raw.get(
            "acquisition_model",
            "independent_gaussian_free_layer_endpoint_voltage",
        )
    )
    if acquisition_model != "independent_gaussian_free_layer_endpoint_voltage":
        raise ValueError(f"Unsupported read-noise acquisition model: {acquisition_model}.")
    if raw and not bool(raw.get("independent_negative_positive_reads", False)):
        raise ValueError("Negative and positive endpoint reads must be independent.")
    if raw and not bool(
        raw.get("matched_standard_normal_draws_across_schemes", False)
    ):
        raise ValueError("Standard-normal draws must be matched across schemes.")
    expected_draw_rule = (
        "seed + 1000*batch_index + 100*phase_index + state_layer_index"
    )
    if raw and str(raw.get("noise_draw_rule")) != expected_draw_rule:
        raise ValueError("The declared endpoint read-noise draw rule changed.")
    return {
        "endpoint_read_noise_std": sigma,
        "endpoint_read_noise_seed": None if seed is None else int(seed),
        "input_read_noise": False,
        "acquisition_model": acquisition_model,
        "independent_negative_positive_reads": True,
        "matched_standard_normal_draws_across_schemes": True,
        "noise_draw_rule": expected_draw_rule,
    }


def _noised_states(
    states: Sequence[torch.Tensor],
    *,
    sigma: float,
    base_seed: int,
    batch_index: int,
    phase_index: int,
) -> tuple[list[torch.Tensor], list[dict[str, Any]]]:
    output: list[torch.Tensor] = []
    rows: list[dict[str, Any]] = []
    for layer_index, state in enumerate(states):
        seed = int(
            base_seed
            + 1000 * int(batch_index)
            + 100 * int(phase_index)
            + layer_index
        )
        exact = state.detach().cpu().to(torch.float64)
        if sigma == 0.0:
            noise = torch.zeros_like(exact)
        else:
            generator = torch.Generator(device="cpu")
            generator.manual_seed(seed)
            noise = torch.randn(
                tuple(exact.shape), generator=generator, dtype=torch.float64
            ) * float(sigma)
        output.append(exact + noise)
        rows.append(
            {
                "state_layer_index": layer_index,
                "seed": seed,
                "element_count": int(noise.numel()),
                "empirical_noise_mean": float(noise.mean()),
                "empirical_noise_rms": float(noise.square().mean().sqrt()),
                "empirical_noise_maximum_absolute": float(noise.abs().max()),
                "standard_normal_sha256": base._tensor_sha256(
                    noise / float(sigma) if sigma > 0.0 else noise
                ),
            }
        )
    return output, rows


def _energy_gradients_at_states(
    runtime: Mapping[str, Any],
    *,
    input_state: torch.Tensor,
    states: Sequence[torch.Tensor],
    device: torch.device,
) -> list[torch.Tensor]:
    input_layer = runtime["energy_fn"].layers()[0]
    input_layer.state = input_state.detach().to(
        device=device, dtype=torch.float64
    ).clone()
    restored = [
        value.detach().to(device=device, dtype=torch.float64).clone()
        for value in states
    ]
    base._restore_states(runtime["free_layers"], restored)
    return [
        value.detach().cpu().to(torch.float64).clone()
        for value in base._energy_gradients(runtime)
    ]


def _weight_dict(
    runtime: Mapping[str, Any], gradients: Sequence[torch.Tensor]
) -> dict[str, torch.Tensor]:
    return {
        str(runtime["parameters"][index].name).strip(): gradients[index]
        .detach()
        .cpu()
        .to(torch.float64)
        .clone()
        for index in runtime["weight_indices"]
    }


def _apply_endpoint_read_noise(
    *,
    output: Mapping[str, Any],
    case: Mapping[str, Any],
    batch: Mapping[str, Any],
    noise_contract: Mapping[str, Any],
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], list[dict[str, Any]], dict[str, Any]]:
    sigma = float(noise_contract["endpoint_read_noise_std"])
    seed = noise_contract["endpoint_read_noise_seed"]
    if sigma <= 0.0 or seed is None:
        return (
            dict(output["gradients"]),
            dict(output["numerators"]),
            [],
            {
                "enabled": False,
                "parameter_tensors_unchanged": True,
                "input_remained_exact": True,
            },
        )
    runtime = output["captured_runtime"]
    input_state = output["captured_input_state"]
    phase_states = output["captured_phase_states"]
    if set(phase_states) != {"negative", "positive"}:
        raise RuntimeError("Centered read-noise replay requires negative and positive states.")
    parameter_hash_before = base._parameter_state_sha256(runtime["parameters"])
    input_hash = base._tensor_sha256(input_state)
    endpoint_gradients: dict[str, dict[str, torch.Tensor]] = {}
    noise_rows: list[dict[str, Any]] = []
    for phase, phase_index in (("negative", 1), ("positive", 2)):
        noised, rows = _noised_states(
            phase_states[phase],
            sigma=sigma,
            base_seed=int(seed),
            batch_index=int(batch["batch_index"]),
            phase_index=phase_index,
        )
        endpoint_gradients[phase] = _weight_dict(
            runtime,
            _energy_gradients_at_states(
                runtime,
                input_state=input_state,
                states=noised,
                device=device,
            ),
        )
        for row in rows:
            noise_rows.append(
                {
                    "architecture": case["architecture"],
                    "scheme": case["scheme"],
                    "checkpoint_role": output["guard"]["checkpoint_role"],
                    "batch_index": int(batch["batch_index"]),
                    "phase": phase,
                    "endpoint_read_noise_std": sigma,
                    **row,
                }
            )

    denominator_scale = float(output["guard"]["output_row_current_scale"])
    noisy_gradients: dict[str, torch.Tensor] = {}
    noisy_numerators: dict[str, torch.Tensor] = {}
    for name in output["gradients"]:
        numerator, estimate = shadow._native_eqprop_estimate(
            eqprop_variant="centered",
            beta=float(case["base_beta"]),
            denominator_scale=denominator_scale,
            zero=output["endpoint_gradients"]["zero"][name],
            negative=endpoint_gradients["negative"][name],
            positive=endpoint_gradients["positive"][name],
        )
        noisy_numerators[name] = numerator.detach().cpu().to(torch.float64).clone()
        noisy_gradients[name] = estimate.detach().cpu().to(torch.float64).clone()

    parameter_hash_after = base._parameter_state_sha256(runtime["parameters"])
    final_input_hash = base._tensor_sha256(runtime["energy_fn"].layers()[0].state)
    guard = {
        "enabled": True,
        "endpoint_read_noise_std": sigma,
        "endpoint_read_noise_seed": int(seed),
        "noise_draw_count": len(noise_rows),
        "parameter_state_sha256_before": parameter_hash_before,
        "parameter_state_sha256_after": parameter_hash_after,
        "parameter_tensors_unchanged": parameter_hash_before == parameter_hash_after,
        "input_state_sha256_before": input_hash,
        "input_state_sha256_after": final_input_hash,
        "input_remained_exact": input_hash == final_input_hash,
        "all_noisy_gradients_float64": all(
            value.dtype == torch.float64 for value in noisy_gradients.values()
        ),
        "all_noisy_gradients_finite": all(
            bool(torch.isfinite(value).all()) for value in noisy_gradients.values()
        ),
    }
    if not all(
        bool(guard[key])
        for key in (
            "parameter_tensors_unchanged",
            "input_remained_exact",
            "all_noisy_gradients_float64",
            "all_noisy_gradients_finite",
        )
    ):
        raise RuntimeError(f"Endpoint read-noise guard failed: {guard}")
    return noisy_gradients, noisy_numerators, noise_rows, guard


def _layer_row(
    *,
    config: Mapping[str, Any],
    case: Mapping[str, Any],
    selected: Mapping[str, Any],
    batch: Mapping[str, Any],
    parameter_name: str,
    output: Mapping[str, Any],
    residual_passed: bool,
) -> dict[str, Any]:
    scored_gradients = output.get("scored_gradients", output["gradients"])
    scored_numerators = output.get("scored_numerators", output["numerators"])
    metrics = shadow._eqprop_bptt_metrics(
        eqprop_gradient=scored_gradients[parameter_name],
        bptt_gradient=output["bptt_gradients"][parameter_name],
    )
    clean_metrics = shadow._eqprop_bptt_metrics(
        eqprop_gradient=output["gradients"][parameter_name],
        bptt_gradient=output["bptt_gradients"][parameter_name],
    )
    acquisition_metrics = shadow._eqprop_bptt_metrics(
        eqprop_gradient=scored_gradients[parameter_name],
        bptt_gradient=output["gradients"][parameter_name],
    )
    cosine = metrics["cosine"]
    norm_delta = float(metrics["symmetric_norm_delta"])
    gradient_passed = bool(
        cosine is not None
        and float(cosine) >= float(config["gradient_contract"]["cosine_minimum"])
        and norm_delta
        <= float(config["gradient_contract"]["symmetric_norm_delta_maximum"])
    )
    acquisition_passed = bool(
        acquisition_metrics["cosine"] is not None
        and float(acquisition_metrics["cosine"])
        >= float(config["gradient_contract"]["cosine_minimum"])
        and float(acquisition_metrics["symmetric_norm_delta"])
        <= float(config["gradient_contract"]["symmetric_norm_delta_maximum"])
    )
    dtype_row = next(
        row
        for row in output["gradient_dtype_rows"]
        if row["parameter_name"] == parameter_name
    )
    return {
        "schema": SCHEMA,
        "architecture": case["architecture"],
        "scheme": case["scheme"],
        "checkpoint_role": selected["checkpoint_role"],
        "T": int(selected["T"]),
        "K": int(selected["K"]),
        "batch_index": int(batch["batch_index"]),
        "batch_payload_sha256": batch["payload_sha256"],
        "batch_source_indices_sha256": batch["source_indices_sha256"],
        "parameter_name": parameter_name,
        "base_beta": float(case["base_beta"]),
        "injected_beta": float(case["injected_beta"]),
        "output_row_current_scale": float(
            output["guard"]["output_row_current_scale"]
        ),
        "endpoint_read_noise_std": float(
            output.get("noise_contract", {}).get("endpoint_read_noise_std", 0.0)
        ),
        "endpoint_read_noise_seed": output.get("noise_contract", {}).get(
            "endpoint_read_noise_seed"
        ),
        "bptt_l2": float(metrics["bptt_l2"]),
        "eqprop_l2": float(metrics["eqprop_l2"]),
        "eqprop_over_bptt_norm_ratio": metrics["eqprop_over_bptt_norm_ratio"],
        "cosine": cosine,
        "symmetric_norm_delta": norm_delta,
        "relative_l2_difference_over_bptt": metrics[
            "relative_l2_difference_over_bptt"
        ],
        "clean_eqprop_over_bptt_norm_ratio": clean_metrics[
            "eqprop_over_bptt_norm_ratio"
        ],
        "clean_eqprop_bptt_cosine": clean_metrics["cosine"],
        "clean_eqprop_bptt_symmetric_norm_delta": clean_metrics[
            "symmetric_norm_delta"
        ],
        "noisy_over_clean_eqprop_norm_ratio": acquisition_metrics[
            "eqprop_over_bptt_norm_ratio"
        ],
        "noisy_clean_eqprop_cosine": acquisition_metrics["cosine"],
        "noisy_clean_eqprop_symmetric_norm_delta": acquisition_metrics[
            "symmetric_norm_delta"
        ],
        "bptt_exact_zero_fraction": float(dtype_row["bptt_exact_zero_fraction"]),
        "eqprop_exact_zero_fraction": float(
            (scored_gradients[parameter_name] == 0.0).to(torch.float64).mean()
        ),
        "finite_difference_numerator_l2": float(
            torch.linalg.vector_norm(
                scored_numerators[parameter_name].detach().to(torch.float64)
            )
        ),
        "read_noise_acquisition_passed": acquisition_passed,
        "gradient_fidelity_passed": gradient_passed,
        "read_noise_usable_gate_passed": bool(acquisition_passed and gradient_passed),
        "all_endpoint_residual_gates_passed": residual_passed,
        "unqualified_launch_gate_passed": bool(gradient_passed and residual_passed),
    }


def _summaries(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["architecture"]), str(row["scheme"]), str(row["checkpoint_role"]))].append(row)
    output = []
    for (architecture, scheme, role), values in grouped.items():
        cosines = [float(row["cosine"]) for row in values if row["cosine"] is not None]
        norm_deltas = [float(row["symmetric_norm_delta"]) for row in values]
        clean_cosines = [
            float(row["clean_eqprop_bptt_cosine"])
            for row in values
            if row["clean_eqprop_bptt_cosine"] is not None
        ]
        acquisition_cosines = [
            float(row["noisy_clean_eqprop_cosine"])
            for row in values
            if row["noisy_clean_eqprop_cosine"] is not None
        ]
        acquisition_norm_deltas = [
            float(row["noisy_clean_eqprop_symmetric_norm_delta"])
            for row in values
        ]
        ratios = [
            float(row["eqprop_over_bptt_norm_ratio"])
            for row in values
            if row["eqprop_over_bptt_norm_ratio"] is not None
        ]
        output.append(
            {
                "architecture": architecture,
                "scheme": scheme,
                "checkpoint_role": role,
                "T": int(values[0]["T"]),
                "K": int(values[0]["K"]),
                "base_beta": float(values[0]["base_beta"]),
                "injected_beta": float(values[0]["injected_beta"]),
                "endpoint_read_noise_std": float(
                    values[0]["endpoint_read_noise_std"]
                ),
                "endpoint_read_noise_seed": values[0]["endpoint_read_noise_seed"],
                "batch_count": len({int(row["batch_index"]) for row in values}),
                "layer_batch_comparison_count": len(values),
                "minimum_clean_eqprop_bptt_cosine": (
                    min(clean_cosines) if clean_cosines else None
                ),
                "minimum_cosine": min(cosines) if cosines else None,
                "median_cosine": float(torch.tensor(cosines, dtype=torch.float64).median())
                if cosines
                else None,
                "maximum_symmetric_norm_delta": max(norm_deltas),
                "minimum_noisy_clean_eqprop_cosine": (
                    min(acquisition_cosines) if acquisition_cosines else None
                ),
                "maximum_noisy_clean_eqprop_symmetric_norm_delta": max(
                    acquisition_norm_deltas
                ),
                "minimum_norm_ratio": min(ratios) if ratios else None,
                "maximum_norm_ratio": max(ratios) if ratios else None,
                "read_noise_acquisition_passed": all(
                    bool(row["read_noise_acquisition_passed"]) for row in values
                ),
                "gradient_fidelity_passed": all(
                    bool(row["gradient_fidelity_passed"]) for row in values
                ),
                "read_noise_usable_gate_passed": all(
                    bool(row["read_noise_usable_gate_passed"]) for row in values
                ),
                "all_endpoint_residual_gates_passed": all(
                    bool(row["all_endpoint_residual_gates_passed"])
                    for row in values
                ),
                "unqualified_launch_gate_passed": all(
                    bool(row["unqualified_launch_gate_passed"]) for row in values
                ),
            }
        )
    return sorted(
        output,
        key=lambda row: (
            ARCH_ORDER[row["architecture"]],
            SCHEME_ORDER[row["scheme"]],
            ROLE_ORDER[row["checkpoint_role"]],
        ),
    )


def _matched_noise_draw_guard(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    grouped: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[
            (
                row["architecture"],
                row["checkpoint_role"],
                int(row["batch_index"]),
                row["phase"],
                int(row["state_layer_index"]),
            )
        ].append(row)
    mismatches = []
    for key, values in grouped.items():
        hashes = {str(row["standard_normal_sha256"]) for row in values}
        seeds = {int(row["seed"]) for row in values}
        shapes = {int(row["element_count"]) for row in values}
        if len(hashes) != 1 or len(seeds) != 1 or len(shapes) != 1:
            mismatches.append(
                {
                    "coordinate": list(key),
                    "schemes": sorted(str(row["scheme"]) for row in values),
                    "hashes": sorted(hashes),
                    "seeds": sorted(seeds),
                    "element_counts": sorted(shapes),
                }
            )
    return {
        "coordinate_count": len(grouped),
        "row_count": len(rows),
        "matched_standard_normal_draws_across_schemes": not mismatches,
        "mismatches": mismatches,
    }


def _format_float(value: Any) -> str:
    if value is None:
        return "--"
    return f"{float(value):.6g}"


def _write_report(
    path: Path,
    *,
    config: Mapping[str, Any],
    summaries: Sequence[Mapping[str, Any]],
    smoke: bool,
) -> None:
    gradient_all = all(bool(row["gradient_fidelity_passed"]) for row in summaries)
    acquisition_all = all(
        bool(row["read_noise_acquisition_passed"]) for row in summaries
    )
    usable_all = all(bool(row["read_noise_usable_gate_passed"]) for row in summaries)
    residual_all = all(bool(row["all_endpoint_residual_gates_passed"]) for row in summaries)
    noise_contract = _read_noise_contract(config)
    lines = [
        "# Zero-Bias EqProp--BPTT Gradient Gate",
        "",
        "This is a read-only ordinary-MNIST diagnostic, not paper-facing accuracy evidence.",
        "It uses centered frozen-current EqProp in true float64, identical post-T states,",
        "the shared paper T/K, exact-zero biases, and no optimizer step or official-test read.",
        "",
        f"Run tier: `{'smoke' if smoke else 'production'}`.",
        f"Endpoint read-noise sigma: `{noise_contract['endpoint_read_noise_std']}`; seed: `{noise_contract['endpoint_read_noise_seed']}`.",
        "",
        "| Architecture | Scheme | Checkpoint | T/K | Injected beta | Clean EqProp/BPTT cosine | Noisy/clean cosine | Noisy EqProp/BPTT cosine | Max noisy/BPTT norm delta | Acquisition gate | Task gate | Residual gate |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---|---|---|",
    ]
    for row in summaries:
        lines.append(
            "| {architecture} | {scheme} | {role} | {T}/{K} | {beta} | {clean_cosine} | {acquisition_cosine} | {cosine} | {delta} | {acquisition} | {gradient} | {residual} |".format(
                architecture=row["architecture"],
                scheme=row["scheme"],
                role=row["checkpoint_role"],
                T=row["T"],
                K=row["K"],
                beta=_format_float(row["injected_beta"]),
                clean_cosine=_format_float(
                    row["minimum_clean_eqprop_bptt_cosine"]
                ),
                acquisition_cosine=_format_float(
                    row["minimum_noisy_clean_eqprop_cosine"]
                ),
                cosine=_format_float(row["minimum_cosine"]),
                delta=_format_float(row["maximum_symmetric_norm_delta"]),
                acquisition=(
                    "pass" if row["read_noise_acquisition_passed"] else "fail"
                ),
                gradient="pass" if row["gradient_fidelity_passed"] else "fail",
                residual="pass" if row["all_endpoint_residual_gates_passed"] else "fail",
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            (
                "Every noisy gradient passes the predeclared task cosine and norm gate against BPTT."
                if gradient_all
                else "At least one noisy gradient fails the predeclared task gate against BPTT."
            ),
            (
                "Every noisy gradient also passes the acquisition gate against its clean EqProp counterpart."
                if acquisition_all
                else "At least one noisy gradient fails the acquisition gate against clean EqProp."
            ),
            (
                "The combined acquisition-and-task read-noise gate passes everywhere."
                if usable_all
                else "The combined acquisition-and-task read-noise gate does not pass everywhere."
            ),
            (
                "Every endpoint also passes the predeclared projected-residual gate."
                if residual_all
                else "At least one endpoint remains residual-limited; this is reported separately from direct gradient agreement."
            ),
            "The old trained-bias cost-nudge comparison is historical warning evidence only; it is not pooled with these measurements.",
            "",
            "## Contract",
            "",
            f"- Gradient cosine minimum: `{config['gradient_contract']['cosine_minimum']}`.",
            f"- Symmetric norm delta maximum: `{config['gradient_contract']['symmetric_norm_delta_maximum']}`.",
            f"- Projected residual p90 threshold: `{config['gradient_contract']['equilibrium_residual_p90_threshold']}`.",
            "- Cohort: first 64 examples of the fixed validation order, in four batches of 16.",
            "- Checkpoints: reconstructed initialization and best validation (production only).",
            "- Parameters: ConvWeight and DenseWeight only; all Bias tensors are proven exact zero but excluded from scoring.",
            "- Read noise: independent Gaussian samples on every negative/positive free-layer endpoint voltage after equilibrium; the input and BPTT reference remain exact.",
            "",
            "Detailed artifacts: `layer_metrics.csv`, `case_summary.csv`, `equilibrium_residuals.csv`,",
            "`state_displacement.csv`, `phase_diagnostics.csv`, `endpoint_read_noise.csv`, `source_inventory.json`, and `read_only_guards.json`.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any] | None:
    config_path = args.config.expanduser().resolve()
    config = _read_json(config_path)
    if config.get("schema_version") != SCHEMA:
        raise ValueError("Unexpected zero-bias gradient-gate config schema.")
    lineage_proof = _lineage_proof(config)
    runtime_source = _runtime_source_proof(config)
    inventory, source_proof = _inventory(config)
    noise_contract = _read_noise_contract(config)
    beta_scale = float(args.beta_scale)
    if not math.isfinite(beta_scale) or beta_scale <= 0.0:
        raise ValueError("--beta-scale must be finite and positive.")
    for case in inventory:
        case["base_beta"] = float(case["base_beta"]) * beta_scale
        case["injected_beta"] = float(case["injected_beta"]) * beta_scale
    batches, cohort = _build_cohort(config, inventory)
    if args.validate_only:
        print(
            json.dumps(
                {
                    "state": "validated",
                    "case_count": len(inventory),
                    "batch_count": len(batches),
                    "beta_scale": beta_scale,
                    "read_noise_contract": noise_contract,
                    "lineage_proof": lineage_proof,
                    "runtime_source": runtime_source,
                    "cohort_sha256": cohort["cohort_sha256"],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return None

    if args.smoke:
        selected_cases = [
            next(
                case
                for case in inventory
                if case["architecture"] == "conv3" and case["scheme"] == "legacy"
            )
        ]
        roles = ["best_validation"]
        selected_batches = batches[:1]
    else:
        selected_cases = inventory
        roles = [str(value) for value in config["checkpoint_roles"]]
        selected_batches = batches

    output_root = args.output_root.expanduser().resolve()
    run_dir = output_root / str(args.run_id)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable.")
    manifest = {
        "study_id": config["study_id"],
        "run_id": str(args.run_id),
        "arm_id": "conv123_zero_bias_adam_centered_float64_eqprop_bptt_gradient_gate",
        "evidence_class": config["evidence_class"],
        "dataset": config["dataset"]["name"],
        "smoke": bool(args.smoke),
        "command": [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]],
        "configuration": {
            "path": str(config_path),
            "sha256": base.sha256_file(config_path),
            "semantic_sha256": _semantic_sha256(config),
            "resolved": config,
            "execution_overrides": {"beta_scale": beta_scale},
        },
        "runtime": {
            **base.runtime_context(target=args.target),
            "device": str(device),
            "hostname": socket.gethostname(),
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "runtime_source": runtime_source,
        },
        "replay": {
            "case_count": len(selected_cases),
            "checkpoint_roles": roles,
            "batch_count": len(selected_batches),
            "expected_replay_count": len(selected_cases) * len(roles) * len(selected_batches),
            "beta_scale": beta_scale,
            "runtime_dtype": "torch.float64",
            "eqprop_variant": "centered",
            "nudging_mode": "current",
            "frozen_force": True,
            "optimizer_constructed": False,
            "optimizer_steps_applied": False,
            "official_test_read": False,
            "biases_fixed_exact_zero": True,
            "bias_gradients_excluded": True,
            "read_noise_contract": noise_contract,
            "lineage_proof": lineage_proof,
        },
        "analyzer": {
            "path": str(Path(__file__).resolve()),
            "sha256": base.sha256_file(Path(__file__).resolve()),
            "shadow_runner_path": str(SHADOW_RUNNER),
            "shadow_runner_sha256": base.sha256_file(SHADOW_RUNNER),
            "repository_head": _git(REPOSITORY_ROOT, "rev-parse", "HEAD"),
            "repository_dirty": bool(_git(REPOSITORY_ROOT, "status", "--porcelain")),
        },
    }
    base.start_run(run_dir, manifest)
    base._write_json(run_dir / "config.resolved.json", config)
    base._write_json(
        run_dir / "execution.resolved.json",
        {
            "beta_scale": beta_scale,
            "read_noise_contract": noise_contract,
            "selected_cases": [
                {
                    "architecture": case["architecture"],
                    "scheme": case["scheme"],
                    "T": int(case["T"]),
                    "K": int(case["K"]),
                    "base_beta": float(case["base_beta"]),
                    "injected_beta": float(case["injected_beta"]),
                }
                for case in selected_cases
            ],
        },
    )
    base._write_json(run_dir / "cohort.json", cohort)
    base._write_json(
        run_dir / "source_inventory.json",
        {
            **source_proof,
            "runtime_source": runtime_source,
            "lineage_proof": lineage_proof,
            "cases": [
                {
                    key: str(value) if isinstance(value, Path) else value
                    for key, value in case.items()
                    if key != "source_config"
                }
                for case in inventory
            ],
        },
    )

    layer_rows: list[dict[str, Any]] = []
    residual_rows: list[dict[str, Any]] = []
    displacement_rows: list[dict[str, Any]] = []
    phase_rows: list[dict[str, Any]] = []
    parameter_guards: list[dict[str, Any]] = []
    noise_rows: list[dict[str, Any]] = []
    noise_guards: list[dict[str, Any]] = []
    dtype_proofs: list[dict[str, Any]] = []
    bias_proofs: list[dict[str, Any]] = []
    replay_count = 0
    expected_replays = len(selected_cases) * len(roles) * len(selected_batches)
    try:
        for case in selected_cases:
            for role in roles:
                bias_proofs.append(
                    _runtime_bias_proof(case, role=role, device=torch.device("cpu"))
                )
                selected = _selected(case, role)
                for batch in selected_batches:
                    output = shadow._run_precision(
                        case=case,
                        selected=selected,
                        batch=batch,
                        precision="float64",
                        device=device,
                        residual_threshold=float(
                            config["gradient_contract"][
                                "equilibrium_residual_p90_threshold"
                            ]
                        ),
                        amplification_exponent=int(case["output_row_exponent_L"]),
                        nudging_mode="current",
                        eqprop_variant="centered",
                        capture_endpoint_states=bool(
                            noise_contract["endpoint_read_noise_std"] > 0.0
                        ),
                    )
                    (
                        output["scored_gradients"],
                        output["scored_numerators"],
                        current_noise_rows,
                        current_noise_guard,
                    ) = _apply_endpoint_read_noise(
                        output=output,
                        case=case,
                        batch=batch,
                        noise_contract=noise_contract,
                        device=device,
                    )
                    output["noise_contract"] = noise_contract
                    noise_rows.extend(current_noise_rows)
                    noise_guards.append(
                        {
                            "architecture": case["architecture"],
                            "scheme": case["scheme"],
                            "checkpoint_role": role,
                            "batch_index": int(batch["batch_index"]),
                            **current_noise_guard,
                        }
                    )
                    residual_passed = bool(
                        output["residual_rows"]
                        and all(bool(row["gate_passed"]) for row in output["residual_rows"])
                    )
                    for parameter_name in output["gradients"]:
                        layer_rows.append(
                            _layer_row(
                                config=config,
                                case=case,
                                selected=selected,
                                batch=batch,
                                parameter_name=parameter_name,
                                output=output,
                                residual_passed=residual_passed,
                            )
                        )
                    residual_rows.extend(output["residual_rows"])
                    displacement_rows.extend(output["displacement_rows"])
                    phase_rows.extend(output["phase_rows"])
                    parameter_guards.append(output["guard"])
                    dtype_proofs.append(
                        {
                            "architecture": case["architecture"],
                            "scheme": case["scheme"],
                            "checkpoint_role": role,
                            "batch_index": int(batch["batch_index"]),
                            **output["dtype_proof"],
                            "gradient_arithmetic": output["gradient_dtype_rows"],
                        }
                    )
                    replay_count += 1
                    current_rows = [
                        row
                        for row in layer_rows
                        if row["architecture"] == case["architecture"]
                        and row["scheme"] == case["scheme"]
                        and row["checkpoint_role"] == role
                        and int(row["batch_index"]) == int(batch["batch_index"])
                    ]
                    base.append_metric(
                        run_dir / "metrics.jsonl",
                        {
                            "stage": "gradient_replay_complete",
                            "split": "validation",
                            "architecture": case["architecture"],
                            "scheme": case["scheme"],
                            "checkpoint_role": role,
                            "batch_index": int(batch["batch_index"]),
                            "T": int(case["T"]),
                            "K": int(case["K"]),
                            "base_beta": float(case["base_beta"]),
                            "injected_beta": float(case["injected_beta"]),
                            "endpoint_read_noise_std": float(
                                noise_contract["endpoint_read_noise_std"]
                            ),
                            "minimum_clean_eqprop_bptt_cosine": min(
                                float(row["clean_eqprop_bptt_cosine"])
                                for row in current_rows
                            ),
                            "minimum_noisy_clean_eqprop_cosine": min(
                                float(row["noisy_clean_eqprop_cosine"])
                                for row in current_rows
                            ),
                            "minimum_cosine": min(float(row["cosine"]) for row in current_rows),
                            "maximum_symmetric_norm_delta": max(
                                float(row["symmetric_norm_delta"]) for row in current_rows
                            ),
                            "gradient_fidelity_passed": all(
                                bool(row["gradient_fidelity_passed"])
                                for row in current_rows
                            ),
                            "read_noise_acquisition_passed": all(
                                bool(row["read_noise_acquisition_passed"])
                                for row in current_rows
                            ),
                            "read_noise_usable_gate_passed": all(
                                bool(row["read_noise_usable_gate_passed"])
                                for row in current_rows
                            ),
                            "all_endpoint_residual_gates_passed": residual_passed,
                        },
                    )
                    base.update_status_progress(
                        run_dir,
                        {
                            "stage": "zero_bias_eqprop_bptt_gradient_replay",
                            "completed_replays": replay_count,
                            "expected_replays": expected_replays,
                            "architecture": case["architecture"],
                            "scheme": case["scheme"],
                            "checkpoint_role": role,
                            "batch_index": int(batch["batch_index"]),
                        },
                    )

        summaries = _summaries(layer_rows)
        source_hashes_after = {}
        for case in inventory:
            key = f"{case['architecture']}/{case['scheme']}"
            source_hashes_after[key] = _source_file_hashes(
                Path(case["run_dir"]), config["source_contract"]["required_source_files"]
            )
        source_bytes_unchanged = (
            source_hashes_after == source_proof["source_file_hashes_before"]
        )
        all_parameter_guards = all(
            bool(row["parameter_tensors_unchanged_after_cast"])
            and bool(row["parameter_values_unchanged_after_cast"])
            and bool(row["all_gradient_arithmetic_target_dtype"])
            and bool(row["iteration_contract_passed"])
            and bool(row["frozen_current_force_matches_post_T_cost_gradient"])
            and bool(row["frozen_current_force_unchanged_across_phases"])
            for row in parameter_guards
        )
        all_dtype_proofs = all(
            bool(row["all_runtime_variables_target_dtype"])
            and bool(row["all_runtime_variables_target_dtype_after_reset"])
            for row in dtype_proofs
        )
        all_bias_zero = all(
            bool(row["all_bias_tensors_exact_zero"]) for row in bias_proofs
        )
        all_noise_guards = all(
            bool(row["parameter_tensors_unchanged"])
            and bool(row["input_remained_exact"])
            and (
                not bool(row["enabled"])
                or (
                    bool(row["all_noisy_gradients_float64"])
                    and bool(row["all_noisy_gradients_finite"])
                )
            )
            for row in noise_guards
        )
        matched_noise_draw_guard = _matched_noise_draw_guard(noise_rows)
        expected_layer_count = sum(
            int(case["scored_interaction_count_L"])
            for case in selected_cases
            for _role in roles
            for _batch in selected_batches
        )
        if not args.smoke:
            if expected_replays != int(config["completion"]["production_replay_count"]):
                raise RuntimeError("Configured production replay count is inconsistent.")
            if expected_layer_count != int(
                config["completion"]["production_layer_comparison_count"]
            ):
                raise RuntimeError("Configured production layer count is inconsistent.")
            expected_noise_draws = int(
                config["completion"].get("production_noise_draw_count", 0)
            )
            if len(noise_rows) != expected_noise_draws:
                raise RuntimeError(
                    "Configured production read-noise draw count is inconsistent: "
                    f"{len(noise_rows)} != {expected_noise_draws}."
                )
        completion = {
            "criteria_met": True,
            "completed_all_expected_replays": replay_count == expected_replays,
            "completed_all_expected_layer_comparisons": len(layer_rows)
            == expected_layer_count,
            "source_bundles_validated": len(inventory) == len(config["cases"]),
            "source_bytes_unchanged": source_bytes_unchanged,
            "all_bias_learning_rates_exact_zero": all(
                bool(row["all_bias_learning_rates_exact_zero"])
                for row in source_proof["bias_learning_rate_proofs"]
            ),
            "all_bias_tensors_exact_zero": all_bias_zero,
            "all_parameter_and_frozen_force_guards_passed": all_parameter_guards,
            "all_float64_dtype_proofs_passed": all_dtype_proofs,
            "all_endpoint_read_noise_guards_passed": all_noise_guards,
            "matched_endpoint_read_noise_draws_across_schemes": bool(
                matched_noise_draw_guard[
                    "matched_standard_normal_draws_across_schemes"
                ]
            ),
            "completed_expected_endpoint_read_noise_draws": bool(
                args.smoke
                or len(noise_rows)
                == int(config["completion"].get("production_noise_draw_count", 0))
            ),
            "fixed_cohort_reproduced": True,
            "optimizer_constructed": False,
            "optimizer_steps_applied": False,
            "official_test_read": False,
        }
        completion["criteria_met"] = bool(
            all(
                bool(value)
                for key, value in completion.items()
                if key
                not in {
                    "criteria_met",
                    "optimizer_constructed",
                    "optimizer_steps_applied",
                    "official_test_read",
                }
            )
            and completion["optimizer_constructed"] is False
            and completion["optimizer_steps_applied"] is False
            and completion["official_test_read"] is False
        )
        if not completion["criteria_met"]:
            raise RuntimeError(f"Operational completion criteria failed: {completion}")

        base._write_csv(run_dir / "layer_metrics.csv", layer_rows)
        base._write_csv(run_dir / "case_summary.csv", summaries)
        base._write_csv(run_dir / "equilibrium_residuals.csv", residual_rows)
        base._write_csv(run_dir / "state_displacement.csv", displacement_rows)
        base._write_csv(run_dir / "phase_diagnostics.csv", phase_rows)
        if noise_rows:
            base._write_csv(run_dir / "endpoint_read_noise.csv", noise_rows)
        base._write_json(
            run_dir / "read_only_guards.json",
            {
                "source_file_hashes_before": source_proof["source_file_hashes_before"],
                "source_file_hashes_after": source_hashes_after,
                "source_bytes_unchanged": source_bytes_unchanged,
                "bias_learning_rate_proofs": source_proof["bias_learning_rate_proofs"],
                "bias_tensor_proofs": bias_proofs,
                "parameter_guards": parameter_guards,
                "dtype_proofs": dtype_proofs,
                "endpoint_read_noise_contract": noise_contract,
                "endpoint_read_noise_guards": noise_guards,
                "matched_endpoint_read_noise_draw_guard": matched_noise_draw_guard,
                "optimizer_constructed": False,
                "optimizer_steps_applied": False,
                "official_test_read": False,
            },
        )
        _write_report(
            run_dir / "report.md", config=config, summaries=summaries, smoke=bool(args.smoke)
        )
        gradient_all = all(bool(row["gradient_fidelity_passed"]) for row in summaries)
        acquisition_all = all(
            bool(row["read_noise_acquisition_passed"]) for row in summaries
        )
        usable_all = all(
            bool(row["read_noise_usable_gate_passed"]) for row in summaries
        )
        residual_all = all(
            bool(row["all_endpoint_residual_gates_passed"]) for row in summaries
        )
        launch_all = all(bool(row["unqualified_launch_gate_passed"]) for row in summaries)
        terminal_metrics = {
            "beta_scale": beta_scale,
            "replay_count": replay_count,
            "layer_comparison_count": len(layer_rows),
            "summary_count": len(summaries),
            "endpoint_read_noise_std": float(
                noise_contract["endpoint_read_noise_std"]
            ),
            "endpoint_read_noise_seed": noise_contract[
                "endpoint_read_noise_seed"
            ],
            "endpoint_read_noise_draw_count": len(noise_rows),
            "minimum_clean_eqprop_bptt_cosine": min(
                float(row["clean_eqprop_bptt_cosine"]) for row in layer_rows
            ),
            "minimum_noisy_clean_eqprop_cosine": min(
                float(row["noisy_clean_eqprop_cosine"]) for row in layer_rows
            ),
            "minimum_cosine": min(float(row["cosine"]) for row in layer_rows),
            "maximum_symmetric_norm_delta": max(
                float(row["symmetric_norm_delta"]) for row in layer_rows
            ),
            "gradient_fidelity_all_passed": gradient_all,
            "read_noise_acquisition_all_passed": acquisition_all,
            "read_noise_usable_gate_all_passed": usable_all,
            "equilibrium_residual_all_passed": residual_all,
            "unqualified_launch_gate_all_passed": launch_all,
            "source_bytes_unchanged": source_bytes_unchanged,
            "all_bias_tensors_exact_zero": all_bias_zero,
            "optimizer_steps_applied": False,
            "official_test_read": False,
        }
        base.append_metric(
            run_dir / "metrics.jsonl",
            {"stage": "analysis_complete", "split": "validation", **terminal_metrics},
        )
        result = base.complete_run(
            run_dir, terminal_metrics=terminal_metrics, completion=completion
        )
        return result
    except BaseException as error:
        status_path = run_dir / "status.json"
        if status_path.is_file():
            status = _read_json(status_path)
            if status.get("state") == "running" and not (run_dir / "result.json").exists():
                base.fail_run(run_dir, error=error)
        raise


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", type=Path, default=Path("results"))
    parser.add_argument("--run-id", default="production")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--target", default="main:RTX3090")
    parser.add_argument(
        "--beta-scale",
        type=float,
        default=1.0,
        help="Multiply every predeclared base and injected beta by this factor.",
    )
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = run(args)
    if result is not None:
        run_dir = args.output_root.expanduser().resolve() / str(args.run_id)
        print(
            json.dumps(
                {
                    "state": "complete",
                    "result_sha256": base.sha256_file(run_dir / "result.json"),
                    "terminal_metrics": result["terminal_metrics"],
                },
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
