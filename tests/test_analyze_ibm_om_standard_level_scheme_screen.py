from __future__ import annotations

import copy
from hashlib import sha256
from itertools import product
import json
from pathlib import Path

import pytest
import torch

import experiments.mnist_relu_drn.analyze_ibm_om_standard_level_scheme_screen as analyzer_module
from experiments.artifacts import sha256_file
from experiments.mnist_relu_drn.analyze_ibm_om_standard_level_scheme_screen import (
    AnalysisIntegrityError,
    analyze_standard_level_screen,
    semantic_summary_sha256,
)
from experiments.mnist_relu_drn.ibm_om_standard_level_scheme_screen import (
    _commission_reset_origin_for_population,
    build_standard_level_grid,
)
from training.ibm_reram_hwa import (
    IbmReramArrayPopulation,
    _native_seed,
    _population_fingerprint,
    load_om_array_population,
    save_om_array_population,
)
from training.ibm_reram_program_verify import OM_PRESET, derive_seed


_SCHEMES = {
    "four_without_fixed_r": (4, False, 0.91),
    "four_with_fixed_r": (4, True, 0.93),
    "eight_without_fixed_r": (8, False, 0.92),
    "eight_with_fixed_r": (8, True, 0.95),
}
_SEEDS = (87001, 87002, 87003)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _digest(label: str) -> str:
    return sha256(label.encode("utf-8")).hexdigest()


def _constant_summary(value: float) -> dict[str, float]:
    return {
        "minimum": value,
        "p01": value,
        "p10": value,
        "median": value,
        "mean": value,
        "p90": value,
        "p99": value,
        "maximum": value,
        "rms": abs(value),
    }


def _population(root: Path, topology: int, seed: int) -> dict[str, object]:
    population_path = root / "populations" / f"{topology}-device-assignment-{seed}.npz"
    keys = (
        ("base.dense_weight.0", "base.dense_weight.1")
        if topology == 4
        else (
            "base.conductance_plus.0",
            "base.conductance_minus.0",
            "base.conductance_plus.1",
            "base.conductance_minus.1",
        )
    )
    shapes = tuple((1, 1) for _key in keys)
    binding_seeds = tuple(
        _native_seed(derive_seed(seed, OM_PRESET, key, "published")) for key in keys
    )
    donor_seeds = tuple(
        _native_seed(derive_seed(seed, OM_PRESET, key, "repair_donor"))
        for key in keys
    )
    size = len(keys)
    tensors = {
        "max_bound": torch.ones(size, dtype=torch.float32),
        "min_bound": -torch.ones(size, dtype=torch.float32),
        "dwmin_up": torch.full((size,), 0.1, dtype=torch.float32),
        "dwmin_down": torch.full((size,), 0.1, dtype=torch.float32),
        "reference": torch.zeros(size, dtype=torch.float32),
        "corrupt": torch.zeros(size, dtype=torch.bool),
        "published_corrupt": torch.tensor(
            [True, *([False] * (size - 1))], dtype=torch.bool
        ),
    }
    scalar_parameters = {
        "aihwkit_version": "1.1.0",
        "nominal_dw_min": 0.0949,
        "dw_min_std": 0.4158,
        "write_noise_std": 1.4113,
    }
    fingerprint = _population_fingerprint(
        assignment_seed=seed,
        corruption_policy="counterfactual_repaired",
        keys=keys,
        shapes=shapes,
        binding_sampling_seeds=binding_seeds,
        donor_sampling_seeds=donor_seeds,
        scalar_parameters=scalar_parameters,
        tensors=tensors,
    )
    population = IbmReramArrayPopulation(
        assignment_seed=seed,
        corruption_policy="counterfactual_repaired",
        binding_keys=keys,
        binding_shapes=shapes,
        binding_sampling_seeds=binding_seeds,
        donor_sampling_seeds=donor_seeds,
        nominal_dw_min=0.0949,
        dw_min_std=0.4158,
        write_noise_std=1.4113,
        max_bound=tensors["max_bound"],
        min_bound=tensors["min_bound"],
        dwmin_up=tensors["dwmin_up"],
        dwmin_down=tensors["dwmin_down"],
        reference=tensors["reference"],
        corrupt=tensors["corrupt"],
        published_corrupt=tensors["published_corrupt"],
        fingerprint=fingerprint,
        aihwkit_version="1.1.0",
    )
    save_om_array_population(population_path, population)
    receipt = {
        "schema": "ebl.ibm_reram.om_array_population_receipt",
        "schema_version": 1,
        "backend": "external_pinned_aihwkit_python",
        "python_executable": "/test/python",
        "python_version": "3.12",
        "torch_version": "2.test",
        "aihwkit_version": "1.1.0",
        "request": {
            "preset": "reram_array_om",
            "assignment_seed": seed,
            "corruption_policy": "counterfactual_repaired",
            "binding_keys": list(keys),
            "binding_shapes": [list(shape) for shape in shapes],
            "required_aihwkit_version": "1.1.0",
        },
        "num_cells": size,
        "population_fingerprint": fingerprint,
        "population_sha256": sha256_file(population_path),
        "sampler_source_sha256": (
            "f938908c70135abe4308e67cb9d366651c81ecef3701b29d42c416e20c02fd2e"
        ),
        "population_implementation_sha256": (
            "337ff76e0b3e3acda9ab6c1728f35141ef399ab2b8cbc2f57e9ce8d39ceb6206"
        ),
    }
    receipt_path = population_path.with_suffix(".receipt.json")
    _write_json(receipt_path, receipt)
    return {
        "topology": topology,
        "assignment_seed": seed,
        "path": str(population_path.resolve()),
        "sha256": sha256_file(population_path),
        "receipt": str(receipt_path.resolve()),
        "receipt_sha256": sha256_file(receipt_path),
        "population_fingerprint": fingerprint,
        "cells": size,
        "published_corrupt_cells_repaired": 1,
        "final_corrupt_cells": 0,
        "nominal_dw_min": 0.0949,
        "declared_dw_min_std_but_disabled": 0.4158,
        "declared_write_noise_std_but_disabled": 1.4113,
    }


def _calibration(gain: float, accuracy: float, kl: float) -> dict[str, object]:
    return {
        "gain": gain,
        "gain_source": "per_scheme_standard_level_development_fit",
        "gain_was_fit_on_this_scheme": True,
        "raw_kl": kl + 0.1,
        "calibrated_kl": kl,
        "score_rms": 0.5,
        "calibrated_score_rms": 0.7,
        "teacher_logit_rms": 1.0,
        "student_accuracy": accuracy,
        "teacher_accuracy": 0.98,
        "teacher_agreement": accuracy,
        "gain_at_grid_boundary": False,
    }


def _test_metrics(accuracy: float, gain: float, label: str) -> dict[str, object]:
    return {
        "examples": 10_000,
        "kl_teacher_student": 0.2,
        "raw_kl_teacher_student": 0.3,
        "student_accuracy": accuracy,
        "teacher_accuracy": 0.98,
        "teacher_agreement": accuracy - 0.01,
        "raw_score_rms": 0.5,
        "calibrated_score_rms": 0.7,
        "teacher_logit_rms": 1.0,
        "fixed_logit_gain": gain,
        "prediction_sha256": _digest(label),
        "voltage": [
            {
                "layer": layer,
                "values": 100,
                "mean": 0.2 + 0.01 * layer,
                "rms": 0.3 + 0.01 * layer,
                "standard_deviation": 0.1,
                "minimum": 0.0,
                "maximum": 0.8,
            }
            for layer in (0, 1, 2)
        ],
    }


def _mapping_layer(layer: int) -> dict[str, object]:
    return {
        "layer": layer,
        "standard_level_spacing": 0.2,
        "standard_level_spacing_delta_multiples": 4,
        "active_target_outside_bounds_count": 0,
        "zero_positive_capacity_quad_fraction": 0.0,
        "per_quad_common_signed_logical_level_count": {
            "p10": 3.0,
            "median": 5.0,
        },
        "active_quantization_error": {"rms": 0.04},
        "logical_contrast": {"rms": 0.25},
        "continuous_logical_contrast": {"rms": 0.26},
        "baseline_logical_contrast": {"rms": 0.01},
        "logical_sign_flip_fraction_nonzero": 0.001,
        "continuous_logical_sign_flip_fraction_nonzero": 0.0005,
        "mean_edge_denominator_loading": 1.25,
        "normalized_total_conductance_proxy": 100.0 + layer,
    }


def _mapping(scheme: str, topology: int, fixed_r: bool, pair: tuple[float, float]) -> dict[str, object]:
    target_count = topology // 2
    return {
        "scheme": scheme,
        "encoding": "single" if topology == 4 else "differential",
        "device_count_per_logical_weight": topology,
        "use_fixed_reference": fixed_r,
        "level_origin_policy": (
            "exact_sampled_reference_center_with_active_a_projection_only"
            if fixed_r
            else "sampled_min_bound_reset_origin"
        ),
        "standard_level_spacing": 0.2,
        "standard_level_spacing_delta_multiples": 4,
        "scale_fractions": list(pair),
        "layers": [_mapping_layer(0), _mapping_layer(1)],
        "hashes": {
            "standard_level_targets": [
                _digest(f"{scheme}:{pair}:standard:{index}")
                for index in range(target_count)
            ],
            "continuous_envelope_targets": [
                _digest(f"{scheme}:{pair}:continuous:{index}")
                for index in range(target_count)
            ],
        },
    }


def _fixture(tmp_path: Path) -> tuple[Path, dict[str, object]]:
    root = tmp_path / "run"
    source_root = tmp_path / "sources"
    source_root.mkdir()
    model_config = source_root / "model.json"
    teacher = source_root / "teacher.pt"
    model_config.write_text("{}\n", encoding="utf-8")
    teacher.write_bytes(b"teacher")
    scale_values = [0.125, 0.25, 0.5, 1.0]
    screen_config = {
        "schema_version": 1,
        "screen_id": "mnist-ibm-om-standard-level-scheme-screen-20260827-v1",
        "evidence_class": "model_based_aihwkit_preset",
        "preset": "reram_array_om",
        "required_aihwkit_version": "1.1.0",
        "corruption_policy": "counterfactual_repaired",
        "source": {
            "teacher_sha256": sha256_file(teacher),
            "model_config_sha256": sha256_file(model_config),
        },
        "development_assignment_seed": 86001,
        "heldout_assignment_seeds": list(_SEEDS),
        "standard_levels": {
            "delta_definition": "nominal_dw_min_in_native_a_coordinate",
            "conductance_coordinate": "clip((a+1)/2,0,1)",
            "minimum_spacing_delta_multiples": 4,
            "without_fixed_r_origin": "sampled_min_bound_reset",
            "with_fixed_r_origin": "exact_sampled_reference_r",
            "with_fixed_r_active_zero": (
                "nearest_in_bounds_integer_level_without_changing_r"
            ),
            "active_programming_direction": (
                "nonnegative_offset_with_dual_rail_sign"
            ),
            "range_policy": "whole_uniform_levels_within_sampled_active_bounds",
            "zero_positive_capacity_group": "retain_zero_level_only_and_report",
            "no_in_bounds_standard_level": (
                "retain_exact_reference_zero_only_and_report"
            ),
            "logical_rounding": "nearest_integer_half_away_from_zero",
            "cycle_to_cycle_random_term": 0.0,
            "apparent_write_noise": 0.0,
        },
        "per_scheme_calibration": {
            "scale_fractions": scale_values,
            "selection_domain": "development_assignment_after_standard_level_mapping",
            "selection_metric": "mapped_accuracy_then_calibrated_kl_then_scale_pair",
            "logit_gain": "per_scheme_positive_kl_fit",
            "heldout_application": "freeze_selected_scale_pair_and_gain_per_scheme",
        },
        "continuous_envelope_control": True,
        "sample_limit": None,
    }
    screen_config_path = source_root / "screen.json"
    _write_json(screen_config_path, screen_config)

    development: dict[str, object] = {}
    selected_by_scheme: dict[str, tuple[tuple[float, float], float]] = {}
    for scheme_index, (scheme, (topology, fixed_r, _test_accuracy)) in enumerate(
        _SCHEMES.items()
    ):
        gain = 1.5 + 0.1 * scheme_index
        candidates = []
        for pair in product(scale_values, repeat=2):
            selected_pair = pair == (0.5, 0.5)
            accuracy = 0.96 if selected_pair else 0.80
            kl = 0.05 if selected_pair else 0.2
            candidates.append(
                {
                    "scale_fractions": list(pair),
                    "calibration": _calibration(gain, accuracy, kl),
                    "target_hashes": _mapping(
                        scheme, topology, fixed_r, (float(pair[0]), float(pair[1]))
                    )["hashes"],
                }
            )
        selected_index = next(
            index
            for index, candidate in enumerate(candidates)
            if candidate["scale_fractions"] == [0.5, 0.5]
        )
        selected = candidates[selected_index]
        pair = (0.5, 0.5)
        selected_by_scheme[scheme] = (pair, gain)
        development[scheme] = {
            "scheme": scheme,
            "topology": topology,
            "selection_domain": "development_assignment_after_standard_level_mapping",
            "selection_metric": "mapped_accuracy_then_calibrated_kl_then_scale_pair",
            "per_scheme_refit_performed": True,
            "selected_index": selected_index,
            "selected": selected,
            "continuous_envelope_calibration_at_selected_gain": {
                "gain": gain,
                "student_accuracy": 0.97,
            },
            "mapping": _mapping(scheme, topology, fixed_r, pair),
            "candidates": candidates,
        }

    development_populations = [
        _population(root, topology, 86001) for topology in (4, 8)
    ]
    heldout = []
    for seed in _SEEDS:
        for topology in (4, 8):
            arms = []
            for scheme, (scheme_topology, fixed_r, test_accuracy) in _SCHEMES.items():
                if scheme_topology != topology:
                    continue
                pair, gain = selected_by_scheme[scheme]
                arms.append(
                    {
                        "scheme": scheme,
                        "assignment_seed": seed,
                        "scale_fractions": list(pair),
                        "fixed_logit_gain": gain,
                        "calibration_source": "frozen_per_scheme_development_assignment",
                        "per_scheme_refit_performed": True,
                        "mapping": _mapping(scheme, topology, fixed_r, pair),
                        "standard_level_test": _test_metrics(
                            test_accuracy, gain, f"{scheme}:{seed}:standard"
                        ),
                        "continuous_envelope_test": _test_metrics(
                            test_accuracy + 0.01, gain, f"{scheme}:{seed}:continuous"
                        ),
                        "continuous_to_standard_prediction_flip_count": 100,
                        "continuous_to_standard_prediction_flip_fraction": 0.01,
                    }
                )
            heldout.append(
                {
                    "assignment_seed": seed,
                    "topology": topology,
                    "population": _population(root, topology, seed),
                    "arms": arms,
                }
            )

    aggregate = {}
    for scheme, (_topology, _fixed_r, accuracy) in _SCHEMES.items():
        aggregate[scheme] = {
            "assignments": list(_SEEDS),
            "standard_level_accuracy": _constant_summary(accuracy),
            "continuous_envelope_accuracy": _constant_summary(accuracy + 0.01),
            "standard_minus_continuous_accuracy": _constant_summary(-0.01),
            "passes_90_percent_mean_gate": accuracy >= 0.9,
        }
    summary = {
        "schema": "ebl.mnist_relu_drn.ibm_om_standard_level_scheme_screen",
        "schema_version": 1,
        "status": "identity_aware_ideal_standard_level_screen_complete",
        "screen_id": screen_config["screen_id"],
        "claim_boundary": (
            "AIHWKit 1.1.0 normalized OM fitted-model control with repaired "
            "identities and analyst-standard uniform levels separated by four "
            "nominal increments. No stochastic write, program-and-verify, HWA, "
            "training, absolute conductance calibration, or fabricated-device claim."
        ),
        "source_precision": "fp32_teacher_and_drn_solver",
        "deployment_precision": "uniform_four_delta_standard_level_grid",
        "screen_config": str(screen_config_path.resolve()),
        "screen_config_sha256": sha256_file(screen_config_path),
        "model_config": str(model_config.resolve()),
        "model_config_sha256": sha256_file(model_config),
        "teacher_weights": str(teacher.resolve()),
        "teacher_weights_sha256": sha256_file(teacher),
        "teacher_architecture": "bias_free_relu_784_50_10",
        "device": "cpu",
        "sample_limit": None,
        "development_assignment_seed": 86001,
        "heldout_assignment_seeds": list(_SEEDS),
        "calibration_policy": {
            "source": "per_scheme_development_assignment",
            "selection_domain": "development_assignment_after_standard_level_mapping",
            "selection_metric": "mapped_accuracy_then_calibrated_kl_then_scale_pair",
            "heldout_application": "freeze_selected_scale_pair_and_gain_per_scheme",
        },
        "scheme_contract": {
            "without_fixed_r_origin": "sampled RESET/lower state",
            "with_fixed_r_origin": "exact intrinsic sampled r",
            "active_level_spacing": "4 * nominal dw_min/2 in x=(a+1)/2",
            "four_fixed_r_positive": "G++=G--=a; G+-=G-+=r",
            "four_fixed_r_negative": "G++=G--=r; G+-=G-+=a",
            "four_edge_transfer_and_loading": "D=G; S=G",
            "eight_edge_transfer": "D=G_a-G_r",
            "eight_edge_loading": "S=G_a+G_r",
        },
        "development_populations": development_populations,
        "development_calibration": development,
        "heldout": heldout,
        "aggregate": aggregate,
        "matched_comparisons": {
            "fixed_r_minus_no_r": {
                "four_devices": 0.93 - 0.91,
                "eight_devices": 0.95 - 0.92,
            },
            "eight_minus_four": {
                "without_fixed_r": 0.92 - 0.91,
                "with_fixed_r": 0.95 - 0.93,
            },
        },
    }
    summary_path = root / "analysis" / "summary.json"
    _write_json(summary_path, summary)
    screen_contract = {
        "screen_config": summary["screen_config"],
        "screen_config_sha256": summary["screen_config_sha256"],
        "model_config": summary["model_config"],
        "model_config_sha256": summary["model_config_sha256"],
        "teacher_weights": summary["teacher_weights"],
        "teacher_weights_sha256": summary["teacher_weights_sha256"],
        "aihwkit_python": "/test/aihwkit/python",
        "device": summary["device"],
        "sample_limit": None,
        "contract": screen_config,
    }
    _write_json(root / "screen_contract.json", screen_contract)
    _write_json(
        root / "launch_contract.json",
        {
            "schema_version": 1,
            "evidence_tier": "exploratory_noncanonical",
            "screen_id": screen_config["screen_id"],
            "started_at_utc": "2026-08-27T00:00:00Z",
            "source_commit": "a" * 40,
            "launcher": {"type": "local_tmux", "handle": "test-main"},
            "command": "test main command",
            "expected_coverage": {
                "development_assignments": [86001],
                "heldout_assignments": list(_SEEDS),
                "schemes": list(_SCHEMES),
                "development_calibrations": 4,
                "heldout_arm_reports": 12,
                "population_receipts": 8,
            },
            "paths": {
                "result_root": str(root.resolve()),
                "log": str((root / "run.log").resolve()),
                "runtime_contract": str((root / "screen_contract.json").resolve()),
                "terminal_result": str(summary_path.resolve()),
            },
            "progress_contract": "test",
            "expected_runtime": "test",
            "safe_retry": "test",
        },
    )
    return summary_path, summary


def _fixture_v2(tmp_path: Path) -> tuple[Path, dict[str, object]]:
    summary_path, summary = _fixture(tmp_path)
    root = summary_path.parent.parent
    screen_config_path = Path(summary["screen_config"])
    screen_config = json.loads(screen_config_path.read_text(encoding="utf-8"))
    screen_config["schema_version"] = 2
    screen_config["screen_id"] = (
        "mnist-ibm-om-standard-level-reset-mean-scheme-screen-20260827-v2"
    )
    levels = screen_config["standard_levels"]
    levels["without_fixed_r_origin"] = (
        "bounded_per_cell_mean_of_repeated_apparent_raw_a_reset_reads"
    )
    levels["deployment_cycle_to_cycle_random_term"] = levels.pop(
        "cycle_to_cycle_random_term"
    )
    levels["deployment_apparent_write_noise"] = levels.pop(
        "apparent_write_noise"
    )
    screen_config["reset_baseline_commissioning"] = {
        "applies_to": "without_fixed_r_only",
        "samples_per_cell": 8,
        "sample_sequence": (
            "initialize_at_sampled_lower_bound_then_one_reset_pulse_and_"
            "apparent_read_per_sample"
        ),
        "read_coordinate": "raw_active_a_before_conductance_mapping",
        "baseline_estimator": "per_cell_arithmetic_mean",
        "cross_cell_pooling": "none",
        "standard_error_guard": 0.0,
        "bound_policy": "clip_mapped_mean_to_sampled_active_conductance_bounds",
        "commissioning_noise": (
            "preset_cycle_to_cycle_and_apparent_write_noise_enabled"
        ),
        "seed_derivation": (
            "derive_seed(assignment_seed,per_cell_raw_a_reset_commissioning_"
            "v2,population_fingerprint,samples_per_cell)"
        ),
        "freeze_policy": (
            "one_baseline_per_topology_assignment_reused_for_every_scale_candidate"
        ),
    }
    _write_json(screen_config_path, screen_config)

    summary["screen_id"] = screen_config["screen_id"]
    summary["screen_config_sha256"] = sha256_file(screen_config_path)
    summary["claim_boundary"] = (
        "AIHWKit 1.1.0 normalized OM fitted-model control with repaired "
        "identities, per-cell raw-a baselines commissioned from eight "
        "stochastic RESET/read observations in the no-r arms, and analyst-"
        "standard uniform levels separated by four nominal increments. Level "
        "deployment itself is ideal and noiseless. No program-and-verify, HWA, "
        "training, absolute conductance calibration, or fabricated-device claim."
    )
    summary["scheme_contract"] = {
        **summary["scheme_contract"],
        "without_fixed_r_origin": (
            "bounded per-cell arithmetic mean of eight sequential apparent "
            "raw-a RESET/read observations"
        ),
        "without_fixed_r_cross_cell_pooling": "none",
        "without_fixed_r_standard_error_guard": 0.0,
    }

    populations = list(summary["development_populations"])
    populations.extend(block["population"] for block in summary["heldout"])
    prior_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(4)
        for population_record in populations:
            population = load_om_array_population(Path(population_record["path"]))
            topology = int(population_record["topology"])
            commissioning, commissioning_report = (
                _commission_reset_origin_for_population(
                    population,
                    topology=topology,
                    read_samples=8,
                    output_dir=root,
                )
            )
            population_record["reset_baseline_commissioning"] = commissioning_report
            population_record["declared_dw_min_std"] = population_record.pop(
                "declared_dw_min_std_but_disabled"
            )
            population_record["declared_write_noise_std"] = population_record.pop(
                "declared_write_noise_std_but_disabled"
            )
            population_record["noise_application"] = {
                "reset_commissioning": (
                    "preset cycle-to-cycle and apparent write noise enabled"
                ),
                "ideal_level_deployment": "both disabled",
            }
            prefix = "four" if topology == 4 else "eight"
            population_record["standard_level_grids"] = {
                f"{prefix}_without_fixed_r": build_standard_level_grid(
                    population,
                    use_fixed_reference=False,
                    no_r_reset_mean_raw_a=commissioning.reset_mean_raw_a,
                    spacing_delta_multiples=4,
                ).report,
                f"{prefix}_with_fixed_r": build_standard_level_grid(
                    population,
                    use_fixed_reference=True,
                    spacing_delta_multiples=4,
                ).report,
            }
    finally:
        torch.set_num_threads(prior_threads)

    for scheme, report in summary["development_calibration"].items():
        if not _SCHEMES[scheme][1]:
            report["mapping"]["level_origin_policy"] = (
                "bounded_per_cell_mean_apparent_raw_a_reset_origin"
            )
    for block in summary["heldout"]:
        for arm in block["arms"]:
            if not _SCHEMES[arm["scheme"]][1]:
                arm["mapping"]["level_origin_policy"] = (
                    "bounded_per_cell_mean_apparent_raw_a_reset_origin"
                )
    _write_json(summary_path, summary)

    runtime_contract_path = root / "screen_contract.json"
    runtime_contract = json.loads(runtime_contract_path.read_text(encoding="utf-8"))
    runtime_contract["screen_config_sha256"] = summary["screen_config_sha256"]
    runtime_contract["contract"] = screen_config
    _write_json(runtime_contract_path, runtime_contract)
    (root / "run.log").write_text("fixture complete\n", encoding="utf-8")
    (root / "exit_code.txt").write_text("0\n", encoding="utf-8")
    _write_json(
        root / "launch_contract.json",
        _v2_launch_contract(root, summary_path, summary["screen_config_sha256"], replay=False),
    )
    return summary_path, summary


def _v2_launch_contract(
    root: Path,
    summary_path: Path,
    screen_config_sha256: str,
    *,
    replay: bool,
) -> dict[str, object]:
    value: dict[str, object] = {
        "schema_version": 1,
        "evidence_tier": "exploratory_noncanonical",
        "screen_id": (
            "mnist-ibm-om-standard-level-reset-mean-scheme-screen-20260827-v2"
        ),
        "started_at_utc": "2026-08-27T00:00:00Z",
        "source_commit": "a" * 40,
        "source_context": {
            "numerical_source_frozen_in_commit": True,
            "screen_config_sha256": screen_config_sha256,
            "working_tree_exception": (
                "Only the independent untracked post-run analyzer and its tests "
                "were under review at launch; neither is imported by the numerical screen."
            ),
        },
        "launcher": {
            "type": "local_tmux",
            "handle": "test-v2-replay" if replay else "test-v2-main",
        },
        "command": "test v2 command",
        "expected_coverage": {
            "development_assignments": [86001],
            "heldout_assignments": list(_SEEDS),
            "schemes": list(_SCHEMES),
            "development_calibrations": 4,
            "heldout_arm_reports": 12,
            "population_receipts": 8,
            "reset_commissioning_receipts": 8,
            "test_examples_per_arm": 10_000,
        },
        "population_reuse": (
            "Byte-identical v1 assignment populations and production sampling "
            "receipts copied before launch so v1-to-v2 changes only the declared "
            "no-r commissioning intervention."
        ),
        "paths": {
            "result_root": str(root.resolve()),
            "log": str((root / "run.log").resolve()),
            "exit_code": str((root / "exit_code.txt").resolve()),
            "runtime_contract": str((root / "screen_contract.json").resolve()),
            "terminal_result": str(summary_path.resolve()),
        },
        "progress_contract": "test",
        "expected_runtime": "test",
        "safe_retry": "test",
    }
    if replay:
        value["replay_success_criterion"] = (
            "Semantic equivalence of every scientific field, population/commissioning "
            "content hash, target hash, calibration, prediction hash, metric, and "
            "diagnostic after normalizing only result-root-dependent artifact path "
            "strings. Raw summary byte identity is not expected because paths are "
            "serialized."
        )
    return value


def _write_replay_launch_contract(
    replay_root: Path, source_summary: Path, replay_summary: Path
) -> None:
    _write_json(
        replay_root / "launch_contract.json",
        {
            "schema_version": 1,
            "evidence_tier": "exploratory_noncanonical_deterministic_replay",
            "screen_id": "mnist-ibm-om-standard-level-scheme-screen-20260827-v1",
            "started_at_utc": "2026-08-27T00:01:00Z",
            "source_commit": "a" * 40,
            "launcher": {"type": "local_tmux", "handle": "test-replay"},
            "command": "test replay command",
            "expected_coverage": {
                "development_calibrations": 4,
                "heldout_arm_reports": 12,
                "population_receipts": 8,
            },
            "reference_summary": str(source_summary.resolve()),
            "terminal_result": str(replay_summary.resolve()),
            "success_criterion": (
                "Terminal exit zero, complete declared coverage, and byte-identical "
                "summary SHA-256 to the reference execution."
            ),
        },
    )


def _relocate_population_metadata(
    summary: dict[str, object], replay_root: Path
) -> None:
    populations = list(summary["development_populations"])
    populations.extend(block["population"] for block in summary["heldout"])
    for index, population in enumerate(populations):
        source_population = Path(population["path"])
        source_receipt = Path(population["receipt"])
        replay_population = replay_root / "populations" / f"population-{index}.npz"
        replay_receipt = replay_root / "populations" / f"population-{index}.receipt.json"
        replay_population.parent.mkdir(parents=True, exist_ok=True)
        replay_population.write_bytes(source_population.read_bytes())
        receipt = json.loads(source_receipt.read_text(encoding="utf-8"))
        _write_json(replay_receipt, receipt)
        population["path"] = str(replay_population.resolve())
        population["receipt"] = str(replay_receipt.resolve())
        population["receipt_sha256"] = sha256_file(replay_receipt)
        commissioning = population.get("reset_baseline_commissioning")
        if commissioning is not None:
            source_artifact = Path(commissioning["path"])
            source_commissioning_receipt = Path(commissioning["receipt"])
            replay_artifact = replay_root / "commissioning" / source_artifact.name
            replay_commissioning_receipt = (
                replay_root / "commissioning" / source_commissioning_receipt.name
            )
            replay_artifact.parent.mkdir(parents=True, exist_ok=True)
            replay_artifact.write_bytes(source_artifact.read_bytes())
            replay_commissioning_receipt.write_bytes(
                source_commissioning_receipt.read_bytes()
            )
            commissioning["path"] = str(replay_artifact.resolve())
            commissioning["receipt"] = str(replay_commissioning_receipt.resolve())
            commissioning["receipt_sha256"] = sha256_file(
                replay_commissioning_receipt
            )


def test_analyzer_verifies_and_emits_without_mutating_source(tmp_path: Path) -> None:
    summary_path, summary = _fixture(tmp_path)
    original_sha = sha256_file(summary_path)
    replay = copy.deepcopy(summary)
    replay_root = tmp_path / "replay"
    _relocate_population_metadata(replay, replay_root)
    replay_path = replay_root / "analysis" / "summary.json"
    _write_json(replay_path, replay)
    source_contract = summary_path.parent.parent / "screen_contract.json"
    (replay_root / "screen_contract.json").write_bytes(source_contract.read_bytes())
    _write_replay_launch_contract(replay_root, summary_path, replay_path)
    assert semantic_summary_sha256(summary) == semantic_summary_sha256(replay)

    output_dir = tmp_path / "post-run"
    analysis = analyze_standard_level_screen(
        summary_path,
        output_dir,
        compare_summary_path=replay_path,
    )

    assert sha256_file(summary_path) == original_sha
    assert analysis["integrity"]["heldout_arms_verified"] == 12
    assert analysis["integrity"]["population_receipt_pairs_verified"] == 8
    assert analysis["semantic_comparison"]["scientific_fields_match"] is True
    assert analysis["semantic_comparison"]["provenance_verified"] is True
    assert analysis["semantic_comparison"]["population_receipt_pairs_verified"] == 8
    assert analysis["semantic_comparison"]["declared_byte_identity_satisfied"] is False
    assert analysis["semantic_comparison"]["declared_replay_success"] is False
    assert (
        analysis["status"]
        == "complete_semantic_replay_match_declared_byte_identity_failed"
    )
    assert analysis["accuracy_aggregate"]["four_with_fixed_r"][
        "standard_level_accuracy"
    ]["mean"] == pytest.approx(0.93)
    for name in (
        "post_run_analysis.json",
        "accuracy_by_assignment.csv",
        "mechanism_by_scheme_layer.csv",
        "post_run_analysis.md",
    ):
        assert (output_dir / name).is_file()
    assert len((output_dir / "accuracy_by_assignment.csv").read_text().splitlines()) == 13
    assert len((output_dir / "mechanism_by_scheme_layer.csv").read_text().splitlines()) == 25
    report = (output_dir / "post_run_analysis.md").read_text(encoding="utf-8")
    assert "difference_rms_over_mean_loading" in report
    assert "predeclared byte-identical-summary criterion failed" in report
    assert "Development calibration accuracy uses 1,024 training examples" in report
    assert "not means of per-assignment ratios" in report
    assert analysis["mechanism_aggregate"][0]["ratio_aggregation"] == (
        "ratio_of_assignment_mean_components_not_mean_of_assignment_ratios"
    )


def test_analyzer_rejects_non_recomputed_aggregate(tmp_path: Path) -> None:
    summary_path, summary = _fixture(tmp_path)
    summary["aggregate"]["four_without_fixed_r"]["standard_level_accuracy"][
        "mean"
    ] = 0.999
    _write_json(summary_path, summary)

    with pytest.raises(AnalysisIntegrityError, match="aggregate.*mean"):
        analyze_standard_level_screen(summary_path, tmp_path / "post-run")


def test_analyzer_no_replay_uses_internal_consistency_status(tmp_path: Path) -> None:
    summary_path, _summary = _fixture(tmp_path)
    output = tmp_path / "post-run"

    analysis = analyze_standard_level_screen(summary_path, output)

    assert analysis["status"] == "complete_internal_consistency_post_run_analysis"
    assert analysis["semantic_comparison"] is None
    assert analysis["integrity"]["independent_replay_supplied"] is False
    report = (output / "post_run_analysis.md").read_text(encoding="utf-8")
    assert "No independent replay was supplied" in report


def test_analyzer_rejects_tampered_receipt_layout(tmp_path: Path) -> None:
    summary_path, summary = _fixture(tmp_path)
    population = summary["development_populations"][0]
    receipt_path = Path(population["receipt"])
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["request"]["binding_shapes"] = [[999, 999], [1, 1]]
    _write_json(receipt_path, receipt)
    population["receipt_sha256"] = sha256_file(receipt_path)
    _write_json(summary_path, summary)

    with pytest.raises(AnalysisIntegrityError, match="frozen OM contract"):
        analyze_standard_level_screen(summary_path, tmp_path / "post-run")


def test_analyzer_rejects_selected_mapping_hash_mismatch(tmp_path: Path) -> None:
    summary_path, summary = _fixture(tmp_path)
    report = summary["development_calibration"]["four_without_fixed_r"]
    report["mapping"]["hashes"]["standard_level_targets"][0] = _digest("tampered")
    _write_json(summary_path, summary)

    with pytest.raises(AnalysisIntegrityError, match="candidate target hashes"):
        analyze_standard_level_screen(summary_path, tmp_path / "post-run")


def test_analyzer_rejects_changed_scheme_contract(tmp_path: Path) -> None:
    summary_path, summary = _fixture(tmp_path)
    summary["scheme_contract"]["eight_edge_loading"] = "S=G_a-G_r"
    _write_json(summary_path, summary)

    with pytest.raises(AnalysisIntegrityError, match="complete frozen source contract"):
        analyze_standard_level_screen(summary_path, tmp_path / "post-run")


def test_receipt_digest_remains_semantic(tmp_path: Path) -> None:
    summary_path, summary = _fixture(tmp_path)
    replay = copy.deepcopy(summary)
    replay_root = tmp_path / "replay"
    _relocate_population_metadata(replay, replay_root)
    assert semantic_summary_sha256(summary) == semantic_summary_sha256(replay)
    receipt_path = Path(replay["development_populations"][0]["receipt"])
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["python_executable"] = "/different/interpreter"
    _write_json(receipt_path, receipt)
    replay["development_populations"][0]["receipt_sha256"] = sha256_file(receipt_path)

    assert semantic_summary_sha256(summary) != semantic_summary_sha256(replay)


def test_staging_failure_preserves_published_bundle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    summary_path, _summary = _fixture(tmp_path)
    output = tmp_path / "post-run"
    analyze_standard_level_screen(summary_path, output)
    before = {
        path.name: sha256_file(path)
        for path in output.iterdir()
        if path.is_file()
    }

    def fail_write(_path: Path, _value: str) -> None:
        raise RuntimeError("injected staged write failure")

    monkeypatch.setattr(analyzer_module, "_atomic_write_text", fail_write)
    with pytest.raises(RuntimeError, match="injected staged write failure"):
        analyze_standard_level_screen(summary_path, output)

    after = {
        path.name: sha256_file(path)
        for path in output.iterdir()
        if path.is_file()
    }
    assert after == before
    assert not list(tmp_path.glob(".post-run.staging.*"))


def test_v2_analyzer_verifies_all_commissioning_pairs_and_projection(
    tmp_path: Path,
) -> None:
    summary_path, _summary = _fixture_v2(tmp_path)
    output = tmp_path / "post-run-v2"

    analysis = analyze_standard_level_screen(summary_path, output)

    assert analysis["source_contract_schema_version"] == 2
    assert analysis["integrity"]["population_receipt_pairs_verified"] == 8
    assert analysis["integrity"]["reset_commissioning_receipt_pairs_verified"] == 8
    assert len(analysis["commissioning_by_population"]) == 8
    assert set(analysis["commissioning_aggregate"]) == {"4", "8"}
    assert all(
        0.0 <= row["projected_cell_fraction"] <= 1.0
        for row in analysis["commissioning_by_population"]
    )
    report = (output / "post_run_analysis.md").read_text(encoding="utf-8")
    assert "Per-cell RESET commissioning" in report


def test_v2_analyzer_rejects_tampered_exact_commissioning_receipt(
    tmp_path: Path,
) -> None:
    summary_path, summary = _fixture_v2(tmp_path)
    commissioning = summary["development_populations"][0][
        "reset_baseline_commissioning"
    ]
    receipt_path = Path(commissioning["receipt"])
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["commissioner_report"]["coordinate"] = "reference_relative"
    _write_json(receipt_path, receipt)
    commissioning["receipt_sha256"] = sha256_file(receipt_path)
    _write_json(summary_path, summary)

    with pytest.raises(AnalysisIntegrityError, match="exact deterministic RESET"):
        analyze_standard_level_screen(summary_path, tmp_path / "post-run-v2")


def test_v2_analyzer_rejects_tampered_reset_mean_grid_hash(
    tmp_path: Path,
) -> None:
    summary_path, summary = _fixture_v2(tmp_path)
    grid = summary["development_populations"][0]["standard_level_grids"][
        "four_without_fixed_r"
    ]
    grid["reset_mean_commissioning"]["hashes"][
        "apparent_raw_a_mean"
    ] = _digest("tampered reset mean")
    _write_json(summary_path, summary)

    with pytest.raises(AnalysisIntegrityError, match="link exactly"):
        analyze_standard_level_screen(summary_path, tmp_path / "post-run-v2")


def test_v2_analyzer_rejects_legacy_population_noise_field(
    tmp_path: Path,
) -> None:
    summary_path, summary = _fixture_v2(tmp_path)
    population = summary["development_populations"][0]
    population["declared_dw_min_std_but_disabled"] = population.pop(
        "declared_dw_min_std"
    )
    _write_json(summary_path, summary)

    with pytest.raises(AnalysisIntegrityError, match="renamed v2 population noise"):
        analyze_standard_level_screen(summary_path, tmp_path / "post-run-v2")
