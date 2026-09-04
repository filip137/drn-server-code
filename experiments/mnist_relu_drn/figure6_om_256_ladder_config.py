"""Fail-closed protocol for the 784-256-10 Figure-6/IBM-OM ladder."""

from __future__ import annotations

import json
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping


EXPERIMENT_ID = "mnist_figure6_om_784_256_10_postpv_fault.v1"
SCHEMA_VERSION = 1


EXPECTED_CONFIG: Mapping[str, Any] = {
    "schema_version": SCHEMA_VERSION,
    "experiment_id": EXPERIMENT_ID,
    "evidence_tier": "exploratory_noncanonical",
    "teacher": {
        "dims": [784, 256, 10],
        "config": "examples/mnist_relu/teacher_256.json",
        "seed": 42,
        "data_seed": 42,
        "epochs": 30,
        "batch_size": 128,
        "optimizer": "adam",
        "learning_rate": 0.001,
        "weight_decay": 0.0,
        "selection": "minimum_validation_cross_entropy",
        "minimum_validation_accuracy": 0.97,
    },
    "student": {
        "physical_dims": [1568, 512, 20],
        "logical_dims": [784, 256, 10],
        "layouts": ["halves", "paired"],
        "encoding": "four_cell_dual_rail",
        "non_linearity": "perfect_diode",
        "include_biases": False,
        "solver": "asynchronous",
        "solver_iterations": 4,
        "overrelaxation_factor": 1.1,
        "input_gain": 100.0,
        "voltage_amp": 4.0,
        "current_amp": 0.25,
        "conductance_min": 0.0,
        "conductance_max": 6.0,
    },
    "gain_calibration": {
        "minimum": 0.001,
        "maximum": 1000.0,
        "steps": 121,
        "examples": 1024,
        "endpoint_seeds": [102001, 102002, 102003],
        "selection": "minimum_macro_teacher_KL_then_lower_gain",
    },
    "endpoint_model": {
        "source": "Figure6_endpoint_marginals",
        "pairing": "independent",
        "invalid_pair_policy": "conditional_redraw_both_endpoints",
        "train_seeds": [101001, 101002, 101003, 101004, 101005, 101006, 101007, 101008],
        "development_seeds": [102001, 102002, 102003],
        "adam_development_seed": 103001,
        "heldout_seeds": [104001, 104002, 104003],
    },
    "om_devices": {
        "preset": "reram_array_om",
        "required_aihwkit_version": "1.1.0",
        "published_corrupt_probability": 0.1348,
        "development_assignment_seed": 103101,
        "heldout_assignment_seeds": [104101, 104102, 104103],
        "program_verify_tolerance_progress": 0.023725,
        "program_verify_maximum_pulses": 128,
    },
    "direct": {
        "master": "teacher_weight_divided_by_layer_absolute_maximum",
    },
    "scratch": {
        "seed": 43,
        "initialization": "independent_Kaiming_then_layer_absolute_maximum_normalization",
    },
    "hwa": {
        "noise": "endpoint_variation_only_held_apparent_view",
        "training_populations": 8,
        "rotations_per_population": 8,
        "effective_bank_size": 64,
        "screen_epochs": 3,
        "main_epochs": 10,
        "batch_size": 16,
        "base_learning_rates": [0.004411914893617021, 0.000011974808510638298],
        "input_multipliers": [0.3333333333333333, 1.0, 3.0],
        "output_multipliers": [1.0, 10.0, 30.0],
        "optimizer": "digital_adam",
        "loss": "teacher_KL",
        "selection": "macro_apparent_validation_accuracy_then_KL_then_epoch_then_multipliers",
        "program_verify_noise_during_training": False,
        "fault_noise_during_training": False,
    },
    "fault": {
        "intervention": "post_pv_reset_stuck_fault",
        "timing": "after_clean_program_and_verify",
        "mask": "paired_published_OM_population",
        "persistent_progress": 0.0,
        "immutable": True,
        "initial_apparent_observation": "one_OM_write_noise_draw",
        "held_between_device_writes": True,
    },
    "adam": {
        "epochs": 10,
        "batch_size": 16,
        "learning_rate_grid": [0.000003, 0.00001, 0.00003, 0.0001, 0.0003],
        "rate_families": ["hwa", "scratch"],
        "screen_batches": 512,
        "screen_write_streams": 3,
        "development_program_verify_seed_base": 103200,
        "development_fault_observation_seed_base": 103300,
        "development_pulse_noise_seed_base": 103400,
        "development_selection_seed_base": 103500,
        "development_data_order_seed_base": 103600,
        "beta1": 0.9,
        "beta2": 0.999,
        "epsilon": 1e-8,
        "nominal_progress_step": 0.04745,
        "pulse_cap_per_cell": 640,
        "maximum_pulses_per_cell_per_minibatch": 1,
        "verify_reads_during_updates": 0,
        "loss": "teacher_KL",
        "forward_state": "held_apparent",
        "selection": "best_apparent_validation_accuracy_then_KL_then_epoch",
        "rate_selection": "macro_clean_and_corrupt_accuracy_then_KL_then_pulses_then_lower_rate",
    },
    "replication": {
        "arrays": 3,
        "writes_per_array": 3,
        "program_verify_seed_base": 105000,
        "fault_observation_seed_base": 106000,
        "adam_pulse_noise_seed_base": 107000,
        "adam_selection_seed_base": 108000,
        "data_order_seed_base": 109000,
        "suffix_formula": "100*array_index+write_index",
    },
    "arms": [
        "direct_pv",
        "direct_pv_reset_stuck",
        "hwa_pv",
        "hwa_pv_reset_stuck",
        "hwa_pv_adam",
        "hwa_pv_reset_stuck_adam",
        "scratch_pv_adam",
        "scratch_pv_reset_stuck_adam",
    ],
    "evaluation": {
        "primary": "held_apparent_state",
        "persistent": "secondary_diagnostic_only",
        "checkpoint": "best_apparent_validation_epoch_then_lower_KL_then_earlier_epoch",
        "test": "selected_checkpoint_only",
        "aggregation": "average_writes_within_array_then_mean_and_full_range_across_arrays",
    },
}


def _first_difference(expected: Any, provided: Any, path: str = "config") -> str | None:
    if isinstance(expected, Mapping):
        if not isinstance(provided, Mapping):
            return f"{path}: expected an object"
        missing = sorted(set(expected) - set(provided))
        unknown = sorted(set(provided) - set(expected))
        if missing:
            return f"{path}: missing keys {missing!r}"
        if unknown:
            return f"{path}: unknown keys {unknown!r}"
        for key in expected:
            difference = _first_difference(expected[key], provided[key], f"{path}.{key}")
            if difference is not None:
                return difference
        return None
    if isinstance(expected, list):
        if not isinstance(provided, list) or len(expected) != len(provided):
            return f"{path}: expected exact list {expected!r}, provided {provided!r}"
        for index, (left, right) in enumerate(zip(expected, provided, strict=True)):
            difference = _first_difference(left, right, f"{path}[{index}]")
            if difference is not None:
                return difference
        return None
    if type(expected) is not type(provided) or expected != provided:
        return f"{path}: expected {expected!r}, provided {provided!r}"
    return None


def parse_ladder_config(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    """Accept only the reviewed full protocol, including every seed and arm."""

    difference = _first_difference(EXPECTED_CONFIG, payload)
    if difference is not None:
        raise ValueError(f"Figure-6 OM 256 ladder config mismatch: {difference}.")
    # A JSON round trip detaches callers from the module constant.
    return MappingProxyType(json.loads(json.dumps(payload)))


def load_ladder_config(path: Path) -> Mapping[str, Any]:
    source = path.expanduser().resolve()
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Expected readable ladder JSON at {source}.") from error
    if not isinstance(payload, Mapping):
        raise ValueError("Expected ladder config to be a JSON object.")
    return parse_ladder_config(payload)


__all__ = [
    "EXPECTED_CONFIG",
    "EXPERIMENT_ID",
    "SCHEMA_VERSION",
    "load_ladder_config",
    "parse_ladder_config",
]
