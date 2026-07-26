from __future__ import annotations

import copy
from pathlib import Path

import pytest

from experiments.mnist_conv.identity import run_fingerprint, sha256_json
from experiments.mnist_conv.lr_optimizer_boundary_spec import (
    OPTIMIZER_BOUNDARY_STUDY_SCHEMA_VERSION,
    OptimizerBoundaryStudySpec,
    optimizer_boundary_study_template,
)
from experiments.mnist_conv.specs import (
    ADAM_OPTIMIZER_V7,
    CONV3_ORDINARY_LR_PROTOCOL_ID,
    OPTIMIZER_BOUNDARY_PROTOCOL_ID,
    SGD_OPTIMIZER_V7,
    RunSpec,
    SpecValidationError,
    normalize_optimizer_v7,
)


REPO_ROOT = Path(__file__).resolve().parents[2]

_SOURCE_BY_COORDINATE = {
    ("conv2_baseline_v1_c1", 0.01): (
        "conv2_baseline_v1_c1--grid-c03-d01",
        "lrstudy_c735d2beead9fcbadda89a65ffe86257"
        "da53f15e6cbfab7b4f04b14b08ff6c2f",
        "run_spec.v5.json",
    ),
    ("conv2_baseline_v1_c1", 0.03): (
        "conv2_baseline_v1_c1--grid-c03-d02",
        "lrstudy_c735d2beead9fcbadda89a65ffe86257"
        "da53f15e6cbfab7b4f04b14b08ff6c2f",
        "run_spec.v5.json",
    ),
    ("conv2_ours_v4_c1", 0.003): (
        "conv2_ours_v4_c1--scheme-rho--c03-d00",
        "lrsweep_07eed332608b684ff77bcf994071b9023"
        "0fd877345b2e5132d9144cc4fa40435",
        "run_spec.v6.json",
    ),
    ("conv2_ours_v4_c1", 0.01): (
        "conv2_ours_v4_c1--scheme-rho--c03-d01",
        "lrsweep_07eed332608b684ff77bcf994071b9023"
        "0fd877345b2e5132d9144cc4fa40435",
        "run_spec.v6.json",
    ),
    ("conv2_legacy_v4_c0p25", 0.003): (
        "conv2_legacy_v4_c0p25--scheme-rho--c03-d00",
        "lrsweep_07eed332608b684ff77bcf994071b9023"
        "0fd877345b2e5132d9144cc4fa40435",
        "run_spec.v6.json",
    ),
    ("conv2_legacy_v4_c0p25", 0.01): (
        "conv2_legacy_v4_c0p25--scheme-rho--c03-d01",
        "lrsweep_07eed332608b684ff77bcf994071b9023"
        "0fd877345b2e5132d9144cc4fa40435",
        "run_spec.v6.json",
    ),
}


def _reuse_cells() -> list[dict[str, object]]:
    cells: list[dict[str, object]] = []
    for index, ((row_id, rho_dense), source) in enumerate(
        _SOURCE_BY_COORDINATE.items(),
        1,
    ):
        entry_id, collection_id, run_spec_name = source
        names = [
            "best_validation.pt",
            "final.pt",
            "minibatches.json",
            "parameter_diagnostics.csv",
            run_spec_name,
            "step_log.csv",
            "summary.json",
            "validation.json",
        ]
        outputs = [
            {
                "name": name,
                "bytes": 1000 + index + output_index,
                "sha256": format(index * 16 + output_index, "064x"),
            }
            for output_index, name in enumerate(names)
        ]
        output_by_name = {output["name"]: output for output in outputs}
        cells.append(
            {
                "row_id": row_id,
                "rho_conv": 0.01,
                "rho_dense": rho_dense,
                "source_entry_id": entry_id,
                "source_collection_id": collection_id,
                "expected_completed_steps": 17190,
                "run_spec_sha256": output_by_name[run_spec_name]["sha256"],
                "candidate_summary_sha256": output_by_name["summary.json"]["sha256"],
                "completion_sha256": format(1000 + index, "064x"),
                "completion_bytes": 1900 + index,
                "outputs": outputs,
            }
        )
    return cells


def _dataset(*, validation_batch_size: int = 128) -> dict[str, object]:
    return {
        "name": "mnist",
        "input_shape": [2, 28, 28],
        "batch_size": 16,
        "normalization": {"mean": 0.1307, "std": 0.3081, "scale": 0.3},
        "affine": {
            "enabled": False,
            "preset": "ordinary_identity",
            "degrees": 0.0,
            "translate": [0.0, 0.0],
            "scale": [1.0, 1.0],
            "shear": 0.0,
            "seed": 1729,
            "interpolation": "bilinear",
            "fill": 0.0,
        },
        "max_batches": None,
        "train_shuffle_seed": 0,
        "validation": {
            "source": "mnist_train",
            "size": 5000,
            "samples_per_class": 500,
            "split_seed": 0,
            "batch_size": validation_batch_size,
            "stratified": True,
            "official_test_enabled": False,
            "indices_sha256": "1" * 64,
        },
    }


def _solver(inference_iterations: int, training_iterations: int) -> dict[str, object]:
    return {
        "inference_iterations": inference_iterations,
        "training_iterations": training_iterations,
        "energy_mode": "asynchronous",
        "minimizer": {
            "double_diode_updater": "CustomExponentialDoubleDiodeUpdater",
            "adaptive_equilibrium": False,
            "overrelaxation_factor": 1.1,
            "single_diode_updater": "custom",
            "iv_data_path": None,
            "experimental_damping": 0.5,
            "experimental_newton_max_steps": 100,
            "settings": {
                "rel_tol": 1e-5,
                "vn_tol": 1e-6,
                "use_polish": True,
                "max_newton_iters": 32,
                "z_thresh": 1e10,
                "exp_clip": 1e5,
                "dynamic_polish": True,
                "overrelaxation_reject_steps": False,
                "overrelaxation_reject_max_tries": 3,
                "overrelaxation_reject_shrink": 0.5,
                "overrelaxation_reject_eps": 0.0,
                "experimental_exponential_newton_tol_progressive": True,
                "experimental_exponential_newton_tol_start": 1e-5,
                "experimental_exponential_newton_tol_end": 1e-5,
                "experimental_exponential_newton_tol_switch_hi": 1e-2,
                "experimental_exponential_newton_tol_switch_lo": 5e-4,
            },
        },
    }


def _training(
    *,
    optimizer: dict[str, object],
    rates: dict[str, float],
    epochs: int = 5,
    total_steps: int = 17190,
) -> dict[str, object]:
    return {
        "algorithm": "BP",
        "epochs": epochs,
        "optimizer": copy.deepcopy(optimizer),
        "learning_rates_by_parameter": copy.deepcopy(rates),
        "schedule": {
            "name": "constant",
            "interval": "optimizer_step",
            "total_steps": total_steps,
            "scheduler_enabled": False,
        },
        "beta": 0.1,
        "checkpoint_rule": "min_validation_loss",
        "pruning": {
            "enabled": False,
            "after_epoch": None,
            "min_best_validation_accuracy": None,
        },
        "batch_state_policy": "reset_each_batch",
        "diagnostics": {
            "enabled": True,
            "bounded_parameter_classes": ["ConvWeight", "DenseWeight"],
            "bias_global_gate": False,
            "range_epsilon": 1e-8,
            "projection_epsilon": 1e-12,
            "early_fraction": 1.0,
            "projection_efficiency_threshold": 0.5,
            "projection_persistence": 16,
            "occupancy_delta_threshold": 0.2,
            "occupancy_persistence": 16,
        },
    }


def _model(
    *,
    voltage_amp: float,
    current_amp: float,
    input_gain: float,
    depth: int,
) -> dict[str, object]:
    return {
        "type": "resistive_conv",
        "non_linearity": "hard_sigmoid",
        "voltage_amp": voltage_amp,
        "current_amp": current_amp,
        "input_gain": input_gain,
        "weight_gains": [1.0] * (depth + 1),
        "weight_min": 0.0,
        "weight_max": 100.0,
        "weight_init_mode": "kaiming_uniform",
        "quadratic_diode_param": {},
        "exponential_diode_param": {},
        "hard_sigmoid_param": {"g_on": 100.0, "g_off": 0.0, "v_off": 4.0},
        "trainable_parameters": {
            "weights": True,
            "biases": True,
            "amplification": False,
            "hard_sigmoid_v_off": False,
        },
        "amplification_min": 1e-6,
        "amplification_max": None,
    }


def _calibration(depth: int, calibration_id: str) -> dict[str, object]:
    measurements = [
        {
            "layer_index": index,
            "measured_saturation": 0.3 if index == 1 else 0.05 * index,
        }
        for index in range(1, depth + 1)
    ]
    return {
        "kind": "hard_sigmoid_saturation",
        "calibration_id": calibration_id,
        "scope": "first_hidden_layer",
        "sample_count": 256,
        "batch_size": 64,
        "model_seed": 0,
        "affine_seed": 1729,
        "settling_iterations": 64,
        "adaptive_equilibrium": False,
        "v_off": 4.0,
        "g_on": 100.0,
        "g_off": 0.0,
        "target_initial_saturation": 0.3,
        "measured_initial_saturation": 0.3,
        "layer_measurements": measurements,
    }


def _boundary_run(
    *,
    optimizer: dict[str, object] | None = None,
    stage: str = "main_grid",
    bias_policy: str = "attached",
) -> dict[str, object]:
    optimizer = copy.deepcopy(optimizer or ADAM_OPTIMIZER_V7)
    units = {
        "ConvWeight_0": 2.0,
        "ConvWeight_1": 4.0,
        "DenseWeight_0": 5.0,
    }
    rates = {
        "ConvWeight_0": 0.0075,
        "ConvWeight_1": 0.00375,
        "DenseWeight_0": 0.002,
        "Bias_0": 0.0075,
        "Bias_1": 0.00375,
    }
    parent_completion = None
    parent_decision = None
    bias_cap = None
    if bias_policy == "capped":
        rates["Bias_0"] = 0.001
        rates["Bias_1"] = 0.0005
        parent_completion = "9" * 64
        parent_decision = "d" * 64
        bias_cap = 0.001
    elif stage == "upper_sentinel":
        parent_decision = "d" * 64

    return {
        "schema_version": "mnist-conv-run/v7",
        "label": "conv2-boundary-test",
        "seed": 0,
        "replicate_id": None,
        "protocol_id": OPTIMIZER_BOUNDARY_PROTOCOL_ID,
        "category": "diagnostic",
        "run": {
            "dataset": _dataset(),
            "architecture": {
                "profile": "conv2",
                "channels": [64, 128],
                "kernel_sizes": [3, 3],
                "strides": [2, 2],
                "paddings": [1, 1],
                "output_dim": 20,
                "pooling": {"mode": "none"},
            },
            "model": _model(
                voltage_amp=1.0,
                current_amp=1.0,
                input_gain=253.302230835,
                depth=2,
            ),
            "solver": _solver(16, 6),
            "training": _training(optimizer=optimizer, rates=rates),
            "initialization": {
                "checkpoint": {
                    "path": "lr_studies/boundary/initialization/conv2.pt",
                    "sha256": "2" * 64,
                    "format": "drn.function.parameters/v1",
                    "source_run_id": None,
                    "role": "initialization",
                }
            },
            "calibration": _calibration(2, "conv2-boundary-test"),
            "lr_provenance": {
                "study_id": "lrstudy_" + "3" * 64,
                "row_id": "conv2_baseline_v1_c1",
                "candidate_stage": stage,
                "execution_mode": "new",
                "optimizer_name": optimizer["name"],
                "optimizer_parameters": copy.deepcopy(optimizer),
                "rho_conv": 0.015,
                "rho_dense": 0.01,
                "normalization_unit_by_weight": units,
                "bias_q90_unit_by_parameter": {
                    "Bias_0": 1.0,
                    "Bias_1": 2.0,
                },
                "bias_weight_mapping": {
                    "Bias_0": "ConvWeight_0",
                    "Bias_1": "ConvWeight_1",
                },
                "bias_lr_policy": bias_policy,
                "bias_rho_cap": bias_cap,
                "raw_learning_rates_by_parameter": copy.deepcopy(rates),
                "probe_sha256": "4" * 64,
                "bias_q90_probe_sha256": "c" * 64,
                "parent_study_config_sha256": "5" * 64,
                "parent_entry_completion_sha256": parent_completion,
                "parent_decision_sha256": parent_decision,
                "split_sha256": "1" * 64,
                "normalization_minibatches_sha256": "6" * 64,
                "replay_minibatches_sha256": "7" * 64,
                "training_batch_order_sha256": "8" * 64,
                "initialization_checkpoint_sha256": "2" * 64,
                "initialization_tensor_sha256": "a" * 64,
                "code_fingerprint_sha256": "b" * 64,
                "official_test_read": False,
            },
        },
    }


def _conv3_v7_run() -> dict[str, object]:
    units = {
        "ConvWeight_0": 0.2,
        "ConvWeight_1": 0.4,
        "ConvWeight_2": 0.5,
        "DenseWeight_0": 2.0,
    }
    rates = {
        "ConvWeight_0": 0.015,
        "ConvWeight_1": 0.0075,
        "ConvWeight_2": 0.006,
        "DenseWeight_0": 0.005,
        "Bias_0": 0.015,
        "Bias_1": 0.0075,
        "Bias_2": 0.006,
    }
    return {
        "schema_version": "mnist-conv-run/v7",
        "label": "conv3-v7-regression-fixture",
        "seed": 0,
        "replicate_id": None,
        "protocol_id": CONV3_ORDINARY_LR_PROTOCOL_ID,
        "category": "diagnostic",
        "run": {
            "dataset": _dataset(validation_batch_size=64),
            "architecture": {
                "profile": "conv3",
                "channels": [64, 128, 256],
                "kernel_sizes": [3, 3, 3],
                "strides": [2, 2, 1],
                "paddings": [1, 1, 1],
                "output_dim": 20,
                "pooling": {"mode": "none"},
            },
            "model": _model(
                voltage_amp=4.0,
                current_amp=1.0,
                input_gain=744.739074707,
                depth=3,
            ),
            "solver": _solver(32, 8),
            "training": _training(
                optimizer=SGD_OPTIMIZER_V7,
                rates=rates,
                epochs=3,
                total_steps=10314,
            ),
            "initialization": {
                "checkpoint": {
                    "path": "lr_studies/conv3/initialization/conv3.pt",
                    "sha256": "c" * 64,
                    "format": "drn.function.parameters/v1",
                    "source_run_id": None,
                    "role": "initialization",
                }
            },
            "calibration": _calibration(3, "conv3-v7-regression"),
            "lr_provenance": {
                "study_id": (
                    "lrstudy_6e56b399cf3e522a2c3f39f66a516111"
                    "47b6171dd3b9e223f20791bc6275caf0"
                ),
                "row_id": "conv3_ours_v4_c1",
                "candidate_stage": "core_grid",
                "rho_conv": 0.003,
                "rho_dense": 0.01,
                "median_unit_by_weight": units,
                "bias_weight_mapping": {
                    "Bias_0": "ConvWeight_0",
                    "Bias_1": "ConvWeight_1",
                    "Bias_2": "ConvWeight_2",
                },
                "probe_sha256": "d" * 64,
                "split_sha256": "1" * 64,
                "batch_order_sha256": "e" * 64,
                "initialization_checkpoint_sha256": "c" * 64,
                "initialization_tensor_sha256": "f" * 64,
                "gain_calibration_dataset": "deterministic_medium_affine_mnist",
                "optimization_dataset": "ordinary_mnist",
                "official_test_read": False,
            },
        },
    }


def test_boundary_study_is_full_strict_content_addressed_contract() -> None:
    spec = OptimizerBoundaryStudySpec.from_dict(
        optimizer_boundary_study_template(_reuse_cells())
    )

    assert spec.data["schema_version"] == OPTIMIZER_BOUNDARY_STUDY_SCHEMA_VERSION
    assert spec.data["optimizer_arms"]["adam"] == ADAM_OPTIMIZER_V7
    assert spec.data["candidate_training"]["total_steps"] == 17190
    assert spec.data["dataset"]["normalization"] == {
        "mean": 0.1307,
        "std": 0.3081,
        "scale": 0.3,
    }
    assert spec.data["dataset"]["official_test"]["read_allowed"] is False
    assert spec.data["solver"]["minimizer"]["settings"]["rel_tol"] == 1e-5
    assert len(spec.data["reuse"]["cells"]) == 6
    assert all(
        cell["expected_completed_steps"] == 17190
        and len(cell["outputs"]) == 8
        for cell in spec.data["reuse"]["cells"]
    )
    assert "/home/" not in str(spec.identity_payload())

    renamed = spec.to_dict()
    renamed["name"] = "display-only"
    assert OptimizerBoundaryStudySpec.from_dict(renamed).study_id == spec.study_id

    changed = spec.to_dict()
    changed["solver"]["minimizer"]["settings"]["rel_tol"] = 1e-4
    with pytest.raises(SpecValidationError, match=r"study\.solver"):
        OptimizerBoundaryStudySpec.from_dict(changed)


def test_boundary_study_reuse_is_exact_and_fails_closed() -> None:
    value = optimizer_boundary_study_template(_reuse_cells())
    value["reuse"]["cells"][0]["completion_sha256"] = "not-a-hash"
    with pytest.raises(SpecValidationError, match="completion_sha256"):
        OptimizerBoundaryStudySpec.from_dict(value)

    value = optimizer_boundary_study_template(_reuse_cells())
    value["reuse"]["cells"][0]["expected_completed_steps"] = 17189
    with pytest.raises(SpecValidationError, match="exactly 17190"):
        OptimizerBoundaryStudySpec.from_dict(value)

    value = optimizer_boundary_study_template(_reuse_cells())
    value["reuse"]["cells"][0]["source_entry_id"] = "/host/dependent/path"
    with pytest.raises(SpecValidationError, match="without path separators"):
        OptimizerBoundaryStudySpec.from_dict(value)


def test_v7_optimizer_union_is_exact_and_flags_are_explicit() -> None:
    from experiments.mnist_conv.lr_engine import (
        FROZEN_ADAM_OPTIMIZER,
        FROZEN_SGD_OPTIMIZER,
        validate_explicit_optimizer_contract,
    )

    assert normalize_optimizer_v7(SGD_OPTIMIZER_V7) == SGD_OPTIMIZER_V7
    assert normalize_optimizer_v7(ADAM_OPTIMIZER_V7) == ADAM_OPTIMIZER_V7
    assert FROZEN_SGD_OPTIMIZER == SGD_OPTIMIZER_V7
    assert FROZEN_ADAM_OPTIMIZER == ADAM_OPTIMIZER_V7
    assert validate_explicit_optimizer_contract(SGD_OPTIMIZER_V7) == (
        SGD_OPTIMIZER_V7
    )
    assert validate_explicit_optimizer_contract(ADAM_OPTIMIZER_V7) == (
        ADAM_OPTIMIZER_V7
    )

    missing = copy.deepcopy(ADAM_OPTIMIZER_V7)
    missing.pop("fused")
    with pytest.raises(SpecValidationError, match="exactly keys"):
        normalize_optimizer_v7(missing)

    enabled = copy.deepcopy(ADAM_OPTIMIZER_V7)
    enabled["foreach"] = True
    with pytest.raises(SpecValidationError, match="exactly false"):
        normalize_optimizer_v7(enabled)

    extra = copy.deepcopy(SGD_OPTIMIZER_V7)
    extra["nesterov"] = False
    with pytest.raises(SpecValidationError, match="exactly keys"):
        normalize_optimizer_v7(extra)


@pytest.mark.parametrize("optimizer", [SGD_OPTIMIZER_V7, ADAM_OPTIMIZER_V7])
def test_v7_boundary_run_binds_optimizer_units_and_all_parameter_rates(
    optimizer: dict[str, object],
) -> None:
    spec = RunSpec.from_dict(_boundary_run(optimizer=optimizer))
    training = spec.data["run"]["training"]
    provenance = spec.data["run"]["lr_provenance"]

    assert training["optimizer"] == optimizer
    assert set(training["learning_rates_by_parameter"]) == {
        "ConvWeight_0",
        "ConvWeight_1",
        "DenseWeight_0",
        "Bias_0",
        "Bias_1",
    }
    assert provenance["optimizer_parameters"] == optimizer
    assert (
        provenance["raw_learning_rates_by_parameter"]
        == training["learning_rates_by_parameter"]
    )
    assert provenance["official_test_read"] is False


def test_v7_capped_confirmation_applies_q90_cap_without_increasing_bias_lr() -> None:
    value = _boundary_run(
        stage="bias_capped_confirmation",
        bias_policy="capped",
    )
    spec = RunSpec.from_dict(value)
    rates = spec.data["run"]["training"]["learning_rates_by_parameter"]

    assert rates["Bias_0"] == min(rates["ConvWeight_0"], 0.001 / 1.0)
    assert rates["Bias_1"] == min(rates["ConvWeight_1"], 0.001 / 2.0)
    assert rates["Bias_0"] <= rates["ConvWeight_0"]
    assert rates["Bias_1"] <= rates["ConvWeight_1"]

    increased = copy.deepcopy(value)
    increased["run"]["training"]["learning_rates_by_parameter"]["Bias_1"] = 0.01
    increased["run"]["lr_provenance"]["raw_learning_rates_by_parameter"][
        "Bias_1"
    ] = 0.01
    with pytest.raises(SpecValidationError, match=r"min\(attached weight rate"):
        RunSpec.from_dict(increased)


def test_v7_rejects_missing_rate_and_optimizer_provenance_drift() -> None:
    missing_rate = _boundary_run()
    del missing_rate["run"]["training"]["learning_rates_by_parameter"][
        "DenseWeight_0"
    ]
    with pytest.raises(SpecValidationError, match="exactly keys"):
        RunSpec.from_dict(missing_rate)

    optimizer_drift = _boundary_run()
    optimizer_drift["run"]["lr_provenance"]["optimizer_name"] = "SGD"
    with pytest.raises(SpecValidationError, match="optimizer_parameters.name"):
        RunSpec.from_dict(optimizer_drift)

    raw_rate_drift = _boundary_run()
    raw_rate_drift["run"]["lr_provenance"]["raw_learning_rates_by_parameter"][
        "DenseWeight_0"
    ] *= 2
    with pytest.raises(
        SpecValidationError,
        match="run.training.learning_rates_by_parameter",
    ):
        RunSpec.from_dict(raw_rate_drift)


def test_revised_conv3_v7_normalized_payload_and_run_id_are_frozen() -> None:
    spec = RunSpec.from_dict(_conv3_v7_run())
    provenance = {
        "git_revision": "1" * 40,
        "dirty_source_digest": "2" * 64,
        "effective_code_fingerprint": "3" * 64,
    }

    assert sha256_json(spec.identity_payload()) == (
        "fd2bfe851f0c2e8362169123f152c406f68d5cb130068e69c841eb20e573df77"
    )
    assert run_fingerprint(spec, provenance) == (
        "run_dffabb9fbd8e010ab0e34ee2627f8541d4c4c21059025feedad303fbdd1a47cd"
    )


def test_existing_v1_v6_and_revised_v7_study_identities_are_frozen() -> None:
    from experiments.mnist_conv.lr_study_spec import LRStudySpec

    expected = {
        "hardsigmoid_lr_study_sgd_bs16_v1.json": (
            "lrstudy_2103c850c10540d5628cd20cbebc0af5"
            "19bd871116062e8b80e50ad0c068a9a4"
        ),
        "hardsigmoid_lr_conv2_two_rho_constant_sgd_bs16_v6.json": (
            "lrstudy_c735d2beead9fcbadda89a65ffe86257"
            "da53f15e6cbfab7b4f04b14b08ff6c2f"
        ),
        "hardsigmoid_lr_conv3_scheme_two_rho_constant_sgd_bs16_v7.json": (
            "lrstudy_6e56b399cf3e522a2c3f39f66a516111"
            "47b6171dd3b9e223f20791bc6275caf0"
        ),
    }
    for filename, expected_id in expected.items():
        path = REPO_ROOT / "configs/conv" / filename
        assert LRStudySpec.from_path(path).study_id == expected_id
