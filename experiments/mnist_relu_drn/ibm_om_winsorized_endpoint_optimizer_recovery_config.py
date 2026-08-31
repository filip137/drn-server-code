"""Strict exploratory config for matched endpoint optimizer recovery.

Endpoint seeds 89401--89405 are stochastic P&V realizations on one frozen
target-87004 device population.  They are not independent array populations.
Seed 89401 is development-only; seeds 89402--89405 are fixed evaluation
realizations opened only after optimizer-specific settings have been frozen.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from experiments.mnist_relu.config import _keys, _object
from experiments.mnist_relu_drn.config import (
    EXPERIMENT_ID as STUDENT_EXPERIMENT_ID,
    StudentConfig,
    StudentTrainSpec,
    parse_student_config,
    resolve_student_spec,
)
from experiments.schema import RunMode, config_error


EXPERIMENT_ID = "mnist_ibm_om_winsorized_endpoint_optimizer_recovery.v1"
SCHEMA_VERSION = 1
TARGET_ASSIGNMENT_SEED = 87004
DEVELOPMENT_ENDPOINT_SEED = 89401
EVALUATION_ENDPOINT_SEEDS = (89402, 89403, 89404, 89405)
ALL_ENDPOINT_SEEDS = (DEVELOPMENT_ENDPOINT_SEED, *EVALUATION_ENDPOINT_SEEDS)

EXPECTED_TEACHER_WEIGHTS_PATH = "data/mnist_relu_teacher_fixed_init_20260816.pt"
EXPECTED_TEACHER_WEIGHTS_SHA256 = (
    "9a961a77628e365b54fdf59304ec7fddf46f1f4834c4c03cecea9c204f837f52"
)
PREDECESSOR_RESULT_ID = (
    "mnist-ibm-om-winsorized-cross-array-open-loop-adam-"
    "exploratory-20260829-v1"
)
PREDECESSOR_RUN_RELATIVE_PATH = (
    "runs/fresh_87004/20260829T121606.793011Z-8adf8460-a5c62c4d"
)
PREDECESSOR_SUMMARY_SHA256 = (
    "4771e2ac2ffd57baed6d6088e335023ab9778883a0d9e5d965a85d25d2c3b323"
)
POPULATION_RELATIVE_PATH = (
    "artifacts/fresh_target_87004/winsorized_identity/winsorized_population.npz"
)
POPULATION_RECEIPT_RELATIVE_PATH = (
    "artifacts/fresh_target_87004/winsorized_identity/"
    "winsorized_population.receipt.json"
)
POPULATION_SHA256 = (
    "d30274ad576bb24cd2c2d17d01bd6c3cd1bb99a908ee3990f8c1d6e12865e741"
)
POPULATION_RECEIPT_SHA256 = (
    "98ede52f7c8a8cda09dd9346d2ca9e3195fd024492f99295f1765aa9d562b76c"
)
POPULATION_FINGERPRINT = (
    "214a4c48c81d4cd1edecef356aae7fb564ab60541cdb4de34caf246dfd0026b8"
)
SOURCE_JOINT_HARDWARE_INSTANCE_ID = (
    "fabbba85777d05cd41f6e08caccd597bc9a32fd0510938a86f3b3c9f51ec5fdb"
)

ENDPOINT_SHA256 = {
    89401: "8bf90e76c924b9b821a9794502df408bfd8b9c3e4bd17892370127e079b1d8dd",
    89402: "db8fa99e3e82f529e213664fe06ac3fc73d49fbce324bf6c4eccfb09f3ef7dfe",
    89403: "4e5f16119ed29341210d36f4786feeab229cabf318aecf6dce13761a90d9a3f4",
    89404: "d877e9df92b6fea16ea772b3e71da36162394145229ff5f0d7e7169dfbcb9466",
    89405: "2ebd2e44cde99393bfb12f475c0fa51efc55ef5f1dd09d93eb024c7ccba3d4d7",
}
ENDPOINT_TEST_CORRECT = {89401: 7424, 89402: 6249, 89403: 6374, 89404: 6629, 89405: 6012}
ENDPOINT_TEST_PREDICTION_SHA256 = {
    89401: "3d185ed28c0de1bc81f26d6b9044e8b19014bdd9f62b74536e99eed4951095aa",
    89402: "cdb43f067ce64d3cd1fdf2db81e40606e1280e8738959aacb988dc736a97b608",
    89403: "4b8fda4c36eca815f075645f6c3220f142c11a8ed115deb1b44e962b99e44026",
    89404: "531d48558ef5de99385acda9d4977ffb07145a8e5a0686bfd4b6ab618d7c3928",
    89405: "c7ff1fa14c9c1b60a75fe396b45542806bd32c8704169cf64f619286bcc53422",
}

NOMINAL_DELTA_X = 0.04745
PULSE_CAP = 64
MAXIMUM_RANDOM_DRAWS = 385
AUXILIARY_MAXIMUM_RANDOM_DRAWS = 1 + 2 * PULSE_CAP
FIXED_LOGIT_GAIN = 14.12537544622754
ADAM_LEARNING_RATE_RAW_X = 3.0e-5
QAT_LAYER_LEARNING_RATES = (0.004411914893617021, 1.1974808510638298e-5)
SGD_SCALE_GRID = (0.3, 1.0, 3.0)
TT_V1_TRANSFER_SCALE_GRID = (3.0, 10.0, 30.0)
TT_V2_TRANSFER_SCALE_GRID = (10.0, 30.0, 100.0)
TT_FAST_LEARNING_RATE_RAW_X = 0.1
DEVELOPMENT_MAXIMUM_BATCHES = 512


@dataclass(frozen=True)
class EndpointArtifactContract:
    endpoint_seed: int
    relative_path: str
    sha256: str
    historical_test_correct: int
    historical_test_prediction_sha256: str


@dataclass(frozen=True)
class PredecessorContract:
    result_id: str
    run_relative_path: str
    summary_relative_path: str
    summary_sha256: str
    population_relative_path: str
    population_receipt_relative_path: str
    population_sha256: str
    population_receipt_sha256: str
    population_fingerprint: str
    source_joint_hardware_instance_id: str
    endpoints: tuple[EndpointArtifactContract, ...]


@dataclass(frozen=True)
class DevelopmentContract:
    endpoint_seed: int
    role: str
    maximum_batches: int
    selection_metric: str
    tie_breakers: tuple[str, ...]
    sgd_scale_grid: tuple[float, ...]
    tt_v1_transfer_scale_grid: tuple[float, ...]
    tt_v2_transfer_scale_grid: tuple[float, ...]
    test_opened: bool


@dataclass(frozen=True)
class OptimizerContract:
    names: tuple[str, ...]
    epochs: int
    nominal_delta_x: float
    pulse_cap_per_cell: int
    c_maximum_random_draws: int
    adam_learning_rate_raw_x: float
    adam_beta1: float
    adam_beta2: float
    adam_epsilon: float
    qat_layer_learning_rates: tuple[float, float]
    tt_fast_learning_rate_raw_x: float
    tt_gamma: float
    tt_transfer_every_minibatches: int
    tt_units_in_mbatch: bool
    tt_reads_per_transfer: int
    tt_random_selection: bool
    tt_reset_probability: float
    tt_chop_probability: float
    tt_desired_bl: int
    tt_h_threshold: float
    tt_h_momentum: float
    tt_auto_scale: bool
    tt_correct_gradient_magnitudes: bool
    tt_transfer_axis: str
    tt_a_read_state: str
    tt_a_zero_initialization: str
    tt_a_maximum_random_draws: int
    verify_reads_during_training: int


@dataclass(frozen=True)
class EvaluationContract:
    endpoint_seeds: tuple[int, ...]
    endpoint_semantics: str
    independent_population_count: int
    validation_role: str
    test_role: str
    selection_after_freeze: bool


@dataclass(frozen=True)
class ExecutionContract:
    evidence_tier: str
    evidence_class: str
    device: str
    state_authority: str
    full_conductance_forward: str
    update_surface: str


@dataclass(frozen=True)
class EndpointOptimizerRecoveryProtocol:
    expected_teacher_weights_path: str
    expected_teacher_weights_sha256: str
    target_assignment_seed: int
    predecessor: PredecessorContract
    development: DevelopmentContract
    optimizers: OptimizerContract
    evaluation: EvaluationContract
    execution: ExecutionContract
    auxiliary_endpoint_seed_by_p0: tuple[tuple[int, int], ...]
    auxiliary_pulse_selection_seed_by_p0: tuple[tuple[int, int], ...]
    slow_pulse_selection_seed_by_p0: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class EndpointOptimizerRecoveryConfig:
    schema_version: int
    experiment_id: str
    protocol: EndpointOptimizerRecoveryProtocol
    student: StudentConfig


@dataclass(frozen=True)
class EndpointOptimizerRecoveryTrainSpec:
    experiment_id: str
    protocol: EndpointOptimizerRecoveryProtocol
    student: StudentTrainSpec


def _student_payload() -> Mapping[str, Any]:
    return {
        "schema_version": 1,
        "experiment_id": STUDENT_EXPERIMENT_ID,
        "runtime": {"seed": 42, "data_seed": 42, "device": "cuda", "dtype": "float32"},
        "data": {"batch_size": 16, "validation_points": 5000, "num_points": None, "shuffle": True},
        "teacher": {"type": "bias_free_relu", "initialization": "signed_weight_mapping"},
        "model": {
            "dims": [1568, 100, 20],
            "input_gain": 100.0,
            "conductance_min": 0.0,
            "conductance_max": 2.0,
            "voltage_amp": 4.0,
            "current_amp": 0.25,
            "encoding": "single",
            "include_biases": False,
            "non_linearity": {
                "type": "perfect_diode",
                "quadratic_diode_param": {},
                "exponential_diode_param": {},
                "hard_sigmoid_param": {},
            },
        },
        "solver": {
            "inference_iterations": 4,
            "training_iterations": 4,
            "mode": "asynchronous",
            "overrelaxation_factor": 1.1,
        },
        "mapping": {
            "scale_fractions": [1.0],
            "scale_fraction_pairs": [[1.0, 1.0]],
            "range_placement": "lower",
            "calibration_examples": 1024,
            "calibration_batch_size": 128,
            "logit_gain_min": 0.001,
            "logit_gain_max": 1000.0,
            "logit_gain_steps": 121,
        },
        "modes": {
            "train": {
                "num_epochs": 1,
                "learning_rates": [0.0, 0.0],
                "temperature": 1.0,
                "log_every": 1,
                "max_batches": None,
                "max_validation_batches": None,
                "minimum_relative_kl_improvement": 0.0,
                "weight_modifier": {"type": "none", "parameters": {}},
                "update_backend": {"type": "ideal", "parameters": {}},
                "selection_evaluation": "clean",
                "selection_metric": "student_accuracy",
                "selection_noise_repeats": 1,
                "selection_weight_modifier": {"type": "none", "parameters": {}},
            }
        },
    }


def _endpoint_contracts() -> tuple[EndpointArtifactContract, ...]:
    return tuple(
        EndpointArtifactContract(
            endpoint_seed=seed,
            relative_path=f"artifacts/deployments/contrast_transfer_seed_{seed}.pt",
            sha256=ENDPOINT_SHA256[seed],
            historical_test_correct=ENDPOINT_TEST_CORRECT[seed],
            historical_test_prediction_sha256=ENDPOINT_TEST_PREDICTION_SHA256[seed],
        )
        for seed in ALL_ENDPOINT_SEEDS
    )


def parse_endpoint_optimizer_recovery_config(
    payload: Mapping[str, Any],
) -> EndpointOptimizerRecoveryConfig:
    raw = _object(payload, "config")
    _keys(
        raw,
        "config",
        {
            "schema_version",
            "experiment_id",
            "target_assignment_seed",
            "development_endpoint_seed",
            "evaluation_endpoint_seeds",
        },
    )
    expected = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "target_assignment_seed": TARGET_ASSIGNMENT_SEED,
        "development_endpoint_seed": DEVELOPMENT_ENDPOINT_SEED,
        "evaluation_endpoint_seeds": list(EVALUATION_ENDPOINT_SEEDS),
    }
    for name, value in expected.items():
        if raw[name] != value:
            raise config_error(f"config.{name}", f"to equal {value!r}", raw[name])

    protocol = EndpointOptimizerRecoveryProtocol(
        expected_teacher_weights_path=EXPECTED_TEACHER_WEIGHTS_PATH,
        expected_teacher_weights_sha256=EXPECTED_TEACHER_WEIGHTS_SHA256,
        target_assignment_seed=TARGET_ASSIGNMENT_SEED,
        predecessor=PredecessorContract(
            result_id=PREDECESSOR_RESULT_ID,
            run_relative_path=PREDECESSOR_RUN_RELATIVE_PATH,
            summary_relative_path="artifacts/scientific_summary.json",
            summary_sha256=PREDECESSOR_SUMMARY_SHA256,
            population_relative_path=POPULATION_RELATIVE_PATH,
            population_receipt_relative_path=POPULATION_RECEIPT_RELATIVE_PATH,
            population_sha256=POPULATION_SHA256,
            population_receipt_sha256=POPULATION_RECEIPT_SHA256,
            population_fingerprint=POPULATION_FINGERPRINT,
            source_joint_hardware_instance_id=SOURCE_JOINT_HARDWARE_INSTANCE_ID,
            endpoints=_endpoint_contracts(),
        ),
        development=DevelopmentContract(
            endpoint_seed=DEVELOPMENT_ENDPOINT_SEED,
            role="validation_only_optimizer_specific_hyperparameter_development",
            maximum_batches=DEVELOPMENT_MAXIMUM_BATCHES,
            selection_metric="maximum_validation_student_correct",
            tie_breakers=(
                "lower_validation_teacher_student_KL",
                "fewer_slow_C_pulses",
                "smaller_candidate_scale",
            ),
            sgd_scale_grid=SGD_SCALE_GRID,
            tt_v1_transfer_scale_grid=TT_V1_TRANSFER_SCALE_GRID,
            tt_v2_transfer_scale_grid=TT_V2_TRANSFER_SCALE_GRID,
            test_opened=False,
        ),
        optimizers=OptimizerContract(
            names=("open_loop_sgd", "open_loop_adam", "tt_v1", "tt_v2"),
            epochs=1,
            nominal_delta_x=NOMINAL_DELTA_X,
            pulse_cap_per_cell=PULSE_CAP,
            c_maximum_random_draws=MAXIMUM_RANDOM_DRAWS,
            adam_learning_rate_raw_x=ADAM_LEARNING_RATE_RAW_X,
            adam_beta1=0.9,
            adam_beta2=0.999,
            adam_epsilon=1e-8,
            qat_layer_learning_rates=QAT_LAYER_LEARNING_RATES,
            tt_fast_learning_rate_raw_x=TT_FAST_LEARNING_RATE_RAW_X,
            tt_gamma=0.0,
            tt_transfer_every_minibatches=1,
            tt_units_in_mbatch=True,
            tt_reads_per_transfer=1,
            tt_random_selection=False,
            tt_reset_probability=0.0,
            tt_chop_probability=0.0,
            tt_desired_bl=1,
            tt_h_threshold=1.0,
            tt_h_momentum=0.0,
            tt_auto_scale=False,
            tt_correct_gradient_magnitudes=True,
            tt_transfer_axis="canonical_logical_[out,in]_input_columns",
            tt_a_read_state=(
                "held_apparent_post_write_raw_x_displacement_from_intrinsic_symmetry;"
                "not_fresh_independent_read_noise"
            ),
            tt_a_zero_initialization="bounded_hidden_model_oracle_intrinsic_symmetry",
            tt_a_maximum_random_draws=AUXILIARY_MAXIMUM_RANDOM_DRAWS,
            verify_reads_during_training=0,
        ),
        evaluation=EvaluationContract(
            endpoint_seeds=EVALUATION_ENDPOINT_SEEDS,
            endpoint_semantics=(
                "four_stochastic_PV_endpoint_realizations_on_one_target_87004_population"
            ),
            independent_population_count=1,
            validation_role="diagnostic_only_after_hyperparameters_frozen",
            test_role="before_and_after_fixed_one_epoch_arm_no_selection",
            selection_after_freeze=False,
        ),
        execution=ExecutionContract(
            evidence_tier="exploratory_noncanonical",
            evidence_class="model_based_winsorized_IBM_OM_control",
            device="cuda",
            state_authority="persistent_slow_C_and_physical_fast_A_plants",
            full_conductance_forward="G=native_raw_active_a+1_without_remap_or_clipping",
            update_surface=(
                "column_serial_open_loop_stochastic_coincidence_emulator;"
                "qualified_OM_plant_TT_minibatch_equation_emulators"
            ),
        ),
        auxiliary_endpoint_seed_by_p0=tuple((seed, seed + 2000) for seed in ALL_ENDPOINT_SEEDS),
        auxiliary_pulse_selection_seed_by_p0=tuple((seed, seed + 3000) for seed in ALL_ENDPOINT_SEEDS),
        slow_pulse_selection_seed_by_p0=tuple((seed, seed + 4000) for seed in ALL_ENDPOINT_SEEDS),
    )
    return EndpointOptimizerRecoveryConfig(
        schema_version=SCHEMA_VERSION,
        experiment_id=EXPERIMENT_ID,
        protocol=protocol,
        student=parse_student_config(_student_payload()),
    )


def resolve_endpoint_optimizer_recovery_spec(
    document: EndpointOptimizerRecoveryConfig, mode: RunMode
) -> EndpointOptimizerRecoveryTrainSpec:
    if mode is not RunMode.TRAIN:
        raise config_error(
            "the requested run mode",
            "to be 'train' for this train-only experiment",
            mode.value,
        )
    student = resolve_student_spec(document.student, mode)
    if not isinstance(student, StudentTrainSpec):  # pragma: no cover
        raise RuntimeError("Expected the parent parser to resolve training.")
    return EndpointOptimizerRecoveryTrainSpec(
        experiment_id=EXPERIMENT_ID,
        protocol=document.protocol,
        student=student,
    )


__all__ = [
    "ALL_ENDPOINT_SEEDS",
    "DEVELOPMENT_ENDPOINT_SEED",
    "EVALUATION_ENDPOINT_SEEDS",
    "EXPERIMENT_ID",
    "EndpointOptimizerRecoveryTrainSpec",
    "NOMINAL_DELTA_X",
    "SCHEMA_VERSION",
    "TARGET_ASSIGNMENT_SEED",
    "parse_endpoint_optimizer_recovery_config",
    "resolve_endpoint_optimizer_recovery_spec",
]
