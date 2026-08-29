"""Strict config for exploratory deterministic QAT on Winsorized IBM OM cells."""

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
from experiments.mnist_relu_drn.ibm_om_baseline_spacing_pv_no_clip_config import (
    CommissioningContract,
    JointAssignmentRepairContract,
    OmDeviceContract,
)
from experiments.schema import RunMode, config_error


EXPERIMENT_ID = "mnist_ibm_om_winsorized_qat.v1"
SCHEMA_VERSION = 1
EXPECTED_TEACHER_WEIGHTS_SHA256 = (
    "9a961a77628e365b54fdf59304ec7fddf46f1f4834c4c03cecea9c204f837f52"
)
EXPECTED_TEACHER_WEIGHTS_PATH = "data/mnist_relu_teacher_fixed_init_20260816.pt"
DEVELOPMENT_ASSIGNMENT_SEED = 86001
EVALUATION_ASSIGNMENT_SEED = 87001
DEVELOPMENT_ENDPOINT_SEEDS = (88101, 88102, 88103, 88104, 88105)
EVALUATION_ENDPOINT_SEEDS = (89101, 89102, 89103, 89104, 89105)
BASELINE_POSITION_FRACTION = 0.0
SPACING_DELTA_X_MULTIPLIERS = (1, 2, 4)
FIXED_SCALE_FRACTIONS = (1.0, 1.0)
FIXED_LOGIT_GAIN = 14.12537544622754
NUM_EPOCHS = 10
LEARNING_RATES = (0.004411914893617021, 0.000011974808510638298)
RESET_READ_SAMPLES = 8

DEVELOPMENT_HARDWARE_INSTANCE_ID = (
    "feddd5a62ae3ace700dc46f155f54236c58fa769c0d4d6603613ed857b7a0150"
)
EVALUATION_HARDWARE_INSTANCE_ID = (
    "58fb1079e7bd60f53ba93ba61a3dca887352846b9f5a5768e9cd7bb72aed8492"
)
WINSORIZED_POPULATION_FINGERPRINT = (
    "7c5ef35d1d6ff77341be162c44372f0e81ada7dbf070567905fdb05b331e06a4"
)
WINSORIZED_POPULATION_SHA256 = (
    "8da388b299c0dc2ca50e1999b5f1891864c035913b119f012755cd00a1bfb551"
)
WINSORIZED_COMMISSIONING_SHA256 = (
    "4ad3406e161a5394aab955257c75e928a50980d98475ff52619e4b5c7862cc0b"
)
WINSORIZED_COMMISSIONING_SEED = 2309854359023799702
DEVELOPMENT_CALIBRATION_SHA256 = (
    "922fa46cab1970ef08e05dbde73ce7c3cc3a270e4daebfd80fdc3a3330deac65"
)
DEVELOPMENT_JOINT_ASSIGNMENT_SHA256 = (
    "e3f6f87ae70c2274c2fc696ee9d6a08b1dc15d3d05f69e7b8153a44245a9ea5f"
)
EVALUATION_JOINT_ASSIGNMENT_SHA256 = (
    "9ae7f255790b69f851d8733b62d23f540f7026ce7c52938baac0caa407c97228"
)
EVALUATION_WINSORIZED_POPULATION_FINGERPRINT = (
    "36fbad76021c7d93bd0a888981b6c062eea938f3839a2d50b1112b181eedcb7d"
)
EVALUATION_WINSORIZED_POPULATION_SHA256 = (
    "21707e1e28636bf10e0f6c6e560117a9ec740a235ac977bc56717df3a36be819"
)
EVALUATION_WINSORIZED_COMMISSIONING_SHA256 = (
    "85db62c9ff01396af0e28e30cdcc2d72596a71d270073c04bc3d8b5050030bd4"
)
EVALUATION_WINSORIZED_COMMISSIONING_SEED = 3612945087974632635

COMMON_SOURCE_WEIGHT_SHA256 = (
    "5d4f43f33f74b0b360e4813866aa3ce49916152e330d29649d6d628a32736ffa",
    "9748ca76aaf7dcdf03287e34bffe66a20d9579a6c4863c9df4a2d8e346d59bec",
)
COMMON_BASELINE_SHA256 = (
    "6dd31ca4b6b2c0bb65aa3ae44f31be30c9fa86c751c85f2dab4103468144228e",
    "c4357804a834c5ffadd4c90c6ab9a5ce718c73ad7d2b12cefad31b2e6c5a1d52",
)
COMMON_CONTINUOUS_CONDUCTANCE_SHA256 = (
    "bdf1f8813054ee174632d8610f86319bf57fb3ac964ebed88f77d5fb22e6b42e",
    "b9f674c3a07fd5e724a6e77844ad533671e0c9516896f5a794dc2e532fa2ef50",
)
DEVELOPMENT_MAPPING_SHA256_BY_SPACING = {
    1: "42c9ff51e292ee44fe3bea1292b47b07183421ecef4162418392c836457c34ab",
    2: "16167653510ae51c54a00a978ed6c2a74490707562e66f689b83a9d2ab037e20",
    4: "bc915f6e9abd32ec84f3ac274d83a6ba2618a181e9ffc5783ea306ff5c752915",
}
INITIAL_QUANTIZED_CONDUCTANCE_SHA256_BY_SPACING = {
    1: (
        "f4d7ca496b26cfdaac2c86a40a00797b1717be0fd316d5db9d839f7bc407afff",
        "6e49fd37cc792167960793024a0212d051e1456d2669dfebfc99b97d5397b8c7",
    ),
    2: (
        "32a87fb27feb380033a47832a8c1333f905cdacc886a1e47e3b53b5e766b3594",
        "06b318d343db4136beae49403795cc86e151acebfef8687a9a45fdf4fd887f30",
    ),
    4: (
        "f90edc1e2290e6911f483b46aff4b267a47ee0cd12f8bee4d1ad3d00d376aae0",
        "ceb774157fe767ef4ac8e17857830f8f2c306d736a63206c6a089810b0ecae72",
    ),
}
EVALUATION_BASELINE_SHA256 = (
    "409dd8881c296660a245c1d0e123a3bcc74cd84c845c93de69e64af939416e32",
    "6d2a6d080e7bb84f9abbfe7c6895254e50c2efc4e330d563f86758b9437eada9",
)
EVALUATION_CONTINUOUS_CONDUCTANCE_SHA256 = (
    "926b0a37902e8e1d8765f1ebfeeea7df651c414eee088029cfa6fe977bec8fb9",
    "3ecce84d3b8d1dbdf3d10da8327513a205506cdb7e965acc821a5ee8ca173729",
)
EVALUATION_MAPPING_SHA256_BY_SPACING = {
    1: "02a33841e95c5444446eea1d174badf204734bf7d5ecf622d773d635d7cca144",
    2: "29cfdf5af9c58db5ad7041b322ccc587b862d55d7a8a82b8eb364856c8dedded",
    4: "27704dbdc47c8fce279fc528fdad15e49ee1c7b7dfbde8ee64bd3ef12fa3899d",
}
EVALUATION_QUANTIZED_CONDUCTANCE_SHA256_BY_SPACING = {
    1: (
        "3d4f882bff945a65860466cbf360d099d43620d1f22f20931b49778f8240ed10",
        "d7d6a12200fd0b803bb183f055e42bda79cc5d92027a5c35257b3af5adf5c15a",
    ),
    2: (
        "312900d94a37a230c864597fead56f0d885a134e22ec5bd16dfc8228998568f7",
        "f7a0ad7ef5ca8ee24ea095eb9fdac4282c3e9df166b45d8ef90d67d77b57446f",
    ),
    4: (
        "74b361d6f50fc6c09f4b74d1e24a077b60b1fe1aa08e7e44cdcc997143673e83",
        "4e563eecfef88dc06ac94980b8151862bf2239e8d6bedc3252acee86dc251d61",
    ),
}


@dataclass(frozen=True)
class WinsorizedQatSourceContract:
    weights_role: str
    expected_teacher_weights_path: str
    expected_teacher_weights_sha256: str


@dataclass(frozen=True)
class WinsorizedQatAssignmentContract:
    development_assignment_seed: int
    evaluation_assignment_seed: int
    development_endpoint_seeds: tuple[int, ...]
    evaluation_endpoint_seeds: tuple[int, ...]


@dataclass(frozen=True)
class WinsorizedQatArtifactContract:
    source_exploratory_result_id: str
    development_hardware_instance_id: str
    evaluation_hardware_instance_id: str
    development_joint_assignment_sha256: str
    evaluation_joint_assignment_sha256: str
    development_winsorized_population_fingerprint: str
    evaluation_winsorized_population_fingerprint: str
    development_winsorized_population_sha256: str
    evaluation_winsorized_population_sha256: str
    development_winsorized_commissioning_sha256: str
    evaluation_winsorized_commissioning_sha256: str
    development_winsorized_commissioning_seed: int
    evaluation_winsorized_commissioning_seed: int
    commissioning_read_samples: int
    development_calibration_sha256: str


@dataclass(frozen=True)
class WinsorizedQatMappingContract:
    baseline_policy: str
    baseline_position_fraction: float
    baseline_definition: str
    destination_column_groups: tuple[tuple[str, str], tuple[str, str]]
    spacing_delta_x_multiplier: int
    delta_x_definition: str
    conductance_formula: str
    fixed_scale_fractions: tuple[float, float]
    fixed_logit_gain: float
    logical_rounding: str
    level_capacity_policy: str
    reference_policy: str
    circuit_accounting: str


@dataclass(frozen=True)
class WinsorizedQatInitializationContract:
    source_weight_sha256_by_layer: tuple[str, str]
    baseline_sha256_by_layer: tuple[str, str]
    continuous_conductance_sha256_by_layer: tuple[str, str]
    initial_quantized_conductance_sha256_by_layer: tuple[str, str]
    development_mapping_sha256: str
    clean_shadow_policy: str


@dataclass(frozen=True)
class WinsorizedQatEvaluationContract:
    same_hardware_role: str
    cross_hardware_role: str
    deployment_program_verify: bool
    inference_read_noise: bool
    development_mapping_sha256: str
    evaluation_mapping_sha256: str
    evaluation_baseline_sha256_by_layer: tuple[str, str]
    evaluation_continuous_conductance_sha256_by_layer: tuple[str, str]
    evaluation_initial_quantized_conductance_sha256_by_layer: tuple[str, str]


@dataclass(frozen=True)
class WinsorizedQatTrainingContract:
    epochs: int
    learning_rates: tuple[float, float]
    quantizer: str
    program_verify: bool
    read_noise: bool


@dataclass(frozen=True)
class WinsorizedQatExecutionContract:
    evidence_tier: str
    profile: str
    training_forward: str
    straight_through_estimator: str
    training_program_verify: bool
    inference_read_noise: bool
    retention_drift: bool
    checkpoint_policy: str


@dataclass(frozen=True)
class WinsorizedQatProtocol:
    source: WinsorizedQatSourceContract
    assignments: WinsorizedQatAssignmentContract
    artifacts: WinsorizedQatArtifactContract
    device: OmDeviceContract
    commissioning: CommissioningContract
    joint_assignment_repair: JointAssignmentRepairContract
    mapping: WinsorizedQatMappingContract
    initialization: WinsorizedQatInitializationContract
    training: WinsorizedQatTrainingContract
    evaluation: WinsorizedQatEvaluationContract
    execution: WinsorizedQatExecutionContract

    @property
    def development_assignment_seed(self) -> int:
        return self.assignments.development_assignment_seed

    @property
    def evaluation_assignment_seed(self) -> int:
        return self.assignments.evaluation_assignment_seed

    @property
    def development_endpoint_seeds(self) -> tuple[int, ...]:
        return self.assignments.development_endpoint_seeds

    @property
    def evaluation_endpoint_seeds(self) -> tuple[int, ...]:
        return self.assignments.evaluation_endpoint_seeds

    @property
    def spacing_delta_x_multiplier(self) -> int:
        return self.mapping.spacing_delta_x_multiplier

    @property
    def fixed_scale_fractions(self) -> tuple[float, float]:
        return self.mapping.fixed_scale_fractions

    @property
    def fixed_logit_gain(self) -> float:
        return self.mapping.fixed_logit_gain

    @property
    def learning_rates(self) -> tuple[float, float]:
        return self.training.learning_rates

    @property
    def epochs(self) -> int:
        return self.training.epochs


@dataclass(frozen=True)
class WinsorizedQatConfig:
    schema_version: int
    experiment_id: str
    protocol: WinsorizedQatProtocol
    student: StudentConfig


@dataclass(frozen=True)
class WinsorizedQatTrainSpec:
    experiment_id: str
    protocol: WinsorizedQatProtocol
    student: StudentTrainSpec


_SOURCE = {
    "weights_role": "frozen_relu_source_teacher_and_logical_master_initializer",
    "expected_teacher_weights_path": EXPECTED_TEACHER_WEIGHTS_PATH,
    "expected_teacher_weights_sha256": EXPECTED_TEACHER_WEIGHTS_SHA256,
}
_ASSIGNMENTS = {
    "development_assignment_seed": DEVELOPMENT_ASSIGNMENT_SEED,
    "evaluation_assignment_seed": EVALUATION_ASSIGNMENT_SEED,
    "development_endpoint_seeds": list(DEVELOPMENT_ENDPOINT_SEEDS),
    "evaluation_endpoint_seeds": list(EVALUATION_ENDPOINT_SEEDS),
}
_ARTIFACTS = {
    "source_exploratory_result_id": (
        "mnist-ibm-om-baseline-spacing-pv-winsorized-nominal-"
        "exploratory-20260828-v2"
    ),
    "development_hardware_instance_id": DEVELOPMENT_HARDWARE_INSTANCE_ID,
    "evaluation_hardware_instance_id": EVALUATION_HARDWARE_INSTANCE_ID,
    "development_joint_assignment_sha256": DEVELOPMENT_JOINT_ASSIGNMENT_SHA256,
    "evaluation_joint_assignment_sha256": EVALUATION_JOINT_ASSIGNMENT_SHA256,
    "development_winsorized_population_fingerprint": (
        WINSORIZED_POPULATION_FINGERPRINT
    ),
    "evaluation_winsorized_population_fingerprint": (
        EVALUATION_WINSORIZED_POPULATION_FINGERPRINT
    ),
    "development_winsorized_population_sha256": WINSORIZED_POPULATION_SHA256,
    "evaluation_winsorized_population_sha256": (
        EVALUATION_WINSORIZED_POPULATION_SHA256
    ),
    "development_winsorized_commissioning_sha256": (
        WINSORIZED_COMMISSIONING_SHA256
    ),
    "evaluation_winsorized_commissioning_sha256": (
        EVALUATION_WINSORIZED_COMMISSIONING_SHA256
    ),
    "development_winsorized_commissioning_seed": (
        WINSORIZED_COMMISSIONING_SEED
    ),
    "evaluation_winsorized_commissioning_seed": (
        EVALUATION_WINSORIZED_COMMISSIONING_SEED
    ),
    "commissioning_read_samples": RESET_READ_SAMPLES,
    "development_calibration_sha256": DEVELOPMENT_CALIBRATION_SHA256,
}
_DEVICE = {
    "evidence_class": "model_based_aihwkit_preset_bound_winsorization_control",
    "preset": "reram_array_om",
    "required_aihwkit_version": "1.1.0",
    "corruption_policy": "counterfactual_repaired",
    "native_coordinate": "raw_a_with_hard_support_truncated_to_[-1,1]",
}
_COMMISSIONING = {
    "samples_per_cell": RESET_READ_SAMPLES,
    "sample_sequence": (
        "after_bound_winsorization_initialize_at_winsorized_lower_bound_then_"
        "one_reset_pulse_and_apparent_read_per_sample"
    ),
    "read_coordinate": "raw_active_a_then_map_to_x=(a+1)/2",
    "baseline_estimator": "per_cell_arithmetic_mean",
    "bound_policy": "bound_mean_to_own_winsorized_support",
    "noise_policy": "preset_cycle_to_cycle_and_apparent_write_noise_enabled",
    "freeze_policy": (
        "deterministic_per_winsorized_identity_reused_across_training_"
        "spacing_and_endpoint_repeats"
    ),
}
_JOINT_ASSIGNMENT_REPAIR = {
    "scope": (
        "reuse_complete_quad_joint_identity_from_predecessor_before_winsorization"
    ),
    "donor_seed_derivation": (
        "derive_seed(assignment_seed,mnist_ibm_om_baseline_selection_joint_donor_v1)"
    ),
    "destination_traversal": "canonical_layer_then_logical_quad_order",
    "donor_traversal": "same_layer_unique_donor_quads_in_canonical_order",
    "eligibility": "reuse_frozen_predecessor_joint_identity_assignment_exactly",
    "identity_copy": (
        "copy_complete_four_cell_quad_then_winsorize_each_cell_bounds"
    ),
    "maximum_candidates_per_quad": 128,
    "exhaustion": "invalidate_assignment",
}
_MAPPING_COMMON = {
    "baseline_policy": "shared_destination_columns",
    "baseline_position_fraction": BASELINE_POSITION_FRACTION,
    "baseline_definition": "B_j=L_j+alpha*(U_j-L_j)",
    "destination_column_groups": [["G++", "G-+"], ["G+-", "G--"]],
    "delta_x_definition": "nominal_dw_min/2_in_x",
    "conductance_formula": "G=a_winsorized+1=2*x=B+n*(2*h*delta_x)",
    "fixed_scale_fractions": list(FIXED_SCALE_FRACTIONS),
    "fixed_logit_gain": FIXED_LOGIT_GAIN,
    "logical_rounding": "nearest_nonnegative_integer_half_away_from_zero",
    "level_capacity_policy": "whole_winsorized_support_levels_no_short_terminal_interval",
    "reference_policy": "absent_no_reference_subtraction",
    "circuit_accounting": "full_G_in_numerator_and_denominator",
}
_TRAINING = {
    "epochs": NUM_EPOCHS,
    "learning_rates": list(LEARNING_RATES),
    "quantizer": "deterministic_uniform_device_codebook_with_ste",
    "program_verify": False,
    "read_noise": False,
}
_EXECUTION = {
    "evidence_tier": "exploratory_noncanonical",
    "profile": "direct_local_cuda",
    "training_forward": "deterministic_ideal_quantized_winsorized_mapping",
    "straight_through_estimator": "identity_gradient_through_quantizer",
    "training_program_verify": False,
    "inference_read_noise": False,
    "retention_drift": False,
    "checkpoint_policy": "report_exact_epoch_10_save_best_development_as_diagnostic",
}

_STUDENT_RUNTIME = {
    "seed": 42,
    "data_seed": 42,
    "device": "cuda",
    "dtype": "float32",
}
_STUDENT_DATA = {
    "batch_size": 16,
    "validation_points": 5000,
    "num_points": None,
    "shuffle": True,
}
_STUDENT_MODEL = {
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
}
_STUDENT_SOLVER = {
    "inference_iterations": 4,
    "training_iterations": 4,
    "mode": "asynchronous",
    "overrelaxation_factor": 1.1,
}
_STUDENT_MAPPING = {
    "scale_fractions": [1.0],
    "scale_fraction_pairs": [[1.0, 1.0]],
    "range_placement": "lower",
    "calibration_examples": 1024,
    "calibration_batch_size": 128,
    "logit_gain_min": 0.001,
    "logit_gain_max": 1000.0,
    "logit_gain_steps": 121,
}
_STUDENT_TEACHER = {
    "type": "bias_free_relu",
    "initialization": "signed_weight_mapping",
}
_STUDENT_TRAIN = {
    "num_epochs": NUM_EPOCHS,
    "learning_rates": list(LEARNING_RATES),
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


def _same_json_value(provided: Any, expected: Any) -> bool:
    """Compare JSON values without bool/number coercion."""

    if type(provided) is not type(expected):
        return False
    if isinstance(expected, list):
        return len(provided) == len(expected) and all(
            _same_json_value(left, right)
            for left, right in zip(provided, expected)
        )
    if isinstance(expected, dict):
        return set(provided) == set(expected) and all(
            _same_json_value(provided[key], value)
            for key, value in expected.items()
        )
    return bool(provided == expected)


def _require_exact(
    raw: Mapping[str, Any], expected: Mapping[str, Any], *, path: str
) -> None:
    _keys(raw, path, set(expected))
    for name, expected_value in expected.items():
        if not _same_json_value(raw[name], expected_value):
            raise config_error(
                f"{path}.{name}", f"to equal {expected_value!r}", raw[name]
            )


def _parse_protocol(value: Any) -> WinsorizedQatProtocol:
    path = "config.winsorized_qat"
    raw = _object(value, path)
    _keys(
        raw,
        path,
        {
            "source",
            "assignments",
            "artifacts",
            "device",
            "commissioning",
            "joint_assignment_repair",
            "mapping",
            "initialization",
            "training",
            "evaluation",
            "execution",
        },
    )

    source = _object(raw["source"], f"{path}.source")
    assignments = _object(raw["assignments"], f"{path}.assignments")
    artifacts = _object(raw["artifacts"], f"{path}.artifacts")
    device = _object(raw["device"], f"{path}.device")
    commissioning = _object(raw["commissioning"], f"{path}.commissioning")
    joint_assignment_repair = _object(
        raw["joint_assignment_repair"], f"{path}.joint_assignment_repair"
    )
    mapping = _object(raw["mapping"], f"{path}.mapping")
    initialization = _object(raw["initialization"], f"{path}.initialization")
    training = _object(raw["training"], f"{path}.training")
    evaluation = _object(raw["evaluation"], f"{path}.evaluation")
    execution = _object(raw["execution"], f"{path}.execution")
    _require_exact(source, _SOURCE, path=f"{path}.source")
    _require_exact(assignments, _ASSIGNMENTS, path=f"{path}.assignments")
    _require_exact(artifacts, _ARTIFACTS, path=f"{path}.artifacts")
    _require_exact(device, _DEVICE, path=f"{path}.device")
    _require_exact(
        commissioning,
        _COMMISSIONING,
        path=f"{path}.commissioning",
    )
    _require_exact(
        joint_assignment_repair,
        _JOINT_ASSIGNMENT_REPAIR,
        path=f"{path}.joint_assignment_repair",
    )
    _require_exact(training, _TRAINING, path=f"{path}.training")
    _require_exact(execution, _EXECUTION, path=f"{path}.execution")

    mapping_fields = {
        **_MAPPING_COMMON,
        "spacing_delta_x_multiplier": mapping.get(
            "spacing_delta_x_multiplier"
        ),
    }
    spacing = mapping_fields["spacing_delta_x_multiplier"]
    if type(spacing) is not int or spacing not in SPACING_DELTA_X_MULTIPLIERS:
        raise config_error(
            f"{path}.mapping.spacing_delta_x_multiplier",
            f"to be one of {SPACING_DELTA_X_MULTIPLIERS!r}",
            spacing,
        )
    _require_exact(mapping, mapping_fields, path=f"{path}.mapping")

    expected_initialization = {
        "source_weight_sha256_by_layer": list(COMMON_SOURCE_WEIGHT_SHA256),
        "baseline_sha256_by_layer": list(COMMON_BASELINE_SHA256),
        "continuous_conductance_sha256_by_layer": list(
            COMMON_CONTINUOUS_CONDUCTANCE_SHA256
        ),
        "initial_quantized_conductance_sha256_by_layer": list(
            INITIAL_QUANTIZED_CONDUCTANCE_SHA256_BY_SPACING[spacing]
        ),
        "development_mapping_sha256": DEVELOPMENT_MAPPING_SHA256_BY_SPACING[
            spacing
        ],
        "clean_shadow_policy": (
            "one_common_alpha_0_continuous_full_G_state_for_all_spacings"
        ),
    }
    _require_exact(
        initialization,
        expected_initialization,
        path=f"{path}.initialization",
    )
    expected_evaluation = {
        "same_hardware_role": (
            "post_epoch_10_deploy_to_frozen_development_assignment_86001"
        ),
        "cross_hardware_role": (
            "post_epoch_10_deploy_to_frozen_heldout_assignment_87001"
        ),
        "deployment_program_verify": True,
        "inference_read_noise": False,
        "development_mapping_sha256": DEVELOPMENT_MAPPING_SHA256_BY_SPACING[
            spacing
        ],
        "evaluation_mapping_sha256": EVALUATION_MAPPING_SHA256_BY_SPACING[
            spacing
        ],
        "evaluation_baseline_sha256_by_layer": list(EVALUATION_BASELINE_SHA256),
        "evaluation_continuous_conductance_sha256_by_layer": list(
            EVALUATION_CONTINUOUS_CONDUCTANCE_SHA256
        ),
        "evaluation_initial_quantized_conductance_sha256_by_layer": list(
            EVALUATION_QUANTIZED_CONDUCTANCE_SHA256_BY_SPACING[spacing]
        ),
    }
    _require_exact(evaluation, expected_evaluation, path=f"{path}.evaluation")

    return WinsorizedQatProtocol(
        source=WinsorizedQatSourceContract(**_SOURCE),
        assignments=WinsorizedQatAssignmentContract(
            development_assignment_seed=DEVELOPMENT_ASSIGNMENT_SEED,
            evaluation_assignment_seed=EVALUATION_ASSIGNMENT_SEED,
            development_endpoint_seeds=DEVELOPMENT_ENDPOINT_SEEDS,
            evaluation_endpoint_seeds=EVALUATION_ENDPOINT_SEEDS,
        ),
        artifacts=WinsorizedQatArtifactContract(**_ARTIFACTS),
        device=OmDeviceContract(**_DEVICE),
        commissioning=CommissioningContract(**_COMMISSIONING),
        joint_assignment_repair=JointAssignmentRepairContract(
            **_JOINT_ASSIGNMENT_REPAIR
        ),
        mapping=WinsorizedQatMappingContract(
            **{
                **_MAPPING_COMMON,
                "destination_column_groups": tuple(
                    tuple(group)
                    for group in _MAPPING_COMMON["destination_column_groups"]
                ),
                "fixed_scale_fractions": FIXED_SCALE_FRACTIONS,
                "spacing_delta_x_multiplier": spacing,
            }
        ),
        initialization=WinsorizedQatInitializationContract(
            source_weight_sha256_by_layer=COMMON_SOURCE_WEIGHT_SHA256,
            baseline_sha256_by_layer=COMMON_BASELINE_SHA256,
            continuous_conductance_sha256_by_layer=(
                COMMON_CONTINUOUS_CONDUCTANCE_SHA256
            ),
            initial_quantized_conductance_sha256_by_layer=(
                INITIAL_QUANTIZED_CONDUCTANCE_SHA256_BY_SPACING[spacing]
            ),
            development_mapping_sha256=(
                DEVELOPMENT_MAPPING_SHA256_BY_SPACING[spacing]
            ),
            clean_shadow_policy=expected_initialization["clean_shadow_policy"],
        ),
        training=WinsorizedQatTrainingContract(
            epochs=NUM_EPOCHS,
            learning_rates=LEARNING_RATES,
            quantizer=_TRAINING["quantizer"],
            program_verify=False,
            read_noise=False,
        ),
        evaluation=WinsorizedQatEvaluationContract(
            same_hardware_role=expected_evaluation["same_hardware_role"],
            cross_hardware_role=expected_evaluation["cross_hardware_role"],
            deployment_program_verify=True,
            inference_read_noise=False,
            development_mapping_sha256=expected_evaluation[
                "development_mapping_sha256"
            ],
            evaluation_mapping_sha256=expected_evaluation[
                "evaluation_mapping_sha256"
            ],
            evaluation_baseline_sha256_by_layer=EVALUATION_BASELINE_SHA256,
            evaluation_continuous_conductance_sha256_by_layer=(
                EVALUATION_CONTINUOUS_CONDUCTANCE_SHA256
            ),
            evaluation_initial_quantized_conductance_sha256_by_layer=(
                EVALUATION_QUANTIZED_CONDUCTANCE_SHA256_BY_SPACING[spacing]
            ),
        ),
        execution=WinsorizedQatExecutionContract(**_EXECUTION),
    )


def _validate_student_contract(student: StudentConfig, raw: Mapping[str, Any]) -> None:
    if set(student.modes) != {RunMode.TRAIN.value}:
        raise config_error(
            "config.modes",
            "to define exactly the train mode",
            tuple(student.modes),
        )
    for name, expected in (
        ("runtime", _STUDENT_RUNTIME),
        ("data", _STUDENT_DATA),
        ("model", _STUDENT_MODEL),
        ("solver", _STUDENT_SOLVER),
        ("mapping", _STUDENT_MAPPING),
        ("teacher", _STUDENT_TEACHER),
    ):
        _require_exact(
            _object(raw[name], f"config.{name}"),
            expected,
            path=f"config.{name}",
        )
    modes = _object(raw["modes"], "config.modes")
    _keys(modes, "config.modes", {"train"})
    _require_exact(
        _object(modes["train"], "config.modes.train"),
        _STUDENT_TRAIN,
        path="config.modes.train",
    )


def parse_winsorized_qat_config(payload: Mapping[str, Any]) -> WinsorizedQatConfig:
    raw = _object(payload, "config")
    student_keys = {
        "schema_version",
        "experiment_id",
        "runtime",
        "data",
        "teacher",
        "model",
        "solver",
        "mapping",
        "modes",
    }
    _keys(raw, "config", student_keys | {"winsorized_qat"})
    if raw["schema_version"] != SCHEMA_VERSION:
        raise config_error(
            "config.schema_version", "to equal 1", raw["schema_version"]
        )
    if raw["experiment_id"] != EXPERIMENT_ID:
        raise config_error(
            "config.experiment_id",
            f"to equal {EXPERIMENT_ID!r}",
            raw["experiment_id"],
        )

    protocol = _parse_protocol(raw["winsorized_qat"])
    student_payload = {key: raw[key] for key in student_keys}
    student_payload["experiment_id"] = STUDENT_EXPERIMENT_ID
    student = parse_student_config(student_payload)
    _validate_student_contract(student, raw)
    return WinsorizedQatConfig(
        schema_version=SCHEMA_VERSION,
        experiment_id=EXPERIMENT_ID,
        protocol=protocol,
        student=student,
    )


def resolve_winsorized_qat_spec(
    document: WinsorizedQatConfig, mode: RunMode
) -> WinsorizedQatTrainSpec:
    if mode is not RunMode.TRAIN:
        raise config_error(
            "the requested run mode",
            "to be 'train' for this train-only experiment",
            mode.value,
        )
    student = resolve_student_spec(document.student, mode)
    if not isinstance(student, StudentTrainSpec):  # pragma: no cover
        raise RuntimeError("Expected the parent parser to resolve training.")
    return WinsorizedQatTrainSpec(
        experiment_id=EXPERIMENT_ID,
        protocol=document.protocol,
        student=student,
    )


__all__ = [
    "BASELINE_POSITION_FRACTION",
    "DEVELOPMENT_ASSIGNMENT_SEED",
    "EXPERIMENT_ID",
    "EXPECTED_TEACHER_WEIGHTS_SHA256",
    "FIXED_LOGIT_GAIN",
    "FIXED_SCALE_FRACTIONS",
    "LEARNING_RATES",
    "NUM_EPOCHS",
    "SCHEMA_VERSION",
    "SPACING_DELTA_X_MULTIPLIERS",
    "WinsorizedQatConfig",
    "WinsorizedQatProtocol",
    "WinsorizedQatTrainSpec",
    "parse_winsorized_qat_config",
    "resolve_winsorized_qat_spec",
]
