"""Strict configuration for the IBM-OM crossbar--digital-ReLU comparator."""

from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Any, Mapping

from experiments.mnist_relu.config import _integer, _keys, _number, _object
from experiments.schema import RunMode, config_error


EXPERIMENT_ID = "mnist_ibm_om_crossbar_relu.v1"
SCHEMA_VERSION = 1
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


@dataclass(frozen=True)
class RuntimeSettings:
    seed: int
    data_seed: int
    device: str
    dtype: str
    required_aihwkit_version: str


@dataclass(frozen=True)
class DataSettings:
    batch_size: int
    validation_points: int
    num_points: int | None
    shuffle: bool


@dataclass(frozen=True)
class ModelSettings:
    dims: tuple[int, int, int]
    bias: bool
    digital_non_linearity: str
    maximum_input_size: int
    maximum_output_size: int


@dataclass(frozen=True)
class SourceSettings:
    expected_teacher_weights_path: str
    expected_teacher_weights_sha256: str


@dataclass(frozen=True)
class DeviceSettings:
    preset: str
    evidence_class: str
    assignment_seed: int
    endpoint_seeds: tuple[int, ...]
    corruption_policy: str
    bound_policy: str
    reference_policy: str
    deterministic_codebook_pulses: int
    maximum_programming_pulses: int
    verify_tolerance_x: float
    controller: str


@dataclass(frozen=True)
class MappingSettings:
    weight_scaling_omega: tuple[float, float]
    scaling_policy: str
    out_of_bounds_policy: str
    codebook_tie_rule: str


@dataclass(frozen=True)
class OffchipForwardNoiseSettings:
    model: str
    base_state: str
    equation: str
    resampling: str
    samples_per_minibatch: int
    relative_scale: float
    seed: int
    gradient_estimator: str


@dataclass(frozen=True)
class OffchipSettings:
    policy: str
    training_protocol: str
    epochs: int
    logical_learning_rates: tuple[float, float]
    beta_1: float
    beta_2: float
    epsilon: float
    objective: str
    master_q_bounds: tuple[float, float]
    maximum_batches: int | None
    checkpoint_policy: str
    deployment_target: str
    epoch_evaluation: str
    forward_noise: OffchipForwardNoiseSettings | None


@dataclass(frozen=True)
class StarRecoverySettings:
    state_representation: str
    capture_forward_state: str
    target_quantization: str
    hidden_gain: float
    output_gain: float
    calibration_examples: int
    calibration_maximum_batches: int | None
    pulse_rule: str
    update_batching: str
    fault_transition: str
    fault_source_corruption_policy: str
    fault_source_preset_default_corrupt_devices_prob: float
    fault_source_enabled_corrupt_devices_prob: float
    fault_source_corrupt_devices_range: float
    fault_mask_access: str


@dataclass(frozen=True)
class SupervisedBpRecoverySettings:
    repair_examples: int
    label_source: str
    update_batching: str
    gradient_engine: str
    fault_transition: str
    fault_source_corruption_policy: str
    fault_source_preset_default_corrupt_devices_prob: float
    fault_source_enabled_corrupt_devices_prob: float
    fault_source_corrupt_devices_range: float
    fault_mask_access: str


@dataclass(frozen=True)
class SupervisedShadowPvRecoverySettings:
    repair_examples: int
    label_source: str
    update_batching: str
    gradient_engine: str
    optimizer_state: str
    optimizer_coordinate: str
    logical_learning_rates: tuple[float, float]
    shadow_initial_state: str
    shadow_bounds: str
    write_schedule: str
    writer: str
    maximum_programming_pulses: int
    verify_tolerance_x: float
    fault_transition: str
    fault_source_corruption_policy: str
    fault_source_preset_default_corrupt_devices_prob: float
    fault_source_enabled_corrupt_devices_prob: float
    fault_source_corrupt_devices_range: float
    fault_mask_access: str


@dataclass(frozen=True)
class SupervisedStochasticBpRecoverySettings:
    repair_examples: int
    label_source: str
    update_batching: str
    gradient_engine: str
    optimizer_state: str
    weight_state: str
    pulse_type: str
    desired_bl: int
    fixed_bl: bool
    update_bl_management: bool
    update_management: bool
    um_grad_scale: float
    bit_line_seed: int
    cumulative_pulse_cap: int | None
    final_program_verify: bool
    fault_transition: str
    fault_source_corruption_policy: str
    fault_source_preset_default_corrupt_devices_prob: float
    fault_source_enabled_corrupt_devices_prob: float
    fault_source_corrupt_devices_range: float
    fault_mask_access: str


@dataclass(frozen=True)
class TikiTakaV1RecoverySettings:
    algorithm: str
    gamma: float
    fast_lr: float
    transfer_every: int
    units_in_mbatch: bool
    n_reads_per_transfer: int
    transfer_selection: str
    transfer_lr: float
    scale_transfer_lr: bool
    transfer_columns: bool
    with_reset_prob: float
    random_selection: bool
    fast_preset: str
    fast_evidence_class: str
    fast_assignment_seed: int
    fast_endpoint_seeds: tuple[int, ...]
    fast_initialization: str
    fast_corruption_policy: str
    fast_preset_default_corrupt_devices_prob: float
    fast_enabled_corrupt_devices_prob: float
    fast_corrupt_devices_range: float


@dataclass(frozen=True)
class RecoverySettings:
    policy: str
    layer_scope: str
    epochs: int
    learning_rates_q: tuple[float, float]
    beta_1: float | None
    beta_2: float | None
    epsilon: float | None
    objective: str
    pulse_cap_per_cell: int | None
    maximum_batches: int | None
    star: StarRecoverySettings | None
    supervised_bp: SupervisedBpRecoverySettings | None
    supervised_shadow_pv: SupervisedShadowPvRecoverySettings | None
    supervised_stochastic_bp: SupervisedStochasticBpRecoverySettings | None
    tiki_taka: TikiTakaV1RecoverySettings | None


@dataclass(frozen=True)
class TransferTargetSettings:
    assignment_seed: int
    endpoint_seeds: tuple[int, ...]
    corruption_policy: str


@dataclass(frozen=True)
class TransferSettings:
    enabled: bool
    source_state: str
    targets: tuple[TransferTargetSettings, ...]


@dataclass(frozen=True)
class EvaluationSettings:
    evaluate_test: bool
    sample_limit: int | None
    maximum_validation_batches: int | None
    selection_metric: str


@dataclass(frozen=True)
class DrnReferenceSettings:
    path: str
    sha256: str
    architecture: str
    assignment_seed: int
    endpoint_seeds: tuple[int, ...]


@dataclass(frozen=True)
class CrossbarConfig:
    schema_version: int
    experiment_id: str
    runtime: RuntimeSettings
    data: DataSettings
    model: ModelSettings
    source: SourceSettings
    device: DeviceSettings
    mapping: MappingSettings
    offchip: OffchipSettings
    recovery: RecoverySettings
    transfer: TransferSettings
    evaluation: EvaluationSettings
    drn_reference: DrnReferenceSettings


@dataclass(frozen=True)
class CrossbarTrainSpec:
    experiment_id: str
    runtime: RuntimeSettings
    data: DataSettings
    model: ModelSettings
    source: SourceSettings
    device: DeviceSettings
    mapping: MappingSettings
    offchip: OffchipSettings
    recovery: RecoverySettings
    transfer: TransferSettings
    evaluation: EvaluationSettings
    drn_reference: DrnReferenceSettings


def _digest(value: Any, path: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise config_error(path, "to be a lowercase SHA-256 digest", value)
    return value


def _text(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise config_error(path, "to be a non-empty string", value)
    return value.strip()


def _boolean(value: Any, path: str) -> bool:
    if not isinstance(value, bool):
        raise config_error(path, "to be a boolean", value)
    return value


def _integer_tuple(
    value: Any,
    path: str,
    *,
    allow_empty: bool = False,
) -> tuple[int, ...]:
    if not isinstance(value, (list, tuple)) or (not value and not allow_empty):
        raise config_error(path, "to be a non-empty list of positive integers", value)
    result = tuple(
        _integer(item, f"{path}[{index}]", minimum=1)
        for index, item in enumerate(value)
    )
    if len(set(result)) != len(result):
        raise config_error(path, "to contain unique values", value)
    return result


def _pair(
    value: Any,
    path: str,
    *,
    minimum: float,
    maximum: float | None = None,
    positive: bool = False,
) -> tuple[float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise config_error(path, "to contain exactly two numbers", value)
    result = tuple(_number(item, f"{path}[{index}]", minimum=minimum) for index, item in enumerate(value))
    if positive and any(item <= 0.0 for item in result):
        raise config_error(path, "to contain two positive numbers", value)
    if maximum is not None and any(item > maximum for item in result):
        raise config_error(path, f"to contain values <= {maximum}", value)
    return (float(result[0]), float(result[1]))


def _optional_positive_integer(value: Any, path: str) -> int | None:
    return None if value is None else _integer(value, path, minimum=1)


def _parse_runtime(value: Any) -> RuntimeSettings:
    path = "config.runtime"
    raw = _object(value, path)
    _keys(raw, path, {"seed", "data_seed", "device", "dtype", "required_aihwkit_version"})
    if raw["device"] not in {"cpu", "cuda"}:
        raise config_error(f"{path}.device", "to be 'cpu' or 'cuda'", raw["device"])
    if raw["dtype"] != "float32":
        raise config_error(f"{path}.dtype", "to equal 'float32'", raw["dtype"])
    if raw["required_aihwkit_version"] != "1.1.0":
        raise config_error(
            f"{path}.required_aihwkit_version", "to equal '1.1.0'", raw["required_aihwkit_version"]
        )
    if raw["seed"] != 42 or raw["data_seed"] != 42:
        raise config_error(path, "to use matched runtime and data seeds 42", dict(raw))
    return RuntimeSettings(
        seed=_integer(raw["seed"], f"{path}.seed"),
        data_seed=_integer(raw["data_seed"], f"{path}.data_seed"),
        device=raw["device"],
        dtype="float32",
        required_aihwkit_version="1.1.0",
    )


def _parse_data(value: Any) -> DataSettings:
    path = "config.data"
    raw = _object(value, path)
    _keys(raw, path, {"batch_size", "validation_points", "num_points", "shuffle"})
    batch_size = _integer(raw["batch_size"], f"{path}.batch_size", minimum=1)
    validation_points = _integer(
        raw["validation_points"], f"{path}.validation_points", minimum=1
    )
    shuffle = _boolean(raw["shuffle"], f"{path}.shuffle")
    if batch_size != 16 or validation_points != 5000 or shuffle is not True:
        raise config_error(
            path,
            "to use matched batch_size=16, validation_points=5000, and shuffle=true",
            dict(raw),
        )
    return DataSettings(
        batch_size=batch_size,
        validation_points=validation_points,
        num_points=_optional_positive_integer(raw["num_points"], f"{path}.num_points"),
        shuffle=shuffle,
    )


def _parse_model(value: Any) -> ModelSettings:
    path = "config.model"
    raw = _object(value, path)
    _keys(
        raw,
        path,
        {"dims", "bias", "digital_non_linearity", "maximum_input_size", "maximum_output_size"},
    )
    if not isinstance(raw["dims"], (list, tuple)) or tuple(raw["dims"]) != (784, 50, 10):
        raise config_error(f"{path}.dims", "to equal [784, 50, 10]", raw["dims"])
    if raw["bias"] is not False:
        raise config_error(f"{path}.bias", "to be false", raw["bias"])
    if raw["digital_non_linearity"] != "relu":
        raise config_error(
            f"{path}.digital_non_linearity", "to equal 'relu'", raw["digital_non_linearity"]
        )
    maximum_input = _integer(raw["maximum_input_size"], f"{path}.maximum_input_size", minimum=1)
    maximum_output = _integer(raw["maximum_output_size"], f"{path}.maximum_output_size", minimum=1)
    if maximum_input != 512 or maximum_output != 512:
        raise config_error(path, "to use the standard 512-by-512 AIHWKit tile limits", dict(raw))
    return ModelSettings((784, 50, 10), False, "relu", 512, 512)


def _parse_source(value: Any) -> SourceSettings:
    path = "config.source"
    raw = _object(value, path)
    _keys(raw, path, {"expected_teacher_weights_path", "expected_teacher_weights_sha256"})
    return SourceSettings(
        expected_teacher_weights_path=_text(raw["expected_teacher_weights_path"], f"{path}.expected_teacher_weights_path"),
        expected_teacher_weights_sha256=_digest(raw["expected_teacher_weights_sha256"], f"{path}.expected_teacher_weights_sha256"),
    )


def _parse_device(value: Any) -> DeviceSettings:
    path = "config.device"
    raw = _object(value, path)
    _keys(
        raw,
        path,
        {
            "preset",
            "evidence_class",
            "assignment_seed",
            "endpoint_seeds",
            "corruption_policy",
            "bound_policy",
            "reference_policy",
            "deterministic_codebook_pulses",
            "maximum_programming_pulses",
            "verify_tolerance_x",
            "controller",
        },
    )
    expected = {
        "preset": "ReRamArrayOMPresetDevice",
        "evidence_class": "model_based_aihwkit_preset",
        "reference_policy": "fixed_sampled_intrinsic_reference_q_equals_a_minus_r",
        "controller": "one_pulse_apparent_verify_persistent_handoff",
    }
    for name, expected_value in expected.items():
        if raw[name] != expected_value:
            raise config_error(f"{path}.{name}", f"to equal {expected_value!r}", raw[name])
    if raw["corruption_policy"] not in {"published", "counterfactual_repaired"}:
        raise config_error(
            f"{path}.corruption_policy", "to be 'published' or 'counterfactual_repaired'", raw["corruption_policy"]
        )
    if raw["bound_policy"] not in {
        "native_sampled_bounds",
        "winsorize_raw_active_a_to_unit_interval",
    }:
        raise config_error(
            f"{path}.bound_policy",
            (
                "to be 'native_sampled_bounds' or "
                "'winsorize_raw_active_a_to_unit_interval'"
            ),
            raw["bound_policy"],
        )
    codebook_pulses = _integer(
        raw["deterministic_codebook_pulses"], f"{path}.deterministic_codebook_pulses", minimum=1
    )
    programming_pulses = _integer(
        raw["maximum_programming_pulses"], f"{path}.maximum_programming_pulses", minimum=1
    )
    if codebook_pulses != 128 or programming_pulses != 128:
        raise config_error(path, "to use the frozen 128-pulse deterministic and P&V caps", dict(raw))
    tolerance = _number(raw["verify_tolerance_x"], f"{path}.verify_tolerance_x")
    if tolerance <= 0.0:
        raise config_error(f"{path}.verify_tolerance_x", "to be positive", tolerance)
    if not math.isclose(float(tolerance), 0.023725, rel_tol=0.0, abs_tol=1e-12):
        raise config_error(
            f"{path}.verify_tolerance_x",
            "to equal the matched half-nominal-step tolerance 0.023725",
            tolerance,
        )
    return DeviceSettings(
        preset=expected["preset"],
        evidence_class=expected["evidence_class"],
        assignment_seed=_integer(raw["assignment_seed"], f"{path}.assignment_seed", minimum=1),
        endpoint_seeds=_integer_tuple(raw["endpoint_seeds"], f"{path}.endpoint_seeds"),
        corruption_policy=raw["corruption_policy"],
        bound_policy=raw["bound_policy"],
        reference_policy=expected["reference_policy"],
        deterministic_codebook_pulses=128,
        maximum_programming_pulses=128,
        verify_tolerance_x=float(tolerance),
        controller=expected["controller"],
    )


def _parse_mapping(value: Any) -> MappingSettings:
    path = "config.mapping"
    raw = _object(value, path)
    _keys(raw, path, {"weight_scaling_omega", "scaling_policy", "out_of_bounds_policy", "codebook_tie_rule"})
    expected = {
        "scaling_policy": "one_shared_absmax_scale_per_logical_layer",
        "out_of_bounds_policy": "retain_request_and_report_endpoint_saturation",
        "codebook_tie_rule": "lowest_pulse_index",
    }
    for name, expected_value in expected.items():
        if raw[name] != expected_value:
            raise config_error(f"{path}.{name}", f"to equal {expected_value!r}", raw[name])
    return MappingSettings(
        weight_scaling_omega=_pair(
            raw["weight_scaling_omega"], f"{path}.weight_scaling_omega", minimum=0.0, maximum=1.0, positive=True
        ),
        scaling_policy=expected["scaling_policy"],
        out_of_bounds_policy=expected["out_of_bounds_policy"],
        codebook_tie_rule=expected["codebook_tie_rule"],
    )


def _parse_offchip_forward_noise(
    value: Any,
    path: str,
) -> OffchipForwardNoiseSettings:
    raw = _object(value, path)
    _keys(
        raw,
        path,
        {
            "model",
            "base_state",
            "equation",
            "resampling",
            "samples_per_minibatch",
            "relative_scale",
            "seed",
            "gradient_estimator",
        },
    )
    expected = {
        "model": "aihwkit_1_1_0_softbounds_reference_additive_write_noise",
        "base_state": "support_clamped_persistent_q",
        "equation": (
            "q_apparent_equals_q_persistent_plus_write_noise_std_times_"
            "nominal_dw_min_times_standard_normal"
        ),
        "resampling": "independent_full_array_draw_per_minibatch",
        "samples_per_minibatch": 1,
        "relative_scale": 1.0,
        "gradient_estimator": "identity_straight_through",
    }
    for key, expected_value in expected.items():
        if raw[key] != expected_value:
            raise config_error(
                f"{path}.{key}",
                f"to equal {expected_value!r}",
                raw[key],
            )
    seed = _integer(raw["seed"], f"{path}.seed", minimum=1)
    return OffchipForwardNoiseSettings(
        model=expected["model"],
        base_state=expected["base_state"],
        equation=expected["equation"],
        resampling=expected["resampling"],
        samples_per_minibatch=1,
        relative_scale=1.0,
        seed=seed,
        gradient_estimator=expected["gradient_estimator"],
    )


def _parse_offchip(value: Any) -> OffchipSettings:
    path = "config.offchip"
    raw = _object(value, path)
    _keys(
        raw,
        path,
        {
            "policy",
            "epochs",
            "logical_learning_rates",
            "betas",
            "epsilon",
            "objective",
            "master_q_bounds",
            "maximum_batches",
            "checkpoint_policy",
        },
        {
            "forward_noise",
            "training_protocol",
            "deployment_target",
            "epoch_evaluation",
        },
    )
    supported_policies = {
        "none",
        "support_clamped_no_update",
        "continuous_hwa",
        "stochastic_apparent_hwa",
        "deterministic_qat",
    }
    if raw["policy"] not in supported_policies:
        raise config_error(
            f"{path}.policy",
            f"to be one of {sorted(supported_policies)!r}",
            raw["policy"],
        )
    policy = raw["policy"]
    epochs = _integer(raw["epochs"], f"{path}.epochs", minimum=0)
    rates = _pair(
        raw["logical_learning_rates"],
        f"{path}.logical_learning_rates",
        minimum=0.0,
    )
    betas = _pair(raw["betas"], f"{path}.betas", minimum=0.0)
    epsilon = _number(raw["epsilon"], f"{path}.epsilon")
    bounds = _pair(raw["master_q_bounds"], f"{path}.master_q_bounds", minimum=-1.0)
    if (
        betas != (0.9, 0.999)
        or not math.isclose(float(epsilon), 1e-8, rel_tol=0.0, abs_tol=1e-16)
        or raw["objective"] != "teacher_kl"
        or bounds != (-1.0, 1.0)
        or raw["checkpoint_policy"] != "fixed_final_epoch_no_selection"
    ):
        raise config_error(
            path,
            (
                "to use teacher-KL, Adam betas [0.9, 0.999], epsilon 1e-8, "
                "master q bounds [-1, 1], and a fixed final checkpoint"
            ),
            dict(raw),
        )
    forward_noise = (
        None
        if raw.get("forward_noise") is None
        else _parse_offchip_forward_noise(
            raw["forward_noise"], f"{path}.forward_noise"
        )
    )
    default_training_protocol = (
        "no_update"
        if policy in {"none", "support_clamped_no_update"}
        else "legacy_five_epoch_fixed_final"
    )
    training_protocol = raw.get(
        "training_protocol", default_training_protocol
    )
    supported_training_protocols = {
        "no_update",
        "legacy_five_epoch_fixed_final",
        "full_mnist_ten_epoch_fixed_final",
    }
    if training_protocol not in supported_training_protocols:
        raise config_error(
            f"{path}.training_protocol",
            f"to be one of {sorted(supported_training_protocols)!r}",
            training_protocol,
        )
    if policy in {"none", "support_clamped_no_update"}:
        if epochs != 0 or rates != (0.0, 0.0):
            raise config_error(
                path,
                "to use zero epochs and rates for a no-update offchip policy",
                dict(raw),
            )
        if training_protocol != "no_update":
            raise config_error(
                f"{path}.training_protocol",
                "to equal 'no_update' for an off-chip no-update policy",
                training_protocol,
            )
    elif training_protocol == "legacy_five_epoch_fixed_final":
        if epochs != 5 or rates != (1e-4, 1e-4):
            raise config_error(
                path,
                (
                    "to use the predeclared five epochs and logical learning "
                    "rates [1e-4, 1e-4] for legacy HWA/QAT"
                ),
                dict(raw),
            )
    elif training_protocol == "full_mnist_ten_epoch_fixed_final":
        if (
            policy not in {"continuous_hwa", "stochastic_apparent_hwa"}
            or epochs != 10
            or rates != (1e-4, 1e-4)
        ):
            raise config_error(
                path,
                (
                    "to use continuous or stochastic HWA for ten full-MNIST "
                    "fixed-final epochs at logical learning rates [1e-4, 1e-4]"
                ),
                dict(raw),
            )
    else:
        raise config_error(
            f"{path}.training_protocol",
            "not to equal 'no_update' for a trained off-chip policy",
            training_protocol,
        )
    if policy == "stochastic_apparent_hwa":
        if forward_noise is None:
            raise config_error(
                f"{path}.forward_noise",
                "to be present for stochastic_apparent_hwa",
                raw.get("forward_noise"),
            )
    elif forward_noise is not None:
        raise config_error(
            f"{path}.forward_noise",
            "to be null or absent outside stochastic_apparent_hwa",
            raw["forward_noise"],
        )
    deployment_target = raw.get(
        "deployment_target", "policy_realized_state"
    )
    if deployment_target not in {
        "policy_realized_state",
        "fixed_final_master_fault_blind_pv",
    }:
        raise config_error(
            f"{path}.deployment_target",
            (
                "to be 'policy_realized_state' or "
                "'fixed_final_master_fault_blind_pv'"
            ),
            deployment_target,
        )
    if (
        deployment_target == "fixed_final_master_fault_blind_pv"
        and policy
        not in {"none", "continuous_hwa", "stochastic_apparent_hwa"}
    ):
        raise config_error(
            f"{path}.deployment_target",
            (
                "to use the exact-master P&V handoff only for no-HWA, "
                "continuous-HWA, or stochastic-apparent-HWA policies"
            ),
            deployment_target,
        )
    epoch_evaluation = raw.get("epoch_evaluation", "validation_only")
    if epoch_evaluation not in {"validation_only", "validation_and_test"}:
        raise config_error(
            f"{path}.epoch_evaluation",
            "to be 'validation_only' or 'validation_and_test'",
            epoch_evaluation,
        )
    if (
        training_protocol == "full_mnist_ten_epoch_fixed_final"
        and epoch_evaluation != "validation_and_test"
    ):
        raise config_error(
            f"{path}.epoch_evaluation",
            "to equal 'validation_and_test' for the ten-epoch HWA diagnostic",
            epoch_evaluation,
        )
    return OffchipSettings(
        policy=policy,
        training_protocol=training_protocol,
        epochs=epochs,
        logical_learning_rates=rates,
        beta_1=betas[0],
        beta_2=betas[1],
        epsilon=float(epsilon),
        objective="teacher_kl",
        master_q_bounds=bounds,
        maximum_batches=_optional_positive_integer(
            raw["maximum_batches"], f"{path}.maximum_batches"
        ),
        checkpoint_policy=raw["checkpoint_policy"],
        deployment_target=deployment_target,
        epoch_evaluation=epoch_evaluation,
        forward_noise=forward_noise,
    )


def _parse_supervised_stochastic_bp(
    value: Any,
    path: str,
) -> SupervisedStochasticBpRecoverySettings:
    raw = _object(value, path)
    _keys(
        raw,
        path,
        {
            "repair_examples",
            "label_source",
            "update_batching",
            "gradient_engine",
            "optimizer_state",
            "weight_state",
            "pulse_type",
            "desired_bl",
            "fixed_bl",
            "update_bl_management",
            "update_management",
            "um_grad_scale",
            "bit_line_seed",
            "cumulative_pulse_cap",
            "final_program_verify",
            "fault_transition",
            "fault_source_corruption_policy",
            "fault_source_preset_default_corrupt_devices_prob",
            "fault_source_enabled_corrupt_devices_prob",
            "fault_source_corrupt_devices_range",
            "fault_mask_access",
        },
    )
    expected = {
        "label_source": "ground_truth",
        "update_batching": "minibatch",
        "gradient_engine": "manual_cross_entropy_backprop",
        "optimizer_state": "none",
        "weight_state": "physical_persistent_and_apparent_device_state_no_shadow",
        "pulse_type": "stochastic_compressed",
        "fault_transition": "post_deployment_published_companion_replay",
        "fault_source_corruption_policy": "published",
        "fault_mask_access": "forbidden",
    }
    for name, expected_value in expected.items():
        if raw[name] != expected_value:
            raise config_error(
                f"{path}.{name}", f"to equal {expected_value!r}", raw[name]
            )
    desired_bl = _integer(raw["desired_bl"], f"{path}.desired_bl", minimum=1)
    if desired_bl != 31:
        raise config_error(
            f"{path}.desired_bl",
            "to equal the frozen stochastic-compressed maximum bit-line length 31",
            raw["desired_bl"],
        )
    boolean_contract = {
        "fixed_bl": True,
        "update_bl_management": True,
        "update_management": True,
        "final_program_verify": False,
    }
    for name, expected_value in boolean_contract.items():
        parsed = _boolean(raw[name], f"{path}.{name}")
        if parsed is not expected_value:
            raise config_error(
                f"{path}.{name}", f"to be {str(expected_value).lower()}", raw[name]
            )
    um_grad_scale = _number(
        raw["um_grad_scale"], f"{path}.um_grad_scale", minimum=0.0
    )
    if not math.isclose(float(um_grad_scale), 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise config_error(
            f"{path}.um_grad_scale",
            "to equal the frozen update-management gradient scale 1.0",
            raw["um_grad_scale"],
        )
    bit_line_seed = _integer(
        raw["bit_line_seed"], f"{path}.bit_line_seed", minimum=1
    )
    if bit_line_seed != 108402:
        raise config_error(
            f"{path}.bit_line_seed",
            "to equal the frozen stochastic bit-line seed 108402",
            raw["bit_line_seed"],
        )
    if raw["cumulative_pulse_cap"] is not None:
        raise config_error(
            f"{path}.cumulative_pulse_cap",
            "to be null because this arm has no cumulative per-cell pulse cap",
            raw["cumulative_pulse_cap"],
        )
    fault_source_scalars = {
        "fault_source_preset_default_corrupt_devices_prob": 0.0,
        "fault_source_enabled_corrupt_devices_prob": 0.1348,
        "fault_source_corrupt_devices_range": 0.01,
    }
    for name, expected_value in fault_source_scalars.items():
        parsed = _number(raw[name], f"{path}.{name}", minimum=0.0)
        if not math.isclose(
            float(parsed), expected_value, rel_tol=0.0, abs_tol=1e-12
        ):
            raise config_error(
                f"{path}.{name}", f"to equal {expected_value!r}", raw[name]
            )
    return SupervisedStochasticBpRecoverySettings(
        repair_examples=_integer(
            raw["repair_examples"], f"{path}.repair_examples", minimum=1
        ),
        label_source=expected["label_source"],
        update_batching=expected["update_batching"],
        gradient_engine=expected["gradient_engine"],
        optimizer_state=expected["optimizer_state"],
        weight_state=expected["weight_state"],
        pulse_type=expected["pulse_type"],
        desired_bl=desired_bl,
        fixed_bl=True,
        update_bl_management=True,
        update_management=True,
        um_grad_scale=float(um_grad_scale),
        bit_line_seed=bit_line_seed,
        cumulative_pulse_cap=None,
        final_program_verify=False,
        fault_transition=expected["fault_transition"],
        fault_source_corruption_policy=expected[
            "fault_source_corruption_policy"
        ],
        fault_source_preset_default_corrupt_devices_prob=(
            fault_source_scalars[
                "fault_source_preset_default_corrupt_devices_prob"
            ]
        ),
        fault_source_enabled_corrupt_devices_prob=(
            fault_source_scalars["fault_source_enabled_corrupt_devices_prob"]
        ),
        fault_source_corrupt_devices_range=fault_source_scalars[
            "fault_source_corrupt_devices_range"
        ],
        fault_mask_access=expected["fault_mask_access"],
    )


def _parse_tiki_taka_v1(value: Any, path: str) -> TikiTakaV1RecoverySettings:
    raw = _object(value, path)
    _keys(
        raw,
        path,
        {
            "algorithm",
            "gamma",
            "fast_lr",
            "transfer_every",
            "units_in_mbatch",
            "n_reads_per_transfer",
            "transfer_selection",
            "transfer_lr",
            "scale_transfer_lr",
            "transfer_columns",
            "with_reset_prob",
            "random_selection",
            "fast_preset",
            "fast_evidence_class",
            "fast_assignment_seed",
            "fast_endpoint_seeds",
            "fast_initialization",
            "fast_corruption_policy",
            "fast_preset_default_corrupt_devices_prob",
            "fast_enabled_corrupt_devices_prob",
            "fast_corrupt_devices_range",
        },
    )
    expected = {
        "algorithm": "tiki_taka_transfer_compound_v1",
        "transfer_selection": "sequential_physical_tile_columns",
        "fast_preset": "ReRamArrayOMPresetDevice",
        "fast_evidence_class": "model_based_aihwkit_preset",
        "fast_initialization": "strict_q_zero_program_verify_mask_blind",
    }
    for name, expected_value in expected.items():
        if raw[name] != expected_value:
            raise config_error(
                f"{path}.{name}", f"to equal {expected_value!r}", raw[name]
            )
    if raw["fast_corruption_policy"] not in {
        "counterfactual_repaired",
        "published",
    }:
        raise config_error(
            f"{path}.fast_corruption_policy",
            "to be 'counterfactual_repaired' or 'published'",
            raw["fast_corruption_policy"],
        )
    scalar_contract = {
        "gamma": 0.0,
        "fast_lr": 1.0,
        "transfer_lr": 1.0,
        "with_reset_prob": 0.0,
        "fast_preset_default_corrupt_devices_prob": 0.0,
        "fast_enabled_corrupt_devices_prob": 0.1348,
        "fast_corrupt_devices_range": 0.01,
    }
    parsed_scalars: dict[str, float] = {}
    for name, expected_value in scalar_contract.items():
        parsed = _number(raw[name], f"{path}.{name}", minimum=0.0)
        if not math.isclose(
            float(parsed), expected_value, rel_tol=0.0, abs_tol=1e-12
        ):
            raise config_error(
                f"{path}.{name}", f"to equal {expected_value!r}", raw[name]
            )
        parsed_scalars[name] = float(parsed)
    integer_contract = {
        "transfer_every": 1,
        "n_reads_per_transfer": 1,
        "fast_assignment_seed": 88004,
    }
    parsed_integers: dict[str, int] = {}
    for name, expected_value in integer_contract.items():
        parsed = _integer(raw[name], f"{path}.{name}", minimum=1)
        if parsed != expected_value:
            raise config_error(
                f"{path}.{name}", f"to equal {expected_value}", raw[name]
            )
        parsed_integers[name] = parsed
    boolean_contract = {
        "units_in_mbatch": True,
        "scale_transfer_lr": True,
        "transfer_columns": True,
        "random_selection": False,
    }
    for name, expected_value in boolean_contract.items():
        parsed = _boolean(raw[name], f"{path}.{name}")
        if parsed is not expected_value:
            raise config_error(
                f"{path}.{name}", f"to be {str(expected_value).lower()}", raw[name]
            )
    fast_endpoint_seeds = _integer_tuple(
        raw["fast_endpoint_seeds"], f"{path}.fast_endpoint_seeds"
    )
    if fast_endpoint_seeds != (98402, 98403, 98404, 98405):
        raise config_error(
            f"{path}.fast_endpoint_seeds",
            "to equal the frozen independent fast-array seeds [98402, 98403, 98404, 98405]",
            raw["fast_endpoint_seeds"],
        )
    return TikiTakaV1RecoverySettings(
        algorithm=expected["algorithm"],
        gamma=parsed_scalars["gamma"],
        fast_lr=parsed_scalars["fast_lr"],
        transfer_every=parsed_integers["transfer_every"],
        units_in_mbatch=True,
        n_reads_per_transfer=parsed_integers["n_reads_per_transfer"],
        transfer_selection=expected["transfer_selection"],
        transfer_lr=parsed_scalars["transfer_lr"],
        scale_transfer_lr=True,
        transfer_columns=True,
        with_reset_prob=parsed_scalars["with_reset_prob"],
        random_selection=False,
        fast_preset=expected["fast_preset"],
        fast_evidence_class=expected["fast_evidence_class"],
        fast_assignment_seed=parsed_integers["fast_assignment_seed"],
        fast_endpoint_seeds=fast_endpoint_seeds,
        fast_initialization=expected["fast_initialization"],
        fast_corruption_policy=raw["fast_corruption_policy"],
        fast_preset_default_corrupt_devices_prob=parsed_scalars[
            "fast_preset_default_corrupt_devices_prob"
        ],
        fast_enabled_corrupt_devices_prob=parsed_scalars[
            "fast_enabled_corrupt_devices_prob"
        ],
        fast_corrupt_devices_range=parsed_scalars["fast_corrupt_devices_range"],
    )


def _parse_recovery(value: Any) -> RecoverySettings:
    path = "config.recovery"
    raw = _object(value, path)
    policy = raw.get("policy")
    common = {
        "policy",
        "layer_scope",
        "epochs",
        "objective",
        "maximum_batches",
    }
    if policy == "star_local_pulse_sgd":
        _keys(raw, path, common | {"learning_rates_q", "pulse_cap_per_cell", "star"})
    elif policy == "supervised_ce_pulse_adam":
        _keys(
            raw,
            path,
            common
            | {
                "learning_rates_q",
                "pulse_cap_per_cell",
                "betas",
                "epsilon",
                "supervised_bp",
            },
        )
    elif policy == "supervised_ce_shadow_program_verify":
        _keys(
            raw,
            path,
            common | {"betas", "epsilon", "supervised_shadow_pv"},
        )
    elif policy == "supervised_ce_stochastic_pulse_sgd":
        _keys(
            raw,
            path,
            common | {"learning_rates_q", "supervised_stochastic_bp"},
        )
    elif policy == "supervised_ce_tiki_taka_v1":
        _keys(
            raw,
            path,
            common
            | {
                "learning_rates_q",
                "supervised_stochastic_bp",
                "tiki_taka",
            },
        )
    else:
        _keys(
            raw,
            path,
            common | {"learning_rates_q", "pulse_cap_per_cell", "betas", "epsilon"},
        )
    if policy not in {
        "none",
        "pulse_adam",
        "star_local_pulse_sgd",
        "supervised_ce_pulse_adam",
        "supervised_ce_shadow_program_verify",
        "supervised_ce_stochastic_pulse_sgd",
        "supervised_ce_tiki_taka_v1",
    }:
        raise config_error(
            f"{path}.policy",
            (
                "to be 'none', 'pulse_adam', 'star_local_pulse_sgd', "
                "'supervised_ce_pulse_adam', or "
                "'supervised_ce_shadow_program_verify', "
                "'supervised_ce_stochastic_pulse_sgd', or "
                "'supervised_ce_tiki_taka_v1'"
            ),
            policy,
        )
    if raw["layer_scope"] not in {"all", "input_only", "output_only"}:
        raise config_error(
            f"{path}.layer_scope", "to be 'all', 'input_only', or 'output_only'", raw["layer_scope"]
        )
    epochs = _integer(raw["epochs"], f"{path}.epochs", minimum=0)
    if policy == "supervised_ce_shadow_program_verify":
        # The physical-coordinate rates depend on the deployed layer scales and
        # are therefore derived by the runtime from the declared logical rates.
        rates = (0.0, 0.0)
        pulse_cap = 0
    elif policy in {
        "supervised_ce_stochastic_pulse_sgd",
        "supervised_ce_tiki_taka_v1",
    }:
        rates = _pair(
            raw["learning_rates_q"],
            f"{path}.learning_rates_q",
            minimum=0.0,
            positive=True,
        )
        pulse_cap = None
    else:
        rates = _pair(raw["learning_rates_q"], f"{path}.learning_rates_q", minimum=0.0)
        pulse_cap = _integer(
            raw["pulse_cap_per_cell"], f"{path}.pulse_cap_per_cell", minimum=0
        )
    star = None
    supervised_bp = None
    supervised_shadow_pv = None
    supervised_stochastic_bp = None
    tiki_taka = None
    beta_1: float | None = None
    beta_2: float | None = None
    epsilon: float | None = None
    if policy in {
        "none",
        "pulse_adam",
        "supervised_ce_pulse_adam",
        "supervised_ce_shadow_program_verify",
    }:
        betas = _pair(raw["betas"], f"{path}.betas", minimum=0.0)
        if any(item >= 1.0 for item in betas):
            raise config_error(f"{path}.betas", "to contain values in [0, 1)", raw["betas"])
        parsed_epsilon = _number(raw["epsilon"], f"{path}.epsilon")
        if parsed_epsilon <= 0.0:
            raise config_error(f"{path}.epsilon", "to be positive", parsed_epsilon)
        if betas != (0.9, 0.999) or not math.isclose(
            float(parsed_epsilon), 1e-8, rel_tol=0.0, abs_tol=1e-16
        ):
            raise config_error(
                path,
                "to use the matched Adam betas [0.9, 0.999] and epsilon 1e-8",
                dict(raw),
            )
        expected_objective = (
            "cross_entropy"
            if policy in {
                "supervised_ce_pulse_adam",
                "supervised_ce_shadow_program_verify",
            }
            else "teacher_kl"
        )
        if raw["objective"] != expected_objective:
            raise config_error(
                f"{path}.objective",
                f"to equal {expected_objective!r}",
                raw["objective"],
            )
        beta_1, beta_2 = betas
        epsilon = float(parsed_epsilon)

    if policy == "none":
        if (
            epochs != 0
            or rates != (0.0, 0.0)
            or raw["layer_scope"] != "all"
            or pulse_cap != 0
        ):
            raise config_error(
                path,
                "to use zero epochs/rates/cap and scope='all' for policy='none'",
                dict(raw),
            )
    elif policy == "pulse_adam" and (
        epochs != 1 or rates != (6e-5, 6e-5) or pulse_cap != 64
    ):
        raise config_error(
            path,
            (
                "to use one epoch, q-coordinate learning rates [6e-5, 6e-5], "
                "and the matched 64-pulse per-cell recovery cap for pulse_adam"
            ),
            dict(raw),
        )
    elif policy == "star_local_pulse_sgd":
        if (
            epochs != 1
            or any(rate <= 0.0 for rate in rates)
            or pulse_cap < 1
            or raw["objective"] != "local_star_state_matching"
            or raw["layer_scope"] != "all"
        ):
            raise config_error(
                path,
                (
                    "to use exactly one epoch, positive rates/cap, scope='all', "
                    "and objective='local_star_state_matching' for STAR"
                ),
                dict(raw),
            )
        star_path = f"{path}.star"
        star_raw = _object(raw["star"], star_path)
        _keys(
            star_raw,
            star_path,
            {
                "state_representation",
                "capture_forward_state",
                "target_quantization",
                "hidden_gain",
                "output_gain",
                "calibration_examples",
                "calibration_maximum_batches",
                "pulse_rule",
                "update_batching",
                "fault_transition",
                "fault_source_corruption_policy",
                "fault_source_preset_default_corrupt_devices_prob",
                "fault_source_enabled_corrupt_devices_prob",
                "fault_source_corrupt_devices_range",
                "fault_mask_access",
            },
        )
        expected = {
            "state_representation": "crossbar.post_relu_hidden_and_output_logits.v1",
            "capture_forward_state": "apparent_q",
            "target_quantization": "fp16",
            "pulse_rule": "stochastic_pulse_sgd",
            "update_batching": "sequential_per_example",
            "fault_transition": "post_deployment_published_companion_replay",
            "fault_source_corruption_policy": "published",
            "fault_mask_access": "forbidden",
        }
        for name, expected_value in expected.items():
            if star_raw[name] != expected_value:
                raise config_error(
                    f"{star_path}.{name}", f"to equal {expected_value!r}", star_raw[name]
                )
        hidden_gain = _number(star_raw["hidden_gain"], f"{star_path}.hidden_gain", minimum=0.0)
        output_gain = _number(star_raw["output_gain"], f"{star_path}.output_gain", minimum=0.0)
        if hidden_gain <= 0.0 or output_gain <= 0.0:
            raise config_error(star_path, "to use positive hidden and output gains", dict(star_raw))
        fault_source_scalars = {
            "fault_source_preset_default_corrupt_devices_prob": 0.0,
            "fault_source_enabled_corrupt_devices_prob": 0.1348,
            "fault_source_corrupt_devices_range": 0.01,
        }
        for name, expected_value in fault_source_scalars.items():
            value = _number(star_raw[name], f"{star_path}.{name}", minimum=0.0)
            if not math.isclose(
                float(value), expected_value, rel_tol=0.0, abs_tol=1e-12
            ):
                raise config_error(
                    f"{star_path}.{name}",
                    f"to equal {expected_value!r}",
                    star_raw[name],
                )
        star = StarRecoverySettings(
            state_representation=expected["state_representation"],
            capture_forward_state=expected["capture_forward_state"],
            target_quantization=expected["target_quantization"],
            hidden_gain=float(hidden_gain),
            output_gain=float(output_gain),
            calibration_examples=_integer(
                star_raw["calibration_examples"],
                f"{star_path}.calibration_examples",
                minimum=1,
            ),
            calibration_maximum_batches=_optional_positive_integer(
                star_raw["calibration_maximum_batches"],
                f"{star_path}.calibration_maximum_batches",
            ),
            pulse_rule=expected["pulse_rule"],
            update_batching=expected["update_batching"],
            fault_transition=expected["fault_transition"],
            fault_source_corruption_policy=expected["fault_source_corruption_policy"],
            fault_source_preset_default_corrupt_devices_prob=(
                fault_source_scalars[
                    "fault_source_preset_default_corrupt_devices_prob"
                ]
            ),
            fault_source_enabled_corrupt_devices_prob=(
                fault_source_scalars["fault_source_enabled_corrupt_devices_prob"]
            ),
            fault_source_corrupt_devices_range=fault_source_scalars[
                "fault_source_corrupt_devices_range"
            ],
            fault_mask_access=expected["fault_mask_access"],
        )
    elif policy == "supervised_ce_pulse_adam":
        if (
            epochs < 1
            or any(rate <= 0.0 for rate in rates)
            or pulse_cap < 1
            or raw["layer_scope"] != "all"
        ):
            raise config_error(
                path,
                (
                    "to use positive epochs/rates/cap and scope='all' for "
                    "supervised cross-entropy retraining"
                ),
                dict(raw),
            )
        bp_path = f"{path}.supervised_bp"
        bp_raw = _object(raw["supervised_bp"], bp_path)
        _keys(
            bp_raw,
            bp_path,
            {
                "repair_examples",
                "label_source",
                "update_batching",
                "gradient_engine",
                "fault_transition",
                "fault_source_corruption_policy",
                "fault_source_preset_default_corrupt_devices_prob",
                "fault_source_enabled_corrupt_devices_prob",
                "fault_source_corrupt_devices_range",
                "fault_mask_access",
            },
        )
        expected = {
            "label_source": "ground_truth",
            "update_batching": "minibatch",
            "gradient_engine": "autograd_full_network_backprop",
            "fault_transition": "post_deployment_published_companion_replay",
            "fault_source_corruption_policy": "published",
            "fault_mask_access": "forbidden",
        }
        for name, expected_value in expected.items():
            if bp_raw[name] != expected_value:
                raise config_error(
                    f"{bp_path}.{name}",
                    f"to equal {expected_value!r}",
                    bp_raw[name],
                )
        fault_source_scalars = {
            "fault_source_preset_default_corrupt_devices_prob": 0.0,
            "fault_source_enabled_corrupt_devices_prob": 0.1348,
            "fault_source_corrupt_devices_range": 0.01,
        }
        for name, expected_value in fault_source_scalars.items():
            parsed = _number(bp_raw[name], f"{bp_path}.{name}", minimum=0.0)
            if not math.isclose(
                float(parsed), expected_value, rel_tol=0.0, abs_tol=1e-12
            ):
                raise config_error(
                    f"{bp_path}.{name}",
                    f"to equal {expected_value!r}",
                    bp_raw[name],
                )
        supervised_bp = SupervisedBpRecoverySettings(
            repair_examples=_integer(
                bp_raw["repair_examples"],
                f"{bp_path}.repair_examples",
                minimum=1,
            ),
            label_source=expected["label_source"],
            update_batching=expected["update_batching"],
            gradient_engine=expected["gradient_engine"],
            fault_transition=expected["fault_transition"],
            fault_source_corruption_policy=expected[
                "fault_source_corruption_policy"
            ],
            fault_source_preset_default_corrupt_devices_prob=(
                fault_source_scalars[
                    "fault_source_preset_default_corrupt_devices_prob"
                ]
            ),
            fault_source_enabled_corrupt_devices_prob=(
                fault_source_scalars[
                    "fault_source_enabled_corrupt_devices_prob"
                ]
            ),
            fault_source_corrupt_devices_range=fault_source_scalars[
                "fault_source_corrupt_devices_range"
            ],
            fault_mask_access=expected["fault_mask_access"],
        )
    elif policy == "supervised_ce_shadow_program_verify":
        if epochs != 1 or raw["layer_scope"] != "all":
            raise config_error(
                path,
                (
                    "to use exactly one epoch and scope='all' for supervised "
                    "continuous-shadow retraining with final program-and-verify"
                ),
                dict(raw),
            )
        shadow_path = f"{path}.supervised_shadow_pv"
        shadow_raw = _object(raw["supervised_shadow_pv"], shadow_path)
        _keys(
            shadow_raw,
            shadow_path,
            {
                "repair_examples",
                "label_source",
                "update_batching",
                "gradient_engine",
                "optimizer_state",
                "optimizer_coordinate",
                "logical_learning_rates",
                "shadow_initial_state",
                "shadow_bounds",
                "write_schedule",
                "writer",
                "maximum_programming_pulses",
                "verify_tolerance_x",
                "fault_transition",
                "fault_source_corruption_policy",
                "fault_source_preset_default_corrupt_devices_prob",
                "fault_source_enabled_corrupt_devices_prob",
                "fault_source_corrupt_devices_range",
                "fault_mask_access",
            },
        )
        expected = {
            "label_source": "ground_truth",
            "update_batching": "minibatch",
            "gradient_engine": "autograd_full_network_backprop",
            "optimizer_state": "digital_fp32_shadow_and_adam",
            "optimizer_coordinate": (
                "logical_weight_learning_rate_converted_to_q"
            ),
            "shadow_initial_state": "post_fault_apparent_q",
            "shadow_bounds": "healthy_source_population_q_bounds",
            "write_schedule": "fixed_final_epoch_program_verify",
            "writer": "one_pulse_apparent_verify_persistent_handoff",
            "fault_transition": "post_deployment_published_companion_replay",
            "fault_source_corruption_policy": "published",
            "fault_mask_access": "forbidden",
        }
        for name, expected_value in expected.items():
            if shadow_raw[name] != expected_value:
                raise config_error(
                    f"{shadow_path}.{name}",
                    f"to equal {expected_value!r}",
                    shadow_raw[name],
                )
        logical_rates = _pair(
            shadow_raw["logical_learning_rates"],
            f"{shadow_path}.logical_learning_rates",
            minimum=0.0,
        )
        if logical_rates != (1e-3, 1e-3):
            raise config_error(
                f"{shadow_path}.logical_learning_rates",
                "to equal the predeclared logical rates [0.001, 0.001]",
                shadow_raw["logical_learning_rates"],
            )
        maximum_programming_pulses = _integer(
            shadow_raw["maximum_programming_pulses"],
            f"{shadow_path}.maximum_programming_pulses",
            minimum=1,
        )
        if maximum_programming_pulses != 128:
            raise config_error(
                f"{shadow_path}.maximum_programming_pulses",
                "to equal the predeclared 128-pulse program-and-verify cap",
                shadow_raw["maximum_programming_pulses"],
            )
        verify_tolerance_x = _number(
            shadow_raw["verify_tolerance_x"],
            f"{shadow_path}.verify_tolerance_x",
            minimum=0.0,
        )
        if not math.isclose(
            float(verify_tolerance_x), 0.023725, rel_tol=0.0, abs_tol=1e-12
        ):
            raise config_error(
                f"{shadow_path}.verify_tolerance_x",
                "to equal the predeclared apparent-x tolerance 0.023725",
                shadow_raw["verify_tolerance_x"],
            )
        fault_source_scalars = {
            "fault_source_preset_default_corrupt_devices_prob": 0.0,
            "fault_source_enabled_corrupt_devices_prob": 0.1348,
            "fault_source_corrupt_devices_range": 0.01,
        }
        for name, expected_value in fault_source_scalars.items():
            parsed = _number(shadow_raw[name], f"{shadow_path}.{name}", minimum=0.0)
            if not math.isclose(
                float(parsed), expected_value, rel_tol=0.0, abs_tol=1e-12
            ):
                raise config_error(
                    f"{shadow_path}.{name}",
                    f"to equal {expected_value!r}",
                    shadow_raw[name],
                )
        supervised_shadow_pv = SupervisedShadowPvRecoverySettings(
            repair_examples=_integer(
                shadow_raw["repair_examples"],
                f"{shadow_path}.repair_examples",
                minimum=1,
            ),
            label_source=expected["label_source"],
            update_batching=expected["update_batching"],
            gradient_engine=expected["gradient_engine"],
            optimizer_state=expected["optimizer_state"],
            optimizer_coordinate=expected["optimizer_coordinate"],
            logical_learning_rates=logical_rates,
            shadow_initial_state=expected["shadow_initial_state"],
            shadow_bounds=expected["shadow_bounds"],
            write_schedule=expected["write_schedule"],
            writer=expected["writer"],
            maximum_programming_pulses=maximum_programming_pulses,
            verify_tolerance_x=float(verify_tolerance_x),
            fault_transition=expected["fault_transition"],
            fault_source_corruption_policy=expected[
                "fault_source_corruption_policy"
            ],
            fault_source_preset_default_corrupt_devices_prob=(
                fault_source_scalars[
                    "fault_source_preset_default_corrupt_devices_prob"
                ]
            ),
            fault_source_enabled_corrupt_devices_prob=(
                fault_source_scalars[
                    "fault_source_enabled_corrupt_devices_prob"
                ]
            ),
            fault_source_corrupt_devices_range=fault_source_scalars[
                "fault_source_corrupt_devices_range"
            ],
            fault_mask_access=expected["fault_mask_access"],
        )
        pulse_cap = maximum_programming_pulses
    elif policy in {
        "supervised_ce_stochastic_pulse_sgd",
        "supervised_ce_tiki_taka_v1",
    }:
        if (
            epochs != 1
            or raw["objective"] != "cross_entropy"
            or raw["layer_scope"] != "all"
        ):
            raise config_error(
                path,
                (
                    "to use exactly one epoch, objective='cross_entropy', "
                    "and scope='all' for stochastic-pulse supervised recovery"
                ),
                dict(raw),
            )
        supervised_stochastic_bp = _parse_supervised_stochastic_bp(
            raw["supervised_stochastic_bp"],
            f"{path}.supervised_stochastic_bp",
        )
        if policy == "supervised_ce_tiki_taka_v1":
            tiki_taka = _parse_tiki_taka_v1(
                raw["tiki_taka"], f"{path}.tiki_taka"
            )
    return RecoverySettings(
        policy=policy,
        layer_scope=raw["layer_scope"],
        epochs=epochs,
        learning_rates_q=rates,
        beta_1=beta_1,
        beta_2=beta_2,
        epsilon=epsilon,
        objective=raw["objective"],
        pulse_cap_per_cell=pulse_cap,
        maximum_batches=_optional_positive_integer(raw["maximum_batches"], f"{path}.maximum_batches"),
        star=star,
        supervised_bp=supervised_bp,
        supervised_shadow_pv=supervised_shadow_pv,
        supervised_stochastic_bp=supervised_stochastic_bp,
        tiki_taka=tiki_taka,
    )


def _parse_transfer(value: Any, *, endpoint_count: int, source_assignment: int) -> TransferSettings:
    path = "config.transfer"
    raw = _object(value, path)
    _keys(raw, path, {"enabled", "source_state", "targets"})
    enabled = _boolean(raw["enabled"], f"{path}.enabled")
    source_state = raw["source_state"]
    targets_raw = raw["targets"]
    if not isinstance(targets_raw, (list, tuple)):
        raise config_error(f"{path}.targets", "to be a list", targets_raw)
    if enabled:
        if source_state not in {
            "offchip_fixed_final_master",
            "same_array_persistent",
        }:
            raise config_error(
                f"{path}.source_state",
                (
                    "to be 'offchip_fixed_final_master' or "
                    "'same_array_persistent' when transfer is enabled"
                ),
                source_state,
            )
        if not targets_raw:
            raise config_error(
                f"{path}.targets",
                "to contain at least one fresh target array when enabled",
                targets_raw,
            )
        targets: list[TransferTargetSettings] = []
        assignments: list[int] = []
        for index, item in enumerate(targets_raw):
            target_path = f"{path}.targets[{index}]"
            target = _object(item, target_path)
            _keys(
                target,
                target_path,
                {"assignment_seed", "endpoint_seeds"},
                {"corruption_policy"},
            )
            assignment = _integer(
                target["assignment_seed"],
                f"{target_path}.assignment_seed",
                minimum=1,
            )
            endpoints = _integer_tuple(
                target["endpoint_seeds"],
                f"{target_path}.endpoint_seeds",
            )
            if assignment == source_assignment:
                raise config_error(
                    f"{target_path}.assignment_seed",
                    "to differ from the source assignment",
                    assignment,
                )
            if len(endpoints) != endpoint_count:
                raise config_error(
                    f"{target_path}.endpoint_seeds",
                    "to contain one target endpoint per source endpoint",
                    target["endpoint_seeds"],
                )
            corruption_policy = target.get("corruption_policy", "inherit_source")
            if corruption_policy not in {
                "inherit_source",
                "published",
                "counterfactual_repaired",
            }:
                raise config_error(
                    f"{target_path}.corruption_policy",
                    (
                        "to be 'inherit_source', 'published', or "
                        "'counterfactual_repaired'"
                    ),
                    corruption_policy,
                )
            assignments.append(assignment)
            targets.append(
                TransferTargetSettings(assignment, endpoints, corruption_policy)
            )
        if len(set(assignments)) != len(assignments):
            raise config_error(
                f"{path}.targets",
                "to contain unique target assignment seeds",
                targets_raw,
            )
        return TransferSettings(True, source_state, tuple(targets))
    if source_state != "none" or targets_raw:
        raise config_error(
            path,
            "to use source_state='none' and targets=[] when disabled",
            dict(raw),
        )
    return TransferSettings(False, "none", ())


def _parse_evaluation(value: Any) -> EvaluationSettings:
    path = "config.evaluation"
    raw = _object(value, path)
    _keys(raw, path, {"evaluate_test", "sample_limit", "maximum_validation_batches", "selection_metric"})
    if raw["selection_metric"] != "fixed_final_epoch_no_selection":
        raise config_error(
            f"{path}.selection_metric",
            "to equal 'fixed_final_epoch_no_selection'",
            raw["selection_metric"],
        )
    return EvaluationSettings(
        evaluate_test=_boolean(raw["evaluate_test"], f"{path}.evaluate_test"),
        sample_limit=_optional_positive_integer(raw["sample_limit"], f"{path}.sample_limit"),
        maximum_validation_batches=_optional_positive_integer(
            raw["maximum_validation_batches"], f"{path}.maximum_validation_batches"
        ),
        selection_metric=raw["selection_metric"],
    )


def _parse_drn_reference(value: Any) -> DrnReferenceSettings:
    path = "config.drn_reference"
    raw = _object(value, path)
    _keys(raw, path, {"path", "sha256", "architecture", "assignment_seed", "endpoint_seeds"})
    if raw["architecture"] != "four_device_full_G_perfect_diode_drn":
        raise config_error(
            f"{path}.architecture", "to equal 'four_device_full_G_perfect_diode_drn'", raw["architecture"]
        )
    return DrnReferenceSettings(
        path=_text(raw["path"], f"{path}.path"),
        sha256=_digest(raw["sha256"], f"{path}.sha256"),
        architecture=raw["architecture"],
        assignment_seed=_integer(raw["assignment_seed"], f"{path}.assignment_seed", minimum=1),
        endpoint_seeds=_integer_tuple(raw["endpoint_seeds"], f"{path}.endpoint_seeds"),
    )


def parse_crossbar_config(payload: Mapping[str, Any]) -> CrossbarConfig:
    raw = _object(payload, "config")
    required = {
        "schema_version",
        "experiment_id",
        "runtime",
        "data",
        "model",
        "source",
        "device",
        "mapping",
        "offchip",
        "recovery",
        "transfer",
        "evaluation",
        "drn_reference",
    }
    _keys(raw, "config", required)
    if raw["schema_version"] != SCHEMA_VERSION:
        raise config_error("config.schema_version", "to equal 1", raw["schema_version"])
    if raw["experiment_id"] != EXPERIMENT_ID:
        raise config_error("config.experiment_id", f"to equal {EXPERIMENT_ID!r}", raw["experiment_id"])
    device = _parse_device(raw["device"])
    transfer = _parse_transfer(
        raw["transfer"], endpoint_count=len(device.endpoint_seeds), source_assignment=device.assignment_seed
    )
    recovery = _parse_recovery(raw["recovery"])
    offchip = _parse_offchip(raw["offchip"])
    data = _parse_data(raw["data"])
    evaluation = _parse_evaluation(raw["evaluation"])
    if offchip.training_protocol == "full_mnist_ten_epoch_fixed_final":
        available = 60_000 - data.validation_points
        expected_batches = math.ceil(available / data.batch_size)
        if (
            data.num_points is not None
            or offchip.maximum_batches != expected_batches
            or not evaluation.evaluate_test
            or evaluation.sample_limit != 1_000
            or evaluation.maximum_validation_batches != math.ceil(
                1_000 / data.batch_size
            )
        ):
            raise config_error(
                "config",
                (
                    "to use all 55,000 training examples for ten fixed-final "
                    f"HWA epochs with maximum_batches={expected_batches}, "
                    "and epoch diagnostics on exactly 1,000 validation and "
                    "1,000 test examples"
                ),
                dict(raw),
            )
    if (
        offchip.epoch_evaluation == "validation_and_test"
        and not evaluation.evaluate_test
    ):
        raise config_error(
            "config.offchip.epoch_evaluation",
            "to require config.evaluation.evaluate_test=true",
            offchip.epoch_evaluation,
        )
    if recovery.policy == "star_local_pulse_sgd" and (
        device.corruption_policy != "counterfactual_repaired"
        or transfer.enabled
        or offchip.policy != "continuous_hwa"
        or recovery.maximum_batches != offchip.maximum_batches
    ):
        raise config_error(
            "config",
            (
                "to start STAR from a counterfactually repaired healthy array "
                "after continuous HWA, reuse its first-epoch batch budget, "
                "and keep fresh-array transfer disabled"
            ),
            dict(raw),
        )
    if recovery.policy in {
        "supervised_ce_pulse_adam",
        "supervised_ce_shadow_program_verify",
        "supervised_ce_stochastic_pulse_sgd",
        "supervised_ce_tiki_taka_v1",
    } and (
        device.corruption_policy != "counterfactual_repaired"
        or transfer.enabled
        or offchip.policy != "continuous_hwa"
    ):
        raise config_error(
            "config",
            (
                "to start supervised post-fault retraining from a "
                "counterfactually repaired healthy array after continuous HWA "
                "and keep fresh-array transfer disabled"
            ),
            dict(raw),
        )
    if recovery.policy in {
        "supervised_ce_stochastic_pulse_sgd",
        "supervised_ce_tiki_taka_v1",
    }:
        stochastic_settings = recovery.supervised_stochastic_bp
        if stochastic_settings is None:  # pragma: no cover - parser invariant
            raise RuntimeError(
                "Expected supervised stochastic-BP settings after parsing."
            )
        available = 60_000 - data.validation_points
        expected_batches = math.ceil(available / data.batch_size)
        if (
            stochastic_settings.repair_examples != available
            or recovery.maximum_batches != expected_batches
            or data.num_points != 16
            or offchip.maximum_batches != 1
        ):
            raise config_error(
                "config.recovery",
                (
                    "to consume exactly one complete 55,000-example "
                    "stochastic-pulse repair epoch with maximum_batches="
                    f"ceil(55000/{data.batch_size}) while preserving the "
                    "exact prior P0 source contract data.num_points=16 and "
                    "offchip.maximum_batches=1"
                ),
                dict(raw["recovery"]),
            )
        if recovery.policy == "supervised_ce_tiki_taka_v1":
            tiki_taka = recovery.tiki_taka
            if tiki_taka is None:  # pragma: no cover - parser invariant
                raise RuntimeError("Expected Tiki-Taka v1 settings after parsing.")
            if (
                tiki_taka.fast_assignment_seed == device.assignment_seed
                or len(tiki_taka.fast_endpoint_seeds)
                != len(device.endpoint_seeds)
                or set(tiki_taka.fast_endpoint_seeds) & set(device.endpoint_seeds)
            ):
                raise config_error(
                    "config.recovery.tiki_taka",
                    (
                        "to use an independent fast-array assignment and one "
                        "disjoint fast endpoint seed per slow endpoint"
                    ),
                    dict(raw["recovery"]["tiki_taka"]),
                )
    if recovery.policy == "supervised_ce_pulse_adam":
        settings = recovery.supervised_bp
        if settings is None:  # pragma: no cover - parser invariant
            raise RuntimeError("Expected supervised-BP settings after parsing.")
        available = 60_000 - data.validation_points
        expected_batches = math.ceil(settings.repair_examples / data.batch_size)
        if (
            settings.repair_examples > available
            or recovery.maximum_batches != expected_batches
        ):
            raise config_error(
                "config.recovery",
                (
                    "to consume the complete declared supervised repair cohort: "
                    f"repair_examples <= {available} and maximum_batches="
                    f"ceil(repair_examples/{data.batch_size})"
                ),
                dict(raw["recovery"]),
            )
    if recovery.policy == "supervised_ce_shadow_program_verify":
        shadow_settings = recovery.supervised_shadow_pv
        if shadow_settings is None:  # pragma: no cover - parser invariant
            raise RuntimeError("Expected supervised shadow/P&V settings after parsing.")
        available = 60_000 - data.validation_points
        expected_batches = math.ceil(
            shadow_settings.repair_examples / data.batch_size
        )
        if (
            shadow_settings.repair_examples != available
            or recovery.maximum_batches != expected_batches
        ):
            raise config_error(
                "config.recovery",
                (
                    "to consume exactly one complete 55,000-example supervised "
                    "repair epoch with maximum_batches="
                    f"ceil(55000/{data.batch_size})"
                ),
                dict(raw["recovery"]),
            )
        if (
            shadow_settings.maximum_programming_pulses
            != device.maximum_programming_pulses
            or not math.isclose(
                shadow_settings.verify_tolerance_x,
                device.verify_tolerance_x,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            or shadow_settings.writer != device.controller
        ):
            raise config_error(
                "config.recovery.supervised_shadow_pv",
                (
                    "to use the same program-and-verify cap, apparent-x "
                    "tolerance, and writer as config.device"
                ),
                dict(raw["recovery"]["supervised_shadow_pv"]),
            )
    reference = _parse_drn_reference(raw["drn_reference"])
    if reference.assignment_seed != device.assignment_seed or reference.endpoint_seeds != device.endpoint_seeds:
        raise config_error(
            "config.drn_reference", "to identify the same assignment and endpoint cohort as the crossbar", dict(raw["drn_reference"])
        )
    return CrossbarConfig(
        schema_version=1,
        experiment_id=EXPERIMENT_ID,
        runtime=_parse_runtime(raw["runtime"]),
        data=data,
        model=_parse_model(raw["model"]),
        source=_parse_source(raw["source"]),
        device=device,
        mapping=_parse_mapping(raw["mapping"]),
        offchip=offchip,
        recovery=recovery,
        transfer=transfer,
        evaluation=evaluation,
        drn_reference=reference,
    )


def resolve_crossbar_spec(document: CrossbarConfig, mode: RunMode) -> CrossbarTrainSpec:
    if mode is not RunMode.TRAIN:
        raise config_error("the requested run mode", "to equal 'train'", mode.value)
    return CrossbarTrainSpec(
        experiment_id=document.experiment_id,
        runtime=document.runtime,
        data=document.data,
        model=document.model,
        source=document.source,
        device=document.device,
        mapping=document.mapping,
        offchip=document.offchip,
        recovery=document.recovery,
        transfer=document.transfer,
        evaluation=document.evaluation,
        drn_reference=document.drn_reference,
    )


__all__ = [
    "CrossbarConfig",
    "CrossbarTrainSpec",
    "EXPERIMENT_ID",
    "OffchipSettings",
    "SCHEMA_VERSION",
    "StarRecoverySettings",
    "SupervisedBpRecoverySettings",
    "SupervisedStochasticBpRecoverySettings",
    "TikiTakaV1RecoverySettings",
    "parse_crossbar_config",
    "resolve_crossbar_spec",
]
