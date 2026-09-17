"""Strict config for the short IBM-OM 784-256-128-10 Adam experiment."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any

from experiments.mnist_relu.config import _integer, _keys, _number, _object
from experiments.schema import RunMode, config_error


EXPERIMENT_ID = "mnist_ibm_om_3fc_adam.v1"
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
    train_examples: int
    test_examples: int
    shuffle: bool
    preprocessing: str


@dataclass(frozen=True)
class ModelSettings:
    dims: tuple[int, int, int, int]
    bias: bool
    hidden_activation: str
    maximum_input_size: int
    weight_mapping: str


@dataclass(frozen=True)
class SourceSettings:
    initialization: str
    initialization_seed: int
    initial_requested_range: tuple[float, float]


@dataclass(frozen=True)
class DeviceSettings:
    preset: str
    evidence_class: str
    assignment_seed: int
    endpoint_seed: int
    corruption_policy: str
    corrupt_devices_probability: float
    corrupt_devices_range: float
    reference_std: float
    bound_policy: str
    reference_policy: str
    maximum_initial_programming_pulses: int
    verify_tolerance_q: float


@dataclass(frozen=True)
class OptimizerSettings:
    policy: str
    epochs: int
    learning_rates_q: tuple[float, float, float]
    beta_1: float
    beta_2: float
    epsilon: float
    objective: str
    maximum_one_pulse_updates_per_cell: int
    checkpoint_policy: str


@dataclass(frozen=True)
class EvaluationSettings:
    teacher_weights_path: str
    teacher_weights_sha256: str
    teacher_preprocessing: str
    evaluate_each_epoch: bool
    apparent_state_is_primary: bool
    ibm_ttv2_reference_accuracy: float
    ibm_fp32_reference_accuracy: float


@dataclass(frozen=True)
class ThreeFcAdamSpec:
    experiment_id: str
    runtime: RuntimeSettings
    data: DataSettings
    model: ModelSettings
    source: SourceSettings
    device: DeviceSettings
    optimizer: OptimizerSettings
    evaluation: EvaluationSettings


def _text(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise config_error(path, "to be a non-empty string", value)
    return value.strip()


def _boolean(value: Any, path: str) -> bool:
    if not isinstance(value, bool):
        raise config_error(path, "to be a boolean", value)
    return value


def _digest(value: Any, path: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise config_error(path, "to be a lowercase SHA-256 digest", value)
    return value


def _exact(value: Any, expected: Any, path: str) -> Any:
    if value != expected:
        raise config_error(path, f"to equal {expected!r}", value)
    return value


def _triple(value: Any, path: str) -> tuple[float, float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise config_error(path, "to contain exactly three positive numbers", value)
    result = tuple(
        float(_number(item, f"{path}[{index}]", minimum=0.0))
        for index, item in enumerate(value)
    )
    if any(item <= 0.0 for item in result):
        raise config_error(path, "to contain exactly three positive numbers", value)
    return (result[0], result[1], result[2])


def parse_three_fc_adam_config(value: Any) -> ThreeFcAdamSpec:
    """Parse the deliberately narrow five-epoch exploratory recipe."""

    raw = _object(value, "config")
    _keys(
        raw,
        "config",
        {
            "schema_version",
            "experiment_id",
            "runtime",
            "data",
            "model",
            "source",
            "device",
            "optimizer",
            "evaluation",
        },
    )
    _exact(raw["schema_version"], SCHEMA_VERSION, "config.schema_version")
    _exact(raw["experiment_id"], EXPERIMENT_ID, "config.experiment_id")

    runtime = _object(raw["runtime"], "config.runtime")
    _keys(
        runtime,
        "config.runtime",
        {"seed", "data_seed", "device", "dtype", "required_aihwkit_version"},
    )
    runtime_settings = RuntimeSettings(
        seed=_integer(runtime["seed"], "config.runtime.seed", minimum=0),
        data_seed=_integer(runtime["data_seed"], "config.runtime.data_seed", minimum=0),
        device=_text(runtime["device"], "config.runtime.device"),
        dtype=_text(runtime["dtype"], "config.runtime.dtype"),
        required_aihwkit_version=_text(
            runtime["required_aihwkit_version"],
            "config.runtime.required_aihwkit_version",
        ),
    )
    _exact(runtime_settings.device, "cuda", "config.runtime.device")
    _exact(runtime_settings.dtype, "float32", "config.runtime.dtype")
    _exact(
        runtime_settings.required_aihwkit_version,
        "1.1.0",
        "config.runtime.required_aihwkit_version",
    )

    data = _object(raw["data"], "config.data")
    _keys(
        data,
        "config.data",
        {"batch_size", "train_examples", "test_examples", "shuffle", "preprocessing"},
    )
    data_settings = DataSettings(
        batch_size=_integer(data["batch_size"], "config.data.batch_size", minimum=1),
        train_examples=_integer(
            data["train_examples"], "config.data.train_examples", minimum=1
        ),
        test_examples=_integer(
            data["test_examples"], "config.data.test_examples", minimum=1
        ),
        shuffle=_boolean(data["shuffle"], "config.data.shuffle"),
        preprocessing=_text(data["preprocessing"], "config.data.preprocessing"),
    )
    _exact(data_settings.batch_size, 64, "config.data.batch_size")
    _exact(data_settings.train_examples, 60_000, "config.data.train_examples")
    _exact(data_settings.test_examples, 10_000, "config.data.test_examples")
    _exact(data_settings.shuffle, True, "config.data.shuffle")
    _exact(
        data_settings.preprocessing,
        "torchvision_to_tensor_raw_0_1",
        "config.data.preprocessing",
    )

    model = _object(raw["model"], "config.model")
    _keys(
        model,
        "config.model",
        {"dims", "bias", "hidden_activation", "maximum_input_size", "weight_mapping"},
    )
    if not isinstance(model["dims"], (list, tuple)) or tuple(model["dims"]) != (
        784,
        256,
        128,
        10,
    ):
        raise config_error(
            "config.model.dims", "to equal [784, 256, 128, 10]", model["dims"]
        )
    model_settings = ModelSettings(
        dims=(784, 256, 128, 10),
        bias=_boolean(model["bias"], "config.model.bias"),
        hidden_activation=_text(
            model["hidden_activation"], "config.model.hidden_activation"
        ),
        maximum_input_size=_integer(
            model["maximum_input_size"],
            "config.model.maximum_input_size",
            minimum=1,
        ),
        weight_mapping=_text(model["weight_mapping"], "config.model.weight_mapping"),
    )
    _exact(model_settings.bias, True, "config.model.bias")
    _exact(model_settings.hidden_activation, "sigmoid", "config.model.hidden_activation")
    _exact(model_settings.maximum_input_size, 512, "config.model.maximum_input_size")
    _exact(model_settings.weight_mapping, "none_direct_q", "config.model.weight_mapping")

    source = _object(raw["source"], "config.source")
    _keys(
        source,
        "config.source",
        {"initialization", "initialization_seed", "initial_requested_range"},
    )
    source_range = source["initial_requested_range"]
    if not isinstance(source_range, (list, tuple)) or len(source_range) != 2:
        raise config_error(
            "config.source.initial_requested_range", "to equal [-1.0, 1.0]", source_range
        )
    source_settings = SourceSettings(
        initialization=_text(source["initialization"], "config.source.initialization"),
        initialization_seed=_integer(
            source["initialization_seed"],
            "config.source.initialization_seed",
            minimum=0,
        ),
        initial_requested_range=(float(source_range[0]), float(source_range[1])),
    )
    _exact(
        source_settings.initialization,
        "aihwkit_analog_linear_default_kaiming_uniform_with_bias",
        "config.source.initialization",
    )
    _exact(
        source_settings.initial_requested_range,
        (-1.0, 1.0),
        "config.source.initial_requested_range",
    )

    device = _object(raw["device"], "config.device")
    _keys(
        device,
        "config.device",
        {
            "preset",
            "evidence_class",
            "assignment_seed",
            "endpoint_seed",
            "corruption_policy",
            "corrupt_devices_probability",
            "corrupt_devices_range",
            "reference_std",
            "bound_policy",
            "reference_policy",
            "maximum_initial_programming_pulses",
            "verify_tolerance_q",
        },
    )
    device_settings = DeviceSettings(
        preset=_text(device["preset"], "config.device.preset"),
        evidence_class=_text(device["evidence_class"], "config.device.evidence_class"),
        assignment_seed=_integer(
            device["assignment_seed"], "config.device.assignment_seed", minimum=1
        ),
        endpoint_seed=_integer(
            device["endpoint_seed"], "config.device.endpoint_seed", minimum=1
        ),
        corruption_policy=_text(
            device["corruption_policy"], "config.device.corruption_policy"
        ),
        corrupt_devices_probability=float(
            _number(
                device["corrupt_devices_probability"],
                "config.device.corrupt_devices_probability",
                minimum=0.0,
            )
        ),
        corrupt_devices_range=float(
            _number(
                device["corrupt_devices_range"],
                "config.device.corrupt_devices_range",
                minimum=0.0,
            )
        ),
        reference_std=float(
            _number(
                device["reference_std"],
                "config.device.reference_std",
                minimum=0.0,
            )
        ),
        bound_policy=_text(device["bound_policy"], "config.device.bound_policy"),
        reference_policy=_text(
            device["reference_policy"], "config.device.reference_policy"
        ),
        maximum_initial_programming_pulses=_integer(
            device["maximum_initial_programming_pulses"],
            "config.device.maximum_initial_programming_pulses",
            minimum=1,
        ),
        verify_tolerance_q=float(
            _number(
                device["verify_tolerance_q"],
                "config.device.verify_tolerance_q",
                minimum=0.0,
            )
        ),
    )
    _exact(device_settings.preset, "ReRamArrayOMPresetDevice", "config.device.preset")
    _exact(
        device_settings.evidence_class,
        "model_based_aihwkit_preset",
        "config.device.evidence_class",
    )
    _exact(device_settings.corruption_policy, "published", "config.device.corruption_policy")
    _exact(
        device_settings.corrupt_devices_probability,
        0.1348,
        "config.device.corrupt_devices_probability",
    )
    _exact(
        device_settings.corrupt_devices_range,
        0.01,
        "config.device.corrupt_devices_range",
    )
    _exact(device_settings.reference_std, 0.05, "config.device.reference_std")
    _exact(device_settings.bound_policy, "native_sampled_bounds", "config.device.bound_policy")
    _exact(
        device_settings.reference_policy,
        "fixed_sampled_intrinsic_reference_q_equals_a_minus_r",
        "config.device.reference_policy",
    )
    if device_settings.verify_tolerance_q <= 0.0:
        raise config_error(
            "config.device.verify_tolerance_q",
            "to be positive",
            device_settings.verify_tolerance_q,
        )

    optimizer = _object(raw["optimizer"], "config.optimizer")
    _keys(
        optimizer,
        "config.optimizer",
        {
            "policy",
            "epochs",
            "learning_rates_q",
            "betas",
            "epsilon",
            "objective",
            "maximum_one_pulse_updates_per_cell",
            "checkpoint_policy",
        },
    )
    betas = optimizer["betas"]
    if not isinstance(betas, (list, tuple)) or len(betas) != 2:
        raise config_error("config.optimizer.betas", "to contain two numbers", betas)
    optimizer_settings = OptimizerSettings(
        policy=_text(optimizer["policy"], "config.optimizer.policy"),
        epochs=_integer(optimizer["epochs"], "config.optimizer.epochs", minimum=1),
        learning_rates_q=_triple(
            optimizer["learning_rates_q"], "config.optimizer.learning_rates_q"
        ),
        beta_1=float(_number(betas[0], "config.optimizer.betas[0]", minimum=0.0)),
        beta_2=float(_number(betas[1], "config.optimizer.betas[1]", minimum=0.0)),
        epsilon=float(
            _number(optimizer["epsilon"], "config.optimizer.epsilon", minimum=0.0)
        ),
        objective=_text(optimizer["objective"], "config.optimizer.objective"),
        maximum_one_pulse_updates_per_cell=_integer(
            optimizer["maximum_one_pulse_updates_per_cell"],
            "config.optimizer.maximum_one_pulse_updates_per_cell",
            minimum=1,
        ),
        checkpoint_policy=_text(
            optimizer["checkpoint_policy"], "config.optimizer.checkpoint_policy"
        ),
    )
    _exact(optimizer_settings.policy, "hybrid_pulse_adam", "config.optimizer.policy")
    _exact(optimizer_settings.epochs, 5, "config.optimizer.epochs")
    if not 0.0 <= optimizer_settings.beta_1 < 1.0:
        raise config_error("config.optimizer.betas[0]", "to be in [0, 1)", betas[0])
    if not 0.0 <= optimizer_settings.beta_2 < 1.0:
        raise config_error("config.optimizer.betas[1]", "to be in [0, 1)", betas[1])
    if optimizer_settings.epsilon <= 0.0:
        raise config_error("config.optimizer.epsilon", "to be positive", optimizer["epsilon"])
    _exact(optimizer_settings.objective, "cross_entropy", "config.optimizer.objective")
    _exact(
        optimizer_settings.checkpoint_policy,
        "fixed_final_epoch_no_selection",
        "config.optimizer.checkpoint_policy",
    )

    evaluation = _object(raw["evaluation"], "config.evaluation")
    _keys(
        evaluation,
        "config.evaluation",
        {
            "teacher_weights_path",
            "teacher_weights_sha256",
            "teacher_preprocessing",
            "evaluate_each_epoch",
            "apparent_state_is_primary",
            "ibm_ttv2_reference_accuracy",
            "ibm_fp32_reference_accuracy",
        },
    )
    evaluation_settings = EvaluationSettings(
        teacher_weights_path=_text(
            evaluation["teacher_weights_path"], "config.evaluation.teacher_weights_path"
        ),
        teacher_weights_sha256=_digest(
            evaluation["teacher_weights_sha256"],
            "config.evaluation.teacher_weights_sha256",
        ),
        teacher_preprocessing=_text(
            evaluation["teacher_preprocessing"],
            "config.evaluation.teacher_preprocessing",
        ),
        evaluate_each_epoch=_boolean(
            evaluation["evaluate_each_epoch"], "config.evaluation.evaluate_each_epoch"
        ),
        apparent_state_is_primary=_boolean(
            evaluation["apparent_state_is_primary"],
            "config.evaluation.apparent_state_is_primary",
        ),
        ibm_ttv2_reference_accuracy=float(
            _number(
                evaluation["ibm_ttv2_reference_accuracy"],
                "config.evaluation.ibm_ttv2_reference_accuracy",
                minimum=0.0,
            )
        ),
        ibm_fp32_reference_accuracy=float(
            _number(
                evaluation["ibm_fp32_reference_accuracy"],
                "config.evaluation.ibm_fp32_reference_accuracy",
                minimum=0.0,
            )
        ),
    )
    _exact(
        evaluation_settings.teacher_preprocessing,
        "normalize_mean_0.1307_std_0.3",
        "config.evaluation.teacher_preprocessing",
    )
    _exact(
        evaluation_settings.apparent_state_is_primary,
        True,
        "config.evaluation.apparent_state_is_primary",
    )
    _exact(
        evaluation_settings.evaluate_each_epoch,
        True,
        "config.evaluation.evaluate_each_epoch",
    )
    if not 0.0 <= evaluation_settings.ibm_ttv2_reference_accuracy <= 1.0:
        raise config_error(
            "config.evaluation.ibm_ttv2_reference_accuracy",
            "to be in [0, 1]",
            evaluation_settings.ibm_ttv2_reference_accuracy,
        )
    if not 0.0 <= evaluation_settings.ibm_fp32_reference_accuracy <= 1.0:
        raise config_error(
            "config.evaluation.ibm_fp32_reference_accuracy",
            "to be in [0, 1]",
            evaluation_settings.ibm_fp32_reference_accuracy,
        )

    return ThreeFcAdamSpec(
        experiment_id=EXPERIMENT_ID,
        runtime=runtime_settings,
        data=data_settings,
        model=model_settings,
        source=source_settings,
        device=device_settings,
        optimizer=optimizer_settings,
        evaluation=evaluation_settings,
    )


def resolve_three_fc_adam_spec(
    document: ThreeFcAdamSpec,
    mode: RunMode,
) -> ThreeFcAdamSpec:
    if mode is not RunMode.TRAIN:
        raise config_error("run mode", "to be 'train'", mode.value)
    return document


__all__ = [
    "EXPERIMENT_ID",
    "SCHEMA_VERSION",
    "ThreeFcAdamSpec",
    "parse_three_fc_adam_config",
    "resolve_three_fc_adam_spec",
]
