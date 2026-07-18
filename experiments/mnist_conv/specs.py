"""Complete scientific JSON contracts for canonical MNIST Conv runs/sweeps."""

from __future__ import annotations

import copy
import itertools
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping


RUN_SCHEMA_VERSION = "mnist-conv-run/v1"
SWEEP_SCHEMA_VERSION = "mnist-conv-sweep/v1"

ARCHITECTURE_PROFILES: dict[str, dict[str, Any]] = {
    "conv1": {"channels": [64], "kernel_sizes": [3], "strides": [2], "paddings": [1], "output_dim": 20},
    "conv2": {"channels": [64, 128], "kernel_sizes": [3, 3], "strides": [2, 2], "paddings": [1, 1], "output_dim": 20},
    "conv3": {"channels": [64, 128, 256], "kernel_sizes": [3, 3, 3], "strides": [2, 2, 1], "paddings": [1, 1, 1], "output_dim": 20},
}

MINIMIZER_FIELDS = {
    "double_diode_updater", "adaptive_equilibrium", "overrelaxation_factor",
    "single_diode_updater", "iv_data_path", "experimental_damping",
    "experimental_newton_max_steps", "settings",
}
MINIMIZER_SETTINGS_FIELDS = {
    "rel_tol", "vn_tol", "use_polish", "max_newton_iters", "z_thresh",
    "exp_clip", "dynamic_polish", "overrelaxation_reject_steps",
    "overrelaxation_reject_max_tries", "overrelaxation_reject_shrink",
    "overrelaxation_reject_eps", "experimental_exponential_newton_tol_progressive",
    "experimental_exponential_newton_tol_start", "experimental_exponential_newton_tol_end",
    "experimental_exponential_newton_tol_switch_hi", "experimental_exponential_newton_tol_switch_lo",
}


class SpecValidationError(ValueError):
    pass


def _error(expected: str, provided: Any, path: str) -> SpecValidationError:
    return SpecValidationError(f"Expected {path} to be {expected}. Provided value: {provided!r}.")


def _strict_json_loads(text: str, source: Path) -> Any:
    def reject_constant(value: str) -> Any:
        raise _error("a finite JSON number", value, str(source))

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise _error("JSON objects with unique keys", key, str(source))
            result[key] = value
        return result

    try:
        return json.loads(
            text,
            parse_constant=reject_constant,
            object_pairs_hook=unique_object,
        )
    except json.JSONDecodeError as exc:
        raise SpecValidationError(
            f"Expected {source} to contain valid strict JSON. Provided error: {exc}."
        ) from exc


def _object(value: Any, path: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise _error("a JSON object", value, path)
    return copy.deepcopy(value)


def _exact_keys(value: Mapping[str, Any], keys: Iterable[str], path: str) -> None:
    expected = set(keys)
    missing, extra = sorted(expected - set(value)), sorted(set(value) - expected)
    if missing or extra:
        raise _error(f"an object with exactly keys {sorted(expected)}", {"missing": missing, "extra": extra}, path)


def _finite(value: Any, path: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise _error("a finite number", value, path)
    result = float(value)
    if not math.isfinite(result) or (positive and result <= 0):
        raise _error("a positive finite number" if positive else "a finite number", value, path)
    return result


def _integer(value: Any, path: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise _error(f"an integer >= {minimum}", value, path)
    return value


def _nullable_positive_int(value: Any, path: str) -> int | None:
    return None if value is None else _integer(value, path, minimum=1)


def _finite_list(value: Any, path: str, length: int, *, positive: bool = False) -> list[float]:
    if not isinstance(value, list) or len(value) != length:
        raise _error(f"an explicit list of exactly {length} numbers", value, path)
    return [_finite(item, f"{path}[{index}]", positive=positive) for index, item in enumerate(value)]


def _string(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise _error("a non-empty string", value, path)
    return value.strip()


def _normalize_dataset(value: Any) -> dict[str, Any]:
    dataset = _object(value, "run.dataset")
    _exact_keys(dataset, {"name", "input_shape", "batch_size", "normalization", "affine", "max_batches", "max_test_batches"}, "run.dataset")
    if dataset["name"] != "mnist":
        raise _error("exactly 'mnist'", dataset["name"], "run.dataset.name")
    if dataset["input_shape"] != [2, 28, 28]:
        raise _error("the signed-channel shape [2, 28, 28]", dataset["input_shape"], "run.dataset.input_shape")
    normalization = _object(dataset["normalization"], "run.dataset.normalization")
    _exact_keys(normalization, {"mean", "std", "scale"}, "run.dataset.normalization")
    normalization = {
        "mean": _finite(normalization["mean"], "run.dataset.normalization.mean"),
        "std": _finite(normalization["std"], "run.dataset.normalization.std", positive=True),
        "scale": _finite(normalization["scale"], "run.dataset.normalization.scale", positive=True),
    }
    affine = _object(dataset["affine"], "run.dataset.affine")
    _exact_keys(affine, {"enabled", "preset", "degrees", "translate", "scale", "shear", "seed", "interpolation", "fill"}, "run.dataset.affine")
    if not isinstance(affine["enabled"], bool):
        raise _error("a boolean", affine["enabled"], "run.dataset.affine.enabled")
    translate = _finite_list(affine["translate"], "run.dataset.affine.translate", 2)
    if any(item < 0 for item in translate):
        raise _error("two non-negative fractions", translate, "run.dataset.affine.translate")
    scale = _finite_list(affine["scale"], "run.dataset.affine.scale", 2, positive=True)
    if scale[0] > scale[1]:
        raise _error("an ordered [minimum, maximum] pair", scale, "run.dataset.affine.scale")
    if affine["interpolation"] != "bilinear":
        raise _error("exactly 'bilinear'", affine["interpolation"], "run.dataset.affine.interpolation")
    affine = {
        "enabled": affine["enabled"], "preset": _string(affine["preset"], "run.dataset.affine.preset"),
        "degrees": _finite(affine["degrees"], "run.dataset.affine.degrees"), "translate": translate,
        "scale": scale, "shear": _finite(affine["shear"], "run.dataset.affine.shear"),
        "seed": _integer(affine["seed"], "run.dataset.affine.seed"), "interpolation": "bilinear",
        "fill": _finite(affine["fill"], "run.dataset.affine.fill"),
    }
    if affine["degrees"] < 0 or affine["shear"] < 0:
        raise _error("non-negative symmetric affine ranges", {"degrees": affine["degrees"], "shear": affine["shear"]}, "run.dataset.affine")
    if affine["fill"] != 0.0:
        raise _error("exactly 0.0 for the v1 backend", affine["fill"], "run.dataset.affine.fill")
    return {
        "name": "mnist", "input_shape": [2, 28, 28],
        "batch_size": _integer(dataset["batch_size"], "run.dataset.batch_size", minimum=1),
        "normalization": normalization, "affine": affine,
        "max_batches": _nullable_positive_int(dataset["max_batches"], "run.dataset.max_batches"),
        "max_test_batches": _nullable_positive_int(dataset["max_test_batches"], "run.dataset.max_test_batches"),
    }


def _normalize_architecture(value: Any) -> dict[str, Any]:
    architecture = _object(value, "run.architecture")
    _exact_keys(architecture, {"profile", "channels", "kernel_sizes", "strides", "paddings", "output_dim", "pooling"}, "run.architecture")
    profile = architecture["profile"]
    if profile not in ARCHITECTURE_PROFILES:
        raise _error(f"one of {sorted(ARCHITECTURE_PROFILES)}", profile, "run.architecture.profile")
    frozen = ARCHITECTURE_PROFILES[profile]
    for field, expected in frozen.items():
        if architecture[field] != expected:
            raise _error(f"the frozen {profile} value {expected!r}", architecture[field], f"run.architecture.{field}")
    pooling = _object(architecture["pooling"], "run.architecture.pooling")
    _exact_keys(pooling, {"mode"}, "run.architecture.pooling")
    if pooling["mode"] != "none":
        raise _error("exactly 'none'", pooling["mode"], "run.architecture.pooling.mode")
    return copy.deepcopy(architecture)


def _validate_diode_params(model: dict[str, Any], non_linearity: str) -> None:
    for key in ("quadratic_diode_param", "exponential_diode_param", "hard_sigmoid_param"):
        model[key] = _object(model[key], f"run.model.{key}")
    if non_linearity == "hard_sigmoid":
        for key in ("quadratic_diode_param", "exponential_diode_param"):
            if model[key]:
                raise _error(
                    "an empty object because this diode family is unused by hard_sigmoid",
                    model[key],
                    f"run.model.{key}",
                )
        params = model["hard_sigmoid_param"]
        _exact_keys(params, {"g_on", "g_off", "v_off"}, "run.model.hard_sigmoid_param")
        params["g_on"] = _finite(params["g_on"], "run.model.hard_sigmoid_param.g_on")
        params["g_off"] = _finite(params["g_off"], "run.model.hard_sigmoid_param.g_off")
        params["v_off"] = _finite(params["v_off"], "run.model.hard_sigmoid_param.v_off")
        if params["v_off"] < 0:
            raise _error("a non-negative number", params["v_off"], "run.model.hard_sigmoid_param.v_off")
    else:
        for key in ("quadratic_diode_param", "exponential_diode_param", "hard_sigmoid_param"):
            if model[key]:
                raise _error(
                    "an empty object because perfect_diode has no configurable diode parameters in v1",
                    model[key],
                    f"run.model.{key}",
                )


def _normalize_model(value: Any, depth: int) -> dict[str, Any]:
    model = _object(value, "run.model")
    keys = {
        "type", "non_linearity", "voltage_amp", "current_amp", "input_gain", "weight_gains",
        "weight_min", "weight_max", "weight_init_mode", "quadratic_diode_param",
        "exponential_diode_param", "hard_sigmoid_param", "trainable_parameters",
        "amplification_min", "amplification_max",
    }
    _exact_keys(model, keys, "run.model")
    if model["type"] != "resistive_conv":
        raise _error("exactly 'resistive_conv'", model["type"], "run.model.type")
    if model["non_linearity"] not in {"hard_sigmoid", "perfect_diode"}:
        raise _error("'hard_sigmoid' or 'perfect_diode'", model["non_linearity"], "run.model.non_linearity")
    trainable = _object(model["trainable_parameters"], "run.model.trainable_parameters")
    _exact_keys(trainable, {"weights", "biases", "amplification", "hard_sigmoid_v_off"}, "run.model.trainable_parameters")
    expected_trainable = {"weights": True, "biases": True, "amplification": False, "hard_sigmoid_v_off": False}
    if trainable != expected_trainable:
        raise _error(f"the fixed v1 settings {expected_trainable!r}", trainable, "run.model.trainable_parameters")
    weight_min, weight_max = _finite(model["weight_min"], "run.model.weight_min"), _finite(model["weight_max"], "run.model.weight_max")
    if weight_min > weight_max:
        raise _error("<= weight_max", weight_min, "run.model.weight_min")
    model.update({
        "voltage_amp": _finite(model["voltage_amp"], "run.model.voltage_amp", positive=True),
        "current_amp": _finite(model["current_amp"], "run.model.current_amp", positive=True),
        "input_gain": _finite(model["input_gain"], "run.model.input_gain", positive=True),
        "weight_gains": _finite_list(model["weight_gains"], "run.model.weight_gains", depth + 1, positive=True),
        "weight_min": weight_min, "weight_max": weight_max,
        "weight_init_mode": _string(model["weight_init_mode"], "run.model.weight_init_mode"),
        "trainable_parameters": expected_trainable,
        "amplification_min": _finite(model["amplification_min"], "run.model.amplification_min", positive=True),
        "amplification_max": None if model["amplification_max"] is None else _finite(model["amplification_max"], "run.model.amplification_max", positive=True),
    })
    if model["amplification_max"] is not None and model["amplification_max"] < model["amplification_min"]:
        raise _error(">= amplification_min", model["amplification_max"], "run.model.amplification_max")
    _validate_diode_params(model, model["non_linearity"])
    return model


def _normalize_minimizer(value: Any) -> dict[str, Any]:
    minimizer = _object(value, "run.solver.minimizer")
    _exact_keys(minimizer, MINIMIZER_FIELDS, "run.solver.minimizer")
    if minimizer["adaptive_equilibrium"] is not False:
        raise _error("false for fixed-step v1", minimizer["adaptive_equilibrium"], "run.solver.minimizer.adaptive_equilibrium")
    for key in ("double_diode_updater", "single_diode_updater"):
        minimizer[key] = _string(minimizer[key], f"run.solver.minimizer.{key}")
    if minimizer["iv_data_path"] is not None:
        raise _error("null for v1 hard-sigmoid/perfect-diode runs", minimizer["iv_data_path"], "run.solver.minimizer.iv_data_path")
    minimizer["overrelaxation_factor"] = _finite(
        minimizer["overrelaxation_factor"],
        "run.solver.minimizer.overrelaxation_factor",
        positive=True,
    )
    minimizer["experimental_damping"] = _finite(
        minimizer["experimental_damping"],
        "run.solver.minimizer.experimental_damping",
        positive=True,
    )
    if minimizer["experimental_damping"] > 1:
        raise _error("a fraction in (0, 1]", minimizer["experimental_damping"], "run.solver.minimizer.experimental_damping")
    minimizer["experimental_newton_max_steps"] = _integer(
        minimizer["experimental_newton_max_steps"],
        "run.solver.minimizer.experimental_newton_max_steps",
        minimum=1,
    )
    settings = _object(minimizer["settings"], "run.solver.minimizer.settings")
    _exact_keys(settings, MINIMIZER_SETTINGS_FIELDS, "run.solver.minimizer.settings")
    boolean_settings = {
        "use_polish", "dynamic_polish", "overrelaxation_reject_steps",
        "experimental_exponential_newton_tol_progressive",
    }
    for key in boolean_settings:
        if not isinstance(settings[key], bool):
            raise _error("a boolean", settings[key], f"run.solver.minimizer.settings.{key}")
    settings["max_newton_iters"] = _integer(
        settings["max_newton_iters"], "run.solver.minimizer.settings.max_newton_iters"
    )
    settings["overrelaxation_reject_max_tries"] = _integer(
        settings["overrelaxation_reject_max_tries"],
        "run.solver.minimizer.settings.overrelaxation_reject_max_tries",
        minimum=1,
    )
    for key in (
        "rel_tol", "vn_tol", "z_thresh", "exp_clip",
        "experimental_exponential_newton_tol_start",
        "experimental_exponential_newton_tol_end",
        "experimental_exponential_newton_tol_switch_hi",
        "experimental_exponential_newton_tol_switch_lo",
    ):
        settings[key] = _finite(settings[key], f"run.solver.minimizer.settings.{key}", positive=True)
    settings["overrelaxation_reject_eps"] = _finite(
        settings["overrelaxation_reject_eps"],
        "run.solver.minimizer.settings.overrelaxation_reject_eps",
    )
    if settings["overrelaxation_reject_eps"] < 0:
        raise _error("a non-negative finite number", settings["overrelaxation_reject_eps"], "run.solver.minimizer.settings.overrelaxation_reject_eps")
    settings["overrelaxation_reject_shrink"] = _finite(
        settings["overrelaxation_reject_shrink"],
        "run.solver.minimizer.settings.overrelaxation_reject_shrink",
    )
    if not 0 < settings["overrelaxation_reject_shrink"] < 1:
        raise _error("a fraction in (0, 1)", settings["overrelaxation_reject_shrink"], "run.solver.minimizer.settings.overrelaxation_reject_shrink")
    if not (
        settings["experimental_exponential_newton_tol_switch_hi"]
        > settings["experimental_exponential_newton_tol_switch_lo"]
    ):
        raise _error(
            "greater than experimental_exponential_newton_tol_switch_lo",
            settings["experimental_exponential_newton_tol_switch_hi"],
            "run.solver.minimizer.settings.experimental_exponential_newton_tol_switch_hi",
        )
    minimizer["settings"] = settings
    return minimizer


def _normalize_solver(value: Any) -> dict[str, Any]:
    solver = _object(value, "run.solver")
    _exact_keys(solver, {"inference_iterations", "training_iterations", "energy_mode", "minimizer"}, "run.solver")
    if solver["energy_mode"] != "asynchronous":
        raise _error("exactly 'asynchronous'", solver["energy_mode"], "run.solver.energy_mode")
    return {
        "inference_iterations": _integer(solver["inference_iterations"], "run.solver.inference_iterations", minimum=1),
        "training_iterations": _integer(solver["training_iterations"], "run.solver.training_iterations", minimum=1),
        "energy_mode": "asynchronous", "minimizer": _normalize_minimizer(solver["minimizer"]),
    }


def _normalize_training(value: Any, depth: int) -> dict[str, Any]:
    training = _object(value, "run.training")
    keys = {
        "algorithm", "epochs", "optimizer", "learning_rate", "beta", "lr_decay",
        "checkpoint_rule", "pruning", "batch_state_policy",
    }
    _exact_keys(training, keys, "run.training")
    if training["algorithm"] != "BP":
        raise _error("exactly 'BP'", training["algorithm"], "run.training.algorithm")
    if training["batch_state_policy"] != "reset_each_batch":
        raise _error("exactly 'reset_each_batch'", training["batch_state_policy"], "run.training.batch_state_policy")
    if training["checkpoint_rule"] != "max_test_accuracy":
        raise _error("exactly 'max_test_accuracy'", training["checkpoint_rule"], "run.training.checkpoint_rule")
    optimizer = _object(training["optimizer"], "run.training.optimizer")
    _exact_keys(optimizer, {"name", "momentum", "weight_decay"}, "run.training.optimizer")
    if optimizer["name"] != "SGD":
        raise _error("exactly 'SGD' for v1", optimizer["name"], "run.training.optimizer.name")
    optimizer["momentum"] = _finite(optimizer["momentum"], "run.training.optimizer.momentum")
    optimizer["weight_decay"] = _finite(optimizer["weight_decay"], "run.training.optimizer.weight_decay")
    if optimizer["momentum"] != 0.0:
        raise _error("exactly 0.0 for the current v1 SGD protocol", optimizer["momentum"], "run.training.optimizer.momentum")
    if optimizer["weight_decay"] != 0.0:
        raise _error("exactly 0.0 for the current v1 SGD protocol", optimizer["weight_decay"], "run.training.optimizer.weight_decay")
    pruning = _object(training["pruning"], "run.training.pruning")
    _exact_keys(pruning, {"enabled", "after_epoch", "min_best_test_accuracy"}, "run.training.pruning")
    if not isinstance(pruning["enabled"], bool):
        raise _error("a boolean", pruning["enabled"], "run.training.pruning.enabled")
    if pruning["enabled"]:
        pruning["after_epoch"] = _integer(pruning["after_epoch"], "run.training.pruning.after_epoch", minimum=1)
        pruning["min_best_test_accuracy"] = _finite(pruning["min_best_test_accuracy"], "run.training.pruning.min_best_test_accuracy")
        if not 0 <= pruning["min_best_test_accuracy"] <= 1:
            raise _error("a fraction in [0, 1]", pruning["min_best_test_accuracy"], "run.training.pruning.min_best_test_accuracy")
    elif pruning["after_epoch"] is not None or pruning["min_best_test_accuracy"] is not None:
        raise _error("null thresholds when disabled", pruning, "run.training.pruning")
    return {
        "algorithm": "BP", "epochs": _integer(training["epochs"], "run.training.epochs", minimum=1),
        "optimizer": {"name": "SGD", "momentum": 0.0, "weight_decay": 0.0},
        "learning_rate": _finite_list(training["learning_rate"], "run.training.learning_rate", 2 * depth + 1, positive=True),
        "beta": _finite(training["beta"], "run.training.beta"),
        "lr_decay": _finite(training["lr_decay"], "run.training.lr_decay", positive=True),
        "checkpoint_rule": "max_test_accuracy", "pruning": pruning, "batch_state_policy": "reset_each_batch",
    }


def _normalize_initialization(value: Any) -> dict[str, Any]:
    initialization = _object(value, "run.initialization")
    _exact_keys(initialization, {"checkpoint"}, "run.initialization")
    checkpoint = initialization["checkpoint"]
    if checkpoint is None:
        return {"checkpoint": None}
    checkpoint = _object(checkpoint, "run.initialization.checkpoint")
    _exact_keys(checkpoint, {"path", "sha256", "format", "source_run_id", "role"}, "run.initialization.checkpoint")
    checkpoint["path"] = _string(checkpoint["path"], "run.initialization.checkpoint.path")
    checkpoint_path = Path(checkpoint["path"])
    if checkpoint_path.is_absolute() or ".." in checkpoint_path.parts or checkpoint_path == Path(".") or "\\" in checkpoint["path"]:
        raise _error("a portable path relative to results_root without '..'", checkpoint["path"], "run.initialization.checkpoint.path")
    checkpoint["path"] = checkpoint_path.as_posix()
    if not isinstance(checkpoint["sha256"], str) or not re.fullmatch(r"[0-9a-f]{64}", checkpoint["sha256"]):
        raise _error("a lowercase SHA-256 digest", checkpoint["sha256"], "run.initialization.checkpoint.sha256")
    checkpoint["format"] = _string(checkpoint["format"], "run.initialization.checkpoint.format")
    allowed_formats = {"drn.function.parameters/v1", "legacy_tensor_list"}
    if checkpoint["format"] not in allowed_formats:
        raise _error(
            f"one of {sorted(allowed_formats)!r}",
            checkpoint["format"],
            "run.initialization.checkpoint.format",
        )
    checkpoint["role"] = _string(checkpoint["role"], "run.initialization.checkpoint.role")
    allowed_roles = {"best", "final", "initialization"}
    if checkpoint["role"] not in allowed_roles:
        raise _error(
            f"one of {sorted(allowed_roles)!r}",
            checkpoint["role"],
            "run.initialization.checkpoint.role",
        )
    if checkpoint["source_run_id"] is not None:
        checkpoint["source_run_id"] = _string(checkpoint["source_run_id"], "run.initialization.checkpoint.source_run_id")
        if not re.fullmatch(r"run_[0-9a-f]{64}", checkpoint["source_run_id"]):
            raise _error(
                "null or a canonical run_ followed by 64 lowercase hexadecimal characters",
                checkpoint["source_run_id"],
                "run.initialization.checkpoint.source_run_id",
            )
    return {"checkpoint": checkpoint}


def _normalize_layer_measurements(
    value: Any,
    *,
    depth: int,
    measurement_key: str,
) -> list[dict[str, Any]]:
    path = "run.calibration.layer_measurements"
    if not isinstance(value, list) or len(value) != depth:
        raise _error(f"a list with exactly one measurement for each of {depth} hidden layers", value, path)
    result = []
    for expected_index, raw in enumerate(value, 1):
        item = _object(raw, f"{path}[]")
        _exact_keys(item, {"layer_index", measurement_key}, f"{path}[]")
        layer_index = _integer(item["layer_index"], f"{path}[].layer_index", minimum=1)
        if layer_index != expected_index:
            raise _error(f"the contiguous hidden-layer index {expected_index}", layer_index, f"{path}[].layer_index")
        measured = _finite(item[measurement_key], f"{path}[].{measurement_key}")
        if not 0 <= measured <= 1:
            raise _error("a fraction in [0, 1]", measured, f"{path}[].{measurement_key}")
        result.append({"layer_index": layer_index, measurement_key: measured})
    return result


def _normalize_calibration(
    value: Any,
    model: dict[str, Any],
    *,
    depth: int,
    affine_seed: int,
) -> dict[str, Any]:
    calibration = _object(value, "run.calibration")
    common = {
        "kind", "calibration_id", "scope", "sample_count", "batch_size",
        "model_seed", "affine_seed", "settling_iterations",
        "adaptive_equilibrium", "layer_measurements",
    }
    if model["non_linearity"] == "hard_sigmoid":
        keys = common | {
            "v_off", "g_on", "g_off", "target_initial_saturation",
            "measured_initial_saturation",
        }
        _exact_keys(calibration, keys, "run.calibration")
        if calibration["kind"] != "hard_sigmoid_saturation":
            raise _error("exactly 'hard_sigmoid_saturation'", calibration["kind"], "run.calibration.kind")
        calibration["calibration_id"] = _string(calibration["calibration_id"], "run.calibration.calibration_id")
        calibration["scope"] = _string(calibration["scope"], "run.calibration.scope")
        calibration["sample_count"] = _integer(calibration["sample_count"], "run.calibration.sample_count", minimum=1)
        calibration["v_off"] = _finite(calibration["v_off"], "run.calibration.v_off")
        if calibration["v_off"] != float(model["hard_sigmoid_param"]["v_off"]):
            raise _error("equal to model hard-sigmoid v_off", calibration["v_off"], "run.calibration.v_off")
        for key in ("g_on", "g_off"):
            calibration[key] = _finite(calibration[key], f"run.calibration.{key}")
            if calibration[key] != float(model["hard_sigmoid_param"][key]):
                raise _error(f"equal to model hard-sigmoid {key}", calibration[key], f"run.calibration.{key}")
        for key in ("target_initial_saturation", "measured_initial_saturation"):
            calibration[key] = _finite(calibration[key], f"run.calibration.{key}")
            if not 0 <= calibration[key] <= 1:
                raise _error("a fraction in [0, 1]", calibration[key], f"run.calibration.{key}")
        measurement_key = "measured_saturation"
    else:
        keys = common | {
            "target_initial_occupancy", "measured_initial_occupancy", "clamp_epsilon",
        }
        _exact_keys(calibration, keys, "run.calibration")
        if calibration["kind"] != "perfect_diode_clamped_occupancy":
            raise _error("exactly 'perfect_diode_clamped_occupancy'", calibration["kind"], "run.calibration.kind")
        calibration["clamp_epsilon"] = _finite(calibration["clamp_epsilon"], "run.calibration.clamp_epsilon", positive=True)
        for key in ("target_initial_occupancy", "measured_initial_occupancy"):
            calibration[key] = _finite(calibration[key], f"run.calibration.{key}")
            if not 0 <= calibration[key] <= 1:
                raise _error("a fraction in [0, 1]", calibration[key], f"run.calibration.{key}")
        measurement_key = "measured_clamped_occupancy"

    calibration["calibration_id"] = _string(calibration["calibration_id"], "run.calibration.calibration_id")
    calibration["scope"] = _string(calibration["scope"], "run.calibration.scope")
    calibration["sample_count"] = _integer(calibration["sample_count"], "run.calibration.sample_count", minimum=1)
    calibration["batch_size"] = _integer(calibration["batch_size"], "run.calibration.batch_size", minimum=1)
    calibration["model_seed"] = _integer(calibration["model_seed"], "run.calibration.model_seed")
    calibration["affine_seed"] = _integer(calibration["affine_seed"], "run.calibration.affine_seed")
    if calibration["affine_seed"] != affine_seed:
        raise _error("equal to the dataset affine seed", calibration["affine_seed"], "run.calibration.affine_seed")
    calibration["settling_iterations"] = _integer(
        calibration["settling_iterations"], "run.calibration.settling_iterations", minimum=1
    )
    if not isinstance(calibration["adaptive_equilibrium"], bool):
        raise _error("a boolean", calibration["adaptive_equilibrium"], "run.calibration.adaptive_equilibrium")
    calibration["layer_measurements"] = _normalize_layer_measurements(
        calibration["layer_measurements"], depth=depth, measurement_key=measurement_key
    )
    initial_key = "measured_initial_saturation" if model["non_linearity"] == "hard_sigmoid" else "measured_initial_occupancy"
    if calibration[initial_key] != calibration["layer_measurements"][0][measurement_key]:
        raise _error(
            f"equal to the first hidden-layer {measurement_key}",
            calibration[initial_key],
            f"run.calibration.{initial_key}",
        )
    return calibration


@dataclass(frozen=True)
class RunSpec:
    data: dict[str, Any]

    @classmethod
    def from_path(cls, path: str | Path) -> "RunSpec":
        source = Path(path)
        try:
            text = source.read_text()
        except OSError as exc:
            raise SpecValidationError(f"Expected {source} to contain valid JSON. Provided error: {exc}.") from exc
        return cls.from_dict(_strict_json_loads(text, source))

    @classmethod
    def from_dict(cls, value: Any) -> "RunSpec":
        top = _object(value, "RunSpec")
        _exact_keys(top, {"schema_version", "label", "seed", "replicate_id", "protocol_id", "category", "run"}, "RunSpec")
        if top["schema_version"] != RUN_SCHEMA_VERSION:
            raise _error(RUN_SCHEMA_VERSION, top["schema_version"], "RunSpec.schema_version")
        label = _string(top["label"], "RunSpec.label")
        protocol_id = _string(top["protocol_id"], "RunSpec.protocol_id")
        if top["category"] not in {"diagnostic", "final"}:
            raise _error("'diagnostic' or 'final'", top["category"], "RunSpec.category")
        run = _object(top["run"], "RunSpec.run")
        _exact_keys(run, {"dataset", "architecture", "model", "solver", "training", "initialization", "calibration"}, "RunSpec.run")
        replicate_id = top["replicate_id"]
        if replicate_id is not None:
            replicate_id = _string(replicate_id, "RunSpec.replicate_id")
        architecture = _normalize_architecture(run["architecture"])
        model = _normalize_model(run["model"], len(architecture["channels"]))
        dataset = _normalize_dataset(run["dataset"])
        normalized_run = {
            "dataset": dataset,
            "architecture": architecture, "model": model, "solver": _normalize_solver(run["solver"]),
            "training": _normalize_training(run["training"], len(architecture["channels"])),
            "initialization": _normalize_initialization(run["initialization"]),
            "calibration": _normalize_calibration(
                run["calibration"],
                model,
                depth=len(architecture["channels"]),
                affine_seed=dataset["affine"]["seed"],
            ),
        }
        if top["category"] == "final" and normalized_run["training"]["pruning"]["enabled"]:
            raise _error("disabled for final runs", normalized_run["training"]["pruning"], "run.training.pruning")
        return cls({
            "schema_version": RUN_SCHEMA_VERSION,
            "label": label,
            "seed": _integer(top["seed"], "RunSpec.seed"),
            "replicate_id": replicate_id,
            "protocol_id": protocol_id,
            "category": top["category"],
            "run": normalized_run,
        })

    def to_dict(self) -> dict[str, Any]:
        return copy.deepcopy(self.data)

    def identity_payload(self) -> dict[str, Any]:
        payload = self.to_dict()
        payload.pop("label")
        checkpoint = payload["run"]["initialization"]["checkpoint"]
        if checkpoint is not None:
            checkpoint.pop("path")
        return payload


def _pointer_parts(pointer: str) -> list[str]:
    if not isinstance(pointer, str) or not pointer.startswith("/") or pointer == "/":
        raise _error("a non-root RFC 6901 JSON pointer", pointer, "JSON pointer")
    return [part.replace("~1", "/").replace("~0", "~") for part in pointer[1:].split("/")]


def pointer_get(document: Any, pointer: str) -> Any:
    current = document
    for token in _pointer_parts(pointer):
        if isinstance(current, list):
            try: current = current[int(token)]
            except (ValueError, IndexError) as exc: raise SpecValidationError(f"Expected pointer {pointer!r} to resolve. Provided token: {token!r}.") from exc
        elif isinstance(current, dict) and token in current: current = current[token]
        else: raise SpecValidationError(f"Expected pointer {pointer!r} to resolve. Missing token: {token!r}.")
    return current


def pointer_set(document: Any, pointer: str, value: Any) -> None:
    parts = _pointer_parts(pointer)
    current = document
    for token in parts[:-1]:
        current = current[int(token)] if isinstance(current, list) else current[token]
    token = parts[-1]
    if isinstance(current, list):
        index = int(token)
        if not 0 <= index < len(current): raise SpecValidationError(f"Expected pointer {pointer!r} to replace an existing list item. Provided index: {index}.")
        current[index] = copy.deepcopy(value)
    elif isinstance(current, dict) and token in current: current[token] = copy.deepcopy(value)
    else: raise SpecValidationError(f"Expected pointer {pointer!r} to replace an existing field. Provided token: {token!r}.")


def _flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for key in sorted(value): out.update(_flatten(value[key], f"{prefix}/{key}"))
        return out
    if isinstance(value, list):
        out = {}
        for index, item in enumerate(value): out.update(_flatten(item, f"{prefix}/{index}"))
        return out
    return {prefix: value}


def _covers(declaration: str, leaf: str) -> bool:
    return leaf == declaration or leaf.startswith(declaration.rstrip("/") + "/")


@dataclass(frozen=True)
class ExpandedRun:
    job_index: int
    logical_key: str
    case_id: str
    axes: dict[str, Any]
    spec: RunSpec


@dataclass(frozen=True)
class SweepSpec:
    data: dict[str, Any]

    @classmethod
    def from_path(cls, path: str | Path) -> "SweepSpec":
        source = Path(path).expanduser().resolve()
        try:
            text = source.read_text()
        except OSError as exc:
            raise SpecValidationError(f"Expected {source} to contain valid JSON. Provided error: {exc}.") from exc
        value = _strict_json_loads(text, source)
        return cls.from_dict(value, source_dir=source.parent)

    @classmethod
    def from_dict(cls, value: Any, *, source_dir: str | Path | None = None) -> "SweepSpec":
        top = _object(value, "SweepSpec")
        _exact_keys(top, {"schema_version", "name", "base_run", "axes", "cases", "varying_fields", "collection"}, "SweepSpec")
        if top["schema_version"] != SWEEP_SCHEMA_VERSION: raise _error(SWEEP_SCHEMA_VERSION, top["schema_version"], "SweepSpec.schema_version")
        name = _string(top["name"], "SweepSpec.name")
        base_value = top["base_run"]
        if isinstance(base_value, str):
            base_path = Path(base_value).expanduser()
            if not base_path.is_absolute():
                if source_dir is None: raise SpecValidationError(f"Expected source_dir for relative base_run. Provided value: {base_value!r}.")
                base_path = Path(source_dir) / base_path
            base = RunSpec.from_path(base_path)
        else: base = RunSpec.from_dict(base_value)
        axes_raw = top["axes"]
        if not isinstance(axes_raw, list): raise _error("a list", axes_raw, "SweepSpec.axes")
        axes, seen = [], set()
        for raw in axes_raw:
            axis = _object(raw, "SweepSpec.axes[]"); _exact_keys(axis, {"path", "values"}, "SweepSpec.axes[]")
            pointer_get(base.to_dict(), axis["path"])
            if axis["path"] in seen: raise SpecValidationError(f"Expected unique axis paths. Provided duplicate: {axis['path']!r}.")
            if not isinstance(axis["values"], list) or not axis["values"]: raise _error("a non-empty list", axis["values"], "SweepSpec.axes[].values")
            seen.add(axis["path"]); axes.append(copy.deepcopy(axis))
        axes.sort(key=lambda item: item["path"])
        cases_raw = top["cases"]
        if not isinstance(cases_raw, list) or not cases_raw: raise _error("a non-empty list", cases_raw, "SweepSpec.cases")
        cases, case_ids = [], set()
        for raw in cases_raw:
            case = _object(raw, "SweepSpec.cases[]"); _exact_keys(case, {"id", "set"}, "SweepSpec.cases[]")
            case_id = _string(case["id"], "SweepSpec.cases[].id")
            if case_id in case_ids: raise SpecValidationError(f"Expected unique case ids. Provided duplicate: {case_id!r}.")
            assignments = _object(case["set"], f"SweepSpec.cases[{case_id}].set")
            if set(assignments) & seen: raise SpecValidationError(f"Expected linked case fields and axes to be disjoint. Provided overlap: {sorted(set(assignments) & seen)!r}.")
            for pointer in assignments: pointer_get(base.to_dict(), pointer)
            case_ids.add(case_id); cases.append({"id": case_id, "set": assignments})
        varying = top["varying_fields"]
        if not isinstance(varying, list) or len(varying) != len(set(varying)) or any(not isinstance(item, str) for item in varying): raise _error("a list of unique JSON pointers", varying, "SweepSpec.varying_fields")
        for pointer in varying: pointer_get(base.to_dict(), pointer)
        assignment_paths = set(seen)
        for case in cases:
            assignment_paths.update(case["set"])
        undeclared_assignments = sorted(
            pointer
            for pointer in assignment_paths
            if not any(_covers(declaration, pointer) for declaration in varying)
        )
        if undeclared_assignments:
            raise SpecValidationError(
                "Expected every case or axis assignment path to be covered by varying_fields. "
                f"Provided undeclared assignment paths: {undeclared_assignments!r}."
            )
        collection = _object(top["collection"], "SweepSpec.collection"); _exact_keys(collection, {"expected_seeds", "required_cases", "group_by"}, "SweepSpec.collection")
        expected_seeds = collection["expected_seeds"]
        if not isinstance(expected_seeds, list) or not expected_seeds: raise _error("a non-empty integer list", expected_seeds, "SweepSpec.collection.expected_seeds")
        collection["expected_seeds"] = [_integer(item, "SweepSpec.collection.expected_seeds[]") for item in expected_seeds]
        if len(set(collection["expected_seeds"])) != len(expected_seeds): raise _error("unique seeds", expected_seeds, "SweepSpec.collection.expected_seeds")
        required_cases = collection["required_cases"]
        expected_cases = [case["id"] for case in cases]
        if required_cases != expected_cases:
            raise _error(
                f"the case ids in file order {expected_cases!r}",
                required_cases,
                "SweepSpec.collection.required_cases",
            )
        if (
            not isinstance(collection["group_by"], list)
            or any(not isinstance(item, str) for item in collection["group_by"])
            or len(collection["group_by"]) != len(set(collection["group_by"]))
        ):
            raise _error("a list of unique JSON pointers", collection["group_by"], "SweepSpec.collection.group_by")
        forbidden_group_fields = {"/seed", "/replicate_id"}
        forbidden_group_by = sorted(set(collection["group_by"]) & forbidden_group_fields)
        if forbidden_group_by:
            raise _error(
                "grouping fields that exclude /seed and /replicate_id so seed coverage is aggregated within each group",
                forbidden_group_by,
                "SweepSpec.collection.group_by",
            )
        for pointer in collection["group_by"]: pointer_get(base.to_dict(), pointer)
        result = cls({"schema_version": SWEEP_SCHEMA_VERSION, "name": name, "base_run": base.to_dict(), "axes": axes, "cases": cases, "varying_fields": sorted(varying), "collection": collection})
        result.expand()
        return result

    def to_dict(self) -> dict[str, Any]: return copy.deepcopy(self.data)

    def expand(self) -> list[ExpandedRun]:
        base, axes = self.data["base_run"], self.data["axes"]
        products = list(itertools.product(*(axis["values"] for axis in axes))) if axes else [tuple()]
        expanded: list[ExpandedRun] = []
        for case in self.data["cases"]:
            case_seeds: set[int] = set()
            for values in products:
                candidate = copy.deepcopy(base)
                for pointer, value in case["set"].items(): pointer_set(candidate, pointer, value)
                axis_map = {}
                for axis, value in zip(axes, values): pointer_set(candidate, axis["path"], value); axis_map[axis["path"]] = copy.deepcopy(value)
                spec = RunSpec.from_dict(candidate); case_seeds.add(spec.data["seed"])
                logical_key = json.dumps({"case_id": case["id"], "axes": axis_map}, sort_keys=True, separators=(",", ":"))
                expanded.append(ExpandedRun(len(expanded), logical_key, case["id"], axis_map, spec))
            expected = set(self.data["collection"]["expected_seeds"])
            if case_seeds != expected: raise SpecValidationError(f"Expected every case to contain seeds {sorted(expected)!r}. Provided case {case['id']!r} seeds: {sorted(case_seeds)!r}.")
        # Compare with the base as well as across jobs.  Otherwise a linked
        # override shared by every job (or the sole job in a one-case sweep)
        # would be mistaken for an unused varying field.
        flattened = [
            _flatten(RunSpec.from_dict(base).identity_payload()),
            *(_flatten(item.spec.identity_payload()) for item in expanded),
        ]
        leaves = sorted(set().union(*(item.keys() for item in flattened))) if flattened else []
        changed = {leaf for leaf in leaves if len({json.dumps(item.get(leaf), sort_keys=True, separators=(",", ":")) for item in flattened}) > 1}
        declared = self.data["varying_fields"]
        uncovered = sorted(leaf for leaf in changed if not any(_covers(item, leaf) for item in declared))
        unused = sorted(item for item in declared if not any(_covers(item, leaf) for leaf in changed))
        if uncovered or unused: raise SpecValidationError(f"Expected varying_fields to cover exactly all scientific variation. Provided uncovered: {uncovered!r}; unused: {unused!r}.")
        return expanded
