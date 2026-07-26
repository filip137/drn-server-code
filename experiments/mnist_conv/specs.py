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
RUN_SCHEMA_VERSION_V2 = "mnist-conv-run/v2"
RUN_SCHEMA_VERSION_V3 = "mnist-conv-run/v3"
RUN_SCHEMA_VERSION_V4 = "mnist-conv-run/v4"
RUN_SCHEMA_VERSION_V5 = "mnist-conv-run/v5"
RUN_SCHEMA_VERSION_V6 = "mnist-conv-run/v6"
RUN_SCHEMA_VERSION_V7 = "mnist-conv-run/v7"
LR_STAGE_RUN_SCHEMA_VERSIONS = frozenset(
    {
        RUN_SCHEMA_VERSION_V2,
        RUN_SCHEMA_VERSION_V3,
        RUN_SCHEMA_VERSION_V4,
        RUN_SCHEMA_VERSION_V5,
        RUN_SCHEMA_VERSION_V6,
        RUN_SCHEMA_VERSION_V7,
    }
)
SWEEP_SCHEMA_VERSION = "mnist-conv-sweep/v1"
WEIGHT_INIT_MODES = {
    "bounded_uniform",
    "kaiming_normal",
    "kaiming_uniform",
    "xavier_normal",
    "xavier_uniform",
}

OPTIMIZER_BOUNDARY_PROTOCOL_ID = (
    "conv-hardsigmoid-lr-conv2-sgd-adam-boundary-constant-bs16-v1"
)
CONV3_ORDINARY_LR_PROTOCOL_ID = (
    "conv-hardsigmoid-lr-conv3-scheme-two-rho-"
    "median-constant-sgd-bs16-v7"
)

SGD_OPTIMIZER_V7: dict[str, Any] = {
    "name": "SGD",
    "momentum": 0.0,
    "weight_decay": 0.0,
}

ADAM_OPTIMIZER_V7: dict[str, Any] = {
    "name": "Adam",
    "betas": [0.9, 0.999],
    "eps": 1e-8,
    "weight_decay": 0.0,
    "amsgrad": False,
    "foreach": False,
    "fused": False,
    "maximize": False,
    "capturable": False,
    "differentiable": False,
}


def require_generic_run_schema(
    schema_version: Any,
    *,
    surface: str,
) -> None:
    """Keep LR-stage candidate bundles off the generic run/sweep surfaces."""

    if schema_version == RUN_SCHEMA_VERSION:
        return
    expected = (
        f"the generic MNIST Conv {surface} surface to receive schema_version "
        f"{RUN_SCHEMA_VERSION!r}; LR-stage candidate schemas "
        f"{sorted(LR_STAGE_RUN_SCHEMA_VERSIONS)!r} must use "
        "'python -m experiments.mnist_conv lr-study'"
    )
    raise SpecValidationError(
        f"Expected {expected}. Provided schema_version: {schema_version!r}."
    )

_V7_RHO_CONV_MAIN_GRID = (0.01, 0.015, 0.03)
_V7_RHO_CONV_SENTINEL = 0.1
_V7_RHO_DENSE_GRID_BY_SCHEME = {
    "baseline": (0.01, 0.03),
    "ours": (0.003, 0.01),
    "legacy": (0.003, 0.01),
}

_V6_RHO_CONV_GRID = (5e-4, 1e-3, 3e-3, 1e-2)
_V6_RHO_DENSE_GRID = (3e-3, 1e-2, 3e-2, 1e-1)
_V6_CONV2_ROWS: dict[str, dict[str, Any]] = {
    "conv2_baseline_v1_c1": {
        "stage": "baseline_grid",
        "voltage_amp": 1.0,
        "current_amp": 1.0,
        "input_gain": 253.302230835,
        "inference_iterations": 16,
        "training_iterations": 6,
    },
    "conv2_ours_v4_c1": {
        "stage": "confirmation",
        "voltage_amp": 4.0,
        "current_amp": 1.0,
        "input_gain": 716.3439331055,
        "inference_iterations": 24,
        "training_iterations": 6,
    },
    "conv2_legacy_v4_c0p25": {
        "stage": "confirmation",
        "voltage_amp": 4.0,
        "current_amp": 0.25,
        "input_gain": 661.4369506836,
        "inference_iterations": 8,
        "training_iterations": 4,
    },
}

_CONV1_SCHEME_RHO_CONV_GRID = (1.0 / 3000.0, 1e-3, 3e-3, 1e-2)
_CONV1_SCHEME_RHO_DENSE_GRID = (3e-3, 1e-2, 3e-2)
_CONV1_SCHEME_RHO_ROWS: dict[str, dict[str, Any]] = {
    "conv1_ours_v4_c1": {
        "voltage_amp": 4.0,
        "current_amp": 1.0,
        "input_gain": 84.8402175903,
        "inference_iterations": 4,
        "training_iterations": 4,
    },
    "conv1_legacy_v4_c0p25": {
        "voltage_amp": 4.0,
        "current_amp": 0.25,
        "input_gain": 31.8188591003,
        "inference_iterations": 4,
        "training_iterations": 4,
    },
}

_CONV2_SCHEME_RHO_CONV_GRID = (5e-4, 1.5e-3, 3e-3, 1e-2)
_CONV2_SCHEME_RHO_DENSE_GRID = (3e-3, 1e-2, 3e-2)
_CONV2_SCHEME_RHO_ROWS: dict[str, dict[str, Any]] = {
    row_id: contract
    for row_id, contract in _V6_CONV2_ROWS.items()
    if row_id in {"conv2_ours_v4_c1", "conv2_legacy_v4_c0p25"}
}

_CONV3_ORDINARY_RHO_CONV_GRID = (5e-4, 3e-3, 1e-2, 3e-2)
_CONV3_ORDINARY_RHO_DENSE_GRID = (3e-3, 1e-2, 3e-2, 1e-1)
_CONV3_LEGACY_RESCUE_RHO_PAIRS = (
    (5e-5, 3e-4),
    (1.5e-5, 1e-4),
    (5e-6, 3e-5),
)
_CONV3_ORDINARY_ROWS: dict[str, dict[str, Any]] = {
    "conv3_baseline_v1_c1": {
        "voltage_amp": 1.0,
        "current_amp": 1.0,
        "input_gain": 251.3061370850,
        "inference_iterations": 24,
        "training_iterations": 8,
    },
    "conv3_ours_v4_c1": {
        "voltage_amp": 4.0,
        "current_amp": 1.0,
        "input_gain": 744.7390747070,
        "inference_iterations": 32,
        "training_iterations": 8,
    },
    "conv3_legacy_v4_c0p25": {
        "voltage_amp": 4.0,
        "current_amp": 0.25,
        "input_gain": 665.0302124023,
        "inference_iterations": 8,
        "training_iterations": 6,
    },
}

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


def _normalize_dataset_v2(value: Any) -> dict[str, Any]:
    """Normalize the validation-only dataset contract used by LR candidates.

    V2 is deliberately separate from the established v1 contract: existing
    bundles retain their original schema and test-split semantics, while LR
    studies must make the train/validation split and loader RNG independent of
    model construction explicit.
    """

    dataset = _object(value, "run.dataset")
    keys = {
        "name", "input_shape", "batch_size", "normalization", "affine",
        "max_batches", "train_shuffle_seed", "validation",
    }
    _exact_keys(dataset, keys, "run.dataset")
    base = _normalize_dataset(
        {
            "name": dataset["name"],
            "input_shape": dataset["input_shape"],
            "batch_size": dataset["batch_size"],
            "normalization": dataset["normalization"],
            "affine": dataset["affine"],
            "max_batches": dataset["max_batches"],
            "max_test_batches": None,
        }
    )
    if base["batch_size"] != 16:
        raise _error("exactly 16 for the frozen LR study", base["batch_size"], "run.dataset.batch_size")
    if base["max_batches"] is not None:
        raise _error("null for complete five-epoch candidate runs", base["max_batches"], "run.dataset.max_batches")
    validation = _object(dataset["validation"], "run.dataset.validation")
    _exact_keys(
        validation,
        {
            "source", "size", "samples_per_class", "split_seed",
            "batch_size", "stratified", "official_test_enabled",
            "indices_sha256",
        },
        "run.dataset.validation",
    )
    if validation["source"] != "mnist_train":
        raise _error("exactly 'mnist_train'", validation["source"], "run.dataset.validation.source")
    if validation["size"] != 5000:
        raise _error("exactly 5000", validation["size"], "run.dataset.validation.size")
    if validation["samples_per_class"] != 500:
        raise _error("exactly 500", validation["samples_per_class"], "run.dataset.validation.samples_per_class")
    if validation["split_seed"] != 0:
        raise _error("exactly 0", validation["split_seed"], "run.dataset.validation.split_seed")
    if validation["batch_size"] != 128:
        raise _error("exactly 128", validation["batch_size"], "run.dataset.validation.batch_size")
    if validation["stratified"] is not True:
        raise _error("true", validation["stratified"], "run.dataset.validation.stratified")
    if validation["official_test_enabled"] is not False:
        raise _error("false for LR selection", validation["official_test_enabled"], "run.dataset.validation.official_test_enabled")
    digest = validation["indices_sha256"]
    if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
        raise _error("a lowercase SHA-256 digest", digest, "run.dataset.validation.indices_sha256")
    base.pop("max_test_batches")
    base["train_shuffle_seed"] = _integer(
        dataset["train_shuffle_seed"], "run.dataset.train_shuffle_seed"
    )
    if base["train_shuffle_seed"] != 0:
        raise _error("exactly 0", base["train_shuffle_seed"], "run.dataset.train_shuffle_seed")
    base["validation"] = {
        "source": "mnist_train",
        "size": 5000,
        "samples_per_class": 500,
        "split_seed": 0,
        "batch_size": 128,
        "stratified": True,
        "official_test_enabled": False,
        "indices_sha256": digest,
    }
    return base


def _normalize_dataset_conv3_v7(value: Any) -> dict[str, Any]:
    """Normalize the v7 ordinary-MNIST split with validation batches of 64."""

    dataset = _object(value, "run.dataset")
    validation = _object(dataset.get("validation"), "run.dataset.validation")
    if validation.get("batch_size") != 64:
        raise _error(
            "exactly 64 for the Conv3 v7 study",
            validation.get("batch_size"),
            "run.dataset.validation.batch_size",
        )
    compatibility = copy.deepcopy(dataset)
    compatibility["validation"]["batch_size"] = 128
    normalized = _normalize_dataset_v2(compatibility)
    normalized["validation"]["batch_size"] = 64
    return normalized


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
    weight_init_mode = _string(model["weight_init_mode"], "run.model.weight_init_mode")
    if weight_init_mode not in WEIGHT_INIT_MODES:
        raise _error(
            f"one of {sorted(WEIGHT_INIT_MODES)!r}",
            weight_init_mode,
            "run.model.weight_init_mode",
        )
    if weight_init_mode == "bounded_uniform" and weight_min >= weight_max:
        raise _error(
            "< weight_max for bounded_uniform initialization",
            weight_min,
            "run.model.weight_min",
        )
    model.update({
        "voltage_amp": _finite(model["voltage_amp"], "run.model.voltage_amp", positive=True),
        "current_amp": _finite(model["current_amp"], "run.model.current_amp", positive=True),
        "input_gain": _finite(model["input_gain"], "run.model.input_gain", positive=True),
        "weight_gains": _finite_list(model["weight_gains"], "run.model.weight_gains", depth + 1, positive=True),
        "weight_min": weight_min, "weight_max": weight_max,
        "weight_init_mode": weight_init_mode,
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


def _normalize_training_v2(value: Any, depth: int) -> dict[str, Any]:
    training = _object(value, "run.training")
    keys = {
        "algorithm", "epochs", "optimizer", "peak_learning_rate", "schedule",
        "beta", "checkpoint_rule", "pruning", "batch_state_policy",
        "diagnostics",
    }
    _exact_keys(training, keys, "run.training")
    if training["algorithm"] != "BP":
        raise _error("exactly 'BP'", training["algorithm"], "run.training.algorithm")
    if training["epochs"] != 5:
        raise _error("exactly 5 for the frozen LR screen", training["epochs"], "run.training.epochs")
    if training["batch_state_policy"] != "reset_each_batch":
        raise _error("exactly 'reset_each_batch'", training["batch_state_policy"], "run.training.batch_state_policy")
    if training["checkpoint_rule"] != "min_validation_loss":
        raise _error("exactly 'min_validation_loss'", training["checkpoint_rule"], "run.training.checkpoint_rule")

    optimizer = _object(training["optimizer"], "run.training.optimizer")
    _exact_keys(optimizer, {"name", "momentum", "weight_decay"}, "run.training.optimizer")
    if optimizer != {"name": "SGD", "momentum": 0.0, "weight_decay": 0.0}:
        raise _error(
            "exactly {'name': 'SGD', 'momentum': 0.0, 'weight_decay': 0.0}",
            optimizer,
            "run.training.optimizer",
        )

    peak = _finite(training["peak_learning_rate"], "run.training.peak_learning_rate", positive=True)
    schedule = _object(training["schedule"], "run.training.schedule")
    _exact_keys(
        schedule,
        {"name", "interval", "warmup_fraction", "warmup_steps", "total_steps", "min_lr_factor"},
        "run.training.schedule",
    )
    expected_schedule = {
        "name": "linear_warmup_cosine",
        "interval": "optimizer_step",
        "warmup_fraction": 0.05,
        "warmup_steps": 860,
        "total_steps": 17190,
        "min_lr_factor": 0.0,
    }
    normalized_schedule = {
        "name": schedule["name"],
        "interval": schedule["interval"],
        "warmup_fraction": _finite(schedule["warmup_fraction"], "run.training.schedule.warmup_fraction"),
        "warmup_steps": _integer(schedule["warmup_steps"], "run.training.schedule.warmup_steps", minimum=1),
        "total_steps": _integer(schedule["total_steps"], "run.training.schedule.total_steps", minimum=1),
        "min_lr_factor": _finite(schedule["min_lr_factor"], "run.training.schedule.min_lr_factor"),
    }
    if normalized_schedule != expected_schedule:
        raise _error(f"the frozen schedule {expected_schedule!r}", normalized_schedule, "run.training.schedule")

    pruning = _object(training["pruning"], "run.training.pruning")
    _exact_keys(pruning, {"enabled", "after_epoch", "min_best_validation_accuracy"}, "run.training.pruning")
    if pruning != {"enabled": False, "after_epoch": None, "min_best_validation_accuracy": None}:
        raise _error("the disabled LR-screen pruning contract", pruning, "run.training.pruning")

    diagnostics = _object(training["diagnostics"], "run.training.diagnostics")
    _exact_keys(
        diagnostics,
        {
            "enabled", "bounded_parameter_classes", "bias_global_gate", "range_epsilon",
            "projection_epsilon", "early_fraction", "projection_efficiency_threshold",
            "projection_persistence", "occupancy_delta_threshold", "occupancy_persistence",
        },
        "run.training.diagnostics",
    )
    expected_diagnostics = {
        "enabled": True,
        "bounded_parameter_classes": ["ConvWeight", "DenseWeight"],
        "bias_global_gate": False,
        "range_epsilon": 1e-8,
        "projection_epsilon": 1e-12,
        "early_fraction": 0.2,
        "projection_efficiency_threshold": 0.5,
        "projection_persistence": 16,
        "occupancy_delta_threshold": 0.2,
        "occupancy_persistence": 16,
    }
    normalized_diagnostics = {
        "enabled": diagnostics["enabled"],
        "bounded_parameter_classes": diagnostics["bounded_parameter_classes"],
        "bias_global_gate": diagnostics["bias_global_gate"],
        "range_epsilon": _finite(diagnostics["range_epsilon"], "run.training.diagnostics.range_epsilon", positive=True),
        "projection_epsilon": _finite(diagnostics["projection_epsilon"], "run.training.diagnostics.projection_epsilon", positive=True),
        "early_fraction": _finite(diagnostics["early_fraction"], "run.training.diagnostics.early_fraction"),
        "projection_efficiency_threshold": _finite(diagnostics["projection_efficiency_threshold"], "run.training.diagnostics.projection_efficiency_threshold"),
        "projection_persistence": _integer(diagnostics["projection_persistence"], "run.training.diagnostics.projection_persistence", minimum=1),
        "occupancy_delta_threshold": _finite(diagnostics["occupancy_delta_threshold"], "run.training.diagnostics.occupancy_delta_threshold"),
        "occupancy_persistence": _integer(diagnostics["occupancy_persistence"], "run.training.diagnostics.occupancy_persistence", minimum=1),
    }
    if normalized_diagnostics != expected_diagnostics:
        raise _error(
            f"the frozen LR diagnostic settings {expected_diagnostics!r}",
            normalized_diagnostics,
            "run.training.diagnostics",
        )
    return {
        "algorithm": "BP",
        "epochs": 5,
        "optimizer": {"name": "SGD", "momentum": 0.0, "weight_decay": 0.0},
        "peak_learning_rate": peak,
        "schedule": expected_schedule,
        "beta": _finite(training["beta"], "run.training.beta"),
        "checkpoint_rule": "min_validation_loss",
        "pruning": {"enabled": False, "after_epoch": None, "min_best_validation_accuracy": None},
        "batch_state_policy": "reset_each_batch",
        "diagnostics": expected_diagnostics,
    }


def _layerwise_parameter_names(depth: int) -> tuple[str, ...]:
    if depth not in (1, 2, 3):
        raise _error(
            "one, two, or three convolution layers", depth, "run.architecture"
        )
    return tuple(
        [f"ConvWeight_{index}" for index in range(depth)]
        + ["DenseWeight_0"]
        + [f"Bias_{index}" for index in range(depth)]
    )


def _normalize_training_v3(value: Any, depth: int) -> dict[str, Any]:
    """Normalize the constant layer-wise five-epoch diagnostic contract."""

    training = _object(value, "run.training")
    keys = {
        "algorithm", "epochs", "optimizer", "learning_rates_by_parameter",
        "schedule", "beta", "checkpoint_rule", "pruning",
        "batch_state_policy", "diagnostics",
    }
    _exact_keys(training, keys, "run.training")
    if training["algorithm"] != "BP":
        raise _error("exactly 'BP'", training["algorithm"], "run.training.algorithm")
    if training["epochs"] != 5:
        raise _error("exactly 5", training["epochs"], "run.training.epochs")
    if training["batch_state_policy"] != "reset_each_batch":
        raise _error(
            "exactly 'reset_each_batch'",
            training["batch_state_policy"],
            "run.training.batch_state_policy",
        )
    if training["checkpoint_rule"] != "min_validation_loss":
        raise _error(
            "exactly 'min_validation_loss'",
            training["checkpoint_rule"],
            "run.training.checkpoint_rule",
        )
    optimizer = _object(training["optimizer"], "run.training.optimizer")
    expected_optimizer = {"name": "SGD", "momentum": 0.0, "weight_decay": 0.0}
    if optimizer != expected_optimizer:
        raise _error(
            f"exactly {expected_optimizer!r}", optimizer, "run.training.optimizer"
        )

    rates = _object(
        training["learning_rates_by_parameter"],
        "run.training.learning_rates_by_parameter",
    )
    expected_names = _layerwise_parameter_names(depth)
    _exact_keys(rates, expected_names, "run.training.learning_rates_by_parameter")
    normalized_rates = {
        name: _finite(
            rates[name],
            f"run.training.learning_rates_by_parameter.{name}",
            positive=True,
        )
        for name in expected_names
    }
    for index in range(depth):
        weight = f"ConvWeight_{index}"
        bias = f"Bias_{index}"
        if normalized_rates[bias] != normalized_rates[weight]:
            raise _error(
                f"exactly the associated {weight} rate {normalized_rates[weight]!r}",
                normalized_rates[bias],
                f"run.training.learning_rates_by_parameter.{bias}",
            )

    schedule = _object(training["schedule"], "run.training.schedule")
    expected_schedule = {
        "name": "constant",
        "interval": "optimizer_step",
        "total_steps": 17190,
        "scheduler_enabled": False,
    }
    if schedule != expected_schedule:
        raise _error(
            f"the frozen constant schedule {expected_schedule!r}",
            schedule,
            "run.training.schedule",
        )
    pruning = _object(training["pruning"], "run.training.pruning")
    expected_pruning = {
        "enabled": False,
        "after_epoch": None,
        "min_best_validation_accuracy": None,
    }
    if pruning != expected_pruning:
        raise _error(
            "the disabled LR-screen pruning contract",
            pruning,
            "run.training.pruning",
        )
    diagnostics = _object(training["diagnostics"], "run.training.diagnostics")
    expected_diagnostics = {
        "enabled": True,
        "bounded_parameter_classes": ["ConvWeight", "DenseWeight"],
        "bias_global_gate": False,
        "range_epsilon": 1e-8,
        "projection_epsilon": 1e-12,
        "early_fraction": 0.2,
        "projection_efficiency_threshold": 0.5,
        "projection_persistence": 16,
        "occupancy_delta_threshold": 0.2,
        "occupancy_persistence": 16,
    }
    if diagnostics != expected_diagnostics:
        raise _error(
            f"the frozen LR diagnostic settings {expected_diagnostics!r}",
            diagnostics,
            "run.training.diagnostics",
        )
    return {
        "algorithm": "BP",
        "epochs": 5,
        "optimizer": expected_optimizer,
        "learning_rates_by_parameter": normalized_rates,
        "schedule": expected_schedule,
        "beta": _finite(training["beta"], "run.training.beta"),
        "checkpoint_rule": "min_validation_loss",
        "pruning": expected_pruning,
        "batch_state_policy": "reset_each_batch",
        "diagnostics": expected_diagnostics,
    }


def _normalize_training_v4(value: Any, depth: int) -> dict[str, Any]:
    """V5 keeps the v3 constant-vector contract but gates the complete run."""

    training = copy.deepcopy(_object(value, "run.training"))
    diagnostics = _object(training.get("diagnostics"), "run.training.diagnostics")
    if diagnostics.get("early_fraction") != 1.0:
        raise _error(
            "exactly 1.0 for full-run v5 safety gates",
            diagnostics.get("early_fraction"),
            "run.training.diagnostics.early_fraction",
        )
    training["diagnostics"]["early_fraction"] = 0.2
    normalized = _normalize_training_v3(training, depth)
    normalized["diagnostics"]["early_fraction"] = 1.0
    return normalized


def _normalize_training_conv3_v7(value: Any, depth: int) -> dict[str, Any]:
    """Normalize the exact three-epoch constant-vector Conv3 LR screen."""

    if depth != 3:
        raise _error(
            "exactly three convolution layers for Conv3 v7",
            depth,
            "run.architecture",
        )
    training = copy.deepcopy(_object(value, "run.training"))
    if training.get("epochs") != 3:
        raise _error(
            "exactly 3 for the Conv3 v7 LR screen",
            training.get("epochs"),
            "run.training.epochs",
        )
    schedule = _object(training.get("schedule"), "run.training.schedule")
    expected_schedule = {
        "name": "constant",
        "interval": "optimizer_step",
        "total_steps": 10314,
        "scheduler_enabled": False,
    }
    if schedule != expected_schedule:
        raise _error(
            f"the frozen constant schedule {expected_schedule!r}",
            schedule,
            "run.training.schedule",
        )
    compatibility = copy.deepcopy(training)
    compatibility["epochs"] = 5
    compatibility["schedule"]["total_steps"] = 17190
    normalized = _normalize_training_v4(compatibility, depth)
    normalized["epochs"] = 3
    normalized["schedule"] = expected_schedule
    return normalized


def normalize_optimizer_v7(value: Any, path: str = "run.training.optimizer") -> dict[str, Any]:
    """Normalize the exact optimizer union frozen by the boundary diagnostic.

    The explicit false-valued Adam flags are scientific inputs.  In
    particular, omitting a flag and relying on a PyTorch-version default is
    rejected rather than silently changing the run identity.
    """

    optimizer = _object(value, path)
    name = optimizer.get("name")
    if name == "SGD":
        _exact_keys(optimizer, SGD_OPTIMIZER_V7, path)
        normalized = {
            "name": "SGD",
            "momentum": _finite(optimizer["momentum"], f"{path}.momentum"),
            "weight_decay": _finite(
                optimizer["weight_decay"], f"{path}.weight_decay"
            ),
        }
        if normalized != SGD_OPTIMIZER_V7:
            raise _error(f"exactly {SGD_OPTIMIZER_V7!r}", normalized, path)
        return normalized

    if name == "Adam":
        _exact_keys(optimizer, ADAM_OPTIMIZER_V7, path)
        betas = _finite_list(optimizer["betas"], f"{path}.betas", 2)
        normalized = {
            "name": "Adam",
            "betas": betas,
            "eps": _finite(optimizer["eps"], f"{path}.eps", positive=True),
            "weight_decay": _finite(
                optimizer["weight_decay"], f"{path}.weight_decay"
            ),
        }
        for flag in (
            "amsgrad",
            "foreach",
            "fused",
            "maximize",
            "capturable",
            "differentiable",
        ):
            if optimizer[flag] is not False:
                raise _error("exactly false", optimizer[flag], f"{path}.{flag}")
            normalized[flag] = False
        if normalized != ADAM_OPTIMIZER_V7:
            raise _error(f"exactly {ADAM_OPTIMIZER_V7!r}", normalized, path)
        return normalized

    raise _error("'SGD' or 'Adam'", name, f"{path}.name")


def _normalize_training_v7(value: Any, depth: int) -> dict[str, Any]:
    """Normalize an optimizer-neutral constant-vector five-epoch run.

    V7 keeps the established v5/v6 schedule and safety diagnostics, while
    allowing the exact SGD/Adam union and independently specified bias rates.
    Every scientific parameter must be represented by one explicit learning
    rate; there is no optimizer-level fallback LR.
    """

    training = _object(value, "run.training")
    expected_names = _layerwise_parameter_names(depth)
    optimizer = normalize_optimizer_v7(training.get("optimizer"))
    rates = _object(
        training.get("learning_rates_by_parameter"),
        "run.training.learning_rates_by_parameter",
    )
    _exact_keys(rates, expected_names, "run.training.learning_rates_by_parameter")
    normalized_rates = {
        name: _finite(
            rates[name],
            f"run.training.learning_rates_by_parameter.{name}",
            positive=True,
        )
        for name in expected_names
    }

    # Reuse the frozen constant-schedule and diagnostics validator without
    # weakening the established v1--v6 contract.  Its two SGD-only checks are
    # supplied with validation-only placeholders, then replaced by the
    # already-normalized v7 scientific values.
    compatibility = copy.deepcopy(training)
    compatibility["optimizer"] = copy.deepcopy(SGD_OPTIMIZER_V7)
    for index in range(depth):
        compatibility["learning_rates_by_parameter"][f"Bias_{index}"] = (
            normalized_rates[f"ConvWeight_{index}"]
        )
    normalized = _normalize_training_v4(compatibility, depth)
    normalized["optimizer"] = optimizer
    normalized["learning_rates_by_parameter"] = normalized_rates
    return normalized


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


def _normalize_lr_provenance(value: Any) -> dict[str, Any]:
    provenance = _object(value, "run.lr_provenance")
    keys = {
        "study_id", "row_id", "candidate_role", "rho_target", "rho_unit",
        "probe_sha256", "range_sha256", "split_sha256", "batch_order_sha256",
        "initialization_tensor_sha256",
    }
    _exact_keys(provenance, keys, "run.lr_provenance")
    if not isinstance(provenance["study_id"], str) or re.fullmatch(
        r"lrstudy_[0-9a-f]{64}", provenance["study_id"]
    ) is None:
        raise _error(
            "lrstudy_ followed by 64 lowercase hexadecimal characters",
            provenance["study_id"],
            "run.lr_provenance.study_id",
        )
    row_id = _string(provenance["row_id"], "run.lr_provenance.row_id")
    role = provenance["candidate_role"]
    if role not in {"fast", "middle", "high"}:
        raise _error("'fast', 'middle', or 'high'", role, "run.lr_provenance.candidate_role")
    result = {
        "study_id": provenance["study_id"],
        "row_id": row_id,
        "candidate_role": role,
        "rho_target": _finite(provenance["rho_target"], "run.lr_provenance.rho_target", positive=True),
        "rho_unit": _finite(provenance["rho_unit"], "run.lr_provenance.rho_unit", positive=True),
    }
    for field in (
        "probe_sha256", "range_sha256", "split_sha256", "batch_order_sha256",
        "initialization_tensor_sha256",
    ):
        digest = provenance[field]
        if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            raise _error("a lowercase SHA-256 digest", digest, f"run.lr_provenance.{field}")
        result[field] = digest
    return result


def _normalize_lr_provenance_v3(value: Any, depth: int) -> dict[str, Any]:
    provenance = _object(value, "run.lr_provenance")
    keys = {
        "study_id", "row_id", "candidate_role", "rho_target",
        "rho_unit_by_weight", "bias_weight_mapping", "probe_sha256",
        "range_sha256", "split_sha256", "batch_order_sha256",
        "initialization_tensor_sha256",
    }
    _exact_keys(provenance, keys, "run.lr_provenance")
    study_id = provenance["study_id"]
    if not isinstance(study_id, str) or re.fullmatch(
        r"lrstudy_[0-9a-f]{64}", study_id
    ) is None:
        raise _error(
            "lrstudy_ followed by 64 lowercase hexadecimal characters",
            study_id,
            "run.lr_provenance.study_id",
        )
    role = provenance["candidate_role"]
    if role not in {"fast", "middle", "high"}:
        raise _error(
            "'fast', 'middle', or 'high'",
            role,
            "run.lr_provenance.candidate_role",
        )
    units = _object(provenance["rho_unit_by_weight"], "run.lr_provenance.rho_unit_by_weight")
    expected_weights = tuple(
        [f"ConvWeight_{index}" for index in range(depth)] + ["DenseWeight_0"]
    )
    _exact_keys(units, expected_weights, "run.lr_provenance.rho_unit_by_weight")
    normalized_units = {
        name: _finite(
            units[name],
            f"run.lr_provenance.rho_unit_by_weight.{name}",
            positive=True,
        )
        for name in expected_weights
    }
    mapping = _object(
        provenance["bias_weight_mapping"],
        "run.lr_provenance.bias_weight_mapping",
    )
    expected_mapping = {
        f"Bias_{index}": f"ConvWeight_{index}" for index in range(depth)
    }
    if mapping != expected_mapping:
        raise _error(
            f"exactly {expected_mapping!r}",
            mapping,
            "run.lr_provenance.bias_weight_mapping",
        )
    result = {
        "study_id": study_id,
        "row_id": _string(provenance["row_id"], "run.lr_provenance.row_id"),
        "candidate_role": role,
        "rho_target": _finite(
            provenance["rho_target"],
            "run.lr_provenance.rho_target",
            positive=True,
        ),
        "rho_unit_by_weight": normalized_units,
        "bias_weight_mapping": expected_mapping,
    }
    for field in (
        "probe_sha256", "range_sha256", "split_sha256",
        "batch_order_sha256", "initialization_tensor_sha256",
    ):
        digest = provenance[field]
        if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            raise _error(
                "a lowercase SHA-256 digest",
                digest,
                f"run.lr_provenance.{field}",
            )
        result[field] = digest
    return result


def _normalize_lr_provenance_v4(value: Any, depth: int) -> dict[str, Any]:
    """Normalize the architecture-alpha provenance recorded by v5 candidates."""

    provenance = _object(value, "run.lr_provenance")
    keys = {
        "study_id", "row_id", "candidate_arm", "alpha_role", "alpha_arch",
        "median_unit_by_weight", "target_multipliers_by_weight",
        "bias_weight_mapping", "probe_sha256", "anchor_audit_sha256",
        "split_sha256", "batch_order_sha256", "initialization_tensor_sha256",
    }
    _exact_keys(provenance, keys, "run.lr_provenance")
    study_id = provenance["study_id"]
    if not isinstance(study_id, str) or re.fullmatch(
        r"lrstudy_[0-9a-f]{64}", study_id
    ) is None:
        raise _error(
            "lrstudy_ followed by 64 lowercase hexadecimal characters",
            study_id,
            "run.lr_provenance.study_id",
        )
    arm = provenance["candidate_arm"]
    if arm not in {"strict_equal", "historical_profile"}:
        raise _error(
            "'strict_equal' or 'historical_profile'",
            arm,
            "run.lr_provenance.candidate_arm",
        )
    alpha_role = provenance["alpha_role"]
    if alpha_role not in {"lower", "center", "upper"}:
        raise _error(
            "'lower', 'center', or 'upper'",
            alpha_role,
            "run.lr_provenance.alpha_role",
        )
    expected_weights = tuple(
        [f"ConvWeight_{index}" for index in range(depth)] + ["DenseWeight_0"]
    )
    units = _object(
        provenance["median_unit_by_weight"],
        "run.lr_provenance.median_unit_by_weight",
    )
    _exact_keys(units, expected_weights, "run.lr_provenance.median_unit_by_weight")
    normalized_units = {
        name: _finite(
            units[name],
            f"run.lr_provenance.median_unit_by_weight.{name}",
            positive=True,
        )
        for name in expected_weights
    }
    multipliers = _object(
        provenance["target_multipliers_by_weight"],
        "run.lr_provenance.target_multipliers_by_weight",
    )
    _exact_keys(
        multipliers,
        expected_weights,
        "run.lr_provenance.target_multipliers_by_weight",
    )
    normalized_multipliers = {
        name: _finite(
            multipliers[name],
            f"run.lr_provenance.target_multipliers_by_weight.{name}",
            positive=True,
        )
        for name in expected_weights
    }
    expected_mapping = {
        f"Bias_{index}": f"ConvWeight_{index}" for index in range(depth)
    }
    mapping = _object(
        provenance["bias_weight_mapping"],
        "run.lr_provenance.bias_weight_mapping",
    )
    if mapping != expected_mapping:
        raise _error(
            f"exactly {expected_mapping!r}",
            mapping,
            "run.lr_provenance.bias_weight_mapping",
        )
    result = {
        "study_id": study_id,
        "row_id": _string(provenance["row_id"], "run.lr_provenance.row_id"),
        "candidate_arm": arm,
        "alpha_role": alpha_role,
        "alpha_arch": _finite(
            provenance["alpha_arch"],
            "run.lr_provenance.alpha_arch",
            positive=True,
        ),
        "median_unit_by_weight": normalized_units,
        "target_multipliers_by_weight": normalized_multipliers,
        "bias_weight_mapping": expected_mapping,
    }
    for field in (
        "probe_sha256", "anchor_audit_sha256", "split_sha256",
        "batch_order_sha256", "initialization_tensor_sha256",
    ):
        digest = provenance[field]
        if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            raise _error(
                "a lowercase SHA-256 digest",
                digest,
                f"run.lr_provenance.{field}",
            )
        result[field] = digest
    return result


def _normalize_lr_provenance_v5(value: Any, depth: int) -> dict[str, Any]:
    """Normalize the direct Conv/Dense relative-update targets used by v6.

    V6 intentionally has no architecture-level scalar, candidate arm, or
    target profile.  The exact-key check prevents any of those retired
    coordinates from being smuggled into a v5 run bundle.
    """

    provenance = _object(value, "run.lr_provenance")
    keys = {
        "study_id",
        "row_id",
        "candidate_stage",
        "rho_conv",
        "rho_dense",
        "median_unit_by_weight",
        "bias_weight_mapping",
        "probe_sha256",
        "split_sha256",
        "batch_order_sha256",
        "initialization_checkpoint_sha256",
        "initialization_tensor_sha256",
    }
    _exact_keys(provenance, keys, "run.lr_provenance")
    study_id = provenance["study_id"]
    if not isinstance(study_id, str) or re.fullmatch(
        r"lrstudy_[0-9a-f]{64}", study_id
    ) is None:
        raise _error(
            "lrstudy_ followed by 64 lowercase hexadecimal characters",
            study_id,
            "run.lr_provenance.study_id",
        )
    row_id = _string(provenance["row_id"], "run.lr_provenance.row_id")
    if row_id not in _V6_CONV2_ROWS:
        raise _error(
            f"one of {sorted(_V6_CONV2_ROWS)!r}",
            row_id,
            "run.lr_provenance.row_id",
        )
    candidate_stage = provenance["candidate_stage"]
    expected_stage = _V6_CONV2_ROWS[row_id]["stage"]
    if candidate_stage != expected_stage:
        raise _error(
            f"exactly {expected_stage!r} for row {row_id!r}",
            candidate_stage,
            "run.lr_provenance.candidate_stage",
        )

    rho_conv = _finite(
        provenance["rho_conv"], "run.lr_provenance.rho_conv", positive=True
    )
    rho_dense = _finite(
        provenance["rho_dense"], "run.lr_provenance.rho_dense", positive=True
    )
    if rho_conv not in _V6_RHO_CONV_GRID:
        raise _error(
            f"one of the frozen values {list(_V6_RHO_CONV_GRID)!r}",
            rho_conv,
            "run.lr_provenance.rho_conv",
        )
    if rho_dense not in _V6_RHO_DENSE_GRID:
        raise _error(
            f"one of the frozen values {list(_V6_RHO_DENSE_GRID)!r}",
            rho_dense,
            "run.lr_provenance.rho_dense",
        )

    expected_weights = tuple(
        [f"ConvWeight_{index}" for index in range(depth)] + ["DenseWeight_0"]
    )
    units = _object(
        provenance["median_unit_by_weight"],
        "run.lr_provenance.median_unit_by_weight",
    )
    _exact_keys(
        units, expected_weights, "run.lr_provenance.median_unit_by_weight"
    )
    normalized_units = {
        name: _finite(
            units[name],
            f"run.lr_provenance.median_unit_by_weight.{name}",
            positive=True,
        )
        for name in expected_weights
    }
    expected_mapping = {
        f"Bias_{index}": f"ConvWeight_{index}" for index in range(depth)
    }
    mapping = _object(
        provenance["bias_weight_mapping"],
        "run.lr_provenance.bias_weight_mapping",
    )
    if mapping != expected_mapping:
        raise _error(
            f"exactly {expected_mapping!r}",
            mapping,
            "run.lr_provenance.bias_weight_mapping",
        )

    result = {
        "study_id": study_id,
        "row_id": row_id,
        "candidate_stage": candidate_stage,
        "rho_conv": rho_conv,
        "rho_dense": rho_dense,
        "median_unit_by_weight": normalized_units,
        "bias_weight_mapping": expected_mapping,
    }
    for field in (
        "probe_sha256",
        "split_sha256",
        "batch_order_sha256",
        "initialization_checkpoint_sha256",
        "initialization_tensor_sha256",
    ):
        digest = provenance[field]
        if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            raise _error(
                "a lowercase SHA-256 digest",
                digest,
                f"run.lr_provenance.{field}",
            )
        result[field] = digest
    return result


def _normalize_lr_provenance_v6(
    value: Any,
    depth: int,
    architecture_profile: str,
) -> dict[str, Any]:
    """Normalize a direct two-rho amplified-scheme diagnostic."""

    if architecture_profile == "conv1":
        row_contracts = _CONV1_SCHEME_RHO_ROWS
        rho_conv_grid = _CONV1_SCHEME_RHO_CONV_GRID
        rho_dense_grid = _CONV1_SCHEME_RHO_DENSE_GRID
    elif architecture_profile == "conv2":
        row_contracts = _CONV2_SCHEME_RHO_ROWS
        rho_conv_grid = _CONV2_SCHEME_RHO_CONV_GRID
        rho_dense_grid = _CONV2_SCHEME_RHO_DENSE_GRID
    else:
        raise _error(
            "one of ['conv1', 'conv2'] for a direct two-rho diagnostic",
            architecture_profile,
            "run.architecture.profile",
        )

    provenance = _object(value, "run.lr_provenance")
    keys = {
        "study_id",
        "row_id",
        "candidate_stage",
        "rho_conv",
        "rho_dense",
        "median_unit_by_weight",
        "bias_weight_mapping",
        "probe_sha256",
        "split_sha256",
        "batch_order_sha256",
        "initialization_checkpoint_sha256",
        "initialization_tensor_sha256",
    }
    _exact_keys(provenance, keys, "run.lr_provenance")
    study_id = provenance["study_id"]
    if not isinstance(study_id, str) or re.fullmatch(
        r"lrstudy_[0-9a-f]{64}", study_id
    ) is None:
        raise _error(
            "lrstudy_ followed by 64 lowercase hexadecimal characters",
            study_id,
            "run.lr_provenance.study_id",
        )
    row_id = _string(provenance["row_id"], "run.lr_provenance.row_id")
    if row_id not in row_contracts:
        raise _error(
            f"one of {sorted(row_contracts)!r}",
            row_id,
            "run.lr_provenance.row_id",
        )
    candidate_stage = provenance["candidate_stage"]
    if candidate_stage != "scheme_grid":
        raise _error(
            "exactly 'scheme_grid'",
            candidate_stage,
            "run.lr_provenance.candidate_stage",
        )
    rho_conv = _finite(
        provenance["rho_conv"], "run.lr_provenance.rho_conv", positive=True
    )
    rho_dense = _finite(
        provenance["rho_dense"], "run.lr_provenance.rho_dense", positive=True
    )
    if rho_conv not in rho_conv_grid:
        raise _error(
            f"one of the frozen values {list(rho_conv_grid)!r}",
            rho_conv,
            "run.lr_provenance.rho_conv",
        )
    if rho_dense not in rho_dense_grid:
        raise _error(
            f"one of the frozen values {list(rho_dense_grid)!r}",
            rho_dense,
            "run.lr_provenance.rho_dense",
        )

    expected_weights = tuple(
        [f"ConvWeight_{index}" for index in range(depth)] + ["DenseWeight_0"]
    )
    units = _object(
        provenance["median_unit_by_weight"],
        "run.lr_provenance.median_unit_by_weight",
    )
    _exact_keys(
        units, expected_weights, "run.lr_provenance.median_unit_by_weight"
    )
    normalized_units = {
        name: _finite(
            units[name],
            f"run.lr_provenance.median_unit_by_weight.{name}",
            positive=True,
        )
        for name in expected_weights
    }
    expected_mapping = {
        f"Bias_{index}": f"ConvWeight_{index}" for index in range(depth)
    }
    mapping = _object(
        provenance["bias_weight_mapping"],
        "run.lr_provenance.bias_weight_mapping",
    )
    if mapping != expected_mapping:
        raise _error(
            f"exactly {expected_mapping!r}",
            mapping,
            "run.lr_provenance.bias_weight_mapping",
        )
    result = {
        "study_id": study_id,
        "row_id": row_id,
        "candidate_stage": candidate_stage,
        "rho_conv": rho_conv,
        "rho_dense": rho_dense,
        "median_unit_by_weight": normalized_units,
        "bias_weight_mapping": expected_mapping,
    }
    for field in (
        "probe_sha256",
        "split_sha256",
        "batch_order_sha256",
        "initialization_checkpoint_sha256",
        "initialization_tensor_sha256",
    ):
        digest = provenance[field]
        if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            raise _error(
                "a lowercase SHA-256 digest",
                digest,
                f"run.lr_provenance.{field}",
            )
        result[field] = digest
    return result


def _normalize_lr_provenance_conv3_v7(
    value: Any,
    depth: int,
) -> dict[str, Any]:
    """Normalize the adaptive ordinary-MNIST Conv3 two-rho provenance."""

    if depth != 3:
        raise _error(
            "exactly three convolution layers for Conv3 v7",
            depth,
            "run.architecture",
        )
    provenance = _object(value, "run.lr_provenance")
    keys = {
        "study_id",
        "row_id",
        "candidate_stage",
        "rho_conv",
        "rho_dense",
        "median_unit_by_weight",
        "bias_weight_mapping",
        "probe_sha256",
        "split_sha256",
        "batch_order_sha256",
        "initialization_checkpoint_sha256",
        "initialization_tensor_sha256",
        "gain_calibration_dataset",
        "optimization_dataset",
        "official_test_read",
    }
    _exact_keys(provenance, keys, "run.lr_provenance")
    study_id = provenance["study_id"]
    if not isinstance(study_id, str) or re.fullmatch(
        r"lrstudy_[0-9a-f]{64}", study_id
    ) is None:
        raise _error(
            "lrstudy_ followed by 64 lowercase hexadecimal characters",
            study_id,
            "run.lr_provenance.study_id",
        )
    row_id = _string(provenance["row_id"], "run.lr_provenance.row_id")
    if row_id not in _CONV3_ORDINARY_ROWS:
        raise _error(
            f"one of {sorted(_CONV3_ORDINARY_ROWS)!r}",
            row_id,
            "run.lr_provenance.row_id",
        )
    candidate_stage = provenance["candidate_stage"]
    if candidate_stage not in {
        "core_grid",
        "extension_grid",
        "legacy_rescue_promotion",
    }:
        raise _error(
            "'core_grid', 'extension_grid', or 'legacy_rescue_promotion'",
            candidate_stage,
            "run.lr_provenance.candidate_stage",
        )
    rho_conv = _finite(
        provenance["rho_conv"], "run.lr_provenance.rho_conv", positive=True
    )
    rho_dense = _finite(
        provenance["rho_dense"], "run.lr_provenance.rho_dense", positive=True
    )
    if candidate_stage == "legacy_rescue_promotion":
        if (rho_conv, rho_dense) not in _CONV3_LEGACY_RESCUE_RHO_PAIRS:
            raise _error(
                "one of the frozen Conv3 legacy rescue rho pairs "
                f"{list(_CONV3_LEGACY_RESCUE_RHO_PAIRS)!r}",
                {"rho_conv": rho_conv, "rho_dense": rho_dense},
                "run.lr_provenance",
            )
    else:
        if rho_conv not in _CONV3_ORDINARY_RHO_CONV_GRID:
            raise _error(
                f"one of the frozen values {list(_CONV3_ORDINARY_RHO_CONV_GRID)!r}",
                rho_conv,
                "run.lr_provenance.rho_conv",
            )
        if rho_dense not in _CONV3_ORDINARY_RHO_DENSE_GRID:
            raise _error(
                f"one of the frozen values {list(_CONV3_ORDINARY_RHO_DENSE_GRID)!r}",
                rho_dense,
                "run.lr_provenance.rho_dense",
            )
    if candidate_stage == "core_grid" and (
        rho_conv == _CONV3_ORDINARY_RHO_CONV_GRID[-1]
        or rho_dense == _CONV3_ORDINARY_RHO_DENSE_GRID[-1]
    ):
        raise _error(
            "a core-grid pair excluding extension-only targets",
            {"rho_conv": rho_conv, "rho_dense": rho_dense},
            "run.lr_provenance.candidate_stage",
        )
    if candidate_stage == "extension_grid" and (
        rho_conv != _CONV3_ORDINARY_RHO_CONV_GRID[-1]
        and rho_dense != _CONV3_ORDINARY_RHO_DENSE_GRID[-1]
    ):
        raise _error(
            "an extension pair containing at least one extension-only target",
            {"rho_conv": rho_conv, "rho_dense": rho_dense},
            "run.lr_provenance.candidate_stage",
        )

    expected_weights = (
        "ConvWeight_0",
        "ConvWeight_1",
        "ConvWeight_2",
        "DenseWeight_0",
    )
    units = _object(
        provenance["median_unit_by_weight"],
        "run.lr_provenance.median_unit_by_weight",
    )
    _exact_keys(units, expected_weights, "run.lr_provenance.median_unit_by_weight")
    normalized_units = {
        name: _finite(
            units[name],
            f"run.lr_provenance.median_unit_by_weight.{name}",
            positive=True,
        )
        for name in expected_weights
    }
    expected_mapping = {
        "Bias_0": "ConvWeight_0",
        "Bias_1": "ConvWeight_1",
        "Bias_2": "ConvWeight_2",
    }
    mapping = _object(
        provenance["bias_weight_mapping"],
        "run.lr_provenance.bias_weight_mapping",
    )
    if mapping != expected_mapping:
        raise _error(
            f"exactly {expected_mapping!r}",
            mapping,
            "run.lr_provenance.bias_weight_mapping",
        )
    if provenance["gain_calibration_dataset"] != "deterministic_medium_affine_mnist":
        raise _error(
            "exactly 'deterministic_medium_affine_mnist'",
            provenance["gain_calibration_dataset"],
            "run.lr_provenance.gain_calibration_dataset",
        )
    if provenance["optimization_dataset"] != "ordinary_mnist":
        raise _error(
            "exactly 'ordinary_mnist'",
            provenance["optimization_dataset"],
            "run.lr_provenance.optimization_dataset",
        )
    if provenance["official_test_read"] is not False:
        raise _error(
            "exactly false",
            provenance["official_test_read"],
            "run.lr_provenance.official_test_read",
        )
    result = {
        "study_id": study_id,
        "row_id": row_id,
        "candidate_stage": candidate_stage,
        "rho_conv": rho_conv,
        "rho_dense": rho_dense,
        "median_unit_by_weight": normalized_units,
        "bias_weight_mapping": expected_mapping,
        "gain_calibration_dataset": "deterministic_medium_affine_mnist",
        "optimization_dataset": "ordinary_mnist",
        "official_test_read": False,
    }
    for field in (
        "probe_sha256",
        "split_sha256",
        "batch_order_sha256",
        "initialization_checkpoint_sha256",
        "initialization_tensor_sha256",
    ):
        result[field] = _sha256_digest(
            provenance[field],
            f"run.lr_provenance.{field}",
        )
    return result


def _sha256_digest(value: Any, path: str, *, nullable: bool = False) -> str | None:
    if nullable and value is None:
        return None
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        expected = "null or a lowercase SHA-256 digest" if nullable else "a lowercase SHA-256 digest"
        raise _error(expected, value, path)
    return value


def _normalize_lr_provenance_v7(
    value: Any,
    *,
    depth: int,
    optimizer: Mapping[str, Any],
    learning_rates_by_parameter: Mapping[str, float],
) -> dict[str, Any]:
    """Normalize the optimizer-boundary run provenance.

    The optimizer and raw LR vector are intentionally duplicated here and in
    ``run.training``.  The exact equality checks below make every standalone
    run artifact self-describing while preventing the two records from
    drifting.
    """

    provenance = _object(value, "run.lr_provenance")
    keys = {
        "study_id",
        "row_id",
        "candidate_stage",
        "execution_mode",
        "optimizer_name",
        "optimizer_parameters",
        "rho_conv",
        "rho_dense",
        "normalization_unit_by_weight",
        "bias_q90_unit_by_parameter",
        "bias_weight_mapping",
        "bias_lr_policy",
        "bias_rho_cap",
        "raw_learning_rates_by_parameter",
        "probe_sha256",
        "bias_q90_probe_sha256",
        "parent_study_config_sha256",
        "parent_entry_completion_sha256",
        "parent_decision_sha256",
        "split_sha256",
        "normalization_minibatches_sha256",
        "replay_minibatches_sha256",
        "training_batch_order_sha256",
        "initialization_checkpoint_sha256",
        "initialization_tensor_sha256",
        "code_fingerprint_sha256",
        "official_test_read",
    }
    _exact_keys(provenance, keys, "run.lr_provenance")

    study_id = provenance["study_id"]
    if not isinstance(study_id, str) or re.fullmatch(
        r"lrstudy_[0-9a-f]{64}", study_id
    ) is None:
        raise _error(
            "lrstudy_ followed by 64 lowercase hexadecimal characters",
            study_id,
            "run.lr_provenance.study_id",
        )

    row_id = _string(provenance["row_id"], "run.lr_provenance.row_id")
    if row_id not in _V6_CONV2_ROWS:
        raise _error(
            f"one of {sorted(_V6_CONV2_ROWS)!r}",
            row_id,
            "run.lr_provenance.row_id",
        )
    scheme = {
        "conv2_baseline_v1_c1": "baseline",
        "conv2_ours_v4_c1": "ours",
        "conv2_legacy_v4_c0p25": "legacy",
    }[row_id]

    stage = provenance["candidate_stage"]
    allowed_stages = {
        "main_grid",
        "upper_sentinel",
        "bias_capped_confirmation",
    }
    if stage not in allowed_stages:
        raise _error(
            f"one of {sorted(allowed_stages)!r}",
            stage,
            "run.lr_provenance.candidate_stage",
        )
    execution_mode = provenance["execution_mode"]
    if execution_mode not in {"new", "reuse"}:
        raise _error(
            "'new' or 'reuse'",
            execution_mode,
            "run.lr_provenance.execution_mode",
        )

    optimizer_parameters = normalize_optimizer_v7(
        provenance["optimizer_parameters"],
        "run.lr_provenance.optimizer_parameters",
    )
    optimizer_name = provenance["optimizer_name"]
    if optimizer_name != optimizer_parameters["name"]:
        raise _error(
            "equal to optimizer_parameters.name",
            optimizer_name,
            "run.lr_provenance.optimizer_name",
        )
    if optimizer_parameters != dict(optimizer):
        raise _error(
            "exactly run.training.optimizer",
            optimizer_parameters,
            "run.lr_provenance.optimizer_parameters",
        )

    rho_conv = _finite(
        provenance["rho_conv"], "run.lr_provenance.rho_conv", positive=True
    )
    rho_dense = _finite(
        provenance["rho_dense"], "run.lr_provenance.rho_dense", positive=True
    )
    if stage == "upper_sentinel":
        if rho_conv != _V7_RHO_CONV_SENTINEL:
            raise _error(
                f"exactly {_V7_RHO_CONV_SENTINEL!r}",
                rho_conv,
                "run.lr_provenance.rho_conv",
            )
    elif rho_conv not in _V7_RHO_CONV_MAIN_GRID + (_V7_RHO_CONV_SENTINEL,):
        raise _error(
            "one of the frozen main-grid or sentinel values "
            f"{list(_V7_RHO_CONV_MAIN_GRID + (_V7_RHO_CONV_SENTINEL,))!r}",
            rho_conv,
            "run.lr_provenance.rho_conv",
        )
    if stage == "main_grid" and rho_conv not in _V7_RHO_CONV_MAIN_GRID:
        raise _error(
            f"one of the frozen main-grid values {list(_V7_RHO_CONV_MAIN_GRID)!r}",
            rho_conv,
            "run.lr_provenance.rho_conv",
        )
    if rho_dense not in _V7_RHO_DENSE_GRID_BY_SCHEME[scheme]:
        raise _error(
            "one of the frozen "
            f"{scheme} dense values {list(_V7_RHO_DENSE_GRID_BY_SCHEME[scheme])!r}",
            rho_dense,
            "run.lr_provenance.rho_dense",
        )

    expected_weights = tuple(
        [f"ConvWeight_{index}" for index in range(depth)] + ["DenseWeight_0"]
    )
    units = _object(
        provenance["normalization_unit_by_weight"],
        "run.lr_provenance.normalization_unit_by_weight",
    )
    _exact_keys(
        units,
        expected_weights,
        "run.lr_provenance.normalization_unit_by_weight",
    )
    normalized_units = {
        name: _finite(
            units[name],
            f"run.lr_provenance.normalization_unit_by_weight.{name}",
            positive=True,
        )
        for name in expected_weights
    }

    expected_biases = tuple(f"Bias_{index}" for index in range(depth))
    bias_units = _object(
        provenance["bias_q90_unit_by_parameter"],
        "run.lr_provenance.bias_q90_unit_by_parameter",
    )
    _exact_keys(
        bias_units,
        expected_biases,
        "run.lr_provenance.bias_q90_unit_by_parameter",
    )
    normalized_bias_units = {
        name: _finite(
            bias_units[name],
            f"run.lr_provenance.bias_q90_unit_by_parameter.{name}",
            positive=True,
        )
        for name in expected_biases
    }

    expected_mapping = {
        f"Bias_{index}": f"ConvWeight_{index}" for index in range(depth)
    }
    mapping = _object(
        provenance["bias_weight_mapping"],
        "run.lr_provenance.bias_weight_mapping",
    )
    if mapping != expected_mapping:
        raise _error(
            f"exactly {expected_mapping!r}",
            mapping,
            "run.lr_provenance.bias_weight_mapping",
        )

    bias_policy = provenance["bias_lr_policy"]
    expected_policy = "capped" if stage == "bias_capped_confirmation" else "attached"
    if bias_policy != expected_policy:
        raise _error(
            f"exactly {expected_policy!r} for {stage!r}",
            bias_policy,
            "run.lr_provenance.bias_lr_policy",
        )
    bias_cap_raw = provenance["bias_rho_cap"]
    if bias_policy == "attached":
        if bias_cap_raw is not None:
            raise _error(
                "null for attached-bias candidates",
                bias_cap_raw,
                "run.lr_provenance.bias_rho_cap",
            )
        bias_rho_cap = None
    else:
        bias_rho_cap = _finite(
            bias_cap_raw,
            "run.lr_provenance.bias_rho_cap",
            positive=True,
        )
        if bias_rho_cap != 1e-3:
            raise _error(
                "exactly 0.001",
                bias_rho_cap,
                "run.lr_provenance.bias_rho_cap",
            )

    expected_parameters = _layerwise_parameter_names(depth)
    raw_rates = _object(
        provenance["raw_learning_rates_by_parameter"],
        "run.lr_provenance.raw_learning_rates_by_parameter",
    )
    _exact_keys(
        raw_rates,
        expected_parameters,
        "run.lr_provenance.raw_learning_rates_by_parameter",
    )
    normalized_raw_rates = {
        name: _finite(
            raw_rates[name],
            f"run.lr_provenance.raw_learning_rates_by_parameter.{name}",
            positive=True,
        )
        for name in expected_parameters
    }
    if normalized_raw_rates != dict(learning_rates_by_parameter):
        raise _error(
            "exactly run.training.learning_rates_by_parameter",
            normalized_raw_rates,
            "run.lr_provenance.raw_learning_rates_by_parameter",
        )

    for weight, unit in normalized_units.items():
        expected_target = rho_conv if weight.startswith("ConvWeight_") else rho_dense
        observed_target = normalized_raw_rates[weight] * unit
        if not math.isclose(
            observed_target,
            expected_target,
            rel_tol=1e-12,
            abs_tol=1e-15,
        ):
            raise _error(
                "a rate satisfying learning_rate * optimizer-specific unit == "
                f"the direct target ({expected_target!r})",
                {
                    "learning_rate": normalized_raw_rates[weight],
                    "normalization_unit": unit,
                    "product": observed_target,
                },
                f"run.training.learning_rates_by_parameter.{weight}",
            )

    for bias, attached_weight in expected_mapping.items():
        attached_rate = normalized_raw_rates[attached_weight]
        if bias_policy == "attached":
            expected_bias_rate = attached_rate
        else:
            expected_bias_rate = min(
                attached_rate,
                1e-3 / normalized_bias_units[bias],
            )
        if not math.isclose(
            normalized_raw_rates[bias],
            expected_bias_rate,
            rel_tol=1e-12,
            abs_tol=1e-15,
        ):
            raise _error(
                "the attached weight rate"
                if bias_policy == "attached"
                else "min(attached weight rate, 0.001 / optimizer-specific "
                "Q90 bias unit)",
                normalized_raw_rates[bias],
                f"run.training.learning_rates_by_parameter.{bias}",
            )
        if bias_policy == "capped" and normalized_raw_rates[bias] > attached_rate:
            raise _error(
                "no greater than the attached weight rate",
                normalized_raw_rates[bias],
                f"run.training.learning_rates_by_parameter.{bias}",
            )

    parent_entry_completion_sha256 = _sha256_digest(
        provenance["parent_entry_completion_sha256"],
        "run.lr_provenance.parent_entry_completion_sha256",
        nullable=True,
    )
    parent_decision_sha256 = _sha256_digest(
        provenance["parent_decision_sha256"],
        "run.lr_provenance.parent_decision_sha256",
        nullable=True,
    )
    if execution_mode == "reuse":
        if (
            optimizer_name != "SGD"
            or stage != "main_grid"
            or rho_conv != 0.01
            or parent_entry_completion_sha256 is None
            or parent_decision_sha256 is not None
        ):
            raise _error(
                "an SGD main-grid rho_conv=0.01 cell with a verified parent "
                "completion hash and no decision parent",
                {
                    "optimizer_name": optimizer_name,
                    "candidate_stage": stage,
                    "rho_conv": rho_conv,
                    "parent_entry_completion_sha256": parent_entry_completion_sha256,
                    "parent_decision_sha256": parent_decision_sha256,
                },
                "run.lr_provenance.execution_mode",
            )
    elif stage == "main_grid":
        if (
            parent_entry_completion_sha256 is not None
            or parent_decision_sha256 is not None
        ):
            raise _error(
                "null completion and decision parents for a newly trained "
                "main-grid entry",
                {
                    "parent_entry_completion_sha256": (
                        parent_entry_completion_sha256
                    ),
                    "parent_decision_sha256": parent_decision_sha256,
                },
                "run.lr_provenance",
            )
    elif stage == "upper_sentinel":
        if (
            parent_entry_completion_sha256 is not None
            or parent_decision_sha256 is None
        ):
            raise _error(
                "no completion parent and the triggering selection decision "
                "SHA-256 for an upper sentinel",
                {
                    "parent_entry_completion_sha256": (
                        parent_entry_completion_sha256
                    ),
                    "parent_decision_sha256": parent_decision_sha256,
                },
                "run.lr_provenance",
            )
    elif (
        parent_entry_completion_sha256 is None
        or parent_decision_sha256 is None
    ):
        raise _error(
            "both the exact selected-run completion SHA-256 and the "
            "selection decision SHA-256 for a capped confirmation",
            {
                "parent_entry_completion_sha256": (
                    parent_entry_completion_sha256
                ),
                "parent_decision_sha256": parent_decision_sha256,
            },
            "run.lr_provenance",
        )

    result = {
        "study_id": study_id,
        "row_id": row_id,
        "candidate_stage": stage,
        "execution_mode": execution_mode,
        "optimizer_name": optimizer_name,
        "optimizer_parameters": optimizer_parameters,
        "rho_conv": rho_conv,
        "rho_dense": rho_dense,
        "normalization_unit_by_weight": normalized_units,
        "bias_q90_unit_by_parameter": normalized_bias_units,
        "bias_weight_mapping": expected_mapping,
        "bias_lr_policy": bias_policy,
        "bias_rho_cap": bias_rho_cap,
        "raw_learning_rates_by_parameter": normalized_raw_rates,
        "parent_entry_completion_sha256": parent_entry_completion_sha256,
        "parent_decision_sha256": parent_decision_sha256,
        "official_test_read": False,
    }
    if provenance["official_test_read"] is not False:
        raise _error(
            "exactly false",
            provenance["official_test_read"],
            "run.lr_provenance.official_test_read",
        )
    for field in (
        "probe_sha256",
        "bias_q90_probe_sha256",
        "parent_study_config_sha256",
        "split_sha256",
        "normalization_minibatches_sha256",
        "replay_minibatches_sha256",
        "training_batch_order_sha256",
        "initialization_checkpoint_sha256",
        "initialization_tensor_sha256",
        "code_fingerprint_sha256",
    ):
        result[field] = _sha256_digest(
            provenance[field],
            f"run.lr_provenance.{field}",
        )
    return result


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
        supported_versions = {
            RUN_SCHEMA_VERSION,
            RUN_SCHEMA_VERSION_V2,
            RUN_SCHEMA_VERSION_V3,
            RUN_SCHEMA_VERSION_V4,
            RUN_SCHEMA_VERSION_V5,
            RUN_SCHEMA_VERSION_V6,
            RUN_SCHEMA_VERSION_V7,
        }
        if top["schema_version"] not in supported_versions:
            raise _error(
                f"one of {sorted(supported_versions)!r}",
                top["schema_version"],
                "RunSpec.schema_version",
            )
        label = _string(top["label"], "RunSpec.label")
        protocol_id = _string(top["protocol_id"], "RunSpec.protocol_id")
        if top["category"] not in {"diagnostic", "final"}:
            raise _error("'diagnostic' or 'final'", top["category"], "RunSpec.category")
        run = _object(top["run"], "RunSpec.run")
        v2 = top["schema_version"] == RUN_SCHEMA_VERSION_V2
        v3 = top["schema_version"] == RUN_SCHEMA_VERSION_V3
        v4 = top["schema_version"] == RUN_SCHEMA_VERSION_V4
        v5 = top["schema_version"] == RUN_SCHEMA_VERSION_V5
        v6 = top["schema_version"] == RUN_SCHEMA_VERSION_V6
        v7 = top["schema_version"] == RUN_SCHEMA_VERSION_V7
        if (v3 or v4 or v5 or v6 or v7) and top["category"] != "diagnostic":
            raise _error(
                "exactly 'diagnostic' for layer-wise LR candidates",
                top["category"],
                "RunSpec.category",
            )
        run_keys = {"dataset", "architecture", "model", "solver", "training", "initialization", "calibration"}
        if v2 or v3 or v4 or v5 or v6 or v7:
            run_keys.add("lr_provenance")
        _exact_keys(run, run_keys, "RunSpec.run")
        replicate_id = top["replicate_id"]
        if replicate_id is not None:
            replicate_id = _string(replicate_id, "RunSpec.replicate_id")
        architecture = _normalize_architecture(run["architecture"])
        model = _normalize_model(run["model"], len(architecture["channels"]))
        dataset = (
            _normalize_dataset_conv3_v7(run["dataset"])
            if v7 and protocol_id == CONV3_ORDINARY_LR_PROTOCOL_ID
            else _normalize_dataset_v2(run["dataset"])
            if (v2 or v3 or v4 or v5 or v6 or v7)
            else _normalize_dataset(run["dataset"])
        )
        if v2 or v3 or v4 or v5 or v6 or v7:
            if model["non_linearity"] != "hard_sigmoid":
                raise _error("exactly 'hard_sigmoid'", model["non_linearity"], "run.model.non_linearity")
            if model["weight_min"] != 0.0 or model["weight_max"] != 100.0:
                raise _error("the frozen [0, 100] conductance bounds", [model["weight_min"], model["weight_max"]], "run.model weight bounds")
            if model["weight_gains"] != [1.0] * (len(architecture["channels"]) + 1):
                raise _error("all frozen weight gains equal to 1", model["weight_gains"], "run.model.weight_gains")
            if model["weight_init_mode"] != "kaiming_uniform":
                raise _error("exactly 'kaiming_uniform'", model["weight_init_mode"], "run.model.weight_init_mode")
            hard = model["hard_sigmoid_param"]
            if hard != {"g_on": 100.0, "g_off": 0.0, "v_off": 4.0}:
                raise _error("the frozen hard-sigmoid parameters", hard, "run.model.hard_sigmoid_param")
        normalized_run = {
            "dataset": dataset,
            "architecture": architecture, "model": model, "solver": _normalize_solver(run["solver"]),
            "training": (
                _normalize_training_conv3_v7(
                    run["training"], len(architecture["channels"])
                )
                if v7 and protocol_id == CONV3_ORDINARY_LR_PROTOCOL_ID
                else _normalize_training_v7(
                    run["training"], len(architecture["channels"])
                )
                if v7
                else
                _normalize_training_v4(run["training"], len(architecture["channels"]))
                if (v4 or v5 or v6)
                else _normalize_training_v3(run["training"], len(architecture["channels"]))
                if v3
                else _normalize_training_v2(run["training"], len(architecture["channels"]))
                if v2
                else _normalize_training(run["training"], len(architecture["channels"]))
            ),
            "initialization": _normalize_initialization(run["initialization"]),
            "calibration": _normalize_calibration(
                run["calibration"],
                model,
                depth=len(architecture["channels"]),
                affine_seed=dataset["affine"]["seed"],
            ),
        }
        if v2:
            normalized_run["lr_provenance"] = _normalize_lr_provenance(run["lr_provenance"])
        elif v3:
            normalized_run["lr_provenance"] = _normalize_lr_provenance_v3(
                run["lr_provenance"], len(architecture["channels"])
            )
            rho_target = normalized_run["lr_provenance"]["rho_target"]
            rho_units = normalized_run["lr_provenance"]["rho_unit_by_weight"]
            layerwise_rates = normalized_run["training"][
                "learning_rates_by_parameter"
            ]
            for weight, rho_unit in rho_units.items():
                observed_target = layerwise_rates[weight] * rho_unit
                if not math.isclose(
                    observed_target,
                    rho_target,
                    rel_tol=1e-12,
                    abs_tol=1e-15,
                ):
                    raise _error(
                        f"a rate satisfying learning_rate * rho_unit == "
                        f"rho_target ({rho_target!r})",
                        {
                            "learning_rate": layerwise_rates[weight],
                            "rho_unit": rho_unit,
                            "product": observed_target,
                        },
                        f"run.training.learning_rates_by_parameter.{weight}",
                    )
        elif v4:
            normalized_run["lr_provenance"] = _normalize_lr_provenance_v4(
                run["lr_provenance"], len(architecture["channels"])
            )
            provenance = normalized_run["lr_provenance"]
            alpha = provenance["alpha_arch"]
            units = provenance["median_unit_by_weight"]
            multipliers = provenance["target_multipliers_by_weight"]
            layerwise_rates = normalized_run["training"][
                "learning_rates_by_parameter"
            ]
            for weight, unit in units.items():
                observed_target = layerwise_rates[weight] * unit
                expected_target = alpha * multipliers[weight]
                if not math.isclose(
                    observed_target,
                    expected_target,
                    rel_tol=1e-12,
                    abs_tol=1e-15,
                ):
                    raise _error(
                        "a rate satisfying learning_rate * median_unit == "
                        f"alpha_arch * target_multiplier ({expected_target!r})",
                        {
                            "learning_rate": layerwise_rates[weight],
                            "median_unit": unit,
                            "product": observed_target,
                        },
                        f"run.training.learning_rates_by_parameter.{weight}",
                    )
        elif v5:
            if protocol_id != (
                "conv-hardsigmoid-lr-conv2-two-rho-median-constant-"
                "sgd-bs16-v6"
            ):
                raise _error(
                    "the frozen v6 two-rho protocol identifier",
                    protocol_id,
                    "RunSpec.protocol_id",
                )
            if top["seed"] != 0:
                raise _error("exactly 0", top["seed"], "RunSpec.seed")
            if architecture["profile"] != "conv2":
                raise _error(
                    "exactly 'conv2' for v6",
                    architecture["profile"],
                    "run.architecture.profile",
                )
            expected_affine = {
                "enabled": False,
                "preset": "ordinary_identity",
                "degrees": 0.0,
                "translate": [0.0, 0.0],
                "scale": [1.0, 1.0],
                "shear": 0.0,
                "seed": 1729,
                "interpolation": "bilinear",
                "fill": 0.0,
            }
            if dataset["affine"] != expected_affine:
                raise _error(
                    f"the frozen ordinary-MNIST affine settings {expected_affine!r}",
                    dataset["affine"],
                    "run.dataset.affine",
                )
            normalized_run["lr_provenance"] = _normalize_lr_provenance_v5(
                run["lr_provenance"], len(architecture["channels"])
            )
            provenance = normalized_run["lr_provenance"]
            row_contract = _V6_CONV2_ROWS[provenance["row_id"]]
            for field in ("voltage_amp", "current_amp", "input_gain"):
                if model[field] != row_contract[field]:
                    raise _error(
                        f"the frozen {provenance['row_id']} value {row_contract[field]!r}",
                        model[field],
                        f"run.model.{field}",
                    )
            for field in ("inference_iterations", "training_iterations"):
                if normalized_run["solver"][field] != row_contract[field]:
                    raise _error(
                        f"the frozen {provenance['row_id']} value {row_contract[field]!r}",
                        normalized_run["solver"][field],
                        f"run.solver.{field}",
                    )
            if provenance["split_sha256"] != dataset["validation"]["indices_sha256"]:
                raise _error(
                    "exactly run.dataset.validation.indices_sha256",
                    provenance["split_sha256"],
                    "run.lr_provenance.split_sha256",
                )
            checkpoint = normalized_run["initialization"]["checkpoint"]
            if checkpoint is None:
                raise _error(
                    "the verified shared Conv2 initialization checkpoint",
                    checkpoint,
                    "run.initialization.checkpoint",
                )
            if provenance["initialization_checkpoint_sha256"] != checkpoint["sha256"]:
                raise _error(
                    "exactly run.initialization.checkpoint.sha256",
                    provenance["initialization_checkpoint_sha256"],
                    "run.lr_provenance.initialization_checkpoint_sha256",
                )
            units = provenance["median_unit_by_weight"]
            rates = normalized_run["training"]["learning_rates_by_parameter"]
            for weight, unit in units.items():
                expected_target = (
                    provenance["rho_conv"]
                    if weight.startswith("ConvWeight_")
                    else provenance["rho_dense"]
                )
                observed_target = rates[weight] * unit
                if not math.isclose(
                    observed_target,
                    expected_target,
                    rel_tol=1e-12,
                    abs_tol=1e-15,
                ):
                    raise _error(
                        "a rate satisfying learning_rate * median_unit == "
                        f"the direct target ({expected_target!r})",
                        {
                            "learning_rate": rates[weight],
                            "median_unit": unit,
                            "product": observed_target,
                        },
                        f"run.training.learning_rates_by_parameter.{weight}",
                    )
        elif v6:
            architecture_profile = architecture["profile"]
            expected_protocol_id = (
                f"conv-hardsigmoid-lr-{architecture_profile}-amplified-scheme-"
                "two-rho-median-constant-sgd-bs16"
            )
            if protocol_id != expected_protocol_id:
                raise _error(
                    f"the frozen {architecture_profile} amplified-scheme "
                    "two-rho protocol identifier",
                    protocol_id,
                    "RunSpec.protocol_id",
                )
            if top["seed"] != 0:
                raise _error("exactly 0", top["seed"], "RunSpec.seed")
            if architecture_profile not in {"conv1", "conv2"}:
                raise _error(
                    "one of ['conv1', 'conv2'] for the amplified-scheme rho diagnostic",
                    architecture_profile,
                    "run.architecture.profile",
                )
            expected_affine = {
                "enabled": False,
                "preset": "ordinary_identity",
                "degrees": 0.0,
                "translate": [0.0, 0.0],
                "scale": [1.0, 1.0],
                "shear": 0.0,
                "seed": 1729,
                "interpolation": "bilinear",
                "fill": 0.0,
            }
            if dataset["affine"] != expected_affine:
                raise _error(
                    f"the frozen ordinary-MNIST affine settings {expected_affine!r}",
                    dataset["affine"],
                    "run.dataset.affine",
                )
            normalized_run["lr_provenance"] = _normalize_lr_provenance_v6(
                run["lr_provenance"],
                len(architecture["channels"]),
                architecture_profile,
            )
            provenance = normalized_run["lr_provenance"]
            row_contracts = (
                _CONV1_SCHEME_RHO_ROWS
                if architecture_profile == "conv1"
                else _CONV2_SCHEME_RHO_ROWS
            )
            row_contract = row_contracts[provenance["row_id"]]
            for field in ("voltage_amp", "current_amp", "input_gain"):
                if model[field] != row_contract[field]:
                    raise _error(
                        f"the frozen {provenance['row_id']} value {row_contract[field]!r}",
                        model[field],
                        f"run.model.{field}",
                    )
            for field in ("inference_iterations", "training_iterations"):
                if normalized_run["solver"][field] != row_contract[field]:
                    raise _error(
                        f"the frozen {provenance['row_id']} value {row_contract[field]!r}",
                        normalized_run["solver"][field],
                        f"run.solver.{field}",
                    )
            if provenance["split_sha256"] != dataset["validation"]["indices_sha256"]:
                raise _error(
                    "exactly run.dataset.validation.indices_sha256",
                    provenance["split_sha256"],
                    "run.lr_provenance.split_sha256",
                )
            checkpoint = normalized_run["initialization"]["checkpoint"]
            if checkpoint is None:
                raise _error(
                    f"the verified shared {architecture_profile} initialization checkpoint",
                    checkpoint,
                    "run.initialization.checkpoint",
                )
            if provenance["initialization_checkpoint_sha256"] != checkpoint["sha256"]:
                raise _error(
                    "exactly run.initialization.checkpoint.sha256",
                    provenance["initialization_checkpoint_sha256"],
                    "run.lr_provenance.initialization_checkpoint_sha256",
                )
            units = provenance["median_unit_by_weight"]
            rates = normalized_run["training"]["learning_rates_by_parameter"]
            for weight, unit in units.items():
                expected_target = (
                    provenance["rho_conv"]
                    if weight.startswith("ConvWeight_")
                    else provenance["rho_dense"]
                )
                observed_target = rates[weight] * unit
                if not math.isclose(
                    observed_target,
                    expected_target,
                    rel_tol=1e-12,
                    abs_tol=1e-15,
                ):
                    raise _error(
                        "a rate satisfying learning_rate * median_unit == "
                        f"the direct target ({expected_target!r})",
                        {
                            "learning_rate": rates[weight],
                            "median_unit": unit,
                            "product": observed_target,
                        },
                        f"run.training.learning_rates_by_parameter.{weight}",
                    )
        elif v7 and protocol_id == CONV3_ORDINARY_LR_PROTOCOL_ID:
            if top["seed"] != 0:
                raise _error("exactly 0", top["seed"], "RunSpec.seed")
            if architecture["profile"] != "conv3":
                raise _error(
                    "exactly 'conv3' for the ordinary-MNIST v7 LR study",
                    architecture["profile"],
                    "run.architecture.profile",
                )
            expected_affine = {
                "enabled": False,
                "preset": "ordinary_identity",
                "degrees": 0.0,
                "translate": [0.0, 0.0],
                "scale": [1.0, 1.0],
                "shear": 0.0,
                "seed": 1729,
                "interpolation": "bilinear",
                "fill": 0.0,
            }
            if dataset["affine"] != expected_affine:
                raise _error(
                    f"the frozen ordinary-MNIST affine settings {expected_affine!r}",
                    dataset["affine"],
                    "run.dataset.affine",
                )
            normalized_run["lr_provenance"] = (
                _normalize_lr_provenance_conv3_v7(
                    run["lr_provenance"],
                    len(architecture["channels"]),
                )
            )
            provenance = normalized_run["lr_provenance"]
            row_contract = _CONV3_ORDINARY_ROWS[provenance["row_id"]]
            for field in ("voltage_amp", "current_amp", "input_gain"):
                if model[field] != row_contract[field]:
                    raise _error(
                        f"the frozen {provenance['row_id']} value "
                        f"{row_contract[field]!r}",
                        model[field],
                        f"run.model.{field}",
                    )
            for field in ("inference_iterations", "training_iterations"):
                if normalized_run["solver"][field] != row_contract[field]:
                    raise _error(
                        f"the frozen {provenance['row_id']} value "
                        f"{row_contract[field]!r}",
                        normalized_run["solver"][field],
                        f"run.solver.{field}",
                    )
            if provenance["split_sha256"] != dataset["validation"]["indices_sha256"]:
                raise _error(
                    "exactly run.dataset.validation.indices_sha256",
                    provenance["split_sha256"],
                    "run.lr_provenance.split_sha256",
                )
            checkpoint = normalized_run["initialization"]["checkpoint"]
            if checkpoint is None:
                raise _error(
                    "the verified shared Conv3 initialization checkpoint",
                    checkpoint,
                    "run.initialization.checkpoint",
                )
            if (
                provenance["initialization_checkpoint_sha256"]
                != checkpoint["sha256"]
            ):
                raise _error(
                    "exactly run.initialization.checkpoint.sha256",
                    provenance["initialization_checkpoint_sha256"],
                    "run.lr_provenance.initialization_checkpoint_sha256",
                )
            rates = normalized_run["training"]["learning_rates_by_parameter"]
            for weight, unit in provenance["median_unit_by_weight"].items():
                expected_target = (
                    provenance["rho_conv"]
                    if weight.startswith("ConvWeight_")
                    else provenance["rho_dense"]
                )
                observed_target = rates[weight] * unit
                if not math.isclose(
                    observed_target,
                    expected_target,
                    rel_tol=1e-12,
                    abs_tol=1e-15,
                ):
                    raise _error(
                        "a rate satisfying learning_rate * median_unit == "
                        f"the direct target ({expected_target!r})",
                        {
                            "learning_rate": rates[weight],
                            "median_unit": unit,
                            "product": observed_target,
                        },
                        f"run.training.learning_rates_by_parameter.{weight}",
                    )
        elif v7:
            if protocol_id != OPTIMIZER_BOUNDARY_PROTOCOL_ID:
                raise _error(
                    f"exactly {OPTIMIZER_BOUNDARY_PROTOCOL_ID!r}",
                    protocol_id,
                    "RunSpec.protocol_id",
                )
            if top["seed"] != 0:
                raise _error("exactly 0", top["seed"], "RunSpec.seed")
            if architecture["profile"] != "conv2":
                raise _error(
                    "exactly 'conv2' for the optimizer-boundary diagnostic",
                    architecture["profile"],
                    "run.architecture.profile",
                )
            expected_affine = {
                "enabled": False,
                "preset": "ordinary_identity",
                "degrees": 0.0,
                "translate": [0.0, 0.0],
                "scale": [1.0, 1.0],
                "shear": 0.0,
                "seed": 1729,
                "interpolation": "bilinear",
                "fill": 0.0,
            }
            if dataset["affine"] != expected_affine:
                raise _error(
                    f"the frozen ordinary-MNIST affine settings {expected_affine!r}",
                    dataset["affine"],
                    "run.dataset.affine",
                )
            normalized_run["lr_provenance"] = _normalize_lr_provenance_v7(
                run["lr_provenance"],
                depth=len(architecture["channels"]),
                optimizer=normalized_run["training"]["optimizer"],
                learning_rates_by_parameter=normalized_run["training"][
                    "learning_rates_by_parameter"
                ],
            )
            provenance = normalized_run["lr_provenance"]
            row_contract = _V6_CONV2_ROWS[provenance["row_id"]]
            for field in ("voltage_amp", "current_amp", "input_gain"):
                if model[field] != row_contract[field]:
                    raise _error(
                        f"the frozen {provenance['row_id']} value "
                        f"{row_contract[field]!r}",
                        model[field],
                        f"run.model.{field}",
                    )
            for field in ("inference_iterations", "training_iterations"):
                if normalized_run["solver"][field] != row_contract[field]:
                    raise _error(
                        f"the frozen {provenance['row_id']} value "
                        f"{row_contract[field]!r}",
                        normalized_run["solver"][field],
                        f"run.solver.{field}",
                    )
            if provenance["split_sha256"] != dataset["validation"]["indices_sha256"]:
                raise _error(
                    "exactly run.dataset.validation.indices_sha256",
                    provenance["split_sha256"],
                    "run.lr_provenance.split_sha256",
                )
            checkpoint = normalized_run["initialization"]["checkpoint"]
            if checkpoint is None:
                raise _error(
                    "the verified shared Conv2 initialization checkpoint",
                    checkpoint,
                    "run.initialization.checkpoint",
                )
            if (
                provenance["initialization_checkpoint_sha256"]
                != checkpoint["sha256"]
            ):
                raise _error(
                    "exactly run.initialization.checkpoint.sha256",
                    provenance["initialization_checkpoint_sha256"],
                    "run.lr_provenance.initialization_checkpoint_sha256",
                )
        if top["category"] == "final" and normalized_run["training"]["pruning"]["enabled"]:
            raise _error("disabled for final runs", normalized_run["training"]["pruning"], "run.training.pruning")
        return cls({
            "schema_version": top["schema_version"],
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
