#!/usr/bin/env python3
"""Replay selected Conv hard-sigmoid checkpoints under finite bounds.

This is a read-only diagnostic for the deterministic medium-affine Conv1/Conv2
learning-rate handoff or the selected ordinary-MNIST v5 diagnostic. Each
requested ``(gmin, gmax)`` case starts from the same pristine checkpoint
tensors and clamps only raw ``ConvWeight`` and ``DenseWeight`` states. Biases
are restored exactly and never clipped.

Accuracy deterioration is measured against an unclipped replay of the same
checkpoint on the same deterministic official-test cohort.  The LR-screen
checkpoints themselves did not read the official test split; this separate
diagnostic does and must not be used to select a learning rate or checkpoint.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import platform
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
for _path in (REPO_ROOT, REPO_ROOT / "labs", REPO_ROOT / "experiments"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from experiments.evaluate_mnist_bp_write_noise_sweep import (  # noqa: E402
    _mean_margin,
    _scores,
)
from experiments.mnist_conv.identity import sha256_file  # noqa: E402
from experiments.mnist_conv.io import (  # noqa: E402
    atomic_write_csv,
    atomic_write_json,
)
from experiments.mnist_conv.specs import RunSpec, SpecValidationError  # noqa: E402
from labs.custom_classes import FlexibleDeepResistiveEnergy  # noqa: E402
from labs.mnist_train import (  # noqa: E402
    _build_tracking_minimizer,
    _reset_name_counters,
    _resolve_callable,
    _resolve_dataset_config,
    _set_seed,
)
from model.function.cost import SquaredErrorPairedOutputs  # noqa: E402
from model.function.network import Network  # noqa: E402
from model.variable.layer import layer_index  # noqa: E402
from model.variable.parameter import Bias, ConvWeight, DenseWeight  # noqa: E402


SCHEMA_VERSION = "conv-hardsigmoid-finite-conductance/v2"
MEDIUM_AFFINE_PROTOCOL_IDS = {
    "conv-hardsigmoid-lr-sgd-bs16-v1",
    "conv-hardsigmoid-lr-sgd-bs16-relative-rho-v3",
}
ORDINARY_MNIST_PROTOCOL_IDS = {
    "conv-hardsigmoid-lr-architecture-relative-median-constant-sgd-bs16-v5",
}
SUPPORTED_PROTOCOL_IDS = MEDIUM_AFFINE_PROTOCOL_IDS | ORDINARY_MNIST_PROTOCOL_IDS
SUPPORTED_RUN_SPEC_FILES = ("run_spec.v2.json", "run_spec.v4.json")
SCHEME_AMPLIFICATION = {
    "baseline": (1.0, 1.0),
    "legacy": (4.0, 0.25),
    "ours": (4.0, 1.0),
}
ARCHITECTURE_CONTRACT = {
    "conv1": {
        "channels": [64],
        "kernel_sizes": [3],
        "strides": [2],
        "paddings": [1],
        "output_dim": 20,
    },
    "conv2": {
        "channels": [64, 128],
        "kernel_sizes": [3, 3],
        "strides": [2, 2],
        "paddings": [1, 1],
        "output_dim": 20,
    },
    "conv3": {
        "channels": [64, 128, 256],
        "kernel_sizes": [3, 3, 3],
        "strides": [2, 2, 1],
        "paddings": [1, 1, 1],
        "output_dim": 20,
    },
}
FROZEN_TK = {
    ("conv1", "baseline"): (4, 4),
    ("conv1", "legacy"): (4, 4),
    ("conv1", "ours"): (4, 4),
    ("conv2", "baseline"): (16, 6),
    ("conv2", "legacy"): (8, 4),
    ("conv2", "ours"): (24, 6),
    ("conv3", "baseline"): (24, 8),
    ("conv3", "legacy"): (8, 6),
    ("conv3", "ours"): (32, 8),
}
FROZEN_INPUT_GAINS = {
    ("conv1", "baseline"): 75.6030807495,
    ("conv1", "legacy"): 31.8188591003,
    ("conv1", "ours"): 84.8402175903,
    ("conv2", "baseline"): 253.302230835,
    ("conv2", "legacy"): 661.4369506836,
    ("conv2", "ours"): 716.3439331055,
    ("conv3", "baseline"): 251.306137085,
    ("conv3", "legacy"): 665.0302124023,
    ("conv3", "ours"): 744.739074707,
}
CHECKPOINT_FILES = {
    "best_validation": "best_validation.pt",
    "final": "final.pt",
}
SUMMARY_COLUMNS = [
    "architecture",
    "scheme",
    "entry_id",
    "checkpoint_kind",
    "checkpoint_sha256",
    "training_iterations",
    "inference_iterations",
    "gmin",
    "gmax",
    "is_reference",
    "num_examples",
    "accuracy",
    "accuracy_drop",
    "loss",
    "loss_increase",
    "mean_margin",
    "margin_drop",
    "disagreement_fraction",
    "paired_score_rmse",
    "equilibrium_residual_mean",
    "equilibrium_residual_median",
    "equilibrium_residual_p90",
    "equilibrium_residual_p99",
    "equilibrium_residual_max",
    "saturation_mean",
    "saturation_p90",
    "saturation_max",
    "weight_scalar_count",
    "lower_clipped_fraction",
    "upper_clipped_fraction",
    "clipped_fraction",
    "mean_abs_conductance_delta",
    "rms_conductance_delta",
    "relative_l2_conductance_distortion",
]
WEIGHT_COLUMNS = [
    "architecture",
    "scheme",
    "entry_id",
    "checkpoint_kind",
    "checkpoint_sha256",
    "training_iterations",
    "inference_iterations",
    "gmin",
    "gmax",
    "parameter",
    "parameter_type",
    "shape",
    "element_count",
    "original_min",
    "original_mean",
    "original_max",
    "bounded_min",
    "bounded_mean",
    "bounded_max",
    "lower_clipped_fraction",
    "upper_clipped_fraction",
    "clipped_fraction",
    "mean_abs_delta",
    "rms_delta",
    "max_abs_delta",
    "relative_l2_distortion",
]
LAYER_COLUMNS = [
    "architecture",
    "scheme",
    "entry_id",
    "checkpoint_kind",
    "checkpoint_sha256",
    "training_iterations",
    "inference_iterations",
    "gmin",
    "gmax",
    "layer",
    "layer_index",
    "layer_role",
    "residual_mode",
    "num_examples",
    "residual_mean",
    "residual_median",
    "residual_p90",
    "residual_p99",
    "residual_max",
    "saturation_mean",
    "saturation_p90",
    "saturation_max",
]


class EvaluationInputError(ValueError):
    """Raised when an input violates the frozen diagnostic contract."""


@dataclass(frozen=True)
class EntryInput:
    scheme: str
    entry_dir: Path
    entry_id: str
    spec: RunSpec
    completion: dict[str, Any]
    checkpoint_kind: str
    checkpoint_path: Path
    checkpoint_sha256: str


@dataclass
class EvaluationResult:
    summary: dict[str, Any]
    layer_rows: list[dict[str, Any]]
    labels: np.ndarray
    predictions: np.ndarray
    scores: np.ndarray


def _error(expected: str, provided: Any, field: str) -> EvaluationInputError:
    return EvaluationInputError(
        f"Expected {field} to be {expected}. Provided value: {provided!r}."
    )


def _bound_text(value: float | None) -> str:
    return "none" if value is None else format(float(value), ".12g")


def parse_optional_bound(value: str) -> float | None:
    text = str(value).strip().lower()
    if text in {"none", "null", "unbounded"}:
        return None
    try:
        result = float(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "Expected a non-negative finite number or 'none'. "
            f"Provided value: {value!r}."
        ) from exc
    if not math.isfinite(result) or result < 0.0:
        raise argparse.ArgumentTypeError(
            "Expected a non-negative finite number or 'none'. "
            f"Provided value: {value!r}."
        )
    return result


def parse_entry_argument(raw: str) -> tuple[str, Path]:
    if "=" not in raw:
        raise _error(
            "'baseline=PATH', 'legacy=PATH', or 'ours=PATH'",
            raw,
            "--entry",
        )
    scheme, path_text = raw.split("=", 1)
    scheme = scheme.strip().lower()
    path_text = path_text.strip()
    if scheme not in SCHEME_AMPLIFICATION or not path_text:
        raise _error(
            "'baseline=PATH', 'legacy=PATH', or 'ours=PATH'",
            raw,
            "--entry",
        )
    return scheme, Path(path_text).expanduser().resolve()


def build_bound_grid(
    gmins: Iterable[float | None],
    gmaxs: Iterable[float | None],
) -> list[tuple[float | None, float | None]]:
    """Build a stable grid with the unclipped reference first."""

    lower = list(dict.fromkeys([None, *gmins]))
    upper = list(dict.fromkeys([None, *gmaxs]))
    grid: list[tuple[float | None, float | None]] = []
    for gmin in lower:
        for gmax in upper:
            if gmin is not None and gmax is not None and gmin > gmax:
                continue
            grid.append((gmin, gmax))
    return grid


def _completion_output(
    completion: dict[str, Any],
    *,
    filename: str,
) -> dict[str, Any]:
    matches = [
        record
        for record in completion.get("outputs", ())
        if Path(str(record.get("path", ""))).name == filename
    ]
    if len(matches) != 1:
        raise _error(
            f"exactly one completion output named {filename!r}",
            matches,
            "complete.json.outputs",
        )
    return dict(matches[0])


def _source_checkpoint_dataset(spec: RunSpec) -> str:
    protocol_id = str(spec.data.get("protocol_id", ""))
    if protocol_id in MEDIUM_AFFINE_PROTOCOL_IDS:
        return "deterministic medium affine MNIST"
    if protocol_id in ORDINARY_MNIST_PROTOCOL_IDS:
        return "ordinary MNIST"
    raise _error(
        f"one of {sorted(SUPPORTED_PROTOCOL_IDS)!r}",
        protocol_id,
        "protocol_id",
    )


def _run_spec_filename(completion: dict[str, Any]) -> str:
    matches = [
        filename
        for filename in SUPPORTED_RUN_SPEC_FILES
        if any(
            Path(str(record.get("path", ""))).name == filename
            for record in completion.get("outputs", ())
        )
    ]
    if len(matches) != 1:
        raise _error(
            f"exactly one of {SUPPORTED_RUN_SPEC_FILES!r}",
            matches,
            "complete.json run-spec outputs",
        )
    return matches[0]


def _conv_spatial(size: int, kernel: int, stride: int, padding: int) -> int:
    return (size + 2 * padding - (kernel - 1) - 1) // stride + 1


def _build_replay_engine_config(
    spec: RunSpec,
    *,
    dataset_root: str | Path,
    device: str,
    download: bool,
    ordinary_mnist: bool = False,
) -> dict[str, Any]:
    """Build only the model/dataset fields needed for read-only v2 replay."""

    value = spec.data
    run = value["run"]
    dataset = run["dataset"]
    architecture = run["architecture"]
    model = run["model"]
    solver = run["solver"]

    height = width = 28
    layer_shapes: list[list[int]] = [[2, height, width]]
    conv_pipeline: list[dict[str, Any]] = []
    for channels, kernel, stride, padding in zip(
        architecture["channels"],
        architecture["kernel_sizes"],
        architecture["strides"],
        architecture["paddings"],
    ):
        height = _conv_spatial(height, kernel, stride, padding)
        width = _conv_spatial(width, kernel, stride, padding)
        if height <= 0 or width <= 0:
            raise _error(
                "positive convolutional spatial dimensions",
                (height, width),
                "run.architecture",
            )
        layer_shapes.append([channels, height, width])
        conv_pipeline.append(
            {
                "kernel": [kernel, kernel],
                "stride": stride,
                "padding": padding,
                "mode": "convolution",
            }
        )
    layer_shapes.append([architecture["output_dim"]])

    dataset_params: dict[str, Any] = {
        "name": "mnist",
        "batch_size": dataset["batch_size"],
        "root": str(Path(dataset_root).expanduser()),
        "train": True,
        "download": bool(download),
        "normalize": True,
        "normalize_mean": dataset["normalization"]["mean"],
        "normalize_std": dataset["normalization"]["std"],
        "normalize_scale": dataset["normalization"]["scale"],
    }
    if dataset["affine"]["enabled"] and not ordinary_mnist:
        dataset_params["affine_config"] = dict(dataset["affine"])
        dataset_factory = "labs.datasets.AffineMnistDataset"
    else:
        dataset_factory = "labs.datasets.MnistDataset"
    model_key = "mnist_resistive_conv_v1"
    return {
        "lab": {
            "model_key": model_key,
            "dataset_key": "mnist",
        },
        "input_mode": "train",
        "batch_state_policy": run["training"]["batch_state_policy"],
        "datasets": {
            "mnist": {
                "factory": dataset_factory,
                "params": dataset_params,
            }
        },
        "model_base": {
            "weight_min": model["weight_min"],
            "weight_max": model["weight_max"],
            "weight_init_mode": model["weight_init_mode"],
            "input_gain": model["input_gain"],
            "voltage_amp": model["voltage_amp"],
            "current_amp": model["current_amp"],
            "trainable_amplification": model["trainable_parameters"]["amplification"],
            "amplification_min": model["amplification_min"],
            "amplification_max": model["amplification_max"],
            "non_linearity": model["non_linearity"],
            "quadratic_diode_param": dict(model["quadratic_diode_param"]),
            "exponential_diode_param": dict(model["exponential_diode_param"]),
            "hard_sigmoid_param": dict(model["hard_sigmoid_param"]),
            "num_iterations_inference": solver["inference_iterations"],
            "num_iterations_training": solver["training_iterations"],
            "minimizer": json.loads(json.dumps(solver["minimizer"])),
        },
        "model_overrides": {
            model_key: {
                "layer_shapes": layer_shapes,
                "conv_pipeline": conv_pipeline,
                "weight_gains": list(model["weight_gains"]),
            }
        },
        "energy_minimizer": {"mode": solver["energy_mode"]},
        "device": device,
    }


def _verify_completion_file(
    entry_dir: Path,
    completion: dict[str, Any],
    *,
    filename: str,
) -> tuple[Path, str]:
    record = _completion_output(completion, filename=filename)
    path = entry_dir / filename
    if not path.is_file():
        raise _error("an existing regular file", str(path), filename)
    expected_bytes = int(record["bytes"])
    observed_bytes = path.stat().st_size
    if observed_bytes != expected_bytes:
        raise _error(str(expected_bytes), observed_bytes, f"{filename} byte count")
    expected_sha = str(record["sha256"])
    observed_sha = sha256_file(path)
    if observed_sha != expected_sha:
        raise _error(expected_sha, observed_sha, f"{filename} SHA-256")
    return path, expected_sha


def _validate_protocol(
    scheme: str,
    spec: RunSpec,
    engine: dict[str, Any],
) -> None:
    data = spec.data
    run = data["run"]
    dataset = run["dataset"]
    architecture = run["architecture"]
    model_cfg = run["model"]
    solver = run["solver"]
    training = run["training"]
    calibration = run["calibration"]
    architecture_profile = str(architecture["profile"])
    if architecture_profile not in ARCHITECTURE_CONTRACT:
        raise _error(
            f"one of {sorted(ARCHITECTURE_CONTRACT)!r}",
            architecture_profile,
            "run.architecture.profile",
        )
    architecture_contract = ARCHITECTURE_CONTRACT[architecture_profile]
    expected_t, expected_k = FROZEN_TK[(architecture_profile, scheme)]
    expected_gain = FROZEN_INPUT_GAINS[(architecture_profile, scheme)]
    expected_voltage, expected_current = SCHEME_AMPLIFICATION[scheme]
    protocol_id = str(data.get("protocol_id", ""))
    source_dataset = _source_checkpoint_dataset(spec)
    if source_dataset == "deterministic medium affine MNIST":
        expected_affine = {
            "enabled": True,
            "preset": "medium",
            "degrees": 25.0,
            "translate": [0.2, 0.2],
            "scale": [0.8, 1.2],
            "shear": 0.0,
            "seed": 1729,
            "interpolation": "bilinear",
            "fill": 0.0,
        }
    else:
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
    checks = [
        ("category", data["category"], "diagnostic"),
        ("seed", data["seed"], 0),
        ("run.dataset.name", dataset["name"], "mnist"),
        ("run.dataset.input_shape", dataset["input_shape"], [2, 28, 28]),
        ("run.dataset.batch_size", dataset["batch_size"], 16),
        ("run.dataset.max_batches", dataset["max_batches"], None),
        ("run.dataset.affine", dataset["affine"], expected_affine),
        (
            "run.architecture.channels",
            architecture["channels"],
            architecture_contract["channels"],
        ),
        (
            "run.architecture.kernel_sizes",
            architecture["kernel_sizes"],
            architecture_contract["kernel_sizes"],
        ),
        (
            "run.architecture.strides",
            architecture["strides"],
            architecture_contract["strides"],
        ),
        (
            "run.architecture.paddings",
            architecture["paddings"],
            architecture_contract["paddings"],
        ),
        (
            "run.architecture.output_dim",
            architecture["output_dim"],
            architecture_contract["output_dim"],
        ),
        ("run.model.non_linearity", model_cfg["non_linearity"], "hard_sigmoid"),
        ("run.model.weight_min", model_cfg["weight_min"], 0.0),
        ("run.model.weight_max", model_cfg["weight_max"], 100.0),
        ("run.model.hard_sigmoid_param.g_on", model_cfg["hard_sigmoid_param"]["g_on"], 100.0),
        ("run.model.hard_sigmoid_param.g_off", model_cfg["hard_sigmoid_param"]["g_off"], 0.0),
        ("run.model.hard_sigmoid_param.v_off", model_cfg["hard_sigmoid_param"]["v_off"], 4.0),
        (
            "run.solver.inference_iterations",
            solver["inference_iterations"],
            expected_t,
        ),
        (
            "run.solver.training_iterations",
            solver["training_iterations"],
            expected_k,
        ),
        (
            "run.solver.minimizer.adaptive_equilibrium",
            solver["minimizer"]["adaptive_equilibrium"],
            False,
        ),
        ("run.calibration.adaptive_equilibrium", calibration["adaptive_equilibrium"], False),
        ("run.calibration.settling_iterations", calibration["settling_iterations"], 64),
        ("run.calibration.target_initial_saturation", calibration["target_initial_saturation"], 0.3),
        ("run.calibration.v_off", calibration["v_off"], 4.0),
        ("run.training.epochs", training["epochs"], 5),
        ("run.training.optimizer.name", training["optimizer"]["name"], "SGD"),
        ("run.training.optimizer.momentum", training["optimizer"]["momentum"], 0.0),
        ("run.training.optimizer.weight_decay", training["optimizer"]["weight_decay"], 0.0),
        ("run.training.batch_state_policy", training["batch_state_policy"], "reset_each_batch"),
        ("run.training.checkpoint_rule", training["checkpoint_rule"], "min_validation_loss"),
        ("engine.batch_state_policy", engine["batch_state_policy"], "reset_each_batch"),
        (
            "engine.model_base.num_iterations_inference",
            engine["model_base"]["num_iterations_inference"],
            expected_t,
        ),
        (
            "engine.model_base.minimizer.adaptive_equilibrium",
            engine["model_base"]["minimizer"]["adaptive_equilibrium"],
            False,
        ),
    ]
    for field, provided, expected in checks:
        if provided != expected:
            raise _error(f"exactly {expected!r}", provided, field)
    if protocol_id not in SUPPORTED_PROTOCOL_IDS:
        raise _error(
            f"one of {sorted(SUPPORTED_PROTOCOL_IDS)!r}",
            protocol_id,
            "protocol_id",
        )
    if dataset["validation"]["official_test_enabled"] is not False:
        raise _error("false", dataset["validation"]["official_test_enabled"], "validation official-test flag")
    if not math.isclose(
        float(model_cfg["input_gain"]),
        expected_gain,
        rel_tol=0.0,
        abs_tol=1e-6,
    ):
        raise _error(
            str(expected_gain),
            model_cfg["input_gain"],
            f"{architecture_profile}.{scheme}.input_gain",
        )
    if not math.isclose(
        float(model_cfg["voltage_amp"]),
        expected_voltage,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise _error(
            str(expected_voltage),
            model_cfg["voltage_amp"],
            f"{scheme}.voltage_amp",
        )
    if not math.isclose(
        float(model_cfg["current_amp"]),
        expected_current,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise _error(
            str(expected_current),
            model_cfg["current_amp"],
            f"{scheme}.current_amp",
        )


def load_entry_input(
    scheme: str,
    entry_dir: Path,
    *,
    checkpoint_kind: str,
) -> EntryInput:
    completion_path = entry_dir / "complete.json"
    if not completion_path.is_file():
        raise _error("an existing completion record", str(completion_path), "entry")
    completion = json.loads(completion_path.read_text())
    if completion.get("state") != "complete":
        raise _error("'complete'", completion.get("state"), "complete.json.state")
    entry_id = str(completion.get("entry_id", "")).strip()
    if not entry_id:
        raise _error("a non-empty string", entry_id, "complete.json.entry_id")
    spec_path, _ = _verify_completion_file(
        entry_dir,
        completion,
        filename=_run_spec_filename(completion),
    )
    checkpoint_name = CHECKPOINT_FILES[checkpoint_kind]
    checkpoint_path, checkpoint_sha = _verify_completion_file(
        entry_dir,
        completion,
        filename=checkpoint_name,
    )
    spec = RunSpec.from_path(spec_path)
    engine = _build_replay_engine_config(
        spec,
        dataset_root=".",
        device="cpu",
        download=False,
        ordinary_mnist=False,
    )
    _validate_protocol(scheme, spec, engine)
    return EntryInput(
        scheme=scheme,
        entry_dir=entry_dir,
        entry_id=entry_id,
        spec=spec,
        completion=completion,
        checkpoint_kind=checkpoint_kind,
        checkpoint_path=checkpoint_path,
        checkpoint_sha256=checkpoint_sha,
    )


def validate_matched_inputs(records: Sequence[EntryInput]) -> None:
    if {record.scheme for record in records} != set(SCHEME_AMPLIFICATION):
        raise _error(
            "exactly one entry for each of baseline, legacy, and ours",
            [record.scheme for record in records],
            "--entry",
        )
    entry_ids = [record.entry_id for record in records]
    if len(entry_ids) != len(set(entry_ids)):
        raise _error("three distinct entry IDs", entry_ids, "--entry")
    reference = records[0].spec.data
    for record in records[1:]:
        candidate = record.spec.data
        if candidate["seed"] != reference["seed"]:
            raise _error(
                f"the shared seed {reference['seed']!r}",
                candidate["seed"],
                f"{record.scheme}.seed",
            )
        for field in ("dataset", "architecture"):
            expected = reference["run"][field]
            if candidate["run"][field] != expected:
                raise _error(
                    "the same cross-scheme configuration",
                    candidate["run"][field],
                    f"{record.scheme}.run.{field}",
                )
        expected_model = {
            key: value
            for key, value in reference["run"]["model"].items()
            if key not in {"voltage_amp", "current_amp", "input_gain"}
        }
        candidate_model = {
            key: value
            for key, value in candidate["run"]["model"].items()
            if key not in {"voltage_amp", "current_amp", "input_gain"}
        }
        if candidate_model != expected_model:
            raise _error(
                "the same model except voltage_amp, current_amp, and input_gain",
                candidate_model,
                f"{record.scheme}.run.model",
            )


def _build_context(
    record: EntryInput,
    *,
    dataset_root: Path,
    device: torch.device,
    download: bool,
    eval_batch_size: int,
    ordinary_mnist: bool,
) -> dict[str, Any]:
    engine = _build_replay_engine_config(
        record.spec,
        dataset_root=dataset_root,
        device=str(device),
        download=download,
        ordinary_mnist=ordinary_mnist,
    )
    _validate_protocol(record.scheme, record.spec, engine)
    _reset_name_counters()
    _set_seed(record.spec.data["seed"])

    model_key = engine["lab"]["model_key"]
    model_cfg = {
        **engine["model_base"],
        **engine["model_overrides"][model_key],
    }
    energy_fn = FlexibleDeepResistiveEnergy(
        layer_shapes=[tuple(shape) for shape in model_cfg["layer_shapes"]],
        conv_pipeline=model_cfg["conv_pipeline"],
        pooling_mode=model_cfg.get("pooling_mode"),
        weight_gains=model_cfg["weight_gains"],
        input_gain=model_cfg["input_gain"],
        non_linearity=model_cfg["non_linearity"],
        exponential_diode_param=model_cfg["exponential_diode_param"],
        quadratic_diode_param=model_cfg["quadratic_diode_param"],
        hard_sigmoid_param=model_cfg["hard_sigmoid_param"],
        voltage_amp=model_cfg["voltage_amp"],
        current_amp=model_cfg["current_amp"],
        weight_min=model_cfg["weight_min"],
        weight_max=model_cfg["weight_max"],
        weight_init_mode=model_cfg["weight_init_mode"],
        input_mode=engine["input_mode"],
        trainable_amplification=model_cfg["trainable_amplification"],
        amplification_min=model_cfg["amplification_min"],
        amplification_max=model_cfg["amplification_max"],
    )
    energy_fn.set_device(device)
    energy_fn.load(record.checkpoint_path)

    network = Network(energy_fn)
    free_layers = network.free_layers()
    output_layer = energy_fn.layers()[-1]
    if tuple(output_layer.shape) != (20,):
        raise _error("exactly a paired 20-output layer", output_layer.shape, "output layer")
    cost_fn = SquaredErrorPairedOutputs(output_layer, num_classes=10)
    minimizer = _build_tracking_minimizer(
        energy_fn,
        free_layers,
        model_cfg,
        engine["energy_minimizer"]["mode"],
        num_iterations=int(record.spec.data["run"]["solver"]["inference_iterations"]),
        voltage_amp=energy_fn._voltage_amp,
        current_amp=energy_fn._current_amp,
        adaptive_equilibrium=False,
    )

    _, dataset_cfg = _resolve_dataset_config(engine, engine["lab"]["dataset_key"])
    dataset_factory = _resolve_callable(dataset_cfg["factory"])
    dataset_params = dict(dataset_cfg["params"])
    dataset_params["batch_size"] = eval_batch_size
    dataset_params["device"] = device
    loaders = dataset_factory(**dataset_params).build()
    if not isinstance(loaders, tuple) or len(loaders) != 2:
        raise _error(
            "a (train_loader, test_loader) tuple",
            type(loaders).__name__,
            "dataset factory result",
        )
    params = list(energy_fn.params())
    unexpected = [
        type(param).__name__
        for param in params
        if not isinstance(param, (ConvWeight, DenseWeight, Bias))
    ]
    if unexpected:
        raise _error(
            "only ConvWeight, DenseWeight, and Bias parameters",
            unexpected,
            "Conv1 checkpoint schema",
        )
    for required in (ConvWeight, DenseWeight, Bias):
        if not any(isinstance(param, required) for param in params):
            raise _error(
                f"at least one {required.__name__}",
                [type(param).__name__ for param in params],
                "parameters",
            )
    pristine = [param.state.detach().clone() for param in params]
    for param, state in zip(params, pristine):
        if not bool(torch.isfinite(state).all().item()):
            raise _error(
                "a finite tensor",
                getattr(param, "name", type(param).__name__),
                "checkpoint",
            )
        if isinstance(param, (ConvWeight, DenseWeight)) and float(state.min().item()) < -1e-8:
            raise _error(
                "non-negative raw conductances",
                float(state.min().item()),
                getattr(param, "name", type(param).__name__),
            )
    return {
        "engine": engine,
        "energy_fn": energy_fn,
        "network": network,
        "free_layers": free_layers,
        "output_layer": output_layer,
        "cost_fn": cost_fn,
        "minimizer": minimizer,
        "test_loader": loaders[1],
        "params": params,
        "pristine": pristine,
    }


def materialize_test_cohort(
    test_loader: Any,
    *,
    max_samples: int | None,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    batches: list[tuple[torch.Tensor, torch.Tensor]] = []
    total = 0
    for images, labels in test_loader:
        if max_samples is not None and total >= max_samples:
            break
        take = int(images.shape[0])
        if max_samples is not None:
            take = min(take, max_samples - total)
        if take <= 0:
            break
        batches.append(
            (
                images[:take].detach().cpu().clone(),
                labels[:take].detach().cpu().clone(),
            )
        )
        total += take
    if total == 0:
        raise _error("at least one test example", total, "evaluation cohort")
    return batches


def restore_pristine(
    params: Sequence[Any],
    pristine: Sequence[torch.Tensor],
) -> None:
    if len(params) != len(pristine):
        raise _error("matching parameter/state lengths", len(pristine), "pristine states")
    for param, state in zip(params, pristine):
        param.state = state.detach().clone()


def _tensor_distortion(
    original: torch.Tensor,
    bounded: torch.Tensor,
    *,
    gmin: float | None,
    gmax: float | None,
) -> dict[str, Any]:
    original64 = original.detach().to(dtype=torch.float64)
    bounded64 = bounded.detach().to(dtype=torch.float64)
    delta = bounded64 - original64
    changed = delta.ne(0)
    lower = (
        torch.zeros_like(changed)
        if gmin is None
        else original64.lt(float(gmin))
    )
    upper = (
        torch.zeros_like(changed)
        if gmax is None
        else original64.gt(float(gmax))
    )
    denominator = max(float(torch.linalg.vector_norm(original64).item()), 1e-30)
    return {
        "element_count": int(original.numel()),
        "original_min": float(original64.min().item()),
        "original_mean": float(original64.mean().item()),
        "original_max": float(original64.max().item()),
        "bounded_min": float(bounded64.min().item()),
        "bounded_mean": float(bounded64.mean().item()),
        "bounded_max": float(bounded64.max().item()),
        "lower_clipped_count": int(lower.sum().item()),
        "upper_clipped_count": int(upper.sum().item()),
        "clipped_count": int(changed.sum().item()),
        "lower_clipped_fraction": float(lower.to(torch.float64).mean().item()),
        "upper_clipped_fraction": float(upper.to(torch.float64).mean().item()),
        "clipped_fraction": float(changed.to(torch.float64).mean().item()),
        "mean_abs_delta": float(delta.abs().mean().item()),
        "rms_delta": float(torch.sqrt(torch.mean(delta * delta)).item()),
        "max_abs_delta": float(delta.abs().max().item()),
        "relative_l2_distortion": float(torch.linalg.vector_norm(delta).item())
        / denominator,
        "_delta_sq_sum": float(torch.sum(delta * delta).item()),
        "_delta_abs_sum": float(torch.sum(delta.abs()).item()),
        "_original_sq_sum": float(torch.sum(original64 * original64).item()),
    }


def apply_conductance_bounds(
    params: Sequence[Any],
    pristine: Sequence[torch.Tensor],
    *,
    gmin: float | None,
    gmax: float | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Restore all states, then clip only raw Conv/Dense conductances."""

    if gmin is not None and gmax is not None and gmin > gmax:
        raise _error("less than or equal to gmax", gmin, "gmin")
    restore_pristine(params, pristine)
    rows: list[dict[str, Any]] = []
    for param, original in zip(params, pristine):
        if not isinstance(param, (ConvWeight, DenseWeight)):
            if not torch.equal(param.state, original):
                raise RuntimeError("Bias restoration changed an immutable pristine state.")
            continue
        bounded = original.detach().clone()
        if gmin is not None:
            bounded.clamp_(min=float(gmin))
        if gmax is not None:
            bounded.clamp_(max=float(gmax))
        param.state = bounded
        stats = _tensor_distortion(original, bounded, gmin=gmin, gmax=gmax)
        rows.append(
            {
                "parameter": str(getattr(param, "name", type(param).__name__)).strip(),
                "parameter_type": type(param).__name__,
                "shape": "x".join(str(dim) for dim in original.shape),
                **stats,
            }
        )
    if not rows:
        raise _error("at least one ConvWeight or DenseWeight", rows, "parameters")
    for param, original in zip(params, pristine):
        if isinstance(param, Bias) and not torch.equal(param.state, original):
            raise RuntimeError("Bias was modified while imposing conductance bounds.")

    count = sum(int(row["element_count"]) for row in rows)
    delta_sq = sum(float(row["_delta_sq_sum"]) for row in rows)
    delta_abs = sum(float(row["_delta_abs_sum"]) for row in rows)
    original_sq = sum(float(row["_original_sq_sum"]) for row in rows)
    aggregate = {
        "weight_scalar_count": count,
        "lower_clipped_fraction": sum(
            int(row["lower_clipped_count"]) for row in rows
        )
        / count,
        "upper_clipped_fraction": sum(
            int(row["upper_clipped_count"]) for row in rows
        )
        / count,
        "clipped_fraction": sum(int(row["clipped_count"]) for row in rows) / count,
        "mean_abs_conductance_delta": delta_abs / count,
        "rms_conductance_delta": math.sqrt(delta_sq / count),
        "relative_l2_conductance_distortion": math.sqrt(delta_sq)
        / max(math.sqrt(original_sq), 1e-30),
    }
    for row in rows:
        for private_key in [key for key in row if key.startswith("_")]:
            row.pop(private_key)
        row.pop("lower_clipped_count")
        row.pop("upper_clipped_count")
        row.pop("clipped_count")
    return rows, aggregate


def _stats(values: Sequence[float] | np.ndarray) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0 or not np.isfinite(array).all():
        raise _error("a non-empty finite array", array.tolist(), "summary values")
    return {
        "mean": float(np.mean(array)),
        "median": float(np.percentile(array, 50)),
        "p90": float(np.percentile(array, 90)),
        "p99": float(np.percentile(array, 99)),
        "max": float(np.max(array)),
    }


def _layer_number(layer: Any, fallback: int) -> int:
    try:
        return int(layer_index(layer))
    except ValueError:
        return fallback


def _saturation_per_example(layer: Any, *, v_off: float) -> torch.Tensor:
    state = layer.state.detach()
    saturated = (state < -float(v_off)) | (state > float(v_off))
    return saturated.reshape(saturated.shape[0], -1).to(torch.float64).mean(dim=1)


def evaluate_context(
    context: dict[str, Any],
    cohort: Sequence[tuple[torch.Tensor, torch.Tensor]],
    *,
    v_off: float,
) -> EvaluationResult:
    network = context["network"]
    minimizer = context["minimizer"]
    energy_fn = context["energy_fn"]
    free_layers = context["free_layers"]
    cost_fn = context["cost_fn"]
    output_layer = context["output_layer"]
    device = context["pristine"][0].device
    fn = getattr(minimizer, "_fn", energy_fn)

    labels_parts: list[np.ndarray] = []
    prediction_parts: list[np.ndarray] = []
    score_parts: list[np.ndarray] = []
    loss_values: list[float] = []
    margin_values: list[float] = []
    residual_by_layer: list[list[float]] = [[] for _ in free_layers]
    saturation_by_layer: list[list[float]] = [[] for _ in free_layers[:-1]]
    overall_residual: list[float] = []

    for images_cpu, labels_cpu in cohort:
        images = images_cpu.to(device)
        labels = labels_cpu.to(device)
        network.set_input(images, reset=True)
        minimizer.compute_equilibrium()
        cost_fn.set_target(labels)

        scores = _scores(output_layer.state.detach(), 10)
        predictions = torch.argmax(scores, dim=1)
        losses = cost_fn.eval().detach()
        margins = _mean_margin(scores, labels).detach()
        gradients = [fn.grad_layer_fn(layer)().detach() for layer in free_layers]
        residual_norms = [
            gradient.abs().reshape(gradient.shape[0], -1).amax(dim=1)
            for gradient in gradients
        ]
        overall = torch.stack(residual_norms).amax(dim=0)

        labels_parts.append(labels.detach().cpu().numpy().astype(np.int64))
        prediction_parts.append(predictions.detach().cpu().numpy().astype(np.int64))
        score_parts.append(scores.detach().cpu().numpy().astype(np.float64))
        loss_values.extend(losses.cpu().numpy().astype(np.float64).tolist())
        margin_values.extend(margins.cpu().numpy().astype(np.float64).tolist())
        overall_residual.extend(overall.cpu().numpy().astype(np.float64).tolist())
        for index, residual_norm in enumerate(residual_norms):
            residual_by_layer[index].extend(
                residual_norm.cpu().numpy().astype(np.float64).tolist()
            )
            if index < len(free_layers) - 1:
                saturation = _saturation_per_example(
                    free_layers[index],
                    v_off=v_off,
                )
                saturation_by_layer[index].extend(
                    saturation.cpu().numpy().astype(np.float64).tolist()
                )

    labels_array = np.concatenate(labels_parts)
    predictions_array = np.concatenate(prediction_parts)
    scores_array = np.concatenate(score_parts)
    if not (len(labels_array) == len(loss_values) == len(margin_values)):
        raise RuntimeError("Evaluation vector lengths diverged.")
    overall_stats = _stats(overall_residual)
    saturation_values = [
        value for layer_values in saturation_by_layer for value in layer_values
    ]
    saturation_stats = _stats(saturation_values)
    layer_rows: list[dict[str, Any]] = []
    for index, (layer, residual_values) in enumerate(
        zip(free_layers, residual_by_layer)
    ):
        residual_stats = _stats(residual_values)
        is_output = index == len(free_layers) - 1
        layer_saturation = (
            {"mean": None, "p90": None, "max": None}
            if is_output
            else _stats(saturation_by_layer[index])
        )
        layer_rows.append(
            {
                "layer": str(getattr(layer, "name", f"layer_{index}")).strip(),
                "layer_index": _layer_number(layer, index + 1),
                "layer_role": "output" if is_output else f"hidden_{index + 1}",
                "residual_mode": "raw",
                "num_examples": int(len(labels_array)),
                **{f"residual_{key}": value for key, value in residual_stats.items()},
                "saturation_mean": layer_saturation["mean"],
                "saturation_p90": layer_saturation["p90"],
                "saturation_max": layer_saturation["max"],
            }
        )
    summary = {
        "num_examples": int(len(labels_array)),
        "accuracy": float(np.mean(predictions_array == labels_array)),
        "loss": float(np.mean(np.asarray(loss_values, dtype=np.float64))),
        "mean_margin": float(np.mean(np.asarray(margin_values, dtype=np.float64))),
        **{f"equilibrium_residual_{key}": value for key, value in overall_stats.items()},
        "saturation_mean": saturation_stats["mean"],
        "saturation_p90": saturation_stats["p90"],
        "saturation_max": saturation_stats["max"],
    }
    return EvaluationResult(
        summary=summary,
        layer_rows=layer_rows,
        labels=labels_array,
        predictions=predictions_array,
        scores=scores_array,
    )


def _case_prefix(
    record: EntryInput,
    gmin: float | None,
    gmax: float | None,
) -> dict[str, Any]:
    run = record.spec.data["run"]
    return {
        "architecture": run["architecture"]["profile"],
        "scheme": record.scheme,
        "entry_id": record.entry_id,
        "checkpoint_kind": record.checkpoint_kind,
        "checkpoint_sha256": record.checkpoint_sha256,
        "training_iterations": run["solver"]["training_iterations"],
        "inference_iterations": run["solver"]["inference_iterations"],
        "gmin": _bound_text(gmin),
        "gmax": _bound_text(gmax),
    }


def evaluate_entry(
    record: EntryInput,
    *,
    grid: Sequence[tuple[float | None, float | None]],
    dataset_root: Path,
    device: torch.device,
    download: bool,
    eval_batch_size: int,
    max_samples: int | None,
    v_off: float,
    ordinary_mnist: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    context = _build_context(
        record,
        dataset_root=dataset_root,
        device=device,
        download=download,
        eval_batch_size=eval_batch_size,
        ordinary_mnist=ordinary_mnist,
    )
    cohort = materialize_test_cohort(
        context["test_loader"],
        max_samples=max_samples,
    )
    summaries: list[dict[str, Any]] = []
    weights: list[dict[str, Any]] = []
    layers: list[dict[str, Any]] = []
    reference: EvaluationResult | None = None
    initial_sha = sha256_file(record.checkpoint_path)
    if initial_sha != record.checkpoint_sha256:
        raise _error(
            record.checkpoint_sha256,
            initial_sha,
            f"{record.checkpoint_kind} checkpoint SHA-256 before replay",
        )
    try:
        for gmin, gmax in grid:
            weight_rows, aggregate = apply_conductance_bounds(
                context["params"],
                context["pristine"],
                gmin=gmin,
                gmax=gmax,
            )
            result = evaluate_context(context, cohort, v_off=v_off)
            for param, original in zip(context["params"], context["pristine"]):
                if isinstance(param, Bias) and not torch.equal(param.state, original):
                    raise RuntimeError("Inference modified a Bias state.")
            if reference is None:
                if (gmin, gmax) != (None, None):
                    raise RuntimeError(
                        "The first conductance-grid case must be the unclipped reference."
                    )
                reference = result
            if not np.array_equal(result.labels, reference.labels):
                raise RuntimeError(
                    "The deterministic test cohort labels changed between cases."
                )
            prefix = _case_prefix(record, gmin, gmax)
            summary = {
                **prefix,
                "is_reference": (gmin, gmax) == (None, None),
                **result.summary,
                "accuracy_drop": reference.summary["accuracy"]
                - result.summary["accuracy"],
                "loss_increase": result.summary["loss"]
                - reference.summary["loss"],
                "margin_drop": reference.summary["mean_margin"]
                - result.summary["mean_margin"],
                "disagreement_fraction": float(
                    np.mean(result.predictions != reference.predictions)
                ),
                "paired_score_rmse": float(
                    np.sqrt(np.mean((result.scores - reference.scores) ** 2))
                ),
                **aggregate,
            }
            summaries.append(summary)
            weights.extend({**prefix, **row} for row in weight_rows)
            layers.extend({**prefix, **row} for row in result.layer_rows)
    finally:
        restore_pristine(context["params"], context["pristine"])
        final_sha = sha256_file(record.checkpoint_path)
        if final_sha != initial_sha:
            raise RuntimeError(
                "Checkpoint SHA-256 changed during read-only replay: "
                f"before={initial_sha!r}, after={final_sha!r}."
            )
    return summaries, weights, layers


def _plot_results(
    output_dir: Path,
    summaries: Sequence[dict[str, Any]],
) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    written: list[str] = []
    colors = {"baseline": "#4C78A8", "legacy": "#F58518", "ours": "#54A24B"}
    for bound_key, fixed_key, filename, xlabel in (
        (
            "gmin",
            "gmax",
            "combined_min_floor_deterioration.png",
            "Minimum conductance",
        ),
        (
            "gmax",
            "gmin",
            "combined_max_ceiling_deterioration.png",
            "Maximum conductance",
        ),
    ):
        fig, axes = plt.subplots(2, 1, figsize=(7.2, 7.0), sharex=True)
        all_x: list[float] = []
        for scheme in SCHEME_AMPLIFICATION:
            rows = [
                row
                for row in summaries
                if row["scheme"] == scheme
                and row[fixed_key] == "none"
                and row[bound_key] != "none"
            ]
            rows.sort(key=lambda row: float(row[bound_key]))
            if not rows:
                continue
            x = [float(row[bound_key]) for row in rows]
            all_x.extend(x)
            axes[0].plot(
                x,
                [100.0 * float(row["accuracy_drop"]) for row in rows],
                marker="o",
                label=scheme,
                color=colors[scheme],
            )
            axes[1].plot(
                x,
                [100.0 * float(row["clipped_fraction"]) for row in rows],
                marker="o",
                label=scheme,
                color=colors[scheme],
            )
        if not all_x:
            plt.close(fig)
            continue
        positive = [value for value in all_x if value > 0.0]
        if len(positive) == len(all_x):
            axes[1].set_xscale("log")
        elif positive:
            axes[1].set_xscale("symlog", linthresh=min(positive) / 10.0)
        axes[0].axhline(1.0, color="black", linestyle="--", linewidth=1.0)
        axes[0].set_ylabel("Accuracy drop (pp)")
        axes[1].set_ylabel("Conductances clipped (%)")
        axes[1].set_xlabel(xlabel)
        architectures = {str(row["architecture"]) for row in summaries}
        architecture = (
            next(iter(architectures)) if len(architectures) == 1 else "mixed Conv"
        )
        axes[0].set_title(
            f"{architecture.capitalize()} hard sigmoid "
            "(diagnostic, row-specific T/K): scheme comparison"
        )
        for axis in axes:
            axis.grid(True, alpha=0.3)
            handles, labels = axis.get_legend_handles_labels()
            if handles:
                axis.legend(handles, labels)
        fig.tight_layout()
        path = output_dir / filename
        fig.savefig(path, dpi=180)
        plt.close(fig)
        written.append(path.name)

    for scheme in SCHEME_AMPLIFICATION:
        rows = [row for row in summaries if row["scheme"] == scheme]
        gmins = list(dict.fromkeys(row["gmin"] for row in rows))
        gmaxs = list(dict.fromkeys(row["gmax"] for row in rows))
        if len(gmins) <= 1 or len(gmaxs) <= 1:
            continue
        matrix = np.full((len(gmins), len(gmaxs)), np.nan, dtype=np.float64)
        for row in rows:
            matrix[gmins.index(row["gmin"]), gmaxs.index(row["gmax"])] = (
                100.0 * float(row["accuracy_drop"])
            )
        fig, ax = plt.subplots(
            figsize=(max(5.0, 0.9 * len(gmaxs)), max(4.2, 0.7 * len(gmins)))
        )
        image = ax.imshow(matrix, aspect="auto", cmap="magma")
        ax.set_xticks(range(len(gmaxs)), labels=gmaxs, rotation=45, ha="right")
        ax.set_yticks(range(len(gmins)), labels=gmins)
        ax.set_xlabel("Maximum conductance")
        ax.set_ylabel("Minimum conductance")
        ax.set_title(f"{scheme}: accuracy drop (percentage points)")
        fig.colorbar(image, ax=ax, label="Accuracy drop (pp)")
        fig.tight_layout()
        path = output_dir / f"{scheme}_accuracy_drop_heatmap.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        written.append(path.name)
    return written


def _runtime_provenance(device: torch.device) -> dict[str, Any]:
    try:
        torchvision_version = importlib.metadata.version("torchvision")
    except importlib.metadata.PackageNotFoundError:
        torchvision_version = None
    device_name = (
        torch.cuda.get_device_name(device)
        if device.type == "cuda"
        else (platform.processor() or platform.machine())
    )
    return {
        "evaluator_path": str(Path(__file__).resolve()),
        "evaluator_sha256": sha256_file(Path(__file__).resolve()),
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "torch_version": str(torch.__version__),
        "torchvision_version": torchvision_version,
        "cuda_runtime_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "device": str(device),
        "device_name": device_name,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.max_samples is not None and args.max_samples < 1:
        raise _error("null or an integer >= 1", args.max_samples, "--max-samples")
    if isinstance(args.eval_batch_size, bool) or args.eval_batch_size < 1:
        raise _error("an integer >= 1", args.eval_batch_size, "--eval-batch-size")
    if args.v_off <= 0.0 or not math.isfinite(args.v_off):
        raise _error("a positive finite number", args.v_off, "--v-off")
    device = torch.device(
        args.device
        if args.device is not None
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    if device.type == "cuda" and not torch.cuda.is_available():
        raise _error("an available device", args.device, "--device")

    parsed = [parse_entry_argument(raw) for raw in args.entry]
    schemes = [scheme for scheme, _ in parsed]
    if len(schemes) != len(set(schemes)):
        raise _error("each scheme exactly once", schemes, "--entry")
    records = [
        load_entry_input(scheme, entry, checkpoint_kind=args.checkpoint)
        for scheme, entry in parsed
    ]
    records.sort(key=lambda record: list(SCHEME_AMPLIFICATION).index(record.scheme))
    validate_matched_inputs(records)
    architecture = str(records[0].spec.data["run"]["architecture"]["profile"])
    output_dir = Path(args.output_dir).expanduser().resolve()
    for record in records:
        try:
            output_dir.relative_to(record.entry_dir)
        except ValueError:
            continue
        raise _error(
            "outside every immutable LR candidate entry",
            str(output_dir),
            "--output-dir",
        )
    grid = build_bound_grid(args.gmin, args.gmax)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict[str, Any]] = []
    weight_rows: list[dict[str, Any]] = []
    layer_rows: list[dict[str, Any]] = []
    for record in records:
        print(
            f"[finite-conductance] scheme={record.scheme} "
            f"entry={record.entry_id} checkpoint={record.checkpoint_kind} "
            f"cases={len(grid)}",
            flush=True,
        )
        summaries, weights, layers = evaluate_entry(
            record,
            grid=grid,
            dataset_root=Path(args.dataset_root).expanduser(),
            device=device,
            download=bool(args.download),
            eval_batch_size=int(args.eval_batch_size),
            max_samples=args.max_samples,
            v_off=float(args.v_off),
            ordinary_mnist=bool(args.ordinary_mnist),
        )
        summary_rows.extend(summaries)
        weight_rows.extend(weights)
        layer_rows.extend(layers)

    atomic_write_csv(
        output_dir / "finite_conductance_results.csv",
        SUMMARY_COLUMNS,
        summary_rows,
    )
    atomic_write_csv(
        output_dir / "finite_conductance_weight_metrics.csv",
        WEIGHT_COLUMNS,
        weight_rows,
    )
    atomic_write_csv(
        output_dir / "finite_conductance_residual_layers.csv",
        LAYER_COLUMNS,
        layer_rows,
    )
    plot_files = _plot_results(output_dir, summary_rows)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "scientific_status": (
            "diagnostic only; official test read for finite-conductance replay; "
            "not an LR/checkpoint selection and not paper-facing evidence"
        ),
        "checkpoint_kind": args.checkpoint,
        "checkpoint_policy": (
            "completion-record SHA-256 and byte count verified; checkpoint "
            "SHA-256 reverified after read-only replay"
        ),
        "conductance_coordinate": "raw ConvWeight/DenseWeight parameter state",
        "bias_policy": "Bias states restored exactly and never clipped",
        "case_policy": "restore pristine checkpoint states before every (gmin,gmax)",
        "architecture": architecture,
        "training_iterations_by_scheme": {
            record.scheme: record.spec.data["run"]["solver"]["training_iterations"]
            for record in records
        },
        "inference_iterations_by_scheme": {
            record.scheme: record.spec.data["run"]["solver"]["inference_iterations"]
            for record in records
        },
        "adaptive_equilibrium": False,
        "batch_state_policy": "reset_each_batch",
        "evaluation_batch_size": int(args.eval_batch_size),
        "paired_score_definition": (
            "output[...,0] - output[...,1] for each class pair"
        ),
        "equilibrium_residual_definition": (
            "per-example maximum raw |dE/dz| over every free hidden/output layer"
        ),
        "saturation_definition": (
            f"hard-sigmoid hidden states outside [-{float(args.v_off):g},"
            f"{float(args.v_off):g}]"
        ),
        "official_test_read": True,
        "source_checkpoint_dataset": _source_checkpoint_dataset(records[0].spec),
        "evaluation_dataset": (
            "ordinary MNIST"
            if args.ordinary_mnist
            or _source_checkpoint_dataset(records[0].spec) == "ordinary MNIST"
            else "deterministic medium affine MNIST"
        ),
        "evaluation_affine_transform_enabled": (
            not bool(args.ordinary_mnist)
            and bool(records[0].spec.data["run"]["dataset"]["affine"]["enabled"])
        ),
        "final_paper_training_authorized": False,
        "max_samples": args.max_samples,
        "dataset_root": str(Path(args.dataset_root).expanduser()),
        "device": str(device),
        "v_off": float(args.v_off),
        "evaluator_provenance": _runtime_provenance(device),
        "grid": [
            {"gmin": _bound_text(gmin), "gmax": _bound_text(gmax)}
            for gmin, gmax in grid
        ],
        "entries": [
            {
                "scheme": record.scheme,
                "entry_id": record.entry_id,
                "entry_dir": str(record.entry_dir),
                "checkpoint": str(record.checkpoint_path),
                "checkpoint_sha256": record.checkpoint_sha256,
                "checkpoint_sha256_verified_after_replay": True,
                "voltage_amp": record.spec.data["run"]["model"]["voltage_amp"],
                "current_amp": record.spec.data["run"]["model"]["current_amp"],
                "input_gain": record.spec.data["run"]["model"]["input_gain"],
                "protocol_id": record.spec.data["protocol_id"],
            }
            for record in records
        ],
        "results": summary_rows,
        "weight_metrics": weight_rows,
        "residual_layers": layer_rows,
        "plots": plot_files,
    }
    atomic_write_json(output_dir / "finite_conductance_results.json", payload)
    return payload


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--entry",
        action="append",
        required=True,
        help=(
            "Completed supported candidate entry as SCHEME=PATH. Repeat exactly once "
            "for baseline, legacy, and ours."
        ),
    )
    parser.add_argument(
        "--checkpoint",
        choices=tuple(CHECKPOINT_FILES),
        default="best_validation",
    )
    parser.add_argument(
        "--gmin",
        nargs="+",
        required=True,
        type=parse_optional_bound,
        help="Minimum raw conductance values; use 'none' for no floor.",
    )
    parser.add_argument(
        "--gmax",
        nargs="+",
        required=True,
        type=parse_optional_bound,
        help="Maximum raw conductance values; use 'none' for no ceiling.",
    )
    parser.add_argument("--dataset-root", default="~/datasets/mnist")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Evaluate a deterministic test prefix; default is all 10,000.",
    )
    parser.add_argument("--v-off", type=float, default=4.0)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument(
        "--ordinary-mnist",
        action="store_true",
        help=(
            "Disable the source medium-affine transform during evaluation. "
            "The saved checkpoint and every other model setting remain unchanged."
        ),
    )
    parser.add_argument("--download", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        payload = run(args)
    except (EvaluationInputError, SpecValidationError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(
        f"[finite-conductance] wrote {len(payload['results'])} cases to "
        f"{Path(args.output_dir).expanduser().resolve()}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
