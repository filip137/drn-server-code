import argparse
from datetime import datetime
import json
import math
import os
import random
import socket
import sys
from importlib import import_module
import importlib.util
from pathlib import Path

import numpy as np
import torch

try:  # Optional TensorBoard support
    from torch.utils.tensorboard import SummaryWriter
except Exception:
    SummaryWriter = None


# Ensure project modules are importable
LABS_DIR = Path(__file__).resolve().parent
REPO_ROOT = LABS_DIR.parent
PROJECT_ROOT = LABS_DIR.parent / "energy-based-learning"
DEFAULT_IMAGE_RUNS_BASE = LABS_DIR.parent / "simulation_results" / "experiments_labs"

for path in (LABS_DIR, REPO_ROOT, PROJECT_ROOT):
    if str(path) not in sys.path:
        sys.path.append(str(path))


from custom_classes import (
    ConvLayer,
    ConvResistive,
    DetailedSumSeparableFunction,
    FlexibleDeepResistiveEnergy,
    TrackingQuadraticMinimizer,
)  # noqa: E402
from custom_minimizer import CustomQuadraticMinimizer as QuadraticMinimizer, MinimizerSettings  # noqa: E402
from model.function.cost import SquaredError, SquaredErrorPairedOutputs  # noqa: E402
from model.function.interaction import load_function_checkpoint_states, scalar_float  # noqa: E402
from model.function.network import Network  # noqa: E402
from model.variable.layer import Layer  # noqa: E402
from model.variable.parameter import ConvWeight  # noqa: E402
from model.variable.parameter import Bias, DenseWeight, HardSigmoidVOff, PoolWeight  # noqa: E402
from training.monitor import Optimizer  # noqa: E402
from training.sgd import AugmentedFunction, Backprop, EquilibriumProp  # noqa: E402


def _set_seed(seed):
    if seed is None:
        return
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _reset_name_counters():
    # Keep parameter names stable when a sweep builds multiple models in one process.
    Layer._counter = 0
    Bias._counter = 0
    DenseWeight._counter = 0
    ConvWeight._counter = 0
    PoolWeight._counter = 0
    HardSigmoidVOff._counter = 0


def _sanitize_npz_key(name, fallback):
    cleaned = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in str(name).strip())
    cleaned = cleaned.strip("_")
    return cleaned or fallback


def _param_schema(params):
    schema = []
    used = set()
    for idx, param in enumerate(params):
        base = _sanitize_npz_key(getattr(param, "name", None), f"param_{idx}")
        key = base
        suffix = 1
        while key in used:
            key = f"{base}_{suffix}"
            suffix += 1
        used.add(key)
        schema.append((key, param))
    return schema


def _checkpoint_to_npz(checkpoint_path, npz_path, param_schema, metadata):
    params = [param for _, param in param_schema]
    tensors, checkpoint_source_format = load_function_checkpoint_states(
        checkpoint_path,
        params,
        map_location="cpu",
    )

    arrays = {}
    names = []
    types = []
    shapes = []
    for (name, param), tensor in zip(param_schema, tensors):
        arrays[name] = tensor.detach().cpu().numpy()
        names.append(name)
        types.append(param.__class__.__name__)
        shapes.append(tuple(int(dim) for dim in tensor.shape))

    arrays["param_names"] = np.asarray(names)
    arrays["param_types"] = np.asarray(types)
    arrays["param_shapes_json"] = np.asarray(json.dumps(shapes, allow_nan=False))
    metadata = {**metadata, "checkpoint_source_format": checkpoint_source_format}
    arrays["metadata_json"] = np.asarray(json.dumps(metadata, sort_keys=True, allow_nan=False))
    target = Path(npz_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez(target, **arrays)
    return target


def _save_history_arrays(run_dir, history):
    paths = {
        "loss_train_path": run_dir / "loss_train.npy",
        "loss_test_path": run_dir / "loss_test.npy",
        "accuracy_train_path": run_dir / "accuracy_train.npy",
        "accuracy_test_path": run_dir / "accuracy_test.npy",
    }
    np.save(paths["loss_train_path"], np.asarray(history["loss"], dtype=np.float64))
    np.save(paths["loss_test_path"], np.asarray(history["test_loss"], dtype=np.float64))
    np.save(paths["accuracy_train_path"], np.asarray(history["accuracy"], dtype=np.float64))
    np.save(paths["accuracy_test_path"], np.asarray(history["test_accuracy"], dtype=np.float64))
    return paths


def _hard_sigmoid_v_off_values(energy_fn):
    values = {}
    for param in energy_fn.params():
        if isinstance(param, HardSigmoidVOff):
            values[param.name] = float(param.state.detach().cpu().reshape(-1)[0].item())
    return values


def _amplification_values(energy_fn):
    return {
        "voltage_amp": scalar_float(getattr(energy_fn, "_voltage_amp")),
        "current_amp": scalar_float(getattr(energy_fn, "_current_amp")),
    }


def _should_log_batch(batch_idx, total_batches, log_interval):
    batch_num = batch_idx + 1
    interval = int(log_interval or 0)
    return batch_num == 1 or batch_num == total_batches or (
        interval > 0 and batch_num % interval == 0
    )


def _normalize_training_algorithm(training_algorithm):
    value = "EP" if training_algorithm is None else str(training_algorithm).upper()
    if value not in ("EP", "BP"):
        raise ValueError(f"training_algorithm must be 'EP' or 'BP'. Got {training_algorithm!r}.")
    return value


def _json_sanitize(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.device):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_sanitize(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_sanitize(item) for item in value]
    return value


class NonFiniteTrainingError(FloatingPointError):
    """Raised as soon as a non-finite training value is observed."""


BATCH_STATE_POLICIES = ("reset_each_batch", "carry_within_epoch")


def _resolve_batch_state_policy(config, explicit_policy=None):
    training_cfg = config.get("training", {})
    configured_policy = training_cfg.get("batch_state_policy") if isinstance(training_cfg, dict) else None
    policy = explicit_policy
    if policy is None:
        policy = config.get("batch_state_policy")
    if policy is None:
        policy = configured_policy
    if policy is None:
        policy = "reset_each_batch"
    policy = str(policy).strip().lower()
    if policy not in BATCH_STATE_POLICIES:
        raise ValueError(
            f"Expected batch_state_policy to be one of {BATCH_STATE_POLICIES}. "
            f"Provided value: {policy!r}."
        )
    return policy


def _batch_reset_flag(batch_state_policy, batch_index):
    return batch_state_policy == "reset_each_batch" or int(batch_index) == 0


def _resolve_optimizer_settings(config, *, optimizer_name=None, momentum=None, weight_decay=None):
    """Resolve explicit optimizer science and reject argument/config drift."""

    optimizer_cfg = config.get("optimizer")
    if optimizer_cfg is None:
        optimizer_cfg = {}
    if not isinstance(optimizer_cfg, dict):
        raise ValueError(
            "Expected config['optimizer'] to be an object with explicit name, momentum, and "
            f"weight_decay fields. Provided value: {optimizer_cfg!r}."
        )

    configured_name = optimizer_cfg.get("name")
    if optimizer_name is None:
        optimizer_name = configured_name
    elif configured_name is not None and optimizer_name != configured_name:
        raise ValueError(
            "Expected optimizer_name to equal config['optimizer']['name']. "
            f"Provided values: optimizer_name={optimizer_name!r}, configured={configured_name!r}."
        )
    if optimizer_name not in {"SGD", "Adam"}:
        raise ValueError(
            "Expected optimizer name to be one of ('SGD', 'Adam'). "
            f"Provided value: {optimizer_name!r}."
        )

    resolved = {}
    for field, explicit in (("momentum", momentum), ("weight_decay", weight_decay)):
        configured = optimizer_cfg.get(field)
        if explicit is None:
            explicit = configured
        elif configured is not None:
            configured_value = _require_finite_scalar(configured, f"configured optimizer {field}")
            explicit_value = _require_finite_scalar(explicit, f"optimizer {field}")
            if explicit_value != configured_value:
                raise ValueError(
                    f"Expected {field} to equal config['optimizer'][{field!r}]. "
                    f"Provided values: argument={explicit_value!r}, configured={configured_value!r}."
                )
        if explicit is None:
            raise ValueError(
                f"Expected optimizer {field} to be supplied explicitly or in config['optimizer']. "
                f"Provided value: {explicit!r}."
            )
        resolved[field] = _require_finite_scalar(explicit, f"optimizer {field}")

    if not 0.0 <= resolved["momentum"] < 1.0:
        raise ValueError(
            "Expected optimizer momentum to be in [0, 1). "
            f"Provided value: {resolved['momentum']!r}."
        )
    if resolved["weight_decay"] < 0.0:
        raise ValueError(
            "Expected optimizer weight_decay to be non-negative. "
            f"Provided value: {resolved['weight_decay']!r}."
        )
    if optimizer_name == "Adam" and resolved["momentum"] != 0.0:
        raise ValueError(
            "Expected optimizer momentum to be 0.0 when name is 'Adam'. "
            f"Provided value: {resolved['momentum']!r}."
        )
    return optimizer_name, resolved["momentum"], resolved["weight_decay"]


def _optimizer_details(config, name, momentum, weight_decay):
    details = {
        "name": name,
        "momentum": momentum,
        "weight_decay": weight_decay,
    }
    if name == "Adam":
        optimizer_cfg = config.get("optimizer", {})
        raw_betas = optimizer_cfg.get("betas", [0.9, 0.999])
        if not isinstance(raw_betas, (list, tuple)) or len(raw_betas) != 2:
            raise ValueError(
                "Expected config['optimizer']['betas'] to contain two values. "
                f"Provided value: {raw_betas!r}."
            )
        betas = tuple(
            _require_finite_scalar(value, f"optimizer beta {index}")
            for index, value in enumerate(raw_betas)
        )
        if not 0.0 <= betas[0] < 1.0 or not 0.0 <= betas[1] < 1.0:
            raise ValueError(
                "Expected optimizer betas to be in [0, 1). "
                f"Provided value: {betas!r}."
            )
        eps = _require_finite_scalar(optimizer_cfg.get("eps", 1e-8), "optimizer eps")
        if eps <= 0.0:
            raise ValueError(f"Expected optimizer eps to be positive. Provided value: {eps!r}.")
        details.update(betas=list(betas), eps=eps)
    return details


def _build_parameter_optimizer(energy_fn, cost_fn, learning_rates, details):
    if details["name"] == "SGD":
        return Optimizer(
            energy_fn,
            cost_fn,
            learning_rates,
            momentum=details["momentum"],
            weight_decay=details["weight_decay"],
        )

    params = [
        param
        for param in energy_fn.params() + cost_fn.params()
        if not isinstance(param, PoolWeight)
    ]
    if len(learning_rates) != len(params):
        raise ValueError(
            f"learning_rates length ({len(learning_rates)}) does not match "
            f"parameter count ({len(params)} after filtering PoolWeight)"
        )
    groups = [
        {"params": [param.state], "lr": rate}
        for param, rate in zip(params, learning_rates)
    ]
    return torch.optim.Adam(
        groups,
        lr=1.0,
        betas=tuple(details["betas"]),
        eps=details["eps"],
        weight_decay=details["weight_decay"],
        amsgrad=False,
        foreach=False,
        maximize=False,
        capturable=False,
        differentiable=False,
        fused=False,
    )


def _validate_optimizer_rate_consistency(config, learning_rates, lr_decay):
    """Ensure canonical arguments and generated optimizer config agree exactly."""

    optimizer_cfg = config.get("optimizer", {})
    configured_rates = optimizer_cfg.get("learning_rate") if isinstance(optimizer_cfg, dict) else None
    if configured_rates is not None:
        if not isinstance(configured_rates, (list, tuple)):
            raise ValueError(
                "Expected config['optimizer']['learning_rate'] to be an explicit vector. "
                f"Provided value: {configured_rates!r}."
            )
        normalized = [
            _require_finite_scalar(value, f"configured learning rate {index}")
            for index, value in enumerate(configured_rates)
        ]
        if normalized != list(learning_rates):
            raise ValueError(
                "Expected supplied learning rates to equal config['optimizer']['learning_rate']. "
                f"Provided values: supplied={list(learning_rates)!r}, configured={normalized!r}."
            )
    configured_decay = optimizer_cfg.get("lr_decay") if isinstance(optimizer_cfg, dict) else None
    if configured_decay is not None:
        configured_decay = _require_finite_scalar(configured_decay, "configured optimizer lr_decay")
        if configured_decay != lr_decay:
            raise ValueError(
                "Expected supplied lr_decay to equal config['optimizer']['lr_decay']. "
                f"Provided values: supplied={lr_decay!r}, configured={configured_decay!r}."
            )


def _require_finite_tensor(value, label, *, epoch=None, batch=None):
    location = []
    if epoch is not None:
        location.append(f"epoch={epoch}")
    if batch is not None:
        location.append(f"batch={batch}")
    where = f" at {', '.join(location)}" if location else ""
    if not torch.is_tensor(value):
        raise TypeError(
            f"Expected {label} to be a torch.Tensor{where}. "
            f"Provided value: {type(value).__name__}."
        )
    if not bool(torch.isfinite(value).all().item()):
        nonfinite = int((~torch.isfinite(value)).sum().item())
        raise NonFiniteTrainingError(
            f"Expected {label} to contain only finite values{where}. "
            f"Provided value: tensor with {nonfinite} non-finite element(s)."
        )


def _require_finite_optimizer_state(optimizer, *, epoch=None, batch=None):
    """Fail immediately when a tensor or scalar optimizer state is non-finite."""

    def visit(value, label):
        if torch.is_tensor(value):
            _require_finite_tensor(value, label, epoch=epoch, batch=batch)
        elif isinstance(value, dict):
            for index, (key, item) in enumerate(value.items()):
                key_label = repr(key) if isinstance(key, (str, int)) else str(index)
                visit(item, f"{label}[{key_label}]")
        elif isinstance(value, (list, tuple)):
            for index, item in enumerate(value):
                visit(item, f"{label}[{index}]")
        elif isinstance(value, (float, np.floating)) and not math.isfinite(
            float(value)
        ):
            location = []
            if epoch is not None:
                location.append(f"epoch={epoch}")
            if batch is not None:
                location.append(f"batch={batch}")
            where = f" at {', '.join(location)}" if location else ""
            raise NonFiniteTrainingError(
                f"Expected {label} to be finite{where}. Provided value: {value!r}."
            )

    visit(optimizer.state, "optimizer state")


def _require_finite_scalar(value, label, *, epoch=None, batch=None):
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Expected {label} to be numeric. Provided value: {value!r}."
        ) from exc
    if not math.isfinite(numeric):
        location = []
        if epoch is not None:
            location.append(f"epoch={epoch}")
        if batch is not None:
            location.append(f"batch={batch}")
        where = f" at {', '.join(location)}" if location else ""
        raise NonFiniteTrainingError(
            f"Expected {label} to be finite{where}. Provided value: {value!r}."
        )
    return numeric


def _require_finite_variables(variables, label, *, epoch=None, batch=None):
    for index, variable in enumerate(variables):
        name = getattr(variable, "name", index)
        _require_finite_tensor(
            variable.state,
            f"{label} {name!r}",
            epoch=epoch,
            batch=batch,
        )


def _write_json(path, payload):
    serialized = json.dumps(_json_sanitize(payload), indent=2, allow_nan=False)
    Path(path).write_text(serialized)


def _checkpoint_best_then_callback(
    *,
    energy_fn,
    best_model_path,
    epoch,
    train_loss,
    train_accuracy,
    test_loss,
    test_accuracy,
    best_accuracy,
    best_epoch,
    optimizer,
    history,
    epoch_callback,
):
    """Persist a new best epoch before exposing that epoch to a callback."""
    candidate = test_accuracy if test_accuracy is not None else train_accuracy
    is_best = candidate > best_accuracy
    if is_best:
        best_accuracy = float(candidate)
        best_epoch = int(epoch)
        energy_fn.save(best_model_path)
        checkpoint_params = getattr(energy_fn, "_params", None)
        if checkpoint_params is None:
            checkpoint_params = energy_fn.params()
        load_function_checkpoint_states(
            best_model_path,
            checkpoint_params,
            map_location="cpu",
        )

    history["learning_rate"].append(
        [float(group["lr"]) for group in optimizer.param_groups]
    )
    should_stop = False
    if epoch_callback:
        epoch_info = {
            "epoch": int(epoch),
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "test_loss": test_loss,
            "test_accuracy": test_accuracy,
            "is_best": is_best,
            "best_epoch": best_epoch,
            "best_accuracy": best_accuracy,
        }
        should_stop = bool(epoch_callback(epoch_info, history))
    return best_accuracy, best_epoch, should_stop


def _evaluate_image_loader(
    *,
    loader,
    network,
    free_layers,
    cost_fn,
    minimizer_inference,
    device,
    max_batches,
    split_label,
    epoch,
    epochs,
    log_interval,
):
    running_loss = 0.0
    running_correct = 0
    seen = 0
    total_batches = len(loader)
    batches_this_evaluation = (
        min(total_batches, int(max_batches))
        if max_batches is not None
        else total_batches
    )
    for batch_idx, (images, labels) in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        images = images.to(device)
        labels = labels.to(device)
        _require_finite_tensor(
            images,
            f"{split_label} input",
            epoch=epoch,
            batch=batch_idx + 1,
        )
        network.set_input(images, reset=True)
        minimizer_inference.compute_equilibrium()
        _require_finite_variables(
            free_layers,
            f"{split_label} inference layer",
            epoch=epoch,
            batch=batch_idx + 1,
        )
        cost_fn.set_target(labels)
        batch_cost = cost_fn.eval()
        _require_finite_tensor(
            batch_cost,
            f"{split_label} cost",
            epoch=epoch,
            batch=batch_idx + 1,
        )
        batch_loss = _require_finite_scalar(
            batch_cost.mean().item(),
            f"{split_label} batch loss",
            epoch=epoch,
            batch=batch_idx + 1,
        )
        errors = cost_fn.error_fn()
        batch_correct = int((~errors).sum().item())
        running_loss += batch_loss * images.size(0)
        running_correct += batch_correct
        seen += images.size(0)

        if _should_log_batch(batch_idx, batches_this_evaluation, log_interval):
            print(
                f"[{split_label}] Epoch {epoch}/{epochs} | "
                f"Batch {batch_idx+1}/{total_batches} | "
                f"running loss={running_loss / seen:.4f} "
                f"acc={running_correct / seen * 100:.2f}%",
                flush=True,
            )
    if not seen:
        raise RuntimeError(
            f"Expected the {split_label} loader to yield at least one example. "
            "Provided value: 0 examples."
        )
    return (
        _require_finite_scalar(
            running_loss / seen,
            f"{split_label} loss",
            epoch=epoch,
        ),
        _require_finite_scalar(
            running_correct / seen,
            f"{split_label} accuracy",
            epoch=epoch,
        ),
        int(seen),
    )


def _terminal_official_test_policy(config):
    evaluation = config.get("evaluation", {})
    if evaluation is None:
        evaluation = {}
    if not isinstance(evaluation, dict):
        raise ValueError(
            "Expected config['evaluation'] to be an object. "
            f"Provided value: {evaluation!r}."
        )
    checkpoint_selection = evaluation.get("checkpoint_selection")
    official_test = evaluation.get("official_test", {})
    if official_test is None:
        official_test = {}
    if not isinstance(official_test, dict):
        raise ValueError(
            "Expected config['evaluation']['official_test'] to be an object. "
            f"Provided value: {official_test!r}."
        )
    return checkpoint_selection, str(official_test.get("policy", "disabled"))


def _make_reporting_epoch_callback(
    reporting_run_dir,
    dataset_provenance,
    epochs,
    user_callback,
):
    if reporting_run_dir is None:
        return user_callback

    from experiments.reporting import append_metric, update_status_progress

    run_dir = Path(reporting_run_dir).expanduser().resolve()
    evaluation_split = (
        "validation"
        if isinstance(dataset_provenance, dict)
        and dataset_provenance.get("schema") == "mnist-train-validation-split/v1"
        else "test"
    )

    def callback(epoch_info, history):
        epoch = int(epoch_info["epoch"])
        append_metric(
            run_dir / "metrics.jsonl",
            {
                "kind": "epoch",
                "epoch": epoch,
                "evaluation_split": evaluation_split,
                "metrics": {
                    "train_loss": epoch_info["train_loss"],
                    "train_accuracy": epoch_info["train_accuracy"],
                    f"{evaluation_split}_loss": epoch_info["test_loss"],
                    f"{evaluation_split}_accuracy": epoch_info["test_accuracy"],
                    "is_best": bool(epoch_info["is_best"]),
                    "best_epoch": epoch_info["best_epoch"],
                    f"best_{evaluation_split}_accuracy": epoch_info[
                        "best_accuracy"
                    ],
                },
            },
        )
        update_status_progress(
            run_dir,
            {
                "stage": "training",
                "epoch": epoch,
                "epochs": int(epochs),
            },
        )
        if user_callback is None:
            return False
        return bool(user_callback(epoch_info, history))

    return callback


def _default_quadratic_params():
    return {"diode_conductance": 1.0, "v_min": -1e6, "v_max": 1e6}


def _default_exponential_params():
    return {"I_s": 1e-6, "V_t": 0.025, "V_off": 0.0}


MINIMIZER_SETTINGS_FIELDS = (
    "rel_tol",
    "vn_tol",
    "use_polish",
    "max_newton_iters",
    "z_thresh",
    "exp_clip",
    "dynamic_polish",
    "overrelaxation_reject_steps",
    "overrelaxation_reject_max_tries",
    "overrelaxation_reject_shrink",
    "overrelaxation_reject_eps",
    "experimental_exponential_newton_tol_progressive",
    "experimental_exponential_newton_tol_start",
    "experimental_exponential_newton_tol_end",
    "experimental_exponential_newton_tol_switch_hi",
    "experimental_exponential_newton_tol_switch_lo",
)

MINIMIZER_CONFIG_FIELDS = (
    "double_diode_updater",
    "adaptive_equilibrium",
    "overrelaxation_factor",
    "single_diode_updater",
    "iv_data_path",
    "experimental_damping",
    "experimental_newton_max_steps",
    "settings",
)


def _require_mapping(config: dict, key: str, owner: str) -> dict:
    if key not in config or not isinstance(config[key], dict):
        raise ValueError(
            f"Expected {owner}.{key} to be an explicit object in the config. "
            f"Provided value: {config.get(key)!r}."
        )
    return config[key]


def _require_key(config: dict, key: str, owner: str):
    if key not in config:
        raise ValueError(
            f"Expected {owner}.{key} to be set explicitly in the config. "
            f"Provided value: {config!r}."
        )
    return config[key]


def _require_minimizer_config(model_cfg: dict) -> dict:
    minimizer_cfg = _require_mapping(model_cfg, "minimizer", "model config")
    missing = [key for key in MINIMIZER_CONFIG_FIELDS if key not in minimizer_cfg]
    if missing:
        raise ValueError(
            "Expected model config minimizer to define all simulator fields: "
            f"{list(MINIMIZER_CONFIG_FIELDS)}. Missing fields: {missing}. "
            f"Provided value: {minimizer_cfg!r}."
        )
    _require_mapping(minimizer_cfg, "settings", "model config minimizer")
    return minimizer_cfg


def _minimizer_settings_from_config(minimizer_cfg: dict) -> MinimizerSettings:
    settings_cfg = _require_mapping(minimizer_cfg, "settings", "model config minimizer")
    missing = [key for key in MINIMIZER_SETTINGS_FIELDS if key not in settings_cfg]
    if missing:
        raise ValueError(
            "Expected model config minimizer.settings to define all solver fields: "
            f"{list(MINIMIZER_SETTINGS_FIELDS)}. Missing fields: {missing}. "
            f"Provided value: {settings_cfg!r}."
        )
    return MinimizerSettings(
        **{field: settings_cfg[field] for field in MINIMIZER_SETTINGS_FIELDS}
    )


def _validate_diode_param_config(model_cfg: dict) -> None:
    for key in ("quadratic_diode_param", "exponential_diode_param", "hard_sigmoid_param"):
        _require_mapping(model_cfg, key, "model config")
    if model_cfg["non_linearity"] == "hard_sigmoid":
        params = model_cfg["hard_sigmoid_param"]
        missing = [key for key in ("g_on", "g_off") if key not in params]
        if missing:
            raise ValueError(
                "Expected model config hard_sigmoid_param to define explicit "
                f"'g_on' and 'g_off'. Missing fields: {missing}. Provided value: {params!r}."
            )


def _sanity_check_minimizer_settings() -> MinimizerSettings:
    return MinimizerSettings(
        rel_tol=1e-5,
        vn_tol=1e-6,
        use_polish=True,
        max_newton_iters=32,
        z_thresh=1e10,
        exp_clip=100000.0,
        dynamic_polish=True,
        overrelaxation_reject_steps=False,
        overrelaxation_reject_max_tries=3,
        overrelaxation_reject_shrink=0.5,
        overrelaxation_reject_eps=0.0,
        experimental_exponential_newton_tol_progressive=True,
        experimental_exponential_newton_tol_start=1e-5,
        experimental_exponential_newton_tol_end=1e-5,
        experimental_exponential_newton_tol_switch_hi=1e-2,
        experimental_exponential_newton_tol_switch_lo=5e-4,
    )


def _build_tracking_minimizer(
    fn,
    free_layers,
    model_cfg,
    mode,
    *,
    num_iterations,
    voltage_amp,
    current_amp,
    adaptive_equilibrium=None,
):
    _validate_diode_param_config(model_cfg)
    minimizer_cfg = _require_minimizer_config(model_cfg)
    double_diode_updater = _require_key(
        minimizer_cfg, "double_diode_updater", "model config minimizer"
    )
    configured_adaptive_equilibrium = _require_key(
        minimizer_cfg, "adaptive_equilibrium", "model config minimizer"
    )
    if adaptive_equilibrium is None:
        adaptive_equilibrium = configured_adaptive_equilibrium
    return TrackingQuadraticMinimizer(
        fn=fn,
        free_layers=free_layers,
        num_iterations=num_iterations,
        mode=mode,
        non_linearity=model_cfg["non_linearity"],
        quadratic_diode_param=model_cfg["quadratic_diode_param"],
        exponential_diode_param=model_cfg["exponential_diode_param"],
        hard_sigmoid_param=model_cfg["hard_sigmoid_param"],
        voltage_amp=voltage_amp,
        current_amp=current_amp,
        iv_data=minimizer_cfg.get("iv_data"),
        iv_data_path=_require_key(minimizer_cfg, "iv_data_path", "model config minimizer"),
        double_diode_updater=double_diode_updater,
        adaptive_equilibrium=adaptive_equilibrium,
        overrelaxation_factor=_require_key(
            minimizer_cfg, "overrelaxation_factor", "model config minimizer"
        ),
        single_diode_updater=_require_key(
            minimizer_cfg, "single_diode_updater", "model config minimizer"
        ),
        minimizer_settings=_minimizer_settings_from_config(minimizer_cfg),
        damping=_require_key(minimizer_cfg, "experimental_damping", "model config minimizer"),
        experimental_newton_max_steps=_require_key(
            minimizer_cfg, "experimental_newton_max_steps", "model config minimizer"
        ),
    )


def _single_conv_gradient_check(device):
    """Sanity check: ensure a standalone conv layer reaches quadratic equilibrium."""
    torch.manual_seed(0)

    batch_size = 2
    input_shape = (2, 6, 6)
    kernel = (3, 3)
    stride = 1
    padding = 0
    out_channels = 4
    h_out = (input_shape[1] + 2 * padding - kernel[0]) // stride + 1
    w_out = (input_shape[2] + 2 * padding - kernel[1]) // stride + 1
    output_shape = (out_channels, h_out, w_out)

    input_layer = ConvLayer(input_shape, device=device)
    output_layer = ConvLayer(output_shape, device=device)

    conv_weight = ConvWeight(
        shape=(out_channels, input_shape[0], kernel[0], kernel[1]),
        gain=0.5,
        device=device,
        clamp=False,
        clamp_min=None,
        clamp_max=None,
    )

    conv_interaction = ConvResistive(
        input_layer, output_layer, conv_weight, padding=padding, stride=stride, dilation=1
    )

    energy_fn = DetailedSumSeparableFunction(
        layers=[input_layer, output_layer],
        params=[conv_weight],
        interactions=[conv_interaction],
    )

    input_layer.state = torch.randn(batch_size, *input_shape, device=device)
    output_layer.state = torch.zeros(batch_size, *output_shape, device=device)

    minimizer = QuadraticMinimizer(
        energy_fn,
        free_layers=[output_layer],
        num_iterations=3,
        mode="forward",
        non_linearity="linear",
        quadratic_diode_param=_default_quadratic_params(),
        exponential_diode_param=_default_exponential_params(),
        voltage_amp=1.0,
        current_amp=1.0,
        hard_sigmoid_param={},
        iv_data=None,
        iv_data_path=None,
        double_diode_updater="CustomExponentialDoubleDiodeUpdater",
        adaptive_equilibrium=False,
        overrelaxation_factor=1.1,
        single_diode_updater="custom",
        minimizer_settings=_sanity_check_minimizer_settings(),
    )

    grad_fn = energy_fn.grad_layer_fn(output_layer)
    grad_before = grad_fn().norm().item()
    minimizer.compute_equilibrium()
    grad_after = grad_fn().norm().item()
    return grad_before, grad_after


def load_config(config_path):
    config_path = Path(config_path)
    if not config_path.is_absolute():
        config_path = LABS_DIR / config_path
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    return json.loads(config_path.read_text())


def _normalize_dataset_key(dataset_key):
    if dataset_key is None:
        return "mnist"
    normalized = str(dataset_key).strip().lower().replace("-", "_")
    if normalized == "fashion_mnist":
        return "fmnist"
    return normalized


def _resolve_dataset_config(config, dataset_key):
    normalized = _normalize_dataset_key(dataset_key)
    datasets_cfg = config.get("datasets", {})

    if normalized in datasets_cfg:
        return normalized, json.loads(json.dumps(datasets_cfg[normalized]))

    if normalized == "fmnist":
        base_cfg = datasets_cfg.get("mnist")
        if base_cfg is None:
            raise KeyError(
                "Config does not define datasets['mnist']; cannot derive Fashion-MNIST dataset settings."
            )

        dataset_cfg = json.loads(json.dumps(base_cfg))
        dataset_cfg["factory"] = "labs.datasets.FashionMnistDataset"

        params = dict(dataset_cfg.get("params", {}))
        params["name"] = "fmnist"
        root = params.get("root")
        if root is None or Path(str(root)).name == "mnist":
            params["root"] = "~/datasets/fashion_mnist"
        params.setdefault("download", True)
        params.setdefault("normalize", True)
        # MNIST's inherited statistics are invalid for Fashion-MNIST. A
        # dataset-specific entry above can still explicitly override these.
        params["normalize_mean"] = 0.2860
        params["normalize_std"] = 0.3530
        dataset_cfg["params"] = params
        return normalized, dataset_cfg

    raise KeyError(f"Unsupported dataset key {dataset_key!r}.")


def _resolve_callable(import_path):
    module_path, attr = import_path.rsplit(".", 1)

    if module_path.startswith("labs."):
        rel_module = module_path[len("labs."):]
        module_base = rel_module.replace(".", "/")
        candidate = LABS_DIR / f"{module_base}.py"
        if not candidate.exists():
            candidate = LABS_DIR / rel_module.replace(".", "/") / "__init__.py"
        if not candidate.exists():
            raise ImportError(f"Cannot locate module '{module_path}' at {candidate}")

        spec = importlib.util.spec_from_file_location(
            f"labs_{rel_module.replace('.', '_')}", candidate
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    else:
        module = import_module(module_path)

    return getattr(module, attr)


def _resolve_run_dir(config_path: Path, output_dir: str | None) -> tuple[Path, str, str]:
    host = socket.gethostname().split(".")[0]
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    if output_dir:
        run_dir = Path(output_dir).expanduser().resolve()
    elif config_path.parent.name.startswith("trial_"):
        run_dir = config_path.parent
    else:
        run_dir = DEFAULT_IMAGE_RUNS_BASE / f"{config_path.stem}_{host}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir, host, timestamp


def _train_image_task(
    config_path,
    epochs,
    lr,
    beta,
    log_interval,
    max_batches,
    dataset_key,
    model_key,
    image_shape,
    device=None,
    sanity_check=False,
    epoch_callback=None,
    output_dir=None,
    training_algorithm=None,
    seed=None,
    lr_decay=None,
    init_checkpoint_path=None,
    max_test_batches=None,
    batch_state_policy=None,
    optimizer_name=None,
    momentum=None,
    weight_decay=None,
    gradient_callback=None,
    optimizer_step_callback=None,
    apply_optimizer_steps=True,
    reporting_run_dir=None,
    skip_terminal_official_test=False,
):
    config_path = Path(config_path).expanduser().resolve()
    config = load_config(config_path)
    seed_value = seed if seed is not None else config.get("seed")
    _set_seed(seed_value)
    _reset_name_counters()
    dataset_key = _normalize_dataset_key(dataset_key)
    run_dir, host, timestamp = _resolve_run_dir(config_path, output_dir)
    config_snapshot_path = run_dir / "config.used.json"
    _write_json(config_snapshot_path, config)
    batch_state_policy = _resolve_batch_state_policy(config, batch_state_policy)
    epochs = int(epochs)
    if epochs <= 0:
        raise ValueError(f"Expected epochs to be a positive integer. Provided value: {epochs!r}.")
    training_algorithm = _normalize_training_algorithm(
        training_algorithm if training_algorithm is not None else config.get("training_algorithm")
    )
    optimizer_name_value, momentum_value, weight_decay_value = _resolve_optimizer_settings(
        config,
        optimizer_name=optimizer_name,
        momentum=momentum,
        weight_decay=weight_decay,
    )
    optimizer_details = _optimizer_details(
        config,
        optimizer_name_value,
        momentum_value,
        weight_decay_value,
    )

    project_root = PROJECT_ROOT
    if str(project_root) not in sys.path:
        sys.path.append(str(project_root))

    requested_device = device if device is not None else config.get("device")
    if requested_device is not None:
        device = torch.device(requested_device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(f"Requested CUDA device {requested_device!r}, but CUDA is not available.")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if sanity_check:
        grad_before, grad_after = _single_conv_gradient_check(device)
        print(
            f"[Sanity] Single conv gradient norm {grad_before:.6e} -> {grad_after:.6e}"
        )
        if not grad_after < grad_before * 1e-3:
            raise RuntimeError(
                "Single conv sanity check failed: gradients did not decrease sufficiently."
            )

    model_overrides = config["model_overrides"]
    resolved_model_key = model_key
    if resolved_model_key not in model_overrides:
        lab_model_key = config.get("lab", {}).get("model_key")
        if lab_model_key in model_overrides:
            resolved_model_key = lab_model_key
        elif dataset_key in model_overrides:
            resolved_model_key = dataset_key
        elif len(model_overrides) == 1:
            resolved_model_key = next(iter(model_overrides))
        else:
            available = ", ".join(sorted(model_overrides))
            raise KeyError(
                f"Unknown model_key {model_key!r}; available model_overrides keys: {available}"
            )

    model_cfg = {**config["model_base"], **model_overrides[resolved_model_key]}
    _validate_diode_param_config(model_cfg)
    _require_minimizer_config(model_cfg)
    dataset_key, dataset_cfg = _resolve_dataset_config(config, dataset_key)
    beta_value = beta
    if beta_value is None:
        beta_value = config.get("beta")
        if beta_value is None:
            beta_value = max(model_cfg.get("nudging", 0.0), 1e-3)
    beta_value = _require_finite_scalar(beta_value, "beta")
    optimizer_cfg = config.get("optimizer", {})
    optimizer_lr_decay = optimizer_cfg.get("lr_decay") if isinstance(optimizer_cfg, dict) else None
    top_level_lr_decay = config.get("lr_decay")
    if optimizer_lr_decay is not None and top_level_lr_decay is not None:
        if _require_finite_scalar(optimizer_lr_decay, "configured optimizer lr_decay") != _require_finite_scalar(
            top_level_lr_decay, "configured top-level lr_decay"
        ):
            raise ValueError(
                "Expected config['optimizer']['lr_decay'] to equal config['lr_decay']. "
                f"Provided values: optimizer={optimizer_lr_decay!r}, top-level={top_level_lr_decay!r}."
            )
    lr_decay_value = lr_decay
    if lr_decay_value is None:
        lr_decay_value = optimizer_lr_decay
    if lr_decay_value is None:
        lr_decay_value = top_level_lr_decay
    lr_decay_value = 1.0 if lr_decay_value is None else _require_finite_scalar(
        lr_decay_value,
        "lr_decay",
    )
    for source, configured in (
        ("config['optimizer']['lr_decay']", optimizer_lr_decay),
        ("config['lr_decay']", top_level_lr_decay),
    ):
        if configured is not None and _require_finite_scalar(configured, source) != lr_decay_value:
            raise ValueError(
                f"Expected supplied lr_decay to equal {source}. "
                f"Provided values: supplied={lr_decay_value!r}, configured={configured!r}."
            )
    if lr_decay_value <= 0.0:
        raise ValueError(f"Expected positive lr_decay, got {lr_decay_value}.")

    height, width = image_shape

    raw_layer_shapes = model_cfg["layer_shapes"]
    layer_shapes = [
        tuple(shape) if isinstance(shape, (list, tuple)) else (shape,)
        for shape in raw_layer_shapes
    ]
    conv_pipeline = model_cfg.get("conv_pipeline", [])
    if conv_pipeline is None:
        conv_pipeline = []

    input_shape = layer_shapes[0]
    if len(input_shape) != 3:
        raise ValueError(
            f"Expected input layer shape with 3 dimensions (channels, H, W); got {input_shape}"
        )
    if input_shape[1:] != (height, width):
        raise ValueError(
            f"Input layer shape {input_shape} does not match dataset image size {(height, width)}."
        )

    output_shape = layer_shapes[-1]
    if len(output_shape) != 1:
        raise ValueError(
            f"Output layer shape must be 1-D (num_classes or 2*num_classes); got {output_shape}"
        )

    num_classes = 10

    print(
        f"Training model '{model_key}' on dataset '{dataset_key}' | layer_shapes={layer_shapes} "
        f"| num_classes={num_classes} | non_linearity={model_cfg['non_linearity']} "
        f"| training_algorithm={training_algorithm} | beta={beta_value}"
    )

    energy_fn = FlexibleDeepResistiveEnergy(
        layer_shapes=layer_shapes,
        conv_pipeline=conv_pipeline,
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
        weight_init_mode=model_cfg.get("weight_init_mode", "kaiming_uniform"),
        input_mode=config.get("input_mode", "train"),
        trainable_amplification=bool(model_cfg.get("trainable_amplification", False)),
        amplification_min=model_cfg.get("amplification_min", 1e-6),
        amplification_max=model_cfg.get("amplification_max"),
    )
    energy_fn.set_device(device)
    _require_finite_scalar(scalar_float(energy_fn._voltage_amp), "initial voltage amplification")
    _require_finite_scalar(scalar_float(energy_fn._current_amp), "initial current amplification")
    init_checkpoint_value = (
        init_checkpoint_path
        if init_checkpoint_path is not None
        else config.get("init_checkpoint_path")
    )
    if init_checkpoint_value:
        init_checkpoint_path = Path(init_checkpoint_value).expanduser().resolve()
        if not init_checkpoint_path.exists():
            raise FileNotFoundError(
                f"Expected init_checkpoint_path to exist, got {init_checkpoint_path}."
            )
        energy_fn.load(init_checkpoint_path)
        print(f"[mnist_train] Initialized model from checkpoint {init_checkpoint_path}")
    else:
        init_checkpoint_path = None
    checkpoint_params = getattr(energy_fn, "_params", energy_fn.params())
    param_schema = _param_schema(checkpoint_params)
    configured_parameter_order = config.get("parameter_order")
    if configured_parameter_order is not None:
        actual_parameter_order = [name for name, _parameter in param_schema]
        if list(configured_parameter_order) != actual_parameter_order:
            raise ValueError(
                "Expected config['parameter_order'] to equal the trainer's runtime "
                "parameter order. "
                f"Provided values: configured={configured_parameter_order!r}, "
                f"runtime={actual_parameter_order!r}."
            )
    _require_finite_variables(checkpoint_params, "initial parameter")

    network = Network(energy_fn)
    free_layers = network.free_layers()
    output_layer = energy_fn.layers()[-1]

    if len(output_layer.shape) != 1:
        raise ValueError("Image setup expects a vector output layer.")

    output_dim = output_layer.shape[0]

    if output_dim == num_classes:
        cost_fn = SquaredError(output_layer)
    elif output_dim == 2 * num_classes:
        cost_fn = SquaredErrorPairedOutputs(output_layer, num_classes=num_classes)
    else:
        raise ValueError(
            f"Unsupported output_dim={output_dim}; expected {num_classes} or {2 * num_classes}."
        )
    
    augmented_fn = AugmentedFunction(energy_fn, cost_fn)
    training_fn = augmented_fn if training_algorithm == "EP" else energy_fn

    training_iterations = int(
        model_cfg.get("num_iterations_training", model_cfg["num_iterations_inference"])
    )
    inference_iterations = int(model_cfg["num_iterations_inference"])
    minimizer_training = _build_tracking_minimizer(
        training_fn,
        free_layers,
        model_cfg,
        config["energy_minimizer"]["mode"],
        num_iterations=training_iterations,
        voltage_amp=energy_fn._voltage_amp,
        current_amp=energy_fn._current_amp,
    )

    minimizer_inference = _build_tracking_minimizer(
        energy_fn,
        free_layers,
        model_cfg,
        config["energy_minimizer"]["mode"],
        num_iterations=inference_iterations,
        voltage_amp=energy_fn._voltage_amp,
        current_amp=energy_fn._current_amp,
    )



    dataset_factory = _resolve_callable(dataset_cfg["factory"])
    dataset_params = dict(dataset_cfg["params"])
    dataset_params.setdefault("device", device)
    if "root" in dataset_params:
        dataset_params["root"] = os.path.expanduser(str(dataset_params["root"]))
    dataset_builder = dataset_factory(**dataset_params)
    loader_result = dataset_builder.build()
    dataset_provenance = None
    if hasattr(loader_result, "train_loader") and hasattr(loader_result, "validation_loader"):
        train_loader = loader_result.train_loader
        test_loader = loader_result.validation_loader
        dataset_provenance = {
            "schema": "mnist-train-validation-split/v1",
            "source_split": "train",
            "split_seed": int(loader_result.split_seed),
            "shuffle_seed": int(loader_result.shuffle_seed),
            "train_indices_sha256": loader_result.train_indices_hash,
            "validation_indices_sha256": loader_result.validation_indices_hash,
            "first_epoch_batch_order_sha256": loader_result.first_epoch_batch_order_hash,
        }
    elif isinstance(loader_result, tuple):
        train_loader, test_loader = loader_result
    else:
        train_loader = loader_result
        test_loader = None
    checkpoint_selection, official_test_policy = _terminal_official_test_policy(config)
    if official_test_policy not in {"disabled", "terminal_once"}:
        raise ValueError(
            "Expected official-test policy to be 'disabled' or 'terminal_once'. "
            f"Provided value: {official_test_policy!r}."
        )
    if official_test_policy == "terminal_once":
        if dataset_provenance is None:
            raise ValueError(
                "Terminal official-test evaluation requires a train/validation dataset."
            )
        if checkpoint_selection != "maximum_validation_accuracy":
            raise ValueError(
                "Terminal official-test evaluation requires "
                "evaluation.checkpoint_selection='maximum_validation_accuracy'."
            )
        if not hasattr(dataset_builder, "build_official_test_loader"):
            raise ValueError(
                "Terminal official-test evaluation requires the dataset factory "
                "to provide build_official_test_loader()."
            )

    epoch_callback = _make_reporting_epoch_callback(
        reporting_run_dir,
        dataset_provenance,
        epochs,
        epoch_callback,
    )

    if training_algorithm == "EP":
        params = augmented_fn.params()
        estimator = EquilibriumProp(
            params,
            free_layers,
            augmented_fn,
            cost_fn,
            minimizer_training,
            variant="centered",
            nudging=beta_value,
        )
    else:
        params = energy_fn.params()
        estimator = Backprop(params, free_layers, cost_fn, minimizer_training)

    energy_params = energy_fn.params()
    cost_params = cost_fn.params()
    if isinstance(lr, (list, tuple)):
        if len(lr) < len(energy_params) + len(cost_params):
            raise ValueError(
                f"Expected {len(energy_params) + len(cost_params)} learning rates, got {len(lr)}."
            )
        learning_rates = [
            _require_finite_scalar(value, f"learning rate {index}")
            for index, value in enumerate(lr)
        ]
    else:
        learning_rate = _require_finite_scalar(lr, "learning rate")
        learning_rates = [learning_rate] * (len(energy_params) + len(cost_params))
    _validate_optimizer_rate_consistency(config, learning_rates, lr_decay_value)
    optimizer = _build_parameter_optimizer(
        energy_fn,
        cost_fn,
        learning_rates,
        optimizer_details,
    )

    history = {
        "loss": [],
        "accuracy": [],
        "test_loss": [],
        "test_accuracy": [],
        "learning_rate": [],
    }
    writer = SummaryWriter(str(run_dir)) if SummaryWriter is not None else None
    if writer is None:
        print("[mnist_train] TensorBoard unavailable; proceeding without event files.")
    else:
        print(f"[mnist_train] Writing TensorBoard events to {run_dir}")

    resolved_run_config = {
        "config_path": str(config_path),
        "init_checkpoint_path": str(init_checkpoint_path) if init_checkpoint_path else None,
        "training_algorithm": training_algorithm,
        "seed": seed_value,
        "dataset": {
            "key": dataset_key,
            "factory": dataset_cfg["factory"],
            "params": dataset_params,
            "provenance": dataset_provenance,
            "preprocessing": {
                "normalize": bool(dataset_params.get("normalize", False)),
                "normalize_mean": dataset_params.get("normalize_mean"),
                "normalize_std": dataset_params.get("normalize_std"),
                "normalize_scale": dataset_params.get("normalize_scale", 1.0),
                "input_mode": config.get("input_mode", "train"),
            },
        },
        "architecture": {
            "model_key": resolved_model_key,
            "layer_shapes": [list(shape) for shape in layer_shapes],
            "conv_pipeline": conv_pipeline,
            "weight_gains": list(model_cfg["weight_gains"]),
            "weight_init_mode": model_cfg.get("weight_init_mode", "kaiming_uniform"),
            "input_gain": model_cfg["input_gain"],
            "non_linearity": model_cfg["non_linearity"],
            "output_dim": output_dim,
            "num_classes": num_classes,
        },
        "optimizer": {
            **optimizer_details,
            "learning_rate": list(learning_rates),
            "lr_decay": lr_decay_value,
        },
        "training": {
            "epochs": int(epochs),
            "batch_size": int(dataset_params.get("batch_size")),
            "max_batches": max_batches,
            "max_test_batches": max_test_batches,
            "batch_state_policy": batch_state_policy,
            "optimizer_steps_applied": bool(apply_optimizer_steps),
            "num_iterations_training": training_iterations,
            "num_iterations_inference": inference_iterations,
            "beta": beta_value,
        },
        "evaluation": {
            "epoch_split": "validation" if dataset_provenance is not None else "test",
            "checkpoint_selection": checkpoint_selection,
            "official_test_policy": official_test_policy,
            "official_test_skipped": bool(skip_terminal_official_test),
        },
        "amplification": {
            "voltage_amp": float(model_cfg["voltage_amp"]),
            "current_amp": float(model_cfg["current_amp"]),
            "trainable": bool(model_cfg.get("trainable_amplification", False)),
            "amplification_min": model_cfg.get("amplification_min", 1e-6),
            "amplification_max": model_cfg.get("amplification_max"),
            "mapping": "voltage_amp=1 and current_amp=1 is the unamplified baseline.",
        },
        "bounds": {
            "weight_min": model_cfg.get("weight_min"),
            "weight_max": model_cfg.get("weight_max"),
        },
        "diode_params": {
            "quadratic_diode_param": model_cfg["quadratic_diode_param"],
            "exponential_diode_param": model_cfg["exponential_diode_param"],
            "hard_sigmoid_param": model_cfg["hard_sigmoid_param"],
        },
        "minimizer": {
            "mode": config["energy_minimizer"]["mode"],
            **json.loads(json.dumps(_json_sanitize(model_cfg["minimizer"]))),
        },
    }
    config_json_path = run_dir / "config.json"
    _write_json(config_json_path, resolved_run_config)

    best_test_accuracy = float("-inf")
    best_epoch = None
    best_model_path = run_dir / "best_model.pt"
    stop_training = False
    for epoch in range(epochs):
        running_loss = 0.0
        running_correct = 0
        seen = 0
        train_total_batches = len(train_loader)
        train_batches_this_epoch = (
            min(train_total_batches, int(max_batches))
            if max_batches is not None
            else train_total_batches
        )

        for batch_idx, (images, labels) in enumerate(train_loader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            optimizer.zero_grad()
            images = images.to(device)
            labels = labels.to(device)
            _require_finite_tensor(
                images,
                "training input",
                epoch=epoch + 1,
                batch=batch_idx + 1,
            )

            network.set_input(
                images,
                reset=_batch_reset_flag(batch_state_policy, batch_idx),
            )
            minimizer_inference.compute_equilibrium()
            _require_finite_variables(
                free_layers,
                "inference layer",
                epoch=epoch + 1,
                batch=batch_idx + 1,
            )
            cost_fn.set_target(labels)

            batch_cost = cost_fn.eval()
            _require_finite_tensor(
                batch_cost,
                "training cost",
                epoch=epoch + 1,
                batch=batch_idx + 1,
            )
            batch_loss = _require_finite_scalar(
                batch_cost.mean().item(),
                "training batch loss",
                epoch=epoch + 1,
                batch=batch_idx + 1,
            )
            errors = cost_fn.error_fn()
            batch_correct = int((~errors).sum().item())

            running_loss += batch_loss * images.size(0)
            running_correct += batch_correct
            seen += images.size(0)

            grads = estimator.compute_gradient()
            if len(grads) < len(params):
                raise RuntimeError(
                    f"Expected at least {len(params)} parameter gradients at epoch={epoch + 1}, "
                    f"batch={batch_idx + 1}. Provided value: {len(grads)}."
                )
            for param_index, (param, grad) in enumerate(zip(params, grads[:len(params)])):
                _require_finite_tensor(
                    grad,
                    f"gradient {getattr(param, 'name', param_index)!r}",
                    epoch=epoch + 1,
                    batch=batch_idx + 1,
                )
                if tuple(grad.shape) != tuple(param.state.shape):
                    raise RuntimeError(
                        f"Expected gradient {param_index} to have shape {tuple(param.state.shape)} "
                        f"at epoch={epoch + 1}, batch={batch_idx + 1}. "
                        f"Provided value: {tuple(grad.shape)}."
                    )
                param.state.grad = grad
            stop_after_batch = False
            if gradient_callback is not None:
                stop_after_batch = bool(
                    gradient_callback(
                        {
                            "epoch": epoch + 1,
                            "batch": batch_idx + 1,
                            "loss": batch_loss,
                            "parameters": tuple(params),
                            "gradients": tuple(grads[:len(params)]),
                        }
                    )
                )
            if apply_optimizer_steps:
                tracked_indices = (
                    [
                        index
                        for index, param in enumerate(params)
                        if str(getattr(param, "name", "")).strip().startswith(
                            ("ConvWeight_", "DenseWeight_")
                        )
                    ]
                    if optimizer_step_callback is not None
                    else []
                )
                pre_optimizer_states = {
                    str(getattr(params[index], "name", "")).strip(): (
                        params[index].state.detach().clone()
                    )
                    for index in tracked_indices
                }
                optimizer.step()
                _require_finite_variables(
                    energy_params + cost_params,
                    "optimizer-updated parameter",
                    epoch=epoch + 1,
                    batch=batch_idx + 1,
                )
                _require_finite_optimizer_state(
                    optimizer,
                    epoch=epoch + 1,
                    batch=batch_idx + 1,
                )
                post_optimizer_states = {
                    str(getattr(params[index], "name", "")).strip(): (
                        params[index].state.detach().clone()
                    )
                    for index in tracked_indices
                }
                for param in energy_params:
                    param.clamp_()
                _require_finite_variables(
                    energy_params + cost_params,
                    "clamped parameter",
                    epoch=epoch + 1,
                    batch=batch_idx + 1,
                )
                if optimizer_step_callback is not None:
                    stop_after_batch = bool(
                        optimizer_step_callback(
                            {
                                "epoch": epoch + 1,
                                "batch": batch_idx + 1,
                                "loss": batch_loss,
                                "parameters": tuple(params),
                                "gradients": tuple(grads[:len(params)]),
                                "pre_optimizer_states": pre_optimizer_states,
                                "post_optimizer_states": post_optimizer_states,
                                "post_projection_states": {
                                    str(
                                        getattr(params[index], "name", "")
                                    ).strip(): (
                                        params[index].state.detach().clone()
                                    )
                                    for index in tracked_indices
                                },
                            }
                        )
                    ) or stop_after_batch

            acc = running_correct / seen
            avg_loss = running_loss / seen
            if _should_log_batch(batch_idx, train_batches_this_epoch, log_interval):
                print(
                    f"Epoch {epoch+1}/{epochs} | Batch {batch_idx+1}/{train_total_batches} | "
                    f"running loss={avg_loss:.4f} acc={acc*100:.2f}%",
                    flush=True,
                )
            if stop_after_batch:
                break

        if not seen:
            raise RuntimeError(
                f"Expected the training loader to yield at least one example in epoch {epoch + 1}. "
                "Provided value: 0 examples."
            )
        epoch_train_loss = _require_finite_scalar(
            running_loss / seen,
            "epoch training loss",
            epoch=epoch + 1,
        )
        epoch_train_acc = _require_finite_scalar(
            running_correct / seen,
            "epoch training accuracy",
            epoch=epoch + 1,
        )
        history["loss"].append(epoch_train_loss)
        history["accuracy"].append(epoch_train_acc)

        if test_loader is not None:
            optimizer.zero_grad()
            epoch_test_loss, epoch_test_acc, _test_seen = _evaluate_image_loader(
                loader=test_loader,
                network=network,
                free_layers=free_layers,
                cost_fn=cost_fn,
                minimizer_inference=minimizer_inference,
                device=device,
                max_batches=max_test_batches,
                split_label="Validation" if dataset_provenance is not None else "Test",
                epoch=epoch + 1,
                epochs=epochs,
                log_interval=log_interval,
            )
            history["test_loss"].append(epoch_test_loss)
            history["test_accuracy"].append(epoch_test_acc)
        else:
            epoch_test_loss = None
            epoch_test_acc = None

        print()

        if writer is not None:
            writer.add_scalar("train/loss", epoch_train_loss, epoch + 1)
            writer.add_scalar("train/accuracy", epoch_train_acc, epoch + 1)
            if epoch_test_loss is not None:
                writer.add_scalar("test/loss", epoch_test_loss, epoch + 1)
            if epoch_test_acc is not None:
                writer.add_scalar("test/accuracy", epoch_test_acc, epoch + 1)
                writer.add_scalar("test/error", 1.0 - epoch_test_acc, epoch + 1)
            writer.flush()

        best_test_accuracy, best_epoch, should_stop = _checkpoint_best_then_callback(
            energy_fn=energy_fn,
            best_model_path=best_model_path,
            epoch=epoch + 1,
            train_loss=epoch_train_loss,
            train_accuracy=epoch_train_acc,
            test_loss=epoch_test_loss,
            test_accuracy=epoch_test_acc,
            best_accuracy=best_test_accuracy,
            best_epoch=best_epoch,
            optimizer=optimizer,
            history=history,
            epoch_callback=epoch_callback,
        )
        if should_stop:
            stop_training = True
            break

        if lr_decay_value != 1.0 and epoch + 1 < epochs:
            for group in optimizer.param_groups:
                group["lr"] *= lr_decay_value

    if stop_training:
        print("Epoch callback requested early stop; summarizing current history.")

    def _last(values):
        return values[-1] if values else float("nan")

    summary = {
        "final_train_loss": _last(history["loss"]),
        "final_train_accuracy": _last(history["accuracy"]),
        "best_train_accuracy": max(history["accuracy"]) if history["accuracy"] else float("nan"),
    }
    if history["test_accuracy"]:
        summary["final_test_accuracy"] = history["test_accuracy"][-1]
        summary["best_test_accuracy"] = best_test_accuracy
        summary["best_epoch"] = best_epoch
        summary["final_test_loss"] = history["test_loss"][-1]
        summary["test_error"] = 1.0 - summary["final_test_accuracy"]
    else:
        summary["best_epoch"] = best_epoch
        summary["test_error"] = 1.0 - summary["final_train_accuracy"]

    _require_finite_variables(energy_params + cost_params, "final parameter")
    final_model_path = run_dir / "final_model.pt"
    energy_fn.save(final_model_path)
    if not best_model_path.exists():
        best_model_path = run_dir / "best_model.pt"
        energy_fn.save(best_model_path)
        load_function_checkpoint_states(
            best_model_path,
            checkpoint_params,
            map_location="cpu",
        )
        best_epoch = len(history["accuracy"]) if history["accuracy"] else None
        if history["test_accuracy"]:
            best_test_accuracy = history["test_accuracy"][-1]
        elif history["accuracy"]:
            best_test_accuracy = history["accuracy"][-1]

    official_test_loss = None
    official_test_accuracy = None
    official_test_examples = 0
    official_test_evaluations = 0
    if official_test_policy == "terminal_once" and not skip_terminal_official_test:
        energy_fn.load(best_model_path)
        _require_finite_variables(
            checkpoint_params,
            "best-validation checkpoint parameter",
        )
        official_test_loader = dataset_builder.build_official_test_loader()
        official_test_loss, official_test_accuracy, official_test_examples = (
            _evaluate_image_loader(
                loader=official_test_loader,
                network=network,
                free_layers=free_layers,
                cost_fn=cost_fn,
                minimizer_inference=minimizer_inference,
                device=device,
                max_batches=None,
                split_label="OfficialTest",
                epoch=best_epoch,
                epochs=epochs,
                log_interval=log_interval,
            )
        )
        official_test_evaluations = 1
        energy_fn.load(final_model_path)
        _require_finite_variables(checkpoint_params, "restored final parameter")
        summary["official_test_loss"] = official_test_loss
        summary["official_test_accuracy"] = official_test_accuracy
        summary["official_test_error"] = 1.0 - official_test_accuracy
        summary["official_test_checkpoint"] = "best_validation"
        summary["official_test_evaluations"] = official_test_evaluations
        summary["official_test_examples"] = official_test_examples

    weights_metadata = {
        "voltage_amp": float(model_cfg["voltage_amp"]),
        "current_amp": float(model_cfg["current_amp"]),
        "seed": seed_value,
        "layer_shapes": [list(shape) for shape in layer_shapes],
        "non_linearity": model_cfg["non_linearity"],
        "training_algorithm": training_algorithm,
        "checkpoint_format": "versioned DRN function checkpoint with exact ordered parameter schema",
        "batch_state_policy": batch_state_policy,
    }
    weights_final_path = run_dir / "weights_final.npz"
    weights_best_path = run_dir / "weights_best.npz"
    _checkpoint_to_npz(final_model_path, weights_final_path, param_schema, weights_metadata)
    _checkpoint_to_npz(best_model_path, weights_best_path, param_schema, weights_metadata)
    history_paths = _save_history_arrays(run_dir, history)
    learned_hard_sigmoid_v_off = _hard_sigmoid_v_off_values(energy_fn)
    learned_amplification = _amplification_values(energy_fn)
    for name, value in learned_hard_sigmoid_v_off.items():
        _require_finite_scalar(value, f"learned hard-sigmoid parameter {name!r}")
    for name, value in learned_amplification.items():
        _require_finite_scalar(value, f"learned amplification {name!r}")

    metrics = {
        "run_dir": str(run_dir),
        "training_algorithm": training_algorithm,
        "seed": seed_value,
        "batch_state_policy": batch_state_policy,
        "voltage_amp": float(model_cfg["voltage_amp"]),
        "current_amp": float(model_cfg["current_amp"]),
        "best_epoch": summary.get("best_epoch", best_epoch),
        "final_train_loss": summary.get("final_train_loss"),
        "final_train_accuracy": summary.get("final_train_accuracy"),
        "best_train_accuracy": summary.get("best_train_accuracy"),
        "best_test_accuracy": summary.get("best_test_accuracy", best_test_accuracy),
        "final_test_accuracy": summary.get("final_test_accuracy"),
        "final_test_loss": summary.get("final_test_loss"),
        "test_error": summary.get("test_error"),
        "best_validation_accuracy": (
            summary.get("best_test_accuracy") if dataset_provenance is not None else None
        ),
        "final_validation_accuracy": (
            summary.get("final_test_accuracy") if dataset_provenance is not None else None
        ),
        "final_validation_loss": (
            summary.get("final_test_loss") if dataset_provenance is not None else None
        ),
        "official_test_accuracy": official_test_accuracy,
        "official_test_loss": official_test_loss,
        "official_test_examples": official_test_examples,
        "official_test_evaluations": official_test_evaluations,
        "official_test_checkpoint": (
            "best_validation" if official_test_evaluations else None
        ),
        "lr_decay": lr_decay_value,
        "optimizer": optimizer_details,
        "optimizer_steps_applied": bool(apply_optimizer_steps),
        "dataset_provenance": dataset_provenance,
        "final_learning_rate": [float(group["lr"]) for group in optimizer.param_groups],
        "checkpoint_path": str(final_model_path),
        "best_checkpoint_path": str(best_model_path),
        "weights_final_path": str(weights_final_path),
        "weights_best_path": str(weights_best_path),
        "config_path": str(config_json_path),
        "history_paths": {key: str(value) for key, value in history_paths.items()},
        "learned_hard_sigmoid_v_off": learned_hard_sigmoid_v_off,
        "learned_amplification": learned_amplification,
    }
    metrics_path = run_dir / "metrics.json"
    _write_json(metrics_path, metrics)

    if writer is not None:
        writer.close()

    event_files = sorted(str(path) for path in run_dir.glob("events.out.tfevents.*"))
    metadata = {
        "config_path": str(config_path),
        "config_used_path": str(config_snapshot_path),
        "run_dir": str(run_dir),
        "event_files": event_files,
        "epochs": int(epochs),
        "lr": list(lr) if isinstance(lr, (list, tuple)) else float(lr),
        "lr_decay": lr_decay_value,
        "optimizer": optimizer_details,
        "optimizer_steps_applied": bool(apply_optimizer_steps),
        "beta": beta_value,
        "training_algorithm": training_algorithm,
        "batch_state_policy": batch_state_policy,
        "seed": seed_value,
        "dataset_key": dataset_key,
        "model_key": model_key,
        "dataset_factory": dataset_cfg["factory"],
        "dataset_provenance": dataset_provenance,
        "device": str(device),
        "host": host,
        "timestamp": timestamp,
        "summary": summary,
        "learned_hard_sigmoid_v_off": learned_hard_sigmoid_v_off,
        "learned_amplification": learned_amplification,
    }
    metadata_path = run_dir / "run_metadata.json"
    _write_json(metadata_path, metadata)

    summary["run_dir"] = str(run_dir)
    summary["event_files"] = event_files
    summary["config_used_path"] = str(config_snapshot_path)
    summary["run_metadata_path"] = str(metadata_path)
    summary["metrics_path"] = str(metrics_path)
    summary["final_model_path"] = str(final_model_path)
    summary["best_model_path"] = str(best_model_path)
    summary["weights_final_path"] = str(weights_final_path)
    summary["weights_best_path"] = str(weights_best_path)
    history["summary"] = summary
    history["run_dir"] = str(run_dir)
    history["event_files"] = event_files
    return history





def train_mnist_conv(
    config_path,
    epochs,
    lr,
    beta,
    log_interval,
    max_batches,
    max_test_batches=None,
    device=None,
    sanity_check=False,
    epoch_callback=None,
    dataset_key="mnist",
    output_dir=None,
    model_key="mnist",
    training_algorithm=None,
    seed=None,
    lr_decay=None,
    init_checkpoint_path=None,
    batch_state_policy=None,
    optimizer_name=None,
    momentum=None,
    weight_decay=None,
    gradient_callback=None,
    optimizer_step_callback=None,
    apply_optimizer_steps=True,
    reporting_run_dir=None,
    skip_terminal_official_test=False,
):
    return _train_image_task(
        config_path=config_path,
        epochs=epochs,
        lr=lr,
        beta=beta,
        log_interval=log_interval,
        max_batches=max_batches,
        max_test_batches=max_test_batches,
        dataset_key=dataset_key,
        model_key=model_key,
        image_shape=(28, 28),
        device=device,
        sanity_check=sanity_check,
        epoch_callback=epoch_callback,
        output_dir=output_dir,
        training_algorithm=training_algorithm,
        seed=seed,
        lr_decay=lr_decay,
        init_checkpoint_path=init_checkpoint_path,
        batch_state_policy=batch_state_policy,
        optimizer_name=optimizer_name,
        momentum=momentum,
        weight_decay=weight_decay,
        gradient_callback=gradient_callback,
        optimizer_step_callback=optimizer_step_callback,
        apply_optimizer_steps=apply_optimizer_steps,
        reporting_run_dir=reporting_run_dir,
        skip_terminal_official_test=skip_terminal_official_test,
    )


def train_tiny_grid(
    config_path,
    epochs,
    lr,
    beta,
    log_interval,
    max_batches,
    max_test_batches=None,
    device=None,
    sanity_check=False,
    epoch_callback=None,
    output_dir=None,
    training_algorithm=None,
    seed=None,
    lr_decay=None,
    init_checkpoint_path=None,
    batch_state_policy=None,
    reporting_run_dir=None,
    skip_terminal_official_test=False,
):
    return _train_image_task(
        config_path=config_path,
        epochs=epochs,
        lr=lr,
        beta=beta,
        log_interval=log_interval,
        max_batches=max_batches,
        max_test_batches=max_test_batches,
        dataset_key="tiny3x3",
        model_key="tiny3x3",
        image_shape=(3, 3),
        device=device,
        sanity_check=sanity_check,
        epoch_callback=epoch_callback,
        output_dir=output_dir,
        training_algorithm=training_algorithm,
        seed=seed,
        lr_decay=lr_decay,
        init_checkpoint_path=init_checkpoint_path,
        batch_state_policy=batch_state_policy,
        reporting_run_dir=reporting_run_dir,
        skip_terminal_official_test=skip_terminal_official_test,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description="Train resistive MNIST model via equilibrium propagation.")
    parser.add_argument("--config", required=True, help="Path to configuration JSON.")
    parser.add_argument(
        "--epochs",
        type=int,
        help="Number of training epochs. Overrides config if provided.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        help="Learning rate for SGD updates. Overrides config if provided.",
    )
    parser.add_argument(
        "--beta",
        type=float,
        help="Nudging strength for equilibrium propagation. Overrides config if provided.",
    )
    parser.add_argument(
        "--lr-decay",
        type=float,
        help="Multiplicative learning-rate decay applied after each epoch. Overrides config if provided.",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        help="Logging interval in batches. Overrides config if provided.",
    )
    parser.add_argument(
        "--max-batches",
        type=int,
        help="Optional cap on number of batches per epoch. Overrides config if provided.",
    )
    parser.add_argument(
        "--max-test-batches",
        "--max-validation-batches",
        dest="max_test_batches",
        type=int,
        help="Optional cap on validation batches. Overrides config if provided.",
    )
    parser.add_argument(
        "--device",
        type=str,
        help="Torch device string to use, e.g. 'cpu', 'cuda', or 'cuda:0'. Overrides config if provided.",
    )
    parser.add_argument(
        "--sanity-check",
        action="store_true",
        help="Run internal single-conv gradient sanity check before training.",
    )
    parser.add_argument(
        "--result-json",
        type=str,
        default=None,
        help="Optional path to write a JSON summary (e.g., for Optuna sweeps).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Optional directory for run artifacts such as config.used.json, run_metadata.json, and TensorBoard events.",
    )
    parser.add_argument(
        "--reporting-run-dir",
        type=str,
        default=None,
        help="Canonical run directory whose metrics.jsonl and status.json are updated.",
    )
    parser.add_argument(
        "--init-checkpoint",
        type=str,
        default=None,
        help="Optional DRN parameter checkpoint to load before training.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Dataset override for MNIST-style image runs. Use 'mnist' or 'fmnist'.",
    )
    parser.add_argument(
        "--training-algorithm",
        choices=("EP", "BP", "ep", "bp"),
        default=None,
        help="Training algorithm. Defaults to config['training_algorithm'] or EP.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for initialization and dataloader order.",
    )
    parser.add_argument(
        "--skip-terminal-official-test",
        action="store_true",
        help="Skip a configured terminal official-test evaluation (smoke runs only).",
    )

    args = parser.parse_args(argv)
    config_path = Path(args.config).expanduser().resolve()
    config = load_config(config_path)
    lab_cfg = config.get("lab", {})

    epochs = args.epochs if args.epochs is not None else lab_cfg.get("epochs")
    if epochs is None:
        raise ValueError("Epochs must be provided via --epochs or config['lab']['epochs'].")
    epochs = int(epochs)

    if args.lr is not None:
        lr = args.lr
    else:
        lr = config.get("lr")
        if lr is None and isinstance(config.get("optimizer"), dict):
            lr = config["optimizer"].get("learning_rate")

    if lr is None:
        raise ValueError("Learning rate must be provided via --lr or config['lr'].")

    if isinstance(lr, list):
        learning_rate_cfg = [float(x) for x in lr]
    else:
        learning_rate_cfg = float(lr)

    beta = args.beta if args.beta is not None else config.get("beta")
    if beta is not None:
        beta = float(beta)

    lr_decay = args.lr_decay
    if lr_decay is None:
        optimizer_cfg = config.get("optimizer", {})
        lr_decay = optimizer_cfg.get("lr_decay", config.get("lr_decay", 1.0))
    lr_decay = float(lr_decay)

    log_interval = args.log_interval if args.log_interval is not None else config.get("log_interval")
    if log_interval is None:
        raise ValueError("Log interval must be provided via --log-interval or config['log_interval'].")
    log_interval = int(log_interval)

    max_batches = args.max_batches if args.max_batches is not None else config.get("max_batches")
    if max_batches is not None:
        max_batches = int(max_batches)
    max_test_batches = (
        args.max_test_batches
        if args.max_test_batches is not None
        else config.get("max_test_batches")
    )
    if max_test_batches is not None:
        max_test_batches = int(max_test_batches)

    model_key = lab_cfg.get("model_key", "mnist")
    default_dataset_key = lab_cfg.get("dataset_key")
    if default_dataset_key is None:
        default_dataset_key = "tiny3x3" if model_key == "tiny3x3" else "mnist"
    dataset_key = _normalize_dataset_key(
        args.dataset if args.dataset is not None else default_dataset_key
    )

    if model_key == "tiny3x3":
        train_fn = train_tiny_grid
        if dataset_key != "tiny3x3":
            raise ValueError(
                f"Model {model_key!r} only supports dataset 'tiny3x3', got {dataset_key!r}."
            )
    else:
        train_fn = train_mnist_conv
        if dataset_key not in ("mnist", "fmnist"):
            raise ValueError(
                f"Unsupported dataset {dataset_key!r}. Use 'mnist' or 'fmnist'."
            )

    train_kwargs = dict(
        config_path=config_path,
        epochs=epochs,
        lr=learning_rate_cfg,
        beta=beta,
        log_interval=log_interval,
        max_batches=max_batches,
        max_test_batches=max_test_batches,
        device=args.device,
        sanity_check=args.sanity_check,
        output_dir=args.output_dir,
        training_algorithm=args.training_algorithm,
        seed=args.seed,
        lr_decay=lr_decay,
        init_checkpoint_path=args.init_checkpoint,
        reporting_run_dir=args.reporting_run_dir,
        skip_terminal_official_test=args.skip_terminal_official_test,
    )
    if model_key != "tiny3x3":
        train_kwargs["dataset_key"] = dataset_key
        train_kwargs["model_key"] = model_key

    history = train_fn(**train_kwargs)
    print("Training history:", history)

    summary = history.get("summary", {})
    print("Final summary:", json.dumps(summary, indent=2))

    if args.result_json:
        result_payload = {
            "config_path": args.config,
            "epochs": epochs,
            "learning_rate": learning_rate_cfg,
            "beta": beta,
            "lr_decay": lr_decay,
            "model_key": lab_cfg.get("model_key", "mnist"),
            "dataset_key": dataset_key,
            "test_error": summary.get("test_error"),
            "final_train_accuracy": summary.get("final_train_accuracy"),
            "final_test_accuracy": summary.get("final_test_accuracy"),
            "best_test_accuracy": summary.get("best_test_accuracy"),
            "run_dir": summary.get("run_dir"),
            "event_files": summary.get("event_files"),
        }
        result_path = Path(args.result_json)
        result_path.parent.mkdir(parents=True, exist_ok=True)
        _write_json(result_path, result_payload)
        print(f"Wrote Optuna summary to {result_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
