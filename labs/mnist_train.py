import argparse
from datetime import datetime
import json
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
PROJECT_ROOT = LABS_DIR.parent / "energy-based-learning"
DEFAULT_IMAGE_RUNS_BASE = LABS_DIR.parent / "simulation_results" / "experiments_labs"

for path in (LABS_DIR, PROJECT_ROOT):
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
from model.function.interaction import scalar_float  # noqa: E402
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
    tensors = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(tensors, (list, tuple)):
        raise ValueError(f"Expected checkpoint to contain a list/tuple of tensors: {checkpoint_path}")
    if len(tensors) != len(param_schema):
        raise ValueError(
            f"Expected {len(param_schema)} tensors in checkpoint {checkpoint_path}; got {len(tensors)}."
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
    arrays["param_shapes_json"] = np.asarray(json.dumps(shapes))
    arrays["metadata_json"] = np.asarray(json.dumps(metadata, sort_keys=True))
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
        adaptive_equilibrium=True,
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
        params.setdefault("normalize_mean", 0.286)
        params.setdefault("normalize_std", 0.353)
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
):
    config_path = Path(config_path).expanduser().resolve()
    config = load_config(config_path)
    seed_value = seed if seed is not None else config.get("seed")
    _set_seed(seed_value)
    _reset_name_counters()
    dataset_key = _normalize_dataset_key(dataset_key)
    run_dir, host, timestamp = _resolve_run_dir(config_path, output_dir)
    config_snapshot_path = run_dir / "config.used.json"
    config_snapshot_path.write_text(json.dumps(config, indent=2))
    training_algorithm = _normalize_training_algorithm(
        training_algorithm if training_algorithm is not None else config.get("training_algorithm")
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
    beta_value = float(beta_value)
    lr_decay_value = lr_decay
    if lr_decay_value is None:
        optimizer_cfg = config.get("optimizer", {})
        lr_decay_value = optimizer_cfg.get("lr_decay", config.get("lr_decay", 1.0))
    lr_decay_value = 1.0 if lr_decay_value is None else float(lr_decay_value)
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
    loader_result = dataset_factory(**dataset_params).build()
    if isinstance(loader_result, tuple):
        train_loader, test_loader = loader_result
    else:
        train_loader = loader_result
        test_loader = None

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
        learning_rates = list(lr)
    else:
        learning_rates = [lr] * (len(energy_params) + len(cost_params))
    optimizer = Optimizer(energy_fn, cost_fn, learning_rates, momentum=0.0, weight_decay=0.0)

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
            "name": "SGD",
            "learning_rate": list(learning_rates),
            "lr_decay": lr_decay_value,
            "momentum": 0.0,
            "weight_decay": 0.0,
        },
        "training": {
            "epochs": int(epochs),
            "batch_size": int(dataset_params.get("batch_size")),
            "max_batches": max_batches,
            "max_test_batches": max_test_batches,
            "num_iterations_training": training_iterations,
            "num_iterations_inference": inference_iterations,
            "beta": beta_value,
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
    config_json_path.write_text(json.dumps(_json_sanitize(resolved_run_config), indent=2))

    best_test_accuracy = float("-inf")
    best_epoch = None
    best_model_path = run_dir / "best_model.pt"
    stop_training = False
    for epoch in range(epochs):
        running_loss = 0.0
        running_correct = 0
        seen = 0
        reset_flag = True
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

            network.set_input(images, reset=reset_flag)
            reset_flag = False
            minimizer_inference.compute_equilibrium()
            cost_fn.set_target(labels)

            batch_loss = cost_fn.eval().mean().item()
            errors = cost_fn.error_fn()
            batch_correct = int((~errors).sum().item())

            running_loss += batch_loss * images.size(0)
            running_correct += batch_correct
            seen += images.size(0)

            grads = estimator.compute_gradient()
            for param, grad in zip(params, grads[:len(params)]):
                param.state.grad = grad
            optimizer.step()
            for param in energy_params:
                param.clamp_()

            acc = running_correct / seen
            avg_loss = running_loss / seen
            if _should_log_batch(batch_idx, train_batches_this_epoch, log_interval):
                print(
                    f"Epoch {epoch+1}/{epochs} | Batch {batch_idx+1}/{train_total_batches} | "
                    f"running loss={avg_loss:.4f} acc={acc*100:.2f}%",
                    flush=True,
                )

        if seen:
            epoch_train_loss = running_loss / seen
            epoch_train_acc = running_correct / seen
        else:
            epoch_train_loss = float("nan")
            epoch_train_acc = float("nan")
        history["loss"].append(epoch_train_loss)
        history["accuracy"].append(epoch_train_acc)

        if test_loader is not None:
            test_running_loss = 0.0
            test_correct = 0
            test_seen = 0
            optimizer.zero_grad()
            test_total_batches = len(test_loader)
            test_batches_this_epoch = (
                min(test_total_batches, int(max_test_batches))
                if max_test_batches is not None
                else test_total_batches
            )
            for test_batch_idx, (images, labels) in enumerate(test_loader):
                if max_test_batches is not None and test_batch_idx >= max_test_batches:
                    break
                images = images.to(device)
                labels = labels.to(device)
                network.set_input(images, reset=True)
                minimizer_inference.compute_equilibrium()
                cost_fn.set_target(labels)
                batch_loss = cost_fn.eval().mean().item()
                errors = cost_fn.error_fn()
                batch_correct = int((~errors).sum().item())
                test_running_loss += batch_loss * images.size(0)
                test_correct += batch_correct
                test_seen += images.size(0)

                avg_loss = test_running_loss / test_seen
                acc = test_correct / test_seen
                if _should_log_batch(test_batch_idx, test_batches_this_epoch, log_interval):
                    print(
                        f"[Test] Epoch {epoch+1}/{epochs} | Batch {test_batch_idx+1}/{test_total_batches} | running loss={avg_loss:.4f} acc={acc*100:.2f}%",
                        flush=True,
                    )
            epoch_test_loss = test_running_loss / test_seen if test_seen else float("nan")
            epoch_test_acc = test_correct / test_seen if test_seen else float("nan")
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

        if epoch_callback:
            epoch_info = {
                "epoch": epoch + 1,
                "train_loss": epoch_train_loss,
                "train_accuracy": epoch_train_acc,
                "test_loss": epoch_test_loss,
                "test_accuracy": epoch_test_acc,
            }
            should_stop = epoch_callback(epoch_info, history)
            if should_stop:
                stop_training = True
                break

        best_candidate = epoch_test_acc if epoch_test_acc is not None else epoch_train_acc
        if best_candidate is not None and np.isfinite(best_candidate) and best_candidate > best_test_accuracy:
            best_test_accuracy = float(best_candidate)
            best_epoch = epoch + 1
            energy_fn.save(best_model_path)

        history["learning_rate"].append([float(group["lr"]) for group in optimizer.param_groups])
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
        summary["best_test_accuracy"] = max(history["test_accuracy"])
        summary["best_epoch"] = int(np.argmax(history["test_accuracy"]) + 1)
        summary["final_test_loss"] = history["test_loss"][-1]
        summary["test_error"] = 1.0 - summary["final_test_accuracy"]
    else:
        summary["best_epoch"] = best_epoch
        summary["test_error"] = 1.0 - summary["final_train_accuracy"]

    final_model_path = run_dir / "final_model.pt"
    energy_fn.save(final_model_path)
    if not best_model_path.exists():
        best_model_path = run_dir / "best_model.pt"
        energy_fn.save(best_model_path)
        best_epoch = len(history["accuracy"]) if history["accuracy"] else None
        if history["test_accuracy"]:
            best_test_accuracy = history["test_accuracy"][-1]
        elif history["accuracy"]:
            best_test_accuracy = history["accuracy"][-1]

    weights_metadata = {
        "voltage_amp": float(model_cfg["voltage_amp"]),
        "current_amp": float(model_cfg["current_amp"]),
        "seed": seed_value,
        "layer_shapes": [list(shape) for shape in layer_shapes],
        "non_linearity": model_cfg["non_linearity"],
        "training_algorithm": training_algorithm,
        "checkpoint_format": "torch list of DRN parameter tensors in param_names order",
    }
    weights_final_path = run_dir / "weights_final.npz"
    weights_best_path = run_dir / "weights_best.npz"
    _checkpoint_to_npz(final_model_path, weights_final_path, param_schema, weights_metadata)
    _checkpoint_to_npz(best_model_path, weights_best_path, param_schema, weights_metadata)
    history_paths = _save_history_arrays(run_dir, history)
    learned_hard_sigmoid_v_off = _hard_sigmoid_v_off_values(energy_fn)
    learned_amplification = _amplification_values(energy_fn)

    metrics = {
        "run_dir": str(run_dir),
        "training_algorithm": training_algorithm,
        "seed": seed_value,
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
        "lr_decay": lr_decay_value,
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
    metrics_path.write_text(json.dumps(metrics, indent=2))

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
        "beta": beta_value,
        "training_algorithm": training_algorithm,
        "seed": seed_value,
        "dataset_key": dataset_key,
        "model_key": model_key,
        "dataset_factory": dataset_cfg["factory"],
        "device": str(device),
        "host": host,
        "timestamp": timestamp,
        "summary": summary,
        "learned_hard_sigmoid_v_off": learned_hard_sigmoid_v_off,
        "learned_amplification": learned_amplification,
    }
    metadata_path = run_dir / "run_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2))

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
    )


def train_tiny_grid(
    config_path,
    epochs,
    lr,
    beta,
    log_interval,
    max_batches,
    device=None,
    sanity_check=False,
    epoch_callback=None,
    output_dir=None,
    training_algorithm=None,
    seed=None,
    lr_decay=None,
    init_checkpoint_path=None,
):
    return _train_image_task(
        config_path=config_path,
        epochs=epochs,
        lr=lr,
        beta=beta,
        log_interval=log_interval,
        max_batches=max_batches,
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
    )


def main():
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

    args = parser.parse_args()
    config = load_config(args.config)
    lab_cfg = config.get("lab", {})

    epochs = args.epochs if args.epochs is not None else lab_cfg.get("epochs")
    if epochs is None:
        raise ValueError("Epochs must be provided via --epochs or config['lab']['epochs'].")
    epochs = int(epochs)

    if args.lr is not None:
        lr = args.lr
    else:
        lr = config.get("lr")

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
        config_path=args.config,
        epochs=epochs,
        lr=learning_rate_cfg,
        beta=beta,
        log_interval=log_interval,
        max_batches=max_batches,
        device=args.device,
        sanity_check=args.sanity_check,
        output_dir=args.output_dir,
        training_algorithm=args.training_algorithm,
        seed=args.seed,
        lr_decay=lr_decay,
        init_checkpoint_path=args.init_checkpoint,
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
        result_path.write_text(json.dumps(result_payload, indent=2))
        print(f"Wrote Optuna summary to {result_path}")


if __name__ == "__main__":
    main()
