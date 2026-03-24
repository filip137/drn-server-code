import json
import os
import resource
import shutil
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional, Sequence

import numpy as np
import torch

LABS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = LABS_DIR.parent
for path in (LABS_DIR, PROJECT_ROOT):
    if str(path) not in sys.path:
        sys.path.append(str(path))

from labs.datasets import DigitsDataset, LinSpaceDataset, MoonsDataset, YinYangDataset  # noqa: E402
from labs.common import export_pt_to_npz, CustomEvaluator, CustomTrainer  # noqa: E402
from model.resistive.network import DeepResistiveEnergy  # noqa: E402
from model.function.network import Network  # noqa: E402
from labs.custom_minimizer import (  # noqa: E402
    CustomQuadraticMinimizer,
    CustomAnderssonMinimizer,
    MinimizerSettings,
)
from training.monitor import Optimizer  # noqa: E402
from training.sgd import AugmentedFunction, Backprop, EquilibriumProp  # noqa: E402
from model.function.cost import SquaredError, SquaredErrorPairedOutputs  # noqa: E402
from model.variable.parameter import Bias  # noqa: E402
from training.statistics import (  # noqa: E402
    Counter,
    EnergyStat,
    CostStat,
    ErrorStat,
    TopFiveErrorStat,
    WeightColumnSumStat,
    WeightDistributionStat,
    WeightRowSumStat,
)

from small_network_config import _set_seed, _parse_layer_shapes  # noqa: E402

try:  # Optional TensorBoard support
    from torch.utils.tensorboard import SummaryWriter
except Exception:
    SummaryWriter = None


def _resolve_double_diode_runtime(double_diode_runtime: str) -> tuple[str, bool]:
    if double_diode_runtime == "float64_fixed":
        return "float64_timed", False
    if double_diode_runtime == "float64_experimental":
        return "float64_experimental", False
    if double_diode_runtime == "float32_adaptive":
        return "float32", True
    if double_diode_runtime == "float32_overrelaxed":
        return "overrelaxed", True
    raise SystemExit(
        "double_diode_runtime must be one of: float32_adaptive, float64_fixed, "
        "float64_experimental, float32_overrelaxed."
    )


def _resolve_minimizer_impl(minimizer_settings_data):
    minimizer_impl = minimizer_settings_data.minimizer_impl
    minimizer_cls = (
        CustomAnderssonMinimizer
        if minimizer_impl == "andersson"
        else CustomQuadraticMinimizer
    )
    extra_args = {}
    if minimizer_cls is CustomAnderssonMinimizer:
        extra_args = dict(
            anderson_m=minimizer_settings_data.anderson_m,
            anderson_omega=minimizer_settings_data.anderson_omega,
            anderson_tol_floor=minimizer_settings_data.anderson_tol_floor,
            anderson_reg=minimizer_settings_data.anderson_reg,
        )
    return minimizer_cls, extra_args


def _skip_layer_for_residual_comparison(layer_name: str) -> bool:
    normalized = str(layer_name).strip().lower().replace("-", "").replace("_", "").replace(" ", "")
    return normalized == "layer0"


def _collect_double_diode_timing_stats(energy_minimizer, reset=False):
    aggregated = None
    for updater in getattr(energy_minimizer, "_updaters", []):
        timing_stats = getattr(updater, "timing_stats", None)
        if not callable(timing_stats):
            continue
        stats = timing_stats(reset=reset)
        if not isinstance(stats, dict):
            continue
        if aggregated is None:
            aggregated = {
                "calls": 0,
                "totals_seconds": {},
                "call_counts": {},
            }
        aggregated["calls"] += int(stats.get("calls", 0))
        for key, value in stats.get("totals_seconds", {}).items():
            aggregated["totals_seconds"][key] = aggregated["totals_seconds"].get(key, 0.0) + float(value)
        for key, value in stats.get("call_counts", {}).items():
            aggregated["call_counts"][key] = aggregated["call_counts"].get(key, 0) + int(value)
    return aggregated


@dataclass(frozen=True)
class LinspaceSettings:
    linspace_min: float
    linspace_max: float
    linspace_samples: int


@dataclass(frozen=True)
class NonLinearityData:
    non_linearity: str
    quadratic_diode_param: Optional[dict]
    exponential_diode_param: Optional[dict]
    hard_sigmoid_param: Optional[dict]
    iv_data_path: Optional[str] = None


@dataclass(frozen=True)
class MinimizerRuntimeSettings:
    exp_clip: float
    double_diode_updater: Optional[str]
    adaptive_equilibrium: bool
    overrelaxation_factor: float
    overrelaxation_reject_steps: bool
    overrelaxation_reject_max_tries: int
    overrelaxation_reject_shrink: float
    overrelaxation_reject_eps: float
    single_diode_updater: Optional[str]
    rel_tol: float
    vn_tol: float
    residual_current_tol: Optional[float]
    use_polish: bool
    max_newton_iters: int
    z_thresh: float
    dynamic_polish: bool
    minimizer_impl: str
    anderson_m: int
    anderson_omega: float
    anderson_tol_floor: float
    anderson_reg: float
    damping: float
    experimental_newton_max_steps: int = 100
    experimental_exponential_newton_tol_progressive: bool = True
    experimental_exponential_newton_tol_start: float = 1e-5
    experimental_exponential_newton_tol_end: float = 1e-5
    experimental_exponential_newton_tol_switch_hi: float = 1e-2
    experimental_exponential_newton_tol_switch_lo: float = 5e-4
    double_diode_runtime: Optional[str] = None


@dataclass(frozen=True)
class LinspaceRunSettings:
    dims: Sequence[int]
    batch_size: int
    num_iterations: int
    num_epochs: int
    seed: Optional[int]
    device: Optional[str]
    voltage_amp: float
    current_amp: float
    input_gain: Optional[float]
    weight_gains: Optional[Sequence[float]]
    weight_min: Optional[float]
    weight_max: Optional[float]
    dataset_name: str
    linspace_settings: LinspaceSettings
    non_linearity_data: NonLinearityData
    minimizer_settings: MinimizerRuntimeSettings


@dataclass
class ValidateRunSettings:
    dims: Sequence[int]
    batch_size: int
    num_iterations: int
    num_epochs: Optional[int]
    num_points: int
    seed: Optional[int]
    device: Optional[str]
    voltage_amp: float
    current_amp: float
    linspace_min: Optional[float]
    linspace_max: Optional[float]
    linspace_samples: Optional[int]
    input_gain: Optional[float]
    weight_gains: Optional[Sequence[float]]
    weight_min: Optional[float]
    weight_max: Optional[float]
    dataset_name: str
    non_linearity_data: NonLinearityData
    minimizer_settings: MinimizerRuntimeSettings


@dataclass(frozen=True)
class TrainRunSettings:
    dims: Sequence[int]
    num_points: int
    batch_size: int
    num_iterations: int
    num_epochs: int
    seed: Optional[int]
    device: Optional[str]
    voltage_amp: float
    current_amp: float
    input_gain: Optional[float]
    weight_gains: Optional[Sequence[float]]
    weight_min: Optional[float]
    weight_max: Optional[float]
    dataset_name: str
    training_algorithm: str
    learning_rate: Optional[Sequence[float]]
    nudging: float
    non_linearity_data: NonLinearityData
    minimizer_settings: MinimizerRuntimeSettings
    early_stop_min_epochs: Optional[int] = None
    early_stop_error_pct: Optional[float] = None
    use_tensorboard: bool = False
    tensorboard_dir: Optional[str] = None
    save_epochs: Optional[Sequence[int]] = None
    save_every_epoch: bool = False
    save_test_accuracy_threshold: Optional[float] = None
    save_test_accuracy_thresholds: Optional[Sequence[float]] = None


def _format_accuracy_tag(value: float) -> str:
    return f"{value:.4f}".rstrip("0").rstrip(".").replace(".", "p")


def _resolve_torch_device(requested_device: Optional[str]) -> torch.device:
    if requested_device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        device = torch.device(requested_device)
    except (TypeError, ValueError) as exc:
        raise SystemExit(
            "Expected device to be a valid torch device string such as 'cpu', 'cuda', or 'cuda:0'. "
            f"Provided value: {requested_device!r}."
        ) from exc
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit(
                "Expected CUDA to be available for device selection. "
                f"Provided value: {requested_device!r}."
            )
        if device.index is not None and device.index >= torch.cuda.device_count():
            raise SystemExit(
                "Expected CUDA device index to be within available range. "
                f"Provided value: {requested_device!r}."
            )
    return device


def _resolve_accuracy_thresholds(settings: TrainRunSettings) -> list[float]:
    thresholds = []
    if settings.save_test_accuracy_thresholds is not None:
        thresholds.extend(float(value) for value in settings.save_test_accuracy_thresholds)
    elif settings.save_test_accuracy_threshold is not None:
        thresholds.append(float(settings.save_test_accuracy_threshold))
    return sorted(set(thresholds))


def _create_run_dir(
    output_root: str,
    run_subdir: Optional[str],
    non_linearity: str,
    linspace_only: bool,
) -> Path:
    output_root_path = Path(output_root).expanduser()
    output_root_path.mkdir(parents=True, exist_ok=True)
    suffix = "_linspace" if linspace_only else ""
    run_name = datetime.now().strftime("%Y%m%d-%H%M%S") + f"_{non_linearity}{suffix}"
    if run_subdir is None:
        run_parent = output_root_path
    else:
        run_subdir_path = Path(run_subdir)
        run_subdir_parts = run_subdir_path.parts
        root_parts = output_root_path.parts
        already_in_root = (
            len(run_subdir_parts) > 0
            and len(root_parts) >= len(run_subdir_parts)
            and root_parts[-len(run_subdir_parts):] == run_subdir_parts
        )
        run_parent = output_root_path if already_in_root else output_root_path / run_subdir_path
    run_dir = run_parent / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _find_latest_model(weights_root: Path, hidden_count: int) -> Path:
    candidates = sorted(
        (weights_root / f"hidden_{hidden_count}").glob("*/model.pt"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(f"No model.pt found under {weights_root}/hidden_{hidden_count}")
    return candidates[0]


def _build_energy_stack(
    *,
    device: torch.device,
    input_dim: int,
    hidden_dims: Sequence[int],
    output_dim: int,
    non_linearity: str,
    voltage_amp: float,
    current_amp: float,
    quadratic_diode_param: Optional[dict],
    exponential_diode_param: Optional[dict],
    hard_sigmoid_param: Optional[dict],
    input_gain: Optional[float],
    weights_path: Optional[str],
    weight_gains: Optional[Sequence[float]],
    weight_min: Optional[float],
    weight_max: Optional[float],
    dataset_name: str = "moons",
):
    if not non_linearity:
        raise SystemExit("non_linearity must be provided via config.")
    layer_shapes = [(input_dim * 2,)] + [(dim,) for dim in hidden_dims] + [(output_dim,)]
    expected = len(layer_shapes) - 1
    if not isinstance(weight_gains, (list, tuple)):
        raise SystemExit(f"weight_gains must be a list with {expected} entries.")
    resolved_weight_gains = list(weight_gains)
    if len(resolved_weight_gains) != expected:
        raise SystemExit(f"weight_gains must have {expected} entries.")
    if input_gain is None:
        raise SystemExit("input_gain must be provided explicitly.")
    if quadratic_diode_param is None or exponential_diode_param is None or hard_sigmoid_param is None:
        raise SystemExit(
            "Expected all diode parameter dicts (quadratic_diode_param, exponential_diode_param, "
            "hard_sigmoid_param) to be provided explicitly."
        )
    quadratic_params = dict(quadratic_diode_param)
    exponential_params = dict(exponential_diode_param)
    hard_sigmoid_params = dict(hard_sigmoid_param)
    if weight_min is None or weight_max is None:
        raise SystemExit("weight_min and weight_max must be provided explicitly.")

    energy_fn = DeepResistiveEnergy(
        layer_shapes=layer_shapes,
        weight_gains=resolved_weight_gains,
        input_gain=input_gain,
        non_linearity=non_linearity,
        exponential_diode_param=exponential_params,
        quadratic_diode_param=quadratic_params,
        hard_sigmoid_param=hard_sigmoid_params,
        voltage_amp=voltage_amp,
        current_amp=current_amp,
        weight_min=weight_min,
        weight_max=weight_max,
    )
    energy_fn.set_device(device)
    if weights_path:
        energy_fn.load(Path(weights_path).expanduser())

    network = Network(energy_fn)
    free_layers = network.free_layers()
    output_layer = energy_fn.layers()[-1]
    if dataset_name == "moons":
        num_classes = 2
    elif dataset_name == "yinyang":
        num_classes = 3
    elif dataset_name == "digits":
        num_classes = 10
    else:
        raise ValueError(f"Unsupported dataset '{dataset_name}'.")
    output_width = int(output_layer.shape[0])
    if output_width == num_classes:
        cost_fn = SquaredError(output_layer)
    elif output_width == 2 * num_classes:
        cost_fn = SquaredErrorPairedOutputs(output_layer, num_classes)
    else:
        raise ValueError(
            f"Unsupported output size {output_layer.shape} for dataset '{dataset_name}'. "
            f"Expected {num_classes} (standard) or {2 * num_classes} (differential)."
        )

    return (
        energy_fn,
        network,
        free_layers,
        cost_fn,
        output_layer,
        layer_shapes,
        input_gain,
        quadratic_params,
        exponential_params,
        hard_sigmoid_params,
    )


def _attach_stats(trainer, evaluator, energy_fn, cost_fn, train_loader, test_loader, output_dim):
    train_counter = Counter(energy_fn, len(train_loader.dataset))
    train_counter_post = Counter(energy_fn, len(train_loader.dataset))
    train_counter_post.display = False
    eval_counter = Counter(energy_fn, len(test_loader.dataset))

    core_stats = [EnergyStat(energy_fn), CostStat(cost_fn), ErrorStat(cost_fn)]
    if output_dim >= 5:
        core_stats.append(TopFiveErrorStat(cost_fn))

    trainer.add_statistic(train_counter, list_idx=0)
    trainer.add_statistic(train_counter_post, list_idx=1)
    for stat in core_stats:
        trainer.add_statistic(stat, list_idx=0)
        trainer.add_statistic(stat, list_idx=1)

    # Weight statistics (train only)
    weight_params = [param for param in energy_fn.params() if "Weight" in param.name]
    if weight_params:
        for param in weight_params:
            trainer.add_statistic(WeightRowSumStat(param), list_idx=0)
            trainer.add_statistic(WeightColumnSumStat(param), list_idx=0)
        for stat_type in ["mean", "std", "min", "max", "abs_mean", "abs_std"]:
            for param in weight_params:
                trainer.add_statistic(WeightDistributionStat(param, stat_type), list_idx=0)

    evaluator.add_statistic(eval_counter)
    for stat in core_stats:
        evaluator.add_statistic(stat)


def _report_non_finite_params(params, label: str) -> None:
    for param in params:
        state = param.state
        if torch.isfinite(state).all():
            continue
        nan_count = torch.isnan(state).sum().item()
        inf_count = torch.isinf(state).sum().item()
        name = getattr(param, "name", param.__class__.__name__)
        print(
            f"[nan-check] {label} {name}: nan={nan_count} inf={inf_count} "
            f"shape={tuple(state.shape)} dtype={state.dtype} device={state.device}"
        )
        break


def _get_stat_value(stats, name: str):
    for stat in stats:
        if getattr(stat, "name", None) == name:
            return stat.get()
    return None


def _write_tensorboard_stats(writer, stats, prefix: str, epoch: int) -> None:
    if writer is None:
        return
    for stat in stats:
        name = getattr(stat, "name", None)
        if not name:
            continue
        tag = f"{name}/{prefix}"
        option = getattr(stat, "option", None)
        if option:
            tag = f"{tag}_{option}"
        value = stat.get()
        if torch.is_tensor(value):
            value = value.item()
        writer.add_scalar(tag, value, epoch)


def _compute_accuracy(*, network, cost_fn, energy_minimizer, dataloader) -> Optional[float]:
    total = 0
    correct = 0
    for batch in dataloader:
        if isinstance(batch, (list, tuple)):
            if len(batch) >= 2:
                x, y = batch[0], batch[1]
            elif len(batch) == 1:
                x, y = batch[0], None
            else:
                continue
        else:
            x, y = batch, None

        network.set_input(x, reset=True)
        energy_minimizer.compute_equilibrium()
        if y is None:
            continue
        cost_fn.set_target(y)
        errors = cost_fn.error_fn()
        batch_total = int(errors.numel())
        batch_correct = batch_total - int(errors.sum().item())
        total += batch_total
        correct += batch_correct

    if total == 0:
        return None
    return correct / total


def train(
    *,
    settings: TrainRunSettings,
    output_root: str,
    run_subdir: Optional[str],
    weights_path: Optional[str],
    return_metrics: bool = False,
    epoch_callback: Optional[Callable[[int, dict], None]] = None,
) -> tuple[Path, Path, Path] | tuple[Path, Path, Path, dict]:
    _set_seed(settings.seed)

    device = _resolve_torch_device(settings.device)
    input_dim, hidden_dims, output_dim, _ = _parse_layer_shapes(settings.dims)
    if settings.dataset_name == "moons":
        dataset = MoonsDataset(
            name="moons",
            batch_size=settings.batch_size,
            device=device,
            num_samples=settings.num_points,
            input_dim=input_dim,
        )
    elif settings.dataset_name == "yinyang":
        dataset = YinYangDataset(
            name="yinyang",
            batch_size=settings.batch_size,
            device=device,
            num_samples=settings.num_points,
        )
    elif settings.dataset_name == "digits":
        dataset = DigitsDataset(
            name="digits",
            batch_size=settings.batch_size,
            device=device,
            num_samples=settings.num_points,
            seed=settings.seed or 0,
        )
    else:
        raise SystemExit(
            f"Unsupported dataset '{settings.dataset_name}'. Expected 'moons', 'yinyang', or 'digits'."
        )
    train_loader, test_loader = dataset.build()
    minimizer_settings_data = settings.minimizer_settings
    non_linearity_data = settings.non_linearity_data

    (
        energy_fn,
        network,
        free_layers,
        cost_fn,
        output_layer,
        _layer_shapes,
        _input_gain,
        quadratic_params,
        exponential_params,
        hard_sigmoid_params,
    ) = _build_energy_stack(
        device=device,
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        output_dim=output_dim,
        non_linearity=non_linearity_data.non_linearity,
        voltage_amp=settings.voltage_amp,
        current_amp=settings.current_amp,
        quadratic_diode_param=non_linearity_data.quadratic_diode_param,
        exponential_diode_param=non_linearity_data.exponential_diode_param,
        hard_sigmoid_param=non_linearity_data.hard_sigmoid_param,
        input_gain=settings.input_gain,
        weights_path=weights_path,
        weight_gains=settings.weight_gains,
        weight_min=settings.weight_min,
        weight_max=settings.weight_max,
        dataset_name=settings.dataset_name,
    )

    training_algorithm = settings.training_algorithm
    if training_algorithm is None:
        training_algorithm = "EP"
    training_algorithm = str(training_algorithm).upper()
    if training_algorithm not in ("EP", "BP"):
        raise SystemExit(
            "training_algorithm must be 'EP' or 'BP'. "
            f"Got {settings.training_algorithm!r}."
        )

    if training_algorithm == "EP":
        augmented_fn = AugmentedFunction(energy_fn, cost_fn)
        training_fn = augmented_fn
    else:
        augmented_fn = None
        training_fn = energy_fn

    minimizer_mode = "asynchronous"
    minimizer_settings = MinimizerSettings(
        rel_tol=minimizer_settings_data.rel_tol,
        vn_tol=minimizer_settings_data.vn_tol,
        use_polish=minimizer_settings_data.use_polish,
        max_newton_iters=minimizer_settings_data.max_newton_iters,
        z_thresh=minimizer_settings_data.z_thresh,
        exp_clip=minimizer_settings_data.exp_clip,
        dynamic_polish=minimizer_settings_data.dynamic_polish,
        overrelaxation_reject_steps=minimizer_settings_data.overrelaxation_reject_steps,
        overrelaxation_reject_max_tries=minimizer_settings_data.overrelaxation_reject_max_tries,
        overrelaxation_reject_shrink=minimizer_settings_data.overrelaxation_reject_shrink,
        overrelaxation_reject_eps=minimizer_settings_data.overrelaxation_reject_eps,
        experimental_exponential_newton_tol_progressive=(
            minimizer_settings_data.experimental_exponential_newton_tol_progressive
        ),
        experimental_exponential_newton_tol_start=(
            minimizer_settings_data.experimental_exponential_newton_tol_start
        ),
        experimental_exponential_newton_tol_end=(
            minimizer_settings_data.experimental_exponential_newton_tol_end
        ),
        experimental_exponential_newton_tol_switch_hi=(
            minimizer_settings_data.experimental_exponential_newton_tol_switch_hi
        ),
        experimental_exponential_newton_tol_switch_lo=(
            minimizer_settings_data.experimental_exponential_newton_tol_switch_lo
        ),
    )
    double_diode_updater = minimizer_settings_data.double_diode_updater
    adaptive_equilibrium = minimizer_settings_data.adaptive_equilibrium
    minimizer_cls, extra_args = _resolve_minimizer_impl(minimizer_settings_data)
    if training_algorithm == "BP" and minimizer_settings_data.minimizer_impl == "andersson":
        print(
            "[train] warning: training_algorithm=BP with minimizer_impl=andersson "
            "is unvalidated; consider minimizer_impl=custom."
        )
    energy_minimizer_training = minimizer_cls(
        fn=training_fn,
        free_layers=free_layers,
        num_iterations=settings.num_iterations,
        mode=minimizer_mode,
        non_linearity=non_linearity_data.non_linearity,
        quadratic_diode_param=quadratic_params,
        exponential_diode_param=exponential_params,
        voltage_amp=settings.voltage_amp,
        current_amp=settings.current_amp,
        hard_sigmoid_param=hard_sigmoid_params,
        iv_data=None,
        iv_data_path=non_linearity_data.iv_data_path,
        double_diode_updater=double_diode_updater,
        adaptive_equilibrium=adaptive_equilibrium,
        overrelaxation_factor=minimizer_settings_data.overrelaxation_factor,
        single_diode_updater=minimizer_settings_data.single_diode_updater,
        damping=minimizer_settings_data.damping,
        experimental_newton_max_steps=minimizer_settings_data.experimental_newton_max_steps,
        minimizer_settings=minimizer_settings,
        **extra_args,
    )

    energy_minimizer_inference = minimizer_cls(
        fn=energy_fn,
        free_layers=free_layers,
        num_iterations=settings.num_iterations,
        mode=minimizer_mode,
        non_linearity=non_linearity_data.non_linearity,
        quadratic_diode_param=quadratic_params,
        exponential_diode_param=exponential_params,
        voltage_amp=settings.voltage_amp,
        current_amp=settings.current_amp,
        hard_sigmoid_param=hard_sigmoid_params,
        iv_data=None,
        iv_data_path=non_linearity_data.iv_data_path,
        double_diode_updater=double_diode_updater,
        adaptive_equilibrium=adaptive_equilibrium,
        overrelaxation_factor=minimizer_settings_data.overrelaxation_factor,
        single_diode_updater=minimizer_settings_data.single_diode_updater,
        damping=minimizer_settings_data.damping,
        experimental_newton_max_steps=minimizer_settings_data.experimental_newton_max_steps,
        minimizer_settings=minimizer_settings,
        **extra_args,
    )
    if hasattr(energy_minimizer_inference, "reset_equilibrium_iteration_stats"):
        energy_minimizer_inference.reset_equilibrium_iteration_stats()
    if hasattr(energy_minimizer_inference, "reset_overrelaxation_reject_stats"):
        energy_minimizer_inference.reset_overrelaxation_reject_stats()

    params = energy_fn.params()
    if training_algorithm == "EP":
        estimator = EquilibriumProp(
            params,
            free_layers,
            augmented_fn,
            cost_fn,
            energy_minimizer_training,
            variant="centered",
            nudging=settings.nudging,
        )
    else:
        estimator = Backprop(
            params,
            free_layers,
            cost_fn,
            energy_minimizer_training,
        )

    non_bias_params = [param for param in params if not isinstance(param, Bias)]
    if not isinstance(settings.learning_rate, (list, tuple)):
        raise SystemExit(f"learning_rate must be a list with {len(non_bias_params)} entries.")
    learning_rates = list(settings.learning_rate)
    if len(learning_rates) != len(non_bias_params):
        raise SystemExit(f"learning_rate must have {len(non_bias_params)} entries.")
    for idx, param in enumerate(params):
        if isinstance(param, Bias):
            learning_rates.insert(idx, 0.0)
    optimizer = Optimizer(
        energy_fn,
        cost_fn,
        learning_rates,
        momentum=0.0,
        weight_decay=0.0,
    )

    record_statistics = ("store_states",) if os.environ.get("DRN_STORE_STATES") == "1" else ()

    trainer = CustomTrainer(
        network,
        cost_fn,
        params,
        train_loader,
        estimator,
        optimizer,
        energy_minimizer_inference,
        record_statistics=record_statistics,
    )
    evaluator = CustomEvaluator(
        network,
        cost_fn,
        test_loader,
        energy_minimizer_inference,
        record_statistics=record_statistics,
    )
    _attach_stats(trainer, evaluator, energy_fn, cost_fn, train_loader, test_loader, output_layer.shape[0])

    run_dir = _create_run_dir(
        output_root,
        run_subdir,
        non_linearity_data.non_linearity,
        linspace_only=False,
    )

    writer = None
    if settings.use_tensorboard:
        if SummaryWriter is None:
            print("[train] TensorBoard unavailable; install torch.utils.tensorboard to enable events.")
        else:
            tb_dir = (
                Path(settings.tensorboard_dir).expanduser()
                if settings.tensorboard_dir
                else run_dir
            )
            tb_dir.mkdir(parents=True, exist_ok=True)
            writer = SummaryWriter(str(tb_dir))

    track_best = settings.dataset_name == "digits"
    best_cost = None
    best_epoch = None
    best_model_path = run_dir / "model_best.pt"
    accuracy_threshold_paths = {}
    saved_accuracy_thresholds = set()
    for accuracy_threshold in _resolve_accuracy_thresholds(settings):
        threshold_tag = _format_accuracy_tag(accuracy_threshold)
        accuracy_threshold_paths[accuracy_threshold] = {
            "model_path": run_dir / f"model_test_accuracy_{threshold_tag}.pt",
            "model_npz_path": run_dir / f"model_test_accuracy_{threshold_tag}.npz",
            "info_path": run_dir / f"model_test_accuracy_{threshold_tag}.json",
        }
    checkpoint_epochs = None
    if settings.save_epochs:
        try:
            parsed_epochs = {int(epoch) for epoch in settings.save_epochs}
        except (TypeError, ValueError) as exc:
            raise SystemExit("save_epochs must be an int or list of ints.") from exc
        parsed_epochs = {epoch for epoch in parsed_epochs if epoch > 0}
        if parsed_epochs:
            checkpoint_epochs = parsed_epochs

    for epoch in range(1, settings.num_epochs + 1):
        trainer.run(verbose=True)
        _report_non_finite_params(params, f"epoch {epoch} after train")
        if (
            settings.early_stop_min_epochs is not None
            and settings.early_stop_error_pct is not None
            and epoch >= settings.early_stop_min_epochs
        ):
            train_error_pct = _get_stat_value(trainer._stats[0], "Error")
            if train_error_pct is not None and train_error_pct > settings.early_stop_error_pct:
                print(
                    f"[early-stop] epoch={epoch} train_error={train_error_pct:.2f}% "
                    f"> {settings.early_stop_error_pct:.2f}%; stopping."
                )
                break
        evaluator.run(verbose=True)
        _report_non_finite_params(params, f"epoch {epoch} after eval")
        if track_best:
            test_cost = _get_stat_value(evaluator._stats[0], "Cost")
            if test_cost is not None and (best_cost is None or test_cost < best_cost):
                best_cost = test_cost
                best_epoch = epoch
                energy_fn.save(best_model_path)
        if accuracy_threshold_paths:
            test_error_pct = _get_stat_value(evaluator._stats[0], "Error")
            if test_error_pct is not None:
                epoch_test_accuracy = 1.0 - float(test_error_pct) / 100.0
                for accuracy_threshold, threshold_paths in accuracy_threshold_paths.items():
                    if accuracy_threshold in saved_accuracy_thresholds:
                        continue
                    if epoch_test_accuracy < accuracy_threshold:
                        continue
                    threshold_model_path = threshold_paths["model_path"]
                    threshold_model_npz_path = threshold_paths["model_npz_path"]
                    threshold_info_path = threshold_paths["info_path"]
                    energy_fn.save(threshold_model_path)
                    export_pt_to_npz(threshold_model_path, threshold_model_npz_path)
                    threshold_info = {
                        "epoch": epoch,
                        "test_accuracy": epoch_test_accuracy,
                        "threshold": accuracy_threshold,
                    }
                    threshold_info_path.write_text(json.dumps(threshold_info, indent=2))
                    saved_accuracy_thresholds.add(accuracy_threshold)
                    print(
                        "[train] saved threshold checkpoint at "
                        f"epoch={epoch} threshold={accuracy_threshold:.4f} "
                        f"test_accuracy={epoch_test_accuracy:.6f} "
                        f"to {threshold_model_path}"
                    )
        if settings.save_every_epoch or (checkpoint_epochs and epoch in checkpoint_epochs):
            checkpoint_path = run_dir / f"model_epoch_{epoch:04d}.pt"
            energy_fn.save(checkpoint_path)
            print(f"[train] saved checkpoint to {checkpoint_path}")
        if writer is not None:
            _write_tensorboard_stats(writer, trainer._stats[0], "train", epoch)
            _write_tensorboard_stats(writer, evaluator._stats[0], "test", epoch)
            writer.flush()
        if epoch_callback is not None:
            epoch_metrics = {
                "train_cost": _get_stat_value(trainer._stats[0], "Cost"),
                "train_error": _get_stat_value(trainer._stats[0], "Error"),
                "test_cost": _get_stat_value(evaluator._stats[0], "Cost"),
                "test_error": _get_stat_value(evaluator._stats[0], "Error"),
            }
            epoch_callback(epoch, epoch_metrics)
        if hasattr(energy_minimizer_inference, "equilibrium_iteration_stats"):
            energy_minimizer_inference.equilibrium_iteration_stats(reset=True)
        if hasattr(energy_minimizer_training, "equilibrium_iteration_stats"):
            energy_minimizer_training.equilibrium_iteration_stats(reset=True)

    model_path = run_dir / "model.pt"
    if track_best and best_model_path.exists():
        shutil.copy2(best_model_path, model_path)
    else:
        energy_fn.save(model_path)
    model_npz_path = run_dir / "model.npz"
    export_pt_to_npz(model_path, model_npz_path)

    try:
        energy_fn.load(model_path)
    except Exception:
        pass
    train_accuracy = _compute_accuracy(
        network=network,
        cost_fn=cost_fn,
        energy_minimizer=energy_minimizer_inference,
        dataloader=train_loader,
    )
    test_accuracy = _compute_accuracy(
        network=network,
        cost_fn=cost_fn,
        energy_minimizer=energy_minimizer_inference,
        dataloader=test_loader,
    )
    final_accuracy = {
        "train_accuracy": train_accuracy,
        "test_accuracy": test_accuracy,
    }
    if train_accuracy is not None:
        final_accuracy["train_accuracy_pct"] = train_accuracy * 100.0
    if test_accuracy is not None:
        final_accuracy["test_accuracy_pct"] = test_accuracy * 100.0
    final_accuracy_path = run_dir / "final_accuracy.json"
    final_accuracy_path.write_text(json.dumps(final_accuracy, indent=2))
    print(f"[train] saved final accuracy to {final_accuracy_path}")

    if return_metrics:
        if track_best and best_model_path.exists():
            energy_fn.load(model_path)
        train_accuracy = _compute_accuracy(
            network=network,
            cost_fn=cost_fn,
            energy_minimizer=energy_minimizer_inference,
            dataloader=train_loader,
        )
        metrics = {"train_accuracy": train_accuracy}
        if writer is not None:
            writer.close()
        return run_dir, model_path, model_npz_path, metrics

    if writer is not None:
        writer.close()
    return run_dir, model_path, model_npz_path


def moons_linspace(
    *,
    settings: LinspaceRunSettings,
    output_root: str,
    run_subdir: Optional[str],
    weights_path: Optional[str],
    cli_command: Optional[str],
    run_dir: Optional[Path] = None,
    model_path: Optional[Path] = None,
    model_npz_path: Optional[Path] = None,
) -> Path:
    _set_seed(settings.seed)
    device = _resolve_torch_device(settings.device)
    minimizer_settings_data = settings.minimizer_settings
    linspace_settings = settings.linspace_settings
    non_linearity_data = settings.non_linearity_data
    input_dim, hidden_dims, output_dim, _ = _parse_layer_shapes(settings.dims)
    minimizer_settings = MinimizerSettings(
        rel_tol=minimizer_settings_data.rel_tol,
        vn_tol=minimizer_settings_data.vn_tol,
        use_polish=minimizer_settings_data.use_polish,
        max_newton_iters=minimizer_settings_data.max_newton_iters,
        z_thresh=minimizer_settings_data.z_thresh,
        exp_clip=minimizer_settings_data.exp_clip,
        dynamic_polish=minimizer_settings_data.dynamic_polish,
        overrelaxation_reject_steps=minimizer_settings_data.overrelaxation_reject_steps,
        overrelaxation_reject_max_tries=minimizer_settings_data.overrelaxation_reject_max_tries,
        overrelaxation_reject_shrink=minimizer_settings_data.overrelaxation_reject_shrink,
        overrelaxation_reject_eps=minimizer_settings_data.overrelaxation_reject_eps,
        experimental_exponential_newton_tol_progressive=(
            minimizer_settings_data.experimental_exponential_newton_tol_progressive
        ),
        experimental_exponential_newton_tol_start=(
            minimizer_settings_data.experimental_exponential_newton_tol_start
        ),
        experimental_exponential_newton_tol_end=(
            minimizer_settings_data.experimental_exponential_newton_tol_end
        ),
        experimental_exponential_newton_tol_switch_hi=(
            minimizer_settings_data.experimental_exponential_newton_tol_switch_hi
        ),
        experimental_exponential_newton_tol_switch_lo=(
            minimizer_settings_data.experimental_exponential_newton_tol_switch_lo
        ),
    )
    double_diode_updater = minimizer_settings_data.double_diode_updater
    adaptive_equilibrium = minimizer_settings_data.adaptive_equilibrium
    minimizer_cls, extra_args = _resolve_minimizer_impl(minimizer_settings_data)
    (
        energy_fn,
        network,
        free_layers,
        _cost_fn,
        _output_layer,
        layer_shapes,
        input_gain,
        quadratic_params,
        exponential_params,
        hard_sigmoid_params,
    ) = _build_energy_stack(
        device=device,
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        output_dim=output_dim,
        non_linearity=non_linearity_data.non_linearity,
        voltage_amp=settings.voltage_amp,
        current_amp=settings.current_amp,
        quadratic_diode_param=non_linearity_data.quadratic_diode_param,
        exponential_diode_param=non_linearity_data.exponential_diode_param,
        hard_sigmoid_param=non_linearity_data.hard_sigmoid_param,
        input_gain=settings.input_gain,
        weights_path=weights_path,
        weight_gains=settings.weight_gains,
        weight_min=settings.weight_min,
        weight_max=settings.weight_max,
        dataset_name=settings.dataset_name,
    )

    energy_minimizer_inference = minimizer_cls(
        fn=energy_fn,
        free_layers=free_layers,
        num_iterations=settings.num_iterations,
        mode="asynchronous",
        non_linearity=non_linearity_data.non_linearity,
        quadratic_diode_param=quadratic_params,
        exponential_diode_param=exponential_params,
        voltage_amp=settings.voltage_amp,
        current_amp=settings.current_amp,
        hard_sigmoid_param=hard_sigmoid_params,
        iv_data=None,
        iv_data_path=non_linearity_data.iv_data_path,
        double_diode_updater=double_diode_updater,
        adaptive_equilibrium=adaptive_equilibrium,
        overrelaxation_factor=minimizer_settings_data.overrelaxation_factor,
        single_diode_updater=minimizer_settings_data.single_diode_updater,
        damping=minimizer_settings_data.damping,
        experimental_newton_max_steps=minimizer_settings_data.experimental_newton_max_steps,
        minimizer_settings=minimizer_settings,
        **extra_args,
    )

    if run_dir is None:
        run_dir = _create_run_dir(output_root, run_subdir, non_linearity_data.non_linearity, linspace_only=True)

    grid_dataset = LinSpaceDataset(
        name="linspace_eval",
        batch_size=settings.batch_size,
        device=device,
        num_inputs=input_dim,
        min=linspace_settings.linspace_min,
        max=linspace_settings.linspace_max,
        num_samples=linspace_settings.linspace_samples,
    )
    grid_loader = grid_dataset.build_mesh(
        xmin=linspace_settings.linspace_min,
        xmax=linspace_settings.linspace_max,
        ymin=linspace_settings.linspace_min,
        ymax=linspace_settings.linspace_max,
    )

    input_layer = network.layers()[0]
    original_input_gain = getattr(input_layer, "_gain", None)
    if original_input_gain is not None:
        input_layer._gain = 1.0
    state_history = {layer.name: [] for layer in network.layers()}
    residual_history = {layer.name: [] for layer in network.layers()}
    input_batches = []
    iteration_count_batches = []

    for batch in grid_loader:
        if isinstance(batch, (list, tuple)):
            x = batch[0]
        else:
            x = batch
        input_batches.append(x.detach().cpu())
        x_model = x
        if settings.dataset_name == "moons" and x.shape[1] == 2 and input_dim != 2:
            x_model = MoonsDataset._expand_xy_features(x, input_dim)
        network.set_input(x_model, reset=True)
        input_layer.set_input(x_model, mode="test")
        energy_minimizer_inference.compute_equilibrium()
        batch_iteration_counts = None
        if hasattr(energy_minimizer_inference, "equilibrium_sample_iterations"):
            sample_iters = energy_minimizer_inference.equilibrium_sample_iterations()
            if sample_iters is not None:
                batch_iteration_counts = sample_iters.detach().cpu().numpy()
        if batch_iteration_counts is None and hasattr(energy_minimizer_inference, "equilibrium_iteration_stats"):
            stats = energy_minimizer_inference.equilibrium_iteration_stats(reset=False)
            last_iterations = int(stats.get("last_iterations", 0))
            batch_size_local = int(x.shape[0]) if hasattr(x, "shape") and len(x.shape) > 0 else 1
            batch_iteration_counts = np.full((batch_size_local,), last_iterations, dtype=np.int64)
        if batch_iteration_counts is not None:
            batch_iteration_counts = np.asarray(batch_iteration_counts).reshape(-1)
            batch_size_local = int(x.shape[0]) if hasattr(x, "shape") and len(x.shape) > 0 else 1
            if batch_iteration_counts.size == 1 and batch_size_local > 1:
                batch_iteration_counts = np.full((batch_size_local,), int(batch_iteration_counts[0]), dtype=np.int64)
            if batch_iteration_counts.size != batch_size_local:
                raise ValueError(
                    f"Iteration count shape mismatch: got {batch_iteration_counts.size} values for batch size {batch_size_local}."
                )
            iteration_count_batches.append(batch_iteration_counts.astype(np.int64, copy=False))
        for layer in network.layers():
            state_history[layer.name].append(layer.state.detach().cpu())
        fn = network._function
        for layer in fn.layers():
            grad = fn.grad_layer_fn(layer)()
            res_current = grad.abs().max()
            residual_history[layer.name].append(float(res_current.item()))
        fn = network._function
        for layer in fn.layers():
            grad = fn.grad_layer_fn(layer)()
            res_current = grad.abs().max()
            residual_history[layer.name].append(float(res_current.item()))
        fn = network._function
        for layer in fn.layers():
            grad = fn.grad_layer_fn(layer)()
            # Residual current = ||dE/dz||_inf
            res_current = grad.abs().max()
            residual_history[layer.name].append(float(res_current.item()))

    linspace_iter_stats = None
    if hasattr(energy_minimizer_inference, "equilibrium_iteration_stats"):
        linspace_iter_stats = energy_minimizer_inference.equilibrium_iteration_stats(reset=True)
        print(
            "[equilibrium][linspace] "
            f"avg_iters={linspace_iter_stats['avg_iterations']:.2f} "
            f"calls={linspace_iter_stats['calls']} "
            f"last={linspace_iter_stats['last_iterations']} "
            f"max={linspace_iter_stats['max_iterations']}"
        )

    if original_input_gain is not None:
        input_layer._gain = original_input_gain

    inputs_array = torch.cat(input_batches, dim=0).numpy()
    states_array = {name: torch.cat(history, dim=0).numpy() for name, history in state_history.items()}

    inputs_path = run_dir / "linspace_inputs.npz"
    states_path = run_dir / "linspace_states.npz"
    residuals_path = run_dir / "linspace_residual_currents.npz"
    iter_counts_path = run_dir / "linspace_iteration_counts.npz"
    np.savez(inputs_path, inputs=inputs_array)
    np.savez(states_path, **states_array)
    np.savez(residuals_path, **{name: np.asarray(values) for name, values in residual_history.items()})
    residual_tol = minimizer_settings_data.residual_current_tol
    residual_stats = {}
    residual_all = []
    for name, values in residual_history.items():
        if _skip_layer_for_residual_comparison(name):
            continue
        arr = np.asarray(values, dtype=float)
        if arr.size == 0:
            continue
        residual_all.append(arr.reshape(-1))
        stats = {
            "count": int(arr.size),
            "max": float(np.max(arr)),
            "mean": float(np.mean(arr)),
            "p50": float(np.percentile(arr, 50)),
            "p90": float(np.percentile(arr, 90)),
            "p99": float(np.percentile(arr, 99)),
        }
        if residual_tol is not None:
            tol_val = float(residual_tol)
            stats["tol"] = tol_val
            stats["pass_rate"] = float(np.mean(arr <= tol_val))
            stats["pass_all"] = bool(np.all(arr <= tol_val))
        residual_stats[name] = stats
    residual_overall = None
    if residual_all:
        all_vals = np.concatenate(residual_all, axis=0)
        residual_overall = {
            "count": int(all_vals.size),
            "max": float(np.max(all_vals)),
            "mean": float(np.mean(all_vals)),
            "p50": float(np.percentile(all_vals, 50)),
            "p90": float(np.percentile(all_vals, 90)),
            "p99": float(np.percentile(all_vals, 99)),
        }
        if residual_tol is not None:
            tol_val = float(residual_tol)
            residual_overall["tol"] = tol_val
            residual_overall["pass_rate"] = float(np.mean(all_vals <= tol_val))
            residual_overall["pass_all"] = bool(np.all(all_vals <= tol_val))
    if iteration_count_batches:
        iteration_counts = np.concatenate(iteration_count_batches, axis=0)
        expected = int(linspace_settings.linspace_samples) * int(linspace_settings.linspace_samples)
        if iteration_counts.size != expected:
            raise ValueError(
                f"Expected {expected} iteration counts for a {linspace_settings.linspace_samples}x{linspace_settings.linspace_samples} grid, got {iteration_counts.size}."
            )
        iteration_grid = iteration_counts.reshape(
            linspace_settings.linspace_samples, linspace_settings.linspace_samples
        )
        np.savez(
            iter_counts_path,
            iteration_counts=iteration_counts,
            iteration_grid=iteration_grid,
            linspace_samples=np.asarray(linspace_settings.linspace_samples),
        )

    dims = [shape[0] for shape in layer_shapes]
    resolved_weight_gains = list(settings.weight_gains)
    metadata = {
        "run_dir": str(run_dir),
        "non_linearity": non_linearity_data.non_linearity,
        "dataset": settings.dataset_name,
        "voltage_amp": settings.voltage_amp,
        "current_amp": settings.current_amp,
        "input_gain": settings.input_gain,
        "weight_gains": resolved_weight_gains,
        "weight_min": settings.weight_min,
        "weight_max": settings.weight_max,
        "quadratic_diode_param": quadratic_params,
        "exponential_diode_param": exponential_params,
        "hard_sigmoid_param": hard_sigmoid_params,
        "layer_shapes": layer_shapes,
        "dims": dims,
        "num_iterations": settings.num_iterations,
        "num_epochs": settings.num_epochs,
        "linspace_min": linspace_settings.linspace_min,
        "linspace_max": linspace_settings.linspace_max,
        "linspace_samples": linspace_settings.linspace_samples,
        "linspace_only": True,
        "weights_path": str(weights_path) if weights_path else None,
        "linspace_residual_currents": str(residuals_path),
        "residual_current_stats": residual_stats,
        "residual_current_overall": residual_overall,
        "residual_comparison_skipped_layers": ["Layer_0"],
        "double_diode_updater": double_diode_updater,
        "adaptive_equilibrium": adaptive_equilibrium,
        "overrelaxation_factor": minimizer_settings_data.overrelaxation_factor,
        "overrelaxation_reject_steps": minimizer_settings_data.overrelaxation_reject_steps,
        "overrelaxation_reject_max_tries": minimizer_settings_data.overrelaxation_reject_max_tries,
        "overrelaxation_reject_shrink": minimizer_settings_data.overrelaxation_reject_shrink,
        "overrelaxation_reject_eps": minimizer_settings_data.overrelaxation_reject_eps,
        "single_diode_updater": minimizer_settings_data.single_diode_updater,
        "minimizer_impl": minimizer_settings_data.minimizer_impl,
        "anderson_m": minimizer_settings_data.anderson_m,
        "anderson_omega": minimizer_settings_data.anderson_omega,
        "anderson_tol_floor": minimizer_settings_data.anderson_tol_floor,
        "anderson_reg": minimizer_settings_data.anderson_reg,
    }
    if iteration_count_batches:
        metadata["linspace_iteration_counts"] = str(iter_counts_path)
    if cli_command:
        metadata["cli_command"] = cli_command
    if linspace_iter_stats is not None:
        metadata["equilibrium_iterations"] = linspace_iter_stats
    if hasattr(energy_minimizer_inference, "anderson_run_stats"):
        metadata["anderson_run_stats"] = energy_minimizer_inference.anderson_run_stats()
    if hasattr(energy_minimizer_inference, "overrelaxation_reject_stats"):
        metadata["overrelaxation_reject_stats"] = energy_minimizer_inference.overrelaxation_reject_stats(reset=True)
    metadata_path = run_dir / "run_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2))

    if model_path is not None or model_npz_path is not None:
        if model_path is not None:
            print(f"Saved model to {model_path}")
        if model_npz_path is not None:
            print(f"Saved weights to {model_npz_path}")
    print(f"Saved linspace inputs to {inputs_path}")
    print(f"Saved linspace states to {states_path}")
    print(f"Saved linspace residual currents to {residuals_path}")
    if iteration_count_batches:
        print(f"Saved linspace iteration counts to {iter_counts_path}")
    print(f"Saved metadata to {metadata_path}")

    return run_dir


def digits_validate(
    *,
    settings: ValidateRunSettings,
    output_root: str,
    run_subdir: Optional[str],
    weights_path: Optional[str],
    cli_command: Optional[str],
    run_dir: Optional[Path] = None,
) -> Path:
    """Run digits validation and save settled layer states."""
    _set_seed(settings.seed)
    timing_start = time.perf_counter()
    usage_start = resource.getrusage(resource.RUSAGE_SELF)
    device = _resolve_torch_device(settings.device)
    skip_residual = os.environ.get("DRN_SKIP_RESIDUAL", "").lower() in ("1", "true", "yes")
    minimizer_settings_data = settings.minimizer_settings
    non_linearity_data = settings.non_linearity_data
    input_dim, hidden_dims, output_dim, _ = _parse_layer_shapes(settings.dims)

    if settings.dataset_name != "digits":
        raise SystemExit("digits_validate requires dataset_name='digits'.")

    timing_start = time.perf_counter()
    dataset = DigitsDataset(
        name="digits",
        batch_size=settings.batch_size,
        device=device,
        num_samples=settings.num_points,
        seed=settings.seed or 0,
    )
    _train_loader, test_loader = dataset.build()

    if run_dir is None:
        run_dir = _create_run_dir(
            output_root,
            run_subdir,
            non_linearity_data.non_linearity,
            linspace_only=False,
        )

    minimizer_settings = MinimizerSettings(
        rel_tol=minimizer_settings_data.rel_tol,
        vn_tol=minimizer_settings_data.vn_tol,
        use_polish=minimizer_settings_data.use_polish,
        max_newton_iters=minimizer_settings_data.max_newton_iters,
        z_thresh=minimizer_settings_data.z_thresh,
        exp_clip=minimizer_settings_data.exp_clip,
        dynamic_polish=minimizer_settings_data.dynamic_polish,
        overrelaxation_reject_steps=minimizer_settings_data.overrelaxation_reject_steps,
        overrelaxation_reject_max_tries=minimizer_settings_data.overrelaxation_reject_max_tries,
        overrelaxation_reject_shrink=minimizer_settings_data.overrelaxation_reject_shrink,
        overrelaxation_reject_eps=minimizer_settings_data.overrelaxation_reject_eps,
        experimental_exponential_newton_tol_progressive=(
            minimizer_settings_data.experimental_exponential_newton_tol_progressive
        ),
        experimental_exponential_newton_tol_start=(
            minimizer_settings_data.experimental_exponential_newton_tol_start
        ),
        experimental_exponential_newton_tol_end=(
            minimizer_settings_data.experimental_exponential_newton_tol_end
        ),
        experimental_exponential_newton_tol_switch_hi=(
            minimizer_settings_data.experimental_exponential_newton_tol_switch_hi
        ),
        experimental_exponential_newton_tol_switch_lo=(
            minimizer_settings_data.experimental_exponential_newton_tol_switch_lo
        ),
    )
    double_diode_updater = minimizer_settings_data.double_diode_updater
    adaptive_equilibrium = minimizer_settings_data.adaptive_equilibrium
    minimizer_cls, extra_args = _resolve_minimizer_impl(minimizer_settings_data)

    (
        energy_fn,
        network,
        free_layers,
        cost_fn,
        _output_layer,
        _layer_shapes,
        _input_gain,
        quadratic_params,
        exponential_params,
        hard_sigmoid_params,
    ) = _build_energy_stack(
        device=device,
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        output_dim=output_dim,
        non_linearity=non_linearity_data.non_linearity,
        voltage_amp=settings.voltage_amp,
        current_amp=settings.current_amp,
        quadratic_diode_param=non_linearity_data.quadratic_diode_param,
        exponential_diode_param=non_linearity_data.exponential_diode_param,
        hard_sigmoid_param=non_linearity_data.hard_sigmoid_param,
        input_gain=settings.input_gain,
        weights_path=weights_path,
        weight_gains=settings.weight_gains,
        weight_min=settings.weight_min,
        weight_max=settings.weight_max,
        dataset_name=settings.dataset_name,
    )

    energy_minimizer_inference = minimizer_cls(
        fn=energy_fn,
        free_layers=free_layers,
        num_iterations=settings.num_iterations,
        mode="asynchronous",
        non_linearity=non_linearity_data.non_linearity,
        quadratic_diode_param=quadratic_params,
        exponential_diode_param=exponential_params,
        voltage_amp=settings.voltage_amp,
        current_amp=settings.current_amp,
        hard_sigmoid_param=hard_sigmoid_params,
        iv_data=None,
        iv_data_path=non_linearity_data.iv_data_path,
        double_diode_updater=double_diode_updater,
        adaptive_equilibrium=adaptive_equilibrium,
        overrelaxation_factor=minimizer_settings_data.overrelaxation_factor,
        single_diode_updater=minimizer_settings_data.single_diode_updater,
        damping=minimizer_settings_data.damping,
        experimental_newton_max_steps=minimizer_settings_data.experimental_newton_max_steps,
        minimizer_settings=minimizer_settings,
        **extra_args,
    )
    if hasattr(energy_minimizer_inference, "reset_equilibrium_iteration_stats"):
        energy_minimizer_inference.reset_equilibrium_iteration_stats()
    if hasattr(energy_minimizer_inference, "reset_overrelaxation_reject_stats"):
        energy_minimizer_inference.reset_overrelaxation_reject_stats()

    inputs_batches = []
    labels_batches = []
    indices_batches = []
    state_history = {layer.name: [] for layer in network.layers()}
    residual_history = {layer.name: [] for layer in network.layers()} if not skip_residual else None
    iteration_count_batches = []
    total = 0
    correct = 0

    for batch in test_loader:
        if isinstance(batch, (list, tuple)):
            if len(batch) == 3:
                x, y, idx = batch
            elif len(batch) == 2:
                x, y = batch
                idx = None
            else:
                x, y, idx = batch[0], None, None
        else:
            x, y, idx = batch, None, None

        network.set_input(x, reset=True)
        energy_minimizer_inference.compute_equilibrium()
        batch_iteration_counts = None
        if hasattr(energy_minimizer_inference, "equilibrium_sample_iterations"):
            sample_iters = energy_minimizer_inference.equilibrium_sample_iterations()
            if sample_iters is not None:
                batch_iteration_counts = sample_iters.detach().cpu().numpy()
        if batch_iteration_counts is None and hasattr(energy_minimizer_inference, "equilibrium_iteration_stats"):
            stats = energy_minimizer_inference.equilibrium_iteration_stats(reset=False)
            last_iterations = int(stats.get("last_iterations", 0))
            batch_size_local = int(x.shape[0]) if hasattr(x, "shape") and len(x.shape) > 0 else 1
            batch_iteration_counts = np.full((batch_size_local,), last_iterations, dtype=np.int64)
        if batch_iteration_counts is not None:
            batch_iteration_counts = np.asarray(batch_iteration_counts).reshape(-1)
            batch_size_local = int(x.shape[0]) if hasattr(x, "shape") and len(x.shape) > 0 else 1
            if batch_iteration_counts.size == 1 and batch_size_local > 1:
                batch_iteration_counts = np.full((batch_size_local,), int(batch_iteration_counts[0]), dtype=np.int64)
            if batch_iteration_counts.size != batch_size_local:
                raise ValueError(
                    f"Iteration count shape mismatch: got {batch_iteration_counts.size} values for batch size {batch_size_local}."
                )
            iteration_count_batches.append(batch_iteration_counts.astype(np.int64, copy=False))

        inputs_batches.append(x.detach().cpu())
        if y is not None:
            labels_batches.append(y.detach().cpu())
        if idx is not None:
            indices_batches.append(idx.detach().cpu())
        for layer in network.layers():
            state_history[layer.name].append(layer.state.detach().cpu())
        if not skip_residual:
            fn = network._function
            for layer in fn.layers():
                grad = fn.grad_layer_fn(layer)()
                res_current = grad.abs().max()
                residual_history[layer.name].append(float(res_current.item()))
        if y is not None:
            cost_fn.set_target(y)
            batch_errors = cost_fn.error_fn()
            batch_total = int(batch_errors.numel())
            batch_correct = batch_total - int(batch_errors.sum().item())
            total += batch_total
            correct += batch_correct

    inputs_array = torch.cat(inputs_batches, dim=0).numpy() if inputs_batches else None
    labels_array = torch.cat(labels_batches, dim=0).numpy() if labels_batches else None
    indices_array = torch.cat(indices_batches, dim=0).numpy() if indices_batches else None
    states_array = {name: torch.cat(history, dim=0).numpy() for name, history in state_history.items()}
    accuracy = (correct / total) if total > 0 else None
    iter_stats = None
    if hasattr(energy_minimizer_inference, "equilibrium_iteration_stats"):
        iter_stats = energy_minimizer_inference.equilibrium_iteration_stats(reset=True)
    hist_stats = None
    if hasattr(energy_minimizer_inference, "anderson_history_stats"):
        hist_stats = energy_minimizer_inference.anderson_history_stats(reset=True)
    newton_stats = None
    if hasattr(energy_minimizer_inference, "_updaters"):
        total_iters = 0
        calls = 0
        for updater in energy_minimizer_inference._updaters:
            if hasattr(updater, "_newton_iter_total") and hasattr(updater, "_newton_iter_calls"):
                total_iters += int(getattr(updater, "_newton_iter_total", 0))
                calls += int(getattr(updater, "_newton_iter_calls", 0))
        if calls >= 0:
            avg_iters = (total_iters / calls) if calls > 0 else 0.0
            newton_stats = {
                "total_iters": total_iters,
                "calls": calls,
                "avg_iters_per_call": avg_iters,
            }

    inputs_path = run_dir / "validation_inputs.npz"
    states_path = run_dir / "validation_states.npz"
    iter_counts_path = run_dir / "validation_iteration_counts.npz"
    np.savez(
        inputs_path,
        inputs=inputs_array,
        labels=labels_array,
        indices=indices_array,
    )
    np.savez(states_path, **states_array)
    if iteration_count_batches:
        iteration_counts = np.concatenate(iteration_count_batches, axis=0)
        expected = int(inputs_array.shape[0]) if inputs_array is not None else int(iteration_counts.size)
        if iteration_counts.size != expected:
            raise ValueError(
                f"Expected {expected} iteration counts for validation, got {iteration_counts.size}."
            )
        np.savez(iter_counts_path, iteration_counts=iteration_counts)

    residuals_path = None
    residual_stats = {}
    residual_overall = None
    if not skip_residual and residual_history is not None:
        residuals_path = run_dir / "validation_residual_currents.npz"
        np.savez(residuals_path, **{name: np.asarray(values) for name, values in residual_history.items()})
        residual_tol = minimizer_settings_data.residual_current_tol
        residual_all = []
        for name, values in residual_history.items():
            if _skip_layer_for_residual_comparison(name):
                continue
            arr = np.asarray(values, dtype=float)
            if arr.size == 0:
                continue
            residual_all.append(arr.reshape(-1))
            stats = {
                "count": int(arr.size),
                "max": float(np.max(arr)),
                "mean": float(np.mean(arr)),
                "p50": float(np.percentile(arr, 50)),
                "p90": float(np.percentile(arr, 90)),
                "p99": float(np.percentile(arr, 99)),
            }
            if residual_tol is not None:
                tol_val = float(residual_tol)
                stats["tol"] = tol_val
                stats["pass_rate"] = float(np.mean(arr <= tol_val))
                stats["pass_all"] = bool(np.all(arr <= tol_val))
            residual_stats[name] = stats
        if residual_all:
            all_vals = np.concatenate(residual_all, axis=0)
            residual_overall = {
                "count": int(all_vals.size),
                "max": float(np.max(all_vals)),
                "mean": float(np.mean(all_vals)),
                "p50": float(np.percentile(all_vals, 50)),
                "p90": float(np.percentile(all_vals, 90)),
                "p99": float(np.percentile(all_vals, 99)),
            }
            if residual_tol is not None:
                tol_val = float(residual_tol)
                residual_overall["tol"] = tol_val
                residual_overall["pass_rate"] = float(np.mean(all_vals <= tol_val))
                residual_overall["pass_all"] = bool(np.all(all_vals <= tol_val))

    metadata = {
        "run_dir": str(run_dir),
        "non_linearity": non_linearity_data.non_linearity,
        "dataset": settings.dataset_name,
        "device": str(device),
        "dims": settings.dims,
        "batch_size": settings.batch_size,
        "num_epochs": settings.num_epochs,
        "num_iterations": settings.num_iterations,
        "rel_tol": minimizer_settings_data.rel_tol,
        "vn_tol": minimizer_settings_data.vn_tol,
        "use_polish": minimizer_settings_data.use_polish,
        "max_newton_iters": minimizer_settings_data.max_newton_iters,
        "z_thresh": minimizer_settings_data.z_thresh,
        "dynamic_polish": minimizer_settings_data.dynamic_polish,
        "linspace_min": settings.linspace_min,
        "linspace_max": settings.linspace_max,
        "linspace_samples": settings.linspace_samples,
        "num_samples": int(inputs_array.shape[0]) if inputs_array is not None else 0,
        "weights_path": str(weights_path) if weights_path else None,
        "validation_inputs": str(inputs_path),
        "validation_states": str(states_path),
        "validation_residual_currents": str(residuals_path) if residuals_path else None,
        "residual_current_stats": residual_stats,
        "residual_current_overall": residual_overall,
        "residual_comparison_skipped_layers": ["Layer_0"],
        "validation_accuracy": accuracy,
        "validation_correct": int(correct),
        "validation_total": int(total),
        "double_diode_updater": double_diode_updater,
        "adaptive_equilibrium": adaptive_equilibrium,
        "overrelaxation_factor": minimizer_settings_data.overrelaxation_factor,
        "overrelaxation_reject_steps": minimizer_settings_data.overrelaxation_reject_steps,
        "overrelaxation_reject_max_tries": minimizer_settings_data.overrelaxation_reject_max_tries,
        "overrelaxation_reject_shrink": minimizer_settings_data.overrelaxation_reject_shrink,
        "overrelaxation_reject_eps": minimizer_settings_data.overrelaxation_reject_eps,
        "minimizer_impl": minimizer_settings_data.minimizer_impl,
        "double_diode_runtime": minimizer_settings_data.double_diode_runtime,
        "damping": minimizer_settings_data.damping,
        "experimental_newton_max_steps": minimizer_settings_data.experimental_newton_max_steps,
        "experimental_exponential_newton_tol_progressive": (
            minimizer_settings_data.experimental_exponential_newton_tol_progressive
        ),
        "experimental_exponential_newton_tol_start": (
            minimizer_settings_data.experimental_exponential_newton_tol_start
        ),
        "experimental_exponential_newton_tol_end": (
            minimizer_settings_data.experimental_exponential_newton_tol_end
        ),
        "experimental_exponential_newton_tol_switch_hi": (
            minimizer_settings_data.experimental_exponential_newton_tol_switch_hi
        ),
        "experimental_exponential_newton_tol_switch_lo": (
            minimizer_settings_data.experimental_exponential_newton_tol_switch_lo
        ),
        "exp_clip": minimizer_settings_data.exp_clip,
        "single_diode_updater": minimizer_settings_data.single_diode_updater,
        "iv_data_path": non_linearity_data.iv_data_path,
        "LABS_IV_CURVE_PATH": os.environ.get("LABS_IV_CURVE_PATH"),
    }
    total_seconds = time.perf_counter() - timing_start
    metadata["validation_total_seconds"] = total_seconds
    if iteration_count_batches:
        metadata["validation_iteration_counts"] = str(iter_counts_path)
    if iter_stats is not None:
        metadata["equilibrium_iterations"] = iter_stats
    if hist_stats is not None:
        metadata["anderson_history"] = hist_stats
    if hasattr(energy_minimizer_inference, "overrelaxation_reject_stats"):
        metadata["overrelaxation_reject_stats"] = energy_minimizer_inference.overrelaxation_reject_stats(reset=True)
    if newton_stats is not None:
        metadata["newton_iteration_stats"] = newton_stats
    double_diode_timing_stats = _collect_double_diode_timing_stats(energy_minimizer_inference, reset=True)
    if double_diode_timing_stats is not None:
        metadata["double_diode_timing_stats"] = double_diode_timing_stats
    if cli_command:
        metadata["cli_command"] = cli_command
    metadata_path = run_dir / "validation_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2))

    usage_end = resource.getrusage(resource.RUSAGE_SELF)
    user_time = usage_end.ru_utime - usage_start.ru_utime
    system_time = usage_end.ru_stime - usage_start.ru_stime
    cpu_pct = 0.0
    if total_seconds > 0:
        cpu_pct = ((user_time + system_time) / total_seconds) * 100.0

    elapsed_seconds = max(total_seconds, 0.0)
    hours = int(elapsed_seconds // 3600)
    minutes = int((elapsed_seconds % 3600) // 60)
    seconds = elapsed_seconds % 60.0
    if hours > 0:
        elapsed_fmt = f"{hours}:{minutes:02d}:{seconds:05.2f}"
    else:
        elapsed_fmt = f"{minutes}:{seconds:05.2f}"

    log_name = "time_digits_validate_skip_residual.log" if skip_residual else "time_digits_validate_with_residual.log"
    timing_log_path = run_dir / log_name
    cmd = cli_command or ""
    page_size = os.sysconf("SC_PAGE_SIZE") if hasattr(os, "sysconf") else 4096
    log_lines = [
        f'\tCommand being timed: "{cmd}"',
        f"\tUser time (seconds): {user_time:.2f}",
        f"\tSystem time (seconds): {system_time:.2f}",
        f"\tPercent of CPU this job got: {cpu_pct:.0f}%",
        f"\tElapsed (wall clock) time (h:mm:ss or m:ss): {elapsed_fmt}",
        "\tAverage shared text size (kbytes): 0",
        "\tAverage unshared data size (kbytes): 0",
        "\tAverage stack size (kbytes): 0",
        "\tAverage total size (kbytes): 0",
        f"\tMaximum resident set size (kbytes): {usage_end.ru_maxrss}",
        "\tAverage resident set size (kbytes): 0",
        f"\tMajor (requiring I/O) page faults: {usage_end.ru_majflt - usage_start.ru_majflt}",
        f"\tMinor (reclaiming a frame) page faults: {usage_end.ru_minflt - usage_start.ru_minflt}",
        f"\tVoluntary context switches: {usage_end.ru_nvcsw - usage_start.ru_nvcsw}",
        f"\tInvoluntary context switches: {usage_end.ru_nivcsw - usage_start.ru_nivcsw}",
        f"\tSwaps: {usage_end.ru_nswap - usage_start.ru_nswap}",
        f"\tFile system inputs: {usage_end.ru_inblock - usage_start.ru_inblock}",
        f"\tFile system outputs: {usage_end.ru_oublock - usage_start.ru_oublock}",
        "\tSocket messages sent: 0",
        "\tSocket messages received: 0",
        f"\tSignals delivered: {usage_end.ru_nsignals - usage_start.ru_nsignals}",
        f"\tPage size (bytes): {page_size}",
        "\tExit status: 0",
    ]
    timing_log_path.write_text("\n".join(log_lines) + "\n")

    print(f"Saved validation inputs to {inputs_path}")
    print(f"Saved validation states to {states_path}")
    if residuals_path is not None:
        print(f"Saved validation residual currents to {residuals_path}")
    if iteration_count_batches:
        print(f"Saved validation iteration counts to {iter_counts_path}")
    if accuracy is not None:
        print(f"Validation accuracy: {accuracy * 100:.2f}% ({correct}/{total})")
    print(f"Saved metadata to {metadata_path}")
    print(f"Saved timing log to {timing_log_path}")
    return run_dir
