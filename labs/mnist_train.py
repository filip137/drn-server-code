import argparse
import json
import os
import sys
from importlib import import_module
import importlib.util
from pathlib import Path

import torch


# Ensure project modules are importable
LABS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = LABS_DIR.parent / "energy-based-learning"

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
from model.resistive.minimizer import QuadraticMinimizer  # noqa: E402
from model.function.cost import SquaredError, SquaredErrorPairedOutputs  # noqa: E402
from model.function.network import Network  # noqa: E402
from model.variable.parameter import ConvWeight  # noqa: E402
from training.monitor import Optimizer  # noqa: E402
from training.sgd import AugmentedFunction, EquilibriumProp  # noqa: E402


def _default_quadratic_params():
    return {"diode_conductance": 1.0, "v_min": -1e6, "v_max": 1e6}


def _default_exponential_params():
    return {"I_s": 1e-6, "V_t": 0.025, "V_off": 0.0}


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
    sanity_check=False,
    epoch_callback=None,
):
    config = load_config(config_path)

    project_root = PROJECT_ROOT
    if str(project_root) not in sys.path:
        sys.path.append(str(project_root))

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

    model_cfg = {**config["model_base"], **config["model_overrides"][model_key]}
    dataset_cfg = config["datasets"][dataset_key]
    beta_value = beta
    if beta_value is None:
        beta_value = config.get("beta")
        if beta_value is None:
            beta_value = max(model_cfg.get("nudging", 0.0), 1e-3)
    beta_value = float(beta_value)

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
        f"Training model '{model_key}' | layer_shapes={layer_shapes} "
        f"| num_classes={num_classes} | non_linearity={model_cfg['non_linearity']} "
        f"| beta={beta_value}"
    )

    energy_fn = FlexibleDeepResistiveEnergy(
        layer_shapes=layer_shapes,
        conv_pipeline=conv_pipeline,
        weight_gains=model_cfg["weight_gains"],
        input_gain=model_cfg["input_gain"],
        non_linearity=model_cfg["non_linearity"],
        exponential_diode_param=model_cfg["exponential_diode_param"],
        quadratic_diode_param=model_cfg["quadratic_diode_param"],
        voltage_amp=model_cfg["voltage_amp"],
        current_amp=model_cfg["current_amp"],
        weight_min=model_cfg["weight_min"],
        weight_max=model_cfg["weight_max"],
        input_mode=config.get("input_mode", "train"),
    )
    energy_fn.set_device(device)

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

    minimizer_training = TrackingQuadraticMinimizer(
        augmented_fn,
        free_layers,
        num_iterations=model_cfg["num_iterations_inference"],
        mode=config["energy_minimizer"]["mode"],
        non_linearity=model_cfg["non_linearity"],
        exponential_diode_param=model_cfg["exponential_diode_param"],
        quadratic_diode_param=model_cfg["quadratic_diode_param"],
        voltage_amp=energy_fn._voltage_amp,
        current_amp=energy_fn._current_amp,
    )


    minimizer_inference= TrackingQuadraticMinimizer(
        energy_fn,
        free_layers,
        num_iterations=model_cfg["num_iterations_inference"],
        mode=config["energy_minimizer"]["mode"],
        non_linearity=model_cfg["non_linearity"],
        exponential_diode_param=model_cfg["exponential_diode_param"],
        quadratic_diode_param=model_cfg["quadratic_diode_param"],
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
    }

    stop_training = False
    for epoch in range(epochs):
        running_loss = 0.0
        running_correct = 0
        seen = 0
        reset_flag = True

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
            print(
                f"Epoch {epoch+1}/{epochs} | Batch {batch_idx+1}/{len(train_loader)} | "
                f"running loss={avg_loss:.4f} acc={acc*100:.2f}%",
                end="\r",
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
            for test_batch_idx, (images, labels) in enumerate(test_loader):
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
                print(
                    f"[Test] Epoch {epoch+1}/{epochs} | Batch {test_batch_idx+1}/{len(test_loader)} | running loss={avg_loss:.4f} acc={acc*100:.2f}%",
                    end="\r",
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
        summary["final_test_loss"] = history["test_loss"][-1]
        summary["test_error"] = 1.0 - summary["final_test_accuracy"]
    else:
        summary["test_error"] = 1.0 - summary["final_train_accuracy"]

    history["summary"] = summary
    return history





def train_mnist_conv(
    config_path,
    epochs,
    lr,
    beta,
    log_interval,
    max_batches,
    epoch_callback=None,
):
    return _train_image_task(
        config_path=config_path,
        epochs=epochs,
        lr=lr,
        beta=beta,
        log_interval=log_interval,
        max_batches=max_batches,
        dataset_key="mnist",
        model_key="mnist",
        image_shape=(28, 28),
        epoch_callback=epoch_callback,
    )


def train_tiny_grid(
    config_path,
    epochs,
    lr,
    beta,
    log_interval,
    max_batches,
    epoch_callback=None,
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
        epoch_callback=epoch_callback,
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

    log_interval = args.log_interval if args.log_interval is not None else config.get("log_interval")
    if log_interval is None:
        raise ValueError("Log interval must be provided via --log-interval or config['log_interval'].")
    log_interval = int(log_interval)

    max_batches = args.max_batches if args.max_batches is not None else config.get("max_batches")
    if max_batches is not None:
        max_batches = int(max_batches)

    model_key = lab_cfg.get("model_key", "mnist")

    if model_key == "tiny3x3":
        train_fn = train_tiny_grid
    else:
        train_fn = train_mnist_conv

    history = train_fn(
        config_path=args.config,
        epochs=epochs,
        lr=learning_rate_cfg,
        beta=beta,
        log_interval=log_interval,
        max_batches=max_batches,
    )
    print("Training history:", history)

    summary = history.get("summary", {})
    print("Final summary:", json.dumps(summary, indent=2))

    if args.result_json:
        result_payload = {
            "config_path": args.config,
            "epochs": epochs,
            "learning_rate": learning_rate_cfg,
            "beta": beta,
            "model_key": lab_cfg.get("model_key", "mnist"),
            "test_error": summary.get("test_error"),
            "final_train_accuracy": summary.get("final_train_accuracy"),
            "final_test_accuracy": summary.get("final_test_accuracy"),
            "best_test_accuracy": summary.get("best_test_accuracy"),
        }
        result_path = Path(args.result_json)
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_text(json.dumps(result_payload, indent=2))
        print(f"Wrote Optuna summary to {result_path}")


if __name__ == "__main__":
    main()
