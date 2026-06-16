import sys
from pathlib import Path


def _append_sys_path(path: Path):
    """Append `path` to `sys.path` if it's not already present."""
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.append(path_str)


LABS_DIR = Path(__file__).resolve().parent


def _find_project_root() -> Path:
    """
    Locate the project root by walking up the directory tree.
    Supports two layouts:
      1. labs/ is inside the project root.
      2. labs/ is a sibling of the project root (energy-based-learning/).
    """
    search_points = [LABS_DIR] + list(LABS_DIR.parents)
    for base in search_points:
        candidate = base
        if (candidate / "model").is_dir():
            return candidate
        alt_candidate = base / "energy-based-learning"
        if (alt_candidate / "model").is_dir():
            return alt_candidate
    raise ModuleNotFoundError(
        "Unable to locate the project root containing the 'model' package."
    )


PROJECT_ROOT = _find_project_root()
MODEL_ROOT = PROJECT_ROOT / "model"

_append_sys_path(LABS_DIR)
_append_sys_path(PROJECT_ROOT)
_append_sys_path(MODEL_ROOT)

from model.function.cost import SquaredErrorPairedOutputs
from model.function.network import Network
from custom_minimizer import CustomQuadraticMinimizer as QuadraticMinimizer
from model.function.interaction import (
    BiasInteraction,
    DoubleExponentialNonLinearInteraction,
    DoubleQuadraticNonLinearInteraction,
    HardSigmoidNonLinearInteraction,
    SumSeparableFunction,
    Function,
    QFunction,
)
from model.resistive.interaction import AveragePoolResistive, DenseResistive, MaxPoolResistive
from model.resistive.layer import NonlinearResistiveLayer, PoolLayer
from model.variable.layer import InputLayer, LinearLayer
from model.variable.parameter import Bias, ConvWeight, DenseWeight, PoolWeight
from training.sgd import Nudging

import torch
import torch.nn.functional as F

POOLING_INTERACTIONS = {
    "avg": AveragePoolResistive,
    "max": MaxPoolResistive,
}


class TrackingQuadraticMinimizer(QuadraticMinimizer):
    def __init__(self, fn, free_layers, *args, **kwargs):
        self._tracked_fn = fn
        super().__init__(fn, free_layers, *args, **kwargs)
        self.gradient_history_before_sample = {}
        self.gradient_history_after_sample = {}
        self.gradient_history_before = []
        self.gradient_history_after = []

    def _step(self,layer_group):
        pre_activations = [updater.pre_activate() for updater in layer_group]
        for updater, pre_activation in zip(layer_group, pre_activations):
            updater._layer.state = pre_activation

    def step(self, layer_group):
        debug = False
        #grads_before = {}
        if debug:
            for updater in layer_group:
                layer = updater._layer
                grad_fn = self._tracked_fn.grad_layer_fn(layer)
                grad = grad_fn().detach()
                grad_norm = float(torch.norm(grad, p=float("inf")).cpu())
                if layer.name not in self.gradient_history_before_sample:
                    self.gradient_history_before_sample[layer.name] = []
                self.gradient_history_before_sample[layer.name].append(grad_norm)




        #prev = updater._layer.state.clone()
        super().step(layer_group)
        #delta = (updater._layer.state - prev).abs().max().item()

        #grads_after = {}
        if debug:
            for updater in layer_group:
                layer = updater._layer
                grad_fn = self._tracked_fn.grad_layer_fn(layer)
                grad = grad_fn().detach()
                grad_norm = float(torch.norm(grad, p=float("inf")).cpu())
                if debug:
                    print("--- per-interaction a/b ---")
                    for interaction in self._tracked_fn._interactions:
                        if layer not in interaction.layers():
                            continue
                        # fetch the individual contribution
                        a_fn = getattr(interaction, "a_coef_fn", None)
                        b_fn = getattr(interaction, "b_coef_fn", None)
                        if a_fn:
                            a_val = a_fn(layer)
                            a_val = a_val() if callable(a_val) else a_val
                            print(f"{interaction.__class__.__name__} a:\n{a_val}")
                        if b_fn:
                            b_val = b_fn(layer)
                            b_val = b_val() if callable(b_val) else b_val
                            print(f"{interaction.__class__.__name__} b:\n{b_val}")

                    a = self._tracked_fn.a_coef_fn(layer)()
                    b = self._tracked_fn.b_coef_fn(layer)()
                    analytical = 2 * a * layer.state + b
                    residual = grad - analytical
                    print("residual:", residual)

                if layer.name not in self.gradient_history_after_sample:
                    self.gradient_history_after_sample[layer.name] = []
                self.gradient_history_after_sample[layer.name].append(grad_norm)
                if len(self.gradient_history_after_sample[layer.name]) == 30:
                    breakpoint()
    
    def update_gradients(self):
        self.gradient_history_before.append(self.gradient_history_before_sample)
        self.gradient_history_after.append(self.gradient_history_after_sample)
        self.gradient_history_before_sample = {}
        self.gradient_history_after_sample = {}


class DetailedSumSeparableFunction(SumSeparableFunction):
    """Sum-separable function that logs individual interaction contributions."""

    debug_enabled = False

    def __init__(self, layers, params, interactions):
        super().__init__(layers, params, interactions)

    def grad_layer_fn(self, layer):
        hooks = [
            (interaction, interaction.grad_layer_fn(layer))
            for interaction in self._interactions
            if layer in interaction.layers()
        ]

        def grad():
            total = None
            for interaction, fn in hooks:
                value = fn()
                if self.debug_enabled:
                    print(
                        f"[DetailedSumSeparableFunction][grad] "
                        f"{interaction.__class__.__name__}: {value}"
                    )
                total = value if total is None else total + value
            if total is None:
                return torch.zeros_like(layer.state)
            if self.debug_enabled:
                print(f"[DetailedSumSeparableFunction][grad] total: {total}")
            return total

        return grad

    def a_coef_fn(self, layer):
        hooks = []
        for interaction in self._interactions:
            if layer not in interaction.layers():
                continue
            fn = interaction.a_coef_fn(layer)
            if fn is None:
                continue
            hooks.append((interaction, fn))

        def a_fn():
            total = None
            for interaction, fn in hooks:
                value = fn()
                if self.debug_enabled:
                    print(
                        f"[DetailedSumSeparableFunction][a] "
                        f"{interaction.__class__.__name__}: {value}"
                    )
                total = value if total is None else total + value
            if total is None:
                return torch.zeros_like(layer.state)
            if self.debug_enabled:
                print(f"[DetailedSumSeparableFunction][a] total: {total}")
            return total

        return a_fn

    def b_coef_fn(self, layer):
        hooks = [
            (interaction, interaction.b_coef_fn(layer))
            for interaction in self._interactions
            if layer in interaction.layers()
        ]

        def b_fn():
            total = None
            for interaction, fn in hooks:
                value = fn()
                if self.debug_enabled:
                    print(
                        f"[DetailedSumSeparableFunction][b] "
                        f"{interaction.__class__.__name__}: {value}"
                    )
                total = value if total is None else total + value
            if total is None:
                return torch.zeros_like(layer.state)
            if self.debug_enabled:
                print(f"[DetailedSumSeparableFunction][b] total: {total}")
            return total

        return b_fn


class FlexibleResistiveInputLayer(InputLayer):
    """Input layer with a configurable default duplication mode."""

    def __init__(self, shape, gain, batch_size=1, device=None, default_mode="train"):
        super().__init__(shape, batch_size=batch_size, device=device)
        self._gain = gain
        self._default_mode = default_mode

    def set_mode(self, mode: str) -> None:
        if mode not in {"train", "test"}:
            raise ValueError(f"Unknown input mode: {mode}")
        self._default_mode = mode

    def mode(self) -> str:
        return self._default_mode

    def set_input(self, input_values, mode=None):
        chosen_mode = self._default_mode if mode is None else mode
        if chosen_mode == "train":
            self._state = self._gain * torch.cat((input_values, -input_values), 1)
        elif chosen_mode == "test":
            self._state = self._gain * torch.cat((input_values, torch.zeros_like(input_values)), 1)
        else:
            raise ValueError(f"Unknown input mode: {chosen_mode}")


class ConvLayer(NonlinearResistiveLayer):
    """Simple linear layer used for convolutional stages."""

    def __init__(self, shape, batch_size = 1, device=None, non_linearity = None):
        super().__init__(shape, batch_size=batch_size, device=device, non_linearity = non_linearity)


class FlexibleDeepResistiveEnergy(DetailedSumSeparableFunction):
    """
    Drop-in replacement for DeepResistiveEnergy using FlexibleResistiveInputLayer.
    """

    def __init__(
        self,
        layer_shapes,
        weight_gains,
        input_gain,
        non_linearity,
        exponential_diode_param,
        quadratic_diode_param,
        hard_sigmoid_param=None,
        voltage_amp=1.0,
        current_amp=1.0,
        weight_min=None,
        weight_max=None,
        weight_init_mode="kaiming_uniform",
        input_mode="train",
        conv_pipeline=None,
        pooling_mode="avg",
    ):
        self._input_amplifier = input_gain
        self._voltage_amp = voltage_amp
        self._current_amp = current_amp
        self._non_linearity = non_linearity

        self._layer_shapes = layer_shapes
        self._weight_gains = weight_gains
        self._weight_min = weight_min
        self._weight_max = weight_max
        self._weight_init_mode = weight_init_mode
        self._conv_pipeline = list(conv_pipeline or [])
        has_pooling_stage = any(
            conf.get("mode", "convolution") == "pooling" for conf in self._conv_pipeline
        )
        self._pooling_mode = pooling_mode if has_pooling_stage else None
        if has_pooling_stage and self._pooling_mode is None:
            self._pooling_mode = "avg"

        num_conv_stages = len(self._conv_pipeline)
        if num_conv_stages and len(layer_shapes) < num_conv_stages + 2:
            raise ValueError(
                "layer_shapes must specify input, conv stages, and at least one dense/output layer."
            )

        input_shape = layer_shapes[0]
        conv_stage_shapes = layer_shapes[1 : 1 + num_conv_stages]
        hidden_shapes = layer_shapes[1 + num_conv_stages : -1]
        output_shape = layer_shapes[-1]

        input_layer = FlexibleResistiveInputLayer(
            input_shape, gain=input_gain, device=None, default_mode=input_mode
        )

        convpool_layers = []
        convpool_modes = []
        pool_interaction_cls = None
        if self._conv_pipeline:
            if has_pooling_stage and self._pooling_mode not in POOLING_INTERACTIONS:
                raise ValueError(
                    f"Unknown pooling_mode '{self._pooling_mode}', "
                    f"expected one of {tuple(POOLING_INTERACTIONS.keys())}"
                )
            if has_pooling_stage:
                pool_interaction_cls = POOLING_INTERACTIONS[self._pooling_mode]
            for conf, shape in zip(self._conv_pipeline, conv_stage_shapes):
                mode = conf.get("mode", "convolution")
                if mode == "convolution":
                    stage_layer = ConvLayer(shape, device=None, non_linearity=non_linearity)
                elif mode == "pooling":
                    stage_layer = PoolLayer(shape, device=None)
                else:
                    raise ValueError(f"Unknown conv_pipeline mode '{mode}'")
                convpool_layers.append(stage_layer)
                convpool_modes.append(mode)

        hidden_layers = [
            NonlinearResistiveLayer(shape, non_linearity=non_linearity)
            for shape in hidden_shapes
        ]
        output_layer = LinearLayer(output_shape, device=None)
        layers = [input_layer] + convpool_layers + hidden_layers + [output_layer]
        free_layers = [
            layer for layer, mode in zip(convpool_layers, convpool_modes) if mode != "pooling"
        ] + hidden_layers

        ### CONV / POOLING PARAMETERS
        conv_specs = []
        if self._conv_pipeline:
            if len(convpool_layers) != len(self._conv_pipeline):
                raise ValueError("conv_pipeline length must match conv stage shapes.")

            prev_layer = input_layer
            for conf, layer in zip(self._conv_pipeline, convpool_layers):
                kernel = tuple(conf["kernel"])
                stride = conf.get("stride", 1)
                padding = conf.get("padding", 0)
                mode = conf.get("mode", "convolution")
                conv_specs.append(
                    {
                        "pre": prev_layer,
                        "post": layer,
                        "kernel": kernel,
                        "stride": stride,
                        "padding": padding,
                        "mode": mode,
                    }
                )
                prev_layer = layer

        self._validate_conv_shapes(conv_specs)

        if len(weight_gains) < len(conv_specs):
            raise ValueError(
                f"Expected at least {len(conv_specs)} weight_gains for conv stages, "
                f"got {len(weight_gains)}."
            )

        conv_weight_gains = weight_gains[: len(conv_specs)]
        convpool_weights = []
        for spec, gain in zip(conv_specs, conv_weight_gains):
            out_channels = spec["post"]._shape[0]
            in_channels = spec["pre"]._shape[0]
            kh, kw = spec["kernel"]
            if spec["mode"] == "convolution":
                stage_weight = ConvWeight(
                    shape=(out_channels, in_channels, kh, kw),
                    gain=gain,
                    device=None,
                    clamp=True,
                    clamp_min=weight_min,
                    clamp_max=weight_max,
                    init_mode=weight_init_mode,
                )
            elif spec["mode"] == "pooling":
                stage_weight = PoolWeight(
                    shape=(out_channels, in_channels, kh, kw),
                    gain=gain,
                    device=None,
                    clamp=True,
                    clamp_min=weight_min,
                    clamp_max=weight_max,
                )
            else:
                raise ValueError(f"Unknown conv_pipeline mode '{spec['mode']}'")
            convpool_weights.append(stage_weight)

        convpool_interactions = []
        for spec, stage_weight in zip(conv_specs, convpool_weights):
            if spec["mode"] == "convolution":
                interaction = ConvResistive(
                    spec["pre"],
                    spec["post"],
                    stage_weight,
                    padding=spec["padding"],
                    stride=spec["stride"],
                    dilation=1,
                    voltage_amp=voltage_amp,
                    current_amp=current_amp,
                )
                interaction._voltage_amp = self._voltage_amp
            elif spec["mode"] == "pooling":
                if pool_interaction_cls is None:
                    raise ValueError(
                        "pooling_mode must be one of {'max', 'avg'} when "
                        "conv_pipeline contains pooling stages."
                    )
                interaction = pool_interaction_cls(
                    spec["pre"],
                    spec["post"],
                    stage_weight,
                    stride=spec["stride"],
                    voltage_amp=voltage_amp,
                    current_amp=current_amp,
                )
            else:
                raise ValueError(f"Unknown conv_pipeline mode '{spec['mode']}'")
            convpool_interactions.append(interaction)

        dense_source = convpool_layers[-1] if convpool_layers else input_layer
        downstream_layers = [dense_source] + hidden_layers + [output_layer]
        dense_pairs = list(zip(downstream_layers[:-1], downstream_layers[1:]))
        dense_weight_gains = weight_gains[len(conv_specs):]

        if len(dense_weight_gains) < len(dense_pairs):
            raise ValueError(
                f"Expected {len(dense_pairs)} dense weight gains after conv stages, "
                f"got {len(dense_weight_gains)}."
            )

        dense_weights = [
            DenseWeight(
                layer_pre.shape,
                layer_post.shape,
                gain,
                device=None,
                clamp=True,
                clamp_min=weight_min,
                clamp_max=weight_max,
                init_mode=weight_init_mode,
            )
            for (layer_pre, layer_post), gain in zip(dense_pairs, dense_weight_gains)
        ]

        biases = [Bias(layer._shape, 0.0, device=None) for layer in free_layers]
        bias_interactions = [BiasInteraction(layer, bias) for layer, bias in zip(free_layers, biases)]

        weight_interactions = [
            DenseResistive(layer_pre, layer_post, weight, self._voltage_amp, self._current_amp)
            for (layer_pre, layer_post), weight in zip(dense_pairs, dense_weights)
        ]

        if non_linearity == "perfect_diode":
            non_linear_interaction = []
        elif non_linearity == "hard_sigmoid":
            non_linear_interaction = [
                HardSigmoidNonLinearInteraction(
                    layer,
                    hard_sigmoid_param or {},
                    voltage_amp=self._voltage_amp,
                    current_amp=self._current_amp,
                )
                for layer in free_layers[:-1]
            ]
        elif non_linearity == "double_diode_quadratic":
            non_linear_interaction = [
                DoubleQuadraticNonLinearInteraction(
                    layer,
                    quadratic_diode_param,
                    voltage_amp=self._voltage_amp,
                    current_amp=self._current_amp,
                )
                for layer in free_layers
            ]
        elif non_linearity == "double_diode_exponential":
            non_linear_interaction = [
                DoubleExponentialNonLinearInteraction(
                    layer,
                    exponential_diode_param,
                    voltage_amp=self._voltage_amp,
                    current_amp=self._current_amp,
                )
                for layer in free_layers
            ]
        else:
            non_linear_interaction = []
            #raise ValueError(f"Unknown non-linearity: {non_linearity}")

        self._all_params = convpool_weights + dense_weights + biases
        self._trainable_params = [
            param for param in self._all_params if not isinstance(param, PoolWeight)
        ]

        interactions = (
            bias_interactions
            + weight_interactions
            + non_linear_interaction
            + convpool_interactions
        )

        DetailedSumSeparableFunction.__init__(self, layers, self._all_params, interactions)

    def set_input_mode(self, mode: str) -> None:
        input_layer = self.layers()[0]
        if isinstance(input_layer, FlexibleResistiveInputLayer):
            input_layer.set_mode(mode)
        else:
            raise TypeError("Input layer is not FlexibleResistiveInputLayer")

    def params(self):
        return self._trainable_params

    @staticmethod
    def _calc_spatial(shape, kernel_size, stride, padding, dilation=1):
        """Compute convolution output height/width for a single stage."""
        _, h_in, w_in = shape
        kh, kw = kernel_size
        h_out = (h_in + 2 * padding - dilation * (kh - 1) - 1) // stride + 1
        w_out = (w_in + 2 * padding - dilation * (kw - 1) - 1) // stride + 1
        return h_out, w_out

    @classmethod
    def _validate_conv_shapes(cls, conv_specs):
        """Ensure declared layer shapes match the geometry implied by conv_specs."""
        if not conv_specs:
            return
        for spec in conv_specs:
            expected_h, expected_w = cls._calc_spatial(
                spec["pre"]._shape,
                spec["kernel"],
                spec["stride"],
                spec["padding"],
            )
            declared = spec["post"]._shape
            if declared[1] != expected_h or declared[2] != expected_w:
                raise ValueError(
                    f"Layer '{spec['post'].name}' expects spatial "
                    f"{declared[1]}x{declared[2]}, but {spec['pre'].name} "
                    f"with kernel {spec['kernel']}, stride {spec['stride']}, padding {spec['padding']} "
                    f"produces {expected_h}x{expected_w}."
                )


class FlexibleConvWeight(ConvWeight):
    def __init__(self, shape, gain, device, clamp,
                 clamp_min, clamp_max):
        ConvWeight.__init__(self, shape, gain, device, clamp, clamp_min, clamp_max)
        self._gain = gain
        self._clamp = clamp
        self._clamp_min = clamp_min
        self._clamp_max = clamp_max

class ConvResistive(QFunction):
    """Convolutional resistive interaction mirroring DenseResistive logic in conv form."""

    def __init__(
        self,
        layer_pre,
        layer_post,
        conv_weight,
        padding,
        stride,
        dilation,
        voltage_amp=1.0,
        current_amp=1.0,
    ):
        self._layer_pre = layer_pre
        self._layer_post = layer_post
        self._weight = conv_weight
        QFunction.__init__(self, [layer_pre, layer_post], [conv_weight])
        self._P = padding
        self._S = stride
        self._D = dilation
        self._voltage_amp = voltage_amp
        self._current_amp = current_amp

    def _amp_factor(self):
        layer_pre_index = int(self._layer_pre._name[-1])
        return (self._current_amp / self._voltage_amp) ** layer_pre_index

    def _pre_scale(self):
        return 1.0 if self._layer_pre.name == 'Layer_0' else self._voltage_amp

    def _conv_geometry(self):
        weight = self._weight.get()
        C_out, C_in, Kh, Kw = weight.shape
        x = self._layer_pre.state
        _, _, H_in, W_in = x.shape

        H_out = (H_in + 2 * self._P - self._D * (Kh - 1) - 1) // self._S + 1
        W_out = (W_in + 2 * self._P - self._D * (Kw - 1) - 1) // self._S + 1

        return weight, C_out, C_in, Kh, Kw, H_in, W_in, H_out, W_out

    def eval(self, per_sample=True):
        weight, C_out, C_in, Kh, Kw, *_ = self._conv_geometry()

        layer_pre = self._layer_pre.state.clone()
        if self._layer_pre.name != 'Layer_0':
            layer_pre = layer_pre * self._voltage_amp
        layer_post = self._layer_post.state  # / self._layer_post.gain
        layer_post_scaled = layer_post * self._current_amp


        cols = F.unfold(layer_pre, (Kh, Kw), padding=self._P, stride=self._S, dilation=self._D)
        N, C_out, H_out, W_out = layer_post_scaled.shape
        K = C_in * Kh * Kw
        L = H_out * W_out

        patches = cols.transpose(1, 2).unsqueeze(2)
        kernels = weight.view(1, 1, C_out, K)
        targets = layer_post_scaled.view(N, C_out, L).transpose(1, 2).unsqueeze(-1)

        diff2 = (patches - targets).pow(2)
        weighted = diff2 * kernels
        E_per = 0.5 * weighted.sum(dim=(1, 2, 3)) * self._amp_factor()
        return E_per if per_sample else E_per.sum()

    def im2col(self):
        x = self._layer_pre.state
        N = x.shape[0]
        weight, C_out, _, Kh, Kw, _, _, H_out, W_out = self._conv_geometry()

        cols = F.unfold(x, (Kh, Kw), padding=self._P, stride=self._S, dilation=self._D)
        Wflat = weight.view(C_out, -1)
        y = torch.matmul(cols.transpose(1, 2), Wflat.t())
        y = y.transpose(1, 2).reshape(N, C_out, H_out, W_out)
        return y

    def col2im(self):
        weight, C_out, C_in, Kh, Kw, H_in, W_in, H_out, W_out = self._conv_geometry()
        y = self._layer_post.state
        N = y.shape[0]

        L = H_out * W_out
        y_cols = y.reshape(N, C_out, L)
        Wflat = weight.view(C_out, -1)
        cols_pre = torch.matmul(Wflat.t().unsqueeze(0), y_cols)

        x_pre = F.fold(
            cols_pre,
            output_size=(H_in, W_in),
            kernel_size=(Kh, Kw),
            padding=self._P,
            stride=self._S,
            dilation=self._D,
        )
        return x_pre

    def a_im2col(self):
        weight, C_out, C_in, Kh, Kw, _, _, H_out, W_out = self._conv_geometry()
        a_per_ch = weight.view(C_out, -1).sum(dim=1).view(1, C_out, 1, 1)
        return a_per_ch.expand(1, C_out, H_out, W_out)

    def a_col2im(self):
        weight, C_out, C_in, Kh, Kw, H_in, W_in, H_out, W_out = self._conv_geometry()
        y = self._layer_post.state
        y_ones = torch.ones_like(y)
        N = y_ones.shape[0]

        L = H_out * W_out
        y_cols = y_ones.reshape(N, C_out, L)
        Wflat = weight.view(C_out, -1)
        cols_pre = torch.matmul(Wflat.t().unsqueeze(0), y_cols)
        x_pre = F.fold(
            cols_pre,
            output_size=(H_in, W_in),
            kernel_size=(Kh, Kw),
            padding=self._P,
            stride=self._S,
            dilation=self._D,
        )
        return x_pre

    def a_coef_fn(self, layer):
        dictionary = {
            self._layer_pre: self._a_coef_layer_pre,
            self._layer_post: self._a_coef_layer_post,
        }
        return dictionary[layer]

    def b_coef_fn(self, layer):
        dictionary = {
            self._layer_pre: self._b_coef_layer_pre,
            self._layer_post: self._b_coef_layer_post,
        }
        return dictionary[layer]

    def _b_coef_layer_pre(self):
        b_coef = -self.col2im()
        b_coef = b_coef * self._amp_factor() * self._pre_scale() * self._current_amp
        return b_coef

    def _b_coef_layer_post(self):
        b_coef = -self.im2col()
        b_coef = b_coef * self._amp_factor() * self._pre_scale() * self._current_amp
        return b_coef

    def _a_coef_layer_pre(self):
        a_map = self.a_col2im()
        pre_scale = self._pre_scale()
        a_coef = a_map * self._amp_factor() * pre_scale * pre_scale
        return 0.5 * a_coef

    def _a_coef_layer_post(self):
        a_map = self.a_im2col()
        a_coef = a_map * self._amp_factor() * self._current_amp * self._current_amp
        return 0.5 * a_coef

    def _grad_weight(self):
        weight, C_out, C_in, Kh, Kw, *_ = self._conv_geometry()
        x = self._layer_pre.state
        if self._layer_pre.name != 'Layer_0':
            x = x * self._voltage_amp
        y = self._layer_post.state.clone()
        y_rescaled = y * self._current_amp

        cols = F.unfold(x, (Kh, Kw), padding=self._P, stride=self._S, dilation=self._D)
        N, C_out, H_out, W_out = y.shape
        K = C_in * Kh * Kw
        L = H_out * W_out

        patches = cols.transpose(1, 2).unsqueeze(2)
        targets = y_rescaled.reshape(N, C_out, L).transpose(1, 2).unsqueeze(-1)
        diff2 = (patches - targets).pow(2)
        grad_weight = 0.5 * diff2.sum(dim=1).mean(dim=0)

        layer_pre_index = int(self._layer_pre._name[-1])
        amp = (self._current_amp / self._voltage_amp) ** layer_pre_index
        return grad_weight.view(C_out, C_in, Kh, Kw) * amp

    def grad_param_fn(self, param):
        dictionary = {self._weight: self._grad_weight}
        return dictionary[param]
    
