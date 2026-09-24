"""Wider CIFAR L8: coupled analog convolutions and analog dense readout.

VoltageClampedConvBlock and AnalogSolverConfig are carried forward from the
exploratory CIFAR L12 implementation, using this checkout's resistive engine.
"""
from __future__ import annotations
from contextlib import contextmanager
from dataclasses import MISSING, dataclass, fields
import math
from typing import Any, Iterator, Mapping, Sequence
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.modules.batchnorm import _BatchNorm
from labs.custom_minimizer import CustomQuadraticMinimizer, MinimizerSettings
from model.function.interaction import BiasInteraction, SumSeparableFunction
from model.resistive.interaction import ConvResistive, DenseResistive
from model.resistive.layer import ConvLayer, ResistiveInputLayer
from model.variable.layer import LinearLayer
from model.variable.parameter import Bias, ConvWeight, DenseWeight

@dataclass(frozen=True)
class AnalogSolverConfig:
    """Validated fixed-iteration perfect-diode solver configuration."""

    mode: str
    non_linearity: str
    voltage_amp: float
    current_amp: float
    quadratic_diode_param: Mapping[str, Any]
    exponential_diode_param: Mapping[str, Any]
    hard_sigmoid_param: Mapping[str, Any]
    minimizer_settings: MinimizerSettings
    adaptive_equilibrium: bool = False
    overrelaxation_factor: float = 1.0
    double_diode_updater: str = "custom"
    single_diode_updater: str = "custom"
    iv_data: Any = None
    iv_data_path: str | None = None
    damping: float = 0.5
    experimental_newton_max_steps: int = 100

    def __post_init__(self) -> None:
        if self.mode != "asynchronous":
            raise ValueError(
                "CIFAR hybrid blocks require minimizer mode='asynchronous'; "
                f"got {self.mode!r}."
            )
        if self.non_linearity != "perfect_diode":
            raise ValueError(
                "The first CIFAR benchmark requires non_linearity='perfect_diode'; "
                f"got {self.non_linearity!r}."
            )
        if self.adaptive_equilibrium:
            raise ValueError("CIFAR BPTT requires fixed iterations, not adaptive equilibrium.")
        for name, value in (
            ("voltage_amp", self.voltage_amp),
            ("current_amp", self.current_amp),
        ):
            if not math.isfinite(float(value)) or float(value) <= 0.0:
                raise ValueError(
                    f"Local CIFAR blocks require finite positive {name}; got {value!r}."
                )
        for name, value in (
            ("quadratic_diode_param", self.quadratic_diode_param),
            ("exponential_diode_param", self.exponential_diode_param),
            ("hard_sigmoid_param", self.hard_sigmoid_param),
        ):
            if not isinstance(value, Mapping):
                raise TypeError(f"Expected explicit {name} dictionary, got {type(value).__name__}.")

    @classmethod
    def from_mapping(cls, config: Mapping[str, Any]) -> "AnalogSolverConfig":
        """Parse a strict solver config while allowing a nested settings block."""

        minimizer_raw = config.get("minimizer", {})
        if not isinstance(minimizer_raw, Mapping):
            raise TypeError("Expected model minimizer configuration to be a mapping.")
        settings_raw = minimizer_raw.get("settings", minimizer_raw)
        if not isinstance(settings_raw, Mapping):
            raise TypeError("Expected minimizer.settings to be a mapping.")

        settings_values: dict[str, Any] = {}
        missing_settings: list[str] = []
        for field in fields(MinimizerSettings):
            if field.name in settings_raw:
                settings_values[field.name] = settings_raw[field.name]
            elif field.default is not MISSING:
                settings_values[field.name] = field.default
            else:
                missing_settings.append(field.name)
        if missing_settings:
            raise KeyError(
                "Missing explicit minimizer settings: " + ", ".join(missing_settings)
            )

        required = (
            "mode",
            "non_linearity",
            "voltage_amp",
            "current_amp",
            "quadratic_diode_param",
            "exponential_diode_param",
            "hard_sigmoid_param",
        )
        missing = [name for name in required if name not in config]
        if missing:
            raise KeyError("Missing explicit analog solver keys: " + ", ".join(missing))

        return cls(
            mode=str(config["mode"]),
            non_linearity=str(config["non_linearity"]),
            voltage_amp=float(config["voltage_amp"]),
            current_amp=float(config["current_amp"]),
            quadratic_diode_param=dict(config["quadratic_diode_param"]),
            exponential_diode_param=dict(config["exponential_diode_param"]),
            hard_sigmoid_param=dict(config["hard_sigmoid_param"]),
            minimizer_settings=MinimizerSettings(**settings_values),
            adaptive_equilibrium=bool(minimizer_raw.get("adaptive_equilibrium", False)),
            overrelaxation_factor=float(minimizer_raw.get("overrelaxation_factor", 1.0)),
            double_diode_updater=str(minimizer_raw.get("double_diode_updater", "custom")),
            single_diode_updater=str(minimizer_raw.get("single_diode_updater", "custom")),
            iv_data=minimizer_raw.get("iv_data"),
            iv_data_path=minimizer_raw.get("iv_data_path"),
            damping=float(minimizer_raw.get("damping", 0.5)),
            experimental_newton_max_steps=int(
                minimizer_raw.get("experimental_newton_max_steps", 100)
            ),
        )


def _inverse_softplus(value: float) -> float:
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"Expected a finite positive softplus value, got {value!r}.")
    # log(expm1(value)) overflows for the gain=100 benchmark.  This equivalent
    # expression is stable over the full positive floating-point range.
    return value + math.log1p(-math.exp(-value))


def _require_positive_int(value: int, name: str) -> int:
    normalized = int(value)
    if normalized <= 0:
        raise ValueError(f"Expected {name} > 0, got {value!r}.")
    return normalized


def _iteration_schedule(
    value: int | Sequence[int], block_count: int, name: str
) -> tuple[int, ...]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        schedule = tuple(
            _require_positive_int(iterations, f"{name}[{index}]")
            for index, iterations in enumerate(value)
        )
        if len(schedule) != block_count:
            raise ValueError(
                f"Expected {name} to contain {block_count} entries, got {schedule!r}."
            )
        return schedule
    iterations = _require_positive_int(value, name)
    return (iterations,) * block_count


class VoltageClampedConvBlock(nn.Module):
    """A voltage-mode local analog block with one or more convolution edges."""

    def __init__(
        self,
        *,
        logical_input_channels: int,
        analog_channels: Sequence[int],
        spatial_size: int,
        solver: AnalogSolverConfig,
        input_gain_init: float,
        weight_gain: float,
        weight_init_mode: str,
        weight_min: float,
        weight_max: float,
        device: torch.device | str,
        input_gain_mode: str = 'softplus',
    ) -> None:
        super().__init__()
        self.logical_input_channels = _require_positive_int(
            logical_input_channels, "logical_input_channels"
        )
        self.analog_channels = tuple(int(value) for value in analog_channels)
        if not self.analog_channels or any(value <= 0 for value in self.analog_channels):
            raise ValueError(
                "Expected at least one positive analog channel count, "
                f"got {analog_channels!r}."
            )
        if any(value % 2 for value in self.analog_channels):
            raise ValueError(
                "Every analog channel count must be even for differential states."
            )
        self.spatial_size = _require_positive_int(spatial_size, "spatial_size")
        self.solver = solver
        self.weight_min = float(weight_min)
        self.weight_max = float(weight_max)
        if not (
            math.isfinite(self.weight_min)
            and math.isfinite(self.weight_max)
            and 0.0 <= self.weight_min < self.weight_max
        ):
            raise ValueError(
                "Expected finite conductance bounds 0 <= weight_min < weight_max; "
                f"got {(weight_min, weight_max)!r}."
            )
        if not math.isfinite(float(weight_gain)) or float(weight_gain) < 0.0:
            raise ValueError(f"Expected finite weight_gain >= 0, got {weight_gain!r}.")

        target_device = torch.device(device)
        physical_input_channels = 2 * self.logical_input_channels
        shape = (physical_input_channels, self.spatial_size, self.spatial_size)
        self.boundary_layer = ResistiveInputLayer(
            shape, gain=1.0, batch_size=1, device=target_device
        )
        self._free_layers = [
            ConvLayer(
                (channels, self.spatial_size, self.spatial_size),
                batch_size=1,
                device=target_device,
                non_linearity="perfect_diode",
            )
            for channels in self.analog_channels
        ]

        # ConvResistive derives amplification position from the layer suffix.
        # Each local circuit must therefore restart its names at Layer_0.
        self.boundary_layer._name = "Layer_0"
        for index, layer in enumerate(self._free_layers, start=1):
            layer._name = f"Layer_{index}"

        channel_pairs = tuple(
            zip(
                (physical_input_channels, *self.analog_channels[:-1]),
                self.analog_channels,
            )
        )
        self.weights: list[ConvWeight] = []
        for index, (channels_in, channels_out) in enumerate(channel_pairs):
            weight = ConvWeight(
                shape=(channels_out, channels_in, 3, 3),
                gain=float(weight_gain),
                device=target_device,
                clamp=True,
                clamp_min=self.weight_min,
                clamp_max=self.weight_max,
                init_mode=str(weight_init_mode),
            )
            weight.name = f"ConvWeight_{index}"
            weight.state.requires_grad_(True)
            self.weights.append(weight)

        self.biases: list[Bias] = []
        for index, layer in enumerate(self._free_layers):
            bias = Bias(layer.shape, gain=0.0, device=target_device)
            bias.name = f"Bias_{index}"
            bias.state.requires_grad_(False)
            self.biases.append(bias)

        layers = [self.boundary_layer, *self._free_layers]
        self.interactions = [
            ConvResistive(
                layers[index],
                layers[index + 1],
                self.weights[index],
                padding=1,
                stride=1,
                dilation=1,
                voltage_amp=solver.voltage_amp,
                current_amp=solver.current_amp,
            )
            for index in range(len(self._free_layers))
        ]
        bias_interactions = [
            BiasInteraction(layer, bias)
            for layer, bias in zip(self._free_layers, self.biases)
        ]
        self.energy = SumSeparableFunction(
            layers=layers,
            params=[*self.weights, *self.biases],
            interactions=[*self.interactions, *bias_interactions],
        )
        self.minimizer = CustomQuadraticMinimizer(
            self.energy,
            free_layers=self._free_layers,
            num_iterations=1,
            mode=solver.mode,
            non_linearity=solver.non_linearity,
            quadratic_diode_param=dict(solver.quadratic_diode_param),
            exponential_diode_param=dict(solver.exponential_diode_param),
            voltage_amp=solver.voltage_amp,
            current_amp=solver.current_amp,
            hard_sigmoid_param=dict(solver.hard_sigmoid_param),
            iv_data=solver.iv_data,
            iv_data_path=solver.iv_data_path,
            double_diode_updater=solver.double_diode_updater,
            adaptive_equilibrium=solver.adaptive_equilibrium,
            overrelaxation_factor=solver.overrelaxation_factor,
            single_diode_updater=solver.single_diode_updater,
            minimizer_settings=solver.minimizer_settings,
            damping=solver.damping,
            experimental_newton_max_steps=solver.experimental_newton_max_steps,
        )

        if input_gain_mode not in ('softplus', 'fixed', 'log'):
            raise ValueError('conv_input_gain_mode must be softplus, fixed or log')
        if not math.isfinite(float(input_gain_init)) or input_gain_init <= 0:
            raise ValueError('input_gain_init must be finite and positive')
        self.input_gain_mode = input_gain_mode
        self.input_gain_base = float(input_gain_init)
        initial_raw = (_inverse_softplus(float(input_gain_init)) if input_gain_mode == 'softplus'
                       else 0. if input_gain_mode == 'log' else float(input_gain_init))
        self._input_gain_raw = nn.Parameter(
            torch.tensor(
                initial_raw,
                dtype=torch.float32,
                device=target_device,
            ), requires_grad=input_gain_mode != 'fixed',
        )

    @property
    def input_gain(self) -> torch.Tensor:
        """Strictly positive digital-to-analog boundary voltage gain."""

        if self.input_gain_mode == 'fixed':
            return self._input_gain_raw
        if self.input_gain_mode == 'log':
            return self.input_gain_base * self._input_gain_raw.exp()
        return F.softplus(self._input_gain_raw)

    @property
    def drive_scale(self) -> torch.Tensor:
        """Compatibility name for the trainable input-mode voltage gain."""

        return self.input_gain

    @property
    def _drive_scale_raw(self) -> nn.Parameter:
        return self._input_gain_raw

    def free_layers(self) -> list[ConvLayer]:
        return list(self._free_layers)

    def analog_parameters(self) -> list[torch.Tensor]:
        return [weight.state for weight in self.weights]

    def reset_state_(self, batch_size: int, device: torch.device | str | None = None) -> None:
        target = torch.device(device) if device is not None else self.weights[0].state.device
        for layer in self._free_layers:
            layer.init_state(int(batch_size), target)

    def detach_state_(self) -> None:
        self.boundary_layer.state = self.boundary_layer.state.detach()
        for layer in self._free_layers:
            layer.state = layer.state.detach()

    def set_input_voltage(self, values: torch.Tensor) -> None:
        expected = (
            values.size(0),
            self.logical_input_channels,
            self.spatial_size,
            self.spatial_size,
        )
        if tuple(values.shape) != expected:
            raise ValueError(
                f"Expected logical boundary input shape {expected}, got {tuple(values.shape)}."
            )
        # ResistiveInputLayer performs the physical [v, -v] duplication.  Its
        # own fixed gain is one; the trainable softplus gain sets the voltage.
        self.boundary_layer.set_input(self.input_gain * values)

    @staticmethod
    def differential_decode(state: torch.Tensor) -> torch.Tensor:
        if state.ndim != 4 or state.size(1) % 2:
            raise ValueError(
                "Expected a four-dimensional state with an even channel count for decode; "
                f"got shape={tuple(state.shape)}."
            )
        positive, negative = state.chunk(2, dim=1)
        return positive - negative

    def forward(
        self,
        values: torch.Tensor,
        *,
        num_iterations: int,
        reset: bool,
    ) -> torch.Tensor:
        iterations = _require_positive_int(num_iterations, "num_iterations")
        if reset:
            self.reset_state_(values.size(0), values.device)
        else:
            expected_batch = values.size(0)
            for layer in self._free_layers:
                if layer.state.size(0) != expected_batch:
                    raise ValueError(
                        "Cannot replay a detached analog state with a different batch size: "
                        f"state batch={layer.state.size(0)}, input batch={expected_batch}."
                    )
        self.set_input_voltage(values)
        self.minimizer.num_iterations = iterations
        self.minimizer.compute_equilibrium()
        return self.differential_decode(self._free_layers[-1].state)

    def project_weights_(self) -> None:
        with torch.no_grad():
            for weight in self.weights:
                weight.clamp_()

    def zero_analog_grad_(self, set_to_none: bool = True) -> None:
        for tensor in self.analog_parameters():
            if set_to_none:
                tensor.grad = None
            elif tensor.grad is not None:
                tensor.grad.zero_()

    def _apply(self, fn):  # noqa: ANN001 - follows torch.nn.Module's internal API
        super()._apply(fn)
        for layer in (self.boundary_layer, *self._free_layers):
            layer.state = fn(layer.state.detach())
        for parameter in (*self.weights, *self.biases):
            requires_grad = parameter.state.requires_grad
            parameter.state = fn(parameter.state.detach()).requires_grad_(requires_grad)
        return self


@contextmanager
def batchnorm_minibatch_stats_only(model: nn.Module) -> Iterator[None]:
    """Use minibatch statistics without updating BatchNorm running buffers."""

    tracking_modes: list[tuple[_BatchNorm, bool]] = []
    for module in model.modules():
        if isinstance(module, _BatchNorm):
            tracking_modes.append((module, bool(module.track_running_stats)))
            module.track_running_stats = False
    try:
        yield
    finally:
        for module, track_running_stats in tracking_modes:
            module.track_running_stats = track_running_stats


class AnalogDenseReadout(nn.Module):
    """A clamped differential input connected to unconstrained linear outputs."""

    def __init__(self, features, classes, config, device):
        super().__init__()
        self.boundary = ResistiveInputLayer((2 * features,), gain=1., device=device)
        self.output = LinearLayer((classes,), device=device)
        self.boundary._name, self.output._name = 'Layer_0', 'Layer_1'
        self.weight = DenseWeight((2 * features,), (classes,), gain=config['weight_gain'],
                                  device=device, clamp=True,
                                  clamp_min=config['weight_min'], clamp_max=config['weight_max'],
                                  init_mode=config['weight_init_mode'])
        self.weight.state.requires_grad_(True)
        self.interaction = DenseResistive(self.boundary, self.output, self.weight,
                                         config['solver']['voltage_amp'], config['solver']['current_amp'])
        self._input_gain_raw = nn.Parameter(torch.tensor(
            _inverse_softplus(config['input_gain_init']), device=device))

    def forward(self, values):
        self.boundary.set_input(F.softplus(self._input_gain_raw) * values)
        # Exact single-node-layer coordinate update of the DenseResistive energy.
        # No digital learned matrix, bias, softmax, or post-logit scaling.
        a = self.interaction.a_coef_fn(self.output)()
        b = self.interaction.b_coef_fn(self.output)()
        self.output.state = -b / (2 * a)
        return self.output.state

    def detach_state_(self):
        self.boundary.state = self.boundary.state.detach()
        self.output.state = self.output.state.detach()


def boundary_normalization_settings(config):
    mode = config.get('boundary_normalization', 'batch_norm')
    eps = config.get('batch_norm_eps', [1e-5] * len(config['blocks']))
    trainable = config.get('bn_affine_trainable', True)
    if mode not in ('batch_norm', 'none'):
        raise ValueError('boundary_normalization must be batch_norm or none')
    if not isinstance(trainable, bool):
        raise ValueError('bn_affine_trainable must be boolean')
    if len(eps) != len(config['blocks']) or any(not math.isfinite(e) or e <= 0 for e in eps):
        raise ValueError('batch_norm_eps requires one positive finite value per block')
    return mode, tuple(eps), trainable


class CifarL8Analog(nn.Module):
    def __init__(self, config, device='cpu'):
        super().__init__()
        solver = AnalogSolverConfig.from_mapping(config['solver'])
        self.boundary_normalization, bn_eps, bn_trainable = boundary_normalization_settings(config)
        normalization = config.get('block_output_normalization', 'none')
        if normalization not in ('none', 'voltage'):
            raise ValueError('block_output_normalization must be none or voltage')
        self.block_output_scales = tuple(
            solver.voltage_amp ** (len(logical) - 1) if normalization == 'voltage' else 1.
            for logical in config['blocks']
        )
        channels, size = 3, config.get('input_size', 32)
        blocks, bridges = [], []
        for index, logical in enumerate(config['blocks']):
            blocks.append(VoltageClampedConvBlock(
                logical_input_channels=channels, analog_channels=[2*c for c in logical],
                spatial_size=size, solver=solver, input_gain_init=config['input_gain_init'],
                input_gain_mode=config.get('conv_input_gain_mode', 'softplus'),
                weight_gain=config['weight_gain'], weight_init_mode=config['weight_init_mode'],
                weight_min=config['weight_min'], weight_max=config['weight_max'], device=device))
            channels, size = logical[-1], size // 2
            norm = (nn.BatchNorm2d(channels, eps=bn_eps[index], affine=True, device=device)
                    if self.boundary_normalization == 'batch_norm' else nn.Identity())
            if isinstance(norm, _BatchNorm):
                norm.requires_grad_(bn_trainable)
            bridges.append(nn.Sequential(nn.MaxPool2d(2), norm))
        self.analog_blocks = nn.ModuleList(blocks)
        self.bridges = nn.ModuleList(bridges)
        self.head = AnalogDenseReadout(channels*size*size, config['num_classes'], config, device)

    def forward(self, x, iterations, reset=True):
        if len(iterations) != len(self.analog_blocks):
            raise ValueError('One iteration count required per convolution block')
        for block, bridge, count, scale in zip(
            self.analog_blocks, self.bridges, iterations, self.block_output_scales
        ):
            x = block(x, num_iterations=count, reset=reset)
            if scale != 1.:
                x = x / scale
            x = bridge(x)
        return self.head(x.flatten(1))

    def detach_state_(self):
        for block in self.analog_blocks:
            block.detach_state_()
        self.head.detach_state_()

    def bptt(self, x, iterations):
        if not self.training:
            raise ValueError('BPTT requires training mode')
        with torch.no_grad():
            free = self(x, iterations, reset=True).detach()
        self.detach_state_()
        with batchnorm_minibatch_stats_only(self):
            tracked = self(x, iterations, reset=False)
        return free, tracked

    def named_conductances(self):
        for i, block in enumerate(self.analog_blocks):
            for j, weight in enumerate(block.weights):
                yield f'blocks.{i}.weights.{j}', weight.state
        yield 'head.weight', self.head.weight.state

    def trainable_tensors(self):
        return {n: p for n, p in (*self.named_parameters(), *self.named_conductances())
                if p.requires_grad}

    def optimizer_groups(self, config):
        rates = [lr for block in config['conv_learning_rates'] for lr in block]
        rates.append(config['head_learning_rate'])
        conductances = list(self.named_conductances())
        if len(rates) != len(conductances):
            raise ValueError('Learning rates do not cover all conductances')
        groups = [{'params': [p], 'lr': lr, 'name': name}
                  for (name,p),lr in zip(conductances,rates)]
        for name,p in self.named_parameters():
            if not p.requires_grad:
                continue
            if '_input_gain_raw' in name:
                rate = (config.get('conv_gain_learning_rate', config['gain_learning_rate'])
                        if name.startswith('analog_blocks.') else config['gain_learning_rate'])
            else:
                rate = config['bn_learning_rate']
            groups.append({'params':[p], 'lr':rate, 'name':name})
        return groups

    def project_(self):
        for block in self.analog_blocks:
            block.project_weights_()
        with torch.no_grad():
            self.head.weight.clamp_()

    def snapshot(self):
        return {'module': {n:p.detach().cpu().clone() for n,p in self.state_dict().items()},
                'conductances': {n:p.detach().cpu().clone() for n,p in self.named_conductances()}}

    def restore(self, state):
        self.load_state_dict(state['module'], strict=True)
        conductances = dict(self.named_conductances())
        if conductances.keys() != state['conductances'].keys():
            raise ValueError('Analog checkpoint keys differ')
        with torch.no_grad():
            for name,p in conductances.items():
                p.copy_(state['conductances'][name])

    def restore_initializer(self, state):
        """Import original softplus epoch-zero state, preserving physical gains."""
        module = dict(state['module'])
        omitted = []
        for index, block in enumerate(self.analog_blocks):
            if block.input_gain_mode == 'softplus':
                continue
            key = f'analog_blocks.{index}._input_gain_raw'
            source = module[key]
            expected = torch.full_like(source, _inverse_softplus(block.input_gain_base))
            if not torch.equal(source, expected):
                raise ValueError('Gain conversion requires the original softplus initializer')
            module[key] = (torch.zeros_like(source) if block.input_gain_mode == 'log'
                           else torch.full_like(source, block.input_gain_base))
        if self.boundary_normalization == 'none':
            permitted = {f'bridges.{i}.1.{suffix}' for i in range(len(self.bridges))
                         for suffix in ('weight', 'bias', 'running_mean', 'running_var',
                                        'num_batches_tracked')}
            omitted = sorted(set(module) & permitted)
            for key in omitted:
                del module[key]
        self.restore({'module': module, 'conductances': state['conductances']})
        return omitted
