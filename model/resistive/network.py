import math
from tokenize import Double
from numbers import Integral, Real

import torch
from torch.nn import Hardsigmoid

from model.function.interaction import HardSigmoidNonLinearInteraction, SumSeparableFunction
from model.resistive.layer import PoolLayer, ResistiveInputLayer, NonlinearResistiveLayer, ConvLayer
from model.variable.layer import LinearLayer
from model.variable.parameter import Bias, DenseWeight, ConvWeight, PoolWeight
# from model.resistive.parameter import TiedDenseWeight as DenseWeight
from model.function.interaction import BiasInteraction
from model.resistive.interaction import (
    BasePoolResistive,
    AveragePoolResistive,
    MaxPoolResistive,
    DenseResistive,
    SignedDenseResistive,
    ConvResistive,
)
from model.resistive.low_rank import (
    PassiveLowRankAdapterConfig,
    PassiveLowRankDenseWeight,
    parse_passive_low_rank_adapter,
)
from model.function.interaction import (
    DoubleQuadraticNonLinearInteraction,
    DoubleExponentialNonLinearInteraction,
    SingleExponentialNonLinearInteraction,
    LpwNonLinearInteraction,
)
POOLING_INTERACTIONS = {
    "max": MaxPoolResistive,
    "avg": AveragePoolResistive,
}



class DeepResistiveEnergy(SumSeparableFunction):
    """Scalar energy of a deep resistive network (DRN).

    For passive edges this is power-like. Differential active two-ports use a
    positive layer-weighted potential and expose the metric separately from
    physical device dissipation. Successive layers are densely connected.
    """

    def __init__(self, layer_shapes, weight_gains, input_gain,
                 non_linearity, exponential_diode_param, quadratic_diode_param, hard_sigmoid_param,
                 voltage_amp, current_amp,
                 weight_min=None, weight_max=None,
                 weight_init_mode='kaiming_uniform', conv_pipeline=None,
                 pooling_mode="avg", passive_low_rank_adapter=None,
                 differential_dense_edges=None, include_biases=True,
                 legacy_process_index_amplification=False):
        """Creates an instance of a dense Hopfield network

        Args:
            layer_shapes (list of tuple of ints): the shapes of the tensors representing the layers of the network
            weight_gains (list of float32): the gains of the weights used at initialization
            input_gain (float): the gain of input variables (input voltage sources)
            non_linearity (str): type of non-linearity ('perfect_diode', 'lpw_diode', 'hard_sigmoid', 'linear')
            voltage_amp (float): voltage amplification factor.
            current_amp (float): current amplification factor.
            weight_min (float, optional): Lower clamp bound applied to the conductance weights.
            weight_max (float, optional): Upper clamp bound applied to the conductance weights.
        """

        adapter_config = parse_passive_low_rank_adapter(
            passive_low_rank_adapter
        )
        self._passive_low_rank_adapter_config = adapter_config
        self._input_amplifier = input_gain
        self._voltage_amp = voltage_amp
        self._current_amp = current_amp
        self._non_linearity = non_linearity
        # Store diode parameter dictionaries so downstream utilities (e.g., Monitor)
        # can introspect saturation bounds without threading them through manually.
        self._quadratic_diode_param = dict(quadratic_diode_param or {})
        self._exponential_diode_param = dict(exponential_diode_param or {})
        self._hardsigmoid_params = dict(hard_sigmoid_param or {})
        self._layer_shapes = layer_shapes
        self._weight_gains = weight_gains
        self._weight_min = weight_min
        self._weight_max = weight_max
        self._weight_init_mode = weight_init_mode
        try:
            raw_differential_edges = (
                ()
                if differential_dense_edges is None
                else tuple(differential_dense_edges)
            )
        except TypeError as exc:
            raise ValueError(
                "Expected differential_dense_edges to be an iterable of "
                "unique non-negative integer edge indices. Provided value: "
                f"{differential_dense_edges!r}."
            ) from exc
        invalid_differential_edge_values = tuple(
            edge
            for edge in raw_differential_edges
            if isinstance(edge, bool)
            or not isinstance(edge, Integral)
            or edge < 0
        )
        if invalid_differential_edge_values or len(
            set(raw_differential_edges)
        ) != len(raw_differential_edges):
            raise ValueError(
                "Expected differential_dense_edges to be an iterable of "
                "unique non-negative integer edge indices. Provided value: "
                f"{differential_dense_edges!r}."
            )
        self._differential_dense_edges = tuple(
            sorted(raw_differential_edges)
        )
        if not isinstance(include_biases, bool):
            raise ValueError(
                "Expected include_biases to be a bool. Provided value: "
                f"{include_biases!r}."
            )
        self._include_biases = include_biases
        if not isinstance(legacy_process_index_amplification, bool):
            raise ValueError(
                "Expected legacy_process_index_amplification to be a bool. "
                f"Provided value: {legacy_process_index_amplification!r}."
            )
        self._legacy_process_index_amplification = (
            legacy_process_index_amplification
        )
        if adapter_config is not None:
            if conv_pipeline is None or (
                isinstance(conv_pipeline, (list, tuple))
                and not conv_pipeline
            ):
                self._conv_pipeline = []
            else:
                raise ValueError(
                    "Expected conv_pipeline to be empty when "
                    "passive_low_rank_adapter is enabled. "
                    f"Provided value: {conv_pipeline!r}."
                )
        else:
            self._conv_pipeline = list(conv_pipeline or [])
        has_pooling_stage = any(conf.get("mode") == "pooling" for conf in self._conv_pipeline)
        self._pooling_mode = pooling_mode if has_pooling_stage else None
        if has_pooling_stage and self._pooling_mode is None:
            self._pooling_mode = "avg"

        if self._differential_dense_edges:
            for name, value in (
                ("voltage_amp", self._voltage_amp),
                ("current_amp", self._current_amp),
            ):
                if (
                    isinstance(value, bool)
                    or not isinstance(value, Real)
                    or not math.isfinite(float(value))
                    or float(value) <= 0.0
                ):
                    raise ValueError(
                        f"Expected {name} to be a finite positive amplifier "
                        "magnitude for differential dense edges. "
                        f"Provided value: {value!r}."
                    )
            if adapter_config is not None:
                raise ValueError(
                    "Expected differential_dense_edges not to be combined "
                    "with a low-rank adapter. Provided value: "
                    f"differential_dense_edges={self._differential_dense_edges!r}, "
                    "enabled_adapters=('passive_low_rank_adapter',)."
                )
            if self._conv_pipeline:
                raise ValueError(
                    "Expected conv_pipeline to be empty when differential "
                    "dense edges are enabled. Provided value: "
                    f"{self._conv_pipeline!r}."
                )
            if self._include_biases:
                raise ValueError(
                    "Expected include_biases=false when differential dense "
                    f"edges are enabled. Provided value: {include_biases!r}."
                )
            if self._legacy_process_index_amplification:
                raise ValueError(
                    "Expected legacy_process_index_amplification=false when "
                    "differential dense edges are enabled. Provided value: "
                    f"{legacy_process_index_amplification!r}."
                )
            if self._non_linearity not in {"linear", "perfect_diode"}:
                raise ValueError(
                    "Expected non_linearity to be 'linear' or "
                    "'perfect_diode' when differential dense edges are "
                    f"enabled. Provided value: {self._non_linearity!r}."
                )
            effective_weight_min = 0.0
            if weight_min is not None:
                if (
                    isinstance(weight_min, bool)
                    or not isinstance(weight_min, Real)
                    or not math.isfinite(float(weight_min))
                    or float(weight_min) < 0.0
                ):
                    raise ValueError(
                        "Expected weight_min to be None or a finite "
                        "non-negative conductance for differential dense "
                        f"edges. Provided value: {weight_min!r}."
                    )
                effective_weight_min = float(weight_min)
            if weight_max is not None and (
                isinstance(weight_max, bool)
                or not isinstance(weight_max, Real)
                or not math.isfinite(float(weight_max))
                or float(weight_max) <= effective_weight_min
            ):
                raise ValueError(
                    "Expected weight_max to be None or a finite conductance "
                    "strictly greater than the non-negative differential "
                    "weight_min. Provided value: "
                    f"weight_min={weight_min!r}, weight_max={weight_max!r}."
                )

        if adapter_config is not None:
            self._init_passive_low_rank_adapter(adapter_config)
            return


        num_conv_stages = len(self._conv_pipeline)
        if num_conv_stages and len(layer_shapes) < num_conv_stages + 2:
            raise ValueError(
                "layer_shapes must specify input, conv stages, and at least one dense/output layer."
            )

        input_shape = layer_shapes[0]
        conv_stage_shapes = layer_shapes[1 : 1 + num_conv_stages]
        hidden_shapes = layer_shapes[1 + num_conv_stages : -1]
        output_shape = layer_shapes[-1]

        input_layer = ResistiveInputLayer(input_shape, gain=input_gain, device=None)  # input layer

        convpool_layers = []
        convpool_modes = []
        PoolInteraction = None
        if self._conv_pipeline:
            if has_pooling_stage and self._pooling_mode not in POOLING_INTERACTIONS:
                raise ValueError(f"Unknown pooling_mode '{self._pooling_mode}', expected one of {tuple(POOLING_INTERACTIONS.keys())}")
            if has_pooling_stage:
                PoolInteraction = POOLING_INTERACTIONS[self._pooling_mode]
            for conf, shape in zip(self._conv_pipeline, conv_stage_shapes):
                mode = conf.get("mode")
                if mode == "convolution":
                    convpool_layer = ConvLayer(shape, device=None, non_linearity=non_linearity)
                elif mode == "pooling":
                    convpool_layer = PoolLayer(shape, device=None)
                else:
                    raise ValueError(f"Unknown conv_pipeline mode '{mode}'")
                convpool_layers.append(convpool_layer)
                convpool_modes.append(mode)

        hidden_layers = [
            NonlinearResistiveLayer(shape, non_linearity=non_linearity) for shape in hidden_shapes
        ]  # hidden layers
        output_layer = LinearLayer(output_shape, device=None)  # output layer
        layers = [input_layer] + convpool_layers + hidden_layers + [output_layer]



        ### CONV / POOLING PARAMETERS
        conv_specs = []
        if self._conv_pipeline:
            if len(convpool_layers) != len(self._conv_pipeline):
                raise ValueError("conv_pipeline length must match conv stage shapes.")

            prev_layer = input_layer
            for conf, layer in zip(self._conv_pipeline, convpool_layers):
                kernel = tuple(conf["kernel"])
                stride = conf.get("stride")
                padding = conf.get("padding")
                mode = conf.get("mode")
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

        self._validate_conv_shapes(conv_specs) ##maybe I do not really need this?

        conv_weight_gains = weight_gains[: len(conv_specs)]
        convpool_weights = []
        for spec, gain in zip(conv_specs, conv_weight_gains):
            out_channels = spec["post"]._shape[0]
            in_channels = spec["pre"]._shape[0]
            kh, kw = spec["kernel"]
            if spec["mode"] == "convolution":
                convpool_weight = ConvWeight(
                    shape=(out_channels, in_channels, kh, kw),
                    gain=gain,
                    device=None,
                    clamp=True,
                    clamp_min=weight_min,
                    clamp_max=weight_max,
                    init_mode = weight_init_mode
                )
            elif spec["mode"] == "pooling":
                convpool_weight = PoolWeight(
                    shape=(out_channels, in_channels, kh, kw),
                    gain=gain,
                    device=None,
                    clamp=True,
                    clamp_min=weight_min,
                    clamp_max=weight_max,
                    
                )
            convpool_weights.append(convpool_weight)

        convpool_interactions = []
        for spec, convpool_weight in zip(conv_specs, convpool_weights):
            if spec["mode"] == "convolution":
                convpool_interaction = ConvResistive(
                    spec["pre"],
                    spec["post"],
                    convpool_weight,
                    padding=spec["padding"],
                    stride=spec["stride"],
                    dilation=1,
                    voltage_amp=voltage_amp,
                    current_amp=current_amp
                )
            elif spec["mode"] == "pooling":
                if PoolInteraction is None:
                    raise ValueError("pooling_mode must be one of {'max', 'avg'} when conv_pipeline contains pooling stages.")
                convpool_interaction = PoolInteraction(
                    spec["pre"],
                    spec["post"],
                    convpool_weight,
                    stride=spec["stride"],
                    voltage_amp=voltage_amp,
                    current_amp=current_amp
                )
            convpool_interactions.append(convpool_interaction)

        dense_source = convpool_layers[-1] if convpool_layers else input_layer
        downstream_layers = [dense_source] + hidden_layers + [output_layer]
        dense_pairs = list(zip(downstream_layers[:-1], downstream_layers[1:]))
        dense_weight_gains = weight_gains[len(conv_specs):]
        if self._differential_dense_edges and (
            len(dense_weight_gains) != len(dense_pairs)
        ):
            raise ValueError(
                "Expected weight_gains to contain exactly one gain per dense "
                "edge when differential dense edges are enabled. Provided "
                f"value: weight_gains={weight_gains!r}, "
                f"dense_edge_count={len(dense_pairs)}."
            )
        invalid_differential_edges = tuple(
            edge
            for edge in self._differential_dense_edges
            if edge >= len(dense_pairs)
        )
        if invalid_differential_edges:
            raise ValueError(
                "Expected differential_dense_edges to contain unique dense "
                f"edge indices in [0, {len(dense_pairs)}). Provided value: "
                f"{self._differential_dense_edges!r}."
            )
        if self._differential_dense_edges != () and (
            self._differential_dense_edges != tuple(range(len(dense_pairs)))
        ):
            raise ValueError(
                "Expected differential_dense_edges to select every dense "
                "edge so one positive layer metric applies to the complete "
                f"network. Provided value: {self._differential_dense_edges!r}."
            )
        free_layers = [layer for layer, mode in zip(convpool_layers, convpool_modes) if mode != "pooling"] + hidden_layers

        # build the biases
        biases = (
            [Bias(layer._shape, 0., device=None) for layer in free_layers]
            if self._include_biases
            else []
        )
        bias_interactions = [BiasInteraction(layer, bias) for layer, bias in zip(free_layers, biases)]

        # build the weights of the network
        # outs = [True] * (len(edges)-1) + [False]
        dense_weights = []
        dense_weight_sets = []
        for edge_index, ((layer_pre, layer_post), gain) in enumerate(
            zip(dense_pairs, dense_weight_gains)
        ):
            plus = DenseWeight(
                layer_pre.shape,
                layer_post.shape,
                gain,
                device=None,
                clamp=True,
                clamp_min=weight_min,
                clamp_max=weight_max,
                init_mode=weight_init_mode,
            )
            if edge_index in self._differential_dense_edges:
                minus = DenseWeight(
                    layer_pre.shape,
                    layer_post.shape,
                    gain,
                    device=None,
                    clamp=True,
                    clamp_min=weight_min,
                    clamp_max=weight_max,
                    init_mode=weight_init_mode,
                )
                plus.checkpoint_role = "conductance_plus"
                minus.checkpoint_role = "conductance_minus"
                dense_weights.extend((plus, minus))
                dense_weight_sets.append((plus, minus))
            else:
                dense_weights.append(plus)
                dense_weight_sets.append((plus,))
        logical_layer_indices = {
            id(layer): index for index, layer in enumerate(layers)
        }
        self._logical_layer_indices = {
            layer: index for index, layer in enumerate(layers)
        }
        if self._differential_dense_edges:
            for layer in layers[1:]:
                self._layer_energy_scale_at(
                    logical_layer_indices[id(layer)],
                    dtype=layer.state.dtype,
                )
        weight_interactions = []
        for (layer_pre, layer_post), weights in zip(
            dense_pairs,
            dense_weight_sets,
        ):
            common = (
                {}
                if self._legacy_process_index_amplification
                else {
                    "logical_pre_index": logical_layer_indices[id(layer_pre)],
                    "logical_post_index": logical_layer_indices[id(layer_post)],
                }
            )
            if len(weights) == 2:
                interaction = SignedDenseResistive(
                    layer_pre,
                    layer_post,
                    weights[0],
                    weights[1],
                    self._voltage_amp,
                    self._current_amp,
                    **common,
                )
            else:
                interaction = DenseResistive(
                    layer_pre,
                    layer_post,
                    weights[0],
                    self._voltage_amp,
                    self._current_amp,
                    **common,
                )
            weight_interactions.append(interaction)
        
        # include nonlinear interactions for all nonlinear layers (conv + hidden)
        non_linear_layers = [layer for layer, mode in zip(convpool_layers, convpool_modes) if mode != "pooling"] + hidden_layers

        if non_linearity == "perfect_diode":
            non_linear_interaction = []

        elif non_linearity == "lpw_diode":
            non_linear_interaction = [
                LpwNonLinearInteraction(
                    layer,
                    quadratic_diode_param,
                    voltage_amp=self._voltage_amp,
                    current_amp=self._current_amp,
                    logical_layer_index=(
                        None
                        if self._legacy_process_index_amplification
                        else logical_layer_indices[id(layer)]
                    ),
                )
                for layer in non_linear_layers
            ]

        elif non_linearity == "hard_sigmoid":
            non_linear_interaction = [
                HardSigmoidNonLinearInteraction(
                    layer,
                    hard_sigmoid_param,
                    voltage_amp=self._voltage_amp,
                    current_amp=self._current_amp,
                    logical_layer_index=(
                        None
                        if self._legacy_process_index_amplification
                        else logical_layer_indices[id(layer)]
                    ),
                )
                for layer in non_linear_layers
            ]
        elif non_linearity == "double_diode_quadratic":
            non_linear_interaction = [
                DoubleQuadraticNonLinearInteraction(
                    layer,
                    quadratic_diode_param,
                    voltage_amp=self._voltage_amp,
                    current_amp=self._current_amp,
                    logical_layer_index=(
                        None
                        if self._legacy_process_index_amplification
                        else logical_layer_indices[id(layer)]
                    ),
                )
                for layer in non_linear_layers
            ]
        elif non_linearity == "double_diode_exponential":
            non_linear_interaction = [
                DoubleExponentialNonLinearInteraction(
                    layer,
                    exponential_diode_param,
                    voltage_amp=self._voltage_amp,
                    current_amp=self._current_amp,
                    logical_layer_index=(
                        None
                        if self._legacy_process_index_amplification
                        else logical_layer_indices[id(layer)]
                    ),
                )
                for layer in non_linear_layers
            ]
        elif non_linearity == "single_diode_exponential":
            non_linear_interaction = [
                SingleExponentialNonLinearInteraction(
                    layer,
                    exponential_diode_param,
                    voltage_amp=self._voltage_amp,
                    current_amp=self._current_amp,
                    logical_layer_index=(
                        None
                        if self._legacy_process_index_amplification
                        else logical_layer_indices[id(layer)]
                    ),
                )
                for layer in non_linear_layers
            ]
        elif non_linearity == "linear":
            non_linear_interaction = []
        else:
            non_linear_interaction = []
            print("Nonlinear interaction for this non-linearity not defined yet")

        # Track all params for device movement, but expose only trainable ones (exclude PoolWeight)
        self._all_params = convpool_weights + dense_weights + biases
        self._trainable_params = [p for p in self._all_params if not isinstance(p, PoolWeight)]
        self._base_params = list(self._all_params)
        self._adapter_params = []
        interactions = bias_interactions + weight_interactions + non_linear_interaction + convpool_interactions

        # creates an instance of Network; pass all params so set_device moves everything to the right device
        SumSeparableFunction.__init__(self, layers, self._all_params, interactions)

    def _layer_energy_scale_at(self, logical_index, *, dtype):
        """Return the positive diagonal metric for one logical layer."""

        exponent = max(int(logical_index) - 1, 0)
        try:
            ratio = (
                1.0
                if exponent == 0
                else float(self._current_amp) / float(self._voltage_amp)
            )
            scale = 1.0 if exponent == 0 else math.pow(ratio, exponent)
        except (OverflowError, ZeroDivisionError, ValueError) as exc:
            raise ValueError(
                "Expected voltage_amp/current_amp and network depth to yield "
                "a finite positive layer-energy scale. Provided value: "
                f"voltage_amp={self._voltage_amp!r}, "
                f"current_amp={self._current_amp!r}, "
                f"logical_layer_index={logical_index!r}, dtype={dtype!s}."
            ) from exc
        represented = torch.as_tensor(scale, dtype=dtype)
        if (
            not math.isfinite(ratio)
            or ratio <= 0.0
            or not torch.isfinite(represented).item()
            or represented.item() <= 0.0
        ):
            raise ValueError(
                "Expected voltage_amp/current_amp and network depth to yield "
                "a finite positive layer-energy scale. Provided value: "
                f"voltage_amp={self._voltage_amp!r}, "
                f"current_amp={self._current_amp!r}, "
                f"logical_layer_index={logical_index!r}, dtype={dtype!s}."
            )
        return scale

    def layer_energy_scale(self, layer):
        """Return the energy-gradient metric multiplying physical KCL."""

        if not self._differential_dense_edges:
            return 1.0
        logical_index = self._logical_layer_indices.get(layer)
        if logical_index is None:
            raise ValueError(
                "Expected layer to belong to this differential resistive "
                f"energy. Provided value: {layer!r}."
            )
        return self._layer_energy_scale_at(
            logical_index,
            dtype=layer.state.dtype,
        )

    def _init_passive_low_rank_adapter(
        self,
        config: PassiveLowRankAdapterConfig,
    ):
        """Build one dense base edge plus a passive two-factor branch."""

        layer_shapes = self._layer_shapes
        valid_shapes = (
            isinstance(layer_shapes, (list, tuple))
            and len(layer_shapes) == 2
            and all(
                isinstance(shape, (list, tuple, torch.Size))
                and len(shape) == 1
                and isinstance(shape[0], Integral)
                and not isinstance(shape[0], bool)
                and shape[0] > 0
                for shape in layer_shapes
            )
        )
        if not valid_shapes:
            raise ValueError(
                "Expected layer_shapes to contain exactly two "
                "one-dimensional positive shapes when "
                "passive_low_rank_adapter is enabled. "
                f"Provided value: {layer_shapes!r}."
            )
        self._layer_shapes = [tuple(shape) for shape in layer_shapes]

        for name, value in (
            ("voltage_amp", self._voltage_amp),
            ("current_amp", self._current_amp),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, Real)
                or float(value) != 1.0
            ):
                raise ValueError(
                    f"Expected {name} to equal 1.0 when "
                    "passive_low_rank_adapter is enabled. "
                    f"Provided value: {value!r}."
                )

        if (
            not isinstance(self._weight_gains, (list, tuple))
            or len(self._weight_gains) != 1
        ):
            raise ValueError(
                "Expected weight_gains to contain exactly one gain for the "
                "base dense matrix when passive_low_rank_adapter is enabled. "
                f"Provided value: {self._weight_gains!r}."
            )

        input_shape, output_shape = self._layer_shapes
        rank_shape = (config.rank,)
        input_layer = ResistiveInputLayer(
            input_shape,
            gain=self._input_amplifier,
            device=None,
        )
        rank_layer = LinearLayer(rank_shape, device=None)
        output_layer = LinearLayer(output_shape, device=None)

        base_weight = DenseWeight(
            input_shape,
            output_shape,
            self._weight_gains[0],
            device=None,
            clamp=True,
            clamp_min=self._weight_min,
            clamp_max=self._weight_max,
            init_mode=self._weight_init_mode,
        )
        input_factor = PassiveLowRankDenseWeight(
            input_shape,
            rank_shape,
            role="input_factor",
            gain=config.input_factor_gain,
            conductance_min=config.input_factor_min,
            conductance_max=config.conductance_max,
        )
        output_initial_value = (
            0.0
            if config.output_factor_init == "zero"
            else config.output_off_conductance
        )
        output_factor = PassiveLowRankDenseWeight(
            rank_shape,
            output_shape,
            role="output_factor",
            gain=0.0,
            conductance_min=output_initial_value,
            conductance_max=config.conductance_max,
            initial_value=output_initial_value,
        )

        layers = [input_layer, rank_layer, output_layer]
        interactions = [
            DenseResistive(
                input_layer,
                output_layer,
                base_weight,
                voltage_amp=1.0,
                current_amp=1.0,
            ),
            DenseResistive(
                input_layer,
                rank_layer,
                input_factor,
                voltage_amp=1.0,
                current_amp=1.0,
            ),
            DenseResistive(
                rank_layer,
                output_layer,
                output_factor,
                voltage_amp=1.0,
                current_amp=1.0,
            ),
        ]
        self._base_params = [base_weight]
        self._adapter_params = [input_factor, output_factor]
        self._all_params = self._base_params + self._adapter_params
        self._trainable_params = list(self._adapter_params)
        SumSeparableFunction.__init__(
            self,
            layers,
            self._all_params,
            interactions,
        )



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



    def __str__(self):
        return 'Deep Resistive Network -- layer shapes={}, weight gains={}, input_gain={}'.format(self._layer_shapes, self._weight_gains, self._input_amplifier)

    @property
    def passive_low_rank_adapter_config(self):
        """Return the normalized adapter configuration, or ``None``."""

        return self._passive_low_rank_adapter_config

    def base_params(self):
        """Return parameters belonging to the original DRN."""

        return list(self._base_params)

    def adapter_params(self):
        """Return passive adapter factors in stable ``[A, B]`` order."""

        return list(self._adapter_params)

    # Expose only trainable params (PoolWeight is kept frozen)
    def params(self):
        return self._trainable_params
