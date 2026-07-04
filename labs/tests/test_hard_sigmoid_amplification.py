import sys
from pathlib import Path

import pytest
import torch

from model.function.interaction import HardSigmoidNonLinearInteraction
from model.resistive.layer import NonlinearResistiveLayer, ResistiveInputLayer
from model.resistive.minimizer import QuadraticMinimizer
from model.variable.layer import Layer
from model.variable.parameter import HardSigmoidVOff


LABS_DIR = Path(__file__).resolve().parents[1]
if str(LABS_DIR) not in sys.path:
    sys.path.insert(0, str(LABS_DIR))

from custom_classes import FlexibleDeepResistiveEnergy  # noqa: E402


class _QuadraticLayerEnergy:
    def __init__(self, layer, *, a, b):
        self._layer = layer
        self._a = torch.as_tensor(a, dtype=layer.state.dtype, device=layer.state.device)
        self._b = torch.as_tensor(b, dtype=layer.state.dtype, device=layer.state.device)

    def layers(self):
        return [self._layer]

    def params(self):
        return []

    def a_coef_fn(self, layer):
        assert layer is self._layer
        return lambda: torch.full_like(layer.state, self._a)

    def b_coef_fn(self, layer):
        assert layer is self._layer
        return lambda: torch.full_like(layer.state, self._b)

    def grad_layer_fn(self, layer):
        assert layer is self._layer
        return lambda: 2.0 * self.a_coef_fn(layer)() * layer.state + self.b_coef_fn(layer)()


def _make_layer_2():
    Layer._counter = 0
    ResistiveInputLayer((2,), gain=1.0, batch_size=1, device="cpu")
    NonlinearResistiveLayer((2,), batch_size=1, device="cpu", non_linearity="hard_sigmoid")
    return NonlinearResistiveLayer((2,), batch_size=1, device="cpu", non_linearity="hard_sigmoid")


def _hard_sigmoid_interactions(energy):
    return [
        interaction
        for interaction in energy._interactions
        if isinstance(interaction, HardSigmoidNonLinearInteraction)
    ]


def _base_energy_kwargs(hard_sigmoid_param):
    return dict(
        input_gain=1.0,
        non_linearity="hard_sigmoid",
        exponential_diode_param={"I_s": 1.0e-6, "V_t": 0.025, "V_off": 0.0},
        quadratic_diode_param=_quadratic_params(),
        hard_sigmoid_param=hard_sigmoid_param,
        voltage_amp=1.0,
        current_amp=4.0,
        weight_min=0.0,
        weight_max=100.0,
    )


def _quadratic_params():
    return {"diode_conductance": 1.0, "v_min": -1.0e6, "v_max": 1.0e6}


def test_hard_sigmoid_legacy_v_min_v_max_boundaries_are_preserved():
    layer = _make_layer_2()
    params = {"g_on": 100.0, "g_off": 0.0, "v_min": -1.5, "v_max": 1.5}
    nonlinear_energy = HardSigmoidNonLinearInteraction(
        layer,
        params,
        voltage_amp=1.0,
        current_amp=1.0,
    )
    minimizer = QuadraticMinimizer(
        _QuadraticLayerEnergy(layer, a=1.0, b=0.0),
        [layer],
        num_iterations=1,
        mode="asynchronous",
        non_linearity="hard_sigmoid",
        quadratic_diode_param=_quadratic_params(),
        exponential_diode_param={},
        hard_sigmoid_param=params,
        voltage_amp=1.0,
        current_amp=1.0,
    )

    assert nonlinear_energy.v_min == pytest.approx(-1.5)
    assert nonlinear_energy.v_max == pytest.approx(1.5)
    assert minimizer._updaters[0]._vmin == pytest.approx(-1.5)
    assert minimizer._updaters[0]._vmax == pytest.approx(1.5)


def test_hard_sigmoid_scalar_v_off_expands_to_symmetric_boundaries():
    layer = _make_layer_2()
    params = {"g_on": 100.0, "g_off": 0.0, "v_off": 1.25}
    nonlinear_energy = HardSigmoidNonLinearInteraction(
        layer,
        params,
        voltage_amp=1.0,
        current_amp=1.0,
    )
    minimizer = QuadraticMinimizer(
        _QuadraticLayerEnergy(layer, a=1.0, b=0.0),
        [layer],
        num_iterations=1,
        mode="asynchronous",
        non_linearity="hard_sigmoid",
        quadratic_diode_param=_quadratic_params(),
        exponential_diode_param={},
        hard_sigmoid_param=params,
        voltage_amp=1.0,
        current_amp=1.0,
    )

    assert nonlinear_energy.v_min == pytest.approx(-1.25)
    assert nonlinear_energy.v_max == pytest.approx(1.25)
    assert minimizer._updaters[0]._vmin == pytest.approx(-1.25)
    assert minimizer._updaters[0]._vmax == pytest.approx(1.25)


def test_hard_sigmoid_v_off_list_maps_to_hidden_layers_and_updaters():
    Layer._counter = 0
    params = {"g_on": 100.0, "g_off": 0.0, "v_off": [1.0, 2.0]}
    energy = FlexibleDeepResistiveEnergy(
        layer_shapes=[(2,), (4,), (6,), (2,)],
        weight_gains=[1.0, 1.0, 1.0],
        **_base_energy_kwargs(params),
    )
    nonlinear_interactions = _hard_sigmoid_interactions(energy)
    free_layers = energy.layers()[1:3]
    minimizer = QuadraticMinimizer(
        energy,
        free_layers,
        num_iterations=1,
        mode="asynchronous",
        non_linearity="hard_sigmoid",
        quadratic_diode_param=_quadratic_params(),
        exponential_diode_param={},
        hard_sigmoid_param=params,
        voltage_amp=1.0,
        current_amp=4.0,
    )

    assert [interaction._layer.name for interaction in nonlinear_interactions] == [
        "Layer_1",
        "Layer_2",
    ]
    assert [(interaction.v_min, interaction.v_max) for interaction in nonlinear_interactions] == [
        pytest.approx((-1.0, 1.0)),
        pytest.approx((-2.0, 2.0)),
    ]
    assert [(updater._vmin, updater._vmax) for updater in minimizer._updaters] == [
        pytest.approx((-1.0, 1.0)),
        pytest.approx((-2.0, 2.0)),
    ]


def test_hard_sigmoid_v_off_list_skips_pooling_layers():
    Layer._counter = 0
    energy = FlexibleDeepResistiveEnergy(
        layer_shapes=[(2, 8, 8), (4, 6, 6), (4, 3, 3), (6,), (2,)],
        weight_gains=[1.0, 1.0, 1.0, 1.0],
        conv_pipeline=[
            {"kernel": (3, 3), "stride": 1, "padding": 0, "mode": "convolution"},
            {"kernel": (2, 2), "stride": 2, "padding": 0, "mode": "pooling"},
        ],
        pooling_mode="avg",
        **_base_energy_kwargs({"g_on": 100.0, "g_off": 0.0, "v_off": [1.25, 2.5]}),
    )
    nonlinear_interactions = _hard_sigmoid_interactions(energy)

    assert [interaction._layer.name for interaction in nonlinear_interactions] == [
        "Layer_1",
        "Layer_3",
    ]
    assert [(interaction.v_min, interaction.v_max) for interaction in nonlinear_interactions] == [
        pytest.approx((-1.25, 1.25)),
        pytest.approx((-2.5, 2.5)),
    ]


def test_hard_sigmoid_v_off_list_length_must_match_nonlinear_layers():
    Layer._counter = 0
    with pytest.raises(ValueError, match="v_off list length"):
        FlexibleDeepResistiveEnergy(
            layer_shapes=[(2,), (4,), (6,), (2,)],
            weight_gains=[1.0, 1.0, 1.0],
            **_base_energy_kwargs({"g_on": 100.0, "g_off": 0.0, "v_off": [1.0]}),
        )


def test_trainable_hard_sigmoid_v_off_is_one_parameter_per_nonlinear_layer():
    Layer._counter = 0
    HardSigmoidVOff._counter = 0
    energy = FlexibleDeepResistiveEnergy(
        layer_shapes=[(2,), (4,), (6,), (2,)],
        weight_gains=[1.0, 1.0, 1.0],
        **_base_energy_kwargs(
            {
                "g_on": 100.0,
                "g_off": 0.0,
                "v_off": [1.0, 2.0],
                "trainable_v_off": True,
            }
        ),
    )

    v_off_params = [param for param in energy.params() if isinstance(param, HardSigmoidVOff)]
    nonlinear_interactions = _hard_sigmoid_interactions(energy)

    assert [param.name for param in v_off_params] == ["HardSigmoidVOff_0", "HardSigmoidVOff_1"]
    assert [float(param.state.item()) for param in v_off_params] == pytest.approx([1.0, 2.0])
    assert [interaction.params() for interaction in nonlinear_interactions] == [
        [v_off_params[0]],
        [v_off_params[1]],
    ]


def test_trainable_hard_sigmoid_v_off_affects_closed_form_update_gradient():
    layer = _make_layer_2()
    HardSigmoidVOff._counter = 0
    v_off_param = HardSigmoidVOff(1.5, device="cpu")
    params = {"g_on": 100.0, "g_off": 0.0, "v_off_param": v_off_param}
    base_energy = _QuadraticLayerEnergy(layer, a=1.0, b=-20.0)
    minimizer = QuadraticMinimizer(
        base_energy,
        [layer],
        num_iterations=1,
        mode="asynchronous",
        non_linearity="hard_sigmoid",
        quadratic_diode_param=_quadratic_params(),
        exponential_diode_param={},
        hard_sigmoid_param=params,
        voltage_amp=1.0,
        current_amp=1.0,
    )

    v_off_param.state.requires_grad = True
    updated_state = minimizer._updaters[0].pre_activate()
    grad = torch.autograd.grad(updated_state.sum(), v_off_param.state)[0]

    assert torch.isfinite(grad).all()
    assert grad.abs().sum() > 0


def test_trainable_hard_sigmoid_energy_boundaries_follow_replaced_state():
    layer = _make_layer_2()
    HardSigmoidVOff._counter = 0
    v_off_param = HardSigmoidVOff(1.5, device="cpu")
    interaction = HardSigmoidNonLinearInteraction(
        layer,
        {"g_on": 100.0, "g_off": 0.0, "v_off_param": v_off_param},
        voltage_amp=1.0,
        current_amp=1.0,
    )

    assert [float(value.item()) for value in interaction._boundary_tensors()] == pytest.approx(
        [-1.5, 1.5]
    )

    v_off_param.state = torch.tensor([3.25])
    assert [float(value.item()) for value in interaction._boundary_tensors()] == pytest.approx(
        [-3.25, 3.25]
    )

    layer.state = torch.tensor([[2.0, -2.0]])
    grad = interaction.grad_layer_fn(layer)()
    assert torch.allclose(grad, torch.zeros_like(grad))


def test_hard_sigmoid_updater_uses_same_amplified_conductance_as_energy():
    """Regression test for current/voltage amp on deeper hard-sigmoid layers.

    Layer_2 must use g * (current_amp / voltage_amp) in both the energy gradient
    and the closed-form hard-sigmoid updater. Without that scaling in the updater,
    the update is not a zero of the amplified energy gradient.
    """

    layer = _make_layer_2()
    layer.state = torch.zeros(1, 2)
    voltage_amp = 1.0
    current_amp = 4.0
    params = {"g_on": 100.0, "g_off": 0.0, "v_min": -1.5, "v_max": 1.5}
    base_energy = _QuadraticLayerEnergy(layer, a=1.0, b=-20.0)
    nonlinear_energy = HardSigmoidNonLinearInteraction(
        layer,
        params,
        voltage_amp=voltage_amp,
        current_amp=current_amp,
    )

    minimizer = QuadraticMinimizer(
        base_energy,
        [layer],
        num_iterations=1,
        mode="asynchronous",
        non_linearity="hard_sigmoid",
        quadratic_diode_param={},
        exponential_diode_param={},
        hard_sigmoid_param=params,
        voltage_amp=voltage_amp,
        current_amp=current_amp,
    )
    updated_state = minimizer._updaters[0].pre_activate()
    layer.state = updated_state

    total_grad = base_energy.grad_layer_fn(layer)() + nonlinear_energy.grad_layer_fn(layer)()
    assert torch.allclose(total_grad, torch.zeros_like(total_grad), atol=1e-5, rtol=1e-5)
    assert minimizer._updaters[0].g_on == params["g_on"] * (current_amp / voltage_amp)


def test_hard_sigmoid_energy_is_not_attached_to_output_layer():
    """The output layer is linear; hard-sigmoid penalties belong to hidden layers."""

    Layer._counter = 0
    energy = FlexibleDeepResistiveEnergy(
        layer_shapes=[(2,), (4,), (2,)],
        weight_gains=[1.0, 1.0],
        input_gain=1.0,
        non_linearity="hard_sigmoid",
        exponential_diode_param={"I_s": 1.0e-6, "V_t": 0.025, "V_off": 0.0},
        quadratic_diode_param={"diode_conductance": 1.0, "v_min": -1.0e6, "v_max": 1.0e6},
        hard_sigmoid_param={"g_on": 100.0, "g_off": 0.0, "v_min": -1.5, "v_max": 1.5},
        voltage_amp=1.0,
        current_amp=4.0,
        weight_min=0.0,
        weight_max=100.0,
    )

    nonlinear_layers = [
        interaction._layer.name
        for interaction in energy._interactions
        if isinstance(interaction, HardSigmoidNonLinearInteraction)
    ]
    assert nonlinear_layers == ["Layer_1"]
    assert energy.layers()[-1].name == "Layer_2"
