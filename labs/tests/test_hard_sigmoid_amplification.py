import sys
from pathlib import Path

import torch

from model.function.interaction import HardSigmoidNonLinearInteraction
from model.resistive.layer import NonlinearResistiveLayer, ResistiveInputLayer
from model.resistive.minimizer import QuadraticMinimizer
from model.variable.layer import Layer


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
