from dataclasses import dataclass

import torch

from labs.small_network_core import _build_energy_stack
from model.function.cost import SquaredError
from model.function.network import Network
from model.resistive.builders import ModelBundle, build_deep_resistive_energy
from model.resistive.minimizer import QuadraticMinimizer
from training.monitor import Optimizer
from training.sgd import AugmentedFunction, EquilibriumProp
from training.tiki_taka import build_optimizer


_SEED = 20260727
_DEVICE = torch.device("cpu")
_QUADRATIC_DIODE = {
    "diode_conductance": 10.0,
    "v_min": -1.0,
    "v_max": 1.0,
}
_EXPONENTIAL_DIODE = {
    "I_s": 1e-6,
    "V_t": 0.05,
    "V_off": 0.5,
}
_HARD_SIGMOID = {
    "g_on": 10.0,
    "g_off": 10.0,
    "v_min": -1.0,
    "v_max": 1.0,
}
_INPUTS = torch.tensor(
    [[0.25, -0.75], [-0.4, 0.6]],
    dtype=torch.float32,
)
_LABELS = torch.tensor([0, 1], dtype=torch.long)


@dataclass
class _Composition:
    energy: object
    network: Network
    free_layers: list
    cost: SquaredError
    output_layer: object
    bundle: ModelBundle | None = None


def _build_compositions() -> tuple[_Composition, _Composition]:
    legacy_kwargs = {
        "device": _DEVICE,
        "input_dim": 2,
        "hidden_dims": (4,),
        "output_dim": 2,
        "non_linearity": "perfect_diode",
        "voltage_amp": 1.0,
        "current_amp": 1.0,
        "quadratic_diode_param": _QUADRATIC_DIODE,
        "exponential_diode_param": _EXPONENTIAL_DIODE,
        "hard_sigmoid_param": _HARD_SIGMOID,
        "input_gain": 1.0,
        "weights_path": None,
        "weight_gains": (0.2, 0.15),
        "weight_min": 1e-5,
        "weight_max": 1.0,
        "dataset_name": "moons",
    }

    torch.manual_seed(_SEED)
    (
        legacy_energy,
        legacy_network,
        legacy_free_layers,
        legacy_cost,
        legacy_output,
        layer_shapes,
        _input_gain,
        _quadratic_params,
        _exponential_params,
        _hard_sigmoid_params,
    ) = _build_energy_stack(**legacy_kwargs)

    torch.manual_seed(_SEED)
    bundle = build_deep_resistive_energy(
        layer_shapes=layer_shapes,
        weight_gains=list(legacy_kwargs["weight_gains"]),
        input_gain=legacy_kwargs["input_gain"],
        non_linearity=legacy_kwargs["non_linearity"],
        exponential_diode_param=_EXPONENTIAL_DIODE,
        quadratic_diode_param=_QUADRATIC_DIODE,
        hard_sigmoid_param=_HARD_SIGMOID,
        voltage_amp=legacy_kwargs["voltage_amp"],
        current_amp=legacy_kwargs["current_amp"],
        weight_min=legacy_kwargs["weight_min"],
        weight_max=legacy_kwargs["weight_max"],
    )
    current_energy = bundle.energy
    current_energy.set_device(_DEVICE)
    current_network = Network(current_energy)
    current_free_layers = current_network.free_layers()
    current_output = current_energy.layers()[-1]
    current_cost = SquaredError(current_output)

    return (
        _Composition(
            energy=legacy_energy,
            network=legacy_network,
            free_layers=legacy_free_layers,
            cost=legacy_cost,
            output_layer=legacy_output,
        ),
        _Composition(
            energy=current_energy,
            network=current_network,
            free_layers=current_free_layers,
            cost=current_cost,
            output_layer=current_output,
            bundle=bundle,
        ),
    )


def _minimizer(function, free_layers):
    return QuadraticMinimizer(
        fn=function,
        free_layers=free_layers,
        num_iterations=5,
        mode="asynchronous",
        non_linearity="perfect_diode",
        quadratic_diode_param=_QUADRATIC_DIODE,
        exponential_diode_param=_EXPONENTIAL_DIODE,
        hard_sigmoid_param=_HARD_SIGMOID,
        voltage_amp=1.0,
        current_amp=1.0,
    )


def _settle_free_phase(composition):
    composition.network.set_input(_INPUTS, reset=True)
    _minimizer(
        composition.energy,
        composition.free_layers,
    ).compute_equilibrium()


def _assert_tensors_identical(actual, expected):
    assert actual.dtype == expected.dtype
    assert actual.device == expected.device
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


def test_builder_matches_legacy_parameters_catalog_and_free_equilibrium():
    legacy, current = _build_compositions()

    legacy_parameters = tuple(legacy.energy._all_params)
    current_parameters = tuple(current.energy._all_params)
    assert current.bundle is not None
    assert current.bundle.parameters.all_parameters == current_parameters
    assert current.bundle.parameters.trainable_parameters == tuple(
        current.energy.params()
    )
    assert [binding.key for binding in current.bundle.parameters.all] == [
        "base.dense_weight.0",
        "base.dense_weight.1",
        "base.bias.0",
    ]
    assert [type(parameter) for parameter in current_parameters] == [
        type(parameter) for parameter in legacy_parameters
    ]
    for current_parameter, legacy_parameter in zip(
        current_parameters,
        legacy_parameters,
    ):
        _assert_tensors_identical(
            current_parameter.state,
            legacy_parameter.state,
        )

    _settle_free_phase(legacy)
    _settle_free_phase(current)

    for current_layer, legacy_layer in zip(
        current.energy.layers(),
        legacy.energy.layers(),
    ):
        _assert_tensors_identical(current_layer.state, legacy_layer.state)
    _assert_tensors_identical(
        current.output_layer.state,
        legacy.output_layer.state,
    )
    assert float(current.output_layer.state.abs().sum()) > 0.0


def test_centered_ep_and_configured_direct_update_match_legacy_path():
    legacy, current = _build_compositions()
    _settle_free_phase(legacy)
    _settle_free_phase(current)
    legacy.cost.set_target(_LABELS)
    current.cost.set_target(_LABELS)

    legacy_augmented = AugmentedFunction(legacy.energy, legacy.cost)
    current_augmented = AugmentedFunction(current.energy, current.cost)
    legacy_estimator = EquilibriumProp(
        legacy.energy.params(),
        legacy.free_layers,
        legacy_augmented,
        legacy.cost,
        _minimizer(legacy_augmented, legacy.free_layers),
        variant="centered",
        nudging=0.1,
    )
    current_estimator = EquilibriumProp(
        current.energy.params(),
        current.free_layers,
        current_augmented,
        current.cost,
        _minimizer(current_augmented, current.free_layers),
        variant="centered",
        nudging=0.1,
    )

    legacy_gradients = legacy_estimator.compute_gradient()
    current_gradients = current_estimator.compute_gradient()
    assert len(current_gradients) == len(legacy_gradients)
    assert any(float(gradient.abs().sum()) > 0.0 for gradient in current_gradients)
    for current_gradient, legacy_gradient in zip(
        current_gradients,
        legacy_gradients,
    ):
        _assert_tensors_identical(current_gradient, legacy_gradient)

    for parameter, gradient in zip(
        legacy.energy.params(),
        legacy_gradients,
    ):
        parameter.state.grad = gradient
    for parameter, gradient in zip(
        current.energy.params(),
        current_gradients,
    ):
        parameter.state.grad = gradient

    learning_rates = [0.03] * len(legacy.energy.params())
    legacy_optimizer = Optimizer(
        legacy.energy,
        legacy.cost,
        learning_rates,
        momentum=0.0,
        weight_decay=0.0,
    )
    configured_direct_optimizer = build_optimizer(
        current.energy,
        current.cost,
        learning_rates,
        update_pipeline={"type": "direct"},
        momentum=0.0,
        weight_decay=0.0,
    )
    assert isinstance(configured_direct_optimizer, Optimizer)

    legacy_optimizer.step()
    configured_direct_optimizer.step()
    for parameter in legacy.energy.params():
        parameter.clamp_()
    for parameter in current.energy.params():
        parameter.clamp_()

    for current_parameter, legacy_parameter in zip(
        current.energy.params(),
        legacy.energy.params(),
    ):
        _assert_tensors_identical(
            current_parameter.state,
            legacy_parameter.state,
        )
