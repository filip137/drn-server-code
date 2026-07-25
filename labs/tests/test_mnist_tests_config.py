from types import SimpleNamespace

from labs import mnist_tests


def test_build_minimizer_uses_top_level_config_and_base_energy_amplification(monkeypatch):
    captured = {}

    class FakeMinimizer:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(mnist_tests, "CustomQuadraticMinimizer", FakeMinimizer)

    base_energy = SimpleNamespace(_voltage_amp=2.5, _current_amp=0.4)
    parts = SimpleNamespace(
        energy_fn=base_energy,
        energy_minimizer_cfg={
            "adaptive_equilibrium": False,
            "use_polish": False,
            "dynamic_polish": False,
            "exp_clip": 165.0,
        },
        training_cfg={"energy_minimizer": {"adaptive_equilibrium": True}},
        model_cfg={
            "non_linearity": "perfect_diode",
            "quadratic_diode_param": {},
            "exponential_diode_param": {},
            "hard_sigmoid_param": {},
        },
        free_layers=[],
        minimizer_mode="asynchronous",
    )

    augmented_energy = object()
    result = mnist_tests._build_minimizer(parts, 4, fn=augmented_energy)

    assert isinstance(result, FakeMinimizer)
    assert captured["fn"] is augmented_energy
    assert captured["adaptive_equilibrium"] is False
    assert captured["voltage_amp"] == 2.5
    assert captured["current_amp"] == 0.4
    assert captured["minimizer_settings"].use_polish is False
    assert captured["minimizer_settings"].dynamic_polish is False
    assert captured["minimizer_settings"].exp_clip == 165.0
