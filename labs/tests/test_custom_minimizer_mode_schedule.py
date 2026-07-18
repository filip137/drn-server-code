from types import SimpleNamespace

import pytest
import torch

from labs.custom_minimizer import CustomMinimizer


def _schedule_minimizer(mode, *, num_iterations=3, adaptive=False, num_updaters=3):
    layers = [
        SimpleNamespace(name=f"u{index}", state=torch.tensor([float(index)]))
        for index in range(num_updaters)
    ]
    updaters = [SimpleNamespace(_layer=layer) for layer in layers]
    minimizer = CustomMinimizer.__new__(CustomMinimizer)
    minimizer._mode = mode
    minimizer._num_iterations = num_iterations
    minimizer._updaters = updaters
    minimizer._layers = layers
    minimizer._adaptive_equilibrium = adaptive
    minimizer._force_fixed_iterations = False
    minimizer._stored_states = {}
    minimizer._equilibrium_iter_total = 0
    minimizer._equilibrium_iter_calls = 0
    minimizer._equilibrium_iter_last = 0
    minimizer._equilibrium_iter_max = 0
    minimizer._equilibrium_sample_iters_last = None
    minimizer._rel_tol = 1e-5
    minimizer._vn_tol = 1e-6
    minimizer._use_polish = False
    minimizer._max_newton_iters = 0
    minimizer._z_thresh = 0.0
    minimizer._dynamic_polish = False
    minimizer._set_experimental_newton_tol = lambda _tol: None
    minimizer._experimental_exponential_newton_tol_policy = lambda _delta: 1e-5

    trace = []

    def record_group(group, *, label, iteration):
        trace.append((iteration, label, tuple(updater._layer.name for updater in group)))

    def record_async(odd, even, *, iteration):
        trace.append((iteration, "asynchronous_0", tuple(u._layer.name for u in odd)))
        trace.append((iteration, "asynchronous_1", tuple(u._layer.name for u in even)))

    minimizer._step_group_with_reject_policy = record_group
    minimizer._step_odd_even_with_reject_policy = record_async
    return minimizer, trace


@pytest.mark.parametrize(
    ("mode", "expected_sweep"),
    [
        ("forward", [("u0",), ("u1",), ("u2",)]),
        ("backward", [("u2",), ("u1",), ("u0",)]),
        ("synchronous", [("u0", "u1", "u2")]),
        ("asynchronous", [("u0", "u2"), ("u1",)]),
    ],
)
def test_fixed_mode_executes_complete_sweeps(mode, expected_sweep):
    minimizer, trace = _schedule_minimizer(mode, num_iterations=3)

    minimizer.compute_equilibrium()

    assert [entry[2] for entry in trace] == expected_sweep * 3
    assert [entry[0] for entry in trace] == [
        iteration
        for iteration in range(3)
        for _ in expected_sweep
    ]
    assert minimizer.equilibrium_iteration_stats() == {
        "calls": 1,
        "avg_iterations": 3.0,
        "last_iterations": 3,
        "max_iterations": 3,
    }


@pytest.mark.parametrize("mode", ["forward", "backward", "synchronous", "asynchronous"])
def test_adaptive_mode_checks_convergence_after_one_complete_sweep(mode):
    minimizer, trace = _schedule_minimizer(mode, num_iterations=5, adaptive=True)

    minimizer.compute_equilibrium()

    expected_group_count = {"forward": 3, "backward": 3, "synchronous": 1, "asynchronous": 2}[mode]
    assert len(trace) == expected_group_count
    assert {entry[0] for entry in trace} == {0}
    assert minimizer.equilibrium_iteration_stats()["last_iterations"] == 1
    assert minimizer.equilibrium_iteration_stats()["max_iterations"] == 5


def test_forward_single_updater_keeps_odd_iteration_count():
    minimizer, trace = _schedule_minimizer(
        "forward",
        num_iterations=3,
        num_updaters=1,
    )

    minimizer.compute_equilibrium()

    assert [entry[2] for entry in trace] == [("u0",), ("u0",), ("u0",)]
