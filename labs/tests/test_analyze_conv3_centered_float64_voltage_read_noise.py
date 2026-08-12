from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import torch


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "experiments/analyze_conv3_centered_float64_voltage_read_noise.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("_voltage_read_noise_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


noise = _load_module()


def test_helper_import_does_not_shadow_the_current_worktree() -> None:
    import experiments.launch
    import experiments.rho_search
    import labs.mnist_train

    expected_root = ROOT.resolve()
    for module in (experiments.launch, experiments.rho_search, labs.mnist_train):
        assert Path(module.__file__).resolve().is_relative_to(expected_root)
    assert not any("signed-scaled-bias" in entry for entry in sys.path)


def test_noised_states_are_matched_across_cases_and_independent_across_phases():
    states = [torch.zeros((2, 3), dtype=torch.float64), torch.zeros((2, 2))]
    left, left_rows = noise._noised_states(
        states, sigma=1.0e-4, base_seed=1234, trial_index=7, phase_index=1
    )
    repeated, repeated_rows = noise._noised_states(
        states, sigma=1.0e-4, base_seed=1234, trial_index=7, phase_index=1
    )
    other_phase, _ = noise._noised_states(
        states, sigma=1.0e-4, base_seed=1234, trial_index=7, phase_index=2
    )

    assert all(torch.equal(a, b) for a, b in zip(left, repeated, strict=True))
    assert left_rows == repeated_rows
    assert any(
        not torch.equal(a, b)
        for a, b in zip(left, other_phase, strict=True)
    )
    assert left_rows[0]["seed"] != left_rows[1]["seed"]


def test_zero_noise_is_exact_identity():
    states = [torch.tensor([[1.0, -2.0]], dtype=torch.float64)]
    observed, rows = noise._noised_states(
        states, sigma=0.0, base_seed=8, trial_index=0, phase_index=1
    )
    assert torch.equal(observed[0], states[0])
    assert rows[0]["empirical_noise_rms"] == 0.0


def _trial_row(*, acquisition_cosine: float, task_cosine: float) -> dict:
    return {
        "scheme": "ours",
        "checkpoint_role": "best_validation",
        "clean_beta_selection_status": "largest_tested_clean_bracketed_by_next_beta",
        "sigma_voltage": 1.0e-7,
        "parameter_name": "ConvWeight_0",
        "acquisition_cosine": acquisition_cosine,
        "acquisition_symmetric_norm_delta": 0.01,
        "task_cosine": task_cosine,
        "task_symmetric_norm_delta": 0.01,
        "acquisition_gate_passed": acquisition_cosine >= 0.99,
        "task_gate_passed": task_cosine >= 0.99,
        "usable_gate_passed": acquisition_cosine >= 0.99 and task_cosine >= 0.99,
    }


def test_trial_aggregation_keeps_acquisition_and_task_gates_separate():
    rows = [
        _trial_row(acquisition_cosine=0.999, task_cosine=0.98)
        for _ in range(32)
    ]
    layer_rows, configuration_rows = noise._aggregate_trials(
        rows, cosine_minimum=0.99, norm_delta_maximum=0.1
    )
    assert layer_rows[0]["acquisition_gate_passed"] is True
    assert layer_rows[0]["task_gate_passed"] is False
    assert layer_rows[0]["usable_gate_passed"] is False
    assert configuration_rows[0]["all_layers_acquisition_gate_passed"] is True
    assert configuration_rows[0]["all_layers_usable_gate_passed"] is False
    assert configuration_rows[0]["clean_beta_selection_status"].startswith(
        "largest_tested_clean"
    )


def test_threshold_is_zero_anchored_and_does_not_resume_after_failure():
    configuration_rows = []
    for sigma, acquisition, usable in (
        (0.0, True, True),
        (1.0e-8, True, True),
        (3.0e-8, True, False),
        (1.0e-7, True, True),
        (3.0e-7, False, False),
    ):
        configuration_rows.append(
            {
                "scheme": "ours",
                "checkpoint_role": "best_validation",
                "sigma_voltage": sigma,
                "clean_beta_selection_status": "largest_tested_clean_bracketed_by_next_beta",
                "all_layers_acquisition_gate_passed": acquisition,
                "all_layers_usable_gate_passed": usable,
            }
        )
    clean_state_rows = [
        {
            "scheme": "ours",
            "checkpoint_role": "best_validation",
            "delta_rms": 2.0e-6,
        }
    ]
    threshold = noise._threshold_rows(configuration_rows, clean_state_rows)[0]
    assert threshold["acquisition_only_maximum_sustained_sigma"] == 1.0e-7
    assert threshold["usable_maximum_sustained_sigma"] == 1.0e-8
    assert threshold["usable_first_failing_sigma"] == 3.0e-8
    assert threshold["clean_beta_selection_status"] == (
        "largest_tested_clean_bracketed_by_next_beta"
    )
