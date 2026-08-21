from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
CONFIG = (
    ROOT
    / "configs/conv/perfectdiode_conv23_zero_bias_adam_eqprop_bptt_gradient_gate_one_decade_sigma_5em4_seed0_20260817_v1.json"
)
SCRIPT = ROOT / "experiments/analyze_conv123_zero_bias_eqprop_bptt_gradient_gate.py"


def _config() -> dict:
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def test_noisy_gate_config_is_exact_conv23_one_decade_surface() -> None:
    value = _config()
    cases = value["cases"]
    assert [(row["architecture"], row["scheme"]) for row in cases] == [
        (architecture, scheme)
        for architecture in ("conv2", "conv3")
        for scheme in ("baseline", "ours", "legacy")
    ]
    assert {
        (row["architecture"], row["scheme"]): row["injected_beta"]
        for row in cases
    } == {
        ("conv2", "baseline"): 100.0,
        ("conv2", "ours"): 10.0,
        ("conv2", "legacy"): 0.03,
        ("conv3", "baseline"): 100.0,
        ("conv3", "ours"): 3.0,
        ("conv3", "legacy"): 0.001,
    }
    assert value["read_noise_contract"] == {
        "endpoint_read_noise_std": 5.0e-4,
        "endpoint_read_noise_seed": 2026081601,
        "input_read_noise": False,
        "acquisition_model": "independent_gaussian_free_layer_endpoint_voltage",
        "independent_negative_positive_reads": True,
        "matched_standard_normal_draws_across_schemes": True,
        "noise_draw_rule": (
            "seed + 1000*batch_index + 100*phase_index + state_layer_index"
        ),
    }
    assert value["completion"]["production_replay_count"] == 48
    assert value["completion"]["production_layer_comparison_count"] == 168
    assert value["completion"]["production_noise_draw_count"] == 336


def test_noisy_gate_parent_contract_differs_only_by_selected_architectures_and_noise() -> None:
    value = _config()
    parent_path = ROOT / value["lineage"]["clean_gradient_gate_config"]
    parent = json.loads(parent_path.read_text(encoding="utf-8"))
    for key in (
        "schema_version",
        "source_contract",
        "dataset",
        "gradient_contract",
        "checkpoint_roles",
    ):
        assert value[key] == parent[key]
    expected_cases = [
        row
        for row in parent["cases"]
        if row["architecture"] in {"conv2", "conv3"}
    ]
    assert value["cases"] == expected_cases


def test_read_noise_draws_are_repeatable_and_phase_independent() -> None:
    code = f"""
import json
import sys
import torch
sys.argv = [str({str(SCRIPT)!r}), '--config', str({str(CONFIG)!r})]
from experiments import analyze_conv123_zero_bias_eqprop_bptt_gradient_gate as gate
states = [torch.zeros((2, 3), dtype=torch.float64), torch.zeros((2, 1), dtype=torch.float64)]
left, left_rows = gate._noised_states(states, sigma=5e-4, base_seed=2026081601, batch_index=2, phase_index=1)
repeat, repeat_rows = gate._noised_states(states, sigma=5e-4, base_seed=2026081601, batch_index=2, phase_index=1)
positive, positive_rows = gate._noised_states(states, sigma=5e-4, base_seed=2026081601, batch_index=2, phase_index=2)
print(json.dumps({{
    'repeat': all(torch.equal(a, b) for a, b in zip(left, repeat)),
    'independent': all(not torch.equal(a, b) for a, b in zip(left, positive)),
    'repeat_hashes': [row['standard_normal_sha256'] for row in left_rows] == [row['standard_normal_sha256'] for row in repeat_rows],
    'phase_hashes_differ': [row['standard_normal_sha256'] for row in left_rows] != [row['standard_normal_sha256'] for row in positive_rows],
}}))
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    observed = json.loads(completed.stdout)
    assert observed == {
        "repeat": True,
        "independent": True,
        "repeat_hashes": True,
        "phase_hashes_differ": True,
    }


def test_noisy_gate_validate_only_reproduces_parent_and_cohort() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--config",
            str(CONFIG),
            "--validate-only",
            "--device",
            "cpu",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    observed = json.loads(completed.stdout)
    assert observed["state"] == "validated"
    assert observed["case_count"] == 6
    assert observed["batch_count"] == 4
    assert observed["read_noise_contract"]["endpoint_read_noise_std"] == 5.0e-4
    assert observed["lineage_proof"][
        "scientific_contract_matches_parent_except_read_noise"
    ] is True
