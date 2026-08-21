from __future__ import annotations

import json
import math
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
CONFIG = (
    ROOT
    / "configs/conv/perfectdiode_conv3_zero_bias_adam_eqprop_bptt_gradient_beta_sweep_one_decade_to_max_sigma_5em4_seed0_20260817_v1.json"
)
RUNNER = ROOT / "experiments/analyze_conv123_zero_bias_eqprop_bptt_gradient_gate.py"


def _config() -> dict:
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def test_sweep_config_covers_conv3_all_schemes_from_one_decade_to_maximum() -> None:
    value = _config()
    assert [(row["architecture"], row["scheme"]) for row in value["cases"]] == [
        ("conv3", "baseline"),
        ("conv3", "ours"),
        ("conv3", "legacy"),
    ]
    assert {
        row["scheme"]: row["injected_beta"] for row in value["cases"]
    } == {"baseline": 100.0, "ours": 3.0, "legacy": 0.001}
    points = value["sweep_contract"]["beta_scale_points"]
    assert len(points) == 9
    assert len({row["run_id"] for row in points}) == 9
    for index, point in enumerate(points):
        assert point["index"] == index
        assert math.isclose(point["beta_scale"], 10.0 ** (index / 8.0))
    assert value["sweep_contract"]["injected_beta_maxima"] == {
        "baseline": 1000.0,
        "ours": 30.0,
        "legacy": 0.01,
    }
    assert value["completion"]["production_replay_count"] == 24
    assert value["completion"]["production_layer_comparison_count"] == 96
    assert value["completion"]["production_noise_draw_count"] == 192


def test_sweep_config_preserves_clean_parent_contract_and_adds_fixed_noise() -> None:
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
    assert value["cases"] == [
        row for row in parent["cases"] if row["architecture"] == "conv3"
    ]
    assert value["read_noise_contract"]["endpoint_read_noise_std"] == 5.0e-4
    assert value["read_noise_contract"]["endpoint_read_noise_seed"] == 2026081601
    assert value["read_noise_contract"]["input_read_noise"] is False


def test_sweep_config_validate_only_reproduces_sources_and_cohort() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            str(RUNNER),
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
    assert observed["case_count"] == 3
    assert observed["batch_count"] == 4
    assert observed["beta_scale"] == 1.0
    assert observed["read_noise_contract"]["endpoint_read_noise_std"] == 5.0e-4
    assert observed["lineage_proof"][
        "scientific_contract_matches_parent_except_read_noise"
    ] is True
