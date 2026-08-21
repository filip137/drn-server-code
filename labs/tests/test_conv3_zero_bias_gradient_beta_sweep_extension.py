from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
CONFIG = (
    ROOT
    / "configs/conv/perfectdiode_conv3_zero_bias_adam_eqprop_bptt_gradient_beta_sweep_above_max_to_10xmax_sigma_5em4_seed0_20260817_v1.json"
)
RUNNER = ROOT / "experiments/analyze_conv123_zero_bias_eqprop_bptt_gradient_gate.py"


def _config() -> dict:
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def test_extension_continues_parent_grid_one_decade_above_maximum() -> None:
    value = _config()
    points = value["sweep_contract"]["beta_scale_points"]
    assert len(points) == 8
    assert len({row["run_id"] for row in points}) == 8
    for expected_index, point in zip(range(9, 17), points, strict=True):
        assert point["index"] == expected_index
        assert math.isclose(
            point["beta_scale"], 10.0 ** (expected_index / 8.0)
        )
    assert value["sweep_contract"]["extension_injected_beta_maxima"] == {
        "baseline": 10000.0,
        "ours": 300.0,
        "legacy": 0.1,
    }


def test_extension_preserves_reviewed_parent_scientific_contract() -> None:
    value = _config()
    contract = value["sweep_contract"]
    parent_path = ROOT / contract["parent_config"]
    parent_bytes = parent_path.read_bytes()
    assert hashlib.sha256(parent_bytes).hexdigest() == contract[
        "parent_config_sha256"
    ]
    parent = json.loads(parent_bytes)
    for key in (
        "schema_version",
        "source_contract",
        "dataset",
        "gradient_contract",
        "read_noise_contract",
        "checkpoint_roles",
        "cases",
        "completion",
    ):
        assert value[key] == parent[key]


def test_combined_grid_has_seventeen_uniform_log_spaced_points() -> None:
    from experiments import (
        analyze_conv3_zero_bias_gradient_beta_sweep_extension_result as analysis,
    )

    value = _config()
    parent_path = ROOT / value["sweep_contract"]["parent_config"]
    parent = json.loads(parent_path.read_text(encoding="utf-8"))
    parent_points, extension_points = analysis._extension_points(parent, value)
    combined = parent_points + extension_points
    assert len(parent_points) == 9
    assert len(extension_points) == 8
    assert len(combined) == 17
    assert [row["index"] for row in combined] == list(range(17))
    for index, point in enumerate(combined):
        assert math.isclose(point["beta_scale"], 10.0 ** (index / 8.0))


def test_extension_validate_only_reproduces_sources_and_extreme_scale() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            str(RUNNER),
            "--config",
            str(CONFIG),
            "--validate-only",
            "--device",
            "cpu",
            "--beta-scale",
            "100",
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
    assert observed["beta_scale"] == 100.0
    assert observed["read_noise_contract"]["endpoint_read_noise_std"] == 5.0e-4
    assert observed["lineage_proof"][
        "scientific_contract_matches_parent_except_read_noise"
    ] is True
