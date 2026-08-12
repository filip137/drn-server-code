import hashlib
import json
from pathlib import Path
import subprocess
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
RUNNER = REPOSITORY_ROOT / "experiments/analyze_conv_eqprop_bptt_checkpoint_gradients.py"
CONFIG = REPOSITORY_ROOT / "configs/conv/perfectdiode_conv123_legacy_adam_eqprop_bptt_beta_cosine_seed0_20260810_v1.json"


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_eqprop_checkpoint_gradient_config_is_weight_only_and_read_only():
    config = json.loads(CONFIG.read_text(encoding="utf-8"))

    assert config["checkpoint_roles"] == [
        "reconstructed_initialization",
        "best_validation",
    ]
    contract = config["gradient_contract"]
    assert contract["parameter_inclusion"] == ["ConvWeight_*", "DenseWeight_*"]
    assert contract["parameter_exclusion"] == ["Bias"]
    assert contract["optimizer_steps_applied"] is False
    assert contract["nudging_mode"] == "cost"
    assert contract["adaptive_equilibrium"] is False
    assert contract["betas"] == [0.003, 0.01, 0.03, 0.1, 0.25, 0.5, 1.0]
    assert config["dataset"]["official_test_read"] is False
    assert [case["gradient_iterations"] for case in config["cases"]] == [4, 6, 8]


def test_declared_source_artifact_hashes_match():
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    source_root = Path(config["source_contract"]["source_study_root"])
    for case in config["cases"]:
        run_dir = source_root / "final_runs" / case["run_id"]
        assert _sha256(run_dir / "config.used.json") == case["config_used_sha256"]
        assert _sha256(run_dir / "result.json") == case["result_sha256"]
        assert _sha256(run_dir / "best_model.pt") == case["best_checkpoint_sha256"]
        assert _sha256(run_dir / "manifest.json") == case["source_manifest_sha256"]


def test_runner_help_bootstraps_exact_runtime():
    result = subprocess.run(
        [sys.executable, str(RUNNER), "--help"],
        cwd=REPOSITORY_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "--runtime-source-root" in result.stdout
    assert "--smoke" in result.stdout


def test_gradient_metric_direction_and_beta_selection_helpers():
    program = f"""
import importlib.util
import pathlib
import sys
import torch

runner = pathlib.Path({str(RUNNER)!r})
sys.argv = [str(runner)]
spec = importlib.util.spec_from_file_location('eqprop_checkpoint_analysis_test', runner)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

same = module._vector_metrics(torch.tensor([1.0, 2.0]), torch.tensor([2.0, 4.0]), zero_epsilon=1e-12)
opposite = module._vector_metrics(torch.tensor([1.0, 2.0]), torch.tensor([-1.0, -2.0]), zero_epsilon=1e-12)
assert abs(same['cosine'] - 1.0) < 1e-12
assert abs(same['norm_ratio_eqprop_over_bptt'] - 2.0) < 1e-12
assert abs(opposite['cosine'] + 1.0) < 1e-12

rows = []
for beta, values in [(0.01, [0.9, 0.8]), (0.1, [0.85, 0.85])]:
    for index, cosine in enumerate(values):
        rows.append({{
            'architecture': 'conv1',
            'checkpoint_role': 'reconstructed_initialization',
            'phase_length': 'operating',
            'eqprop_variant': 'centered',
            'beta': beta,
            'cosine': cosine,
            'parameter_name': f'weight_{{index}}',
            'coverage_complete': True,
        }})
selected = [row for row in module._select_betas(rows) if row['selected']]
assert len(selected) == 1
assert selected[0]['beta'] == 0.1
"""
    result = subprocess.run(
        [sys.executable, "-c", program],
        cwd=REPOSITORY_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
