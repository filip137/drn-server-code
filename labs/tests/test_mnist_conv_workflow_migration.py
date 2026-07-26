from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import subprocess
from pathlib import Path

import pytest

from experiments.mnist_conv.layout import ResultLayout
from experiments.mnist_conv.manifest import publish_manifest
from experiments.mnist_conv.specs import SweepSpec
from experiments.submit_mnist_conv_slurm import load_profile, submit_sweep


REPO_ROOT = Path(__file__).resolve().parents[2]
BUILDER_PATH = REPO_ROOT / "experiments" / "build_mnist_bp_conv_amp_calibrated_training_manifest.py"


def _load_builder():
    spec = importlib.util.spec_from_file_location("canonical_calibration_builder", BUILDER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _diagnostic_run() -> dict:
    return {
        "schema_version": "mnist-conv-run/v1",
        "label": "schema-test-only",
        "seed": 0,
        "replicate_id": None,
        "protocol_id": "diagnostic-test-fixture",
        "category": "diagnostic",
        "run": {
            "dataset": {
                "name": "mnist",
                "input_shape": [2, 28, 28],
                "batch_size": 1,
                "normalization": {"mean": 0.1307, "std": 0.3081, "scale": 0.3},
                "affine": {
                    "enabled": True,
                    "preset": "medium",
                    "degrees": 25.0,
                    "translate": [0.2, 0.2],
                    "scale": [0.8, 1.2],
                    "shear": 0.0,
                    "seed": 1729,
                    "interpolation": "bilinear",
                    "fill": 0.0,
                },
                "max_batches": 1,
                "max_test_batches": 1,
            },
            "architecture": {
                "profile": "conv1",
                "channels": [64],
                "kernel_sizes": [3],
                "strides": [2],
                "paddings": [1],
                "output_dim": 20,
                "pooling": {"mode": "none"},
            },
            "model": {
                "type": "resistive_conv",
                "non_linearity": "hard_sigmoid",
                "voltage_amp": 1.0,
                "current_amp": 1.0,
                "input_gain": 10.0,
                "weight_gains": [1.0, 1.0],
                "weight_min": 0.0,
                "weight_max": 100.0,
                "weight_init_mode": "kaiming_uniform",
                "quadratic_diode_param": {},
                "exponential_diode_param": {},
                "hard_sigmoid_param": {"g_on": 100.0, "g_off": 0.0, "v_off": 4.0},
                "trainable_parameters": {
                    "weights": True,
                    "biases": True,
                    "amplification": False,
                    "hard_sigmoid_v_off": False,
                },
                "amplification_min": 1e-6,
                "amplification_max": None,
            },
            "solver": {
                "inference_iterations": 1,
                "training_iterations": 1,
                "energy_mode": "asynchronous",
                "minimizer": {
                    "double_diode_updater": "newton",
                    "adaptive_equilibrium": False,
                    "overrelaxation_factor": 1.0,
                    "single_diode_updater": "closed_form",
                    "iv_data_path": None,
                    "experimental_damping": 1.0,
                    "experimental_newton_max_steps": 16,
                    "settings": {
                        "rel_tol": 1e-6,
                        "vn_tol": 1e-6,
                        "use_polish": False,
                        "max_newton_iters": 50,
                        "z_thresh": 1e-9,
                        "exp_clip": 40.0,
                        "dynamic_polish": False,
                        "overrelaxation_reject_steps": False,
                        "overrelaxation_reject_max_tries": 4,
                        "overrelaxation_reject_shrink": 0.5,
                        "overrelaxation_reject_eps": 0.0,
                        "experimental_exponential_newton_tol_progressive": False,
                        "experimental_exponential_newton_tol_start": 1e-3,
                        "experimental_exponential_newton_tol_end": 1e-6,
                        "experimental_exponential_newton_tol_switch_hi": 1e-2,
                        "experimental_exponential_newton_tol_switch_lo": 1e-4,
                    },
                },
            },
            "training": {
                "algorithm": "BP",
                "epochs": 1,
                "optimizer": {"name": "SGD", "momentum": 0.0, "weight_decay": 0.0},
                "learning_rate": [0.01, 0.01, 0.01],
                "beta": 1.0,
                "lr_decay": 1.0,
                "checkpoint_rule": "max_test_accuracy",
                "pruning": {"enabled": False, "after_epoch": None, "min_best_test_accuracy": None},
                "batch_state_policy": "reset_each_batch",
            },
            "initialization": {"checkpoint": None},
            "calibration": _calibration("calibration-base", 0.30),
        },
    }


def _calibration(calibration_id: str, measured: float) -> dict:
    return {
        "kind": "hard_sigmoid_saturation",
        "calibration_id": calibration_id,
        "scope": "first_hidden_layer",
        "sample_count": 256,
        "batch_size": 64,
        "model_seed": 0,
        "affine_seed": 1729,
        "settling_iterations": 64,
        "adaptive_equilibrium": False,
        "layer_measurements": [{"layer_index": 1, "measured_saturation": measured}],
        "v_off": 4.0,
        "g_on": 100.0,
        "g_off": 0.0,
        "target_initial_saturation": 0.30,
        "measured_initial_saturation": measured,
    }


def _record(case_id: str, voltage_amp: float, current_amp: float, gain: float, measured: float) -> dict:
    return {
        "case_id": case_id,
        "status": "complete",
        "dataset_name": "mnist",
        "architecture_profile": "conv1",
        "non_linearity": "hard_sigmoid",
        "voltage_amp": voltage_amp,
        "current_amp": current_amp,
        "input_gain": gain,
        "calibration": _calibration(f"cal-{case_id}", measured),
    }


def _write_inputs(tmp_path: Path) -> tuple[Path, Path]:
    base = tmp_path / "run.json"
    base.write_text(json.dumps(_diagnostic_run()))
    records = {
        "schema_version": "mnist-conv-calibration-records/v1",
        "records": [
            _record("mnist_bp_amp_v4_c0p25", 4.0, 0.25, 31.0, 0.31),
            _record("mnist_bp_amp_v1_c1", 1.0, 1.0, 11.0, 0.29),
            _record("mnist_bp_amp_v4_c1", 4.0, 1.0, 21.0, 0.30),
        ],
    }
    calibration_path = tmp_path / "calibrations.json"
    calibration_path.write_text(json.dumps(records))
    return base, calibration_path


def test_calibration_builder_preserves_case_order_and_training_settings(tmp_path: Path) -> None:
    builder = _load_builder()
    base, records = _write_inputs(tmp_path)

    sweep = builder.build_sweep(
        base_run_path=base,
        calibration_records_path=records,
        name="diagnostic-schema-test",
        seeds=[0, 2],
    )

    assert [case["id"] for case in sweep.data["cases"]] == [
        "mnist_bp_amp_v4_c0p25",
        "mnist_bp_amp_v1_c1",
        "mnist_bp_amp_v4_c1",
    ]
    assert sweep.data["collection"]["required_cases"] == [case["id"] for case in sweep.data["cases"]]
    expanded = sweep.expand()
    assert len(expanded) == 6
    assert {tuple(item.spec.data["run"]["training"]["learning_rate"]) for item in expanded} == {
        (0.01, 0.01, 0.01)
    }
    assert {
        (item.spec.data["run"]["solver"]["inference_iterations"], item.spec.data["run"]["solver"]["training_iterations"])
        for item in expanded
    } == {(1, 1)}
    assert all(item.spec.data["run"]["calibration"]["settling_iterations"] == 64 for item in expanded)


def test_calibration_builder_rejects_incomplete_or_lr_bearing_records(tmp_path: Path) -> None:
    builder = _load_builder()
    base, records_path = _write_inputs(tmp_path)
    value = json.loads(records_path.read_text())
    value["records"][0]["calibration"]["layer_measurements"] = []
    records_path.write_text(json.dumps(value))
    with pytest.raises(builder.ConversionError, match="one explicit measurement"):
        builder.build_sweep(
            base_run_path=base,
            calibration_records_path=records_path,
            name="invalid",
            seeds=[0, 1],
        )

    value = json.loads(_write_inputs(tmp_path)[1].read_text())
    value["records"][0]["learning_rate"] = 0.5
    records_path.write_text(json.dumps(value))
    with pytest.raises(builder.ConversionError, match="exactly keys"):
        builder.build_sweep(
            base_run_path=base,
            calibration_records_path=records_path,
            name="invalid",
            seeds=[0, 1],
        )


def test_slurm_submit_dry_run_is_manifest_driven_and_afterany(tmp_path: Path) -> None:
    sweep = SweepSpec.from_dict({
        "schema_version": "mnist-conv-sweep/v1",
        "name": "slurm-contract-test",
        "base_run": _diagnostic_run(),
        "axes": [{"path": "/seed", "values": [0, 1, 2]}],
        "cases": [{"id": "baseline", "set": {}}],
        "varying_fields": ["/seed"],
        "collection": {
            "expected_seeds": [0, 1, 2],
            "required_cases": ["baseline"],
            "group_by": ["/run/architecture/profile"],
        },
    })
    layout = ResultLayout(tmp_path / "results")
    _, manifest_path = publish_manifest(
        sweep,
        layout,
        {
            "git_revision": "1" * 40,
            "dirty_source_digest": "2" * 64,
            "effective_code_fingerprint": "3" * 64,
        },
    )
    sweep_dir = manifest_path.parent
    result = submit_sweep(
        sweep_dir=sweep_dir,
        profile=REPO_ROOT / "configs" / "executors" / "jeanzay-v100.json",
        results_root=layout.root,
        dataset_root=tmp_path / "data",
        device="cuda",
        dry_run=True,
        retry_failed=True,
        recover_stale=True,
        repo_root=REPO_ROOT,
    )

    assert result["submitted"] is False
    assert "--array=0-2%3" in result["worker"]
    assert "--dependency=afterany:{worker_job_id}" in result["collector"]
    export = next(item for item in result["worker"] if item.startswith("--export="))
    assert "MNIST_CONV_RETRY_FAILED=1" in export
    assert "MNIST_CONV_RECOVER_STALE=1" in export
    forbidden = (
        "learning_rate",
        "input_gain",
        "inference_iterations",
        "training_iterations",
        "diode",
    )
    option_names = [
        token.split("=", 1)[0].lower()
        for token in result["worker"] + result["collector"]
        if token.startswith("--")
    ]
    export_keys = [
        assignment.split("=", 1)[0].lower()
        for assignment in export.removeprefix("--export=").split(",")
    ]
    operational_surface = option_names + export_keys
    assert not any(
        item in name for item in forbidden for name in operational_surface
    )


def test_executor_profile_and_wrappers_are_operational_only() -> None:
    profile_path = REPO_ROOT / "configs" / "executors" / "jeanzay-v100.json"
    profile = load_profile(profile_path)
    profile_text = json.dumps(profile).lower()
    for forbidden in ("learning_rate", "input_gain", "architecture", "iteration", "diode", "seed"):
        assert forbidden not in profile_text

    wrappers = [
        REPO_ROOT / "experiments" / "run_mnist_conv_local.sh",
        REPO_ROOT / "experiments" / "run_mnist_conv_slurm_array.sh",
        REPO_ROOT / "experiments" / "collect_mnist_conv_slurm.sh",
    ]
    subprocess.run(["bash", "-n", *map(str, wrappers)], check=True)
    local_text = wrappers[0].read_text()
    assert "--executor local" in local_text
    assert "--config" in local_text


@pytest.mark.parametrize(
    "path",
    [
        "run_mnist_bp_conv_amp_calibrated_manifest_lane.sh",
        "run_mnist_bp_conv_amp_calibrated_manifest_jeanzay.slurm",
        "run_mnist_bp_conv_amp_calibrated_local_pipeline.sh",
        "run_mnist_bp_conv2_amp_calibrated_calibrate_and_submit_jeanzay.slurm",
    ],
)
def test_deprecated_calibrated_launchers_fail_closed(path: str) -> None:
    completed = subprocess.run(
        ["bash", str(REPO_ROOT / "experiments" / path)],
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 2
    assert "Expected" in completed.stderr
    assert "Provided" in completed.stderr
