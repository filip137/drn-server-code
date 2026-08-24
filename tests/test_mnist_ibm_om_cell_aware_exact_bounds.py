from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments.artifacts import sha256_file
from experiments.definitions import resolve_experiment_config
from experiments.schema import ConfigError, RunMode
from experiments.study_workflow import load_study_plan
from experiments.mnist_relu_drn.runtime import (
    _modifier_forward_gain,
    _selection_improved,
)
from experiments.mnist_relu_drn.ibm_om_cell_aware_launcher import (
    DESIGN_RECEIPT,
    DESIGN_RECEIPT_SHA256,
    DEVICE_MODEL,
    FULL_SPAN,
    SCHEDULE_RECEIPT,
    SCHEDULE_RECEIPT_SHA256,
    TEACHER,
    _command,
    _completed_run,
    _task_environment,
    development_tasks,
    heldout_tasks,
)
from experiments.mnist_relu_drn.ibm_om_cell_aware_analysis import (
    _paired_bootstrap,
)
from training.ibm_reram_hwa import IbmReramHwaParameterModifier


_ROOT = Path(__file__).resolve().parents[1]
_CONFIG_ROOT = (
    _ROOT
    / "examples"
    / "mnist_relu_drn"
    / "ibm_om_cell_aware_exact_bounds"
)
_STUDY = (
    _ROOT
    / "studies"
    / "mnist-ibm-om-cell-aware-exact-bounds-transfer-20260824-v1.json"
)
_LAYOUTS = {
    "base.dense_weight.0": "halves",
    "base.dense_weight.1": "paired",
}


def _assert_modifier(
    parameters,
    *,
    execution: str,
    assignment_seed: int,
    endpoint_seed: int,
    mode: str,
) -> None:
    assert parameters["execution"] == execution
    assert parameters["assignment_seed"] == assignment_seed
    assert parameters["endpoint_seed"] == endpoint_seed
    assert parameters["corruption_policy"] == "counterfactual_repaired"
    assert parameters["noisy_evaluation"] is (execution == "pulse_resolved")
    assert parameters["target_mapping"] == "cell_aware_exact_bounds_quad"
    assert dict(parameters["dual_rail_layout_by_parameter"]) == _LAYOUTS
    assert parameters["common_window_margin_fraction"] == 0.0
    assert parameters["cell_aware_mode"] == mode
    assert parameters["cell_aware_signed_levels"] == 9
    assert parameters["forward_logit_gain"] == (
        3.5481338923357533 if mode == "continuous" else 2.818382931264453
    )
    assert parameters["maximum_program_pulses"] == 128
    assert parameters["target_out_of_support"] == "error"


def test_development_configs_reuse_reset_schedules_and_accuracy_selection() -> None:
    expected = {
        "zero_update_quantized.json": (0, (0.0, 0.0), None, "quantized_9_level"),
        "clean_bptt_quantized_deploy.json": (
            10,
            (0.1388429752066116, 0.00037685950413223146),
            None,
            "quantized_9_level",
        ),
        "continuous_hwa.json": (
            10,
            (0.0017531914893617025, 0.000004757787234042554),
            "continuous",
            "continuous",
        ),
        "quantized_qat.json": (
            10,
            (0.004411914893617021, 0.000011974808510638298),
            "quantized_9_level",
            "quantized_9_level",
        ),
    }
    specs = {}
    for name, (epochs, rates, training_mode, selection_mode) in expected.items():
        definition, spec = resolve_experiment_config(
            _CONFIG_ROOT / name,
            RunMode.TRAIN,
        )
        assert definition.experiment_id == "mnist_relu_drn_kd.v1"
        specs[name] = spec
        assert spec.settings.num_epochs == epochs
        assert spec.settings.learning_rates == rates
        assert spec.settings.selection_evaluation == "modifier"
        assert spec.settings.selection_metric == "student_accuracy"
        assert spec.settings.selection_noise_repeats == 3
        if training_mode is None:
            assert spec.settings.weight_modifier.type == "none"
        else:
            _assert_modifier(
                spec.settings.weight_modifier.parameters,
                execution="compact_endpoint",
                assignment_seed=84001,
                endpoint_seed=84002,
                mode=training_mode,
            )
        _assert_modifier(
            spec.settings.selection_weight_modifier.parameters,
            execution="pulse_resolved",
            assignment_seed=84001,
            endpoint_seed=84003,
            mode=selection_mode,
        )

    reference = specs["zero_update_quantized.json"]
    for spec in specs.values():
        for field in ("runtime", "data", "teacher", "model", "solver", "mapping"):
            assert getattr(spec, field) == getattr(reference, field)


def test_design_receipt_locks_development_population_and_gain_fit() -> None:
    receipt = json.loads(DESIGN_RECEIPT.read_text(encoding="utf-8"))
    assert sha256_file(DESIGN_RECEIPT) == DESIGN_RECEIPT_SHA256
    assert receipt["heldout_assignment_consulted"] is False
    assert receipt["assignment_seed"] == 84001
    assert receipt["population_fingerprint"] == (
        "4887ad89abdd16193448c54a0cbe97be915cb4b9bc04cf1151c5e2a96ee5a3ac"
    )
    assert receipt["gain_fit"]["continuous"]["gain"] == 3.5481338923357533
    assert (
        receipt["gain_fit"]["quantized_9_level"]["gain"]
        == 2.818382931264453
    )
    assert receipt["diagnostic_only_ideal_full_test"]["selection_use"] is False
    assert sha256_file(SCHEDULE_RECEIPT) == SCHEDULE_RECEIPT_SHA256


@pytest.mark.parametrize("mode", ["continuous", "quantized"])
@pytest.mark.parametrize("endpoint_seed", range(85101, 85106))
def test_heldout_configs_use_fresh_assignment_and_one_exact_endpoint_seed(
    mode: str,
    endpoint_seed: int,
) -> None:
    definition, spec = resolve_experiment_config(
        _CONFIG_ROOT / f"heldout_{mode}_seed_{endpoint_seed}.json",
        RunMode.VALIDATE,
    )
    assert definition.experiment_id == "mnist_relu_drn_kd.v1"
    assert spec.settings.split == "test"
    assert spec.settings.sample_limit is None
    assert spec.settings.noise_repeats == 1
    _assert_modifier(
        spec.settings.weight_modifier.parameters,
        execution="pulse_resolved",
        assignment_seed=85001,
        endpoint_seed=endpoint_seed,
        mode=("continuous" if mode == "continuous" else "quantized_9_level"),
    )


def test_cell_aware_config_rejects_wrong_levels_layout_or_missing_gain(
    tmp_path: Path,
) -> None:
    source = _CONFIG_ROOT / "quantized_qat.json"
    payload = json.loads(source.read_text(encoding="utf-8"))
    parameters = payload["modes"]["train"]["weight_modifier"]["parameters"]
    parameters["cell_aware_signed_levels"] = 7
    path = tmp_path / "wrong-levels.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ConfigError, match="to equal 9"):
        resolve_experiment_config(path, RunMode.TRAIN)

    payload = json.loads(source.read_text(encoding="utf-8"))
    parameters = payload["modes"]["train"]["weight_modifier"]["parameters"]
    parameters["dual_rail_layout_by_parameter"]["base.dense_weight.1"] = "halves"
    path = tmp_path / "wrong-layout.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ConfigError, match="canonical W1/W2 layout"):
        resolve_experiment_config(path, RunMode.TRAIN)

    payload = json.loads(source.read_text(encoding="utf-8"))
    parameters = payload["modes"]["train"]["weight_modifier"]["parameters"]
    parameters.pop("forward_logit_gain")
    path = tmp_path / "missing-gain.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ConfigError, match="forward_logit_gain"):
        resolve_experiment_config(path, RunMode.TRAIN)


def test_study_declares_four_development_and_twenty_heldout_runs() -> None:
    study = load_study_plan(_STUDY)
    assert study["study_id"] == (
        "mnist-ibm-om-cell-aware-exact-bounds-transfer-20260824-v1"
    )
    assert study["evidence_class"] == "model_based_aihwkit_preset"
    assert len(study["arms"]) == 8
    train = [arm for arm in study["arms"] if arm["mode"] == "train"]
    validate = [arm for arm in study["arms"] if arm["mode"] == "validate"]
    assert len(train) == 4
    assert sum(len(arm["configs"]) for arm in validate) == 20
    joined = "\n".join(
        [study["motivation"], *study["completion_criteria"], *study["analysis_plan"]]
    )
    assert "hidden bounds were consumed" in joined
    assert "Assignment seed 85001 must not be sampled" in joined
    assert "0.1388429752066116" in joined
    assert "3.5481338923357533" in joined
    assert "2.818382931264453" in joined
    assert "at least 90 percent" in joined
    assert "No on-chip" in joined


def test_device_forward_gain_is_temporary_and_selection_ties_keep_earliest() -> None:
    stack = SimpleNamespace(cost=SimpleNamespace(gain=4.46683592150963))
    modifier = IbmReramHwaParameterModifier.__new__(
        IbmReramHwaParameterModifier
    )
    modifier._config = SimpleNamespace(forward_logit_gain=2.818382931264453)
    with _modifier_forward_gain(stack, modifier, evaluation=True):
        assert stack.cost.gain == 2.818382931264453
    assert stack.cost.gain == 4.46683592150963

    incumbent = {"student_accuracy": 0.7, "kl_teacher_student": 0.4}
    assert not _selection_improved(
        dict(incumbent), incumbent, metric="student_accuracy"
    )
    assert _selection_improved(
        {"student_accuracy": 0.71}, incumbent, metric="student_accuracy"
    )
    assert _selection_improved(
        {"kl_teacher_student": 0.39}, incumbent, metric="kl_teacher_student"
    )


def test_launcher_declares_four_development_then_twenty_heldout_public_cli_tasks() -> None:
    development = development_tasks()
    assert len(development) == 4
    assert all(task.mode == "train" for task in development)
    assert all(task.weights == FULL_SPAN for task in development)
    frozen = {
        "checkpoints": {
            source_arm: {
                "weights": str(FULL_SPAN),
                "weights_sha256": (
                    "a99995a3e5b321a56bb7e00e29840c3b76f3fee95b8d8c80fdf0fa16e93b3563"
                ),
            }
            for source_arm in (
                "train-zero-update-quantized",
                "train-clean-bptt-quantized-deploy",
                "train-continuous-cell-aware-hwa",
                "train-quantized-cell-aware-qat",
            )
        }
    }
    heldout = heldout_tasks(frozen)
    assert len(heldout) == 20
    assert all(task.mode == "validate" for task in heldout)
    assert {
        int(task.config.stem.rsplit("_", 1)[1]) for task in heldout
    } == set(range(85101, 85106))
    command = _command(
        heldout[0],
        python=Path("/pinned/python"),
        study_dir=Path("/study"),
    )
    assert command[:4] == ["/pinned/python", "-m", "ebl", "validate"]
    assert command[command.index("--teacher-weights") + 1] == str(TEACHER)
    assert command[command.index("--device-model") + 1] == str(DEVICE_MODEL)


def test_launcher_can_inherit_or_explicitly_pin_cuda_visibility() -> None:
    inherited = _task_environment(
        {"CUDA_VISIBLE_DEVICES": "GPU-existing"},
        aihwkit_python="/pinned/aihwkit-python",
        cuda_visible_devices="inherit",
    )
    assert inherited["CUDA_VISIBLE_DEVICES"] == "GPU-existing"
    assert inherited["EBL_AIHWKIT_PYTHON"] == "/pinned/aihwkit-python"
    assert inherited["EBL_DEFER_CURRENT_SIMULATIONS"] == "1"
    assert inherited["PYTHONUNBUFFERED"] == "1"

    explicit = _task_environment(
        {},
        aihwkit_python="/pinned/aihwkit-python",
        cuda_visible_devices="0,2",
    )
    assert explicit["CUDA_VISIBLE_DEVICES"] == "0,2"
    with pytest.raises(RuntimeError, match="comma-separated"):
        _task_environment(
            {},
            aihwkit_python="/pinned/aihwkit-python",
            cuda_visible_devices="",
        )


def test_launcher_recognizes_native_complete_status(tmp_path: Path) -> None:
    config = tmp_path / "config.json"
    config.write_text("{}\n", encoding="utf-8")
    run_dir = tmp_path / "arm" / "run"
    run_dir.mkdir(parents=True)
    (run_dir / "status.json").write_text(
        json.dumps({"status": "complete"}), encoding="utf-8"
    )
    (run_dir / "manifest.json").write_text(
        json.dumps(
            {
                "study": {
                    "source_config_sha256": sha256_file(config),
                }
            }
        ),
        encoding="utf-8",
    )
    assert _completed_run(tmp_path / "arm", config) == run_dir


def test_primary_paired_bootstrap_is_deterministic_and_endpoint_paired() -> None:
    values = [0.01, 0.02, 0.03, 0.04, 0.05]
    first = _paired_bootstrap(values)
    second = _paired_bootstrap(values)
    assert first == second
    assert first["seed"] == 20260824
    assert first["replicates"] == 100_000
    assert first["lower"] <= sum(values) / 5 <= first["upper"]
