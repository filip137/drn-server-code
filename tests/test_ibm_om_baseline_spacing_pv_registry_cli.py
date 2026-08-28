from __future__ import annotations

import json
from pathlib import Path
import sys
from types import ModuleType

import pytest

from ebl.cli import ValidateRequest, main
from experiments.definitions import (
    EXPERIMENT_REGISTRY,
    resolve_experiment_config,
)
from experiments.mnist_relu_drn.ibm_om_baseline_spacing_pv_config import (
    EXPERIMENT_ID,
    STUDY_ID,
    BaselineSpacingPvValidateSpec,
)
from experiments.schema import ConfigError, RunMode
from experiments.study_workflow import load_study_plan, prepare_study


ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = (
    ROOT / "examples" / "mnist_relu_drn" / "ibm_om_baseline_spacing_pv"
)
CONFIG_PATH = CONFIG_ROOT / "alpha_000_spacing_4delta-heldout-87001.json"
STUDY_PATH = ROOT / "studies" / f"{STUDY_ID}.json"
RUNTIME_MODULE = (
    "experiments.mnist_relu_drn.ibm_om_baseline_spacing_pv_runtime"
)


def test_registry_exposes_validate_only_baseline_spacing_pv() -> None:
    definition = EXPERIMENT_REGISTRY[EXPERIMENT_ID]

    assert definition.experiment_id == EXPERIMENT_ID
    assert definition.schema_version == 1
    assert definition.supported_modes == (RunMode.VALIDATE,)

    registered, spec = resolve_experiment_config(CONFIG_PATH, RunMode.VALIDATE)
    assert registered is definition
    assert isinstance(spec, BaselineSpacingPvValidateSpec)
    assert spec.student.runtime.device == "cuda"
    assert spec.protocol.baseline_position_fraction == 0.0
    assert spec.protocol.spacing_delta_x_multiplier == 4

    with pytest.raises(ConfigError, match="to be one of validate"):
        definition.resolve(
            definition.parse(json.loads(CONFIG_PATH.read_text())),
            RunMode.TRAIN,
        )


def test_default_cli_lazily_dispatches_baseline_spacing_validate(
    tmp_path: Path,
    monkeypatch,
) -> None:
    runtime = ModuleType(RUNTIME_MODULE)
    seen: list[ValidateRequest] = []

    def run_validate(request: ValidateRequest) -> int:
        seen.append(request)
        return 37

    runtime.run_validate = run_validate  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, RUNTIME_MODULE, runtime)

    weights = ROOT / "data" / "mnist_relu_teacher_fixed_init_20260816.pt"
    result = main(
        [
            "validate",
            "--config",
            str(CONFIG_PATH),
            "--output-dir",
            str(tmp_path / "runs"),
            "--weights",
            str(weights),
            "--teacher-weights",
            str(weights),
        ]
    )

    assert result == 37
    assert len(seen) == 1
    request = seen[0]
    assert request.definition is EXPERIMENT_REGISTRY[EXPERIMENT_ID]
    assert isinstance(request.spec, BaselineSpacingPvValidateSpec)
    assert request.config_path == CONFIG_PATH
    assert request.output_dir == tmp_path / "runs"
    assert request.weights == weights
    assert request.teacher_weights == weights
    assert request.device_model is None


def test_study_plan_prepares_and_every_declared_config_resolves(
    tmp_path: Path,
) -> None:
    plan = load_study_plan(STUDY_PATH)
    declared = [
        config
        for arm in plan["arms"]
        for config in arm["configs"]
    ]

    assert plan["study_id"] == STUDY_ID
    assert len(plan["arms"]) == 9
    assert len(declared) == 27
    assert all(
        "smoke-" not in Path(item["resolved_path"]).name
        for item in declared
    )
    for item in declared:
        definition, spec = resolve_experiment_config(
            item["resolved_path"],
            RunMode.VALIDATE,
        )
        assert definition is EXPERIMENT_REGISTRY[EXPERIMENT_ID]
        assert isinstance(spec, BaselineSpacingPvValidateSpec)

    study_root = prepare_study(STUDY_PATH, tmp_path / "results")
    assert study_root == (tmp_path / "results" / STUDY_ID).resolve()
    prepared = json.loads((study_root / "study.json").read_text())
    assert prepared["study_id"] == STUDY_ID
    assert len(prepared["arms"]) == 9
    assert sum(len(arm["configs"]) for arm in prepared["arms"]) == 27
    assert all(
        (study_root / "runs" / arm["arm_id"]).is_dir()
        for arm in prepared["arms"]
    )
