from __future__ import annotations

import json
from pathlib import Path
import sys
from types import ModuleType

import pytest

from ebl.cli import TrainRequest, main
from experiments.definitions import EXPERIMENT_REGISTRY, resolve_experiment_config
from experiments.mnist_relu_drn.ibm_om_population_hwa_cross_array_config import (
    ARRAY_ASSIGNMENT_SEEDS,
    EXPERIMENT_ID,
    HWA_EPOCHS,
    NOMINAL_DELTA_X,
    PopulationHwaCrossArrayTrainSpec,
    parse_population_hwa_cross_array_config,
)
from experiments.schema import ConfigError, RunMode


ROOT = Path(__file__).resolve().parents[1]
FULL_CONFIG = (
    ROOT
    / "examples/mnist_relu_drn/ibm_om_population_hwa_cross_array/full.json"
)
SMOKE_CONFIG = (
    ROOT
    / "examples/mnist_relu_drn/ibm_om_population_hwa_cross_array/smoke.json"
)
RUNTIME_MODULE = (
    "experiments.mnist_relu_drn.ibm_om_population_hwa_cross_array_runtime"
)


@pytest.mark.parametrize(
    ("path", "profile", "endpoint_repeats"),
    (
        (FULL_CONFIG, "exploratory_full", 4),
        (SMOKE_CONFIG, "smoke", 1),
    ),
)
def test_config_resolves_positive_g_population_hwa_ladder(
    path: Path,
    profile: str,
    endpoint_repeats: int,
) -> None:
    definition, spec = resolve_experiment_config(path, RunMode.TRAIN)
    assert definition is EXPERIMENT_REGISTRY[EXPERIMENT_ID]
    assert isinstance(spec, PopulationHwaCrossArrayTrainSpec)
    protocol = spec.protocol
    assert protocol.profile == profile
    assert protocol.evidence_tier == "exploratory_noncanonical"
    assert protocol.initialization["global_physical_bounds"] == [0.0, 2.0]
    assert protocol.initialization["array_A_role"].endswith(
        "not_HWA_state_or_calibration"
    )
    assert protocol.hwa["epochs"] == HWA_EPOCHS
    assert protocol.hwa["strength_schedule"] == "linear_one_indexed_epoch_ramp"
    assert protocol.hwa["sample_scope"] == (
        "one_fresh_draw_per_physical_cell_per_minibatch"
    )
    assert protocol.hwa["fixed_array_identity"] is False
    assert protocol.hwa["fixed_per_cell_bounds"] is False
    assert protocol.hwa["fixed_reference"] is False
    assert protocol.hwa["corrupt_mask"] is False
    assert [arm.arm_id for arm in protocol.hwa_arms] == [
        "population_hwa_continuous",
        "population_hwa_one_delta_qat",
    ]
    assert protocol.hwa_arms[0].quantization_spacing_raw_x is None
    assert protocol.hwa_arms[1].quantization_spacing_raw_x == pytest.approx(
        NOMINAL_DELTA_X
    )
    assert [array.label for array in protocol.arrays] == ["A", "B", "C", "D"]
    assert [array.assignment_seed for array in protocol.arrays] == [
        ARRAY_ASSIGNMENT_SEEDS[label] for label in ("A", "B", "C", "D")
    ]
    assert len(protocol.arrays[0].endpoint_seeds) == endpoint_repeats
    assert protocol.arrays[0].opening_phase == (
        "before_hwa_raw_relu_baseline_only"
    )
    assert all(
        array.opening_phase == "after_both_hwa_checkpoints_are_frozen"
        for array in protocol.arrays[1:]
    )
    assert protocol.deployment["physical_populations"] == [
        "counterfactual_repaired",
        "published_corrupt",
    ]
    assert protocol.deployment["inference_state"] == (
        "persistent_full_G_equals_2x"
    )
    assert protocol.deployment["apparent_endpoint_applied_to_drn"] is False
    assert protocol.deployment["target_clipping"] is False
    assert spec.student.model.conductance_min == 0.0
    assert spec.student.model.conductance_max == 2.0


@pytest.mark.parametrize(
    "payload",
    (
        {
            "schema_version": 1,
            "experiment_id": EXPERIMENT_ID,
            "profile": "canonical",
        },
        {
            "schema_version": 1,
            "experiment_id": EXPERIMENT_ID,
            "profile": "smoke",
            "array_A": 87004,
        },
    ),
)
def test_config_rejects_undeclared_protocol_drift(
    payload: dict[str, object],
) -> None:
    with pytest.raises(ConfigError):
        parse_population_hwa_cross_array_config(payload)


def test_default_cli_dispatch_forwards_all_three_explicit_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = ModuleType(RUNTIME_MODULE)
    seen: list[TrainRequest] = []

    def run_train(request: TrainRequest) -> int:
        seen.append(request)
        return 97

    runtime.run_train = run_train  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, RUNTIME_MODULE, runtime)
    teacher = tmp_path / "teacher.pt"
    endpoint = tmp_path / "raw_active_endpoint.json"
    estimator = tmp_path / "raw_active_estimators.json"
    for path in (teacher, endpoint, estimator):
        path.write_text(json.dumps({"placeholder": True}), encoding="utf-8")
    result = main(
        [
            "train",
            "--config",
            str(SMOKE_CONFIG),
            "--output-dir",
            str(tmp_path / "run"),
            "--teacher-weights",
            str(teacher),
            "--device-model",
            str(endpoint),
            "--device-data",
            str(estimator),
        ]
    )
    assert result == 97
    assert len(seen) == 1
    request = seen[0]
    assert request.definition is EXPERIMENT_REGISTRY[EXPERIMENT_ID]
    assert request.teacher_weights == teacher
    assert request.device_model == endpoint
    assert request.device_data == estimator
    assert request.weights is None
    assert request.base_weights is None
