from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import sys
from types import ModuleType

import pytest

from ebl.cli import TrainRequest, main
from experiments.definitions import EXPERIMENT_REGISTRY, parse_experiment_config
from experiments.mnist_relu_drn.ibm_om_winsorized_multi_assignment_qat_config import (
    DEVELOPMENT_ASSIGNMENT_SEED,
    EXPECTED_TEACHER_WEIGHTS_PATH,
    EXPERIMENT_ID,
    FINAL_ASSIGNMENT_SEED,
    FINAL_ENDPOINT_SEEDS,
    LEARNING_RATES,
    SPACING_DELTA_X_MULTIPLIERS,
    TRAINING_ASSIGNMENT_SEEDS,
    WinsorizedMultiAssignmentQatTrainSpec,
    parse_winsorized_multi_assignment_qat_config,
    resolve_winsorized_multi_assignment_qat_spec,
)
from experiments.schema import ConfigError, RunMode


ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = (
    ROOT
    / "examples"
    / "mnist_relu_drn"
    / "ibm_om_winsorized_multi_assignment_qat"
)
RUNTIME_MODULE = (
    "experiments.mnist_relu_drn."
    "ibm_om_winsorized_multi_assignment_qat_runtime"
)


def _paths() -> tuple[Path, ...]:
    return tuple(
        CONFIG_ROOT / f"alpha_000_spacing_{spacing}delta.json"
        for spacing in SPACING_DELTA_X_MULTIPLIERS
    )


def _payload(path: Path | None = None) -> dict:
    selected = path or _paths()[0]
    return json.loads(selected.read_text(encoding="utf-8"))


def test_exact_three_multi_assignment_configs_resolve_with_disjoint_roles() -> None:
    assert set(CONFIG_ROOT.glob("*.json")) == set(_paths())
    observed = set()
    for path in _paths():
        spec = resolve_winsorized_multi_assignment_qat_spec(
            parse_winsorized_multi_assignment_qat_config(_payload(path)),
            RunMode.TRAIN,
        )
        assert isinstance(spec, WinsorizedMultiAssignmentQatTrainSpec)
        assert spec.student.runtime.device == "cuda"
        assert spec.student.settings.num_epochs == 10
        assert spec.protocol.training.optimizer == "sgd"
        assert spec.protocol.training.learning_rates == LEARNING_RATES
        assert spec.protocol.training.program_verify is False
        assert spec.protocol.training.read_noise is False
        assert spec.protocol.assignments.training_assignment_seeds == (
            TRAINING_ASSIGNMENT_SEEDS
        )
        assert spec.protocol.assignments.development_assignment_seed == (
            DEVELOPMENT_ASSIGNMENT_SEED
        )
        assert spec.protocol.assignments.final_assignment_seed == (
            FINAL_ASSIGNMENT_SEED
        )
        assert spec.protocol.assignments.final_endpoint_seeds == FINAL_ENDPOINT_SEEDS
        roles = {item.assignment_seed: item.role for item in spec.protocol.predecessors}
        assert roles == {
            86001: "training_cycle_0",
            87001: "training_cycle_1",
            87002: "development_selection",
            87003: "final_heldout",
        }
        assert set(TRAINING_ASSIGNMENT_SEEDS).isdisjoint(
            {DEVELOPMENT_ASSIGNMENT_SEED, FINAL_ASSIGNMENT_SEED}
        )
        assert DEVELOPMENT_ASSIGNMENT_SEED != FINAL_ASSIGNMENT_SEED
        observed.add(spec.protocol.mapping.spacing_delta_x_multiplier)
    assert observed == set(SPACING_DELTA_X_MULTIPLIERS)


@pytest.mark.parametrize(
    ("section", "field", "replacement"),
    [
        ("assignments", "development_assignment_seed", 87003),
        ("assignments", "training_assignment_seeds", [86001, 87002]),
        ("training", "optimizer", "adam"),
        ("training", "program_verify", True),
        ("evaluation", "fixed_headline_epoch", 9),
    ],
)
def test_multi_assignment_protocol_rejects_role_or_training_drift(
    section: str, field: str, replacement: object
) -> None:
    payload = deepcopy(_payload())
    payload["winsorized_multi_assignment_qat"][section][field] = replacement
    with pytest.raises(ConfigError, match=field):
        parse_winsorized_multi_assignment_qat_config(payload)


def test_predecessor_contract_rejects_numeric_type_coercion() -> None:
    payload = deepcopy(_payload())
    payload["winsorized_multi_assignment_qat"]["predecessors"][0][
        "assignment_seed"
    ] = 86001.0
    with pytest.raises(ConfigError, match="predecessor"):
        parse_winsorized_multi_assignment_qat_config(payload)


def test_registry_and_cli_dispatch_are_train_only_and_lazy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    definition = EXPERIMENT_REGISTRY[EXPERIMENT_ID]
    assert definition.supported_modes == (RunMode.TRAIN,)
    selected, document = parse_experiment_config(_payload())
    assert selected is definition
    assert document.experiment_id == EXPERIMENT_ID

    runtime = ModuleType(RUNTIME_MODULE)
    seen: list[TrainRequest] = []

    def run_train(request: TrainRequest) -> int:
        seen.append(request)
        return 29

    runtime.run_train = run_train  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, RUNTIME_MODULE, runtime)
    result = main(
        [
            "train",
            "--config",
            str(_paths()[0]),
            "--output-dir",
            str(tmp_path / "run"),
            "--teacher-weights",
            str(ROOT / EXPECTED_TEACHER_WEIGHTS_PATH),
        ]
    )
    assert result == 29
    assert len(seen) == 1
    assert seen[0].definition is definition
    assert isinstance(seen[0].spec, WinsorizedMultiAssignmentQatTrainSpec)
