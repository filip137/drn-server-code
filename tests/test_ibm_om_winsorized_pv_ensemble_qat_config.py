from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import sys
from types import ModuleType

import pytest

from ebl.cli import TrainRequest, main
from experiments.definitions import EXPERIMENT_REGISTRY, parse_experiment_config
from experiments.mnist_relu_drn.ibm_om_winsorized_pv_ensemble_qat_config import (
    ARMS,
    DEVELOPMENT_ASSIGNMENT_SEED,
    DEVELOPMENT_ENDPOINT_SEEDS,
    EXPECTED_TEACHER_WEIGHTS_PATH,
    EXPERIMENT_ID,
    FINAL_ASSIGNMENT_SEED,
    FINAL_ENDPOINT_SEEDS,
    NUM_EPOCHS,
    TRAINING_ASSIGNMENT_SEEDS,
    TRAINING_ENDPOINT_SEEDS,
    WinsorizedPvEnsembleQatTrainSpec,
    parse_winsorized_pv_ensemble_qat_config,
    resolve_winsorized_pv_ensemble_qat_spec,
)
from experiments.schema import ConfigError, RunMode


ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = (
    ROOT
    / "examples"
    / "mnist_relu_drn"
    / "ibm_om_winsorized_pv_ensemble_qat"
)
RUNTIME_MODULE = (
    "experiments.mnist_relu_drn.ibm_om_winsorized_pv_ensemble_qat_runtime"
)
EXPECTED_LOSS = {
    "deterministic": (0, 1.0, 0.0, 0.0, 0),
    "mean2": (2, 0.25, 0.75, 0.0, 4),
    "tail4": (4, 0.25, 0.50, 0.25, 4),
}


def _paths() -> tuple[Path, ...]:
    return tuple(CONFIG_ROOT / f"{arm}.json" for arm in ARMS)


def _payload(arm: str = "deterministic") -> dict:
    return json.loads((CONFIG_ROOT / f"{arm}.json").read_text(encoding="utf-8"))


def test_exact_three_pv_ensemble_configs_resolve_with_frozen_roles_and_losses() -> None:
    assert set(CONFIG_ROOT.glob("*.json")) == set(_paths())
    for arm, path in zip(ARMS, _paths()):
        spec = resolve_winsorized_pv_ensemble_qat_spec(
            parse_winsorized_pv_ensemble_qat_config(
                json.loads(path.read_text(encoding="utf-8"))
            ),
            RunMode.TRAIN,
        )
        assert isinstance(spec, WinsorizedPvEnsembleQatTrainSpec)
        assert spec.student.runtime.device == "cuda"
        assert spec.student.settings.num_epochs == NUM_EPOCHS == 3
        assert spec.protocol.mapping.baseline_position_fraction == 0.0
        assert spec.protocol.mapping.spacing_delta_x_multiplier == 1
        assert spec.protocol.assignments.training_assignment_seeds == (
            TRAINING_ASSIGNMENT_SEEDS
        )
        assert spec.protocol.assignments.development_assignment_seed == (
            DEVELOPMENT_ASSIGNMENT_SEED
        )
        assert spec.protocol.assignments.final_assignment_seed == FINAL_ASSIGNMENT_SEED
        assert spec.protocol.assignments.training_endpoint_seeds == (
            TRAINING_ENDPOINT_SEEDS
        )
        assert spec.protocol.assignments.development_endpoint_seeds == (
            DEVELOPMENT_ENDPOINT_SEEDS
        )
        assert spec.protocol.assignments.final_endpoint_seeds == FINAL_ENDPOINT_SEEDS
        training = spec.protocol.training
        assert (
            training.persistent_samples_per_minibatch,
            training.ideal_loss_weight,
            training.mean_pv_loss_weight,
            training.tail_pv_loss_weight,
            training.training_endpoint_bank_size,
        ) == EXPECTED_LOSS[arm]
        assert training.apparent_verify_applied_to_drn is False
        assert training.post_program_projection == "none"
        assert training.read_noise is False
        assert spec.protocol.evaluation.fixed_headline_epoch == 3

        assignment_roles = set(TRAINING_ASSIGNMENT_SEEDS)
        assert assignment_roles.isdisjoint(
            {DEVELOPMENT_ASSIGNMENT_SEED, FINAL_ASSIGNMENT_SEED}
        )
        endpoint_roles = [
            set(TRAINING_ENDPOINT_SEEDS),
            set(DEVELOPMENT_ENDPOINT_SEEDS),
            set(FINAL_ENDPOINT_SEEDS),
        ]
        assert all(
            left.isdisjoint(right)
            for index, left in enumerate(endpoint_roles)
            for right in endpoint_roles[index + 1 :]
        )


@pytest.mark.parametrize(
    ("arm", "field", "replacement"),
    [
        ("deterministic", "persistent_samples_per_minibatch", 1),
        ("mean2", "mean_pv_loss_weight", 0.50),
        ("tail4", "tail_pv_loss_weight", 0.0),
        ("tail4", "post_program_projection", "clip_to_[0,1]"),
    ],
)
def test_pv_ensemble_config_rejects_arm_loss_or_projection_drift(
    arm: str, field: str, replacement: object
) -> None:
    payload = deepcopy(_payload(arm))
    payload["winsorized_pv_ensemble_qat"]["training"][field] = replacement
    with pytest.raises(ConfigError, match=field):
        parse_winsorized_pv_ensemble_qat_config(payload)


def test_pv_ensemble_config_rejects_endpoint_role_overlap() -> None:
    payload = deepcopy(_payload("mean2"))
    payload["winsorized_pv_ensemble_qat"]["assignments"][
        "development_endpoint_seeds"
    ][0] = TRAINING_ENDPOINT_SEEDS[0]
    with pytest.raises(ConfigError, match="development_endpoint_seeds"):
        parse_winsorized_pv_ensemble_qat_config(payload)


def test_registry_and_cli_dispatch_are_train_only_and_lazy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    definition = EXPERIMENT_REGISTRY[EXPERIMENT_ID]
    assert definition.supported_modes == (RunMode.TRAIN,)
    selected, document = parse_experiment_config(_payload("mean2"))
    assert selected is definition
    assert document.experiment_id == EXPERIMENT_ID

    runtime = ModuleType(RUNTIME_MODULE)
    seen: list[TrainRequest] = []

    def run_train(request: TrainRequest) -> int:
        seen.append(request)
        return 31

    runtime.run_train = run_train  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, RUNTIME_MODULE, runtime)
    result = main(
        [
            "train",
            "--config",
            str(CONFIG_ROOT / "mean2.json"),
            "--output-dir",
            str(tmp_path / "run"),
            "--teacher-weights",
            str(ROOT / EXPECTED_TEACHER_WEIGHTS_PATH),
        ]
    )
    assert result == 31
    assert len(seen) == 1
    assert seen[0].definition is definition
    assert isinstance(seen[0].spec, WinsorizedPvEnsembleQatTrainSpec)
