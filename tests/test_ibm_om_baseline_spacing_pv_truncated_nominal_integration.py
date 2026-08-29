from __future__ import annotations

import json
from pathlib import Path
import sys
from types import ModuleType

import pytest

from ebl.cli import ValidateRequest, main
from experiments.definitions import EXPERIMENT_REGISTRY, parse_experiment_config
from experiments.mnist_relu_drn.ibm_om_baseline_spacing_pv_truncated_nominal_config import (
    BASELINE_POSITION_FRACTIONS,
    ENDPOINT_SEEDS_BY_ASSIGNMENT,
    EXPERIMENT_ID,
    HELDOUT_ASSIGNMENT_SEEDS,
    SPACING_DELTA_X_MULTIPLIERS,
    BaselineSpacingPvTruncatedNominalValidateSpec,
    parse_baseline_spacing_pv_truncated_nominal_config,
    resolve_baseline_spacing_pv_truncated_nominal_spec,
)
from experiments.schema import ConfigError, RunMode


ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = (
    ROOT
    / "examples"
    / "mnist_relu_drn"
    / "ibm_om_baseline_spacing_pv_truncated_nominal"
)
RUNTIME_MODULE = (
    "experiments.mnist_relu_drn."
    "ibm_om_baseline_spacing_pv_truncated_nominal_runtime"
)


def _alpha_stem(alpha: float) -> str:
    return {0.0: "000", 0.25: "025", 0.5: "050"}[alpha]


def _production_paths() -> tuple[Path, ...]:
    return tuple(
        CONFIG_ROOT
        / (
            f"alpha_{_alpha_stem(alpha)}_spacing_{spacing}delta-"
            f"heldout-{seed}.json"
        )
        for alpha in BASELINE_POSITION_FRACTIONS
        for spacing in SPACING_DELTA_X_MULTIPLIERS
        for seed in HELDOUT_ASSIGNMENT_SEEDS
    )


def _payload(path: Path | None = None) -> dict:
    selected = path or _production_paths()[0]
    return json.loads(selected.read_text(encoding="utf-8"))


def test_exact_twenty_seven_cuda_configs_resolve_with_full_matrix() -> None:
    expected = set(_production_paths())
    assert set(CONFIG_ROOT.glob("*.json")) == expected
    observed = set()
    for path in sorted(expected):
        document = parse_baseline_spacing_pv_truncated_nominal_config(
            _payload(path)
        )
        spec = resolve_baseline_spacing_pv_truncated_nominal_spec(
            document, RunMode.VALIDATE
        )
        assert isinstance(spec, BaselineSpacingPvTruncatedNominalValidateSpec)
        assert spec.student.runtime.device == "cuda"
        assert spec.student.settings.sample_limit is None
        assert spec.protocol.execution.evidence_tier == "exploratory_noncanonical"
        assert spec.protocol.assignments.endpoint_seeds == (
            ENDPOINT_SEEDS_BY_ASSIGNMENT[
                spec.protocol.assignments.heldout_seed
            ]
        )
        observed.add(
            (
                spec.protocol.baseline_position_fraction,
                spec.protocol.spacing_delta_x_multiplier,
                spec.protocol.assignments.heldout_seed,
            )
        )
    assert observed == {
        (alpha, spacing, seed)
        for alpha in BASELINE_POSITION_FRACTIONS
        for spacing in SPACING_DELTA_X_MULTIPLIERS
        for seed in HELDOUT_ASSIGNMENT_SEEDS
    }


def test_contract_declares_counterfactual_preprogram_winsorization() -> None:
    spec = resolve_baseline_spacing_pv_truncated_nominal_spec(
        parse_baseline_spacing_pv_truncated_nominal_config(_payload()),
        RunMode.VALIDATE,
    )
    assert spec.protocol.device.evidence_class == (
        "model_based_aihwkit_preset_bound_winsorization_control"
    )
    assert spec.protocol.truncation.policy == (
        "nominal_bound_winsorization_no_rejection_or_resampling"
    )
    assert spec.protocol.truncation.operation_order == (
        "sample_frozen_identity_then_winsorize_bounds_then_commission_then_pv"
    )
    assert spec.protocol.truncation.raw_a_minimum == -1.0
    assert spec.protocol.truncation.raw_a_maximum == 1.0
    assert spec.protocol.truncation.conductance_formula == "G=a_truncated+1=2*x"
    assert spec.protocol.program_verify.circuit_handoff == (
        "G=2*x_persistent_without_projection"
    )
    assert spec.protocol.truncation.endpoint_policy.endswith(
        "no_post_handoff_clipping"
    )


def test_registry_is_additive_and_validate_only() -> None:
    definition = EXPERIMENT_REGISTRY[EXPERIMENT_ID]
    assert definition.supported_modes == (RunMode.VALIDATE,)
    assert "counterfactual" in definition.description
    selected, document = parse_experiment_config(_payload())
    assert selected is definition
    assert document.experiment_id == EXPERIMENT_ID
    with pytest.raises(ConfigError, match="to be one of validate"):
        definition.resolve(document, RunMode.TRAIN)


def test_default_cli_lazily_dispatches_winsorized_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = ModuleType(RUNTIME_MODULE)
    seen: list[ValidateRequest] = []

    def run_validate(request: ValidateRequest) -> int:
        seen.append(request)
        return 37

    runtime.run_validate = run_validate  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, RUNTIME_MODULE, runtime)
    config_path = _production_paths()[0]
    weights = ROOT / "data" / "mnist_relu_teacher_fixed_init_20260816.pt"
    result = main(
        [
            "validate",
            "--config",
            str(config_path),
            "--output-dir",
            str(tmp_path / "run"),
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
    assert isinstance(
        request.spec, BaselineSpacingPvTruncatedNominalValidateSpec
    )
    assert request.config_path == config_path
    assert request.output_dir == tmp_path / "run"
    assert request.weights == weights
    assert request.teacher_weights == weights
