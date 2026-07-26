from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from experiments.mnist_conv import lr_stages
from experiments.mnist_conv.lr_conv3_legacy_gain200_bias2_rescue import (
    RESULT_SCHEMA_VERSION,
    SOURCE_RATES,
    TARGET_RATES,
    Bias2RescueSpec,
)
from experiments.mnist_conv.lr_protocol import (
    LRProtocolValidationError,
    layerwise_learning_rate_report,
)
from experiments.mnist_conv.lr_stages import execute_candidate_entry
from experiments.mnist_conv.lr_study_spec import LRStudySpec


REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG = (
    REPO_ROOT
    / "configs/conv/"
    "hardsigmoid_lr_conv3_legacy_gain200_bias2_0p1x_bs16_v1.json"
)
V7_CONFIG = (
    REPO_ROOT
    / "configs/conv/"
    "hardsigmoid_lr_conv3_scheme_two_rho_constant_sgd_bs16_v7.json"
)


def test_bias2_rescue_has_stable_separate_identity() -> None:
    spec = Bias2RescueSpec.from_path(CONFIG)
    assert spec.rescue_id == (
        "lrbias2_0135c2403354462edfe121ed198c0ff9"
        "73132303ff4f28ed7ca5d13bda29ef81"
    )
    assert spec.data["controlled_change"]["input_gain"] == 200.0
    assert spec.data["controlled_change"]["learning_rate_multiplier"] == 0.1
    assert spec.data["artifacts"]["mutate_parent_study"] is False
    assert spec.data["artifacts"]["mutate_source_rescue"] is False


def test_only_bias2_learning_rate_changes() -> None:
    changed = {
        name
        for name in SOURCE_RATES
        if SOURCE_RATES[name] != TARGET_RATES[name]
    }
    assert changed == {"Bias_2"}
    assert TARGET_RATES["Bias_2"] == pytest.approx(
        0.1 * SOURCE_RATES["Bias_2"], rel=0.0, abs=1e-20
    )


def test_bias2_rescue_rejects_an_extra_rate_change() -> None:
    value = json.loads(CONFIG.read_text())
    value["controlled_change"]["learning_rates_by_parameter"][
        "DenseWeight_0"
    ] *= 0.1
    with pytest.raises(
        ValueError,
        match="controlled_change.learning_rates_by_parameter.DenseWeight_0",
    ):
        Bias2RescueSpec.from_dict(value)


def test_progress_artifact_requires_controlled_wrapper(tmp_path: Path) -> None:
    study = LRStudySpec.from_path(V7_CONFIG)
    with pytest.raises(ValueError, match="progress_path only"):
        execute_candidate_entry(
            study.data,
            tmp_path,
            "conv3_legacy_v4_c0p25",
            "not-controlled",
            data_root=tmp_path,
            download=False,
            device="cpu",
            progress_path=tmp_path / "progress.json",
        )


def test_default_report_still_rejects_untied_bias_rate() -> None:
    with pytest.raises(
        LRProtocolValidationError,
        match=r"learning_rates\['Bias_2'\].*associated weight rate",
    ):
        layerwise_learning_rate_report(TARGET_RATES)


def test_report_accepts_only_the_explicit_controlled_bias_rate() -> None:
    report = layerwise_learning_rate_report(
        TARGET_RATES,
        bias_learning_rate_overrides={"Bias_2": TARGET_RATES["Bias_2"]},
    )
    assert report["bias_learning_rate_overrides"] == {
        "Bias_2": TARGET_RATES["Bias_2"]
    }
    assert report["weight_learning_rates"] == {
        name: rate
        for name, rate in TARGET_RATES.items()
        if name.startswith(("ConvWeight_", "DenseWeight_"))
    }
    assert "bias_learning_rate_overrides" not in (
        layerwise_learning_rate_report(SOURCE_RATES)
    )


def test_report_rejects_wrong_declared_bias_override_rate() -> None:
    with pytest.raises(
        LRProtocolValidationError,
        match=r"learning_rates\['Bias_2'\].*declared controlled override",
    ):
        layerwise_learning_rate_report(
            TARGET_RATES,
            bias_learning_rate_overrides={
                "Bias_2": SOURCE_RATES["Bias_2"]
            },
        )


@pytest.mark.parametrize("name", ["ConvWeight_2", "Bias_3"])
def test_report_rejects_noncanonical_bias_override(name: str) -> None:
    with pytest.raises(
        LRProtocolValidationError,
        match="bias_learning_rate_overrides",
    ):
        layerwise_learning_rate_report(
            TARGET_RATES,
            bias_learning_rate_overrides={name: TARGET_RATES["Bias_2"]},
        )


def test_controlled_bias_override_requires_controlled_wrapper(
    tmp_path: Path,
) -> None:
    study = LRStudySpec.from_path(V7_CONFIG)
    with pytest.raises(
        ValueError,
        match="controlled_bias_learning_rate_overrides only",
    ):
        execute_candidate_entry(
            study.data,
            tmp_path,
            "conv3_legacy_v4_c0p25",
            "not-controlled",
            data_root=tmp_path,
            download=False,
            device="cpu",
            controlled_bias_learning_rate_overrides={
                "Bias_2": TARGET_RATES["Bias_2"]
            },
        )


def _controlled_executor_case() -> tuple[
    LRStudySpec,
    dict[str, object],
    dict[str, object],
    dict[str, object],
    dict[str, object],
]:
    study = LRStudySpec.from_path(V7_CONFIG)
    row = next(
        value
        for value in study.data["rows"]
        if value["row_id"] == "conv3_legacy_v4_c0p25"
    )
    runtime_row = copy.deepcopy(row)
    runtime_row["input_gain"] = 200.0
    weights = (
        "ConvWeight_0",
        "ConvWeight_1",
        "ConvWeight_2",
        "DenseWeight_0",
    )
    payload = {
        "row_id": row["row_id"],
        "candidate_role": "legacy-gain200-bias2-0p1x",
        "candidate_stage": "legacy_gain_sensitivity",
        "rho_conv": 5e-5,
        "rho_dense": 3e-4,
        "median_units_by_weight": {name: 1.0 for name in weights},
        "learning_rates_by_parameter": dict(TARGET_RATES),
        "learning_rates_by_weight": {
            name: TARGET_RATES[name] for name in weights
        },
        "peak_learning_rate": max(TARGET_RATES.values()),
    }
    run_spec = {
        "schema_version": "test-controlled-bias-lr/v1",
        "learning_rates_by_parameter": dict(TARGET_RATES),
    }
    return study, row, runtime_row, payload, run_spec


def test_controlled_bias_override_passes_real_executor_lr_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    study, row, runtime_row, payload, run_spec = _controlled_executor_case()
    observed: dict[str, object] = {}

    class ReachedRuntime(Exception):
        pass

    def stop_after_report(*_args: object, **kwargs: object) -> None:
        observed["learning_rate"] = kwargs["learning_rate"]
        raise ReachedRuntime

    monkeypatch.setattr(lr_stages, "build_model_runtime", stop_after_report)
    with pytest.raises(ReachedRuntime):
        lr_stages.execute_candidate_entry(
            study.data,
            tmp_path,
            str(row["row_id"]),
            str(payload["candidate_role"]),
            candidate_payload=payload,
            output_stage="bias2-test",
            output_root=tmp_path,
            data_root=tmp_path / "mnist",
            download=False,
            device="cpu",
            controlled_runtime_row=runtime_row,
            controlled_run_spec=run_spec,
            controlled_summary_schema=RESULT_SCHEMA_VERSION,
            controlled_bias_learning_rate_overrides={
                "Bias_2": TARGET_RATES["Bias_2"]
            },
        )
    assert observed["learning_rate"] == TARGET_RATES


def test_real_executor_rejects_undeclared_bias_override_before_runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    study, row, runtime_row, payload, run_spec = _controlled_executor_case()
    reached_runtime = False

    def stop_after_report(*_args: object, **_kwargs: object) -> None:
        nonlocal reached_runtime
        reached_runtime = True
        raise AssertionError("runtime should not be reached")

    monkeypatch.setattr(lr_stages, "build_model_runtime", stop_after_report)
    with pytest.raises(
        LRProtocolValidationError,
        match=r"learning_rates\['Bias_2'\].*associated weight rate",
    ):
        lr_stages.execute_candidate_entry(
            study.data,
            tmp_path,
            str(row["row_id"]),
            str(payload["candidate_role"]),
            candidate_payload=payload,
            output_stage="bias2-test",
            output_root=tmp_path,
            data_root=tmp_path / "mnist",
            download=False,
            device="cpu",
            controlled_runtime_row=runtime_row,
            controlled_run_spec=run_spec,
            controlled_summary_schema=RESULT_SCHEMA_VERSION,
        )
    assert reached_runtime is False


def test_controlled_bias_override_must_match_run_spec(
    tmp_path: Path,
) -> None:
    study, row, runtime_row, payload, run_spec = _controlled_executor_case()
    run_spec["learning_rates_by_parameter"] = dict(SOURCE_RATES)
    with pytest.raises(
        ValueError,
        match="same complete learning_rates_by_parameter mapping",
    ):
        lr_stages.execute_candidate_entry(
            study.data,
            tmp_path,
            str(row["row_id"]),
            str(payload["candidate_role"]),
            candidate_payload=payload,
            output_stage="bias2-test",
            output_root=tmp_path,
            data_root=tmp_path / "mnist",
            download=False,
            device="cpu",
            controlled_runtime_row=runtime_row,
            controlled_run_spec=run_spec,
            controlled_summary_schema=RESULT_SCHEMA_VERSION,
            controlled_bias_learning_rate_overrides={
                "Bias_2": TARGET_RATES["Bias_2"]
            },
        )
