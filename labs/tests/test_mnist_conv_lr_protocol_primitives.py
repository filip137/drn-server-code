from __future__ import annotations

import math
from dataclasses import replace

import pytest

from experiments.mnist_conv import lr_protocol as lr


def _tensor(
    *,
    name: str = "ConvWeight_0",
    bounded: bool = True,
    gradient_rms: float = 1.0,
    proposed_update_rms: float = 1.0,
    normalized_update: float = 0.01,
    lower_bound_occupancy: float = 0.5,
    upper_bound_occupancy: float = 0.0,
    projection_efficiency: float | None = 1.0,
    proposed_bound_crossing_fraction: float = 0.0,
) -> lr.RangeTensorRecord:
    return lr.RangeTensorRecord(
        name=name,
        bounded=bounded,
        gradient_rms=gradient_rms,
        proposed_update_rms=proposed_update_rms,
        normalized_update=normalized_update,
        lower_bound_occupancy=lower_bound_occupancy,
        upper_bound_occupancy=upper_bound_occupancy,
        projection_efficiency=projection_efficiency,
        proposed_bound_crossing_fraction=proposed_bound_crossing_fraction,
    )


def _records(
    count: int,
    *,
    tensor_at_step=None,
    loss_at_step=None,
    rhos: tuple[float, ...] | None = None,
) -> list[lr.RangeStepRecord]:
    result = []
    for step in range(1, count + 1):
        tensor = tensor_at_step(step) if tensor_at_step else _tensor()
        loss = loss_at_step(step) if loss_at_step else 1.0
        rho = rhos[step - 1] if rhos is not None else step * 1e-5
        result.append(
            lr.RangeStepRecord(
                step=step,
                learning_rate=rho / 0.01,
                rho=rho,
                loss=loss,
                tensors=(tensor,),
            )
        )
    return result


def test_frozen_six_row_contract_and_settings_are_exact() -> None:
    assert lr.LR_STUDY_SCHEMA_VERSION == "mnist-conv-lr-study/v1"
    assert lr.LR_RUN_SCHEMA_VERSION == "mnist-conv-run/v2"
    assert len(lr.FROZEN_STUDY_ROWS) == 6
    assert tuple(row.input_gain for row in lr.FROZEN_STUDY_ROWS) == (
        75.6030807495,
        84.8402175903,
        31.8188591003,
        253.3022308350,
        716.3439331055,
        661.4369506836,
    )
    assert lr.frozen_study_row("conv1", "ours") == lr.FrozenStudyRow(
        "conv1", "ours", "mnist_bp_amp_v4_c1", 4.0, 1.0, 84.8402175903, 4, 4
    )
    assert lr.frozen_study_row("conv2", "legacy").input_gain == 661.4369506836
    assert lr.validate_frozen_study_rows(
        reversed(lr.FROZEN_STUDY_ROWS)
    ) == lr.FROZEN_STUDY_ROWS
    assert lr.validate_frozen_study_row(
        vars(lr.FROZEN_STUDY_ROWS[0])
    ) == lr.FROZEN_STUDY_ROWS[0]
    settings = dict(lr.FROZEN_STUDY_SETTINGS)
    settings["unrelated_executor_setting"] = "local"
    assert lr.validate_frozen_study_settings(settings)["batch_size"] == 16


def test_frozen_contract_rejects_scientific_drift_with_expected_first_message() -> None:
    rows = list(lr.FROZEN_STUDY_ROWS)
    rows[0] = replace(rows[0], input_gain=rows[0].input_gain + 1e-9)
    with pytest.raises(
        lr.LRProtocolValidationError,
        match=r"^Expected row .* frozen row .*Provided value:",
    ):
        lr.validate_frozen_study_rows(rows)

    settings = dict(lr.FROZEN_STUDY_SETTINGS)
    settings["batch_size"] = 4
    with pytest.raises(
        lr.LRProtocolValidationError,
        match=r"^Expected study settings.batch_size to be 16. Provided value: 4.$",
    ):
        lr.validate_frozen_study_settings(settings)


def test_update_scale_projection_and_linear_q90_math() -> None:
    assert lr.rms([3.0, 4.0]) == pytest.approx(math.sqrt(12.5))
    assert lr.normalized_update_rho([100.0, -100.0]) == 1.0
    assert lr.normalized_update_from_rms(0.25) == 0.0025
    assert lr.normalized_bias_update([1e-8], [0.0]) == 1.0
    assert lr.projection_efficiency([2.0, 0.0], [1.0, 0.0]) == 0.5
    assert lr.projection_efficiency([0.0], [0.0]) is None
    assert lr.linear_quantile([0.0, 10.0], 0.9) == 9.0
    assert lr.probe_rho_unit(range(1, 12)) == 10.0
    assert lr.target_learning_rates(0.01)[1e-5] == 0.001


def test_geometric_range_schedule_has_exact_endpoints_and_optional_extension() -> None:
    main = lr.range_normalized_update_schedule()
    extended = lr.range_normalized_update_schedule(include_extension=True)
    assert len(main) == 600
    assert main[0] == 1e-5
    assert main[-1] == 1e-2
    assert all(left < right for left, right in zip(main, main[1:]))
    assert len(extended) == 728
    assert extended[:600] == main
    assert extended[599] == 1e-2
    assert extended[600] == 1e-2
    assert extended[-1] == 3e-2
    rates = lr.range_learning_rate_schedule(0.02, include_extension=True)
    assert rates[0] == 5e-4
    assert rates[-1] == 1.5


def test_rescue_range_schedule_preserves_the_v1_tail_bit_for_bit() -> None:
    legacy = lr.range_normalized_update_schedule()
    rescue_kwargs = {
        "main_steps": 1_199,
        "start_rho": 1e-8,
        "end_rho": 1e-2,
        "exact_tail_start_rho": 1e-5,
        "exact_tail_steps": 600,
    }

    rescue = lr.range_normalized_update_schedule(**rescue_kwargs)
    assert len(rescue) == 1_199
    assert rescue[0] == 1e-8
    assert rescue[599] == 1e-5
    assert rescue[-1] == 1e-2
    assert rescue[599:] == legacy
    assert all(left < right for left, right in zip(rescue, rescue[1:]))

    # A single interpolated 1e-8--1e-2 schedule is mathematically equivalent
    # but does not preserve the seam or legacy floating-point values exactly.
    direct = lr.geometric_schedule(1e-8, 1e-2, 1_199)
    assert direct[599] < 1e-5
    assert direct[599:] != legacy

    expected_crossings = {
        1e-5: 600,
        3e-5: 696,
        1e-4: 800,
        3e-4: 895,
        1e-3: 1_000,
        3e-3: 1_095,
        1e-2: 1_199,
    }
    for target, expected_step in expected_crossings.items():
        crossing_step = next(
            step for step, value in enumerate(rescue, start=1) if value >= target
        )
        assert crossing_step == expected_step

    extended = lr.range_normalized_update_schedule(
        include_extension=True,
        **rescue_kwargs,
    )
    assert len(extended) == 1_327
    assert extended[:1_199] == rescue
    assert extended[1_198] == 1e-2
    assert extended[1_199] == 1e-2
    assert extended[-1] == 3e-2

    rates = lr.range_learning_rate_schedule(
        0.02,
        include_extension=True,
        **rescue_kwargs,
    )
    assert rates[0] == 5e-7
    assert rates[599] == 5e-4
    assert rates[-1] == 1.5


def test_exact_tail_schedule_rejects_a_non_geometric_piecewise_contract() -> None:
    with pytest.raises(
        lr.LRProtocolValidationError,
        match="Expected exact tail geometric ratio",
    ):
        lr.range_normalized_update_schedule(
            main_steps=1_198,
            start_rho=1e-8,
            end_rho=1e-2,
            exact_tail_start_rho=1e-5,
            exact_tail_steps=600,
        )


def test_candidate_schedule_has_exact_860_step_warmup_and_zero_cosine_endpoint() -> None:
    assert math.ceil(0.05 * lr.CANDIDATE_TOTAL_STEPS) == lr.CANDIDATE_WARMUP_STEPS
    schedule = lr.candidate_learning_rate_schedule(0.2)
    assert len(schedule) == 17_190
    assert schedule[0] == pytest.approx(0.2 / 860)
    assert schedule[858] == pytest.approx(0.2 * 859 / 860)
    assert schedule[859] == 0.2
    assert schedule[860] < 0.2
    assert schedule[-1] == 0.0
    assert all(
        left >= right
        for left, right in zip(schedule[859:], schedule[860:])
    )


@pytest.mark.parametrize(
    ("kind", "tensor_at_step", "loss_at_step", "confirmed_step"),
    [
        (
            "gradient_rms_explosion",
            lambda step: _tensor(gradient_rms=101.0 if step >= 33 else 1.0),
            None,
            40,
        ),
        (
            "bound_occupancy_increase",
            lambda step: _tensor(lower_bound_occupancy=0.71 if step >= 33 else 0.5),
            None,
            48,
        ),
        (
            "projection_efficiency",
            lambda step: _tensor(projection_efficiency=0.49 if step >= 33 else 1.0),
            None,
            48,
        ),
        (
            "loss_ema_explosion",
            None,
            lambda step: 5.0 if step >= 33 else 1.0,
            40,
        ),
    ],
)
def test_range_sustained_gates_backdate_failure_to_first_bad_step(
    kind, tensor_at_step, loss_at_step, confirmed_step
) -> None:
    records = _records(
        confirmed_step,
        tensor_at_step=tensor_at_step,
        loss_at_step=loss_at_step,
    )
    result = lr.analyze_range_stop_gates(
        records,
        {"ConvWeight_0": 0.5},
        ema_decay=0.0,
    )
    assert result.failure is not None
    assert result.failure.kind == kind
    assert result.failure.onset_step == 33
    assert result.failure.confirmed_step == confirmed_step
    assert result.maximum_admissible_step == 32


def test_range_nonfinite_is_immediate_even_inside_reference_window() -> None:
    records = _records(32)
    records[6] = replace(records[6], loss=float("nan"))
    result = lr.analyze_range_stop_gates(records, {"ConvWeight_0": 0.5})
    assert result.failure == lr.RangeFailure(
        "non_finite", 7, 7, detail="loss"
    )
    assert result.maximum_admissible_step == 6


def test_zero_proposals_are_ignored_without_clearing_projection_streak() -> None:
    def tensor_at_step(step: int) -> lr.RangeTensorRecord:
        if 41 <= step <= 44:
            return _tensor(
                proposed_update_rms=0.0,
                normalized_update=0.0,
                projection_efficiency=None,
            )
        if 33 <= step <= 52:
            return _tensor(projection_efficiency=0.49)
        return _tensor()

    result = lr.analyze_range_stop_gates(
        _records(52, tensor_at_step=tensor_at_step),
        {"ConvWeight_0": 0.5},
    )
    assert result.failure is not None
    assert result.failure.kind == "projection_efficiency"
    assert result.failure.onset_step == 33
    assert result.failure.confirmed_step == 52


def test_crossings_are_diagnostic_only_and_safe_main_range_allows_extension() -> None:
    records = _records(
        600,
        tensor_at_step=lambda step: _tensor(
            proposed_bound_crossing_fraction=1.0,
        ),
        rhos=lr.range_normalized_update_schedule(),
    )
    result = lr.analyze_range_stop_gates(records, {"ConvWeight_0": 0.5})
    assert result.failure is None
    assert result.maximum_admissible_step == 600
    assert result.extension_allowed is True


def test_an_active_unsustained_gate_at_step_600_blocks_extension() -> None:
    records = _records(
        600,
        tensor_at_step=lambda step: _tensor(
            projection_efficiency=0.49 if step >= 595 else 1.0
        ),
    )
    result = lr.analyze_range_stop_gates(records, {"ConvWeight_0": 0.5})
    assert result.failure is None
    assert result.extension_allowed is False
    assert result.active_gates_at_main_end == (
        "projection_efficiency:ConvWeight_0",
    )


def test_rescue_gate_uses_step_1199_as_the_main_range_boundary() -> None:
    cleared_old_boundary_streak = _records(
        1_199,
        tensor_at_step=lambda step: _tensor(
            projection_efficiency=0.49 if 595 <= step <= 600 else 1.0
        ),
    )
    result = lr.analyze_range_stop_gates(
        cleared_old_boundary_streak,
        {"ConvWeight_0": 0.5},
        main_range_steps=1_199,
    )
    assert result.failure is None
    assert result.maximum_admissible_step == 1_199
    assert result.extension_allowed is True
    assert result.active_gates_at_main_end == ()

    active_rescue_boundary_streak = _records(
        1_199,
        tensor_at_step=lambda step: _tensor(
            projection_efficiency=0.49 if step >= 1_194 else 1.0
        ),
    )
    result = lr.analyze_range_stop_gates(
        active_rescue_boundary_streak,
        {"ConvWeight_0": 0.5},
        main_range_steps=1_199,
    )
    assert result.failure is None
    assert result.extension_allowed is False
    assert result.active_gates_at_main_end == (
        "projection_efficiency:ConvWeight_0",
    )


def test_range_candidate_extraction_uses_next_eight_steps_and_log_middle() -> None:
    rhos = lr.range_normalized_update_schedule(include_extension=True)
    records = _records(
        len(rhos),
        rhos=rhos,
        loss_at_step=lambda step: 10.0 if step <= 201 else 8.0,
    )
    result = lr.extract_range_candidates(records, ema_decay=0.0)
    assert result.status == "resolved"
    assert result.reason is None
    assert result.selected_targets == (1e-4, 1e-3, 1e-2)
    assert result.high is not None
    assert result.high.crossing_step == 600
    assert result.high.window_start_step == 601
    assert result.high.window_end_step == 608


def test_range_candidate_extraction_leaves_non_decreasing_trace_unresolved() -> None:
    rhos = lr.range_normalized_update_schedule(include_extension=True)
    result = lr.extract_range_candidates(
        _records(len(rhos), rhos=rhos),
        ema_decay=0.0,
    )
    assert result.status == "unresolved"
    assert result.reason == "no_decreasing_loss_region"
    assert result.selected_targets == ()


def test_range_candidate_duplicate_roles_are_filled_deterministically() -> None:
    rhos = lr.range_normalized_update_schedule(include_extension=True)
    records = _records(
        len(rhos),
        rhos=rhos,
        loss_at_step=lambda step: 10.0 if step <= 590 else 8.0,
    )
    result = lr.extract_range_candidates(records, ema_decay=0.0)
    assert result.status == "resolved"
    assert result.fast is not None and result.high is not None and result.middle is not None
    assert result.fast.rho_target == 1e-2
    assert result.high.rho_target == 3e-3
    assert result.middle.rho_target == 1e-3


def _candidate(
    candidate_id: str,
    rate: float,
    loss: float | None,
    accuracy: float | None,
    efficiency: float | None,
    *,
    admissible: bool = True,
) -> lr.CandidateRunResult:
    return lr.CandidateRunResult(
        candidate_id=candidate_id,
        peak_learning_rate=rate,
        admissible=admissible,
        final_validation_loss=loss,
        final_validation_accuracy=accuracy,
        median_projection_efficiency=efficiency,
        inadmissible_reason=None if admissible else "numerical_failure",
    )


def test_final_selection_uses_two_percent_plateau_log_center_and_ties() -> None:
    selection = lr.select_final_candidate(
        [
            _candidate("low", 1e-3, 1.0, 0.91, 0.8),
            _candidate("high", 4e-3, 1.01, 0.92, 0.7),
            _candidate("outside", 16e-3, 1.021, 0.99, 1.0),
        ]
    )
    assert selection.status == "frozen_seed0_screen"
    assert [candidate.candidate_id for candidate in selection.plateau] == ["low", "high"]
    assert selection.selected is not None
    assert selection.selected.candidate_id == "high"

    efficiency_tie = lr.select_final_candidate(
        [
            _candidate("low", 1e-3, 1.0, 0.9, 0.9),
            _candidate("high", 4e-3, 1.0, 0.9, 0.8),
        ]
    )
    assert efficiency_tie.selected is not None
    assert efficiency_tie.selected.candidate_id == "low"

    raw_lr_tie = lr.select_final_candidate(
        [
            _candidate("low", 1e-3, 1.0, 0.9, 0.9),
            _candidate("high", 4e-3, 1.0, 0.9, 0.9),
        ]
    )
    assert raw_lr_tie.selected is not None
    assert raw_lr_tie.selected.candidate_id == "low"


def test_final_selection_excludes_inadmissible_and_can_be_unresolved() -> None:
    bad = _candidate("bad", 1e-3, None, None, None, admissible=False)
    good = _candidate("good", 2e-3, 2.0, 0.5, 0.8)
    selection = lr.select_final_candidate([bad, good])
    assert selection.selected == good
    unresolved = lr.select_final_candidate([bad])
    assert unresolved.status == "unresolved"
    assert unresolved.reason == "no_admissible_candidates"
