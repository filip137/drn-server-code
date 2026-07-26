from __future__ import annotations

import math

import pytest
import torch

from experiments.mnist_conv.lr_engine import transition_rows
from experiments.mnist_conv.lr_protocol import (
    LRProtocolValidationError,
    linear_quantile,
    parameter_relative_update_from_rms,
    probe_parameter_relative_rho_unit,
)
from experiments.mnist_conv.lr_step import ParameterTransition, SGDTransition


def _transition(
    name: str,
    *,
    bounded: bool,
    proposed_update_rms: float,
    normalized_update: float,
) -> ParameterTransition:
    state = torch.tensor([100.0, 100.0], dtype=torch.float64)
    return ParameterTransition(
        name=name,
        parameter_kind="bounded_weight" if bounded else "bias",
        bounded_gate=bounded,
        report_only=not bounded,
        pre_update=state,
        post_sgd_pre_projection=state.clone(),
        post_projection=state.clone(),
        gradient_rms=proposed_update_rms,
        proposed_update_rms=proposed_update_rms,
        normalized_update=normalized_update,
        proposed_bound_crossing_fraction=0.0 if bounded else None,
        lower_bound_occupancy=0.5 if bounded else None,
        upper_bound_occupancy=0.0 if bounded else None,
        combined_bound_occupancy=0.5 if bounded else None,
        projection_efficiency=1.0,
        proposal_is_numerically_zero=False,
        projection_gate_eligible=bounded,
    )


def test_parameter_relative_update_uses_frozen_initial_rms() -> None:
    assert parameter_relative_update_from_rms(
        0.25,
        initial_parameter_rms=0.005,
    ) == pytest.approx(50.0)


@pytest.mark.parametrize(
    ("proposed_update_rms", "initial_parameter_rms"),
    [
        (-1.0, 1.0),
        (1.0, 0.0),
        (1.0, -1.0),
        (math.inf, 1.0),
        (1.0, math.nan),
        (True, 1.0),
    ],
)
def test_parameter_relative_update_rejects_invalid_values(
    proposed_update_rms: float,
    initial_parameter_rms: float,
) -> None:
    with pytest.raises(LRProtocolValidationError):
        parameter_relative_update_from_rms(
            proposed_update_rms,
            initial_parameter_rms=initial_parameter_rms,
        )


def test_relative_rho_unit_is_max_of_per_parameter_linear_q90() -> None:
    updates = {
        "ConvWeight_0": [1.0] * 32,
        "DenseWeight_0": list(range(1, 33)),
        "ConvWeight_1": [0.25] * 32,
    }

    unit, by_parameter = probe_parameter_relative_rho_unit(updates)

    expected_dense = linear_quantile(range(1, 33), 0.9)
    assert by_parameter == {
        "ConvWeight_0": 1.0,
        "DenseWeight_0": expected_dense,
        "ConvWeight_1": 0.25,
    }
    assert unit == expected_dense

    # A flattened Q90 would dilute the limiting Dense tensor with values from
    # the two convolution tensors.  The v3 aggregate must not do that.
    flattened = [value for values in updates.values() for value in values]
    assert unit != linear_quantile(flattened, 0.9)


def test_relative_rho_unit_is_unchanged_by_extra_non_limiting_parameters() -> None:
    candidate_warmup_updates = {
        "DenseWeight_0": [0.1, 0.2, 0.3, 0.4],
        "ConvWeight_0": [0.01, 0.01, 0.01, 0.01],
    }
    original, original_by_parameter = probe_parameter_relative_rho_unit(
        candidate_warmup_updates
    )
    extended, extended_by_parameter = probe_parameter_relative_rho_unit(
        {
            **candidate_warmup_updates,
            "ConvWeight_1": [0.001, 0.001, 0.001, 0.001],
            "ConvWeight_2": [0.002, 0.002, 0.002, 0.002],
        }
    )

    assert original == pytest.approx(0.37)
    assert original_by_parameter["DenseWeight_0"] == pytest.approx(0.37)
    assert extended == original
    assert extended_by_parameter["DenseWeight_0"] == original


@pytest.mark.parametrize(
    "updates",
    [
        {},
        {"DenseWeight_0": []},
        {"DenseWeight_0": [0.0, 0.0]},
        {"DenseWeight_0": [1.0], " DenseWeight_0 ": [2.0]},
    ],
)
def test_relative_rho_unit_rejects_invalid_parameter_collections(updates) -> None:
    with pytest.raises(LRProtocolValidationError):
        probe_parameter_relative_rho_unit(updates)


def test_transition_rows_logs_span_and_frozen_relative_coordinates() -> None:
    transition = SGDTransition(
        parameters=(
            _transition(
                "DenseWeight_0",
                bounded=True,
                proposed_update_rms=2.0,
                normalized_update=0.02,
            ),
            _transition(
                "Bias_0",
                bounded=False,
                proposed_update_rms=0.5,
                normalized_update=5.0,
            ),
        ),
        restored=False,
    )

    rows = transition_rows(
        transition,
        step=7,
        learning_rate=0.125,
        rho_schedule=0.3,
        rho_schedule_relative=0.3,
        rho_schedule_span=2e-5,
        bounded_initial_rms_scales={"DenseWeight_0": 4.0},
    )

    weight, bias = rows
    assert weight["scheduled_rho"] == 0.3
    assert weight["scheduled_rho_relative"] == 0.3
    assert weight["scheduled_rho_span"] == 2e-5
    assert weight["normalized_update"] == 0.02
    assert weight["span_normalized_update"] == 0.02
    assert weight["initial_parameter_rms"] == 4.0
    assert weight["parameter_relative_update"] == 0.5

    # The captured pre-update tensor has RMS 100.  A result of 0.5 therefore
    # proves that logging used the supplied frozen W0 scale, not live state.
    assert weight["parameter_relative_update"] != pytest.approx(2.0 / 100.0)

    assert bias["normalized_update"] == 5.0
    assert bias["span_normalized_update"] is None
    assert bias["initial_parameter_rms"] is None
    assert bias["parameter_relative_update"] is None


@pytest.mark.parametrize(
    "scales",
    [
        {},
        {"DenseWeight_0": 4.0, "ConvWeight_0": 1.0},
        {"DenseWeight_0": 0.0},
        {"DenseWeight_0": -1.0},
        {"DenseWeight_0": math.inf},
        {"DenseWeight_0": math.nan},
    ],
)
def test_transition_rows_rejects_invalid_frozen_bounded_scales(scales) -> None:
    transition = SGDTransition(
        parameters=(
            _transition(
                "DenseWeight_0",
                bounded=True,
                proposed_update_rms=2.0,
                normalized_update=0.02,
            ),
        ),
        restored=False,
    )

    with pytest.raises((ValueError, FloatingPointError)):
        transition_rows(
            transition,
            step=1,
            learning_rate=0.1,
            rho_schedule=0.01,
            bounded_initial_rms_scales=scales,
        )
